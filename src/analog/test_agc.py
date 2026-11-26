"""验证：若在每层乘法器前做"自动增益控制(AGC)"，器件退化能否消除？

物理动机：
    实测 a-grp4 的残差流从 0.20 涨到 89.8，Gilbert cell 在 V_SAT=16 下必然饱和，
    导致 44.7× 精度退化。真实模拟电路有两种应对：
      (a) 提高电源电压（V_SAT 大 4~8 倍）-> 功耗/面积爆炸
      (b) 加可变增益放大器(VGA/AGC)，把每层信号拉回 O(1) 动态范围
          -> 这是模拟前端里的标准做法，成本远低于高压工艺

本脚本验证 (b)：对每层输入按其 RMS 做归一化（增益可配置），
乘法器看到的信号始终在 O(1)，再用等量反向增益恢复。
数学上等价于给每层加一个"尺度不变"的乘法器，不改变网络函数。
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/root/autodl-tmp")
import analog_sim_generic as A  # noqa: E402
from analog_sim_generic import DevSim, load_ckpt  # noqa: E402
from nam.models.losses import esr as official_esr  # noqa: E402

DEV = "cuda"
BASE_P = dict(A.P)
RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/n3_a-grp4_s7"

ck = sorted(Path(RUN).glob("*/lightning_logs/version_0/checkpoints/epoch=*.ckpt"))[-1]
cfg, sd = load_ckpt(str(ck), str(ck.parent.parent.parent.parent / "config_model.json"))
lc = cfg["layers_configs"][0]
K, dils = lc["kernel_size"], lc["dilations"]
g_in = lc.get("groups_input", 1)
g_l1c = lc.get("layer_1x1_config", {})
g_l1 = g_l1c.get("groups", 1) if g_l1c.get("active") else 1
act = lc.get("activation", "Tanh")
rf = 1 + sum((K - 1) * d for d in dils)
head_scale = cfg.get("head_scale", 0.01)

x, _ = sf.read("data/tsa15_test_input.wav", dtype="float32")
yt, _ = sf.read("data/tsa15_test_target.wav", dtype="float32")
x = torch.from_numpy(x[:48000 * 3]).to(DEV)
yt = torch.from_numpy(yt[:48000 * 3]).to(DEV)


def fwd(dev, agc=False):
    """可选 AGC：每层乘法前把残差流按 RMS 归一化，乘法后恢复。"""
    pre0 = "_layer_arrays.0."
    y = x[None, None, :]
    cond = y
    L_in = min(y.shape[2], cond.shape[2])
    out_len = L_in - (rf - 1)
    rf_no_head = 1 + sum((K - 1) * d for d in dils)
    out_len_no_head = L_in - (rf_no_head - 1)

    y = dev.mac1x1(y, sd[pre0 + "_rechannel.weight"], None, groups=1)
    head_in = None
    for li, d in enumerate(dils):
        p = f"{pre0}_layers.{li}"
        if agc:
            # 每层输入按 RMS 归一（RMS 用全序列统计，模拟里可由 AGC 环路估计）
            rms = y.pow(2).mean().sqrt().clamp(min=1e-6)
            y_in = y / rms
            gscale = rms
        else:
            y_in, gscale = y, 1.0
        z = dev.conv(y_in, sd[f"{p}._conv.weight"], sd.get(f"{p}._conv.bias"), d, g_in) * gscale
        mix = dev.conv(cond, sd[f"{p}._input_mixer.weight"], None, 1, 1)[:, :, -z.shape[2]:]
        z1 = min(z.shape[2], mix.shape[2])
        z = dev.act(z[:, :, -z1:] + mix[:, :, -z1:], act)
        res_out = dev.conv(z, sd[f"{p}._layer1x1.weight"], sd.get(f"{p}._layer1x1.bias"), 1, g_l1)
        y = y[:, :, -res_out.shape[2]:] + res_out
        term = z[:, :, -out_len_no_head:]
        head_in = term if head_in is None else head_in[:, :, -out_len_no_head:] + term

    out = dev.conv(head_in, sd[pre0 + "_head_rechannel.weight"], sd.get(pre0 + "_head_rechannel.bias"), 1, 1)
    return (head_scale * out)[:, 0, :]


def ev(agc, v_sat, label):
    p = dict(BASE_P)
    p["V_SAT"] = v_sat
    A.P.clear()
    A.P.update(p)
    dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(0))
    with torch.no_grad():
        pr = fwd(dev, agc=agc)
    off = yt.numel() - pr.numel()
    t = yt[off:]
    e = float(official_esr(pr.float().reshape(1, -1), t.float().reshape(1, -1)))
    A.P.clear()
    A.P.update(BASE_P)
    print(f"{label:34s} V_SAT={v_sat:5.0f}  ESR={e:.4e}  NRMS={np.sqrt(max(e,0)):.4f}")
    return e


print(f"[cfg] {RUN}")
print(f"{'方案':<34s} {'V_SAT':>10s}")
print("-" * 62)
ev(False, 4096, "无 AGC（≈理想）")
ev(False, 16,   "无 AGC + V_SAT16")
ev(True,  16,   "AGC + V_SAT16")
ev(True,  4,    "AGC + V_SAT4")
ev(True,  2,    "AGC + V_SAT2")
