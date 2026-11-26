"""定量规律：累加项数（conv 输入通道数）如何影响器件退化？

假设：Gilbert cell 的输出是 V_sat·tanh(wx/V_sat)，对 wx 做三次展开
      tanh(u) ≈ u - u³/3，则乘法失真项与 (wx)³ 成正比。
      求和的失真 ∝ Σ(w_i x_i)³，当各项同量级时随"项数"变化复杂，
      但直觉上项数越多、合成信号幅度越大，越容易饱和。

实测对比（V_SAT=16，同一 test 段）：
    a-grp4       每输出累加 3 项（groups=4, ch=12）
    A2-causal    每输出累加 12 项
    a-dw-ch16    每输出累加 1 项（depthwise）但 ch=16
    a-dw-ch24    每输出累加 1 项，ch=24

本脚本统一测量：各模型的 (a) 残差流 RMS 增长倍数 (b) 器件退化倍数。
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/root/autodl-tmp")
from analog_sim_generic import DevSim, load_ckpt, sim_forward  # noqa: E402
from nam.models.losses import esr as official_esr  # noqa: E402
from nam.models.wavenet._wavenet import WaveNet  # noqa: E402

DEV = "cuda"
x, _ = sf.read("data/tsa15_test_input.wav", dtype="float32")
yt, _ = sf.read("data/tsa15_test_target.wav", dtype="float32")
x = torch.from_numpy(x[:48000 * 3]).to(DEV)
yt = torch.from_numpy(yt[:48000 * 3]).to(DEV)

RUNS = [
    ("a-grp4", "runs/n3_a-grp4_s7"),
    ("a-grp6-ch18", "runs/n3_a-grp6-ch18_s7"),
    ("a-dw-ch16", "runs/n5_a-dw-ch16_s7"),
    ("a-dw-ch24", "runs/n5_a-dw-ch24_s7"),
    ("A2-causal", "runs/n4_a2causal-300_s7"),
]


def find_ckpt(run):
    fs = sorted(
        Path(run).glob("*/lightning_logs/version_0/checkpoints/epoch=*.ckpt"),
        key=lambda p: int(p.name.split("epoch=")[1].split("_")[0]),
    )
    return fs[-1] if fs else None


def esr_of(pred, tgt):
    if pred.ndim == 1:
        pred = pred[None]
    off = tgt.numel() - pred.shape[-1]
    return float(official_esr(pred.float(), tgt[off:].float()[None]))


print(f"{'model':<16s} {'acc_terms':>9s} {'|y|rms_max':>11s} {'digital':>9s} "
      f"{'analog':>9s} {'退化':>7s}")
print("-" * 68)

for name, run in RUNS:
    ck = find_ckpt(run)
    if ck is None:
        print(f"{name:<16s} no ckpt")
        continue
    cfg_p = ck.parent.parent.parent.parent / "config_model.json"
    cfg, sd = load_ckpt(str(ck), str(cfg_p))
    lc = cfg["layers_configs"][0]
    ch = lc["channels"]
    g = lc.get("groups_input", 1)
    terms = ch // g

    # 数字参考
    m = WaveNet.init_from_config(json.load(open(cfg_p))["net"]["config"]).to(DEV).eval()
    m.load_state_dict(sd, strict=False)
    with torch.no_grad():
        pred = m(x[None, None])[0, 0]
        # 残差流 RMS 增长
        la = m._layer_arrays[0]
        la_out = la(x[None, None], x[None, None], head_input=None)
        rms_max = la_out[1].pow(2).mean().sqrt().item()
    e_dig = esr_of(pred, yt)

    # 器件
    dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(0))
    with torch.no_grad():
        pa = sim_forward(dev, sd, cfg, x)
    e_ana = esr_of(pa, yt)

    print(f"{name:<16s} {terms:>9d} {rms_max:>11.2f} {np.sqrt(max(e_dig,0)):>9.4f} "
          f"{np.sqrt(max(e_ana,0)):>9.4f} {e_ana/max(e_dig,1e-12):>6.1f}×")
