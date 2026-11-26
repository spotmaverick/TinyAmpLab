"""通用器件级物理仿真：适配任意官方 WaveNet 配置（含分组卷积）

与 `analog_sim_wavenet.py` 的区别：后者硬编码旧 lite 架构（2 组、3-tap、tanh 门控）。
本脚本从 `config_model.json` 直接构建，因此能仿真族 A 的所有变体
（`groups_input` / `bottleneck` / `layer_1x1 groups` / `kernel_size` / `LeakyReLU`）。

## 每个算子 → 电路块

| 数学操作 | 电路实现 | 非理想性 |
|---|---|---|
| 权重存储 | 6-bit 浮栅 | 量化 + 高斯失配 σ_m + 全局温漂 |
| 乘法 | Gilbert cell | 三次项 β·x² + 饱和 V_sat |
| 求和 | KCL 电流相加 | 无源、理想 |
| 卷积 | 每个 delay tap 一套乘法器阵列 | 同 Gilbert |
| 激活 | 差分对饱和（tanh）/ 二极管整流（LeakyReLU） | 输入失调 σ_off |
| 节点 | 热噪声 | SNR |
| 延迟线 | 数字 SRAM（数模混合） | 无损 |

## 用法

    python analog_sim_generic.py <ckpt> [--mc 8] [--quick] [--sanity]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

P = dict(
    BITS=6,
    SIGMA_MISMATCH=0.01,
    DRIFT=0.03,
    BETA_H3=0.002,
    V_SAT=16.0,          # 按残差流动范围定标；sanity 模式无限大
    SIGMA_OFFSET=0.005,
    SNR_DB=70,
    MC=8,
)

DATA = Path("/root/autodl-tmp/data")


def quant_sym(w, bits):
    """逐输出通道对称量化。"""
    if w.dim() == 1:
        return w
    levels = 2 ** (bits - 1) - 1
    dims = tuple(range(1, w.dim()))
    scale = w.abs().amax(dim=dims, keepdim=True).clamp(min=1e-12) / levels
    return (w / scale).round().clamp(-(levels + 1), levels) * scale


class DevSim:
    """一次蒙特卡洛采样的器件实例。"""

    def __init__(self, sd, gen, sanity=False):
        self.g = gen
        self.sanity = sanity
        p = dict(P)
        if sanity:
            p.update(SIGMA_MISMATCH=0.0, DRIFT=0.0, BETA_H3=0.0,
                     SIGMA_OFFSET=0.0, SNR_DB=999, V_SAT=1e9)
        self.p = p
        self.w = {}
        for k, v in sd.items():
            if not torch.is_tensor(v):
                continue
            v = v.to(DEVICE).float()
            wq = v if sanity else quant_sym(v, p["BITS"])
            if p["SIGMA_MISMATCH"] > 0:
                mm = torch.empty_like(wq).normal_(0, p["SIGMA_MISMATCH"], generator=gen)
            else:
                mm = 0.0
            if p["DRIFT"] > 0:
                dr = torch.empty(1, device=DEVICE).uniform_(-p["DRIFT"], p["DRIFT"], generator=gen)
            else:
                dr = 0.0
            self.w[k] = (wq * (1 + mm) * (1 + dr)).float()
        self._off = {}

    def noise(self, shape):
        if self.p["SNR_DB"] >= 200:
            return 0.0
        return torch.empty(shape, device=DEVICE).normal_(
            0, 0.25 / (10 ** (self.p["SNR_DB"] / 20)), generator=self.g)

    def _gilbert(self, prod):
        """Gilbert cell: 含三次项与饱和。prod 形状 (B, Cout, Cin, T)。"""
        if self.p["BETA_H3"] <= 0:
            return prod.sum(dim=2)
        vs = self.p["V_SAT"]
        xs = prod / vs
        return (vs * torch.tanh(prod * (1 + self.p["BETA_H3"] * xs.square()) / vs)).sum(dim=2)

    def conv(self, x, w, bias, dilation, groups):
        """通用 1D 因果分组卷积（支持任意 kernel_size 与 groups）。"""
        if x.ndim == 1:                      # (T,) -> (1,1,T)
            x = x[None, None, :]
        elif x.ndim == 2:                    # (B,T) -> (B,1,T)
            x = x[:, None, :]
        B, Cin, T = x.shape
        Cout, CinPerG, K = w.shape
        G = groups
        CinG = Cin // G
        CoutG = Cout // G
        xp = F.pad(x, ((K - 1) * dilation, 0))
        y = 0
        for k in range(K):
            tap = xp[:, :, k * dilation: k * dilation + T]            # (B,Cin,T)
            # 分组：把 tap 与 w 都 reshaped 成 (G, ...) 后逐组做 Gilbert
            tg = tap.reshape(B, G, CinG, T)
            wk = w[:, :, k].reshape(G, CoutG, CinG)                   # (G,CoutG,CinG)
            prod = tg.unsqueeze(2) * wk.view(1, G, CoutG, CinG, 1)    # (B,G,CoutG,CinG,T)
            prod = prod.reshape(B, G * CoutG, CinG, T)
            yk = self._gilbert(prod).reshape(B, G, CoutG, T).reshape(B, Cout, T)
            y = y + yk
        if bias is not None:
            y = y + bias[None, :, None]
        return y

    def mac1x1(self, x, w, bias=None, groups=1):
        return self.conv(x, w, bias, dilation=1, groups=groups)

    def act(self, z, kind):
        p = self.p
        if p["SIGMA_OFFSET"] > 0 and kind not in self._off:
            self._off[kind] = torch.empty((1, z.shape[1], 1), device=DEVICE).normal_(
                0, p["SIGMA_OFFSET"], generator=self.g)
        if p["SIGMA_OFFSET"] > 0:
            z = z + self._off[kind]
        if kind.startswith("tanh"):
            return torch.tanh(z)
        if kind.startswith("LeakyReLU"):
            sl = 0.01
            return torch.where(z > 0, z, sl * z)
        if kind.startswith("ReLU"):
            return torch.clamp(z, min=0)
        raise ValueError(f"未支持的激活: {kind}")


def sim_forward(dev, sd, cfg, x):
    """复刻官方 WaveNet.forward，算子换成器件实现。"""
    lc = cfg["layers_configs"][0]
    ch = lc["channels"]
    K = lc["kernel_size"]
    dils = lc["dilations"]
    act = lc.get("activation", "Tanh")
    g_in = lc.get("groups_input", 1)
    g_l1 = lc.get("layer_1x1_config", {}).get("groups", 1) if lc.get("layer_1x1_config", {}).get("active") else 1
    rf = 1 + sum((K - 1) * d for d in dils)
    head_scale = cfg.get("head_scale", 0.01)

    y = x[:, None, :] if x.ndim == 2 else (x[None, None, :] if x.ndim == 1 else x)
    cond = y                                  # condition = 原始输入（1 通道）
    # 官方 out_length 用完整 receptive_field（含 head_rechannel 的 -1）
    L_in = min(y.shape[2], cond.shape[2])
    rf_no_head = 1 + sum((K - 1) * d for d in dils)     # 不含 head_rechannel
    out_len = L_in - (rf - 1)                           # 含 head_rechannel
    out_len_no_head = L_in - (rf_no_head - 1)

    pre0 = "_layer_arrays.0."
    y = dev.mac1x1(y, sd[pre0 + "_rechannel.weight"], None, groups=1)
    head_in = None

    for li, d in enumerate(dils):
        pre = f"{pre0}_layers.{li}"
        z = dev.conv(y, sd[f"{pre}._conv.weight"], sd.get(f"{pre}._conv.bias"), d, g_in)
        # input_mixer: condition(1ch) -> channels，逐通道一个乘法器
        mix = dev.conv(cond, sd[f"{pre}._input_mixer.weight"], None, 1, 1)[:, :, -z.shape[2]:]
        z1len = min(z.shape[2], mix.shape[2])
        z = dev.act(z[:, :, -z1len:] + mix[:, :, -z1len:], act)
        z = z + dev.noise(z.shape)
        # layer1x1（无偏置裁剪），残差对齐到 layer_output 长度
        res_out = dev.conv(z, sd[f"{pre}._layer1x1.weight"], sd.get(f"{pre}._layer1x1.bias"), 1, g_l1)
        y = y[:, :, -res_out.shape[2]:] + res_out + dev.noise(res_out.shape)
        # head_term 裁到 out_len_no_head
        term = z[:, :, -out_len_no_head:]
        head_in = term if head_in is None else head_in[:, :, -out_len_no_head:] + term

    out = dev.conv(head_in, sd[pre0 + "_head_rechannel.weight"], sd.get(pre0 + "_head_rechannel.bias"), 1, 1)
    # 官方 WaveNet.forward: head_input 是 LayerArray 的返回值（已过 _head_rechannel），
    # 之后才乘 head_scale
    return (head_scale * out)[:, 0, :]


def load_ckpt(ckpt, cfg_path):
    cfg = json.load(open(cfg_path))["net"]["config"]
    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    clean = {}
    for k, v in sd.items():
        while k.startswith("_net."):
            k = k[len("_net."):]
        if "_has_sample" in k or "_sample_rate" in k:
            continue
        clean[k] = v
    return cfg, clean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--cfg")
    ap.add_argument("--mc", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="只取前 5 秒")
    ap.add_argument("--sanity", action="store_true", help="关闭全部非理想性对拍")
    args = ap.parse_args()

    cfg_path = args.cfg
    if not cfg_path:
        for parent in Path(args.ckpt).resolve().parents:
            c = parent / "config_model.json"
            if c.exists():
                cfg_path = str(c)
                break
    cfg, sd = load_ckpt(args.ckpt, cfg_path)

    xi, _ = sf.read(str(DATA / "tsa15_test_input.wav"), dtype="float32")
    yi, _ = sf.read(str(DATA / "tsa15_test_target.wav"), dtype="float32")
    if args.quick:
        xi, yi = xi[:48000 * 5], yi[:48000 * 5]
    x = torch.from_numpy(xi).to(DEVICE)
    y = torch.from_numpy(yi).to(DEVICE)

    lc = cfg["layers_configs"][0]
    rf = 1 + sum((lc["kernel_size"] - 1) * d for d in lc["dilations"])
    print(f"[cfg] ch={lc['channels']} K={lc['kernel_size']} layers={len(lc['dilations'])} "
          f"g_in={lc.get('groups_input',1)} act={lc.get('activation')} RF={rf}")

    from nam.models.losses import esr as official_esr

    def esr_of(pred, tgt):
        # pred: (T,) 或 (1,T)；官方 esr 要求 (B, T)
        if pred.ndim == 1:
            pred = pred[None]
        off = tgt.numel() - pred.shape[-1]
        t = tgt[off:]
        return float(official_esr(pred.float(), t.float()[None]))

    # 数字参考（直接用器件模型 + sanity，等于理想算子）
    dev0 = DevSim(sd, torch.Generator(device=DEVICE).manual_seed(0), sanity=True)
    with torch.no_grad():
        pred_ideal = sim_forward(dev0, sd, cfg, x)
    d = esr_of(pred_ideal, y)
    print(f"[sanity] 理想器件 ESR={d:.6e} NRMS={np.sqrt(max(d,0)):.6f} "
          f"(应≈0，非 0 说明仿真与配置不匹配)")

    if args.sanity:
        return

    res = []
    for i in range(args.mc):
        gen = torch.Generator(device=DEVICE).manual_seed(1000 + i)
        dev = DevSim(sd, gen)
        with torch.no_grad():
            pr = sim_forward(dev, sd, cfg, x)
        e = esr_of(pr, y)
        res.append(e)
        print(f"  MC[{i:02d}] ESR={e:.6e} NRMS={np.sqrt(max(e,0)):.6f}")
    a = np.array(res)
    print(f"\n[analog] MC={args.mc}  ESR mean={a.mean():.6e} std={a.std():.2e} "
          f"NRMS mean={np.sqrt(a.mean()):.6f}")
    print(f"[退化] 相对理想 ESR 倍数 = {a.mean()/max(d,1e-12):.2f}×")


if __name__ == "__main__":
    sys.exit(main())
