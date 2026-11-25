#!/usr/bin/env python3
"""preact 峰值探针：验证"加宽降低模拟摆幅需求"的机制猜测。

Gilbert 乘法器失真项 ∝ β·(prod/Vsat)^2。prod 的规模由每层 preact
（tanh 之前的 z+mix）峰值决定。本脚本统计各变体在真实测试输入上的
preact 峰值分布，直接给出摆幅需求，不再用 NRMS 间接推断。

用法:
  python3 preact_probe.py VARIANT WEIGHT_PATH [SECONDS]
输出:
  JSON 到 stdout，字段含每层 preact 的 max/abs-mean/p99。
"""
import sys
import json
import torch
import torch.nn.functional as F
import soundfile as sf
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VARIANTS = {
    "dw_ch12":     dict(ch=12, rf_pct=100, depthwise=True),
    "dw_ch24":     dict(ch=24, rf_pct=100, depthwise=True),
    "wd_ch24_r6":  dict(ch=24, rf_pct=100, depthwise=True),
    "wd_ch32_r8":  dict(ch=32, rf_pct=100, depthwise=True),
    "lite":        dict(ch=12, rf_pct=100, depthwise=False),
}


def build_groups(ch, rf_pct):
    head_ch = max(1, ch // 2)
    d1 = [1, 2, 4, 8, 16, 32, 64]
    d2 = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    if rf_pct != 100:
        d1 = [max(1, int(d * rf_pct / 100)) for d in d1]
        d2 = [max(1, int(d * rf_pct / 100)) for d in d2]
    return [(1, ch, d1, head_ch, False), (ch, head_ch, d2, 1, True)]


def is_lowrank(sd, pre):
    return f"{pre}.layer1x1.0.weight" in sd


def main():
    variant = sys.argv[1]
    wpath = sys.argv[2]
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    cfg = VARIANTS[variant]
    GROUPS = build_groups(cfg["ch"], cfg["rf_pct"])
    depthwise = cfg["depthwise"]

    try:
        sd = torch.load(wpath, map_location="cpu", weights_only=False)
    except TypeError:  # torch 1.10 无 weights_only 参数
        sd = torch.load(wpath, map_location="cpu")
    sd = {k: (v.float().to(DEVICE) if torch.is_tensor(v) else v) for k, v in sd.items()}

    data_root = Path(wpath).parents[1] / "data"
    x, sr = sf.read(str(data_root / "tsa15_test_input.wav"))
    n = int(seconds * sr)
    xt = torch.from_numpy(x[:n].astype("float32"))[None, None].to(DEVICE)

    peaks = []
    c = xt
    y = xt
    with torch.no_grad():
        for gi, (in_size, ch, dils, head_out, head_bias) in enumerate(GROUPS):
            y = F.conv1d(y, sd[f"groups.{gi}.rechannel.weight"], None, 1)
            for li, d in enumerate(dils):
                pre = f"groups.{gi}.layers.{li}"
                groups = ch if depthwise else 1
                z = F.conv1d(F.pad(y, (2 * d, 0)),
                             sd[f"{pre}.conv.weight"], sd[f"{pre}.conv.bias"],
                             dilation=d, groups=groups)
                Lz = z.shape[-1]
                mix = F.conv1d(c, sd[f"{pre}.input_mixer.weight"], None, 1)[:, :, -Lz:]
                preact = z + mix  # 进入 tanh 之前 —— Gilbert 失真的输入规模
                peaks.append({
                    "group": gi, "layer": li, "dilation": d,
                    "max": round(float(preact.abs().max()), 4),
                    "p99": round(float(preact.abs().flatten().kthvalue(
                        max(1, int(preact.numel() * 0.99))).values), 4),
                    "mean_abs": round(float(preact.abs().mean()), 4),
                })
                z = torch.tanh(preact)
                if is_lowrank(sd, pre):
                    h = F.conv1d(z, sd[f"{pre}.layer1x1.0.weight"], None, 1)
                    mix1 = F.conv1d(h, sd[f"{pre}.layer1x1.1.weight"],
                                    sd[f"{pre}.layer1x1.1.bias"], 1)
                else:
                    mix1 = F.conv1d(z, sd[f"{pre}.layer1x1.weight"],
                                    sd[f"{pre}.layer1x1.bias"], 1)
                y = y[:, :, -Lz:] + mix1
                # preact 统计不依赖 head 累加；此处只保留残差流 y 即可
                # （原实现把 ch 通道的 z 直接累加到 head_input，组间 head_out 不同会维度不匹配）

    global_max = max(p["max"] for p in peaks)
    global_p99 = max(p["p99"] for p in peaks)
    res = {
        "variant": variant, "params": sum(v.numel() for v in sd.values() if torch.is_tensor(v)),
        "device": str(DEVICE), "seconds": seconds,
        "preact_global_max": global_max, "preact_global_p99_max": global_p99,
        # 三次项相对误差 β·(x/V)^2 ≤ 1% → V ≥ x·sqrt(β/0.01) = x·sqrt(0.2)
        "vsat_for_1pct_h3_by_max": round(global_max * (0.002 / 0.01) ** 0.5, 2),
        "vsat_for_1pct_h3_by_p99": round(global_p99 * (0.002 / 0.01) ** 0.5, 2),
        "layers": peaks,
    }
    print(json.dumps(res))


if __name__ == "__main__":
    main()
