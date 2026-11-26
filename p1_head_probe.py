"""P-1a：head 累加链幅度诊断（零训练成本，纯前向测量）

## 为什么必须先测这个
head 累加是 L 级串行电流求和。加法器会不会饱和，完全取决于
「累加值 RMS」与「加法器饱和摆幅 V_add」的比值（摆幅余量）。
不知道各级幅度，就无法判断加法器失真该按多大摆幅建模，
P-1 的主判据（加法失真占比）也就无从谈起。

## 测量内容
对每个种子、每个误差配置，记录
  - 每一级 head term 的 RMS（a_i）
  - 串行累加后每级的 RMS（h_i = h_{i-1} + a_i）
  - 相干增益：h_L / sqrt(sum a_i^2)，=1 表示纯随机相位，=sqrt(L) 表示完全相干
  - 末级 head_in 的 RMS 与峰值

## 用法
  python p1_head_probe.py                 # 5 种子，快速（0.5s）
  python p1_head_probe.py --seconds 3
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "github_mirrors" / "neural-amp-modeler"))

import analog_sim_generic as A          # noqa: E402
from analog_sim_generic import DevSim   # noqa: E402
from p1_head_accum import SEED_PATHS, resolve_nam, load_nam   # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DATA = ROOT / "data"


@torch.no_grad()
def probe_forward(dev, sd, cfg, x):
    """跑一遍前向，返回 (各级 term RMS, 串行累加各级 RMS, 末级统计)。"""
    lc = cfg["layers_configs"][0]
    K, dils = lc["kernel_size"], lc["dilations"]
    act = lc.get("activation", "Tanh")
    g_in = lc.get("groups_input", 1)
    l1 = lc.get("layer_1x1_config", {}) or {}
    g_l1 = l1.get("groups", 1) if l1.get("active") else 1

    y = x[None, None, :] if x.ndim == 1 else x[:, None, :]
    cond = y
    rf_no_head = 1 + sum((K - 1) * d for d in dils)
    out_len = y.shape[2] - (rf_no_head - 1)
    pre0 = "_layer_arrays.0."

    y = dev.mac1x1(y, sd[pre0 + "_rechannel.weight"], None, groups=1)

    terms = []
    for li, d in enumerate(dils):
        pre = f"{pre0}_layers.{li}"
        z = dev.conv(y, sd[f"{pre}._conv.weight"], sd.get(f"{pre}._conv.bias"), d, g_in)
        mix = dev.conv(cond, sd[f"{pre}._input_mixer.weight"], None, 1, 1)[:, :, -z.shape[2]:]
        n = min(z.shape[2], mix.shape[2])
        z = dev.act(z[:, :, -n:] + mix[:, :, -n:], act)
        res_out = dev.conv(z, sd[f"{pre}._layer1x1.weight"], sd.get(f"{pre}._layer1x1.bias"), 1, g_l1)
        y = y[:, :, -res_out.shape[2]:] + res_out
        terms.append(z[:, :, -out_len:])

    a_rms = [float(t.pow(2).mean().sqrt()) for t in terms]
    h = terms[0]
    h_rms = [a_rms[0]]
    for t in terms[1:]:
        h = h + t
        h_rms.append(float(h.pow(2).mean().sqrt()))
    final = terms[0]
    for t in terms[1:]:
        final = final + t
    return dict(a_rms=a_rms, h_rms=h_rms,
                head_in_rms=float(final.pow(2).mean().sqrt()),
                head_in_absmax=float(final.abs().max()),
                head_in_p99=float(final.abs().quantile(0.99)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", default=list(SEED_PATHS))
    ap.add_argument("--seconds", type=float, default=0.5)
    ap.add_argument("--out", default="p1_probe.json")
    args = ap.parse_args()

    xi, _ = sf.read(str(DATA / "tsa15_test_input.wav"), dtype="float32")
    n = int(48000 * args.seconds)
    xt = torch.from_numpy(xi[:n]).to(DEV)

    base = dict(A.P)
    rows = []
    print(f"[probe] dev={DEV} seconds={args.seconds} head_scale will be applied AFTER head_rechannel")
    for s in args.seeds:
        cfg, sd = load_nam(resolve_nam(s))
        rec = {"seed": s}
        # 理想算子（只关心幅度，先排除器件噪声）
        dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(0), sanity=True)
        r = probe_forward(dev, sd, cfg, xt)
        rec["ideal"] = r
        # 器件全非理想
        A.P.clear(); A.P.update(base)
        dev2 = DevSim(sd, torch.Generator(device=DEV).manual_seed(1000))
        r2 = probe_forward(dev2, sd, cfg, xt)
        rec["device"] = r2
        A.P.clear(); A.P.update(base)

        for tag in ("ideal", "device"):
            a = np.array(rec[tag]["a_rms"]); h = rec[tag]["h_rms"]
            coh = h[-1] / max(float(np.sqrt((a ** 2).sum())), 1e-12)
            rec[tag]["coherence_gain"] = float(coh)
            rec[tag]["L"] = len(a)
        rows.append(rec)

        d = rec["device"]
        print(f"\n[{s}] head_scale={cfg.get('head_scale')} L={d['L']}")
        print("  级  a_i(RMS)   h_i(RMS)   a_i/V_SAT  h_i/V_SAT")
        for i, (a, hv) in enumerate(zip(d["a_rms"], d["h_rms"])):
            print(f"  {i:2d}  {a:9.4f}  {hv:9.4f}   {a/16:8.3f}   {hv/16:8.3f}")
        print(f"  末级 head_in RMS={d['head_in_rms']:.3f}  |max|={d['head_in_absmax']:.2f} "
              f"p99={d['head_in_p99']:.3f}")
        print(f"  相干增益 h_L/√Σa² = {d['coherence_gain']:.3f}  "
              f"(√L={np.sqrt(d['L']):.2f}, 1.0=随机相位)")

    json.dump(rows, open(ROOT / args.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {args.out}")

    print("\n=== 幅度汇总（器件态）===")
    print(f"{'seed':>4} {'h1':>8} {'hL':>9} {'|max|':>8} {'p99':>8} {'相干':>6} {'hL/16':>7}")
    for r in rows:
        d = r["device"]
        print(f"{r['seed']:>4} {d['h_rms'][0]:8.3f} {d['head_in_rms']:9.3f} "
              f"{d['head_in_absmax']:8.2f} {d['head_in_p99']:8.3f} "
              f"{d['coherence_gain']:6.2f} {d['head_in_rms']/16:7.2f}")


if __name__ == "__main__":
    main()
