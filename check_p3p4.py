"""核对 A=P3 head 全旁路 与 B=P4 head V_SAT→∞ 是否等价，
并用**配对口径**判断 s7 的 head 贡献到底是正还是负。

## 为什么要配对
P3 的口径是「先对 mc 次 NRMS 做 RMS 聚合，再用两个聚合值相减」。
P2 早已记录 MC 单次波动极大（s7 三次 0.1016/0.0919/0.1171，±11%），
所以「聚合后相减」在效应量接近噪声时会得出随机符号。
正确做法是**同一 MC 种子下配对求差**，再对差值聚合——这样公共噪声抵消。

本脚本同时给出两种口径，看它们是否一致。
"""
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import numpy as np
import torch
import soundfile as sf
import analog_sim_generic as A
from analog_sim_generic import DevSim
import p1_vsat_sweep as V
import p3_sharp as P3

DEV = V.DEV
MC = 8                     # 与 P3 一致
SEED = sys.argv[1] if len(sys.argv) > 1 else "s7"

xi, _ = sf.read(str(V.DATA / "tsa15_test_input.wav"), dtype="float32")
yi, _ = sf.read(str(V.DATA / "tsa15_test_target.wav"), dtype="float32")
n = int(48000 * 5.0)
xt = torch.from_numpy(np.ascontiguousarray(xi[:n])).to(DEV)
yt = torch.from_numpy(np.ascontiguousarray(yi[:n])).to(DEV)

cfg, sd = V.load_nam(V.nam_path(SEED))
base_p = dict(A.P)
DIALECT = dict(BITS=6, SIGMA_MISMATCH=0.01, DRIFT=0.03,
               BETA_H3=0.002, SIGMA_OFFSET=0.005, SNR_DB=70.0, V_SAT=16.0)

gmap, _cmap, ng, _nc = P3.build_layout(cfg, sd)
head_idx = ng - 1
print(f"=== {SEED}  ng={ng}  head_idx={head_idx}  gmap[head_idx]={gmap[head_idx]}")

A.P.clear(); A.P.update(base_p)
P3.base_p = base_p


def run(mc_seed, mode):
    """mode: base / head_bypass / head_vinf / all_bypass"""
    A.P.clear(); A.P.update(base_p); A.P.update(DIALECT)
    try:
        dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(mc_seed))
        orig_g = dev._gilbert
        cnt = {"i": 0}

        def g_w(prod):
            i = cnt["i"]; cnt["i"] = i + 1
            if mode == "all_bypass":
                return prod.sum(dim=2)
            if i == head_idx and mode == "head_bypass":
                return prod.sum(dim=2)
            if i == head_idx and mode == "head_vinf":
                old = dev.p["V_SAT"]; dev.p["V_SAT"] = 1e5
                try:
                    return orig_g(prod)
                finally:
                    dev.p["V_SAT"] = old
            return orig_g(prod)

        dev._gilbert = g_w
        with torch.no_grad():
            p = V.run_forward(dev, cfg, xt)
        dev._gilbert = orig_g
        assert cnt["i"] == ng, f"layout mismatch {cnt['i']} vs {ng}"
        return V.nrms_of(p, yt), p
    finally:
        A.P.clear(); A.P.update(base_p)


modes = ["base", "head_bypass", "head_vinf", "all_bypass"]
vals = {m: [] for m in modes}
preds = {}
for i in range(MC):
    for m in modes:
        r, p = run(2000 + i, m)
        vals[m].append(r)
        if i == 0:
            preds[m] = p

print(f"\n=== 逐 MC 原始 NRMS（mc_seed=2000..{2000+MC-1}）===")
print("  " + "".join(f"{m:>14}" for m in modes))
for i in range(MC):
    print(f"  " + "".join(f"{vals[m][i]:>14.6f}" for m in modes))

agg = {m: float(np.sqrt((np.array(vals[m]) ** 2).mean())) for m in modes}
print(f"\n=== 口径 1：先 RMS 聚合再相减（P3 口径，会产生符号随机）===")
for m in modes[1:]:
    print(f"  base − {m:<12} = {agg['base'] - agg[m]:+.6f}"
          f"   (base={agg['base']:.6f} {m}={agg[m]:.6f})")

print(f"\n=== 口径 2：配对求差（正确做法，公共噪声抵消）===")
for m in modes[1:]:
    d = np.array(vals["base"]) - np.array(vals[m])
    print(f"  base − {m:<12} = mean {d.mean():+.6f} ± {d.std(ddof=1):.6f}"
          f"   RMS {np.sqrt((d**2).mean()):+.6f}   (n={len(d)})")

print(f"\n=== 配对差值的符号一致性（多少个 MC 次里 head 旁路是改善）===")
for m in ["head_bypass", "head_vinf"]:
    d = np.array(vals["base"]) - np.array(vals[m])
    n_pos = int((d > 0).sum())
    print(f"  {m:<12} 改善 {n_pos}/{len(d)} 次   逐次: "
          + " ".join(f"{x:+.4f}" for x in d))

print(f"\n=== A 与 B 是否等价（mc_seed=2000 的预测张量）===")
print(f"  max|Δ| = {float((preds['head_bypass'] - preds['head_vinf']).abs().max()):.3e}")

print(f"\n=== 可加性（同口径 2 配对）===")
d_head = np.array(vals["base"]) - np.array(vals["head_bypass"])
d_all = np.array(vals["base"]) - np.array(vals["all_bypass"])
print(f"  head 占比 = mean(head)/mean(all) = {d_head.mean() / d_all.mean():+.1%}")
print(f"  总退化 mean = {d_all.mean():+.6f}  (base={agg['base']:.6f} floor={agg['all_bypass']:.6f})")
