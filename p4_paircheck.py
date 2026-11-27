"""P4-check：用配对口径重算 head 的真实贡献，回答「P3 的 head 占比 85% 是否稳健」。

## 为什么必须重算
P3 的口径是「先对 mc 次 NRMS 做 RMS 聚合，再用两个聚合值相减」。
在 s7 上实测：
  · 口径 1（P3）：base − head_bypass = −0.0038（head 是"有害"的）
  · 口径 2（配对）：            = −0.0040 ± 0.0138，改善仅 3/8 次
两者数值巧合接近，但**符号一致性只有 3/8**，说明这个量在噪声之内。
而 ALL（全旁路）的配对差 = +0.0168 ± 0.0065，8/8 全正 → 真实效应。

因此：单组归因必须报「配对均值 ± σ 与符号一致性」，否则符号毫无意义。

## 本脚本做两件事
1. 对 5 种子 × [head, ALL] 做 8 次配对，给出 mean±σ 与符号一致性
2. 给出 P3 口径与配对口径的对比表，标出哪些「head 占比」是噪声
"""
import json
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
MC = 8

# 硬性要求加速卡：CPU 上跑虽然结果一致，但慢 ~30 倍，
# 且历史上出现过节点 libhydmi.so 缺库导致静默降级 CPU 的情况
import os
if os.environ.get("P4_REQUIRE_ACCEL", "0") == "1" and DEV != "cuda":
    raise SystemExit("P4_REQUIRE_ACCEL=1 但无可用加速卡（节点 DCU 不可见），拒绝在 CPU 上跑")

DIALECT = dict(BITS=6, SIGMA_MISMATCH=0.01, DRIFT=0.03,
               BETA_H3=0.002, SIGMA_OFFSET=0.005, SNR_DB=70.0, V_SAT=16.0)

xi, _ = sf.read(str(V.DATA / "tsa15_test_input.wav"), dtype="float32")
yi, _ = sf.read(str(V.DATA / "tsa15_test_target.wav"), dtype="float32")
n = int(48000 * 5.0)
xt = torch.from_numpy(np.ascontiguousarray(xi[:n])).to(DEV)
yt = torch.from_numpy(np.ascontiguousarray(yi[:n])).to(DEV)

base_p = dict(A.P)
P3.base_p = base_p

# 旁路目标：单组与全旁路。head 之外逐层也测（看哪些层稳健）
GROUPS = ["rechannel"] + [f"L{i}" for i in range(10)] + ["head", "ALL"]


def nrms_once(sd, cfg, group):
    """group="BASE" 或某个组名 / "ALL"。返回 8 次配对的原始 NRMS 列表。"""
    A.P.clear(); A.P.update(base_p); A.P.update(DIALECT)
    try:
        gmap, _cmap, ng, _nc = P3.build_layout(cfg, sd)
        vals = []
        for i in range(MC):
            dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(2000 + i))
            orig_g = dev._gilbert
            cnt = {"i": 0}

            def g_w(prod):
                k = cnt["i"]; cnt["i"] = k + 1
                grp = gmap.get(k, "?")
                if group == "ALL" or grp == group:
                    return prod.sum(dim=2)
                return orig_g(prod)

            dev._gilbert = g_w
            with torch.no_grad():
                p = V.run_forward(dev, cfg, xt)
            dev._gilbert = orig_g
            if cnt["i"] != ng:
                raise RuntimeError(f"[layout] {cnt['i']} != {ng}")
            vals.append(V.nrms_of(p, yt))
        return vals
    finally:
        A.P.clear(); A.P.update(base_p)


def main():
    out = {"meta": {"mc": MC, "v_sat": 16.0, "dialect": DIALECT,
                    "groups": GROUPS}, "seeds": {}}
    print(f"=== 配对口径重算（MC={MC}，逐次配对求差）===", flush=True)

    for s in V.SEEDS:
        cfg, sd = V.load_nam(V.nam_path(s))
        print(f"\n--- {s} ---", flush=True)
        base = np.array(nrms_once(sd, cfg, "BASE"))
        rec = {"base_mc": base.tolist(),
               "base_rms": float(np.sqrt((base ** 2).mean())),
               "base_mean": float(base.mean()), "groups": {}}
        for g in GROUPS:
            v = np.array(nrms_once(sd, cfg, g))
            d = base - v                      # 正 = 旁路后变好（该组有害）
            n_pos = int((d > 0).sum())
            sem = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
            # 零效应判定：|均值| 与 σ 都比浮点噪声还小 → 该组对退化无贡献，
            # 此时"改善 k/n"只是浮点末位的符号统计，不能当证据
            zero = bool(abs(d.mean()) < 1e-9 and d.std(ddof=1) < 1e-9)
            # 稳健性：均值超过 2 倍 SEM 且符号一致性 >= 6/8
            robust = bool(not zero and abs(d.mean()) > 2 * sem
                          and max(n_pos, len(d) - n_pos) >= 6)
            rec["groups"][g] = {
                "nrms_mc": v.tolist(),
                "nrms_rms": float(np.sqrt((v ** 2).mean())),
                "d_pair_mean": float(d.mean()), "d_pair_std": float(d.std(ddof=1)),
                "d_pair_sem": sem, "d_pair_rms": float(np.sqrt((d ** 2).mean())),
                "n_better": n_pos, "n": int(len(d)),
                "zero_effect": zero, "robust": robust,
                "d_agg_p3": float(rec["base_rms"] - np.sqrt((v ** 2).mean())),
            }
            tag = "○零" if zero else ("✓" if robust else "✗噪声")
            print(f"  {g:<10} Δ配对={d.mean():+8.5f}±{d.std(ddof=1):7.5f} "
                  f"改善{n_pos:>2}/{len(d)}  Δ聚合(P3)={rec['groups'][g]['d_agg_p3']:+8.5f} {tag}",
                  flush=True)
        out["seeds"][s] = rec

    # 汇总：哪些组的归因是稳健的
    print("\n=== 汇总：稳健性矩阵（✓ = |mean|>2SEM 且 符号一致性>=6/8）===")
    print("  %-10s" % "组" + "".join("%8s" % s for s in V.SEEDS))
    for g in GROUPS:
        row = "".join(
            "%8s" % ("○" if out["seeds"][s]["groups"][g].get("zero_effect")
                     else ("✓" if out["seeds"][s]["groups"][g]["robust"] else "✗"))
            for s in V.SEEDS)
        print("  %-10s%s" % (g, row))

    print("\n=== head 贡献：配对 vs 聚合，以及占 ALL 的比例 ===")
    print("  %-6s%12s%12s%12s%12s%10s" % ("seed", "headΔ配对", "headΔ聚合", "ALLΔ配对", "head/ALL", "稳健?"))
    for s in V.SEEDS:
        g = out["seeds"][s]["groups"]
        h, a = g["head"], g["ALL"]
        ratio = h["d_pair_mean"] / a["d_pair_mean"] if abs(a["d_pair_mean"]) > 1e-12 else float("nan")
        print("  %-6s%+12.5f%+12.5f%+12.5f%11.1f%%%10s" % (
            s, h["d_pair_mean"], h["d_agg_p3"], a["d_pair_mean"], 100 * ratio,
            "○" if h.get("zero_effect") else ("✓" if h["robust"] else "✗噪声")))

    json.dump(out, open(V.OUT / "p4_paircheck.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n[saved] {V.OUT / 'p4_paircheck.json'}", flush=True)


if __name__ == "__main__":
    main()
