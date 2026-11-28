"""P6-D2：13 种子的"标定后器件精度"分布 —— 决定选型的那个数

## 为什么要单独跑这个

P6-D（k 扫描）只回答了"head 增益重平衡能不能帮每个种子"——答案是能，但残余差距天差地别：

    seed s7   k=1 0.1056 → k=2  0.0968
    seed s23  k=1 0.1235 → k=2  0.0960
    seed s37  k=1 0.1567 → k=4  0.1462   ← 修不动
    seed s31  k=1 0.2386 → k=32 0.1109
    seed s11  k=1 0.2147 → k=4  0.1151
    seed s41  k=1 0.2051 → k=32 0.1131

单看 k=1 或单看"最优 k"都会得出错误结论。真正要对标官方 NAM 的量是：

    **每个种子在"免费手段(k 重平衡) + 付费手段(分层摆幅)"都用上之后，最多能做到多少**

本脚本对每个种子在 (k, V_SAT) 二维网格上测 device NRMS，取该种子的最优值，
再跨种子取中位/分布。这样才能回答："目标拓扑的 13 种子中位，在付得起的价格下，
到底能不能追上官方 standard 的 0.0852"。

## 成本控制

- **只测 device，不重复测 digital**（digital 与 k/V 无关，P5-C 已验证 spread=0）。
  单次 eval ≈ 3 段 × mc 次前向，比 p5c 的 eval_k 省一半。
- 网格取 k ∈ {1, 4, 32}（覆盖小/中/大三档，P6-D 实测最优 k 落在 2~32），
  V ∈ {16, 32, 64, 96}（16 是不加钱，96 是"基本关掉削顶"）。

## 用法

    $PY p6_calibrate.py --seeds all --mc 4 --out out/p6_calib_all.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_analog_sim as O                                  # noqa: E402
from official_analog_sim import DevSim, forward, esr_nrms         # noqa: E402
from p5c_head_rebalance import rebalance                          # noqa: E402

DEVICE = O.DEVICE
SEGMENTS = [("seg0", 0.0), ("seg1", 8.0), ("seg2", 16.0)]


def load_model(nam_path):
    nam = json.load(open(nam_path))
    sys.path.insert(0, os.environ.get("NAM_SRC", "/public/home/spotmaverick/nam_proj"))
    from nam.models import init_from_nam
    net = init_from_nam(nam)
    sd = {k: v for k, v in net.state_dict().items()
          if "_has_sample" not in k and "_sample_rate" not in k}
    sd = {k[5:] if k.startswith("_net.") else k: v for k, v in sd.items()}
    return nam, sd


def device_nrms(nam, sd, seg_data, mc, v_sat):
    vals = []
    for _nm, xt, yt in seg_data:
        dl = []
        for i in range(mc):
            dv = DevSim(sd, torch.Generator(device=DEVICE).manual_seed(1000 + i))
            dv.p["V_SAT"] = float(v_sat)
            with torch.no_grad():
                pr = forward(dv, sd, nam, xt)
            dl.append(esr_nrms(pr, yt)[1])
        vals.append(float(np.sqrt(np.mean(np.square(dl)))))
    return float(np.median(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="all", help="逗号分隔的种子名，或 all")
    ap.add_argument("--model-dir", default="models/seeds")
    ap.add_argument("--mc", type=int, default=4)
    ap.add_argument("--ks", default="1,4,32")
    ap.add_argument("--vsat", default="16,32,64,96")
    ap.add_argument("--seg-len", type=float, default=5.0)
    ap.add_argument("--out", default="out/p6_calib.json")
    args = ap.parse_args()

    if args.seeds == "all":
        seeds = sorted(Path(args.model_dir).glob("*.nam"))
    else:
        seeds = [Path(args.model_dir) / f"{s}.nam" for s in args.seeds.split(",")]
    ks = [float(x) for x in args.ks.split(",")]
    vs = [float(x) for x in args.vsat.split(",")]

    xi_all, sr = O.sf.read(str(O.DATA / "tsa15_test_input.wav"), dtype="float32")
    yi_all, _ = O.sf.read(str(O.DATA / "tsa15_test_target.wav"), dtype="float32")
    assert sr == O.SR
    seg_data = [(nm, *O.load_seg(t0, args.seg_len, xi_all, yi_all)) for nm, t0 in SEGMENTS]

    print(f"[env] DEVICE={DEVICE} seeds={len(seeds)} ks={ks} vsat={vs}", flush=True)
    out = {"mc": args.mc, "ks": ks, "vsat": vs, "rows": {}}

    for sp in seeds:
        name = sp.stem
        nam, sd = load_model(str(sp))
        # sanity 硬门槛（每种子都过一遍，防止结构复刻在某类模型上失效）
        xt0, yt0 = seg_data[0][1], seg_data[0][2]
        d0 = DevSim(sd, torch.Generator(device=DEVICE).manual_seed(0), sanity=True)
        with torch.no_grad():
            pi = forward(d0, sd, nam, xt0)
        if pi.ndim == 2:
            pi = pi[0]
        oo, _ = O.official_path(str(sp), xt0, yt0)
        n = min(pi.shape[0], oo.shape[0])
        maxd = float((pi[-n:] - oo[-n:]).abs().max())
        if maxd > 1e-5:
            print(f"[{name}] sanity FAIL max|Δ|={maxd:.2e} — 跳过", flush=True)
            out["rows"][name] = {"sanity_fail": maxd}
            continue

        row = {"sanity": maxd, "grid": {}}
        best = (1e9, None, None)
        for k in ks:
            sd2, nam2 = rebalance(sd, nam, k)
            for v in vs:
                d = device_nrms(nam2, sd2, seg_data, args.mc, v)
                row["grid"][f"k{int(k)}_v{int(v)}"] = d
                if d < best[0]:
                    best = (d, k, v)
        row["best"] = {"device": best[0], "k": best[1], "v_sat": best[2]}
        row["free"] = row["grid"]["k1_v16"]            # 什么都不加价的基线
        row["k_only"] = min(row["grid"][f"k{int(k)}_v16"] for k in ks)   # 只做免费 k
        out["rows"][name] = row
        print(f"[{name}] k1v16={row['free']:.4f}  k-only={row['k_only']:.4f}  "
              f"best={best[0]:.4f} @k={int(best[1])},V={int(best[2])}", flush=True)

    ok = [r for r in out["rows"].values() if "best" in r]
    if ok:
        for key, lab in (("free", "免费基线(k1,V16)"), ("k_only", "仅免费 k"),
                         ("best", "k+V 全用上")):
            v = [r["best"]["device"] if key == "best" else r[key] for r in ok]
            print(f"[summary] {lab:18s} n={len(v)} median={np.median(v):.4f} "
                  f"mean={np.mean(v):.4f} min={min(v):.4f} max={max(v):.4f}", flush=True)
        out["summary"] = {
            lab: {"median": float(np.median([r["best"]["device"] if key == "best" else r[key]
                                            for r in ok])),
                  "min": float(min(r["best"]["device"] if key == "best" else r[key] for r in ok)),
                  "max": float(max(r["best"]["device"] if key == "best" else r[key] for r in ok))}
            for key, lab in (("free", "free_k1v16"), ("k_only", "k_only"), ("best", "best_kv"))}

    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print("[done] " + args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
