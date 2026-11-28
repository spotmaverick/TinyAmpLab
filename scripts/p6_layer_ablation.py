"""P6-B2：固定候选摆幅规格在 13 个种子上的验证 —— 把"分层预算"变成可下单的规格

## 为什么不能直接用 P6-B 的结论

P6-B（`p6_layer_budget.py`）在 s7 上给出"只要 5 层高摆幅就拿到 90% 收益"，
在 s71 上给出"3 层拿到 80%"。但这两个是**单种子**结论，而且两个种子的最优集还不一样：

    s7 : {head, L8, L6, L5, L9}
    s71: {head, L8, L9}

硅片上只能定一套规格。必须回答的是：**一套固定规格，在 13 个种子上分别能拿到多少**，
而不是"某个种子最好的那套能拿多少"。

## 候选规格（都是"哪几个 scope 给 V=96，其余 16"）

    S0  none          全 16（不加钱的基线）
    S1  head          只给最后一级 head_rechannel
    S2  L8,L9         只给两个中间层（P6-B 里 s71 的次大项）
    S3  head+L8,L9    组合
    S4  head+L5,L6,L8,L9   s7 的 90% 集
    S5  all           全部 scope（P6-D2 的天花板，作为参考）

每个种子的 k 取 `p6_calib_all.json` 里该种子的最优 k（k 是免费的，直接用它）。

## 成本

6 个规格 × 3 段 × mc 次 = 72 次前向 / 种子，13 个种子约 15~20 分钟。

## 用法

    $PY p6_layer_ablation.py --seeds all --calib out/p6_calib_all.json --mc 4 \
        --out out/p6b2_specs.json
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


def all_scopes(nam, sd):
    from official_analog_sim import parse_arrays
    out = []
    for ai, _ in enumerate(parse_arrays(nam["config"])):
        out.append(f"A{ai}.rc")
        out += [f"A{ai}.L{li}" for li in sorted({int(k.split(".")[3]) for k in sd
                                                if k.startswith(f"_layer_arrays.{ai}._layers.")})]
        out.append(f"A{ai}.head")
    return out


def specs_for(scopes):
    """候选规格：键名 → 需要给 V_HIGH 的 scope 列表（空列表 = 全 16）。

    scope 形如 `A0.L8` / `A1.head` / `A0.rc`，取最后一段做名字匹配，
    这样单 array（目标拓扑）和双 array（官方）用同一套规格名都能对上。
    """
    def pick(*names):
        want = set(names)
        return [s for s in scopes if s.split(".")[-1] in want]

    return {
        "S0_none": [],
        "S1_head": pick("head"),
        "S2_L8L9": pick("L8", "L9"),
        "S3_head_L8L9": pick("head", "L8", "L9"),
        "S4_head_L5L6L8L9": pick("head", "L5", "L6", "L8", "L9"),
        "S5_all": list(scopes),
    }


def device_nrms(nam, sd, seg_data, mc, vsat_map, v_high):
    vals = []
    for _nm, xt, yt in seg_data:
        dl = []
        for i in range(mc):
            dv = DevSim(sd, torch.Generator(device=DEVICE).manual_seed(1000 + i))
            dv.p["V_SAT"] = 16.0
            dv.vsat_map = {s: v_high for s in vsat_map}
            with torch.no_grad():
                pr = forward(dv, sd, nam, xt)
            dl.append(esr_nrms(pr, yt)[1])
        vals.append(float(np.sqrt(np.mean(np.square(dl)))))
    return float(np.median(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="all")
    ap.add_argument("--model-dir", default="models/seeds")
    ap.add_argument("--calib", default="out/p6_calib_all.json")
    ap.add_argument("--mc", type=int, default=4)
    ap.add_argument("--v-high", type=float, default=96.0)
    ap.add_argument("--seg-len", type=float, default=5.0)
    ap.add_argument("--out", default="out/p6b2_specs.json")
    args = ap.parse_args()

    best_k = {}
    if Path(args.calib).is_file():
        cal = json.load(open(args.calib))
        best_k = {s: r["best"]["k"] for s, r in cal["rows"].items() if "best" in r}
        print(f"[calib] 读入 {len(best_k)} 个种子的最优 k", flush=True)

    if args.seeds == "all":
        seeds = sorted(Path(args.model_dir).glob("*.nam"))
    else:
        seeds = [Path(args.model_dir) / f"{s}.nam" for s in args.seeds.split(",")]

    xi_all, sr = O.sf.read(str(O.DATA / "tsa15_test_input.wav"), dtype="float32")
    yi_all, _ = O.sf.read(str(O.DATA / "tsa15_test_target.wav"), dtype="float32")
    assert sr == O.SR
    seg_data = [(nm, *O.load_seg(t0, args.seg_len, xi_all, yi_all)) for nm, t0 in SEGMENTS]

    out = {"mc": args.mc, "v_high": args.v_high, "rows": {}}
    for sp in seeds:
        name = sp.stem
        nam, sd = load_model(str(sp))
        scopes = all_scopes(nam, sd)
        specs = specs_for(scopes)
        k = float(best_k.get(name, 32.0))
        sd2, nam2 = rebalance(sd, nam, k)
        row = {"k": k, "scopes": specs, "values": {}}
        for sname, sel in specs.items():
            d = device_nrms(nam2, sd2, seg_data, args.mc, sel, args.v_high)
            row["values"][sname] = d
        base = row["values"]["S0_none"]
        row["recovery"] = {k2: (base - v) / max(base - row["values"]["S5_all"], 1e-9)
                           for k2, v in row["values"].items()}
        out["rows"][name] = row
        print(f"[{name}] k={int(k)} " + "  ".join(
            f"{s}={v:.4f}({row['recovery'][s]*100:3.0f}%)" for s, v in row["values"].items()),
            flush=True)

    ok = [r for r in out["rows"].values() if r["values"]]
    if ok:
        print("\n[summary] 跨种子中位（括号内为该规格占全层收益的比例）", flush=True)
        for sname in specs_of(out):
            v = [r["values"][sname] for r in ok]
            rec = [r["recovery"][sname] for r in ok]
            print(f"  {sname:20s} median={np.median(v):.4f} max={max(v):.4f} "
                  f"回收率median={np.median(rec)*100:5.1f}% min={min(rec)*100:5.1f}%", flush=True)

    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print("[done] " + args.out, flush=True)
    return 0


def specs_of(out):
    for r in out["rows"].values():
        if r["values"]:
            return list(r["values"].keys())
    return []


if __name__ == "__main__":
    raise SystemExit(main())
