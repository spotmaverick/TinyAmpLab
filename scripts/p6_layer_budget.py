"""P6-B：分层 V_SAT 预算分配 —— 到底是哪几层需要高摆幅？

## 要回答的问题

P5-A 已经证明：目标拓扑要把 V_SAT 掉到 64~96 才追平官方，而官方 16~48 就够。
但"全层 V_SAT=96"这个结论对硅片是不可接受的 —— 摆幅每翻一倍，Gilbert cell 的
尾电流就要翻倍，功耗和面积按平方级恶化（差分对的线性区宽度 ∝ I_TAIL/gm）。

真正要下单的规格是**分层摆幅**：

    head 那一级给 64，其余 13 层给 16 —— 能不能拿到同样的精度？

P5-B 的旁路归因已经暗示答案是"能"：s71 的 head 一项就占 Gilbert 总退化的 74%。
但那只是旁路（把非线性完全关掉）的上界，不等于"抬高摆幅"能拿到的实际收益。
本脚本把这件事做实。

## 三个量

1. **单层边际（marginal）**：只把该 scope 抬到 V_HIGH，其余保持 V_BASE，测改善。
   这是该层"欠摆幅"的绝对量，也是排序依据。
2. **贪心累积（cumulative）**：按边际从大到小依次抬到 V_HIGH，测每一步的 device NRMS。
   得到一条"花钱 vs 精度"的阶梯曲线。
3. **最小充分集（m_min）**：达到全层抬满收益的 90% / 95% 所需的最少层数。

## 两种规格 regime

    regime=pure   k=1（不做 head 增益重平衡），纯粹问"硅上该给哪层高摆幅"
    regime=kbal   head 先按 P5-C 重平衡（零成本），再看**剩下的**缺口在哪层

两个都要，因为它们是两条不同的工程路线：
  pure   = 多花功耗，不动权重
  kbal   = 不动功耗，改 head 这一级的增益分配（但需要输出级能放大 k 倍）

## scope 粒度

    A{ai}.L{li}   该层的 conv + input_mixer + layer1x1（P5 的层粒度）
    A{ai}.head    head_rechannel
    A{ai}.rc      rechannel

## 用法

    $PY p6_layer_budget.py --nam models/seeds/s7.nam --mc 4 --regime pure --out out/p6b_s7_pure.json
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
from official_analog_sim import DevSim, forward, esr_nrms, parse_arrays  # noqa: E402
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


def scopes_for(nam, sd):
    """层粒度 scope 列表，顺序从输入到输出（便于人读）。"""
    arrays = parse_arrays(nam["config"])
    out = []
    for ai, _arr in enumerate(arrays):
        layers = sorted({int(k.split(".")[3]) for k in sd
                         if k.startswith(f"_layer_arrays.{ai}._layers.")})
        out.append(f"A{ai}.rc")
        out += [f"A{ai}.L{li}" for li in layers]
        out.append(f"A{ai}.head")
    return out


def eval_cfg(nam, sd, seg_data, mc, vsat_map, base_vsat):
    """在给定分层摆幅下测 device NRMS（3 段取中位）。MC 种子固定 → 配对可比。"""
    vals = []
    for _nm, xt, yt in seg_data:
        dl = []
        for i in range(mc):
            dv = DevSim(sd, torch.Generator(device=DEVICE).manual_seed(1000 + i))
            dv.p["V_SAT"] = float(base_vsat)
            dv.vsat_map = dict(vsat_map)
            with torch.no_grad():
                pr = forward(dv, sd, nam, xt)
            dl.append(esr_nrms(pr, yt)[1])
        vals.append(float(np.sqrt(np.mean(np.square(dl)))))
    return float(np.median(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nam", required=True)
    ap.add_argument("--mc", type=int, default=4)
    ap.add_argument("--regime", choices=["pure", "kbal"], default="pure")
    ap.add_argument("--k", type=float, default=8.0, help="kbal regime 下的 head 增益除子")
    ap.add_argument("--seg-len", type=float, default=5.0)
    ap.add_argument("--v-base", type=float, default=16.0)
    ap.add_argument("--v-high", type=float, default=96.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    nam, sd = load_model(args.nam)
    print(f"[env] DEVICE={DEVICE} cuda={torch.cuda.is_available()}", flush=True)
    print(f"[cfg] params={sum(v.numel() for v in sd.values())} regime={args.regime}", flush=True)

    if args.regime == "kbal":
        sd, nam = rebalance(sd, nam, args.k)
        print(f"[kbal] head_rechannel /{args.k:g}, head_scale *{args.k:g}", flush=True)

    # ---- sanity 硬门槛（结构复刻错则拒绝出数）
    xi_all, sr = O.sf.read(str(O.DATA / "tsa15_test_input.wav"), dtype="float32")
    yi_all, _ = O.sf.read(str(O.DATA / "tsa15_test_target.wav"), dtype="float32")
    assert sr == O.SR
    xt0, yt0 = O.load_seg(SEGMENTS[0][1], args.seg_len, xi_all, yi_all)
    d0 = DevSim(sd, torch.Generator(device=DEVICE).manual_seed(0), sanity=True)
    with torch.no_grad():
        pi = forward(d0, sd, nam, xt0)
    if pi.ndim == 2:
        pi = pi[0]
    oo, _ = O.official_path(args.nam, xt0, yt0)
    n = min(pi.shape[0], oo.shape[0])
    maxd = float((pi[-n:] - oo[-n:]).abs().max())
    print(f"[sanity] max|Δ|={maxd:.3e}", flush=True)
    if maxd > 1e-5:
        print("[sanity] FAIL — 结构复刻有误，拒绝出数。", flush=True)
        return 2

    seg_data = [(nm, *O.load_seg(t0, args.seg_len, xi_all, yi_all)) for nm, t0 in SEGMENTS]
    scopes = scopes_for(nam, sd)
    res = {"nam": args.nam, "regime": args.regime, "mc": args.mc,
           "v_base": args.v_base, "v_high": args.v_high,
           "sanity_max_abs_diff": maxd, "scopes": scopes}

    # ---- 1. 基线（全层 V_BASE）
    base = eval_cfg(nam, sd, seg_data, args.mc, {}, args.v_base)
    res["baseline"] = base
    print(f"[base] all V={args.v_base:g} → device={base:.4f}", flush=True)

    # ---- 2. 全层抬满（参考天花板）
    full = eval_cfg(nam, sd, seg_data, args.mc,
                    {sc: args.v_high for sc in scopes}, args.v_base)
    res["all_high"] = full
    total_gain = base - full
    res["total_gain"] = total_gain
    print(f"[ceiling] all V={args.v_high:g} → device={full:.4f}  "
          f"总收益={total_gain:+.4f}", flush=True)

    # ---- 3. 单层边际（只抬这一层）
    singles = []
    for sc in scopes:
        d = eval_cfg(nam, sd, seg_data, args.mc, {sc: args.v_high}, args.v_base)
        singles.append({"scope": sc, "device_median": d, "relief": base - d,
                        "share_of_total": (base - d) / total_gain if total_gain > 1e-9 else 0.0})
        print(f"  [single] {sc:12s} device={d:.4f}  relief={base-d:+.4f}  "
              f"占总量{(base-d)/total_gain*100 if total_gain>1e-9 else 0:5.1f}%", flush=True)
    singles.sort(key=lambda r: r["relief"], reverse=True)
    res["singles"] = singles

    # ---- 4. 贪心累积（按单层边际排序，依次抬满）
    cum = []
    chosen = {}
    for r in singles:
        chosen[r["scope"]] = args.v_high
        d = eval_cfg(nam, sd, seg_data, args.mc, chosen, args.v_base)
        frac = (base - d) / total_gain if total_gain > 1e-9 else 0.0
        cum.append({"n_high": len(chosen), "added": r["scope"],
                    "device_median": d, "frac_of_total": frac,
                    "set": sorted(chosen.keys())})
        print(f"  [cum] m={len(chosen):2d} +{r['scope']:12s} device={d:.4f}  "
              f"已达总量 {frac*100:5.1f}%", flush=True)
        if frac >= 0.999:
            break
    res["cumulative"] = cum

    # ---- 5. 最小充分集
    res["m_min"] = {}
    for thr in (0.80, 0.90, 0.95, 0.99):
        hit = next((c for c in cum if c["frac_of_total"] >= thr), None)
        res["m_min"][f"{int(thr*100)}pct"] = (
            {"n_high": hit["n_high"], "set": hit["set"], "device_median": hit["device_median"]}
            if hit else None)

    # ---- 6. 推荐集的摆幅降级曲线：能降到多少还够用
    rec = res["m_min"]["90pct"]
    if rec:
        curve = []
        for v in (16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 1e5):
            d = eval_cfg(nam, sd, seg_data, args.mc,
                         {sc: v for sc in rec["set"]}, args.v_base)
            curve.append({"v_high": v, "device_median": d})
            print(f"  [grade] rec_set @ V={v:>8.0f} device={d:.4f}", flush=True)
        res["rec_set_grade"] = curve

    out = Path(args.out) if args.out else Path(f"p6b_{Path(args.nam).stem}_{args.regime}.json")
    json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
    print("[done] " + str(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
