"""P5-A / P5-B：V_SAT 同扫 + 逐层 Gilbert 旁路归因

## 背景

`OFFICIAL_NAM_COMPARE_20261127.md` 的结论：数字精度目标拓扑领先，但器件精度官方反超
（standard 0.0900 / lite 0.0975 vs 种子中位 0.1637）。两个未决问题：

  P5-A  官方两档与目标拓扑在 **同一 V_SAT 网格**上各自的最优工作点在哪？
        统一 V_SAT=16 是否偏袒某一方？若给目标拓扑更高摆幅，差距能收窄多少？
  P5-B  目标拓扑的退化集中在哪些层？是否又是 head（P3 结论）还是分散在分组卷积？

## 设计要点

- **零训练**：全部复用已训权重，纯前向仿真。
- **配对比较**：同一 (model, seg, MC seed) 下改变 V_SAT 或旁路集合，权重与 MC 噪声完全一致，
  差值只反映被改的那一项。避免"两次独立采样相减"引入的 ±10% 噪声（P4 已踩过）。
- **V_SAT 网格含 1e5**（视作无穷），用于取"该模型的工作点饱和上限"。
- **旁路语义**：把该 scope 的 Gilbert 换成理想求和，等于关掉该层非线性（含 tanh 削顶，
  与 P2 的 BETA_H3=0 语义一致）。

## scope 命名

    A{ai}.L{li}.conv    第 ai 个 array 第 li 层的分组卷积
    A{ai}.L{li}.mix     input_mixer
    A{ai}.L{li}.l1x1    layer1x1（残差投影）
    A{ai}.L{li}.h1x1    head1x1（若启用）
    A{ai}.head          head_rechannel
    A{ai}.rc            rechannel（仅首层 array 有实际作用）

前缀匹配：`A0.L3` 覆盖该层全部三处；`A0` 覆盖整个 array0。

## 用法

    source ~/p26/bw2env.sh
    export PYTHONPATH=/public/home/spotmaverick/nam_proj:/public/home/spotmaverick/t29:$PYTHONPATH
    export OFFICIAL_SIM_DATA=/public/home/spotmaverick/nam_proj/data
    export NAM_SRC=/public/home/spotmaverick/nam_proj

    # P5-A：V_SAT 同扫（默认网格）
    $PY p5_vsat_layer.py --mode vsat --nam models/seeds/s7.nam --mc 4 --out out/p5a_s7.json
    # P5-B：逐层旁路归因
    $PY p5_vsat_layer.py --mode bypass --nam models/seeds/s7.nam --mc 4 --out out/p5b_s7.json

## 硬门槛

脚本启动即做一次 sanity（理想路径 vs 官方 PyTorch 前向），max|Δ| > 1e-5 直接拒绝出数。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_analog_sim as O                                  # noqa: E402
from official_analog_sim import DevSim, forward, esr_nrms, parse_arrays  # noqa: E402

DEVICE = O.DEVICE
DATA = O.DATA
SR = O.SR

SEGMENTS = [("seg0", 0.0), ("seg1", 8.0), ("seg2", 16.0)]
SEG_LEN = 5.0

VSAT_GRID = [16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 1e5]


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
    """枚举可旁路的 scope，用于 P5-B。粒度取"层"（合并 conv/mix/l1x1）。"""
    arrays = parse_arrays(nam["config"])
    out = []
    for ai, arr in enumerate(arrays):
        layers = sorted({int(k.split(".")[3]) for k in sd
                         if k.startswith(f"_layer_arrays.{ai}._layers.")})
        for li in layers:
            out.append(f"A{ai}.L{li}")
        out.append(f"A{ai}.head")
        out.append(f"A{ai}.rc")
    return out


def run_seg(nam, sd, xt, yt, mc, vsat=None, bypass=(), vsat_map=None, over=None):
    """同一 (seg, MC) 下跑一次，返回 (device NRMS, digital NRMS)。配对：MC 种子固定。"""
    dev_list, dig_list = [], []
    for i in range(mc):
        dg = DevSim(sd, torch.Generator(device=DEVICE).manual_seed(1000 + i), sanity=True)
        with torch.no_grad():
            pr = forward(dg, sd, nam, xt)
        dig_list.append(esr_nrms(pr, yt)[1])

        dv = DevSim(sd, torch.Generator(device=DEVICE).manual_seed(1000 + i), over=over)
        if vsat is not None:
            dv.p["V_SAT"] = float(vsat)
        if vsat_map:
            dv.vsat_map = dict(vsat_map)
        if bypass:
            dv.bypass = set(bypass)
        with torch.no_grad():
            pr = forward(dv, sd, nam, xt)
        dev_list.append(esr_nrms(pr, yt)[1])
    return (float(np.sqrt(np.mean(np.square(dev_list)))),
            float(np.sqrt(np.mean(np.square(dig_list)))))


def run_decomp(nam, sd, seg_data, mc):
    """误差源分解阶梯：逐项关掉非理想性，看每一项各自贡献多少。

    阶梯（每一步只关一项）：
      1 baseline           全部开启，V_SAT=16
      2 gilbert_off        关 Gilbert 非线性（tanh+β），保留量化/失配/漂移/噪声
      3 quant_off          + 关 6-bit 量化
      4 mismatch_drift_off + 关 权重失配与温漂
      5 pure_digital       + 关热噪声 → 应等于 digital 参考（自检）
    """
    arrays = parse_arrays(nam["config"])
    all_scopes = []
    for ai, arr in enumerate(arrays):
        layers = sorted({int(k.split(".")[3]) for k in sd
                         if k.startswith(f"_layer_arrays.{ai}._layers.")})
        all_scopes += [f"A{ai}.L{li}" for li in layers] + [f"A{ai}.head", f"A{ai}.rc"]

    steps = [
        ("baseline", dict(), dict()),
        ("gilbert_off", dict(bypass=tuple(all_scopes)), dict()),
        ("quant_off", dict(bypass=tuple(all_scopes)), dict(BITS=64)),
        ("mismatch_drift_off", dict(bypass=tuple(all_scopes)),
         dict(BITS=64, SIGMA_MISMATCH=0.0, DRIFT=0.0)),
        ("pure_digital", dict(bypass=tuple(all_scopes)),
         dict(BITS=64, SIGMA_MISMATCH=0.0, DRIFT=0.0, SNR_DB=999.0)),
    ]
    out = []
    for name, kw, over in steps:
        vals, digs = [], []
        for _nm, xt, yt in seg_data:
            d, g = run_seg(nam, sd, xt, yt, mc, over=over, **kw)
            vals.append(d)
            digs.append(g)
        med = float(np.median(vals))
        out.append({"step": name, "device_median": med,
                    "digital_median": float(np.median(digs))})
        print(f"  [decomp] {name:20s} device={med:.4f}", flush=True)
    # 逐项贡献（相邻两级之差）
    for i in range(len(out) - 1):
        out[i]["contribution"] = out[i]["device_median"] - out[i + 1]["device_median"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nam", required=True)
    ap.add_argument("--mode", choices=["vsat", "bypass", "both", "decomp"], default="both")
    ap.add_argument("--mc", type=int, default=4)
    ap.add_argument("--seg-len", type=float, default=SEG_LEN)
    ap.add_argument("--vsat", default=",".join(str(v) for v in VSAT_GRID))
    ap.add_argument("--head-vsat", action="store_true",
                    help="额外做 head 级 V_SAT 扫描（其余层固定 16）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    nam, sd = load_model(args.nam)
    arrays = parse_arrays(nam["config"])
    print(f"[env] DEVICE={DEVICE} cuda={torch.cuda.is_available()}", flush=True)
    print(f"[cfg] arrays={len(arrays)} params={sum(v.numel() for v in sd.values())}", flush=True)

    xi_all, sr = sf.read(str(DATA / "tsa15_test_input.wav"), dtype="float32")
    yi_all, _ = sf.read(str(DATA / "tsa15_test_target.wav"), dtype="float32")
    assert sr == SR

    # ---- sanity 硬门槛
    xt, yt = O.load_seg(SEGMENTS[0][1], args.seg_len, xi_all, yi_all)
    d0 = DevSim(sd, torch.Generator(device=DEVICE).manual_seed(0), sanity=True)
    with torch.no_grad():
        pi = forward(d0, sd, nam, xt)
    if pi.ndim == 2:
        pi = pi[0]
    oo, (_, _) = O.official_path(args.nam, xt, yt)
    n = min(pi.shape[0], oo.shape[0])
    maxd = float((pi[-n:] - oo[-n:]).abs().max())
    print(f"[sanity] max|Δ|={maxd:.3e}", flush=True)
    if maxd > 1e-5:
        print("[sanity] FAIL — 结构复刻有误，拒绝出数。", flush=True)
        return 2

    # 预算每段数据
    seg_data = [(nm, *O.load_seg(t0, args.seg_len, xi_all, yi_all)) for nm, t0 in SEGMENTS]

    res = {"nam": args.nam, "mc": args.mc, "dialect": O.DIALECT,
           "sanity_max_abs_diff": maxd, "segments": [s[0] for s in SEGMENTS]}

    # ---- 基线（V_SAT=16，全开）
    base = {}
    for nm, xt, yt in seg_data:
        d, g = run_seg(nam, sd, xt, yt, args.mc)
        base[nm] = {"dev": d, "dig": g}
    base_dev = float(np.median([v["dev"] for v in base.values()]))
    base_dig = float(np.median([v["dig"] for v in base.values()]))
    res["baseline"] = {"device_median": base_dev, "digital_median": base_dig,
                       "per_seg": base}
    print(f"[base] V_SAT=16 device={base_dev:.4f} digital={base_dig:.4f}", flush=True)

    # ---- P5-A：V_SAT 全层扫描
    if args.mode in ("vsat", "both"):
        grid = [float(v) for v in args.vsat.split(",")]
        rows = []
        for vs in grid:
            vals = []
            for nm, xt, yt in seg_data:
                d, _ = run_seg(nam, sd, xt, yt, args.mc, vsat=vs)
                vals.append(d)
            med = float(np.median(vals))
            rows.append({"v_sat": vs, "device_median": med,
                         "per_seg": vals,
                         "gain_vs_base": base_dev - med})
            print(f"  [vsat] {vs:>9.0f}  device={med:.4f}  Δvs16={base_dev-med:+.4f}", flush=True)
        res["vsat_sweep"] = rows
        best = min(rows, key=lambda r: r["device_median"])
        res["vsat_best"] = best
        print(f"[vsat] best V_SAT={best['v_sat']:.0f} device={best['device_median']:.4f}", flush=True)

    # ---- P5-A2：head 级 V_SAT（其余固定 16）
    if args.head_vsat and args.mode in ("vsat", "both"):
        heads = [f"A{ai}.head" for ai in range(len(arrays))]
        rows = []
        for vs in [16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 1e5]:
            vals = []
            for nm, xt, yt in seg_data:
                d, _ = run_seg(nam, sd, xt, yt, args.mc,
                               vsat_map={h: vs for h in heads})
                vals.append(d)
            med = float(np.median(vals))
            rows.append({"head_v_sat": vs, "device_median": med,
                         "gain_vs_base": base_dev - med})
            print(f"  [head-vsat] {vs:>9.0f}  device={med:.4f}  Δ={base_dev-med:+.4f}", flush=True)
        res["head_vsat_sweep"] = rows

    # ---- P5-B：逐层旁路归因
    if args.mode in ("bypass", "both"):
        scopes = scopes_for(nam, sd)
        rows = []
        for sc in scopes:
            vals = []
            for nm, xt, yt in seg_data:
                d, _ = run_seg(nam, sd, xt, yt, args.mc, bypass=(sc,))
                vals.append(d)
            med = float(np.median(vals))
            rows.append({"scope": sc, "device_median": med,
                         "drop": med - base_dev,          # 负 = 关掉它后变好 = 它贡献退化
                         "relief": base_dev - med})
            print(f"  [bypass] {sc:16s} device={med:.4f}  relief={base_dev-med:+.4f}", flush=True)
        res["bypass"] = sorted(rows, key=lambda r: r["relief"], reverse=True)
        # 全部旁路的上限
        allv = []
        for nm, xt, yt in seg_data:
            d, _ = run_seg(nam, sd, xt, yt, args.mc, bypass=tuple(scopes))
            allv.append(d)
        res["bypass_all"] = {"device_median": float(np.median(allv))}
        print(f"[bypass-all] device={np.median(allv):.4f} (下限参考)", flush=True)

    # ---- P5-C：误差源分解（关掉 Gilbert 后的地板是什么造成的）
    if args.mode == "decomp":
        res["decomp"] = run_decomp(nam, sd, seg_data, args.mc)
        out = Path(args.out) if args.out else Path("p5d_" + Path(args.nam).stem + ".json")
        json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
        print("[done] " + str(out), flush=True)
        return 0

    out = Path(args.out) if args.out else Path("p5_" + Path(args.nam).stem + ".json")
    json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
    print("[done] " + str(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
