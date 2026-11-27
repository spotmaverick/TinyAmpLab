"""P2：误差源留一消融 + 双口径对照 + 逐层饱和归因

## 为什么做这个

上一轮 `p1_vsat_sweep.py` 扫 V_SAT 时暴露两件事：

1. **种子间方差随工作点骤降**：V_SAT=16 时 5 种子 CV=30.5%（max/min 2.26x），
   到 V_SAT>=96 降到 CV=6.8%（1.17x）。最好种子只降 22%（0.1041->0.0808），
   最差种子降 60%（0.2357->0.0949）。**「38.7% 器件方差」几乎全是饱和贡献的**，
   这动摇了 P0（训练侧正则）的必要性。
2. **s11 在 V_SAT>=64 后反而变差**（0.0908->0.0949）：说明高 V_SAT 时主角换了，
   从 tanh 饱和切换到失配/失调/噪声底噪。

但比这两条更要紧的是**口径矛盾**：`analog_sim_generic.P` 的实际默认值
（BITS=6, BETA_H3=0.002, SIGMA_MISMATCH=0.01, SIGMA_OFFSET=0.005, SNR_DB=70）
与 `HANDOVER_20261126.md` 记录的默认档位
（BITS=8, BETA_H3=0.3, SIGMA_MISMATCH=0.02, SIGMA_OFFSET=0.001, SNR_DB=60）
严重不一致。**β 差 150 倍**，而 β 正是「Gilbert cell 三次失真是否决定性」的关键。
上轮 V_SAT 扫描继承的是代码口径（β=0.002），却拿文档口径的 A2 基准 0.6076 对比。
本脚本把两套口径都跑，用同一批权重、同一份音频、同一组 MC 种子，得到可直接相减的差值。

## 三个模式

- `ablate`：留一消融。基准 + 逐个关闭 6 个误差源（量化/失配/漂移/三次失真/失调/噪声）
  + 全关（数字模型）。每个配置 × 5 种子 × MC 次。
- `var`：逐层饱和归因。用 _gilbert 猴补丁记录每层的 pre-tanh 幅度 RMS、absmax、
  超 V_SAT 比例，解释「为什么 s31/s11 在最差工作点退化更重」。
- `all`：两者都跑。

## 口径定义

| 名称 | BITS | SIGMA_MISMATCH | DRIFT | BETA_H3 | SIGMA_OFFSET | SNR_DB |
|---|---:|---:|---:|---:|---:|---:|
| code（代码当前默认） | 6 | 0.01 | 0.03 | 0.002 | 0.005 | 70 |
| doc（HANDOVER 记录）  | 8 | 0.02 | 0.03 | 0.3   | 0.001 | 60 |

V_SAT 统一 16.0（即历史报数的工作点）。

## 用法

  python p2_ablate.py --mode all --mc 8
  python p2_ablate.py --mode ablate --seeds s7 s31 --mc 2      # 快验
  P2_REQUIRE_ACCEL=1 python p2_ablate.py --mode all             # 强制加速卡

## 注意

本脚本的前向直接 import `p1_vsat_sweep`，**不复制一份**——保证与产出历史数字的
forward 完全同源。改前向只改那一处。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# 先 import p1_vsat_sweep，由它完成 sys.path 布置（ROOT / mirrors / pylibs / stubs），
# 这样后续 import torch/numpy 才能拿到正确版本。顺序不能动。
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import p1_vsat_sweep as V                                       # noqa: E402

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
import soundfile as sf                                          # noqa: E402
import analog_sim_generic as A                                  # noqa: E402
from analog_sim_generic import DevSim                           # noqa: E402

DEV = V.DEV
_REQ = os.environ.get("P2_REQUIRE_ACCEL", "0") == "1"
if _REQ and DEV != "cuda":
    raise SystemExit("P2_REQUIRE_ACCEL=1 但无可用加速卡，拒绝在 CPU 上运行")

SEEDS = V.SEEDS
DATA = V.DATA
OUT = V.OUT
OUT.mkdir(parents=True, exist_ok=True)

V_SAT = 16.0

# 两套口径。差异逐项列出，不用继承，避免「以为对齐了其实没有」。
DIALECTS = {
    "code": dict(BITS=6, SIGMA_MISMATCH=0.01, DRIFT=0.03,
                 BETA_H3=0.002, SIGMA_OFFSET=0.005, SNR_DB=70.0, V_SAT=V_SAT),
    "doc":  dict(BITS=8, SIGMA_MISMATCH=0.02, DRIFT=0.03,
                 BETA_H3=0.3,   SIGMA_OFFSET=0.001, SNR_DB=60.0, V_SAT=V_SAT),
}

# 留一消融：把某一项关掉时覆盖成什么
#
# ⚠ 语义陷阱（冒烟实测发现）：`BETA_H3=0` 在 `_gilbert` 里走的是
#     `return prod.sum(dim=2)` —— **直接绕过 tanh**，等于同时关掉
#     「tanh 削顶」和「三次项」。所以不能把这个 case 叫 "关三次失真"，
#     它实际是「关整套 Gilbert 非线性」。冒烟里它降幅最大（0.1026->0.0771）
#     证明菜单里真正主导的是 tanh 饱和，不是 β 的取值。
#     要在现有实现下分离两者，必须做单变量 β 扫描（见 sweep 模式）。
OFF = {
    "QUANT":    dict(BITS=32),
    "MISMATCH": dict(SIGMA_MISMATCH=0.0),
    "DRIFT":    dict(DRIFT=0.0),
    "NONLIN":   dict(BETA_H3=0.0),   # 注意：关的是 tanh+β 整体，不是只关 β
    "OFFSET":   dict(SIGMA_OFFSET=0.0),
    "NOISE":    dict(SNR_DB=999.0),
}
OFF_CN = {"QUANT": "量化", "MISMATCH": "权重失配", "DRIFT": "温漂",
          "NONLIN": "Gilbert非线性(tanh+β)", "OFFSET": "差分对失调", "NOISE": "热噪声"}

# 单变量扫描：回答「口径矛盾里到底是哪个参数在起作用」
# 两套口径同时改了 6 个参数，直接相减无法归因。必须逐参数扫。
SWEEP = {
    "BETA_H3": [0.002, 0.01, 0.05, 0.1, 0.3],
    "BITS":    [4, 5, 6, 7, 8],
}
ALL_OFF = dict(BITS=32, SIGMA_MISMATCH=0.0, DRIFT=0.0,
               BETA_H3=0.0, SIGMA_OFFSET=0.0, SNR_DB=999.0, V_SAT=1e9)


def set_p(over):
    """把 A.P 设成 over 覆盖后的状态。DevSim 在构造时读 P，所以必须先设再建。"""
    A.P.clear()
    A.P.update(over)


def one_run(sd, cfg, xt, yt, over, mc_seed):
    set_p(over)
    try:
        dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(mc_seed))
        with torch.no_grad():
            pred = V.run_forward(dev, cfg, xt)
        return V.nrms_of(pred, yt)
    finally:
        A.P.clear()
        A.P.update(base_p)


def agg(vals):
    """MC 聚合用 RMS，与 p1_run.py / p1_vsat_sweep.py 同口径（算术平均会偏 14%）。"""
    return float(np.sqrt((np.array(vals, dtype=np.float64) ** 2).mean()))


# ---------------------------------------------------------------- ablate

def run_ablate(cfg, sd, xt, yt, mc, seed_name, res):
    for dname, dpar in DIALECTS.items():
        base = dict(dpar)
        # 基准 + 逐个留一 + 全关
        cases = [("BASE", {})]
        cases += [(k, v) for k, v in OFF.items()]
        cases += [("ALL_OFF", ALL_OFF)]

        slot = res.setdefault("ablate", {}).setdefault(dname, {})
        for cname, over in cases:
            p = dict(base)
            p.update(over)
            vals = [one_run(sd, cfg, xt, yt, p, 2000 + i) for i in range(mc)]
            m = agg(vals)
            slot.setdefault(cname, {})[seed_name] = {"nrms": m, "mc_vals": vals}
            print(f"  [{dname:4}] {seed_name:4} {cname:9} NRMS={m:.4f}  "
                  f"({min(vals):.4f}~{max(vals):.4f})", flush=True)


def summarize_ablate(res):
    """对每个口径算：留一后 NRMS 相对基准降了多少 -> 该项贡献占比。"""
    out = {}
    for dname, cases in res.get("ablate", {}).items():
        base = cases.get("BASE", {})
        if not base:
            continue
        seeds = sorted(base)
        bmed = float(np.median([base[s]["nrms"] for s in seeds]))
        rows = {}
        for cname, per_seed in cases.items():
            m = float(np.median([per_seed[s]["nrms"] for s in seeds]))
            # 贡献度：关掉此项后误差还剩多少（剩余方差比），以及绝对降幅
            rows[cname] = {
                "median": m,
                "drop_vs_base": bmed - m,
                "residual_frac": (m / bmed) ** 2 if bmed else None,
                "per_seed": {s: per_seed[s]["nrms"] for s in seeds},
            }
        # 各源按降幅排序，给出"关掉它误差残留多少"
        ranked = sorted([(k, v["drop_vs_base"]) for k, v in rows.items()
                         if k not in ("BASE", "ALL_OFF")],
                        key=lambda x: -x[1])
        out[dname] = {"base_median": bmed, "cases": rows,
                      "ranking": [{"source": k, "drop": d} for k, d in ranked]}
    return out


# ---------------------------------------------------------------- var

def diag_forward(dev, cfg, xt):
    """跑一次前向，同时记录每次 _gilbert 的 pre-tanh 统计 + head 累加和幅度。

    调用顺序（可预测，用于定位）：
      idx 0                          -> _rechannel (mac1x1)
      idx 1+li*8 .. 1+li*8+7         -> layer li: [conv tap0..tap5, input_mixer, layer1x1]
      最后 1 个                       -> _head_rechannel

    head 摆幅必须单独测：`run_forward` 里 head 累加是**普通加法**（`h = h + t`），
    不经过 _gilbert，所以不能从 _gilbert 记录里取。这里包装 `dev.conv`，
    记录最后一次调用（head_rechannel）的**输入**幅度——那正是 10 个 term 的累加和，
    也就是 P-1 关心的 `h_末`（历史值 RMS 17.5~22.9）。
    """
    K = cfg["layers_configs"][0]["kernel_size"]
    nL = len(cfg["layers_configs"][0]["dilations"])
    recs = []
    conv_inputs = []
    orig_g = dev._gilbert
    orig_c = dev.conv

    def wrapped_g(prod):
        with torch.no_grad():
            p = prod.detach()
            rms = float(p.pow(2).mean().sqrt())
            amax = float(p.abs().amax())
            vs = dev.p["V_SAT"]
            if vs < 1e8:
                sat = float((p.abs() > vs).float().mean())
                over = rms / vs
            else:
                sat, over = 0.0, 0.0
        recs.append({"rms": rms, "absmax": amax, "sat_frac": sat, "rms_over_vsat": over})
        return orig_g(prod)

    def wrapped_c(x, w, bias, dilation, groups):
        with torch.no_grad():
            xx = x.detach()
            conv_inputs.append({
                "rms": float(xx.pow(2).mean().sqrt()),
                "absmax": float(xx.abs().amax()),
            })
        return orig_c(x, w, bias, dilation, groups)

    dev._gilbert = wrapped_g
    dev.conv = wrapped_c
    with torch.no_grad():
        pred = V.run_forward(dev, cfg, xt)
    dev._gilbert = orig_g
    dev.conv = orig_c

    # 定位：按 K 与层数重排（conv 的 K 个 tap 连续，但 mixer/layer1x1 是 K=1）
    layers = []
    i = 1
    for li in range(nL):
        conv_taps = recs[i:i + K]
        i += K
        mixer = recs[i]; i += 1
        l1 = recs[i]; i += 1
        if conv_taps:
            layers.append({
                "layer": li,
                "conv_rms_over_vsat": max(t["rms_over_vsat"] for t in conv_taps),
                "conv_absmax_over_vsat": max(t["absmax"] / dev.p["V_SAT"] for t in conv_taps),
                "conv_sat_frac": max(t["sat_frac"] for t in conv_taps),
                "mixer_rms_over_vsat": mixer["rms_over_vsat"],
                "layer1x1_rms_over_vsat": l1["rms_over_vsat"],
                "layer1x1_sat_frac": l1["sat_frac"],
            })
    # 最后一次 conv 调用就是 head_rechannel，其输入 = 10 个 term 的累加和
    h = conv_inputs[-1] if conv_inputs else {"rms": 0.0, "absmax": 0.0}
    head = {"rms": h["rms"], "absmax": h["absmax"]}
    return pred, layers, head, len(recs)


def run_var(cfg, sd, xt, yt, seed_name, res):
    """逐层饱和深度诊断 + head 累加摆幅。"""
    slot = res.setdefault("var", {})
    for dname, dpar in DIALECTS.items():
        base = dict(dpar)
        set_p(base)
        try:
            dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(2000))
            pred, layers, head, ncall = diag_forward(dev, cfg, xt)
            nr = V.nrms_of(pred, yt)
            vs = dev.p["V_SAT"]
        finally:
            A.P.clear()
            A.P.update(base_p)
        slot.setdefault(dname, {})[seed_name] = {
            "nrms": nr,
            "layers": layers,
            "head_sum_rms": head["rms"],            # 10 term 累加和 RMS（P-1 的 h_末）
            "head_sum_rms_over_vsat": head["rms"] / vs if vs else None,
            "head_sum_absmax": head["absmax"],
            "n_gilbert_calls": ncall,
        }
        worst = max(layers, key=lambda r: r["conv_rms_over_vsat"]) if layers else None
        print(f"  [var {dname:4}] {seed_name:4} NRMS={nr:.4f}  "
              f"最深饱和层=#{worst['layer'] if worst else '?'} "
              f"(rms/V_SAT={worst['conv_rms_over_vsat']:.2f}, "
              f"sat_frac={worst['conv_sat_frac']:.4f})  "
              f"head累加和 RMS={head['rms']:.2f} ({head['rms']/vs:.2f}xV_SAT)",
              flush=True)


# ---------------------------------------------------------------- sweep

def run_sweep(cfg, sd, xt, yt, mc, seed_name, res):
    """单变量扫描：在 code 口径基础上，逐参数单独改成 doc 值，看各自贡献多少。

    这是解开「口径矛盾」的唯一正确做法。两套口径同时改 6 个参数，
    直接相减得到的是 6 个变量的混杂效应，无法判断是 β 还是 BITS 在起作用。
    """
    slot = res.setdefault("sweep", {})
    code = dict(DIALECTS["code"])
    doc = dict(DIALECTS["doc"])

    # 基线（code）、终点（doc），以及 6 条单变量路径
    cases = [("code_all", {}), ("doc_all", doc)]
    for k in ("BITS", "SIGMA_MISMATCH", "DRIFT", "BETA_H3", "SIGMA_OFFSET", "SNR_DB"):
        cases.append((f"only_{k}", {k: doc[k]}))

    for label, over in cases:
        p = dict(code)
        p.update(over)
        vals = [one_run(sd, cfg, xt, yt, p, 3000 + i) for i in range(mc)]
        m = agg(vals)
        slot.setdefault(label, {})[seed_name] = {"nrms": m, "mc_vals": vals}
        print(f"  [swp] {seed_name:4} {label:17} NRMS={m:.4f}", flush=True)


def summarize_sweep(res):
    """单变量贡献：code 基线到 doc 全量之间的差值，有多少能被单个参数解释。"""
    sw = res.get("sweep", {})
    if not sw:
        return {}
    out = {}
    for label, per_seed in sw.items():
        seeds = sorted(per_seed)
        out[label] = float(np.median([per_seed[s]["nrms"] for s in seeds]))
    if "code_all" not in out or "doc_all" not in out:
        return out
    gap = out["doc_all"] - out["code_all"]
    contrib = {}
    for label, v in out.items():
        if label.endswith("_all"):
            continue
        contrib[label] = {"delta": v - out["code_all"],
                          "share_of_gap": (v - out["code_all"]) / gap if gap else None}
    return {"medians": out, "code_to_doc_gap": gap, "single_param": contrib}


# ---------------------------------------------------------------- main

base_p = None   # 原始 A.P 快照，供 one_run/diag 恢复


def main():
    global base_p
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ablate", "var", "sweep", "all"], default="all")
    ap.add_argument("--seeds", nargs="*", default=SEEDS)
    ap.add_argument("--mc", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    base_p = dict(A.P)
    print(f"[device] {DEV}  torch={torch.__version__}  "
          f"hip={getattr(torch.version, 'hip', None)}", flush=True)

    xi, _ = sf.read(str(DATA / "tsa15_test_input.wav"), dtype="float32")
    yi, _ = sf.read(str(DATA / "tsa15_test_target.wav"), dtype="float32")
    n = int(48000 * args.seconds)
    xt = torch.from_numpy(xi[:n]).to(DEV)
    yt = torch.from_numpy(yi[:n]).to(DEV)
    print(f"[data] {args.seconds}s = {n} samples, seeds={args.seeds}, mc={args.mc}",
          flush=True)

    res = {"meta": {"dev": DEV, "torch": torch.__version__,
                    "hip": str(getattr(torch.version, "hip", None)),
                    "mc": args.mc, "seconds": args.seconds,
                    "v_sat": V_SAT, "dialects": DIALECTS, "off": OFF,
                    "sweep": SWEEP, "tag": args.tag,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S")}}
    t0 = time.time()

    for s in args.seeds:
        path = V.nam_path(s)
        cfg, sd = V.load_nam(path)
        print(f"\n=== {s}  ({path})", flush=True)
        if args.mode in ("ablate", "all"):
            run_ablate(cfg, sd, xt, yt, args.mc, s, res)
        if args.mode in ("sweep", "all"):
            run_sweep(cfg, sd, xt, yt, args.mc, s, res)
        if args.mode in ("var", "all"):
            run_var(cfg, sd, xt, yt, s, res)

    if "ablate" in res:
        res["ablate_summary"] = summarize_ablate(res)
    if "sweep" in res:
        res["sweep_summary"] = summarize_sweep(res)

    tag = args.tag or time.strftime("%m%d_%H%M")
    outp = OUT / f"p2_ablate_{tag}.json"
    json.dump(res, open(outp, "w"), indent=1, ensure_ascii=False)
    print(f"\n[saved] {outp}  总耗时 {time.time()-t0:.0f}s")

    # 控制台摘要：两个口径下谁最该打
    for dname, s2 in res.get("ablate_summary", {}).items():
        print(f"\n--- 口径 {dname}: 基准中位 NRMS={s2['base_median']:.4f} ---")
        print(f"{'关掉这一项':<20}{'中位NRMS':>10}{'降幅':>10}{'剩余方差比':>12}")
        for cname, row in s2["cases"].items():
            if cname == "BASE":
                continue
            lbl = "全关(数字)" if cname == "ALL_OFF" else OFF_CN.get(cname, cname)
            print(f"{lbl:<20}{row['median']:>10.4f}{row['drop_vs_base']:>10.4f}"
                  f"{row['residual_frac']:>12.3f}")
        print("按降幅排序：" + " > ".join(
            f"{OFF_CN.get(r['source'], r['source'])}({r['drop']:.4f})"
            for r in s2["ranking"]))

    sw = res.get("sweep_summary") or {}
    if sw.get("single_param"):
        print(f"\n--- 口径矛盾归因（code={sw['medians']['code_all']:.4f} "
              f"-> doc={sw['medians']['doc_all']:.4f}，gap={sw['code_to_doc_gap']:+.4f}）---")
        print(f"{'单改这一项为 doc 值':<24}{'中位NRMS':>10}{'相对code':>10}{'占总gap':>10}")
        for label, c in sorted(sw["single_param"].items(),
                               key=lambda kv: -abs(kv[1]["delta"])):
            share = c["share_of_gap"]
            pct = f"{100*share:>9.0f}%" if share is not None else "        -"
            print(f"{label:<24}{sw['medians'][label]:>10.4f}"
                  f"{c['delta']:>+10.4f}{pct}")


if __name__ == "__main__":
    main()
