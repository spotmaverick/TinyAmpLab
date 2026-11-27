"""P3：为什么 s11 饱和最浅却退化最重——逐算子注入失真归因 + 设计指标筛选

## 要回答的唯一问题

P2 发现两个互相矛盾的事实：

1. 器件退化 67% 来自 Gilbert 非线性（tanh 削顶），这没问题；
2. **但「饱和深度」不能预测退化**：r = −0.47（越深反而越轻），
   s11 饱和最浅（rms/V_SAT = 0.19）却退化最重（NRMS 0.2111）。

P2 的候选解释是「preact 分布锐度」，但 s23 是例外，当时明确写了「需专门实验才能定论」。
本脚本就是那个实验。

## 三个候选机理，各自可测

| 机理 | 物理含义 | 观测量 |
|---|---|---|
| **H1 重尾** | 削顶不由 rms 决定，而由尾部决定。rms 小但峰均比高的分布照样被削 | `kurt`、`crest` |
| **H2 符号消去** | head 把 10 个 term 直接相加；正负相消后「大信号里的小误差」在输出端折算的相对误差暴涨 | `cancel_ratio = ‖h‖ / Σ‖z_i‖` |
| **H3 层位置** | 误差注入在哪层决定它被下游放大多少 | 逐层旁路 ΔNRMS |

**H2 最被看重**：它天然解释「饱和浅但退化重」——相消越强，同样的绝对失真
在输出端折算的相对误差越大。而且有直接硬件含义（累加器动态范围需求）。

## 三个模式

### `stats`：逐调用统计

单次前向，记录全部 82 次 `_gilbert` 调用的标度无关统计量：

| 量 | 定义 | 为什么选它 |
|---|---|---|
| `rms_over_vsat` | rms(x) / V_SAT | 旧指标，用来证明它不行 |
| `sat_frac` | mean(abs(x) > V_SAT) | 旧指标的另一形态 |
| `crest` | max(abs(x)) / rms(x) | 一阶重尾指示 |
| `kurt` | E[x⁴] / E[x²]² | 二阶重尾指示（高斯 = 3） |
| `err_out_frac` | ‖g(x) − Σx‖ / ‖Σx‖ | **真正注入信号链的失真**，主候选 |
| `err_elem_frac` | ‖g(x) − x‖ / ‖x‖ | 逐元素失真，相消前有多脏 |

外加 head 两条通路：`head_sum_rms`（10 term 累加和）、
`cancel_ratio`（相消强度，1 = 完全同相，越小消去越强）。

### `layer`：逐层旁路消融

把某层的 8 次 `_gilbert` 换成理想线性求和，其余层保持非线性。
`_gilbert` 里 `BETA_H3=0` 分支就是 `prod.sum(dim=2)`，即精确理想值，**无标度失配**。
同 MC 种子对齐，ΔNRMS 是纯归因。粒度：rechannel + L0..L9 + head + ALL = 13 组。

### `call`：逐调用旁路

82 个 gilbert 调用逐个旁路，看是否某个 dilation tap 独大。种子少一些，成本大。

## 指标筛选的正确做法

不拿 5 个种子做相关性（n = 5 什么都说明不了）。正确做法：
**逐层旁路的 ΔNRMS 当 ground truth，逐层统计量当预测变量，n = 种子数 × 层数 = 50**。
谁的 |r| 最高，谁才是能用的设计指标。本检验自洽——统计量与 ΔNRMS 在同一次
器件实现（同一 MC 种子）下测得。

## 用法

  python p3_sharp.py --mode all --mc 8 --seconds 5.0
  P3_REQUIRE_ACCEL=1 python p3_sharp.py --mode layer     # 强制加速卡
  python p3_sharp.py --mode stats --seeds s11 --mc 1     # 快验

## 与 P1/P2 的关系

前向直接 import `p1_vsat_sweep.run_forward`，**不复制**。三层脚本共用同一份前向，
保证所有历史数字同源可比。改前向只改那一处。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import p1_vsat_sweep as V                                       # noqa: E402

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
import soundfile as sf                                          # noqa: E402
import analog_sim_generic as A                                  # noqa: E402
from analog_sim_generic import DevSim                           # noqa: E402

DEV = V.DEV
_REQ = os.environ.get("P3_REQUIRE_ACCEL", "0") == "1"
if _REQ and DEV != "cuda":
    raise SystemExit("P3_REQUIRE_ACCEL=1 但无可用加速卡，拒绝在 CPU 上运行")

SEEDS = V.SEEDS
DATA = V.DATA
OUT = V.OUT
OUT.mkdir(parents=True, exist_ok=True)

V_SAT = 16.0
DIALECT = dict(BITS=6, SIGMA_MISMATCH=0.01, DRIFT=0.03,
               BETA_H3=0.002, SIGMA_OFFSET=0.005, SNR_DB=70.0, V_SAT=V_SAT)


# ---------------------------------------------------------------- 布局

def build_layout(cfg, sd):
    """算出每次 _gilbert / conv 调用属于哪个逻辑组。

    ⚠ 关键：`_gilbert` 的调用次数 = 该次 conv 的**权重** kernel 长度，不是层的
    `kernel_size`。`rechannel / input_mixer / layer1x1 / head_rechannel` 的权重都是
    1x1（w.shape[-1] == 1），所以各只触发 1 次；只有层的 `_conv` 是 K 次。

    于是 82 次 _gilbert 的分布是（K=6，nL=10）：
      1 (rechannel) + 10 × (6 + 1 + 1) + 1 (head) = 82   ← 与 P2 实测一致
    若按「每个 conv 都 K 次」去算会得到 192，全错。

    返回 (gmap, cmap, n_gilbert, n_conv)
    """
    lc = cfg["layers_configs"][0]
    nL = len(lc["dilations"])
    pre0 = "_layer_arrays.0."
    gmap, cmap = {}, {}
    gi = ci = 0

    def emit(group, slot, klen):
        nonlocal gi, ci
        cmap[ci] = (group, slot)
        ci += 1
        for _ in range(klen):
            gmap[gi] = group
            gi += 1

    emit("rechannel", "rechannel", int(sd[pre0 + "_rechannel.weight"].shape[-1]))
    for li in range(nL):
        pre = f"{pre0}_layers.{li}"
        for slot, key in (("conv", "_conv"), ("mixer", "_input_mixer"),
                          ("l1x1", "_layer1x1")):
            w = sd.get(f"{pre}.{key}.weight")
            if w is None:
                continue
            emit(f"L{li}", slot, int(w.shape[-1]))
    emit("head", "head", int(sd[pre0 + "_head_rechannel.weight"].shape[-1]))
    return gmap, cmap, gi, ci


def rf_and_outlen(cfg, T):
    lc = cfg["layers_configs"][0]
    K, dils = lc["kernel_size"], lc["dilations"]
    rf = 1 + sum((K - 1) * d for d in dils)
    return rf, T - (rf - 1)


# ---------------------------------------------------------------- 统计量

def gilbert_stats(prod, vs, beta, chunk=16384):
    """分块累计标度无关统计量，避免一次性物化多个全量中间张量。

    prod: (B, C, Cin, T)。误差口径统一用能量比（等价 RMS 比），
    与 `nrms_of` 的加权方式一致，避免 max 口径被 outlier 主导。
    """
    T = prod.shape[-1]
    n = 0
    ss = ss4 = 0.0                        # Σx², Σx⁴（x 已归一化到 V_SAT）
    amax = 0.0
    nclip = 0
    e_ideal = e_out = 0.0                 # ‖Σ理想‖², ‖g−Σ理想‖²
    e_in = e_elem = 0.0                   # ‖x‖², ‖g_elem−x‖²
    for s in range(0, T, chunk):
        p = prod[..., s:s + chunk].detach()
        xs = p / vs
        pe = vs * torch.tanh(xs * (1 + beta * xs.square()))
        ideal = p.sum(dim=2)
        got = pe.sum(dim=2)
        n += p.numel()
        ss += float((xs * xs).sum())
        ss4 += float((xs * xs).square().sum())
        amax = max(amax, float(p.abs().amax()))
        nclip += int((p.abs() > vs).sum())
        e_in += float((p * p).sum())
        e_elem += float(((pe - p) * (pe - p)).sum())
        e_ideal += float((ideal * ideal).sum())
        d = got - ideal
        e_out += float((d * d).sum())
    eps = 1e-30
    rms = (ss / n) ** 0.5 * vs
    return {
        "rms_over_vsat": float(rms / vs),
        "absmax_over_vsat": float(amax / vs),
        "crest": float(amax / rms) if rms > 0 else 0.0,
        "kurt": float((ss4 / n) / ((ss / n) ** 2 + eps)),
        "sat_frac": float(nclip / n),
        "err_out_frac": float((e_out / (e_ideal + eps)) ** 0.5),
        "err_elem_frac": float((e_elem / (e_in + eps)) ** 0.5),
    }


# ---------------------------------------------------------------- 单次前向

def run_once(sd, cfg, xt, over=None, mc_seed=2000, bypass=(), stats=False,
             stat_every=None):
    """跑一次前向。bypass 是逻辑组名集合（"ALL" = 全部旁路）。

    返回 (pred, stats_dict|None, head_dict)
    """
    A.P.clear()
    A.P.update(base_p)
    if over:
        A.P.update(over)
    try:
        dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(mc_seed))
        gmap, _, ng, _ = build_layout(cfg, sd)
        _, out_len = rf_and_outlen(cfg, xt.shape[-1])
        sink = {} if stats else None
        acts, conv_in, terms = [], [], []

        orig_act, orig_conv, orig_g = dev.act, dev.conv, dev._gilbert

        def act_w(z, kind):
            out = orig_act(z, kind)
            if stats:
                zz = out[:, :, -out_len:].detach()
                acts.append(float(zz.pow(2).mean().sqrt()))
                # 存完整 term 向量（10 × out_len float ≈ 9.6 MB），
                # 以便精确算相干度与消去比。只记 RMS 无法分离「相位」与「幅度」。
                terms.append(zz.detach().reshape(-1).clone())
            return out

        def conv_w(x, w, bias, dilation, groups):
            if stats:
                conv_in.append(float(x.detach().pow(2).mean().sqrt()))
            return orig_conv(x, w, bias, dilation, groups)

        cnt = {"i": 0}

        def g_wrapped(prod):
            i = cnt["i"]
            cnt["i"] = i + 1
            grp = gmap.get(i, "?")
            if sink is not None:
                p = prod if stat_every is None else prod[..., :stat_every]
                st = gilbert_stats(p, dev.p["V_SAT"], dev.p["BETA_H3"])
                st["group"] = grp
                sink[i] = st
            if "ALL" in bypass or grp in bypass:
                return prod.sum(dim=2)        # 精确理想线性求和
            return orig_g(prod)

        dev.act, dev.conv, dev._gilbert = act_w, conv_w, g_wrapped
        with torch.no_grad():
            pred = V.run_forward(dev, cfg, xt)
        dev.act, dev.conv, dev._gilbert = orig_act, orig_conv, orig_g

        # 布局自校验：算出来的 gilbert 次数必须等于实际调用次数。
        # 静默错位会让全部归因失效且不报错，这是本脚本最危险的失效模式，
        # 所以每次运行都验一遍（开销可忽略）。
        if cnt["i"] != ng:
            raise RuntimeError(
                f"[layout] 预测 {ng} 次 _gilbert，实际 {cnt['i']} 次。"
                f"调用布局假设失效，归因结果不可信，请重核 build_layout。")

        head = {}
        if stats and conv_in:
            h_rms = conv_in[-1]               # head_rechannel 的输入 = 10 term 累加和
            s_sum = float(np.sum(acts)) if acts else 0.0
            head = {
                "head_sum_rms": h_rms,
                "head_sum_rms_over_vsat": h_rms / dev.p["V_SAT"],
                "term_rms_sum": s_sum,
                # ⚠ 这个是「幅度加权」的消去比，会被幅度不均严重污染：
                # 各层 term 从 0.13 涨到 12.9（差 50 倍），Σ 被末几层主导，
                # 即使各项完全独立它也会趋近于 1。冒烟实测独立情形 = 0.547
                # （理论 1/sqrt(10)=0.316），偏差 73%。**不能单独用来判断消去。**
                "cancel_ratio_raw": (h_rms / s_sum) if s_sum > 0 else None,
                "n_terms": len(acts),
                "n_gilbert": ng,
            }
            # ---- 干净的相位指标：先把每项按自身 RMS 归一，再求能量 ----
            # phase_coh = ‖Σ û_i‖² / n ∈ [0, n]。1 = 完全独立，n = 完全同相。
            # 幅度信息被刻意抹掉，所以它只反映「符号/相位是否对齐」。
            if terms:
                u = [t / (t.pow(2).mean().sqrt() + 1e-30) for t in terms]
                svec = torch.stack(u).sum(dim=0)
                coh = float(svec.pow(2).mean()) / len(u)
                head["phase_coh"] = coh
                head["phase_coh_norm"] = coh / len(u)      # 归一化到 [0,1]
                head["n_terms_vec"] = len(u)
                # 也记录幅度集中度：最大 term 能量占总能量的比例。
                # 高集中度 + 低相干度 = 「一根独大」，会掩盖其余项
                e = np.array([float(t.pow(2).mean()) for t in terms])
                head["amp_top1_frac"] = float(e.max() / e.sum())
                head["amp_entropy"] = float(
                    -(e / e.sum() * np.log(e / e.sum() + 1e-30)).sum()
                    / np.log(len(e)))                  # 归一化熵，1 = 幅度均匀
        return pred, sink, head
    finally:
        A.P.clear()
        A.P.update(base_p)


def _agg(vals):
    """MC 聚合用 RMS，与 p1_run / p1_vsat_sweep / p2_ablate 同口径。"""
    return float(np.sqrt((np.array(vals, dtype=np.float64) ** 2).mean()))


# ---------------------------------------------------------------- 模式

def run_stats(cfg, sd, xt, yt, mc, seed_name, res, stat_every):
    slot = res.setdefault("stats", {}).setdefault(seed_name, [])
    for i in range(mc):
        pred, sink, head = run_once(sd, cfg, xt, DIALECT, 2000 + i,
                                    stats=True, stat_every=stat_every)
        slot.append({"nrms": V.nrms_of(pred, yt), "calls": sink,
                     "head": head, "mc_seed": 2000 + i})
    last = slot[-1]
    top = sorted(last["calls"].items(),
                 key=lambda kv: -kv[1]["err_out_frac"])[:3]
    print(f"  [stats] {seed_name:4} NRMS={last['nrms']:.4f}  "
          f"coh={last['head'].get('phase_coh', float('nan')):.2f}  "
          f"head={last['head'].get('head_sum_rms_over_vsat', 0):.2f}xV_SAT  "
          "err_out top3: " +
          ", ".join(f"{v['group']}#{k}={v['err_out_frac']:.3f}" for k, v in top),
          flush=True)


def run_layer(cfg, sd, xt, yt, mc, seed_name, res):
    lc = cfg["layers_configs"][0]
    nL = len(lc["dilations"])
    groups = ["rechannel"] + [f"L{i}" for i in range(nL)] + ["head"]
    ss = res.setdefault("layer", {}).setdefault(seed_name, {})
    for gname in ["BASE"] + groups + ["ALL"]:
        bypass = () if gname == "BASE" else (
            {"ALL"} if gname == "ALL" else {gname})
        vals = [V.nrms_of(run_once(sd, cfg, xt, DIALECT, 2000 + i,
                                   bypass=bypass)[0], yt) for i in range(mc)]
        ss[gname] = {"nrms": _agg(vals), "mc_vals": vals}
        base = ss.get("BASE", {}).get("nrms")
        d = (base - ss[gname]["nrms"]) if base is not None else 0.0
        print(f"  [layer] {seed_name:4} {gname:<10} "
              f"NRMS={ss[gname]['nrms']:.4f}  Δ={d:+.4f}", flush=True)
    return ss


def run_call(cfg, sd, xt, yt, mc, seed_name, res):
    """逐调用旁路：只旁路第 idx 次 _gilbert，其余保持非线性。

    用 `_only{idx}` 作为标记，run_once 里需要特殊处理——见 bypass 支持。
    """
    gmap, _, ng, _ = build_layout(cfg, sd)
    ss = res.setdefault("call", {}).setdefault(seed_name, {})
    for idx in [-1] + list(range(ng)):
        label = "BASE" if idx < 0 else f"{idx}:{gmap.get(idx, '?')}"
        vals = []
        for i in range(mc):
            only = None if idx < 0 else idx
            vals.append(V.nrms_of(
                run_once_idx(sd, cfg, xt, DIALECT, 2000 + i, only)[0], yt))
        ss[label] = {"nrms": _agg(vals)}
    print(f"  [call]  {seed_name:4} 完成（单调用粒度，{ng} 次旁路）", flush=True)
    return ss


def run_once_idx(sd, cfg, xt, over, mc_seed, only_idx):
    """只旁路第 only_idx 次 _gilbert 的前向（call 模式专用）。"""
    A.P.clear()
    A.P.update(base_p)
    if over:
        A.P.update(over)
    try:
        dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(mc_seed))
        orig_g = dev._gilbert
        cnt = {"i": 0}

        def g_wrapped(prod):
            i = cnt["i"]
            cnt["i"] = i + 1
            return prod.sum(dim=2) if i == only_idx else orig_g(prod)

        dev._gilbert = g_wrapped
        with torch.no_grad():
            pred = V.run_forward(dev, cfg, xt)
        dev._gilbert = orig_g
        return pred, None, {}
    finally:
        A.P.clear()
        A.P.update(base_p)


# ---------------------------------------------------------------- 汇总

def summarize_layer(res):
    """每层 ΔNRMS 与占比（相对 ALL 旁路 = 数字模型）。"""
    out = {}
    for s, ss in res.get("layer", {}).items():
        if "BASE" not in ss or "ALL" not in ss:
            continue
        base, dig = ss["BASE"]["nrms"], ss["ALL"]["nrms"]
        tot = base - dig
        rows = {}
        for g, v in ss.items():
            if g in ("BASE", "ALL"):
                continue
            d = base - v["nrms"]
            rows[g] = {"nrms": v["nrms"], "delta": d,
                       "share": (d / tot) if tot else None}
        out[s] = {"base": base, "digital": dig, "total_drop": tot, "groups": rows}
    return out


def _energy_weight(sel):
    """层内能量加权：同一层里 conv tap 与 mixer/l1x1 量级差很大，
    简单平均会被小项稀释。权重用 rms_over_vsat²（即能量）。"""
    w = np.array([max(c["rms_over_vsat"], 1e-9) ** 2 for c in sel])
    return w / w.sum()


def _pcorr(x, y, z):
    """偏相关：控制 z 后 x 与 y 的相关。三层混杂（层深）必须扣掉，
    否则「越深的层越容易被下游放大」会伪装成任何统计量的预测能力。"""
    if len(x) < 4:
        return None
    z = np.asarray(z, dtype=np.float64)
    A1 = np.vstack([z, np.ones_like(z)]).T
    def _resid(v):
        beta, *_ = np.linalg.lstsq(A1, v, rcond=None)
        return v - A1 @ beta
    rx, ry = _resid(x), _resid(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def screen_indicators(res, lay_sum):
    """指标筛选：逐层统计量 vs 逐层 ΔNRMS，n = 种子 × 层数。

    同时给两个数：
      r_spearman —— 原始秩相关，但被「层深」混杂污染
      r_partial  —— 控制层深后的偏相关，**这才是选指标该看的数**
    层深混杂为什么严重：深层信号本来就大、ΔNRMS 本来就大，
    任何随深度单调增长的统计量都会白捡一个高相关。
    """
    keys = ["rms_over_vsat", "sat_frac", "crest", "kurt",
            "err_out_frac", "err_elem_frac", "absmax_over_vsat"]
    pairs = {k: [] for k in keys}
    depth_of = {}
    detail = {}
    for s, recs in res.get("stats", {}).items():
        lay = lay_sum.get(s)
        if not lay:
            continue
        calls = recs[0]["calls"]
        for g, row in lay["groups"].items():
            sel = [v for v in calls.values() if v.get("group") == g]
            if not sel:
                continue
            # 层深：L<i> -> i；rechannel -> -1；head -> nL
            if g == "rechannel":
                dep = -1.0
            elif g == "head":
                dep = 99.0
            else:
                dep = float(g[1:])
            d = detail.setdefault(s, {}).setdefault(g, {"delta": row["delta"],
                                                        "depth": dep})
            w = _energy_weight(sel)
            for k in keys:
                v = float(np.dot([c[k] for c in sel], w))
                pairs[k].append((v, row["delta"], dep))
                d[k] = v
            for hk in ("phase_coh", "phase_coh_norm", "cancel_ratio_raw",
                       "amp_top1_frac", "amp_entropy"):
                v = recs[0]["head"].get(hk)
                if v is not None:
                    d[hk] = v
    out = {}
    for k, pr in pairs.items():
        if len(pr) < 5:
            out[k] = {"n": len(pr), "r_spearman": None, "r_partial": None}
            continue
        x = np.array([p[0] for p in pr], dtype=np.float64)
        y = np.array([p[1] for p in pr], dtype=np.float64)
        z = np.array([p[2] for p in pr], dtype=np.float64)
        if x.std() < 1e-12 or y.std() < 1e-12:
            out[k] = {"n": len(pr), "r_spearman": None, "r_partial": None}
            continue
        out[k] = {"n": int(len(pr)),
                  "r_pearson": float(np.corrcoef(x, y)[0, 1]),
                  "r_spearman": float(np.corrcoef(np.argsort(np.argsort(x)),
                                                  np.argsort(np.argsort(y)))[0, 1]),
                  "r_partial": _pcorr(x, y, z)}
    return {"screen": out, "detail": detail}


def cross_seed_review(res):
    """种子层面：相位相干度 / head 摆幅 与整网 NRMS 的关系（n=5，仅看方向）。

    这是回答「s11 饱和浅却退化重」的主线之一：若坏种子的 phase_coh 明显更低
    或 head 摆幅更高，则 H2（符号消去）成立；否则回去查 H1（重尾）与 H3（层位置）。
    """
    rows = []
    for s, recs in res.get("stats", {}).items():
        h = recs[0]["head"]
        rows.append({"seed": s, "nrms": recs[0]["nrms"],
                     "phase_coh": h.get("phase_coh"),
                     "phase_coh_norm": h.get("phase_coh_norm"),
                     "cancel_ratio_raw": h.get("cancel_ratio_raw"),
                     "amp_top1_frac": h.get("amp_top1_frac"),
                     "head_sum_rms_over_vsat": h.get("head_sum_rms_over_vsat")})
    if len(rows) < 3:
        return {"rows": rows}
    def _c(key):
        v = np.array([r[key] for r in rows], dtype=np.float64)
        nr = np.array([r["nrms"] for r in rows], dtype=np.float64)
        return {"r_pearson": float(np.corrcoef(v, nr)[0, 1]),
                "r_spearman": float(np.corrcoef(np.argsort(np.argsort(v)),
                                                np.argsort(np.argsort(nr)))[0, 1])}
    out = {"rows": rows, "head_vs_nrms": _c("head_sum_rms_over_vsat")}
    for k in ("phase_coh", "cancel_ratio_raw", "amp_top1_frac"):
        try:
            out[f"{k}_vs_nrms"] = _c(k)
        except Exception:
            pass
    return out


# ---------------------------------------------------------------- main

base_p = None


def main():
    global base_p
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stats", "layer", "call", "all"],
                    default="all")
    ap.add_argument("--seeds", nargs="*", default=SEEDS)
    ap.add_argument("--mc", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--stat-every", type=int, default=None,
                    help="统计窗口只取前 N 个时间点（省时；比值口径自洽）")
    ap.add_argument("--call-mc", type=int, default=2,
                    help="call 模式的 MC 次数，默认更小")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    base_p = dict(A.P)
    print(f"[device] {DEV}  torch={torch.__version__}  "
          f"hip={getattr(torch.version, 'hip', None)}", flush=True)
    if DEV == "cuda":
        print(f"[device] {torch.cuda.get_device_name(0)}", flush=True)

    xi, _ = sf.read(str(DATA / "tsa15_test_input.wav"), dtype="float32")
    yi, _ = sf.read(str(DATA / "tsa15_test_target.wav"), dtype="float32")
    n = int(48000 * args.seconds)
    xt = torch.from_numpy(xi[:n]).to(DEV)
    yt = torch.from_numpy(yi[:n]).to(DEV)
    print(f"[data] {args.seconds}s = {n} samples, seeds={args.seeds}, "
          f"mc={args.mc}", flush=True)

    res = {"meta": {"dev": DEV, "torch": torch.__version__,
                    "hip": str(getattr(torch.version, "hip", None)),
                    "mc": args.mc, "call_mc": args.call_mc,
                    "seconds": args.seconds, "v_sat": V_SAT,
                    "dialect": DIALECT, "tag": args.tag,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S")}}
    t0 = time.time()

    for s in args.seeds:
        path = V.nam_path(s)
        cfg, sd = V.load_nam(path)
        print(f"\n=== {s}  ({path})", flush=True)
        if args.mode in ("stats", "all"):
            run_stats(cfg, sd, xt, yt, args.mc, s, res, args.stat_every)
        if args.mode in ("layer", "all"):
            run_layer(cfg, sd, xt, yt, args.mc, s, res)
        if args.mode in ("call", "all"):
            run_call(cfg, sd, xt, yt, args.call_mc, s, res)

    lay_sum = summarize_layer(res)
    res["layer_summary"] = lay_sum
    if "stats" in res and lay_sum:
        res["indicator_screen"] = screen_indicators(res, lay_sum)
    if "stats" in res:
        res["cross_seed"] = cross_seed_review(res)

    tag = args.tag or time.strftime("%m%d_%H%M")
    outp = OUT / f"p3_sharp_{tag}.json"
    json.dump(res, open(outp, "w"), indent=1, ensure_ascii=False)
    print(f"\n[saved] {outp}  总耗时 {time.time()-t0:.0f}s")

    # ---------------- 控制台摘要 ----------------
    if res.get("cross_seed"):
        print("\n--- 种子层面（n=5，仅看方向）---")
        print(f"{'seed':<6}{'NRMS':>8}{'coh':>7}{'top1':>7}{'head/Vsat':>11}")
        for r in res["cross_seed"]["rows"]:
            pc = r.get("phase_coh")
            t1 = r.get("amp_top1_frac")
            print(f"{r['seed']:<6}{r['nrms']:>8.4f}"
                  f"{(pc if pc is not None else float('nan')):>7.2f}"
                  f"{(t1 if t1 is not None else float('nan')):>7.2f}"
                  f"{r['head_sum_rms_over_vsat']:>11.2f}")
        for k in ("phase_coh", "cancel_ratio_raw", "amp_top1_frac",
                  "head_sum_rms_over_vsat"):
            c = res["cross_seed"].get(f"{k}_vs_nrms")
            if c:
                print(f"  {k:<22} vs NRMS: r={c['r_pearson']:+.2f} "
                      f"rho={c['r_spearman']:+.2f}")

    if lay_sum:
        print("\n--- 逐层旁路：ΔNRMS 占比（各层独立旁路，占比之和可 >100%，"
              "说明层间有耦合）---")
        seeds = sorted(lay_sum)
        head_g = sorted({g for s in lay_sum.values() for g in s["groups"]},
                        key=lambda x: (x != "rechannel", x != "head", x))
        print(f"{'层':<11}" + "".join(f"{s:>9}" for s in seeds))
        for g in head_g:
            row = ""
            for s in seeds:
                sh = lay_sum[s]["groups"].get(g, {}).get("share")
                row += f"{100*sh:>8.1f}%" if sh is not None else f"{'-':>9}"
            print(f"{g:<11}{row}")

    scr = (res.get("indicator_screen") or {}).get("screen")
    if scr:
        print("\n--- 设计指标筛选（n = 种子×层数，|rho| 越大越能预测退化）---")
        print(f"{'指标':<20}{'n':>5}{'r':>9}{'rho':>9}{'partial':>10}")
        print("  (partial = 扣掉层深后的偏相关，选指标看这一列)")
        for k, v in sorted(scr.items(),
                           key=lambda kv: -abs(kv[1].get("r_partial") or 0)):
            pa = v.get("r_partial")
            pas = f"{pa:>+10.3f}" if pa is not None else f"{'-':>10}"
            if v.get("r_spearman") is None:
                print(f"{k:<20}{v['n']:>5}{'-':>9}{'-':>9}{pas}")
            else:
                print(f"{k:<20}{v['n']:>5}{v['r_pearson']:>+9.3f}"
                      f"{v['r_spearman']:>+9.3f}{pas}")


if __name__ == "__main__":
    main()
