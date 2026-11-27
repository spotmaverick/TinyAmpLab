"""P4：amp_top1_frac 代理量的跨音频验证 + head 级削顶的因果干预

## 为什么做这个

P3（n=5 种子）证明 head term 幅度集中度 `amp_top1_frac` 与整网退化 ρ=+0.80，
但那是**相关性**，且只有 5 个样本、单一音频片段。两个未决问题：

  1. 这个相关是不是音频片段的伪影？
  2. head 处的削顶到底是不是那些退化的因？需要多少余量才够？

本脚本零训练成本（复用已训 5 种子），用两个实验回答。

### mode=proxy（代理量跨音频验证）
5 种子 × 3 段互不重叠的 tsa15 音频 = 15 样本。每段算：
  device NRMS（V_SAT=16）/ digital NRMS（全 gilbert 旁路成理想求和）/ 退化 = 两者之差，
  以及 head 的 amp_top1_frac / phase_coh / err_out_frac / crest。
对 15 样本做 pooled 相关，**并对每段单独做 n=5 的相关**——
判据是：三段各自都成立才算「不是单片段伪影」。
seg0 与 P3 完全同输入，可直接交叉校验（P3 报 s7 seg0 dev=0.1045 / top1=0.366）。

### mode=causal（head 级削顶的因果干预）★ 这是 P0-1 的直接验证

**设计动机：必须换旋钮。**
最初的方案是把 10 个 head term 按 (gmean/rms_i)^θ 拉平（见 mode=flat）。
冒烟实测发现：θ=0→1 时理想路径 NRMS 从 0.0775 崩到 0.4134（5.3 倍）。
拉平 term 幅度改变了 h = Σz_i 这个函数本身，所谓「零成本」是假的，
退化 = device − digital 变成两个坏东西之差，因果链断了。**该方案作废**（代码保留作反例）。

改用**只提升 head 这一级的 V_SAT**（`vsat_head`），其余各级固定 16：
  · 模型函数完全不变（不改权重、不改幅度、不改拓扑），只改一个模拟量
  · 极限清晰：V_SAT_head → ∞ 时 head 削顶消失，NRMS 必然收敛到「全旁路」下限
  · 直接对应 P0-1「head 独立宽摆幅电源域」这条硬件建议

定义回收率（同 seed、同 MC 种子对齐）：

    base       = NRMS(V_SAT_head=16, 其余 16)     # 现状
    floor      = NRMS(全 gilbert 旁路)             # 理想下限
    native(V)  = NRMS(V_SAT_head=V, 其余 16)
    recovery(V) = (base − native(V)) / (base − floor)

判据：
  · recovery(V→∞) → 1：head 削顶是主要退化源 → P3 的「85% 在 head」获因果支撑
  · recovery 只到 0.6~0.8：head 之外还有不可忽略的贡献
  · head 的 err_out_frac 应随 V_SAT_head 单调降、且 NRMS 跟着它走 →
    构成 P3「err_out_frac 是最强指标」的因果闭环（P3 只有相关）
  · 跨种子看：amp_top1_frac 越高，recovery 曲线应越陡（需更大余量才够）

### mode=flat（已作废的反例，仅存档）
把 head term 按 θ 拉平并保持求和 RMS。**不要用它的 drop 下结论**——
它的 dig 自己就崩了。保留是为了在报告里记录「为什么这个干预不可用」。

## 用法
  python p4_proxy.py --mode proxy  --mc 4
  python p4_proxy.py --mode causal --mc 4
  python p4_proxy.py --mode all    --mc 4
  P4_REQUIRE_ACCEL=1 python p4_proxy.py --mode all     # 强制加速卡

## 度量口径
- NRMS 用 V.nrms_of（与 P1/P2/P3 完全一致），MC 用 RMS 聚合（不是 mean）
- 统计量沿用 P3.gilbert_stats，布局用 P3.build_layout（按权重形状推导）
- head 的 gilbert 索引 = ng − 1（head_rechannel 是 1×1，只触发 1 次）
- 每次前向断言实际 gilbert 调用数 == 预测值，防静默错位
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
import p3_sharp as P3                                           # noqa: E402

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
import soundfile as sf                                          # noqa: E402
import analog_sim_generic as A                                  # noqa: E402
from analog_sim_generic import DevSim                           # noqa: E402

DEV = V.DEV
_REQ = os.environ.get("P4_REQUIRE_ACCEL", "0") == "1"
if _REQ and DEV != "cuda":
    raise SystemExit("P4_REQUIRE_ACCEL=1 但无可用加速卡，拒绝在 CPU 上运行")

SEEDS = V.SEEDS
DATA = V.DATA
OUT = V.OUT
V_SAT = 16.0

DIALECT = dict(BITS=6, SIGMA_MISMATCH=0.01, DRIFT=0.03,
               BETA_H3=0.002, SIGMA_OFFSET=0.005, SNR_DB=70.0, V_SAT=V_SAT)

# 三段互不重叠的 tsa15 test 片段（各 5s，起点 0/8/16s；RF=5411≈0.113s，留足余量）
SEGMENTS = [("seg0", 0.0), ("seg1", 8.0), ("seg2", 16.0)]
SEG_LEN = 5.0

# head 级 V_SAT 扫描点；1e5 视作「实际无穷」（tanh 在该幅度下已近似线性）
VSAT_HEADS = [16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 100000.0]
THETAS = [0.0, 0.25, 0.5, 0.75, 1.0]      # 仅 mode=flat 使用（已作废）

base_p = None


# ---------------------------------------------------------------- 布局
def _layout_n(cfg, sd):
    """返回 (gmap, ng, nL, head_idx)。head 必须是最后一次 gilbert。"""
    gmap, _cmap, ng, _nc = P3.build_layout(cfg, sd)
    nL = len(cfg["layers_configs"][0]["dilations"])
    head_idx = ng - 1
    if gmap.get(head_idx) != "head":
        raise RuntimeError(f"[layout] 末次 gilbert 不属于 head（gmap[{head_idx}]="
                           f"{gmap.get(head_idx)}），布局假设失效")
    return gmap, ng, nL, head_idx


# ---------------------------------------------------------------- 前向（唯一入口）
def run_once(sd, cfg, xt, mc_seed, over=None, stats=False, bypass=(),
             vsat_head=None, keep_terms=False):
    """跑一次前向。返回 (pred, head_stats, terms_or_empty)。

    bypass={"ALL"}    → 所有 gilbert 换成理想线性求和（= digital 参考）
    vsat_head=<float> → 仅 head 那一次 gilbert 用指定 V_SAT，其余用 base_p 的
    keep_terms=True   → 同时返回 10 个 head term（仅 mode=flat 用）
    """
    global base_p
    A.P.clear(); A.P.update(base_p)
    if over:
        A.P.update(over)
    try:
        dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(mc_seed))
        _gmap, ng, _nL, head_idx = _layout_n(cfg, sd)
        orig_g = dev._gilbert
        sink = {} if stats else None
        is_bypass = "ALL" in bypass
        cnt = {"i": 0}

        def g_wrapped(prod):
            i = cnt["i"]
            cnt["i"] = i + 1
            if is_bypass:
                return prod.sum(dim=2)
            v = dev.p["V_SAT"]
            v_here = v
            if vsat_head is not None and i == head_idx:
                v_here = float(vsat_head)
            if stats:
                sink[i] = P3.gilbert_stats(prod[..., :16384], v_here, dev.p["BETA_H3"])
            if v_here != v:
                dev.p["V_SAT"] = v_here
                try:
                    return orig_g(prod)
                finally:
                    dev.p["V_SAT"] = v
            return orig_g(prod)

        dev._gilbert = g_wrapped
        with torch.no_grad():
            if keep_terms:
                pred, terms = _forward_terms(dev, cfg, xt)
            else:
                pred = V.run_forward(dev, cfg, xt)
                terms = []
        dev._gilbert = orig_g
        if cnt["i"] != ng:
            raise RuntimeError(f"[layout] 预测 {ng} 次 gilbert，实际 {cnt['i']} 次")

        head = {}
        if stats:
            head = _head_stats(terms if keep_terms else None, sink, head_idx)
        store = [t.detach().reshape(-1).clone() for t in terms] if keep_terms else []
        return pred, head, store
    finally:
        A.P.clear(); A.P.update(base_p)


def _head_stats(terms, sink, head_idx):
    """head 处的注入失真 +（可选）term 幅度分布统计。"""
    h = {}
    if terms:
        e = np.array([float(t.pow(2).mean()) for t in terms])
        h["amp_top1_frac"] = float(e.max() / e.sum())
        h["amp_entropy"] = float(
            -(e / e.sum() * np.log(e / e.sum() + 1e-30)).sum() / np.log(len(e)))
        h["term_rms"] = [float(np.sqrt(x)) for x in e]
        u = [t / (t.pow(2).mean().sqrt() + 1e-30) for t in terms]
        svec = torch.stack(u).sum(dim=0)
        h["phase_coh"] = float(svec.pow(2).mean()) / len(u)
    if sink:
        last = sink.get(head_idx)
        if last is not None:
            h["head_err_out_frac"] = last["err_out_frac"]
            h["head_crest"] = last["crest"]
            h["head_rms_over_vsat"] = last["rms_over_vsat"]
            h["head_sat_frac"] = last["sat_frac"]
        h["max_err_out_any"] = max(v["err_out_frac"] for v in sink.values())
        # head 之外的次严重者（看 head 是否一枝独秀）
        others = [v["err_out_frac"] for i, v in sink.items() if i != head_idx]
        h["max_err_out_other"] = max(others) if others else None
        h["n_calls"] = len(sink)
    return h


def _forward_terms(dev, cfg, xt, scales=None):
    """V.run_forward 的逐 term 版本（仅 mode=flat 用，保留作反例）。

    唯一改动：head 求和前对第 i 个 term 乘 scales[i]；residual 路径用未缩放的 z。
    """
    lc = cfg["layers_configs"][0]
    K, dils = lc["kernel_size"], lc["dilations"]
    act = lc.get("activation", "Tanh")
    g_in = lc.get("groups_input", 1)
    l1 = lc.get("layer_1x1_config", {}) or {}
    g_l1 = l1.get("groups", 1) if l1.get("active") else 1
    head_scale = cfg.get("head_scale", 0.01)

    y = xt[None, None, :] if xt.ndim == 1 else xt[:, None, :]
    cond = y
    rf = 1 + sum((K - 1) * d for d in dils)
    out_len = y.shape[2] - (rf - 1)
    pre0 = "_layer_arrays.0."
    W = dev.w
    y = dev.mac1x1(y, W[pre0 + "_rechannel.weight"], None, groups=1)
    terms = []
    for li, d in enumerate(dils):
        pre = f"{pre0}_layers.{li}"
        z = dev.conv(y, W[f"{pre}._conv.weight"], W.get(f"{pre}._conv.bias"), d, g_in)
        mix = dev.conv(cond, W[f"{pre}._input_mixer.weight"], None, 1, 1)[:, :, -z.shape[2]:]
        n = min(z.shape[2], mix.shape[2])
        z = dev.act(z[:, :, -n:] + mix[:, :, -n:], act)
        z = z + dev.noise(z.shape)
        ro = dev.conv(z, W[f"{pre}._layer1x1.weight"], W.get(f"{pre}._layer1x1.bias"), 1, g_l1)
        y = y[:, :, -ro.shape[2]:] + ro + dev.noise(ro.shape)
        t = z[:, :, -out_len:]
        if scales is not None:
            t = scales[li] * t
        terms.append(t)
    h = terms[0]
    for t in terms[1:]:
        h = h + t
    out = dev.conv(h, W[pre0 + "_head_rechannel.weight"],
                   W.get(pre0 + "_head_rechannel.bias"), 1, 1)
    return (head_scale * out)[:, 0, :], terms


# ---------------------------------------------------------------- 工具
def _agg(vals):
    """MC 聚合用 RMS（与 P1/P2/P3 口径一致，不是 mean）。"""
    return float(np.sqrt((np.asarray(vals, dtype=np.float64) ** 2).mean()))


def _corr(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return {"n": int(len(x)), "r": None, "rho": None}
    return {"n": int(len(x)),
            "r": float(np.corrcoef(x, y)[0, 1]),
            "rho": float(np.corrcoef(np.argsort(np.argsort(x)),
                                     np.argsort(np.argsort(y)))[0, 1])}


def _seg(seg_name):
    """读一段音频，返回 (xt, yt)。"""
    t0 = dict(SEGMENTS)[seg_name]
    xi, _ = sf.read(str(DATA / "tsa15_test_input.wav"), dtype="float32")
    yi, _ = sf.read(str(DATA / "tsa15_test_target.wav"), dtype="float32")
    i0 = int(48000 * t0)
    i1 = min(len(xi), i0 + int(48000 * SEG_LEN))
    return (torch.from_numpy(np.ascontiguousarray(xi[i0:i1])).to(DEV),
            torch.from_numpy(np.ascontiguousarray(yi[i0:i1])).to(DEV))


# ---------------------------------------------------------------- 自检
def assert_equiv(cfg, sd, xt):
    """run_once 在无干预时必须与原前向逐位一致；vsat_head=V_SAT 也须一致。

    这是防「包装层静默改行为」的唯一保险——P3 那次布局错位就是靠断言才发现的。
    """
    global base_p
    A.P.clear(); A.P.update(base_p)
    d1 = DevSim(sd, torch.Generator(device=DEV).manual_seed(4242))
    with torch.no_grad():
        p1 = V.run_forward(d1, cfg, xt)

    p2, _, _ = run_once(sd, cfg, xt, 4242, stats=False)
    p3, _, _ = run_once(sd, cfg, xt, 4242, over=DIALECT, stats=False)
    p4, _, _ = run_once(sd, cfg, xt, 4242, over=DIALECT, stats=False,
                        vsat_head=V_SAT)     # 显式传同值，应完全等价
    d_a = float((p1 - p2).abs().max())
    d_b = float((p2 - p3).abs().max())       # base_p 与 DIALECT 应等价
    d_c = float((p3 - p4).abs().max())
    A.P.clear(); A.P.update(base_p)
    for name, d in (("passthrough", d_a), ("base==DIALECT", d_b),
                    ("vsat_head=same", d_c)):
        if d > 1e-7:
            raise RuntimeError(f"[自检] {name} 不等价，max|Δ|={d:.3e}。停止。")
    return max(d_a, d_b, d_c)


# ---------------------------------------------------------------- mode=proxy
def run_proxy(cfg, sd, seed_name, mc, res):
    for seg_name, _t0 in SEGMENTS:
        xt, yt = _seg(seg_name)
        devv, dig, heads = [], [], []
        for i in range(mc):
            p, h, _ = run_once(sd, cfg, xt, 2000 + i, DIALECT, stats=True,
                               keep_terms=True)
            devv.append(V.nrms_of(p, yt)); heads.append(h)
            pd, _, _ = run_once(sd, cfg, xt, 2000 + i, DIALECT, bypass={"ALL"})
            dig.append(V.nrms_of(pd, yt))
        dv, dg = np.asarray(devv), np.asarray(dig)
        d_pair = dv - dg                     # 配对：同 MC 种子下逐次求差
        row = {"seed": seed_name, "seg": seg_name,
               "nrms_device": _agg(devv), "nrms_digital": _agg(dig),
               "mc_device": [float(x) for x in devv],
               "mc_digital": [float(x) for x in dig]}
        # 主口径：配对差（公共噪声抵消）。聚合差仅用于与 P1/P3 横向对照
        row["drop"] = float(d_pair.mean())
        row["drop_std"] = float(d_pair.std(ddof=1)) if len(d_pair) > 1 else None
        row["drop_agg"] = row["nrms_device"] - row["nrms_digital"]
        row["drop"] = row["drop"] if len(d_pair) > 1 else row["drop_agg"]
        row["ratio"] = row["nrms_device"] / max(row["nrms_digital"], 1e-12)
        for k in ("amp_top1_frac", "amp_entropy", "phase_coh",
                  "head_err_out_frac", "head_crest", "head_rms_over_vsat",
                  "head_sat_frac", "max_err_out_any", "max_err_out_other"):
            row[k] = heads[-1].get(k)
        res.setdefault("proxy", {}).setdefault(seed_name, {})[seg_name] = row
        def _f(v, spec=".4f"):
            return format(v, spec) if v is not None else "-"
        print(f"  [proxy] {seed_name:5} {seg_name} dev={row['nrms_device']:.4f} "
              f"dig={row['nrms_digital']:.4f} drop={row['drop']:+.4f} "
              f"top1={_f(row['amp_top1_frac'], '.3f')} "
              f"crest={_f(row['head_crest'], '.2f')} "
              f"errOut={_f(row['head_err_out_frac'])}", flush=True)
    return res


# ---------------------------------------------------------------- mode=causal
def run_causal(cfg, sd, seed_name, mc, res):
    """head 级 V_SAT 扫描。模型函数不变，只改 head 那一级削顶强度。

    ★ 全部用**配对口径**：同一 MC 种子下逐次求差，公共噪声抵消。
    绝对 NRMS 的 mc 间波动达 ±11%，用聚合值相减会得出随机符号（见 p4_paircheck）。
    """
    xt, yt = _seg("seg0")       # 与 P3 / P1 同输入，便于横向对照

    dig_mc, base_mc, native_mc = [], [], {}
    for i in range(mc):
        pd, _, _ = run_once(sd, cfg, xt, 2000 + i, DIALECT, bypass={"ALL"})
        dig_mc.append(V.nrms_of(pd, yt))
        p, h, _ = run_once(sd, cfg, xt, 2000 + i, DIALECT, stats=True,
                           vsat_head=V_SAT, keep_terms=True)
        base_mc.append(V.nrms_of(p, yt))
        if i == mc - 1:
            base_head = h

    slot = res.setdefault("causal", {}).setdefault(seed_name, {})
    slot["floor_mc"] = [float(x) for x in dig_mc]
    slot["base_mc"] = [float(x) for x in base_mc]
    slot["native"] = {}

    base_arr = np.asarray(base_mc)
    dig_arr = np.asarray(dig_mc)
    # 逐 MC 的「总退化」配对差（base - floor）
    tot_d = base_arr - dig_arr
    slot["total_drop_mc"] = [float(x) for x in tot_d]
    slot["total_drop_mean"] = float(tot_d.mean())
    slot["total_drop_std"] = float(tot_d.std(ddof=1))

    for vh in VSAT_HEADS:
        vals, heads = [], []
        for i in range(mc):
            p, h, _ = run_once(sd, cfg, xt, 2000 + i, DIALECT, stats=True,
                               vsat_head=vh, keep_terms=True)
            vals.append(V.nrms_of(p, yt)); heads.append(h)
        v = np.asarray(vals)
        key = f"v{vh:.0f}"
        d = base_arr - v                    # 配对：相比 V_head=16 改善多少
        row = {"vsat_head": vh, "mc_native": [float(x) for x in vals],
               "nrms_native_rms": _agg(vals),
               "recov_pair_mean": float(d.mean()),
               "recov_pair_std": float(d.std(ddof=1)),
               "recov_pair_sem": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
               "n_improved": int((d > 0).sum()), "n": int(len(d))}
        # 回收率 = 配对改善 / 配对总退化（逐 MC 求比值再聚合，避免分母为 0 的极端）
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(np.abs(tot_d) > 1e-12, d / tot_d, np.nan)
        row["recovery_mean"] = float(np.nanmean(r)) if np.any(np.isfinite(r)) else None
        row["recovery_std"] = float(np.nanstd(r, ddof=1)) if np.any(np.isfinite(r)) else None
        # 以聚合总退化作分母的版本（与 P3 的 head/ALL 口径可比）
        denom_agg = float(np.sqrt((base_arr ** 2).mean()) - np.sqrt((dig_arr ** 2).mean()))
        row["recovery_agg"] = (float(d.mean()) / denom_agg) if abs(denom_agg) > 1e-12 else None
        row["denom_agg"] = denom_agg
        hh = heads[-1]
        for k in ("head_err_out_frac", "head_crest", "head_rms_over_vsat",
                  "head_sat_frac", "max_err_out_any", "max_err_out_other",
                  "amp_top1_frac", "phase_coh"):
            row[k] = hh.get(k)
        slot["native"][key] = row
        def _f(x, spec=".4f"):
            return format(x, spec) if x is not None else "-"
        print(f"  [causal] {seed_name:5} V_head={vh:>6.0f} "
              f"Δ配对={row['recov_pair_mean']:+.5f}±{row['recov_pair_std']:.5f} "
              f"改善{row['n_improved']}/{row['n']} "
              f"recov(x̄)={_f(row['recovery_mean'], '.3f')} "
              f"errOut={_f(row['head_err_out_frac'])} "
              f"crest={_f(row['head_crest'], '.2f')}", flush=True)

    # 汇总标量（供 summary 用）
    slot["base_agg"] = float(np.sqrt((base_arr ** 2).mean()))
    slot["floor_agg"] = float(np.sqrt((dig_arr ** 2).mean()))
    slot["top1"] = slot["native"]["v16"].get("amp_top1_frac")
    return res


# ---------------------------------------------------------------- mode=flat（作废反例）
def run_flat(cfg, sd, seed_name, mc, res):
    xt, yt = _seg("seg0")
    _, h0, terms0 = run_once(sd, cfg, xt, 2000, DIALECT, stats=True, keep_terms=True)
    rms0 = np.array(h0["term_rms"], dtype=float)
    sum0 = float(torch.stack([t.reshape(1, -1) for t in terms0]).sum(dim=0)
                 .pow(2).mean().sqrt())
    gmean = float(np.exp(np.log(rms0).mean()))
    slot = res.setdefault("flat", {}).setdefault(
        seed_name, {"rms0": rms0.tolist(), "sum0": sum0, "theta": {}})

    for th in THETAS:
        s = (gmean / rms0) ** th
        s = s * (sum0 / max(float(torch.stack(
            [torch.tensor(float(si)) * t.reshape(1, -1)
             for si, t in zip(s, terms0)]).sum(dim=0).pow(2).mean().sqrt()), 1e-12))
        devv, dig, heads = [], [], []
        for i in range(mc):
            # flat 前向必须走带 scales 的专用版本
            p, h, _ = _run_flat_once(sd, cfg, xt, s.tolist(), 2000 + i)
            devv.append(V.nrms_of(p, yt))
            # 关键：flat 模式下 scales 也进理想线性路径（h = Σ s_i z_i），
            # 所以 digital 参考必须是「带同一 scales + 全 gilbert 旁路」，
            # 不能复用 θ=0 的结果
            pd, _, _ = _run_flat_once(sd, cfg, xt, s.tolist(), 2000 + i, pure=True)
            dig.append(V.nrms_of(pd, yt))
            heads.append(h)
        row = {"theta": th, "scales": s.tolist(),
               "nrms_device": _agg(devv), "nrms_digital": _agg(dig)}
        row["drop"] = row["nrms_device"] - row["nrms_digital"]
        for k in ("amp_top1_frac", "head_err_out_frac", "head_crest"):
            row[k] = heads[-1].get(k)
        slot["theta"][f"t{th:.2f}"] = row
        print(f"  [flat] {seed_name:5} θ={th:.2f} dev={row['nrms_device']:.4f} "
              f"dig={row['nrms_digital']:.4f} drop={row['drop']:+.4f} "
              f"top1={row['amp_top1_frac']:.3f}", flush=True)
    return res


def _run_flat_once(sd, cfg, xt, scales, mc_seed, pure=False):
    """flat 干预的前向。pure=True 时把所有 gilbert 换成理想线性求和（digital 参考）。"""
    global base_p
    A.P.clear(); A.P.update(base_p); A.P.update(DIALECT)
    try:
        dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(mc_seed))
        _gmap, ng, _nL, head_idx = _layout_n(cfg, sd)
        orig_g = dev._gilbert
        sink, cnt = {}, {"i": 0}

        def g_wrapped(prod):
            i = cnt["i"]; cnt["i"] = i + 1
            if pure:
                return prod.sum(dim=2)
            sink[i] = P3.gilbert_stats(prod[..., :16384], dev.p["V_SAT"], dev.p["BETA_H3"])
            return orig_g(prod)

        dev._gilbert = g_wrapped
        with torch.no_grad():
            pred, terms = _forward_terms(dev, cfg, xt, scales)
        dev._gilbert = orig_g
        if cnt["i"] != ng:
            raise RuntimeError(f"[layout] 预测 {ng} 次 gilbert，实际 {cnt['i']} 次")
        h = {} if pure else _head_stats(
            [t.detach().reshape(-1).clone() for t in terms], sink, head_idx)
        return pred, h, []
    finally:
        A.P.clear(); A.P.update(base_p)


# ---------------------------------------------------------------- 汇总
def summarize(res, args):
    if "proxy" in res:
        rows = [v for sd in res["proxy"].values() for v in sd.values()]
        keys = ("amp_top1_frac", "head_err_out_frac", "head_crest",
                "max_err_out_any", "amp_entropy", "phase_coh")
        s = {"n_pooled": len(rows), "n_per_seg": {},
             "pooled": {}, "per_seg": {k: {} for k in keys}, "rows": rows}
        for k in keys:
            s["pooled"][k] = _corr([r[k] for r in rows], [r["drop"] for r in rows])
        # 每段单独做，判据：三段都成立才算非伪影
        for seg_name, _t0 in SEGMENTS:
            sel = [r for r in rows if r["seg"] == seg_name]
            s["n_per_seg"][seg_name] = len(sel)
            for k in keys:
                s["per_seg"][k][seg_name] = _corr([r[k] for r in sel],
                                                  [r["drop"] for r in sel])
        # ratio 口径（与 P1 的「退化比」对齐）
        s["pooled_ratio"] = {
            k: _corr([r[k] for r in rows], [r["ratio"] for r in rows]) for k in keys}
        res["proxy_summary"] = s

    if "causal" in res:
        vs = [f"v{vh:.0f}" for vh in VSAT_HEADS]
        seeds = {}
        for k, v in res["causal"].items():
            seeds[k] = {
                "base_agg": v["base_agg"], "floor_agg": v["floor_agg"],
                "total_drop_mean": v["total_drop_mean"],
                "total_drop_std": v["total_drop_std"],
                "top1": v.get("top1"),
                "recovery_agg": {key: v["native"][key].get("recovery_agg") for key in vs},
                "recovery_mean": {key: v["native"][key].get("recovery_mean") for key in vs},
                "recov_pair": {key: (v["native"][key]["recov_pair_mean"],
                                     v["native"][key]["recov_pair_std"]) for key in vs},
                "n_improved": {key: v["native"][key]["n_improved"] for key in vs},
            }
        c = {"vsat_heads": VSAT_HEADS, "seeds": seeds}

        # 平均 recovery 曲线（用聚合口径，和 P3 的 head/ALL 可比）
        mean_curve, sd_curve = [], []
        for key in vs:
            vals = [x["recovery_agg"][key] for x in seeds.values()
                    if x["recovery_agg"].get(key) is not None]
            mean_curve.append(float(np.mean(vals)) if vals else None)
            sd_curve.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else None)
        c["mean_recovery_agg"] = mean_curve
        c["std_recovery_agg"] = sd_curve

        # 跨种子：top1 越高，head 的贡献应该越大？
        tops = [x["top1"] for x in seeds.values() if x["top1"] is not None]
        d50 = [seeds[k]["recovery_agg"]["v48"] for k in seeds
               if seeds[k]["recovery_agg"].get("v48") is not None]
        c["top1_vs_recovery_at48"] = _corr(tops, d50)
        c["top1_vs_total_drop"] = _corr(
            [x["top1"] for x in seeds.values() if x["top1"] is not None],
            [x["total_drop_mean"] for x in seeds.values() if x["top1"] is not None])
        res["causal_summary"] = c

    if "flat" in res:
        drops = [[res["flat"][s]["theta"][f"t{th:.2f}"]["drop"] for th in THETAS]
                 for s in res["flat"]]
        digs = [[res["flat"][s]["theta"][f"t{th:.2f}"]["nrms_digital"] for th in THETAS]
                for s in res["flat"]]
        res["flat_summary"] = {
            "theta": THETAS,
            "drop_by_theta_mean": [float(np.mean([d[i] for d in drops])) for i in range(len(THETAS))],
            "dig_by_theta_mean": [float(np.mean([d[i] for d in digs])) for i in range(len(THETAS))],
            "note": "作废：dig 自身随 θ 崩塌，drop 的差值无因果意义",
        }


def main():
    global base_p
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["proxy", "causal", "flat", "all"], default="all")
    ap.add_argument("--seeds", nargs="*", default=SEEDS)
    ap.add_argument("--mc", type=int, default=4)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    base_p = dict(A.P)

    print(f"[device] {DEV} torch={torch.__version__} "
          f"hip={getattr(torch.version, 'hip', None)}", flush=True)
    if DEV == "cuda":
        print(f"[device] {torch.cuda.get_device_name(0)}", flush=True)

    res = {"meta": {"dev": DEV, "torch": torch.__version__,
                    "hip": str(getattr(torch.version, "hip", None)),
                    "mc": args.mc, "v_sat": V_SAT, "dialect": DIALECT,
                    "segments": SEGMENTS, "seg_len": SEG_LEN,
                    "vsat_heads": VSAT_HEADS, "thetas": THETAS,
                    "tag": args.tag, "time": time.strftime("%Y-%m-%d %H:%M:%S")}}
    t0 = time.time()
    checked = set()
    for s in args.seeds:
        path = V.nam_path(s)
        cfg, sd = V.load_nam(path)
        print(f"\n=== {s} ({path})", flush=True)
        if s not in checked:
            xt_probe, _ = _seg("seg0")
            dmax = assert_equiv(cfg, sd, xt_probe[:8192])
            print(f"  [selfcheck] 等价 max|Δ|={dmax:.2e} ✓", flush=True)
            checked.add(s)
        if args.mode in ("proxy", "all"):
            run_proxy(cfg, sd, s, args.mc, res)
        if args.mode in ("causal", "all"):
            run_causal(cfg, sd, s, args.mc, res)
        if args.mode in ("flat", "all"):
            run_flat(cfg, sd, s, min(args.mc, 2), res)

    summarize(res, args)
    tag = args.tag or time.strftime("%m%d_%H%M")
    outp = OUT / f"p4_proxy_{tag}.json"
    json.dump(res, open(outp, "w"), indent=1, ensure_ascii=False)
    print(f"\n[saved] {outp}  耗时 {time.time()-t0:.0f}s", flush=True)

    ps = res.get("proxy_summary")
    if ps:
        print(f"\n--- proxy: pooled n={ps['n_pooled']}（判据看 per_seg 三段是否都成立）---")
        for k in ("amp_top1_frac", "head_err_out_frac", "head_crest", "max_err_out_any"):
            v = ps["pooled"][k]
            if v["r"] is None:
                print(f"  {k:<20} n/a")
                continue
            segs = "  ".join(
                f"{sg}:{(ps['per_seg'][k][sg]['rho'] if ps['per_seg'][k][sg]['rho'] is not None else float('nan')):+.2f}"
                for sg, _ in SEGMENTS)
            print(f"  {k:<20} pooled r={v['r']:+.3f} rho={v['rho']:+.3f} | per-seg rho: {segs}")

    cs = res.get("causal_summary")
    if cs:
        vs = [f"v{vh:.0f}" for vh in cs["vsat_heads"]]
        hdr = ("%-6s%7s%10s" % ("seed", "top1", "总Δ配对")
               + "".join("%10s" % ("V" + ("inf" if vh >= 1e5 else "%.0f" % vh))
                         for vh in cs["vsat_heads"]))
        print("\n--- causal: head V_SAT 扫描，回收率 = 配对改善 / 总退化 ---")
        print("  （V16 是现状基线=0；Vinf 即 P3 的 head 全旁路。1.0=退化全在 head）")
        print("  " + hdr)
        for name, x in cs["seeds"].items():
            t1 = x["top1"]
            row = "  %-6s%7s%10.4f" % (
                name, "%.3f" % t1 if t1 is not None else "-", x["total_drop_mean"])
            for key in vs:
                r = x["recovery_agg"].get(key)
                row += "%10s" % ("-" if r is None else "%.2f" % r)
            print(row)
        print("  %-6s%7s%10s" % ("mean", "", "")
              + "".join("%10s" % ("-" if m is None else "%.2f" % m)
                        for m in cs["mean_recovery_agg"]))
        print("  %-6s%7s%10s" % ("std", "", "")
              + "".join("%10s" % ("-" if m is None else "%.2f" % m)
                        for m in cs["std_recovery_agg"]))

        print("\n  head err_out_frac 随 V_head 单调降（因果闭环的证据）：")
        for name in cs["seeds"]:
            eo = [res["causal"][name]["native"][k].get("head_err_out_frac") for k in vs]
            print("    %-6s" % name
                  + "".join("%10s" % ("-" if v is None else "%.4f" % v) for v in eo))

        print("\n  配对改善的符号一致性（多少 MC 次里提高 V_head 真的变好）：")
        for name in cs["seeds"]:
            ni = [cs["seeds"][name]["n_improved"][k] for k in vs]
            n = res["causal"][name]["native"]["v16"]["n"]
            print("    %-6s" % name
                  + "".join("%10s" % ("%d/%d" % (v, n) if v is not None else "-")
                            for v in ni))

        c48 = cs.get("top1_vs_recovery_at48") or {}
        if c48.get("rho") is not None:
            print("\n  top1 vs recovery@V48 : r=%+.3f rho=%+.3f (n=%d)"
                  % (c48["r"], c48["rho"], c48["n"]))
        ctd = cs.get("top1_vs_total_drop") or {}
        if ctd.get("rho") is not None:
            print("  top1 vs 总退化配对Δ   : r=%+.3f rho=%+.3f (n=%d)"
                  % (ctd["r"], ctd["rho"], ctd["n"]))

    fs = res.get("flat_summary")
    if fs:
        print("\n--- flat（已作废，仅存档反例：注意 dig 自身随 θ 崩塌）---")
        print("  theta      " + "".join("%10.2f" % t for t in fs["theta"]))
        print("  dig_mean   " + "".join("%10.4f" % v for v in fs["dig_by_theta_mean"]))
        print("  drop_mean  " + "".join("%10.4f" % v for v in fs["drop_by_theta_mean"]))


if __name__ == "__main__":
    main()
