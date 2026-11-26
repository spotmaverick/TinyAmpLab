"""P-1：head 累加链加法器失真的零训练成本实验

## 要回答的问题
`analog_sim_generic.sim_forward` 里 `head_in = head_in[:, :, -L:] + term` 这行
是无噪声、无失真的理想加法。但真实模拟硬件里 head 累加是 L 级（本例 L=10）
级联的 Gilbert 电流求和，每一级都会引入 tanh 饱和 + 三次失真。
如果这个失真占比可观，那么「串行累加 → 树形累加」立刻有价值；
如果占比 <5%，这条路线作废。

## 判据（预先登记，避免事后找解释）
  ratio_serial   = NRMS(串行累加 + 加法失真) / NRMS(全理想)
  delta_add      = NRMS(仅加法失真) / NRMS(无加法失真)   ← 主判据
    delta_add < 1.05  → 加法失真 <5%，树形累加作废
    delta_add > 1.20  → 加法失真 >20%，树形累加进 P1
    中间地带          → 需结合 tail 分析判断

## 三种累加拓扑（同一组权重、同一 MC 种子）
  serial : h_1 = f(a_1); h_i = f(h_{i-1} + a_i)        串行，失真路径深度 L
  tree   : 两两配对 + 奇数项直通                       失真路径深度 ceil(log2 L)
  ideal  : 纯浮点加和                                  基线

## 用法
  python p1_head_accum.py --mc 8                 # 全部 5 种子
  python p1_head_accum.py --mc 8 --seeds s7 s31
  python p1_head_accum.py --mc 8 --head-vsat 16  # 指定加法器摆幅
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "github_mirrors" / "neural-amp-modeler"))

import analog_sim_generic as A          # noqa: E402  复用器件模型
from analog_sim_generic import DevSim, quant_sym   # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DATA = ROOT / "data"

# ---- 五个种子的 model.nam 定位（本地 reclaimed 目录） ----
SEED_PATHS = {
    "s7":  "reclaimed/unpack_20261126/runs/n12_L10-ch32-bigRF_s7/*/model.nam",
    "s11": "reclaimed/seeds_20261126/runs/sd_L10-ch32-bigRF_s11/*/model.nam",
    "s23": "reclaimed/seeds_20261126/runs/sd_L10-ch32-bigRF_s23/*/model.nam",
    "s31": "reclaimed/seeds_20261126/runs/sd_L10-ch32-bigRF_s31/*/model.nam",
    "s37": "reclaimed/seeds_20261126/runs/sd_L10-ch32-bigRF_s37/*/model.nam",
}

# 上一次（无加法失真）的官方口径数值，用于校准本次仿真器是否跑偏
BASELINE_NRMS = {"s7": 0.0894, "s11": 0.2155, "s23": 0.1259, "s31": 0.2376, "s37": 0.1393}


def resolve_nam(seed):
    import glob
    pat = SEED_PATHS[seed]
    hits = sorted(glob.glob(str(ROOT / pat)))
    if not hits:
        raise FileNotFoundError(f"找不到 {seed} 的 model.nam: {pat}")
    return hits[-1]


def load_nam(path):
    """加载 .nam，返回 (cfg, state_dict)。state_dict 键名与 analog_sim_generic 对齐。"""
    from nam.models._from_nam import init_from_nam
    raw = json.load(open(path))
    net = init_from_nam(raw)
    cfg = raw["config"]
    sd = {}
    for k, v in net.state_dict().items():
        k2 = k
        while k2.startswith("_net."):
            k2 = k2[len("_net."):]
        if k2 in ("_has_sample_rate", "_sample_rate", "_has_sample"):
            continue        # 标量元数据，不是网络权重
        sd[k2] = v.detach().cpu()
    # .nam 用 kernel_sizes(逐层) / layer1x1(而非 layer_1x1_config)，
    # 这里归一成 analog_sim_generic 期望的 layers_configs 形式
    L = dict(cfg["layers"][0])
    ks = L.pop("kernel_sizes")
    assert len(set(ks)) == 1, f"暂只支持各层同 kernel: {set(ks)}"
    L["kernel_size"] = ks[0]
    l1 = L.pop("layer1x1", None)
    if l1 is not None:
        L["layer_1x1_config"] = l1
    act = L.get("activation")
    if isinstance(act, list):
        assert len({a["type"] for a in act}) == 1, f"各层激活不一致: {act}"
        L["activation"] = act[0]["type"]
    L.pop("gating_mode", None)
    L.pop("secondary_activation", None)
    cfg2 = {"layers_configs": [L], "head_scale": cfg.get("head_scale", 0.01)}
    return cfg2, sd


# ---------------- head 累加拓扑 ----------------
def head_accum_serial(terms, f_add):
    h = terms[0]
    for t in terms[1:]:
        h = f_add(h + t)
    return h


def head_accum_tree(terms, f_add):
    level = list(terms)
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(f_add(level[i] + level[i + 1]))
        if len(level) % 2:
            nxt.append(level[-1])          # 奇数项直通，不加失真
        level = nxt
    return level[0]


def head_accum_ideal(terms, f_add):
    h = terms[0]
    for t in terms[1:]:
        h = h + t
    return h


TOPOS = {"serial": head_accum_serial, "tree": head_accum_tree, "ideal": head_accum_ideal}


def make_adder(p, mode, gen):
    """加法器失真：与 Gilbert cell 同族模型，但摆幅独立（加法器是电流求和节点）。"""
    if mode == "off" or p.get("HEAD_BETA", 0.0) <= 0 and p.get("HEAD_VSAT", 1e9) >= 1e8:
        return lambda v: v

    vs = p["HEAD_VSAT"]
    beta = p["HEAD_BETA"]
    off = None

    def f(v):
        nonlocal off
        if off is None:
            off = torch.empty((1, v.shape[1], 1), device=v.device).normal_(
                0, p.get("HEAD_SIGMA_OFF", 0.0), generator=gen)
        x = v + off
        xs = x / vs
        return vs * torch.tanh(x * (1 + beta * xs.square()) / vs)

    return f


def sim_forward_head(dev, sd, cfg, x, topo, f_add):
    """复刻 analog_sim_generic.sim_forward，但 head 累加换成可插拔拓扑。

    与上游唯一的差异：把所有 layer 的 head term 先收集，最后统一按 topo 合并。
    其它算子（conv / 1x1 / 激活 / 噪声）完全走 DevSim，保证与既有口径可比。
    """
    lc = cfg["layers_configs"][0]
    ch = lc["channels"]
    K = lc["kernel_size"]
    dils = lc["dilations"]
    act = lc.get("activation", "Tanh")
    g_in = lc.get("groups_input", 1)
    l1 = lc.get("layer1x1_config", {}) or {}
    g_l1 = l1.get("groups", 1) if l1.get("active") else 1
    head_scale = cfg.get("head_scale", 0.01)
    nL = len(dils)

    y = x[None, None, :] if x.ndim == 1 else (x[:, None, :] if x.ndim == 2 else x)
    cond = y
    rf_no_head = 1 + sum((K - 1) * d for d in dils)
    out_len_no_head = y.shape[2] - (rf_no_head - 1)

    pre0 = "_layer_arrays.0."
    y = dev.mac1x1(y, sd[pre0 + "_rechannel.weight"], None, groups=1)

    terms = []
    for li, d in enumerate(dils):
        pre = f"{pre0}_layers.{li}"
        z = dev.conv(y, sd[f"{pre}._conv.weight"], sd.get(f"{pre}._conv.bias"), d, g_in)
        mix = dev.conv(cond, sd[f"{pre}._input_mixer.weight"], None, 1, 1)[:, :, -z.shape[2]:]
        n = min(z.shape[2], mix.shape[2])
        z = dev.act(z[:, :, -n:] + mix[:, :, -n:], act)
        z = z + dev.noise(z.shape)
        res_out = dev.conv(z, sd[f"{pre}._layer1x1.weight"], sd.get(f"{pre}._layer1x1.bias"), 1, g_l1)
        y = y[:, :, -res_out.shape[2]:] + res_out + dev.noise(res_out.shape)
        terms.append(z[:, :, -out_len_no_head:])

    head_in = TOPOS[topo](terms, f_add)
    out = dev.conv(head_in, sd[pre0 + "_head_rechannel.weight"], sd.get(pre0 + "_head_rechannel.bias"), 1, 1)
    return (head_scale * out)[:, 0, :]


def nrms_of(pred, tgt):
    off = tgt.numel() - pred.numel()
    t = tgt[off:].double()
    p = pred.double()
    return float((p - t).norm() / t.norm())


def run_seed(seed, args, xt, yt, base_p):
    cfg, sd = load_nam(resolve_nam(seed))
    lc = cfg["layers_configs"][0]
    rf = 1 + sum((lc["kernel_size"] - 1) * d for d in lc["dilations"])

    out = {"seed": seed, "RF": rf, "ch": lc["channels"], "L": len(lc["dilations"]),
           "g_in": lc.get("groups_input", 1), "cfg": str(Path(resolve_nam(seed)).parent)}

    # 1) 理想基线（关全部非理想性）
    dev0 = DevSim(sd, torch.Generator(device=DEV).manual_seed(0), sanity=True)
    with torch.no_grad():
        p_ideal = sim_forward_head(dev0, sd, cfg, xt, "ideal", lambda v: v)
    out["nrms_ideal"] = nrms_of(p_ideal, yt)

    # 2) 器件全非理想 + 三种 head 累加拓扑
    for topo in ["serial", "tree"]:
        vals_no_add, vals_with_add = [], []
        for i in range(args.mc):
            # 无加法失真（复现原口径）
            A.P.clear(); A.P.update(base_p)
            dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(1000 + i))
            with torch.no_grad():
                pr = sim_forward_head(dev, sd, cfg, xt, topo, lambda v: v)
            vals_no_add.append(nrms_of(pr, yt))

            # 加 head 加法失真
            A.P.clear(); A.P.update(base_p)
            dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(1000 + i))
            gen_h = torch.Generator(device=DEV).manual_seed(5000 + i)
            f_add = make_adder(args.p_head, "on", gen_h)
            with torch.no_grad():
                pr2 = sim_forward_head(dev, sd, cfg, xt, topo, f_add)
            vals_with_add.append(nrms_of(pr2, yt))

        a0 = np.array(vals_no_add); a1 = np.array(vals_with_add)
        out[f"{topo}_no_add"] = float(np.sqrt((a0 ** 2).mean()))
        out[f"{topo}_with_add"] = float(np.sqrt((a1 ** 2).mean()))
        out[f"{topo}_delta"] = out[f"{topo}_with_add"] / max(out[f"{topo}_no_add"], 1e-12)
        out[f"{topo}_mc_std"] = float(a1.std())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=8)
    ap.add_argument("--seeds", nargs="*", default=list(SEED_PATHS))
    ap.add_argument("--seconds", type=float, default=3.0, help="评测时长（秒）")
    ap.add_argument("--head-vsat", type=float, default=16.0)
    ap.add_argument("--head-beta", type=float, default=0.002)
    ap.add_argument("--head-offset", type=float, default=0.005)
    ap.add_argument("--out", default="p1_results.json")
    args = ap.parse_args()

    xi, _ = sf.read(str(DATA / "tsa15_test_input.wav"), dtype="float32")
    yi, _ = sf.read(str(DATA / "tsa15_test_target.wav"), dtype="float32")
    n = int(48000 * args.seconds)
    xt = torch.from_numpy(xi[:n]).to(DEV)
    yt = torch.from_numpy(yi[:n]).to(DEV)

    args.p_head = dict(HEAD_VSAT=args.head_vsat, HEAD_BETA=args.head_beta,
                       HEAD_SIGMA_OFF=args.head_offset)
    base_p = dict(A.P)

    print(f"[P-1] dev={DEV} mc={args.mc} seconds={args.seconds} "
          f"add_vsat={args.head_vsat} add_beta={args.head_beta}", flush=True)
    print(f"{'seed':>4} {'ideal':>8} {'ser_no':>8} {'ser_add':>8} {'Δser':>7} "
          f"{'tre_no':>8} {'tre_add':>8} {'Δtre':>7}", flush=True)

    rows = []
    for s in args.seeds:
        r = run_seed(s, args, xt, yt, base_p)
        r["ref_prev"] = BASELINE_NRMS.get(s)
        r["drift_vs_prev"] = (r["serial_no_add"] / r["ref_prev"]) if r["ref_prev"] else None
        rows.append(r)
        print(f"{s:>4} {r['nrms_ideal']:8.4f} {r['serial_no_add']:8.4f} "
              f"{r['serial_with_add']:8.4f} {r['serial_delta']:6.3f}x "
              f"{r['tree_no_add']:8.4f} {r['tree_with_add']:8.4f} "
              f"{r['tree_delta']:6.3f}x", flush=True)

    # 汇总
    ds = np.array([r["serial_delta"] for r in rows])
    dt = np.array([r["tree_delta"] for r in rows])
    summary = {
        "n_seed": len(rows),
        "serial_delta_median": float(np.median(ds)),
        "tree_delta_median": float(np.median(dt)),
        "tree_vs_serial_gain_median": float(np.median(dt / ds)),
        "verdict": ("tree_valuable" if np.median(ds) > 1.20 else
                    "tree_marginal" if np.median(ds) > 1.05 else "tree_drop"),
        "drift_median": float(np.median([r["drift_vs_prev"] for r in rows if r["drift_vs_prev"]])),
        "params": args.p_head | {"mc": args.mc, "seconds": args.seconds},
    }
    json.dump({"summary": summary, "rows": rows},
              open(ROOT / args.out, "w"), indent=1, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    print(f"串行累加 Δ 中位 = {summary['serial_delta_median']:.3f}x "
          f"（加法失真贡献 {(summary['serial_delta_median']-1)*100:.1f}%）")
    print(f"树形累加 Δ 中位 = {summary['tree_delta_median']:.3f}x")
    print(f"树形相对串行收益 = {summary['tree_vs_serial_gain_median']:.3f}x")
    print(f"仿真器漂移（本次/上轮）= {summary['drift_median']:.3f}x  (应≈1.00)")
    print(f"判定: {summary['verdict']}")


if __name__ == "__main__":
    main()
