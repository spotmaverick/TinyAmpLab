"""P-1 一体化实验（z100 远程版，自包含）

回答一个问题：head 累加链的加法器失真，在 L10-ch32-bigRF 上到底占多少？
以及「串行累加改树形累加」能不能降低这份失真。

## 三个子实验
  1. AMP   : head 累加各级幅度谱（串行 h_i 与树形中间节点），看相干增益与末级工作域
  2. SWEEP : 加法器摆幅 V_add 扫描，找 Δ_add<1.05 的门槛 V*，与 V_SAT=16 比较
  3. TOPO  : 固定 V_add 在若干代表值，串行 vs 树形 vs 理想 的 NRMS 对比

## 关键设计（与上一轮 P-1 草案的修正）
- 加法器是一个独立器件，它有**自己的**饱和摆幅 V_add，不应强行复用乘法器的 V_SAT。
  上一轮把 V_add 直接设成 16 得出的 "Δ=6.6×" 是摆幅失配的产物，不是拓扑问题。
- 树形累加的收益只在「节点幅度峰值」下降时出现。若末级节点必须承受全部和，
  树形对饱和失真天然无效 —— 这一点必须用实测幅度证明，不能靠直觉。

## 用法
  python p1_run.py                 # 全部：AMP + SWEEP + TOPO
  python p1_run.py --skip-sweep
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__import__("os").environ.get(
    "P1_ROOT", "/work/home/spotmaverick/nam_project"))
if not ROOT.exists():                       # 本地跑时回退到脚本所在目录
    ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "local_mirrors" / "neural-amp-modeler"))
sys.path.insert(0, str(ROOT / "github_mirrors" / "neural-amp-modeler"))
# 私有依赖目录：--target 安装的 numpy/scipy/soundfile，不污染集群共享环境。
# 按解释器版本只挂对应那一份，绝不能同时挂上去——两份 numpy 会互相踩，
# 表现为 ImportError: No module named 'numpy.core._multiarray_umath'
_pl = "pylibs_p310" if sys.version_info[:2] == (3, 10) else "pylibs_p38"
_p = ROOT / _pl
if _p.exists():
    sys.path.insert(0, str(_p))
# 极简 stub 目录（librosa/wavio 等），必须最高优先级：
# nam.__init__ 会无条件 import data.py，而 P-1 从 .nam JSON 直接读权重，不走音频 IO
sys.path.insert(0, str(ROOT / "stubs"))

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
import soundfile as sf                                          # noqa: E402
import analog_sim_generic as A                                   # noqa: E402
from analog_sim_generic import DevSim                           # noqa: E402

# 强制走加速卡：P1_REQUIRE_ACCEL=1 时若无可用 DCU/GPU 直接失败，不静默回落 CPU
_REQ = __import__("os").environ.get("P1_REQUIRE_ACCEL", "0") == "1"
if torch.cuda.is_available():
    DEV = "cuda"
elif _REQ:
    raise SystemExit("P1_REQUIRE_ACCEL=1 但当前无可用加速卡，拒绝在 CPU 上运行")
else:
    DEV = "cpu"
print(f"[device] {DEV}  torch={torch.__version__}  "
      f"hip={getattr(torch.version, 'hip', None)}  "
      f"count={torch.cuda.device_count() if DEV == 'cuda' else 0}", flush=True)
if DEV == "cuda":
    print(f"[device] {torch.cuda.get_device_name(0)}", flush=True)
DATA = ROOT / "data"
SEED_DIR = ROOT / "p1" / "seeds" if (ROOT / "p1" / "seeds").exists() else ROOT
OUT = ROOT / "p1" / "out" if (ROOT / "p1").exists() else ROOT
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = ["s7", "s11", "s23", "s31", "s37"]
PREV_NRMS = {"s7": 0.0894, "s11": 0.2155, "s23": 0.1259, "s31": 0.2376, "s37": 0.1393}

LOCAL_NAMS = {
    "s7":  "reclaimed/unpack_20261126/runs/n12_L10-ch32-bigRF_s7/*/model.nam",
    "s11": "reclaimed/seeds_20261126/runs/sd_L10-ch32-bigRF_s11/*/model.nam",
    "s23": "reclaimed/seeds_20261126/runs/sd_L10-ch32-bigRF_s23/*/model.nam",
    "s31": "reclaimed/seeds_20261126/runs/sd_L10-ch32-bigRF_s31/*/model.nam",
    "s37": "reclaimed/seeds_20261126/runs/sd_L10-ch32-bigRF_s37/*/model.nam",
}


def nam_path(seed):
    """优先用 p1/seeds/<seed>.nam（远端），否则在本地 reclaimed 里找。"""
    p = SEED_DIR / f"{seed}.nam"
    if p.exists():
        return str(p)
    import glob
    hits = sorted(glob.glob(str(ROOT / LOCAL_NAMS[seed])))
    if not hits:
        raise FileNotFoundError(f"{seed}: {p} 与 {LOCAL_NAMS[seed]} 都不存在")
    return hits[-1]


# ---------------- 模型加载 ----------------
def load_nam(path):
    from nam.models._from_nam import init_from_nam
    raw = json.load(open(path))
    net = init_from_nam(raw)
    sd = {}
    for k, v in net.state_dict().items():
        k2 = k
        while k2.startswith("_net."):
            k2 = k2[len("_net."):]
        if k2 in ("_has_sample_rate", "_sample_rate", "_has_sample"):
            continue
        sd[k2] = v.detach().cpu()
    L = dict(raw["config"]["layers"][0])
    ks = L.pop("kernel_sizes")
    L["kernel_size"] = ks[0]
    l1 = L.pop("layer1x1", None)
    if l1 is not None:
        L["layer_1x1_config"] = l1
    act = L.get("activation")
    if isinstance(act, list):
        L["activation"] = act[0]["type"]
    L.pop("gating_mode", None)
    L.pop("secondary_activation", None)
    return {"layers_configs": [L], "head_scale": raw["config"].get("head_scale", 0.01)}, sd


# ---------------- 核心前向：可插拔累加拓扑 ----------------
def accum(terms, topo, f):
    if topo == "ideal":
        h = terms[0]
        for t in terms[1:]:
            h = h + t
        return h
    if topo == "serial":
        h = terms[0]
        for t in terms[1:]:
            h = f(h + t)
        return h
    if topo == "tree":
        lvl = list(terms)
        while len(lvl) > 1:
            nxt = [f(lvl[i] + lvl[i + 1]) for i in range(0, len(lvl) - 1, 2)]
            if len(lvl) % 2:
                nxt.append(lvl[-1])
            lvl = nxt
        return lvl[0]
    raise ValueError(topo)


def run_forward(dev, sd, cfg, x, topo="ideal", f=None, collect=False):
    """与 analog_sim_generic.sim_forward 等价，但 head 累加可换拓扑、可收集中间幅度。"""
    lc = cfg["layers_configs"][0]
    K, dils = lc["kernel_size"], lc["dilations"]
    act = lc.get("activation", "Tanh")
    g_in = lc.get("groups_input", 1)
    l1 = lc.get("layer_1x1_config", {}) or {}
    g_l1 = l1.get("groups", 1) if l1.get("active") else 1
    head_scale = cfg.get("head_scale", 0.01)

    y = x[None, None, :] if x.ndim == 1 else x[:, None, :]
    cond = y
    rf = 1 + sum((K - 1) * d for d in dils)
    out_len = y.shape[2] - (rf - 1)
    pre0 = "_layer_arrays.0."
    # 必须从 dev.w 取权重：DevSim 已按 DEVICE 搬好并做过量化/失配注入，
    # 直接用外部 sd（CPU 上的原始浮点）会导致 cuda/cpu 混算
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
        terms.append(z[:, :, -out_len:])

    f = f or (lambda v: v)
    head_in = accum(terms, topo, f)
    out = dev.conv(head_in, W[pre0 + "_head_rechannel.weight"],
                   W.get(pre0 + "_head_rechannel.bias"), 1, 1)
    res = (head_scale * out)[:, 0, :]
    if collect:
        rms = lambda t: float(t.pow(2).mean().sqrt())
        info = {"a_rms": [rms(t) for t in terms]}
        h = terms[0]
        ser = [rms(h)]
        for t in terms[1:]:
            h = h + t
            ser.append(rms(h))
        info["serial_rms"] = ser
        lvl = list(terms); tree = []
        while len(lvl) > 1:
            nxt = []
            for i in range(0, len(lvl) - 1, 2):
                v = lvl[i] + lvl[i + 1]; nxt.append(v); tree.append(rms(v))
            if len(lvl) % 2:
                nxt.append(lvl[-1])
            lvl = nxt
        info["tree_rms"] = tree
        fin = accum(terms, "ideal", None)
        info["head_in_rms"] = rms(fin)
        info["head_in_absmax"] = float(fin.abs().max())
        info["head_in_p99"] = float(fin.abs().quantile(0.99))
        return res, info
    return res


def nrms_of(pred, tgt):
    off = tgt.numel() - pred.numel()
    t = tgt[off:].double()
    return float((pred.double() - t).norm() / t.norm())


def make_adder(vadd, beta, sigma_off, gen):
    off = [None]

    def f(v):
        if off[0] is None:
            off[0] = torch.empty((1, v.shape[1], 1), device=v.device).normal_(
                0, sigma_off, generator=gen)
        x = v + off[0]
        xs = x / vadd
        return vadd * torch.tanh(x * (1 + beta * xs.square()) / vadd)

    return f


# ---------------- 主体 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", default=SEEDS)
    ap.add_argument("--mc", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    xi, _ = sf.read(str(DATA / "tsa15_test_input.wav"), dtype="float32")
    yi, _ = sf.read(str(DATA / "tsa15_test_target.wav"), dtype="float32")
    n = int(48000 * args.seconds)
    xt = torch.from_numpy(xi[:n]).to(DEV)
    yt = torch.from_numpy(yi[:n]).to(DEV)
    base = dict(A.P)
    print(f"[P-1] dev={DEV} mc={args.mc} sec={args.seconds} torch={torch.__version__}", flush=True)

    result = {"env": {"dev": DEV, "mc": args.mc, "seconds": args.seconds,
                      "torch": torch.__version__}, "seeds": {}}

    for s in args.seeds:
        cfg, sd = load_nam(nam_path(s))
        rec = {"seed": s}

        # --- 1. 幅度谱（理想算子，排除噪声干扰）---
        dev0 = DevSim(sd, torch.Generator(device=DEV).manual_seed(0), sanity=True)
        with torch.no_grad():
            p_id, info = run_forward(dev0, sd, cfg, xt, "ideal", None, collect=True)
        rec["amp"] = info
        rec["nrms_ideal"] = nrms_of(p_id, yt)

        # --- 2. 器件基线（head 累加理想）---
        ser_base, tree_base = [], []
        for i in range(args.mc):
            for topo, bag in (("serial", ser_base), ("tree", tree_base)):
                A.P.clear(); A.P.update(base)
                dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(1000 + i))
                with torch.no_grad():
                    pr = run_forward(dev, sd, cfg, xt, topo, None)
                bag.append(nrms_of(pr, yt))
        A.P.clear(); A.P.update(base)
        b_ser = float(np.sqrt((np.array(ser_base) ** 2).mean()))
        b_tree = float(np.sqrt((np.array(tree_base) ** 2).mean()))
        rec["base"] = {"serial": b_ser, "tree": b_tree, "prev": PREV_NRMS.get(s),
                       "drift": b_ser / PREV_NRMS[s] if s in PREV_NRMS else None}
        print(f"\n[{s}] ideal={rec['nrms_ideal']:.4f} base(serial)={b_ser:.4f} "
              f"prev={PREV_NRMS.get(s)} drift={rec['base']['drift']:.3f}x "
              f"head_in_rms={info['head_in_rms']:.2f} max={info['head_in_absmax']:.1f}", flush=True)

        # --- 3. 摆幅扫描 ---
        if not args.skip_sweep:
            grid = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
            rows = []
            for va in grid:
                row = {"vadd": va}
                for topo in ("serial", "tree"):
                    vals = []
                    for i in range(args.mc):
                        A.P.clear(); A.P.update(base)
                        dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(1000 + i))
                        f = make_adder(va, 0.002, 0.005,
                                       torch.Generator(device=DEV).manual_seed(5000 + i))
                        with torch.no_grad():
                            pr = run_forward(dev, sd, cfg, xt, topo, f)
                        vals.append(nrms_of(pr, yt))
                    A.P.clear(); A.P.update(base)
                    m = float(np.sqrt((np.array(vals) ** 2).mean()))
                    row[f"{topo}"] = m
                    row[f"{topo}_d"] = m / (b_ser if topo == "serial" else b_tree)
                rows.append(row)
                print(f"    V_add={va:4d}  ser Δ={row['serial_d']:5.3f}  tree Δ={row['tree_d']:5.3f}",
                      flush=True)
            rec["sweep"] = rows
            thr = next((r["vadd"] for r in rows if r["serial_d"] < 1.05), None)
            rec["threshold_vadd"] = thr
            print(f"    → 门槛 V_add={thr} ({thr/16:.0f}×V_SAT)" if thr else "    → 门槛未达 >512")

        result["seeds"][s] = rec
        json.dump(result, open(OUT / f"p1{args.tag}.json", "w"), indent=1, ensure_ascii=False)

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 74)
    print("A. head 累加幅度谱（理想算子）")
    print(f"{'seed':>4} {'a_末':>7} {'h_末':>8} {'|max|':>7} {'p99':>7} {'相干':>6} {'h_末/16':>8}")
    for s in args.seeds:
        a = result["seeds"][s]["amp"]
        coh = a["head_in_rms"] / max(float(np.sqrt(np.sum(np.array(a["a_rms"]) ** 2))), 1e-12)
        print(f"{s:>4} {a['a_rms'][-1]:7.2f} {a['head_in_rms']:8.2f} {a['head_in_absmax']:7.1f} "
              f"{a['head_in_p99']:7.2f} {coh:6.2f} {a['head_in_rms']/16:8.2f}")

    print("\nB. 加法器摆幅门槛（Δ_serial<1.05）")
    for s in args.seeds:
        r = result["seeds"][s]
        thr = r.get("threshold_vadd")
        print(f"  {s:>4}  基线={r['base']['serial']:.4f}  门槛V_add={thr}  "
              f"{('=%.0f×V_SAT' % (thr/16)) if thr else '(>512)'}")

    print("\nC. 树形 vs 串行（各摆幅下的 Δ 差）")
    for s in args.seeds:
        r = result["seeds"][s]
        if "sweep" not in r:
            continue
        gains = [row["tree_d"] / row["serial_d"] for row in r["sweep"]]
        print(f"  {s:>4}  tree/serial Δ 比 = {min(gains):.3f}~{max(gains):.3f}  "
              f"中位 {np.median(gains):.3f}  ({'树形有收益' if np.median(gains) < 0.95 else '树形无收益'})")

    print("\nD. 仿真器漂移校验（本次/上轮，应≈1.00）")
    dr = [result["seeds"][s]["base"]["drift"] for s in args.seeds
          if result["seeds"][s]["base"]["drift"]]
    print(f"  中位 {np.median(dr):.3f}x   范围 {min(dr):.3f}~{max(dr):.3f}")

    json.dump(result, open(OUT / f"p1{args.tag}.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {OUT}/p1{args.tag}.json")


if __name__ == "__main__":
    main()
