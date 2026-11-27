"""P1-VSAT：5 种子 × V_SAT 多档扫描（中位口径的 V_SAT 需求曲线）

## 为什么做这个
`TARGET_TOPOLOGY_STATUS.md` §3 的 V_SAT 曲线只有 s7 单种子，是「对外报 L10-ch32 比 A2
优 8 倍」这条结论的**唯一依据**。器件 5 种子中位是 s7 的 1.56 倍，所以中位口径的曲线
必然上移——上移多少、8 倍还剩几倍，必须实测，不能外推。

## 与 p1_run.py 的区别
- `p1_run.py` 扫的是**加法器摆幅 V_add**（head 累加级），回答「加法器要不要独立电源域」
- 本脚本扫的是**乘法器饱和摆幅 V_SAT**（Gilbert cell 工作点），回答「工艺需要多高的供电」
两者是不同器件、不同物理量，不能混。

## 口径
- V_SAT ∈ {16, 24, 32, 48, 64, 96, 128}，共 7 档
- BITS=8、β=0.3、含失配/失调/噪声（与 analog_sim_generic 默认一致）
- 每档 MC 次平均（默认 2，与历史 s7 单种子曲线同口径以便直接对比）
- 参考基准：A2-causal 在同档 V_SAT 下的器件 NRMS（写死在 A2_CURVE，来自历史 measurement）

## 用法
  python p1_vsat_sweep.py --mc 2                 # 5 种子全跑
  python p1_vsat_sweep.py --seeds s7 --mc 1      # 单种子快验
  P1_REQUIRE_ACCEL=1 python p1_vsat_sweep.py     # 强制加速卡，不允许回落 CPU
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("P1_ROOT", "/work/home/spotmaverick/nam_project"))
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# 兼容三种部署：本地 mirrors / 远端 local_mirrors / bw 的 /workspace 布局
for _cand in ("local_mirrors", "github_mirrors"):
    _p = ROOT / _cand / "neural-amp-modeler"
    if _p.exists():
        sys.path.insert(0, str(_p))
# 私有依赖：按解释器版本只挂一份（两份 numpy 会互相踩）
for _v, _d in (((3, 11), "pylibs_p311"), ((3, 10), "pylibs_p310"), ((3, 8), "pylibs_p38")):
    if sys.version_info[:2] == _v and (ROOT / _d).exists():
        sys.path.insert(0, str(ROOT / _d))
if (ROOT / "stubs").exists():
    sys.path.insert(0, str(ROOT / "stubs"))

import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import soundfile as sf                                         # noqa: E402
import analog_sim_generic as A                                 # noqa: E402
from analog_sim_generic import DevSim                          # noqa: E402

_REQ = os.environ.get("P1_REQUIRE_ACCEL", "0") == "1"
if torch.cuda.is_available():
    DEV = "cuda"
elif _REQ:
    raise SystemExit("P1_REQUIRE_ACCEL=1 但无可用加速卡，拒绝在 CPU 上运行")
else:
    DEV = "cpu"
print(f"[device] {DEV}  torch={torch.__version__}  "
      f"hip={getattr(torch.version, 'hip', None)}  "
      f"count={torch.cuda.device_count() if DEV == 'cuda' else 0}", flush=True)
if DEV == "cuda":
    print(f"[device] {torch.cuda.get_device_name(0)}", flush=True)

SEED_DIR = ROOT / "p1" / "seeds" if (ROOT / "p1" / "seeds").exists() else ROOT
OUT = ROOT / "p1" / "out" if (ROOT / "p1").exists() else ROOT
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"

SEEDS = ["s7", "s11", "s23", "s31", "s37"]
# 历史 s7 单种子 V_SAT 曲线（TARGET_TOPOLOGY_STATUS §3.2），用于校验本次口径一致
S7_HIST = {16: 0.0894, 24: 0.0719, 32: 0.0676, 48: 0.0646, 64: 0.0612,
           96: 0.0602, 128: 0.0591}
# A2-causal 官方在同档 V_SAT 下的器件 NRMS（历史 measurement）
A2_CURVE = {16: 0.6076, 24: 0.4464, 32: 0.3341, 48: 0.2205, 64: 0.1653,
            96: 0.1214, 128: 0.1024}
VSATS = [16, 24, 32, 48, 64, 96, 128]

LOCAL_NAMS = {
    "s7":  "reclaimed/unpack_20261126/runs/n12_L10-ch32-bigRF_s7/*/model.nam",
    "s11": "reclaimed/seeds_20261126/runs/sd_L10-ch32-bigRF_s11/*/model.nam",
    "s23": "reclaimed/seeds_20261126/runs/sd_L10-ch32-bigRF_s23/*/model.nam",
    "s31": "reclaimed/seeds_20261126/runs/sd_L10-ch32-bigRF_s31/*/model.nam",
    "s37": "reclaimed/seeds_20261126/runs/sd_L10-ch32-bigRF_s37/*/model.nam",
}


def nam_path(seed):
    p = SEED_DIR / f"{seed}.nam"
    if p.exists():
        return str(p)
    import glob
    hits = sorted(glob.glob(str(ROOT / LOCAL_NAMS[seed])))
    if not hits:
        raise FileNotFoundError(f"{seed}: {p} 与 {LOCAL_NAMS[seed]} 都不存在")
    return hits[-1]


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


def run_forward(dev, cfg, x):
    """理想拓扑（串行浮动累加）的器件前向，head 累加不做额外饱和建模。"""
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
    h = terms[0]
    for t in terms[1:]:
        h = h + t
    out = dev.conv(h, W[pre0 + "_head_rechannel.weight"],
                   W.get(pre0 + "_head_rechannel.bias"), 1, 1)
    return (head_scale * out)[:, 0, :]


def nrms_of(pred, tgt):
    off = tgt.numel() - pred.numel()
    t = tgt[off:].double()
    return float((pred.double() - t).norm() / t.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", default=SEEDS)
    ap.add_argument("--mc", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    xi, _ = sf.read(str(DATA / "tsa15_test_input.wav"), dtype="float32")
    yi, _ = sf.read(str(DATA / "tsa15_test_target.wav"), dtype="float32")
    n = int(48000 * args.seconds)
    xt = torch.from_numpy(xi[:n]).to(DEV)
    yt = torch.from_numpy(yi[:n]).to(DEV)
    base = dict(A.P)

    result = {"meta": {"dev": DEV, "torch": torch.__version__,
                       "hip": str(getattr(torch.version, "hip", None)),
                       "mc": args.mc, "seconds": args.seconds,
                       "vsats": VSATS, "tag": args.tag,
                       "time": time.strftime("%Y-%m-%d %H:%M:%S")},
              "seeds": {}}
    t_all = time.time()

    for seed in args.seeds:
        path = nam_path(seed)
        cfg, sd = load_nam(path)
        ideal = None
        rec = {"nam": path, "vsat": {}}
        for v in VSATS:
            vals = []
            for i in range(args.mc):
                p = dict(base)
                p["V_SAT"] = v
                A.P.clear()
                A.P.update(p)
                dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(1000 + i))
                with torch.no_grad():
                    pred = run_forward(dev, cfg, xt)
                if ideal is None:
                    # 数字基准：同一前向、V_SAT 拉到无穷大
                    p2 = dict(base)
                    p2["V_SAT"] = 1e9
                    A.P.clear()
                    A.P.update(p2)
                    d2 = DevSim(sd, torch.Generator(device=DEV).manual_seed(1000 + i))
                    with torch.no_grad():
                        ideal = nrms_of(run_forward(d2, cfg, xt), yt)
                vals.append(nrms_of(pred, yt))
            # MC 聚合口径与 p1_run.py 保持一致：RMS（不是算术平均），
            # 否则与历史 s7 单种子曲线对不上（实测差 14%）
            m = float(np.sqrt((np.array(vals) ** 2).mean()))
            rec["vsat"][v] = {"nrms": m, "mc_vals": vals,
                              "retio": m / ideal if ideal else None}
            print(f"  {seed} V_SAT={v:4d}  NRMS={m:.4f}  "
                  f"(理想 {ideal:.4f}, 退化 {m/ideal:.2f}x)", flush=True)
        rec["ideal_nrms"] = ideal
        result["seeds"][seed] = rec
        A.P.clear()
        A.P.update(base)

    # 中位口径
    med, med_ratio = {}, {}
    for v in VSATS:
        cols = [result["seeds"][s]["vsat"][v]["nrms"]
                for s in result["seeds"] if v in result["seeds"][s]["vsat"]]
        rt = [result["seeds"][s]["vsat"][v]["retio"]
              for s in result["seeds"] if v in result["seeds"][s]["vsat"]]
        med[v] = float(np.median(cols)) if cols else None
        med_ratio[v] = float(np.median(rt)) if rt else None
    result["median"] = med
    result["median_ratio"] = med_ratio

    print("\n=== 中位口径 ===")
    print(f"{'V_SAT':>6} {'L10-ch32中位':>12} {'vs A2':>8} {'退化比':>8}")
    for v in VSATS:
        a2 = A2_CURVE.get(v)
        gain = f"{a2/med[v]:.1f}x" if (a2 and med[v]) else "-"
        print(f"{v:>6} {med[v]:>12.4f} {gain:>8} {med_ratio[v]:>8.2f}x")

    tag = args.tag or time.strftime("%m%d_%H%M")
    out = OUT / f"p1_vsat_{tag}.json"
    json.dump(result, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"\n[saved] {out}  总耗时 {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
