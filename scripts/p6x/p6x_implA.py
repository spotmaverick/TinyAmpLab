"""P6X 实现 A —— 独立重写 #1

本文件按 `PROTOCOL.md` 从零写成，不引用项目里任何既有仿真代码。
实现风格（刻意与后续 B/C 不同，便于交叉发现错误）：
  - 分组卷积用 **einsum** 收缩分组轴，而不是 reshape 拼轴
  - 每个抽头独立抽取时间片后立刻做 Gilbert，累加到输出
  - 时间轴 **分块** 处理，控制峰值显存
  - 权重劣化 mask 在跑之前一次性全部预生成（按名字种子，与调用顺序无关）
  - Gilbert 用展开式  Vs·tanh(u + β·u³)，u = p/Vs

判据：理想参数下与官方 `nam` 包前向 max|Δ| < 1e-5，否则拒绝出数。
"""
import argparse
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ 协议常量
SR = 48000
SEG_T0 = (0.0, 8.0, 16.0)
SEG_LEN = 5.0
MC_N = 4
CHUNK = 1 << 16

P_BITS = 6
P_SIGMA = 0.01
P_DRIFT = 0.03
P_BETA = 0.002
P_VSAT = 16.0
P_OFF = 0.005
P_SNR = 70.0

LEVELS = 2 ** (P_BITS - 1) - 1          # 31
CLAMP_LO, CLAMP_HI = -(LEVELS + 1), LEVELS


def seed_of(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


def gen_for(*parts):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed_of(*parts))
    return g


# ------------------------------------------------------------------ 结构解析
def parse(nam):
    layers = []
    for a in nam["config"]["layers"]:
        ks = a.get("kernel_sizes")
        if ks is None:
            v = a.get("kernel_size", 3)
            ks = [v] * len(a["dilations"]) if isinstance(v, int) else list(v)
        act = a.get("activation", "Tanh")
        if isinstance(act, list):
            act = act[0]
        if isinstance(act, dict):
            act = act.get("type") or act.get("name") or "Tanh"
        layers.append(dict(ch=a["channels"], cin=a["input_size"], K=ks,
                           dil=list(a["dilations"]), g=a.get("groups_input", 1),
                           act=act, n=len(a["dilations"])))
    return layers


def rf_of(arr):
    return 1 + sum((arr["K"][i] - 1) * arr["dil"][i] for i in range(arr["n"]))


# ------------------------------------------------------------------ 权重准备
def split_weights(sd):
    return {k: v for k, v in sd.items() if k.endswith(".weight") or k.endswith(".bias")}


def quantize(w):
    """逐输出通道对称量化（协议 6.1）。偏置不量化，由调用方保证。"""
    if w.dim() == 1:
        return w
    dims = tuple(range(1, w.dim()))
    scale = w.abs().amax(dim=dims, keepdim=True).clamp(min=1e-12) / LEVELS
    q = (w / scale).round().clamp(CLAMP_LO, CLAMP_HI)
    return q * scale


def degrade(weights, mc, ideal):
    """按协议第 7 节预生成全部劣化。返回可直接做前向的权重字典。"""
    if ideal:
        return {k: v.to(DEV).float() for k, v in weights.items()}
    out = {}
    for name, w in weights.items():
        v = w.to(DEV).float()
        if v.dim() >= 2:                                  # 权重才量化/失配，偏置不动
            v = quantize(v)
            g = gen_for("mm", name, mc)
            mm = torch.empty(v.shape).normal_(0, P_SIGMA, generator=g).to(DEV)
            g2 = gen_for("drift", name, mc)
            dr = torch.empty(1).uniform_(-P_DRIFT, P_DRIFT, generator=g2).to(DEV)
            v = v * (1.0 + mm) * (1.0 + dr)
        out[name] = v
    return out


# ------------------------------------------------------------------ 器件算子
def gilbert(prod, vsat, beta):
    """prod: (..., n_in, T) → 沿 n_in 求和 → (..., T)"""
    if beta <= 0:
        return prod.sum(dim=-2)
    u = prod / vsat
    return (vsat * torch.tanh(u + beta * u * u * u)).sum(dim=-2)


def act_fn(z, kind):
    k = kind.lower()
    if k.startswith("tanh"):
        return torch.tanh(z)
    if k.startswith("leakyrelu"):
        return torch.where(z > 0, z, 0.01 * z)
    if k.startswith("relu"):
        return z.clamp(min=0)
    raise ValueError(kind)


def noise_like(node, mc, shape):
    g = gen_for("noise", node, mc)
    s = 0.25 / (10 ** (P_SNR / 20))
    return torch.empty(shape).normal_(0, s, generator=g).to(DEV)


def offset_for(kind, ch, mc):
    g = gen_for("off", kind, ch, mc)
    return torch.empty((1, ch, 1)).normal_(0, P_OFF, generator=g).to(DEV)


def conv_grouped(x, w, bias, dil, groups, vsat, beta, mc, node, ideal):
    """因果 valid 分组卷积 + Gilbert。

    x:(1,Cin,T)  w:(Cout,Cin/g,K)  → (1,Cout,Tout)，Tout = T-(K-1)*dil
    """
    _, cin, T = x.shape
    cout, ipg, K = w.shape
    opg = cout // groups
    tout = T - (K - 1) * dil
    if tout <= 0:
        raise ValueError("序列短于感受野")
    acc = torch.zeros((1, cout, tout), device=DEV)
    xg = x.view(groups, ipg, T)                          # 分组轴显式化
    wg = w.view(groups, opg, ipg, K)
    for k in range(K):
        wk = wg[:, :, :, k]                              # (g,opg,ipg)
        for t0 in range(0, tout, CHUNK):
            t1 = min(t0 + CHUNK, tout)
            xk = xg[:, :, k * dil + t0: k * dil + t1]    # (g,ipg,n)
            p = torch.einsum("gin,goi->goin", xk, wk)    # (g,opg,ipg,n)
            acc[:, :, t0:t1] += gilbert(p, vsat, beta).reshape(1, cout, t1 - t0)
    if bias is not None:
        acc = acc + bias.view(1, -1, 1)
    if not ideal:
        acc = acc + noise_like(node, mc, acc.shape)
    return acc


def mac1x1(x, w, bias, vsat, beta, mc, node, ideal, groups=1):
    """1x1（分组时按 groups 切）：x:(1,Cin,T) w:(Cout,Cin/g,1)"""
    _, cin, T = x.shape
    cout, ipg, _ = w.shape
    opg = cout // groups
    xg = x.view(groups, ipg, T)
    wg = w.view(groups, opg, ipg)
    acc = torch.zeros((1, cout, T), device=DEV)
    for t0 in range(0, T, CHUNK):
        t1 = min(t0 + CHUNK, T)
        p = torch.einsum("gin,goi->goin", xg[:, :, t0:t1], wg)
        acc[:, :, t0:t1] = gilbert(p, vsat, beta).reshape(1, cout, t1 - t0)
    if bias is not None:
        acc = acc + bias.view(1, -1, 1)
    if not ideal:
        acc = acc + noise_like(node, mc, acc.shape)
    return acc


# ------------------------------------------------------------------ 前向
def forward(sd, nam, x, vsat_default=P_VSAT, bypass=(), vsat_map=None,
            ideal=False, mc=0):
    """返回 (1,T) 输出。bypass/vsat_map 用 scope 前缀匹配（协议未用，保留给消融）。"""
    W = degrade(split_weights(sd), mc, ideal)
    # 理想路径必须同时关掉 Gilbert 的 tanh 饱和与三次项，否则"数字精度"不是数字精度
    vs_eff = 1e9 if ideal else float(vsat_default)
    beta_eff = 0.0 if ideal else P_BETA
    arrays = parse(nam)
    head_scale = float(nam["config"].get("head_scale", 1.0))
    rf = 1
    for a in arrays:
        rf += rf_of(a) - 1
    x = x.to(DEV).float()
    if x.dim() == 1:
        x = x[None, :]
    x = torch.nn.functional.pad(x, (rf - 1, 0))          # pad_start
    cond = x[:, None, :]
    y = x[:, None, :]
    head = None
    for ai, a in enumerate(arrays):
        pre = f"_layer_arrays.{ai}."
        y = mac1x1(y, W[pre + "_rechannel.weight"], None, vs_eff, beta_eff, mc,
                   f"{ai}.rc", ideal)
        rfa = rf_of(a)
        L = min(y.shape[2], cond.shape[2])
        out_len = L - (rfa - 1)
        for li in range(a["n"]):
            lp = f"{pre}_layers.{li}."
            zc = conv_grouped(y, W[lp + "_conv.weight"], W.get(lp + "_conv.bias"),
                              a["dil"][li], a["g"], vs_eff, beta_eff, mc,
                              f"{ai}.{li}.conv", ideal)
            mx = mac1x1(cond, W[lp + "_input_mixer.weight"], None, vs_eff,
                        beta_eff, mc, f"{ai}.{li}.mix", ideal)
            mx = mx[:, :, -zc.shape[2]:]
            n = min(zc.shape[2], mx.shape[2])
            z = zc[:, :, -n:] + mx[:, :, -n:]
            post = act_fn(z + (0.0 if ideal else offset_for(a["act"], a["ch"], mc)), a["act"])
            lw = W.get(lp + "_layer1x1.weight")
            if lw is not None:
                lo = mac1x1(post, lw, W.get(lp + "_layer1x1.bias"), vs_eff,
                            beta_eff, mc, f"{ai}.{li}.res", ideal)
            else:
                lo = post
            term = post[:, :, -out_len:]
            head = term if head is None else head[:, :, -out_len:] + term
            y = y[:, :, -lo.shape[2]:] + lo
        hw = W[pre + "_head_rechannel.weight"]
        head = mac1x1(head, hw, W.get(pre + "_head_rechannel.bias"), vs_eff,
                      beta_eff, mc, f"{ai}.head", ideal, groups=1)
        y = y[:, :, -out_len:]
    out = head * head_scale
    if nam["config"].get("head") is not None:
        raise NotImplementedError("协议未覆盖独立 head 网络")
    return out[0, 0, :]


# ------------------------------------------------------------------ 指标
def nrms(p, y):
    n = min(p.shape[-1], y.shape[-1])
    p, y = p[-n:].to(DEV), y[-n:].to(DEV)      # 必须同设备，否则跨 DCU 会报 two devices
    return float(torch.linalg.vector_norm(p - y) / torch.linalg.vector_norm(y))


def load_pair(data):
    xi, sr = sf.read(str(Path(data) / "tsa15_test_input.wav"), dtype="float32")
    yi, _ = sf.read(str(Path(data) / "tsa15_test_target.wav"), dtype="float32")
    assert sr == SR, sr
    return xi, yi


def seg(x, y, t0, ln=None):
    # 注意：不能写 ln=SEG_LEN —— 默认参数在 def 时就绑定，--seg-len 会被吃掉
    ln = SEG_LEN if ln is None else ln
    i = int(t0 * SR)
    j = int((t0 + ln) * SR)
    return torch.from_numpy(x[i:j].copy()), torch.from_numpy(y[i:j].copy())


# ------------------------------------------------------------------ 主流程
def official_forward(path, xt):
    sys.path.insert(0, os.environ.get("NAM_SRC", "/public/home/spotmaverick/nam_proj"))
    from nam.models import init_from_nam
    net = init_from_nam(json.load(open(path)))
    net = net.to(DEV).eval()
    with torch.no_grad():
        return net(xt[None, :].to(DEV))[0]


def main():
    global SEG_LEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("P6X_DATA",
                                                    "/public/home/spotmaverick/nam_proj/data"))
    ap.add_argument("--models", default="all")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--mc", type=int, default=MC_N)
    ap.add_argument("--seg-len", type=float, default=SEG_LEN)
    ap.add_argument("--out", default="out/p6x_A.json")
    ap.add_argument("--only", default=None, help="只跑单个模型，用于快速自检")
    ap.add_argument("--sanity-only", action="store_true", help="只做结构对拍，不跑网格")
    args = ap.parse_args()
    SEG_LEN = args.seg_len

    md = Path(args.model_dir)
    if args.only:
        items = [("solo", md / args.only)]
    elif args.models == "all":
        items = [("std", md / "nam_standard_tsa15.nam"), ("lite", md / "nam_lite_tsa15.nam")]
        items += [(p.stem, p) for p in sorted((md / "seeds").glob("*.nam"))]
    else:
        items = [(s, md / "seeds" / f"{s}.nam") for s in args.models.split(",")]

    xi, yi = load_pair(args.data)
    segs = [seg(xi, yi, t0) for t0 in SEG_T0]
    res = {"impl": "A", "device": DEV, "mc": args.mc, "ks": [1, 32], "vs": [16, 96],
           "rows": {}}

    print(f"[env] impl=A device={DEV} cuda={torch.cuda.is_available()} "
          f"torch={torch.__version__} n_models={len(items)}", flush=True)

    for name, path in items:
        nam = json.load(open(path))
        from nam.models import init_from_nam
        net = init_from_nam(nam).to(DEV).eval()
        sd = {k.removeprefix("_net."): v for k, v in net.state_dict().items()
              if k.endswith(".weight") or k.endswith(".bias")}

        # --- sanity：理想前向 vs 官方前向
        xt0, yt0 = segs[0]
        with torch.no_grad():
            mine = forward(sd, nam, xt0, ideal=True)
            theirs = net(xt0[None, :].to(DEV))[0]
        n = min(mine.shape[0], theirs.shape[0])
        maxd = float((mine[-n:] - theirs[-n:]).abs().max())
        print(f"[{name}] sanity max|Δ|={maxd:.3e}", flush=True)
        if maxd > 1e-5:
            res["rows"][name] = {"sanity_fail": maxd}
            print(f"[{name}] SANITY FAIL —— 不出数", flush=True)
            continue

        row = {"sanity": maxd}
        with torch.no_grad():
            dg = [nrms(forward(sd, nam, xt, ideal=True), yt) for xt, yt in segs]
        row["digital"] = float(np.median(dg))
        if args.sanity_only:
            res["rows"][name] = row
            continue

        grid = {}
        for k in (1, 32):
            sd_k, nam_k = rebalance(sd, nam, k)
            for v in (16.0, 96.0):
                vals = []
                for xt, yt in segs:
                    mc_v = []
                    for m in range(args.mc):
                        with torch.no_grad():
                            mc_v.append(nrms(forward(sd_k, nam_k, xt, vsat_default=v, mc=m), yt))
                    vals.append(float(np.sqrt(np.mean(np.square(mc_v)))))
                grid[f"k{k}_v{int(v)}"] = float(np.median(vals))
        row["grid"] = grid
        row["free"] = grid["k1_v16"]
        row["best"] = min(grid.values())
        res["rows"][name] = row
        print(f"[{name}] digital={row['digital']:.4f} free={row['free']:.4f} "
              f"best={row['best']:.4f}", flush=True)

    ok = {k: v for k, v in res["rows"].items() if "grid" in v}
    seeds = [v["best"] for k, v in ok.items() if k not in ("std", "lite")]  # std 也以 s 开头
    if seeds:
        res["median_best_13"] = float(np.median(seeds))
        res["n_seeds"] = len(seeds)
        print(f"[summary] seeds n={len(seeds)} median_best={res['median_best_13']:.4f} "
              f"max={max(seeds):.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("[done] " + args.out, flush=True)
    return 0


def rebalance(sd, nam, k):
    """head_rechannel 增益重平衡：最后一个 array 的 head_rechannel /k，head_scale *k。"""
    if k == 1:
        return sd, nam
    n_arr = len(nam["config"]["layers"])
    last = n_arr - 1
    sd2 = dict(sd)
    for suf in ("weight", "bias"):
        key = f"_layer_arrays.{last}._head_rechannel.{suf}"
        if key in sd2:
            sd2[key] = sd2[key] / k
    nam2 = json.loads(json.dumps(nam))
    nam2["config"]["head_scale"] = float(nam2["config"].get("head_scale", 1.0)) * k
    return sd2, nam2


if __name__ == "__main__":
    raise SystemExit(main())
