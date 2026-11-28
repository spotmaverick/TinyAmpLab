"""P6X 实现 B —— 独立重写 #2（用于和 A 交叉验证）

按 `PROTOCOL.md` 重写，不看 A 的代码。为了能真正暴露错误，几乎所有内部机制都换成另一套：

| 环节 | A 的做法 | B 的做法（本文件） |
|---|---|---|
| 抽头取数 | 时间轴切片 `x[..., k*d : k*d+n]` | `index_select` 按下标 gather |
| 分组 | `view(g, ipg, T)` + einsum | `torch.chunk` 切通道后逐组 matmul |
| tanh | `torch.tanh` | sigmoid 恒等式 `2σ(2z)-1` |
| 量化 | `round(w/scale)*scale` | 建码本 + `searchsorted` 最近邻 |
| 累加 | float32 原地累加 | float64 累加后回落 float32 |
| head 和 | 逐层原地相加 | 列表收集后一次性 stack+sum |
| 卷积乘加 | einsum 一步出 | 分组内 `torch.bmm`（把通道当 batch） |

判据同 A：理想参数 vs 官方前向 max|Δ| < 1e-5。
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

SR = 48000
SEG_T0 = (0.0, 8.0, 16.0)
SEG_LEN = 5.0
MC_N = 4

BITS = 6
SIGMA = 0.01
DRIFT = 0.03
BETA = 0.002
VSAT = 16.0
OFFSET = 0.005
SNR = 70.0
LEVELS = 2 ** (BITS - 1) - 1
QLO, QHI = -(LEVELS + 1), LEVELS
ACC = torch.float64                      # B 的累加精度（故意与 A 不同）


def seed_of(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0xFFFFFFFF


def rng(*parts):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed_of(*parts))
    return g


# ---------------------------------------------------------------- 量化（码本法）
def quant_via_codebook(w):
    """最近邻查表量化：与 round(w/scale) 数学等价，但走完全不同的代码路径。

    实现要点：每个输出通道一条电平表，searchsorted 找插入点，再在相邻两个电平里取更近的一个
    （等距时取偶数下标 —— 与 torch.round 的 half-to-even 对齐）。
    """
    if w.dim() == 1:
        return w
    cw = w.reshape(w.shape[0], -1)                       # (Cout, rest)
    dims = tuple(range(1, w.dim()))
    scale = w.abs().amax(dim=dims, keepdim=True).reshape(-1, 1).clamp(min=1e-12) / LEVELS
    lv = torch.arange(QLO, QHI + 1, device=w.device, dtype=w.dtype)   # (L,)
    grid = lv[None, :] * scale                            # (Cout, L)
    x = cw
    idx = torch.searchsorted(grid.contiguous(), x.contiguous())      # (Cout, rest)
    idx = idx.clamp(1, grid.shape[1] - 1)
    lo = grid.gather(1, idx - 1)
    hi = grid.gather(1, idx)
    take_lo = (x - lo).abs() <= (hi - x).abs()
    # 等距（tie）时按 half-to-even 修正：下标为偶数的那侧胜出
    tie = (x - lo).abs() == (hi - x).abs()
    even = ((idx - 1) % 2 == 0)
    take_lo = torch.where(tie, even, take_lo)
    q = torch.where(take_lo, lo, hi)
    return q.reshape(w.shape)


# ---------------------------------------------------------------- 劣化
def make_weights(sd, mc, ideal):
    src = {k: v for k, v in sd.items() if k.endswith((".weight", ".bias"))}
    if ideal:
        return {k: v.to(DEV).float() for k, v in src.items()}
    out = {}
    for name, w in src.items():
        v = w.to(DEV).float()
        if v.dim() >= 2:
            v = quant_via_codebook(v)
            mm = torch.empty(v.shape).normal_(0, SIGMA, generator=rng("mm", name, mc))
            dr = torch.empty(1).uniform_(-DRIFT, DRIFT, generator=rng("drift", name, mc))
            v = v * (1.0 + mm.to(DEV)) * (1.0 + dr.to(DEV))
        out[name] = v
    return out


# ---------------------------------------------------------------- 器件算子
def tanh_via_sigmoid(z):
    """2σ(2z)−1 ≡ tanh(z)。与 A 的 torch.tanh 是两条不同的实现路径。"""
    return 2.0 * torch.sigmoid(2.0 * z) - 1.0


def gilbert(prod, vsat, beta):
    """prod: (..., n_in, N) → (..., N)。沿倒数第二维（输入通道）求和。

    关键：tanh 必须作用在**每一个乘积**上，不能先把乘积线性和起来再非线性。
    """
    if beta <= 0:
        return prod.to(ACC).sum(dim=-2)
    u = (prod / vsat).to(ACC)
    y = vsat * tanh_via_sigmoid(u + beta * u * u * u)
    return y.sum(dim=-2)


def act(z, kind):
    k = kind.lower()
    if k.startswith("tanh"):
        return tanh_via_sigmoid(z)
    if k.startswith("leakyrelu"):
        return torch.where(z > 0, z, 0.01 * z)
    if k.startswith("relu"):
        return z.clamp(min=0)
    raise ValueError(kind)


def add_noise(node, mc, t):
    if SNR >= 200:
        return t
    s = 0.25 / (10 ** (SNR / 20))
    return t + torch.empty(t.shape).normal_(0, s, generator=rng("noise", node, mc)).to(DEV)


def get_offset(kind, ch, mc):
    g = rng("off", kind, ch, mc)
    return torch.empty((1, ch, 1)).normal_(0, OFFSET, generator=g).to(DEV)


def conv1d_sim(x, w, bias, dilation, groups, vsat, beta, mc, node, ideal):
    """x:(1,Cin,T) w:(Cout,Cin/g,K) → (1,Cout,Tout)，Tout = T-(K-1)*dilation"""
    _, cin, T = x.shape
    cout, ipg, K = w.shape
    opg = cout // groups
    tout = T - (K - 1) * dilation
    if tout <= 0:
        raise ValueError("序列短于感受野")
    base = torch.arange(tout, device=DEV)
    xgrp = torch.chunk(x, groups, dim=1)                  # groups 个 (1,ipg,T)
    wgrp = torch.chunk(w, groups, dim=0)                  # groups 个 (opg,ipg,K)
    outs = []
    for gi in range(groups):
        xi = xgrp[gi][0]                                  # (ipg, T)
        wi = wgrp[gi]                                     # (opg, ipg, K)
        acc = torch.zeros(opg, tout, dtype=ACC, device=DEV)
        for k in range(K):
            idx = base + k * dilation
            xt = xi.index_select(1, idx)                  # (ipg, tout) —— gather 取抽头
            p = wi[:, :, k].unsqueeze(-1) * xt.unsqueeze(0)   # (opg, ipg, tout) 逐乘积
            acc = acc + gilbert(p, vsat, beta)            # 逐乘积 tanh 后再求和
        o = acc.to(torch.float32)
        if bias is not None:
            o = o + bias[gi * opg:(gi + 1) * opg].view(-1, 1)
        outs.append(o.unsqueeze(0))
    y = torch.cat(outs, dim=1)                            # (1,Cout,tout)
    if not ideal:
        y = add_noise(node, mc, y)
    return y


def mac1x1_sim(x, w, bias, groups, vsat, beta, mc, node, ideal):
    _, cin, T = x.shape
    cout, ipg, _ = w.shape
    opg = cout // groups
    wg = w[:, :, 0]
    if groups == 1:
        p = (wg.unsqueeze(-1) * x[0].unsqueeze(0))         # (Cout,Cin,T)
        y = gilbert(p.reshape(cout, cin, T), vsat, beta).view(1, cout, T)
    else:
        chunks = list(torch.chunk(x, groups, dim=1))
        out = []
        for gi in range(groups):
            wi = wg[gi * opg:(gi + 1) * opg]
            p = wi.unsqueeze(-1) * chunks[gi][0].unsqueeze(0)
            out.append(gilbert(p.reshape(opg, ipg, T), vsat, beta).view(1, opg, T))
        y = torch.cat(out, dim=1)
    if bias is not None:
        y = y + bias.view(1, -1, 1)
    if not ideal:
        y = add_noise(node, mc, y)
    return y


# ---------------------------------------------------------------- 结构解析
def arrays_of(nam):
    res = []
    for a in nam["config"]["layers"]:
        ks = a.get("kernel_sizes")
        if ks is None:
            v = a.get("kernel_size", 3)
            ks = [v] * len(a["dilations"]) if isinstance(v, int) else list(v)
        actv = a.get("activation", "Tanh")
        if isinstance(actv, list):
            actv = actv[0]
        if isinstance(actv, dict):
            actv = actv.get("type") or actv.get("name") or "Tanh"
        res.append(dict(ch=a["channels"], cin=a["input_size"], K=list(ks),
                        d=list(a["dilations"]), g=a.get("groups_input", 1),
                        act=actv, n=len(a["dilations"])))
    return res


def rfield(a):
    return 1 + sum((a["K"][i] - 1) * a["d"][i] for i in range(a["n"]))


def run(sd, nam, x, vsat=VSAT, ideal=False, mc=0):
    W = make_weights(sd, mc, ideal)
    vs = 1e9 if ideal else float(vsat)
    be = 0.0 if ideal else BETA
    arrs = arrays_of(nam)
    rf = 1 + sum(rfield(a) - 1 for a in arrs)
    x = x.to(DEV).float()
    if x.dim() == 1:
        x = x[None, :]
    x = torch.nn.functional.pad(x, (rf - 1, 0))
    c = x[:, None, :]
    y = x[:, None, :]
    hprev = None
    for ai, a in enumerate(arrs):
        pf = f"_layer_arrays.{ai}."
        y = mac1x1_sim(y, W[pf + "_rechannel.weight"], None, 1, vs, be, mc, f"{ai}.rc", ideal)
        rf_a = rfield(a)
        outlen = min(y.shape[2], c.shape[2]) - (rf_a - 1)
        terms = [] if hprev is None else [hprev[:, :, -outlen:]]
        for li in range(a["n"]):
            lf = f"{pf}_layers.{li}."
            zc = conv1d_sim(y, W[lf + "_conv.weight"], W.get(lf + "_conv.bias"),
                            a["d"][li], a["g"], vs, be, mc, f"{ai}.{li}.conv", ideal)
            mx = mac1x1_sim(c, W[lf + "_input_mixer.weight"], None, 1, vs, be, mc,
                            f"{ai}.{li}.mix", ideal)
            mx = mx[:, :, -zc.shape[2]:]
            L = min(zc.shape[2], mx.shape[2])
            z = zc[:, :, -L:] + mx[:, :, -L:]
            if not ideal:
                z = z + get_offset(a["act"], a["ch"], mc)
            post = act(z, a["act"])
            if (lf + "_layer1x1.weight") in W:
                lo = mac1x1_sim(post, W[lf + "_layer1x1.weight"], W.get(lf + "_layer1x1.bias"),
                                1, vs, be, mc, f"{ai}.{li}.res", ideal)
            else:
                lo = post
            terms.append(post[:, :, -outlen:])              # 列表收集，最后一次性求和
            y = y[:, :, -lo.shape[2]:] + lo
        hs = torch.stack(terms, dim=0).to(ACC).sum(dim=0).to(torch.float32)
        hprev = mac1x1_sim(hs, W[pf + "_head_rechannel.weight"],
                           W.get(pf + "_head_rechannel.bias"), 1, vs, be, mc, f"{ai}.head", ideal)
        y = y[:, :, -outlen:]
    if nam["config"].get("head") is not None:
        raise NotImplementedError("协议未覆盖独立 head 网络")
    return (hprev * float(nam["config"].get("head_scale", 1.0)))[0, 0, :]


# ---------------------------------------------------------------- 打分
def nrms(p, y):
    n = min(p.shape[-1], y.shape[-1])
    p, y = p[-n:].to(ACC), y[-n:].to(ACC)
    return float(torch.linalg.vector_norm(p - y) / torch.linalg.vector_norm(y))


def rebalance(sd, nam, k):
    if k == 1:
        return sd, nam
    last = len(nam["config"]["layers"]) - 1
    sd2 = dict(sd)
    for suf in ("weight", "bias"):
        kk = f"_layer_arrays.{last}._head_rechannel.{suf}"
        if kk in sd2:
            sd2[kk] = sd2[kk] / k
    n2 = json.loads(json.dumps(nam))
    n2["config"]["head_scale"] = float(n2["config"].get("head_scale", 1.0)) * k
    return sd2, n2


def main():
    global SEG_LEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("P6X_DATA",
                                                    "/public/home/spotmaverick/nam_proj/data"))
    ap.add_argument("--models", default="all")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--mc", type=int, default=MC_N)
    ap.add_argument("--seg-len", type=float, default=SEG_LEN)
    ap.add_argument("--out", default="out/p6x_B.json")
    ap.add_argument("--only", default=None)
    ap.add_argument("--sanity-only", action="store_true")
    a = ap.parse_args()
    SEG_LEN = a.seg_len

    md = Path(a.model_dir)
    if a.only:
        items = [("solo", md / a.only)]
    elif a.models == "all":
        items = [("std", md / "nam_standard_tsa15.nam"), ("lite", md / "nam_lite_tsa15.nam")]
        items += [(p.stem, p) for p in sorted((md / "seeds").glob("*.nam"))]
    else:
        items = [(s, md / "seeds" / f"{s}.nam") for s in a.models.split(",")]

    xi, sr = sf.read(str(Path(a.data) / "tsa15_test_input.wav"), dtype="float32")
    yi, _ = sf.read(str(Path(a.data) / "tsa15_test_target.wav"), dtype="float32")
    assert sr == SR
    segs = []
    for t0 in SEG_T0:
        i, j = int(t0 * SR), int((t0 + SEG_LEN) * SR)
        segs.append((torch.from_numpy(xi[i:j].copy()), torch.from_numpy(yi[i:j].copy())))

    # 自检：码本量化必须与 round 口径完全一致，否则 B 的 device 数与 A 不可比
    probe = torch.randn(8, 5, 3) * 0.7
    sc = probe.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-12) / LEVELS
    ref = (probe / sc).round().clamp(QLO, QHI) * sc
    got = quant_via_codebook(probe)
    qdiff = float((ref - got).abs().max())
    print(f"[check] codebook vs round: max|Δ|={qdiff:.3e}", flush=True)

    res = {"impl": "B", "device": DEV, "mc": a.mc, "ks": [1, 32], "vs": [16, 96],
           "codebook_check": qdiff, "rows": {}}
    print(f"[env] impl=B device={DEV} cuda={torch.cuda.is_available()} "
          f"torch={torch.__version__}", flush=True)

    for name, path in items:
        nam = json.load(open(path))
        from nam.models import init_from_nam
        net = init_from_nam(nam).to(DEV).eval()
        sd = {k.removeprefix("_net."): v for k, v in net.state_dict().items()
              if k.endswith((".weight", ".bias"))}

        xt0 = segs[0][0]
        mine = run(sd, nam, xt0, ideal=True)
        with torch.no_grad():
            off = net(xt0[None, :].to(DEV))[0]
        n = min(mine.shape[0], off.shape[0])
        maxd = float((mine[-n:] - off[-n:]).abs().max())
        print(f"[{name}] sanity max|Δ|={maxd:.3e}", flush=True)
        if maxd > 1e-5:
            res["rows"][name] = {"sanity_fail": maxd}
            continue

        row = {"sanity": maxd}
        row["digital"] = float(np.median([nrms(run(sd, nam, xt, ideal=True), yt)
                                          for xt, yt in segs]))
        if a.sanity_only:
            res["rows"][name] = row
            continue

        grid = {}
        for k in (1, 32):
            sdk, namk = rebalance(sd, nam, k)
            for v in (16.0, 96.0):
                per_seg = []
                for xt, yt in segs:
                    vv = [nrms(run(sdk, namk, xt, vsat=v, mc=m), yt) for m in range(a.mc)]
                    per_seg.append(float(np.sqrt(np.mean(np.square(vv)))))
                grid[f"k{k}_v{int(v)}"] = float(np.median(per_seg))
        row["grid"] = grid
        row["free"] = grid["k1_v16"]
        row["best"] = min(grid.values())
        res["rows"][name] = row
        print(f"[{name}] digital={row['digital']:.4f} free={row['free']:.4f} "
              f"best={row['best']:.4f}", flush=True)

    seeds = [v["best"] for k, v in res["rows"].items() if "grid" in v and k.startswith("s")]
    if seeds:
        res["median_best_13"] = float(np.median(seeds))
        res["n_seeds"] = len(seeds)
        print(f"[summary] n={len(seeds)} median_best={res['median_best_13']:.4f} "
              f"max={max(seeds):.4f}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("[done] " + a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
