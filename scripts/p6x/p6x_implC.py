"""P6X 实现 C —— 独立重写 #3（流式分块执行）

按 `PROTOCOL.md` 重写。与 A/B 的**执行模型**根本不同：

  A：整段一次性前向，时间轴切片取抽头，einsum 收缩分组轴
  B：整段一次性前向，index_select 取抽头，chunk+bmm 分组，float64 累加
  C：**流式**——把信号切成 BLK 大小的块，每块带 RF-1 的历史重跑整个网络，
     只保留该块对应的输出再拼起来

为什么值得这么做：本项目最容易写错的就是**因果对齐**（"valid 卷积 + 末尾裁剪" vs
"左侧零填充 + 开头裁剪"两种口径）。流式实现如果因果对齐写错，块与块接缝处会出现
周期性台阶——这与整段实现是两种完全不同的失效模式，交叉验证价值最大。

其它差异：
  - 逐乘积用 unsqueeze 广播，不用 einsum/bmm
  - 量化 scale 用 amax 的 reshape 视图计算（同一公式、不同代码路径）
  - 累加全程 float32，不做 float64 提升
  - head 累加用 `functools.reduce`
"""
import argparse
import functools
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
BLK = 1 << 16

BITS = 6
SIGMA = 0.01
DRIFT = 0.03
BETA = 0.002
VSAT = 16.0
OFFSET = 0.005
SNR = 70.0
LEVELS = 2 ** (BITS - 1) - 1
QLO, QHI = -(LEVELS + 1), LEVELS


def sd_seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0xFFFFFFFF


def G(*parts):
    g = torch.Generator(device="cpu")
    g.manual_seed(sd_seed(*parts))
    return g


def qsym(w):
    """协议 6.1：逐输出通道对称量化。"""
    if w.dim() == 1:
        return w
    C = w.shape[0]
    flat = w.reshape(C, -1)
    amax = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    sc = amax / LEVELS
    q = (flat / sc).round().clamp(QLO, QHI) * sc
    return q.reshape(w.shape)


def prep(sd, mc, ideal):
    out = {}
    for name, w in sd.items():
        v = w.to(DEV).float()
        if ideal:
            out[name] = v
            continue
        if v.dim() >= 2:
            v = qsym(v)
            mm = torch.empty(v.shape).normal_(0, SIGMA, generator=G("mm", name, mc)).to(DEV)
            dr = torch.empty(1).uniform_(-DRIFT, DRIFT, generator=G("drift", name, mc)).to(DEV)
            v = v * (1.0 + mm) * (1.0 + dr)
        out[name] = v
    return out


def gil(prod, vsat, beta):
    """prod: (n_out, n_in, T) → (n_out, T)，tanh 逐乘积施加。"""
    if beta <= 0:
        return prod.sum(dim=1)
    u = prod / vsat
    return (vsat * torch.tanh(u + beta * u.pow(3))).sum(dim=1)


def actv(z, kind):
    k = kind.lower()
    if k.startswith("tanh"):
        return torch.tanh(z)
    if k.startswith("leakyrelu"):
        return torch.where(z > 0, z, 0.01 * z)
    if k.startswith("relu"):
        return z.clamp(min=0)
    raise ValueError(kind)


def nterm(node, mc, shape):
    return torch.empty(shape).normal_(0, 0.25 / (10 ** (SNR / 20)),
                                     generator=G("noise", node, mc)).to(DEV)


def offv(kind, ch, mc):
    return torch.empty((1, ch, 1)).normal_(0, OFFSET, generator=G("off", kind, ch, mc)).to(DEV)


def conv_sim(x, w, b, dil, g, vsat, beta, mc, node, ideal):
    """x:(1,Cin,T) w:(Cout,Cin/g,K)。逐抽头逐乘积做 tanh，再求和。"""
    _, cin, T = x.shape
    cout, ipg, K = w.shape
    opg = cout // g
    tout = T - (K - 1) * dil
    if tout <= 0:
        raise ValueError("段太短")
    out = torch.zeros(1, cout, tout, device=DEV)
    for gi in range(g):
        xg = x[:, gi * ipg:(gi + 1) * ipg, :]                 # (1,ipg,T)
        wg = w[gi * opg:(gi + 1) * opg]                        # (opg,ipg,K)
        acc = torch.zeros(opg, tout, device=DEV)
        for k in range(K):
            xn = xg[:, :, k * dil:k * dil + tout][0]           # (ipg,tout)
            # tensordot：w 的第 k 个抽头 (opg,ipg) 与 (ipg,tout) 做外积再广播乘
            wk = wg[:, :, k]                                   # (opg,ipg)
            # 逐乘积：p[o,i,t] = w[o,i,k]·x[i,t+k·d]
            p = wk.unsqueeze(-1) * xn.unsqueeze(0)              # (opg,ipg,tout)
            acc = acc + gil(p, vsat, beta)
        if b is not None:
            acc = acc + b[gi * opg:(gi + 1) * opg].view(-1, 1)
        out[0, gi * opg:(gi + 1) * opg, :] = acc
    if not ideal:
        out = out + nterm(node, mc, out.shape)
    return out


def mac_sim(x, w, b, vsat, beta, mc, node, ideal):
    """1x1，groups 恒 1。"""
    _, cin, T = x.shape
    cout = w.shape[0]
    p = w[:, :, 0].unsqueeze(-1) * x[0].unsqueeze(0)           # (cout,cin,T)
    y = gil(p, vsat, beta).unsqueeze(0)
    if b is not None:
        y = y + b.view(1, -1, 1)
    if not ideal:
        y = y + nterm(node, mc, y.shape)
    return y


def struct(nam):
    res = []
    for a in nam["config"]["layers"]:
        ks = a.get("kernel_sizes")
        if ks is None:
            v = a.get("kernel_size", 3)
            ks = [v] * len(a["dilations"]) if isinstance(v, int) else list(v)
        ac = a.get("activation", "Tanh")
        if isinstance(ac, list):
            ac = ac[0]
        if isinstance(ac, dict):
            ac = ac.get("type") or "Tanh"
        res.append(dict(ch=a["channels"], cin=a["input_size"], K=list(ks),
                        d=list(a["dilations"]), g=a.get("groups_input", 1),
                        act=ac, n=len(a["dilations"])))
    return res


def rfa(a):
    return 1 + sum((a["K"][i] - 1) * a["d"][i] for i in range(a["n"]))


def net_once(W, nam, arrs, seg, vsat, ideal, mc):
    """对**一整段（已补零）**跑一遍网络，返回 (1,T_out)。流式复用这个函数。"""
    c = seg[:, None, :]
    y = seg[:, None, :]
    h = None
    for ai, a in enumerate(arrs):
        pf = f"_layer_arrays.{ai}."
        y = mac_sim(y, W[pf + "_rechannel.weight"], None, vsat, BETA if not ideal else 0.0,
                    mc, f"{ai}.rc", ideal)
        outlen = min(y.shape[2], c.shape[2]) - (rfa(a) - 1)
        parts = [] if h is None else [h[:, :, -outlen:]]
        for li in range(a["n"]):
            lf = f"{pf}_layers.{li}."
            zc = conv_sim(y, W[lf + "_conv.weight"], W.get(lf + "_conv.bias"), a["d"][li],
                          a["g"], vsat, BETA if not ideal else 0.0, mc, f"{ai}.{li}.conv", ideal)
            mx = mac_sim(c, W[lf + "_input_mixer.weight"], None, vsat,
                         BETA if not ideal else 0.0, mc, f"{ai}.{li}.mix", ideal)
            mx = mx[:, :, -zc.shape[2]:]
            L = min(zc.shape[2], mx.shape[2])
            z = zc[:, :, -L:] + mx[:, :, -L:]
            if not ideal:
                z = z + offv(a["act"], a["ch"], mc)
            post = actv(z, a["act"])
            if (lf + "_layer1x1.weight") in W:
                lo = mac_sim(post, W[lf + "_layer1x1.weight"], W.get(lf + "_layer1x1.bias"),
                             vsat, BETA if not ideal else 0.0, mc, f"{ai}.{li}.res", ideal)
            else:
                lo = post
            parts.append(post[:, :, -outlen:])
            y = y[:, :, -lo.shape[2]:] + lo
        h = mac_sim(functools.reduce(lambda p, q: p + q, parts),
                    W[pf + "_head_rechannel.weight"], W.get(pf + "_head_rechannel.bias"),
                    vsat, BETA if not ideal else 0.0, mc, f"{ai}.head", ideal)
        y = y[:, :, -outlen:]
    if nam["config"].get("head") is not None:
        raise NotImplementedError("协议未覆盖独立 head 网络")
    return (h * float(nam["config"].get("head_scale", 1.0)))[0, 0, :]


def forward_stream(W, nam, arrs, x, vsat, ideal, mc, blk=BLK):
    """流式：把输入切成 blk 块，每块带 (RF-1) 历史整体重跑，只取本块输出。"""
    rf = 1 + sum(rfa(a) - 1 for a in arrs)
    base = x.to(DEV).float()
    if base.dim() == 1:
        base = base[None, :]
    pad = torch.nn.functional.pad(base, (rf - 1, 0))            # (1, T+rf-1)
    T = base.shape[1]
    chunks = []
    for s in range(0, T, blk):
        e = min(s + blk, T)
        ws = max(0, s - (rf - 1))
        win = pad[:, ws:e + rf - 1]                             # 历史 + 本块 + 尾部对齐所需
        y = net_once(W, nam, arrs, win, vsat, ideal, mc)
        # win 覆盖的原始下标是 [ws, e)，输出 y 与 win 末尾对齐
        take = e - ws
        chunks.append(y[-take:])
    return torch.cat(chunks)[-T:]


def nrms(p, y):
    n = min(p.shape[-1], y.shape[-1])
    p, y = p[-n:].to(DEV), y[-n:].to(DEV)
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
    global SEG_LEN, BLK
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("P6X_DATA",
                                                    "/public/home/spotmaverick/nam_proj/data"))
    ap.add_argument("--models", default="all")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--mc", type=int, default=MC_N)
    ap.add_argument("--seg-len", type=float, default=SEG_LEN)
    ap.add_argument("--blk", type=int, default=BLK)
    ap.add_argument("--out", default="out/p6x_C.json")
    ap.add_argument("--only", default=None)
    ap.add_argument("--sanity-only", action="store_true")
    a = ap.parse_args()
    SEG_LEN, BLK = a.seg_len, a.blk

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

    res = {"impl": "C", "mode": "streaming", "blk": BLK, "device": DEV, "mc": a.mc,
           "ks": [1, 32], "vs": [16, 96], "rows": {}}
    print(f"[env] impl=C streaming blk={BLK} device={DEV} torch={torch.__version__}", flush=True)

    for name, path in items:
        nam = json.load(open(path))
        from nam.models import init_from_nam
        net = init_from_nam(nam).to(DEV).eval()
        sd = {k.removeprefix("_net."): v for k, v in net.state_dict().items()
              if k.endswith((".weight", ".bias"))}
        arrs = struct(nam)

        W = prep(sd, 0, True)
        mine = forward_stream(W, nam, arrs, segs[0][0], 1e9, True, 0)
        with torch.no_grad():
            ref = net(segs[0][0][None, :].to(DEV))[0]
        n = min(mine.shape[0], ref.shape[0])
        maxd = float((mine[-n:] - ref[-n:]).abs().max())
        print(f"[{name}] sanity max|Δ|={maxd:.3e}", flush=True)
        if maxd > 1e-5:
            res["rows"][name] = {"sanity_fail": maxd}
            continue

        row = {"sanity": maxd}
        row["digital"] = float(np.median(
            [nrms(forward_stream(prep(sd, 0, True), nam, arrs, xt, 1e9, True, 0), yt)
             for xt, yt in segs]))
        if a.sanity_only:
            res["rows"][name] = row
            continue

        grid = {}
        for k in (1, 32):
            sdk, namk = rebalance(sd, nam, k)
            Wm = {m: prep(sdk, m, False) for m in range(a.mc)}   # 每个 mc 只准备一次
            for v in (16.0, 96.0):
                per_seg = []
                for xt, yt in segs:
                    vv = [nrms(forward_stream(Wm[m], namk, arrs, xt, v, False, m), yt)
                          for m in range(a.mc)]
                    per_seg.append(float(np.sqrt(np.mean(np.square(vv)))))
                grid[f"k{k}_v{int(v)}"] = float(np.median(per_seg))
        row["grid"] = grid
        row["free"] = grid["k1_v16"]
        row["best"] = min(grid.values())
        res["rows"][name] = row
        print(f"[{name}] digital={row['digital']:.4f} free={row['free']:.4f} "
              f"best={row['best']:.4f}", flush=True)

    seeds = [v["best"] for k, v in res["rows"].items() if "grid" in v and k not in ("std", "lite")]   # 注意 std 也以 s 开头
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
