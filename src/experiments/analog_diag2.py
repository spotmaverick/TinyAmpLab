# 根因锁定: dw_nl 的 tap 顺序 + bias 丢失验证 (DCU)
import torch, torch.nn.functional as F, numpy as np, soundfile as sf
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
W_PATH = "/work/home/spotmaverick/nam_project/lgw_v_tcn812h1_p3.pt"
DATA = "/work/home/spotmaverick/nam_project/data"
CH, LAYERS, DIL_EXP = 8, 12, 7
sd = {k: v.float().to(DEVICE) for k, v in torch.load(W_PATH, map_location="cpu").items()}
x, sr = sf.read(f"{DATA}/tsa15_test_input.wav"); y, _ = sf.read(f"{DATA}/tsa15_test_target.wav")
x, y = x[:240000], y[:240000]
xt = torch.from_numpy(x.astype(np.float32))[None, None].to(DEVICE)
yt = torch.from_numpy(y.astype(np.float32)).to(DEVICE)
n = yt.shape[0]

def dw_manual(h, w, dil, order, use_bias):
    """order='wrong': taps=[x[t],x[t-d],x[t-2d]]配w[0,1,2]; 'right': 反序"""
    n_ = h.shape[-1]
    xp = F.pad(h, (2*dil, 0))
    a = xp[:, :, 2*dil:2*dil+n_]; b = xp[:, :, dil:dil+n_]; c = xp[:, :, 0:n_]
    taps = [a, b, c] if order == "wrong" else [c, b, a]  # right: [x[t-2d],x[t-d],x[t]]
    ww = w.squeeze(1)  # (C,3)
    out = sum(taps[k] * ww[:, k][None, :, None] for k in range(3))
    if use_bias:
        out = out + 0  # bias 在 conv1d 里单独加, 此处占位
    return out

def fwd(mode):
    h = F.conv1d(xt, sd["inp.weight"], sd["inp.bias"], 1)
    skip = torch.zeros_like(h)
    for i in range(LAYERS):
        dil = 2 ** (i % (DIL_EXP + 1))
        dw_w = sd[f"blocks.{i}.dw.weight"]; dw_b = sd[f"blocks.{i}.dw.bias"]
        if mode == "ideal":
            z = F.conv1d(F.pad(h,(2*dil,0)), dw_w, dw_b, dilation=dil, groups=CH)
        else:
            order = "right" if mode == "right" else "wrong"
            z = dw_manual(h, dw_w, dil, order, False)
            z = z + dw_b[None, :, None]  # 补回 bias
        z = F.conv1d(z, sd[f"blocks.{i}.expand.weight"], sd[f"blocks.{i}.expand.bias"], 1)
        z = F.relu(z)
        zc = F.conv1d(z, sd[f"blocks.{i}.collapse.weight"], sd[f"blocks.{i}.collapse.bias"], 1)
        h = h + zc
        skip = skip + zc * sd[f"skip_gates.{i}"]
    o = F.relu(skip)
    return F.conv1d(o, sd["head.1.weight"], sd["head.1.bias"])[-n:, 0]

def nrms(p, r): return float((p.double()-r.double()).norm()/r.double().norm())
with torch.no_grad():
    print("=== dw tap 顺序验证 (5s) ===")
    for m in ["ideal", "wrong", "right"]:
        print(f"{m:10s} NRMS={nrms(fwd(m), yt):.4f}")
print("DONE")
