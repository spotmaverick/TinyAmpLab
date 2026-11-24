# 逐层对比: F.conv1d vs 手动 prod.sum — 定位最后一处 bug (DCU)
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

def mul_manual(x, w):
    # w: (Cout,Cin,1) → (B,Cout,T); 与 conv1d 等价性待验证
    prod = x.unsqueeze(1) * w.view(1, -1, x.shape[1], 1)
    return prod.sum(2)

with torch.no_grad():
    # 单独验证 mul_manual == conv1d
    z = torch.randn(1, CH, 1000, device=DEVICE)
    w = sd["blocks.0.expand.weight"]
    a = F.conv1d(z, w, sd["blocks.0.expand.bias"], 1)
    b = mul_manual(z, w) + sd["blocks.0.expand.bias"][None, :, None]
    print(f"mul_manual vs conv1d: max_err={float((a-b).abs().max()):.6f}  shape={tuple(b.shape)}")

    # 逐层对比两种实现
    h1 = F.conv1d(xt, sd["inp.weight"], sd["inp.bias"], 1)
    h2 = h1.clone()
    s1 = torch.zeros_like(h1); s2 = torch.zeros_like(h1)
    print(f"{'layer':6s} {'h1_rms':>8s} {'h2_rms':>8s} {'skip1':>8s} {'skip2':>8s}")
    for i in range(LAYERS):
        dil = 2 ** (i % (DIL_EXP + 1))
        dw_w = sd[f"blocks.{i}.dw.weight"]; dw_b = sd[f"blocks.{i}.dw.bias"]
        z1 = F.conv1d(F.pad(h1,(2*dil,0)), dw_w, dw_b, dilation=dil, groups=CH)
        xp = F.pad(h2, (2*dil, 0)); nn = h2.shape[-1]
        taps = torch.stack([xp[:,:,:nn], xp[:,:,dil:dil+nn], xp[:,:,2*dil:2*dil+nn]], -1)
        z2 = (taps * dw_w.squeeze(1)[:,None,None,:]).sum(-1) + dw_b[None,:,None]
        ex_w = sd[f"blocks.{i}.expand.weight"]; ex_b = sd[f"blocks.{i}.expand.bias"]
        z1 = F.relu(F.conv1d(z1, ex_w, ex_b, 1))
        z2 = F.relu(mul_manual(z2, ex_w) + ex_b[None,:,None])
        co_w = sd[f"blocks.{i}.collapse.weight"]; co_b = sd[f"blocks.{i}.collapse.bias"]
        zc1 = F.conv1d(z1, co_w, co_b, 1)
        zc2 = mul_manual(z2, co_w) + co_b[None,:,None]
        h1 = h1 + zc1; h2 = h2 + zc2
        s1 = s1 + zc1*sd[f"skip_gates.{i}"]; s2 = s2 + zc2*sd[f"skip_gates.{i}"]
        if i in [0,1,3,7,11]:
            print(f"{i:6d} {float(h1.square().mean().sqrt()):8.4f} {float(h2.square().mean().sqrt()):8.4f} "
                  f"{float(s1.square().mean().sqrt()):8.4f} {float(s2.square().mean().sqrt()):8.4f}  "
                  f"dw_err={float((z1-z2).abs().max()):.5f}")
    o1 = F.conv1d(F.relu(s1), sd["head.1.weight"], sd["head.1.bias"])[-n:,0]
    o2 = F.conv1d(F.relu(s2), sd["head.1.weight"], sd["head.1.bias"])[-n:,0]
    def nr(p): return float((p.double()-yt.double()).norm()/yt.double().norm())
    print(f"final: conv1d NRMS={nr(o1):.4f}  manual NRMS={nr(o2):.4f}")
print("DONE")
