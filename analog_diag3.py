# 精确诊断: 用与 analog_sim.py 完全相同的 DeviceBatch, 逐项开关定位 (DCU)
import torch, torch.nn.functional as F, numpy as np, soundfile as sf
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
W_PATH = "/work/home/spotmaverick/nam_project/lgw_v_tcn812h1_p3.pt"
DATA = "/work/home/spotmaverick/nam_project/data"
CH, LAYERS, DIL_EXP = 8, 12, 7
V_SAT = 1.5

sd = {k: v.float().to(DEVICE) for k, v in torch.load(W_PATH, map_location="cpu").items()}
x, sr = sf.read(f"{DATA}/tsa15_test_input.wav"); y, _ = sf.read(f"{DATA}/tsa15_test_target.wav")
x, y = x[:240000], y[:240000]
xt = torch.from_numpy(x.astype(np.float32))[None, None].to(DEVICE)
yt = torch.from_numpy(y.astype(np.float32)).to(DEVICE)
n = yt.shape[0]

def quant6(w):
    scale = w.abs().amax(dim=tuple(range(1, w.dim())), keepdim=True).clamp(min=1e-12) / 31.0
    return (w / scale).round().clamp(-32, 31) * scale

def nrms(p, r): return float((p.double()-r.double()).norm()/r.double().norm())

class Dev:
    def __init__(self, quant=True, mismatch=0., drift=0.):
        g = torch.Generator(device=DEVICE).manual_seed(1000)
        self.w = {}
        for k, v in sd.items():
            vv = quant6(v).to(DEVICE) if quant else v
            mm = torch.normal(0, mismatch, size=vv.shape, generator=g, device=DEVICE) if mismatch else 0
            dr = torch.empty(1, device=DEVICE).uniform_(-drift, drift, generator=g) if drift else 0
            self.w[k] = (vv * (1 + mm) * (1 + dr)).float()
        self.noise = 0.  # 先关噪声

    def mul(self, x, w, tanh):
        prod = x.unsqueeze(1) * w.view(1, -1, x.shape[1], 1)
        if tanh:
            return V_SAT * torch.tanh(prod / V_SAT).sum(2)
        return prod.sum(2)

    def dw(self, x, w, b, dil, tanh):
        n_ = x.shape[-1]; xp = F.pad(x, (2*dil, 0))
        taps = torch.stack([xp[:, :, :n_], xp[:, :, dil:dil+n_], xp[:, :, 2*dil:2*dil+n_]], -1)
        prod = taps * w.squeeze(1)[:, None, None, :]
        if tanh:
            return V_SAT * torch.tanh(prod / V_SAT).sum(-1) + b[None, :, None]
        return prod.sum(-1) + b[None, :, None]

    def fwd(self, tanh_dw, tanh_mul):
        w = self.w
        h = F.conv1d(xt, w["inp.weight"], w["inp.bias"], 1) + self.noise
        skip = torch.zeros_like(h)
        for i in range(LAYERS):
            dil = 2 ** (i % (DIL_EXP + 1))
            z = self.dw(h, w[f"blocks.{i}.dw.weight"], w[f"blocks.{i}.dw.bias"], dil, tanh_dw)
            z = self.mul(z, w[f"blocks.{i}.expand.weight"], tanh_mul) + w[f"blocks.{i}.expand.bias"][None, :, None]
            z = F.relu(z)
            zc = self.mul(z, w[f"blocks.{i}.collapse.weight"], tanh_mul) + w[f"blocks.{i}.collapse.bias"][None, :, None]
            h = h + zc
            skip = skip + zc * w[f"skip_gates.{i}"]
        o = F.relu(skip)
        return F.conv1d(o, w["head.1.weight"], w["head.1.bias"])[-n:, 0]

with torch.no_grad():
    print("=== 精确消融 (5s, device=%s) ===" % DEVICE)
    cfgs = [
        ("quant+t anhDW+tanhMUL", dict(), True, True),
        ("quant, no tanh", dict(), False, False),
        ("quant, tanh_dw only", dict(), True, False),
        ("quant, tanh_mul only", dict(), False, True),
        ("fp32, no tanh", dict(quant=False), False, False),
        ("fp32, tanh_dw only", dict(quant=False), True, False),
        ("fp32, tanh_mul only", dict(quant=False), False, True),
        ("quant+mismatch1%+drift3%, no tanh", dict(mismatch=.01, drift=.03), False, False),
    ]
    for name, kw, td, tm in cfgs:
        d = Dev(**kw)
        print(f"{name:40s} NRMS={nrms(d.fwd(td, tm), yt):.4f}")
print("DONE")
