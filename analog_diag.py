# 物理层消融诊断: 逐个打开非理想因素, 定位 NRMS 暴涨根因 (DCU)
import torch, torch.nn.functional as F, numpy as np, soundfile as sf, sys
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
W_PATH = "/work/home/spotmaverick/nam_project/lgw_v_tcn812h1_p3.pt"
DATA = "/work/home/spotmaverick/nam_project/data"
CH, LAYERS, DIL_EXP = 8, 12, 7
V_SAT = 1.5

sd = {k: v.float().to(DEVICE) for k, v in torch.load(W_PATH, map_location="cpu").items()}
x, sr = sf.read(f"{DATA}/tsa15_test_input.wav")
y, _ = sf.read(f"{DATA}/tsa15_test_target.wav")
x, y = x[:240000], y[:240000]
xt = torch.from_numpy(x.astype(np.float32))[None, None].to(DEVICE)
yt = torch.from_numpy(y.astype(np.float32)).to(DEVICE)
n = yt.shape[0]

def quant6(w):
    scale = w.abs().amax(dim=tuple(range(1, w.dim())), keepdim=True).clamp(min=1e-12) / 31.0
    return (w / scale).round().clamp(-32, 31) * scale

def nrms(pred, ref):
    return float((pred.double() - ref.double()).norm() / ref.double().norm())

def fwd(w, mode):
    """mode: ideal / quant / tanh_mul / tanh_dw / all_nonlin"""
    h = F.conv1d(xt, w["inp.weight"], w["inp.bias"], 1)
    skip = torch.zeros_like(h)
    for i in range(LAYERS):
        dil = 2 ** (i % (DIL_EXP + 1))
        dw_w = w[f"blocks.{i}.dw.weight"]; dw_b = w[f"blocks.{i}.dw.bias"]
        z = F.conv1d(F.pad(h, (2*dil, 0)), dw_w, dw_b, dilation=dil, groups=CH)
        if "tanh_dw" in mode:
            z = V_SAT * torch.tanh(z / V_SAT)
        ex_w = w[f"blocks.{i}.expand.weight"]; ex_b = w[f"blocks.{i}.expand.bias"]
        if "tanh_mul" in mode:
            prod = z.unsqueeze(1) * ex_w.view(1, -1, CH, 1)
            z = V_SAT * torch.tanh(prod / V_SAT).sum(2) + ex_b[None, :, None]
        else:
            z = F.conv1d(z, ex_w, ex_b, 1)
        z = F.relu(z)
        co_w = w[f"blocks.{i}.collapse.weight"]; co_b = w[f"blocks.{i}.collapse.bias"]
        if "tanh_mul" in mode:
            prod = z.unsqueeze(1) * co_w.view(1, -1, CH, 1)
            zc = V_SAT * torch.tanh(prod / V_SAT).sum(2) + co_b[None, :, None]
        else:
            zc = F.conv1d(z, co_w, co_b, 1)
        h = h + zc
        skip = skip + zc * w[f"skip_gates.{i}"]
    o = F.relu(skip)
    return F.conv1d(o, w["head.1.weight"], w["head.1.bias"])[-n:, 0]

with torch.no_grad():
    print("=== 消融诊断 (5s, device=%s) ===" % DEVICE)
    print(f"{'mode':20s} {'NRMS':>8s}")
    print(f"{'ideal':20s} {nrms(fwd(sd,'ideal'), yt):8.4f}")
    sdq = {k: quant6(v) for k, v in sd.items()}
    print(f"{'quant6':20s} {nrms(fwd(sdq,'ideal'), yt):8.4f}")
    print(f"{'quant6+tanh_dw':20s} {nrms(fwd(sdq,'tanh_dw'), yt):8.4f}")
    print(f"{'quant6+tanh_mul':20s} {nrms(fwd(sdq,'tanh_mul'), yt):8.4f}")
    print(f"{'quant6+tanh_both':20s} {nrms(fwd(sdq,'tanh_dw+tanh_mul'), yt):8.4f}")
    print(f"{'fp32+tanh_both':20s} {nrms(fwd(sd,'tanh_dw+tanh_mul'), yt):8.4f}")
    # 检查信号幅值
    h = F.conv1d(xt, sdq["inp.weight"], sdq["inp.bias"], 1)
    print(f"\n输入RMS={float(xt.square().mean().sqrt()):.3f}  inp后h RMS={float(h.square().mean().sqrt()):.3f}  |h|max={float(h.abs().max()):.3f}")
print("DONE")
