"""量化复检 v2：逐张量 vs 逐通道（按输出滤波器）缩放对比。用法: python quant_check2.py <variant>"""
import sys, importlib.util
from pathlib import Path
import numpy as np, soundfile as sf, torch

VARIANT = sys.argv[1]
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("lgw2", ROOT / "lgw_dcu2.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ch, L, gate, leaky, skg, dexp, head2 = m.CONFIGS[VARIANT]
model = m.LGNet(ch, L, gate, leaky, skg, dexp, head2)
sd = torch.load(f"dcu_runs/{VARIANT}/lgw_{VARIANT}.pt", map_location="cpu")
model.load_state_dict(sd); model.eval()

tx = torch.from_numpy(sf.read("data/tsa15_test_input.wav")[0].astype(np.float32))
ty = sf.read("data/tsa15_test_target.wav")[0].astype(np.float32)

def eval_nrms(net):
    with torch.no_grad():
        pred = net(tx).numpy()
    return float(np.linalg.norm(pred - ty) / np.linalg.norm(ty))

def quant(sd, bits, per_channel):
    qmax = 2**(bits-1) - 1
    out = {}
    for k, v in sd.items():
        if v.dtype in (torch.float32, torch.float64) and v.numel() > 1:
            if per_channel and v.dim() >= 2:
                # 沿输出通道维(0)分别算 scale
                flat = v.reshape(v.shape[0], -1)
                scale = flat.abs().max(dim=1).values / qmax
                scale = torch.clamp(scale, min=1e-12).reshape(-1, *([1]*(v.dim()-1)))
                out[k] = torch.clamp(torch.round(v/scale), -qmax, qmax) * scale
            else:
                scale = v.abs().max() / qmax
                out[k] = v if scale == 0 else torch.clamp(torch.round(v/scale), -qmax, qmax)*scale
        else:
            out[k] = v
    return out

base = eval_nrms(model)
print(f"[{VARIANT}] params={sum(p.numel() for p in model.parameters())} fp32 test NRMS={base:.4f}")
for bits in [8, 6, 5]:
    for pc in [False, True]:
        model.load_state_dict(quant(sd, bits, pc))
        nr = eval_nrms(model)
        tag = "逐通道" if pc else "逐张量"
        print(f"  {bits}-bit {tag}: {nr:.4f} (Δ{nr-base:+.4f})")
