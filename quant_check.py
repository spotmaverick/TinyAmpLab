"""权重量化敏感性分析：模拟浮栅/Crossbar 阵列的有限精度现实。

对 DCU 训练好的权重做逐张量对称量化（b-bit），在本地 CPU 上评估 TSA15 test 段 NRMS。
用法: python quant_check.py <variant> [bits...]   例: python quant_check.py v_tcn812 8 6 5 4
"""
import sys
import importlib.util
import numpy as np
import soundfile as sf
import torch

VARIANT = sys.argv[1]
BITS = [int(b) for b in sys.argv[2:]] or [8, 6, 5, 4]

spec = importlib.util.spec_from_file_location("lgw2", "lgw_dcu2.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

ch, L, gate, leaky, skg, dexp, head2 = m.CONFIGS[VARIANT]
model = m.LGNet(ch, L, gate, leaky, skg, dexp, head2)
sd = torch.load(f"dcu_runs/{VARIANT}/lgw_{VARIANT}.pt", map_location="cpu")
model.load_state_dict(sd)
model.eval()

tx = torch.from_numpy(sf.read("data/tsa15_test_input.wav")[0].astype(np.float32))
ty = torch.from_numpy(sf.read("data/tsa15_test_target.wav")[0].astype(np.float32))


def eval_nrms(net):
    with torch.no_grad():
        pred = net(tx).numpy()
    ref = ty.numpy()
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def quantize_state(sd, bits):
    qmax = 2 ** (bits - 1) - 1
    out = {}
    for k, v in sd.items():
        if v.dtype in (torch.float32, torch.float64) and v.numel() > 1:
            scale = v.abs().max() / qmax
            if scale == 0:
                out[k] = v
            else:
                out[k] = torch.clamp(torch.round(v / scale), -qmax, qmax) * scale
        else:
            out[k] = v
    return out


base = eval_nrms(model)
print(f"[{VARIANT}] params={sum(p.numel() for p in model.parameters())} fp32 test NRMS = {base:.4f}")
for b in BITS:
    model.load_state_dict(quantize_state(sd, b))
    nr = eval_nrms(model)
    print(f"  {b:2d}-bit: test NRMS {nr:.4f}  (Δ {nr-base:+.4f})")
