"""验证残差缩放 α 对"幅度"与"精度"的影响（不重训，只改前向）

方法：加载已训练的 a-grp4 权重，前向时把每层 layer1x1 输出乘 α。
     α=1.0 是原模型（精度正常但幅度爆）
     α<1.0 幅度被压制，但精度会掉（因为权重是按 α=1 训的）

这个扫描告诉我们两件事：
1. α 与残差幅度的定量关系（能否压到 O(1)）
2. 不重训时 α 的代价有多大 -> 决定是否值得投入重训
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/root/autodl-tmp")
from nam.models.losses import esr as official_esr  # noqa: E402
from nam.models.wavenet._wavenet import WaveNet  # noqa: E402

DEV = "cuda"
RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/n3_a-grp4_s7"
ck = sorted(Path(RUN).glob("*/lightning_logs/version_0/checkpoints/epoch=*.ckpt"))[-1]
cfg_p = ck.parent.parent.parent.parent / "config_model.json"
cfg = json.load(open(cfg_p))["net"]["config"]
sd = torch.load(ck, map_location=DEV, weights_only=False)["state_dict"]
clean = {}
for k, v in sd.items():
    while k.startswith("_net."):
        k = k[len("_net."):]
    if "_has_sample" in k or "_sample_rate" in k:
        continue
    clean[k] = v

x, _ = sf.read("data/tsa15_test_input.wav", dtype="float32")
yt, _ = sf.read("data/tsa15_test_target.wav", dtype="float32")
x = torch.from_numpy(x[:48000 * 3]).to(DEV)
yt = torch.from_numpy(yt[:48000 * 3]).to(DEV)


def run_alpha(alpha):
    m = WaveNet.init_from_config(cfg).to(DEV).eval()
    m.load_state_dict(clean, strict=False)
    # 打补丁：每层 layer1x1 输出乘 alpha
    for la in m._layer_arrays:
        for lay in la._layers:
            l1 = getattr(lay, "_layer1x1", None)
            if l1 is None:
                continue
            orig = l1.forward
            l1.forward = (lambda o: (lambda z, *a, **k: alpha * o(z, *a, **k)))(orig)

    la = m._layer_arrays[0]
    with torch.no_grad():
        xc = x[None, None]                       # (1,1,T)
        h, yres = la(xc, xc, head_input=None)    # 需要 3D
        amp = yres.abs().max().item()
        pred = m(xc)[0, 0]                       # 官方 WaveNet 要 3D 输入
    off = yt.numel() - pred.numel()
    t = yt[off:]
    e = float(official_esr(pred.float().reshape(1, -1), t.float().reshape(1, -1)))
    return amp, e


print(f"[cfg] {RUN}")
print(f"{'alpha':>6s} {'残差|y|max':>12s} {'ESR':>11s} {'NRMS':>8s}")
print("-" * 42)
for a in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
    amp, e = run_alpha(a)
    print(f"{a:6.1f} {amp:12.2f} {e:11.4e} {np.sqrt(max(e,0)):8.4f}")
