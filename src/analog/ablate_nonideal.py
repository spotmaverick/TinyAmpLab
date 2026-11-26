"""非理想性分解：逐项关闭，看谁主导精度退化

对 a-grp4（12ch 分组，LeakyReLU）逐项消融：
  量化 / 权重失配+温漂 / Gilbert 三次失真 / 差分对失调 / 热噪声
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/root/autodl-tmp")
import analog_sim_generic as A  # noqa: E402
from analog_sim_generic import DevSim, load_ckpt, sim_forward  # noqa: E402
from nam.models.losses import esr as official_esr  # noqa: E402

DEV = "cuda"
BASE_P = dict(A.P)

ck = sorted(Path("runs/n3_a-grp4_s7").glob(
    "*/lightning_logs/version_0/checkpoints/epoch=*.ckpt"))[-1]
cfg, sd = load_ckpt(str(ck), str(ck.parent.parent.parent.parent / "config_model.json"))

x, _ = sf.read("data/tsa15_test_input.wav", dtype="float32")
yt, _ = sf.read("data/tsa15_test_target.wav", dtype="float32")
x = torch.from_numpy(x[:48000 * 3]).to(DEV)
yt = torch.from_numpy(yt[:48000 * 3]).to(DEV)


def ev(label, **over):
    p = dict(BASE_P)
    p.update(over)
    A.P.clear()
    A.P.update(p)
    dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(0))
    with torch.no_grad():
        pr = sim_forward(dev, sd, cfg, x)
    off = yt.numel() - pr.numel()
    t = yt[off:]
    e = float(official_esr(pr.float().reshape(1, -1), t.float().reshape(1, -1)))
    print(f"{label:24s} ESR={e:.5e}  NRMS={np.sqrt(max(e,0)):.5f}")
    A.P.clear()
    A.P.update(BASE_P)
    return e


base = ev("全开（基准）")
ev("关量化 (BITS=32)", BITS=32)
ev("关失配+温漂", SIGMA_MISMATCH=0.0, DRIFT=0.0)
ev("关 Gilbert 失真", BETA_H3=0.0)
ev("关失调", SIGMA_OFFSET=0.0)
ev("关热噪声", SNR_DB=999)
print()
ev("全部关闭（应≈数字模型）", BITS=32, SIGMA_MISMATCH=0.0, DRIFT=0.0,
   BETA_H3=0.0, SIGMA_OFFSET=0.0, SNR_DB=999)
print(f"\n基准 ESR={base:.5e}，为数字模型的 {base/4.29e-3:.1f}×")
