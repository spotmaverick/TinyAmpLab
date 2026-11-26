"""V_SAT 扫描：Gilbert cell 的饱和电压决定模拟实现的动态范围需求

物理含义：V_SAT 越大，乘法器越线性（失真小），但电路摆幅/功耗越高。
目标：找到"精度够用"的最小 V_SAT，即电路可接受的最低电源电压。
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
RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/n3_a-grp4_s7"

ck = sorted(Path(RUN).glob("*/lightning_logs/version_0/checkpoints/epoch=*.ckpt"))[-1]
cfg_p = ck.parent.parent.parent.parent / "config_model.json"
cfg, sd = load_ckpt(str(ck), str(cfg_p))

x, _ = sf.read("data/tsa15_test_input.wav", dtype="float32")
yt, _ = sf.read("data/tsa15_test_target.wav", dtype="float32")
x = torch.from_numpy(x[:48000 * 3]).to(DEV)
yt = torch.from_numpy(yt[:48000 * 3]).to(DEV)

# 先看残差流 / preact 的实际幅度，用来定标
m_rf = 1 + sum((cfg["layers_configs"][0]["kernel_size"] - 1)
               * d for d in cfg["layers_configs"][0]["dilations"])
print(f"[cfg] {RUN}  RF={m_rf}")


def ev(vsat, mc=2):
    p = dict(BASE_P)
    p["V_SAT"] = vsat
    A.P.clear()
    A.P.update(p)
    es = []
    for i in range(mc):
        dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(100 + i))
        with torch.no_grad():
            pr = sim_forward(dev, sd, cfg, x)
        off = yt.numel() - pr.numel()
        t = yt[off:]
        es.append(float(official_esr(pr.float().reshape(1, -1), t.float().reshape(1, -1))))
    A.P.clear()
    A.P.update(BASE_P)
    return float(np.mean(es))


print(f"{'V_SAT':>8s}  {'ESR':>11s}  {'NRMS':>8s}  {'相对理想':>9s}")
print("-" * 44)
ideal = None
for vs in (2, 4, 8, 12, 16, 24, 32, 48, 64, 128, 4096):
    e = ev(vs)
    if vs == 4096:
        ideal = e
    print(f"{vs:8.0f}  {e:11.5e}  {np.sqrt(max(e,0)):8.5f}  {e/4.27145e-3:8.1f}×")
print(f"\n（4096 ≈ 无失真参考 ESR={ideal:.5e}）")
