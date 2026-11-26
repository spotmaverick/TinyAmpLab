"""检查残差缩放训练后的实际残差幅度（对比未缩放模型）"""
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, "/root/autodl-tmp")
import analog_scaled_wavenet  # noqa: E402
from nam.models import factory  # noqa: E402
from nam.train import util as u  # noqa: E402

factory.register(
    "ScaledWaveNet", analog_scaled_wavenet.ScaledWaveNet.init_from_config, overwrite=True
)

DEV = "cuda"
run = sys.argv[1]
fs = sorted(
    Path(run).glob("*/lightning_logs/version_0/checkpoints/epoch=*.ckpt"),
    key=lambda p: int(p.name.split("epoch=")[1].split("_")[0]),
)
ck = fs[-1]
cfg_p = ck.parent.parent.parent.parent / "config_model.json"
cfg = json.load(open(cfg_p))

cls = u.resolve_lightning_module_class(cfg)
m = cls.init_from_config(cfg).to(DEV).eval()
sd = torch.load(ck, map_location=DEV, weights_only=False)["state_dict"]
clean = {}
for k, v in sd.items():
    while k.startswith("_net."):
        k = k[len("_net."):]
    if "_has_sample" in k or "_sample_rate" in k:
        continue
    clean[k] = v
m.load_state_dict(clean, strict=False)

inner = m.net
print(f"[{run}] alpha={getattr(inner, '_residual_scale', None)}")
target = getattr(inner, "_net", inner)
la = target._layer_arrays[0]

x, _ = sf.read("data/tsa15_test_input.wav", dtype="float32")
x = torch.from_numpy(x[: 48000 * 2]).to(DEV)
with torch.no_grad():
    h, yr = la(x[None, None], x[None, None], head_input=None)
print(f"残差 |y| max = {yr.abs().max().item():.2f}, p99 = {yr.abs().quantile(0.99).item():.2f}")
