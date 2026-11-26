"""诊断残差流幅度：定位 V_SAT 需求高的根因

若残差流 |y| 随层数单调增长，说明架构缺少归一化，
模拟实现时 Gilbert 乘法器必然饱和 —— 这是"模拟友好"的硬约束。
"""
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, "/root/autodl-tmp")
from analog_sim_generic import load_ckpt  # noqa: E402
from nam.models.wavenet._wavenet import WaveNet  # noqa: E402

DEV = "cuda"
RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/n3_a-grp4_s7"
ck = sorted(Path(RUN).glob("*/lightning_logs/version_0/checkpoints/epoch=*.ckpt"))[-1]
cfg_p = ck.parent.parent.parent.parent / "config_model.json"
cfg, sd = load_ckpt(str(ck), str(cfg_p))

x, _ = sf.read("data/tsa15_test_input.wav", dtype="float32")
x = torch.from_numpy(x[:48000 * 2]).to(DEV)

m = WaveNet.init_from_config(json.load(open(cfg_p))["net"]["config"]).to(DEV).eval()
m.load_state_dict(sd, strict=False)
la = m._layer_arrays[0]

print(f"{'layer':>5s} {'dil':>5s} {'|y| p99':>10s} {'|y| max':>10s} "
      f"{'|z| p99':>10s} {'|z| max':>10s}")
print("-" * 60)
with torch.no_grad():
    c = x[None, None]
    y = la._rechannel(c)
    dils = cfg["layers_configs"][0]["dilations"]
    for li, d in enumerate(dils):
        lay = la._layers[li]
        conv_out = lay.conv(y)
        mix = lay._input_mixer(c)[:, :, -conv_out.shape[2]:]
        z1 = conv_out[:, :, -min(conv_out.shape[2], mix.shape[2]):] + \
            mix[:, :, -min(conv_out.shape[2], mix.shape[2]):]
        post = lay._activation(z1)
        lo = lay._layer1x1(post)
        yp99 = y.abs().quantile(0.99).item()
        ymx = y.abs().max().item()
        zp99 = post.abs().quantile(0.99).item()
        zmx = post.abs().max().item()
        if li % 3 == 0 or li == len(dils) - 1:
            print(f"{li:5d} {d:5d} {yp99:10.3f} {ymx:10.3f} {zp99:10.3f} {zmx:10.3f}")
        y = y[:, :, -lo.shape[2]:] + lo
    print("-" * 60)
    print(f"最终残差流 |y| max = {y.abs().max().item():.3f}, "
          f"p99 = {y.abs().quantile(0.99).item():.3f}")
    hs = la._head_rechannel
    hi = None
    # head 输入的幅度
    print(f"head_scale = {cfg.get('head_scale')}")
