"""包络支路消融实验：同一配对、同一初始化，分别训练有/无包络版本并对比。"""
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import importlib.util
from pathlib import Path

# 归档后相对路径变了：从脚本位置回溯到项目根目录
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("tr", ROOT / "train_nam.py")
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)

DRY = str(ROOT / "data" / "b1m3_input.wav")
TARGET = str(ROOT / "data" / "b1m3_target.wav")


def run(use_env: bool, epochs: int = 800):
    torch.manual_seed(3)
    d = torch.from_numpy(sf.read(DRY)[0].astype(np.float32)).unsqueeze(0)
    t = torch.from_numpy(sf.read(TARGET)[0].astype(np.float32)).unsqueeze(0)
    split = int(d.shape[-1] * 0.85)
    dt, tt = d[:, :split], t[:, :split]

    model = tr.GatedBlockChain(tr.NUM_BLOCKS, tr.MAT_DIM)
    if not use_env:
        model.env_w.requires_grad_(False)
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=tr.LR)
    lf = nn.MSELoss()

    for ep in range(epochs):
        opt.zero_grad()
        pred = model(dt)
        loss = lf(pred, tt) + tr.LAMBDA_STFT * tr.stft_mag_loss(pred, tt)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at epoch {ep}")
        loss.backward()
        opt.step()

    with torch.no_grad():
        out = model(d).squeeze().numpy()
    ref = t.squeeze().numpy()
    a, b = out - out.mean(), ref - ref.mean()
    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    nrms = float(np.linalg.norm(out - ref) / np.linalg.norm(ref))

    a_dry = d.squeeze().numpy()
    a2 = a_dry - a_dry.mean()
    corr_dry = float(np.dot(a2, b) / (np.linalg.norm(a2) * np.linalg.norm(b) + 1e-12))
    return corr, nrms, corr_dry, model.env_w.detach().numpy()


if __name__ == "__main__":
    print(f"数据：{DRY} / {TARGET}")
    for use_env in (False, True):
        corr, nrms, corr_dry, env_w = run(use_env)
        tag = "WITH ENV" if use_env else "NO ENV  "
        print(f"{tag}  corr={corr:+.4f}  NRMS={nrms:.4f}  (dry corr={corr_dry:+.4f})")
        if use_env:
            print(f"           env tanh range [{np.tanh(env_w).min():.3f}, {np.tanh(env_w).max():.3f}], "
                  f"mean {np.tanh(env_w).mean():.3f}")
