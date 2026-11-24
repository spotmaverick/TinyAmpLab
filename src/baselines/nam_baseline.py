"""NAM 标准架构基线，用于与本项目的 dual-gate 前馈链公平对比。

架构（NAM v2 standard，公开定义）：
    输入 = [x(t-15), ..., x(t)]  共 16 个采样（receptive field=16）
    4 层 MLP（hidden=24, ReLU）+ 线性 skip connection（NAM 的直连路径）
损失 = MSE（NAM 默认口径，不加 STFT）
数据 = 与正式训练完全相同的 TSA15 train/val/test（48kHz float32）
指标 = NRMS + corr，test 为独立未见段

用法: python nam_baseline.py [epochs] [seed]
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # 项目根，兼容归档后的相对路径
import sys
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

SR = 48000
RF = 16
HIDDEN = 24
LAYERS = 4
BATCH_SAMPLES = 4 * SR
LR = 3e-3
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(path):
    return torch.from_numpy(sf.read(path)[0].astype(np.float32))


class NamStandard(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        d = RF
        for _ in range(LAYERS - 1):
            layers += [nn.Linear(d, HIDDEN), nn.ReLU()]
            d = HIDDEN
        layers += [nn.Linear(HIDDEN, 1)]
        self.net = nn.Sequential(*layers)
        # NAM skip：初始化为 x(t) 直通（窗口最后一个采样）
        self.skip = nn.Linear(RF, 1)
        nn.init.zeros_(self.skip.weight)
        with torch.no_grad():
            self.skip.weight[0, -1] = 1.0
        nn.init.zeros_(self.skip.bias)

    def forward(self, windows):
        return (self.net(windows) + self.skip(windows)).squeeze(-1)


def make_windows(x):
    return x.unfold(0, RF, 1)


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(make_windows(x)).cpu().numpy()
        ref = y[RF - 1:].cpu().numpy()
    a, b = pred - pred.mean(), ref - ref.mean()
    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    nrms = float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))
    model.train()
    return corr, nrms


if __name__ == "__main__":
    dx = load(str(ROOT / "data" / "tsa15_train_input.wav")).to(DEVICE)
    dy = load(str(ROOT / "data" / "tsa15_train_target.wav")).to(DEVICE)
    vx, vy = load(str(ROOT / "data" / "tsa15_val_input.wav")).to(DEVICE), load(str(ROOT / "data" / "tsa15_val_target.wav")).to(DEVICE)
    tx, ty = load(str(ROOT / "data" / "tsa15_test_input.wav")).to(DEVICE), load(str(ROOT / "data" / "tsa15_test_target.wav")).to(DEVICE)
    X, Y = make_windows(dx), dy[RF - 1:]
    n_batches = len(X) // BATCH_SAMPLES

    model = NamStandard().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    print(f"NAM standard: params={sum(p.numel() for p in model.parameters())} "
          f"| batches/epoch={n_batches} | epochs={EPOCHS} | seed={SEED} | device={DEVICE}", flush=True)

    for ep in range(EPOCHS):
        starts = torch.randint(0, len(X) - BATCH_SAMPLES, (n_batches,))
        ep_loss = 0.0
        for s in starts:
            s = int(s)
            opt.zero_grad()
            pred = model(X[s:s + BATCH_SAMPLES])
            loss = loss_fn(pred, Y[s:s + BATCH_SAMPLES])
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if ep % 10 == 0 or ep == EPOCHS - 1:
            vc, vn = evaluate(model, vx, vy)
            print(f"ep {ep:3d} | loss {ep_loss / n_batches:.6f} | val corr {vc:.4f} NRMS {vn:.4f}",
                  flush=True)

    print("=== 最终结果 ===", flush=True)
    for tag, x, y in [("train", dx, dy), ("val", vx, vy), ("test", tx, ty)]:
        c, n = evaluate(model, x, y)
        print(f"[{tag}] corr {c:+.4f} | NRMS {n:.4f}", flush=True)

    # DCU 推理交付：保存权重 + 生成 test 段全量输出 wav
    torch.save(model.state_dict(), "nam_mlp_tsa15.pt")
    model.eval()
    with torch.no_grad():
        pred = model(make_windows(tx)).cpu().numpy()
    sf.write("nam_mlp_test_out.wav", pred, SR)
    print("✅ 权重 nam_mlp_tsa15.pt / 输出 nam_mlp_test_out.wav 已生成", flush=True)
