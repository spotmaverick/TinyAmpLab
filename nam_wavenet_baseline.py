"""NAM WaveNet 架构基线（对标 A1 standard / A2 同类拓扑）。

NAM 公开定义的 WaveNet 结构：
    输入 = x(t) 单采样 + 因果 dilated conv 堆叠
    每层 = depthwise dilated conv(kernel=3) -> 1x1 扩展 -> gated tanh*sigma
          -> 1x1 收缩 -> residual + skip
    输出 = skip 求和 -> 1x1 head -> y(t)
这是 NAM 官方最常用、精度最高的架构系列（A1 WaveNet / A2 Slimmable 同族）。

数据与指标口径与 dual 完全一致：TSA15 train/val/test 48k float32, NRMS/corr。
用法: python nam_wavenet_baseline.py [epochs] [seed] [channels] [layers]
"""
import sys
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
CHANNELS = int(sys.argv[3]) if len(sys.argv) > 3 else 8      # A2-Full 量级
LAYERS = int(sys.argv[4]) if len(sys.argv) > 4 else 10       # 感受野随层数增加
SR = 48000
BATCH_SAMPLES = 4 * SR
KERNEL = 3
LR = 3e-3

torch.manual_seed(SEED)
np.random.seed(SEED)


def load(p):
    return torch.from_numpy(sf.read(p)[0].astype(np.float32))


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilation = dilation
        self.pad = (KERNEL - 1) * dilation
        self.depthwise = nn.Conv1d(channels, channels, KERNEL,
                                   dilation=dilation, groups=channels, padding=0)
        self.expand = nn.Conv1d(channels, 2 * channels, 1)
        self.collapse = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        # 因果卷积：左侧补零，右侧裁掉
        xp = F.pad(x, (self.pad, 0))
        h = self.depthwise(xp)
        h = self.expand(h)
        gate, filt = h.chunk(2, dim=1)
        h = torch.tanh(filt) * torch.sigmoid(gate)
        h = self.collapse(h)
        # residual + skip：skip 返回给外层累加
        return x + h, h


class NamWaveNet(nn.Module):
    def __init__(self, channels=8, layers=10):
        super().__init__()
        self.input = nn.Conv1d(1, channels, 1)
        self.blocks = nn.ModuleList()
        for i in range(layers):
            dilation = 2 ** (i % 8)
            self.blocks.append(DilatedBlock(channels, dilation))
        self.head = nn.Sequential(nn.ReLU(), nn.Conv1d(channels, 1, 1))

    def forward(self, x):
        # 统一为 (B, 1, N)
        if x.dim() == 1:
            x = x.unsqueeze(0)      # (N,) -> (1, N)
        h = self.input(x.unsqueeze(1))   # (B, N) -> (B, 1, N)
        skip = 0
        for block in self.blocks:
            h, s = block(h)
            skip = skip + s
        return self.head(skip).squeeze(1)


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x.unsqueeze(0)).squeeze(0).numpy()
        ref = y.numpy()
    a, b = pred - pred.mean(), ref - ref.mean()
    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    nrms = float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))
    model.train()
    return corr, nrms


if __name__ == "__main__":
    dx = load("data/tsa15_train_input.wav")
    dy = load("data/tsa15_train_target.wav")
    vx, vy = load("data/tsa15_val_input.wav"), load("data/tsa15_val_target.wav")
    tx, ty = load("data/tsa15_test_input.wav"), load("data/tsa15_test_target.wav")
    n_batches = len(dx) // BATCH_SAMPLES

    model = NamWaveNet(CHANNELS, LAYERS)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    print(f"NAM WaveNet: channels={CHANNELS} layers={LAYERS} "
          f"params={sum(p.numel() for p in model.parameters())} "
          f"| batches/epoch={n_batches} | epochs={EPOCHS} | seed={SEED}", flush=True)

    for ep in range(EPOCHS):
        starts = torch.randint(0, len(dx) - BATCH_SAMPLES, (n_batches,))
        ep_loss = 0.0
        for s in starts:
            s = int(s)
            opt.zero_grad()
            pred = model(dx[s:s + BATCH_SAMPLES])
            loss = loss_fn(pred, dy[s:s + BATCH_SAMPLES])
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
