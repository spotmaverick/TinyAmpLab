"""LGW: Leaky-Gated WaveNet —— 自研架构（双组门控 + LeakyReLU 塌缩 + skip 门控）。

对照：a2lite 复现（ch7×L11，vanilla tanh·sigmoid 门控，≈1.9k 参数，对标 NAM A2 Lite 规模）。
统一协议：TSA15 train/val/test、MSE、恒定 LR 3e-3、seed 7、batch 4s、手动 RIFF IO（无 libsndfile 依赖）。

用法: python lgw_dcu.py <a2lite|lgw1|lgw2> <epochs> <seed>
"""
import sys
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "a2lite"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 7
BATCH = 4 * 48000
LR = 3e-3
SR = 48000

CONFIGS = {
    # (channels, layers, use_dual_gate, use_leaky, use_skip_gates)
    "a2lite": (7, 11, False, False, False),
    "lgw1":   (8, 14, True,  True,  True),
    "lgw2":   (12, 14, True, True,  True),
}

torch.manual_seed(SEED)


def load_wav_float(path):
    with open(path, "rb") as f:
        f.read(12)
        while True:
            tag = f.read(4)
            if len(tag) < 4:
                raise ValueError("no data chunk in " + path)
            size = struct.unpack("<I", f.read(4))[0]
            if tag == b"data":
                raw = f.read(size)
                n = size // 4
                return torch.tensor(struct.unpack("<%df" % n, raw[: n * 4]), dtype=torch.float32)
            f.read(size + (size & 1))


def write_wav_float(path, x_np, sr=SR):
    data = x_np.astype("<f4").tobytes()
    hdr = struct.pack("<4sI4s", b"RIFF", 36 + len(data), b"WAVE")
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 3, 1, sr, sr * 4, 4, 32)
    dat = struct.pack("<4sI", b"data", len(data))
    with open(path, "wb") as f:
        f.write(hdr + fmt + dat + data)


class GatedDilatedBlock(nn.Module):
    """扩张因果卷积 + 门控单元。a2lite: 单组 tanh·sigmoid；lgw: 双组门控 + LeakyReLU 塌缩。"""

    def __init__(self, ch, dil, dual_gate, leaky):
        super().__init__()
        self.dual_gate = dual_gate
        self.leaky = leaky
        self.pad = (3 - 1) * dil
        self.dw = nn.Conv1d(ch, ch, 3, dilation=dil, groups=ch)
        if dual_gate:
            self.expand = nn.Conv1d(ch, 4 * ch, 1)   # g1,f1,g2,f2
        else:
            self.expand = nn.Conv1d(ch, 2 * ch, 1)   # g,f
        self.collapse = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        h = self.dw(F.pad(x, (self.pad, 0)))
        if self.dual_gate:
            g1, f1, g2, f2 = self.expand(h).chunk(4, dim=1)
            z = torch.tanh(f1) * torch.sigmoid(g1) + torch.tanh(f2) * torch.sigmoid(g2)
        else:
            g, f = self.expand(h).chunk(2, dim=1)
            z = torch.tanh(f) * torch.sigmoid(g)
        z = self.collapse(z)
        if self.leaky:
            z = F.leaky_relu(z, 0.15)
        return x + z, z


class LGNet(nn.Module):
    def __init__(self, ch, layers, dual_gate, leaky, skip_gates):
        super().__init__()
        self.inp = nn.Conv1d(1, ch, 1)
        self.blocks = nn.ModuleList(
            [GatedDilatedBlock(ch, 2 ** (i % 8), dual_gate, leaky) for i in range(layers)]
        )
        self.skip_gates = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(layers)]) \
            if skip_gates else None
        if dual_gate:
            self.head = nn.Sequential(nn.ReLU(), nn.Conv1d(ch, ch, 1), nn.Conv1d(ch, 1, 1))
        else:
            self.head = nn.Sequential(nn.ReLU(), nn.Conv1d(ch, 1, 1))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.inp(x.unsqueeze(1))
        skip = 0
        for i, b in enumerate(self.blocks):
            h, z = b(h)
            skip = skip + z * self.skip_gates[i] if self.skip_gates is not None else skip + z
        return self.head(skip).squeeze(1)


def metrics(pred, ref):
    p, r = pred.double() - pred.double().mean(), ref.double() - ref.double().mean()
    corr = float((p * r).sum() / (p.norm() * r.norm() + 1e-12))
    nrms = float((pred.double() - ref.double()).norm() / ref.double().norm())
    return corr, nrms


if __name__ == "__main__":
    dev = "cuda"
    ch, layers, dual, leaky, skg = CONFIGS[VARIANT]
    t0 = __import__("time").time()

    dx = load_wav_float("data/tsa15_train_input.wav").to(dev)
    dy = load_wav_float("data/tsa15_train_target.wav").to(dev)
    vx = load_wav_float("data/tsa15_val_input.wav").to(dev)
    vy = load_wav_float("data/tsa15_val_target.wav").to(dev)
    tx = load_wav_float("data/tsa15_test_input.wav").to(dev)
    ty = load_wav_float("data/tsa15_test_target.wav").to(dev)

    model = LGNet(ch, layers, dual, leaky, skg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    nbatch = len(dx) // BATCH
    nparam = sum(p.numel() for p in model.parameters())
    print("[%s] ch=%d layers=%d dual=%s leaky=%s skipgates=%s params=%d ep=%d seed=%d" % (
        VARIANT, ch, layers, dual, leaky, skg, nparam, EPOCHS, SEED), flush=True)

    for ep in range(EPOCHS):
        starts = torch.randint(0, len(dx) - BATCH, (nbatch,), device=dev)
        tot = 0.0
        for s in starts:
            s = int(s)
            opt.zero_grad()
            pred = model(dx[s:s + BATCH])
            loss = F.mse_loss(pred, dy[s:s + BATCH])
            loss.backward()
            opt.step()
            tot += loss.item()
        if ep % 10 == 0 or ep == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                vc, vn = metrics(model(vx), vy)
            model.train()
            print("ep %3d loss %.6f val_corr %.4f val_NRMS %.4f (%.0fs)" % (
                ep, tot / nbatch, vc, vn, __import__("time").time() - t0), flush=True)

    print("=== FINAL [%s] ===" % VARIANT, flush=True)
    model.eval()
    with torch.no_grad():
        for tag, x, y in [("train", dx, dy), ("val", vx, vy), ("test", tx, ty)]:
            c, n = metrics(model(x), y)
            print("[%s] corr %+.4f NRMS %.4f" % (tag, c, n), flush=True)
        pred = model(tx).cpu().numpy()

    torch.save(model.state_dict(), "lgw_%s.pt" % VARIANT)
    write_wav_float("lgw_%s_test_out.wav" % VARIANT, pred)
    print("saved lgw_%s.pt / lgw_%s_test_out.wav" % (VARIANT, VARIANT), flush=True)
