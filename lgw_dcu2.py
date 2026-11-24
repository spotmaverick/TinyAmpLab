"""LGW v2：地毯式架构搜索的统一训练脚本。

变体维度：gate ∈ {none, classic, sgate, dual} × leaky × skip_gates × dil_exp × 宽度/深度。
统一协议：TSA15、MSE、恒定 LR 3e-3、seed、batch 4s、手动 RIFF IO。
每 run 打印完整配置行，产物 = lgw_<variant>.pt + lgw_<variant>_test_out.wav + <variant>.out。

用法: python lgw_dcu2.py <variant> <epochs> <seed>
"""
import sys
import struct
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "a2lite"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 7
LRSCHED = sys.argv[4] if len(sys.argv) > 4 else "const"   # const | cosine
BATCH = 4 * 48000
LR = 3e-3
SR = 48000

# variant -> (ch, layers, gate, leaky, skip_gates, dil_exp, head2)
CONFIGS = {
    "a2lite":        (7,  11, "classic", False, False, 7, False),
    "lgw1":          (8,  14, "dual",    True,  True,  7, True),
    "lgw2":          (12, 14, "dual",    True,  True,  7, True),
    "v_tcn812":      (8,  12, "none",    False, True,  7, True),
    "v_classic814":  (8,  14, "classic", True,  True,  7, True),
    "v_sgate814":    (8,  14, "sgate",   True,  True,  7, True),
    "v_rf814d9":     (8,  14, "dual",    True,  True,  9, True),
    # P1: 预算内对决（全部 ≤2178；1476 档 = 官方 A2 Lite 预算的 79%）
    "v_dual418":     (4,  18, "dual",    True,  True,  7, False),
    "v_dual512":     (5,  12, "dual",    True,  True,  7, False),
    "v_classic419":  (4,  19, "classic", True,  True,  7, False),
    "v_classic614":  (6,  14, "classic", True,  True,  7, False),
    "v_sgate419":    (4,  19, "sgate",   True,  True,  7, False),
    "v_sgate614":    (6,  14, "sgate",   True,  True,  7, False),
    # P2: TCN 冠军精修（无门控预算内最强，堆宽度/感受野/非对称塑形）
    "v_tcn814":      (8,  14, "none",    False, True,  7, False),
    "v_tcn814d9":    (8,  14, "none",    False, True,  9, False),
    "v_tcn814d9l":   (8,  14, "none",    True,  True,  9, False),
    # P2b: 预算内精修（冠军 v_tcn812 的等参 RF 对照 + 头部消融）
    "v_tcn812d9":    (8,  12, "none",    False, True,  9, True),
    "v_tcn812h1":    (8,  12, "none",    False, True,  7, False),
    # P3: 官方预算压线（1849 ≤ 1871）+ 冠军长训练精修
    "v_tcn713h1":    (7,  13, "none",    False, True,  7, False),
    "v_tcn812h1_p3": (8,  12, "none",    False, True,  7, False),
    "v_tcn713h1_p3": (7,  13, "none",    False, True,  7, False),
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


class Block(nn.Module):
    def __init__(self, ch, dil, gate, leaky):
        super().__init__()
        self.gate, self.leaky = gate, leaky
        self.pad = 2 * dil
        self.dw = nn.Conv1d(ch, ch, 3, dilation=dil, groups=ch)
        nexp = {"dual": 4, "classic": 2, "sgate": 2, "none": 1}[gate]
        self.expand = nn.Conv1d(ch, nexp * ch, 1)
        self.collapse = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        h = self.dw(F.pad(x, (self.pad, 0)))
        if self.gate == "dual":
            g1, f1, g2, f2 = self.expand(h).chunk(4, dim=1)
            z = torch.tanh(f1) * torch.sigmoid(g1) + torch.tanh(f2) * torch.sigmoid(g2)
        elif self.gate == "classic":
            g, f = self.expand(h).chunk(2, dim=1)
            z = torch.tanh(f) * torch.sigmoid(g)
        elif self.gate == "sgate":
            g, f = self.expand(h).chunk(2, dim=1)
            z = f * torch.tanh(g)
        else:
            z = self.expand(h)
            z = F.leaky_relu(z, 0.15) if self.leaky else F.relu(z)
        z = self.collapse(z)
        if self.gate != "none" and self.leaky:
            z = F.leaky_relu(z, 0.15)
        return x + z, z


class LGNet(nn.Module):
    def __init__(self, ch, layers, gate, leaky, skip_gates, dil_exp, head2):
        super().__init__()
        self.inp = nn.Conv1d(1, ch, 1)
        self.blocks = nn.ModuleList(
            [Block(ch, 2 ** (i % (dil_exp + 1)), gate, leaky) for i in range(layers)]
        )
        self.skip_gates = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(layers)]) \
            if skip_gates else None
        if head2:
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
    ch, layers, gate, leaky, skg, dexp, head2 = CONFIGS[VARIANT]
    t0 = time.time()

    dx = load_wav_float("data/tsa15_train_input.wav").to(dev)
    dy = load_wav_float("data/tsa15_train_target.wav").to(dev)
    vx = load_wav_float("data/tsa15_val_input.wav").to(dev)
    vy = load_wav_float("data/tsa15_val_target.wav").to(dev)
    tx = load_wav_float("data/tsa15_test_input.wav").to(dev)
    ty = load_wav_float("data/tsa15_test_target.wav").to(dev)

    model = LGNet(ch, layers, gate, leaky, skg, dexp, head2).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-4) \
        if LRSCHED == "cosine" else None
    nbatch = len(dx) // BATCH
    nparam = sum(p.numel() for p in model.parameters())
    dils = [2 ** (i % (dexp + 1)) for i in range(layers)]
    rf = 1 + 2 * sum(dils)
    macs = sum(3 * ch + {"dual": 4, "classic": 2, "sgate": 2, "none": 1}[gate] * ch * ch + ch * ch
               for _ in range(layers)) + ch + (2 * ch * ch + ch if head2 else ch)
    print("[CFG] %s ch=%d L=%d gate=%s leaky=%s skipgates=%s dil_exp=%d head2=%d sched=%s "
          "params=%d RF=%d(%.1fms) MACs/s=%.0f state=%.1fKB" % (
              VARIANT, ch, layers, gate, leaky, skg, dexp, head2, LRSCHED, nparam, rf, rf / SR * 1000,
              macs, ch * sum(dils) * 4 / 1024), flush=True)

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
        if sched is not None:
            sched.step()
        if ep % 10 == 0 or ep == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                vc, vn = metrics(model(vx), vy)
            model.train()
            print("ep %3d loss %.6f val_corr %.4f val_NRMS %.4f (%.0fs)" % (
                ep, tot / nbatch, vc, vn, time.time() - t0), flush=True)

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
