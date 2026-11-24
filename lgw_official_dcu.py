# LGW-TCN 冠军架构 + 官方 NAM 0.13 训练协议 (DCU, torch 1.10 兼容)
# 目的: 分离 "架构 vs 训练协议" 的贡献 —— 与官方包同协议下直接对比 LGW-TCN 与官方 WaveNet
#
# 用法: python lgw_official_dcu.py <variant> <dataset> [epochs]
#   variant ∈ CONFIGS 键; 默认协议: 官方固定网格窗口(ny=8192)/MSE/Adam 0.004/ExponentialLR 0.993/batch16
# 评价: 官方 esr 与 NRMS 双口径 (NRMS = sqrt(esr))
import sys, json, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path("/work/home/spotmaverick/nam_project/data")
OUT = Path("/work/home/spotmaverick/nam_project/e2_lgw_official")
NY = 8192
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIGS = {
    "v_tcn812h1": (8, 12, "none", False, True, 7, False),   # 冠军架构
    "v_tcn713h1": (7, 13, "none", False, True, 7, False),   # 压线 1855
    "v_tcn814":   (8, 14, "none", False, True, 7, False),
    "v_tcn1016":  (10, 16, "none", False, True, 7, False),  # 加宽探索
    "v_tcn1218":  (12, 18, "none", False, True, 7, False),  # 对标 feather 规模
}

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
        return x + z, z

class LGNet(nn.Module):
    def __init__(self, ch, layers, gate, leaky, skip_gates, dil_exp, head2):
        super().__init__()
        self.inp = nn.Conv1d(1, ch, 1)
        self.blocks = nn.ModuleList(
            [Block(ch, 2 ** (i % (dil_exp + 1)), gate, leaky) for i in range(layers)])
        self.skip_gates = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(layers)]) \
            if skip_gates else None
        self.head = nn.Sequential(nn.ReLU(), nn.Conv1d(ch, 1, 1))

    def forward(self, x):
        # x: (B,T) -> (B,T) 全长因果输出
        h = self.inp(x.unsqueeze(1))
        skip = 0
        for i, b in enumerate(self.blocks):
            h, z = b(h)
            skip = skip + z * self.skip_gates[i] if self.skip_gates is not None else skip + z
        return self.head(skip).squeeze(1)

def receptive_field(layers, dil_exp):
    dils = [2 ** (i % (dil_exp + 1)) for i in range(layers)]
    return 1 + 2 * sum(dils)

def load_wav(p):
    x, sr = sf.read(p)
    assert sr == 48000, f"{p}: sr={sr}"
    return x.astype(np.float32)

class GridDataset(torch.utils.data.Dataset):
    """官方协议: i=idx*ny; x 窗 [i, i+nx+ny-1); y 窗 [i+nx-1, +ny)"""
    def __init__(self, x, y, nx, ny):
        self.x, self.y, self.nx, self.ny = x, y, nx, ny
    def __getitem__(self, idx):
        i = idx * self.ny
        return (torch.from_numpy(self.x[i: i + self.nx + self.ny - 1]),
                torch.from_numpy(self.y[i + self.nx - 1: i + self.nx - 1 + self.ny]))
    def __len__(self):
        return (len(self.x) - self.nx + 1) // self.ny

def esr_official(p, t):
    return torch.mean(torch.mean(torch.square(p - t), dim=1)
                      / torch.mean(torch.square(t), dim=1)).item()

def metrics(pred, ref):
    esr = esr_official(pred[None], ref[None])
    nrms = float((pred.double() - ref.double()).norm() / ref.double().norm())
    return esr, nrms

def train(variant, dataset, epochs):
    t0 = time.time()
    cfg = CONFIGS[variant]
    ch, layers, gate, leaky, skg, dexp, head2 = cfg
    outdir = OUT / f"{variant}_{dataset}_official"; outdir.mkdir(parents=True, exist_ok=True)
    model = LGNet(ch, layers, gate, leaky, skg, dexp, head2).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    rf = receptive_field(layers, dexp)
    print(f"[{variant}] params={n_params} RF={rf}({rf/48000*1000:.1f}ms) device={DEVICE}")

    x = load_wav(DATA / f"{dataset}_train_input.wav")
    y = load_wav(DATA / f"{dataset}_train_target.wav")
    ds = GridDataset(x, y, rf, NY)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, drop_last=True, num_workers=2)
    print(f"train windows={len(ds)} steps/epoch={len(dl)}")

    opt = torch.optim.Adam(model.parameters(), lr=0.004)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.993)
    log = open(outdir / "train.log", "w")
    xv = torch.from_numpy(load_wav(DATA / f"{dataset}_val_input.wav"))[None].to(DEVICE)
    yv = torch.from_numpy(load_wav(DATA / f"{dataset}_val_target.wav")).to(DEVICE)
    best_val, best_state = None, None
    for ep in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)[:, -NY:]          # (B,T_full) 取尾 ny 对齐
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            o = model(xv)[0, -NY:]              # val 全段尾对齐 (LGW 因果全输出, val 20.6s)
            v = esr_official(o[None], yv[-NY:][None])
        line = f"epoch {ep} loss={tot/max(nb,1):.6f} val_esr={v:.6f} lr={sched.get_last_lr()[0]:.6f}"
        print(line); log.write(line + "\n"); log.flush()
        if best_val is None or v < best_val:
            best_val = v
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    torch.save({"state_dict": best_state, "variant": variant, "config": cfg,
                "params": n_params, "rf": rf, "dataset": dataset,
                "best_val_esr": best_val, "protocol": "official-0.13"},
               outdir / "weights.pt")
    # 全段评价 (train/val/test), 官方 esr + NRMS 双口径
    model.load_state_dict(best_state)
    model.eval()
    res = {"variant": variant, "dataset": dataset, "params": n_params, "rf": rf,
           "epochs": epochs, "train_seconds": round(time.time() - t0, 1),
           "protocol": "official-0.13"}
    for split in ["train", "val", "test"]:
        xi = torch.from_numpy(load_wav(DATA / f"{dataset}_{split}_input.wav"))[None].to(DEVICE)
        yi = torch.from_numpy(load_wav(DATA / f"{dataset}_{split}_target.wav")).to(DEVICE)
        with torch.no_grad():
            o = model(xi)[0]
        e, n = metrics(o.cpu(), yi.cpu())
        res[f"esr_{split}"] = round(e, 6)
        res[f"nrms_{split}"] = round(n, 6)
    json.dump(res, open(outdir / "result.json", "w"), indent=2)
    print("RESULT " + json.dumps(res))

if __name__ == "__main__":
    variant, dataset = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    train(variant, dataset, epochs)
