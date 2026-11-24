"""架构对照实验：三个正交改进，在统一数据/种子/超参下公平对比。

变体（--variant）：
  uniform       基线：K=4 相邻抽头，单组门控，无状态（等价当前 40块）
  dilated       +扩张抽头：抽头间隔 1,2,4,8,16,32,64 循环，感受野指数放大
  dual          +双组门控：用满 4×4 输出两路 sum/gate，640 权重全有效
  supply        +输出侧电源状态：一阶 IIR 跟踪输出能量，压缩增益（真 sag）

所有变体共享：40 block、4 抽头、seed、800ep、lr、损失、数据划分。
"""
import argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim

SR = 48000
ALPHA = 0.15
GAIN_PRE = 2.2
BLOCK_CLAMP = 4.0
OUTPUT_SAT = 1.0
NUM_BLOCKS = 40
MAT_DIM = 4
DILATIONS = [1, 2, 4, 8, 16, 32, 64]   # 周期循环；单块窗口随 b 增大，长时依赖由此进入
SUPPLY_TAU = 0.02                        # 电源状态时间常数 20ms（对应电容充放电）
LR = 5e-3
LAMBDA_STFT = 0.01
N_FFT, HOP, WIN = 1024, 256, 1024


def causal_iir_energy(x: torch.Tensor, tau_s: float, sr: int = SR) -> torch.Tensor:
    """一阶 IIR 能量跟踪：s[t]=a*s[t-1]+(1-a)*|x[t]|，用 conv1d 因果实现（O(N)）。
    对应模拟二极管整流+RC 低通，电源能量的物理模型。"""
    a = np.exp(-1.0 / (tau_s * sr))
    L = min(512, int(tau_s * sr * 6) + 1)          # 核长取 6 倍时间常数，尾部可忽略
    k = (1 - a) * (a ** np.arange(L - 1, -1, -1))
    k = torch.tensor(k, dtype=x.dtype, device=x.device).view(1, 1, -1)
    pad = torch.nn.functional.pad(x.abs(), (L - 1, 0))
    return torch.nn.functional.conv1d(pad, k)


def stft_loss(p, t):
    w = torch.hann_window(WIN, device=p.device, dtype=p.dtype)
    sp = torch.stft(p, N_FFT, HOP, WIN, window=w, return_complex=True)
    st = torch.stft(t, N_FFT, HOP, WIN, window=w, return_complex=True)
    return torch.mean((sp.abs() - st.abs()) ** 2) / (torch.mean(st.abs() ** 2) + 1e-8)


class Arch(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        self.variant = variant
        self.use_dilated = "dilated" in variant or "dual" in variant or "supply" in variant
        self.use_dual = "dual" in variant or "supply" in variant
        self.use_supply = "supply" in variant
        rows = 4 if self.use_dual else 2       # 双组门控：sum1,g1,sum2,g2
        self.W = nn.Parameter(torch.randn(NUM_BLOCKS, rows, MAT_DIM) * 0.22)
        if self.use_supply:
            # 电源压缩深度：softplus 保证 >=0（物理上电源只会压缩、不会放大）
            # 初始化使 softplus(w)≈0.15，避免冷启动无梯度，也不会一上来过度压缩
            self.supply_w = nn.Parameter(torch.full((NUM_BLOCKS,), -1.7))

    def pass_audio(self, x, supply):
        audio = x * GAIN_PRE
        for b in range(NUM_BLOCKS):
            d = DILATIONS[b % len(DILATIONS)] if self.use_dilated else 1
            taps = []
            for k in range(MAT_DIM):
                xk = torch.roll(audio, k * d, dims=-1)
                if k > 0:
                    xk[..., :k * d] = 0.0
                taps.append(xk)
            x_vec = torch.stack(taps, dim=-2)
            out = self.W[b] @ x_vec                  # (rows, N)
            s1, g1 = out[..., 0, :], out[..., 1, :]
            s = s1 * torch.tanh(g1)
            if self.use_dual:
                s2, g2 = out[..., 2, :], out[..., 3, :]
                s = s + s2 * torch.tanh(g2)
            if self.use_supply and supply is not None:
                # softplus 保证 depth>=0：电源能量高 -> 增益下降（压缩），符合物理
                depth = torch.nn.functional.softplus(self.supply_w[b])
                gain = 1.0 / (1.0 + depth * supply)   # 压缩而非放大，恒 <=1
                s = s * gain
            y_nl = torch.where(s >= 0, s, ALPHA * s)
            audio = torch.clamp(y_nl + audio, -BLOCK_CLAMP, BLOCK_CLAMP)
        return OUTPUT_SAT * torch.tanh(audio / OUTPUT_SAT)

    def forward(self, x):
        if not self.use_supply:
            return self.pass_audio(x, None)
        # 两遍法：先无状态前向得到输出能量，再用它调制第二遍（一次迭代近似反馈稳态）
        with torch.no_grad():
            y0 = self.pass_audio(x, None)
        energy = causal_iir_energy(y0, SUPPLY_TAU)
        energy = energy / (energy.mean(dim=-1, keepdim=True) + 1e-8)   # 归一化到均值 1
        supply = torch.clamp(energy - 1.0, 0.0, 3.0)                  # 仅高于均值的部分触发压缩
        return self.pass_audio(x, supply)


def metrics(p, t):
    a, b = p - p.mean(), t - t.mean()
    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    nrms = float(np.linalg.norm(p - t) / np.linalg.norm(t))
    return corr, nrms


def load(path):
    w, sr = sf.read(path)
    if w.ndim > 1:
        w = w[:, 0]
    if sr != SR:
        raise RuntimeError(f"expect {SR}Hz, got {sr}")
    return w.astype(np.float32)


def run(variant, dry_path, target_path, epochs, seed=3):
    torch.manual_seed(seed)
    d = load(dry_path)
    t = load(target_path)
    n = min(len(d), len(t))
    d, t = d[:n], t[:n]
    split = int(n * 0.85)
    dt = torch.from_numpy(d[:split]).unsqueeze(0)
    tt = torch.from_numpy(t[:split]).unsqueeze(0)
    dv = torch.from_numpy(d[split:]).unsqueeze(0)
    tv = torch.from_numpy(t[split:]).unsqueeze(0)

    model = Arch(variant)
    opt = optim.Adam(model.parameters(), lr=LR)
    lf = nn.MSELoss()
    for ep in range(epochs):
        opt.zero_grad()
        p = model(dt)
        loss = lf(p, tt) + LAMBDA_STFT * stft_loss(p, tt)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{variant} ep{ep} NaN")
        loss.backward()
        opt.step()
        if ep % 200 == 0:
            print(f"  [{variant}] ep{ep:4d} loss {loss.item():.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        pv = model(dv).squeeze().numpy()
        pa = model(dt).squeeze().numpy()
    cv, nv = metrics(pv, tv.squeeze().numpy())
    ca, na = metrics(pa, tt.squeeze().numpy())
    extra = ""
    if model.use_supply:
        sw = torch.nn.functional.softplus(model.supply_w).detach().numpy()
        extra = f" | supply depth mean {sw.mean():.3f} range [{sw.min():.3f},{sw.max():.3f}]"
    print(f"[{variant:8s}] train corr {ca:+.4f} NRMS {na:.4f} | val corr {cv:+.4f} NRMS {nv:.4f}{extra}", flush=True)
    return dict(variant=variant, train_corr=ca, train_nrms=na, val_corr=cv, val_nrms=nv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="uniform")
    ap.add_argument("--dry", default="data/nam_input.wav")
    ap.add_argument("--target", default="data/nam_output.wav")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    run(a.variant, a.dry, a.target, a.epochs, a.seed)
