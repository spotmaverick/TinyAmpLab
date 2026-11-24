"""正式训练脚本：双组门控 Gated-Block（架构搜索胜出方案）。

架构：40 Block × 4 抽头 × 4×4 Crossbar，4 路输出接成两组 sum/gate，
s = sum1*tanh(g1) + sum2*tanh(g2)，640 权重全部有效。
搜结论见 arch_search*.log：uniform(val 0.871 三种子均值) < dual(0.806)，
dilated 与输出侧电源状态均无稳健收益，故不采用。
"""
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# -------------------------- 模型超参（和芯片方案对齐） --------------------------
NUM_BLOCKS = 40
MAT_DIM = 4
WEIGHT_PER_BLOCK = MAT_DIM * MAT_DIM          # 16：4×4 Crossbar
TOTAL_WEIGHTS = NUM_BLOCKS * WEIGHT_PER_BLOCK  # 640，全部有效（双组门控）
ALPHA = 0.15
SAMPLE_RATE = 48000
GAIN_PRE = 2.2
BLOCK_CLAMP = 4.0         # 块内电压轨，防止 40 级残差链数值溢出
OUTPUT_SAT = 1.0          # 输出级 tanh 饱和幅度

LR = 5e-3
EPOCHS = 800
PRINT_INTERVAL = 50
LAMBDA_STFT = 0.01
VAL_SPLIT = 0.15

N_FFT, HOP_LENGTH, WIN_LENGTH = 1024, 256, 1024

# 设备自适应：智算 DCU 识别为 cuda(HIP)，本地无卡时回落 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dry_path = "data/nam_input.wav"
target_path = "data/nam_output.wav"
save_weight_path = "trained_weights_dual40.npy"
infer_out_path = "student_dual40.wav"

# 数据集选择：dafx（默认，单段内切 val）或 tsa15（官方 train/val/test 划分）
DATASET = sys.argv[1] if len(sys.argv) > 1 else "dafx"
if DATASET == "tsa15":
    dry_path = "data/tsa15_train_input.wav"
    target_path = "data/tsa15_train_target.wav"
    val_dry_path, val_target_path = "data/tsa15_val_input.wav", "data/tsa15_val_target.wav"
    test_dry_path, test_target_path = "data/tsa15_test_input.wav", "data/tsa15_test_target.wav"
    save_weight_path = "trained_weights_dual40_tsa15.npy"
    infer_out_path = "student_dual40_tsa15.wav"

SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7   # 架构搜索显示 seed7/11 收敛更稳，默认取 7

print(f"双组门控架构：{NUM_BLOCKS} Block × {MAT_DIM}×{MAT_DIM} = {TOTAL_WEIGHTS} 权重（全部有效）")


class GatedBlockChain(nn.Module):
    """双组门控：Crossbar 4 路输出 = sum1, g1, sum2, g2。"""

    def __init__(self, num_blocks, mat_dim):
        super().__init__()
        self.num_blocks = num_blocks
        self.mat_dim = mat_dim
        self.W_flat = nn.Parameter(torch.randn(num_blocks * mat_dim * mat_dim) * 0.22)

    def forward(self, x: torch.Tensor):
        audio = x * GAIN_PRE
        for b in range(self.num_blocks):
            start = b * WEIGHT_PER_BLOCK
            W = self.W_flat[start:start + WEIGHT_PER_BLOCK].reshape(self.mat_dim, self.mat_dim)
            taps = []
            for k in range(self.mat_dim):
                xk = torch.roll(audio, k, dims=-1)
                if k > 0:
                    xk[..., :k] = 0.0
                taps.append(xk)
            out = W @ torch.stack(taps, dim=-2)      # (4, N)
            s1, g1 = out[..., 0, :], out[..., 1, :]
            s2, g2 = out[..., 2, :], out[..., 3, :]
            s = s1 * torch.tanh(g1) + s2 * torch.tanh(g2)
            y_nl = torch.where(s >= 0, s, ALPHA * s)
            audio = torch.clamp(y_nl + audio, -BLOCK_CLAMP, BLOCK_CLAMP)
        return OUTPUT_SAT * torch.tanh(audio / OUTPUT_SAT)


def stft_mag_loss(pred, target):
    w = torch.hann_window(WIN_LENGTH, device=pred.device, dtype=pred.dtype)
    sp = torch.stft(pred, N_FFT, HOP_LENGTH, WIN_LENGTH, window=w, return_complex=True)
    st = torch.stft(target, N_FFT, HOP_LENGTH, WIN_LENGTH, window=w, return_complex=True)
    return torch.mean((sp.abs() - st.abs()) ** 2) / (torch.mean(st.abs() ** 2) + 1e-8)


def load_audio(path, sr_target=48000):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在：{path}")
    wav, sr = sf.read(path)
    if len(wav.shape) > 1:
        wav = wav[:, 0]
    if sr != sr_target:
        raise RuntimeError(f"要求 {sr_target}Hz，当前文件 {sr}Hz")
    return wav.astype(np.float32)


def metrics(pred, target):
    a, b = pred - pred.mean(), target - target.mean()
    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    nrms = float(np.linalg.norm(pred - target) / np.linalg.norm(target))
    return corr, nrms


BATCH_SAMPLES = 48000 * 4


def batches(dry_t, target_t, batch=BATCH_SAMPLES):
    n = min(dry_t.shape[-1], target_t.shape[-1])
    for s in range(0, n, batch):
        e = min(s + batch, n)
        if e - s < 1920:
            break
        yield dry_t[:, s:e], target_t[:, s:e]


@torch.no_grad()
def evaluate(model, dry_path_eval, target_path_eval, tag, max_samples=48000 * 12):
    d = torch.from_numpy(load_audio(dry_path_eval)[:max_samples]).unsqueeze(0).to(DEVICE)
    t = torch.from_numpy(load_audio(target_path_eval)[:max_samples]).unsqueeze(0).to(DEVICE)
    out = model(d).squeeze().cpu().numpy()
    ref = t.squeeze().cpu().numpy()
    c, n = metrics(out, ref)
    cd, _ = metrics(d.squeeze().cpu().numpy(), ref)
    print(f"   [{tag}] dry corr {cd:+.3f} | pred corr {c:+.3f} | NRMS {n:.4f}", flush=True)
    return c, n


if __name__ == "__main__":
    torch.manual_seed(SEED)
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else EPOCHS
    print(f"数据集 {DATASET} | seed {SEED} | epochs {epochs}", flush=True)

    dry_np = load_audio(dry_path)
    target_np = load_audio(target_path)
    n = min(len(dry_np), len(target_np))
    dry_np, target_np = dry_np[:n], target_np[:n]
    print(f"✅ 训练对 {dry_path}，{n} 采样（{n/SAMPLE_RATE:.1f}s）", flush=True)

    if DATASET == "tsa15":
        dry_val = load_audio(val_dry_path); target_val = load_audio(val_target_path)
        nv = min(len(dry_val), len(target_val))
        dry_val_t = torch.from_numpy(dry_val[:nv]).unsqueeze(0).to(DEVICE)
        target_val_t = torch.from_numpy(target_val[:nv]).unsqueeze(0).to(DEVICE)
    else:
        split = int(n * (1 - VAL_SPLIT))
        dry_val_t = torch.from_numpy(dry_np[split:]).unsqueeze(0).to(DEVICE)
        target_val_t = torch.from_numpy(target_np[split:]).unsqueeze(0).to(DEVICE)
        dry_np, target_np = dry_np[:split], target_np[:split]

    dry_train_t = torch.from_numpy(dry_np).unsqueeze(0).to(DEVICE)
    target_train_t = torch.from_numpy(target_np).unsqueeze(0).to(DEVICE)
    model = GatedBlockChain(NUM_BLOCKS, MAT_DIM).to(DEVICE)
    print(f"训练设备：{DEVICE}（{torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}）", flush=True)
    opt = optim.Adam(model.parameters(), lr=LR)
    lf = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        tot_t = tot_f = 0.0; nb = 0
        for d_batch, t_batch in batches(dry_train_t, target_train_t):
            opt.zero_grad()
            pred = model(d_batch)
            lt = lf(pred, t_batch)
            lfq = stft_mag_loss(pred, t_batch)
            loss = lt + LAMBDA_STFT * lfq
            if not torch.isfinite(loss):
                raise RuntimeError(f"epoch {ep} 非有限损失，降低 LR 或检查数据")
            loss.backward(); opt.step()
            tot_t += lt.item(); tot_f += lfq.item(); nb += 1
        if ep % PRINT_INTERVAL == 0 or ep == epochs - 1:
            print(f"Epoch {ep:4d} | Train T:{tot_t/max(1,nb):.6f} F:{tot_f/max(1,nb):.6f}", flush=True)

    np.save(save_weight_path, model.W_flat.detach().cpu().numpy())
    print(f"✅ 权重保存：{save_weight_path}（{TOTAL_WEIGHTS} 权重）", flush=True)

    model.eval()
    with torch.no_grad():
        preview = model(dry_train_t[:, :BATCH_SAMPLES]).squeeze().cpu().numpy()
    sf.write(infer_out_path, preview, SAMPLE_RATE)
    print(f"✅ 输出预览：{infer_out_path}", flush=True)

    evaluate(model, dry_path, target_path, "train")
    if DATASET == "tsa15":
        evaluate(model, val_dry_path, val_target_path, "val")
        evaluate(model, test_dry_path, test_target_path, "test")
