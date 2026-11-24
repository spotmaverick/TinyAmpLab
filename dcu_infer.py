"""DCU 上的正式推理：加载 dual-gate 权重，对指定数据集全量前向，写 wav + 指标。

用法: python dcu_infer.py <dafx|b1m3|tsa15> <weight.npy> <out.wav>
tsa15 用独立 test 段；dafx/b1m3 用各自完整配对。
"""
import sys
import numpy as np
import soundfile as sf
import torch

NUM_BLOCKS, MAT_DIM = 40, 4
ALPHA, GAIN_PRE, BLOCK_CLAMP, OUTPUT_SAT = 0.15, 2.2, 4.0, 1.0
SR = 48000
DATASETS = {
    "dafx":  ("data/nam_input.wav",  "data/nam_output.wav"),
    "b1m3":  ("data/b1m3_input.wav", "data/b1m3_target.wav"),
    "tsa15": ("data/tsa15_test_input.wav", "data/tsa15_test_target.wav"),
}


def forward(x, W):
    """x: (N,) float32 已在目标设备；W: (40,4,4) 权重。与 train_nam.py 完全一致的前向。"""
    audio = x * GAIN_PRE
    for b in range(NUM_BLOCKS):
        Wb = W[b]
        taps = []
        for k in range(MAT_DIM):
            xk = torch.roll(audio, k, dims=-1)
            if k > 0:
                xk[..., :k] = 0.0
            taps.append(xk)
        out = Wb @ torch.stack(taps, dim=-2)
        s1, g1 = out[..., 0, :], out[..., 1, :]
        s2, g2 = out[..., 2, :], out[..., 3, :]
        s = s1 * torch.tanh(g1) + s2 * torch.tanh(g2)
        y_nl = torch.where(s >= 0, s, ALPHA * s)
        audio = torch.clamp(y_nl + audio, -BLOCK_CLAMP, BLOCK_CLAMP)
    return OUTPUT_SAT * torch.tanh(audio / OUTPUT_SAT)


def metrics(pred, target):
    a, b = pred - pred.mean(), target - target.mean()
    corr = float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
    nrms = float((pred - target).norm() / target.norm())
    return corr, nrms


if __name__ == "__main__":
    tag, weight_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    W = torch.from_numpy(np.load(weight_path).astype(np.float32).reshape(NUM_BLOCKS, MAT_DIM, MAT_DIM)).to(dev)
    dx, ty_path = DATASETS[tag]
    x_np, sr = sf.read(dx)
    t_np, _ = sf.read(ty_path)
    if x_np.ndim > 1:
        x_np = x_np[:, 0]
    if t_np.ndim > 1:
        t_np = t_np[:, 0]
    n = min(len(x_np), len(t_np))
    x = torch.from_numpy(x_np[:n].astype(np.float32)).to(dev)
    t = torch.from_numpy(t_np[:n].astype(np.float32)).to(dev)

    with torch.no_grad():
        y = forward(x, W)
    y_cpu = y.cpu().numpy()
    sf.write(out_path, y_cpu, SR)

    c, nr = metrics(y.double(), t.double())
    cd, _ = metrics(x.double(), t.double())
    print(f"[{tag}] device={dev} | {n} samples ({n/SR:.1f}s) | weights={weight_path}", flush=True)
    print(f"[{tag}] dry corr {cd:+.4f} | pred corr {c:+.4f} | NRMS {nr:.4f} | finite={bool(np.all(np.isfinite(y_cpu)))}", flush=True)
    print(f"[{tag}] output saved: {out_path}", flush=True)
