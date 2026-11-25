"""严格检验：线性 FIR 能拟合到什么程度？（正确划分训练/测试，避免数据泄漏）"""
import numpy as np
import soundfile as sf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
x, sr = sf.read(str(ROOT / 'data/tsa15_train_input.wav'))
y, _ = sf.read(str(ROOT / 'data/tsa15_train_target.wav'))
xt, _ = sf.read(str(ROOT / 'data/tsa15_test_input.wav'))
yt, _ = sf.read(str(ROOT / 'data/tsa15_test_target.wav'))

print(f'train {len(x)/sr:.1f}s  test {len(xt)/sr:.1f}s')
print(f'y rms train={np.sqrt((y**2).mean()):.4f} test={np.sqrt((yt**2).mean()):.4f}')

# 用训练段估计 FIR（ Wiener-Hopf，带正则），在测试段评价
for L in [64, 256, 1024, 4096]:
    # 构造 Toeplitz 相关矩阵（用 FFT 加速的自相关）
    n = len(x)
    # 自相关 rxx[k] = sum x[t]x[t-k]
    Xf = np.fft.rfft(x, n=2*n)
    rxx_full = np.fft.irfft(Xf * np.conj(Xf), n=2*n)[:L]
    # 互相关 rxy[k] = sum y[t]x[t-k]
    Xf2 = np.fft.rfft(x, n=2*n)
    Yf2 = np.fft.rfft(y, n=2*n)
    rxy_full = np.fft.irfft(Yf2 * np.conj(Xf2), n=2*n)[:L]
    # Toeplitz
    R = np.empty((L, L))
    for i in range(L):
        for j in range(L):
            R[i, j] = rxx_full[abs(i - j)]
    # 正则化（ridge）
    lam = 1e-4 * np.trace(R) / L
    h = np.linalg.solve(R + lam * np.eye(L), rxy_full)
    # 测试段卷积预测
    pred = np.convolve(xt, h)[:len(xt)]
    # 对齐：找最佳延迟
    best = 1e9
    bestlag = 0
    for lag in range(-20, 21):
        if lag >= 0:
            p = pred[:len(pred)-lag] if lag else pred
            t = yt[lag:]
        else:
            p = pred[-lag:]
            t = yt[:len(yt)+lag]
        nn = min(len(p), len(t))
        r = np.linalg.norm(p[:nn] - t[:nn]) / np.linalg.norm(t[:nn])
        if r < best:
            best, bestlag = r, lag
    print(f'FIR L={L:>5}: test NRMS={best:.4f} (lag={bestlag})')

print()
print('参考：WaveNet standard 0.0653, lite 0.0785')
print('若线性 FIR 就能到 0.1 以下，则大网络的必要性需重新论证')
