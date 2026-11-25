"""误差结构分解：单一 NRMS 是否掩盖了关键差异？

数学视角：NRMS = ||e||/||y|| 是 L2 范数比。但音频建模中：
- 误差的时间分布（是否集中在瞬态/静音/高电平）
- 误差的频谱分布（是否集中在人耳敏感的 1-4kHz）
- 线性 vs 非线性分量（系统识别视角）

这些决定"效果"是否真的达到，而单一标量 NRMS 看不出来。
运行：本地（需 numpy/soundfile）
"""
import numpy as np
import soundfile as sf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

x, sr = sf.read(str(ROOT / "data/tsa15_test_input.wav"))
y, _ = sf.read(str(ROOT / "data/tsa15_test_target.wav"))

print("=" * 70)
print("1. 信号的能量集中度（决定 NRMS 是否被少数峰值主导）")
print("=" * 70)
e = y ** 2
n = len(e)
order = np.argsort(e)[::-1]
for frac in [0.01, 0.05, 0.10, 0.20]:
    k = int(n * frac)
    share = e[order[:k]].sum() / e.sum()
    print(f"  能量最高的 {frac*100:>4.0f}% 样本贡献了 {share*100:.1f}% 的总能量")
print("=> 若少数样本主导能量，NRMS 主要反映这些点的精度，")

print()
print("=" * 70)
print("2. 非线性任务的本质（严格线性基准见 linear_check.py）")
print("=" * 70)
print("  曾用 numpy 复数直除 H=Y/X 得到 NRMS=0.0000 —— 那是数值假象：")
print("  在 N 点上用 N 个自由度拟合，任何信号都能被'完美'表示。")
print("  严格做法（train 拟合 / test 评价 / 带正则的 Wiener-Hopf）见")
print("  src/tools/linear_check.py，结果：最优线性 FIR 仅到 0.75。")
print("  => 该任务是强非线性系统辨识，非线性建模确有必要。")

print()
print("=" * 70)
print("3. 频带能量分布（决定模拟电路的带宽要求）")
print("=" * 70)
Yf = np.abs(np.fft.rfft(y))
freqs = np.fft.rfftfreq(len(y), 1/sr)
total = (Yf ** 2).sum()
for lo, hi in [(0, 100), (100, 500), (500, 2000), (2000, 5000), (5000, 10000), (10000, 24000)]:
    m = (freqs >= lo) & (freqs < hi)
    share = (Yf[m] ** 2).sum() / total
    print(f"  {lo:>5}-{hi:<5} Hz: {share*100:5.2f}%")
