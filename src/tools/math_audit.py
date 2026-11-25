"""数学审计：严格检验现有实验结论的统计有效性与比较公平性

以数学/科学态度重新审视，不做任何预设。检查：
1. 每个"结论"背后的对照是否真的只变了一个变量
2. 效应量 vs 噪声（种子方差）的比值
3. 参数效率曲线的幂律拟合与外推
4. "模拟可实现"的约束是否被量化

运行：本地（纯标准库 + 已有 json）
"""
import json, glob, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def load_arch():
    out = {}
    for f in glob.glob(str(ROOT / "arch_v2_local" / "*.json")):
        d = json.load(open(f))
        out[d["variant"] + "_s" + str(d["seed"])] = d
    return out

A = load_arch()

print("=" * 78)
print("审计 1：参数效率曲线的幂律拟合（我方可比协议，全 RF，seed 7）")
print("=" * 78)
# 数据点 (参数, NRMS)
pts = [
    (841, 0.1485, "nano"),
    (3025, 0.1014, "feather"),
    (6553, 0.0785, "lite"),
    (13801, 0.0653, "standard(d1)"),
]
# 拟合 NRMS = a * P^(-b)  => log(NRMS) = log(a) - b log(P)
xs = [math.log(p) for p, n, _ in pts]
ys = [math.log(n) for p, n, _ in pts]
mx = sum(xs) / len(xs)
my = sum(ys) / len(ys)
b = -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
loga = my + b * mx
a = math.exp(loga)
print(f"拟合 NRMS = {a:.4f} * P^(-{b:.4f})")
for p, n, name in pts:
    pred = a * p ** (-b)
    print(f"  {name:<16} P={p:>6}  实测 {n:.4f}  拟合 {pred:.4f}  残差 {(n/pred-1)*100:+.1f}%")
# R^2
ss_res = sum((n - a * p ** (-b)) ** 2 for p, n, _ in pts)
ss_tot = sum((n - my) ** 2 for p, n, _ in pts)  # 注意 my 是对数均值，这里用原始均值
ym = sum(n for _, n, _ in pts) / len(pts)
ss_tot = sum((n - ym) ** 2 for _, n, _ in pts)
print(f"R² = {1 - ss_res/ss_tot:.4f}")
print(f"=> 幂律指数 b≈{b:.3f}。若 b<0.5，说明参数收益递减很快；")
print(f"   要 NRMS 减半需参数增 {2**(1/b):.1f} 倍，这是架构选择的硬约束。")

print()
print("=" * 78)
print("审计 2：‘架构效率是主瓶颈’的对照是否公平？（d1 的核心论断）")
print("=" * 78)
print("原文论断：官方 standard(13801) 0.0653 vs LGW 冠军(2149) 0.1736，")
print("          同协议差 2.7 倍，只能归因于架构。")
print()
print("问题：这是 6.4 倍参数量的对比，不是等参对比！")
p_std, n_std = 13801, 0.0653
p_lgw, n_lgw = 2149, 0.1736
pred_lgw_if_wavenet = a * p_lgw ** (-b)
print(f"  按拟合幂律，2149 参数的 WaveNet 预期 NRMS ≈ {pred_lgw_if_wavenet:.4f}")
print(f"  实测 LGW(2149) = {n_lgw:.4f}")
print(f"  等参差距 = {n_lgw/pred_lgw_if_wavenet:.2f} 倍（不是 2.7 倍）")
print("=> 原论断的‘2.7倍’混入了 6.4 倍参数优势。等参后架构差距约 "
      f"{n_lgw/pred_lgw_if_wavenet:.1f} 倍，仍显著但结论强度需下调。")
print()
print("更严重：两者 RF 也不同 —— LGW 11.3ms vs WaveNet 85.3ms（7.5 倍）。")
print("  WaveNet 用了 7.5 倍长的历史窗口。这不是纯架构对比，")
print("  而是‘架构 + 感受野’的联合对比。要分离需等 RF 实验（已做 d2：")
print("  LGW 拉到 85ms 反而 0.1893 更差，说明 RF 不是 WaveNet 优势来源）。")

print()
print("=" * 78)
print("审计 3：种子方差 vs 效应量（决定哪些结论可信）")
print("=" * 78)
pairs = [("nano_rf50", 0.1412, 0.1793),
         ("feather_rf50", 0.1005, 0.1065),
         ("lite_rf50", 0.0885, 0.0840)]
for name, s7, s11 in pairs:
    spread = abs(s7 - s11) / ((s7 + s11) / 2) * 100
    print(f"  {name:<16} s7={s7:.4f} s11={s11:.4f}  种子差 {spread:.1f}%")
print("=> nano 种子差 27%，任何 <27% 的‘改进’都不可信；")
print("   lite 种子差 5%，12.8% 的退化大于噪声，方向可信。")

print()
print("=" * 78)
print("审计 4：目标定义审视 —— ‘达到 A2 Lite’到底指什么？")
print("=" * 78)
print("官方 lite = 6553 参数，85.3ms RF，官方协议 NRMS 0.0852")
print("我方 lite = 6553 参数，85.3ms RF，我方协议 NRMS 0.0785")
print()
print("关键混淆：")
print("  (a) ‘达到 A2 效果’若指 NRMS≤0.0852 —— 我方 lite 已达到，")
print("      但这是 300ep 余弦 vs 官方 100ep 指数，多花了 3 倍训练。")
print("  (b) 若指‘同等或更少参数’—— lite 就是 A2 lite 本身，无超越。")
print("  (c) 若指‘模拟可实现’—— 物理仿真显示 lite 需 Vsat=16（10× 摆幅），")
print("      器件 3×、延迟线 12×。按此定义，A2 lite 本身就不是好的模拟拓扑。")
print("=> 目标 (c) 与 (a)(b) 冲突：数字精度最优的架构，恰是模拟最难实现的。")
print("   必须明确：我们要的是‘模拟约束下的最优’，而非‘复刻 A2 的数字精度’。")

print()
print("=" * 78)
print("审计 5：模拟可实现性的量化 —— 谁真正适合模拟？")
print("=" * 78)
data = [
    ("LGW-TCN 冠军", 2149, 511, 1.5, 0.1831, 0.3273),  # RF~541? 用 511 近似
    ("WaveNet lite", 6553, 4093, 16.0, 0.0784, 0.1114),
]
print(f"{'架构':<16}{'参数':>7}{'RF':>7}{'Vsat':>7}{'fp32':>8}{'物理':>8}{'物理/参数效率':>16}")
for name, p, rf, v, f, ph in data:
    eff = ph * p / 1000  # 越小越好：物理误差 × 参数（千）
    print(f"{name:<16}{p:>7}{rf:>7}{v:>7.1f}{f:>8.4f}{ph:>8.4f}{eff:>16.3f}")
print("=> 若目标是‘模拟约束下最优’，评价指标应是物理误差，不是数字 NRMS。")
print("   lite 物理 0.1114 优于 LGW 0.3273，但代价是 10× 摆幅 + 12× 延迟线。")
print("   摆幅和延迟线是模拟实现的硬约束，不能只比 NRMS。")
