"""判定 depthwise 路线的参数效率 —— 与全局幂律对照

问题: dw_ch12 (2611 参数, depthwise) 的 test NRMS 落在幂律曲线的哪里?
若正好落在曲线上, 说明 depthwise 没有浪费参数, 是"参数效率达标"的实现.
若明显落在曲线下方, 说明 depthwise 结构劣于同参数全连接 WaveNet.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A2 = ROOT / "arch_v2_local"
AA = ROOT / "arch_analog_local"

# ---- 1. 拟合全局幂律 (官方四档, 我方协议, 全 RF) ----
P_FULL = [
    ("nano", 841, 0.1485),
    ("feather", 3025, 0.1014),
    ("lite", 6553, 0.0785),
    ("standard", 13801, 0.0653),
]
import math

xs = [math.log(p) for _, p, _ in P_FULL]
ys = [math.log(e) for _, _, e in P_FULL]
n = len(xs)
mx = sum(xs) / n
my = sum(ys) / n
sxx = sum((x - mx) ** 2 for x in xs)
sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
b = sxy / sxx
a = my - b * mx
A = math.exp(a)
r2 = 1 - sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / sum((y - my) ** 2 for y in ys)

print("=" * 76)
print(f"全局参数效率幂律: NRMS = {A:.4f} * P^({b:.4f})   R2 = {r2:.4f}")
print("=" * 76)


def predict(p):
    return A * p ** b


print("\n【幂律对四档的回代检验】")
for name, p, e in P_FULL:
    pr = predict(p)
    print(f"  {name:<10} P={p:>6}  实测 {e:.4f}  预测 {pr:.4f}  残差 {(e/pr-1)*100:+.1f}%")

# ---- 2. dw_ch12 对照幂律 ----
print("\n" + "=" * 76)
print("【关键检验】dw_ch12 落在幂律曲线的哪里?")
print("=" * 76)
dws = []
for f in sorted(AA.glob("dw_ch12_s*.json")):
    d = json.load(open(f))
    dws.append(d)
    p, e = d["params"], d["test_NRMS"]
    pr = predict(p)
    print(f"  {d['variant']} seed{d['seed']:<3} P={p:>5} MACs={d['macs']:>5}  实测 {e:.4f}  "
          f"幂律预测 {pr:.4f}  残差 {(e/pr-1)*100:+.1f}%")

if dws:
    mean_e = sum(d["test_NRMS"] for d in dws) / len(dws)
    mean_p = predict(dws[0]["params"])
    print(f"\n  depthwise 平均 {mean_e:.4f} vs 幂律预期 {mean_p:.4f}  "
          f"→ 偏离 {(mean_e/mean_p-1)*100:+.1f}%")
    print("  解读: 偏离≈0 表示 depthwise 未浪费参数, 是参数效率达标的实现")
    print("        明显为正表示同参数下 depthwise 结构劣于全连接 WaveNet")

# ---- 3. 与官方 lite 直接对比 (跨框架, 仅作参考) ----
print("\n" + "=" * 76)
print("【跨框架参考】dw_ch12 vs 官方 lite (框架不同, 结论待 full_ch12 校准)")
print("=" * 76)
lite = json.load(open(A2 / "lite_s7.json"))
for d in dws:
    print(f"  vs lite: {d['test_NRMS']:.4f} / {lite['test_NRMS']:.4f} = "
          f"{(d['test_NRMS']/lite['test_NRMS']-1)*100:+.1f}%  "
          f"(参数 {d['params']}/{lite['params']} = {d['params']/lite['params']*100:.0f}%)")

print("\n" + "=" * 76)
print("【预先登记的判定规则回顾】")
print("=" * 76)
print("  dw 差 <15%  -> depthwise 几乎免费, 路线成立")
print("  dw 差 15-40% -> 中间地带, 需靠宽度/其它手段补偿")
print("  dw 差 >40%  -> depthwise 路线证伪")
print("  (以上阈值以 full_ch12 同框架复现 0.0785 为基准校准)")
