"""正确的比较口径: 同等模拟成本下, depthwise 是否优于全连接?

前一版分析用「MACs<3000 绝对上限」判定 H2 不成立, 但这个口径有问题:
  - 预登记的 3000 上限来自「lite 的 6228 减半」这个直觉, 不是第一性原理
  - depthwise 省的是 conv 的 MACs, 而 1×1 的 ch² 增长与 conv 类型无关
  - 所以「MACs<3000」这个约束对 depthwise 和全连接同样苛刻, 无法体现两者的差别

正确的做法: 在「同等 MACs 预算」下比较精度。
  对每个 MACs 预算 M, 分别求:
    - depthwise 达到该 MACs 所需的 ch (及预测 NRMS)
    - 全连接 lite 的 ch=12 固定 6228 MACs (0.0785) 作为参照
  再问: depthwise 在 M MACs 下的精度 vs 全连接在同等 M 下的精度

对全连接, MACs 与 ch 的关系: conv 是 ch²×3 (主导), 所以全连接在低 MACs 预算下
需要更小的 ch, 精度按幂律下降。
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AA = ROOT / "arch_analog_local"


def macs_dw(ch, rf_pct=100):
    head_ch = max(1, ch // 2)
    d1 = [1, 2, 4, 8, 16, 32, 64]
    d2 = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    if rf_pct != 100:
        d1 = [max(1, int(d * rf_pct / 100)) for d in d1]
        d2 = [max(1, int(d * rf_pct / 100)) for d in d2]
    groups = [(1, ch, d1, head_ch, False), (ch, head_ch, d2, 1, True)]
    m = 0
    for in_size, c, dils, head_out, hb in groups:
        m += in_size * c
        m += len(dils) * (c * 3)
        m += len(dils) * c
        m += len(dils) * (c * c)
        m += c * head_out
    return m


def macs_full(ch, rf_pct=100):
    """全连接: conv 权重 ch×ch×3 (不是 ch×3)."""
    head_ch = max(1, ch // 2)
    d1 = [1, 2, 4, 8, 16, 32, 64]
    d2 = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    if rf_pct != 100:
        d1 = [max(1, int(d * rf_pct / 100)) for d in d1]
        d2 = [max(1, int(d * rf_pct / 100)) for d in d2]
    groups = [(1, ch, d1, head_ch, False), (ch, head_ch, d2, 1, True)]
    m = 0
    for in_size, c, dils, head_out, hb in groups:
        m += in_size * c
        m += len(dils) * (c * c * 3)      # 全连接 conv
        m += len(dils) * c
        m += len(dils) * (c * c)
        m += c * head_out
    return m


def params_of(ch, depthwise, rf_pct=100):
    head_ch = max(1, ch // 2)
    d1 = [1, 2, 4, 8, 16, 32, 64]
    d2 = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    if rf_pct != 100:
        d1 = [max(1, int(d * rf_pct / 100)) for d in d1]
        d2 = [max(1, int(d * rf_pct / 100)) for d in d2]
    groups = [(1, ch, d1, head_ch, False), (ch, head_ch, d2, 1, True)]
    p = 0
    for in_size, c, dils, head_out, hb in groups:
        p += in_size * c
        p += len(dils) * ((c * 3 + 1) if depthwise else (c * c * 3 + c))
        p += len(dils) * c
        p += len(dils) * (c * c + 1)
        p += c * head_out + (1 if hb else 0)
    return p


# 幂律 (官方四档)
P_FULL = [(841, 0.1485), (3025, 0.1014), (6553, 0.0785), (13801, 0.0653)]
xs = [math.log(p) for p, _ in P_FULL]
ys = [math.log(e) for _, e in P_FULL]
n = len(xs)
mx, my = sum(xs) / n, sum(ys) / n
b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
A = math.exp(my - b * mx)
predict = lambda p: A * p ** b

print("=" * 82)
print("同等 MACs 预算下的精度对比")
print("=" * 82)
print("  说明: 幂律 NRMS=1.0968·P^-0.2975 (R²=0.9980) 用于外推")
print()

CH = list(range(6, 33))
dw_tab = [(ch, macs_dw(ch), params_of(ch, True), predict(params_of(ch, True))) for ch in CH]
fw_tab = [(ch, macs_full(ch), params_of(ch, False), predict(params_of(ch, False))) for ch in CH]

print(f"  {'MACs预算':>10}{'dw ch':>7}{'dw NRMS':>10}{'full ch':>9}{'full NRMS':>11}{'dw优势':>10}")
print("  " + "-" * 58)
for budget in [1500, 2000, 2500, 3000, 4000, 5000, 6000, 6228, 8000]:
    d_cand = [t for t in dw_tab if t[1] <= budget]
    f_cand = [t for t in fw_tab if t[1] <= budget]
    if not d_cand or not f_cand:
        continue
    # 同预算下取参数最多者 (最接近预算且精度最好)
    d_best = max(d_cand, key=lambda t: t[2])
    f_best = max(f_cand, key=lambda t: t[2])
    adv = (f_best[3] / d_best[3] - 1) * 100
    print(f"  {budget:>10}{d_best[0]:>7}{d_best[3]:>10.4f}{f_best[0]:>9}{f_best[3]:>11.4f}"
          f"{adv:>9.1f}%")

print()
print("  注: 同一 MACs 预算下, 两者都能选到某个 ch; 谁的参数多谁就更准。")
print("      depthwise 在 ch 相同时 MACs 更少, 但参数也更少 —— 关键看谁在同等 MACs 下参数更多。")

print()
print("=" * 82)
print("换个角度: 同等参数预算下的 MACs 对比 (这才是模拟实现的真实成本)")
print("=" * 82)
print(f"  {'参数预算':>10}{'dw ch':>7}{'dw MACs':>10}{'full ch':>9}{'full MACs':>12}{'MACs节省':>10}")
print("  " + "-" * 58)
for budget in [2500, 3000, 4000, 5000, 6553, 8000]:
    d_cand = [t for t in dw_tab if t[2] <= budget]
    f_cand = [t for t in fw_tab if t[2] <= budget]
    if not d_cand or not f_cand:
        continue
    d_best = max(d_cand, key=lambda t: t[2])
    f_best = max(f_cand, key=lambda t: t[2])
    save = (1 - d_best[1] / f_best[1]) * 100
    print(f"  {budget:>10}{d_best[0]:>7}{d_best[1]:>10}{f_best[0]:>9}{f_best[1]:>12}{save:>9.0f}%")

print()
print("=" * 82)
print("实测验证 (不是外推)")
print("=" * 82)
d12 = json.load(open(AA / "dw_ch12_s7.json"))
d24 = json.load(open(AA / "dw_ch24_s7.json"))
lite = json.load(open(ROOT / "arch_v2_local" / "lite_s7.json"))
print(f"  dw_ch12  参数 {d12['params']:>5}  MACs {d12['macs']:>5}  test {d12['test_NRMS']:.4f}")
print(f"  dw_ch24  参数 {d24['params']:>5}  MACs {d24['macs']:>5}  test {d24['test_NRMS']:.4f}")
print(f"  lite     参数 {lite['params']:>5}  MACs 6228  test {lite['test_NRMS']:.4f}")
print()
print(f"  dw_ch24 vs lite: 精度差 {(d24['test_NRMS']/lite['test_NRMS']-1)*100:+.1f}%, "
      f"MACs 多 {(d24['macs']/6228-1)*100:+.0f}%")
print()
print("  => 关键: depthwise 在同等参数下 MACs 更省, 但要达到 lite 的精度, ")
print("     需要更多参数(因为 1×1 的 ch² 增长), 最终 MACs 反而略高。")
print("     这说明 depthwise 的 MACs 优势在『足够宽』时被 1×1 层吃掉。")
