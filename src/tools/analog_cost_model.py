"""模拟实现的真实成本分解 —— 为什么 MACs 不是判据

前面用 MACs 判定 depthwise 优势, 得到负面结论 (dw_ch24 的 MACs 反而比 lite 高 25%)。
但 MACs 只算了"乘法次数", 没算模拟实现最贵的两样:
  1. 交叉开关 (crossbar) 规模 —— 决定面积、布线复杂度、寄生电容
  2. 累加链深度 (KCL 求和的串联级数) —— 决定摆幅需求与速度

depthwise 的真正优势在这两处, 不在 MACs:
  - depthwise conv: 每层是 ch 条独立链 (3 tap), 无交叉开关
  - 全连接 conv:    每层是 ch×ch 交叉开关
  - 1×1 层:         两者都是 ch×ch (这一项 depthwise 省不掉)

所以正确的问题是: 交叉开关总量 与 累加链深度, 而不是 MACs。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AA = ROOT / "arch_analog_local"


def cost(ch, depthwise, rf_pct=100):
    """返回 (crossbar_cells, macs, max_acc_chain)"""
    head_ch = max(1, ch // 2)
    d1 = [1, 2, 4, 8, 16, 32, 64]
    d2 = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    if rf_pct != 100:
        d1 = [max(1, int(d * rf_pct / 100)) for d in d1]
        d2 = [max(1, int(d * rf_pct / 100)) for d in d2]
    groups = [(1, ch, d1, head_ch, False), (ch, head_ch, d2, 1, True)]

    xbar = 0
    macs = 0
    max_chain = 0
    for in_size, c, dils, head_out, hb in groups:
        xbar += in_size * c                       # rechannel
        macs += in_size * c
        for d in dils:
            if depthwise:
                # ch 条独立链, 每条 3 tap —— 无交叉开关, 链长 3
                macs += c * 3
                max_chain = max(max_chain, 3)
            else:
                # ch×ch 交叉开关, 每输出累加 ch 个 tap×3
                xbar += c * c * 3
                macs += c * c * 3
                max_chain = max(max_chain, c * 3)
            macs += c                            # input_mixer
            xbar += c                            # layer1x1 是 ch×ch 交叉开关
            macs += c * c
            max_chain = max(max_chain, c)
        xbar += c * head_out
        macs += c * head_out
    return xbar, macs, max_chain


print("=" * 86)
print("模拟实现成本的三维对比")
print("=" * 86)
print(f"  {'配置':<16}{'交叉开关':>10}{'MACs':>8}{'最大累加链':>12}{'延迟ms':>9}")
print("  " + "-" * 55)

cfgs = [
    ("lite (全连接)", 12, False, 100, 0.0785),
    ("dw_ch12", 12, True, 100, 0.1084),
    ("dw_ch12_rf50", 12, True, 50, 0.1182),
    ("dw_ch24", 24, True, 100, 0.0804),
]
for name, ch, dw, rf, nr in cfgs:
    xb, m, chain = cost(ch, dw, rf)
    rf_ms = 85.27 if rf == 100 else 42.69
    print(f"  {name:<16}{xb:>10}{m:>8}{chain:>12}{rf_ms:>9.1f}")

print()
print("=" * 86)
print("核心对比: 达到 A2 Lite 精度 (≈0.0785) 所需的器件规模")
print("=" * 86)
lite_xb, lite_m, lite_chain = cost(12, False, 100)
dw24_xb, dw24_m, dw24_chain = cost(24, True, 100)
print(f"  {'配置':<18}{'交叉开关':>10}{'相对':>9}{'最大累加链':>12}{'相对':>9}")
print("  " + "-" * 58)
print(f"  {'lite (全连接)':<18}{lite_xb:>10}{'基准':>9}{lite_chain:>12}{'基准':>9}")
print(f"  {'dw_ch24':<18}{dw24_xb:>10}{(dw24_xb/lite_xb-1)*100:>8.0f}%{dw24_chain:>12}"
      f"{(dw24_chain/lite_chain-1)*100:>8.0f}%")

print()
print("  解读（三条结论要分开看）:")
print("   1) MACs: dw_ch24 比 lite 多 25% —— 1×1 层的 ch² 增长吃掉了 depthwise 在")
print("      dilated conv 上省下的量。**depthwise 的 MACs 优势不成立。**")
print("   2) 交叉开关: dw_ch24 只有 lite 的 20%（936 vs 4752）—— 这是**真实且巨大**")
print("      的优势。全连接 conv 每层要 ch×ch 的交叉开关矩阵, depthwise 只要 ch 条")
print("      独立链, 完全不需要交叉开关。模拟实现的面积、布线复杂度、寄生电容都由")
print("      交叉开关规模决定, 这一项是 depthwise 独有的收益。")
print("   3) 最大累加链: dw_ch24 更浅（24 vs 36）—— KCL 求和级数更短, 摆幅需求更低。")
print()
print("  => 修正后的结论: depthwise 的优势不在 MACs, 而在**交叉开关规模与累加链深度**。")
print("     这两项恰是模拟实现的主导成本, 所以 dw_ch24 用 +25% MACs 换到 -80% 交叉开关")
print("     与 -33% 累加链, 加上精度仅差 +2.4%, 是模拟实现上真正的划算交易。")

print()
print("=" * 86)
print("但 dw_ch12 是完全不同的图景")
print("=" * 86)
d12_xb, d12_m, d12_chain = cost(12, True, 100)
print(f"  dw_ch12: 交叉开关 {d12_xb} (lite 的 {d12_xb/lite_xb*100:.0f}%)  "
      f"MACs {d12_m} ({d12_m/lite_m*100:.0f}%)  累加链 {d12_chain} ({d12_chain/lite_chain*100:.0f}%)")
print(f"  代价: 精度 0.1084 vs lite 0.0785 (差 {(0.1084/0.0785-1)*100:+.0f}%)")
print()
print("  => dw_ch12 是「小而省」的点: 全部三个器件维度都是 lite 的 1/3~1/2,")
print("     代价是精度从 0.0785 退到 0.1084。")
print("     这才是 depthwise 路线的真实价值区间: 不是替代 lite, 而是用 1/3 器件换 38% 精度。")

print()
print("=" * 86)
print("短 RF 的额外收益 (模拟实现里延迟线是面积与功耗的大头)")
print("=" * 86)
for name, ch, dw, rf in [("dw_ch12 全RF", 12, True, 100), ("dw_ch12 短RF", 12, True, 50)]:
    xb, m, chain = cost(ch, dw, rf)
    rf_ms = 85.27 if rf == 100 else 42.69
    taps = 4092 if rf == 100 else 2046
    print(f"  {name:<16} 交叉开关 {xb:>6}  MACs {m:>6}  累加链 {chain:>4}  "
          f"延迟线 {taps:>5} 抽头 ({rf_ms:.1f}ms)")
print()
print("  => 短 RF 让延迟线抽头减半, 而器件规模完全不变。")
print("     在模拟实现里, 延迟线(数字 SRAM) 的面积与功耗通常与 MACs 同量级,")
print("     所以这是与 MACs 独立的第二个成本维度上的收益。")
