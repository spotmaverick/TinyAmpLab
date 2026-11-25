"""参数流分解与「宽 depthwise + 窄 1×1」设计空间

动机
----
二维标度律揭示了一个此前被忽略的事实：

    dw_ch12 (2611 参数) 偏离幂律 +0.5%
    dw_ch24 (8461 参数) 偏离幂律 +6.9%

depthwise 的参数效率**随通道数递减**。若这是真的，"加宽补参"这条路就是死胡同，
因为加到 8461 参数它已经比同参数的全连接差 7%。

本脚本回答：为什么会递减？钱花在哪了？

方法
----
按 arch_analog_friendly.py 的结构精确复算各算子的参数份额，
同时给出模拟实现的三维成本（交叉开关 / MACs / 累加链），
然后构造候选改进方案并比较。

用法: python src/tools/param_flow.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

D1 = [1, 2, 4, 8, 16, 32, 64]
D2 = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def breakdown(ch, depthwise, rf_pct=100, mix_rank=None, mix_groups=1):
    """精确复算参数与成本。

    口径严格对齐 arch_analog_friendly.py 的 AFGroup：
      rechannel   Conv1d(in_size, ch, 1, bias=False)   -> in_size*ch
      layer.conv  Conv1d(ch, ch, 3, groups, bias=True) -> 全连接 ch*ch*3 + ch；depthwise ch*3 + ch
      input_mixer Conv1d(1, ch, 1, bias=False)         -> ch
      layer1x1    Conv1d(ch, ch, 1, bias=True)         -> ch*ch + ch
      head        Conv1d(ch, head_out, 1, bias=head_bias) -> ch*head_out (+head_out)

    mix_rank: 若给定，layer1x1 改为低秩 ch->r->ch
    mix_groups: layer1x1 改分组卷积（交叉开关按组数下降，秩不降）
    """
    d1 = [max(1, round(d * rf_pct / 100)) for d in D1]
    d2 = [max(1, round(d * rf_pct / 100)) for d in D2]
    head_ch = max(1, ch // 2)
    # (in_size, 通道, 膨胀序列, head_out, head_bias)
    groups = [(1, ch, d1, head_ch, False), (ch, head_ch, d2, 1, True)]

    tot = dict(rechannel=0, dwconv=0, fullconv=0, mixer=0, layer1x1=0,
               head=0, bias=0)
    xbar = macs = chain = 0
    layers_total = 0

    for in_size, c, dils, head_out, hb in groups:
        tot["rechannel"] += in_size * c
        xbar += in_size * c
        macs += in_size * c
        for _ in dils:
            layers_total += 1
            if depthwise:
                tot["dwconv"] += c * 3
                tot["bias"] += c
                macs += c * 3 + c
                chain = max(chain, 3)
            else:
                tot["fullconv"] += c * c * 3
                tot["bias"] += c
                xbar += c * c * 3
                macs += c * c * 3 + c
                chain = max(chain, c * 3)
            tot["mixer"] += c
            macs += c
            # layer1x1（含 bias，偏置计入 bias 项）
            tot["bias"] += c
            macs += c
            if mix_rank:
                w = c * mix_rank + mix_rank * c
            else:
                per = c // mix_groups
                w = mix_groups * per * per
            tot["layer1x1"] += w
            xbar += w
            macs += w
            chain = max(chain, c // (mix_groups if mix_groups > 1 else 1))
        tot["head"] += c * head_out
        xbar += c * head_out
        macs += c * head_out
        if hb:
            tot["bias"] += head_out
            macs += head_out

    tot["total"] = sum(v for k, v in tot.items())
    # 与原脚本口径统一：MACs = 权重乘法次数 = 参数量 − 偏置数（偏置加法不记 MAC）
    macs = tot["total"] - tot["bias"]
    return tot, xbar, macs, chain, layers_total


def show(name, br, ref=None):
    tot, xbar, macs, chain, _layers = br
    p = tot["total"]
    conv = tot["dwconv"] + tot["fullconv"]
    mix = tot["layer1x1"] + tot["mixer"]
    print(f"\n  {name}")
    print(f"    参数 {p:>6}    交叉开关 {xbar:>6}    MACs {macs:>6}    累加链 {chain:>3}")
    print(f"    ├ 卷积部分(depthwise/full) {conv:>6}  ({conv/p*100:>5.1f}%)")
    print(f"    ├ 1×1 混合(mixer+layer1x1) {mix:>6}  ({mix/p*100:>5.1f}%)")
    print(f"    └ 其余(rechannel/head/bias) {p-conv-mix:>5}  ({(p-conv-mix)/p*100:>5.1f}%)")
    if ref:
        print(f"    相对 lite: 参数 {p/ref['p']:.2f}×  交叉开关 {xbar/ref['xbar']:.2f}×  "
              f"MACs {macs/ref['macs']:.2f}×")
    return dict(p=p, xbar=xbar, macs=macs, chain=chain,
                conv_frac=conv / p, mix_frac=mix / p)


def main():
    print("=" * 92)
    print("参数流分解：depthwise 的参数效率为何随宽度递减")
    print("=" * 92)

    lite = show("lite (全连接 ch=12)  ← 基准", breakdown(12, False))
    d12 = show("dw_ch12 (depthwise ch=12)", breakdown(12, True), lite)
    d24 = show("dw_ch24 (depthwise ch=24)", breakdown(24, True), lite)

    ref = dict(p=lite["p"], xbar=lite["xbar"], macs=lite["macs"])

    print()
    print("=" * 92)
    print("诊断：ch 从 12 加到 24，参数花在哪")
    print("=" * 92)
    t12, x12, m12, c12, _ = breakdown(12, True)
    t24, x24, m24, c24, _ = breakdown(24, True)
    print(f"  {'算子':<16}{'ch12':>8}{'ch24':>8}{'增量':>8}{'占增量':>9}")
    print("  " + "-" * 50)
    dsum = t24["total"] - t12["total"]
    for k in ("rechannel", "dwconv", "mixer", "layer1x1", "head", "bias"):
        d = t24[k] - t12[k]
        print(f"  {k:<16}{t12[k]:>8}{t24[k]:>8}{d:>+8}{d/dsum*100:>8.1f}%")
    print("  " + "-" * 50)
    print(f"  {'合计':<16}{t12['total']:>8}{t24['total']:>8}{dsum:>+8}{100.0:>8.1f}%")
    print()
    print(f"  ⇒ 1×1 混合（mixer+layer1x1）贡献了 {((t24['layer1x1']+t24['mixer'])-(t12['layer1x1']+t12['mixer']))/dsum*100:.1f}% 的参数增量，")
    print(f"    而它同时是 ch² 的交叉开关——这是模拟实现最贵的东西。")
    print(f"  ⇒ 纯 depthwise 卷积只占 dw_ch24 全部参数的 {t24['dwconv']/t24['total']*100:.1f}%。")
    print(f"    『可分离』这件事已经名不副实：真正主导的是 1×1 层。")

    print()
    print("=" * 92)
    print("结论：加宽是在给 1×1 交叉开关付钱，不是在给卷积付钱")
    print("=" * 92)
    print("  这解释了 dw_ch24 偏离幂律 +6.9%：它花 74% 的参数去买 ch×ch 全交叉开关，")
    print("  而这些交叉开关在模拟实现里既贵（面积∝ch²）又慢（累加链长）。")
    print()
    print("  反过来说：depthwise 卷积本身极其便宜（3ch 参数，无交叉开关）。")
    print("  ⇒ **正确方向是让通道数变大、但 1×1 混合变窄**，而不是两者一起变大。")

    print()
    print("=" * 92)
    print("候选方案：宽 depthwise + 窄 1×1")
    print("=" * 92)
    print("  记 ch = 通道数, r = 1×1 低秩瓶颈维数（r < ch）。")
    print("  1×1 层从 ch→ch 改为 ch→r→ch，参数与交叉开关都从 ch² 降到 2·ch·r。")
    print()

    cands = []
    for ch, r in [(24, 6), (24, 8), (32, 8), (32, 10), (48, 12), (64, 16)]:
        t, x, m, c, _ = breakdown(ch, True, 100, mix_rank=r)
        cands.append((f"wd_ch{ch}_r{r}", ch, r, t, x, m, c))

    print(f"  {'配置':<16}{'参数':>7}{'交叉开关':>9}{'MACs':>7}{'链':>4}"
          f"{'参/基准':>9}{'开关/基准':>10}")
    print("  " + "-" * 68)
    print(f"  {'lite 基准':<16}{lite['p']:>7}{lite['xbar']:>9}{lite['macs']:>7}"
          f"{lite['chain']:>4}{'1.00×':>9}{'1.00×':>10}")
    print(f"  {'dw_ch24':<16}{d24['p']:>7}{d24['xbar']:>9}{d24['macs']:>7}"
          f"{d24['chain']:>4}{d24['p']/lite['p']:>8.2f}×"
          f"{d24['xbar']/lite['xbar']:>9.0%}")
    print("  " + "-" * 68)
    for name, ch, r, t, x, m, c in cands:
        print(f"  {name:<16}{t['total']:>7}{x:>9}{m:>7}{c:>4}"
              f"{t['total']/lite['p']:>8.2f}×{x/lite['xbar']:>9.0%}")
    print()

    # 挑出参数不超过 lite 的方案
    ok = [c for c in cands if c[3]["total"] <= lite["p"]]
    print("  参数不超过 lite（6553）的方案：")
    if ok:
        for name, ch, r, t, x, m, c in ok:
            print(f"    {name}: 参数 {t['total']} ({t['total']/lite['p']:.2f}× lite), "
                  f"交叉开关 {x} ({x/lite['xbar']:.0%} of lite), "
                  f"通道 {ch} ({ch/12:.1f}× lite 的通道数)")
        print()
        print("  这一类的关键特征是：**通道数远多于 lite，但交叉开关远少于 lite**。")
        print("  模拟含义：更多并行的窄链（省面积、省布线），而非更宽的矩阵（费面积）。")
        print("  这正是差分对阵列天然的形态——每个通道一条独立支路。")

    print()
    print("=" * 92)
    print("为什么这个方向在数学上站得住（而不是又一厢情愿）")
    print("=" * 92)
    print("  1. depthwise 卷积的表达力来自『每通道独立滤波 + 1×1 跨通道混合』。")
    print("     低秩 1×1 只削减『跨通道混合』的秩，不削减『每通道滤波』。")
    print("     而 H3 已证明：depthwise 对单 tap 信息的依赖本就低，跨通道混合的边际价值在下降。")
    print("  2. 参数效率递减的根源是 1×1 的 ch²，低秩直接移除这一项。")
    print("  3. 但它有明确风险：若跨通道混合的秩确实是瓶颈，低秩会伤精度。")
    print("     ⇒ 这是**可证伪的**，见下方预登记判据。")

    print()
    print("=" * 92)
    print("预登记判据（在这些结果出来之前写定，避免事后合理化）")
    print("=" * 92)
    print("  ⚠ 关键事实：ch 加宽会让 1×1 的 ch² 迅速吃掉预算。参数 ≤ lite 的方案只有")
    print("    wd_ch24_r6（6445 参数）。wd_ch32_r8 已经是 10577 参数（1.61× lite），")
    print("    所以它**不是**等参数超越，而是『用更多参数换更少交叉开关』。")
    print("    目标必须按这个现实重新表述，不能含糊。")
    print()
    print("  实验 A：wd_ch24_r6（6445 参数, 4500 交叉开关 = lite 的 74%, 通道 24）")
    print("    · 若 NRMS ≤ 0.0804（dw_ch24 实测）→ 低秩几乎免费，同参数下交叉开关降 31%")
    print("      这是**模拟版图上的实质收益**，且不需要更多参数")
    print("    · 若 0.0804 < NRMS ≤ 0.090 → 低秩有代价但可接受，进入参数-开关 trade-off 曲线")
    print("    · 若 > 0.095 → 低秩假设被证伪，跨通道混合的秩是真瓶颈")
    print()
    print("  实验 B：wd_ch32_r8（10577 参数, 7984 交叉开关, 通道 32）")
    print("    · 用于检验『宽通道 + 窄混合』能否把 NRMS 压到 0.075 以下")
    print("    · 但必须诚实标注：这是 1.61× lite 的参数，属于「以规模换精度」，")
    print("      不满足原始目标『同等或更小参数』。仅作为**上界探针**。")
    print()
    print("  方案 B（若低秩失败）：保持全秩 1×1，但把 1×1 换成**分组**（mix_groups=2/4），")
    print("    交叉开关按组数下降而秩不降，代价是组间不混合。")

    print()
    print("=" * 92)
    print("与现有最优点的完整对比表")
    print("=" * 92)
    print(f"  {'方案':<18}{'参数':>7}{'开关':>7}{'MACs':>7}{'链':>4}{'实测/预测NRMS':>16}")
    print("  " + "-" * 62)
    rows = [
        ("lite (实测)", lite, 0.0772, "实测"),
        ("dw_ch12 (实测)", d12, 0.1084, "实测"),
        ("dw_ch24 (实测)", d24, 0.0804, "实测"),
    ]
    for n, r_, nr, tag in rows:
        print(f"  {n:<18}{r_['p']:>7}{r_['xbar']:>7}{r_['macs']:>7}{r_['chain']:>4}"
              f"{nr:>16.4f}")
    print("  " + "-" * 62)
    # 用二维标度律 + dw 偏离因子预测新方案
    A, a, b = 3.3625, 0.3073, 0.1228
    dev24 = 0.080378 / (A * 8461 ** (-a) * 4093 ** (-b))
    for name, ch, r, t, x, m, c in cands:
        pred = A * t["total"] ** (-a) * 4093 ** (-b) * dev24
        print(f"  {name:<18}{t['total']:>7}{x:>7}{m:>7}{c:>4}{pred:>13.4f} 预测")
    print()
    print(f"  （预测用二维标度律 × dw_ch24 的偏离因子 {dev24:.3f} 保守修正）")
    print(f"   注意：低秩会改变 1×1 的有效容量，此预测**未包含**低秩损失，")
    print(f"   因此这些数字是乐观上界，实测应更差。")

    out = dict(
        lite=lite, dw_ch12=d12, dw_ch24=d24,
        breakdown_ch12=t12, breakdown_ch24=t24,
        candidates=[dict(name=n, ch=ch, r=r, **t) for n, ch, r, t, x, m, c in cands],
        dev_dw24=float(dev24),
    )
    json.dump(out, open(ROOT / "arch_analog_local" / "param_flow.json", "w"),
              indent=2, default=float)
    print(f"\n结果已存 arch_analog_local/param_flow.json")


if __name__ == "__main__":
    main()
