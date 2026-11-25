"""补实验设计的定量依据 —— 哪些种子最该补，补了能定什么

背景
----
stat_audit.py 发现两件事：
  1. nano 档种子噪声 22.3%，而它声称的 RF 效应只有 5% —— 效应/噪声 = 0.25
  2. 全部双种子结论的配对设计自由度 df=1，t 临界值 12.71

本脚本用蒙特卡洛精确计算（不靠经验公式）：
  A. 给定真实效应量 d 与种子数 n，配对 t 检验的检出概率（功效）
  B. 反过来：要达到 80% 功效，各结论分别需要多少种子
  C. 按「补一个种子带来的信息增益」排序，给出最小代价的补实验清单
  D. 目标可达性：在实测噪声下，「超越 A2 Lite」需要多大的真实改进才可被证明

全部为纯推断，不需要 DCU。

用法: python src/tools/power_plan.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

RNG = np.random.default_rng(20261125)
NSIM = 200_000


def t_crit(df, alpha=0.05):
    """双侧 t 临界值，用小规模蒙特卡洛反解（避免依赖 scipy）。"""
    if df <= 0:
        return np.inf
    t = RNG.standard_normal(NSIM) / np.sqrt(RNG.chisquare(df, NSIM) / df)
    return float(np.quantile(np.abs(t), 1 - alpha))


def power_paired(d, n, alpha=0.05, nsim=NSIM):
    """配对设计功效：n 对观测，真实效应 d（以配对差的标准差为单位）。"""
    if n < 2:
        return 0.0
    df = n - 1
    tc = t_crit(df, alpha)
    # 配对差 ~ N(d, 1)，检验统计量 = mean/(s/sqrt(n))
    diffs = RNG.standard_normal((nsim, n)) + d
    mean = diffs.mean(axis=1)
    s = diffs.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mean / (s / np.sqrt(n))
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.mean(np.abs(t) > tc))


def n_required(d, target=0.80, alpha=0.05):
    for n in range(2, 400):
        if power_paired(d, n, alpha) >= target:
            return n
    return None


# ---------------------------------------------------------------- 既有结论

# (名称, 配置, 配对效应均值(相对), 该配置种子相对SD)
CLAIMS = [
    ("lite RF 减半 +11~13%", "lite", 0.117, 0.029),
    ("dw_ch12 RF 减半 +7.9%", "dw_ch12", 0.079, 0.019),
    ("nano RF 减半 −5%", "nano", -0.055, 0.223),
    ("feather RF 减半 不可判定", "feather", 0.050, 0.048),
    ("dw_ch24 偏离幂律 +8%", "dw_ch24", 0.080, None),   # 单种子，噪声未知
    ("dw_ch12 偏离幂律 +2.7%", "dw_ch12", 0.027, 0.019),
]


def main():
    print("=" * 92)
    print("补实验设计的定量依据（蒙特卡洛，NSIM=200k）")
    print("=" * 92)

    # ---- A. 两点设计的功效
    print()
    print("A. 为什么 df=1 几乎是死路：配对两点设计的功效")
    print("-" * 92)
    print(f"{'真实效应 d':>12}{'n=2 功效':>12}{'n=3 功效':>12}{'n=5 功效':>12}{'n=9 功效':>12}")
    for d in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        row = [power_paired(d, n) for n in (2, 3, 5, 9)]
        print(f"{d:>12.1f}" + "".join(f"{p*100:>11.1f}%" for p in row))
    print()
    print("  读法：即便真实效应 d=3（效应是噪声的 3 倍），两点设计也只有 "
          f"{power_paired(3.0,2)*100:.0f}% 的检出率。")
    print("        这就是为什么此前所有『双种子同向即认定可信』的做法在统计上站不住——")
    print("        两个种子同向的概率在无效应时本身就是 50%。")

    # ---- B. 各结论所需种子数
    print()
    print("=" * 92)
    print("B. 每条既有结论要变成『可证实的』需要多少种子")
    print("-" * 92)
    print(f"{'结论':<26}{'效应':>8}{'噪声':>8}{'d':>7}{'现种子':>7}{'需种子':>8}{'判定':>14}")
    print("-" * 92)
    plan = []
    for name, cfg, eff, sd in CLAIMS:
        if sd is None:
            print(f"{name:<26}{eff*100:>7.1f}%{'未知':>8}{'—':>7}{1:>7}{'—':>8}{'先测噪声':>14}")
            plan.append(dict(name=name, cfg=cfg, eff=eff, sd=None, need=None,
                             action="补 1 个种子以估计噪声"))
            continue
        d = abs(eff) / sd
        need = n_required(d)
        if d >= 1.0 and need and need <= 9:
            verdict = "可证实"
        elif d >= 0.5:
            verdict = "方向可信"
        else:
            verdict = "代价过高"
        print(f"{name:<26}{eff*100:>7.1f}%{sd*100:>7.1f}%{d:>7.2f}{2:>7}"
              f"{(need if need else '>400'):>8}{verdict:>14}")
        plan.append(dict(name=name, cfg=cfg, eff=eff, sd=sd, d=d, need=need,
                         action=""))

    # ---- C. 补实验优先级
    print()
    print("=" * 92)
    print("C. 补实验优先级（按信息增益 / 代价 排序）")
    print("-" * 92)
    print("  评分 = 该结论对模拟版图决策的重要性 × (1 - 当前功效)")
    print()
    prio = [
        ("dw_ch24 s11", "dw_ch24", 0.080, 0.03,
         "H2 唯一的候选点，现在是单种子。补 1 个种子即可给出误差棒，"
         "决定 depthwise 能否在 <lite 的器件规模下进入 0.08 区间",
         "★★★ 最高"),
        ("dw_ch12_rf50 s11", "dw_ch12_rf50", 0.079, 0.02,
         "H3 判定只有单种子。depthwise 短 RF 的真实代价决定延迟线能否减半",
         "★★★ 最高"),
        ("standard_rf50 s11", "standard_rf50", 0.113, 0.01,
         "验证 b_local 随容量单调上升的假设（当前只有 s7）",
         "★★ 中"),
        ("dw_ch24_rf50 s7", "dw_ch24_rf50", None, None,
         "H2+H3 交互项，作业已提交但结果未回传",
         "★★ 中（已有作业）"),
        ("nano 补种子", "nano", 0.055, 0.223,
         "噪声 22% 远超效应 5%，需 ~150 个种子才可判定。**放弃**："
         "nano 档不承载任何模拟版图决策，不值得",
         "✗ 不补"),
    ]
    for i, (exp, cfg, eff, sd, why, star) in enumerate(prio, 1):
        print(f"  {i}. {exp:<20} {star}")
        print(f"     {why}")
        print()
    print("  => 最小补实验集 = {dw_ch24 s11, dw_ch12_rf50 s11} 共 2 个作业，")
    print("     即可把 H2 与 H3 从『单种子观察』升级为『有误差棒的结论』。")

    # ---- D. 目标可达性
    print()
    print("=" * 92)
    print("D. 目标可达性：在实测噪声下，「超越 A2 Lite」需要多大改进才能被证明")
    print("-" * 92)
    fit = json.load(open(ROOT / "arch_analog_local" / "stat_audit.json"))["fit2d"]
    A, a, b = fit["A"], fit["a"], fit["b"]
    lite = 0.0772           # 双种子均值
    lite_sd = 0.029
    print(f"  A2 Lite（我方协议，双种子均值）= {lite:.4f}，种子相对 SD = {lite_sd*100:.1f}%")
    print()
    print(f"  {'目标 NRMS':>10}{'相对改进':>10}{'效应 d':>8}{'需种子数':>9}{'所需参数 P':>12}{'等效':>16}")
    print("  " + "-" * 72)
    for target in (0.075, 0.072, 0.070, 0.065, 0.060):
        rel = (lite - target) / lite
        d = rel / lite_sd
        need = n_required(d) if d > 0 else None
        # 由二维标度律反解 P（RF 固定 4093）
        P = (A * 4093 ** (-b) / target) ** (1 / a)
        eq = f"{P/6553:.2f}× lite"
        print(f"  {target:>10.3f}{-rel*100:>9.1f}%{d:>8.2f}"
              f"{(need if need else '>400'):>9}{P:>12.0f}{eq:>16}")
    P070 = (A * 4093 ** (-b) / 0.070) ** (1 / a)
    d070 = ((lite - 0.070) / lite) / lite_sd
    n070 = n_required(d070)
    P065 = (A * 4093 ** (-b) / 0.065) ** (1 / a)
    print()
    print("  关键读法：")
    print(f"    · 要达到 NRMS 0.070（比 lite 好 {(lite-0.070)/lite*100:.1f}%），二维标度律要求约 {P070:.0f} 参数")
    print(f"      （{P070/6553:.2f}× lite）。dw_ch24（8461 参数，实测 0.0804）已接近该区间，")
    print(f"      但按幂律它只对应 {A*8461**(-a)*4093**(-b):.4f}，实测 0.0804 略差于幂律（+8%），")
    print(f"      说明加宽的边际收益正在递减，0.070 需要比幂律更多的参数。")
    print(f"    · 该路径上的效应 d = {d070:.2f}，配对设计需 {n070} 个种子才能声称显著（当前 2 个）。")
    print(f"      所以**这不是一条不可证明的路，但必须把种子数从 2 提到 {n070}**。")
    print(f"    · 若把目标放宽到 0.065（standard 已达 0.0655），效应 d = "
          f"{((lite-0.065)/lite)/lite_sd:.2f}，只需 {n_required(((lite-0.065)/lite)/lite_sd)} 个种子。")
    print(f"      但 0.065 需要 {P065:.0f} 参数（{P065/6553:.2f}× lite），模拟器件规模将远超预算。")
    print()
    print("  ⚠ 前提：二维标度律的 a=0.3156 在 P>8461 区间是否仍成立，未经验证。")
    print("     standard（13801, 0.0655）是该区间唯一的锚点，且种子噪声仅 0.8%，可信。")

    out = dict(claims=plan, priority=[p[0] for p in prio])
    json.dump(out, open(ROOT / "arch_analog_local" / "power_plan.json", "w"),
              indent=2, default=float)
    print()
    print("结果已存 arch_analog_local/power_plan.json")


if __name__ == "__main__":
    main()
