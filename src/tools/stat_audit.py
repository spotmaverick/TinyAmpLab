"""统计功效审计 —— 我们的实验究竟能分辨什么

动机
----
项目全部结论都建立在每个配置 1~2 个随机种子上，但从未回答一个前置问题：

    给定实测的种子噪声，多大的效应才能被可靠检测到？

如果不回答这个问题，"lite 的 RF 效应 +11% 可信"这类断言就没有依据——
因为 +11% 可能落在该配置自身的种子波动范围内。

本脚本不产生新的训练结果，只做纯数学推断，回答三件事：

  A. 种子噪声的相对标准差在各档位上分别是多少（用配对/重复设计估计）
  B. 每个既有结论的效应量相对于噪声的信噪比，以及"重复实验后仍同号"的概率
  C. 二维标度律：把 RF 从"配对百分比"提升为标度律的一个自变量，
     检验 RF 敏感度是否随容量单调变化（这是此前只有定性描述的部分）

判据
----
效应量 d = |效应| / 种子标准差（Cohen's d 的配对版本）
  d > 1.0   结论稳健，可写入报告
  0.5~1.0   方向可信但幅度不可引用
  d < 0.5   不可判定，必须补种子

用法: python src/tools/stat_audit.py
"""
import json
import glob
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AA = ROOT / "arch_analog_local"
AV = ROOT / "arch_v2_local"

# ---------------------------------------------------------------- 数据装载

def load_points():
    recs = []
    for f in sorted(glob.glob(str(AA / "*.json")) + glob.glob(str(AV / "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if isinstance(d, list):
            d = d[0]
        if d.get("params") is None or not d.get("test_NRMS"):
            continue
        if d.get("dataset") not in (None, "tsa15"):
            continue
        recs.append(dict(variant=d["variant"], P=d["params"], RF=d["rf"],
                         seed=d["seed"], nr=d["test_NRMS"]))
    return recs


def index(recs):
    """(P, RF) -> {seed: nr}，同时带上 variant 名"""
    g = {}
    for r in recs:
        key = (r["P"], r["RF"])
        g.setdefault(key, {})[r["seed"]] = (r["nr"], r["variant"])
    return g


# ---------------------------------------------------------------- A. 种子噪声

def seed_noise(g):
    """对有双种子的 (P,RF) 组合，估计相对标准差。

    两点估计的样本标准差会严重低估真值（n=2 时 s 的期望只有 sigma 的 0.8 倍），
    因此这里给出去偏修正 c4(n)=sqrt(2/(n-1))*Gamma(n/2)/Gamma((n-1)/2)。
    两点时 c4 = 1/0.7979 = 1.253。
    """
    rows = []
    for (P, RF), seeds in sorted(g.items()):
        if len(seeds) < 2:
            continue
        vals = np.array([v for v, _ in seeds.values()])
        name = list(seeds.values())[0][1]
        mean = vals.mean()
        rel_sd_raw = vals.std(ddof=1) / mean          # 样本相对标准差
        rel_sd = rel_sd_raw * 1.2533                  # n=2 去偏
        rows.append(dict(P=P, RF=RF, name=name, vals=vals, mean=mean,
                         rel_sd=rel_sd, rel_sd_raw=rel_sd_raw))
    return rows


# ---------------------------------------------------------------- B. 效应量

def paired_effect(g, P, rf_hi, rf_lo):
    """同一 seed 内 rf_hi -> rf_lo 的配对效应（相对），返回每 seed 的相对变化。"""
    a = g.get((P, rf_hi), {})
    b = g.get((P, rf_lo), {})
    common = sorted(set(a) & set(b))
    if not common:
        return None
    rel = [(b[s][0] - a[s][0]) / a[s][0] for s in common]
    return dict(seeds=common, rel=rel, mean=float(np.mean(rel)),
                name=a[common[0]][1])


def effect_table(g, noise_by_P):
    print("=" * 90)
    print("B. RF 减半效应的信噪比 (配对设计，同 seed 内比较)")
    print("=" * 90)
    print(f"{'配置':<14}{'P':>7}{'seed':>6}{'全RF':>10}{'半RF':>10}{'相对变化':>11}{'该档种子噪声':>13}{'效应/噪声':>11}")
    print("-" * 90)

    verdicts = []
    for (P, rf_hi) in sorted({k for k in g if k[1] == 4093}):
        eff = paired_effect(g, P, 4093, 2049)
        if eff is None:
            continue
        sd = noise_by_P.get(P)
        for s, r in zip(eff["seeds"], eff["rel"]):
            hi = g[(P, 4093)][s][0]
            lo = g[(P, 2049)][s][0]
            ratio = abs(r) / sd if sd else float("nan")
            print(f"{eff['name']:<14}{P:>7}{s:>6}{hi:>10.6f}{lo:>10.6f}"
                  f"{r*100:>10.1f}%{sd*100:>12.1f}%{ratio:>11.2f}")
        # 判定：用配对效应自身的跨种子一致性
        consistent = len(eff["rel"]) > 1 and np.all(np.array(eff["rel"]) > 0) or \
                     len(eff["rel"]) > 1 and np.all(np.array(eff["rel"]) < 0)
        verdicts.append(dict(name=eff["name"], P=P, mean=eff["mean"],
                             sd=sd, consistent=consistent,
                             n_seeds=len(eff["rel"])))
    return verdicts


def power_analysis(noise_rows):
    """两点配对设计能检测到的最小效应（80% 功效）。

    配对设计：检验均值是否偏离 0。效应 d = mu / sigma（同一配置内的种子标准差）。
    两点时检验统计量 t = mu / (sigma/sqrt(2))，自由度 1 —— 这几乎没有功效。
    所以除了算 d，还要说明"两点设计根本不足以做统计检验"这一事实。
    """
    print()
    print("=" * 90)
    print("A'. 一个必须正视的事实：两点配对设计的统计功效")
    print("=" * 90)
    print("  配对两点：t = (mu/sigma)*sqrt(2)，自由度 df=1。")
    print("  df=1 时 t 分布尾部极厚，双侧 5% 临界值 t_crit = 12.71。")
    print("  即：要声称显著，需要 mu/sigma > 12.71/1.414 = 8.99。")
    print()
    print("  => 任何 mu/sigma < 9 的两点配对，都**无法**通过统计显著检验。")
    print("     这意味着本项目所有双种子结论，严格来说只能算『方向性观察』，")
    print("     不能算『统计显著』。这不是数据质量问题，是设计自由度问题。")
    print()
    print("  若要把可检测效应降到 mu/sigma = 2（80% 功效, df=n-1），需要的种子数：")
    for d in (2.0, 1.5, 1.0):
        # 近似：n ≈ 8/d^2 （配对 t 检验经验公式, alpha=.05, power=.8）
        n = int(np.ceil(8.0 / d ** 2)) + 1
        print(f"     效应 d={d:.1f}  →  需 {n} 个种子（当前 2 个）")


# ---------------------------------------------------------------- C. 二维标度律

def fit_2d(recs):
    """ln NRMS = ln A - a*ln P - b*ln RF  的多元线性回归。"""
    P = np.array([r["P"] for r in recs], float)
    R = np.array([r["RF"] for r in recs], float)
    y = np.log([r["nr"] for r in recs])

    X = np.column_stack([np.ones_like(P), np.log(P), np.log(R)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    n, k = len(y), 2
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    # 残差标准误
    sigma = np.sqrt(ss_res / (n - k - 1))
    return dict(A=np.exp(beta[0]), a=-beta[1], b=-beta[2], r2=r2,
                r2_adj=r2_adj, sigma=sigma, n=n)


def main():
    recs = load_points()
    g = index(recs)

    print("=" * 90)
    print("统计功效审计 —— 基于现有 22 个实验点，不需要新训练")
    print("=" * 90)
    print(f"数据点 {len(recs)} 个，(P,RF) 组合 {len(g)} 个")
    print()

    # ---- A. 噪声
    noise_rows = seed_noise(g)
    print("=" * 90)
    print("A. 种子噪声（相对标准差，n=2 已去偏 1.253×）")
    print("=" * 90)
    print(f"{'配置':<14}{'P':>7}{'RF':>7}{'s7/首':>10}{'s11/次':>10}{'均值':>10}{'相对SD':>10}")
    print("-" * 90)
    noise_by_P = {}
    for r in noise_rows:
        v = r["vals"]
        print(f"{r['name']:<14}{r['P']:>7}{r['RF']:>7}"
              f"{(v[0] if len(v) > 0 else float('nan')):>10.6f}"
              f"{(v[1] if len(v) > 1 else float('nan')):>10.6f}"
              f"{r['mean']:>10.6f}{r['rel_sd']*100:>9.1f}%")
        # 同一 P 取双 RF 中较大者（保守）
        noise_by_P[r["P"]] = max(noise_by_P.get(r["P"], 0), r["rel_sd"])

    if noise_by_P:
        allsd = np.array(list(noise_by_P.values()))
        print("-" * 90)
        print(f"{'合计':<14}{'':>7}{'':>7}{'':>10}{'':>10}"
              f"{'':>10}{allsd.mean()*100:>9.1f}%")
        print(f"  噪声范围 {allsd.min()*100:.1f}% ~ {allsd.max()*100:.1f}%，中位 {np.median(allsd)*100:.1f}%")
        print(f"  ⚠ 以中位 {np.median(allsd)*100:.0f}% 计，任何小于 {np.median(allsd)*100:.0f}% 的效应")
        print(f"    都淹没在噪声里，必须靠配对设计（而非绝对比较）才能提取。")

    # ---- B. 效应量
    print()
    effect_table(g, noise_by_P)

    # ---- 功效
    power_analysis(noise_rows)

    # ---- C. 二维标度律
    print()
    print("=" * 90)
    print("C. 二维标度律：NRMS = A · P^(-a) · RF^(-b)")
    print("=" * 90)
    fit = fit_2d(recs)
    print(f"  A  = {fit['A']:.4f}")
    print(f"  a  = {fit['a']:.4f}   (P 指数，此前单变量拟合为 0.2975)")
    print(f"  b  = {fit['b']:.4f}   (RF 指数)")
    print(f"  R² = {fit['r2']:.5f}   调整 R² = {fit['r2_adj']:.5f}   n = {fit['n']}")
    print(f"  残差标准误 σ = {fit['sigma']:.4f}（对数域，≈ {fit['sigma']*100:.1f}% 相对误差）")
    print()
    print("  解读 b 的物理含义：")
    print(f"    RF 减半（×0.5）的预测代价 = 0.5^(-b) - 1 = {0.5**(-fit['b'])-1:+.1%}")
    print(f"    实测 lite RF50 配对均值 = +11.7%，model 给出 {0.5**(-fit['b'])-1:+.1%}")
    print()
    print("  ⚠ 单一 b 意味着『RF 代价与容量无关』，但实测如表 B 显示小容量档符号相反。")
    print("    这说明需要 b(P) —— RF 敏感度随容量变化。下面直接估计：")

    # 分容量的局部 RF 指数
    print()
    print(f"  {'配置':<14}{'P':>7}{'b_local':>10}{'解释':>40}")
    print("  " + "-" * 72)
    for P in sorted({k[0] for k in g}):
        if (P, 4093) in g and (P, 2049) in g:
            hi = np.mean([v for v, _ in g[(P, 4093)].values()])
            lo = np.mean([v for v, _ in g[(P, 2049)].values()])
            name = list(g[(P, 4093)].values())[0][1]
            b_loc = -np.log(lo / hi) / np.log(2049 / 4093)
            note = ("RF 越长越好" if b_loc > 0.01
                    else "RF 无关" if abs(b_loc) <= 0.01
                    else "RF 越短越好")
            print(f"  {name:<14}{P:>7}{b_loc:>10.4f}{note:>40}")

    # ---- 保存
    out = dict(noise={str(k): v for k, v in noise_by_P.items()},
               fit2d=fit, n_points=len(recs))
    op = ROOT / "arch_analog_local" / "stat_audit.json"
    json.dump(out, open(op, "w"), indent=2, default=float)
    print()
    print(f"结果已存 {op}")


if __name__ == "__main__":
    main()
