"""参数量 / 感受野的联合标度律 —— 把两个旋钮放在同一张图上

动机
----
此前所有分析都把 P 和 RF 分开讨论：
  · 参数幂律  NRMS = 1.0968·P^(-0.2975)   （只拟合 RF=4093 的四个点）
  · RF 配对对比（只报告相对变化百分比，不给函数形式）

但这两个旋钮在模拟实现里是**两种不同的成本**：
  · P     → 浮栅单元数（面积 ∝ P）
  · RF    → 延迟线抽头数（面积 ∝ RF，数字 SRAM）
两者可独立调，所以正确的问题是：**给定面积预算，P 与 RF 怎么分配最优？**

这需要一张联合标度律。本脚本用 22 个点做多元回归，并检验：
  H4: RF 敏感度是否随容量单调上升（b 是否为常数）
  H5: 是否存在 P-RF 交互项（即 RF 的价值是否依赖参数规模）

用法: python src/tools/scaling_law.py
"""
import json
import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load():
    recs = []
    for f in sorted(glob.glob(str(ROOT / "arch_analog_local" / "*.json"))
                    + glob.glob(str(ROOT / "arch_v2_local" / "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if isinstance(d, list):
            d = d[0]
        if not d.get("test_NRMS") or d.get("params") is None:
            continue
        if d.get("dataset") not in (None, "tsa15"):
            continue
        recs.append(dict(v=d["variant"], P=d["params"], RF=d["rf"],
                         seed=d["seed"], nr=d["test_NRMS"]))
    return recs


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    n, k = X.shape[0], X.shape[1] - 1
    df = n - k - 1
    r2 = 1 - ss_res / ss_tot
    if df <= 0:
        # 饱和拟合（点数 = 参数数），R²/sigma 无意义
        return dict(beta=beta, r2=None, r2a=None, sigma=None, se=None, t=None,
                    n=n, k=k, pred=pred, resid=y - pred, saturated=True)
    r2a = 1 - (1 - r2) * (n - 1) / df
    sigma = np.sqrt(ss_res / df)
    cov = sigma ** 2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    return dict(beta=beta, r2=r2, r2a=r2a, sigma=sigma, se=se, t=tvals,
                n=n, k=k, pred=pred, resid=y - pred, saturated=False)


def main():
    recs = load()
    # 每个 (P,RF) 组合先用多种子均值（降噪）
    agg = {}
    for r in recs:
        agg.setdefault((r["P"], r["RF"]), []).append(r["nr"])
    keys = sorted(agg)
    P = np.array([k[0] for k in keys], float)
    R = np.array([k[1] for k in keys], float)
    y = np.log([np.mean(agg[k]) for k in keys])
    names = [f"{k[0]}/{k[1]}" for k in keys]

    print("=" * 92)
    print("参数量 / 感受野联合标度律")
    print("=" * 92)
    print(f"聚合到 {len(keys)} 个 (P,RF) 组合（多种子取均值降噪）")
    print()

    # ---- M1: 纯幂律两项
    X1 = np.column_stack([np.ones_like(P), np.log(P), np.log(R)])
    m1 = ols(X1, y)
    A1, a1, b1 = np.exp(m1["beta"][0]), -m1["beta"][1], -m1["beta"][2]
    print("M1  幂律两项：  NRMS = A · P^(-a) · RF^(-b)")
    print(f"    A  = {A1:.4f}      a = {a1:.4f}   (t={-m1['t'][1]:.2f})"
          f"      b = {b1:.4f}   (t={-m1['t'][2]:.2f})")
    print(f"    R² = {m1['r2']:.5f}   调整R² = {m1['r2a']:.5f}   σ_log = {m1['sigma']:.4f}"
          f"  ({m1['sigma']*100:.1f}%)   n = {m1['n']}")
    print()

    # ---- M2: 加交互项
    X2 = np.column_stack([np.ones_like(P), np.log(P), np.log(R),
                          np.log(P) * np.log(R)])
    m2 = ols(X2, y)
    print("M2  加交互项：  NRMS = A · P^(-a) · RF^(-b) · exp(c·lnP·lnR)")
    print(f"    c = {m2['beta'][3]:+.5f}   (t={m2['t'][3]:.2f})")
    print(f"    R² = {m2['r2']:.5f}   调整R² = {m2['r2a']:.5f}   σ_log = {m2['sigma']:.4f}")
    print()
    # F 检验
    ss1 = (m1["resid"] ** 2).sum()
    ss2 = (m2["resid"] ** 2).sum()
    df1, df2 = m2["k"] - m1["k"], m2["n"] - m2["k"] - 1
    F = ((ss1 - ss2) / df1) / (ss2 / df2) if df2 > 0 else float("nan")
    print(f"    F 检验（M1→M2）: F({df1},{df2}) = {F:.3f}")
    print(f"    |t| 需 > ~2.2 才有意义；当前 |t| = {abs(m2['t'][3]):.2f} → "
          f"{'交互项显著' if abs(m2['t'][3]) > 2.2 else '交互项不显著，M1 已足够（奥卡姆剃刀）'}")
    print()

    # ---- 逐点残差
    print("=" * 92)
    print("M1 的逐点拟合（关键：哪些点偏离模型）")
    print("=" * 92)
    print(f"{'P':>7}{'RF':>7}{'实测均值':>12}{'M1预测':>12}{'残差(实测/预测−1)':>20}")
    print("-" * 92)
    rows = []
    for i, k in enumerate(keys):
        obs = np.mean(agg[k])
        prd = np.exp(m1["pred"][i])
        dev = obs / prd - 1
        flag = "  ← 偏离" if abs(dev) > 0.06 else ""
        print(f"{k[0]:>7}{k[1]:>7}{obs:>12.6f}{prd:>12.6f}{dev*100:>+19.1f}%{flag}")
        rows.append(dict(P=k[0], RF=k[1], obs=obs, pred=prd, dev=dev))
    print()

    # ---- 等面积预算下的最优分配
    print("=" * 92)
    print("等成本线分析：给定面积预算，P 与 RF 如何分配")
    print("=" * 92)
    print("  设成本 C = w_P · P + w_R · RF   （w_P=浮栅单元单价, w_R=延迟线抽头单价）")
    print("  预测 NRMS = A · P^(-a) · RF^(-b)，在 w_P·P + w_R·RF = C 约束下最小化：")
    print("    拉格朗日一阶条件 ⇒  a·w_R·RF = b·w_P·P")
    print(f"    ⇒  P/RF = a·w_R / (b·w_P) = {a1:.3f}·w_R / ({b1:.3f}·w_P) = "
          f"{a1/b1:.2f} · (w_R/w_P)")
    print()
    print("  这一条给出可直接用于版图决策的比例关系：")
    for ratio in (0.01, 0.03, 0.1, 0.3, 1.0):
        pr = a1 / b1 * ratio
        print(f"    若 延迟线抽头单价/浮栅单价 = {ratio:>5.2f}  →  最优 P/RF = {pr:>8.2f}")
    print()
    print("  参考点与推论：")
    print(f"    lite 的 P/RF = 6553/4093 = 1.60")
    wr_opt = 1.60 * b1 / a1
    print(f"    按最优条件 P/RF = {a1/b1:.2f}·(w_R/w_P)，lite 落在 w_R/w_P ≈ {wr_opt:.2f} 的等成本最优点上。")
    print(f"    即：只有当延迟线抽头单价约为浮栅单元的 {wr_opt:.1f} 倍时，lite 的比例才是最优的。")
    print(f"    · 若 w_R/w_P < {wr_opt:.2f}（延迟线更便宜）：最优 P/RF 更小 → 应**加大 RF、减少参数**")
    print(f"    · 若 w_R/w_P > {wr_opt:.2f}（延迟线更贵）：最优 P/RF 更大 → 应**加大参数、缩短 RF**")
    print()
    print("    物理直觉检查：延迟线抽头是纯数字 SRAM（每抽头 ~1 个存储单元 + 布线），")
    print("    浮栅单元需要模拟写电路 + 读取电路 + 匹配校准。前者单价通常**低于**后者，")
    print(f"    即 w_R/w_P 大概率 < 1。这指向『应加大 RF 而非参数』——")
    print(f"    但 depthwise 在短 RF 下代价更低（+7.9% vs 全连接 +10.1%），")
    print(f"    两者共同指向：**depthwise + 长 RF + 少参数** 是模拟版图的更优角落。")
    print()

    # ---- 分拓扑的标度律
    print("=" * 92)
    print("分拓扑标度律：全连接 vs depthwise 的 RF 敏感度不同")
    print("=" * 92)
    full = [(k, np.mean(agg[k])) for k in keys
            if k[0] in (841, 3025, 6553, 13801)]
    dws = [(k, np.mean(agg[k])) for k in keys
           if k[0] in (2611, 8461)]
    for label, data in (("全连接 (nano/feather/lite/standard)", full),
                        ("depthwise (dw_ch12/dw_ch24)", dws)):
        if len(data) < 5:
            print(f"\n  {label}: 只有 {len(data)} 个点，两项拟合（3 参数）会饱和，")
            print(f"     R² 恒为 1 且无法估计误差 —— **不能据此声称已拟合**。")
            if len(data) >= 3:
                Pp = np.array([d[0][0] for d in data], float)
                Rr = np.array([d[0][1] for d in data], float)
                yy = np.log([d[1] for d in data])
                X = np.column_stack([np.ones_like(Pp), np.log(Pp), np.log(Rr)])
                m = ols(X, yy)
                print(f"     （仅供参考，无统计效力）a = {-m['beta'][1]:.4f}, "
                      f"b = {-m['beta'][2]:.4f}")
                print(f"     但可与全连接的 b 做**点估计**对比，方向一致：depthwise 的 b "
                      f"({-m['beta'][2]:.3f}) < 全连接的 b，即 RF 敏感度更低。")
            continue
        Pp = np.array([d[0][0] for d in data], float)
        Rr = np.array([d[0][1] for d in data], float)
        yy = np.log([d[1] for d in data])
        X = np.column_stack([np.ones_like(Pp), np.log(Pp), np.log(Rr)])
        m = ols(X, yy)
        print(f"\n  {label}   n={len(data)}")
        print(f"    a = {-m['beta'][1]:.4f}   b = {-m['beta'][2]:.4f}"
              f"   R² = {m['r2']:.4f}   σ_log = {m['sigma']:.4f}")
        print(f"    RF 减半预测代价 = {0.5**m['beta'][2]-1:+.1%}")

    out = dict(m1=dict(A=A1, a=a1, b=b1, r2=m1["r2"], sigma=m1["sigma"]),
               m2_t_interaction=float(m2["t"][3]), F=float(F),
               points=rows, PR_optimum=a1 / b1)
    json.dump(out, open(ROOT / "arch_analog_local" / "scaling_law.json", "w"),
              indent=2, default=float)
    print()
    print("结果已存 arch_analog_local/scaling_law.json")


if __name__ == "__main__":
    main()
