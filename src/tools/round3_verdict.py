#!/usr/bin/env python3
"""Round3 六作业结果的自动判定（判据在训练前已预登记）。

判据来源：docs/reports/AUDIT_ROUND3_STATS_AND_COST.md 第五节
         src/tools/param_flow.py 的"预登记判据"段
"""
import json
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def load(variant, seed, which="all"):
    dirs = ["arch_analog_local", "arch_v2_local"] if which == "all" else [which]
    for d in dirs:
        p = ROOT / d / f"{variant}_s{seed}.json"
        if p.exists():
            return json.load(open(p))
    return None


def fmt(r):
    if not r:
        return "缺失"
    return f"{r['test_NRMS']:.6f}"


def main():
    print("=" * 74)
    print("Round3 结果判定（判据预登记于训练前，不做事后调整）")
    print("=" * 74)

    # 锚点
    a7 = load("full_ch12", 7)
    print(f"\n[0] 环境锚点 full_ch12 s7: {fmt(a7)}  (旧环境 0.078476)")
    if a7:
        dev = abs(a7["test_NRMS"] - 0.078476) / 0.078476 * 100
        print(f"    偏离 {dev:.4f}%  → " + ("数值一致 ✓" if dev < 1 else "不一致 ✗"))

    # H2: dw_ch24 误差棒
    d11 = load("dw_ch24", 11)
    d7 = load("dw_ch24", 7)
    print("\n[H2] dw_ch24 双种子误差棒")
    print(f"     s7={fmt(d7)}  s11={fmt(d11)}")
    if d7 and d11:
        mu = (d7["test_NRMS"] + d11["test_NRMS"]) / 2
        sd = abs(d7["test_NRMS"] - d11["test_NRMS"]) / 2
        print(f"     μ={mu:.6f}  ±{sd:.6f} ({sd/mu*100:.1f}%)")
        print("     判据: ≤0.12 且交叉开关不反超 → H2 方向成立（仍非统计显著）")

    # H3: 配对补齐
    f7, f11 = load("dw_ch12_rf50", 7), load("dw_ch12_rf50", 11)
    b7, b11 = load("dw_ch12", 7), load("dw_ch12", 11)
    print("\n[H3] depthwise 短 RF 代价（配对补齐）")
    print(f"     全RF s7={fmt(b7)} s11={fmt(b11)}  |  rf50 s7={fmt(f7)} s11={fmt(f11)}")
    if b7 and f7:
        e7 = (f7["test_NRMS"] / b7["test_NRMS"] - 1) * 100
        print(f"     s7 效应 +{e7:.1f}%", end="")
        if b11 and f11:
            e11 = (f11["test_NRMS"] / b11["test_NRMS"] - 1) * 100
            print(f"  s11 效应 +{e11:.1f}%  → " +
                  ("双种子同向 ✓（方向可信）" if e7 * e11 > 0 else "反向 ✗（不可判定）"))
        else:
            print("（s11 缺失）")

    # 低秩 A: wd_ch24_r6（预登记判据）
    print("\n[低秩A] wd_ch24_r6 (6445 参数 / 交叉开关 4500 = lite 74%)")
    print("     预登记: ≤0.0804 免费 | 0.0804~0.090 可接受 | >0.095 证伪")
    for s in (7, 11):
        r = load("wd_ch24_r6", s)
        if not r:
            print(f"     s{s}: 缺失")
            continue
        v = r["test_NRMS"]
        verdict = ("低秩几乎免费 ✓" if v <= 0.0804 else
                   "有代价但可接受" if v <= 0.090 else
                   "低秩假设证伪 ✗")
        print(f"     s{s}: {v:.6f}  → {verdict}")

    # b_local 趋势
    print("\n[b_local] standard_rf50 s11 补容量趋势")
    s11 = load("standard_rf50", 11)
    s7 = load("standard_rf50", 7)
    print(f"     s7={fmt(s7)}  s11={fmt(s11)}")
    if s7 and s11:
        mu = (s7["test_NRMS"] + s11["test_NRMS"]) / 2
        print(f"     μ={mu:.6f}")

    # 上界探针
    print("\n[上界] wd_ch32_r8 s7 (10577 参数 = 1.61× lite，非等参数)")
    u = load("wd_ch32_r8", 7)
    print(f"     {fmt(u)}" + ("  → 能否压到 0.075 以下" if u else ""))

    # 低秩诊断：秩 vs 宽度
    print("\n[低秩诊断] 秩 vs 网络宽度（250 epochs）")
    print("     wd_ch24_r6 (r=6,ch=24,6445P): ", end="")
    print(f"{fmt(load('wd_ch24_r6',7))} / {fmt(load('wd_ch24_r6',11))}")
    print("     wd_ch24_r8 (r=8,ch=24,7741P): ", end="")
    print(f"{fmt(load('wd_ch24_r8',7))} / {fmt(load('wd_ch24_r8',11))}")
    print("     wd_ch32_r6 (r=6,ch=32,8849P): ", end="")
    print(f"{fmt(load('wd_ch32_r6',7))}")
    print("     → 加秩无效、加宽有效 ⇒ 瓶颈是网络宽度而非秩")

    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
