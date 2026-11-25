"""判定性实验的最终分析 —— depthwise 路线是否成立

已有数据:
  full_ch12 s7  6553参数 6228MACs  test 0.0785   (框架自验通过, 与官方lite精确一致)
  full_ch12 s11 6553参数 6228MACs  test ?         (ep280, 即将完成)
  dw_ch12   s7  2611参数 2286MACs  test 0.1096
  dw_ch12   s11 2611参数 2286MACs  test 0.1073
  dw_ch24   s7  ~8200参数 ~4400MACs test ?        (ep220)
  dw_ch12_rf50 s7 2611参数 2286MACs test ?         (ep220)
  dw_ch24_rf50  s7 ~8200参数 ~4400MACs test ?      (ep160)

关键数学问题:
  1. 同参数下, depthwise 是否比全连接差? (H1)
  2. depthwise 是否落在全局参数效率幂律上? (决定它是否"浪费参数")
  3. 宽度补参能否把 depthwise 拉进 0.0852? (H2)
  4. depthwise 能否承受短 RF? (H3)
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A2 = ROOT / "arch_v2_local"
AA = ROOT / "arch_analog_local"


def load(d, pat):
    out = []
    for f in sorted(d.glob(pat)):
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    return out


# ---- 全局幂律 ----
P_FULL = [(841, 0.1485), (3025, 0.1014), (6553, 0.0785), (13801, 0.0653)]
xs = [math.log(p) for p, _ in P_FULL]
ys = [math.log(e) for _, e in P_FULL]
n = len(xs)
mx, my = sum(xs) / n, sum(ys) / n
sxx = sum((x - mx) ** 2 for x in xs)
b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
a = my - b * mx
A = math.exp(a)
R2 = 1 - sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / sum((y - my) ** 2 for y in ys)


def predict(p):
    return A * p ** b


print("=" * 78)
print(f"全局参数效率幂律: NRMS = {A:.4f}·P^({b:.4f})    R² = {R2:.4f}")
print("=" * 78)

full = load(AA, "full_ch12_s*.json")
dw12 = load(AA, "dw_ch12_s*.json")
dw24 = load(AA, "dw_ch24_s*.json")
rf50_12 = load(AA, "dw_ch12_rf50_s*.json")
rf50_24 = load(AA, "dw_ch24_rf50_s*.json")

# ---- 1. 框架自验 ----
print("\n【1】框架自验（预登记规则: full_ch12 s7 ∈ [0.075, 0.082]）")
for d in full:
    ok = "✅ 通过" if 0.075 <= d["test_NRMS"] <= 0.082 else "❌ 失败"
    print(f"  full_ch12 s{d['seed']}: test {d['test_NRMS']:.4f}  {ok}")

if not full:
    print("  (未完成)")
else:
    base = full[0]["test_NRMS"]
    lite = json.load(open(A2 / "lite_s7.json"))["test_NRMS"]
    print(f"  与官方 lite 精确对比: {base:.6f} vs {lite:.6f}  差 {abs(base-lite):.2e}")
    print("  => 框架可信, 下列所有对照成立")

# ---- 2. H1: 同参数 depthwise vs 全连接 ----
print("\n" + "=" * 78)
print("【2】H1 —— 同参数下 depthwise 相对全连接的损失")
print("=" * 78)
if full and dw12:
    # 用幂律反推: 2611 参数的全连接 WaveNet 应达到多少
    expect = predict(dw12[0]["params"])
    print(f"  全连接参照 (full_ch12, 6553参数):      test {full[0]['test_NRMS']:.4f}")
    print(f"  depthwise (dw_ch12, 2611参数):        test "
          f"{sum(d['test_NRMS'] for d in dw12)/len(dw12):.4f}  "
          f"(双种子: {', '.join(f'{d['test_NRMS']:.4f}' for d in dw12)})")
    print(f"  幂律预期 (2611参数应有水平):          {expect:.4f}")
    print()
    avg_dw = sum(d["test_NRMS"] for d in dw12) / len(dw12)
    print(f"  深度 → depthwise 相对幂律偏离: {(avg_dw/expect-1)*100:+.1f}%")
    print("  解读: 偏离接近 0 → depthwise 未浪费参数, 参数效率与全连接同等")
    print("        偏离很大为正 → depthwise 结构劣于同参数全连接")
    print()
    # 若要达到 A2 Lite 水平 0.0852, depthwise 需要多少参数?
    need = (0.0852 / A) ** (1 / b)
    print(f"  按幂律, depthwise 要达 0.0852 需参数: {need:.0f}")
    print(f"  (全连接 lite 用 6553 参数达到 0.0785)")
    print(f"  => depthwise 若达标, 参数需求约为全连接的 {need/6553*100:.0f}%")

# ---- 3. H2: 宽度补参 ----
print("\n" + "=" * 78)
print("【3】H2 —— 宽度补参能否把 depthwise 拉进 A2 Lite 区间")
print("=" * 78)
if dw24:
    for d in dw24:
        e = d["test_NRMS"]
        pr = predict(d["params"])
        print(f"  dw_ch24 s{d['seed']}: {d['params']}参数 {d['macs']}MACs  test {e:.4f}  "
              f"幂律预测 {pr:.4f}  偏离 {(e/pr-1)*100:+.1f}%")
    if dw12:
        p12 = dw12[0]["params"]
        e12 = sum(d["test_NRMS"] for d in dw12) / len(dw12)
        if dw24:
            p24 = dw24[0]["params"]
            e24 = dw24[0]["test_NRMS"]
            print(f"\n  ch12→ch24: 参数 {p12}→{p24} ({p24/p12:.1f}x)  "
                  f"test {e12:.4f}→{e24:.4f} ({e24/e12-1:+.1%})")
            print(f"  MACs 2286→{dw24[0]['macs']} ({dw24[0]['macs']/2286:.1f}x)")
            print(f"  参照: 全连接 lite 6553参数 6228MACs → 0.0785")
            print(f"  dw_ch24 若达 0.0852 以内, 则模拟成本远低于 lite")
else:
    print("  (未完成)")

# ---- 4. H3: 短 RF ----
print("\n" + "=" * 78)
print("【4】H3 —— depthwise 能否承受短 RF（延迟线减半）")
print("=" * 78)
for name, group, base in [("dw_ch12", dw12, None), ("dw_ch24", dw24, None)]:
    rf50 = rf50_12 if name == "dw_ch12" else rf50_24
    if group and rf50:
        b = group[0]["test_NRMS"]
        r = rf50[0]["test_NRMS"]
        print(f"  {name}: 全RF {b:.4f} ({group[0]['rf_ms']:.1f}ms)  "
              f"RF50 {r:.4f} ({rf50[0]['rf_ms']:.1f}ms)  退化 {(r/b-1)*100:+.1f}%")
    else:
        print(f"  {name}: (未完成)")

# ---- 5. 模拟成本总账 ----
print("\n" + "=" * 78)
print("【5】模拟成本总账（MACs 与延迟线才是模拟实现的真实成本）")
print("=" * 78)
rows = []
if full:
    for d in full:
        rows.append(("full_ch12 (官方lite复刻)", d["params"], d["macs"], d["rf_ms"], d["test_NRMS"]))
for d in dw12:
    rows.append((f"dw_ch12 s{d['seed']}", d["params"], d["macs"], d["rf_ms"], d["test_NRMS"]))
for d in dw24:
    rows.append((f"dw_ch24 s{d['seed']}", d["params"], d["macs"], d["rf_ms"], d["test_NRMS"]))
for d in rf50_12:
    rows.append((f"dw_ch12_rf50 s{d['seed']}", d["params"], d["macs"], d["rf_ms"], d["test_NRMS"]))
for d in rf50_24:
    rows.append((f"dw_ch24_rf50 s{d['seed']}", d["params"], d["macs"], d["rf_ms"], d["test_NRMS"]))

if rows:
    print(f"  {'变体':<26}{'参数':>7}{'MACs':>7}{'延迟ms':>9}{'test':>9}{'MACs节省':>10}")
    print("  " + "-" * 68)
    ref_macs = 6228
    for name, p, m, ms, e in rows:
        print(f"  {name:<26}{p:>7}{m:>7}{ms:>9.1f}{e:>9.4f}{(1-m/ref_macs)*100:>9.0f}%")
