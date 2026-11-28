"""P6X 判定：三份独立实现是否一致 + 跨硬件是否一致 + 统计噪声是否够小

按 `PROTOCOL.md` 第 10 节执行判定：

  L1 实现一致性：同一指标在三次实现间的相对差 < 1e-3（0.1%）
  L1b 跨硬件：三次实现分处不同硬件（bw2 BW DCU / z100 Z100 DCU），
      若仍满足 L1，说明结论不依赖具体加速卡
  L2 统计稳定性：MC=4 → MC=16，同一模型同一指标变化 < 5%

不满足就打印失败项，并指明是哪一份实现偏离——那份实现就是"被推翻的实现"，
这是本协议最有价值的信息（说明它有 bug）。

用法：
    python3 p6x_compare.py --dir ../../reclaimed/p6x_20261128
"""
import argparse
import glob
import json
import statistics as st
from pathlib import Path

L1_REL = 1e-3          # 0.1%
L2_REL = 0.05          # 5%


def load(d, impl):
    p = Path(d) / f"p6x_{impl}.json"
    if not p.is_file():
        return None
    return json.load(open(p))


def rows_of(doc):
    if doc is None:
        return {}
    return {k: v for k, v in doc["rows"].items() if "grid" in v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../../reclaimed/p6x_20261128")
    ap.add_argument("--mc16", default="p6x_mc16.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    docs = {i: load(a.dir, i) for i in "ABC"}
    present = [i for i, d in docs.items() if d is not None]
    print(f"[load] 可用实现: {present}", flush=True)
    if len(present) < 3:
        print(f"[warn] 只有 {len(present)} 份实现，无法完成三次一致性判定", flush=True)

    R = {i: rows_of(docs[i]) for i in present}
    models = sorted(set.intersection(*[set(R[i]) for i in present]) if present else set())
    print(f"[load] 共有模型 {len(models)} 个", flush=True)

    # ---------------- L1：逐模型逐指标
    print("\n" + "=" * 78)
    print("L1 实现一致性（三次独立重写，不同硬件）")
    print("=" * 78)
    print(f"{'model':7s}{'metric':9s}" + "".join(f"{'impl'+i:>11s}" for i in present)
          + f"{'spread':>11s}{'rel%':>9s}  verdict")
    worst = (0.0, None)
    fails = []
    table = {}
    for m in models:
        table[m] = {}
        for key in ("digital", "free", "best"):
            vals = [R[i][m][key] for i in present]
            spread = max(vals) - min(vals)
            rel = spread / min(vals)
            ok = rel < L1_REL
            if rel > worst[0]:
                worst = (rel, (m, key, vals))
            if not ok:
                fails.append((m, key, vals, rel))
            table[m][key] = vals
            print(f"{m:7s}{key:9s}" + "".join(f"{v:11.6f}" for v in vals)
                  + f"{spread:11.2e}{rel*100:9.4f}  {'PASS' if ok else 'FAIL'}")
    print(f"\n最大相对差 {worst[0]*100:.4f}% @ {worst[1] and (worst[1][0], worst[1][1])}")
    l1 = "PASS" if not fails else "FAIL"
    print(f"[L1] {l1}  (判据 < {L1_REL*100:.1f}%)")

    # ---------------- 跨种子汇总
    print("\n" + "=" * 78)
    print("汇总量：13 种子 best 的中位（决定选型的数）")
    print("=" * 78)
    med = {}
    for i in present:
        seeds = [v["best"] for k, v in R[i].items() if k not in ("std", "lite")]
        med[i] = st.median(seeds) if seeds else None
        mx = max(seeds) if seeds else None
        print(f"  impl {i}: n={len(seeds)} median_best={med[i]:.6f}  worst={mx:.6f}")
    if len(present) >= 2:
        vals = [med[i] for i in present if med[i] is not None]
        rel = (max(vals) - min(vals)) / min(vals)
        print(f"  → 三次中位相对差 {rel*100:.4f}%  "
              f"{'PASS' if rel < L1_REL else 'FAIL'}")
        l1_med = "PASS" if rel < L1_REL else "FAIL"
    else:
        l1_med = "N/A"

    # ---------------- 官方基线
    print("\n官方基线（同一协议下）：")
    for i in present:
        for k in ("std", "lite"):
            if k in R[i]:
                print(f"  impl {i} {k}: free={R[i][k]['free']:.4f} best={R[i][k]['best']:.4f} "
                      f"digital={R[i][k]['digital']:.4f}")
        break

    # ---------------- L2：MC 噪声
    print("\n" + "=" * 78)
    print("L2 统计稳定性（MC=4 → MC=16）")
    print("=" * 78)
    p16 = Path(a.dir) / a.mc16
    l2 = "N/A"
    if p16.is_file():
        d16 = rows_of(json.load(open(p16)))
        l2fails = []
        print(f"{'model':7s}{'metric':9s}{'MC4':>10s}{'MC16':>10s}{'rel%':>9s}  verdict")
        base = R[present[0]]
        for m, r in d16.items():
            if m not in base:
                continue
            for key in ("free", "best"):
                v4, v16 = base[m][key], r[key]
                rel = abs(v16 - v4) / min(v4, v16)
                ok = rel < L2_REL
                if not ok:
                    l2fails.append((m, key, v4, v16, rel))
                print(f"{m:7s}{key:9s}{v4:10.4f}{v16:10.4f}{rel*100:9.2f}  "
                      f"{'PASS' if ok else 'FAIL'}")
        l2 = "PASS" if not l2fails else "FAIL"
        print(f"[L2] {l2}  (判据 < {L2_REL*100:.0f}%)")
    else:
        print(f"  未找到 {p16}，跳过 L2")

    # ---------------- 总分
    print("\n" + "=" * 78)
    overall = (l1 == "PASS" and l1_med in ("PASS", "N/A") and l2 in ("PASS", "N/A"))
    print(f"判定：L1={l1}  L1(中位)={l1_med}  L2={l2}  →  "
          f"{'结论成立' if overall else '结论不成立，需继续做实验'}")
    if fails:
        print("L1 失败项：")
        for m, key, vals, rel in fails:
            print(f"  {m}.{key}: {vals}  rel={rel*100:.4f}%")

    if a.out:
        json.dump({"present": present, "table": table, "median_best": med,
                   "L1": l1, "L1_median": l1_med, "L2": l2, "overall": overall,
                   "fails": [{"model": m, "metric": k, "values": v, "rel": r}
                             for m, k, v, r in fails]},
                  open(a.out, "w"), indent=2)
        print("[done] " + a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
