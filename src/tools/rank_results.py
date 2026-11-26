"""汇总所有评测结果，输出排行榜（在 4090 上跑）"""
import glob
import json
import os

REF = 0.1786  # A2 官方基线 test NRMS

rows = []
for tf in glob.glob("/root/autodl-tmp/results/*_test.json"):
    base = tf[: -len("_test.json")]
    name = os.path.basename(base).replace("cand_", "")
    try:
        t = json.load(open(tf))
    except Exception:
        continue
    if not t.get("nrms"):
        continue
    v = None
    vp = base + "_val.json"
    if os.path.exists(vp):
        try:
            v = json.load(open(vp)).get("nrms")
        except Exception:
            pass
    rows.append(dict(name=name, val=v, test=t["nrms"], corr=t.get("corr"),
                     rf=t.get("rf")))

rows.sort(key=lambda r: r["test"])

print(f"{'候选':<22}{'val':>8}{'test':>8}{'corr':>8}{'RF':>7}{'vs A2':>9}")
print("-" * 64)
for r in rows:
    v = f"{r['val']:.4f}" if r["val"] else "  —   "
    delta = (r["test"] - REF) / REF * 100
    tag = "基准" if abs(delta) < 1 else f"{delta:+.1f}%"
    print(f"{r['name']:<22}{v:>8}{r['test']:>8.4f}"
          f"{(r['corr'] or 0):>8.4f}{(r['rf'] or 0):>7}{tag:>9}")
print(f"\nA2 官方基线 test NRMS = {REF}")
print(f"共 {len(rows)} 个候选")
