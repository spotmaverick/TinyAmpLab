"""把 results_v2.jsonl 汇总成排行榜（区分 Full / Lite 分支）"""
import json
import sys
from pathlib import Path

rows = []
for line in Path(sys.argv[1] if len(sys.argv) > 1 else "results_v2.jsonl").read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "error" in d or "nrms" not in d:
        continue
    tag = d.get("tag") or Path(d.get("ckpt", "?")).parts[-5]
    # 从 ckpt 路径推 run 名
    parts = Path(d.get("ckpt", "")).parts
    run = parts[1] if len(parts) > 1 else tag
    slim = d.get("slim")
    branch = "Full" if slim in (1.0, None) else "Lite"
    rows.append(dict(run=run, branch=branch, nrms=d["nrms"], corr=d.get("corr", 0),
                     params=d.get("params", 0), rf=d.get("rf", 0)))

rows.sort(key=lambda r: r["nrms"])
print(f"{'run':38s} {'branch':6s} {'NRMS':>8s} {'corr':>7s} {'params':>8s} {'RF':>6s}")
print("-" * 80)
for r in rows:
    print(f"{r['run']:38s} {r['branch']:6s} {r['nrms']:8.4f} {r['corr']:7.4f} "
          f"{r['params']:8d} {r['rf']:6d}")
