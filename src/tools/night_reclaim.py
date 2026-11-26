#!/usr/bin/env python3
"""夜间批次产物自动回收 + 结果汇总。

从 z100 (可选 bw2) 拉回 arch_v2 下新产的 JSON/日志/权重，
汇总成一张跨管箱对比表存到 experiments/z100_night1/。

用法:
    python night_reclaim.py            # 回收 + 汇总
    python night_reclaim.py --dry      # 只看会做什么
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DCU = os.path.expanduser("~/.workbuddy/skills/sugon-dcu-fleet/scripts/dcu.sh")
LOCAL = Path(__file__).resolve().parents[2]
OUT = LOCAL / "experiments" / "z100_night1"
REMOTE = {
    "z100": "/work/home/spotmaverick/nam_project/arch_v2",
    "bw2": "/public/home/spotmaverick/nam_project/arch_v2",
}


def dcu(machine, *args, timeout=600):
    cmd = [DCU, "-m", machine] + list(args)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def remote_list(machine):
    rc, out = dcu(machine, "ssh",
                  f"cd {REMOTE[machine]} 2>/dev/null && ls -1 *.json 2>/dev/null")
    if rc != 0:
        return []
    return [x.strip() for x in out.splitlines() if x.strip().endswith(".json")]


def pull(machine, names, dest):
    """逐个拉取；批量失败则回退到 tar 打包。"""
    if not names:
        return []
    dest.mkdir(parents=True, exist_ok=True)
    got = []
    # 打包后一次拉，比逐个 scp 快
    files = " ".join(names)
    rc, out = dcu(machine, "ssh",
                  f"cd {REMOTE[machine]} && tar czf /tmp/_reclaim.tar.gz {files}")
    if rc != 0:
        print(f"  [!] 打包失败: {out[-300:]}")
        return []
    rc, out = dcu(machine, "pull", "/tmp/_reclaim.tar.gz", str(dest))
    if rc != 0:
        print(f"  [!] 拉取失败: {out[-300:]}")
        return []
    tar = dest / "_reclaim.tar.gz"
    if tar.exists():
        subprocess.run(["tar", "xzf", str(tar), "-C", str(dest)], check=False)
        tar.unlink()
        got = names
    dcu(machine, "ssh", "rm -f /tmp/_reclaim.tar.gz")
    return got


def summarize(dest):
    rows = []
    for f in sorted(dest.glob("*.json")):
        if f.name.startswith("vsat_scan"):
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        rfp = d.get("rf_scale_pct", 100)
        arch = d.get("variant") or d.get("arch")
        # rf 缩放变体单独成一个"架构"列，避免和标准配置混在同一个单元格
        # 注意 variant 本身可能已带 rf 后缀（如 "standard_rf50"），不要重复拼接
        if rfp in (None, 100) or str(arch).endswith(f"rf{rfp}"):
            label = arch
        else:
            label = f"{arch}_rf{rfp}"
        rows.append({
            "stem": f.stem,
            "variant": label,
            "params": d.get("params"),
            "dataset": d.get("dataset", "tsa15"),
            "epochs": d.get("epochs"),
            "val": d.get("best_val_NRMS"),
            "test": d.get("test_NRMS"),
            "test_esr": d.get("test_esr"),
            "sec": d.get("train_seconds"),
        })
    return rows


def write_report(rows, path):
    lines = ["# 夜间批次（多管箱泛化）结果汇总", "",
             f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}", "",
             "**口径**：val NRMS 取 JSON 的 `best_val_NRMS`（300 ep 内最优）；"
             "test NRMS 为留出测试段。数值越小越好。", "",
             "**来源**：z100 凌晨批次（34 个） + bw 补跑（5 个，2026-11-26 09:26）。"
             "`训练秒` 一列跨机不可比：z100 为 18 作业挤 7 节点的实测（28~46 s/ep），"
             "bw 为单卡 5 路并发（2.6~3.2 s/ep），同素材 bw 约快 10 倍。", "",
             "## 一、全量结果", "",
             "| 模型 | 管箱 | 参数 | ep | val NRMS | test NRMS | test ESR | 训练秒 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda x: (x["val"] if x["val"] is not None else 9)):
        val = f"{r['val']:.4f}" if r["val"] is not None else "-"
        tst = f"{r['test']:.4f}" if r["test"] is not None else "-"
        esr = f"{r['test_esr']:.4f}" if r["test_esr"] is not None else "-"
        lines.append(
            f"| {r['stem']} | {r['dataset']} | {r['params']} | {r['epochs']} | "
            f"{val} | {tst} | {esr} | {r['sec']} |")

    # 管箱 × 架构 的 pivoted 矩阵，直接看跨箱泛化
    # 只铺主配置（rf=100）的四个规模档；rf 缩放变体单独列在后面
    order = ["nano", "feather", "lite", "standard"]
    base_rows = [r for r in rows if r["variant"] in order]
    archs = [a for a in order if any(r["variant"] == a for r in base_rows)]

    cells = {}
    for r in base_rows:
        if r["val"] is not None:
            cells.setdefault(r["dataset"], {})[r["variant"]] = r

    lines += ["", "## 二、管箱 × 规模矩阵（val NRMS，rf=100 主配置）", "",
              "| 管箱 | " + " | ".join(archs) + " |",
              "|---" * (len(archs) + 1) + "|"]
    for ds in sorted(cells, key=lambda d: min(
            v["val"] for v in cells[d].values())):
        cells_txt = []
        for a in archs:
            r = cells[ds].get(a)
            cells_txt.append(f"{r['val']:.4f}" if r else "—")
        lines.append(f"| {ds} | " + " | ".join(cells_txt) + " |")

    rf_rows = [r for r in rows if r["variant"] not in order and r["val"] is not None]
    if rf_rows:
        lines += ["", "## 三、RF 缩放变体（非主配置，仅供参考）", ""]
        for r in sorted(rf_rows, key=lambda x: x["val"]):
            lines.append(f"- {r['dataset']:<24s} {r['variant']:<14s} "
                         f"{r['val']:.4f}")

    lines += ["", "## 四、按 val NRMS 排序（越小越好）", ""]
    for r in sorted(rows, key=lambda x: (x["val"] if x["val"] is not None else 9)):
        if r["val"] is not None:
            lines.append(f"- {r['dataset']:<30s} {r['stem']:<44s} {r['val']:.4f}")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--machines", default="z100")
    args = ap.parse_args()

    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    OUT.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for m in machines:
        names = remote_list(m)
        print(f"[{m}] 远端 JSON {len(names)} 个")
        if args.dry:
            for n in names:
                print(f"   would pull {n}")
            continue
        dest = OUT / m
        got = pull(m, names, dest)
        print(f"[{m}] 已回收 {len(got)} 个")
        all_rows += summarize(dest)

    if args.dry:
        return

    if all_rows:
        rep = OUT / "SUMMARY.md"
        write_report(all_rows, rep)
        print(f"\n汇总已写 {rep}")
        print(f"共 {len(all_rows)} 条结果")
        # 打印前几名
        ranked = [r for r in all_rows if r["val"] is not None]
        ranked.sort(key=lambda x: x["val"])
        print("\nval NRMS 最好的 8 个:")
        for r in ranked[:8]:
            print(f"   {r['dataset']:<30s} {r['stem']:<44s} {r['val']:.4f}")


if __name__ == "__main__":
    main()
