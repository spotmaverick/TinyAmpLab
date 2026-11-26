#!/usr/bin/env python3
"""夜间批次监控：汇总 z100 / bw2 上所有作业状态与训练指标。

用法:
    python night_watch.py            # 单次快照
    python night_watch.py --loop 600 # 每 600 秒刷新一次
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

DCU = os.path.expanduser("~/.workbuddy/skills/sugon-dcu-fleet/scripts/dcu.sh")
Z100_ROOT = "/work/home/spotmaverick/nam_project"


def run(remote, cmd, timeout=120):
    """在远端机器执行命令，返回 stdout。"""
    full = [DCU, "-m", remote, "ssh", cmd]
    try:
        p = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return p.stdout
    except subprocess.TimeoutExpired:
        return ""


def z100_snapshot():
    cmd = (
        f"cd {Z100_ROOT} && "
        "echo '@@QUEUE@@'; squeue -u spotmaverick -h -o '%T' | sort | uniq -c; "
        "echo '@@LOGS@@'; "
        "for f in arch_v2/lite_*_s7.log arch_v2/standard_*_s7.log "
        "arch_v2/feather_*_s7.log arch_v2/nano_*_s7.log; do "
        "[ -f \"$f\" ] || continue; "
        "b=$(basename $f .log); "
        # 最后一条 ep 记录的 epoch 序号（不是行数：每 20ep 才打一行）
        "ep=$(grep '^ep' $f | tail -1 | awk '{print $2}'); "
        # 日志末行是 RESULT JSON，val_NRMS 只在 ep 行上；取最后一条 ep 行
        "v=$(grep '^ep' $f | tail -1 | grep -o 'val_NRMS [0-9.]*' | cut -d' ' -f2); "
        "echo \"$b|$ep|$v\"; "
        "done"
    )
    return run("z100", cmd, timeout=180)


def parse_z100(out):
    queue, logs = {}, []
    section = None
    for line in out.splitlines():
        line = line.strip()
        if line == "@@QUEUE@@":
            section = "q"
            continue
        if line == "@@LOGS@@":
            section = "l"
            continue
        if section == "q" and line:
            parts = line.split()
            if len(parts) >= 2:
                queue[parts[1]] = int(parts[0])
        elif section == "l" and "|" in line:
            b, ep, v = line.split("|", 2)
            try:
                epn = int(ep)
            except ValueError:
                epn = -1
            logs.append((b, epn, v or "-"))
    return queue, logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0)
    args = ap.parse_args()

    while True:
        t0 = time.strftime("%H:%M:%S")
        print(f"\n{'=' * 78}\n 夜间批次快照  {t0}\n{'=' * 78}")

        out = z100_snapshot()
        queue, logs = parse_z100(out)

        if queue:
            total = sum(queue.values())
            print(f"\n[z100] 作业总数 {total}: " +
                  ", ".join(f"{k}={v}" for k, v in sorted(queue.items())))
        else:
            print("\n[z100] 无作业或连接失败")

        if logs:
            print(f"\n{'模型':<44s} {'ep':>4s} {'val_NRMS':>9s}")
            print("-" * 60)
            done = []
            for b, ep, v in sorted(logs):
                mark = ""
                if v != "-":
                    mark = "  <- 已收敛" if ep >= 280 else ""
                print(f"{b:<44s} {ep:>4d} {v:>9s}{mark}")
                if v != "-":
                    try:
                        if float(v) < 0.09:
                            done.append((b, float(v)))
                    except ValueError:
                        pass
            if done:
                print(f"\n已达 NRMS<0.09 的模型 ({len(done)} 个):")
                for b, v in sorted(done, key=lambda x: x[1]):
                    print(f"   {b:<44s} {v:.4f}")

        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
