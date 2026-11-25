#!/usr/bin/env python3
"""实验产物回收：从智算机器增量同步权重/日志/JSON 到 MacBook。

每次实验结束后调用，或在本地定期运行。只做增量复制，不删远端文件。
用法:
  python3 sync_back.py bw        # 回收 BW 产物
  python3 sync_back.py z100      # 回收 8×Z100 产物
  python3 sync_back.py all       # 两边都回收（默认）
"""
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]

# 远端产物目录 -> 本地目标目录
SPEC = {
    "bw": {
        "host": "ssh.zzai.scnet.cn",
        "port": 58222,
        "user": "spotmaverick.2125517769658642433",
        "remote_root": "/workspace/TinyAmpLab",
        "socket": "/tmp/bw_ctrl.sock",
        "pairs": [
            ("arch_v2", "arch_v2_local"),
            ("arch_analog", "arch_analog_local"),
            ("runs_bw", "runs_bw"),
            ("analog_sim", "experiments/analog_sim_bw"),
        ],
    },
    "z100": {
        "host": "wuzh02.hpccube.com",
        "port": 65091,
        "user": "spotmaverick",
        "key": str(Path.home() / ".ssh/wuzh02_spotmaverick.pem"),
        "remote_root": "/work/home/spotmaverick/nam_project",
        "pairs": [
            ("arch_v2", "experiments/z100_arch_v2"),
            ("arch_analog", "experiments/z100_arch_analog"),
            ("analog_sim", "experiments/z100_analog_sim"),
        ],
    },
}


def ssh_opts(m):
    """返回 rsync 用的 -e 参数。BW 用复用通道，Z100 用密钥。"""
    if "socket" in m:
        return f"ssh -S {m['socket']} -o BatchMode=yes -o ConnectTimeout=10"
    return (f"ssh -i {m['key']} -o BatchMode=yes -o ConnectTimeout=10 "
            f"-p {m['port']}")


def remote_host(m):
    return f"{m['user']}@{m['host']}"


def sync_one(machine, m, rdir, ldir):
    local = ROOT / ldir
    local.mkdir(parents=True, exist_ok=True)
    src = f"{remote_host(m)}:{m['remote_root']}/{rdir}/"
    cmd = ["rsync", "-az", "--stats", "-e", ssh_opts(m), src, str(local) + "/"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        return False, f"rsync失败 rc={p.returncode}: {p.stderr.strip()[:200]}"
    # 统计本次传了多少文件（Mac rsync 2.6.9 的 --stats 字段格式不同，取不到就标 ok）
    n = None
    for line in p.stdout.splitlines():
        if "Number of regular files transferred" in line:
            n = line.split(":")[-1].strip()
            break
    return True, f"{n + ' 文件' if n else '已同步'}"


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    machines = ["bw", "z100"] if which == "all" else which.split(",")
    machines = [m.strip() for m in machines]
    if any(m not in SPEC for m in machines):
        print(f"未知机器: {machines}，可选 bw/z100/all")
        sys.exit(1)

    ts = datetime.now().strftime("%F %T")
    print(f"=== 实验产物回收 {ts} ===")
    ok_all = True
    for name in machines:
        m = SPEC[name]
        print(f"\n[{name}] {m['user']}@{m['host']}")
        for rdir, ldir in m["pairs"]:
            ok, msg = sync_one(name, m, rdir, ldir)
            print(f"  {rdir:16s} -> {ldir:30s} {msg}")
            if not ok:
                ok_all = False

    # 生成本地产物清单（便于追溯版本）
    manifest = ROOT / "experiments" / "SYNC_MANIFEST.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "a") as f:
        f.write(f"\n=== {ts} {'/'.join(machines)} ===\n")
        for name in machines:
            for rdir, ldir in SPEC[name]["pairs"]:
                d = ROOT / ldir
                if d.exists():
                    files = sorted(d.rglob("*"))
                    nfile = sum(1 for p in files if p.is_file())
                    f.write(f"  {name}/{rdir:16s} -> {ldir:30s} {nfile} files\n")
    print(f"\n清单已追加: {manifest.relative_to(ROOT)}")
    if not ok_all:
        print("⚠ 部分回收失败，请检查连接后重试")
        sys.exit(2)
    print("回收完成 ✓")


if __name__ == "__main__":
    main()
