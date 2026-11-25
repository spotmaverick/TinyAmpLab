#!/usr/bin/env python3
"""物理仿真任务调度器：并发度锁定 2，避免单卡过载。
每个作业独立日志 + 状态文件，支持本地轮询。
"""
import subprocess, time, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJ = "/workspace/TinyAmpLab"
OUT = f"{PROJ}/analog_sim"
os.makedirs(OUT, exist_ok=True)
STATUS = f"{OUT}/_supervisor_status.txt"

# 作业列表：标签 -> 命令
JOBS = [
    ("lite_full",       "src/analog/analog_sim_wavenet.py"),
    ("dw_ch12_full",    "src/analog/analog_sim_dw.py --variant dw_ch12"),
    ("dw_ch24_full",    "src/analog/analog_sim_dw.py --variant dw_ch24"),
    ("dw_ch12_rf50_full",  "src/analog/analog_sim_dw.py --variant dw_ch12_rf50"),
    ("dw_ch24_rf50_full",  "src/analog/analog_sim_dw.py --variant dw_ch24_rf50"),
    ("dw_ch12_vsat",    "src/analog/analog_sim_dw.py --variant dw_ch12 --vsat-scan"),
    ("dw_ch24_vsat",    "src/analog/analog_sim_dw.py --variant dw_ch24 --vsat-scan"),
]

MAX_PARALLEL = 2


def run_job(tag, cmd):
    t0 = time.time()
    log = f"{OUT}/log_{tag}.txt"
    with open(log, "w") as f:
        p = subprocess.Popen(f"cd {PROJ} && python3 -u {cmd}", shell=True,
                             stdout=f, stderr=subprocess.STDOUT)
        rc = p.wait()
    dt = time.time() - t0
    line = f"{tag} rc={rc} {dt:.1f}s"
    with open(STATUS, "a") as s:
        s.write(line + "\n")
    print(line, flush=True)
    return tag, rc, dt


def main():
    with open(STATUS, "w") as s:
        s.write(f"START {time.strftime('%F %T')} max_parallel={MAX_PARALLEL}\n")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        futs = {ex.submit(run_job, tag, cmd): tag for tag, cmd in JOBS}
        for f in as_completed(futs):
            results.append(f.result())
    with open(STATUS, "a") as s:
        s.write(f"ALL_DONE {time.strftime('%F %T')}\n")
    print("ALL_DONE")


if __name__ == "__main__":
    main()
