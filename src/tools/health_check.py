#!/usr/bin/env python
"""数据集健康检查：时长配对 + 电平 + 相关性参考。

用法: python health_check.py [--full]

判据说明（重要）:
  - 真正的配对错误判据是 **input 与 target 时长不匹配**（>2s 差异）。
    曾出现 Blackstar-HT1: DRY 152.5s 而 target 340s，互相关峰值仅 0.06，
    模型学不动（val corr≈0.01, NRMS≈1.0）。
  - raw corr **不能**作为判据：高增益/强失真箱天然低相关，
    如 Marshall (corr≈0.05) 却训练出 val NRMS 0.074 的最优结果。
    低 corr 只作提示，需结合训练日志判断。
"""
import sys
import glob
import os

import soundfile as sf
import numpy as np

DATA = "data"
SPLITS = ("train", "val", "test")
LOW_CORR = 0.05
DUR_TOL = 2.0  # 秒


def tags():
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, "*_train_input.wav"))):
        out.append(os.path.basename(f)[: -len("_train_input.wav")])
    return out


def main():
    full = "--full" in sys.argv
    bad = []
    hdr = f"{'dataset':30s} {'split':6s} {'corr':>8s} {'in_s':>7s} {'out_s':>7s}  verdict"
    print(hdr)
    print("-" * len(hdr))
    for t in tags():
        for sp in SPLITS:
            xi = os.path.join(DATA, f"{t}_{sp}_input.wav")
            yt = os.path.join(DATA, f"{t}_{sp}_target.wav")
            if not (os.path.exists(xi) and os.path.exists(yt)):
                print(f"{t:30s} {sp:6s} {'-':>8s} {'-':>7s} {'-':>7s}  MISSING")
                bad.append((t, sp, "missing file"))
                continue
            x, _ = sf.read(xi)
            y, _ = sf.read(yt)
            n = min(len(x), len(y))
            c = float(np.corrcoef(x[:n], y[:n])[0, 1]) if n > 1 else 0.0
            dx, dy = len(x) / 48000.0, len(y) / 48000.0
            if abs(dx - dy) >= DUR_TOL:
                v = f"DURATION MISMATCH"
                bad.append((t, sp, f"{dx:.0f}s vs {dy:.0f}s"))
            elif abs(c) < LOW_CORR:
                v = "low corr (见说明,可能正常)"
            else:
                v = "OK"
            print(f"{t:30s} {sp:6s} {c:+8.4f} {dx:7.1f} {dy:7.1f}  {v}")
            if not full:
                break
    print()
    if bad:
        print(f"!! {len(bad)} 处确认为配对错误:")
        for t, sp, why in bad:
            print(f"   {t} [{sp}] {why}")
        sys.exit(1)
    print("时长配对全部通过")


if __name__ == "__main__":
    main()
