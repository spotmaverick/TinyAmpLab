"""同容量公平对比：A2 Lite (3ch) vs 族 A 3ch

Round1 的问题在于拿 12 通道的族 A 比"只训练了 3 通道的 A2"。
正确做法是在**同等有效容量**下比：
  - A2 Lite：allowed_channels=[3,12]，裁到 3 通道，1090 参数
  - a3ch：族 A 3 通道，1087 参数

用法：python cmp_same_capacity.py
"""
import glob
import json
import subprocess
import sys
from pathlib import Path


def last_ckpt(pattern):
    files = sorted(
        glob.glob(pattern, recursive=True),
        key=lambda p: int(p.split("epoch=")[1].split("_")[0]) if "epoch=" in p else -1,
    )
    return files[-1] if files else None


def eval_ckpt(ckpt, slim):
    r = subprocess.run(
        [sys.executable, "eval_v2.py", ckpt, "--slim", str(slim)],
        capture_output=True, text=True, cwd="/root/autodl-tmp",
    )
    line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    try:
        return json.loads(line)
    except Exception:
        return {"error": line or r.stderr[-200:]}


def main():
    print(f"{'model':<28s} {'branch':>6s} {'params':>7s} {'NRMS':>8s} {'corr':>7s}")
    print("-" * 62)

    # A2 Lite：需要一个用 smallest_and_zeros 或 causal 训的 slimmable 模型
    a2_cands = [
        ("A2-causal(slim=0)", "runs/n4_a2causal-300_s7/**/epoch=*.ckpt", 0.0),
        ("A2-causal(slim=1)", "runs/n4_a2causal-300_s7/**/epoch=*.ckpt", 1.0),
        ("A2-orig100(slim=0)", "runs/cand_a2/**/epoch=0099*.ckpt", 0.0),
    ]
    rows = []
    for name, pat, slim in a2_cands:
        ck = last_ckpt(pat)
        if not ck:
            print(f"{name:<28s} {'—':>6s} {'—':>7s} {'no ckpt':>8s}")
            continue
        r = eval_ckpt(ck, slim)
        if "nrms" in r:
            rows.append((name, "Lite" if slim == 0 else "Full", r["params"], r["nrms"], r["corr"]))
            print(f"{name:<28s} {'Lite' if slim==0 else 'Full':>6s} "
                  f"{r['params']:>7d} {r['nrms']:>8.4f} {r['corr']:>7.4f}")
        else:
            print(f"{name:<28s} err: {r.get('error','?')[:40]}")

    # 族 A
    for name, pat in [
        ("a3ch-300 (族A 3ch)", "runs/n4_a3ch-300_s7/**/epoch=*.ckpt"),
    ]:
        ck = last_ckpt(pat)
        if not ck:
            print(f"{name:<28s} no ckpt")
            continue
        r = eval_ckpt(ck, 1.0)
        if "nrms" in r:
            rows.append((name, "3ch", r["params"], r["nrms"], r["corr"]))
            print(f"{name:<28s} {'3ch':>6s} {r['params']:>7d} {r['nrms']:>8.4f} {r['corr']:>7.4f}")

    print()
    lite = [r for r in rows if r[1] == "Lite"]
    three = [r for r in rows if r[1] == "3ch"]
    if lite and three:
        a, b = lite[0][3], three[0][3]
        print(f"同容量（~1090 参数）对比：A2 Lite={a:.4f}  vs  族A 3ch={b:.4f}")
        print(f"=> 族A 相对改善 {(a-b)/a*100:+.1f}%")


if __name__ == "__main__":
    main()
