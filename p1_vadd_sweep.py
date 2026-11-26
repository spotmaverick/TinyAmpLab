"""P-1b：head 加法器摆幅扫描 —— 找出让加法失真「可忽略」的摆幅门槛

## 逻辑
P-1 的主判据（加法失真占比）依赖于一个自由参数：加法器的饱和摆幅 V_add。
V_add 定得太低，加法器全程饱和，结论必然是「树形累加有价值」——
但那是摆幅没设计好，不是拓扑缺点的证据。
正确做法：扫 V_add，找出门槛 V*，使得 Δ_add < 1.05（失真<5%）。
  - 若 V* 与层内乘法器摆幅 V_SAT=16 处于同一量级 → 加法器可以和乘法器共用电源域，
    加法失真不是独立风险（因为 head 幅度本来就在 20 左右，需要同样大的摆幅）
  - 若 V* 需要比 16 高一个数量级 → 加法器必须独立高压域，是真实成本

## 输出
每个种子 × 每个 V_add → (serial NRMS, tree NRMS)
再算 Δ_add = NRMS(with add) / NRMS(no add)，找 Δ<1.05 的最小 V_add。

## 用法
  python p1_vadd_sweep.py --seeds s7 s31 --mc 2
  python p1_vadd_sweep.py --mc 4          # 全部 5 种子
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "github_mirrors" / "neural-amp-modeler"))

import analog_sim_generic as A                                   # noqa: E402
from analog_sim_generic import DevSim                           # noqa: E402
from p1_head_accum import (SEED_PATHS, resolve_nam, load_nam,   # noqa: E402
                           sim_forward_head, nrms_of, make_adder)

DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DATA = ROOT / "data"

# head_in 实测 RMS ≈ 17~22，p99 ≈ 55~76，|max| ≈ 81~98
# 门槛扫描覆盖「与 V_SAT 同量级」到「高 30 倍」
VADD_GRID = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", default=list(SEED_PATHS))
    ap.add_argument("--mc", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--out", default="p1_vadd_sweep.json")
    args = ap.parse_args()

    xi, _ = sf.read(str(DATA / "tsa15_test_input.wav"), dtype="float32")
    yi, _ = sf.read(str(DATA / "tsa15_test_target.wav"), dtype="float32")
    n = int(48000 * args.seconds)
    xt = torch.from_numpy(xi[:n]).to(DEV)
    yt = torch.from_numpy(yi[:n]).to(DEV)
    base = dict(A.P)

    all_rows = []
    print(f"[sweep] dev={DEV} mc={args.mc} sec={args.seconds} grid={VADD_GRID}", flush=True)

    for s in args.seeds:
        cfg, sd = load_nam(resolve_nam(s))
        rec = {"seed": s, "grid": []}

        # 基线：器件非理想，但 head 累加理想
        base_vals = {"serial": [], "tree": []}
        for i in range(args.mc):
            for topo in ("serial", "tree"):
                A.P.clear(); A.P.update(base)
                dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(1000 + i))
                with torch.no_grad():
                    pr = sim_forward_head(dev, sd, cfg, xt, topo, lambda v: v)
                base_vals[topo].append(nrms_of(pr, yt))
        A.P.clear(); A.P.update(base)
        b = {k: float(np.sqrt((np.array(v) ** 2).mean())) for k, v in base_vals.items()}
        rec["baseline"] = b

        for va in VADD_GRID:
            row = {"vadd": va}
            for topo in ("serial", "tree"):
                vals = []
                for i in range(args.mc):
                    A.P.clear(); A.P.update(base)
                    dev = DevSim(sd, torch.Generator(device=DEV).manual_seed(1000 + i))
                    gen_h = torch.Generator(device=DEV).manual_seed(5000 + i)
                    f_add = make_adder(dict(HEAD_VSAT=va, HEAD_BETA=0.002, HEAD_SIGMA_OFF=0.005),
                                       "on", gen_h)
                    with torch.no_grad():
                        pr = sim_forward_head(dev, sd, cfg, xt, topo, f_add)
                    vals.append(nrms_of(pr, yt))
                A.P.clear(); A.P.update(base)
                m = float(np.sqrt((np.array(vals) ** 2).mean()))
                row[f"{topo}_nrms"] = m
                row[f"{topo}_delta"] = m / max(b[topo], 1e-12)
            rec["grid"].append(row)
            print(f"  {s} V_add={va:4d}  ser Δ={row['serial_delta']:5.3f}  "
                  f"tree Δ={row['tree_delta']:5.3f}", flush=True)
        all_rows.append(rec)

    # 门槛：Δ_serial < 1.05 的最小 V_add
    print("\n=== 门槛汇总（Δ_serial 首次 <1.05 的 V_add）===")
    for r in all_rows:
        thr = None
        for row in r["grid"]:
            if row["serial_delta"] < 1.05:
                thr = row["vadd"]; break
        r["threshold_vadd"] = thr
        b = r["baseline"]["serial"]
        print(f"  {r['seed']:>4}  基线NRMS={b:.4f}  门槛V_add={thr}  "
              f"= {thr/16:.1f}×V_SAT" if thr else
              f"  {r['seed']:>4}  基线NRMS={b:.4f}  门槛V_add=未达到(>512)")

    json.dump(all_rows, open(ROOT / args.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
