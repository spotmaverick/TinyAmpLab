"""P5-C：head 增益重平衡 —— 零训练成本、数字函数精确不变

## 动机（来自 P5-B 的实测归因）

P5-B 逐层旁路显示，目标拓扑的退化**集中在 head_rechannel 这一级**：

    seed s71   A0.head relief = +0.0581   （总 Gilbert 贡献 0.0785 的 74%）
    seed s7    A0.head relief = +0.0073   （总量 0.0456 的 16%）
    official   A1.head relief = +0.0049   （总量 0.0049 的 100%，但绝对值很小）

机理：head_input 是 10 个 term 的**相干和**（P-1 实测 RMS 17.5~22.9、p99 55~79），
进入 head_rechannel 的 Gilbert cell 时远超 V_SAT=16，被 tanh 狠狠削顶。

## 关键观察：这个削顶可以免费消掉

WaveNet 的输出是 `head_scale * head_rechannel(head_input)`。而 `head_scale` 是
**Gilbert 之后的一个自由标量**（电路上就是输出放大器的增益／数字乘一个常数），
不参与任何非线性。

因此对任意 k > 0 做变换：

    head_rechannel.weight  ← weight / k
    head_rechannel.bias    ← bias   / k
    head_scale             ← head_scale * k

**数字函数逐位不变**（线性等价），但送进 Gilbert 的乘积缩小 k 倍，
饱和深度大幅降低；削顶后的结果再被 head_scale 放大回原量级。

这是一个**硬件设计选择**（把 head 求和节点的增益压低、后级再放大），
不是重新训练，不需要改权重数值以外的任何东西。

## 与 residual_scale(α) 的区别

α 作用在残差流上，而残差是 `x + layer_output` 的**串联结构**，
缩放 α 会改变整个网络的函数 → 必须重训（P2 已验证）。
head 的 head_scale 是**末端串联的自由标量**，不存在这个问题。

## 判据

1. `--k 1` 必须与基线逐位一致（自检）。
2. 数字 NRMS 对所有 k 必须**完全相同**（差 < 1e-6）——否则说明变换不精确。
3. device NRMS 随 k 单调下降并趋于「head 全旁路」的极限。

## 用法

    $PY p5c_head_rebalance.py --nam models/seeds/s71.nam --mc 4 --ks 1,2,4,8,16
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_analog_sim as O                                  # noqa: E402
from official_analog_sim import DevSim, forward, esr_nrms, parse_arrays  # noqa: E402

DEVICE = O.DEVICE
SEGMENTS = [("seg0", 0.0), ("seg1", 8.0), ("seg2", 16.0)]


def rebalance(sd, nam, k):
    """把**最后一个** layer array 的 head_rechannel weight/bias 除以 k，head_scale 乘 k。

    为什么只动最后一个 array（多 array 模型必读）：
        官方 forward 里 `head_input, y = layer_array(y, c, head_input=head_input)`，
        而 array 返回的第一项是 `head_rechannel(head_input)`。也就是说：

           array0 的 head_rechannel 输出（如 8 通道）会作为 array1 的 head_input 初值，
           再与 array1 自己的 head term **相加**，最后经 array1 的 head_rechannel(8→1)。

        若把 array0 的 head_rechannel 也除以 k，它输出的 8 通道幅度变小，
        与 array1 未缩放的 head term 相加时**相对权重就变了** —— 函数被改写。
        实测：standard 全缩放时 digital spread = 7.79e-02（自检 FAIL），正是这个原因。

        只有最后一个 array 的 head_rechannel 输出直接进入 `head_scale *`，
        它是**串在末端的线性算子**，除以 k 再用 head_scale 乘 k 补偿，数学上逐位等价。
    """
    arrays = parse_arrays(nam["config"])
    last = len(arrays) - 1
    sd2 = {kk: vv.clone() if torch.is_tensor(vv) else vv for kk, vv in sd.items()}
    for suf in ("weight", "bias"):
        key = f"_layer_arrays.{last}._head_rechannel.{suf}"
        if key in sd2:
            sd2[key] = sd2[key] / k
    nam2 = json.loads(json.dumps(nam))
    nam2["config"]["head_scale"] = float(nam2["config"].get("head_scale", 1.0)) * k
    return sd2, nam2


def eval_k(sd, nam, seg_data, mc):
    vals, digs = [], []
    for _nm, xt, yt in seg_data:
        dv_list, dg_list = [], []
        for i in range(mc):
            dg = DevSim(sd, torch.Generator(device=DEVICE).manual_seed(1000 + i), sanity=True)
            with torch.no_grad():
                pr = forward(dg, sd, nam, xt)
            dg_list.append(esr_nrms(pr, yt)[1])

            dv = DevSim(sd, torch.Generator(device=DEVICE).manual_seed(1000 + i))
            with torch.no_grad():
                pr = forward(dv, sd, nam, xt)
            dv_list.append(esr_nrms(pr, yt)[1])
        vals.append(float(np.sqrt(np.mean(np.square(dv_list)))))
        digs.append(float(np.sqrt(np.mean(np.square(dg_list)))))
    return float(np.median(vals)), float(np.median(digs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nam", required=True)
    ap.add_argument("--mc", type=int, default=4)
    ap.add_argument("--ks", default="1,2,4,8,16,32")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    nam, sd = O.load_model(args.nam) if hasattr(O, "load_model") else (None, None)
    if nam is None:
        nam = json.load(open(args.nam))
        sys.path.insert(0, __import__("os").environ.get("NAM_SRC", "/public/home/spotmaverick/nam_proj"))
        from nam.models import init_from_nam
        net = init_from_nam(nam)
        sd = {k: v for k, v in net.state_dict().items()
              if "_has_sample" not in k and "_sample_rate" not in k}
        sd = {k[5:] if k.startswith("_net.") else k: v for k, v in sd.items()}

    xi_all, sr = O.sf.read(str(O.DATA / "tsa15_test_input.wav"), dtype="float32")
    yi_all, _ = O.sf.read(str(O.DATA / "tsa15_test_target.wav"), dtype="float32")
    seg_data = [(nm, *O.load_seg(t0, 5.0, xi_all, yi_all)) for nm, t0 in SEGMENTS]

    # sanity
    xt = seg_data[0][1]
    yt = seg_data[0][2]
    d0 = DevSim(sd, torch.Generator(device=DEVICE).manual_seed(0), sanity=True)
    with torch.no_grad():
        pi = forward(d0, sd, nam, xt)
    if pi.ndim == 2:
        pi = pi[0]
    oo, _ = O.official_path(args.nam, xt, yt)
    n = min(pi.shape[0], oo.shape[0])
    maxd = float((pi[-n:] - oo[-n:]).abs().max())
    print(f"[sanity] max|Δ|={maxd:.3e}", flush=True)
    if maxd > 1e-5:
        print("[sanity] FAIL", flush=True)
        return 2

    res = {"nam": args.nam, "mc": args.mc, "rows": []}
    base_dev = None
    for k in [float(x) for x in args.ks.split(",")]:
        sd2, nam2 = rebalance(sd, nam, k)
        dev, dig = eval_k(sd2, nam2, seg_data, args.mc)
        if base_dev is None:
            base_dev = dev
        res["rows"].append({"k": k, "device_median": dev, "digital_median": dig,
                            "gain_vs_k1": base_dev - dev})
        print(f"  [k={k:>5.0f}] device={dev:.4f}  digital={dig:.4f}  Δ={base_dev-dev:+.4f}",
              flush=True)

    # 自检：所有 k 的数字精度必须一致
    digs = [r["digital_median"] for r in res["rows"]]
    spread = max(digs) - min(digs)
    res["digital_invariance_spread"] = spread
    print(f"[check] digital spread across k = {spread:.2e} "
          f"({'PASS' if spread < 1e-5 else 'FAIL — 变换不精确'})", flush=True)

    out = Path(args.out) if args.out else Path("p5c_" + Path(args.nam).stem + ".json")
    json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
    print("[done] " + str(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
