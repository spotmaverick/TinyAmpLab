"""官方 NAM 模型的器件级（Gilbert cell）仿真：与目标拓扑同口径的公平对比

## 为什么需要这个脚本

项目里 `analog_sim_generic.py` 只支持**单 layer array** 的自研 `config_model.json`。
官方 NAM（lite/standard/feather/nano）是**两层 layer array 级联**、且 `head_input` 跨
array 累加，结构不同，无法直接套用。没有这个适配层，就只能拿"官方模型的数字精度"
去比"我方模型的器件精度"，口径错配。

本脚本严格复刻官方 `nam/models/wavenet/_wavenet.py:forward` 与
`_layer_array.py:forward` 的算子顺序，只把乘法器换成 Gilbert cell、权重换成
6-bit 量化存储、激活换成带失调的差分对——即 `analog_sim_generic.DevSim` 的同一套
非理想参数（DIALECT），保证两者唯一的差别是**拓扑/权重**，不是仿真器。

## 结构对齐（逐项核对过官方源码）

    WaveNet.forward:
      y, head_input = x, None
      for layer_array in self._layer_arrays:
          head_input, y = layer_array(y, c, head_input=head_input)   # head 跨 array 累加
      head_input = head_scale * head_input
      return head_input if head is None else head(head_input)

    LayerArray.forward:
      out_length            = min(len(x), len(c)) - (RF_no_head_rechannel - 1)
      out_length_no_head    = min(len(x), len(c)) - (RF_no_head_rechannel - 1)   # 给 layer 裁 term
      x = rechannel(x)                       # 1x1，模拟路径也要算乘法器
      for layer:
          zconv = conv(x)                    # 因果 SAME（左 pad K-1 倍 dilation）
          mix   = input_mixer(c)[-len(zconv):]
          z1    = zconv + mix
          post  = activation(z1)
          layer_output = layer1x1(post)      # 若 active
          head_term    = head1x1(post)[-out_length:] if head1x1 else post[-out_length:]
          residual     = x[-len(layer_output):] + layer_output
      head_rechannel_output = head_rechannel(head_input)   # 1x1+可带 bias
      return head_rechannel_output, x[-out_length:]

注意 `head_input` 在 array 内是 **head_term 的跨层和**（Python 里 `head_input` 是
list，最后 `torch.cat(...).sum(dim=1)` —— 见 `_layer_array.py` 1069-1080 行的实际写法，
本脚本按"逐层累加、长度对齐 out_length_no_head"实现，与官方等价）。

## 非理想性（与 analog_sim_generic 完全一致的 DIALECT）

| 算子 | 电路 | 非理想项 |
|---|---|---|
| 权重存储 | 6-bit 浮栅 | 对称量化 + 高斯失配 σ / 全局温漂 |
| 乘法 | Gilbert cell | 三次项 β·x̂² + tanh 饱和 V_SAT |
| 求和 | KCL | 理想无源 |
| 激活 | 差分对 | 输入失调 σ_off + tanh |
| 节点 | 热噪声 SNR |
| 延迟线 | 数字 SRAM | 无损 |

## 用法（在计算节点，bw2/z100）

    source ~/p26/bw2env.sh
    export PYTHONPATH=/public/home/spotmaverick/nam_proj:/public/home/spotmaverick/t29:$PYTHONPATH
    $PY official_analog_sim.py --nam <model.nam> --mc 4 [--segs 0,8,16] [--seg-len 5]
    $PY official_analog_sim.py --sanity --nam <model.nam>     # 关非理想，与官方 PyTorch 对拍

判据：`--sanity` 必须给出 `max|Δ| < 1e-5`（官方前向 vs 本仿真理想路径），
否则说明结构复刻错了，此时**任何 NRMS 都不能用**。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA = Path(os.environ.get("OFFICIAL_SIM_DATA", "/public/home/spotmaverick/nam_proj/data"))

# 与 analog_sim_generic.P 完全一致，保证口径可比
DIALECT = dict(
    BITS=6,
    SIGMA_MISMATCH=0.01,
    DRIFT=0.03,
    BETA_H3=0.002,
    V_SAT=16.0,
    SIGMA_OFFSET=0.005,
    SNR_DB=70.0,
)

SEGMENTS = [("seg0", 0.0), ("seg1", 8.0), ("seg2", 16.0)]
SEG_LEN = 5.0
SR = 48000


def quant_sym(w, bits):
    """逐输出通道对称量化（与 analog_sim_generic.quant_sym 一致）。"""
    if w.dim() == 1:
        return w
    levels = 2 ** (bits - 1) - 1
    dims = tuple(range(1, w.dim()))
    scale = w.abs().amax(dim=dims, keepdim=True).clamp(min=1e-12) / levels
    return (w / scale).round().clamp(-(levels + 1), levels) * scale


class DevSim:
    """一次蒙特卡洛采样的器件实例（与 analog_sim_generic.DevSim 同实现）。"""

    def __init__(self, sd, gen, sanity=False):
        self.g = gen
        self.sanity = sanity
        p = dict(DIALECT)
        if sanity:
            p.update(SIGMA_MISMATCH=0.0, DRIFT=0.0, BETA_H3=0.0,
                     SIGMA_OFFSET=0.0, SNR_DB=999, V_SAT=1e9, BITS=64)
        self.p = p
        self.w = {}
        for k, v in sd.items():
            if not torch.is_tensor(v):
                continue
            v = v.to(DEVICE).float()
            wq = v if sanity else quant_sym(v, p["BITS"])
            mm = (torch.empty_like(wq).normal_(0, p["SIGMA_MISMATCH"], generator=gen)
                  if p["SIGMA_MISMATCH"] > 0 else 0.0)
            dr = (torch.empty(1, device=DEVICE).uniform_(-p["DRIFT"], p["DRIFT"], generator=gen)
                  if p["DRIFT"] > 0 else 0.0)
            self.w[k] = (wq * (1 + mm) * (1 + dr)).float()
        self._off = {}

    def noise(self, shape):
        if self.p["SNR_DB"] >= 200:
            return 0.0
        return torch.empty(shape, device=DEVICE).normal_(
            0, 0.25 / (10 ** (self.p["SNR_DB"] / 20)), generator=self.g)

    def _gilbert(self, prod):
        if self.p["BETA_H3"] <= 0:
            return prod.sum(dim=2)
        vs = self.p["V_SAT"]
        xs = prod / vs
        return (vs * torch.tanh(prod * (1 + self.p["BETA_H3"] * xs.square()) / vs)).sum(dim=2)

    def mac1x1(self, x, w, bias, groups=1):
        """1x1 conv（groups 恒 1）：每个 (out,in) 一对一个 Gilbert 乘法器。"""
        if x.ndim == 2:
            x = x[:, None, :]
        B, Cin, T = x.shape
        ww = self.w[w] if isinstance(w, str) else w
        Cout = ww.shape[0]
        prod = x[:, None, :, :] * ww.view(1, Cout, Cin, 1)      # (B,Cout,Cin,T)
        y = self._gilbert(prod)
        if bias is not None:
            bb = self.w[bias] if isinstance(bias, str) else bias
            y = y + bb.view(1, -1, 1)
        return y

    def conv(self, x, wkey, bkey, dilation, groups):
        """因果分组 1D 卷积。

        注意：官方用 `nn.Conv1d(padding=0)`（**valid，不 pad**），再靠"末尾对齐"实现因果
        （`_conv.py` 无 padding 参数，`_layer_array.py` 用 out_length = L-(RF-1) 裁剪）。
        因此这里也必须 valid，不能用左 pad —— 否则输出长度差 (K-1)*d，后续对齐全错。
        """
        if x.ndim == 2:
            x = x[:, None, :]
        B, Cin, T = x.shape
        w = self.w[wkey]
        Cout, CinPerG, K = w.shape
        G = groups
        CinG, CoutG = Cin // G, Cout // G
        T_out = T - (K - 1) * dilation
        y = 0
        for k in range(K):
            tap = x[:, :, k * dilation: k * dilation + T_out]        # valid 抽取
            tg = tap.reshape(B, G, CinG, T_out)
            wk = w[:, :, k].reshape(G, CoutG, CinG)
            prod = tg.unsqueeze(2) * wk.view(1, G, CoutG, CinG, 1)   # (B,G,CoutG,CinG,T)
            # _gilbert 沿 dim=2 求和，必须把输入通道搬到 dim=2（与参考实现一致）
            prod = prod.reshape(B, G * CoutG, CinG, T_out)
            y = y + self._gilbert(prod).reshape(B, G, CoutG, T_out).reshape(B, Cout, T_out)
        if bkey is not None and bkey in self.w:
            y = y + self.w[bkey].view(1, -1, 1)
        return y

    def act(self, z, kind):
        """差分对激活。失调缓存 key 必须含通道数：官方模型两层 array 通道不同
        （standard 16→8、lite 12→6），只按 kind 缓存会拿到错误尺寸的失调向量。
        """
        p = self.p
        key = (kind, z.shape[1])
        if p["SIGMA_OFFSET"] > 0 and key not in self._off:
            self._off[key] = torch.empty((1, z.shape[1], 1), device=DEVICE).normal_(
                0, p["SIGMA_OFFSET"], generator=self.g)
        if p["SIGMA_OFFSET"] > 0:
            z = z + self._off[key]
        k = kind.lower()
        if k.startswith("tanh"):
            return torch.tanh(z)
        if k.startswith("leakyrelu"):
            return torch.where(z > 0, z, 0.01 * z)
        if k.startswith("relu"):
            return torch.clamp(z, min=0)
        raise ValueError(f"未支持的激活: {kind}")


# ---------------------------------------------------------------- 结构解析

def parse_arrays(nam_cfg):
    """从 .nam 的 config 解析出 layer array 列表，字段就地规范化。"""
    arrays = []
    for lc in nam_cfg["layers"]:
        ks = lc.get("kernel_sizes")
        if ks is None:
            k0 = lc.get("kernel_size", 3)
            ks = [k0] * len(lc["dilations"]) if isinstance(k0, int) else list(k0)
        act = lc.get("activation", "Tanh")
        if isinstance(act, list):
            act = act[0]
        if isinstance(act, dict):
            act = act.get("type") or act.get("name") or "Tanh"
        head1x1 = lc.get("head1x1") or {}
        layer1x1 = lc.get("layer1x1") or {}
        arrays.append(dict(
            channels=lc["channels"],
            kernel_sizes=list(ks),
            dilations=list(lc["dilations"]),
            activation=str(act),
            groups_input=lc.get("groups_input", 1),
            groups_input_mixin=lc.get("groups_input_mixin", 1),
            layer1x1_active=bool(layer1x1.get("active", False)),
            layer1x1_groups=layer1x1.get("groups", 1),
            head1x1_active=bool(head1x1.get("active", False)),
            head1x1_out=int(head1x1.get("out_channels", 1)),
            head=dict(lc["head"]),          # head_rechannel: out_channels/kernel_size/bias
        ))
    return arrays


def rf_no_head_rechannel(arr):
    return 1 + sum((k - 1) * d for k, d in zip(arr["kernel_sizes"], arr["dilations"]))


# ---------------------------------------------------------------- 前向

def forward_array(dev, sd, arr, ai, x, cond, head_input=None):
    """复刻官方 LayerArray.forward（含 head 跨层累加、跨 array 续接、残差对齐）。

    官方关键点（_layer_array.py 1069-1080）：
        head_input = head_term if head_input is None else head_input[..., -out_len_nh:] + head_term
    即 array 之间**续接累加**：上一个 array 的 head_rechannel 输出作为下一个 array 的初值。
    """
    pre = f"_layer_arrays.{ai}."
    g_in = arr["groups_input"]
    Lx, Lc = x.shape[2], cond.shape[2]
    rf_nh = rf_no_head_rechannel(arr)
    out_len_no_head = min(Lx, Lc) - (rf_nh - 1)
    out_len = out_len_no_head            # 本层 head_rechannel 是 1-tap → RF 不额外增加

    # rechannel: 1x1，输入 → channels
    x = dev.mac1x1(x, pre + "_rechannel.weight", None, groups=1)

    layers = sorted({int(k.split(".")[3]) for k in sd
                     if k.startswith(pre + "_layers.")})
    for li in layers:
        lp = f"{pre}_layers.{li}."
        d = arr["dilations"][li]
        zconv = dev.conv(x, lp + "_conv.weight", lp + "_conv.bias", d, g_in)

        # input_mixer: cond(1ch) → channels，每通道一个乘法器，按官方裁到 zconv 长度
        mix = dev.mac1x1(cond, lp + "_input_mixer.weight", None, groups=1)
        mix = mix[:, :, -zconv.shape[2]:]
        z1len = min(zconv.shape[2], mix.shape[2])
        z1 = zconv[:, :, -z1len:] + mix[:, :, -z1len:]

        post = dev.act(z1, arr["activation"])
        post = post + dev.noise(post.shape)

        if arr["layer1x1_active"]:
            lo = dev.mac1x1(post, lp + "_layer1x1.weight", lp + "_layer1x1.bias",
                            groups=arr["layer1x1_groups"])
        else:
            lo = post

        # head_term：head1x1 若 active 则先 1x1 再裁；否则直接裁
        if arr["head1x1_active"]:
            hb = lp + "_head1x1.bias"
            term = dev.mac1x1(post, lp + "_head1x1.weight", hb if hb in sd else None, 1)
            term = term[:, :, -out_len_no_head:]
        else:
            term = post[:, :, -out_len_no_head:]
        head_input = (term if head_input is None
                      else head_input[:, :, -out_len_no_head:] + term)

        # residual 对齐 layer_output 长度
        x = x[:, :, -lo.shape[2]:] + lo + dev.noise(lo.shape)

    # head_rechannel：1x1，可带 bias
    hw = pre + "_head_rechannel.weight"
    hb = pre + "_head_rechannel.bias"
    head_out = dev.mac1x1(head_input, hw, hb if hb in sd else None, groups=1)
    x = x[:, :, -out_len:]
    assert head_out.shape[2] == x.shape[2], f"array{ai} 长度不齐 {head_out.shape} vs {x.shape}"
    return head_out, x


def total_receptive_field(arrays):
    """总 RF = 各 array 的 (RF_no_head + head_rechannel_tap - 1) 逐级累减，与官方一致。

    官方 WaveNet.receptive_field 是各 layer_array.receptive_field 的串联：
        RF_i = RF_no_head_rechannel_i + head_rechannel_kernel_i - 1
    串联总 RF = 1 + Σ(RF_i - 1)
    """
    return 1 + sum(rf_no_head_rechannel(a) + a["head"].get("kernel_size", 1) - 1 - 1
                   for a in arrays)


def forward(dev, sd, nam, x):
    """复刻官方 WaveNet.forward（含 pad_start=True 的前置零填充）。

    官方 BaseNet.forward：pad_start 默认 True，在输入左侧补 (RF-1) 个零，
    输出与输入等长（因果、末尾对齐）。这里一并复刻，否则长度口径对不上。
    """
    arrays = parse_arrays(nam["config"])
    head_scale = float(nam["config"].get("head_scale", 1.0))
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim == 3:
        x = x[:, 0, :]
    rf = total_receptive_field(arrays)
    x = F.pad(x, (rf - 1, 0))                      # pad_start=True
    cond = x[:, None, :]
    y, head_input = x[:, None, :], None
    for ai, arr in enumerate(arrays):
        head_input, y = forward_array(dev, sd, arr, ai, y, cond, head_input=head_input)
    head = head_input * head_scale
    hcfg = nam["config"].get("head")
    if hcfg:
        raise NotImplementedError("本脚本暂不支持带独立 head 网络的 .nam（当前四档均无）")
    return head[:, 0, :]


# ---------------------------------------------------------------- 评测

def esr_nrms(pred, tgt):
    if pred.ndim == 1:
        pred = pred[None]
    off = tgt.numel() - pred.shape[-1]
    t = tgt[off:].float()
    err = pred.float() - t[None]
    mse = float(torch.mean(err ** 2))
    ms = float(torch.mean(t ** 2))
    return mse / (ms + 1e-20), float(np.sqrt(mse / (ms + 1e-20)))


def load_seg(t0, seg_len, xi_all, yi_all):
    i0 = int(t0 * SR)
    i1 = min(len(xi_all), i0 + int(SR * seg_len))
    return (torch.from_numpy(xi_all[i0:i1]).to(DEVICE),
            torch.from_numpy(yi_all[i0:i1]).to(DEVICE))


def official_path(nam_path, x, y):
    """官方 PyTorch 前向（用 nam 包），作 sanity 基准。

    官方 BaseNet.forward 接受 (B,T) 二维，内部 pad_start=True 补零后走 WaveNet._forward。
    """
    sys.path.insert(0, os.environ.get("NAM_SRC", "/public/home/spotmaverick/nam_proj"))
    from nam.models import init_from_nam
    net = init_from_nam(json.load(open(nam_path)))
    net = net.to(DEVICE).eval()
    with torch.no_grad():
        out = net(x[None, :], pad_start=True)[0]
    return out, esr_nrms(out, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nam", required=True)
    ap.add_argument("--mc", type=int, default=4)
    ap.add_argument("--segs", default="0,8,16")
    ap.add_argument("--seg-len", type=float, default=SEG_LEN)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    nam = json.load(open(args.nam))
    # 官方 .nam 权重是扁平 list，需要重建成 state_dict 键名
    sys.path.insert(0, os.environ.get("NAM_SRC", "/public/home/spotmaverick/nam_proj"))
    from nam.models import init_from_nam
    net = init_from_nam(nam)
    sd = {k: v for k, v in net.state_dict().items() if "_has_sample" not in k and "_sample_rate" not in k}
    sd = {k[5:] if k.startswith("_net.") else k: v for k, v in sd.items()}
    net = net.to(DEVICE).eval()

    arrays = parse_arrays(nam["config"])
    print(f"[cfg] arrays={len(arrays)} " + " | ".join(
        f"ch={a['channels']} L={len(a['dilations'])} K={a['kernel_sizes'][0]} "
        f"RF={rf_no_head_rechannel(a)} act={a['activation']}" for a in arrays), flush=True)
    print(f"[cfg] head_scale={nam['config'].get('head_scale')} params={sum(v.numel() for v in sd.values())}", flush=True)

    xi_all, sr = sf.read(str(DATA / "tsa15_test_input.wav"), dtype="float32")
    yi_all, _ = sf.read(str(DATA / "tsa15_test_target.wav"), dtype="float32")
    assert sr == SR

    segs = [(f"seg{i}", float(t)) for i, t in enumerate(args.segs.split(","))]

    # ---- sanity：理想器件路径 vs 官方前向
    xt, yt = load_seg(segs[0][1], args.seg_len, xi_all, yi_all)
    dev0 = DevSim(sd, torch.Generator(device=DEVICE).manual_seed(0), sanity=True)
    with torch.no_grad():
        pred_ideal = forward(dev0, sd, nam, xt)
        if pred_ideal.ndim == 2:
            pred_ideal = pred_ideal[0]
        out_off, (_, nr_off) = official_path(args.nam, xt, yt)
    n = min(pred_ideal.shape[0], out_off.shape[0])
    maxd = float((pred_ideal[-n:] - out_off[-n:]).abs().max())
    print(f"[sanity] max|Δ|(ideal vs official)={maxd:.3e}  "
          f"official NRMS={nr_off:.6f}", flush=True)
    if maxd > 1e-5:
        print("[sanity] FAIL —— 结构复刻有误，任何 NRMS 都不可用。停止。", flush=True)
        return 2
    if args.sanity:
        return 0

    # ---- 数字 + 器件（同 MC 种子对齐，与 P4 口径一致）
    res = {"nam": args.nam, "mc": args.mc, "dialect": DIALECT,
           "sanity_max_abs_diff": maxd, "segments": []}
    for name, t0 in segs:
        xt, yt = load_seg(t0, args.seg_len, xi_all, yi_all)
        devv, dig = [], []
        for i in range(args.mc):
            dg = DevSim(sd, torch.Generator(device=DEVICE).manual_seed(1000 + i), sanity=True)
            with torch.no_grad():
                pr = forward(dg, sd, nam, xt)
            dig.append(esr_nrms(pr, yt)[1])
            dv = DevSim(sd, torch.Generator(device=DEVICE).manual_seed(1000 + i))
            with torch.no_grad():
                pr = forward(dv, sd, nam, xt)
            devv.append(esr_nrms(pr, yt)[1])
        nrms_device = float(np.sqrt(np.mean(np.square(devv))))
        nrms_digital = float(np.sqrt(np.mean(np.square(dig))))
        row = {"seg": name, "t0": t0,
               "nrms_device": nrms_device, "nrms_digital": nrms_digital,
               "drop": nrms_device - nrms_digital,
               "mc_device": devv, "mc_digital": dig}
        res["segments"].append(row)
        print(f"  [{name}] dev={nrms_device:.4f} dig={nrms_digital:.4f} "
              f"drop={row['drop']:+.4f} (mc={args.mc})", flush=True)

    # 汇总（与 P0a 一致：3 段池化）
    dev_pool = [r["nrms_device"] for r in res["segments"]]
    dig_pool = [r["nrms_digital"] for r in res["segments"]]
    res["summary"] = {
        "nrms_device_median": float(np.median(dev_pool)),
        "nrms_device_mean": float(np.mean(dev_pool)),
        "nrms_digital_median": float(np.median(dig_pool)),
        "drop_median": float(np.median(dev_pool) - np.median(dig_pool)),
    }
    print("[summary] " + json.dumps(res["summary"], ensure_ascii=False), flush=True)

    out = Path(args.out) if args.out else Path("official_analog_" + Path(args.nam).stem + ".json")
    json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
    print("[done] " + str(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
