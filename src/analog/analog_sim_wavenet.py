# analog_sim_wavenet.py — 档位1: 器件抽象级物理仿真 (官方 WaveNet lite, 6553 参数)
# 运行: 智算服务器 DCU (DEVICE 自适应 cuda)
#
# 与 analog_sim.py 的区别: 后者硬编码 LGW-TCN 冠军 (depthwise + expand/collapse + ReLU),
# 本脚本对应官方 WaveNet lite 的真实算子 (全连接 3-tap conv + tanh + 1x1 残差),
# 因为 lite 才是当前精度最优解, 物理可行性必须按它的算子重新算一遍.
#
# 每个数学操作 → 带非理想性的电路块:
#   1. 权重存储 = 6-bit 浮栅: 逐通道对称量化 + 高斯写入失配 σ_m + 全局温漂
#   2. 乘法器   = Gilbert cell: y = V_sat·tanh(w·x·(1+β·x̂²)/V_sat)
#   3. 求和     = KCL 电流求和 (无源, 理想)
#   4. 激活     = 差分对饱和 (天然产生 tanh, 无需额外电路) + 输入失调 σ_off
#   5. 节点噪声 = 热噪声 (SNR_DB)
#   6. 延迟线   = 数字 SRAM (数模混合, 无损)
#
# 重要: 官方 WaveNet 用 tanh, 差分对的 tanh 传递函数天然吻合 —— 这是选它做物理实现的
# 依据之一. 相比之下 LGW 的 ReLU 需要额外整流电路且没有天然饱和.
#
# 用法:
#   python analog_sim_wavenet.py --sanity   # 关闭全部非理想性, 与 digital 对拍 (必须一致)
#   python analog_sim_wavenet.py --quick    # 5s 快速冒烟 MC×3
#   python analog_sim_wavenet.py            # 全段 MC×16
import sys, json
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE = "/work/home/spotmaverick/nam_project"
DATA = BASE + "/data"
W_PATH = BASE + "/arch_v2/lite_s7.pt"
OUT = BASE + "/analog_sim"

# 官方 lite 架构 (与 wavenet_official_dcu.py OFFICIAL_ARCHS["lite"] 一致)
GROUPS = [
    (1, 12, [1, 2, 4, 8, 16, 32, 64], 6, False),
    (12, 6, [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512], 1, True),
]
HEAD_SCALE = 0.02

P = dict(
    BITS=6,
    SIGMA_MISMATCH=0.01,
    DRIFT=0.03,
    BETA_H3=0.002,
    V_SAT=1.5,
    SIGMA_OFFSET=0.005,
    SNR_DB=70,
    MC=16,
    CHUNK=480000,
)


def group_rf(dils):
    return 2 * sum(dils) + 1


RF = sum(group_rf(g[2]) for g in GROUPS) - (len(GROUPS) - 1)
print(f"[cfg] lite RF = {RF} samples = {RF/48000*1000:.1f} ms")


def quant6(w):
    """逐输出通道对称量化到 6-bit (31 级)."""
    if w.dim() == 1:
        return w
    flat_dims = tuple(range(1, w.dim()))
    scale = w.abs().amax(dim=flat_dims, keepdim=True).clamp(min=1e-12) / 31.0
    return (w / scale).round().clamp(-32, 31) * scale


class Dev:
    """一次蒙特卡洛采样的器件实例."""

    def __init__(self, sd, gen):
        self.g = gen
        self.w = {}
        for k, v in sd.items():
            if not torch.is_tensor(v):
                continue
            v = v.to(DEVICE).float()
            wq = quant6(v)
            if P["SIGMA_MISMATCH"] > 0:
                mm = torch.empty_like(wq).normal_(0, P["SIGMA_MISMATCH"], generator=gen)
            else:
                mm = 0.0
            if P["DRIFT"] > 0:
                drift = torch.empty(1, device=DEVICE).uniform_(-P["DRIFT"], P["DRIFT"], generator=gen)
            else:
                drift = 1.0
            self.w[k] = (wq * (1 + mm) * (1 + drift)).float()
        self.head_scale = float(sd["head_scale"]) if "head_scale" in sd else HEAD_SCALE
        self._sn = None

    def noise_sig(self):
        if self._sn is None:
            self._sn = 0.25 / (10 ** (P["SNR_DB"] / 20))
        return self._sn

    def node_noise(self, shape):
        if P["SNR_DB"] >= 200:  # sanity 模式关闭噪声
            return 0.0
        return torch.empty(shape, device=DEVICE).normal_(0, self.noise_sig(), generator=self.g)

    def mac(self, x, w, bias=None, channels_out=None):
        """Gilbert 乘法阵列 + KCL 求和. x: (B,Cin,T), w: (Cout,Cin) 或 (Cout,Cin,1)."""
        w2 = w.reshape(w.shape[0], -1)
        prod = x.unsqueeze(1) * w2.view(1, w2.shape[0], w2.shape[1], 1)
        if P["BETA_H3"] <= 0:
            y = prod.sum(dim=2)
        else:
            xs = prod / P["V_SAT"]
            y = (P["V_SAT"] * torch.tanh(prod * (1 + P["BETA_H3"] * xs.square()) / P["V_SAT"])).sum(dim=2)
        if bias is not None:
            y = y + bias[None, :, None]
        return y

    def conv3_mac(self, x, w, bias, dilation):
        """3-tap 全连接卷积 (Cout×Cin×3): 每个 tap 是独立乘法器阵列."""
        B, Cin, T = x.shape
        Cout = w.shape[0]
        xp = F.pad(x, (2 * dilation, 0))  # causal
        taps = [xp[:, :, k * dilation: k * dilation + T] for k in range(3)]
        y = 0
        for k in range(3):
            w_k = w[:, :, k]
            prod = taps[k].unsqueeze(1) * w_k.view(1, Cout, Cin, 1)
            if P["BETA_H3"] <= 0:
                y = y + prod.sum(dim=2)
            else:
                xs = prod / P["V_SAT"]
                y = y + (P["V_SAT"] * torch.tanh(prod * (1 + P["BETA_H3"] * xs.square()) / P["V_SAT"])).sum(dim=2)
        if bias is not None:
            y = y + bias[None, :, None]
        return y

    def tanh_act(self, z):
        """差分对饱和: 天然 tanh + 输入失调电压."""
        if P["SIGMA_OFFSET"] > 0:
            off = torch.empty((1, z.shape[1], 1), device=DEVICE).normal_(0, P["SIGMA_OFFSET"], generator=self.g)
            z = z + off
        return torch.tanh(z)


def forward_group(dev, gi, y, c, head_input, sd_keys, cond=1):
    """复刻官方 _LayerArray.forward, 但算子换成器件实现."""
    (in_size, ch, dils, head_out, head_bias) = GROUPS[gi]
    rf_g = group_rf(dils)

    y = dev.mac(y, dev.w[f"layer_arrays.{gi}.rechannel.weight"], channels_out=ch)
    out_len = y.shape[-1] - rf_g + 1

    for li, d in enumerate(dils):
        pre = f"layer_arrays.{gi}.layers.{li}"
        z = dev.conv3_mac(y, dev.w[f"{pre}.conv.weight"], dev.w.get(f"{pre}.conv.bias"), d)
        Lz = z.shape[-1]
        mix = dev.mac(c, dev.w[f"{pre}.input_mixer.weight"])[:, :, -Lz:]
        z = dev.tanh_act(z + mix)
        z = z + dev.node_noise(z.shape)
        resid = y[:, :, -Lz:] + dev.mac(z, dev.w[f"{pre}.layer1x1.weight"], dev.w.get(f"{pre}.layer1x1.bias"))
        resid = resid + dev.node_noise(resid.shape)
        y = resid
        term = z[:, :, -out_len:]
        head_input = term if head_input is None else head_input[:, :, -out_len:] + term

    hb = dev.w.get(f"layer_arrays.{gi}.head_rechannel.bias") if head_bias else None
    head_out = dev.mac(head_input, dev.w[f"layer_arrays.{gi}.head_rechannel.weight"], hb)
    return head_out, y[:, :, -out_len:]


def physical_fwd(dev, x):
    c = x
    head_input, y = None, x
    for gi in range(len(GROUPS)):
        head_input, y = forward_group(dev, gi, y, c, head_input, None)
    return dev.head_scale * head_input


def digital_fwd(sd, x):
    """无扰数字基线, 顺序与官方 forward 逐行对应."""
    c = x
    head_input, y = None, x
    for gi, (in_size, ch, dils, head_out, head_bias) in enumerate(GROUPS):
        rf_g = group_rf(dils)
        y = F.conv1d(y, sd[f"layer_arrays.{gi}.rechannel.weight"], None, 1)
        out_len = y.shape[-1] - rf_g + 1
        for li, d in enumerate(dils):
            pre = f"layer_arrays.{gi}.layers.{li}"
            z = F.conv1d(F.pad(y, (2 * d, 0)), sd[f"{pre}.conv.weight"], sd[f"{pre}.conv.bias"], dilation=d)
            Lz = z.shape[-1]
            mix = F.conv1d(c, sd[f"{pre}.input_mixer.weight"], None, 1)[:, :, -Lz:]
            z = torch.tanh(z + mix)
            resid = y[:, :, -Lz:] + F.conv1d(z, sd[f"{pre}.layer1x1.weight"], sd[f"{pre}.layer1x1.bias"], 1)
            y = resid
            term = z[:, :, -out_len:]
            head_input = term if head_input is None else head_input[:, :, -out_len:] + term
        hb = sd.get(f"layer_arrays.{gi}.head_rechannel.bias") if head_bias else None
        head_input = F.conv1d(head_input, sd[f"layer_arrays.{gi}.head_rechannel.weight"], hb, 1)
    return sd["head_scale"] * head_input if "head_scale" in sd else HEAD_SCALE * head_input


def chunked(dev, fn, xt):
    """分段推理, 段间保留 RF-1 历史, 拼接后与整段推理等价 (因果+平移不变)."""
    T = xt.shape[-1]
    outs, pos = [], 0
    while pos < T:
        a = max(0, pos - RF + 1)
        seg = xt[:, :, a:min(pos + P["CHUNK"], T)]
        o = fn(dev, seg)
        drop = pos - a
        outs.append(o[:, :, drop:] if drop > 0 else o)
        pos += P["CHUNK"]
    return torch.cat(outs, dim=-1)


def nrms(pred, ref):
    return float((pred.double() - ref.double()).norm() / ref.double().norm())


def macs_per_sample():
    """每采样模拟 MAC 数 (不含延迟线搬移)."""
    m = 0
    for in_size, ch, dils, head_out, head_bias in GROUPS:
        m += in_size * ch                      # rechannel
        m += len(dils) * (ch * ch * 3)         # conv 3-tap
        m += len(dils) * (ch * 1)              # input_mixer
        m += len(dils) * (ch * ch)             # layer1x1
        m += ch * head_out                     # head_rechannel
    return m


def main():
    sanity = "--sanity" in sys.argv
    quick = "--quick" in sys.argv
    if sanity:
        P["BETA_H3"] = 0.0
        P["SIGMA_OFFSET"] = 0.0
        P["SIGMA_MISMATCH"] = 0.0
        P["DRIFT"] = 0.0
        P["SNR_DB"] = 999
        P["MC"] = 1
    if quick:
        P["MC"] = 3

    sd = torch.load(W_PATH, map_location="cpu")
    sd = {k: (v.float() if torch.is_tensor(v) else v) for k, v in sd.items()}
    x, sr = sf.read(f"{DATA}/tsa15_test_input.wav")
    assert sr == 48000
    y, _ = sf.read(f"{DATA}/tsa15_test_target.wav")
    if quick or sanity:
        x, y = x[:240000], y[:240000]
    xt = torch.from_numpy(x.astype(np.float32))[None, None].to(DEVICE)
    yt = torch.from_numpy(y.astype(np.float32)).to(DEVICE)

    nparam = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
    print(f"device={DEVICE}  参数 {nparam}  评估 {len(x)/48000:.1f}s  MC×{P['MC']}")

    with torch.no_grad():
        o_dig = chunked(None, digital_fwd, xt)[0, 0]
        n = o_dig.shape[0]
        ref = yt[-n:]
        base = nrms(o_dig, ref)
        print(f"[fp32 数字基线]  NRMS = {base:.4f}")

        # 量化器版本走同一套数字路径, 只为隔离量化影响
        sdq = {k: (quant6(v) if torch.is_tensor(v) else v) for k, v in sd.items()}
        o_q = chunked(None, digital_fwd, sdq)[0, 0]
        q = nrms(o_q, ref)
        print(f"[6-bit 仅量化]   NRMS = {q:.4f}")

        if sanity:
            g = torch.Generator(device=DEVICE).manual_seed(0)
            dev = Dev(sdq, g)  # 用已量化权重, 参考值即 sdq 数字路径, 隔离器件非理想性
            o_p = chunked(dev, physical_fwd, xt)[0, 0]
            p = nrms(o_p, ref)
            print(f"[物理路径(无扰)] NRMS = {p:.4f}   (应与 6-bit 量化 {q:.4f} 一致)")
            d = abs(p - q)
            print(f"[对拍] 物理 vs 量化 差值 = {d:.6f}  {'PASS' if d < 1e-3 else 'FAIL'}")
            return

        mc = []
        for k in range(P["MC"]):
            g = torch.Generator(device=DEVICE).manual_seed(1000 + k)
            dev = Dev(sd, g)
            o = chunked(dev, physical_fwd, xt)[0, 0]
            r = nrms(o, ref)
            mc.append(r)
            print(f"  MC#{k+1}: NRMS = {r:.4f}")
        arr = np.array(mc)
        print(f"[全物理层 MC]    NRMS = {arr.mean():.4f} ± {arr.std():.4f}")

    res = {
        "device": DEVICE, "arch": "lite", "params": nparam, "rf": RF,
        "rf_ms": round(RF / 48000 * 1000, 2), "seconds": len(x) / 48000, "mc": P["MC"],
        "nrms_fp32": round(base, 4), "nrms_quant_only": round(q, 4),
        "nrms_physical_mean": round(float(arr.mean()), 4),
        "nrms_physical_std": round(float(arr.std()), 4),
        "nrms_physical_min": round(float(arr.min()), 4),
        "nrms_physical_max": round(float(arr.max()), 4),
        "macs_per_sample": macs_per_sample(),
        "params_dev": P,
    }
    print("\n=== 元器件规模核算 (lite) ===")
    print(f"权重存储: {nparam} 浮栅 2T cell ≈ {2*nparam} 管")
    print(f"每采样模拟 MACs: {res['macs_per_sample']}")
    print(f"延迟线: RF {RF} 抽头 × 6ch 数字 SRAM ≈ {RF*6/1024:.1f} Kb")
    import os
    os.makedirs(OUT, exist_ok=True)
    tag = "lite_quick" if quick else "lite_full"
    json.dump(res, open(f"{OUT}/result_{tag}.json", "w"), indent=2)
    print(f"\n结果已写 {OUT}/result_{tag}.json")


if __name__ == "__main__":
    main()
