# analog_sim_dw.py — 档位2: depthwise 模拟友好拓扑的器件抽象级物理仿真
#
# 为什么需要独立脚本:
#   analog_sim_wavenet.py 的 conv3_mac 处理全连接 conv (权重 ch×ch×3),
#   跨通道混流发生在 3-tap 卷积本身. depthwise conv (权重 ch×1×3) 的
#   跨通道混流只发生在 1×1 层 (layer1x1 / rechannel), 因此:
#     - 每通道独立累加链更短 → 残差流动态范围可能显著小于全连接
#     - Gilbert 三次失真的累积方式不同
#   这两点直接决定模拟摆幅需求, 是模拟可行性的核心指标, 不能套用全连接脚本.
#
# 器件映射 (与 analog_sim_wavenet.py 一致):
#   1. 权重存储 = 6-bit 浮栅: 逐通道对称量化 + 高斯写入失配 + 全局温漂
#   2. 乘法器   = Gilbert cell: y = V_sat·tanh(w·x·(1+β·x̂²)/V_sat)
#   3. 求和     = KCL 电流求和 (无源, 理想)
#   4. 激活     = 差分对饱和 (天然 tanh) + 输入失调
#   5. 节点噪声 = 热噪声 (SNR_DB)
#   6. 延迟线   = 数字 SRAM (数模混合, 无损)
#
# 用法:
#   python analog_sim_dw.py --sanity              # 对拍: 物理路径 == 数字路径
#   python analog_sim_dw.py --quick               # 5s 快速冒烟 MC×3
#   python analog_sim_dw.py --variant dw_ch24     # 全段 MC×16
#   python analog_sim_dw.py --variant dw_ch24 --vsat-scan  # 摆幅定标扫描
import sys, json, os
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 可移植：默认从脚本所在项目根推导，可用 ANALOG_BASE 覆盖（旧服务器布局）
import os as _os
BASE = _os.environ.get("ANALOG_BASE", str(__import__("pathlib").Path(__file__).resolve().parents[2]))
DATA = BASE + "/data"
OUT = BASE + "/analog_sim"
ARCH_DIR = BASE + "/arch_analog"

# 变体定义: 与 arch_analog_friendly.build() 一致
VARIANTS = {
    "dw_ch12": dict(ch=12, rf_pct=100, depthwise=True),
    "dw_ch24": dict(ch=24, rf_pct=100, depthwise=True),
    "dw_ch12_rf50": dict(ch=12, rf_pct=50, depthwise=True),
    "dw_ch24_rf50": dict(ch=24, rf_pct=50, depthwise=True),
    # 低秩变体：1x1 为两段式，forward/physical 按 is_lowrank 自动处理
    "wd_ch32_r8": dict(ch=32, rf_pct=100, depthwise=True),
    "wd_ch32_r6": dict(ch=32, rf_pct=100, depthwise=True),
    "wd_ch24_r6": dict(ch=24, rf_pct=100, depthwise=True),
}
HEAD_SCALE = 0.02

P = dict(
    BITS=6,
    SIGMA_MISMATCH=0.01,
    DRIFT=0.03,
    BETA_H3=0.002,
    V_SAT=16.0,          # 会被 --vsat-scan 覆盖
    SIGMA_OFFSET=0.005,
    SNR_DB=70,
    MC=16,
    CHUNK=480000,
)


def group_rf(dils):
    return 2 * sum(dils) + 1


def build_groups(ch, rf_pct):
    """复刻 arch_analog_friendly.build 的分组与 dilation 缩放."""
    head_ch = max(1, ch // 2)
    d1 = [1, 2, 4, 8, 16, 32, 64]
    d2 = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    if rf_pct != 100:
        d1 = [max(1, int(d * rf_pct / 100)) for d in d1]
        d2 = [max(1, int(d * rf_pct / 100)) for d in d2]
    return [(1, ch, d1, head_ch, False), (ch, head_ch, d2, 1, True)]


def quant6(w):
    """逐通道对称 6-bit 量化 (浮栅写入精度)."""
    if not torch.is_tensor(w):
        return w
    qmax = 2 ** (P["BITS"] - 1) - 1
    mx = w.abs().flatten(1).max(dim=1).values.clamp(min=1e-12) if w.dim() > 1 else w.abs().max().clamp(min=1e-12)
    shape = (-1,) + (1,) * (w.dim() - 1)
    scale = (mx / qmax).reshape(shape)
    return torch.round(w / scale).clamp(-qmax, qmax) * scale


class Dev:
    """器件实例: 量化 + 失配 + 温漂后的权重, 以及各类非理想算子."""

    def __init__(self, sd, gen):
        self.g = gen
        self.w = {}
        for k, v in sd.items():
            if not torch.is_tensor(v):
                continue
            v = v.to(DEVICE).float()
            wq = quant6(v)
            mm = torch.empty_like(wq).normal_(0, P["SIGMA_MISMATCH"], generator=gen)
            if P["DRIFT"] > 0:
                drift = torch.empty(1, device=DEVICE).uniform_(-P["DRIFT"], P["DRIFT"], generator=gen)
            else:
                drift = torch.zeros(1, device=DEVICE)
            self.w[k] = (wq * (1 + mm) * (1 + drift)).float()
        self.head_scale = float(sd["head_scale"]) if "head_scale" in sd else HEAD_SCALE
        self._sn = None

    def noise_sig(self):
        if self._sn is None:
            self._sn = 10 ** (-P["SNR_DB"] / 20.0)
        return self._sn

    def node_noise(self, shape):
        if P["SNR_DB"] >= 200:
            return 0.0
        return torch.empty(shape, device=DEVICE).normal_(0, self.noise_sig(), generator=self.g)

    def mac(self, x, w, bias=None):
        """Gilbert 乘法阵列 + KCL 求和. x:(B,Cin,T), w:(Cout,Cin) 或 (Cout,Cin,1)."""
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

    def dw_conv3_mac(self, x, w, bias, dilation):
        """depthwise 3-tap 卷积: 权重 (ch,1,3), 每通道独立, 无跨通道累加.

        与全连接的物理差异: 乘法器阵列是 ch 组独立链, 每链只累加 3 个 tap,
        没有 KCL 多路求和的电流叠加 → 动态范围与失真特性不同.
        """
        B, Cin, T = x.shape
        xp = F.pad(x, (2 * dilation, 0))
        y = 0
        for k in range(3):
            taps = xp[:, :, k * dilation: k * dilation + T]          # (B,Cin,T)
            w_k = w[:, 0, k].view(1, Cin, 1)                         # (1,Cin,1)
            prod = taps * w_k
            if P["BETA_H3"] <= 0:
                y = y + prod
            else:
                xs = prod / P["V_SAT"]
                y = y + P["V_SAT"] * torch.tanh(prod * (1 + P["BETA_H3"] * xs.square()) / P["V_SAT"])
        if bias is not None:
            y = y + bias[None, :, None]
        return y

    def tanh_act(self, z):
        if P["SIGMA_OFFSET"] > 0:
            off = torch.empty((1, z.shape[1], 1), device=DEVICE).normal_(0, P["SIGMA_OFFSET"], generator=self.g)
            z = z + off
        return torch.tanh(z)


def is_lowrank(sd_keys, pre):
    """判断该层的 1x1 是否为低秩两段式 (layer1x1.0/.1)。"""
    return f"{pre}.layer1x1.0.weight" in sd_keys


def apply_1x1_physical(dev, z, pre):
    """1x1 混合层：全秩单段或低秩两段，器件非理想性逐段施加。"""
    if is_lowrank(dev.w, pre):
        # ch -> r -> ch，第一段无 bias，第二段有 bias
        h = dev.mac(z, dev.w[f"{pre}.layer1x1.0.weight"])
        return dev.mac(h, dev.w[f"{pre}.layer1x1.1.weight"],
                       dev.w.get(f"{pre}.layer1x1.1.bias"))
    return dev.mac(z, dev.w[f"{pre}.layer1x1.weight"],
                   dev.w.get(f"{pre}.layer1x1.bias"))


def forward_group(dev, gi, y, c, head_input, GROUPS):
    (in_size, ch, dils, head_out, head_bias) = GROUPS[gi]
    rf_g = group_rf(dils)

    y = dev.mac(y, dev.w[f"groups.{gi}.rechannel.weight"])
    out_len = y.shape[-1] - rf_g + 1

    for li, d in enumerate(dils):
        pre = f"groups.{gi}.layers.{li}"
        z = dev.dw_conv3_mac(y, dev.w[f"{pre}.conv.weight"], dev.w.get(f"{pre}.conv.bias"), d)
        Lz = z.shape[-1]
        mix = dev.mac(c, dev.w[f"{pre}.input_mixer.weight"])[:, :, -Lz:]
        z = dev.tanh_act(z + mix)
        z = z + dev.node_noise(z.shape)
        resid = y[:, :, -Lz:] + apply_1x1_physical(dev, z, pre)
        resid = resid + dev.node_noise(resid.shape)
        y = resid
        term = z[:, :, -out_len:]
        head_input = term if head_input is None else head_input[:, :, -out_len:] + term

    hb = dev.w.get(f"groups.{gi}.head_rechannel.bias") if head_bias else None
    head_out = dev.mac(head_input, dev.w[f"groups.{gi}.head_rechannel.weight"], hb)
    return head_out, y


def physical_fwd(dev, x, GROUPS):
    c = x
    head_input, y = None, x
    for gi in range(len(GROUPS)):
        head_input, y = forward_group(dev, gi, y, c, head_input, GROUPS)
    return dev.head_scale * head_input


def digital_fwd(sd, x, GROUPS):
    """无扰数字基线, 逐行对应 arch_analog_friendly.AFNet.forward."""
    dev = x.device
    sd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in sd.items()}
    c = x
    head_input, y = None, x
    for gi, (in_size, ch, dils, head_out, head_bias) in enumerate(GROUPS):
        rf_g = group_rf(dils)
        y = F.conv1d(y, sd[f"groups.{gi}.rechannel.weight"], None, 1)
        out_len = y.shape[-1] - rf_g + 1
        for li, d in enumerate(dils):
            pre = f"groups.{gi}.layers.{li}"
            z = F.conv1d(F.pad(y, (2 * d, 0)), sd[f"{pre}.conv.weight"], sd[f"{pre}.conv.bias"],
                         dilation=d, groups=ch)
            Lz = z.shape[-1]
            mix = F.conv1d(c, sd[f"{pre}.input_mixer.weight"], None, 1)[:, :, -Lz:]
            z = torch.tanh(z + mix)
            if is_lowrank(sd, pre):
                h = F.conv1d(z, sd[f"{pre}.layer1x1.0.weight"], None, 1)
                mix1 = F.conv1d(h, sd[f"{pre}.layer1x1.1.weight"],
                                sd[f"{pre}.layer1x1.1.bias"], 1)
            else:
                mix1 = F.conv1d(z, sd[f"{pre}.layer1x1.weight"],
                                sd[f"{pre}.layer1x1.bias"], 1)
            resid = y[:, :, -Lz:] + mix1
            y = resid
            term = z[:, :, -out_len:]
            head_input = term if head_input is None else head_input[:, :, -out_len:] + term
        hb = sd.get(f"groups.{gi}.head_rechannel.bias") if head_bias else None
        head_input = F.conv1d(head_input, sd[f"groups.{gi}.head_rechannel.weight"], hb, 1)
    return sd["head_scale"] * head_input if "head_scale" in sd else HEAD_SCALE * head_input


def chunked(dev, fn, xt, GROUPS, RF):
    """分段推理, 段间保留 RF-1 历史, 拼接后与整段等价."""
    T = xt.shape[-1]
    overlap = RF - 1
    trim = group_rf(GROUPS[-1][2]) - 1
    drop = overlap - trim
    outs, pos = [], 0
    while pos < T:
        a = max(0, pos - overlap)
        seg = xt[:, :, a:min(pos + P["CHUNK"], T)]
        o = fn(dev, seg, GROUPS)
        d = min(drop, o.shape[-1])
        outs.append(o[:, :, d:])
        pos += P["CHUNK"]
    return torch.cat(outs, dim=-1)


def nrms(pred, ref):
    return float((pred.double() - ref.double()).norm() / ref.double().norm())


def macs_of(GROUPS):
    m = 0
    for in_size, ch, dils, head_out, head_bias in GROUPS:
        m += in_size * ch
        m += len(dils) * (ch * 3)        # depthwise: ch×1×3
        m += len(dils) * ch              # input_mixer
        m += len(dils) * (ch * ch)       # layer1x1
        m += ch * head_out
    return m


def run_once(sd, GROUPS, RF, xt, yt, valid, mc, seed0=1000, verbose=True):
    """跑 mc 次蒙特卡洛, 返回 (fp32, quant, [各次NRMS])."""
    with torch.no_grad():
        o_dig = chunked(sd, digital_fwd, xt, GROUPS, RF)[0, 0][-valid:]
        ref = yt[-valid:]
        base = nrms(o_dig, ref)
        sdq = {k: (quant6(v) if torch.is_tensor(v) else v) for k, v in sd.items()}
        o_q = chunked(sdq, digital_fwd, xt, GROUPS, RF)[0, 0][-valid:]
        q = nrms(o_q, ref)
        arr = []
        for k in range(mc):
            g = torch.Generator(device=DEVICE).manual_seed(seed0 + k)
            dev = Dev(sd, g)
            o = chunked(dev, physical_fwd, xt, GROUPS, RF)[0, 0][-valid:]
            r = nrms(o, ref)
            arr.append(r)
            if verbose:
                print(f"  MC#{k+1}: NRMS = {r:.4f}", flush=True)
    return base, q, arr


def main():
    sanity = "--sanity" in sys.argv
    quick = "--quick" in sys.argv
    scan = "--vsat-scan" in sys.argv
    variant = "dw_ch24"
    if "--variant" in sys.argv:
        variant = sys.argv[sys.argv.index("--variant") + 1]

    cfg = VARIANTS[variant]
    GROUPS = build_groups(cfg["ch"], cfg["rf_pct"])
    RF = sum(group_rf(g[2]) for g in GROUPS) - (len(GROUPS) - 1)

    if sanity:
        P["BETA_H3"] = 0.0
        P["SIGMA_OFFSET"] = 0.0
        P["SIGMA_MISMATCH"] = 0.0
        P["DRIFT"] = 0.0
        P["SNR_DB"] = 999
        P["MC"] = 1
    if quick:
        P["MC"] = 3

    # 支持 --weight 指定权重文件（默认 variant_s7.pt），便于跑双种子物理仿真
    wpath = f"{ARCH_DIR}/{variant}_s7.pt"
    if "--weight" in sys.argv:
        wname = sys.argv[sys.argv.index("--weight") + 1]
        wpath = wname if "/" in wname else f"{ARCH_DIR}/{wname}"
    if not os.path.exists(wpath):
        print(f"[ABORT] 权重不存在: {wpath}")
        sys.exit(1)
    try:
        sd = torch.load(wpath, map_location="cpu", weights_only=False)
    except TypeError:  # torch 1.10 无 weights_only 参数
        sd = torch.load(wpath, map_location="cpu")
    sd = {k: (v.float() if torch.is_tensor(v) else v) for k, v in sd.items()}

    x, sr = sf.read(f"{DATA}/tsa15_test_input.wav")
    assert sr == 48000
    y, _ = sf.read(f"{DATA}/tsa15_test_target.wav")
    if quick or sanity:
        x, y = x[:240000], y[:240000]
    xt = torch.from_numpy(x.astype(np.float32))[None, None].to(DEVICE)
    yt = torch.from_numpy(y.astype(np.float32)).to(DEVICE)

    nparam = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
    valid = xt.shape[-1] - RF + 1
    print(f"device={DEVICE}  variant={variant}  参数 {nparam}  RF {RF} "
          f"({RF/48000*1000:.1f}ms)  MACs {macs_of(GROUPS)}  评估 {len(x)/48000:.1f}s  MC×{P['MC']}")

    if sanity:
        base, q, arr = run_once(sd, GROUPS, RF, xt, yt, valid, 1, verbose=False)
        print(f"[fp32 数字基线]  NRMS = {base:.4f}")
        print(f"[6-bit 仅量化]   NRMS = {q:.4f}")
        # 物理路径(无非理想) 应等于 6-bit 量化路径
        with torch.no_grad():
            sdq = {k: (quant6(v) if torch.is_tensor(v) else v) for k, v in sd.items()}
            g = torch.Generator(device=DEVICE).manual_seed(0)
            dev = Dev(sdq, g)
            o_p = chunked(dev, physical_fwd, xt, GROUPS, RF)[0, 0][-valid:]
            p = nrms(o_p, yt[-valid:])
        d = abs(p - q)
        print(f"[物理路径(无扰)] NRMS = {p:.4f}   (应与 6-bit 量化 {q:.4f} 一致)")
        print(f"[对拍] 物理 vs 量化 差值 = {d:.6f}  {'PASS' if d < 1e-3 else 'FAIL'}")
        print(f"\n=== 总判定: {'PASS' if d < 1e-3 else 'FAIL'} ===")
        return

    if scan:
        print("\n=== Vsat 摆幅定标扫描 ===")
        orig = P["V_SAT"]
        results = []
        for vs in [1.5, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0]:
            P["V_SAT"] = vs
            base, q, arr = run_once(sd, GROUPS, RF, xt, yt, valid, 3, verbose=False)
            m = float(np.mean(arr))
            results.append({"vsat": vs, "quant": q, "physical_mean": round(m, 4),
                            "physical_std": round(float(np.std(arr)), 4),
                            "degrade_vs_quant_pct": round((m / q - 1) * 100, 1)})
            print(f"  Vsat={vs:>6}: 6bit {q:.4f} → 全器件 {m:.4f} ± {np.std(arr):.4f} "
                  f"({(m/q-1)*100:+.1f}%)", flush=True)
        P["V_SAT"] = orig
        os.makedirs(OUT, exist_ok=True)
        json.dump({"variant": variant, "vsat_scan": results},
                  open(f"{OUT}/vsat_scan_{variant}.json", "w"), indent=2)
        print(f"\n结果已写 {OUT}/vsat_scan_{variant}.json")
        return

    base, q, arr = run_once(sd, GROUPS, RF, xt, yt, valid, P["MC"])
    a = np.array(arr)
    print(f"[fp32 数字基线]  NRMS = {base:.4f}")
    print(f"[6-bit 仅量化]   NRMS = {q:.4f}")
    print(f"[全物理层 MC]    NRMS = {a.mean():.4f} ± {a.std():.4f}")

    res = {
        "device": DEVICE, "variant": variant, "params": nparam, "rf": RF,
        "rf_ms": round(RF / 48000 * 1000, 2), "seconds": len(x) / 48000, "mc": P["MC"],
        "nrms_fp32": round(base, 4), "nrms_quant_only": round(q, 4),
        "nrms_physical_mean": round(float(a.mean()), 4),
        "nrms_physical_std": round(float(a.std()), 4),
        "nrms_physical_min": round(float(a.min()), 4),
        "nrms_physical_max": round(float(a.max()), 4),
        "macs_per_sample": macs_of(GROUPS),
        "params_dev": P,
    }
    print("\n=== 元器件规模核算 ===")
    print(f"权重存储: {nparam} 浮栅 2T cell ≈ {2*nparam} 管")
    print(f"每采样模拟 MACs: {res['macs_per_sample']}")
    print(f"延迟线: RF {RF} 抽头 × {GROUPS[1][1]}ch 数字 SRAM ≈ {RF*GROUPS[1][1]/1024:.1f} Kb")
    os.makedirs(OUT, exist_ok=True)
    stem = f"{variant}_full" if not quick else f"{variant}_quick"
    if "--weight" in sys.argv:
        # 带权重种子后缀，避免 s7/s11 结果互相覆盖
        wname = sys.argv[sys.argv.index("--weight") + 1]
        seed_tag = wname.replace(".pt", "").split("_")[-1]
        stem = f"{stem}_{seed_tag}"
    tag = stem
    json.dump(res, open(f"{OUT}/result_{tag}.json", "w"), indent=2)
    print(f"\n结果已写 {OUT}/result_{tag}.json")


if __name__ == "__main__":
    main()
