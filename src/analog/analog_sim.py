# analog_sim.py — 档位1: 器件抽象级物理仿真 (冠军架构 v_tcn812h1_p3, 2149 参数)
# 运行: 智算服务器 DCU (DEVICE 自适应 cuda/cpu)
#
# 每个数学操作 → 带非理想性的电路块:
#   1. 权重存储 = 6-bit 浮栅: 逐通道对称量化 + 高斯写入失配 σ_m + 全局温漂
#   2. 乘法器   = Gilbert cell (逐乘法器建模): y = V_sat·tanh(w·x·(1+β·x̂²)/V_sat)
#   3. 求和     = KCL 电流求和 (无源, 理想)
#   4. 激活     = 跨导线性 ReLU + 每神经元输入失调 σ_off
#   5. 节点噪声 = 热噪声 (SNR_DB)
#   6. 延迟线   = 数字 SRAM (数模混合, 无损)
#
# 数据流 (与训练架构逐块对应): dw → expand → ReLU → collapse → 残差 + skip 门控
import sys, json
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA = "/work/home/spotmaverick/nam_project/data"
W_PATH = "/work/home/spotmaverick/nam_project/lgw_v_tcn812h1_p3.pt"
OUT = "/work/home/spotmaverick/nam_project/analog_sim"

CH, LAYERS, DIL_EXP, SKIP_GATES = 8, 12, 7, True

P = dict(
    BITS=6,               # 浮栅存储位宽
    SIGMA_MISMATCH=0.01,  # 浮栅写入失配 σ=1%
    DRIFT=0.03,           # 全局温漂 ±3%
    BETA_H3=0.002,        # Gilbert 三阶失真 0.2%
    V_SAT=1.5,            # 归一化摆幅
    SIGMA_OFFSET=0.005,   # ReLU 失调 0.5%
    SNR_DB=70,            # 节点热噪声 SNR
    MC=16,                # 蒙特卡洛次数
    CHUNK=480000,         # 推理分段 10s
)

def quant6(w):
    scale = w.abs().amax(dim=tuple(range(1, w.dim())), keepdim=True).clamp(min=1e-12) / 31.0
    return (w / scale).round().clamp(-32, 31) * scale

class DeviceBatch:
    """一次蒙特卡洛采样的器件实例"""
    def __init__(self, sd, gen):
        self.g = gen
        self.sigma_n = None
        self.w = {}
        for k, v in sd.items():
            v = v.to(DEVICE)
            wq = quant6(v)
            mm = torch.normal(0, P["SIGMA_MISMATCH"], size=v.shape, generator=gen, device=DEVICE)
            drift = torch.empty(1, device=DEVICE).uniform_(-P["DRIFT"], P["DRIFT"], generator=gen)
            self.w[k] = (wq * (1 + mm) * (1 + drift)).float()

    def _noise_sig(self):
        if self.sigma_n is None:
            self.sigma_n = 0.25 / (10 ** (P["SNR_DB"] / 20))
        return self.sigma_n

    def mul_nl(self, x, w):
        """乘法器阵列: prod_ij = w_ij·x_j → Gilbert 非线性 → KCL 求和"""
        prod = x.unsqueeze(1) * w.view(1, -1, x.shape[1], 1)  # (B,Cout,Cin,T)
        xs = prod / P["V_SAT"]
        y = P["V_SAT"] * torch.tanh(prod * (1 + P["BETA_H3"] * xs.square()) / P["V_SAT"])
        return y.sum(dim=2)

    def dw_nl(self, x, w, b, dil):
        """深度卷积 3-tap 延迟线 (dilation=d): w=[w0,w1,w2] 对应 [x[t-2d],x[t-d],x[t]]"""
        n = x.shape[-1]
        xp = F.pad(x, (2 * dil, 0))
        # causal conv1d 语义: 输出[t] = w0*x[t-2d] + w1*x[t-d] + w2*x[t]
        taps = torch.stack([xp[:, :, :n], xp[:, :, dil:dil + n], xp[:, :, 2 * dil:2 * dil + n]], dim=-1)
        prod = taps * w.squeeze(1)[None, :, None, :]  # (B,C,T,3); w 需 batch 维, 否则广播成 (C,C,T,3)
        xs = prod / P["V_SAT"]
        y = P["V_SAT"] * torch.tanh(prod * (1 + P["BETA_H3"] * xs.square()) / P["V_SAT"])
        return y.sum(-1) + b[None, :, None]  # bias 是物理加法节点, 不经过乘法器

    def fwd_chunk(self, x):
        w = self.w
        h = F.conv1d(x, w["inp.weight"], w["inp.bias"], 1)
        h = h + torch.normal(0, self._noise_sig(), size=h.shape, generator=self.g, device=DEVICE)
        skip = torch.zeros_like(h)
        for i in range(LAYERS):
            dil = 2 ** (i % (DIL_EXP + 1))
            z = self.dw_nl(h, w[f"blocks.{i}.dw.weight"], w[f"blocks.{i}.dw.bias"], dil)
            z = z + torch.normal(0, self._noise_sig(), size=z.shape, generator=self.g, device=DEVICE)
            z = self.mul_nl(z, w[f"blocks.{i}.expand.weight"]) + w[f"blocks.{i}.expand.bias"][None, :, None]
            z = F.relu(z + torch.normal(0, P["SIGMA_OFFSET"], size=(1, CH, 1), generator=self.g, device=DEVICE))
            zc = self.mul_nl(z, w[f"blocks.{i}.collapse.weight"]) + w[f"blocks.{i}.collapse.bias"][None, :, None]
            zc = zc + torch.normal(0, self._noise_sig(), size=zc.shape, generator=self.g, device=DEVICE)
            h = h + zc
            skip = skip + zc * w[f"skip_gates.{i}"]
        o = F.relu(skip + torch.normal(0, P["SIGMA_OFFSET"], size=(1, 1, 1), generator=self.g, device=DEVICE))
        return F.conv1d(o, w["head.1.weight"], w["head.1.bias"])

def digital_forward(sd, x):
    """无扰数字基线 (fp32) — 顺序与训练一致: dw → expand → ReLU → collapse"""
    h = F.conv1d(x, sd["inp.weight"], sd["inp.bias"], 1)
    skip = torch.zeros_like(h)
    for i in range(LAYERS):
        dil = 2 ** (i % (DIL_EXP + 1))
        z = F.conv1d(F.pad(h, (2 * dil, 0)), sd[f"blocks.{i}.dw.weight"],
                     sd[f"blocks.{i}.dw.bias"], dilation=dil, groups=CH)
        z = F.conv1d(z, sd[f"blocks.{i}.expand.weight"], sd[f"blocks.{i}.expand.bias"], 1)
        z = F.relu(z)
        zc = F.conv1d(z, sd[f"blocks.{i}.collapse.weight"], sd[f"blocks.{i}.collapse.bias"], 1)
        h = h + zc
        skip = skip + zc * sd[f"skip_gates.{i}"]
    o = F.relu(skip)
    return F.conv1d(o, sd["head.1.weight"], sd["head.1.bias"]).squeeze(1)

def physical_forward(dev, x):
    """分段物理推理 (相邻段重叠 RF-1 采样, 延迟线历史连续无缝)"""
    RF = 1 + 2 * sum(2 ** (i % (DIL_EXP + 1)) for i in range(LAYERS))  # 541
    T = x.shape[-1]
    outs, pos = [], 0
    while pos < T:
        a = max(0, pos - RF + 1)
        seg = x[:, :, a:min(pos + P["CHUNK"], T)]
        o = dev.fwd_chunk(seg)
        drop = pos - a
        if drop > 0:
            o = o[:, :, drop:]
        outs.append(o)
        pos += P["CHUNK"]
    return torch.cat(outs, dim=-1).squeeze(1)

def nrms(pred, ref):
    return float((pred.double() - ref.double()).norm() / ref.double().norm())

def main():
    quick = "--quick" in sys.argv
    if quick:
        P["MC"] = 3
    sd = {k: v.float().to(DEVICE) for k, v in torch.load(W_PATH, map_location="cpu").items()}
    x, sr = sf.read(f"{DATA}/tsa15_test_input.wav"); assert sr == 48000
    y, _ = sf.read(f"{DATA}/tsa15_test_target.wav")
    if quick:
        x, y = x[:240000], y[:240000]
    xt = torch.from_numpy(x.astype(np.float32))[None, None].to(DEVICE)
    yt = torch.from_numpy(y.astype(np.float32)).to(DEVICE)
    n = yt.shape[0]
    print(f"device={DEVICE}  评估时长 {len(x)/48000:.1f}s  MC×{P['MC']}")

    res = {"device": DEVICE, "seconds": len(x) / 48000, "mc": P["MC"]}
    with torch.no_grad():
        o = digital_forward(sd, xt)[-n:]
        res["nrms_fp32"] = round(nrms(o, yt[-n:]), 4)
        print(f"[fp32 数字基线]  NRMS = {res['nrms_fp32']}")

        sd_q = {k: quant6(v) for k, v in sd.items()}
        o = digital_forward(sd_q, xt)[-n:]
        res["nrms_quant_only"] = round(nrms(o, yt[-n:]), 4)
        print(f"[6-bit 仅量化]   NRMS = {res['nrms_quant_only']}")

        mc = []
        for k in range(P["MC"]):
            g = torch.Generator(device=DEVICE).manual_seed(1000 + k)
            dev = DeviceBatch(sd, g)
            o = physical_forward(dev, xt)[-n:]
            r = nrms(o, yt[-n:])
            mc.append(r)
            print(f"  MC#{k+1}: NRMS = {r:.4f}")
        r = np.array(mc)
        res["nrms_physical_mean"] = round(float(r.mean()), 4)
        res["nrms_physical_std"] = round(float(r.std()), 4)
        res["nrms_physical_min"] = round(float(r.min()), 4)
        res["nrms_physical_max"] = round(float(r.max()), 4)
        print(f"[全物理层 MC]    NRMS = {r.mean():.4f} ± {r.std():.4f}  (min {r.min():.4f}, max {r.max():.4f})")

    # 元器件规模核算
    n_w = sum(v.numel() for v in sd.values())
    res["devices"] = {
        "params": n_w,
        "floating_gate_cells_2T": 2 * n_w,
        "gm_units": "dw 288 + expand/collapse 1536 ≈ 1824",
        "macs_per_sample": 1840,
        "delay_line": "256 tap × 8ch 数字 SRAM ≈ 2Kb",
    }
    print("\n=== 元器件规模核算 ===")
    print(f"权重存储: {n_w} 浮栅 2T cell ≈ {2*n_w} 管")
    print(f"计算阵列: ~1824 跨导单元 (dw 288 + expand/collapse 1536)")
    print(f"每采样 MACs: 1840 — 模拟域单周期完成")
    print(f"延迟线: 256 抽头 × 8ch 数字 SRAM ≈ 2Kb")
    print(f"对比数字方案: 同 MACs 的 16-bit 乘加阵列 ≈ 55 万管量级")

    import os
    os.makedirs(OUT, exist_ok=True)
    tag = "quick" if quick else "full"
    json.dump(res, open(f"{OUT}/result_{tag}.json", "w"), indent=2)
    print(f"\n结果已写 {OUT}/result_{tag}.json")

if __name__ == "__main__":
    main()
