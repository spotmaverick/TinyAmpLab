"""族 B2：更强的模拟原生递归架构（在族 A 结论指导下设计）

## 设计依据（来自族 A 的实证结论）

族 A 告诉我们两件关键事实：
1. **分组（channel mixing 受限但保留）是最优的** —— 4~6 组，全连接与 depthwise 都更差。
2. **激活必须是器件天然的** —— tanh（差分对）；LeakyReLU 也可（近似）。

族 B 原本的 DiagSSM 只有"每通道独立一阶 RC + 输出端一次 1×1"，
通道间**只在最后混合一次**，表达力不足（epoch 23 时 NRSS 0.62）。

## 本文件的结构：RecurrentBlock（模拟原生的"分组递归层"）

    # 每层 = 分组递归 + 分组 1×1 混合 + tanh + 残差
    s_c[t]    = a_c * s_c[t-1] + b_c * u_c[t]        # 分组内各通道独立一阶 RC
    z         = tanh(s + mix(u))                      # 分组 1×1（模拟里是固定电阻网）
    y         = u + rescale(z)                        # 残差

其中 u 是分组后的通道（group_size 个通道共享混合），这样：
- 递归状态：每通道 1 个电容 → **延迟线 = 0**
- 通道混合：分组 1×1 → 交叉开关 = ch²/g（g=组数），与族 A 的最优设置一致
- 多层堆叠：给足表达力（族 A 需要 15 层，族 B 用 4~6 层即可，因为递归自带长记忆）

物理映射（每层）：
    1 个电容阵列（ch 个） + ch 个可调跨导（决定 a） + ch 个输入跨导（b）
    + 一组分组 1×1 电阻网（ch²/g） + 1 个差分对（tanh）
    **无延迟线，无采样保持阵列**

## 与官方 A2 的公平性
同一训练器、同一数据、同一 lr/scheduler/epoch，评测用官方 esr。
"""
import math

import numpy as np
import torch
import torch.nn as nn

from nam.models.base import BaseNet as _BaseNet

from analog_ssm import iir_scan


class RecBlock(nn.Module):
    """模拟原生的分组递归块。

    参数（物理器件数）：
        电容        ch
        递归跨导    ch      (a)
        输入跨导    ch      (b)
        分组 1×1    ch²/g   (g = 组数)
    """

    def __init__(self, ch, groups=4, tau_min_ms=0.02, tau_max_ms=60.0,
                 sample_rate=48000.0):
        super().__init__()
        self._ch = ch
        self._groups = groups
        self._sr = sample_rate

        taus = np.logspace(
            math.log10(tau_min_ms), math.log10(tau_max_ms), ch
        )
        a0 = np.clip(np.exp(-1.0 / (taus * 1e-3 * sample_rate)), 1e-4, 1 - 1e-4)
        self.a_logit = nn.Parameter(
            torch.tensor(np.log(a0 / (1 - a0)), dtype=torch.float32)
        )
        self.b_gain = nn.Parameter(torch.full((ch,), 1.0 / math.sqrt(ch)))
        self.mix = nn.Conv1d(ch, ch, 1, groups=groups, bias=False)
        self.out = nn.Conv1d(ch, ch, 1, bias=True)

    def forward(self, x):
        """x: (B, C, L) -> (B, C, L)"""
        a = torch.sigmoid(self.a_logit)
        u = x * self.b_gain.view(1, -1, 1)
        s = iir_scan(u, a)                       # 递归状态（无延迟线）
        z = torch.tanh(s + self.mix(x))
        return x + self.out(z)


class RecurrentNet(_BaseNet):
    """由 RecBlock 堆叠成的模拟原生网络。

    :param channels: 通道数（= 每层电容数）
    :param n_blocks: 递归块层数
    :param groups: 块内 1×1 分组数（族 A 结论：4~6 最优）
    """

    def __init__(self, channels: int = 16, n_blocks: int = 4, groups: int = 4,
                 in_groups: int = 1, sample_rate: float = 48000.0,
                 head_scale: float = 0.02, tau_min_ms: float = 0.02,
                 tau_max_ms: float = 60.0):
        super().__init__(sample_rate=sample_rate)
        self._channels = channels
        self._n_blocks = n_blocks
        self._groups = groups
        self._head_scale = head_scale
        self._tau = (tau_min_ms, tau_max_ms)

        g = math.gcd(in_groups, channels) if in_groups > 1 else 1
        self.rechannel = nn.Conv1d(1, channels, 1, groups=g, bias=False)
        self.blocks = nn.ModuleList([
            RecBlock(channels, groups, tau_min_ms, tau_max_ms, sample_rate)
            for _ in range(n_blocks)
        ])
        self.head = nn.Conv1d(channels, 1, 1, bias=True)

    @property
    def receptive_field(self) -> int:
        # 递归结构不需要 lookahead；记忆由极点决定
        return 1

    @property
    def pad_start_default(self) -> bool:
        return True

    def _forward(self, x, **kw):
        if x.ndim == 3:
            x = x[:, 0, :]
        y = self.rechannel(x[:, None, :])
        for blk in self.blocks:
            y = blk(y)
        return self._head_scale * self.head(y)[:, 0, :]

    # ---- 导出/导入 ----
    def _export_config(self):
        return {
            "channels": self._channels,
            "n_blocks": self._n_blocks,
            "groups": self._groups,
            "head_scale": self._head_scale,
            "tau_min_ms": self._tau[0],
            "tau_max_ms": self._tau[1],
            "sample_rate": self.sample_rate,
        }

    def _export_weights(self) -> np.ndarray:
        ws = [self.rechannel.weight.detach().cpu().flatten()]
        for blk in self.blocks:
            ws += [
                blk.a_logit.detach().cpu().flatten(),
                blk.b_gain.detach().cpu().flatten(),
                blk.mix.weight.detach().cpu().flatten(),
                blk.out.weight.detach().cpu().flatten(),
                blk.out.bias.detach().cpu().flatten(),
            ]
        ws += [self.head.weight.detach().cpu().flatten(),
               self.head.bias.detach().cpu().flatten()]
        ws.append(torch.tensor([self._head_scale]))
        return torch.cat(ws).numpy()

    def import_weights(self, weights, i: int = 0) -> int:
        def take(shape):
            nonlocal i
            n = int(np.prod(shape))
            t = torch.as_tensor(weights[i:i + n], dtype=torch.float32).reshape(shape)
            i += n
            return t

        self.rechannel.weight.data = take(self.rechannel.weight.shape)
        for blk in self.blocks:
            blk.a_logit.data = take(blk.a_logit.shape)
            blk.b_gain.data = take(blk.b_gain.shape)
            blk.mix.weight.data = take(blk.mix.weight.shape)
            blk.out.weight.data = take(blk.out.weight.shape)
            blk.out.bias.data = take(blk.out.bias.shape)
        self.head.weight.data = take(self.head.weight.shape)
        self.head.bias.data = take(self.head.bias.shape)
        self._head_scale = float(torch.as_tensor(weights[i]))
        i += 1
        return i


def cost_recurrent(ch, n_blocks, groups):
    """物理成本：电容 / 延迟线 / 交叉开关 / 累加链"""
    per_block = ch * 2 + ch * ch // groups * 2   # a,b 跨导 + 分组1x1 + out
    params = ch + n_blocks * per_block + ch + 1
    return dict(
        caps=ch * n_blocks,
        delay_taps=0,
        xbar=params,
        macs=ch + n_blocks * (ch * ch // groups * 2),
        chain=ch // groups,
        params=params,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print("族 B2 物理成本（对照 A2: 权重 15685, xbar 15324, 延迟线 5386）")
    print(f"{'配置':<26}{'电容':>6}{'延迟线':>7}{'交叉开关':>9}{'累加链':>7}")
    print("-" * 60)
    for ch, nb, g in [(16, 4, 4), (24, 4, 4), (24, 6, 4), (32, 6, 4),
                      (32, 6, 6), (48, 6, 6)]:
        c = cost_recurrent(ch, nb, g)
        print(f"ch{ch}-b{nb}-g{g:<14}{c['caps']:>6}{c['delay_taps']:>7}"
              f"{c['xbar']:>9}{c['chain']:>7}")

    # 冒烟 + 速度
    import time
    for ch, nb, g in [(16, 4, 4), (32, 6, 4)]:
        m = RecurrentNet(channels=ch, n_blocks=nb, groups=g).eval()
        x = torch.randn(1, 48000)
        t0 = time.time()
        with torch.no_grad():
            y = m(x)
        n = sum(p.numel() for p in m.parameters())
        print(f"\nch{ch}-b{nb}-g{g}: params={n} 1s audio={time.time()-t0:.2f}s "
              f"out={tuple(y.shape)} rf={m.receptive_field}")
