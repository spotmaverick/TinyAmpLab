"""A2 与自研架构（严格对标 A2 Lite/Full 双档）

设计原则：
1. **slimmable 双档训练**：一套权重，每步随机切到某个 allowed_channels 档位训练，
   推理时按档位切片。这是 A2 官方做法，也是 Lite/Full 共存的唯一正确方式。
2. **模拟友好**：depthwise 代替全连接（交叉开关 ch² → ch），短 RF 变体。
3. 所有对比必须在**同数据同协议同种子**下，且 ≥2 种子。

核心 API：
  A2Slim      A2 官方等价（12ch，可按 3ch 切片）
  build(name) 按名称构造
  slimmable_forward  训练时随机切片
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

A2_DILATIONS = [1, 5, 29, 97, 227] * 3
A2_KERNEL = 6
A2_HEAD_SCALE = 0.01


def ratio_to_channels(ratio, max_ch):
    """官方 _ratio_to_channels: 1 + min(floor(r*max), max-1)，即 [1, max]。"""
    return 1 + min(int(ratio * max_ch), max_ch - 1)


class SlimLayer(nn.Module):
    """单层，支持通道切片。conv/mixer/x1 均按前 k 通道切片。"""

    def __init__(self, ch, cond, dilation, kernel=A2_KERNEL, depthwise=False):
        super().__init__()
        g = ch if depthwise else 1
        self.conv = nn.Conv1d(ch, ch, kernel, dilation=dilation, groups=g)
        self.mixer = nn.Conv1d(cond, ch, 1, bias=False)
        self.x1 = nn.Conv1d(ch, ch, 1)
        self.kernel = kernel
        self.depthwise = depthwise
        self.ch = ch

    def forward(self, x, c, k):
        """k = 生效通道数。权重取前 k 通道（官方 slice_channels_uniform）。"""
        w = self.conv.weight
        b = self.conv.bias
        if k < self.ch:
            if self.depthwise:
                # depthwise: weight (ch,1,K) → 取前 k 个通道
                w = w[:k]
            else:
                w = w[:k, :k]
            b = b[:k]
        z = F.conv1d(x[:, :k], w, b, dilation=self.conv.dilation,
                     padding=self.conv.padding,
                     groups=(k if self.depthwise else 1))
        mw = self.mixer.weight[:k]
        m = F.conv1d(c, mw, None)
        m = m[:, :, -z.shape[2]:]
        z = F.leaky_relu(z + m, 0.01)
        # 1x1 残差
        x1w, x1b = self.x1.weight[:k, :k], self.x1.bias[:k]
        r = F.conv1d(z, x1w, x1b)
        return x[:, :, -r.shape[2]:] + r, z


class A2Slim(nn.Module):
    """A2 官方等价 + 可选 depthwise + 通道切片。

    训练时若 slimmable=True，每次 forward 随机取某个 allowed_channels 档位。
    """

    def __init__(self, input_size=1, channels=12, cond=1, dilations=None,
                 kernel=A2_KERNEL, head_scale=A2_HEAD_SCALE, depthwise=False,
                 allowed_channels=None, slimmable=False):
        super().__init__()
        dilations = dilations or A2_DILATIONS
        self.channels = channels
        self.rechannel = nn.Conv1d(input_size, channels, 1)
        self.layers = nn.ModuleList(
            [SlimLayer(channels, cond, d, kernel, depthwise) for d in dilations]
        )
        # head: channels -> 1，取前 k 通道（官方 _SlimmableHeadRechannel: w[:, :adj]）
        self.head = nn.Conv1d(channels, 1, 1)
        self.head_scale = head_scale
        self._rf = 1 + sum((kernel - 1) * d for d in dilations)
        self.allowed_channels = allowed_channels or [channels]
        self.slimmable = slimmable
        self.depthwise = depthwise

    @property
    def receptive_field(self):
        return self._rf

    def _head(self, z, k):
        w = self.head.weight[:, :k]
        return F.conv1d(z, w, self.head.bias) * self.head_scale

    def forward(self, x, c=None, k=None):
        if x.ndim == 2:
            x = x[:, None, :]
        if c is None:
            c = x
        if k is None:
            if self.slimmable and self.training:
                k = random.choice(self.allowed_channels)
            else:
                k = self.allowed_channels[-1]
        out_len = x.shape[2] - (self._rf - 1)
        # rechannel 取前 k 通道
        rw = self.rechannel.weight[:k]
        rb = self.rechannel.bias[:k] if self.rechannel.bias is not None else None
        z = F.conv1d(x, rw, rb)
        skip = None
        for layer in self.layers:
            z, h = layer(z, c, k)
            h = h[:, :, -out_len:] if h.shape[2] >= out_len else \
                F.pad(h, (out_len - h.shape[2], 0))
            skip = h if skip is None else skip + h
        return self._head(skip, k)


def build(name):
    """名称 → (model, desc)。

    a2                  官方 A2 等价 ch12（单档）
    a2_slim             A2 官方 slimmable（ch12，档位 3/12）★主基线
    a2_dw_slim          depthwise 版 A2 slimmable
    a2_ch<n>            固定通道
    a2_dw_ch<n>         depthwise 固定通道
    ..._rf<pct>         dilation 按比例缩放
    """
    import re
    m = re.match(r"^a2(_dw)?(_slim)?(_ch(\d+))?(_rf(\d+))?$", name)
    if not m:
        raise ValueError(f"未知架构: {name}")
    dw = bool(m.group(1))
    slim = bool(m.group(2))
    ch = int(m.group(4)) if m.group(4) else 12
    pct = int(m.group(6)) if m.group(6) else 100
    dil = [max(1, round(d * pct / 100)) for d in A2_DILATIONS]

    if slim:
        # 官方比例：3/12 = 0.25
        allowed = sorted({max(1, round(ch * r)) for r in (0.25, 1.0)})
        model = A2Slim(channels=ch, dilations=dil, depthwise=dw,
                       allowed_channels=allowed, slimmable=True)
        desc = (f"A2{'深度可分离' if dw else '全连接'}slimmable ch={ch} "
                f"档位={allowed} RF={model.receptive_field}"
                f"({model.receptive_field/48000*1000:.1f}ms)")
    else:
        model = A2Slim(channels=ch, dilations=dil, depthwise=dw,
                       allowed_channels=[ch], slimmable=False)
        desc = (f"A2{'深度可分离' if dw else '全连接'} ch={ch} "
                f"RF={model.receptive_field}"
                f"({model.receptive_field/48000*1000:.1f}ms)")
    return model, desc


def active_params(model, k):
    """该档位实际使用的权重数（模拟实现成本）。"""
    m = model
    ch = m.channels
    dil_n = len(m.layers)
    p = 1 * k + k                       # rechannel
    if m.depthwise:
        p += dil_n * (k * A2_KERNEL + k)     # conv
    else:
        p += dil_n * (k * k * A2_KERNEL + k)
    p += dil_n * 1 * k                   # mixer
    p += dil_n * (k * k + k)             # 1x1
    p += k * 1 + 1                       # head
    return p


if __name__ == "__main__":
    for nm in ["a2", "a2_slim", "a2_dw_slim", "a2_dw_ch24"]:
        mod, d = build(nm)
        x = torch.randn(1, 1, 20000)
        y = mod(x)
        print(f"{nm:14s} total={sum(p.numel() for p in mod.parameters()):6d}  {d}")
        if getattr(mod, "slimmable", False):
            for k in mod.allowed_channels:
                y = mod(x, k=k)
                print(f"    k={k:2d} 首次输出={tuple(y.shape)}  "
                      f"激活参数≈{active_params(mod, k)}")
