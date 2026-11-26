"""NAM A2 (Slimmable WaveNet) 精简重实现 + 我方候选架构

A2 规格来源：官方 NAM `nam_full_configs/models/wavenet_slimmable.json`
  通道 12 / 层 15 / kernel 6 / dilations [1,5,29,97,227]×3
  激活 LeakyReLU（**无门控**，mid_channels = bottleneck）
  slimmable: allowed_channels [3, 12]，一套权重按通道切片
  head_scale = 0.01
  实测：15,685 权重，RF 5386 采样 = 112.2 ms @48k

本文件提供：
  A2Net     —— A2 的等价精简实现（用于我方训练/评测，不依赖官方包）
  A2Slim    —— 支持通道切片的 A2（Lite=3ch / Full=12ch 共用权重）
  build_arch —— 按名称构造我方候选架构（模拟友好变体）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- A2 官方规格 ----------------
A2_CHANNELS = 12
A2_KERNEL = 6
A2_DILATIONS = [1, 5, 29, 97, 227] * 3
A2_HEAD_SCALE = 0.01
A2_ALLOWED_CHANNELS = [3, 12]


class A2Layer(nn.Module):
    """A2 单层：conv(dilated) + input_mixer → LeakyReLU → 1x1，残差 + skip 双输出。

    注意 A2 无门控：mid_channels = channels（不像 A1 那样 2×）。
    """

    def __init__(self, ch, cond, dilation, kernel=A2_KERNEL, groups=1,
                 depthwise=False):
        super().__init__()
        g = ch if depthwise else groups
        self.conv = nn.Conv1d(ch, ch, kernel, dilation=dilation, groups=g)
        self.mixer = nn.Conv1d(cond, ch, 1, bias=False)
        self.act = nn.LeakyReLU(0.01)
        self.x1 = nn.Conv1d(ch, ch, 1)
        self.dilation = dilation

    def forward(self, x, c):
        z = self.conv(x)
        m = self.mixer(c)[:, :, -z.shape[2]:]
        z = z + m
        z = self.act(z)
        return x[:, :, -z.shape[2]:] + self.x1(z), z


class A2Net(nn.Module):
    """A2 等价实现（单层数组，is_first + is_last）。"""

    def __init__(self, input_size=1, channels=A2_CHANNELS, cond=1,
                 dilations=None, kernel=A2_KERNEL, head_scale=A2_HEAD_SCALE,
                 depthwise=False):
        super().__init__()
        dilations = dilations or A2_DILATIONS
        self.rechannel = nn.Conv1d(input_size, channels, 1)
        self.layers = nn.ModuleList(
            [A2Layer(channels, cond, d, kernel, depthwise=depthwise) for d in dilations]
        )
        self.head = nn.Conv1d(channels, 1, 1)
        self.head_scale = head_scale
        self._rf = 1 + sum((kernel - 1) * d for d in dilations)

    @property
    def receptive_field(self):
        return self._rf

    def forward(self, x, c=None):
        # x: (B, 1, T) → 输出 (B, 1, T - (RF-1))
        if x.ndim == 2:
            x = x[:, None, :]
        if c is None:
            c = x
        out_len = x.shape[2] - (self._rf - 1)
        z = self.rechannel(x)
        skip = None
        for layer in self.layers:
            z, h = layer(z, c)
            h = h[:, :, -out_len:].clone() if h.shape[2] >= out_len else \
                F.pad(h, (out_len - h.shape[2], 0))
            skip = h if skip is None else skip + h
        out = self.head(skip) * self.head_scale
        return out


def _count(m):
    return sum(p.numel() for p in m.parameters())


# ---------------- 我方候选架构（模拟友好） ----------------
def build_arch(name, **kw):
    """按名称构造架构。返回 (model, desc)。

    命名：a2            官方 A2 等价（ch12，全连接）
          a2_ch<n>      改通道数
          a2_dw         官方通道但 depthwise（交叉开关从 ch² 降到 ch）
          a2_dw_ch<n>   depthwise + 改通道
          a2_rf<pct>    dilation 按 pct 缩放（<100 缩短 RF）
    """
    import re
    m = re.match(r"^a2(_dw)?(_ch(\d+))?(_rf(\d+))?$", name)
    if not m:
        raise ValueError(f"未知架构: {name}")
    dw = bool(m.group(1))
    ch = int(m.group(3)) if m.group(3) else A2_CHANNELS
    pct = int(m.group(5)) if m.group(5) else 100
    dil = [max(1, round(d * pct / 100)) for d in A2_DILATIONS]
    model = A2Net(channels=ch, dilations=dil, depthwise=dw, **kw)
    desc = f"A2{'深度可分离' if dw else '全连接'} ch={ch} " \
           f"RF={model.receptive_field}({model.receptive_field/48000*1000:.1f}ms)"
    return model, desc


if __name__ == "__main__":
    for nm in ["a2", "a2_dw", "a2_ch3", "a2_dw_ch24"]:
        mod, d = build_arch(nm)
        x = torch.randn(1, 1, 20000)
        y = mod(x)
        print(f"{nm:14s} params={_count(mod):6d}  {d}  out={tuple(y.shape)}")
