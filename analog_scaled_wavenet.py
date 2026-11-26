"""模拟原生 WaveNet：逐层增益渐变（Layer Gain Taper），抑制幅度爆炸

## 为什么需要它

器件仿真实测（`ANALOG_SIM_FINDINGS.md` + 2026-11-26 逐层诊断）：

    a-grp4 残差流 RMS: 0.049 -> 22.6（460×）
    逐层 preact 幅度: 0.24 -> 13.2（55×），head 累加后 30.8
    Gilbert cell 在 V_SAT=16 下饱和 -> 精度退化 45×

关键发现（2026-11-26 晚，推翻了 α 方案）：

    官方 `_Layer.forward` 有两条独立路径
        residual    = x + layer1x1(post_activation)     # 过 layer1x1
        head_output = post_activation                   # **不过 layer1x1**
    所以把 α 乘在 layer1x1 上只压住了残差链，
    head 累加路径完全不受影响 —— 实测 V_SAT 需求没有下降（仍需 64）。

## 本架构：在激活输出处做逐层增益渐变

    正确的作用点是「每层激活之后」，一个点同时覆盖两条路径：

        residual    = x + γ_i · layer1x1(z_i)
        head_output = γ_i · z_i

    γ_i 随层号递减（前期层 γ≈1，后期层 γ 小）。物理意义：

        γ_i = 第 i 层激活输出后的固定电阻分压比
        —— 模拟电路里是**零成本**的（一个电阻比值），
           却能换来动态范围需求的数量级下降。

## 为什么用「渐变」而不是固定 α

固定 α 会被训练完全补偿：layer1x1 权重可以整体放大 1/α 抵消掉缩放。
实测 α=0.5 只把残差增长从 480× 降到 270×（预期 4× 实际 1.8×）。

渐变 γ_i 则是**结构性约束**：后期层要补偿得连续放大 1/γ，15 层累计
是天文数字，梯度会爆炸，训练自然不会走这条路。同时前期层幅度本来就
很小（0.05~0.5），不需要压制，所以溺水点放在后期。

## 配置

    residual_scale : float   兼容旧配置，等价于所有 γ_i 相同
    layer_gains    : dict | list
        {"mode": "exp",   "start": 1.0, "end": 0.3}   指数递减（推荐）
        {"mode": "const", "value": 0.5}               常数
        [1.0, 0.9, 0.8, ...]                          显式列表

## 用法

    from analog_scaled_wavenet import ScaledWaveNet
    from nam.models import factory
    factory.register("ScaledWaveNet", ScaledWaveNet.init_from_config, overwrite=True)
"""
import math
from typing import Any, Dict, List, Optional

import torch

from nam.models.base import BaseNet as _BaseNet
from nam.models.wavenet._wavenet import WaveNet as _WaveNet


def _resolve_gains(cfg: Dict[str, Any], n_layers: int) -> List[float]:
    """把配置里的增益描述解析成逐层列表。"""
    lg = cfg.get("layer_gains")
    if lg is None:
        alpha = float(cfg.get("residual_scale", 1.0))
        return [alpha] * n_layers

    if isinstance(lg, (list, tuple)):
        gains = [float(g) for g in lg]
        if len(gains) != n_layers:
            raise ValueError(
                f"layer_gains 长度 {len(gains)} != 层数 {n_layers}"
            )
        return gains

    mode = str(lg.get("mode", "exp"))
    if mode == "const":
        return [float(lg.get("value", 1.0))] * n_layers
    if mode == "exp":
        start = float(lg.get("start", 1.0))
        end = float(lg.get("end", 0.3))
        if start <= 0 or end <= 0:
            raise ValueError("layer_gains 的 start/end 必须 > 0（exp 模式取对数）")
        if n_layers == 1:
            return [start]
        ratio = end / start
        return [start * ratio ** (i / (n_layers - 1)) for i in range(n_layers)]
    raise ValueError(f"未知 layer_gains.mode: {mode}")


def _make_scaled(orig_forward, gain: float):
    """闭包工厂（避免循环变量捕获问题）。"""
    def fwd(*args, **kw):
        return gain * orig_forward(*args, **kw)
    return fwd


class ScaledWaveNet(_BaseNet):
    """包装官方 WaveNet，在每层激活输出处乘一个固定增益 γ_i。

    实现方式：替换 `_layer_arrays[*]._layers[*]._activation.forward`。
    不新增参数、不改权重形状，导出的 .nam 与官方格式完全兼容
    （增益是前向常数，可从 config 的 layer_gains 还原）。
    """

    def __init__(self, wavenet: _WaveNet, layer_gains: Optional[List[float]] = None,
                 sample_rate: float = 48000.0):
        super().__init__(sample_rate=sample_rate)
        self._net = wavenet
        n_layers = sum(len(la._layers) for la in wavenet._layer_arrays)
        gains = list(layer_gains) if layer_gains is not None else [1.0] * n_layers
        if len(gains) != n_layers:
            raise ValueError(f"layer_gains 长度 {len(gains)} != 层数 {n_layers}")
        self._layer_gains = gains
        self._patch()

    @property
    def layer_gains(self) -> List[float]:
        return list(self._layer_gains)

    def _patch(self):
        li = 0
        for layer_array in self._net._layer_arrays:
            for layer in layer_array._layers:
                gain = self._layer_gains[li]
                li += 1
                if gain == 1.0:
                    continue
                act = getattr(layer, "_activation", None)
                if act is None:
                    continue
                act.forward = _make_scaled(act.forward, gain)

    @property
    def receptive_field(self) -> int:
        return self._net.receptive_field

    @property
    def pad_start_default(self) -> bool:
        return True

    def _forward(self, x, **kw):
        # BaseNet.forward 传入 2D (B, L) 并已完成 pad_start；
        # 内部 WaveNet 需要 3D (B, 1, L)，返回 (B, 1, L) -> 压成 (B, L)
        if x.ndim == 2:
            x = x[:, None, :]
        y = self._net(x)
        return y[:, 0, :] if y.ndim == 3 else y

    # ---- 与官方一致性所需的接口 ----

    def _export_config(self):
        cfg = self._net.export_config(sample_rate=self.sample_rate)
        cfg["layer_gains"] = list(self._layer_gains)
        return cfg

    def _export_weights(self):
        return self._net.export_weights()

    def import_weights(self, weights, i: int = 0) -> int:
        return self._net.import_weights(weights, i)

    @classmethod
    def init_from_config(cls, config: Dict[str, Any]):
        cfg = dict(config)
        cfg.pop("residual_scale", None)   # 由 layer_gains 统一表达
        lg = cfg.pop("layer_gains", None)
        inner = _WaveNet.init_from_config(cfg)
        n_layers = sum(len(la._layers) for la in inner._layer_arrays)
        # 用含 layer_gains 的临时 dict 解析
        gains = _resolve_gains({"layer_gains": lg}, n_layers)
        return cls(inner, layer_gains=gains)


if __name__ == "__main__":
    # 冒烟测试：验证两条路径都被缩放
    from nam.models.wavenet._wavenet import WaveNet

    base = {
        "layers_configs": [{
            "condition_size": 1, "input_size": 1, "channels": 6,
            "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
            "kernel_size": 3, "dilations": [1, 2, 4, 8],
            "activation": "LeakyReLU", "groups_input": 3,
        }],
        "head_scale": 0.01,
    }

    torch.manual_seed(0)
    ref = WaveNet.init_from_config(dict(base))
    x2d = torch.randn(1, 300)
    x3d = x2d[:, None, :]          # 内部 WaveNet 要 (B, 1, L)
    with torch.no_grad():
        y_ref = ref(x3d)

    # 同一个 net 对象上挂 γ=0.5，比较两条路径是否都被缩放
    m = ScaledWaveNet(ref, layer_gains=[0.5] * 4)
    with torch.no_grad():
        y_half = m(x2d)            # ScaledWaveNet 走 BaseNet 接口 (B, L)
    with torch.no_grad():
        y_half3 = ref(x3d)         # 已 patch，直接调内层

    print(f"ref(3D)   out rms={y_ref.pow(2).mean().sqrt():.6e}")
    print(f"gamma.5   out rms={y_half.pow(2).mean().sqrt():.6e}  (BaseNet 2D 接口)")
    print(f"gamma.5   out rms={y_half3.pow(2).mean().sqrt():.6e}  (内层 3D 接口)")
    ratio = (y_half3.pow(2).mean() / y_ref.pow(2).mean()).sqrt().item()
    print(f"幅度比 = {ratio:.4f}   (gamma=0.5 时预期 < 1，验证缩放生效)")
    print(f"gains={[round(g,3) for g in m.layer_gains]}")
    print(f"params={sum(p.numel() for p in m.parameters())} rf={m.receptive_field}")

    # exp 模式解析
    g = _resolve_gains({"layer_gains": {"mode": "exp", "start": 1.0, "end": 0.3}}, 15)
    print("exp 1.0->0.3 over 15:", [round(v, 3) for v in g])
