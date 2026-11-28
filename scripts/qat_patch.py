"""P6-A：器件感知训练（QAT）—— 让权重在训练时就"看见"6-bit 量化与器件失配

## 为什么需要它（P5 分解给出的证据）

P5 的误差分解把目标拓扑的退化拆成四项（s71，V_SAT=16）：

    基线                0.1637
    − Gilbert 非线性     0.0785   ← 可用 head 增益重平衡(P5-C) + 分层摆幅(P6-B) 免费/低价消掉
    − 6-bit 量化         0.0044
    − 失配 + 漂移        0.0155   ← **校准后剩下的最大一项**
    − 激活失调           0.0066
    = 数字精度地板        0.0587

也就是说：把 Gilbert 削顶按 P5-C/P6-B 消掉之后，剩下的瓶颈是**量化 + 失配**，
这两项**事后缩放无法改善**（它们是权重本身的量化误差与工艺偏差），只能靠训练侧。

## 做法

不改官方代码，只在 `_ConvNd._conv_forward` 外面套一层"权重劣化"，用 STE 反传：

    w_eff = w + (quant_sym(w, bits) - w).detach()          # 前向量化，梯度直通
    w_eff = w_eff * mismatch * drift                       # 静态器件偏差（每 module 一份）

关键设计选择（会影响结论怎么读，必须写清）：

1. **量化口径与 DevSim 完全一致**：`quant_sym` 复用 `official_analog_sim.quant_sym`
   的同一公式（逐输出通道对称量化、levels = 2^(bits-1)-1），否则训练与评估不自洽。
2. **失配/漂移是静态的**（每个 module 在首次前向时抽样一次，之后固定），
   因为工艺失配在流片后就固定了，不是每步重抽的噪声。
   代价是会"记住"这一颗芯片；为了检验泛化，评估时在 MC 0~3 四颗芯片上分别测，
   若 MC 0（训练用的那颗）明显好于 MC 1~3，说明过拟合到单颗芯片——这本身是结论。
3. **只在 training 模式生效**。validation 保持全精度，使 val_loss 仍是数字精度参考，
   与历史 13 种子实验的曲线可直接对比。真正的器件精度由 DevSim 后验管线测。

## 开关

由环境变量 `NAM_QAT` 控制（JSON），`train_seeded.py` 在调用官方 main 之前启用：

    NAM_QAT='{"bits":6}'                                   # 只做量化感知
    NAM_QAT='{"bits":6,"sigma":0.01,"drift":0.03}'         # 量化 + 失配/漂移感知
    NAM_QAT='{"bits":6,"sigma":0.01,"drift":0.03,"chip_seed":1}'

不设该环境变量时，本模块不产生任何行为改变（默认关闭，历史实验可复现）。
"""
import json
import os

import torch
import torch.nn as nn

# 注意：`_conv_forward` 的**实际实现**在 `Conv1d` / `Conv2d` 上，`_ConvNd` 里只是
# abstractmethod。只 patch `_ConvNd` 不会生效（Conv1d 自己覆盖了它）——这里必须
# 逐个具体类替换，并保存原函数用于还原。
_PATCHED = {}
_ORIG = {}


_STATE = {
    "on": False,
    "bits": 64,
    "sigma": 0.0,
    "drift": 0.0,
    "chip_seed": 0,
    "n": 0,
}


def quant_sym(w, bits):
    """逐输出通道对称量化（与 official_analog_sim.quant_sym 同一公式）。"""
    if w.dim() == 1 or bits >= 32:
        return w
    levels = 2 ** (bits - 1) - 1
    dims = tuple(range(1, w.dim()))
    scale = w.abs().amax(dim=dims, keepdim=True).clamp(min=1e-12) / levels
    return (w / scale).round().clamp(-(levels + 1), levels) * scale


def _device_bias(module, weight):
    """每个 module 的静态失配/漂移（首次前向抽样后固定 = 一颗芯片）。

    种子用**首次前向的全局序号**而不是 id(module)：id 跨进程不稳定，会让"同一 chip_seed"
    在不同 run 里抽到不同的失配，失去可复现性。前向顺序在固定结构下是确定的。
    """
    if (getattr(module, "_qat_mm", None) is None
            or module._qat_mm.shape != weight.shape
            or module._qat_mm.device != weight.device):
        if not hasattr(module, "_qat_idx"):
            module._qat_idx = _STATE["n"]
            _STATE["n"] += 1
        g = torch.Generator(device="cpu").manual_seed(
            _STATE["chip_seed"] * 100003 + module._qat_idx)
        mm = (torch.empty(weight.shape, device="cpu")
              .normal_(0, _STATE["sigma"], generator=g) if _STATE["sigma"] > 0 else
              torch.zeros(weight.shape, device="cpu"))
        dr = (torch.empty(1, device="cpu").uniform_(
            -_STATE["drift"], _STATE["drift"], generator=g) if _STATE["drift"] > 0 else
            torch.zeros(1, device="cpu"))
        module._qat_mm = (1.0 + mm).to(weight.device)
        module._qat_dr = (1.0 + dr).to(weight.device)
    return module._qat_mm, module._qat_dr


def _make_qat(cls):
    """为具体卷积类生成 _conv_forward 包装（保留该类的原函数）。"""
    orig = _ORIG[cls]

    def wrapper(self, input, weight, bias):
        if not _STATE["on"] or not self.training or weight is None:
            return orig(self, input, weight, bias)
        w = weight
        if _STATE["bits"] < 32:
            w = w + (quant_sym(w, _STATE["bits"]) - w).detach()
        if _STATE["sigma"] > 0 or _STATE["drift"] > 0:
            mm, dr = _device_bias(self, w)
            w = w * mm * dr
        return orig(self, input, w, bias)

    return wrapper


def enable(bits=64, sigma=0.0, drift=0.0, chip_seed=0):
    _STATE.update(on=True, bits=int(bits), sigma=float(sigma),
                  drift=float(drift), chip_seed=int(chip_seed), n=0)
    for cls in (nn.Conv1d, nn.Conv2d):
        if cls not in _ORIG:
            _ORIG[cls] = cls._conv_forward
        cls._conv_forward = _make_qat(cls)
        _PATCHED[cls] = True
    print(f"[qat_patch] enabled bits={bits} sigma={sigma} drift={drift} "
          f"chip_seed={chip_seed} classes={list(_PATCHED)}", flush=True)


def enable_from_env(var="NAM_QAT"):
    """由环境变量启用；未设置则完全不介入。"""
    raw = os.environ.get(var)
    if not raw:
        return None
    cfg = json.loads(raw)
    enable(**cfg)
    return cfg


def disable():
    _STATE["on"] = False
    for cls, orig in _ORIG.items():
        cls._conv_forward = orig
    _PATCHED.clear()
