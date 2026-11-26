"""证据固化：A2 slimmable 死锁的完整验证

结论：官方 `nam_full_configs/models/wavenet_slimmable.json` 使用
`init_strategy=smallest_and_zeros`，该策略在初始化时把"超出最小通道数"的
weight 与 bias 全部置零。由于每个 layer 的 `_input_mixer` 是 `bias=False`，
被置零通道的输入恒为 0，梯度也恒为 0，因此 **大通道分支永久不更新**。

改为官方另一个策略 `init_strategy=channel_causal` 后完全正常。

运行：python diag_deadlock.py
"""
import copy
import json
import sys
from pathlib import Path

import torch

from nam.models.wavenet._wavenet import WaveNet

CFG = Path("nam_full_configs/models/wavenet_slimmable.json")


def grad_at(model, x, ratio):
    model.zero_grad()
    model.set_slimming(ratio)
    out = model(x)
    target = torch.zeros_like(out)
    ((out - target) ** 2).mean().backward()
    return sum(
        p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None
    )


def main():
    cfg = json.load(open(CFG))
    torch.manual_seed(0)
    x = torch.randn(1, 1, 6000)

    print("=" * 70)
    print("证据 1：两种 init_strategy 的梯度活性")
    print("=" * 70)
    for strat in ("smallest_and_zeros", "channel_causal"):
        c = copy.deepcopy(cfg["net"]["config"])
        c["layers_configs"][0]["slimmable"]["kwargs"]["init_strategy"] = strat
        m = WaveNet.init_from_config(c)
        m.train()
        print(f"\n{strat}:")
        for r in (0.0, 0.4, 0.5, 0.6, 0.8, 1.0):
            g = grad_at(m, x, r)
            flag = "  <<< 梯度为零：死锁" if g == 0.0 else ""
            print(f"  ratio={r:.1f}  total_grad={g:.4e}{flag}")

    print()
    print("=" * 70)
    print("证据 2：用 smallest_and_zeros 训练后导出权重的零值比例")
    print("=" * 70)
    for nam in sorted(Path("runs").glob("cand_a2/*/model.nam")):
        d = json.load(open(nam))
        w = d["weights"]
        nz = sum(1 for v in w if v != 0)
        print(f"{nam}: total={len(w)}  nonzero={nz}  zero={len(w) - nz} "
              f"({(len(w) - nz) / len(w) * 100:.1f}% 为零)")

    print()
    print("=" * 70)
    print("证据 3：大通道权重是否严格为零（逐层）")
    print("=" * 70)
    ckpts = sorted(Path("runs/cand_a2").glob("*/lightning_logs/version_*/checkpoints/*.ckpt"))
    if not ckpts:
        print("（无 ckpt，跳过）")
        return
    ck = ckpts[-1]
    cfg_p = ck.parent.parent.parent.parent / "config_model.json"
    m = WaveNet.init_from_config(json.load(open(cfg_p))["net"]["config"])
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    clean = {}
    for k, v in sd.items():
        while k.startswith("_net."):
            k = k[len("_net."):]
        if "_has_sample" in k or "_sample_rate" in k:
            continue
        clean[k] = v
    m.load_state_dict(clean, strict=False)
    zero_cnt = nonzero_cnt = 0
    for _, p in m.named_parameters():
        if p.dim() >= 2 and p.shape[0] == 12:
            if p[3:].abs().sum().item() == 0:
                zero_cnt += 1
            else:
                nonzero_cnt += 1
    print(f"12-输出维参数张量：大通道全零 {zero_cnt} 个 / 大通道有非零 {nonzero_cnt} 个")
    print("=> 若 nonzero_cnt == 0，则 12 通道分支从未被训练")


if __name__ == "__main__":
    sys.exit(main())
