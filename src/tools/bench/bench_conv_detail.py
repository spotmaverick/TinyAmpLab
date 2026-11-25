"""深入分解 conv1d 加速来源：序列长度 × 层数 × dilation。
在 Z100 与 BW 上跑完全相同的代码。"""
import sys, time
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1)


def bench(fn, warmup=3, rep=15):
    for _ in range(warmup):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(rep):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t) / rep * 1000


def test_conv(L, nlayer, dilation):
    layers = nn.ModuleList([nn.Conv1d(12, 12, 3, dilation=dilation) for _ in range(nlayer)]).to(DEV)
    x = torch.randn(1, 12, L, device=DEV)

    def step():
        for l in layers:
            if l.weight.grad is not None:
                l.weight.grad.detach_()
                l.weight.grad.zero_()
                l.bias.grad.detach_()
                l.bias.grad.zero_()
        z = x
        for l in layers:
            z = torch.tanh(l(z))
        z.sum().backward()
    ms = bench(step)
    return ms

print("BENCH_INFO torch=%s" % torch.__version__)
print("DETAIL_START")
configs = [
    (196092, 1, 1),    # 长序列单层
    (48000, 1, 1),     # 中序列单层
    (4800, 1, 1),      # 短序列单层
    (4800, 20, 1),     # 短序列20层（dilation=1）
    (196092, 20, 512), # 长序列20层大dilation
    (196092, 20, 1),   # 长序列20层（无dilation）
]
for L, nl, dil in configs:
    ms = test_conv(L, nl, dil)
    print("DETAIL L=%d layers=%d dil=%d %.2f ms" % (L, nl, dil, ms))
print("DETAIL_DONE")
