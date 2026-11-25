"""对等 benchmark：Z100(torch1.10) 与 BW(torch2.9) 跑完全相同的代码。
输出统一格式 BENCH <name> <ms> 便于脚本解析。
"""
import sys, time
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("BENCH_INFO torch=%s dev=%s name=%s" % (torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", DEV))
if DEV == "cuda":
    print("BENCH_INFO vram_gb=%.1f" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))

torch.manual_seed(1)


def bench(fn, warmup=3, rep=20):
    for _ in range(warmup):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(rep):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t) / rep * 1000  # ms


class Net20(nn.Module):
    """对照实验的 20 层 conv1d，与 BW 上 8.1ms/step 的 benchmark 完全一致。"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv1d(12, 12, 3, dilation=1 << (i % 10)) for i in range(20)])

    def forward(self, x):
        for l in self.layers:
            x = torch.tanh(l(x))
        return x.sum()


def run_conv_bench(L):
    m = Net20().to(DEV)
    x = torch.randn(1, 12, L, device=DEV)

    def step():
        m.zero_grad()
        y = m(x)
        y.backward()
    ms = bench(step)
    print("BENCH conv20_fwd_bwd L=%d %.2f ms" % (L, ms))
    return ms


def run_full_step(L):
    """完整训练步：fwd + mse + bwd + Adam step，最接近真实训练的一 step。"""
    m = Net20().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = torch.randn(1, 12, L, device=DEV)
    tgt = torch.randn(1, device=DEV)

    def step():
        opt.zero_grad()
        y = m(x)
        loss = ((y - tgt) ** 2).mean()
        loss.backward()
        opt.step()
    ms = bench(step)
    print("BENCH full_step L=%d %.2f ms" % (L, ms))
    return ms


def run_matmul(N):
    """纯算力 benchmark：不受 Python 循环影响。"""
    a = torch.randn(N, N, device=DEV)
    b = torch.randn(N, N, device=DEV)

    def step():
        return a @ b
    ms = bench(step, warmup=3, rep=10)
    tflops = 2 * N**3 / (ms / 1000) / 1e12
    print("BENCH matmul N=%d %.2f ms %.3f TFLOPS" % (N, ms, tflops))
    return ms, tflops


print("BENCH_START")
run_conv_bench(196092)   # 真实训练的输入长度（win = RF + BATCH - 1）
run_conv_bench(48000)    # 短序列，看长度缩放
run_full_step(196092)
run_matmul(4000)         # 对照历史记录 Z100 4000x4000 = 3.36 TFLOPS
print("BENCH_DONE")
