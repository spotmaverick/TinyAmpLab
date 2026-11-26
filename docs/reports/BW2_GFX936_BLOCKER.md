# bw2 为什么跑不了 NAM 训练（gfx936 缺 code object）

**结论**：bw2 的 GPU 对 PyTorch 不可用，且不是环境配置问题 —— 现有 wheel 的二进制里没有 gfx936 的 code object，任何 kernel 都无法下发。
**复现时间**：2026-11-26 08:00–08:16，作业 1218056 / 1218060 / 1218065 / 1218072 / 1218107。

---

## 一、事实链

| 检查 | 命令 / 作业 | 结果 |
|---|---|---|
| 连通、队列 | `dcu.sh -m bw2 ping` | ✅ 3s，分区 `hx1hdnormal` |
| torch 可用性 | `~/mc310/bin/python -c import torch` | ✅ 2.1.0（需先 `export LD_LIBRARY_PATH=""; source $DTKROOT/env.sh`，否则 `libgalaxyhip.so.5` 找不到） |
| 卡被识别 | `rocm-smi` / `hipGetDeviceCount` | ✅ `Name=BW`，65520 MiB，n=1 |
| **卡的 gfx 目标** | `hipcc` 编译 `hipGetDeviceProperties` 探针（1218072） | **`gcnArchName=gfx936:sramecc+:xnack-`，cc 9.3** |
| **wheel 内架构** | `torch.cuda.get_arch_list()` | **`['gfx906','gfx926','gfx928']`** —— 无 gfx936 |
| 实际计算 | `randn(256,256).sum()` / `Conv1d` | ❌ `RuntimeError: HIP error: invalid device function`（1218056） |
| 架构覆盖绕过 | `HSA_OVERRIDE_GFX_VERSION=9.0.6` | ❌ 段错误，rc=139，无输出（1218107） |
| 集群 wheel 仓库 | `/public/software/apps/DeepLearning/whl` | ❌ 在 bw2 上不存在；`~/whl_py310` 是从 z100 拷来的同一份 |

`get_arch_list()` 只列 gfx906/926/928，而设备要 gfx936 —— 这是二进制缺 ISA 的直接证据，与 `LD_LIBRARY_PATH`、DTK 版本、`--mem`、module 体系都无关。

## 二、可走的四条路（按代价排序）

| 路线 | 代价 | 说明 |
|---|---|---|
| **向管理员要 BW 版 wheel** | 等待，最稳 | 需按 gfx936 编译的 torch 2.x + torchaudio；拿到后 `pip install --no-deps` 装入 `~/mc310` 即可 |
| **自编译 torch（gfx936）** | 1~2 天，风险高 | 需 torch 源码 + DTK 的 clang 支持 `--offload-arch=gfx936`（未验证）+ 海光算子补丁；无 vendor 补丁时即便编过也可能算子不对 |
| **apptainer 镜像** | 半天，取决于有没有现成镜像 | bw2 的 `apptainer` 不在默认 PATH，需先确认可用；镜像内必须自带 gfx936 版 torch |
| 依赖 z100 | 零成本 | 短序列 NAM 在 z100 单卡与 BW 单卡速度接近（L≤4800 时约 1.1×），bw2 的优势是 712 卡并发，对 5 个作业无意义 |

## 三、给下次的提醒

- **不要在 bw2 上重试「换 DTK 版本 / 改 LD_LIBRARY_PATH / 加 module load」** —— DTK 24.04.2/24.04.3/25.04.x/26.04 全部试过，报错完全相同。
- `torch.cuda.is_available()` 在 bw2 上返回 **True**，不要用它判断可用性；正确的判据是**跑一次 `randn().sum()`**。
- 探针脚本已留在 `bw2:~/probe/gpucheck.py`（torch 层）与 `~/probe/devarch.cpp`（HIP 层读 gcnArchName），环境就绪后直接复跑即可。
