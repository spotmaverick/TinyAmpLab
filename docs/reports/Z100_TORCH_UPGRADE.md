# Z100 升级 torch 的可行性与收益实测（2026-11-25）

> 回答用户问题：**8 张 Z100 同时可用是优势；Z100 能否靠升级 torch 提升性能？**
> 结论先行：**能升级，也有收益（2.3~2.8×），但 8 卡 Z100 升级后总吞吐仍不及单卡 BW。**

## 一、可行性：能升级，但要换软件栈组合

旧卡平台提供完整的 DTK 版本矩阵（21.04 → 26.04），wheel 仓库在
`/public/software/apps/DeepLearning/whl/dtk-*/pytorch/`：

| 组合 | torch | Python | 状态 |
|---|---|---|---|
| 现状 DTK22.04 | 1.10.0a0 | 3.8 | 已用 |
| DTK23.04 | 1.13.1 | 3.8 | 可选 |
| **DTK24.04** | **2.1.0+das1.0.dtk2404** | **3.10** | **本次实测成功** |

搭建步骤（已验证）：
1. `conda create -p torch21_env python=3.10 --offline`（conda 缓存有 py3.10.12/18）
2. 重命名 wheel 去掉多余 `+` 号（新版 pip 拒绝多 `+` 版本号），`pip install --no-deps`
3. `conda install -c defaults openssl=1.1`（torch2.1 需要 libssl1.1，系统只有 1.0.2/3.0）
4. 从 nam 环境复制纯 Python 依赖（typing_extensions/sympy/networkx/jinja2/filelock/numpy）
5. **关键坑**：`LD_LIBRARY_PATH` 不能含 conda env 的 lib（旧 MKL 符号冲突），
   只保留 `torch/lib` + openssl；numpy 需 <2（torch2.1 按 numpy1.x 编译）

作业脚本：`nam_probe/bench_t21_v3.slurm`，Slurm 作业 49912972 / 49913113。
设备名从 `C878180` 变为 `Z100SM`（DTK24 新命名），**确认 Z100 硬件支持 DTK24**。

## 二、实测收益：conv 快 2.2×，但 dilation 几乎没改善

同一 benchmark 脚本（MD5 一致）：

| 测试项 | Z100 torch1.10 | Z100 torch2.1 | 加速 | BW torch2.9 |
|---|---:|---:|---:|---:|
| conv20 fwd+bwd L=196092 | 359.58 ms | 160.42 ms | **2.24×** | 8.09 ms |
| full step | 360.73 ms | 159.79 ms | 2.26× | 8.30 ms |
| matmul 4000² | 9.45 TFLOPS | 9.59 TFLOPS | **1.01×** | 49.96 TFLOPS |
| 20层 conv dil=1 | 252.92 ms | 28.79 ms | **8.79×** | 8.36 ms |
| **20层 conv dil=512** | **852.23 ms** | **646.45 ms** | **1.32×** | **8.04 ms** |

**两个关键发现**：

1. **纯算力纹丝不动**（matmul 9.45→9.59 TFLOPS）：升级 torch 不改变硬件算力，
   符合预期。
2. **dil=1 快 8.8×，但 dil=512 只快 1.3×**：torch 版本修复了普通长序列卷积的
   实现，但**没有修复大 dilation 的访存缺陷**。这正是本项目最依赖的路径
   （RF=4093，dilation 最大 512），所以升级收益被大幅限制。

对照 BW：dil=512 只要 8.04ms，比 Z100 torch2.1 快 **80×**。
这部分不是 torch 版本能补的——是硬件/驱动层的访存能力差异。

## 三、8 卡 Z100 升级后 vs 单卡 BW

训练吞吐估算（两种口径，避免单点假设）：

| 口径 | Z100 升级收益 | 单卡 s/ep | 8 卡吞吐 | vs 单卡 BW |
|---|---:|---:|---:|---:|
| 实测 full_step 缩放 | 2.26× | 13.57 s/ep | 0.589 ep/s | **0.62×** |
| 混合 dilation 估算 | 2.82× | 10.85 s/ep | 0.737 ep/s | **0.78×** |
| BW 实测（基准） | — | 1.06 s/ep | 0.947 ep/s | 1.00× |

> 混合口径假设 20 层里 3 层是大 dilation（与本项目 D1/D2 分布一致）；
> full_step 口径用实测完整训练步缩放，但它的 Net20 只用 dil=1，可能低估收益。
> 两者给出 0.62~0.78× 的区间，**结论稳健：8 卡 Z100 升级后仍不及单卡 BW**。

旧栈对照：8 卡 Z100 吞吐 0.261 ep/s，只有单卡 BW 的 0.28×。

### 为什么 8 卡也追不上

单卡差距 29×，8 卡只能补 8×，剩下 3.6× 的缺口。升级 torch 补 2.3~2.8×，
仍差 1.3~1.6×。这是"广度换不来深度"的典型案例：
**当单卡性能差近一个数量级时，卡数优势（8×）不够填平。**

## 四、什么情况下 8 卡 Z100 反而更划算

虽然训练吞吐不及，但 8 卡仍有其价值：

| 场景 | 建议 |
|---|---|
| 大 batch 多卡数据并行（模型复制到 8 卡） | Z100 集群理论加速接近 8×，但需要 NCCL/HCCL 通信验证 |
|  embarrassingly parallel 的超参搜索 | 8 个独立作业可同时跑，总吞吐 8× 单卡 Z100 |
| 显存需求 > 单卡 | Z100 16GB vs BW 64GB，反而是劣势 |
| 本项目训练（重 dilation、长序列） | **BW 明显更优**，即使只有 1 卡 |
| 短序列小 conv（L≤4800） | 两卡几乎同速，8 卡 Z100 完胜 |

**结论**：如果目标是继续当前 TinyAmpLab 路线（长序列 + 重 dilation），
升级 torch 是性价比不高的投入——收益 2.3× 但仍追不上 BW，且搭建环境花了
约 1 小时。建议直接在 BW 上用并发多进程（单卡 6 并发已实现 1.20 ep/s）。

## 五、可复现性

- 旧卡环境：`/work/home/spotmaverick/torch21_env/`（torch2.1 + py3.10 + deps）
- 作业脚本：`/work/home/spotmaverick/nam_probe/bench_t21_v3.slurm`（含完整环境设置）
- benchmark 脚本：`src/tools/bench/bench_peer.py`、`bench_conv_detail.py`
- 本地对照：BW 上 `/tmp/bench_peer.py`，同为 MD5 校验一致的副本
