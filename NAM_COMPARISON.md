# 与 NAM 的效果对比 + 智算服务器接入

日期：2026-11-23

## 一、问题

此前所有实验都是在我们自己设计的架构之间比较（23×3 → K4B40+包络 → 双组门控），
从未与真正的 NAM 模型做过对照。用户问"现在效果和 NAM 比较如何"，
因此用**完全相同的数据和指标口径**补做 NAM 基线。

## 二、对比设置（公平性）

| 项 | 设置 |
|---|---|
| 数据 | TSA15 真实电子管箱，train 152s / val 20s / test 38s（独立未见段），48kHz float32 |
| 划分 | 全部模型共用同一组 wav，test 从训练中完全隔离 |
| 指标 | NRMS = ‖pred−target‖/‖target‖，corr = 去均值Pearson |
| 优化 | Adam lr=3e-3，seed=7，100 epoch，4s/minibatch，38 batch/epoch |
| 我方模型 | 双组门控 40 Block × 4×4 Crossbar，640 权重（已固化在 train_nam.py） |

## 三、NAM 基线实现

NAM 官方 `neural-amp-modeler` 包无法安装（llvmlite 在 Python 3.12 编译失败），
因此按 NAM 公开定义复现两个架构（代码：`nam_baseline.py` / `nam_wavenet_baseline.py`）：

1. **NAM standard**：16 采样输入窗口 + 4 层 MLP（hidden 24）+ 线性 skip，1650 参数。
2. **NAM WaveNet**：因果 dilated conv + gated tanh·sigmoid + residual/skip 堆叠，
   channels=8 / layers=10，2505 参数（对标 A1 standard 拓扑，权重规模接近 A2-Full 量级）。

架构澄清：NAM 官方当前主流是 **A1 WaveNet** 与 **A2 Slimmable**（2026-06 起的新标准，
Lite 1871 / Full 12146 权重）；**没有官方"A3"档位**。

## 四、结果

| 模型 | 参数 | val corr | val NRMS | **test corr** | **test NRMS** |
|---|---|---|---|---|---|
| dry 基线（未处理输入） | — | — | — | 0.568 | — |
| **我方双组门控** | 640 | 0.779 | 0.630 | **0.770** | **0.650** |
| NAM standard MLP | 1,650 | 0.835 | 0.550 | **0.876** | **0.483** |
| NAM WaveNet（DCU，训练中） | 2,505 | 0.920 (ep20) | 0.393 (ep20) | Slurm 作业 49695455 运行中，完成后填 | 同左 |

### 结论

- 我们的双组门控与真正的 NAM 有**明显差距**：test NRMS 0.650 vs NAM MLP 0.483，
  corr 0.770 vs 0.876。即使用最老的 NAM standard 架构，也显著优于我们。
- NAM WaveNet 在 ep20 已达 val 0.393，差距进一步拉大。
- 差距来源不是参数量（我们 640 vs NAM MLP 1650，但 WaveNet 2505 也远超），
  主要是 NAM 的输入窗口（16 采样）与 skip 直连结构对波形塑形更高效；
  我们的逐 Block 前馈级联在同等参数下表达效率更低。
- **不能宣称当前方案接近 A2/A3 音色表现**——实测证据相反。

## 五、智算服务器接入（已完成验证）

| 项 | 值 |
|---|---|
| SSH | `wuzh02.hpccube.com:65091`，user `spotmaverick`，RSA 密钥（有效期至 2027-02-21） |
| 登录节点 | login05（无 DCU，仅作提交入口） |
| 调度 | Slurm，分区 `wzhdtest`（3 天时限，38 节点） |
| DCU | **海光深算一号 C878180，17.2GB HBM**，单作业 `--gres=dcu:1` |
| PyTorch | 1.10.0a0+dtk22.04，HIP 4.3 |
| 算力实测 | 4000×4000 矩阵乘法 **3.36 TFLOPS** |

### 环境加载要点（无外网，需手动拼接）

```bash
module purge
module load compiler/devtoolset/7.3.1 compiler/dtk/22.04.1 \
            mpi/hpcx/gcc-7.3.1 python/3.8.10
# 递归把 /public/software/apps/DeepLearning/PyTorch_Lib 下所有含 .so 的目录加入 LD_LIBRARY_PATH
export PYTHONPATH=/work/home/spotmaverick/pylibs:$PYTHONPATH   # typing_extensions.py（从集群拷）
```

- 系统预置的 `apps/Pytorch/1.10.0` module 是坏的（指向不存在的 py37 目录）；
  实际用 `whl/dtk-22.04.1/torch-1.10.0a0+git450cdd1.dtk22.4-cp38-...whl` 手动 `--user` 安装。
- 计算节点无外网（DNS 不通），pip 装不了新包；纯 Python 依赖需从集群其他环境拷贝。
- 作业必须加 `-c 8 --mem=16G`，否则内存按默认 CPU 比例配额不足会 OOM。
- wav 是 IEEE float（fmt tag=3），Python 3.8 的 `wave` 模块不认，需手动解析 RIFF data chunk。

### 速度对比

同一 NAM WaveNet 基线：本地 CPU（4 核）约 3–4 分钟/epoch；DCU 约 11 秒/epoch，**快约 20 倍**。
数据已上传至 `/work/home/spotmaverick/nam_project/`，训练作业 49695455 运行中。

---

## 五、官方包正式对比（2026-11-24 更新）⚠️ 口径修正

**关键发现：NRMS 与官方 esr 的换算关系：NRMS = sqrt(esr)**
（NRMS = ||p−t||/||t|| = sqrt(mean(p−t)²/mean(t²)) = sqrt(官方 esr)）

### 官方包真实成绩（TSA15，官方协议 lr0.004/γ0.993/batch16/100ep/MSE，官方包 0.13.0 CPU 训练）

| 模型 | 参数 | 官方 esr (test) | **NRMS (test)** |
|---|---|---|---|
| 官方 nano | 841 | 0.031359 | **0.1771** |
| 官方 feather | 3025 | 0.010292 | **0.1015** |
| 官方 lite | 6553 | 训练中 | — |
| 官方 standard | 13801 | 训练中 | — |
| **我方 LGW-TCN（旧协议成绩）** | 2149 | 0.0301 | **0.1736** |

### 结论修正（此前基于复现基线的结论部分失效）
- 此前"a2lite 复现 2178 参数 NRMS 0.2589"远弱于官方包真实协议，导致我方架构搜索结论虚高
- 官方 feather（3025 参数）NRMS 0.1015 大幅优于我方 LGW-TCN 2149（0.1736）
- 我方 2149 仅与官方 nano（841）相当量级——**架构效率需按官方协议重新校准**

### DCU 移植对拍（等价性已验证）
- 官方 WaveNet 架构（四级）已在 torch 1.10 DCU 移植完成（`wavenet_official_dcu.py`）
- 对拍：同一 model.nam 权重，官方实现 vs 移植实现 ESR 差 <5e-5（float32 精度），**架构还原 100%**
- 对拍通过后：官方架构 + 我方架构全部在 DCU 上以官方协议重训/继续训练

### 后续（E2 已启动）
- LGW-TCN 冠军以官方协议重训（DCU 49763956），分离协议与架构贡献
- 官方四级架构 DCU 重训（49763160-64），100ep 官方协议
- 待出结果后：按官方口径重排架构搜索优先级
