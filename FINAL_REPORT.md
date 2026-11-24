# 最终报告：自研架构超越 NAM A2 Lite（同等/更小参数）

## 结论

**目标达成，双交付**。在同一份 TSA15 管箱数据、同一指标口径（独立 test 段 NRMS/corr）下：

| 交付候选 | 参数 | test NRMS | test corr | 对比 a2lite 复现 (2178, 0.2589) |
|---|---|---|---|---|
| **效果优先：v_tcn812h1_p3** | 2149 | **0.1736** | **0.9850** | 参数 −1.3%，NRMS **−33%** |
| **预算优先：v_tcn713h1_p3** | **1855** | **0.1801** | **0.9838** | **1855 < 官方 A2 Lite 的 1871**，NRMS −30% |

6-bit 逐通道量化后：2149 档 0.1802、1855 档 0.1848——**模拟实现按 6-bit 浮栅设计仍全面超越 a2lite 的 fp32 成绩**。

（seed11 复核已确认：test NRMS 0.1831 / corr 0.9831，与 seed7 的 0.1736/0.9850 高度一致，冠军结论跨种子稳健。）

## 最终架构（LGW-TCN）

```
输入 x(t)
  → 1×1 卷积升维到 ch
  → L 层 [扩张因果深度卷积(k=3, dilation=2^0..2^7 循环) → 1×1 卷积 → ReLU] 残差块
     每层输出乘一个可学标量 skip 门控，累加成 skip 总线
  → ReLU → 1×1 卷积 → 输出 y(t)
```

- **无门控乘法器**：搜索证明预算内门控是负资产（TCN 无门控 0.2089 vs 各类门控 0.26-0.39）
- 电路映射：延迟线抽头 + Crossbar MAC + 逐层标量增益 + ReLU，无归一化/注意力/FFT
- 流式激活状态 ~8.4KB，MACs/sample ≈ 1840（2149 档实测记账）
- 训练配方：MSE、Adam、LR 3e-3 余弦退火至 1e-4、300 epoch、batch 4s（配方带来 0.2089→0.1736 的提升）

## 22 个 run 的完整搜索足迹（全部 DCU 训练）

| 阶段 | run 数 | 关键裁决 |
|---|---|---|
| P0 门控拓扑 | 7 | 等参下 classic > sgate(f·tanh g)（0.2124 vs 0.2475）；无门控 TCN 与 WaveNet 门控打平 |
| P1 预算内矩阵 | 6 | 深窄(ch4-6)全败；**宽度是预算内的生命线** |
| P2 冠军精修 | 5 | 单层头更优（0.2089）；RF 对 TCN 有害、对 dual 有益（等参确认）；LeakyReLU 对无门控路径有害 |
| P3 长训练+压线 | 4 | 300ep 余弦 → 0.1736；1855 ≤1871 → 0.1801 |

## 复现命令

```bash
# 服务器环境（wuzh02.hpccube.com:65091, Slurm wzhdtest 分区）
source /work/home/spotmaverick/nam_probe/run_env.sh
cd /work/home/spotmaverick/nam_project
python -u lgw_dcu2.py v_tcn812h1_p3 300 7 cosine   # 效果优先档
python -u lgw_dcu2.py v_tcn713h1_p3 300 7 cosine   # 预算优先档
```

- 训练脚本：`lgw_dcu2.py`（统一变体矩阵）；权重与 test 音频在 `dcu_runs/<variant>/`
- 量化分析：`quant_check2.py <variant>`（逐张量/逐通道对比）
- 全部账本：`dcu_runs/runs.csv` / `runs.md`；过程记录 `EXPERIMENT_LOG.md`

## 已知边界（诚实声明）

1. 对比基线是**同协议复现**（a2lite 拓扑同类、2178 参数；官方包因 llvmlite 编译失败未能安装），不是官方 A2 Lite 的官方训练数字；官方 A2 Lite 在 NAM 官方损失（ESR+MSTFT）与官方数据上训练，口径不同不可直接横比。
2. 单数据集（TSA15 Crunch）结论；跨管箱泛化未验证（b1m3 试探 corr 仅 +0.43）。
3. seed7 全程；seed11 复核进行中。
4. 长 sag 类动态仍非前馈架构强项（感受野 11.3ms）；本项目早前实验已证实 sag 需要状态型结构。
