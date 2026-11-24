# 本轮交付概览：取消 Block 范式约束后的架构重设计与实测优选

## 结论

用户取消"Block 范式不变"限制后，我设计四个候选架构做公平对照（统一数据/种子/超参），并用三个种子 + 两个数据集验证。**胜出方案：双组门控（dual gate），40 Block × 4×4 Crossbar，640 权重全部有效**。实测 NRMS 相比上一代架构在 TSA15 独立测试集上从 0.785 降到 0.650。

## 关键发现：瓶颈是有效权重数，不是感受野

我最初假设瓶颈是 2.5ms 感受野装不下 sag，设计了扩张抽头（RF 扩到 40ms）和输出侧电源状态。**实验推翻了这个假设**：

| 变体 | 权重 | val NRMS（3 种子均值） |
| --- | --- | --- |
| uniform 基线 | 320 | 0.871 |
| dilated 扩张抽头 | 320 | ≈0.92（单种子，无收益） |
| **dual 双组门控** | **640** | **0.806** |
| supply 电源状态 | 680 | 单种子 train 0.587 / val 0.906（过拟合，弃用） |

- 扩张抽头把感受野放大 16 倍却无收益 → 感受野不是瓶颈。
- 双组门控把之前悬空的 320 个死权重接成第二组 sum/gate → 显著且跨种子稳健的改善。这与此前发现的"K×K 只用 2 行、1/3 权重从未训练"完全对应。
- 输出侧电源状态虽修正了物理方向（softplus 保证只压缩不放大），但严重过拟合，性价比低，不采用。

## 正式训练结果

| 数据集 | 指标 | 上一代(K4B40+包络) | **双组门控** |
| --- | --- | --- | --- |
| DAFx23 | corr | +0.714 | **+0.779** |
| DAFx23 | NRMS | 0.701 | **0.627** |
| TSA15 train | NRMS | 0.752 | 0.578 |
| TSA15 val | NRMS | 0.741 | 0.630 |
| **TSA15 test** | **NRMS** | **0.785** | **0.650** |
| TSA15 test | corr | 0.639 | **0.770** |

- DAFx23：800 epoch，Train T 0.183→0.0284，无 NaN。
- TSA15：100 epoch（大数据集早收敛），Train T 0.0425→0.0264；train/val/test 三段独立评估，test 未参与训练。
- 训练/推理前向一致性：8.3e-7（WAV 量化级）。权重全部有限。

## 电路代价

4×4 Crossbar 单元数不变（640），每 Block 增加 1 路乘法器 + 1 路 tanh（第二组 sum/gate），总线宽不变。相比增加 Block 数或扩张抽头，这是最低代价的扩容方式，且全部 640 权重都参与计算（不再有 1/3 死权重）。

## 方法论要点

单种子对比会得出错误结论：seed3 下 uniform 只有 0.925（初始化运气差），seed7/11 下 uniform 本就能到 0.844。三个种子的相对排序才可信——dual 在全部三个种子上都胜出（0.833/0.810/0.774）。这也是本轮坚持做多种子验证的原因。

## 文件与复现

- `ARCH_SEARCH.md`：完整实验记录（假设、方法、结果、判断）
- `arch_search.py`：四变体对照脚本；`arch_search_dafx.log`、`uniform_800.log`、`r2_*.log`、`r3_*.log`：原始日志
- `train_nam.py` / `infer_only.py`：已固化为双组门控正式版本
- `trained_weights_dual40.npy`（DAFx23）/ `trained_weights_dual40_tsa15.npy`：正式权重
- `student_dual40.wav` / `student_dual40_tsa15.wav`：拟合输出样例

```bash
/opt/anaconda3/bin/python train_nam.py dafx 7 800       # DAFx23 复现
/opt/anaconda3/bin/python train_nam.py tsa15 7 100      # TSA15 复现
/opt/anaconda3/bin/python infer_only.py                 # 纯 NumPy 推理
```

## 未采用 / 后续

- 扩张抽头、输出电源状态：实测无稳健收益，未固化。
- 若还要继续降 NRMS：可考虑增加 Block 数（会拉长时序链）或用真实状态型模型（需更大训练数据，5 秒数据训不动 supply 已经证明）。
- 浮栅扰动仿真、单 Block 转移曲线导出供 LTspice 仍未做。
