# TinyAmpLab 当前状态（2026-11-25 13:50）

> 目标：用物理元器件可实现的神经网络做数字管箱/效果器建模，在参数规模、响应时间、
> 音频连续性上争取优势，对标并力争达到/超越 NAM A2 Lite。所有训练/推理/仿真在智算服务器 DCU 上。
> 完整实验账本见 `docs/reports/MODEL_SUMMARY.md`；**最新数学审计与重设计见 `docs/reports/MATH_AUDIT_AND_REDESIGN.md`**。

## ⚠️ 本阶段四个结论（其中两个推翻了此前的判断）

### 1. 「架构效率差 2.7 倍」是错的 —— 等参差距仅 1.55 倍

四档拟合出极干净的幂律 `NRMS=1.0968·P^(-0.2975)`（R²=0.9980，四档回代残差 ≤2.2%）。
按此外推，2149 参数的 WaveNet 应为 0.1119，实测 LGW 0.1736 → **等参差距 1.55 倍**。
原来的 2.7 倍混入了 6.4 倍参数优势。

### 2. H1 证伪：depthwise 不是精度损失主因

| 证据 | 数值 |
|---|---|
| `full_ch12` s7（框架自验） | 0.078476，与官方 lite **精确到小数点后六位一致** |
| `full_ch12` s11（框架自验） | 0.075958，同样在预登记区间内 |
| `dw_ch12` 双种子（2611 参数、depthwise） | 0.1096 / 0.1073，平均 **0.1084** |
| 幂律在 2611 参数处的预测 | **0.1056** |
| **偏离** | **仅 +2.7%** |

**depthwise 精确落在参数效率幂律曲线上，未浪费任何参数。** 此前 depthwise 复现落后到 0.2207，
是协议/对齐/门控三处同时改动所致，与 conv 类型无关——我此前的归因是错的。

### 3. H2 按预登记规则不成立，但绝对精度已接近 A2 Lite

`dw_ch24`（8461 参数、7812 MACs、85.3ms）test **0.0804**，vs lite 的 0.0785 只差 **+2.4%**。
但预登记规则要求 `MACs < 3000`，实测 7812 → **H2 不成立**。不事后放宽规则。

补充说明：该约束源自「lite 的 6228 减半」的直觉，不是第一性原理；depthwise 省的是 conv 的
MACs，而 1×1 层的 ch² 增长与 conv 类型无关。

### 4. H3 成立：depthwise 短 RF 代价低于全连接

| 变体 | RF | test NRMS | 退化 |
|---|---:|---:|---:|
| `dw_ch12` → `dw_ch12_rf50` | 85.3→42.7ms | 0.1096 → 0.1182 | **+7.9%** |
| `lite` → `lite_rf50`（双种子） | 85.3→42.7ms | 0.0785 → 0.0885/0.0840 | +12.8% / +10.6% |

机理：depthwise 单 tap 依赖低于全连接（后者需 12 通道共同累加才表达一个 tap 的效果）。

## 最重要的修正：depthwise 的真实优势不在 MACs，而在交叉开关

模拟实现的主导成本不是乘法次数，而是**交叉开关规模**与**累加链深度**：

| 配置 | 交叉开关 | MACs | 最大累加链 | test |
|---|---:|---:|---:|---:|
| lite（全连接） | 4752 | 6228 | 36 | 0.0785 |
| **dw_ch24** | **936（−80%）** | 7812（+25%） | **24（−33%）** | **0.0804** |
| dw_ch12 | **324（−93%）** | 2286（−63%） | **12（−67%）** | 0.1084 |
| dw_ch12_rf50 | 324（−93%） | 2286（−63%） | 12（−67%） | 0.1182，延迟线减半 |

全连接 conv 每层要 ch×ch 的交叉开关矩阵；depthwise 只要 ch 条独立链，完全不需要交叉开关。
**dw_ch24 用 +25% MACs 换到 −80% 交叉开关、−33% 累加链，精度只差 +2.4%** —— 这是模拟实现上划算的交易。

## 一句话结论

最优解仍是 **WaveNet lite（0.0785，全器件 0.1063±0.0114）**，但**模拟友好的候选解是
`dw_ch24`（0.0804，交叉开关少 80%、累加链浅 33%）**，以及 `dw_ch12_rf50`（0.1182，
交叉开关少 93%、延迟线减半到 42.7ms）。原始目标「模拟可实现且达到 A2 效果」**首次具备可行性**。

## 已确立（有实测支撑）

| 结论 | 关键数据 |
|---|---|
| 协议红利真实 | 我方协议给官方架构 −9.7%、给 LGW −26% |
| **架构效率差距是 1.55 倍不是 2.7 倍** | 幂律 R²=0.9980，四档回代残差 ≤2.2% |
| **depthwise 参数效率达标** | dw_ch12 双种子平均 0.1084 vs 幂律预测 0.1056，偏离 +2.7% |
| **框架自验双种子通过** | full_ch12 s7=0.078476（与官方精确到六位）、s11=0.075958 |
| **depthwise 短 RF 代价更低** | +7.9% vs 全连接 +11~13% |
| **depthwise 交叉开关少 80~93%** | dw_ch24 936 vs lite 4752；dw_ch12 324 |
| **depthwise 摆幅需求低约 40%** | 同等退化下所需 Vsat 约为 lite 的 60% |
| LGW 族的扩张均为负收益 | 拉长 RF 11.3→85.3ms：0.1736→0.1893；dual 门控 4741 参数仅 0.2259 |
| lite 的 RF 拐点 | 85.3ms 0.0785 → 64ms 0.0796（+1.4%）→ 42.7ms 0.0885 → 21.4ms 0.1029 |
| 物理器件退化（两架构已测，全段 MC×16） | lite：0.0785→0.0850→**0.1063±0.0114**（Vsat 16）；LGW：0.1831→0.1976→0.3273±0.0198（Vsat 1.5） |

## 已撤回 / 未成立

- **「depthwise 是精度损失主因」（H1）**：**已证伪**，depthwise 落在幂律曲线上（+2.7%）。
- **H2（宽度补参进入低 MACs 区）**：按预登记规则**不成立**（MACs 7812 ≥ 3000），但精度达标。
- **「RF 缩放代价随容量反转」（nano −4.9%、feather −0.8%）**：补测发现 nano 种子方差 27%，是噪声。
- **「depthwise 天然可用更小摆幅」**：**部分成立**——摆幅需求确实低约 40%，但机制不同（累加链短使
  Gilbert 失真增长慢，非动态范围小）；且 depthwise 退化**饱和在 +24.2%**，lite 在 Vsat=16 仅 +11.4%
  且未饱和，物理鲁棒性略差。
- **「同等参数超越 A2 Lite」**：早期结论基于复现基线（0.2589），官方包真实水平远高于复现，不成立。

## 正在跑（DCU，待换服务器）

| 作业 ID | 变体 | 状态 |
|---|---|---|
| 49861834 | `dw_ch24_rf50` s7（H2+H3 联合） | RUNNING ep260，约 10 分钟 |

其余 6 个作业全部收敛并已回传。

## 暂停与迁移状态

- **已停止提交新作业**，等换服务器信息。
- **迁移包已就绪并端到端验证通过**：
  - `TinyAmpLab_migrate_20261125_1317.tar.gz`（512KB，MD5 `f5e88b7af71aaa7d9abbb9e2d7b087fc`）
  - `TinyAmpLab_data_20261125_1310.tar.gz`（1.1GB，MD5 `3dd5cb1c9706da5fa433058181ab4bfb`）
  - 两者都在旧服务器 `/work/home/spotmaverick/migrate_out/`
  - **已实测**：全新目录部署 → 11 项关键文件自检通过 → 数据 94 wav 就位 → 两套物理仿真 sanity 均 PASS
- **16 个数据集（94 wav，1.2GB）已全部回传本地** `data/`（此前只有 3 个，这是遗漏项，已补）。
- **`.git` 从 1.4GB 清理到 344KB**（reflog 保留的历史大对象已清除，`git fsck` 无问题）。

## 数据集（全部在本地 `data/`）

`tsa15`、`marshall_jvm410h_chod1`(+`_spk`)、`blackstar_ht1`、`bsht5m`、`englrt50`、`markv_clean`、
`markvx`、`mesa550_burn`、`mesa550_clean`、`mesa550_crunch`、`ts9_s1`、`ts9_s2`、`ua1176`、`bigmuff_diy`
——共 16 个、94 个 wav。

## 关键限制（接手者必读）

1. **H2 未按规则成立**，需决定是否调整模拟成本口径（交叉开关优先于 MACs）。
2. **WaveNet 阶梯架构不存在「只缩短 RF」的干净对照**（13 层互异 2 的幂最小和 8191）。
3. **Vsat 必须按架构动态范围定标**：lite 需 16、LGW 需 1.5；depthwise 的最优 Vsat 约 8~16。
4. **depthwise 物理鲁棒性略差**（退化饱和 +24.2% vs lite <20%），最终变体需实测不能外推。
5. 本地 Mac 无 torch；登录节点有 torch 但无 DCU，可跑 CPU 物理仿真。训练必须 sbatch 到计算节点。

## 复现入口

```bash
# 判定性实验
sbatch sbatch_analog.sh dw_ch24 300 7 tsa15

# 物理器件仿真（depthwise 版有独立的 Vsat 扫描）
python src/analog/analog_sim_dw.py --variant dw_ch12 --sanity
python src/analog/analog_sim_dw.py --variant dw_ch12 --quick --vsat-scan

# 分析工具
python src/tools/analog_verdict.py     # 判定性实验总分析
python src/tools/macs_frontier.py      # MACs-精度前沿
python src/tools/analog_cost_model.py  # 模拟成本三维分解
python src/tools/swing_compare.py      # 摆幅需求对照

# 数据转换 / 轮询 / 清理 / 迁移
python src/tools/convert_marshall.py --list
./src/tools/poll_arch2.sh 900 12
./src/tools/cleanup.sh --apply
./src/tools/migrate_pack.sh /tmp/mp
./src/tools/migrate_data.sh /tmp/mp
```

迁移详见 `docs/MIGRATION.md`。仓库：`https://github.com/spotmaverick/TinyAmpLab`（Public，SSH remote）。
