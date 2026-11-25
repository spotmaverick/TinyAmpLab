# TinyAmpLab — 微型吉他管箱神经建模实验室

轻量级吉他管箱（guitar amp）神经建模研究项目：**架构搜索 × 官方 NAM 基准 × DCU 训练 × 物理器件级仿真**。

---

## 从哪里开始看

| 你想知道 | 看这里 |
|---|---|
| 项目当前在做什么、下一步 | [`overview.md`](overview.md) |
| 全部模型架构与参数实验结果（**主账本**） | [`docs/reports/MODEL_SUMMARY.md`](docs/reports/MODEL_SUMMARY.md) |
| 与官方 NAM 的口径与对比 | [`docs/reports/NAM_COMPARISON.md`](docs/reports/NAM_COMPARISON.md) |
| 当前 WaveNet 架构族并行搜索计划 | [`docs/plans/ARCH_V2_PLAN.md`](docs/plans/ARCH_V2_PLAN.md) |
| 三诊断实验裁决与 bug 记录 | [`docs/plans/DIAG_PLAN_REVIEW.md`](docs/plans/DIAG_PLAN_REVIEW.md) |
| 如何跑起来 | 本文件下方「复现」章节 |

> **注意**：`docs/reports/FINAL_REPORT.md` 是早期阶段报告，标题里的「超越 A2 Lite」结论已被官方包对比推翻，仅作历史记录。以 `MODEL_SUMMARY.md` 为准。

---

## 当前结论（截至 2026-11-25）

统一口径：`NRMS = ‖pred−target‖ / ‖target‖ = sqrt(官方 esr)`。数据集 TSA15，test 段 38.1s。

| 模型 | 参数 | RF | 协议 | test NRMS |
|---|---:|---:|---|---:|
| 官方 standard | 13801 | 85.3ms | 官方 100ep | 0.0723 |
| **官方 standard** | 13801 | 85.3ms | **我方 300ep** | **0.0653** |
| 官方 lite | 6553 | 85.3ms | 我方 300ep | 0.0785 |
| 官方 standard | 13801 | 42.7ms | 我方 300ep | 0.0727 |
| 官方 lite | 6553 | 64.0ms | 我方 300ep | 0.0796 |
| 官方 lite | 6553 | 42.7ms | 我方 300ep | 0.0885 |
| 官方 lite | 6553 | 21.4ms | 我方 300ep | 0.1029 |
| 官方 feather | 3025 | 85.3ms | 我方 300ep | 0.1014 |
| 官方 feather | 3025 | 42.7ms | 我方 300ep | 0.1005 |
| 官方 nano | 841 | 85.3ms | 我方 300ep | 0.1485 |
| 官方 nano | 841 | 42.7ms | 我方 300ep | 0.1412 |
| LGW-TCN 冠军 | 2149 | 11.3ms | 我方 300ep | 0.1736 |

**已验证结论**：

1. **协议红利**：我方协议（300ep 余弦 + batch192000 + lr3e-3）对官方架构（−9.7%）和 LGW（−26%）都有效。
2. **架构效率的差距是 1.55 倍，不是 2.7 倍**（2026-11-25 数学审计修正）：四档拟合出幂律 `NRMS = 1.0968·P^(−0.2975)`（R²=0.9980）。按此外推，2149 参数的 WaveNet 应为 0.1119，而实测 LGW 是 0.1736——等参后差距仅 1.55 倍，原来的 2.7 倍混入了 6.4 倍参数优势。
3. **LGW 族的 RF 与门控扩张均为负收益**（等参单变量）：拉长 RF 11.3→85.3ms，0.1736→0.1893；dual 门控 4741 参数仅 0.2259。
4. **lite 的 RF 拐点在 64ms 与 42.7ms 之间**（seed 7 四点曲线）：85.3ms 0.0785 → 64ms 0.0796（+1.4%，几乎无损）→ 42.7ms 0.0885（+12.8%）→ 21.4ms 0.1029（+31.1%）。
5. **大容量档同样依赖长 RF**：standard 缩到 42.7ms 退化 +11.3%（0.0653→0.0727），与 lite +12.8% 同向。
6. **depthwise 不是精度损失的主因**（2026-11-25 判定性实验，框架自验精确到小数点后六位）：`dw_ch12`（2611 参数、2286 MACs、depthwise）双种子 test 0.1096 / 0.1073，平均 0.1084，而幂律在 2611 参数处的预测是 **0.1056**——**偏离仅 +2.7%，depthwise 未浪费任何参数**。此前 depthwise 复现落后到 0.2207，是协议/对齐/门控三处同时改动所致，与 conv 类型无关。

> **已撤回的结论**：曾据 seed 7 单点写出「短 RF 代价随容量反转（nano −4.9% 反而更好）」。补测发现 nano 种子方差达 27%（s7 0.1412 vs s11 0.1793），该结论被噪声淹没；且 `_rf50` 的 dilation 缩放产生重复序列（13 层仅 9 个唯一值），混淆了 RF 与结构两个变量。详见 `docs/reports/MODEL_SUMMARY.md` G 节。

**模拟友好拓扑（depthwise）对照表**：

| 变体 | 参数 | MACs | RF | test NRMS | MACs vs lite |
|---|---:|---:|---:|---:|---:|
| full_ch12（官方 lite 复刻，框架自验） | 6553 | 6228 | 85.3ms | 0.0785 | 基准 |
| dw_ch12 s7 | 2611 | 2286 | 85.3ms | 0.1096 | **−63%** |
| dw_ch12 s11 | 2611 | 2286 | 85.3ms | 0.1073 | **−63%** |

depthwise 以 63% 更少的 MACs 达到了同等参数效率。按幂律推算，达到 A2 Lite 的 0.0852 需约 5368 参数（约 4400 MACs，比 lite 少 29%）。完整论证见 `docs/reports/MATH_AUDIT_AND_REDESIGN.md` 第七节。

**物理器件级仿真**（全器件非理想，MC×16，38.1s 全段）：

| 模型 | fp32 | 6-bit 量化 | 全器件非理想 | 模拟摆幅 |
|---|---:|---:|---:|---:|
| LGW-TCN 冠军（2149 参数，RF 11.3ms） | 0.1831 | 0.1976 | 0.3273 ± 0.0198 | Vsat 1.5 |
| WaveNet lite（6553 参数，RF 85.3ms） | 0.0785 | 0.0850 | **0.1063 ± 0.0114** | **Vsat 16** |

lite 的物理退化更小（+25% vs LGW +66%）、绝对精度更好，但代价是 **3× 器件规模 + 约 10× 模拟摆幅**。摆幅不足时 Gilbert 三次失真会让 NRMS 恶化到 0.6361——这是真实工程约束，不是脚本问题。

一个待验证的优势：depthwise 的残差流累加深度远小于全连接 conv（每层累加 3 个 tap 而非 12 通道），**可能天然需要更小的模拟摆幅**。若成立，depthwise 会在精度、MACs、摆幅三个维度同时优于全连接 lite——这正是模拟实现最需要的性质。

lite 的仿真脚本是 `src/analog/analog_sim_wavenet.py`。它与 LGW 版的差别不只是权重，**算子也按官方 WaveNet 重写**：全连接 3-tap conv + tanh + 1×1 残差，而非 depthwise + ReLU。这点对物理实现有实际意义——差分对饱和天然产生 tanh，官方 WaveNet 的激活正好用它，不需要额外整流电路。脚本带两级自检：`--sanity` 先与官方 `OfficialWaveNet` 逐样本对拍（抓自家前向实现的错误），再关掉全部非理想性验证物理路径等于数字路径。

## 数据集

| 数据集 | 来源 | 时长（train/val/test） | 用途 |
|---|---|---|---|
| `tsa15` | 已有 | 152.5 / 20.6 / 38.1 s | 主实验集，全部结论基于它 |
| `marshall_jvm410h_chod1` | Zenodo 10892012，前级输出 | 210 / 30 / 60 s | 新音箱泛化验证 |
| `marshall_jvm410h_chod1_spk` | 同上，喇叭输出（含功放） | 210 / 30 / 60 s | 全链路泛化验证 |

Marshall 数据集 2.76 GB，MD5 与官方值一致、全条目 CRC 通过。**该数据集没有独立 test 录音**（test 目录的 9 个录音都是其他旋钮组合，与建模任务不同），因此 trainval 300s 按时序切 70/10/20，test 取末尾 60s。标准设置 `B050_M050_T050_G050` 是各旋钮中间值，符合 NAM 一权重一设置的惯例。另有 28 组旋钮扫描 × 双采集点可用（`convert_marshall.py --list` / `--convert-all`），供多设置联合建模。

---

## 目录结构

```
.
├── README.md                    本文件：导航 + 结论摘要
├── overview.md                  当前状态总览（作业/下载进度）
│
├── docs/                        所有文档
│   ├── reports/                 结论性报告
│   │   ├── MODEL_SUMMARY.md     ★ 全部实验结果主账本
│   │   ├── NAM_COMPARISON.md    官方 NAM 对比与口径
│   │   └── FINAL_REPORT.md      早期报告（已被推翻，历史记录）
│   └── plans/                   计划与裁决
│       ├── ARCH_V2_PLAN.md      当前 WaveNet 架构族搜索
│       ├── DIAG_PLAN_REVIEW.md  三诊断裁决 + bug 记录
│       ├── ARCH_SEARCH.md       早期架构搜索足迹
│       └── EXPANSION_PLAN.md    数据扩展与官方基线计划
│
├── *.py / *.sh                  活跃入口脚本（详见下方「脚本索引」）
│
├── src/                         辅助与历史代码
│   ├── baselines/               NAM 基线复现（MLP/WaveNet/推理）
│   ├── experiments/             一次性诊断、消融、早期推理
│   ├── analog/                  物理器件仿真脚本
│   └── tools/                   数据转换、同步、轮询工具
│
├── experiments/                 实验产物（结果/权重/日志）
│   ├── lgw_search/runs/         22 组 LGW 搜索账本与权重
│   ├── diagnostics/diag3/       d1/d2/d3 结果 JSON
│   ├── architecture_v2/         WaveNet 架构族搜索结果（进行中）
│   ├── analog_sim/              物理仿真结果 JSON
│   └── early_block/             早期 Gated-Block 权重与音频
│
├── data/                        本地工作数据（48kHz float32 wav）
├── dataset_zenodo/              原始 zip 与下载/转换工具
├── sync/                        从智算回传的数据/权重/结果
├── github_mirrors/              官方 NAM 源码镜像（仅参考）
└── nam_baseline/                早期基线数据

（大文件 wav/pt/npy 不入库，由 .gitignore 排除）
```

---

## 脚本索引

**活跃入口**（项目根目录，直接可跑）：

| 脚本 | 用途 |
|---|---|
| `arch_search_v2.py` | WaveNet 架构族搜索（当前主线），支持 `<arch>_rf<scale>` |
| `wavenet_official_dcu.py` | 官方 WaveNet 四档 DCU 训练 + `.nam` 权重对拍 |
| `diag_three.py` | 三诊断实验（已完成，保留复现能力） |
| `lgw_dcu2.py` | LGW-TCN 架构搜索（22 run，历史主线） |
| `lgw_official_dcu.py` | LGW-TCN + 官方协议训练 |
| `train_nam.py` | 早期 23 级 Gated-Block 行为级仿真 |

**辅助脚本**：

| 脚本 | 用途 |
|---|---|
| `src/analog/analog_sim.py` | LGW-TCN 冠军物理器件仿真（DCU） |
| `src/analog/analog_sim_wavenet.py` | WaveNet lite 物理器件仿真（`--sanity` 含两级对拍） |
| `arch_search_v2.py` / `sbatch_arch2.sh` | WaveNet 架构族搜索（支持第 4 参数指定数据集） |
| `src/tools/convert_new_amps.py` | ToneTwist AFx 数据统一转换 |
| `src/tools/convert_ua.py` / `convert_marshall.py` | UA1176 / Marshall 数据转换（后者支持 `--list`/`--convert-all`） |
| `src/tools/sync_back.sh` | 智算服务器 → 本地结果回传 |
| `src/tools/poll_dcu.sh` / `poll_arch2.sh` | DCU 作业轮询 |
| `src/tools/cleanup.sh` | 缓存清理（默认预演；`--apply` 执行；已装 pre-commit 钩子自动跑） |

---

## 数据集

全部来自 [ToneTwist AFx Dataset](https://github.com/mcomunita/tonetwist-afx-dataset)（Zenodo），统一转 48kHz 单声道 float32，train/val/test 三段，命名 `<tag>_{train,val,test}_{input,target}.wav`。

已覆盖 12+ 设备：TSA15、Engl RT50、Mesa MarkV Extreme、Blackstar HT5M/HT1、Mesa 550 三通道、MarkV Clean、TS9、Big Muff、UA1176、Marshall JVM410H（下载中）。

wav 不入库。按 `dataset_zenodo/dl_list.txt` 的 record ID 下载后，用 `src/tools/convert_*.py` 转换。

---

## 环境

| 侧 | 配置 |
|---|---|
| DCU 训练 | 曙光智算 wuzh02，海光深算一号，DTK 22.04，torch 1.10，Slurm `-p wzhdtest -c 8 --mem=16G --gres=dcu:1` |
| 官方包对比 | Miniconda py3.10 + torch 2.6.0+cpu + neural-amp-modeler 0.13.0（`pip install --only-binary=:all:`） |
| 本地 Mac | Python 3.12 / torch 2.2.2 / soundfile 0.14 |

---

## 复现

```bash
# 1. 官方 WaveNet 架构族搜索（我方协议，300ep）
python arch_search_v2.py lite 300 7

# 2. 官方协议训练（100ep）
python wavenet_official_dcu.py standard tsa15 100

# 3. 官方权重对拍（验证 DCU 移植等价性）
python wavenet_official_dcu.py --eval <model.nam> tsa15

# 4. LGW-TCN 官方协议
python lgw_official_dcu.py v_tcn812h1 tsa15 100

# 5. 物理器件级仿真（DCU）
python src/analog/analog_sim.py              # LGW-TCN 冠军
python src/analog/analog_sim_wavenet.py --sanity   # WaveNet lite，先做两级对拍
python src/analog/analog_sim_wavenet.py            # 全量 MC×16

# 6. 检查作业进度
./src/tools/poll_dcu.sh 180 20
./src/tools/poll_arch2.sh 900 12            # 架构族 + Marshall 泛化作业

# 7. 清理缓存（默认只预演，不删）
./src/tools/cleanup.sh                 # 看将清理什么
./src/tools/cleanup.sh --apply         # 真删
./src/tools/cleanup.sh --apply --gc    # 顺带回收 git 对象
```

---

## License

MIT
