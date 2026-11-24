# TinyAmpLab — 微型吉他管箱神经建模实验室

轻量级吉他管箱（guitar amp）神经建模的**架构搜索 × 官方基准 × DCU 训练管线**研究项目。

在真实电子管箱录音数据上，系统对比官方 Neural Amp Modeler (NAM) 全部模型等级与自研轻量架构（LGW-TCN），全部训练/推理迁移至海光 DCU（曙光智算，DTK 22.04）。

## 核心成果

### 官方包基准（neural-amp-modeler 0.13.0，TSA15 数据，官方协议 100 epochs）

| 模型 | 参数 | 官方 esr (test) | NRMS (test) |
|---|---|---|---|
| 官方 nano | 841 | 0.0314 | 0.1771 |
| 官方 feather | 3025 | 0.0103 | 0.1015 |
| 官方 lite | 6553 | 0.0073 | 0.0852 |
| 官方 standard | 13801 | 见 results | 见 results |
| **LGW-TCN（官方协议）** | 2149 | 0.0558 | 0.2361 |
| **LGW-TCN（300ep 余弦协议）** | 2149 | — | **0.1736** |

**口径说明**：`NRMS = ‖pred−target‖ / ‖target‖ = sqrt(官方 esr)`。两种口径必须换算后才能横比。

**关键发现**：
1. 官方 WaveNet 架构（nano/feather/lite）在同协议下参数效率显著高于 LGW-TCN——841 参数的 nano 在官方协议下优于 2149 参数的 LGW-TCN
2. 训练协议贡献巨大：LGW-TCN 从官方协议换到 300ep 余弦协议后 NRMS 0.2361 → 0.1736（−26%）
3. 架构搜索必须先统一协议口径，否则结论虚高——本项目此前的复现基线（NRMS 0.2589）严重低估了官方真实水平

### 架构还原与 DCU 移植

官方 WaveNet 四级架构（nano/feather/lite/standard）已 100% 还原至 torch 1.10/DCU：

- 层结构：dilated conv → 条件混入（input_mixer）→ tanh → 残差（layer1x1）+ skip 激活累加 → 组间 rechannel
- 参数量逐级验算一致（841/3025/6553/13801），RF = 4093
- **对拍验证**：同一 model.nam 权重下，官方实现与移植实现的 ESR 差 < 5e-5（float32 精度）

## 代码结构

```
├── wavenet_official_dcu.py   # 官方 WaveNet 架构 DCU 移植（训练 + 官方权重对拍）
├── lgw_official_dcu.py       # LGW-TCN 架构 + 官方训练协议（DCU）
├── lgw_dcu2.py               # LGW-TCN 架构搜索全套（22 run 网格）
├── train_nam.py              # Gated-Block 行为级仿真训练（早期版本）
├── nam_baseline.py           # NAM standard MLP 复现基线
├── nam_wavenet_baseline.py   # NAM WaveNet 复现基线
├── infer_only.py             # NumPy 纯推理（门控链）
├── dcu_infer.py              # DCU 权重加载 + 全段推理 + 写 wav
├── quant_check2.py           # 逐张量/逐通道 INT 量化对比
├── convert_new_amps.py       # ToneTwist AFx 数据集统一转换（48k/float32）
├── sync_back.sh              # 智算服务器 → 本地 数据/权重/结果 回传
├── dataset_zenodo/dl_list.txt # zenodo record 下载清单
└── docs
    ├── NAM_COMPARISON.md     # 官方包对比全记录（含口径修正）
    ├── EXPANSION_PLAN.md     # E1-E5 拓展实验矩阵
    ├── ARCH_SEARCH.md        # 架构搜索足迹
    └── FINAL_REPORT.md       # 阶段交付报告
```

## 数据集

全部来自 [ToneTwist AFx Dataset](https://github.com/mcomunita/tonetwist-afx-dataset)（zenodo），统一转换 48kHz 单声道 float32，train/val/test 三段：

| tag | 设备 | 类型 |
|---|---|---|
| tsa15 | Ibanez TSA15 Crunch | 管箱 |
| englrt50 | Engl Retro Tube 50 Drive | 管箱 |
| markvx | Mesa Mark V Extreme | 管箱 |
| bsht5m | Blackstar HT5 Metal OD | 管箱 |
| blackstar_ht1 | Blackstar HT1 OD | 管箱 |
| mesa550_clean/crunch/burn | Mesa 550 三通道 | 管箱 |
| markv_clean | Mesa Mark V Clean | 管箱 |
| ts9_s1/s2 | Ibanez TS9 双 setting | 过载 |
| bigmuff_diy | DIY Big Muff | 失真 |

wav 数据不入库（约 1GB），按 `dataset_zenodo/dl_list.txt` 的 record ID 从 zenodo 下载后用 `convert_new_amps.py` 统一转换。

## 环境

| 侧 | 配置 |
|---|---|
| DCU 训练 | 曙光智算 wuzh02，海光深算一号，DTK 22.04，torch 1.10.0a0+git450cdd1.dtk22.4，Slurm (`-c8 --mem=16G --gres=dcu:1`) |
| 官方包 | Miniconda py3.10 env + torch 2.6.0+cpu + neural-amp-modeler 0.13.0（`pip install --only-binary=:all:` 绕过编译死结） |

## 复现

```bash
# 官方架构 DCU 训练（官方协议）
python wavenet_official_dcu.py standard tsa15 100
# 官方权重对拍（加载官方 .nam 验证移植等价性）
python wavenet_official_dcu.py --eval <model.nam> tsa15
# LGW-TCN 官方协议
python lgw_official_dcu.py v_tcn812h1 tsa15 100
```

## License

MIT
