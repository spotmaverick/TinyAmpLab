# 拓展对比计划（软件模拟阶段）

用户指令：官方包对比才有效；对齐 A2 Full 及更大规模官方协议；数据充分拓展；暂不进入电路设计阶段。

## 已就绪的基础设施

- 官方 NAM 环境：登录节点 Miniconda（glibc 2.17 兼容老版）+ py3.10 env + torch 2.6 CPU + neural-amp-modeler
- 官方包 0.13.0 安装完成（`--only-binary=:all:` 策略，llvmlite/numba/soxr/pillow 全部 wheel 化）
- 计算节点执行：CPU-only Slurm 作业（共享 FS，conda env 直接可用；QOS 强制 --gres=dcu:1，实际跑 CPU）
- 多管箱数据（全部 ToneTwist AFx 系列，同作者同结构，DRY 共享）：
  | 管箱 | 风格 | train/val/test | 状态 |
  |---|---|---|---|
  | Ibanez TSA15 Crunch | 英式 crunch | 152.5/20.6/38.1s | ✅ 已有 |
  | Engl Retro Tube 50 Drive | 英式 drive | 152.5/20.6/38.1s | ✅ 已转换 |
  | MesaBoogie Mark V Extreme | 美式高增益 | 152.5/20.6/38.1s | ✅ 已转换 |
  | Blackstar HT5M Overdrive | 英式金属 | 152.5/20.6/38.1s | ✅ 已转换 |
  - 全部已转 48kHz 单声道 float32，命名 `<tag>_{train,val,test}_{input,target}.wav`
  - 注：4 个 amp 的 input.wav 是同一扫描信号（文件大小完全一致），跨 amp 只需下载 target

## 实验矩阵（按优先级）

### E1 官方基线（最高优先级）✅ 已启动（2026-11-24）
- [x] 官方包确认：neural-amp-modeler 0.13.0（含 WaveNet/LSTM/ConvNet/PackedWaveNet，torch 2.6.0+cpu + PL 2.6.1）
- [x] 官方架构定义考古：v0.8.0 `nam/train/core.py::_get_wavenet_config()` 四级架构 STANDARD/LITE/FEATHER/NANO，转译到 0.13 `layers_configs` schema（RF 全部 4093，与官方模型卡片一致）
- [x] 0.13 schema 实测参数量（一手证据，服务器上实例化统计）：standard=13801、lite=6553、feather=3025、nano=841
- [x] 官方训练协议：nam_full_configs（lr 0.004、ExponentialLR gamma 0.993、batch 16、100 epochs、官方损失内建 ESR+MRSTFT）
- [x] 冒烟通过：nano 2 epochs CPU 跑通并导出 model.nam（2.84 it/s）
- [🔄] 四作业并行训练 TSA15（Slurm 49756750/49756752/49756754/49756755，后台轮询中）：nano / feather / lite / standard
- 评价：官方 esr（mean per-sample MSE/target MSE）统一算 train/val/test 三段，与 LGW-TCN 2149 参数同口径对比
- 服务器端位置：`/work/home/spotmaverick/nam_project/e1/`（e1_setup.py、e1_run.py、sbatch_e1.sh）
- 预期：CPU 训练，nano 约 30min，standard 数小时

### E2 损失变量消融
- [ ] 我方冠军 v_tcn812h1_p3 改用官方损失（ESR+MSTFT）重训——分离"架构 vs 损失"贡献
- [ ] 我方 MVP 损失（MSE+STFT）下的官方架构复现数字（已有）作为交叉参照

### E3 多管箱矩阵
- [ ] 我方冠军在 4 个管箱上分别训练+评测（架构不变，数据切换）
- [ ] NAM A2 Lite/Full 在 4 个管箱上同协议训练+评测
- [ ] 产出 4×3 对比表（管箱 × 架构）
- [ ] 跨管箱泛化：TSA15 训练的模型直接评测其他管箱 test 段

### E4 规模阶梯
- [ ] 我方架构按官方 A2 Full 规模（~12k）重构并训练，验证"我方在同等规模下是否仍优"
- [ ] 与 A1 WaveNet（~13.8k）、LSTM（官方支持）对照

### E5 数据规模拓展（E1-E4 之后）
- [ ] Aalto 1 小时长录音数据集（RAT/DFZ/CS-3，record 17053707，dry=ch1/wet=ch2）
- [ ] DAFx23 的 18 对短配对作为额外 eval 套件

## 风险与依赖

- 官方 NAM 对 torch 版本要求（≥2.x）与 DCU torch 1.10 不兼容 → 官方模型走 CPU 训练，我方模型仍可在 DCU 上训；对比评测统一在 CPU 或统一指标口径下进行
- CPU 训练速度：A2 Full 在 152s 数据上可能需 6-12h → 提交长时限 Slurm 作业
- Zenodo 下载不稳（CentOS 7 curl 老旧）→ wget 重试策略已验证可行
