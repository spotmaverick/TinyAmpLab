# 智算服务器迁移手册

> 场景：从旧服务器（`wuzh02.hpccube.com:65091`）迁移到新服务器。
> 生成时间：2026-11-25 13:50。迁移包已在旧服务器 `/work/home/spotmaverick/migrate_out/` 打好。

## 一、已准备好的迁移产物

| 产物 | 大小 | MD5 | 内容 |
|---|---:|---|---|
| `TinyAmpLab_migrate_20261125_1317.tar.gz` | 512 KB | `f5e88b7af71aaa7d9abbb9e2d7b087fc` | 代码 + 全部结果 + 权重 + 环境规格 + 数据清单 + 一键部署脚本 |
| `TinyAmpLab_data_20261125_1310.tar.gz` | 1.1 GB | `3dd5cb1c9706da5fa433058181ab4bfb` | 94 个已校验的 wav 数据集（16 个音箱） |

两个包都在旧服务器 `/work/home/spotmaverick/migrate_out/`。

> **注意**：代码包必须用 **1317** 这个版本。1309 版是修复前的——旧服务器上是整理前的散落目录结构，
> 打包时 `rsync src/` 落空，导致迁移包缺失所有 `src/` 子目录（工具与物理仿真脚本），
> 新服务器部署后会跑不起来。1317 版改为从 GitHub clone 取最新结构，已修。

**Marshall 原始 zip（2.76GB）不在包内**——如需重建，用 `dataset/MANIFEST.md` 里的分段续传命令（已验证可靠，本项目就是这么下的）。

### 迁移链路已实测验证（2026-11-25 13:40）

在旧服务器上模拟了一次完整迁移并全部通过：

| 验证项 | 结果 |
|---|---|
| 全新目录解压 + `deploy.sh` | ✅ 通过 |
| 11 项关键文件自检 | ✅ 全过（`arch_analog_friendly.py`、`src/analog/analog_sim_dw.py` 等） |
| 数据包解压 | ✅ 94 个 wav 就位 |
| 环境激活 | ✅ torch 1.10.0a0+git450cdd1-dtk22.04 |
| `analog_sim_dw.py --sanity` | ✅ PASS（物理 vs 6-bit 差值 0.000000） |
| `analog_sim_wavenet.py --sanity` | ✅ PASS（官方类交叉验证误差 0.000000e+00） |

验证完已清理测试目录，不留垃圾。

## 二、迁移步骤

### 步骤 1：把包拉到本地（MacBook）

```bash
KEY=/Users/wangguan/Downloads/spotmaverick_wuzh02.hpccube.com_RsaKeyExpireTime_2027-02-21_21-34-03.txt
mkdir -p ~/TinyAmpLab_migrate
scp -P 65091 -i "$KEY" 'spotmaverick@wuzh02.hpccube.com:/work/home/spotmaverick/migrate_out/*.tar.gz*' ~/TinyAmpLab_migrate/
```

### 步骤 2：上传到新服务器

```bash
scp -P <新端口> -i <新密钥> ~/TinyAmpLab_migrate/*.tar.gz* <新用户>@<新主机>:~/migrate_in/
```

### 步骤 3：在新服务器解压并部署

```bash
mkdir -p ~/migrate_in && cd ~/migrate_in
tar -xzf TinyAmpLab_migrate_*.tar.gz
tar -xzf TinyAmpLab_data_*.tar.gz -C ~/nam_project/     # 数据放到项目根
bash TinyAmpLab_migrate_*/deploy.sh ~/nam_project
```

### 步骤 4：验证环境

```bash
bash src/tools/env_check_bw.sh --full   # 一次性打出 DTK/torch/Python/DCU/算子全部事实
```

这个脚本会明确告诉你三件事：
1. torch 版本是否带 `+das.opt1.dtkNNNN` 后缀（**不带就说明被公版覆盖，DCU 会失效**）
2. `dtkNNNN` 与本机 `/opt/dtk` 版本是否一致
3. 本项目用到的算子（`conv1d` 带 dilation / `tanh` / `sigmoid` / 反向传播 / 设备端 Generator / float64）能否正常工作

**期望**：DTK 版本与 torch 后缀一致、`cuda.is_available()` 在计算节点为 `True`。
**若 `torch.cuda.is_available()` 是 False** → 若在登录节点属正常，DCU 只在计算节点可用。
**若 torch 版本不带 dtk 后缀** → 停手，环境已被污染，不要在此基础上跑实验。

> BW 系列（海光深算三号）仍是 DTK + ROCm 栈，`torch.cuda` API 映射不变，
> 代码里的 `DEVICE = "cuda" if ...` **一行都不用改**。
> 版本选型详见 **`docs/BW_ENV_SELECTION.md`**。

### 步骤 4.5：升级 torch 版本后必做的锚点复现

换 torch 版本会改变数值结果，**历史结论不能无条件继承**。重跑框架自验锚点：

```bash
sbatch sbatch_analog.sh full_ch12 300 7 tsa15
```

| 结果 | 判定 |
|---|---|
| 0.075 ~ 0.082 | ✅ 数值行为一致（旧环境 0.078476），历史结论可继承 |
| 偏差 5~15% | ⚠️ 数值有差异但不影响排序，文档需注明 |
| 偏差 >15% 或不收敛 | ❌ 环境有问题，先排查 DTK/torch 匹配与算子后端 |

### 步骤 5：跑一次 sanity 确认一切正常

```bash
# 在登录节点用 CPU 跑（不需要 DCU）
python src/analog/analog_sim_wavenet.py --sanity    # 期望: 官方类交叉验证 PASS, 物理 vs 量化差值 0
python src/analog/analog_sim_dw.py --variant dw_ch12 --sanity   # 期望: PASS
```

> 若新环境设备端 `torch.Generator(device=...)` 不可用，`env_check_bw.sh` 会告警，
> 兜底改法见 `BW_ENV_SELECTION.md` 第五节风险 2。

### 步骤 6：提交训练作业

```bash
sbatch sbatch_analog.sh dw_ch24 300 7 tsa15
```

## 三、环境规格要点

### 旧服务器实测（对照基准）

| 项 | 值 |
|---|---|
| python | 3.8.10（集群 module `python/3.8.10`） |
| torch | 集群 DTK 自带 `1.10.0a0+git450cdd1-dtk22.04`，**勿用 pip 覆盖** |
| 关键 module | `compiler/devtoolset/7.3.1 compiler/dtk/22.04.1 mpi/hpcx/gcc-7.3.1 python/3.8.10` |
| LD_LIBRARY_PATH | 需包含 `/public/software/apps/DeepLearning/PyTorch_Lib` 与 `/public/software/apps/deeplearning-depend` 下所有 `.so` 所在目录 |
| Slurm | `sbatch -p wzhdtest -c 8 --mem=16G --gres=dcu:1` |
| DCU device | torch 层面用 `cuda`（**不是 `dpu`**） |
| neural-amp-modeler | **不需要**（项目为手写复刻，不 import nam 包） |

### BW 系列新机器推荐（详见 `docs/BW_ENV_SELECTION.md`）

| 项 | 推荐 |
|---|---|
| DTK | 26.04 或 25.04 |
| torch | `2.9.0+das.opt1.dtk2604`（配 DTK26.04）或 `2.5.1+das.opt1.dtk25042`（配 DTK25.04） |
| Python | 3.11（可接受 3.9~3.12） |
| 硬约束 | torch 版本必须带 `+das.opt1.dtkNNNN` 且与 `/opt/dtk` 版本一致；**禁止 `pip install torch` 覆盖** |

**若新服务器的 module 名或分区名不同** → 改 `~/nam_project/env/run_env.sh` 与 `sbatch_*.sh` 里的 `#SBATCH` 头即可，其余脚本无需动。

完整规格见迁移包内 `env/ENV_SPEC.md`（含旧环境实测值与 BW 选型两节）。

## 四、DCU 使用踩坑清单（务必阅读）

1. **device 字符串是 `cuda` 不是 `dpu`** —— DCU 后端映射到 `torch.cuda` API。
2. **登录节点没有 DCU** —— 训练必须 `sbatch` 到计算节点；但 CPU 可在登录节点跑 sanity/quick 验证。
3. **`--mem` 上限 16G** —— 超了作业不调度。
4. **必须带 `--gres=dcu:1`** —— 否则不分配 DCU。
5. **配额限制** —— 账户 DCU 配额有限，作业会因 `AssocGrpGRES` 排队，这是预期行为（说明配额用满了）。
6. **`torch.load` 的 device 不匹配** —— 权重从 CPU 载入但输入在 DCU 时会报错，需按输入设备转换权重。`analog_sim_wavenet.py` 的 `digital_fwd` 已处理。
7. **Miniconda 版本** —— 必须用 `py310_23.5.2` 旧版，新版与集群 glibc 冲突。
8. **不要在运行中的脚本上原地编辑** —— shell 会边读边执行，读到损坏内容导致任务失败（本次踩过）。

## 五、当前实验状态（迁移后接着做）

**已完成**：
- 官方 NAM 四档基线（nano/feather/lite/standard）+ RF 缩放曲线 + 双种子配对判定
- Marshall / UA 数据接入与转换
- lite 物理器件级仿真全量（0.0785 → 0.0850 → 0.1063±0.0114）
- 框架自验双种子通过（full_ch12 s7=0.078476 与官方精确一致）
- depthwise 物理仿真与 Vsat 扫描

**未完成（作业在跑，等收敛）**：
- `dw_ch24` s7 —— **H2 决定性**：宽度补参能否进 0.10–0.12
- `dw_ch12_rf50` s7 —— H3：短 RF 代价
- `dw_ch24_rf50` s7 —— H2+H3

**下一步（依赖 dw_ch24 结果）**：
1. 若 `dw_ch24` 达标 → 对它做物理仿真（`analog_sim_dw.py --variant dw_ch24`），确认摆幅与鲁棒性
2. 补 `dw_ch24` 的第二种子（s11）做配对
3. 把 depthwise 架构接入 `analog_sim_dw.py` 的完整器件级核算，与 lite 做最终对比

## 六、GitHub 备份

代码与结论文档已全部推到 `https://github.com/spotmaverick/TinyAmpLab`（Public）。
即使迁移包丢失，从 GitHub 也能恢复全部代码与文档——只有 `.pt` 权重和数据集不在 GitHub 里（`.gitignore` 排除），需靠上面的迁移包或重新训练。
