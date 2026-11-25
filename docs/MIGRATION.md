# 智算服务器迁移手册

> 场景：从旧服务器（`wuzh02.hpccube.com:65091`）迁移到新服务器。
> 生成时间：2026-11-25 13:15。迁移包已在旧服务器 `/work/home/spotmaverick/migrate_out/` 打好。

## 一、已准备好的迁移产物

| 产物 | 大小 | MD5 | 内容 |
|---|---:|---|---|
| `TinyAmpLab_migrate_20261125_1309.tar.gz` | 717 KB | `bff7711d5ecea944a08604d96bb5d49e` | 代码 + 全部结果 + 权重 + 环境规格 + 数据清单 + 一键部署脚本 |
| `TinyAmpLab_data_20261125_1310.tar.gz` | 1.1 GB | `3dd5cb1c9706da5fa433058181ab4bfb` | 94 个已校验的 wav 数据集 |

两个包都在旧服务器 `/work/home/spotmaverick/migrate_out/`。

**Marshall 原始 zip（2.76GB）不在包内**——如需重建，用 `dataset/MANIFEST.md` 里的分段续传命令（已验证可靠，本项目就是这么下的）。

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
source ~/nam_project/env/run_env.sh
cd ~/nam_project
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**期望**：打印 DTK torch 版本 + `True`。
**若 `torch.cuda.is_available()` 是 False** → 说明当前在登录节点，DCU 只在计算节点可用，这是正常的。

### 步骤 5：跑一次 sanity 确认一切正常

```bash
# 在登录节点用 CPU 跑（不需要 DCU）
python src/analog/analog_sim_wavenet.py --sanity    # 期望: 官方类交叉验证 PASS, 物理 vs 量化差值 0
python src/analog/analog_sim_dw.py --variant dw_ch12 --sanity   # 期望: PASS
```

### 步骤 6：提交训练作业

```bash
sbatch sbatch_analog.sh dw_ch24 300 7 tsa15
```

## 三、环境规格要点

| 项 | 值 |
|---|---|
| python | 3.8.10（集群 module `python/3.8.10`） |
| torch | 集群 DTK 自带 `1.10.0a0+git450cdd1-dtk22.04`，**勿用 pip 覆盖** |
| 关键 module | `compiler/devtoolset/7.3.1 compiler/dtk/22.04.1 mpi/hpcx/gcc-7.3.1 python/3.8.10` |
| LD_LIBRARY_PATH | 需包含 `/public/software/apps/DeepLearning/PyTorch_Lib` 与 `/public/software/apps/deeplearning-depend` 下所有 `.so` 所在目录 |
| Slurm | `sbatch -p wzhdtest -c 8 --mem=16G --gres=dcu:1` |
| DCU device | torch 层面用 `cuda`（**不是 `dpu`**） |
| neural-amp-modeler | `pip install --only-binary=:all: neural-amp-modeler`（源码编译会失败） |

**若新服务器的 module 名或分区名不同** → 改 `~/nam_project/env/run_env.sh` 与 `sbatch_*.sh` 里的 `#SBATCH` 头即可，其余脚本无需动。

完整规格见迁移包内 `env/ENV_SPEC.md`（含所有踩坑记录）。

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
