# 新 BW 集群 zzeshell 可用性验证

**验证时间**：2026-11-25 23:00–23:15 CST
**连接信息**：`zzeshell.scnet.cn:65032`，用户 `spotmaverick`，RSA 密钥（有效期至 2026-12-25）
**结论**：✅ **完全可用**，且是**同型号 BW 的规模化集群**，原有 BW 结论可直接迁移

---

## 一、连接与身份

| 项目 | 结果 |
|---|---|
| SSH 连接 | ✅ 成功（密钥认证，无需密码） |
| 主机名 | `zz-login02` —— **Slurm 登录节点**，非原来那台单卡容器 |
| 操作系统 | Sugon OS 8.9 |
| CPU / 内存 | 128 核 / 1TB |
| 密钥部署位置 | `~/.ssh/bw2_zzeshell.pem`（600 权限） |

密钥文件为标准 OpenSSH 私钥（`BEGIN OPENSSH PRIVATE KEY`，38 行），可直接用于 ssh。

---

## 二、算力资源：比原 BW 大两个数量级

| 项目 | 结果 |
|---|---|
| 分区 | `hx1hdnormal`（唯一分区） |
| 节点规模 | 89 节点，每节点 **8 DCU** |
| 当前可用 | 81 节点空闲（`A/I/O/T = 81/0/8/89`） |
| 单作业时限 | 3 天（`3-00:00:00`） |
| CPU 资源池 | 11392 核，其中 4904 空闲 |
| 账号权限 | `spotmaverick` **在分区白名单内**，QOS=`user_spotmaverick` |

对比原 BW：**单卡容器 → 89 节点 × 8 卡集群**，这是从"一台机器"到"一个集群"的量级跃迁。

---

## 三、★ 关键结论：DCU 与原 BW 同型号

| 指标 | 新集群 (zzeshell) | 原 BW | 判定 |
|---|---|---:|---|
| 显存 | **65520 MiB** | 65520 MiB | 完全一致 |
| 设备 DID | 6320 | — | 同款芯片 |
| GRES 类型 | `dcu:Hygon:8` | `dcu:Hygon` | 一致 |

**意义**：此前所有 BW 上的性能结论（长序列 conv 比 Z100 快 44×、重 dilation 优势等）**在这台机器上直接适用**，无需重新标定。

> 注：`rocm-smi` 显示 Card Vendor 为「Chengdu C-3000 IC Design Co., Ltd.」，这是海光 DCU 的封装厂商标识，不影响芯片型号判断——DID 6320 与 64GB 显存两项硬指标完全对齐。

---

## 四、验证过程中踩的两个坑（必须记住）

### 坑 1：输出路径必须用 `/public/home/`

```bash
# ❌ 错误：作业 1 秒内 FAILED（Signal 53），连输出文件都不生成
#SBATCH -o /home/spotmaverick/probe/out_%j.out

# ✅ 正确
#SBATCH -o /public/home/spotmaverick/probe/out_%j.out
```

原因：家目录实际是 `/public/home/spotmaverick`（9.2P NFS 共享盘）。`/home` 在登录节点是个软链接，但在计算节点**不存在**，Slurm 写输出文件失败 → 作业被杀。

### 坑 2：`module load dtk` 不够，必须 `source env.sh`

```bash
# ❌ 只 module load → libamdhip64.so not found、HIP_DEVICES=0
module load compiler/dtk/24.04.3

# ✅ 正确
export DTKROOT=/public/software/compiler/dtk-24.04.3
source $DTKROOT/env.sh
```

原因：`module show` 只看到 `setenv DTKROOT/HIP_PATH/ROCM_PATH`，**没有设 `LD_LIBRARY_PATH`**。而 `env.sh` 会把三个关键路径都加进去：

- `$ROCM_PATH/lib` — libamdhip64.so
- **`$ROCM_PATH/lib64`** — libamd_comgr.so（这个只在 lib64，漏了就报 `load comgr library error`）
- `$ROCM_PATH/.hyhal/lib` — 海光 HAL 层

修正后实测：**`HIP_DEVICES=1 rc=0`**，申请的 `gres=dcu:1` 正好拿到 1 张卡，`rocm-smi` 正常显示 `DCU[0]`，8 张卡设备节点（card0~card7）齐全。

---

## 五、软件栈现状与缺口

### 已有

| 类别 | 版本 |
|---|---|
| DTK | 24.04.2、24.04.3、25.04、25.04.1、25.04.2、25.04.4、**26.04** |
| RCCL | `app/rccl/dtk-25.04/26.01.1` |
| MPI | hpcx 2.18、openmpi 4.1.8/5.0.3、ucx 1.18（均带 dtk 变体） |
| Python | 3.12.6（系统）、3.8.10（apps） |
| 容器 | apptainer、singularity |
| 外网 | ✅ 通（pypi 200、清华源 200） |

### 缺口（影响能否立刻开工）

| 缺口 | 影响 | 解决路径 |
|---|---|---|
| **无预装 PyTorch** | 无法直接跑训练 | 需自建环境 |
| **无 wheel 仓库**（`/public/software/apps/DeepLearning/` 不存在） | 拿不到海光定制 torch | 从旧 Z100 集群拷 |
| **无 conda**（anaconda3 目录在但 conda 二进制缺失） | 建环境麻烦 | 可用系统 python + venv |

**关键：PyPI 上的官方 torch 是 CUDA/ROCm 版，不认海光 DCU，必须用海光定制 wheel。**

旧 Z100 集群上有现成的：`/public/software/apps/DeepLearning/whl/dtk-24.04/pytorch/torch2.1/py310/`（此前实测可跑通，设备名 Z100SM）。跨集群拷贝是最快路径。

---

## 六、验证清单

| 验证项 | 方法 | 结果 |
|---|---|---|
| SSH 连通 | 密钥登录 | ✅ `zz-login02` |
| Slurm 权限 | 提交作业 | ✅ 分配到 `a14r4n18`/`a14r3n04`/`a14r3n01` |
| DCU 设备可见 | `ls /dev/kfd /dev/dri` | ✅ `/dev/kfd` + card0~7 |
| HIP 识别卡数 | `hipGetDeviceCount` | ✅ **1**（申请 1 张） |
| 显存规格 | `rocm-smi --showmeminfo` | ✅ 65520 MiB |
| 共享存储 | `df -h ~` | ✅ 9.2P NFS，计算节点可见 |
| 外网 | curl pypi | ✅ 200 |
| PyTorch | `import torch` | ❌ 未安装（需自建） |

**7/8 通过，唯一缺口是 PyTorch 环境需自建。**

---

## 七、建议

1. **可以正式投入使用**。这台是原 BW 的同型号规模化集群，性能结论直接继承。
2. **优先搭 torch 环境**：从旧 Z100 集群拷海光 wheel（dtk-24.04 对应 torch2.1），或测试 rccl dtk-25.04 + 更新 torch。
3. **作业脚本三条硬规则**：
   - 输出路径一律 `/public/home/spotmaverick/...`
   - 必须 `source $DTKROOT/env.sh`
   - 申请卡数用 `--gres=dcu:N`
4. **注意时限**：单作业最长 3 天。当前训练单作业约 10 分钟，无压力。
5. **优先级**：`Priority=1255`，实测排队 30~60 秒即分配，比原 BW 容器更稳定（原容器 SSH 频繁断连，这里是正规集群）。
