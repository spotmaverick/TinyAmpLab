# bw2 环境搭建成功：移植 BW 容器的 gfx936 torch

> **2026-11-27 08:00–08:45 完成。bw2 现在可以训练了。**
> 验证作业：1252043（torch 验证）、1252431 / 1252507（端到端仿真）

## 一、一句话结论

**bw2 已可用。** 从 BW 容器移植 torch 2.9.0（`arch` 含 gfx936）+ DTK 26.04，
`Conv1d` / 大 dilation 卷积 / 反向传播全部通过，端到端跑出与 bw 一致的结果。

**为什么这条路之前没走通**：11-26 那轮只试了「从 z100 拷 wheel」和「换 DTK 版本」。
z100 的 wheel 只含 gfx906/926/928，换哪个 DTK 都没用。真正的解法是
**BW 容器的 torch 本身就是按 gfx936 编译的**（`arch = ['gfx906','gfx926','gfx928','gfx936','gfx938']`）。

---

## 二、迁移链（bw 与 bw2 之间无直连，必须经 Mac 中转）

```
[bw 容器]                    [Mac]                    [bw2]
torch 2.9.0 (1.9G)  ──pull──>  /tmp  ──push──>  /public/home/spotmaverick/t29/
torch.libs   (129M)             48s 传输          解包后 1.8G
纯 Python 依赖 ────────────>                   同上
```

| 包 | 内容 | 大小 |
|---|---|---|
| `torch_gfx936.tgz` | torch + torch.libs（已剪掉 test/ 和 include/） | 978 MB |
| `deps.tgz` | typing_extensions, filelock, fsspec, jinja2, markupsafe, mpmath, networkx, sympy | 27 MB |
| `torchgen.tgz` | torchgen（torch 内部 import 需要） | 0.9 MB |
| `d3.tgz` | numpy 1.26.4, scipy 1.17.1, cffi, pycparser, `_soundfile_data` | 68 MB |
| `d4/d5.tgz` | `soundfile.py`、`_soundfile.py`（都是单文件，容易漏） | 17 KB |
| `d6.tgz` | pydantic 2.12.5 + pydantic_core（nam 0.11 必须要 v2） | 9.4 MB |

**踩坑顺序（下次照这个清单一次配齐）**：
1. `typing_extensions` 太旧 → 缺 `TypeIs`（torch 2.9 需要）
2. `torchgen` 缺失 → `torch.utils._python_dispatch` 直接 import 它
3. `soundfile` 缺失 → **注意它是 `soundfile.py` 单文件，不是目录**
4. `_soundfile` 缺失 → **同样是 `_soundfile.py` 单文件**，另有 `_soundfile_data/libsndfile_x86_64.so`
5. `pydantic` 是 v1 → `_LayerArrayHeadConfig has no attribute 'model_validate'`，必须 v2
6. 装 scipy 时 pip 会把 numpy 升到 2.x → torch 2.9 是 numpy 1.x ABI，必须 `numpy<2`
7. `nam/__init__.py` 无条件 import `train` → 缺 pytorch_lightning 会崩，已改成可选导入

---

## 三、环境激活（已固化为 `~/p26/bw2env.sh`）

```bash
source ~/p26/bw2env.sh
# 内部等价于：
#   export LD_LIBRARY_PATH=""
#   source /public/software/compiler/dtk-26.04/env.sh
#   export PYTHONPATH=/public/home/spotmaverick/t29:$PYTHONPATH
#   export PY=/public/software/apps/anaconda3/2023.09/bin/python3.11
```

⚠️ **必须在计算节点用**（`--gres=dcu:1` 的作业）。登录节点缺
`libmsgpackc.so.2`，import torch 会报错——那是登录节点的问题，不是环境没配好。

---

## 四、验证结果（作业 1252507）

```
sf ok 0.14.0
[device] cuda  torch=2.9.0  hip=6.3.26093  count=1
[device] BW
  s7 V_SAT=  16  NRMS=0.1016  (理想 0.0828, 退化 1.23x)
  s7 V_SAT=  24  NRMS=0.0869  ...
  s7 V_SAT= 128  NRMS=0.0827  (理想 0.0828, 退化 1.00x)
[saved] ... 总耗时 6s
```

**与 bw 的交叉验证**：bw 上同样口径 s7 是 0.1041，bw2 是 0.1016，差 2.4%（MC 噪声量级）。
**两个异构入口结果一致 → 环境可信。**

性能对比（单种子 7 档 V_SAT）：

| 机器 | 耗时 |
|---|---:|
| bw（容器，1 卡） | 2s（1 档单种子）/ 38s（5 种子全扫） |
| bw2（Slurm，1 卡） | 6s（1 档单种子） |

**量级一致。bw2 的真正价值是 712 卡并发，不是单卡更快。**

---

## 五、现在怎么用 bw2

**标准作业模板**（`~/p26/runp1.slurm` 是可用样例）：

```bash
#!/bin/bash
#SBATCH -J myjob
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=01:00:00
#SBATCH --gres=dcu:1              # ← 必须显式申请，否则 QOS 拒绝
#SBATCH -o /public/home/spotmaverick/p26/myjob-%j.out   # ← 必须 /public/home，不能 /home
source ~/p26/bw2env.sh
cd /public/home/spotmaverick/nam_proj
$PY -u your_script.py
```

**项目位置**：`/public/home/spotmaverick/nam_proj/`
- `p1/seeds/*.nam`（5 种子）
- `p1_vsat_sweep.py`、`analog_sim_generic.py`
- `nam/`（已打可选导入补丁）、`stubs/`
- `data/` → 软链到 `/public/home/spotmaverick/nam_project/data`

**批量并发的用法**：同一脚本改 `--seeds`/`--tag` 提交 N 个作业即可，
bw2 有 712 卡，`squeue` 会并行调度。

---

## 六、遗留与注意

- **torch 2.9.0 而非 2.1.0**。与 z100 的 `torch21_env`（2.1.0）版本不同，
  跨机器比较绝对耗时要注意；精度结论不受影响（同一份权重、同一前向逻辑）。
- **`/public` 配额**：`df` 显示 `/public` 池有 5.0P 空闲（集群级共享），
  但**个人配额限制需另查**（`du -sh ~` 当前约 5G）。
- **中转包已删除**（`torch_gfx936.tgz` 等），只保留解包后的 `t29/`。
  若以后 `t29/` 被清理，需重新从 bw 打包（bw 开机时才能做）。
- **bw2 密钥 2026-12-25 到期**，跨年实验要提前续。
