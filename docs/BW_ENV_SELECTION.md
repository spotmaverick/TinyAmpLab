# BW 系列服务器镜像与 PyTorch 版本选型

> 结论先行：**torch 版本不要自己挑，由镜像的 DTK 代次决定。**
> 你要向平台确认的是「DTK 版本」和「Python 版本」，torch 跟着它们走。
> 本文给出推荐档位、必须满足的硬约束、以及为什么不能升到最新版。

## 一、先纠正一个前提：BW 不是另一套软件栈

海光 DCU 目前两代产品在**同一套软件栈**上（DTK = DCU Toolkit，基于 ROCm）：

| 代次 | 型号 | 说明 |
|---|---|---|
| 深算一号 | **DCU Z100**（C878180，17.2GB HBM） | **旧服务器 `wuzh02` 的实际卡型**，见 `README.md` / `NAM_COMPARISON.md` |
| 深算三号 | **BW**（本次实测 `Card Series: BW`，64GB HBM） | 本次迁入目标 |

> **勘误（2026-11-25）**：本表此前把旧卡写成"深算二号 K100_AI"，是错的。
> 项目原始记录（README.md、NAM_COMPARISON.md）一致指向深算一号 Z100。
> 用户口语中的"Z100"即指这张旧卡。实测性能对比见
> `docs/reports/Z100_VS_BW_BENCHMARK.md`。

关键点：**BW 系列仍然走 HIP + DTK + `torch.cuda` API 映射**。所以代码里的
`DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` **一行都不用改**。
BW 带来的实际变化是更宽的显存带宽、PCIe 5.0、以及更高 FP16 算力。
**注意**：实测 29× 的加速并非全部来自硬件代差（带宽差仅约 2×），
软件栈从 torch 1.10/DTK22 升级到 torch 2.9/DTK26 贡献了大部分，详见性能报告。

## 二、推荐档位：按优先级选

| 优先级 | 镜像 | 组成 | 适配本项目 |
|---|---|---|---|
| **首选** | DTK 25.04 或 26.04 + PyTorch 2.x | 例：`pytorch-2.5.1-ubuntu22.04-dtk25.04.2-py3.11` | ✅ 推荐。Python 3.11、torch 2.5/2.6，生态最新且 DTK 成熟 |
| 次选 | DTK 24.04 + PyTorch 2.4 | 社区镜像常见档 | ✅ 可用，无问题 |
| 保守 | DTK 23.10 + PyTorch 2.1~2.3 | 老一点但稳定 | ✅ 可用 |
| **避免** | DTK 22.04 + torch 1.10（旧服务器现状） | Python 3.8 | ⚠️ 能跑但不建议再用，见第四节 |
| **禁止** | 镜像里 `pip install torch` 覆盖 | 社区公版 | ❌ 会覆盖带 `dtk` 后缀的适配版，**直接失去 DCU** |

### 具体建议

**首选：DTK 26.04 + torch 2.9.0（Python 3.11）**
- 这是目前海光官方较新的稳定组合，社区验证充分
- 版本号形如 `torch 2.9.0+das.opt1.dtk2604`
- 注意：`+das.opt1.dtk2604` 后缀是海光适配标识，**必须保留**

**如果平台只给 DTK 25.04：DTK 25.04 + torch 2.5.1 或 2.6（Python 3.11）**
- 形如 `torch 2.5.1+das.opt1.dtk25042`
- 这是一个经过大量实测验证的组合，最稳

**如果平台给 DTK 24.04：torch 2.4.x**

## 三、对本项目的硬约束（我逐条查过代码）

### 必须满足
| 约束 | 来源 | 说明 |
|---|---|---|
| Python ≥ 3.9 | 项目自身无 3.10 专属语法（已查，无 `match`/PEP604 用法） | 但见下方 NAM 说明 |
| numpy 任意 1.x / 2.x | 已查：无 `np.float`/`np.int`/`np.NaN` 等 numpy2 移除的 API | 可放心用 numpy 2 |
| torch ≥ 1.10 | 用到的算子只有 `conv1d`/`pad`/`tanh`/`sigmoid`/`matmul`，全是老 API | 版本上界很宽 |
| **soundfile** | 读写 wav | 任何版本均可，0.12+ 建议 |
| DTK 与 torch 后缀必须匹配 | `torch x.y.z+das.opt1.dtkNNNN` 的 `dtkNNNN` == 本机 DTK | **不匹配会直接报错或静默错误** |

### 不需要满足（重要，能省你很多事）
- ❌ **不需要装 `neural-amp-modeler`**。我查了代码：`wavenet_official_dcu.py` 是**手写复刻**官方结构（注释写明「结构还原自 neural-amp-modeler 0.13.0」），
  并自己用 `json.load` 解析 `.nam` 权重文件。**项目不 import `nam` 包**。
- ❌ 不需要 `pytorch_lightning`、`pydantic`。
- 这很重要：NAM 官方包要求 Python ≥3.10 + pydantic≥2 + lightning≤2.6.1，
  如果项目依赖它，Python 版本就被锁死了。**实际不受此约束**。

## 四、为什么建议离开 torch 1.10（旧服务器现状）

旧服务器是 `torch 1.10.0a0+git450cdd1-dtk22.04` / Python 3.8.10。能跑，但：

1. **torch 1.10 已停止维护**（2021 年发布），且是 `1.10.0a0` alpha 版
2. Python 3.8 已于 2024-10 EOL
3. 迁移正好是升级窗口——但**必须重跑基线**，见下节

## 五、升级到 torch 2.x 的两个已确认风险点

我扫了全部自有代码（排除 `github_mirrors/`），只有两处需要留意：

### 风险 1：`torch.load` 默认 `weights_only=True`（torch ≥ 2.6）

自有代码有 **12 处 `torch.load` 未显式指定 `weights_only`**。

| 权重文件 | 保存内容 | torch≥2.6 能否直接 load |
|---|---|---|
| `arch_v2/*.pt`、`arch_analog/*.pt` | 纯 state_dict（`torch.save(best_state)`） | ✅ 可以 |
| `wavenet_official_dcu.py` 的 `weights.pt` | **混合 dict**（state_dict + str/int/float 元数据） | ⚠️ **会报错** |
| `lgw_official_dcu.py` 的 `weights.pt` | **混合 dict** | ⚠️ **会报错** |

修法（两行，新机器上一次搞定）：
```python
# 混合 dict 的加载处，加 weights_only=False（自己产出的文件，可信）
sd = torch.load(path, map_location="cpu", weights_only=False)
```
> 注意：`weights_only=False` 只用于本项目自己生成的权重，**不要**用于外部下载的 `.pt`。

### 风险 2：设备端随机数生成器

物理仿真用 `torch.Generator(device=DEVICE)` 做蒙特卡洛失配采样。
这在新旧 DCU 上都工作，但**若新卡的 HIP 版本对设备端 RNG 支持有差异**，
最省事的兜底是把生成器放 CPU：

```python
g = torch.Generator()                    # CPU 生成器
g.manual_seed(seed0 + k)
# 采样时用 CPU 张量再搬到设备
mm = torch.empty(wq.shape).normal_(0, sigma, generator=g).to(wg.device)
```
这会改变随机序列，但只要 **sanity 对拍仍 PASS**（物理 vs 数字路径差值 0），
就说明改动没破坏正确性——这是现成的验收手段。

## 六、迁移后必须做的事：基线重跑验证

**换 torch 版本会改变数值结果**，这不是可选项。必须重跑锚点：

```bash
# 锚点实验：full_ch12 s7，官方 lite 等参复刻
# 旧环境（torch 1.10）结果：test NRMS = 0.078476
sbatch sbatch_analog.sh full_ch12 300 7 tsa15
```

| 新结果 | 判定 |
|---|---|
| 0.075 ~ 0.082 | ✅ 数值行为一致，历史结论可继承 |
| 偏离 5~15% | ⚠️ 数值差异但不影响结论排序，需注明 |
| 偏离 >15% 或训练不收敛 | ❌ 环境有问题（多半是 DTK/torch 版本不匹配或算子走错后端） |

同理，物理仿真的 sanity 也必须重跑：
```bash
python src/analog/analog_sim_wavenet.py --sanity   # 需 PASS
python src/analog/analog_sim_dw.py --variant dw_ch12 --sanity
```

## 七、向平台要镜像时，按这个清单问

把下面几条直接发给平台管理员，能避免来回扯：

1. **DTK 版本是多少？**（例：DTK 25.04.2 / 26.04）——这是第一优先，其他都跟它走
2. **镜像里预装的 PyTorch 是什么版本？完整版本号？**（要带 `+das.opt1.dtkNNNN` 后缀）
3. **Python 版本？**（建议 3.11）
4. **PyTorch 是镜像预装还是我自己装？** 若是自己装，我需要 DTK 对应版本的 whl 下载地址
5. **有没有 conda/venv 隔离？** 还是共用系统 Python（影响能否安全 `pip install soundfile`）
6. **Slurm 分区名和资源申请方式是否与旧集群一致**（旧：`-p wzhdtest -c 8 --mem=16G --gres=dcu:1`）
7. **单账户 DCU 配额几张卡？**（旧集群是 8 张，直接决定并行作业数）

## 八、一句话总结

要 **DTK 25.04/26.04 + 预装 PyTorch 2.5~2.9 + Python 3.11** 的镜像。
如果你只能给一句话给平台：**「要海光官方适配的 PyTorch，DTK 25.04 以上，Python 3.11，我自己不装 torch」**。
拿到机器后第一件事不是跑训练，是跑 `full_ch12 s7` 复现 0.078476 这个锚点。
