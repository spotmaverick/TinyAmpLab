# Checkpoint 续训与故障自愈（实战记录）

> 日期：2026-11-27
> 起因：上一轮结论说"官方训练不存 checkpoint，所以不能跨机迁移"。**这条结论是错的，本文件是对它的更正。**

## 一句话结论

**官方 nam 一直在存 checkpoint，而且存的是完整可续训的 checkpoint（含 optimizer/scheduler 状态）。**
真正缺的只是"续训入口"——`train_seeded.py` 里的 `outdir.mkdir(exist_ok=False)` 把任何写回已有 run 目录的尝试直接堵死。
补上入口后，故障恢复从"整颗种子作废"变成"只损失 1 个 epoch"。

## 更正记录：上一轮错在哪

| 上一轮的说法 | 事实 | 错因 |
|---|---|---|
| "官方训练不存 checkpoint（只导出 nam）" | `checkpoint_epoch_{epoch}.ckpt` **每 epoch 覆盖一份**；`save_top_k=3` 的 best 另存一组 | 只看 `find -name "*.ckpt"` 的顶层目录，没往 `lightning_logs/version_N/checkpoints/` 里看 |
| "转机要从 ep0 重训 3.3h" | 从 ep94 续训只需约 1.9h（甚至跨机也一样） | 结论建立在错误前提上 |

实测 ckpt 内容（`checkpoint_epoch_epoch=0069.ckpt`，289 KB）：

```
top-level keys: callbacks, epoch, global_step, loops, lr_schedulers,
                optimizer_states, pytorch-lightning_version, sample_rate, state_dict
epoch=69  global_step=3850
optimizer_states[0]: param_groups + state（lr=0.0024463 已恢复）
lr_schedulers: 1
param count: 18947
```

**不是空壳**——权重、优化器动量、学习率调度器全部在里面。

## 续训为什么技术上可行

三个前提都已验证：

1. **ckpt 完整**：见上表。
2. **数据侧确定性**：`Dataset.__getitem__` 是 `i = idx * self._ny` 的固定切片，无随机采样，换机后取到同样的数据。
3. **官方代码不需要改**：`full.main` 里本来就是
   `trainer.fit(..., **learning_config.get("trainer_fit_kwargs", {}))`，
   所以只要往 `trainer_fit_kwargs` 里塞 `{"ckpt_path": ...}` 就能续训。

## 改动内容

### `train_seeded.py`：从"只能从头跑"改成"可续训"

- 新增 `--resume-from <ckpt>`，**自动从 ckpt 反推 run 目录**
  （向上找第一个含 `config_learning.json` 的目录——用 `full.main` 自己写的锚点，不靠固定层级硬猜）。
- 新增 `--max-epochs N`，用于续训时延长目标 epoch。
- **修掉了 `exist_ok=False`**：续训写回已有 run 目录不再抛 `FileExistsError`。
- 参数解析改成手写：`--max-epochs 120` 里的 `120` 不会被当成位置参数。
  （朴素写法 `[a for a in argv if not a.startswith("--")]` 会把这个 120 算进位置参数，把个数搞乱。）

### `scripts/p0a_run_resilient.sh`：崩溃自动续训

- 启动前找同 run 目录的最新 ckpt（**按 epoch 号排序，不是 mtime**——续训会产生新 version 目录，mtime 会骗人）。
- 有 ckpt 就 `--resume-from`，没有就全新训练；失败自动重试，默认 3 次。
- 成功后导出 `model.nam` 到 `p0a/seeds/s{seed}.nam`。
- **单例保护**：同一 run 目录只认一个主进程。
  判据不能用 `pgrep -f train_seeded`——dataloader worker（`num_workers=4`）的 cmdline 与主进程逐字相同，
  会被误判成"多开"。正确判据是 **ppid 不在同 run 进程集合里**的那个。

## 实战演练：故障注入 + 自愈

在 s47 上真实做了一次，不是纸上推演：

| 步骤 | 结果 |
|---|---|
| 基线：s47 到 ep94，ckpt 也是 ep94 | — |
| `kill -9` 主进程 7556（模拟节点故障） | 进程归零 |
| 启动 `p0a_run_resilient.sh 47` | `attempt#1 RESUME from ep94` |
| 恢复后 | `Restored all states from the checkpoint` |
| 70 秒后 | **ep98，1.70 it/s（与中断前一致）**；s53 全程未受影响 |

**净损失 = 1 个 epoch（约 32 秒），而不是整颗种子重跑 2 小时。**

## 两个代价（必须记住）

### 1. `_plot` 的前向推理是真正的尾部开销，不是 export

`--no-plots` 关掉的 `_plot` 里有两次前向推理：`Took 79.99 (0.26x)`——
**20.64 秒验证音频跑了 80 秒**，两次合计约 160 秒/种子。

而 `export` 本身只是把权重 dump 成 JSON（`_json.dump(model_dict, fp)`），瞬间完成。
所以"让导出变快"这个方向是错的，要省时间只能省 `_plot`。

### 2. 续训后该 epoch 的 shuffle 序列与"不中断连跑"不同

`train_dataloader.shuffle=True`，但 batch 顺序**不在** ckpt 里。
续训后从该 epoch 起，序列与连跑不一致。

这是统计学上的微小差异，不是 bug；但**严格逐 batch 复现做不到**。做敏感性分析时要记住这一点。

## 15 种子统计的可比性

- 迁移/续训不会改变 `epoch` 数、模型结构、优化器超参，**`top1` 与 head 收益的统计口径不受影响**。
- 唯一差异是中断点之后的部分 epoch 的 batch 顺序。对 300 epoch 的收敛量级影响可忽略。
- 若将来要做严格的逐种子 bit-wise 复现，需在 `train_dataloader` 里传 `generator` 并把其状态存进 ckpt——目前没做。

## 主动迁移为什么不划算（用真实速率算过）

有了续训能力，重算了"bw 的 s53 搬到即将空出的机器"：

| 方案 | 完成时刻 | 对比 |
|---|---|---|
| [A] 现状：bw 双路跑完 | **1.94 h** | 基准 |
| [B] s53 转 bw2（0.89h 后空出，2.16 it/s） | 2.42 h | **+0.48 h（更慢）** |
| [C] s53 转 z100（0.98h 后空出，1.9 it/s） | 2.72 h | **+0.78 h（更慢）** |

**瓶颈不在"能不能迁"，在"等机器空出来的时间 > 迁移省下的时间"。**
所以续训能力的价值定位是**故障自愈**，不是主动优化调度。

## 复现命令

```bash
# 从头训练（行为与之前完全一致）
python train_seeded.py <data_cfg> <model_cfg> <learn_cfg> <outdir> <seed> --no-show --no-plots

# 续训（outdir 从 ckpt 自动反推，不必传）
python train_seeded.py <data_cfg> <model_cfg> <learn_cfg> \
    --resume-from <ckpt路径> <seed> --max-epochs 300 --no-show --no-plots

# 带自愈的启动（推荐）
bash scripts/p0a_run_resilient.sh <seed>
```

## 部署状态

| 机器 | train_seeded.py | p0a_run_resilient.sh |
|---|---|---|
| bw | `/workspace/TinyAmpLab/p0a/cfg/` | `/workspace/TinyAmpLab/p0a/run_resilient.sh` |
| bw2 | `/public/home/spotmaverick/nam_proj/cfg/` | `/public/home/spotmaverick/p0a_run_resilient.sh` |
| z100 | `/work/home/spotmaverick/nam_project/p0a_train/` | `/work/home/spotmaverick/nam_project/p0a_run_resilient.sh` |

## 踩过的坑（避免重复）

1. **`set -u` + 未定义变量**：`${2:-$P0A_ROOT/...}` 在 `P0A_ROOT` 未设时直接退出，脚本静默不干活。必须先给默认值再引用。
2. **flag 判断写成 `"--no-plots" in flags`**：`flags` 里存的是去掉 `--` 的名字，这样写恒为 False，
   结果 `make_plots` 一直是 True，白跑了 160 秒/种子的绘图。**已修，并加了 6 个组合的回归测试。**
3. **改了脚本却没覆盖正式路径**：只传成 `train_seeded_resume.py` 验证，实际运行的 `train_seeded.py` 还是旧版，
   于是自愈时还在报 `NotADirectoryError`。验证完必须覆盖正式路径。
4. **`setsid nohup ... &` 在 SSH 会话结束时仍被杀**：要用 `setsid -f bash -c "..."` 双重分离。
