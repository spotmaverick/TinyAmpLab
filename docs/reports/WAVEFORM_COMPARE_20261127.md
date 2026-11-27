# 干琴输入下的三模型输出波形对比（2026-11-27）

## 1. 口径与假设

| 项 | 设置 |
|---|---|
| 干琴输入 | TSA15 `test_input.wav`，取 8.0–13.0 s（与 P0a `seg1` 同段），48 kHz，5 s / 240000 样本 |
| 真实音箱基线 | TSA15 `test_target.wav` 同段（麦克风采集的音箱输出） |
| 种子拓扑 | `L10-ch32-g8-bigRF`，**s71**（13 种子器件 NRMS 0.1605，最接近中位 0.1613；不是最好种子 s7） |
| NAM Lite | 官方 `lite_tsa15`，6553 参数，RF=4093 |
| NAM Full | 官方 `standard_tsa15`，13801 参数，本项目口径中的 Full / 最大档，RF=4093 |
| 精度与对齐 | FP32；官方 NAM `pad_start=True`，三者 `best_shift=0`，无需额外时间平移 |
| 执行位置 | bw2 单卡 BW，作业 1265438（推理+绘图）与 1265478（差距定位）；本地只保存回传结果 |

**重要边界**：本次是模型的**数字输出**与真实录音对比，不叠加 Gilbert 饱和、漂移、量化等模拟器件退化。
s71 在此段的 NRMS 0.0580 是数字结果，不等同于 P0a 的器件 NRMS 0.1605。
另：这里的 Lite/Full 是官方 NAM 四档中的 `lite` 与 `standard`；A2-slimmable 的 3ch/12ch 档位没有已导出的模型，若需要须另训。

## 2. 全段波形差距（真实音箱为基线）

| 输出 | 参数 | RF | NRMS ↓ | corr ↑ | RMS gain（越接近 1 越好） | 包络 RMS 误差 p95 ↓ | 最差 250 ms 窗口 |
|---|---:|---:|---:|---:|---:|---:|---|
| **Seed s71 L10-ch32-g8-bigRF** | 18945 | 5411 | **0.0580** | **0.9983** | **0.998** | **0.0195** | 0.1180 @ 0.00–0.25 s |
| NAM Lite | 6553 | 4093 | 0.0805 | 0.9969 | 0.989 | 0.0288 | 0.1305 @ 0.00–0.25 s |
| NAM Full/Standard | 13801 | 4093 | 0.0698 | 0.9976 | 0.988 | 0.0246 | 0.1223 @ 0.00–0.25 s |

结论：**种子拓扑在这段干琴上全面领先**。相对 NAM Lite，NRMS 低 28%；相对 Full/Standard，低 17%。
三个模型的最大单样本误差都落在 0.001–0.002 s 的起音瞬间（seed 0.308、Lite 0.264、Full 0.298）；
seed 的尖峰数值最大，但只集中在起音，按包络 p95 和分段 NRMS 它仍是最小的。

## 3. 差距定位：按秒 NRMS

| 秒（段内） | Seed s71 | NAM Lite | NAM Full |
|---:|---:|---:|---:|
| 0 | 0.0786 | 0.0966 | 0.0844 |
| 1 | 0.0566 | 0.0835 | 0.0690 |
| 2 | 0.0510 | 0.0778 | 0.0659 |
| 3 | 0.0511 | 0.0725 | 0.0643 |
| 4 | 0.0486 | 0.0708 | 0.0640 |

定位结果：

1. **起音段（0–0.25 s）是三者共同的最差窗口**，与“最大单样本误差都在 t≈0.001 s”一致；
   这是真实音箱的瞬态建立，模型都有约 0.12–0.13 的相对误差。
2. **1.5–1.875 s 是第二差窗口**（换音/间隙处）。seed 0.0663，Lite 0.1037，Full 0.0868。
3. **尾段（3.5–5 s）差距最稳定**：seed 0.0487，Lite 0.0694，Full 0.0626。
   从包络图看，Lite/Full 在 3.5 s 后持续略高于真实音箱，seed 基本贴合。
4. **RMS gain 漂移**：seed 全程 0.995–1.001；Lite 从 0.979 升到 0.997；Full 从 0.985 升到 0.993。
   NAM 两档在前半段系统性低估输出幅度约 1–2%，seed 没有这个偏置。

## 4. 交付物

目录：`reclaimed/waveform_cmp_20261127/out/`

- `dry_input_8s_13s.wav`：干琴输入
- `real_amp_target_8s_13s.wav`：真实音箱基线
- `seed_s71_L10-ch32-g8-bigRF.wav`：种子拓扑输出
- `nam_lite.wav`：官方 NAM Lite 输出
- `nam_full_standard.wav`：官方 NAM Full/Standard 输出
- `diff_*.wav`：各模型减真实音箱的差值音频
- `waveform_full_5s.png`：5 s 干琴波形 + 三模型与基线的 RMS 包络对比，标注 NRMS/corr/gain/最大误差时刻
- `waveform_zoom_transient.png`：0.805 s 附近 50 ms 原始波形叠加与误差曲线
- `waveform_error_scores.png`：样本误差与评分柱状图
- `metrics.json`、`post_analysis.json`：量化指标与差距定位

脚本：`scripts/waveform_compare.py`、`scripts/waveform_post_analysis.py`

## 4.5 0.5 秒窗口放大对比（本地纯后处理）

对已回传的 5 路 WAV 做放大，不重新推理。取差距最大的 **0–0.5 s 起音段**，并以 **3.5–4.0 s 稳态段**为对照。

| 窗口 / 模型 | NRMS ↓ | corr ↑ | gain | max\|err\| | \|err\|>10% RMS 的样本占比 | 4×125 ms 子窗 NRMS |
|---|---:|---:|---:|---:|---:|---|
| 起音 0–0.5 s · Seed s71 | **0.0953** | **0.9955** | **1.000** | 0.308 | **0.164** | 0.167 / 0.057 / 0.064 / 0.061 |
| 起音 0–0.5 s · NAM Lite | 0.1117 | 0.9940 | 0.979 | 0.264 | 0.271 | 0.168 / 0.092 / 0.086 / 0.089 |
| 起音 0–0.5 s · NAM Full | 0.1002 | 0.9951 | 0.986 | 0.298 | 0.181 | 0.170 / 0.066 / 0.070 / 0.069 |
| 稳态 3.5–4.0 s · Seed s71 | **0.0489** | **0.9988** | **1.000** | **0.098** | **0.052** | 0.043 / 0.053 / 0.047 / 0.053 |
| 稳态 3.5–4.0 s · NAM Lite | 0.0665 | 0.9979 | 1.000 | 0.148 | 0.117 | 0.061 / 0.068 / 0.065 / 0.072 |
| 稳态 3.5–4.0 s · NAM Full | 0.0595 | 0.9982 | 0.996 | 0.144 | 0.094 | 0.052 / 0.064 / 0.057 / 0.065 |

放大后的形态结论：
- **起音段的差距是波形形状差，不是简单增益差**。Seed 在这半秒 gain=1.000，但 NRMS 仍有 0.095；NAM Lite gain=0.979，系统性 undershoot。
- **误差集中在前 125 ms**：三家首个 125 ms 子窗 NRMS 都在 0.167–0.170，几乎相同；此后 seed 迅速降到 0.057，Lite 仍在 0.086–0.092，Full 在 0.066–0.070。即起音起始沿是共同难点，之后 seed 收敛更快。
- **25 ms 原始波形细窗**显示：真实音箱在跳变沿有削顶/折角，三种模型都能跟上大轮廓，差异主要落在折角的位置与幅度；NAM Lite 在起音段整段偏低，Full 介于两者之间，seed 的包络最贴合。
- **稳态段同样排序**：seed 0.0489 < Full 0.0595 < Lite 0.0665，且 seed 的增益偏差最小（0.000 vs 0.004 / 0.000，但 Lite 此处 gain 接近 1 是靠后段平均，逐段 NRMS 仍偏高）。

新增产物：`waveform_zoom_0p5s_attack.png`、`waveform_zoom_0p5s_steady.png`、`waveform_zoom_0p5s_detail.png`、`zoom_0p5s_metrics.json`；脚本 `scripts/waveform_zoom_0p5s.py`（本地运行，`.venv_nam/bin/python`）。

## 5. 复现

在 bw2 计算节点（源模型仅临时存在，作业后已清理）：

```bash
source ~/p26/bw2env.sh
export WAVECMP_ROOT=/public/home/spotmaverick/nam_proj
export PYTHONPATH=/public/home/spotmaverick/t29:/public/home/spotmaverick/nam_proj:$PYTHONPATH
$PY -u waveform_compare.py
$PY -u waveform_post_analysis.py
```

## 6. 待确认

- 若“Full”指的是 A2-slimmable 12ch/15697 参数档，而不是官方 Standard，需要提供或重训导出的 `.nam`，本对比可原样复跑。
- 若要对比**模拟器件输出**（含 V_SAT、Gilbert 三阶失真、漂移），应另跑 P0-C 两级摆幅扫描后替换 s71 输出；本次数字输出不能直接代表流片后波形。
