# TinyAmpLab 当前状态（2026-11-25 08:05）

> 目标：用物理元器件可实现的神经网络做数字管箱/效果器建模，在参数规模、响应时间、
> 音频连续性上争取优势，对标并力争达到/超越 NAM A2 Lite。所有训练/推理/仿真在智算服务器 DCU 上。
> 完整实验账本见 `docs/reports/MODEL_SUMMARY.md`；本文件只给当前快照。

## 一句话结论

自研 LGW-TCN 族已被证实效率显著低于官方 WaveNet 族（同协议 standard 0.0653 vs LGW 冠军 0.1736，2.7 倍）。当前最优解是 **WaveNet lite（6553 参数、85.3ms RF、test NRMS 0.0785）**。已撤回早前「短 RF 代价随容量反转」的错误结论。下一步是把 lite 放进物理器件级仿真，验证其在真实非理想下的可用性。

## 已确立（有实测支撑）

| 结论 | 关键数据 |
|---|---|
| 协议红利真实 | 我方协议（300ep 余弦 + batch192000）给官方架构 −9.7%、给 LGW −26% |
| 架构效率是主瓶颈 | 同协议 standard 0.0653 vs LGW 冠军 0.1736（2.7×） |
| LGW 族的扩张均为负收益 | 拉长 RF 11.3→85.3ms：0.1736→0.1893；dual 门控 4741 参数仅 0.2259 |
| lite 是性价比拐点 | 6553→13801 参数翻 2.1×，NRMS 只降 16.8%（0.0785→0.0653） |
| lite 的 RF 拐点 | 85.3ms 0.0785 → 64ms 0.0796（+1.4%）→ 42.7ms 0.0885（+12.8%）→ 21.4ms 0.1029（+31.1%） |
| 大容量档也依赖长 RF | standard 42.7ms 0.0727 vs 85.3ms 0.0653（+11.3%），与 lite 同向 |
| 物理器件退化很硬（LGW） | fp32 0.1831 → 6-bit 0.1976 → 全器件 0.3273±0.0198 |

## 已撤回 / 未成立

- **「RF 缩放代价随容量反转」（nano −4.9%、feather −0.8%）**：补测发现 nano 种子方差 27%（s7 0.1412 vs s11 0.1793），该差异是噪声。且补测只跑 seed11 的 rf50、缺同种子全 RF 基线，配对口径不成立。
- **「同等参数超越 A2 Lite」**：早期结论基于复现基线（0.2589），官方包真实水平远高于复现，不成立。
- **`_rf50` 的"结构退化"**：我初判过度——官方原序列本就有重复（唯一值 10/13），rf50 后 9/13，主体结构保留。真正原因是种子方差与配对缺失。

## 正在跑（DCU，8 卡满载）

| 作业组 | 数量 | 目的 | 状态 |
|---|---:|---|---|
| seed 11 全 RF 基线（nano/feather/lite/standard） | 4 | 补齐配对，判定 RF 规律 | RUNNING |
| Marshall 泛化（lite/standard × 前级/喇叭） | 4 | 新音箱泛化验证 | RUNNING |
| lite 物理器件级仿真（含 sanity 对拍） | 1 | 验证 lite 真实非理想下的可用性 | PENDING（等卡） |

## 数据集

- `tsa15`：主实验集，152.5/20.6/38.1s。全部架构结论基于它。
- `marshall_jvm410h_chod1` / `_spk`：新接入，Zenodo 10892012，2.76GB，MD5+全条目 CRC 校验通过。210/30/60s 按时序切（无独立 test 录音）。28 组旋钮 × 双采集点可用。
- UA1176 已接入；Marshall 已接入并已上传 DCU。

## 关键限制（接手者必读）

1. **RF 规律未定案**：等 seed11 全 RF 基线收敛，做同种子配对后再判。
2. **WaveNet 阶梯架构不存在「只缩短 RF」的干净对照**：13 层互异 2 的幂最小和 8191，远超 RF2049 所需；缩放必引入轻微 dilation 重复，只能靠多种子压噪声。
3. **物理仿真尚未验证 lite**：现有 0.3273 是 LGW 冠军的数字，不是 lite 的。
4. 本地 Mac 无 torch，所有训练/仿真只能在 DCU 跑；登录节点无 DCU，必须 sbatch 到计算节点。

## 复现入口

```bash
# 架构搜索（第 4 参数指定数据集）
sbatch sbatch_arch2.sh lite 300 7 marshall_jvm410h_chod1

# 物理器件仿真：先对拍，再全量
python src/analog/analog_sim_wavenet.py --sanity
python src/analog/analog_sim_wavenet.py

# 数据转换 / 轮询 / 清理
python src/tools/convert_marshall.py --list
./src/tools/poll_arch2.sh 900 12
./src/tools/cleanup.sh --apply
```

仓库：`https://github.com/spotmaverick/TinyAmpLab`（Public，SSH remote）。
最新提交见 `git log --oneline -5`。
