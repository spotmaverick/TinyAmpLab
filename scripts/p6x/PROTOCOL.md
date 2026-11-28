# P6X 实验协议：三次独立实现 + 双硬件交叉验证

> 本文件是**契约**。三次实验必须严格遵守同一份协议，代码必须独立手写、互不参考。
> 协议本身可以先写，因为协议规定的是"测什么、怎么定义"，不是"怎么算"。

## 1. 为什么要这样做

用户要求：每个实验至少三遍、代码每次重写、三次一致才算结论、且至少用两种硬件。

- **独立重写能抓什么**：实现错误（索引错位、分组轴错、因果对齐错、量化口径错、求和维度错）。
  这是真实存在的高发风险——本项目 P5 阶段就犯过 `_conv_forward` patch 到抽象基类、
  Gilbert 求和维度选错、condition 长度错位三类错误。
- **独立重写抓不了什么**：蒙特卡洛噪声、模型假设偏差（V_SAT∝I_TAIL 的电路假设、
  数据集采集链路）。这些必须靠**同实现内加大 MC**与**电路级复核**解决，不靠重写。

因此判据分两层，两层都过才算结论：

| 层 | 判据 | 抓什么 |
|---|---|---|
| L1 实现一致性 | 三次实现的同一指标差 **< 1e-3**（相对），sanity 各自均 < 1e-5 | 实现错误 |
| L2 统计稳定性 | 同实现内 MC=4 → MC=16，中位变化 **< 5%** | 蒙特卡洛噪声 |

## 2. 硬件矩阵

至少两种硬件，已确定：

| 实现 | 硬件 | 说明 |
|---|---|---|
| Impl A | bw2（海光 BW，gfx936，torch 2.9.0，DTK 26.04） | 主算力 |
| Impl B | z100（海光 Z100SM，17 GB，torch 2.1.0 + HIP 5.7） | **不同代 DCU** |
| Impl C | bw2（不同节点）或 Mac CPU（缩小规模） | — |

跨代 DCU 的 float32 BLAS 结果会有微小差异，L1 的 1e-3 门槛已覆盖。

## 3. 被测对象

15 个模型：目标拓扑 `L10-ch32-g8-bigRF` 的 13 个种子（s7 s11 s23 s31 s37 s41 s43 s47 s53 s59 s61 s67 s71），
以及官方训练的 `standard`（13801 参数）与 `lite`（6553 参数）。

权重来源：`.nam` 文件，统一用 `nam.models.init_from_nam` 解析（这是权重格式的权威读法，不属于"上一次实验的代码"）。

## 4. 信号与分段

    采样率      48000 Hz
    输入        data/tsa15_test_input.wav
    目标        data/tsa15_test_target.wav
    分段起点    0.0 / 8.0 / 16.0 s，每段 5.0 s（与 P0a 的 seg0/seg1/seg2 同口径）
    聚合        每段内 MC 值取 rms → 3 段取中位（median）

## 5. 指标定义

    NRMS(p, y) = ||p - y||_2 / ||y||_2
    digital NRMS  : 关掉全部非理想性（理想浮点前向）后与 target 的 NRMS
    device  NRMS  : 打开全部非理想性后的 NRMS

## 6. 非理想性参数（DIALECT，必须逐字一致）

    BITS            = 6          # 权重存储位宽
    SIGMA_MISMATCH  = 0.01       # 权重失配 1σ（逐权重元素）
    DRIFT           = 0.03       # 全局温漂（逐权重张量一个标量，U(-D, D)）
    BETA_H3         = 0.002      # Gilbert 三次项
    V_SAT           = 16.0       # 摆幅（可覆盖）
    SIGMA_OFFSET    = 0.005      # 差分对输入失调（逐通道）
    SNR_DB          = 70.0       # 节点热噪声

### 6.1 量化（口径必须与器件的存储行为一致）

逐输出通道（权重的 dim 1 及以后求 amax）：

    levels = 2**(BITS-1) - 1            # = 31
    scale  = amax / levels
    q      = clamp(round(w / scale), -32, 31) * scale

偏置不量化。

### 6.2 Gilbert cell（乘法器）

权重元素 w 与激活 x 的乘积 p = w·x，按输入通道维求和：

    u = p / V_SAT
    y = Σ_i  V_SAT · tanh( u_i + BETA_H3 · u_i³ )

（等价于 V_SAT·tanh(p·(1+BETA_H3·(p/V_SAT)²)/V_SAT)，两种写法互为代数变形，
 允许不同实现选用不同写法——这正是独立验证的价值之一。）

### 6.3 差分对激活

    z ← z + offset_c ,  offset_c ~ N(0, SIGMA_OFFSET)，逐通道固定
    Tanh      : tanh(z)
    LeakyReLU : z>0 ? z : 0.01·z

### 6.4 热噪声

    n ~ N(0, 0.25 / 10**(SNR_DB/20))，加在激活输出与 layer1x1 输出上

## 7. 随机数契约（关键：让三次实现可逐位对拍）

若随机数按调用顺序抽取，三份实现的前向顺序不同就会拿到不同 MC 样本，永远对不上。
因此**随机数的种子必须由名字决定，而不是由调用顺序决定**：

    seed_of(*parts) = CRC32("|".join(str(p) for p in parts))
    gen = torch.Generator(device="cpu").manual_seed(seed_of(...))

| 随机量 | 种子 parts | 形状 |
|---|---|---|
| 权重失配 | `("mm", tensor_name, mc)` | 同权重 |
| 权重漂移 | `("drift", tensor_name, mc)` | 标量 |
| 激活失调 | `("off", activation_kind, channels, mc)` | (1, C, 1) |
| 热噪声 | `("noise", node_id, mc)` | 同被加量 |

`tensor_name` 用 state_dict 的键名（去掉 `_net.` 前缀），`node_id` 形如 `"0.3.post"` / `"0.3.res"`。

所有 mask 在 CPU 上生成后搬到计算设备——保证跨硬件拿到**同一组** mask。

## 8. 前向结构（契约级描述）

    RF        = 1 + Σ_arrays Σ_layers (K-1)·d
    输入左补 RF-1 个零（因果、pad_start）
    cond      = 补零后的输入
    head_acc  = 空
    for 每个 layer_array a:
        x = rechannel_a(x)                                  # 1x1，无 bias
        for 每层 l:
            zc  = causal_valid_conv(x, w_conv, d, groups=g)  # 无 pad，靠末尾对齐裁剪
            m   = input_mixer(c)                            # 1x1 无 bias
            m   = m[..., -len(zc):]
            z   = zc + m
            post= act(z + offset)
            lo  = layer1x1(post)                            # 1x1 带 bias
            term= post[..., -(min(len(x),len(cond)) - (RF_a - 1)):]
            head_acc = term            if 为空
                     = head_acc[..., -len(term):] + term   否则
            x   = x[..., -len(lo):] + lo
        head_out = head_rechannel_a(head_acc)                # 1x1，bias 视模型而定
        x        = x[..., -out_length_a:]                    # out_length_a = min(len, len) - (RF_a - 1)
        head_acc = head_out                                  # 下一个 array 的初值
    y = head_scale · head_acc

其中 `RF_a` 是该 array 自身的感受野（不含 head_rechannel，因其为 1 tap）。

> 每个实现都必须**独立**把这段结构写成代码，并通过与官方 `nam` 包前向的
> `max|Δ| < 1e-5` 对拍（理想参数下）。对拍不过就不出数。

## 9. 三次实验要产出的表

对每个模型：

| 字段 | 含义 |
|---|---|
| `sanity` | 理想前向 vs 官方前向的 max\|Δ\| |
| `digital` | 数字 NRMS |
| `grid[k][v]` | k∈{1,32} × V∈{16,96} 的 device NRMS |
| `free` | grid[1][16]（不加任何手段） |
| `best` | grid 中最小值（免费 k + 付费摆幅都用上） |

汇总量：

- `median_best_13` = 13 个种子 `best` 的中位 → **决定选型的数**
- `head_concentration` = 官方 std / lite：`head_only@96` 与 `all@96` 相对 `free` 的改善占比

## 10. 判定

满足 L1 + L2 后，若三次实现的 `median_best_13` 差 < 1e-3 且与既有结果（0.0843）一致，
则结论成立；若不一致，继续做第四次、第五次，直到至少三次一致，并把不一致的那次
作为"被推翻的实现"记录在案（这是最有价值的信息，说明有一份实现写错了）。
