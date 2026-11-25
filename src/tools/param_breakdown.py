"""参数分解：A2 Lite 的 6553 参数花在哪里？哪些是模拟实现的真实成本？

数学视角：模拟电路的成本不是参数总数，而是：
- 乘法器数量（MACs/sample）—— Gilbert cell 数量
- 延迟线长度 —— 模拟存储单元
- 权重存储 —— 浮栅单元

一个 1x1 conv 和 3-tap dilated conv 的数字参数数相同，但模拟实现成本差 3 倍。
"""
from pathlib import Path
import json

# 官方 lite 架构（来自 wavenet_official_dcu.py）
GROUPS = [
    (1, 12, [1,2,4,8,16,32,64], 6, False),
    (12, 6, [128,256,512,1,2,4,8,16,32,64,128,256,512], 1, True),
]

print("=" * 74)
print("A2 Lite 参数分解（按算子类型）")
print("=" * 74)
total = 0
for gi, (in_size, ch, dils, head_out, head_bias) in enumerate(GROUPS):
    rechannel = in_size * ch
    conv = len(dils) * (ch * ch * 3 + ch)  # 3-tap 全连接 + bias
    mixer = len(dils) * (1 * ch)           # input_mixer: cond=1 -> ch
    layer1x1 = len(dils) * (ch * ch + ch)
    head = ch * head_out + (head_out if head_bias else 0)
    sub = rechannel + conv + mixer + layer1x1 + head
    total += sub
    print(f"Group {gi}: ch={ch}, layers={len(dils)}")
    print(f"  rechannel 1x1  : {rechannel:>5}  (模拟: 1x MAC)")
    print(f"  dilated conv   : {conv:>5}  (模拟: 3x MAC + {len(dils)*2*max(dils)} 延迟抽头)")
    print(f"  input mixer    : {mixer:>5}  (模拟: 1x MAC)")
    print(f"  layer 1x1      : {layer1x1:>5}  (模拟: 1x MAC)")
    print(f"  head           : {head:>5}")
    print(f"  小计           : {sub:>5}")
print(f"总计: {total}")

print()
print("=" * 74)
print("模拟实现成本分解")
print("=" * 74)
macs = 0
delay_taps = 0
for gi, (in_size, ch, dils, head_out, head_bias) in enumerate(GROUPS):
    macs += in_size * ch
    macs += len(dils) * ch * ch * 3   # conv 是 3-tap
    macs += len(dils) * ch            # mixer
    macs += len(dils) * ch * ch       # 1x1
    macs += ch * head_out
    delay_taps += sum(2 * d for d in dils)
print(f"每采样 MACs（Gilbert cell 数）: {macs}")
print(f"延迟线抽头总数: {delay_taps} = {delay_taps/48000*1000:.1f}ms 历史")
print(f"权重存储: {total} 浮栅单元")
print()
print("关键观察：")
print(f"  dilated conv 占参数 {len(GROUPS[0][2])*12*12*3 + len(GROUPS[1][2])*6*6*3} / {total} "
      f"= {(len(GROUPS[0][2])*12*12*3 + len(GROUPS[1][2])*6*6*3)/total*100:.0f}%")
print("  但它的延迟线成本最高（85ms 历史）")
print("  1x1 conv 参数多但模拟便宜（无延迟线）")
