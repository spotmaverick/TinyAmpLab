"""Gilbert 乘法器摆幅 Vsat 定标扫描 (analog_sim_wavenet.py 配套工具)

用途: 换架构/换权重时, 必须先跑本脚本确定 Vsat, 否则物理仿真结果无意义。
  lite 残差流从 0.04 累积到 2.2、preact 峰值 14.6，沿用 LGW 的 Vsat=1.5
  会让 Gilbert 三次失真把 NRMS 从 0.087 恶化到 0.636。

运行: 智算服务器 (CPU 即可, 无需 DCU)
  cd /work/home/spotmaverick/nam_project && python vsat_scan.py

输出: Vsat → NRMS 曲线。选取 NRMS 相对 6-bit 基线退化 <15% 的最小 Vsat。
  当前 lite 选 Vsat=16 (退化 +11%), 已写入 analog_sim_wavenet.py 的 P 字典。
"""
import sys, torch, soundfile as sf
sys.argv = ['x']
sys.path.insert(0, '/work/home/spotmaverick/nam_project')
import analog_sim_wavenet as A

BASE = '/work/home/spotmaverick/nam_project'
sd = torch.load(f'{BASE}/arch_v2/lite_s7.pt', map_location='cpu')
sd = {k: (v.float() if torch.is_tensor(v) else v) for k, v in sd.items()}
sdq = {k: (A.quant6(v) if torch.is_tensor(v) else v) for k, v in sd.items()}
x, _ = sf.read(f'{BASE}/data/tsa15_test_input.wav')
x = x[:240000].astype('float32')
xt = torch.from_numpy(x)[None, None]
y, _ = sf.read(f'{BASE}/data/tsa15_test_target.wav')
yt = torch.from_numpy(y[:240000].astype('float32'))
valid = xt.shape[-1] - A.RF + 1
ref = yt[-valid:]

# 关掉其他非理想因素，只留 Gilbert 失真，隔离 Vsat 影响
A.P['SIGMA_MISMATCH'] = 0
A.P['DRIFT'] = 0
A.P['SIGMA_OFFSET'] = 0
A.P['SNR_DB'] = 999

print('=== Vsat 扫描（仅 Gilbert 失真，其余关闭）===')
print(f'{"Vsat":>8} {"prod/Vsat@peak14":>18} {"NRMS":>10}')
for vsat in [1.5, 3, 5, 8, 12, 16, 20, 30]:
    A.P['V_SAT'] = vsat
    g = torch.Generator().manual_seed(1000)
    dev = A.Dev(sdq, g)
    with torch.no_grad():
        o = A.chunked(dev, A.physical_fwd, xt)[0, 0]
        o = o[-(xt.shape[-1]-A.RF+1):]
        nrms = float((o.double()-ref.double()).norm()/ref.double().norm())
    print(f'{vsat:>8.1f} {14/vsat:>18.2f} {nrms:>10.4f}')

print('\n=== 参考: LGW 的 Vsat=1.5 下 NRMS = 0.3273 (fp32 0.1831) ===')
