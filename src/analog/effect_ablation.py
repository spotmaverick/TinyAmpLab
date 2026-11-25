"""器件非理想因素消融 (analog_sim_wavenet.py 配套工具)

用途: 物理仿真退化超标时, 逐项隔离量化/Gilbert/失配/温漂/失调/噪声的贡献,
  定位主因。当前结论: lite 的退化几乎全部来自 Gilbert 三阶失真, 其余 <2%。

运行: 智算服务器 (CPU 即可, 无需 DCU)
  cd /work/home/spotmaverick/nam_project && python effect_ablation.py
"""
import sys, torch, soundfile as sf, numpy as np

sys.argv = ['x']
sys.path.insert(0, '/work/home/spotmaverick/nam_project')
import analog_sim_wavenet as A

BASE = '/work/home/spotmaverick/nam_project'
sd = torch.load(f'{BASE}/arch_v2/lite_s7.pt', map_location='cpu')
sd = {k: (v.float() if torch.is_tensor(v) else v) for k, v in sd.items()}
x, _ = sf.read(f'{BASE}/data/tsa15_test_input.wav')
x = x[:240000].astype('float32')
xt = torch.from_numpy(x)[None, None]
y, _ = sf.read(f'{BASE}/data/tsa15_test_target.wav')
y = y[:240000].astype('float32')
yt = torch.from_numpy(y)
valid = xt.shape[-1] - A.RF + 1
ref = yt[-valid:]

# 逐一打开单个非理想因素
configs = {
    'baseline(no quant,no effect)': dict(BETA_H3=0, SIGMA_OFFSET=0, SIGMA_MISMATCH=0, DRIFT=0, SNR_DB=999, quant=False),
    'quant only':                   dict(BETA_H3=0, SIGMA_OFFSET=0, SIGMA_MISMATCH=0, DRIFT=0, SNR_DB=999, quant=True),
    'quant + Gilbert(BETA)':        dict(BETA_H3=0.002, SIGMA_OFFSET=0, SIGMA_MISMATCH=0, DRIFT=0, SNR_DB=999, quant=True),
    'quant + offset':               dict(BETA_H3=0, SIGMA_OFFSET=0.005, SIGMA_MISMATCH=0, DRIFT=0, SNR_DB=999, quant=True),
    'quant + mismatch':             dict(BETA_H3=0, SIGMA_OFFSET=0, SIGMA_MISMATCH=0.01, DRIFT=0, SNR_DB=999, quant=True),
    'quant + drift':                dict(BETA_H3=0, SIGMA_OFFSET=0, SIGMA_MISMATCH=0, DRIFT=0.03, SNR_DB=999, quant=True),
    'quant + noise(SNR70)':         dict(BETA_H3=0, SIGMA_OFFSET=0, SIGMA_MISMATCH=0, DRIFT=0, SNR_DB=70, quant=True),
    'ALL':                          dict(BETA_H3=0.002, SIGMA_OFFSET=0.005, SIGMA_MISMATCH=0.01, DRIFT=0.03, SNR_DB=70, quant=True),
}

for name, cfg in configs.items():
    A.P['BETA_H3'] = cfg['BETA_H3']
    A.P['SIGMA_OFFSET'] = cfg['SIGMA_OFFSET']
    A.P['SIGMA_MISMATCH'] = cfg['SIGMA_MISMATCH']
    A.P['DRIFT'] = cfg['DRIFT']
    A.P['SNR_DB'] = cfg['SNR_DB']
    src = {k: (A.quant6(v) if cfg['quant'] and torch.is_tensor(v) else v) for k, v in sd.items()}
    g = torch.Generator().manual_seed(1000)
    dev = A.Dev(src, g)
    with torch.no_grad():
        o = A.chunked(dev, A.physical_fwd, xt)[0, 0]
        o = o[-(xt.shape[-1]-A.RF+1):]
        nrms = float((o.double()-ref.double()).norm()/ref.double().norm())
    print(f'{name:<32} NRMS = {nrms:.4f}')
