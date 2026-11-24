# 新管箱统一转换: 48kHz 单声道 float32, 命名 <tag>_{train,val,test}_{input,target}.wav
import numpy as np, soundfile as sf, os
from pathlib import Path
ND = Path("/work/home/spotmaverick/nam_data")
OUT = Path("/work/home/spotmaverick/nam_project/data")

def conv(src, dst):
    x, sr = sf.read(src)
    if x.ndim > 1: x = x[:, 0]
    n = int(len(x) * 48000 / sr)
    y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    sf.write(dst, y, 48000, subtype="FLOAT")
    return round(n / 48000, 1)

# 类型A: trainval/test 双目录 (DRY: trainval/{train,val} + test)
AMPS_A = [
    ("blackstar_ht1",  "Blackstar-HT1-ChOverdrive",  "G050_V100"),
    ("mesa550_burn",   "MesaBoogie-550-ChBurn",      "G050"),
    ("mesa550_crunch", "MesaBoogie-550-ChCrunch",    "G050"),
    ("mesa550_clean",  "MesaBoogie-550-ChClean",     "G050"),
    ("markv_clean",    "MesaBoogie-MarkV-ChClean",   "G050"),
]
# 类型B: trainvaltest 单文件, 自切分 80/10/10 (DRY: trainvaltest.input.wav)
AMPS_B = [
    ("ts9_s1",     "Ibanez-TS9/trainvaltest/V050_T050_D050/V050_T050_D050.trainvaltest.target.wav"),
    ("ts9_s2",     "Ibanez-TS9/trainvaltest/V050_T050_D100/V050_T050_D100.trainvaltest.target.wav"),
    ("bigmuff_diy","DIY-ElectroHarmonix-BigMuff/Vol=6_Tone=2_Sustain=5/target.wav"),
]

for tag, d, s in AMPS_A:
    pairs = [
        ("train", f"{ND}/DRY/trainval/train.input.wav", f"{ND}/{d}/trainval/{s}/{s}.train.target.wav"),
        ("val",   f"{ND}/DRY/trainval/val.input.wav",   f"{ND}/{d}/trainval/{s}/{s}.val.target.wav"),
        ("test",  f"{ND}/DRY/test/test.input.wav",      f"{ND}/{d}/test/{s}/{s}.test.target.wav"),
    ]
    for sp, xi, yi in pairs:
        if not os.path.exists(yi):
            print(f"[SKIP] {tag}/{sp}: target 不存在 {yi}")
            continue
        n = conv(xi, f"{OUT}/{tag}_{sp}_input.wav")
        m = conv(yi, f"{OUT}/{tag}_{sp}_target.wav")
        print(f"{tag}_{sp}: {n}s input / {m}s target")

X_FULL = f"{ND}/DRY/trainvaltest/trainvaltest.input.wav"
for tag, yp in AMPS_B:
    if not os.path.exists(yp):
        print(f"[SKIP] {tag}: target 不存在")
        continue
    x, sr = sf.read(X_FULL)
    y, srt = sf.read(yp)
    if y.ndim > 1: y = y[:, 0]
    # 统一转 48k
    n = int(len(x) * 48000 / sr); x48 = np.interp(np.linspace(0, len(x)-1, n), np.arange(len(x)), x).astype(np.float32)
    m = int(len(y) * 48000 / srt); y48 = np.interp(np.linspace(0, len(y)-1, m), np.arange(len(y)), y).astype(np.float32)
    L = min(len(x48), len(y48))
    x48, y48 = x48[:L], y48[:L]
    t_train, t_val = int(L * 0.8), int(L * 0.9)
    sp = {"train": (0, t_train), "val": (t_train, t_val), "test": (t_val, L)}
    for spname, (a, b) in sp.items():
        sf.write(f"{OUT}/{tag}_{spname}_input.wav", x48[a:b], 48000, subtype="FLOAT")
        sf.write(f"{OUT}/{tag}_{spname}_target.wav", y48[a:b], 48000, subtype="FLOAT")
    print(f"{tag}: {L/48000:.1f}s 全段 80/10/10 切分完成")
print("=== 转换完成 ===")
