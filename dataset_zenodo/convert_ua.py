# UA 1176LN Limiter: 多段配对 → 统一 48k float32, <tag>_{train,val,test}_{input,target}.wav
# 取单段内连续区间避免拼接跳变: bass1 前 152.5s(train)+20.6s(val), bass2 前 38.1s(test)
import numpy as np, soundfile as sf, glob, re
from pathlib import Path
ND = Path("ua_extract")
OUT = Path("../sync/data")
OUT.mkdir(parents=True, exist_ok=True)

def conv48(p):
    x, sr = sf.read(str(p))
    if x.ndim > 1: x = x[:, 0]
    n = int(len(x) * 48000 / sr)
    return np.interp(np.linspace(0, len(x)-1, n), np.arange(len(x)), x).astype(np.float32)

def find_target(split, base):
    hits = glob.glob(str(ND / "1176LN-Limiter" / split / "*" / f"*.{base}.target.wav"))
    return hits[0] if hits else None

plan = [
    ("train", "trainval", "idmt-afx-bass1", 152.5),
    ("val",   "trainval", "idmt-afx-bass1", 20.6),   # 紧接 train 之后
    ("test",  "test",     "idmt-afx-bass2", 38.1),
]
for sp, src_split, base, dur in plan:
    xi = ND / "DRY" / src_split / f"{base}.input.wav"
    yi = find_target(src_split, base)
    if not yi:
        print(f"[SKIP] {sp}: target 缺失"); continue
    x = conv48(xi); y = conv48(yi)
    start = int(152.5*48000) if sp == "val" else 0
    n = int(dur*48000)
    x, y = x[start:start+n], y[start:start+n]
    L = min(len(x), len(y))
    sf.write(OUT/f"ua1176_{sp}_input.wav", x[:L], 48000, subtype="FLOAT")
    sf.write(OUT/f"ua1176_{sp}_target.wav", y[:L], 48000, subtype="FLOAT")
    print(f"ua1176_{sp}: {L/48000:.1f}s  <- {base}")
print("=== UA 转换完成 ===")
