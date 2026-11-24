"""纯 NumPy 推理：双组门控 Gated-Block，与 train_nam.py 同一前向。"""
import numpy as np
import soundfile as sf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EARLY = ROOT / "experiments" / "early_block"

NUM_BLOCKS = 40
MAT_DIM = 4
ALPHA = 0.15
GAIN_PRE = 2.2
BLOCK_CLAMP = 4.0
OUTPUT_SAT = 1.0
SAMPLE_RATE = 48000

weight_file = str(EARLY / "trained_weights_dual40.npy")
input_wav = str(EARLY / "test_dry.wav")
output_wav = str(EARLY / "nam_infer_out.wav")

W_flat = None


def load_weights(path=weight_file):
    w = np.load(path)
    expected = NUM_BLOCKS * MAT_DIM * MAT_DIM
    if w.size != expected:
        raise ValueError(f"权重数量不匹配：期望 {expected}（{NUM_BLOCKS}×{MAT_DIM}×{MAT_DIM}），实际 {w.size}")
    return w.reshape(NUM_BLOCKS, MAT_DIM, MAT_DIM)


def gated_chain(x: np.ndarray, W=None):
    global W_flat
    if W is None:
        if W_flat is None:
            W_flat = load_weights()
        W = W_flat
    audio = x.astype(np.float32).copy() * GAIN_PRE
    for b in range(NUM_BLOCKS):
        taps = []
        for k in range(MAT_DIM):
            xk = np.roll(audio, k)
            if k > 0:
                xk[:k] = 0
            taps.append(xk)
        out = W[b] @ np.stack(taps, axis=0)       # (4, N)
        s1, g1 = out[0], out[1]
        s2, g2 = out[2], out[3]
        s = s1 * np.tanh(g1) + s2 * np.tanh(g2)
        y_nl = np.where(s >= 0, s, ALPHA * s)
        audio = np.clip(y_nl + audio, -BLOCK_CLAMP, BLOCK_CLAMP)
    return OUTPUT_SAT * np.tanh(audio / OUTPUT_SAT)


if __name__ == "__main__":
    wav, sr = sf.read(input_wav)
    if len(wav.shape) > 1:
        wav = wav[:, 0]
    out = gated_chain(wav)
    sf.write(output_wav, out, sr)
    print(f"推理完成，输出：{output_wav}")
