"""官方口径评测：直接复用 nam.models.losses.esr + init_from_nam

评测口径与官方训练器的 validation 完全一致：
    esr = mean_over_batch( mean((pred-target)^2) / mean(target^2) )
区别仅在于：本脚本在**独立 test 段**上做全段推理，而不是 val 段。

用法（在 4090 / bw 上）：
    python eval_nam_official.py <ckpt_or_nam> <test_input.wav> <test_target.wav>

ckpt 处理：Lightning checkpoint 内 state_dict 的 key 带 "_net." 前缀，本脚本自动剥离并
用 config_model.json 重建模型，再走官方 import/export 链路。

输出：单行 JSON（esr / nrms / corr / n_samples / rf）
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch

from nam.models.losses import esr as official_esr
from nam.models.wavenet._wavenet import WaveNet


def load_wav(p):
    x, sr = sf.read(str(p), dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    return torch.from_numpy(x), sr


def _find_model_cfg(ckpt: Path) -> Path:
    """从 checkpoint 路径反推 config_model.json。

    官方 nam_full 的目录布局：
        <outdir>/<timestamp>/config_model.json
        <outdir>/<timestamp>/lightning_logs/version_N/checkpoints/*.ckpt
    """
    p = ckpt.resolve()
    for parent in p.parents:
        cand = parent / "config_model.json"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"找不到 config_model.json（从 {ckpt} 向上查找）")


def build_from_ckpt(ckpt_path: Path, model_cfg_path: Path):
    cfg = json.load(open(model_cfg_path))
    net_cfg = cfg["net"]["config"]
    model = WaveNet.init_from_config(net_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
    # Lightning 包了一层 "_net."，并带 BaseNet 的采样率元数据
    clean = {}
    for k, v in sd.items():
        while k.startswith("_net."):
            k = k[len("_net."):]
        if "_has_sample" in k or "_sample_rate" in k:
            continue
        clean[k] = v
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if unexpected:
        print(f"[warn] unexpected: {unexpected[:5]}", file=sys.stderr)
    if missing:
        print(f"[warn] missing: {missing[:5]}", file=sys.stderr)
    return model, cfg


def eval_model(model, x, y, device="cuda"):
    """全段推理（不分块，保证与训练时一致的因果对齐）。

    注意：官方 WaveNet 是**因果**的，整段一次前向与分块前向在数学上等价，
    但分块会因感受野边界重复计算而引入对齐误差，因此这里优先整段前向。
    """
    model = model.to(device).eval()
    rf = model.receptive_field
    with torch.no_grad():
        pred = model(x[None, None].to(device))[0, 0].float().cpu()
    off = y.numel() - pred.numel()
    if off < 0:
        raise RuntimeError(f"pred 比 target 长: {pred.numel()} vs {y.numel()}")
    tgt = y[off:].float()

    e = float(official_esr(pred[None], tgt[None]))
    nrms = float(np.sqrt(max(e, 0.0)))
    p, t = pred.double() - pred.double().mean(), tgt.double() - tgt.double().mean()
    corr = float((p * t).sum() / (p.norm() * t.norm() + 1e-12))
    return dict(esr=e, nrms=nrms, corr=corr, n_samples=int(pred.numel()), rf=int(rf))


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    ckpt = Path(sys.argv[1])
    x_path, y_path = Path(sys.argv[2]), Path(sys.argv[3])
    model_cfg = Path(sys.argv[4]) if len(sys.argv) > 4 else _find_model_cfg(ckpt)

    model, _ = build_from_ckpt(ckpt, model_cfg)
    x, sr = load_wav(x_path)
    y, sr2 = load_wav(y_path)
    assert sr == sr2 == 48000, f"采样率不符 {sr}/{sr2}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = eval_model(model, x, y, device)
    res["ckpt"] = str(ckpt)
    res["model_cfg"] = str(model_cfg)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
