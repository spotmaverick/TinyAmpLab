"""官方口径评测（v2）：支持 slimmable 双分支 + 任意 factory 注册架构

与 v1 的区别：
1. **支持 slimmable**：A2 是 `slice_channels_uniform` 可裁剪模型，同一份权重能裁成
   Lite(3ch) / Full(12ch)。`--slim 0` 取 Lite，`--slim 1` 取 Full。
2. **支持任意 factory 架构**：族 B（DiagSSM / RecurrentNet）不是 WaveNet 子类，
   走 `nam.models.factory.init` 重建。
3. 批量模式：一次加载 test 音频，循环评测多个 ckpt，省去重复 IO。

评测口径与官方训练器 validation 完全一致：
    esr = mean_over_batch( mean((pred-target)^2) / mean(target^2) )
区别仅在于在**独立 test 段**上整段推理（不分块，避免感受野边界误差）。

用法：
    python eval_v2.py <ckpt> [--slim 0|1] [--cfg path]
    python eval_v2.py --batch <glob> [--slim 0|1]
"""
import argparse
import glob
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nam.models.losses import esr as official_esr  # noqa: E402
from nam.models import factory as _factory  # noqa: E402

# 族 B 注册（配置里若用到）
try:
    import analog_ssm  # noqa: E402

    _factory.register("DiagSSM", analog_ssm.DiagSSM.init_from_config, overwrite=True)
    _factory.register("GatedIIR", analog_ssm.GatedIIR.init_from_config, overwrite=True)
except Exception:  # noqa: BLE001
    pass
try:
    import analog_rec  # noqa: E402

    _factory.register(
        "RecurrentNet", analog_rec.RecurrentNet.init_from_config, overwrite=True
    )
except Exception:  # noqa: BLE001
    pass

TEST_X = "/root/autodl-tmp/data/tsa15_test_input.wav"
TEST_Y = "/root/autodl-tmp/data/tsa15_test_target.wav"


def load_wav(p):
    x, sr = sf.read(str(p), dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    return torch.from_numpy(x), sr


def find_model_cfg(ckpt: Path) -> Path:
    for parent in ckpt.resolve().parents:
        cand = parent / "config_model.json"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"找不到 config_model.json（从 {ckpt} 向上查找）")


def build_model(model_cfg_path: Path, ckpt_path: Path, device: str):
    cfg = json.load(open(model_cfg_path))
    name = cfg["net"]["name"]
    net_cfg = cfg["net"]["config"]
    if name == "WaveNet":
        from nam.models.wavenet._wavenet import WaveNet

        model = WaveNet.init_from_config(net_cfg)
    else:
        model = _factory.init(name, kwargs={"config": net_cfg})
    model = model.to(device)

    sd = torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
    clean = {}
    for k, v in sd.items():
        while k.startswith("_net."):
            k = k[len("_net."):]
        if "_has_sample" in k or "_sample_rate" in k:
            continue
        clean[k] = v
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if len(missing) > 3 or len(unexpected) > 3:
        print(f"[warn] missing={missing[:3]} unexpected={unexpected[:3]}", file=sys.stderr)
    return model, cfg


def apply_slim(model, slim):
    """slim: 0.0 -> 最小裁剪（Lite），1.0 -> 全宽（Full）。非 slimmable 模型自动跳过。"""
    if not getattr(model, "is_slimmable", False):
        return False
    model.set_slimming(float(slim))
    return True


def eval_model(model, x, y, device="cuda"):
    model = model.to(device).eval()
    with torch.no_grad():
        pred = model(x[None, None].to(device))[0, 0].float().cpu()
    off = y.numel() - pred.numel()
    if off < 0:
        raise RuntimeError(f"pred 比 target 长: {pred.numel()} vs {y.numel()}")
    tgt = y[off:].float()
    e = float(official_esr(pred[None], tgt[None]))
    p = pred.double() - pred.double().mean()
    t = tgt.double() - tgt.double().mean()
    corr = float((p * t).sum() / (p.norm() * t.norm() + 1e-12))
    nparam = sum(p_.numel() for p_ in model.parameters())
    return dict(esr=e, nrms=float(np.sqrt(max(e, 0.0))), corr=corr,
                n_samples=int(pred.numel()), rf=int(model.receptive_field),
                params=int(nparam))


def run_one(ckpt, model_cfg, x, y, device, slim):
    model, _ = build_model(model_cfg, ckpt, device)
    slimmed = apply_slim(model, slim)
    res = eval_model(model, x, y, device)
    res["slimmed"] = slimmed
    res["slim"] = slim if slimmed else None
    res["ckpt"] = str(ckpt)
    if slimmed:
        res["effective_ch"] = None  # 由调用方从 allowed_channels 推断
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--batch", help="glob 模式，批量评测")
    ap.add_argument("--slim", type=float, default=1.0, help="0=Lite 1=Full")
    ap.add_argument("--cfg", help="显式指定 config_model.json")
    ap.add_argument("--x", default=TEST_X)
    ap.add_argument("--y", default=TEST_Y)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, sr = load_wav(args.x)
    y, sr2 = load_wav(args.y)
    assert sr == sr2 == 48000, f"采样率不符 {sr}/{sr2}"

    if args.batch:
        items = []
        for pat in args.batch.split(","):
            items.extend(sorted(glob.glob(pat)))
        # 每个 run 只取最后一个 ckpt
        best = {}
        for c in items:
            ckpt = Path(c)
            key = str(ckpt.parent.parent.parent)  # <outdir>/<ts>/
            best[key] = ckpt
        targets = list(best.values())
    else:
        targets = [Path(args.ckpt)]

    for ckpt in targets:
        try:
            mcfg = Path(args.cfg) if args.cfg else find_model_cfg(ckpt)
            res = run_one(ckpt, mcfg, x, y, device, args.slim)
            res["tag"] = ckpt.parent.parent.parent.name if "runs" in str(ckpt) else ckpt.parent.name
            print(json.dumps(res, ensure_ascii=False), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ckpt": str(ckpt), "error": f"{type(exc).__name__}: {exc}"},
                             ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
