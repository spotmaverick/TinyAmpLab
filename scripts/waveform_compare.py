"""Waveform comparison: seeded topology vs official NAM lite/full vs real amp.

Run on a DCU compute node only (no local inference). The script loads:
  - seed: L10-ch32-g8-bigRF, seed s71, exported .nam
  - nam_lite: official NAM lite_tsa15 (6553 params)
  - nam_full: official NAM standard_tsa15 (13801 params; "Full/最大档" in this project)
  - real amp: TSA15 test target.wav (microphoned speaker output)

It writes aligned WAVs, metrics JSON and waveform comparison PNGs.
"""
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(os.environ.get("WAVECMP_ROOT", "/public/home/spotmaverick/nam_proj"))
DATA = ROOT / "data"
MODELS = ROOT / "waveform_cmp" / "models"
OUT = ROOT / "waveform_cmp" / "out"

SEG_START_S = 8.0
SEG_LEN_S = 5.0
SR = 48000

MODEL_SPECS = [
    ("seed_s71_L10-ch32-g8-bigRF", MODELS / "s71.nam", "Seeded topological candidate"),
    ("nam_lite", MODELS / "nam_lite_tsa15.nam", "Official NAM Lite"),
    ("nam_full_standard", MODELS / "nam_standard_tsa15.nam", "Official NAM Standard (Full)"),
]


def load_nam(path: Path):
    from nam.models._from_nam import init_from_nam

    with open(path, "r") as fp:
        model = init_from_nam(json.load(fp))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


@torch.inference_mode()
def render(model, x: np.ndarray) -> np.ndarray:
    xt = torch.from_numpy(x.astype(np.float32))[None, :]
    if torch.cuda.is_available():
        xt = xt.cuda()
    y = model(xt, pad_start=True)
    if isinstance(y, (tuple, list)):
        y = y[0]
    y = y[0].float().cpu().numpy()
    assert y.shape == x.shape, f"model output {y.shape} != input {x.shape}"
    return y


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def best_shift_corr(pred: np.ndarray, target: np.ndarray, max_shift: int = 16):
    best = (-2.0, 0)
    n = len(target)
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            p = pred[s:]
            t = target[: n - s]
        else:
            p = pred[: n + s]
            t = target[-s:]
        r = pearson(p, t)
        if r > best[0]:
            best = (r, s)
    return best


def frame_rms(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    if len(x) < win:
        return np.array([np.sqrt(np.mean(x * x))])
    n = 1 + (len(x) - win) // hop
    return np.sqrt(np.array([np.mean(x[i * hop : i * hop + win] ** 2) for i in range(n)]))


def metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    err = pred - target
    r0, shift = best_shift_corr(pred, target)
    if shift >= 0:
        p_aligned, t_aligned = pred[shift:], target[: len(target) - shift]
    else:
        p_aligned, t_aligned = pred[: len(target) + shift], target[-shift:]
    err_aligned = p_aligned - t_aligned
    rms_ref = float(np.sqrt(np.mean(target**2)))
    rms_pred = float(np.sqrt(np.mean(pred**2)))
    win, hop = int(0.02 * SR), int(0.01 * SR)
    e_env = frame_rms(err, win, hop)
    return {
        "nrms": float(np.sqrt(np.mean(err**2)) / (rms_ref + 1e-12)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "corr": pearson(pred, target),
        "corr_best_shift": r0,
        "best_shift_samples": int(shift),
        "nrms_best_shift": float(np.sqrt(np.mean(err_aligned**2)) / (rms_ref + 1e-12)),
        "max_abs_err": float(np.max(np.abs(err))),
        "max_err_time_s": float(np.argmax(np.abs(err)) / SR),
        "rms_pred": rms_pred,
        "rms_target": rms_ref,
        "gain_pred_over_target": rms_pred / (rms_ref + 1e-12),
        "crest_pred": float(np.max(np.abs(pred)) / (rms_pred + 1e-12)),
        "envelope_rms_err_mean": float(np.mean(e_env)),
        "envelope_rms_err_p95": float(np.percentile(e_env, 95)),
    }


def make_plots(x, target, outputs, metas, rng):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(x)) / SR
    key_map = {
        "seed": "seed_s71_L10-ch32-g8-bigRF",
        "nam_lite": "nam_lite",
        "nam_full": "nam_full_standard",
    }
    colors = {"seed": "#d62728", "nam_lite": "#1f77b4", "nam_full": "#2ca02c"}
    labels = {
        "seed": "Seed s71 (median seed)",
        "nam_lite": "NAM Lite (6553)",
        "nam_full": "NAM Full/Standard (13801)",
    }

    # Full 5 s: dry waveform + RMS envelopes of target/models (legible at this scale)
    env_win, env_hop = int(0.005 * SR), int(0.0025 * SR)
    env_target = frame_rms(target, env_win, env_hop)
    env_t = (np.arange(len(env_target)) * env_hop + env_win / 2) / SR
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True, constrained_layout=True)
    axes[0].plot(t, x, color="#777777", lw=0.6)
    axes[0].set_ylabel("Dry guitar")
    axes[0].set_title("TSA15 test, dry input 8.0-13.0 s (same segment as P0a seg1)")
    for ax, key in zip(axes[1:], ["seed", "nam_lite", "nam_full"]):
        p = outputs[key_map[key]]
        env_p = frame_rms(p, env_win, env_hop)
        ax.plot(env_t, env_target, color="black", lw=1.1, label="Real amp target", alpha=0.9)
        ax.plot(env_t, env_p, color=colors[key], lw=1.1, label=labels[key], alpha=0.95)
        ax.fill_between(env_t, env_target, env_p, color=colors[key], alpha=0.16, linewidth=0)
        m = metas[key_map[key]]
        ax.set_ylabel("RMS envelope")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
        ax.set_title(
            f"{labels[key]}  NRMS={m['nrms']:.4f}  r={m['corr']:.4f}  "
            f"gain={m['gain_pred_over_target']:.3f}  max err@{m['max_err_time_s']:.2f}s"
        )
        ax.axvline(m["max_err_time_s"], color=colors[key], ls="--", lw=0.9, alpha=0.8)
    axes[-1].set_xlabel("Time in segment (s)")
    fig.savefig(OUT / "waveform_full_5s.png", dpi=150)
    plt.close(fig)

    # Zoom around the strongest real-amp transient: raw samples overlaid
    center = int(np.argmax(np.abs(target)))
    half = int(0.025 * SR)
    lo, hi = max(0, center - half), min(len(x), center + half)
    tz_rel = (np.arange(lo, hi) - lo) / SR
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
    axes[0].plot(tz_rel, target[lo:hi], color="black", lw=1.6, label="Real amp target", zorder=5)
    for key in ["seed", "nam_lite", "nam_full"]:
        axes[0].plot(tz_rel, outputs[key_map[key]][lo:hi], color=colors[key], lw=1.0,
                     alpha=0.85, label=labels[key])
    axes[0].set_title(f"Zoom: 50 ms around real-amp peak at t={center/SR:.3f}s (raw samples)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("Amplitude")
    for key in ["seed", "nam_lite", "nam_full"]:
        axes[1].plot(tz_rel, outputs[key_map[key]][lo:hi] - target[lo:hi],
                     color=colors[key], lw=1.0, label=labels[key])
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_ylabel("Error vs real amp")
    axes[1].set_xlabel("Time in zoom window (s)")
    axes[1].legend(loc="upper right", fontsize=9)
    fig.savefig(OUT / "waveform_zoom_transient.png", dpi=150)
    plt.close(fig)

    # Error traces and score bars
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)
    keys = ["seed", "nam_lite", "nam_full"]
    for key in keys:
        axes[0].plot(t, outputs[key_map[key]] - target, color=colors[key], lw=0.35, alpha=0.8, label=labels[key])
    axes[0].set_ylabel("Sample error")
    axes[0].set_title("Sample-wise error against the real amp (positive = model louder)")
    axes[0].legend(loc="upper right", fontsize=9)
    nrms_vals = [metas[key_map[k]]["nrms"] for k in keys]
    corr_vals = [metas[key_map[k]]["corr"] for k in keys]
    gain_vals = [metas[key_map[k]]["gain_pred_over_target"] for k in keys]
    xpos = np.arange(len(keys))
    b1 = axes[1].bar(xpos - 0.25, nrms_vals, width=0.25, color=[colors[k] for k in keys], alpha=0.9, label="NRMS (lower better)")
    axes[1].bar_label(b1, fmt="%.4f", fontsize=8)
    b2 = axes[1].bar(xpos, corr_vals, width=0.25, color="#17becf", alpha=0.75, label="Corr (higher better)")
    axes[1].bar_label(b2, fmt="%.4f", fontsize=8)
    b3 = axes[1].bar(xpos + 0.25, gain_vals, width=0.25, color="#ff7f0e", alpha=0.75, label="RMS gain (closer to 1 better)")
    axes[1].bar_label(b3, fmt="%.3f", fontsize=8)
    axes[1].set_xticks(xpos, [labels[k] for k in keys])
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].set_ylabel("Score")
    fig.savefig(OUT / "waveform_error_scores.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("No DCU/GPU detected; waveform_compare must run on a compute node.")

    x_all, sr_x = sf.read(DATA / "tsa15_test_input.wav", dtype="float32")
    y_all, sr_y = sf.read(DATA / "tsa15_test_target.wav", dtype="float32")
    assert sr_x == sr_y == SR
    assert len(x_all) == len(y_all)
    lo, hi = int(SEG_START_S * SR), int((SEG_START_S + SEG_LEN_S) * SR)
    x, target = x_all[lo:hi], y_all[lo:hi]

    sf.write(OUT / "dry_input_8s_13s.wav", x, SR, subtype="FLOAT")
    sf.write(OUT / "real_amp_target_8s_13s.wav", target, SR, subtype="FLOAT")

    report = {
        "segment": {"start_s": SEG_START_S, "len_s": SEG_LEN_S, "sr": SR, "samples": len(x)},
        "source": {
            "dry": str(DATA / "tsa15_test_input.wav"),
            "real_amp": str(DATA / "tsa15_test_target.wav"),
        },
        "models": {},
        "device": torch.cuda.get_device_name(0),
    }
    outputs = {}
    metas = {}
    for key, path, desc in MODEL_SPECS:
        print(f"[render] {key}: {path}", flush=True)
        model = load_nam(path)
        pred = render(model, x)
        outputs[key] = pred
        metas[key] = metrics(pred, target)
        cfg = json.load(open(path))
        metas[key].update(
            {
                "path": str(path),
                "description": desc,
                "params": len(cfg["weights"]) - 1,
                "receptive_field": int(getattr(model, "receptive_field", -1)),
            }
        )
        sf.write(OUT / f"{key}.wav", pred, SR, subtype="FLOAT")
        sf.write(OUT / f"diff_{key}.wav", pred - target, SR, subtype="FLOAT")
        print("[metric] %s %s" % (key, json.dumps(metas[key], ensure_ascii=False)), flush=True)
        del model

    make_plots(x, target, outputs, metas, None)
    report["models"] = metas
    with open(OUT / "metrics.json", "w") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)
    print("[done] " + str(OUT), flush=True)


if __name__ == "__main__":
    main()
