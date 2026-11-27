"""Zoom a 0.5 s window of the compared waveforms and quantify the gaps.

Post-processing only: reads the WAVs already fetched from the server, no model
inference. Produces:
  - waveform_zoom_0p5s_attack.png   : 0.0-0.5 s (worst half-second), per-model overlays
  - waveform_zoom_0p5s_steady.png   : 3.5-4.0 s (steady state) control
  - waveform_zoom_0p5s_detail.png   : 25 ms raw-waveform close-ups inside each window
  - zoom_0p5s_metrics.json          : metrics for both windows + 125 ms sub-windows
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "reclaimed" / "waveform_cmp_20261127" / "out"
SR = 48000
MODELS = [
    ("seed_s71_L10-ch32-g8-bigRF", "seed_s71_L10-ch32-g8-bigRF.wav", "#d62728", "Seed s71 L10-ch32-g8-bigRF"),
    ("nam_lite", "nam_lite.wav", "#1f77b4", "NAM Lite"),
    ("nam_full_standard", "nam_full_standard.wav", "#2ca02c", "NAM Full (Standard)"),
]


def stats(pred, ref):
    err = pred - ref
    rms_ref = float(np.sqrt(np.mean(ref ** 2)))
    return {
        "nrms": float(np.sqrt(np.mean(err ** 2)) / (rms_ref + 1e-12)),
        "corr": float(np.corrcoef(pred, ref)[0, 1]),
        "gain": float(np.sqrt(np.mean(pred ** 2)) / (rms_ref + 1e-12)),
        "max_abs_err": float(np.max(np.abs(err))),
        "max_err_time_s": float(np.argmax(np.abs(err)) / SR),
        "err_over_10pct_frac": float(np.mean(np.abs(err) > 0.1 * rms_ref)),
    }


def sub_windows(pred, ref, lo, hi, msec=125):
    n = int(msec / 1000 * SR)
    return [stats(pred[a:a + n], ref[a:a + n]) for a in range(lo, hi, n)]


def overlay_figure(target, dry, preds, lo, hi, title, path):
    t = (np.arange(lo, hi) - lo) / SR
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 1.15])
    axes = [fig.add_subplot(gs[i]) for i in range(5)]
    axes[0].plot(t, dry[lo:hi], color="#888888", lw=0.55)
    axes[0].set_ylabel("Dry guitar")
    axes[0].set_title(title + "  |  dry input (TSA15 test)")
    for ax, (name, _, color, label) in zip(axes[1:4], MODELS):
        p = preds[name][lo:hi]
        r = target[lo:hi]
        ax.plot(t, r, color="black", lw=0.55, label="Real amp target", alpha=0.9)
        ax.plot(t, p, color=color, lw=0.55, label=label, alpha=0.95)
        ax.fill_between(t, r, p, color=color, alpha=0.16, linewidth=0)
        m = stats(p, r)
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9, ncol=2)
        ax.set_title(
            f"{label}: NRMS={m['nrms']:.4f}  corr={m['corr']:.4f}  "
            f"gain={m['gain']:.3f}  max|err|={m['max_abs_err']:.3f} @ {m['max_err_time_s']*1000:.1f} ms"
        )
    ax = axes[4]
    for name, _, color, label in MODELS:
        ax.plot(t, preds[name][lo:hi] - target[lo:hi], color=color, lw=0.6, label=label, alpha=0.9)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel("Error vs real amp")
    ax.set_xlabel("Time in window (s)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, ncol=3)
    ax.set_title("Sample-wise error (same amplitude scale as waveforms above)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def detail_figure(target, dry, preds, windows, path):
    """Two 25 ms raw-waveform close-ups, one per window, with error row."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), constrained_layout=True)
    for col, (lo, hi, wname) in enumerate(windows):
        t = (np.arange(lo, hi) - lo) / SR * 1000
        ax = axes[0, col]
        ax.plot(t, target[lo:hi], color="black", lw=1.6, label="Real amp target", zorder=5)
        for name, _, color, label in MODELS:
            ax.plot(t, preds[name][lo:hi], color=color, lw=1.1, label=label, alpha=0.95)
        ax.set_title(f"{wname}: raw waveform, {len(t)/SR*1000:.0f} ms")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.2)
        ax = axes[1, col]
        for name, _, color, label in MODELS:
            ax.plot(t, preds[name][lo:hi] - target[lo:hi], color=color, lw=1.0, label=label)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xlabel("Time in close-up (ms)")
        ax.set_ylabel("Error")
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    target, sr = sf.read(OUT / "real_amp_target_8s_13s.wav", dtype="float32")
    dry, sr2 = sf.read(OUT / "dry_input_8s_13s.wav", dtype="float32")
    assert sr == sr2 == SR and len(target) == len(dry) == 5 * SR
    preds = {}
    for name, fn, _, _ in MODELS:
        p, s = sf.read(OUT / fn, dtype="float32")
        assert s == SR and len(p) == len(target)
        preds[name] = p

    windows = [("attack 0.0-0.5 s", 0, int(0.5 * SR)), ("steady 3.5-4.0 s", int(3.5 * SR), int(4.0 * SR))]
    result = {"sr": SR, "source_segment_s": [8.0, 13.0], "windows": {}}
    for wname, lo, hi in windows:
        r = target[lo:hi]
        result["windows"][wname] = {
            "start_s_in_segment": lo / SR,
            "end_s_in_segment": hi / SR,
            "models": {
                name: {
                    **stats(preds[name][lo:hi], r),
                    "sub_125ms_nrms": [round(x["nrms"], 5) for x in sub_windows(preds[name], target, lo, hi)],
                }
                for name, _, _, _ in MODELS
            },
        }
        overlay_figure(
            target, dry, preds, lo, hi, wname,
            OUT / ("waveform_zoom_0p5s_attack.png" if lo == 0 else "waveform_zoom_0p5s_steady.png"),
        )

    # 25 ms close-up centred on the largest real-amp peak inside each half-second.
    closeups = []
    for wname, lo, hi in windows:
        peak = lo + int(np.argmax(np.abs(target[lo:hi])))
        half = int(0.0125 * SR)
        a, b = max(0, peak - half), min(len(target), peak + half)
        closeups.append((a, b, wname))
    detail_figure(target, dry, preds, closeups, OUT / "waveform_zoom_0p5s_detail.png")
    result["raw_waveform_closeups"] = [
        {"window": wname, "start_s": a / SR, "end_s": b / SR, "center_peak_s": (a + b) / 2 / SR}
        for a, b, wname in closeups
    ]
    with open(OUT / "zoom_0p5s_metrics.json", "w") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
    print("[done] " + str(OUT), flush=True)


if __name__ == "__main__":
    main()
