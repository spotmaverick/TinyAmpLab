"""Locate where each model's waveform diverges from the real amp.

Reads the WAVs produced by waveform_compare.py on the server and writes
post_analysis.json. No model inference here; run on the server per project rule.
"""
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(os.environ.get("WAVECMP_ROOT", "/public/home/spotmaverick/nam_proj"))
OUT = ROOT / "waveform_cmp" / "out"
SR = 48000
MODELS = [
    ("seed_s71_L10-ch32-g8-bigRF", "seed_s71_L10-ch32-g8-bigRF.wav"),
    ("nam_lite", "nam_lite.wav"),
    ("nam_full_standard", "nam_full_standard.wav"),
]


def stats(pred, ref):
    err = pred - ref
    rms_ref = float(np.sqrt(np.mean(ref**2)))
    return {
        "nrms": float(np.sqrt(np.mean(err**2)) / (rms_ref + 1e-12)),
        "gain": float(np.sqrt(np.mean(pred**2)) / (rms_ref + 1e-12)),
        "max_abs_err": float(np.max(np.abs(err))),
    }


def main():
    target, sr = sf.read(OUT / "real_amp_target_8s_13s.wav", dtype="float32")
    assert sr == SR
    result = {"sr": SR, "duration_s": len(target) / SR, "models": {}}
    for name, fn in MODELS:
        pred, sr_p = sf.read(OUT / fn, dtype="float32")
        assert sr_p == SR and len(pred) == len(target)
        per_sec = []
        for s in range(5):
            lo, hi = s * SR, (s + 1) * SR
            per_sec.append({"second": s, **stats(pred[lo:hi], target[lo:hi])})
        win, hop = int(0.25 * SR), int(0.125 * SR)
        windows = []
        for lo in range(0, len(target) - win + 1, hop):
            st = stats(pred[lo : lo + win], target[lo : lo + win])
            windows.append({"start_s": round(lo / SR, 3), "end_s": round((lo + win) / SR, 3),
                            "nrms": st["nrms"], "gain": st["gain"]})
        worst = sorted(windows, key=lambda w: w["nrms"], reverse=True)[:3]
        result["models"][name] = {
            "per_second": per_sec,
            "first_0_5s": stats(pred[: int(0.5 * SR)], target[: int(0.5 * SR)]),
            "late_3_5_to_5s": stats(pred[int(3.5 * SR) :], target[int(3.5 * SR) :]),
            "worst_250ms_windows": worst,
        }
    with open(OUT / "post_analysis.json", "w") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
    print("[done] " + str(OUT / "post_analysis.json"), flush=True)


if __name__ == "__main__":
    main()
