"""Task 1: is the Cellplus CTC-007/008 (.npy) mapping fixed, or does it carry an internal AGC?

Reads thermal_specs/{lab,cali_board}/thermal/*.npy (int16, 16383 - raw 14-bit count per
thermal_info.json) and, for the lab sequence, the 8-bit thermal_color/*.png written next to them.
Writes per-frame statistics to bench/camera_check/out/ and prints the numbers the conclusion rests on.

    /home/jumasaet/isaac_sim/python.sh bench/camera_check/agc_check.py
"""

from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(ROOT))
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)


def stamp(f: str) -> float:
    return float(os.path.basename(f).split("_")[1][:-4])


def index(f: str) -> int:
    return int(os.path.basename(f).split("_")[0])


def load_set(name: str):
    fs = sorted(glob.glob(os.path.join(REPO, "thermal_specs", name, "thermal", "*.npy")))
    stack = np.stack([np.load(f, allow_pickle=False) for f in fs])
    return fs, stack


def frame_stats(stack: np.ndarray) -> dict[str, np.ndarray]:
    flat = stack.reshape(len(stack), -1)
    p = np.percentile(flat, [0.5, 1, 50, 99, 99.5], axis=1)
    return {
        "min": flat.min(1), "p0.5": p[0], "p1": p[1], "p50": p[2],
        "p99": p[3], "p99.5": p[4], "max": flat.max(1), "mean": flat.mean(1),
    }


def write_csv(path, fs, cols: dict[str, np.ndarray]):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "index", "t_s", *cols])
        t0 = stamp(fs[0])
        for i, f in enumerate(fs):
            w.writerow([os.path.basename(f), index(f), f"{stamp(f) - t0:.3f}",
                        *[f"{v[i]:.3f}" for v in cols.values()]])


def main() -> None:
    report: dict = {}

    # ---------------------------------------------------------------- lab sequence
    fs, raw = load_set("lab")
    n, h, w = raw.shape
    t = np.array([stamp(f) for f in fs]); t -= t[0]
    idx = np.array([index(f) for f in fs])
    report["lab"] = {
        "n_frames": n, "dtype": str(raw.dtype), "shape_hw": [h, w],
        "duration_s": float(t[-1]), "median_dt_s": float(np.median(np.diff(t))),
        "index_gaps": int((np.diff(idx) > 1).sum()), "frames_missing_by_index": int(idx[-1] - idx[0] + 1 - n),
        "global_min": int(raw.min()), "global_max": int(raw.max()),
    }
    s = frame_stats(raw)
    span = s["max"] - s["min"]
    report["lab"]["per_frame_span_counts"] = {
        "min": float(span.min()), "median": float(np.median(span)), "max": float(span.max()),
        "fraction_of_14bit": float(np.median(span) / 16383),
    }
    report["lab"]["per_frame_p50_counts"] = {"min": float(s["p50"].min()), "max": float(s["p50"].max())}

    # The 8-bit PNG written by the capture tool: does it stretch every frame?
    png = np.stack([np.array(Image.open(f.replace("/thermal/", "/thermal_color/").replace(".npy", ".png")))
                    for f in fs]) if all(os.path.exists(f.replace("/thermal/", "/thermal_color/").replace(".npy", ".png")) for f in fs) else None
    pcols = {}
    if png is not None:
        ps = frame_stats(png)
        pcols = {f"png_{k}": v for k, v in ps.items()}
        # Per-frame linear fit png = a*raw + b. A fixed map has one (a, b) for the whole sequence.
        a = np.empty(n); b = np.empty(n); r = np.empty(n)
        for i in range(n):
            x = raw[i].ravel().astype(np.float64); y = png[i].ravel().astype(np.float64)
            A = np.vstack([x, np.ones_like(x)]).T
            (a[i], b[i]), *_ = np.linalg.lstsq(A, y, rcond=None)
            r[i] = np.sqrt(np.mean((A @ [a[i], b[i]] - y) ** 2))
        pcols.update({"png_gain_per_count": a, "png_offset": b, "png_fit_rms": r})
        # What raw count maps to png 0 and 255 in each frame (the window the AGC chose)
        lo = -b / a; hi = (255 - b) / a
        pcols.update({"png_window_lo_raw": lo, "png_window_hi_raw": hi})
        report["lab_png"] = {
            "dtype": str(png.dtype), "shape_hw": list(png.shape[1:]),
            "per_frame_min": [int(ps["min"].min()), int(ps["min"].max())],
            "per_frame_max": [int(ps["max"].min()), int(ps["max"].max())],
            "per_frame_p0.5_median": float(np.median(ps["p0.5"])),
            "per_frame_p99.5_median": float(np.median(ps["p99.5"])),
            "gain_codes_per_count": [float(a.min()), float(np.median(a)), float(a.max())],
            "window_lo_raw": [float(lo.min()), float(np.median(lo)), float(lo.max())],
            "window_hi_raw": [float(hi.min()), float(np.median(hi)), float(hi.max())],
            "fit_rms_codes": [float(r.min()), float(np.median(r)), float(r.max())],
            "window_vs_frame_p0.5_p99.5_corr": [float(np.corrcoef(lo, s["p0.5"])[0, 1]),
                                               float(np.corrcoef(hi, s["p99.5"])[0, 1])],
        }

    # Motion: median |frame-to-frame difference|, in counts. Static stretches have it at the noise floor.
    d = np.median(np.abs(np.diff(raw.astype(np.int32), axis=0)).reshape(n - 1, -1), axis=1)
    motion = np.concatenate([[np.nan], d])
    # Identical consecutive frames = frozen output (shutter/FFC freeze)
    frozen = np.concatenate([[False], (np.diff(raw.astype(np.int32), axis=0) == 0).reshape(n - 1, -1).all(1)])
    report["lab"]["frozen_repeats"] = int(frozen.sum())
    report["lab"]["motion_median_absdiff_counts"] = {
        "p5": float(np.nanpercentile(motion, 5)), "p50": float(np.nanpercentile(motion, 50)),
        "p95": float(np.nanpercentile(motion, 95)),
    }
    write_csv(os.path.join(OUT, "lab_per_frame.csv"), fs,
              {**{f"raw_{k}": v for k, v in s.items()}, "motion": motion, **pcols})

    # Longest static stretches: motion below 1.5x the sequence's 5th percentile for >= 10 frames
    thr = 1.5 * np.nanpercentile(motion, 5)
    quiet = np.nan_to_num(motion, nan=1e9) < thr
    runs = []; start = None
    for i, q in enumerate(np.append(quiet, False)):
        if q and start is None: start = i
        if not q and start is not None:
            if i - start >= 10: runs.append((start - 1, i - 1))  # include the frame before the first quiet diff
            start = None
    report["lab"]["static_threshold_counts"] = float(thr)
    report["lab"]["static_runs"] = [{"from": int(a_), "to": int(b_), "t0_s": float(t[a_]),
                                     "len": int(b_ - a_ + 1)} for a_, b_ in runs]
    json.dump(report, open(os.path.join(OUT, "report_partial.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))
    np.save(os.path.join(OUT, "lab_motion.npy"), motion)


if __name__ == "__main__":
    main()
