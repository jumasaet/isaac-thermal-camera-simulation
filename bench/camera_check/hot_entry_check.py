"""Task 1, second half: does the raw .npy level move when something hot enters or leaves the view?

Two tests on the lab sequence (handheld walk under ceiling fluorescent tubes) and one on cali_board.

A. Consecutive pairs (0.1 s apart). For each pair, the "background" is every pixel that is below the
   99th percentile in BOTH frames and is not within 15 px of a hot pixel in either. Its median
   difference (frame k+1 minus k) is what an AGC would shift when a tube enters; scene motion makes it
   noisy but uncorrelated with the tube. Regressed against the change in hot-pixel count.
B. The two static stretches found by agc_check.py: ROI drift and per-pixel temporal sigma.
C. cali_board: where the cold board covers more or less of the frame, the warm-wall level (mode of the
   upper half of the histogram) should not follow the board's coverage under a fixed map.

    /home/jumasaet/isaac_sim/python.sh bench/camera_check/hot_entry_check.py
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(ROOT))
OUT = os.path.join(ROOT, "out")

HOT = 15700  # stored counts; tubes reach 16000-16960, the room sits at 15000-15500


def load(name):
    fs = sorted(glob.glob(os.path.join(REPO, "thermal_specs", name, "thermal", "*.npy")))
    return fs, np.stack([np.load(f, allow_pickle=False) for f in fs]).astype(np.int32)


def main() -> None:
    rep: dict = {}
    fs, raw = load("lab")
    n = len(raw)
    idx = np.array([int(os.path.basename(f).split("_")[0]) for f in fs])

    # ---- A
    hot = raw > HOT
    near = np.stack([ndimage.binary_dilation(h, iterations=15) for h in hot])
    rows = []
    for k in range(n - 1):
        if idx[k + 1] - idx[k] != 1:
            continue
        a, b = raw[k], raw[k + 1]
        bg = ~(near[k] | near[k + 1])
        if bg.mean() < 0.3:
            continue
        dbg = float(np.median(b[bg] - a[bg]))
        rows.append((k, int(hot[k + 1].sum() - hot[k].sum()), int(hot[k].sum()), int(hot[k + 1].sum()),
                     dbg, float(np.median(np.abs(b[bg] - a[bg])))))
    r = np.array(rows, dtype=float)
    dh, dbg, motion = r[:, 1], r[:, 4], r[:, 5]
    big = np.abs(dh) > 2000  # a tube (or most of one) entering / leaving between two frames
    rep["A_pairs"] = {
        "n_pairs": len(r), "n_tube_events_|dhot|>2000px": int(big.sum()),
        "corr_dbg_vs_dhot_all": float(np.corrcoef(dh, dbg)[0, 1]),
        "slope_counts_per_1000_hot_px": float(np.polyfit(dh, dbg, 1)[0] * 1000),
        "events_median_dbg": float(np.median(dbg[big])), "events_mean_abs_dbg": float(np.mean(np.abs(dbg[big]))),
        "quiet_pairs_mean_abs_dbg": float(np.mean(np.abs(dbg[~big]))),
        "events_enter_mean_dbg": float(np.mean(dbg[big & (dh > 0)])),
        "events_leave_mean_dbg": float(np.mean(dbg[big & (dh < 0)])),
        "events": [{"k": int(x[0]), "dhot_px": int(x[1]), "hot_before": int(x[2]), "hot_after": int(x[3]),
                    "dbg_counts": x[4], "bg_motion_counts": x[5]} for x in r[big]],
    }
    # Same test restricted to low-motion pairs, where the scene itself barely moved
    calm = motion <= 4
    if (big & calm).sum() >= 3:
        rep["A_pairs"]["calm_events"] = {"n": int((big & calm).sum()), "mean_dbg": float(dbg[big & calm].mean()),
                                         "max_abs_dbg": float(np.abs(dbg[big & calm]).max())}

    # ---- B
    part = json.load(open(os.path.join(OUT, "report_partial.json")))
    rep["B_static"] = []
    for run in part["lab"]["static_runs"]:
        s = raw[run["from"]: run["to"] + 1].astype(np.float64)
        h, w = s.shape[1:]
        roi = s[:, h // 2 - 40: h // 2 + 40, w // 2 - 40: w // 2 + 40]
        roi_mean = roi.reshape(len(s), -1).mean(1)
        # temporal sigma per pixel after removing a per-frame global offset (tiny motion / drift)
        resid = s - s.reshape(len(s), -1).mean(1)[:, None, None]
        sig = resid.std(0, ddof=1)
        rep["B_static"].append({
            **run, "roi80_mean_range_counts": float(roi_mean.max() - roi_mean.min()),
            "roi80_frame_to_frame_abs_counts": float(np.mean(np.abs(np.diff(roi_mean)))),
            "frame_mean_range_counts": float(np.ptp(s.mean((1, 2)))),
            "pixel_temporal_sigma_counts_median": float(np.median(sig)),
            "max_count_range": float(np.ptp(s.max((1, 2)))),
        })

    # ---- C
    cfs, cal = load("cali_board")
    out = []
    for f, a in zip(cfs, cal):
        v = a.ravel()
        cold = float((v < np.percentile(v, 50) - 150).mean())  # board circles sit ~300+ counts below the wall
        hist = np.bincount(v - v.min())
        wall = int(np.argmax(ndimage.uniform_filter1d(hist.astype(float), 9)) + v.min())
        out.append((os.path.basename(f), float(np.stack([v]).mean()), cold, wall, float(np.median(v))))
    walls = np.array([o[3] for o in out]); colds = np.array([o[2] for o in out])
    sess = np.array([o[0].split("_")[1].startswith("17925810") or o[0].split("_")[1].startswith("17925811")
                     for o in out])
    rep["C_cali_board"] = {
        "frames": [{"file": o[0], "cold_fraction": round(o[2], 4), "wall_mode": o[3], "median": o[4]} for o in out],
        "per_session": {},
    }
    for name, m in (("session_A_t=...7734-7853", ~sess), ("session_B_t=...1050-1161", sess)):
        rep["C_cali_board"]["per_session"][name] = {
            "n": int(m.sum()), "wall_mode_range": [int(walls[m].min()), int(walls[m].max())],
            "cold_fraction_range": [float(colds[m].min()), float(colds[m].max())],
            "corr_wall_vs_cold_fraction": float(np.corrcoef(walls[m], colds[m])[0, 1]),
        }
    json.dump(rep, open(os.path.join(OUT, "hot_entry_report.json"), "w"), indent=1)
    show = dict(rep); show["A_pairs"] = {k: v for k, v in rep["A_pairs"].items() if k != "events"}
    show["C_cali_board"] = rep["C_cali_board"]["per_session"]
    print(json.dumps(show, indent=1))
    print("tube events (k, dhot, dbg, motion):")
    for e in rep["A_pairs"]["events"]:
        print(f"  {e['k']:5d} {e['dhot_px']:7d} {e['dbg_counts']:6.1f} {e['bg_motion_counts']:5.1f}")


if __name__ == "__main__":
    main()
