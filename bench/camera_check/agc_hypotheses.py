"""Task 1, the decisive comparison: measured background shift at each tube entry/exit versus what two
concrete AGC laws would have produced on the same frame pair.

  hold-mean AGC  : offset moves so the frame mean is constant  -> bg shifts by -(delta of frame mean)
  min-max AGC    : gain rescales the frame span to a fixed range -> bg spread scales by span_k/span_k+1

Pairs are consecutive frames (0.1 s), with >= 1000 px crossing the hot threshold between them.

    /home/jumasaet/isaac_sim/python.sh bench/camera_check/agc_hypotheses.py
"""
import glob, json, os
import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(ROOT))
fs = sorted(glob.glob(os.path.join(REPO, "thermal_specs/lab/thermal/*.npy")))
idx = np.array([int(os.path.basename(f).split("_")[0]) for f in fs])
HOT = 15700
rows = []
prev = None
for k, f in enumerate(fs):
    a = np.load(f).astype(np.int32)
    if prev is not None and idx[k] - idx[k - 1] == 1:
        p = prev
        hp, ha = p > HOT, a > HOT
        dh = int(ha.sum() - hp.sum())
        if abs(dh) >= 1000:
            bg = ~(ndimage.binary_dilation(hp | ha, iterations=15))
            d = a[bg] - p[bg]
            meas = float(np.median(d))
            # hold-mean: the hot change moves the frame mean; an AGC holding the mean shifts bg by -that
            pred_mean = -float(a.mean() - p.mean() - np.median(d))
            # min-max: background spread (p5-p95) ratio vs span ratio
            sp_p, sp_a = np.ptp(p), np.ptp(a)
            w_p = np.subtract(*np.percentile(p[bg], [95, 5])); w_a = np.subtract(*np.percentile(a[bg], [95, 5]))
            rows.append(dict(k=k - 1, dhot_px=dh, bg_shift_meas=meas, bg_shift_pred_holdmean=round(pred_mean, 2),
                             bg_motion=float(np.median(np.abs(d))), span_ratio=round(sp_p / sp_a, 3),
                             bg_spread_ratio=round(w_a / w_p, 3) if w_p else None))
    prev = a
r = rows
m = np.array([x["bg_shift_meas"] for x in r]); pm = np.array([x["bg_shift_pred_holdmean"] for x in r])
sr = np.array([x["span_ratio"] for x in r]); br = np.array([x["bg_spread_ratio"] for x in r])
summ = {
    "n_events": len(r),
    "mean_abs_measured_bg_shift_counts": float(np.mean(np.abs(m))),
    "mean_abs_predicted_holdmean_shift_counts": float(np.mean(np.abs(pm))),
    "corr_measured_vs_holdmean_prediction": float(np.corrcoef(m, pm)[0, 1]),
    "span_ratio_range": [float(sr.min()), float(sr.max())],
    "bg_spread_ratio_range": [float(br.min()), float(br.max())],
    "corr_log_spread_ratio_vs_log_span_ratio": float(np.corrcoef(np.log(sr), np.log(br))[0, 1]),
    "events_span_changes_over_25pct": int((np.abs(np.log(sr)) > np.log(1.25)).sum()),
    "those_events_bg_spread_ratio": [float(x) for x in br[np.abs(np.log(sr)) > np.log(1.25)]],
    "those_events_span_ratio": [float(x) for x in sr[np.abs(np.log(sr)) > np.log(1.25)]],
}
json.dump({"summary": summ, "events": r}, open(os.path.join(ROOT, "out/agc_hypotheses.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
