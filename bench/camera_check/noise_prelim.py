"""Preliminary noise numbers from the two static stretches of the lab sequence (agc_check.py).

Frame DIFFERENCES cancel the (static) scene, so temporal components come out without a uniform target:
  sigma_tvh  = 1.4826 * MAD(d) / sqrt(2)                   per-pixel temporal noise
  sigma_tv   = robust std over frames of the row means of d / sqrt(2), minus the tvh leak
  sigma_th   = same for column means
The spatial row/column pattern (v, h) is read from the temporal MEAN after removing the scene with a
15-px median filter along the other axis -- scene edges leak into it, so it is an upper bound.
These are AFTER the camera's own denoise (thermal_info.json: denoise 40) and in raw counts; the
counts-per-kelvin that would turn them into mK is not measured (no blackbody in the data).

    /home/jumasaet/isaac_sim/python.sh bench/camera_check/noise_prelim.py
"""
import glob, json, os
import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(ROOT))
fs = sorted(glob.glob(os.path.join(REPO, "thermal_specs/lab/thermal/*.npy")))
runs = json.load(open(os.path.join(ROOT, "out/report_partial.json")))["lab"]["static_runs"]
res = []
for run in runs:
    s = np.stack([np.load(f) for f in fs[run["from"]: run["to"] + 1]]).astype(np.float64)
    d = np.diff(s, axis=0)
    d -= np.median(d, axis=(1, 2), keepdims=True)          # drop the global drift step
    tvh = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
    H, W = s.shape[1:]
    rows = d.mean(2); cols = d.mean(1)                        # (frames, H), (frames, W)
    tv = np.sqrt(max(np.median(rows.std(1)) ** 2 - (tvh * np.sqrt(2)) ** 2 / W, 0)) / np.sqrt(2)
    th = np.sqrt(max(np.median(cols.std(1)) ** 2 - (tvh * np.sqrt(2)) ** 2 / H, 0)) / np.sqrt(2)
    m = s.mean(0)
    rowprof = m.mean(1); colprof = m.mean(0)
    v = np.std(rowprof - ndimage.median_filter(rowprof, 15))
    h = np.std(colprof - ndimage.median_filter(colprof, 15))
    res.append(dict(run=f"{run['from']}-{run['to']}", n_frames=len(s), sigma_tvh_counts=round(tvh, 2),
                    sigma_tv_rows_counts=round(tv, 2), sigma_th_cols_counts=round(th, 2),
                    ratio_tv=round(tv / tvh, 3), ratio_th=round(th / tvh, 3),
                    spatial_row_pattern_v_upper_counts=round(v, 2), spatial_col_pattern_h_upper_counts=round(h, 2)))
json.dump(res, open(os.path.join(ROOT, "out/noise_prelim.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
