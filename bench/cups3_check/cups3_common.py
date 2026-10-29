"""Shared by the cups3_check scripts: the three-cup capture, its frame times, the cup regions, the probe
temperatures per frame, the board reference and the band radiance.

Data: thermal_specs/three_cups_experiments/sync_cups/ -- thermal (94 x int16 (480, 640) stored counts of the
Cellplus CTC-006-VGA, 14-bit raw, fixed mapping, higher = hotter, `denoise: 80`), RGB (93 PNG, 1280 x 960) and
LiDAR (94 Livox PCD), synchronised to the LiDAR by the capture node. The file names are
<index>_<unix time s>.<ns>; that time (the LiDAR stamp of the set) is the frame time used everywhere here. The
thermal frame's own stamp (sync_index.csv) is within 33 ms of it, which is nothing for a water temperature.

Band radiance: L_B(T) = integral of R(lambda) B(lambda, T) over the YAML's response (docs/physics-model.md §3.2 (b));
the bolometer output is linear in L_B, not in T (§9.3). Radiometry and the LiDAR/thermal geometry are imported from
bench/cups_check/cups_common.py (same rig, same calibration files), pointed at the CTC-006 YAML.
"""
from __future__ import annotations

import glob
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "out"
FIG = HERE / "figures"
DATA = REPO / "thermal_specs" / "three_cups_experiments" / "sync_cups"
# Since 2026-10-29: the denoise-80 noise (task 6) and the housing calibrated in place (task 7). Same band and
# spectral response as cellplus_ctc006_lwir_cups.yaml, so the band LUT and tasks 0-1 do not change.
SENSOR_YAML = REPO / "bench" / "camera_check" / "cellplus_ctc006_lwir_cups3.yaml"
# A variant run (task 7: another sensor YAML) renders into out/<CUPS3_SIM_DIR> and task2_compare writes
# task2_comparison_<CUPS3_SIM_DIR>.{json,png}; unset, everything is as before (out/sim, task2_comparison.*).
SENSOR_YAML = pathlib.Path(os.environ.get("CUPS3_SENSOR_YAML", SENSOR_YAML))
SIM_DIR = os.environ.get("CUPS3_SIM_DIR", "sim")
COMPARISON = "task2_comparison" if SIM_DIR == "sim" else f"task2_comparison_{SIM_DIR}"
sys.path.insert(0, str(REPO / "bench" / "cups_check"))
import cups_common as cc  # noqa: E402

cc.SENSOR_YAML = SENSOR_YAML  # same response CSV as the CTC-007 draft; set before the first band_l call
band_l, band_l_inv, c2k = cc.band_l, cc.band_l_inv, cc.c2k
read_pcd, lidar_to_thermal, project_thermal, undistort_points = (cc.read_pcd, cc.lidar_to_thermal,
                                                                  cc.project_thermal, cc.undistort_points)
K_TH, DIST_TH = cc.K_TH, cc.DIST_TH

# The cup as measured by the user (180 ml paper cup), the same model as bench/cups_check.
CUP_TOP_DIAMETER_M = 0.070
CUP_BOTTOM_DIAMETER_M = 0.050
CUP_HEIGHT_M = 0.060

CUPS = ("cold", "warm", "hot")  # left to right in the image
CALIBRATION_CUPS = ("cold", "hot")  # task 1: the warm cup is the independent test
T_AIR_C = 23.9
# Probe readings, C, before -> after the recording. The reading times were NOT logged, and the data show the
# cups moved far less during the 29 s than the readings differ (hot: -1.15 K, not -7 K), so the readings cannot
# be placed at the first and last frame. Each cup is held at the mean of its two readings (bench/GUIA.md, 5).
T_PROBE_C = {"hot": (69.3, 62.3), "warm": (36.1, 34.8), "cold": (6.8, 7.9)}
T_MEAN_C = {cup: (a + b) / 2 for cup, (a, b) in T_PROBE_C.items()}  # hot 65.8, warm 35.45, cold 7.35
# Column window of each cup in the thermal frame (a search window, not a measurement).
CUP_WINDOW_COLS = {"cold": (40, 200), "warm": (210, 370), "hot": (390, 560)}


def r_at(z: float | np.ndarray) -> float | np.ndarray:
    """Cup outer radius at height z above its base."""
    return CUP_BOTTOM_DIAMETER_M / 2 + (CUP_TOP_DIAMETER_M - CUP_BOTTOM_DIAMETER_M) / 2 * np.asarray(z) / CUP_HEIGHT_M


# -- frames ---------------------------------------------------------------------------------------------


def frame_files(kind: str = "thermal") -> list[str]:
    ext = {"thermal": "npy", "lidar": "pcd", "rgb": "png", "thermal_color": "png"}[kind]
    return sorted(glob.glob(str(DATA / kind / f"*.{ext}")))


def frame_time_s(path: str) -> float:
    return float(os.path.basename(path).split("_", 1)[1].rsplit(".", 1)[0])


def frame_index(path: str) -> int:
    return int(os.path.basename(path).split("_", 1)[0])


def load_thermal() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(counts (N, 480, 640) float64, unix times (N,), file names)."""
    fs = frame_files("thermal")
    a = np.stack([np.load(f) for f in fs])
    assert a.dtype == np.int16 and a.shape[1:] == (480, 640), (a.dtype, a.shape)
    return a.astype(np.float64), np.array([frame_time_s(f) for f in fs]), [os.path.basename(f) for f in fs]


def probe_t_c(cup: str, t: np.ndarray) -> np.ndarray:
    """Water temperature per frame: constant, the mean of the 'before' and 'after' readings (the times of the
    readings are unknown, see T_PROBE_C)."""
    return np.full(np.shape(t), T_MEAN_C[cup], np.float64)


# -- regions --------------------------------------------------------------------------------------------


def _run_containing(idx: np.ndarray, centre: int) -> tuple[int, int]:
    return cc._run_containing(idx, centre)


def find_cups(mean: np.ndarray) -> dict:
    """Locate the three cups in the mean frame. Regions are (y0, y1, x0, x1), [lo, hi).

    For each cup: s = +-(mean - board level) (sign: colder or hotter than the board); the peak is the extreme
    pixel inside its column window, rows 150-330. The WATER plateau is where s >= 90 % of the peak along the
    centre column (the wetted wall: not the dry band under the rim, not the table reflection below) and the
    central half of the 50 % width at the plateau's middle row; shrunk by 2 rows at the top and 1 at the bottom.
    `mask` is the cup and its reflection, dilated, for the board fit.
    """
    board0 = float(np.median(mean[100:180, 150:420]))
    out = {}
    for cup in CUPS:
        x0w, x1w = CUP_WINDOW_COLS[cup]
        sub = mean[150:330, x0w:x1w] - board0
        sign = -1.0 if sub.min() < -abs(sub.max()) else 1.0
        s = (mean - board0) * sign
        sm = np.full_like(s, -np.inf)
        sm[150:330, x0w:x1w] = s[150:330, x0w:x1w]
        yy, xx = np.unravel_index(np.argmax(sm), s.shape)
        peak = float(np.median(s[yy - 2:yy + 3, xx - 2:xx + 3]))
        col = s[:, xx - 2:xx + 3].mean(1)
        rows = np.nonzero(col >= 0.9 * peak)[0]
        rows = rows[(rows > yy - 60) & (rows < yy + 60)]
        r0, r1 = _run_containing(rows, yy)
        row = s[(r0 + r1) // 2, :]
        cols = np.nonzero(row >= 0.5 * peak)[0]
        c0, c1 = _run_containing(cols, xx)
        w = c1 - c0 + 1
        water = (r0 + 2, r1, c0 + w // 4, c1 - w // 4 + 1)
        out[cup] = {"sign": int(sign), "peak_rc": (int(yy), int(xx)), "peak_contrast_counts": round(peak, 1),
                    "plateau_rows": (int(r0), int(r1)), "half_width_cols": (int(c0), int(c1)),
                    "water": tuple(int(v) for v in water)}
    return out


def board_mask(regions: dict, shape=(480, 640)) -> np.ndarray:
    """Board pixels for the reference plane: rows 90-250 (above the table line at ~262), cols 100-560 (inside
    the board's edges at ~85 and ~575), minus each cup and its halo (50 px around its 50 % outline, which keeps
    the hot cup's halo under ~10 counts) and minus the column above the hot cup (its warm plume)."""
    m = np.zeros(shape, bool)
    m[90:250, 100:560] = True
    for cup, r in regions.items():
        c0, c1 = r["half_width_cols"]
        top = r["plateau_rows"][0] - 30
        m[max(top - 50, 0):, max(c0 - 50, 0):c1 + 51] = False
        if cup == "hot":
            m[:, max(c0 - 30, 0):c1 + 31] = False
    return m


def roi(a: np.ndarray, r: tuple) -> np.ndarray:
    y0, y1, x0, x1 = r
    return a[..., y0:y1, x0:x1]


def _basis(y, x) -> list:
    x = (np.asarray(x, float) - 320.0) / 320.0
    y = (np.asarray(y, float) - 240.0) / 240.0
    return [np.ones_like(x), x, y, x * x, x * y, y * y]


def board_plane(frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per frame, least-squares quadratic surface c = sum_k a_k x^i y^j (i + j <= 2) through the board pixels
    -> coefficients (N, 6).

    The real board's level varies ~20 counts (~0.4 K) left to right, and the simulated camera adds a bowl
    (vignetting x housing emission: +0.8 W m^-2 sr^-1 at the left edge, +0.1 at the centre); the surface carries
    either under each cup instead of the nearest board patch, which sits in a neighbour's halo between cups.
    (A plane, tried first, left the bowl's curvature in the simulated cups: -0.16 K on the centred warm cup.)"""
    yy, xx = np.nonzero(mask)
    A = np.stack(_basis(yy, xx), 1)
    f = frames if frames.ndim == 3 else frames[None]
    vals = f[:, yy, xx]
    coef, *_ = np.linalg.lstsq(A, vals.T, rcond=None)
    return coef.T


def plane_at(coef: np.ndarray, y: np.ndarray | float, x: np.ndarray | float) -> np.ndarray:
    """Evaluate board surfaces (N, 6) at pixel (y, x) -> (N,) or (N, ...)."""
    b = _basis(y, x)
    c = coef.reshape(coef.shape + (1,) * np.ndim(b[0]))
    return sum(c[:, k] * b[k] for k in range(len(b)))


def board_field(coef: np.ndarray, shape=(480, 640)) -> np.ndarray:
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return plane_at(coef, yy, xx)


def per_frame_levels(frames: np.ndarray, regions: dict, coef: np.ndarray) -> dict:
    """Per cup: median counts of its water ROI, the board surface at the ROI's centre, and their difference."""
    out = {}
    n = len(frames)
    for cup, r in regions.items():
        y0, y1, x0, x1 = r["water"]
        c = np.median(roi(frames, r["water"]).reshape(n, -1), axis=1)
        bg = plane_at(coef, 0.5 * (y0 + y1 - 1), 0.5 * (x0 + x1 - 1))
        out[cup] = {"cup": c, "bg": bg, "delta": c - bg}
    return out
