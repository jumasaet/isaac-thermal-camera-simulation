"""Shared by the cups_check scripts: the four capture sets, their frames, the cup ROIs and the band radiance.

Data: thermal_specs/cali_board_new/{hot,cold}_water_{50,95}/thermal/*.npy -- int16 (480, 640), stored
counts of the Cellplus CTC-007 (14-bit raw, fixed mapping, higher = hotter; bench/camera_check/RESULTADOS.md).
File names are <index>_<unix time s>.npy; the time is the capture clock (UTC epoch, so 07:18 JST on 2026-10-24
for the first hot frame).

Band radiance: L_B(T) = integral of R(lambda) B(lambda, T) over the camera's response (docs/physics-model.md
§3.2 (b)), with the draft YAML's R(lambda) (8-14 um, ESTIMATED shape). The bolometer output is linear in
L_B, not in T (§9.3), which is why the calibration below is fitted in radiance.
"""
from __future__ import annotations

import glob
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
DATA = REPO / "thermal_specs" / "cali_board_new"
SENSOR_YAML = REPO / "bench" / "camera_check" / "cellplus_ctc007_lwir.yaml"
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

CASES = ("hot_water_50", "hot_water_95", "cold_water_50", "cold_water_95")
NOMINAL_DISTANCE_M = {"hot_water_50": 0.50, "hot_water_95": 0.95, "cold_water_50": 0.50, "cold_water_95": 0.95}

# The cup as measured by the user (180 ml paper cup).
CUP_TOP_DIAMETER_M = 0.070
CUP_BOTTOM_DIAMETER_M = 0.050
CUP_HEIGHT_M = 0.060

T_AIR_C = 22.0
T_COLD_C = 11.0
T_HOT_RANGE_C = (58.0, 66.0)

# Thermal camera calibration (thermal_specs/cali_board_new/RGBT_calibration_result_matrix.txt), 640 x 480.
K_TH = np.array([[626.94618350, 0.0, 332.60285388], [0.0, 620.62145336, 222.85948387], [0.0, 0.0, 1.0]])
DIST_TH = np.array([-0.2738138890, -0.5834928420, 0.0088117352, 0.0012314182, 0.7123230428])  # k1 k2 p1 p2 k3
# P_thermal = T_TH_RGB @ P_rgb ; P_rgb = T_RGB_LIDAR @ P_lidar (RGB-LiDAR_calibration_result_matrix.txt)
T_TH_RGB = np.array([[0.9984634221, 0.0067074101, -0.0550073212, -0.0438037458],
                     [-0.0071402231, 0.9999450503, -0.0076755247, -0.0053343485],
                     [0.0549528157, 0.0080564952, 0.9984564492, -0.0508196610], [0, 0, 0, 1.0]])
T_RGB_LIDAR = np.array([[-0.041393, -0.998775, -0.027098, 0.038807], [0.367053, 0.010024, -0.930146, -0.075211],
                        [0.929278, -0.048448, 0.366189, 0.010891], [0, 0, 0, 1.0]])


# -- frames ---------------------------------------------------------------------------------------------


def frame_files(case: str, kind: str = "thermal") -> list[str]:
    ext = {"thermal": "npy", "lidar": "pcd", "rgb": "png", "thermal_gray": "png"}[kind]
    return sorted(glob.glob(str(DATA / case / kind / f"*.{ext}")))


def frame_time_s(path: str) -> float:
    return float(os.path.basename(path).split("_", 1)[1].rsplit(".", 1)[0])


def load_case(case: str) -> tuple[np.ndarray, np.ndarray]:
    """(frames (N, 480, 640) float64 counts, unix times (N,))."""
    fs = frame_files(case)
    a = np.stack([np.load(f) for f in fs])
    assert a.dtype == np.int16 and a.shape[1:] == (480, 640), (a.dtype, a.shape)
    return a.astype(np.float64), np.array([frame_time_s(f) for f in fs])


# -- radiometry -----------------------------------------------------------------------------------------

_GRID_K = np.arange(230.0, 380.0, 0.01)
_GRID_L: np.ndarray | None = None


def _response():
    from irsim.config.loader import load_sensor_config
    from irsim.radiometry.lut_files import load_band_response_for_config

    return load_band_response_for_config(load_sensor_config(SENSOR_YAML), REPO / "data")


def band_l(t_k):
    """L_B(T) in W m^-2 sr^-1 over the draft YAML's 8-14 um response. docs/physics-model.md §3.2 (b)."""
    global _GRID_L
    if _GRID_L is None:
        from irsim.radiometry.band_integration import band_radiance

        _GRID_L = band_radiance(_response(), _GRID_K)
    return np.interp(np.asarray(t_k, np.float64), _GRID_K, _GRID_L)


def band_l_inv(l):
    """L_B^-1: the blackbody (apparent) temperature in K of a band radiance. docs/physics-model.md §3.3."""
    band_l(300.0)
    return np.interp(np.asarray(l, np.float64), _GRID_L, _GRID_K)


def c2k(t_c):
    return np.asarray(t_c, np.float64) + 273.15


# -- cup and background regions -------------------------------------------------------------------------


def find_cup(mean: np.ndarray, hot: bool) -> dict:
    """Locate the cup in a set's mean frame and return its regions (row/col slices as [lo, hi)).

    contrast s = +-(mean - board level); the cup is the strongest blob in rows 150-330. The WATER plateau is
    where s >= 90 % of the cup's peak along the centre column and the centre row -- the wetted wall, not
    the dry band above the water line nor the table reflection below the cup. The board reference is two
    rectangles left and right of the cup, 60-120 px clear of its edges (the halo of the hot cup reaches
    ~1 % of its contrast at 30 px), above the table line.
    """
    board0 = np.median(mean[100:180, 300:500])
    s = (mean - board0) * (1 if hot else -1)
    win = np.zeros_like(s, bool)
    win[150:330, 100:540] = True
    sm = np.where(win, s, -np.inf)
    # peak = robust: the 99.9th percentile pixel inside the window
    yy, xx = np.unravel_index(np.argmax(sm), s.shape)
    peak = float(np.median(s[yy - 2:yy + 3, xx - 2:xx + 3]))
    # Plateau along the centre column and the centre row through the peak.
    col = s[:, xx]
    rows = np.nonzero(col >= 0.9 * peak)[0]
    rows = rows[(rows > yy - 60) & (rows < yy + 60)]
    r0, r1 = _run_containing(rows, yy)
    row = s[(r0 + r1) // 2, :]
    cols = np.nonzero(row >= 0.5 * peak)[0]
    c0, c1 = _run_containing(cols, xx)
    # 50 % width at the plateau's centre row; water ROI = central half of the width, plateau rows shrunk by 2.
    w = c1 - c0 + 1
    water = (r0 + 2, r1 - 1, c0 + w // 4, c1 - w // 4 + 1)
    gap0, gap1 = 60, 120
    top = max(r0 - 40, 120)
    bgrows = (top, r1 - 8)
    bg_left = (bgrows[0], bgrows[1], c0 - gap1, c0 - gap0)
    bg_right = (bgrows[0], bgrows[1], c1 + gap0, c1 + gap1)
    return {"peak_rc": (int(yy), int(xx)), "peak_contrast": peak, "plateau_rows": (int(r0), int(r1)),
            "half_width_cols": (int(c0), int(c1)), "water": tuple(int(v) for v in water),
            "bg_left": tuple(int(v) for v in bg_left), "bg_right": tuple(int(v) for v in bg_right)}


def _run_containing(idx: np.ndarray, centre: int) -> tuple[int, int]:
    """[first, last] of the consecutive run of indices that contains centre."""
    idx = np.sort(idx)
    if centre not in idx:
        centre = int(idx[np.argmin(np.abs(idx - centre))])
    k = int(np.nonzero(idx == centre)[0][0])
    lo = k
    while lo > 0 and idx[lo - 1] == idx[lo] - 1:
        lo -= 1
    hi = k
    while hi < len(idx) - 1 and idx[hi + 1] == idx[hi] + 1:
        hi += 1
    return int(idx[lo]), int(idx[hi])


def roi(a: np.ndarray, r: tuple) -> np.ndarray:
    """Pixels of a (y0, y1, x0, x1) region; a may be (H, W) or (N, H, W)."""
    y0, y1, x0, x1 = r
    return a[..., y0:y1, x0:x1]


def per_frame_levels(frames: np.ndarray, regions: dict) -> dict:
    """Median counts of the water ROI and of the board reference, per frame."""
    cup = np.median(roi(frames, regions["water"]).reshape(len(frames), -1), axis=1)
    bl = roi(frames, regions["bg_left"]).reshape(len(frames), -1)
    br = roi(frames, regions["bg_right"]).reshape(len(frames), -1)
    bg = np.median(np.concatenate([bl, br], axis=1), axis=1)
    return {"cup": cup, "bg": bg, "delta": cup - bg}


# -- LiDAR ----------------------------------------------------------------------------------------------

_PCD_DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("i", "<f4"), ("tag", "u1"), ("line", "u1"),
                    ("ts", "<f8")])


def read_pcd(path: str) -> np.ndarray:
    """Livox binary PCD (x y z intensity tag line timestamp) -> (N, 3) float64 metres, LiDAR frame."""
    b = pathlib.Path(path).read_bytes()
    head = b.index(b"DATA binary\n") + len(b"DATA binary\n")
    assert b"FIELDS x y z intensity tag line timestamp" in b[:head]
    p = np.frombuffer(b[head:], dtype=_PCD_DT)
    xyz = np.stack([p["x"], p["y"], p["z"]], 1).astype(np.float64)
    return xyz[np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 0)]


def lidar_to_thermal(xyz: np.ndarray) -> np.ndarray:
    h = np.c_[xyz, np.ones(len(xyz))]
    return (T_TH_RGB @ T_RGB_LIDAR @ h.T).T[:, :3]


def project_thermal(p_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV pinhole + Brown-Conrady (k1 k2 p1 p2 k3) projection with the thermal calibration -> (u, v) px."""
    x, y = p_cam[:, 0] / p_cam[:, 2], p_cam[:, 1] / p_cam[:, 2]
    k1, k2, p1, p2, k3 = DIST_TH
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return K_TH[0, 0] * xd + K_TH[0, 2], K_TH[1, 1] * yd + K_TH[1, 2]


def undistort_points(u: np.ndarray, v: np.ndarray, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Pixel -> normalised ideal (x, y) = (X/Z, Y/Z), inverting project_thermal by fixed-point iteration."""
    xd = (np.asarray(u, float) - K_TH[0, 2]) / K_TH[0, 0]
    yd = (np.asarray(v, float) - K_TH[1, 2]) / K_TH[1, 1]
    k1, k2, p1, p2, k3 = DIST_TH
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x, y = (xd - dx) / rad, (yd - dy) / rad
    return x, y
