#!/usr/bin/env python
"""Weekly-presentation figures for the cups experiment (host only; run after render_rgb_world.py).

    /home/jumasaet/isaac_sim/python.sh bench/cups_check/presentation/make_presentation.py

1. RGB vs thermal, real vs simulated (four panels per case). The simulated RGB view was rendered as a pinhole
   with the calibrated fx on a wide canvas (render_rgb_world.py); here every pixel of the real 1280 x 960 image is
   undistorted with the calibrated Brown-Conrady model (thermal_specs/cali_board_new/
   RGBT_calibration_result_matrix.txt) and looked up in that canvas, so real and simulated RGB share the real
   lens's pixels. The thermal pair is the one the comparison sheets use: the real reference frame calibrated to
   apparent temperature (task 1), and the CTC-006 render (out/sim6_<case>/), on one fixed window.
2. Cup misalignment, real RGB vs simulated RGB, in real-image pixels: the simulated cup is the instance mask; the
   real cup is located on the photo by its printed brown foot band (centre column and width), the table under
   it (bottom edge) and the shadow under the rim lip (top edge). The real rig's stereo calibration is quoted at
   ~2 px RMS (in thermal pixels, i.e. ~4 px here), the fitted camera pose at 2.2 px RMS (task 2).
3. Outside view of the simulated world with both cameras' fields of view drawn: each image border is
   undistorted with that camera's calibration, cast from its optical centre, and stopped at the table top or
   the board, whichever it meets first.
4. The results table (PNG + Markdown) from the English comparison sheets' figures.json and the task-6 edges.

Writes into bench/cups_check/presentation/: rgb_thermal_<case>_<palette>.png, rgb_alignment_<case>.png,
world_<case>.png, results_table.{png,md}, presentation.json.
"""
from __future__ import annotations

import glob
import json
import pathlib
import re
import sys

import numpy as np
from scipy.ndimage import map_coordinates

HERE = pathlib.Path(__file__).resolve().parent
CUPS = HERE.parent
REPO = CUPS.parents[1]
sys.path.insert(0, str(CUPS))
import cups_common as cc  # noqa: E402
import task5_figures as t5  # noqa: E402

CASES = ("hot_water_50", "cold_water_95")
LABEL = {"hot_water_50": "Hot water, 50 cm", "hot_water_95": "Hot water, 95 cm",
         "cold_water_50": "Cold water, 50 cm", "cold_water_95": "Cold water, 95 cm"}
CALIB = REPO / "thermal_specs/cali_board_new/RGBT_calibration_result_matrix.txt"
F = np.diag([1.0, -1.0, -1.0, 1.0])
WINDOW_C = t5.WINDOW_C
PALETTES = t5.PALETTES


# -- calibration ------------------------------------------------------------------------------------------


def rgb_calibration() -> tuple[np.ndarray, np.ndarray]:
    """(K 3x3, dist k1 k2 p1 p2 k3) of the RGB camera, parsed from the calibration report."""
    txt = CALIB.read_text()
    m = re.search(r"\[RGB Intrinsic Matrix K\].*?fx=([\d.]+) fy=([\d.]+) cx=([\d.]+) cy=([\d.]+)", txt, re.S)
    fx, fy, cx, cy = (float(v) for v in m.groups())
    d = re.search(r"\[RGB Distortion Coefficients\][^\n]*\n\s*\[([^\]]+)\]", txt)
    dist = np.array([float(v) for v in d.group(1).split()])
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), dist


def distort(x, y, dist):
    k1, k2, p1, p2, k3 = dist
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    return x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y


def undistort(xd, yd, dist, iters: int = 60):
    """Normalised distorted -> undistorted coordinates (fixed-point iteration) and a validity mask: the
    polynomial is only invertible inside the radius where r (1 + k1 r^2 + ...) stops growing, which for the RGB
    lens lies inside the image corners (the corners were never seen by the calibration)."""
    x, y = xd.copy(), yd.copy()
    old = np.seterr(over="ignore", invalid="ignore")
    for _ in range(iters):
        dx, dy = distort(x, y, dist)
        x, y = x + (xd - dx), y + (yd - dy)
    ex, ey = distort(x, y, dist)
    ok = (np.hypot(ex - xd, ey - yd) < 1e-4) & (np.hypot(x, y) < radial_peak(dist))
    np.seterr(**old)
    x, y = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
    return x, y, ok


def radial_peak(dist) -> float:
    r = np.linspace(0, 3, 30001)
    k1, k2, _p1, _p2, k3 = dist
    rd = r * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)
    d = np.diff(rd)
    return float(r[np.argmax(d <= 0)] if np.any(d <= 0) else r[-1])


def undistort_clamped(xd, yd, dist):
    """For drawing a field of view: radial-only inversion clamped at the polynomial's peak (tangential terms are
    < 0.2 % here and are dropped)."""
    k1, k2, _p1, _p2, k3 = dist
    rpk = radial_peak(dist)
    r = np.linspace(0, rpk, 20001)
    rd = r * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)
    rdn = np.hypot(xd, yd)
    ru = np.interp(rdn, rd, r, right=rpk)
    s = np.where(rdn > 0, ru / np.maximum(rdn, 1e-12), 1.0)
    return xd * s, yd * s


# -- RGB: real vs simulated ---------------------------------------------------------------------------------


def sim_rgb_in_real_pixels(case: str, k: np.ndarray, dist: np.ndarray):
    """(simulated RGB 960x1280x3 uint8, simulated cup mask, valid mask) in the real lens's pixel grid."""
    d = HERE / "render" / case
    cams = json.loads((d / "cams.json").read_text())
    from PIL import Image

    canvas = np.asarray(Image.open(d / "rgb_pinhole.png").convert("RGB"), np.float32)
    ids = np.load(d / "rgb_ids.npy")
    labels = {int(i): p for i, p in json.loads((d / "rgb_labels.json").read_text()).items()}
    cup_ids = [i for i, p in labels.items() if p.endswith("/CupWet") or p.endswith("/CupDry")]
    w_r, h_r = cams["rgb"]["w"], cams["rgb"]["h"]
    fx_r, fy_r = cams["rgb"]["fx_px"], cams["rgb"]["fy_px"]
    assert abs(cams["rgb"]["proj_2_0"]) < 1e-9 and abs(cams["rgb"]["proj_2_1"]) < 1e-9, "principal point not centred"
    vv, uu = np.mgrid[0:960, 0:1280].astype(np.float64)
    xd, yd = (uu - k[0, 2]) / k[0, 0], (vv - k[1, 2]) / k[1, 1]
    x, y, ok = undistort(xd, yd, dist)
    ur, vr = fx_r * x + (w_r / 2 - 0.5), fy_r * y + (h_r / 2 - 0.5)
    ok &= (ur >= 0) & (ur <= w_r - 1) & (vr >= 0) & (vr <= h_r - 1)
    out = np.zeros((960, 1280, 3), np.float32)
    for c in range(3):
        out[..., c] = map_coordinates(canvas[..., c], [vr, ur], order=1, mode="constant")
    out[~ok] = 0
    cup = map_coordinates(np.isin(ids, cup_ids).astype(np.float32), [vr, ur], order=0, mode="constant") > 0.5
    return out.astype(np.uint8), cup & ok, ok


def real_rgb(case: str, t_ref: float) -> tuple[np.ndarray, str]:
    """The RGB photo closest in time to the thermal reference frame."""
    from PIL import Image

    files = sorted(glob.glob(str(cc.DATA / case / "rgb" / "*.png")))
    best = min(files, key=lambda f: abs(cc.frame_time_s(f) - t_ref))
    return np.asarray(Image.open(best).convert("RGB")), pathlib.Path(best).name


def sim_cup_features(mask: np.ndarray) -> dict:
    rows = np.nonzero(mask.any(1))[0]
    top, bot = rows.min() - 0.5, rows.max() + 0.5
    h = bot - top

    def span(row):
        c = np.nonzero(mask[int(round(row))])[0]
        return c.min() - 0.5, c.max() + 0.5

    b0, b1 = span(bot - 0.08 * h)
    t0, t1 = span(top + 0.08 * h)
    return {"top_row": float(top), "bottom_row": float(bot), "height_px": float(h),
            "bottom_width_px": float(b1 - b0), "top_width_px": float(t1 - t0),
            "centre_col_bottom": float((b0 + b1) / 2), "centre_col_top": float((t0 + t1) / 2)}


def real_cup_features(img: np.ndarray, guess: dict) -> dict:
    """Locate the real cup on the photo near where the simulated one is (`guess`, the sim features)."""
    a = img.astype(np.float64) / 255
    v = a.max(2)
    h0, cx0 = guess["height_px"], guess["centre_col_bottom"]
    r0, r1 = int(guess["top_row"] - 0.4 * h0), int(guess["bottom_row"] + 0.4 * h0)
    # 1. The printed brown foot band. Per row, the dark pixels inside a window around the expected cup; a band
    # row has many of them and its dark run stays clear of the window's edges (the board's bottom shadow line
    # and the ruler run across the whole window). Light letters printed in the band break it into pieces, so the
    # extent is the median of the per-row first and last dark columns, not the longest run.
    dark = v < 0.45
    c0, c1 = int(cx0 - 0.8 * guess["top_width_px"]), int(cx0 + 0.8 * guess["top_width_px"])
    rows, mins, maxs = [], [], []
    for r in range(r0, r1):
        cols = np.nonzero(dark[r, c0:c1])[0] + c0
        if len(cols) >= 0.35 * guess["top_width_px"] and cols.min() > c0 + 2 and cols.max() < c1 - 3:
            rows.append(r)
            mins.append(cols.min())
            maxs.append(cols.max())
    br0, br1 = min(rows), max(rows)
    b0, b1 = float(np.median(mins)) - 0.5, float(np.median(maxs)) + 0.5
    cx = (b0 + b1) / 2
    w = b1 - b0
    # 2. Bottom edge. Under the band is a white paper strip (a brightness plateau), then the cup's contact shadow
    # on the table: the edge is where the brightness falls half-way from the plateau to the contact minimum. At
    # 95 cm the contact shadow is not resolved (the foot sits on the table's far line); there the strip is
    # scaled from its 50 cm measurement, 5.0 % of the cup height (6 px of 121), and flagged.
    mid = slice(int(cx - 0.25 * w), int(cx + 0.25 * w))
    pv = v[:, mid].mean(1)
    win = np.arange(br1 + 1, int(br1 + 0.07 * h0) + 2)             # the strip is ~5 % of the cup height
    plateau_row = int(win[np.argmax(pv[win])])
    after = np.arange(plateau_row, int(plateau_row + 0.06 * h0) + 2)
    vmin = pv[after].min()
    drop = pv[plateau_row] - vmin
    if drop > 0.06:
        half = pv[plateau_row] - 0.5 * drop
        r = plateau_row
        while pv[r + 1] > half:
            r += 1
        frac = (pv[r] - half) / (pv[r] - pv[r + 1])
        bottom, bottom_method = r + frac, "contact-shadow half drop"
    else:
        bottom, bottom_method = br1 + 0.5 + 0.050 * h0, "band + strip scaled from 50 cm (estimate)"
    # 3. Top edge: the dark line the rim lip throws on the cup, searched in the left part of the cup (the
    # printed drawing is on the right), then the lip's own height above it. The lip is ~2.5 % of the cup height.
    lcols = slice(int(cx - 0.40 * w), int(cx - 0.15 * w))
    prof = v[:, lcols].mean(1)
    lo, hi = int(bottom - 1.25 * h0), int(bottom - 0.75 * h0)
    shadow = lo + int(np.argmin(prof[lo:hi]))
    above = prof[lo:shadow]
    # the top of the lip: the last row above the shadow that still differs from the board level above it
    board = np.median(prof[lo - 15:lo])
    lip = shadow
    while lip > lo and abs(prof[lip - 1] - board) > 0.012:
        lip -= 1
    del above
    top = lip - 0.5
    return {"top_row": float(top), "bottom_row": float(bottom), "height_px": float(bottom - top),
            "bottom_width_px": float(w), "centre_col_bottom": float(cx), "rim_shadow_row": float(shadow),
            "band_rows": [int(br0), int(br1)], "bottom_method": bottom_method,
            "note": "foot width = printed band extent (the band stops ~1-3 px short of the foot's edges)"}


def thermal_pair(case: str):
    t1 = json.loads((cc.OUT / "task1_calibration.json").read_text())
    real, water, t_s = t5.real_frame_c(case, t1)
    sim = np.load(cc.OUT / f"sim6_{case}" / "apparent_t_k.npy") - np.float32(273.15)
    check = np.load(HERE / "render" / case / "thermal_apparent_t_k.npy") - np.float32(273.15)
    reg = cc.find_cup(cc.band_l(sim + 273.15), case.startswith("hot"))["water"]
    real_mean = float(cc.roi(real, water).mean())
    sim_mean = float(cc.roi(sim, reg).mean())
    return real, sim, t_s, real_mean, sim_mean, float(np.abs(check - sim).max())


def four_panel(case: str, pal: str, rgb_r, rgb_s, cup_s, feat_r, feat_s, th_r, th_s, tm_r, tm_s, photo: str):
    import matplotlib.pyplot as plt

    lo, hi = WINDOW_C["hot" if case.startswith("hot") else "cold"]
    fig = plt.figure(figsize=(14.0, 11.0), dpi=130)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.04], left=0.02, right=0.93, top=0.90, bottom=0.07,
                          wspace=0.05, hspace=0.16)
    ax = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    ax[0].imshow(rgb_r)
    ax[0].set_title(f"Real RGB (frame {photo.split('_')[0]}, nearest the thermal reference frame)", fontsize=12)
    ax[1].imshow(rgb_s)
    ax[1].set_title("Simulated RGB (calibrated intrinsics + RGB→thermal extrinsics)", fontsize=12)
    for a in ax[:2]:
        a.contour(cup_s.astype(float), levels=[0.5], colors=["#00e5ff"], linewidths=1.2)
        a.plot([feat_r["centre_col_bottom"] - feat_r["bottom_width_px"] / 2,
                feat_r["centre_col_bottom"] + feat_r["bottom_width_px"] / 2],
               [feat_r["bottom_row"]] * 2, color="#ff2d95", lw=1.4)
        a.plot([feat_r["centre_col_bottom"]] * 2, [feat_r["top_row"], feat_r["bottom_row"]], color="#ff2d95", lw=1.0)
    dx = feat_s["centre_col_bottom"] - feat_r["centre_col_bottom"]
    dyb = feat_s["bottom_row"] - feat_r["bottom_row"]
    ax[0].text(0.01, 0.01, "cyan: simulated cup outline · magenta: real cup (foot width, axis)", color="w",
               fontsize=9, transform=ax[0].transAxes, va="bottom", bbox=dict(fc="k", alpha=0.5, lw=0))
    ax[1].text(0.01, 0.01, f"misalignment sim − real: Δx = {dx:+.1f} px, Δy(bottom) = {dyb:+.1f} px",
               color="w", fontsize=9, transform=ax[1].transAxes, va="bottom", bbox=dict(fc="k", alpha=0.5, lw=0))
    cmap = PALETTES[pal]
    ax[2].imshow(th_r, cmap=cmap, vmin=lo, vmax=hi)
    ax[2].set_title(f"Real thermal, calibrated · mean water {tm_r:.1f} °C", fontsize=12)
    im = ax[3].imshow(th_s, cmap=cmap, vmin=lo, vmax=hi)
    ax[3].set_title(f"Simulated thermal (CTC-006, measured blur) · mean water {tm_s:.1f} °C", fontsize=12)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[1, 2]))
    cb.set_label(f"Apparent temperature (°C), ε = 1 · fixed window {lo:g}–{hi:g} °C", fontsize=10)
    fig.suptitle(f"{LABEL[case]}: RGB and thermal, real vs. simulated", fontsize=16, fontweight="bold", y=0.965)
    fig.text(0.475, 0.925, "Simulated RGB: geometry and framing only (flat colours, no room around the board; black "
             "corners: outside the calibration's valid radius). Thermal pair: same window and palette.", ha="center",
             fontsize=10, color="#444")
    fig.text(0.475, 0.025, "RGB→thermal calibration (16/26 views): stereo reprojection error 2.2 px (thermal pixels). "
             "Real thermal: the frame the thermometer reading refers to; simulated: the scene at that reading.",
             ha="center", fontsize=9, color="#444")
    fig.savefig(HERE / f"rgb_thermal_{case}_{pal}.png", facecolor="white")
    plt.close(fig)


def alignment_zoom(case: str, rgb_r, rgb_s, cup_s, feat_r, feat_s) -> None:
    import matplotlib.pyplot as plt

    cx, cy = feat_r["centre_col_bottom"], 0.5 * (feat_r["top_row"] + feat_r["bottom_row"])
    half = 1.1 * max(feat_r["height_px"], feat_r["bottom_width_px"])
    box = (cx - half, cx + half, cy + 0.75 * half, cy - 0.75 * half)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.9), dpi=130)
    for a, img, name in ((axs[0], rgb_r, "Real RGB"), (axs[1], rgb_s, "Simulated RGB")):
        a.imshow(img, interpolation="nearest")
        a.contour(cup_s.astype(float), levels=[0.5], colors=["#00e5ff"], linewidths=1.4)
        for f, col in ((feat_r, "#ff2d95"), (feat_s, "#00e5ff")):
            a.axhline(f["bottom_row"], color=col, lw=0.8, ls="--")
            a.axhline(f["top_row"], color=col, lw=0.8, ls="--")
            a.axvline(f["centre_col_bottom"], color=col, lw=0.8, ls="--")
        a.set_xlim(box[0], box[1])
        a.set_ylim(box[2], box[3])
        a.set_title(name, fontsize=12)
        a.set_xticks([])
        a.set_yticks([])
    d = {k: feat_s[k] - feat_r[k] for k in ("centre_col_bottom", "bottom_row", "top_row", "bottom_width_px",
                                             "height_px")}
    fig.suptitle(f"{LABEL[case]}: cup position, real vs. simulated RGB (magenta = real, cyan = simulated)",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.03, f"sim − real: axis {d['centre_col_bottom']:+.1f} px · bottom {d['bottom_row']:+.1f} px · "
             f"top {d['top_row']:+.1f} px · foot width {d['bottom_width_px']:+.1f} px · height {d['height_px']:+.1f} px "
             "(real-image pixels, 1280 x 960)", ha="center", fontsize=10, color="#444")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(HERE / f"rgb_alignment_{case}.png", facecolor="white")
    plt.close(fig)


# -- outside view ---------------------------------------------------------------------------------------


def fov_border(m_wc_usd: np.ndarray, k: np.ndarray, dist: np.ndarray, w: int, h: int, depth_m: float,
               n: int = 60):
    """The image border (true, distorted FOV) on the plane `depth_m` in front of the camera, in world
    coordinates, and the four corners' indices. Border pixels beyond the radius where the calibration's
    polynomial stops being invertible (the RGB lens's corners) are drawn at that radius."""
    t = np.linspace(0, 1, n, endpoint=False)
    u = np.r_[t * (w - 1), np.full(n, w - 1.0), (1 - t) * (w - 1), np.zeros(n)]
    v = np.r_[np.zeros(n), t * (h - 1), np.full(n, h - 1.0), (1 - t) * (h - 1)]
    x, y = undistort_clamped((u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], dist)
    d_cv = np.stack([x, y, np.ones_like(x)], 1) * depth_m
    p_w = (m_wc_usd[:3, :3] @ (F[:3, :3] @ d_cv.T)).T + m_wc_usd[:3, 3]
    return p_w, [0, n, 2 * n, 3 * n]


def project(m_wc_usd: np.ndarray, f_px: float, w: int, h: int, p: np.ndarray) -> np.ndarray:
    pc = (np.linalg.inv(m_wc_usd) @ np.c_[p, np.ones(len(p))].T).T[:, :3]
    z = -pc[:, 2]
    return np.stack([w / 2 - 0.5 + f_px * pc[:, 0] / z, h / 2 - 0.5 - f_px * pc[:, 1] / z], 1)


def world_view(case: str, k_rgb, dist_rgb) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    d = HERE / "render" / case
    cams = json.loads((d / "cams.json").read_text())
    ext = np.asarray(Image.open(d / "ext.png").convert("RGB"))
    e = cams["ext"]
    m_ext = np.array(e["world_from_camera_usd"])
    cup_depth = -next(o for o in cams["objects"] if o["name"] == "CupWet")["centre"][0]
    k_th = cc.K_TH
    rigs = (("Thermal camera (CTC-006, 14.6 mm)", np.array(cams["thermal"]["world_from_camera_usd"]), k_th,
             cc.DIST_TH, 640, 480, "#ff5a1f"),
            ("RGB camera (1280 x 960)", np.array(cams["rgb"]["world_from_camera_usd"]), k_rgb, dist_rgb, 1280, 960,
             "#2f7bff"))
    fig, ax = plt.subplots(figsize=(14, 8.9), dpi=130)
    ax.imshow(ext)
    for name, m, k, dist, w, h, col in rigs:
        o = m[:3, 3]
        border, corners = fov_border(m, k, dist, w, h, cup_depth)
        uv = project(m_ext, e["fx_px"], e["w"], e["h"], border)
        uo = project(m_ext, e["fx_px"], e["w"], e["h"], o[None])[0]
        ax.fill(uv[:, 0], uv[:, 1], color=col, alpha=0.12, lw=0)
        ax.plot(np.r_[uv[:, 0], uv[0, 0]], np.r_[uv[:, 1], uv[0, 1]], color=col, lw=1.6, label=name)
        for i in corners:
            ax.plot([uo[0], uv[i, 0]], [uo[1], uv[i, 1]], color=col, lw=1.0, alpha=0.9)
        ax.plot(*uo, "o", color=col, ms=4)
    ax.set_xlim(0, e["w"] - 1)
    ax.set_ylim(e["h"] - 1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=11, framealpha=0.85)
    ax.set_title(f"Simulated world, outside view — {LABEL[case]}", fontsize=15, fontweight="bold")
    ax.text(0.99, 0.01, f"Fields of view: each camera's image border (true, distorted FOV) drawn on the plane "
            f"{cup_depth:.2f} m in front of it, through the cup.\nRGB corners clipped where its distortion model "
            "stops being invertible. Boxes mark the camera centres (not modelled bodies). Board 1.4 x 0.9 m.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="w",
            bbox=dict(fc="k", alpha=0.45, lw=0))
    fig.tight_layout()
    fig.savefig(HERE / f"world_{case}.png", facecolor="white")
    plt.close(fig)


# -- results table --------------------------------------------------------------------------------------


def results_table() -> list[dict]:
    import matplotlib.pyplot as plt

    fj = json.loads((HERE / "figures_en" / "figures.json").read_text())["mean_apparent_t_c_water_region_and_edges"]
    rows = []
    for c in ("hot_water_50", "hot_water_95", "cold_water_50", "cold_water_95"):
        r = fj[c]
        rows.append({"case": LABEL[c], "real_c": r["real_mean"], "sim_c": r["sim_mean"], "diff_k": r["diff"],
                     "independent": "yes" if c.endswith("95") else "no (calibration fitted here)",
                     "edge_real_px": r["edge_real_px"], "edge_sim_px": r["edge_sim_px"]})
    head = ["Case", "Real T (°C)", "Simulated T (°C)", "Real − sim (K)", "Independent test",
            "10–90 % edge, real (px)", "10–90 % edge, sim (px)"]
    cells = [[r["case"], f"{r['real_c']:.1f}", f"{r['sim_c']:.1f}", f"{r['diff_k']:+.1f}".replace("-", "−"),
              r["independent"], f"{r['edge_real_px']:.1f}", f"{r['edge_sim_px']:.1f}"] for r in rows]
    notes = ("T: mean apparent temperature (ε = 1) over the water plateau, same region definition on both images. "
             "Real: calibrated frame the thermometer reading refers to.\nSimulated: CTC-006 camera with the measured "
             "blur, surface at the reading (66 / 58 / 11 / 11 °C), ε = 0.93. Edge: model-free 10–90 % width of the "
             "cup's side edges.")
    md = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] + ["---:"] * 3 + ["---"] + ["---:"] * 2) + "|"]
    md += ["| " + " | ".join(c) + " |" for c in cells]
    (HERE / "results_table.md").write_text("# Cups: real vs. simulated\n\n" + "\n".join(md) + "\n\n"
                                           + notes.replace("\n", " ") + "\n")
    fig, ax = plt.subplots(figsize=(14, 2.9), dpi=150)
    ax.axis("off")
    tab = ax.table(cellText=cells, colLabels=head, loc="center", cellLoc="center", colLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(11)
    tab.scale(1, 1.9)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        if r == 0:
            cell.set_facecolor("#2b2b2b")
            cell.set_text_props(color="w", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f4f4f4")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left")
    tab.auto_set_column_width(list(range(len(head))))
    ax.set_title("Water cups: real vs. simulated (LWIR 8–14 µm)", fontsize=14, fontweight="bold", pad=6)
    fig.text(0.5, 0.02, notes, ha="center", fontsize=8.5, color="#444")
    fig.savefig(HERE / "results_table.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return rows


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    t5.LANG = "en"
    k_rgb, dist_rgb = rgb_calibration()
    summary = {"rgb_calibration": {"K": k_rgb.tolist(), "dist": dist_rgb.tolist(),
                                   "polynomial_invertible_to_r": radial_peak(dist_rgb)}, "cases": {}}
    for case in CASES:
        th_r, th_s, t_ref, tm_r, tm_s, chk = thermal_pair(case)
        rgb_r, photo = real_rgb(case, t_ref)
        rgb_s, cup_s, valid = sim_rgb_in_real_pixels(case, k_rgb, dist_rgb)
        feat_s = sim_cup_features(cup_s)
        feat_r = real_cup_features(rgb_r, feat_s)
        for pal in PALETTES:
            four_panel(case, pal, rgb_r, rgb_s, cup_s, feat_r, feat_s, th_r, th_s, tm_r, tm_s, photo)
        alignment_zoom(case, rgb_r, rgb_s, cup_s, feat_r, feat_s)
        world_view(case, k_rgb, dist_rgb)
        delta = {k: round(feat_s[k] - feat_r[k], 2) for k in
                 ("centre_col_bottom", "bottom_row", "top_row", "bottom_width_px", "height_px")}
        summary["cases"][case] = {"photo": photo, "real_cup_px": feat_r, "sim_cup_px": feat_s,
                                  "sim_minus_real_px": delta, "valid_fraction_of_real_frame": float(valid.mean()),
                                  "thermal_render_check_max_abs_k": chk,
                                  "mean_water_t_c": {"real": tm_r, "sim": tm_s}}
        print(case, photo, json.dumps(delta), f"valid {valid.mean():.3f}", f"thermal check {chk:.3g} K")
    summary["table"] = results_table()
    (HERE / "presentation.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
