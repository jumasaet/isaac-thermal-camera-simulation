#!/usr/bin/env python
"""Task 2: the cup's size in pixels at 50 and 95 cm -- measured, from the camera's calibration, from the simulator.

    /home/jumasaet/isaac_sim/python.sh bench/cups_check/task2_geometry.py

Measured: on each set's mean frame (blur 10-90 % ~ 14 px, symmetric, so a 50 % crossing is an unbiased edge).
  - width per row: the two 50 % crossings between the board level and the cup's own level on that row
    (median of its central columns), sub-pixel by linear interpolation;
  - top and bottom: the steepest step of the column profile averaged over the central columns (the dry
    band under the rim and the table reflection under the cup have their own levels, so 50 % of what is
    ill-defined there; the steepest step is not);
  - reported: width 10 % of the height below the top (the rim, 7 cm), width 10 % above the bottom (5 cm),
    height top-to-bottom.
Predicted: the same three numbers off a 3-D frustum (7 / 5 cm diameters, 6 cm tall) standing on the table,
  placed and viewed with what the LiDAR says (table plane -> camera height, pitch and roll; cup front
  surface -> cup axis), projected
  (a) with the thermal calibration (fx 626.9, fy 620.6, Brown-Conrady k1 k2 p1 p2 k3) -- "intrinsic";
  (b) with the simulator's pinhole from the draft YAML: f = 22.2 mm / 24 um = 925 px, format centre, no
      distortion -- what render_two_cups.py would draw at the same pose ("simulator").
  Both also at the nominal distance (axis at 0.50 / 0.95 m) in place of the LiDAR one.

Writes out/task2_geometry.json, out/task2_geometry.png and out/lidar_geometry.json (camera pose and cup
positions, read by task3_render_cups.py).
"""
from __future__ import annotations

import json

import numpy as np

import cups_common as cc

SIM_F_PX = 22.2e-3 / 24e-6
SIM_C = (319.5, 239.5)


# -- measurement --------------------------------------------------------------------------------------


def _crossing(p: np.ndarray, x0: int, step: int, level: float) -> float:
    """First sub-pixel index from x0 in direction step where p crosses level."""
    x = x0
    s0 = p[x] - level
    while 0 < x + step < len(p) - 1:
        s1 = p[x + step] - level
        if np.sign(s1) != np.sign(s0):
            return x + step * (s0 / (s0 - s1))
        x += step
    return float("nan")


def measure_silhouette(img: np.ndarray, regions: dict, board: float, hot: bool) -> dict:
    s = (img - board) * (1 if hot else -1)
    c0, c1 = regions["half_width_cols"]
    cm = (c0 + c1) // 2
    w = c1 - c0
    colprof = s[:, cm - w // 6:cm + w // 6 + 1].mean(1)
    g = np.convolve(np.gradient(colprof), np.ones(3) / 3, mode="same")
    pr0, pr1 = regions["plateau_rows"]
    # Rising step board -> cup: the rim, which is the FIRST significant rising step from above -- the water
    # line below it is often the steeper one (the dry band of the cold cup is barely off the board).
    win = g[pr0 - 70:pr0 + 1]
    gmax = float(win.max())
    cand = [i for i in range(1, len(win) - 1) if win[i] >= 0.2 * gmax and win[i] >= win[i - 1] and win[i] >= win[i + 1]]
    top = cand[0] + pr0 - 70
    bot = int(np.argmin(g[pr1:pr1 + 40])) + pr1  # falling step cup -> table
    # sub-pixel: parabola through the gradient peak
    top_f = top + _vertex(g[top - 1:top + 2])
    bot_f = bot + _vertex(-g[bot - 1:bot + 2])
    rows = np.arange(int(np.ceil(top_f)) + 3, int(np.floor(bot_f)) - 2)
    widths, lefts, rights = [], [], []
    for r in rows:
        p = s[r]
        inner = float(np.median(p[cm - w // 6:cm + w // 6 + 1]))
        lv = 0.5 * inner
        lefts.append(_crossing(p, cm, -1, lv))
        rights.append(_crossing(p, cm, +1, lv))
    lefts, rights = np.array(lefts), np.array(rights)
    widths = rights - lefts
    ok = np.isfinite(widths)
    k, b = np.polyfit(rows[ok], widths[ok], 1)
    hgt = bot_f - top_f
    r_top, r_bot = top_f + 0.1 * hgt, bot_f - 0.1 * hgt
    return {"top_row": round(float(top_f), 2), "bottom_row": round(float(bot_f), 2), "height_px": round(float(hgt), 2),
            "width_top10_px": round(float(k * r_top + b), 2), "width_bottom10_px": round(float(k * r_bot + b), 2),
            "centre_col": round(float(np.nanmedian((lefts + rights) / 2)), 2),
            "_rows": rows.tolist(), "_lefts": lefts.tolist(), "_rights": rights.tolist()}


def _vertex(y3: np.ndarray) -> float:
    a, b, c = (float(v) for v in y3)
    den = a - 2 * b + c
    return 0.0 if den == 0 else 0.5 * (a - c) / den


def silhouette_of_points(u: np.ndarray, v: np.ndarray) -> dict:
    """Same three numbers off projected surface points of the model (rows = integer v bins of 0.25 px)."""
    top, bot = float(v.min()), float(v.max())
    hgt = bot - top

    def width_at(row: float) -> float:
        m = np.abs(v - row) < 0.25
        return float(u[m].max() - u[m].min())

    return {"top_row": round(top, 2), "bottom_row": round(bot, 2), "height_px": round(hgt, 2),
            "width_top10_px": round(width_at(top + 0.1 * hgt), 2), "width_bottom10_px": round(width_at(bot - 0.1 * hgt), 2),
            "centre_col": round(float(0.5 * (u.min() + u.max())), 2)}


# -- LiDAR geometry -------------------------------------------------------------------------------------


def lidar_geometry(case: str, regions: dict) -> dict:
    p = np.concatenate([cc.read_pcd(f) for f in cc.frame_files(case, "lidar")])
    pt = cc.lidar_to_thermal(p)
    pt = pt[pt[:, 2] > 0.15]
    u, v = cc.project_thermal(pt)
    img = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
    # Table plane: lower image, nearer than the board; RANSAC then least squares.
    q = pt[img & (v > 300) & (pt[:, 2] < 0.95)]
    rng = np.random.default_rng(0)
    best = (0, None)
    for _ in range(3000):
        s = q[rng.choice(len(q), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((q - s[0]) @ n) < 0.006
        if inl.sum() > best[0]:
            best = (int(inl.sum()), inl)
    x = q[best[1]]
    c = x.mean(0)
    n = np.linalg.svd(x - c, full_matrices=False)[2][2]  # full_matrices=True would build an N x N U
    up = n if n[1] < 0 else -n  # OpenCV camera: +y is down, so "up" has negative y
    h_cam = float(-(c @ up))  # camera origin above the plane
    # Board: a vertical plane ~1 m ahead.
    mb = img & (v > 60) & (v < 220) & (u > 150) & (u < 500) & (pt[:, 2] > 0.98) & (pt[:, 2] < 1.2)
    b = pt[mb]
    cb = b.mean(0)
    nb = np.linalg.svd(b - cb, full_matrices=False)[2][2]
    nb *= np.sign(nb[2])
    # Cup front surface: points inside the cup's half-width columns and plateau-ish rows, 1-5 cm above the
    # table, in front of the board.
    c0, c1 = regions["half_width_cols"]
    r0, r1 = regions["plateau_rows"]
    hz = pt @ up + h_cam
    m = img & (u > c0 + 3) & (u < c1 - 3) & (v > r0 - 25) & (v < r1 + 5) & (hz > 0.01) & (hz < 0.055)
    m &= (pt @ nb) < (cb @ nb) - 0.03
    cup = pt[m]
    zs = np.median(cup[:, 2])
    cup = cup[np.abs(cup[:, 2] - zs) < 0.03]
    front = np.median(cup, axis=0)
    zf = float(front @ up + h_cam)
    r_at = cc.CUP_BOTTOM_DIAMETER_M / 2 + (cc.CUP_TOP_DIAMETER_M - cc.CUP_BOTTOM_DIAMETER_M) / 2 * zf / cc.CUP_HEIGHT_M
    ray = front - (front @ up) * up
    ray /= np.linalg.norm(ray)
    axis_pt = front + r_at * ray
    base = axis_pt - (axis_pt @ up + h_cam) * up  # foot of the axis on the table
    fwd = np.array([0, 0, 1.0])
    fwd_h = fwd - (fwd @ up) * up
    fwd_h /= np.linalg.norm(fwd_h)
    foot_cam = -h_cam * up
    rel = base - foot_cam
    return {"n_points_cup": int(len(cup)), "camera_height_above_table_m": round(h_cam, 4),
            "camera_pitch_up_deg": round(float(np.degrees(np.arcsin(fwd @ up))), 3),
            "camera_roll_deg": round(float(np.degrees(np.arcsin(np.array([1.0, 0, 0]) @ up))), 3),
            "board_distance_along_normal_m": round(float(abs(cb @ nb)), 4),
            "board_distance_horizontal_m": round(float((cb - foot_cam) @ fwd_h), 4),
            "cup_front_depth_m": round(float(front[2]), 4), "cup_front_height_m": round(zf, 4),
            "cup_axis_horizontal_distance_m": round(float(rel @ fwd_h), 4),
            "cup_axis_lateral_m": round(float(rel @ np.cross(up, fwd_h)), 4),
            "_up": up.tolist(), "_base": base.tolist()}


def frustum_points(base: np.ndarray, up: np.ndarray, n_phi: int = 1440, n_z: int = 241) -> np.ndarray:
    e1 = np.cross(up, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    z = np.linspace(0, cc.CUP_HEIGHT_M, n_z)
    r = cc.CUP_BOTTOM_DIAMETER_M / 2 + (cc.CUP_TOP_DIAMETER_M - cc.CUP_BOTTOM_DIAMETER_M) / 2 * z / cc.CUP_HEIGHT_M
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    rr = r[:, None]
    pts = (base[None, None, :] + zz[..., None] * up + rr[..., None] * (np.cos(pp)[..., None] * e1 + np.sin(pp)[..., None] * e2))
    return pts.reshape(-1, 3)


def project_sim(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return SIM_F_PX * p[:, 0] / p[:, 2] + SIM_C[0], SIM_F_PX * p[:, 1] / p[:, 2] + SIM_C[1]


def image_pose(sil: dict, geo: dict) -> dict:
    """Camera height and pitch above the table from the image itself.

    The LiDAR pose puts every cup 15-23 px lower than it is (a ~1-2 deg pitch error somewhere in the
    LiDAR -> RGB -> thermal chain, whose own RMSE is 1.4 cm). The rows of the front rim (z = 6 cm, at the
    axis distance minus 3.5 cm) and of the front bottom edge (z = 0, minus 2.5 cm) in the four sets fix
    height and pitch with the LiDAR's horizontal distances and the calibrated intrinsics + distortion.
    """
    from scipy.optimize import least_squares

    obs = []
    for c in cc.CASES:
        d, lat = geo[c]["cup_axis_horizontal_distance_m"], geo[c]["cup_axis_lateral_m"]
        obs.append((d - cc.CUP_TOP_DIAMETER_M / 2, lat, cc.CUP_HEIGHT_M, sil[c]["top_row"]))
        obs.append((d - cc.CUP_BOTTOM_DIAMETER_M / 2, lat, 0.0, sil[c]["bottom_row"]))
    obs = np.array(obs)

    def rows(p):
        h, th = p[0], np.radians(p[1])
        x, y, z = obs[:, 0], obs[:, 1], obs[:, 2]
        zc = x * np.cos(th) + (z - h) * np.sin(th)
        yc = x * np.sin(th) - (z - h) * np.cos(th)
        xc = -y
        return cc.project_thermal(np.stack([xc, yc, zc], 1))[1]

    sol = least_squares(lambda p: rows(p) - obs[:, 3], x0=[0.035, 2.5])
    res = rows(sol.x) - obs[:, 3]
    return {"camera_height_above_table_m": round(float(sol.x[0]), 4), "camera_pitch_up_deg": round(float(sol.x[1]), 3),
            "row_residuals_px": [round(float(r), 2) for r in res], "rms_px": round(float(np.sqrt(np.mean(res ** 2))), 2)}


def main() -> None:
    cc.OUT.mkdir(exist_ok=True)
    rows, geo = {}, {}
    sil_meas = {}
    for case in cc.CASES:
        frames, _t = cc.load_case(case)
        mean = frames.mean(0)
        reg = cc.find_cup(mean, case.startswith("hot"))
        lv = cc.per_frame_levels(frames, reg)
        meas = measure_silhouette(mean, reg, float(np.mean(lv["bg"])), case.startswith("hot"))
        sil_meas[case] = meas
        g = lidar_geometry(case, reg)
        geo[case] = g
        up, base = np.array(g["_up"]), np.array(g["_base"])
        pts = frustum_points(base, up)
        pred_int = silhouette_of_points(*cc.project_thermal(pts))
        pred_sim = silhouette_of_points(*project_sim(pts))
        # Nominal distance: slide the cup along the table to put its axis at 0.50 / 0.95 m.
        fwd = np.array([0, 0, 1.0]) - up[2] * up
        fwd /= np.linalg.norm(fwd)
        base_nom = base + (cc.NOMINAL_DISTANCE_M[case] - g["cup_axis_horizontal_distance_m"]) * fwd
        pts_n = frustum_points(base_nom, up)
        pred_int_n = silhouette_of_points(*cc.project_thermal(pts_n))
        pred_sim_n = silhouette_of_points(*project_sim(pts_n))
        keys = ("width_top10_px", "width_bottom10_px", "height_px")
        rows[case] = {
            "measured": {k: meas[k] for k in (*keys, "top_row", "bottom_row", "centre_col")},
            "intrinsic_lidar_distance": pred_int, "simulator_lidar_distance": pred_sim,
            "intrinsic_nominal_distance": pred_int_n, "simulator_nominal_distance": pred_sim_n,
            "error_pct_intrinsic_lidar": {k: round(100 * (meas[k] / pred_int[k] - 1), 1) for k in keys},
            "error_pct_simulator_lidar": {k: round(100 * (meas[k] / pred_sim[k] - 1), 1) for k in keys},
            "error_pct_intrinsic_nominal": {k: round(100 * (meas[k] / pred_int_n[k] - 1), 1) for k in keys},
            "error_pct_simulator_nominal": {k: round(100 * (meas[k] / pred_sim_n[k] - 1), 1) for k in keys},
            "error_px_intrinsic_lidar_position": {"row_of_top": round(meas["top_row"] - pred_int["top_row"], 1),
                                                  "row_of_bottom": round(meas["bottom_row"] - pred_int["bottom_row"], 1),
                                                  "centre_col": round(meas["centre_col"] - pred_int["centre_col"], 1)},
        }
    # Ratio 50 -> 95 cm needs no intrinsic at all: a check on the distances.
    ratio = {kind: {k: round(rows[f"{kind}_water_50"]["measured"][k] / rows[f"{kind}_water_95"]["measured"][k], 3)
                    for k in ("width_top10_px", "width_bottom10_px", "height_px")} for kind in ("hot", "cold")}
    ratio["lidar_distance_ratio_hot"] = round(geo["hot_water_95"]["cup_axis_horizontal_distance_m"]
                                              / geo["hot_water_50"]["cup_axis_horizontal_distance_m"], 3)
    ratio["lidar_distance_ratio_cold"] = round(geo["cold_water_95"]["cup_axis_horizontal_distance_m"]
                                               / geo["cold_water_50"]["cup_axis_horizontal_distance_m"], 3)
    report = {"sim_f_px": round(SIM_F_PX, 2), "calibrated_fx_fy_px": [cc.K_TH[0, 0], cc.K_TH[1, 1]],
              "focal_mm_implied_by_calibration_at_24um": round(cc.K_TH[0, 0] * 24e-3, 2),
              "hfov_deg_sim_vs_calibrated_pinhole": [round(float(np.degrees(2 * np.arctan(320 / SIM_F_PX))), 2),
                                                     round(float(np.degrees(2 * np.arctan(320 / cc.K_TH[0, 0]))), 2)],
              "cases": rows, "ratio_50_over_95_measured": ratio,
              "lidar": {c: {k: v for k, v in g.items() if not k.startswith("_")} for c, g in geo.items()}}
    report["image_pose"] = image_pose(sil_meas, geo)
    (cc.OUT / "task2_geometry.json").write_text(json.dumps(report, indent=1))
    (cc.OUT / "lidar_geometry.json").write_text(json.dumps({"cases": geo, "image_pose": report["image_pose"]}, indent=1))
    print("image pose", report["image_pose"])
    plot(sil_meas, geo)
    for c in cc.CASES:
        r = rows[c]
        print(c, "meas", r["measured"])
        print("   intrinsic(lidar)", {k: r["intrinsic_lidar_distance"][k] for k in ("width_top10_px", "width_bottom10_px", "height_px", "top_row", "bottom_row")},
              "err%", r["error_pct_intrinsic_lidar"], "pos", r["error_px_intrinsic_lidar_position"])
        print("   sim(lidar)", {k: r["simulator_lidar_distance"][k] for k in ("width_top10_px", "width_bottom10_px", "height_px")}, "err%", r["error_pct_simulator_lidar"])
        print("   err% nominal: intrinsic", r["error_pct_intrinsic_nominal"], "sim", r["error_pct_simulator_nominal"])
    print(json.dumps(ratio), json.dumps(report["lidar"], indent=1))


def plot(sil: dict, geo: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, case in zip(axs, cc.CASES):
        frames, _ = cc.load_case(case)
        m = frames.mean(0)
        s = sil[case]
        cm = int(s["centre_col"])
        r0, r1 = int(s["top_row"]) - 40, int(s["bottom_row"]) + 40
        crop = m[r0:r1, cm - 80:cm + 80]
        ax.imshow(crop, cmap="gray", extent=(cm - 80, cm + 80, r1, r0))
        ax.plot(s["_lefts"], s["_rows"], "c-", lw=0.8)
        ax.plot(s["_rights"], s["_rows"], "c-", lw=0.8)
        ax.axhline(s["top_row"], color="y", lw=0.8)
        ax.axhline(s["bottom_row"], color="y", lw=0.8)
        up, base = np.array(geo[case]["_up"]), np.array(geo[case]["_base"])
        u, v = cc.project_thermal(frustum_points(base, up, 360, 61))
        ax.plot(u, v, ",", color="r", alpha=0.3)
        ax.set_title(f"{case}\nmeasured (cyan/yellow), intrinsic+LiDAR model (red)", fontsize=8)
    fig.tight_layout()
    fig.savefig(cc.OUT / "task2_geometry.png", dpi=120)


if __name__ == "__main__":
    main()
