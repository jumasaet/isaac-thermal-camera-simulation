#!/usr/bin/env python
"""Scene geometry for the three cups: LiDAR positions, camera pose fitted to the thermal silhouettes, water level.

    /home/jumasaet/isaac_sim/python.sh bench/cups3_check/task0_scene.py

1. Regions (cups3_common.find_cups) on the mean of the 94 thermal frames.
2. LiDAR (all 94 scans, ~0.96 M points) taken to the thermal frame with the rig's two extrinsics
   (thermal_specs/cali_board_new/*matrix.txt, the same rig as bench/cups_check):
     - table plane by RANSAC under the table line -> camera height, pitch and roll;
     - board plane ~1 m ahead -> distance, yaw, and its left/right/top edges;
     - each cup: its front surface points (inside its columns, 1-5.5 cm above the table, in front of the
       board) -> the axis one outer radius behind them -> the base on the table, as (forward, left) from the
       camera's foot.
3. Silhouettes measured on the mean frame (bench/cups_check/task2_geometry.measure_silhouette: rim row,
   bottom row, centre column) and the pose fitted to them: camera height, pitch, roll and yaw, with the
   cups' table positions from the LiDAR held fixed. The LiDAR's own pose is reported next to it (the LiDAR ->
   RGB -> thermal chain carried ~1.4 deg of pitch error in bench/cups_check).
4. Water level: a forward model of the column profile through each cup's centre -- board above the rim, a dry
   band between rim and water line, the wetted wall below, blurred by a Gaussian -- fitted for rim row, water
   line row, the two levels and the blur. The dry band's height in pixels -> millimetres at the cup's distance.

Writes out/scene_geometry.json (read by task2_render.py) and out/scene_geometry.png.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import least_squares

import cups3_common as c3
import task2_geometry as g2  # bench/cups_check (on sys.path via cups3_common)


# -- LiDAR ------------------------------------------------------------------------------------------------


def lidar_points() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.concatenate([c3.read_pcd(f) for f in c3.frame_files("lidar")])
    pt = c3.lidar_to_thermal(p)
    pt = pt[pt[:, 2] > 0.15]
    u, v = c3.project_thermal(pt)
    ok = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
    return pt[ok], u[ok], v[ok]


def fit_plane(q: np.ndarray, tol: float, seed: int = 0, iters: int = 3000) -> tuple[np.ndarray, np.ndarray, int]:
    """RANSAC then SVD -> (centroid, unit normal, inliers)."""
    rng = np.random.default_rng(seed)
    best = (0, None)
    for _ in range(iters):
        s = q[rng.choice(len(q), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((q - s[0]) @ n) < tol
        if inl.sum() > best[0]:
            best = (int(inl.sum()), inl)
    x = q[best[1]]
    c = x.mean(0)
    n = np.linalg.svd(x - c, full_matrices=False)[2][2]  # full_matrices=True would build an N x N U (~13 GB)
    return c, n, best[0]


def lidar_geometry(regions: dict) -> dict:
    pt, u, v = lidar_points()
    # Table: below the table line, nearer than the board.
    c_t, n_t, n_in = fit_plane(pt[(v > 300) & (pt[:, 2] < 0.95)], 0.006)
    up = n_t if n_t[1] < 0 else -n_t  # OpenCV camera: +y down
    h_cam = float(-(c_t @ up))
    hz = pt @ up + h_cam  # height above the table
    fwd = np.array([0, 0, 1.0]) - up[2] * up
    fwd /= np.linalg.norm(fwd)
    left = np.cross(up, fwd)
    foot = -h_cam * up
    # Board: vertical plane ~1 m ahead, above the table.
    mb = (v > 40) & (v < 230) & (u > 120) & (u < 540) & (pt[:, 2] > 0.85) & (pt[:, 2] < 1.3) & (hz > 0.08)
    c_b, n_b, nb_in = fit_plane(pt[mb], 0.01, seed=1)
    n_b *= np.sign(n_b @ fwd)
    board_d = float((c_b - foot) @ n_b) / float(fwd @ n_b)  # along fwd from the foot, at the foot's line
    # Board extent: points within 2 cm of its plane, above the table.
    onb = (np.abs((pt - c_b) @ n_b) < 0.02) & (hz > 0.02)
    lat_b = (pt[onb] - foot) @ left
    top_b = hz[onb]
    board = {"distance_m": round(board_d, 4), "yaw_deg": round(float(np.degrees(np.arctan2(n_b @ left, n_b @ fwd))), 2),
             "left_edge_m": round(float(np.percentile(lat_b, 99.5)), 3),
             "right_edge_m": round(float(np.percentile(lat_b, 0.5)), 3),
             "top_m": round(float(np.percentile(top_b, 99.5)), 3), "n_points": int(onb.sum())}
    cups = {}
    for cup, r in regions.items():
        c0, c1 = r["half_width_cols"]
        r0, r1 = r["plateau_rows"]
        m = (u > c0 + 4) & (u < c1 - 4) & (v > r0 - 25) & (v < r1 + 5) & (hz > 0.01) & (hz < 0.055)
        m &= ((pt - c_b) @ n_b) < -0.05
        q = pt[m]
        zs = np.median(q[:, 2])
        q = q[np.abs(q[:, 2] - zs) < 0.03]
        front = np.median(q, axis=0)
        zf = float(front @ up + h_cam)
        ray = front - (front @ up) * up
        ray /= np.linalg.norm(ray)
        axis = front + float(c3.r_at(zf)) * ray
        base = axis - (axis @ up + h_cam) * up
        rel = base - foot
        cups[cup] = {"n_points": int(len(q)), "front_depth_m": round(float(front[2]), 4),
                     "front_height_m": round(zf, 4), "axis_forward_m": round(float(rel @ fwd), 4),
                     "axis_left_m": round(float(rel @ left), 4),
                     "nearest_base_point_range_m": round(float(np.linalg.norm(rel) - c3.CUP_BOTTOM_DIAMETER_M / 2), 4)}
    return {"camera_height_above_table_m": round(h_cam, 4),
            "camera_pitch_up_deg": round(float(np.degrees(np.arcsin(up @ np.array([0, 0, 1.0])))), 3),
            "camera_roll_deg": round(float(np.degrees(-np.arcsin(up[0] / np.sqrt(1 - up[2] ** 2)))), 3),
            "table_inliers": n_in, "board": board, "cups": cups, "_up": up.tolist(), "_board_n": n_b.tolist()}


# -- the camera pose in the table frame (x forward, y left, z up) -----------------------------------------


def camera_axes(pitch_deg: float, roll_deg: float, yaw_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, down, forward) of the camera in the table frame. Positive roll turns the camera's right toward
    its down (clockwise seen from behind the camera), which lowers the left of the scene in the image."""
    th, ph, ps = np.radians([pitch_deg, roll_deg, yaw_deg])
    f = np.array([np.cos(th) * np.cos(ps), np.cos(th) * np.sin(ps), np.sin(th)])
    r0 = np.cross(f, [0, 0, 1.0])
    r0 /= np.linalg.norm(r0)
    d0 = np.cross(f, r0)
    return np.cos(ph) * r0 + np.sin(ph) * d0, -np.sin(ph) * r0 + np.cos(ph) * d0, f


def project_table(p: np.ndarray, h: float, pitch: float, roll: float, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    r, d, f = camera_axes(pitch, roll, yaw)
    q = p - np.array([0, 0, h])
    return c3.project_thermal(np.stack([q @ r, q @ d, q @ f], 1))


def frustum_table(x: float, y: float, n_phi: int = 720, n_z: int = 121) -> np.ndarray:
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    z = np.linspace(0, c3.CUP_HEIGHT_M, n_z)
    rr = c3.r_at(z)[:, None]
    pts = np.stack([x + rr * np.cos(phi), y + rr * np.sin(phi), np.broadcast_to(z[:, None], (n_z, n_phi))], -1)
    return pts.reshape(-1, 3)


def silhouette_table(x: float, y: float, pose: tuple) -> dict:
    u, v = project_table(frustum_table(x, y), *pose)
    return g2.silhouette_of_points(u, v)


def fit_pose(sil: dict, geo: dict) -> dict:
    """Pitch and roll from the measured rim and bottom rows, each cup's lateral position from its centre column.

    With every cup at ~0.5 m the rows fix only (z - h)/x + pitch, so camera height and pitch are degenerate
    (a free fit drifts to h = -1 mm): the height stays the LiDAR's. Range comes from the LiDAR; bearing comes
    from the image, which pins it to < 1 px (0.8 mm at 0.5 m) where the LiDAR's front-surface median scatters
    by ~1 cm -- so yaw is folded into the laterals (yaw = 0). 9 numbers, 5 unknowns.
    """
    h = geo["camera_height_above_table_m"]
    obs = []
    for cup in c3.CUPS:
        obs += [sil[cup]["top_row"], sil[cup]["bottom_row"], sil[cup]["centre_col"]]
    obs = np.array(obs)

    def model(p, lat=None):
        out = []
        for k, cup in enumerate(c3.CUPS):
            y = geo["cups"][cup]["axis_left_m"] if lat is None else lat[k]
            u, v = project_table(frustum_table(geo["cups"][cup]["axis_forward_m"], y, 360, 61), h, p[0], p[1], 0.0)
            out += [v.min(), v.max(), 0.5 * (u.min() + u.max())]  # unrounded, for the finite differences
        return np.array(out)

    lat0 = [geo["cups"][c]["axis_left_m"] for c in c3.CUPS]
    p0 = [geo["camera_pitch_up_deg"], geo["camera_roll_deg"], *lat0]
    sol = least_squares(lambda p: model(p[:2], p[2:]) - obs, x0=p0, x_scale=[1.0, 1.0, 0.01, 0.01, 0.01])
    res = model(sol.x[:2], sol.x[2:]) - obs
    res0 = model(p0[:2]) - obs
    return {"camera_height_above_table_m": h, "camera_pitch_up_deg": round(float(sol.x[0]), 3),
            "camera_roll_deg": round(float(sol.x[1]), 3), "camera_yaw_left_deg": 0.0,
            "axis_left_m": {c: round(float(y), 4) for c, y in zip(c3.CUPS, sol.x[2:])},
            "axis_left_minus_lidar_mm": {c: round(1000 * float(y - y0), 1) for c, y, y0 in zip(c3.CUPS, sol.x[2:], lat0)},
            "residuals_px_top_bottom_col": [round(float(x), 2) for x in res],
            "rms_px": round(float(np.sqrt(np.mean(res ** 2))), 2),
            "lidar_pose_residuals_px": [round(float(x), 2) for x in res0],
            "lidar_pose_rms_px": round(float(np.sqrt(np.mean(res0 ** 2))), 2)}


# -- water level ------------------------------------------------------------------------------------------


def _blur_steps(v: np.ndarray, edges: list[float], levels: list[float], sigma: float) -> np.ndarray:
    """Sum of Gaussian-blurred steps: level k between edges[k] and edges[k+1] (above edges[0]: 0)."""
    from scipy.special import erf

    out = np.zeros_like(v, float)
    for k, lv in enumerate(levels):
        a, b = edges[k], edges[k + 1]
        out += lv * 0.5 * (erf((v - a) / (np.sqrt(2) * sigma)) - erf((v - b) / (np.sqrt(2) * sigma)))
    return out


def water_level(mean: np.ndarray, board: np.ndarray, r: dict, sil: dict, fwd_m: float, pose: dict,
                band_mm: float | None = None) -> dict:
    """Fit board | dry band | wetted wall to the column profile through the cup centre (central third).

    band_mm: hold the water line that far below the fitted rim and fit only the levels (used for the warm cup,
    whose dry band is too close to the water's level for the line to be found)."""
    c0, c1 = r["half_width_cols"]
    cm, w = (c0 + c1) // 2, c1 - c0
    s = ((mean - board) * r["sign"])[:, cm - w // 6:cm + w // 6 + 1].mean(1)
    top = int(round(sil["top_row"]))
    rows = np.arange(top - 30, r["plateau_rows"][0] + 18)
    y = s[rows]
    wet0 = float(np.median(s[r["plateau_rows"][0]:r["plateau_rows"][0] + 15]))
    # pixels per metre of height on the cup's front wall at that height (finite difference of the projection)
    x = fwd_m - c3.r_at(c3.CUP_HEIGHT_M)
    pz = np.array([[x, 0.0, c3.CUP_HEIGHT_M], [x, 0.0, c3.CUP_HEIGHT_M - 0.01]])
    _, vv = project_table(pz, pose["camera_height_above_table_m"], pose["camera_pitch_up_deg"], 0.0, 0.0)
    px_per_m = float(vv[1] - vv[0]) / 0.01

    def model(p):
        rim, wl, dry, wet, sig, slope = p
        if band_mm is not None:
            wl = rim + band_mm * px_per_m / 1000
        # the wetted wall's level runs linearly with the row (stratification)
        lv = _blur_steps(rows.astype(float), [rim, wl, rows[-1] + 40.0], [dry, wet], sig)
        return lv + np.where(rows >= wl, slope * (rows - wl), 0.0)

    best = None
    for wl0 in (top + 3, top + 8, top + 14):
        p0 = [top - 1.0, float(wl0), 0.5 * wet0, wet0, 5.0, 0.0]
        sol = least_squares(lambda p: model(p) - y, x0=p0,
                            bounds=([top - 15, top - 10, -0.5 * wet0 - 1, 0.5 * wet0, 2.0, -50], [top + 15, top + 40, 1.5 * wet0 + 1, 1.5 * wet0, 9.0, 50]))
        if best is None or sol.cost < best.cost:
            best = sol
    rim, wl, dry, wet, sig, slope = best.x
    if band_mm is not None:
        wl = rim + band_mm * px_per_m / 1000
    band_px = max(wl - rim, 0.0)
    res = model(best.x) - y
    return {"rim_row": round(float(rim), 2), "water_line_row": round(float(wl), 2),
            "dry_band_px": round(float(band_px), 2), "px_per_mm_on_wall": round(px_per_m / 1000, 3),
            "dry_band_mm": round(1000 * band_px / px_per_m, 1),
            "water_depth_mm": round(1000 * (c3.CUP_HEIGHT_M - band_px / px_per_m), 1),
            "dry_level_fraction_of_wet": round(float(dry / wet), 3), "blur_sigma_px": round(float(sig), 2),
            "fit_rms_counts": round(float(np.sqrt(np.mean(res ** 2))), 2), "band_held_mm": band_mm, "_rows": rows.tolist(), "_y": y.tolist(),
            "_model": model(best.x).tolist()}


def main() -> None:
    c3.OUT.mkdir(exist_ok=True)
    frames, t, _ = c3.load_thermal()
    mean = frames.mean(0)
    regions = c3.find_cups(mean)
    coef = c3.board_plane(mean[None], c3.board_mask(regions))
    board_img = c3.board_field(coef)[0]
    sil = {}
    for cup in c3.CUPS:
        r = regions[cup]
        bl = float(c3.plane_at(coef, np.mean(r["plateau_rows"]), np.mean(r["half_width_cols"]))[0])
        sil[cup] = g2.measure_silhouette(mean, r, bl, r["sign"] > 0)
    geo = lidar_geometry(regions)
    pose = fit_pose(sil, geo)
    pz = (pose["camera_height_above_table_m"], pose["camera_pitch_up_deg"], pose["camera_roll_deg"], pose["camera_yaw_left_deg"])
    for cup in c3.CUPS:  # the cup positions used from here on: LiDAR range, image bearing
        geo["cups"][cup]["axis_left_lidar_m"] = geo["cups"][cup]["axis_left_m"]
        geo["cups"][cup]["axis_left_m"] = pose["axis_left_m"][cup]
    pred = {cup: silhouette_table(geo["cups"][cup]["axis_forward_m"], geo["cups"][cup]["axis_left_m"], pz) for cup in c3.CUPS}
    water = {cup: water_level(mean, board_img, regions[cup], sil[cup], geo["cups"][cup]["axis_forward_m"], pose)
             for cup in ("cold", "hot")}
    band = 0.5 * (water["cold"]["dry_band_mm"] + water["hot"]["dry_band_mm"])
    water["warm_free"] = water_level(mean, board_img, regions["warm"], sil["warm"], geo["cups"]["warm"]["axis_forward_m"], pose)
    water["warm"] = water_level(mean, board_img, regions["warm"], sil["warm"], geo["cups"]["warm"]["axis_forward_m"], pose,
                                band_mm=band)
    water = {k: water[k] for k in ("cold", "warm", "hot", "warm_free")}
    rep_extra = {"water_line_below_rim_mm_used": round(band, 1)}
    keys = ("width_top10_px", "width_bottom10_px", "height_px", "top_row", "bottom_row", "centre_col")
    rep = {
        "frames": len(frames), "duration_s": round(float(t[-1] - t[0]), 3),
        "regions": regions, "board_plane_mean_frame": coef[0].round(3).tolist(),
        "lidar": {k: v for k, v in geo.items() if not k.startswith("_")},
        "image_pose": pose,
        "silhouettes": {cup: {"measured": {k: sil[cup][k] for k in keys}, "model_at_image_pose": {k: pred[cup][k] for k in keys},
                              "size_error_pct": {k: round(100 * (sil[cup][k] / pred[cup][k] - 1), 1) for k in keys[:3]}}
                        for cup in c3.CUPS},
        "water": {cup: {k: v for k, v in w.items() if not k.startswith("_")} for cup, w in water.items()},
        **rep_extra,
    }
    (c3.OUT / "scene_geometry.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("lidar", "image_pose", "silhouettes", "water")}, indent=1))
    plot(mean, sil, geo, pz, water)


def plot(mean, sil, geo, pz, water) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 7.5))
    ax = fig.add_subplot(2, 1, 1)
    lo, hi = np.percentile(mean, [0.5, 99.8])
    ax.imshow(mean[150:330], cmap="gray", vmin=lo, vmax=hi, extent=(0, 640, 330, 150))
    for cup in c3.CUPS:
        s = sil[cup]
        ax.plot(s["_lefts"], s["_rows"], "c-", lw=0.8)
        ax.plot(s["_rights"], s["_rows"], "c-", lw=0.8)
        g = geo["cups"][cup]
        u, v = project_table(frustum_table(g["axis_forward_m"], g["axis_left_m"], 360, 61), *pz)
        ax.plot(u, v, ",", color="r", alpha=0.25)
        ax.axhline(water[cup]["water_line_row"], xmin=(s["centre_col"] - 40) / 640, xmax=(s["centre_col"] + 40) / 640,
                   color="lime", lw=1)
    ax.set_title("Mean thermal frame: measured edges (cyan), LiDAR-placed 7/5/6 cm frustums at the fitted pose (red), "
                 "fitted water line (green)", fontsize=9)
    for k, cup in enumerate(c3.CUPS):
        a = fig.add_subplot(2, 3, 4 + k)
        w = water[cup]
        a.plot(w["_rows"], w["_y"], "k.", ms=3, label="profile (central third)")
        a.plot(w["_rows"], w["_model"], "r-", lw=1, label="board | dry band | wet wall, blurred")
        a.axvline(w["rim_row"], color="c", lw=0.8)
        a.axvline(w["water_line_row"], color="lime", lw=0.8)
        held = " (line held)" if w["band_held_mm"] else ""
        a.set_title(f"{cup}: dry band {w['dry_band_mm']} mm{held}, at {w['dry_level_fraction_of_wet']:.2f} of the wet "
                    f"contrast", fontsize=9)
        a.set_xlabel("row")
        a.set_ylabel("|counts - board|")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(c3.OUT / "scene_geometry.png", dpi=110)


if __name__ == "__main__":
    main()
