#!/usr/bin/env python
"""Task 4: the Cellplus's noise from the cup recordings -> proposed noise block for the sensor YAML.

    /home/jumasaet/isaac_sim/python.sh bench/cups_check/task4_noise.py      (after task1 and task2)

docs/physics-model.md §10.2 (NVESD 3-D noise), §10.3, §10.4 (bad pixels and their replacement), §9.4 (NETD).

What the data allow. 63 frames in four static sets (cup cooling or warming, offset steps from FFC between
some frames), captured seconds apart, not as video; the core's own `denoise: 40` was on. So:
  - FRAME QC: a frame whose difference to its neighbours has thousands of |e| > 10-count pixels is
    corrupt (hot_water_50 frames 14 and 15) and is dropped everywhere.
  - TEMPORAL components (tvh, tv, th, t) come from differences of frames of the same set, which cancel the
    scene and the fixed pattern. A plane + quadratic is fitted out of each difference first (housing drift
    shades the frame slowly; left in, it would read as row/column noise). The core's denoise has MEMORY:
    two frames 0.1 s apart differ by sigma 1.0-1.2 counts, 0.3 s apart by 1.4, and from ~1.5 s on the
    difference stops growing (1.6-1.7). So tvh, tv and th use only pairs >= 1.5 s apart -- independent
    frames, i.e. the noise one output frame carries -- and the curve sigma(dt) is reported as the filter's
    footprint. The frame bounce t uses pairs <= 0.35 s apart (between FFCs the level drifts ~1 count per
    frame), so it is the filtered value and a lower bound.
  - SPATIAL (fixed) components (vh, v, h) come from each set's mean frame (temporal noise / sqrt(N)), with
    the scene removed by a high-pass: running median along the profile for v and h, a 5x5 median for vh.
    Each estimator's own bias on white noise is measured by pushing synthetic white noise through it.
    The board has no texture at the 1-2 px scale (the lens blur is 14 px 10-90 %), but its edges and the
    window strip do: hence robust (MAD) sigmas.
  - Region: rows 10-180, columns 330-530: board and wall, above and to the right of every cup. Not over
    the cups: the hot cup sends up a plume of warm humid air that LWIR sees and that moves from frame to
    frame (look/hot95_diff01.png; it read as sigma_th = 1.3 counts in hot_water_95). Not the window strip
    on the left nor the curtain on the right, which can move.
  - FFC SEGMENTS: an FFC re-measures the offset table, so frames on either side of one differ by a new
    fixed pattern, not only by temporal noise (cold_water_50 frames 2 -> 3: +15 counts and sigma 2.2 instead
    of 1.6). Consecutive frames with a level step > 8 counts start a new segment; temporal pairs are taken
    inside one segment only.
  - BAD PIXELS: the camera replaces its defects internally. Looked for two footprints: a pixel equal to a
    neighbour in all 61 frames (a copy; two independent pixels agree in one frame ~14 % of the time, in
    61 never), and a pixel whose high-passed temporal residual has low variance and correlates > 0.8 with
    its 4-neighbour mean (an interpolation). Neither is found (only 4 equal pairs, in the last row's last
    5 columns: an edge artefact), i.e. the denoise after the replacement erases the footprint. Defects
    NOT replaced (fixed-pattern outliers > 8 sigma_vh in all four sets): none.
  - TAILS: the distribution of the temporal residual against a Gaussian of the same robust sigma, and
    pixels whose temporal sigma is > 4x the median (flickering / RTS candidates that are NOT replaced).

Writes out/task4_noise.json, out/task4_noise.png and bench/camera_check/cellplus_ctc007_lwir_cups.yaml
(the original is not touched).
"""
from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import median_filter

import cups_common as cc

REGION = (10, 180, 330, 530)
RNG = np.random.default_rng(7)


def detrend(d: np.ndarray) -> np.ndarray:
    """Remove a full quadratic surface (6 terms) from a 2-D array."""
    h, w = d.shape
    yy, xx = np.mgrid[0:h, 0:w]
    y, x = (yy - h / 2) / h, (xx - w / 2) / w
    a = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1).reshape(-1, 6)
    coef, *_ = np.linalg.lstsq(a, d.reshape(-1), rcond=None)
    return d - (a @ coef).reshape(h, w)


def mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float).ravel()
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def temporal_from_difference(d: np.ndarray) -> dict:
    """NVESD temporal components from one difference image d = F_i - F_j (variances are 2x a frame's)."""
    d = detrend(d)
    nr, nc = d.shape
    rv = d.mean(1)
    rh = d.mean(0)
    e = d - rv[:, None] - rh[None, :] + d.mean()
    var_tvh = np.var(e) / 2 / ((1 - 1 / nc) * (1 - 1 / nr))
    var_tv = max(np.var(rv) / 2 - var_tvh / nc, 0.0)
    var_th = max(np.var(rh) / 2 - var_tvh / nr, 0.0)
    return {"tvh": var_tvh, "tv": var_tv, "th": var_th, "e": e / np.sqrt(2), "mad_tvh": mad_sigma(e) / np.sqrt(2)}


def spatial_from_mean(m: np.ndarray) -> dict:
    """Fixed-pattern sigmas (raw, before bias and temporal corrections) from a mean frame region."""
    rv = m.mean(1)
    rh = m.mean(0)
    rv_hp = rv - median_filter(rv, size=9, mode="nearest")
    rh_hp = rh - median_filter(rh, size=9, mode="nearest")
    resid = m - median_filter(m, size=5, mode="nearest")
    resid = resid - resid.mean(1, keepdims=True) - resid.mean(0, keepdims=True) + resid.mean()
    return {"v": mad_sigma(rv_hp), "h": mad_sigma(rh_hp), "vh": mad_sigma(resid), "_vh_map": resid}


def estimator_bias(shape: tuple[int, int]) -> dict:
    """Ratio estimated / true for each spatial estimator, on pure white V, H and VH noise of sigma 1."""
    out = {"v": [], "h": [], "vh": []}
    nr, nc = shape
    for _ in range(20):
        v = RNG.normal(size=nr)[:, None] * np.ones((1, nc))
        h = np.ones((nr, 1)) * RNG.normal(size=nc)[None, :]
        vh = RNG.normal(size=shape)
        out["v"].append(spatial_from_mean(v)["v"])
        out["h"].append(spatial_from_mean(h)["h"])
        out["vh"].append(spatial_from_mean(vh)["vh"])
    return {k: float(np.mean(x)) for k, x in out.items()}


def replaced_pixels(sets_frames: list[np.ndarray]) -> dict:
    """Look for the footprint of the camera's own defect replacement (see the module docstring)."""
    a = np.concatenate(sets_frames).astype(np.int32)
    eq_h = (a[:, :, 1:] == a[:, :, :-1]).all(0)
    eq_v = (a[:, 1:, :] == a[:, :-1, :]).all(0)
    # Interpolation footprint, on each frame's 5x5 high-pass with each set's temporal mean removed.
    z = []
    for f in sets_frames:
        hp = np.stack([fr - median_filter(fr, size=5, mode="nearest") for fr in f])
        z.append(hp - hp.mean(0))
    z = np.concatenate(z)
    s = z.std(0)
    nb = (np.roll(z, 1, 1) + np.roll(z, -1, 1) + np.roll(z, 1, 2) + np.roll(z, -1, 2)) / 4
    corr = (z * nb).mean(0) / (s * nb.std(0) + 1e-12)
    inner = np.zeros_like(s, bool)
    inner[3:-3, 3:-3] = True
    interp = inner & (corr > 0.8) & (s < 0.7 * np.median(s[inner]))
    return {"exact_equal_pairs_horizontal": np.argwhere(eq_h).tolist(), "exact_equal_pairs_vertical": np.argwhere(eq_v).tolist(),
            "interpolation_footprint_pixels": int(interp.sum()),
            "neighbour_corr_percentiles_50_99_999": [round(float(v), 3) for v in np.percentile(corr[inner], [50, 99, 99.9])],
            "n_frames": int(len(a))}


def frame_qc(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(corrupt frame indices, FFC segment id per frame).

    Corrupt: a frame whose detrended difference to BOTH neighbours has > 1 % of the region beyond 10 counts
    (thousands of pixels; an FFC moves ~50). Segments: a new one starts where consecutive good frames step
    by > 8 counts in level.
    """
    y0, y1, x0, x1 = REGION
    reg = f[:, y0:y1, x0:x1]
    n = len(reg)
    lim = 0.01 * reg[0].size
    big = np.array([(np.abs(detrend(reg[i + 1] - reg[i])) > 10).sum() for i in range(n - 1)])
    bad = [i for i in range(n) if (i == 0 or big[i - 1] > lim) and (i == n - 1 or big[i] > lim)]
    good = [i for i in range(n) if i not in bad]
    seg = np.full(n, -1)
    sid = 0
    for k, i in enumerate(good):
        if k > 0 and abs(float(np.mean(reg[i] - reg[good[k - 1]]))) > 8:
            sid += 1
        seg[i] = sid
    return np.array(bad, int), seg


def main() -> None:
    task1 = json.loads((cc.OUT / "task1_calibration.json").read_text())
    sitf_300k = task1["sitf_counts_per_k"]["at_300k"]
    g = task1["gain_counts_per_w_m2_sr"]
    sitf_range = [round(sitf_300k * g["cold_point_only"] / g["joint_geometric_mean"], 2),
                  round(sitf_300k * g["hot_point_only"] / g["joint_geometric_mean"], 2)]
    y0, y1, x0, x1 = REGION
    nr, nc = y1 - y0, x1 - x0
    per_set, good_frames, e_pool, std_maps, means = {}, [], [], [], {}
    t_bounce, dt_curve, dropped, segments = [], [], {}, {}
    for case in cc.CASES:
        f, t = cc.load_case(case)
        bad, seg = frame_qc(f)
        dropped[case] = bad.tolist()
        segments[case] = seg.tolist()
        f, t, seg = np.delete(f, bad, 0), np.delete(t, bad, 0), np.delete(seg, bad, 0)
        good_frames.append(f)
        reg = f[:, y0:y1, x0:x1]
        comps = {"tvh": [], "tv": [], "th": []}
        for i in range(len(f)):
            for j in range(i + 1, len(f)):
                dt = t[j] - t[i]
                if seg[i] != seg[j]:
                    continue  # an FFC between them
                d = reg[j] - reg[i]
                if np.abs(d).max() == 0:
                    continue  # the capture tool wrote the same frame twice
                if dt >= 1.5 or dt <= 0.35 or j == i + 1:
                    r = temporal_from_difference(d)
                    dt_curve.append((dt, np.sqrt(r["tvh"])))
                    if dt >= 1.5:
                        for k in comps:
                            comps[k].append(r[k])
                        if j == i + 1 or len(e_pool) < 400:
                            e_pool.append(r["e"].astype(np.float32))
                    if dt <= 0.35:
                        t_bounce.append(float(d.mean()))
        # Mean and per-pixel sigma of the largest FFC segment (one fixed pattern).
        big_seg = np.bincount(seg).argmax()
        keep = seg == big_seg
        z = np.stack([detrend(fr - np.median(fr)) for fr in reg[keep]])
        std_maps.append(z.std(0, ddof=1))
        # Effective number of independent frames in the mean: groups separated by > 1.5 s.
        n_eff = int(1 + np.sum(np.diff(t[keep]) > 1.5))
        means[case] = (z.mean(0), n_eff)
        per_set[case] = {"frames_used": len(f), "ffc_segments": int(seg.max() + 1),
                         "frames_in_mean": int(keep.sum()), "independent_groups_in_mean": n_eff, "pairs_ge_1p5s": len(comps["tvh"]),
                         "sigma_tvh": round(float(np.sqrt(np.median(comps["tvh"]))), 3),
                         "sigma_tv": round(float(np.sqrt(np.median(comps["tv"]))), 3),
                         "sigma_th": round(float(np.sqrt(np.median(comps["th"]))), 3)}
    s_tvh = float(np.sqrt(np.mean([per_set[c]["sigma_tvh"] ** 2 for c in cc.CASES])))
    s_tv = float(np.sqrt(np.mean([per_set[c]["sigma_tv"] ** 2 for c in cc.CASES])))
    s_th = float(np.sqrt(np.mean([per_set[c]["sigma_th"] ** 2 for c in cc.CASES])))
    s_t = float(np.std(t_bounce) / np.sqrt(2)) if len(t_bounce) > 2 else float("nan")
    dtc = np.array(dt_curve)
    curve = {}
    for lo, hi in ((0.0, 0.2), (0.2, 0.5), (0.5, 1.5), (1.5, 5.0), (5.0, 60.0)):
        m = (dtc[:, 0] > lo) & (dtc[:, 0] <= hi)
        if m.any():
            sg = float(np.median(dtc[m, 1]))
            curve[f"{lo:g}-{hi:g}s"] = {"pairs": int(m.sum()), "sigma_counts": round(sg, 3),
                                         "frame_correlation": round(1 - (sg / s_tvh) ** 2, 3)}

    e_all = np.stack(e_pool)
    corr_h = float(np.mean([np.corrcoef(e[:, :-1].ravel(), e[:, 1:].ravel())[0, 1] for e in e_pool]))
    corr_v = float(np.mean([np.corrcoef(e[:-1].ravel(), e[1:].ravel())[0, 1] for e in e_pool]))
    corr_h2 = float(np.mean([np.corrcoef(e[:, :-2].ravel(), e[:, 2:].ravel())[0, 1] for e in e_pool]))

    # Tails of the temporal residual (independent pairs only).
    x = e_all.ravel().astype(np.float64)
    sig = mad_sigma(x)
    z = (x - np.median(x)) / sig
    from math import erfc, sqrt

    tails = {"robust_sigma": round(sig, 3), "plain_std": round(float(x.std()), 3),
             "excess_kurtosis": round(float(np.mean(z ** 4) - 3), 3), "n_samples": int(x.size)}
    for k in (3, 4, 5, 6, 8):
        tails[f"frac_beyond_{k}sigma"] = float(np.mean(np.abs(z) > k))
        tails[f"gaussian_beyond_{k}sigma"] = float(erfc(k / sqrt(2)))
    smap = np.median(np.stack(std_maps), 0)
    med = float(np.median(smap))
    for k in (1.5, 2, 3, 4):
        tails[f"fraction_pixels_temporal_sigma_gt_{k:g}x_median"] = round(float((smap > k * med).mean()), 5)
    # How much of the tail the noisy-pixel population carries: tails with those pixels removed.
    quiet = (smap <= 1.5 * med)
    xq = np.stack([e[quiet] for e in e_pool]).ravel().astype(np.float64)
    zq = (xq - np.median(xq)) / mad_sigma(xq)
    tails["quiet_pixels_only"] = {f"frac_beyond_{k}sigma": float(np.mean(np.abs(zq) > k)) for k in (3, 4, 5, 6)}
    tails["quiet_pixels_only"]["excess_kurtosis"] = round(float(np.mean(zq ** 4) - 3), 3)

    # Fixed pattern per set, bias-corrected, the temporal residue (sigma^2 / N_eff) removed.
    bias = estimator_bias((nr, nc))
    spatial, vh_maps = {}, {}
    for case, (m, n) in means.items():
        s = spatial_from_mean(m)
        vh_maps[case] = s["_vh_map"]
        v = np.sqrt(max((s["v"] / bias["v"]) ** 2 - (s_tv ** 2 + s_tvh ** 2 / nc) / n, 0))
        h = np.sqrt(max((s["h"] / bias["h"]) ** 2 - (s_th ** 2 + s_tvh ** 2 / nr) / n, 0))
        vh = np.sqrt(max((s["vh"] / bias["vh"]) ** 2 - s_tvh ** 2 / n, 0))
        spatial[case] = {"raw_v": round(s["v"], 3), "raw_h": round(s["h"], 3), "raw_vh": round(s["vh"], 3),
                         "sigma_v": round(float(v), 3), "sigma_h": round(float(h), 3), "sigma_vh": round(float(vh), 3)}
    s_v = float(np.sqrt(np.mean([spatial[c]["sigma_v"] ** 2 for c in cc.CASES])))
    s_h = float(np.sqrt(np.mean([spatial[c]["sigma_h"] ** 2 for c in cc.CASES])))
    s_vh = float(np.sqrt(np.mean([spatial[c]["sigma_vh"] ** 2 for c in cc.CASES])))
    keys = list(vh_maps)
    vh_corr = {f"{a}~{b}": round(float(np.corrcoef(vh_maps[a].ravel(), vh_maps[b].ravel())[0, 1]), 3)
               for i, a in enumerate(keys) for b in keys[i + 1:]}
    # The part of VH that survives the FFCs between sets: corr * total variance.
    stable_vh = float(np.sqrt(max(np.mean(list(vh_corr.values())), 0) * np.mean([spatial[c]["raw_vh"] ** 2 for c in cc.CASES]))
                      / bias["vh"])

    rep = replaced_pixels(good_frames)
    stack = np.stack([vh_maps[c] for c in cc.CASES])
    late = (np.abs(stack) > 8 * s_vh).all(0) & ((np.sign(stack) == np.sign(stack[0])).all(0))

    ratios = {"tvh": 1.0, "vh": s_vh / s_tvh, "h": s_h / s_tvh, "v": s_v / s_tvh, "tv": s_tv / s_tvh,
              "th": s_th / s_tvh, "t": s_t / s_tvh}
    netd_mk = 1e3 * s_tvh / sitf_300k
    report = {
        "region_rows_cols": REGION, "frames_dropped_corrupt": dropped, "ffc_segment_per_frame": segments,
        "temporal_counts": {"per_set": per_set, "sigma_tvh": round(s_tvh, 3), "sigma_tv": round(s_tv, 3),
                            "sigma_th": round(s_th, 3), "sigma_t_pairs_le_0p35s": round(s_t, 3), "n_pairs_t": len(t_bounce)},
        "denoise_memory_sigma_vs_dt": curve,
        "temporal_residual_neighbour_correlation": {"horizontal_1px": round(corr_h, 3), "horizontal_2px": round(corr_h2, 3),
                                                    "vertical_1px": round(corr_v, 3)},
        "tails": tails,
        "spatial_counts": {"per_set": spatial, "sigma_v": round(s_v, 3), "sigma_h": round(s_h, 3),
                           "sigma_vh_within_set": round(s_vh, 3), "sigma_vh_stable_across_sets": round(stable_vh, 3),
                           "estimator_bias_on_white_noise": {k: round(v, 3) for k, v in bias.items()},
                           "vh_pattern_correlation_between_sets": vh_corr},
        "replaced_pixels_search": rep,
        "unreplaced_outliers_in_region": int(late.sum()), "region_pixels": int(nr * nc),
        "sitf_counts_per_k_300k": sitf_300k, "sitf_range_from_task1": sitf_range,
        "netd_mk_measured_output": round(netd_mk, 1),
        "netd_mk_range": [round(1e3 * s_tvh / sitf_range[1], 1), round(1e3 * s_tvh / sitf_range[0], 1)],
        "ratios_3d_proposed": {k: round(v, 3) for k, v in ratios.items()},
    }
    (cc.OUT / "task4_noise.json").write_text(json.dumps(report, indent=1))
    write_yaml(report)
    plot(e_all, z, smap, dtc, vh_maps, report)
    print(json.dumps({k: report[k] for k in report if k not in ("tails",)}, indent=1))
    print(json.dumps(tails, indent=1))


def write_yaml(r: dict) -> None:
    """Original draft + measured noise, the measured geometry and the measured radiometric anchors."""
    src = cc.SENSOR_YAML.read_text(encoding="utf-8")
    rat = r["ratios_3d_proposed"]
    tc = r["temporal_counts"]
    sc = r["spatial_counts"]
    tl = r["tails"]
    mem = r["denoise_memory_sigma_vs_dt"]
    corr = r["temporal_residual_neighbour_correlation"]
    old_noise_start = src.index("  noise:\n")
    old_noise_end = src.index("  nuc:\n")
    noise = f"""  noise:
    netd_mk_at_300k: 50.0                        # [DS] "<= 50 mK". NOT the anchor any more: the measured
                                                 # calibration.radiometric.netd_k below replaces it (ADR 0025)
    ratios_3d:
      # [DATA cups] bench/cups_check/task4_noise.py, the 63 frames of thermal_specs/cali_board_new, rows
      # {REGION[0]}-{REGION[1]} x cols {REGION[2]}-{REGION[3]} (board and wall, clear of the cups and of the hot cup's
      # plume), inside FFC segments, AFTER the core's denoise: 40. In counts:
      #   temporal (pairs >= 1.5 s apart = independent frames): tvh {tc['sigma_tvh']}, tv {tc['sigma_tv']}, th {tc['sigma_th']}
      #   frame bounce t {tc['sigma_t_pairs_le_0p35s']} (pairs <= 0.35 s; filtered, a lower bound)
      #   fixed (set means, scene high-passed, bias-corrected): vh {sc['sigma_vh_within_set']} within one FFC period
      #   ({sc['sigma_vh_stable_across_sets']} of it survives the FFCs between sets), v {sc['sigma_v']}, h {sc['sigma_h']}.
      # Fixed rows ~2x fixed columns (v ~ 2 h), as in the lab set: the readout is rotated 90 deg.
      # NOT representable in this schema (PENDIENTE, schema gap):
      #   - the denoise has memory: frames 0.1 s apart are correlated r = {mem.get('0-0.2s', {}).get('frame_correlation')}, 0.2-0.5 s r = {mem.get('0.2-0.5s', {}).get('frame_correlation')};
      #     the simulator's tvh is white in time;
      #   - and in space: neighbour r = {corr['horizontal_1px']} (h), {corr['vertical_1px']} (v); the simulator's tvh is white in space;
      #   - heavy tails: beyond 5 sigma {tl['frac_beyond_5sigma']:.2e} of samples (Gaussian {tl['gaussian_beyond_5sigma']:.1e}),
      #     {100 * tl['fraction_pixels_temporal_sigma_gt_2x_median']:.2f} % of pixels have > 2x the median temporal sigma.
      tvh: 1.00
      vh:  {rat['vh']:.3f}
      h:   {rat['h']:.3f}
      v:   {rat['v']:.3f}
      tv:  {rat['tv']:.3f}
      th:  {rat['th']:.3f}
      t:   {rat['t']:.3f}
    fpn_drift_tau_s: 120.0                       # PENDIENTE (Boson). FFCs reset part of VH between sets (r = 0.3-0.4
                                                 # between sets minutes apart); no FFC-free minute in the data
    # [DATA cups] The camera replaces its defects, and its denoise erases the footprint: no pixel copies a
    # neighbour, none shows an interpolation's low variance + high neighbour correlation, and no fixed-pattern
    # outlier (> 8 sigma_vh in all four sets) is left unreplaced. So the replaced population cannot be
    # counted here: the datasheet's 99.5 % operability stays, and nothing is visible after replacement.
    bad_pixel_fraction: 0.005                    # [DS]
    bad_pixel_cluster_lambda: 1.6                # PENDIENTE (Boson)
    bad_pixel_late_fraction: 0.0                 # [DATA cups] none of the ~1500 defects 0.5 % implies is visible;
                                                 # 0 seen -> < 0.2 % of them at 95 % confidence

"""
    out = src[:old_noise_start] + noise + src[old_noise_end:]
    out = out.replace(
        "# DRAFT for bench/camera_check -- not part of configs/sensors/.",
        "# DRAFT for bench/camera_check -- not part of configs/sensors/.\n"
        "# cellplus_ctc007_lwir_cups.yaml = cellplus_ctc007_lwir.yaml + what bench/cups_check measured on the\n"
        "# cup recordings (2026-10-26): the noise block, calibration.geometric, calibration.radiometric.\n"
        "# GENERATED by bench/cups_check/task4_noise.py; edit that, not this. Tag [DATA cups].",
    )
    out = out.replace("  name: cellplus_ctc007_lwir\n", "  name: cellplus_ctc007_lwir_cups\n")
    netd_k = r["netd_mk_measured_output"] * 1e-3
    calib = f"""
  # [DATA cups] The lens is not the datasheet's 22.2 mm: the RGB-thermal calibration (16 views, 0.40 px
  # reprojection, thermal_specs/cali_board_new/RGBT_calibration_result_matrix.txt) gives fx = 626.9 px,
  # i.e. 15.0 mm at 24 um (HFOV 54 deg, not 38), and the cups' sizes agree with it within -7..+8 %
  # (bench/cups_check/task2_geometry.py) where the 22.2 mm pinhole is off by -28..-39 %. Present, this
  # block replaces optics.focal_length_mm and optics.distortion everywhere the projection is used.
  calibration:
    geometric:
      fx_px: 626.94618350
      fy_px: 620.62145336
      cx_px: 332.60285388
      cy_px: 222.85948387
      k1: -0.2738138890
      k2: -0.5834928420
      p1: 0.0088117352
      p2: 0.0012314182
      k3: 0.7123230428
    radiometric:
      # sigma_tvh {tc['sigma_tvh']} counts / SITF {r['sitf_counts_per_k_300k']} counts/K at 300 K (bench/cups_check/task1,
      # radiance-linear, board-referenced; SITF range {r['sitf_range_from_task1']} from the disagreement of the
      # 11 / 22 / 66 C readings) = {r['netd_mk_measured_output']} mK (range {r['netd_mk_range']}). OUTPUT noise, after denoise: 40.
      netd_k: {netd_k:.4f}
      sitf_dn_per_k: {r['sitf_counts_per_k_300k']}
"""
    out = out.replace("\n  extensions:\n", calib + "\n  extensions:\n")
    dst = cc.SENSOR_YAML.with_name("cellplus_ctc007_lwir_cups.yaml")
    dst.write_text(out, encoding="utf-8")
    from irsim.config.loader import load_sensor_config

    spec = load_sensor_config(dst).sensor  # the loader must accept it
    assert spec.calibration is not None and spec.calibration.geometric is not None
    print("wrote", dst, "| loader OK:", spec.name, spec.noise.ratios_3d)


def plot(e_all, z, smap, dtc, vh_maps, r) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    bins = np.linspace(-12, 12, 241)
    hst, _ = np.histogram(z, bins=bins, density=True)
    c = 0.5 * (bins[1:] + bins[:-1])
    ax[0, 0].semilogy(c, hst, label="temporal residual (robust z)")
    ax[0, 0].semilogy(c, np.exp(-c ** 2 / 2) / np.sqrt(2 * np.pi), "--", label="Gaussian")
    ax[0, 0].set_ylim(1e-8, 1)
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title(f"Temporal tails (independent pairs), excess kurtosis {r['tails']['excess_kurtosis']}", fontsize=9)
    ax[0, 1].semilogx(dtc[:, 0], dtc[:, 1], ".", ms=3, alpha=0.5)
    ax[0, 1].axhline(r["temporal_counts"]["sigma_tvh"], color="k", lw=0.8, ls="--", label="sigma_tvh (pairs >= 1.5 s)")
    ax[0, 1].set_xlabel("time between the two frames (s)")
    ax[0, 1].set_ylabel("sigma_tvh of the pair (counts)")
    ax[0, 1].set_title("The core's denoise has memory", fontsize=9)
    ax[0, 1].legend(fontsize=8)
    im = ax[0, 2].imshow(smap, cmap="magma", vmin=0, vmax=3 * np.median(smap))
    ax[0, 2].set_title("per-pixel temporal sigma (counts, median of sets)", fontsize=9)
    plt.colorbar(im, ax=ax[0, 2], fraction=0.03)
    vh = vh_maps["hot_water_50"]
    s_vh = r["spatial_counts"]["sigma_vh_within_set"]
    im = ax[1, 0].imshow(vh, cmap="gray", vmin=-4 * s_vh, vmax=4 * s_vh)
    ax[1, 0].set_title("fixed pattern (hot_water_50 mean, high-passed)", fontsize=9)
    plt.colorbar(im, ax=ax[1, 0], fraction=0.03)
    ax[1, 1].plot(vh.mean(1), label="row means")
    ax[1, 1].plot(np.linspace(0, len(vh) - 1, vh.shape[1]), vh.mean(0), alpha=0.7, label="column means (x rescaled)")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("row / column structure of the fixed pattern (counts)", fontsize=9)
    im = ax[1, 2].imshow(e_all[0], cmap="gray", vmin=-6, vmax=6)
    ax[1, 2].set_title("one temporal difference / sqrt2 (counts)", fontsize=9)
    plt.colorbar(im, ax=ax[1, 2], fraction=0.03)
    fig.tight_layout()
    fig.savefig(cc.OUT / "task4_noise.png", dpi=110)


if __name__ == "__main__":
    main()
