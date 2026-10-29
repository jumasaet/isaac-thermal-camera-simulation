#!/usr/bin/env python
"""Task 6: the Cellplus's 3-D noise at `denoise: 80`, from the synchronised three-cup recording.

    /home/jumasaet/isaac_sim/python.sh bench/cups3_check/task6_noise.py      (after task1)

docs/physics-model.md §10.2 (NVESD 3-D noise), §9.4 (NETD). Same estimators as bench/cups_check/task4_noise.py
(imported from there, not copied), so the numbers compare one to one with the `denoise: 40` values in
bench/camera_check/cellplus_ctc006_lwir_cups.yaml. What differs is the data:

  - 94 frames of thermal_specs/three_cups_experiments/sync_cups/thermal over 28.8 s, irregularly spaced
    (13 ms to 5.6 s: one frame per synchronised LiDAR set). Two pairs are the SAME camera frame written twice
    (sets 80/81 and 88/89 share a thermal_stamp in sync_index.csv: two LiDAR packets 14-16 ms apart paired
    with one camera frame); the second of each is dropped here, leaving 92.
  - One recording, no FFC inside it (no level step > 8 counts), so one fixed pattern: the spatial components
    are the fixed pattern of one FFC period ("within set"); nothing here says how much survives an FFC.
  - REGION rows 10-150 x cols 160-420: the board above the cups. Not above the hot cup (cols >= 440: its
    plume reads as 4-7 counts of temporal sigma), not the left edge (the room past the board's edge), and
    50+ rows clear of every rim.
  - TEMPORAL components come from pairs 1.5-5 s apart, not from every pair >= 1.5 s as in task4. This recording is
    continuous, and the pair sigma keeps growing with separation (1.25 counts at 1.5-3 s, 1.41 at 24-30 s, the
    same at the start, middle and end of the recording): that growth is the fixed pattern drifting, not temporal
    noise, and the simulator carries it separately (`fpn_drift_tau_s`). 1.5 s is past the denoise's memory; 5 s is
    before the drift weighs. The task4-style value (every pair >= 1.5 s) is reported too, for a like-for-like
    comparison with denoise 40.
  - The temporal residue left in the mean frame is measured, not modelled: the spatial estimators run on half the
    difference of the two halves' means. task4's "groups > 1.5 s apart" count is 3 here (the frames are ~0.3 s
    apart), and would subtract ~6x too much.
  - SITF is this recording's own calibration (task1: 48.25 counts/K at 300 K), so NETD is in the same units
    as the warm-cup test.

Control (`sim_control`): the simulated camera's own temporal sigma in counts, uniform 23.9 C scene through the
same chain as task2_render.py, with the old and the new YAML -- what the measured numbers turn into.

Writes out/task6_noise.json and out/task6_noise.png. The proposed YAML is hand-made from the JSON
(bench/camera_check/cellplus_ctc006_lwir_cups3.yaml); this script does not write any YAML.
"""
from __future__ import annotations

import json
import pathlib
import sys
from math import erfc, sqrt

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cups_check"))
import cups3_common as c3  # noqa: E402
from task4_noise import (  # noqa: E402
    detrend,
    estimator_bias,
    mad_sigma,
    spatial_from_mean,
    temporal_from_difference,
)

REGION = (10, 150, 160, 420)
# The denoise: 40 numbers this compares with (bench/camera_check/cellplus_ctc006_lwir_cups.yaml, noise block).
OLD_COUNTS = {"tvh": 1.391, "tv": 0.299, "th": 0.39, "t": 0.101, "vh": 1.429, "v": 0.575, "h": 0.28}
OLD_SITF, OLD_NETD_MK = 52.91, 26.3
OLD_MEMORY_R = {"0-0.2s": 0.485, "0.2-0.5s": 0.28}
OLD_NEIGHBOUR_R = {"horizontal_1px": 0.374, "vertical_1px": 0.297}


def main() -> None:
    task1 = json.loads((c3.OUT / "task1_calibration.json").read_text())
    sitf = float(task1["calibration"]["sitf_counts_per_k_at_300k"])
    f, t, names = c3.load_thermal()
    dup = [i for i in range(1, len(f)) if np.array_equal(f[i], f[i - 1])]
    keep = np.setdiff1d(np.arange(len(f)), dup)
    f, t = f[keep], t[keep]
    y0, y1, x0, x1 = REGION
    reg = f[:, y0:y1, x0:x1]
    nr, nc = reg.shape[1:]
    level = reg.mean((1, 2))
    steps = np.abs(np.diff(level))
    assert steps.max() <= 8, f"an FFC step inside the recording ({steps.max():.1f} counts): segment it first"

    comps = {"tvh": [], "tv": [], "th": []}
    like_task4 = []
    dt_curve, bounce, e_pool = [], [], []
    for i in range(len(f)):
        for j in range(i + 1, len(f)):
            dt = t[j] - t[i]
            if not (dt >= 1.5 or dt <= 0.35 or j == i + 1):
                continue
            r = temporal_from_difference(reg[j] - reg[i])
            dt_curve.append((dt, np.sqrt(r["tvh"])))
            if dt >= 1.5:
                like_task4.append(r["tvh"])
            if 1.5 <= dt <= 5.0:
                for k in comps:
                    comps[k].append(r[k])
                if j == i + 1 or len(e_pool) < 400:
                    e_pool.append(r["e"].astype(np.float32))
            if dt <= 0.35:
                bounce.append(float((reg[j] - reg[i]).mean()))
    s_tvh = float(np.sqrt(np.median(comps["tvh"])))
    s_tv = float(np.sqrt(np.median(comps["tv"])))
    s_th = float(np.sqrt(np.median(comps["th"])))
    s_t = float(np.std(bounce) / np.sqrt(2))
    dtc = np.array(dt_curve)
    curve = {}
    for lo, hi in ((0.0, 0.2), (0.2, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, 12.0), (12.0, 18.0),
                   (18.0, 24.0), (24.0, 30.0)):
        m = (dtc[:, 0] > lo) & (dtc[:, 0] <= hi)
        if m.any():
            sg = float(np.median(dtc[m, 1]))
            curve[f"{lo:g}-{hi:g}s"] = {"pairs": int(m.sum()), "sigma_counts": round(sg, 3),
                                         "frame_correlation": round(1 - (sg / s_tvh) ** 2, 3)}
    corr_h = float(np.mean([np.corrcoef(e[:, :-1].ravel(), e[:, 1:].ravel())[0, 1] for e in e_pool]))
    corr_v = float(np.mean([np.corrcoef(e[:-1].ravel(), e[1:].ravel())[0, 1] for e in e_pool]))

    x = np.stack(e_pool).ravel().astype(np.float64)
    sig = mad_sigma(x)
    z = (x - np.median(x)) / sig
    tails = {"robust_sigma": round(sig, 3), "plain_std": round(float(x.std()), 3),
             "excess_kurtosis": round(float(np.mean(z ** 4) - 3), 3)}
    for k in (3, 4, 5, 6):
        tails[f"frac_beyond_{k}sigma"] = float(np.mean(np.abs(z) > k))
        tails[f"gaussian_beyond_{k}sigma"] = float(erfc(k / sqrt(2)))
    zt = np.stack([detrend(fr - np.median(fr)) for fr in reg])
    smap = zt.std(0, ddof=1)
    med = float(np.median(smap))
    tails["fraction_pixels_temporal_sigma_gt_2x_median"] = round(float((smap > 2 * med).mean()), 5)

    # Fixed pattern: the mean of all frames, bias-corrected, minus the temporal residue of that mean, measured as
    # the same estimator on (mean of first half - mean of second half) / 2.
    n = len(zt)
    mean = zt.mean(0)
    half_a, half_b = zt[: n // 2].mean(0), zt[n // 2:].mean(0)
    s = spatial_from_mean(mean)
    sd = spatial_from_mean((half_a - half_b) / 2)
    bias = estimator_bias((nr, nc))
    fixed = {k: float(np.sqrt(max((s[k] / bias[k]) ** 2 - (sd[k] / bias[k]) ** 2, 0))) for k in ("vh", "v", "h")}
    s_vh, s_v, s_h = fixed["vh"], fixed["v"], fixed["h"]
    # How stable the fixed pattern is over the recording: the two halves' high-passed means, ~14 s apart.
    vh_halves_corr = float(np.corrcoef(spatial_from_mean(half_a)["_vh_map"].ravel(),
                                       spatial_from_mean(half_b)["_vh_map"].ravel())[0, 1])
    # Drift of the pattern: growth of the per-frame pair variance with separation, beyond the memory.
    lo = (dtc[:, 0] >= 1.5) & (dtc[:, 0] <= 30.0)
    slope, icpt = np.polyfit(dtc[lo, 0], dtc[lo, 1] ** 2, 1)

    new = {"tvh": s_tvh, "tv": s_tv, "th": s_th, "t": s_t, "vh": s_vh, "v": s_v, "h": s_h}
    ratios = {k: v / s_tvh for k, v in new.items()}
    old_ratios = {k: v / OLD_COUNTS["tvh"] for k, v in OLD_COUNTS.items()}
    netd_mk = 1e3 * s_tvh / sitf
    report = {
        "data": str(c3.DATA / "thermal"), "denoise": 80, "region_rows_cols": REGION, "region_pixels": int(nr * nc),
        "frames_in_folder": int(len(keep) + len(dup)), "duplicate_frames_dropped": [names[i] for i in dup],
        "frames_used": int(len(f)), "span_s": round(float(t[-1] - t[0]), 2), "max_level_step_counts": round(float(steps.max()), 2),
        "pairs_1p5_to_5s": len(comps["tvh"]), "pairs_le_0p35s": len(bounce),
        "sigma_tvh_every_pair_ge_1p5s_like_task4": round(float(np.sqrt(np.median(like_task4))), 3),
        "pattern_drift": {"pair_variance_slope_counts2_per_s": round(float(slope), 4),
                          "pair_variance_intercept_counts2": round(float(icpt), 3),
                          "vh_map_correlation_between_halves": round(vh_halves_corr, 3),
                          "tau_s_if_the_whole_vh_drifts": round(float(s_vh ** 2 / slope), 0)},
        "sigma_counts": {k: round(v, 3) for k, v in new.items()},
        "sigma_counts_denoise40": OLD_COUNTS,
        "new_over_old": {k: round(new[k] / OLD_COUNTS[k], 3) for k in new},
        "ratios_3d_proposed": {k: round(v, 3) for k, v in ratios.items()},
        "ratios_3d_denoise40": {k: round(v, 3) for k, v in old_ratios.items()},
        "denoise_memory_sigma_vs_dt": curve, "denoise40_memory_r": OLD_MEMORY_R,
        "temporal_residual_neighbour_correlation": {"horizontal_1px": round(corr_h, 3), "vertical_1px": round(corr_v, 3)},
        "denoise40_neighbour_r": OLD_NEIGHBOUR_R,
        "tails": tails,
        "estimator_bias_on_white_noise": {k: round(v, 3) for k, v in bias.items()},
        "raw_spatial_counts": {k: round(float(s[k]), 3) for k in ("v", "h", "vh")},
        "temporal_residue_in_mean_counts": {k: round(float(sd[k] / bias[k]), 3) for k in ("v", "h", "vh")},
        "sitf_counts_per_k_300k": sitf, "netd_mk_output": round(netd_mk, 1),
        "denoise40_sitf_and_netd_mk": [OLD_SITF, OLD_NETD_MK],
        "netd_mk_output_at_denoise40_sitf": round(1e3 * s_tvh / OLD_SITF, 1),
    }
    report["sim_control_temporal_sigma_counts"] = sim_control()
    (c3.OUT / "task6_noise.json").write_text(json.dumps(report, indent=1))
    plot(dtc, smap, mean, s_tvh, report)
    print(json.dumps(report, indent=1))


def sim_control() -> dict:
    """Temporal sigma (counts, consecutive-frame differences / sqrt 2) of the simulated camera on a uniform scene."""
    import yaml

    sys.path.insert(0, str(c3.REPO / "src"))
    from irsim.config.sensor import SensorConfig
    from irsim.materials.table import MaterialTable
    from irsim.pipeline import PipelineConfig, PipelineState, attach_sensor_chain, run_frame
    from irsim.radiometry.lut_files import load_band_lut_for_config

    t_air = c3.T_AIR_C + 273.15
    out = {}
    for name in ("cellplus_ctc006_lwir_cups.yaml", "cellplus_ctc006_lwir_cups3.yaml"):
        d = yaml.safe_load((c3.REPO / "bench" / "camera_check" / name).read_text())
        d["sensor"]["optics"]["supersample_factor"] = 1  # white noise is added on the native grid
        sensor = SensorConfig.model_validate(d)
        lut = load_band_lut_for_config(sensor, c3.REPO / "bench" / "camera_preview" / "lut")
        cfg = PipelineConfig.from_sensor(sensor, MaterialTable.constant(1.0), lut=lut, noise_enabled=True,
                                         flat_field_enabled=True, sensor_seed=7)
        cfg = attach_sensor_chain(cfg, ambient_provider=lambda t: t_air)
        h, w = sensor.sensor.fpa_shape
        planes = {"temperature_k": np.full((h, w), t_air, np.float32), "material_id": np.ones((h, w), np.int32),
                  "distance_m": np.full((h, w), 1.0, np.float32)}
        st, frames = PipelineState(), []
        for i in range(8):
            st.t_s = i / 30.0
            frames.append(np.asarray(run_frame(planes, cfg, st).dn16, np.float64)[100:300, 200:450])
        out[name] = round(float(np.median([np.std(b - a) / np.sqrt(2) for a, b in zip(frames, frames[1:], strict=False)])), 3)
    return out


def plot(dtc, smap, mean, s_tvh, r) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].semilogx(dtc[:, 0], dtc[:, 1], ".", ms=3, alpha=0.4)
    ax[0].axhline(s_tvh, color="k", lw=0.8, ls="--", label=f"sigma_tvh denoise 80 = {s_tvh:.3f}")
    ax[0].axhline(OLD_COUNTS["tvh"], color="C3", lw=0.8, ls=":", label=f"sigma_tvh denoise 40 = {OLD_COUNTS['tvh']}")
    ax[0].set_xlabel("time between the two frames (s)")
    ax[0].set_ylabel("sigma_tvh of the pair (counts)")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Temporal noise vs frame separation (sync_cups)", fontsize=9)
    im = ax[1].imshow(smap, cmap="magma", vmin=0, vmax=3 * np.median(smap))
    ax[1].set_title("per-pixel temporal sigma, region (counts)", fontsize=9)
    plt.colorbar(im, ax=ax[1], fraction=0.03)
    hp = mean - np.median(mean)
    im = ax[2].imshow(hp, cmap="gray", vmin=-4 * r["sigma_counts"]["vh"], vmax=4 * r["sigma_counts"]["vh"])
    ax[2].set_title("mean of 92 frames, detrended (fixed pattern + scene)", fontsize=9)
    plt.colorbar(im, ax=ax[2], fraction=0.03)
    fig.tight_layout()
    fig.savefig(c3.OUT / "task6_noise.png", dpi=110)


if __name__ == "__main__":
    main()
