#!/usr/bin/env python
"""Task 3 (comparison) and the simulator half of task 2: real calibrated cups against the renders.

    /home/jumasaet/isaac_sim/python.sh bench/cups_check/task3_compare.py     (after task1, task2, the renders)

Real: the blackbody apparent temperature of the cup's water region, T_app = L_B^-1(L_B(T_air) + dc / G), with
the task-1 calibration (G fitted on the 50 cm sets only), on the frame the temperature reading refers to
(first frame of hot_50, cold_50, cold_95; last frame of hot_95 -- the time-order assumption).
Simulated: scene_apparent_t (before the camera: emission + reflection + atmosphere, no PSF, no noise) over
the water region of the render, eps = 0.93. It is also L_B^-1(eps L_B(T_s) + (1 - eps) L_B(T_air)), which
the script checks against the render and then uses to carry eps through the Monte Carlo: the same eps goes
into the real side (through G) and the simulated side in each draw.

At 50 cm the comparison is NOT independent: those sets made G. What is left there is the disagreement of
the three readings (11 / 22 / 66 C) that a single G cannot absorb. At 95 cm it is the test.

Writes out/task3_comparison.json and out/task3_comparison.png.
"""
from __future__ import annotations

import json

import numpy as np

import cups_common as cc

RNG = np.random.default_rng(20261027)
N = 20000
EPS_RANGE = (0.90, 0.96)
T_SET = {"hot_water_50": 66.0, "hot_water_95": 58.0, "cold_water_50": 11.0, "cold_water_95": 11.0}
REF_FRAME = {"hot_water_50": 0, "hot_water_95": -1, "cold_water_50": 0, "cold_water_95": 0}


def main() -> None:
    t1 = json.loads((cc.OUT / "task1_calibration.json").read_text())
    t2 = json.loads((cc.OUT / "task2_geometry.json").read_text())
    L, Li = cc.band_l, cc.band_l_inv
    g_nom = t1["gain_counts_per_w_m2_sr"]["joint_geometric_mean"]
    dc50 = {c: t1["calibration_points_50cm"][c]["delta_counts"] for c in ("cold_water_50", "hot_water_50")}

    # One set of draws shared by every case (the calibration is one object).
    eps = RNG.uniform(*EPS_RANGE, N)
    t_air = RNG.normal(cc.T_AIR_C, 0.5, N)
    t_cold = RNG.normal(cc.T_COLD_C, 0.5, N)
    t_hot = 66.0 - RNG.uniform(0, 1.5, N) + RNG.normal(0, 0.5, N)
    la = L(cc.c2k(t_air))
    gc = dc50["cold_water_50"] / (eps * (L(cc.c2k(t_cold)) - la))
    gh = dc50["hot_water_50"] / (eps * (L(cc.c2k(t_hot)) - la))
    w = RNG.uniform(0, 1, N)
    g = np.exp((1 - w) * np.log(gc) + w * np.log(gh))

    rows, geo = {}, {}
    for case in cc.CASES:
        sim = json.loads((cc.OUT / f"sim_{case}" / "report.json").read_text())
        res = t1["results"][case]
        dcs = res["delta_counts_first_last_mean"]
        dc = dcs[0] if REF_FRAME[case] == 0 else dcs[1]
        l22 = float(L(cc.c2k(cc.T_AIR_C)))
        real_nom = float(Li(l22 + dc / g_nom)) - 273.15
        sim_analytic = float(Li(0.93 * L(cc.c2k(T_SET[case])) + 0.07 * l22)) - 273.15
        real_mc = Li(la + dc / g) - 273.15
        sim_mc = Li(eps * L(cc.c2k(T_SET[case])) + (1 - eps) * la) - 273.15
        err = real_mc - sim_mc
        p = lambda x: [round(float(v), 2) for v in np.percentile(x, [16, 50, 84])]  # noqa: E731
        rows[case] = {
            "t_surface_set_in_sim_c": T_SET[case],
            "real_t_apparent_c": round(real_nom, 2), "real_t_apparent_c_mc_16_50_84": p(real_mc),
            "sim_scene_apparent_t_c": sim["cup_scene_apparent_t_c"], "sim_scene_apparent_t_c_analytic": round(sim_analytic, 3),
            "sim_camera_apparent_t_c": sim["cup_apparent_t_c"],
            "error_real_minus_sim_k": round(real_nom - sim["cup_scene_apparent_t_c"], 2),
            "error_k_mc_16_50_84": p(err),
            "real_t_surface_c_eps093": res["nominal_joint"]["t_surface_c_first_last_mean"][0 if REF_FRAME[case] == 0 else 1],
            "independent": case.endswith("95"),
        }
        if case.endswith("95"):
            # Variant: the sim fed the surface temperature the 50 cm set's own cooling/warming predicts for
            # this frame (task 1 Newton extrapolation; no 95 cm data in it). Analytic: the render matches it.
            nx = t1["newton_extrapolation_from_50cm"][case]["predicted_surface_c_at_target_first_last"]
            t_nx = nx[0] if REF_FRAME[case] == 0 else nx[1]
            sim_nx = float(Li(0.93 * L(cc.c2k(t_nx)) + 0.07 * l22)) - 273.15
            rows[case]["variant_sim_at_newton_temperature"] = {
                "t_surface_c": t_nx, "sim_scene_apparent_t_c": round(sim_nx, 2),
                "error_real_minus_sim_k": round(real_nom - sim_nx, 2)}
        meas = t2["cases"][case]["measured"]
        sil = sim["silhouette"]
        orig = cc.OUT / f"simorig_{case}" / "report.json"
        silo = json.loads(orig.read_text())["silhouette"] if orig.exists() else None
        keys = ("width_top10_px", "width_bottom10_px", "height_px")
        geo[case] = {"measured": {k: meas[k] for k in keys},
                     "render_measured_calibration": {k: sil[k] for k in keys},
                     "error_pct_render_measured_calibration": {k: round(100 * (meas[k] / sil[k] - 1), 1) for k in keys}}
        if silo:
            geo[case]["render_original_yaml_22mm"] = {k: silo[k] for k in keys}
            geo[case]["error_pct_render_original_yaml"] = {k: round(100 * (meas[k] / silo[k] - 1), 1) for k in keys}
    report = {"comparison": rows, "geometry_from_renders": geo,
              "note": "error = real calibrated apparent T - simulated scene_apparent_t, both blackbody-equivalent "
                      "(eps = 1 readout). 50 cm: not independent (the calibration was fitted there). 95 cm: the test."}
    (cc.OUT / "task3_comparison.json").write_text(json.dumps(report, indent=1))
    plot(rows)
    for c, r in rows.items():
        print(f"{c:14s} sim set {r['t_surface_set_in_sim_c']:5.1f} C | real T_app {r['real_t_apparent_c']:6.2f} "
              f"{r['real_t_apparent_c_mc_16_50_84']} | sim scene {r['sim_scene_apparent_t_c']:6.2f} (analytic "
              f"{r['sim_scene_apparent_t_c_analytic']:6.2f}, camera {r['sim_camera_apparent_t_c']:6.2f}) | error "
              f"{r['error_real_minus_sim_k']:+5.2f} K {r['error_k_mc_16_50_84']}", r.get("variant_sim_at_newton_temperature", ""))
    print(json.dumps(geo, indent=1))


def plot(rows: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 4, figsize=(16, 7.2), gridspec_kw={"height_ratios": [1, 1]})
    for i, case in enumerate(cc.CASES):
        real = np.load(cc.OUT / f"apparent_t_{case}.npy") - 273.15
        sim = np.load(cc.OUT / f"sim_{case}" / "scene_apparent_t_k.npy") - 273.15
        lo, hi = (5, 70) if case.startswith("hot") else (8, 25)
        axs[0, i].imshow(real, cmap="inferno", vmin=lo, vmax=hi)
        axs[0, i].set_title(f"real, calibrated: {case}\nT_app {rows[case]['real_t_apparent_c']:.1f} C", fontsize=9)
        im = axs[1, i].imshow(sim, cmap="inferno", vmin=lo, vmax=hi)
        axs[1, i].set_title(f"sim scene_apparent_t (set {rows[case]['t_surface_set_in_sim_c']:g} C, eps .93)\n"
                            f"{rows[case]['sim_scene_apparent_t_c']:.1f} C, error {rows[case]['error_real_minus_sim_k']:+.1f} K",
                            fontsize=9)
        for a in (axs[0, i], axs[1, i]):
            a.set_xlim(140, 360)
            a.set_ylim(320, 160)
        plt.colorbar(im, ax=axs[:, i], fraction=0.04, label="C")
    fig.savefig(cc.OUT / "task3_comparison.png", dpi=110)


if __name__ == "__main__":
    main()
