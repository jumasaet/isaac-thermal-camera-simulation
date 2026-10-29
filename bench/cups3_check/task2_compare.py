#!/usr/bin/env python
"""Task 2 (compare): apparent temperature of each cup, real (calibrated, task 1) against simulated (task2_render).

    /home/jumasaet/isaac_sim/python.sh bench/cups3_check/task2_compare.py

Same definitions on both sides, at the 8 instants rendered (real frame i <-> render j at the same time):
  - ROI: cups3_common.find_cups on each side's own mean image (the water plateau's central half);
  - board reference: the board plane (cups3_common.board_plane) at the ROI's centre, per frame;
  - real:  T = L_B^-1( L_B(T_air) + (c_cup - c_board) / G ),  G = (G eps) / 0.93   (task 1's calibration);
  - sim:   T = L_B^-1( L_B(T_air) + L_B(T_app,cup) - L_B(T_app,board) )  on the simulated camera's apparent_t
    ("board-referenced", the treatment the real data get). The sim camera's raw apparent_t reads the board
    +0.38 K above the scene (the housing/optics self-emission term, whose parameters are still the Boson's
    placeholders in the YAML); the real calibration removes the real camera's offset the same way.
    Reported as well: the raw apparent_t and the noise-free scene_apparent_t (no camera at all).

Independence: the hot and cold cups set the real calibration (task 1) with the same probe temperatures the
simulator is given, so their real - sim errors are the calibration's residual, not a test. The WARM cup is
independent: nothing about it entered the calibration.

Writes out/task2_comparison.json, out/task2_comparison.png and out/sim/apparent_t_ref_<j>.npy (the simulated
frames board-referenced like the real image of task 1: one radiance offset per frame puts the board's mean at
L_B(T_air); read by tasks 3 and 5).
"""
from __future__ import annotations

import json

import numpy as np

import cups3_common as c3


def main() -> None:
    simrep = json.loads((c3.OUT / "sim" / "report.json").read_text())
    t1 = json.loads((c3.OUT / "task1_calibration.json").read_text())
    pf = np.load(c3.OUT / "task1_per_frame.npz")
    tr = pf["t_unix"] - pf["t_unix"][0]
    n = len(simrep["renders"])
    app = np.stack([np.load(c3.OUT / "sim" / f"apparent_t_{j}.npy").astype(np.float64) for j in range(n)])
    scn = np.stack([np.load(c3.OUT / "sim" / f"scene_apparent_t_{j}.npy").astype(np.float64) for j in range(n)])
    l_app = c3.band_l(app)
    reg = c3.find_cups(l_app.mean(0))
    mask = c3.board_mask(reg)
    coef = c3.board_plane(l_app, mask)
    lv = c3.per_frame_levels(l_app, reg, coef)
    l_air = c3.band_l(c3.c2k(c3.T_AIR_C))
    idx = np.array([r["real_frame_index"] for r in simrep["renders"]])
    rows = {}
    for cup in c3.CUPS:
        sim_ref = c3.band_l_inv(l_air + lv[cup]["delta"]) - 273.15
        sim_raw = np.array([np.median(c3.roi(app[j], reg[cup]["water"])) for j in range(n)]) - 273.15
        sim_scn = np.array([np.median(c3.roi(scn[j], reg[cup]["water"])) for j in range(n)]) - 273.15
        real = pf[f"app_{cup}"][idx]
        exp = pf[f"app_exp_{cup}"][idx]
        err = real - sim_ref
        rows[cup] = {
            "independent": cup == "warm",
            "probe_c": [round(float(v), 3) for v in pf[f"tw_{cup}"][idx]],
            "real_c": [round(float(v), 3) for v in real],
            "sim_board_referenced_c": [round(float(v), 3) for v in sim_ref],
            "sim_raw_apparent_t_c": [round(float(v), 3) for v in sim_raw],
            "sim_scene_apparent_t_c": [round(float(v), 3) for v in sim_scn],
            "expected_from_probe_c": [round(float(v), 3) for v in exp],
            "error_real_minus_sim_k_mean": round(float(err.mean()), 3),
            "error_real_minus_sim_k_first_last": [round(float(err[0]), 3), round(float(err[-1]), 3)],
            "error_real_minus_sim_raw_k_mean": round(float((real - sim_raw).mean()), 3),
            "error_real_minus_sim_scene_k_mean": round(float((real - sim_scn).mean()), 3),
            "sim_camera_minus_scene_k_mean": round(float((sim_ref - sim_scn).mean()), 3),
            "sim_scene_minus_expected_k_max_abs": round(float(np.abs(sim_scn - exp).max()), 3),
        }
    # Board-referenced simulated IMAGES, the treatment task 1 gives the real image: one radiance offset per frame
    # so that the board (mean over the board mask) reads L_B(T_air). Used by task 3 and task 5.
    for j in range(n):
        off = float(np.mean(l_app[j][mask])) - l_air
        t_ref = c3.band_l_inv(l_app[j] - off).astype(np.float32)
        assert t_ref.dtype == np.float32
        np.save(c3.OUT / "sim" / f"apparent_t_ref_{j}.npy", t_ref)
    board_raw = np.array([np.median(app[j][mask]) for j in range(n)]) - 273.15
    board_scn = np.array([np.median(scn[j][mask]) for j in range(n)]) - 273.15
    rep = {"instants_s": [round(float(tr[i]), 3) for i in idx], "real_frame_index": idx.tolist(),
           "sim_regions": reg, "cups": rows,
           "sim_board_raw_minus_scene_k": round(float((board_raw - board_scn).mean()), 3),
           "warm_uncertainty_from_task1_apparent_k_p16_50_84": t1["mc_A_mean_of_readings"]["warm_err_apparent_k_p2.5_16_50_84_97.5"][1:4],
           "warm_uncertainty_from_task1_apparent_k_p16_50_84_scenario_B": t1["mc_B_anywhere_between_readings"]["warm_err_apparent_k_p2.5_16_50_84_97.5"][1:4]}
    (c3.OUT / "task2_comparison.json").write_text(json.dumps(rep, indent=1))
    for cup, r in rows.items():
        print(cup, "indep" if r["independent"] else "calib", {k: v for k, v in r.items() if k.startswith(("error", "sim_camera", "sim_scene_minus"))})
    print("board raw - scene", rep["sim_board_raw_minus_scene_k"])
    plot(tr, idx, pf, rows)


def plot(tr, idx, pf, rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"cold": "#2166ac", "warm": "#b8860b", "hot": "#b2182b"}
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, cup in zip(axs, c3.CUPS):
        r = rows[cup]
        ax.plot(tr, pf[f"app_{cup}"], ".", ms=3, color=col[cup], label="real, calibrated (every frame)")
        ax.plot(tr[idx], r["sim_board_referenced_c"], "ks", ms=5, mfc="none", label="simulated camera, board-referenced")
        ax.plot(tr[idx], r["sim_scene_apparent_t_c"], "k+", ms=7, label="simulated scene (no camera)")
        ax.plot(tr, pf[f"tw_{cup}"], "-", color="0.6", lw=1, label="probe, mean of the two readings (sim input)")
        tag = "independent" if r["independent"] else "used in the calibration"
        ax.set_title(f"{cup} ({tag}): real − sim {r['error_real_minus_sim_k_mean']:+.2f} K", fontsize=10)
        ax.set_xlabel("time since first frame [s]")
        ax.set_ylabel("°C")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(c3.OUT / "task2_comparison.png", dpi=110)


if __name__ == "__main__":
    main()
