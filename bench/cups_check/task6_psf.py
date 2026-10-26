#!/usr/bin/env python
"""Task 6: the lens's real blur, measured off the cups' side edges, and the defocus model that reproduces it.

    /home/jumasaet/isaac_sim/python.sh bench/cups_check/task6_psf.py          (measure; real data only)
    /home/jumasaet/isaac_sim/python.sh bench/cups_check/task6_psf.py --sim    (the same measurement on sim_*/)

Edge -> ESF -> LSF (docs/physics-model.md §8.3, where MTF_defocus is a slot the spec leaves to a fit):

* Counts are linear in band radiance (§9.3), and the blur acts on radiance, so the edge is measured in counts:
  every frame has its board reference subtracted and is divided by its own water contrast, which removes the
  FFC offset steps and the cooling during a series (a linear system's edge SHAPE does not depend on the
  amplitude). The 12-19 frames of a set are then averaged: the mean image has 1/sqrt(N) of a frame's noise.
* The cup's side walls lean 9.5 deg (7 cm top, 5 cm bottom, 6 cm tall), so the two side edges are slanted
  edges: across the water rows each row samples the edge at a different sub-pixel phase. The model is fitted to
  every pixel of every water row at once -- a rectangle between two straight edges x_l(y), x_r(y), convolved
  with a 1-D LSF:  N(x, y) = B + A [CDF((x - x_l(y)) cos t) - CDF((x - x_r(y)) cos t)].  Using the two edges
  together is what lets the halo be measured at 95 cm, where the cup is only ~40 px wide and each edge's tail
  reaches the other.
* LSF models: one Gaussian (what the schema can carry as aberration_sigma_um, or the Gaussian equivalent of a
  defocus disc), and a Gaussian core plus a wide Gaussian halo carrying a fraction h of the energy (what the
  data show at 95 cm; not representable in the schema).
* 10-90 % width: on the fitted ESF (core + halo), and model-free on the binned half-ESF of the outer side.

The pixel footprint, diffraction and the camera's denoise (neighbour r = 0.37, task 4) are all inside the
measured LSF. The simulated camera has the first two, so the comparison that matters is the same measurement
run on the renders (--sim), not a deconvolution.

Writes out/task6_psf.json and out/task6_psf.png (or out/task6_psf_sim_<tag>.json with --sim).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

import cups_common as cc

PAD = 45            # columns either side of the cup in the fit window
BIN_PX = 0.25       # half-ESF bin (supersampled thanks to the slant)


def normalised_mean(case: str, t1: dict) -> tuple[np.ndarray, int]:
    """Real: mean over the set of (frame - board) / water contrast."""
    frames, _t = cc.load_case(case)
    reg = t1["regions_px_y0_y1_x0_x1"][case]
    bg = np.array([np.median(np.concatenate([cc.roi(f, reg["bg_left"]).ravel(),
                                             cc.roi(f, reg["bg_right"]).ravel()])) for f in frames])
    amp = np.array([np.median(cc.roi(f, reg["water"])) for f in frames]) - bg
    return ((frames - bg[:, None, None]) / amp[:, None, None]).mean(0), len(frames)


def normalised_sim(case: str, tag: str) -> tuple[np.ndarray, tuple, int]:
    """Sim: the camera's apparent_t taken back to band radiance (linear, like counts), board-referenced."""
    d = cc.OUT / f"{tag}_{case}"
    t_k = np.load(d / "apparent_t_k.npy").astype(np.float64)
    lr = cc.band_l(t_k)
    l_air = float(cc.band_l(cc.c2k(cc.T_AIR_C)))
    # The same water region as on the real frames (cups_common.find_cup on a radiance-linear plane).
    water = tuple(cc.find_cup(lr, case.startswith("hot"))["water"])
    amp = np.median(cc.roi(lr, water)) - l_air
    return (lr - l_air) / amp, water, 1


def cdf_gauss(x: np.ndarray, s: float) -> np.ndarray:
    return 0.5 * (1.0 + erf(x / (np.sqrt(2.0) * s)))


def cdf_model(x: np.ndarray, p: np.ndarray, two: bool) -> np.ndarray:
    if not two:
        return cdf_gauss(x, p[0])
    s1, s2, h = p
    return (1.0 - h) * cdf_gauss(x, s1) + h * cdf_gauss(x, s2)


def fit_case(n_img: np.ndarray, water: tuple, two: bool) -> dict:
    y0, y1, x0, x1 = water
    xc = 0.5 * (x0 + x1)
    # first guess of the edges: 50 % crossings of the row-mean profile
    prof = n_img[y0:y1].mean(0)
    left = np.nonzero(prof[: int(xc)] < 0.5)[0].max()
    right = int(xc) + np.nonzero(prof[int(xc):] < 0.5)[0].min()
    c0, c1 = left - PAD, right + PAD
    yy, xx = np.mgrid[y0:y1, c0:c1]
    z = n_img[y0:y1, c0:c1]
    ym = 0.5 * (y0 + y1 - 1)
    # The wall leans outward going up: x_l moves left and x_r right as y decreases (image rows grow down).
    def model(q):
        b, a, al, ar, slope = q[:5]
        xl = al + slope * (yy - ym)
        xr = ar - slope * (yy - ym)
        ct = np.cos(np.arctan(slope))
        p = q[5:]
        return b + a * (cdf_model((xx - xl) * ct, p, two) - cdf_model((xx - xr) * ct, p, two))

    lsf0 = [2.5, 12.0, 0.1] if two else [4.0]
    q0 = np.r_[0.0, 1.0, left + 0.5, right - 0.5, 0.16, lsf0]
    lo = np.r_[-0.5, 0.2, c0, xc, -1.0, [0.3, 3.0, 0.0] if two else [0.3]]
    hi = np.r_[0.5, 2.0, xc, c1, 1.0, [12.0, 60.0, 0.6] if two else [20.0]]
    r = least_squares(lambda q: (model(q) - z).ravel(), q0, bounds=(lo, hi))
    res = r.fun.reshape(z.shape)
    out = {"B": r.x[0], "A": r.x[1], "x_left": r.x[2], "x_right": r.x[3], "slope_px_per_row": r.x[4],
           "rms_residual": float(np.sqrt(np.mean(res**2))), "n_px": int(z.size), "window_cols": [int(c0), int(c1)]}
    if two:
        out.update({"sigma_core_px": r.x[5], "sigma_halo_px": r.x[6], "halo_fraction": r.x[7]})
        p = r.x[5:]
    else:
        out["sigma_px"] = r.x[5]
        p = r.x[5:]
    out["w10_90_px"] = width_10_90(p, two)
    # half-ESF, model-free, outer side of each edge (binned on the slant-corrected distance)
    b, a, al, ar, slope = r.x[:5]
    ct = np.cos(np.arctan(slope))
    dl = (xx - (al + slope * (yy - ym))) * ct           # inside positive
    dr = ((ar - slope * (yy - ym)) - xx) * ct
    zn = (z - b) / a
    d = np.r_[dl[xx < xc], dr[xx >= xc]]
    v = np.r_[zn[xx < xc], zn[xx >= xc]]
    edges = np.arange(-PAD, 0.5 * (ar - al) + BIN_PX, BIN_PX)
    k = np.digitize(d, edges)
    centres, means = [], []
    for i in range(1, len(edges)):
        m = k == i
        if m.sum() >= 3:
            centres.append(0.5 * (edges[i - 1] + edges[i]))
            means.append(float(v[m].mean()))
    out["esf_bins"] = {"d_px": [round(c, 3) for c in centres], "value": [round(m, 4) for m in means]}
    out["w10_90_px_modelfree"] = modelfree_10_90(np.array(centres), np.array(means))
    out = {k2: (round(float(v2), 4) if isinstance(v2, (float, np.floating)) else v2) for k2, v2 in out.items()}
    return out


def width_10_90(p: np.ndarray, two: bool) -> float:
    x = np.linspace(-80, 80, 64001)
    e = cdf_model(x, np.asarray(p), two)
    return float(x[np.searchsorted(e, 0.9)] - x[np.searchsorted(e, 0.1)])


def modelfree_10_90(d: np.ndarray, v: np.ndarray) -> float:
    """10-90 % of the binned half-ESF, levels taken as the median far outside (< -30 px) and the plateau
    (the innermost 3 px). At 95 cm the plateau is the cup centre, which the halo has not fully reached."""
    lo = np.median(v[d < -30]) if np.any(d < -30) else 0.0
    hi = np.median(v[d > d.max() - 3])
    s = np.convolve((v - lo) / (hi - lo), np.ones(5) / 5, mode="same")
    i90 = np.nonzero(s >= 0.9)[0].min()
    i10 = np.nonzero(s[:i90] <= 0.1)[0].max()
    return float(d[i90] - d[i10])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=None, help="measure the renders in out/<tag>_<case>/ instead (e.g. sim, sim6)")
    args = ap.parse_args()
    t1 = json.loads((cc.OUT / "task1_calibration.json").read_text())
    rows = {}
    for case in cc.CASES:
        if args.sim:
            n_img, water, nf = normalised_sim(case, args.sim)
        else:
            n_img, nf = normalised_mean(case, t1)
            water = tuple(t1["regions_px_y0_y1_x0_x1"][case]["water"])
        g1 = fit_case(n_img, water, two=False)
        g2 = fit_case(n_img, water, two=True)
        rows[case] = {"frames": nf, "water_rows": [water[0], water[1]], "gaussian": g1, "core_plus_halo": g2,
                      "axis_distance_m": lidar_distance(case)}
        print(f"{case:14s} N={nf:2d} gauss sigma {g1['sigma_px']:5.2f} px w10-90 {g1['w10_90_px']:5.2f} "
              f"rms {g1['rms_residual']:.4f} | core {g2['sigma_core_px']:5.2f} halo {g2['sigma_halo_px']:5.2f} "
              f"h {g2['halo_fraction']:.3f} w10-90 {g2['w10_90_px']:5.2f} rms {g2['rms_residual']:.4f} | "
              f"model-free w10-90 {g2['w10_90_px_modelfree']:5.2f}")
    name = f"task6_psf_sim_{args.sim}.json" if args.sim else "task6_psf.json"
    (cc.OUT / name).write_text(json.dumps({"cases": rows}, indent=1))
    if not args.sim:
        plot(rows)


def lidar_distance(case: str) -> float:
    """Horizontal camera-to-cup-axis distance from the LiDAR (task 2). The side edges are the wall's tangent
    points, within 0.3 % of it (radius 3 cm at 50 cm)."""
    g = json.loads((cc.OUT / "lidar_geometry.json").read_text())
    return float(g["cases"][case]["cup_axis_horizontal_distance_m"])


def plot(rows: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), dpi=130)
    col = {"hot_water_50": "#c0392b", "cold_water_50": "#2471a3", "hot_water_95": "#e67e22", "cold_water_95": "#5dade2"}
    for k, dist in enumerate(("50", "95")):
        ax = axs[k]
        for case in (f"hot_water_{dist}", f"cold_water_{dist}"):
            r = rows[case]
            e = r["core_plus_halo"]["esf_bins"]
            ax.plot(e["d_px"], e["value"], ".", ms=2.5, color=col[case], label=f"{case} (N = {r['frames']})")
            d = np.array(e["d_px"])
            g = r["gaussian"]
            ax.plot(d, cdf_gauss(d, g["sigma_px"]), "-", lw=1.0, color=col[case], alpha=0.7)
            h = r["core_plus_halo"]
            ax.plot(d, cdf_model(d, np.array([h["sigma_core_px"], h["sigma_halo_px"], h["halo_fraction"]]), True),
                    "--", lw=1.0, color="k", alpha=0.6)
        ax.set_title(f"{dist} cm: half-ESF (dots), Gaussian fit (line), core + halo (dashed)", fontsize=10)
        ax.set_xlabel("distance from the edge, px (inside > 0)")
        ax.set_ylabel("normalised signal")
        ax.set_xlim(-45, 25)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(cc.OUT / "task6_psf.png")


if __name__ == "__main__":
    main()
