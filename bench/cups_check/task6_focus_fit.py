#!/usr/bin/env python
"""Task 6b: which focus distance (and residual aberration) the simulator needs to reproduce the measured edges.

    /home/jumasaet/isaac_sim/python.sh bench/cups_check/task6_focus_fit.py     (after task6_psf.py)

Forward model, with the simulator's own optics (irsim.optics.mtf.lens_otf, docs/physics-model.md §8.3, ADR 0129):
for a lens of focal length f = 14.6 mm at F/1.3 focused at vergence x = 1/s_f (dioptres; 0 = infinity), an
object at s has a blur circle

    c(s) = f^2 |1/s - x| / (F (1 - f x))        (thin lens; = irsim.optics.defocus.blur_circle_um for x >= 0)

and W020 = c / (8F). The line-spread function is the 1-D inverse transform of the Hopkins OTF times the
aberration Gaussian times the pixel box -- the same OTF the camera's DefocusKernelBank builds, at the band
centre (11 um). Its ESF goes into the rectangle model of task6_psf.py and is fitted to every water-row pixel of
all four sets at once; each set keeps its own level, contrast, edge positions and slant (nuisance), the lens
parameters are shared.

Models:
  M0  in focus, aberration sigma only              (schema: mtf.aberration_sigma_um)
  M1  focused at s_f >= f (x >= 0) + aberration     (schema: focus.mode fixed/infinity + defocus_model hopkins)
  M2  x free, negative allowed = focused "beyond infinity" (the sensor closer to the lens than the infinity
      image plane) + aberration. Diagnostic only: the schema cannot carry a negative focus distance.

Score: the mean over the four sets of each set's mean-square residual (normalised signal), so the cold sets,
whose contrast is 1/6 of the hot ones, weigh as much as the hot ones.

Writes out/task6_focus_fit.json and out/task6_focus_fit.png.
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.optimize import least_squares, minimize

import cups_common as cc
import task6_psf as t6

F_MM, F_NUM, PITCH_UM, LAM_UM = 14.6, 1.3, 24.0, 11.0
DX = 0.05                      # px, LSF sampling
NX = 8192                      # +-204.8 px
_X = (np.arange(NX) - NX // 2) * DX
_XI = np.fft.rfftfreq(NX, d=DX * PITCH_UM * 1e-3)  # cyc/mm
_CACHE: dict[tuple, np.ndarray] = {}


def blur_um(s_m: float, x_dpt: float) -> float:
    f = F_MM * 1e-3
    return float(f * f * abs(1.0 / s_m - x_dpt) / (F_NUM * (1.0 - f * x_dpt)) * 1e6)


def esf_px(w020_um: float, sigma_um: float, model: str = "hopkins") -> np.ndarray:
    """ESF on _X (px) of diffraction + defocus (lens_otf) x aberration Gaussian x pixel box."""
    # W020 on a 0.25 um grid = 2.6 um of blur circle (0.1 px); finer than the data can tell apart
    w020_um, sigma_um = round(w020_um * 4) / 4, round(sigma_um * 2) / 2
    key = (w020_um, sigma_um, model)
    if key in _CACHE:
        return _CACHE[key]
    from irsim.optics.mtf import cutoff_frequency_cyc_per_mm, lens_otf, mtf_gaussian

    # The OTF is zero past the diffraction cut-off; evaluate the quadrature only below it.
    below = _XI < cutoff_frequency_cyc_per_mm(LAM_UM, F_NUM)
    otf = np.zeros_like(_XI)
    otf[below] = lens_otf(_XI[below], LAM_UM, F_NUM, w020_um, model)
    if sigma_um > 0:
        otf = otf * mtf_gaussian(_XI, sigma_um * 1e-3)
    otf = otf * np.sinc(_XI * PITCH_UM * 1e-3)              # 1-px detector footprint
    lsf = np.fft.fftshift(np.fft.irfft(otf, n=NX))
    lsf = np.clip(lsf, 0, None)
    e = np.cumsum(lsf)
    e /= e[-1]
    # cumsum puts each sample's mass at its right edge; shift by half a sample so ESF(0) = 0.5
    e = np.interp(_X, _X + 0.5 * DX, e)
    _CACHE[key] = e
    return e


class CaseData:
    def __init__(self, case: str, n_img: np.ndarray, water: tuple, g: dict, dist: float):
        self.case, self.dist = case, dist
        y0, y1, _x0, _x1 = water
        c0, c1 = g["window_cols"]
        self.yy, self.xx = np.mgrid[y0:y1, c0:c1]
        self.z = n_img[y0:y1, c0:c1]
        self.ym = 0.5 * (y0 + y1 - 1)
        self.q0 = np.array([g["B"], g["A"], g["x_left"], g["x_right"], g["slope_px_per_row"]])

    def fit(self, esf: np.ndarray) -> tuple[float, np.ndarray]:
        def model(q):
            b, a, al, ar, sl = q
            ct = np.cos(np.arctan(sl))
            dl = (self.xx - (al + sl * (self.yy - self.ym))) * ct
            dr = (self.xx - (ar - sl * (self.yy - self.ym))) * ct
            return b + a * (np.interp(dl, _X, esf) - np.interp(dr, _X, esf))

        r = least_squares(lambda q: (model(q) - self.z).ravel(), self.q0, method="lm", max_nfev=200)
        return float(np.mean(r.fun**2)), r.x


def score(cases: list[CaseData], x_dpt: float, sigma_um: float, detail: bool = False):
    ms = []
    out = {}
    for c in cases:
        cb = blur_um(c.dist, x_dpt)
        e = esf_px(cb / (8 * F_NUM), sigma_um)
        m, q = c.fit(e)
        ms.append(m)
        if detail:
            out[c.case] = {"distance_m": c.dist, "blur_circle_um": round(cb, 1), "blur_circle_px": round(cb / PITCH_UM, 2),
                           "rms": round(float(np.sqrt(m)), 4), "w10_90_px_model": round(w1090(e), 2)}
    return (float(np.mean(ms)), out) if detail else float(np.mean(ms))


def w1090(e: np.ndarray) -> float:
    return float(_X[np.searchsorted(e, 0.9)] - _X[np.searchsorted(e, 0.1)])


def grid_then_refine(cases, xs, sigmas, x_bounds, s_bounds):
    best = (np.inf, None)
    grid = np.full((len(xs), len(sigmas)), np.nan)
    for i, x in enumerate(xs):
        for j, s in enumerate(sigmas):
            grid[i, j] = score(cases, x, s)
            if grid[i, j] < best[0]:
                best = (grid[i, j], (x, s))
    r = minimize(lambda p: score(cases, p[0], p[1]), np.array(best[1]), method="L-BFGS-B",
                 bounds=[x_bounds, s_bounds], options={"eps": 1e-3})
    # L-BFGS-B on a quantised cache can stop early; keep whichever is better
    x, s = (r.x if r.fun <= best[0] else best[1])
    return float(x), float(s), grid


def main() -> None:
    t1 = json.loads((cc.OUT / "task1_calibration.json").read_text())
    meas = json.loads((cc.OUT / "task6_psf.json").read_text())["cases"]
    cases = []
    for case in cc.CASES:
        n_img, _n = t6.normalised_mean(case, t1)
        water = tuple(t1["regions_px_y0_y1_x0_x1"][case]["water"])
        cases.append(CaseData(case, n_img, water, meas[case]["gaussian"], t6.lidar_distance(case)))
    t0 = time.time()
    sig = np.arange(0.0, 161.0, 10.0)
    res = {}
    # M0: no defocus
    best0 = min(((score_nodefocus(cases, s), s) for s in np.arange(0.0, 200.0, 2.0)))
    r0 = minimize(lambda p: score_nodefocus(cases, p[0]), [best0[1]], bounds=[(0, 300)], method="L-BFGS-B")
    s0 = float(r0.x[0]) if r0.fun <= best0[0] else best0[1]
    res["M0_in_focus_aberration_only"] = detail_nodefocus(cases, s0)
    # M1: x >= 0
    x1, s1, g1 = grid_then_refine(cases, np.arange(0.0, 2.01, 0.1), sig, (0.0, 3.0), (0.0, 300.0))
    res["M1_focus_at_or_inside_infinity"] = detail(cases, x1, s1)
    # M2: x free
    x2, s2, g2 = grid_then_refine(cases, np.arange(-2.5, 2.01, 0.1), sig, (-4.0, 3.0), (0.0, 300.0))
    res["M2_focus_free_incl_beyond_infinity"] = detail(cases, x2, s2)
    res["focus_profile_best_aberration_per_vergence"] = focus_profile(cases)
    res["reference_free_core_plus_halo_rms"] = {c: meas[c]["core_plus_halo"]["rms_residual"] for c in meas}
    res["reference_free_gaussian_rms"] = {c: meas[c]["gaussian"]["rms_residual"] for c in meas}
    res["measured_w10_90_px_modelfree"] = {c: meas[c]["core_plus_halo"]["w10_90_px_modelfree"] for c in meas}
    res["lens"] = {"focal_length_mm": F_MM, "f_number": F_NUM, "pitch_um": PITCH_UM, "wavelength_um": LAM_UM}
    res["infinity_focus_blur_px"] = {c.case: round(blur_um(c.dist, 0.0) / PITCH_UM, 2) for c in cases}
    (cc.OUT / "task6_focus_fit.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"{time.time() - t0:.0f} s")
    plot(cases, res, sig, np.arange(-2.5, 2.01, 0.1), g2)


def score_nodefocus(cases, sigma_um):
    return float(np.mean([c.fit(esf_px(0.0, sigma_um))[0] for c in cases]))


def detail_nodefocus(cases, s):
    e = esf_px(0.0, s)
    per = {c.case: {"rms": round(float(np.sqrt(c.fit(e)[0])), 4), "w10_90_px_model": round(w1090(e), 2)} for c in cases}
    return {"aberration_sigma_um": round(s, 1), "aberration_sigma_px": round(s / PITCH_UM, 2),
            "score_rms": round(float(np.sqrt(score_nodefocus(cases, s))), 4), "cases": per}


def detail(cases, x, s):
    sc, per = score(cases, x, s, detail=True)
    return {"focus_vergence_dpt": round(x, 3),
            "focus_distance_m": (round(1.0 / x, 3) if x > 1e-6 else ("infinity" if abs(x) <= 1e-6 else f"virtual, {1.0 / x:.3f}")),
            "aberration_sigma_um": round(s, 1), "aberration_sigma_px": round(s / PITCH_UM, 2),
            "score_rms": round(float(np.sqrt(sc)), 4), "cases": per}


def plot(cases, res, sig, xs, grid) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), dpi=120)
    ax = axs[0]
    im = ax.imshow(np.sqrt(grid).T, origin="lower", aspect="auto", cmap="viridis",
                   extent=(xs[0] - 0.05, xs[-1] + 0.05, sig[0] - 5, sig[-1] + 5))
    fig.colorbar(im, ax=ax, label="rms residual")
    ax.axvline(0, color="w", lw=0.8)
    ax.set_xlabel("focus vergence 1/s_f (dpt); < 0 = beyond infinity")
    ax.set_ylabel("aberration sigma (um)")
    ax.set_title("joint fit, 4 sets (Hopkins defocus)", fontsize=10)
    for k, dist in enumerate(("50", "95")):
        ax = axs[k + 1]
        for c in cases:
            if not c.case.endswith(dist):
                continue
            m = json.loads((cc.OUT / "task6_psf.json").read_text())["cases"][c.case]["core_plus_halo"]["esf_bins"]
            ax.plot(m["d_px"], m["value"], ".", ms=2, label=f"{c.case} measured")
        d = np.linspace(-45, 25, 600)
        for name, ls in (("M1_focus_at_or_inside_infinity", "-"), ("M2_focus_free_incl_beyond_infinity", "--")):
            r = res[name]
            dist_m = cases[0 if dist == "50" else 1].dist
            e = esf_px(blur_um(dist_m, r["focus_vergence_dpt"]) / (8 * F_NUM), r["aberration_sigma_um"])
            ax.plot(d, np.interp(d, _X, e), ls, color="k", lw=1.1, label=f"{name.split('_')[0]} model")
        r = res["M0_in_focus_aberration_only"]
        ax.plot(d, np.interp(d, _X, esf_px(0.0, r["aberration_sigma_um"])), ":", color="m", lw=1.1, label="M0 model")
        ax.set_xlim(-45, 25)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_title(f"{dist} cm: half-ESF and the fitted lens models", fontsize=10)
        ax.set_xlabel("distance from the edge, px (inside > 0)")
    fig.tight_layout()
    fig.savefig(cc.OUT / "task6_focus_fit.png")


def focus_profile(cases, xs=(-0.4, -0.2, 0.0, 0.1, 0.147, 0.2, 0.3, 0.5, 0.75, 1.0)) -> dict:
    """Best score at each focus vergence, aberration re-optimised: how sharply the data pin the focus."""
    from scipy.optimize import minimize_scalar

    out = {}
    for x in xs:
        r = minimize_scalar(lambda s: score(cases, x, s), bounds=(0.0, 200.0), method="bounded",
                            options={"xatol": 1.0})
        out[f"{x:+.3f}"] = {"aberration_sigma_um": round(float(r.x), 1), "score_rms": round(float(np.sqrt(r.fun)), 5)}
    return out


if __name__ == "__main__":
    main()
