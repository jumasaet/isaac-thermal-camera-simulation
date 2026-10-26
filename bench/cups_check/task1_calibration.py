#!/usr/bin/env python
"""Task 1: counts -> temperature for the Cellplus, fitted in RADIANCE on the 50 cm sets only, tested at 95 cm.

    /home/jumasaet/isaac_sim/python.sh bench/cups_check/task1_calibration.py

Model (docs/physics-model.md §3.2 (b), §3.3, §9.3): the bolometer's output is linear in the band radiance
it receives, so in every frame

    c = G * L_scene + O(t)

with O(t) moving between frames (FFC steps of ~15 counts inside a set, ~50 counts between the hot and the
cold session). The board is the third point: it stands in a closed room at T_air, so its apparent radiance
is L_B(T_air) whatever its emissivity, and differencing each frame against its own board removes O(t):

    dc = c_cup - c_board = G * [eps * L_B(T_s) + (1 - eps) * L_B(T_air) - L_B(T_air)]
                         = G * eps * (L_B(T_s) - L_B(T_air))

G and the paper emissivity eps only appear as a product: the 50 cm points fix G*eps. The emissivity then
sets the camera's gain G (and with it the blackbody apparent temperature of anything else), while the
SURFACE temperature of the same paper at 95 cm, L_B^-1(L_B(T_air) + dc / (G eps)), does not depend on it.

Which hot temperature goes with which set (the water only cools, the captures are time-stamped):
hot_50 07:18:48-07:19:32 JST, hot_95 07:24:51-07:25:34, so hot_50 is the earlier and hotter set. The
assumption is that 66 C was read at the start of hot_50 and 58 C at the end of hot_95, i.e. that the two
readings bracket the hot session. The cold readings are taken as the start of cold_50 (11 C).

Calibration points: the frames in the first 5 s of hot_50 (66 C) and of cold_50 (11 C). Two points and
one unknown (G eps) leave one degree of freedom: the check that a radiance-linear camera agrees with all
three temperatures. It does not (6.2 measured vs 5.1 predicted for the hot/cold contrast ratio), so the
calibration carries that disagreement as its dominant uncertainty (see `uncertainty`).

Writes out/task1_calibration.json, out/task1_calibration.png and out/apparent_t_<case>.npy (float32 K,
blackbody apparent temperature of each set's mean frame, for task 3).
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np

import cups_common as cc

RNG = np.random.default_rng(20261026)
N_MC = 20000
EPS_NOMINAL, EPS_RANGE = 0.93, (0.90, 0.96)  # paper / PE-coated paper cup, LWIR (ESTIMATED: no measurement)
START_WINDOW_S = 5.0
SIGMA_T_READING_C = 0.5  # the thermometer is not named; 0.5 C (1 sigma) assumed for each reading
SIGMA_T_AIR_C = 0.5
HOT_READING_LAG_C = (0.0, 1.5)  # the 66 C reading may precede the first frame: at the ~0.05 K/s the set
#                                cools at, up to 30 s earlier = up to 1.5 K hotter than the first frame


def jst(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone(dt.timedelta(hours=9))).strftime("%H:%M:%S")


def main() -> None:
    cc.OUT.mkdir(exist_ok=True)
    sets = {}
    for case in cc.CASES:
        frames, t = cc.load_case(case)
        mean = frames.mean(0)
        regions = cc.find_cup(mean, case.startswith("hot"))
        lv = cc.per_frame_levels(frames, regions)
        sets[case] = dict(t=t, regions=regions, mean=mean, **lv)

    order = sorted(cc.CASES, key=lambda c: sets[c]["t"][0])
    timeline = [{"case": c, "start_jst": jst(sets[c]["t"][0]), "end_jst": jst(sets[c]["t"][-1]),
                 "frames": len(sets[c]["t"]), "duration_s": round(float(sets[c]["t"][-1] - sets[c]["t"][0]), 1),
                 "gap_from_previous_s": None if i == 0 else round(float(sets[c]["t"][0] - sets[order[i - 1]]["t"][-1]), 1)}
                for i, c in enumerate(order)]

    L = cc.band_l
    l_air = float(L(cc.c2k(cc.T_AIR_C)))

    def start_delta(case: str) -> float:
        s = sets[case]
        m = s["t"] - s["t"][0] <= START_WINDOW_S
        return float(np.mean(s["delta"][m]))

    dc_cold50, dc_hot50 = start_delta("cold_water_50"), start_delta("hot_water_50")
    t_hot50_c = cc.T_HOT_RANGE_C[1]

    def gains(eps, t_air_c, t_cold_c, t_hot_c):
        la = L(cc.c2k(t_air_c))
        g_c = dc_cold50 / (eps * (L(cc.c2k(t_cold_c)) - la))
        g_h = dc_hot50 / (eps * (L(cc.c2k(t_hot_c)) - la))
        return g_c, g_h

    # -- nominal ----------------------------------------------------------------------------------------
    g_c, g_h = (float(v) for v in gains(EPS_NOMINAL, cc.T_AIR_C, cc.T_COLD_C, t_hot50_c))
    # Inverse-variance weights from the temperature readings alone (what each point's G would scatter by).
    dldt = lambda tc: float((L(cc.c2k(tc) + 0.05) - L(cc.c2k(tc) - 0.05)) / 0.1)  # noqa: E731
    sig_lc = np.hypot(dldt(cc.T_COLD_C) * SIGMA_T_READING_C, dldt(cc.T_AIR_C) * SIGMA_T_AIR_C)
    sig_lh = np.hypot(dldt(t_hot50_c) * np.hypot(SIGMA_T_READING_C, np.ptp(HOT_READING_LAG_C) / np.sqrt(12)),
                      dldt(cc.T_AIR_C) * SIGMA_T_AIR_C)
    rel_c = sig_lc / abs(float(L(cc.c2k(cc.T_COLD_C))) - l_air)
    rel_h = sig_lh / abs(float(L(cc.c2k(t_hot50_c))) - l_air)
    w_c, w_h = 1 / rel_c ** 2, 1 / rel_h ** 2
    log_g_w = (w_c * np.log(g_c) + w_h * np.log(g_h)) / (w_c + w_h)
    chi2 = w_c * (np.log(g_c) - log_g_w) ** 2 + w_h * (np.log(g_h) - log_g_w) ** 2
    birge = float(np.sqrt(chi2))  # 1 dof
    # The readings' own uncertainties cannot explain the disagreement (Birge ratio ~3), so the
    # inverse-variance weights (which would hand the result to the hot point) are not trusted: the
    # nominal G is the geometric mean of the two single-point gains, the centre of the Monte Carlo below.
    g_joint = float(np.sqrt(g_c * g_h))

    # The same test in kelvin: a camera linear in T would have one SITF for both points.
    sitf_c = dc_cold50 / (cc.T_COLD_C - cc.T_AIR_C)
    sitf_h = dc_hot50 / (t_hot50_c - cc.T_AIR_C)

    # What each single temperature would have to be for the three to agree (others held at nominal).
    ratio = dc_hot50 / dc_cold50
    lc, lh = float(L(cc.c2k(cc.T_COLD_C))), float(L(cc.c2k(t_hot50_c)))
    consistent = {
        "measured_hot_over_cold_contrast": round(ratio, 3),
        "radiance_model_predicts": round((lh - l_air) / (lc - l_air), 3),
        "kelvin_model_predicts": round((t_hot50_c - cc.T_AIR_C) / (cc.T_COLD_C - cc.T_AIR_C), 3),
        "cold_surface_that_would_agree_c": round(float(cc.band_l_inv(l_air + (lh - l_air) / ratio)) - 273.15, 2),
        "hot_surface_that_would_agree_c": round(float(cc.band_l_inv(l_air + ratio * (lc - l_air))) - 273.15, 2),
        "board_that_would_agree_c": round(float(cc.band_l_inv((lh - ratio * lc) / (1 - ratio))) - 273.15, 2),
    }

    def apply(delta, g, eps, t_air_c=cc.T_AIR_C):
        la = L(cc.c2k(t_air_c))
        t_app = cc.band_l_inv(la + delta / g) - 273.15
        t_surf = cc.band_l_inv(la + delta / (g * eps)) - 273.15
        return t_app, t_surf

    # -- Monte Carlo ------------------------------------------------------------------------------------
    def draw(scenario: str):
        eps = RNG.uniform(*EPS_RANGE, N_MC)
        t_air = RNG.normal(cc.T_AIR_C, SIGMA_T_AIR_C, N_MC)
        t_cold = RNG.normal(cc.T_COLD_C, SIGMA_T_READING_C, N_MC)
        if scenario == "A_time_order":
            t_hot = t_hot50_c - RNG.uniform(*HOT_READING_LAG_C, N_MC) + RNG.normal(0, SIGMA_T_READING_C, N_MC)
        else:  # B: no assumption, the first hot_50 frame anywhere in the 8 C range
            t_hot = RNG.uniform(*cc.T_HOT_RANGE_C, N_MC)
        gc_, gh_ = gains(eps, t_air, t_cold, t_hot)
        # The two points disagree by more than their readings allow; the truth is taken to lie anywhere
        # between the cold-only and the hot-only calibration (w uniform), i.e. we do not know which of the
        # three temperatures is off.
        w = RNG.uniform(0, 1, N_MC)
        g = np.exp((1 - w) * np.log(gc_) + w * np.log(gh_))
        return dict(eps=eps, t_air=t_air, g=g, gc=gc_, gh=gh_)

    def q(x):
        p = np.percentile(x, [2.5, 16, 50, 84, 97.5])
        return {"median": round(float(p[2]), 2), "68": [round(float(p[1]), 2), round(float(p[3]), 2)],
                "95": [round(float(p[0]), 2), round(float(p[4]), 2)]}

    results = {}
    mc = {s: draw(s) for s in ("A_time_order", "B_hot_anywhere_58_66")}
    for case in cc.CASES:
        s = sets[case]
        d_first, d_last, d_mean = float(s["delta"][0]), float(s["delta"][-1]), float(np.mean(s["delta"]))
        row = {"delta_counts_first_last_mean": [round(d_first, 1), round(d_last, 1), round(d_mean, 1)],
               "board_counts_mean": round(float(np.mean(s["bg"])), 1)}
        for name, g in (("joint", g_joint), ("cold_point_only", g_c), ("hot_point_only", g_h)):
            ta, ts = apply(np.array([d_first, d_last, d_mean]), g, EPS_NOMINAL)
            row[f"nominal_{name}"] = {"t_apparent_c_first_last_mean": [round(float(v), 2) for v in ta],
                                      "t_surface_c_first_last_mean": [round(float(v), 2) for v in ts]}
        # Reference frame: the one the reading refers to under the time-order assumption.
        i_ref = -1 if case == "hot_water_95" else 0
        for sc, m in mc.items():
            ta, ts = apply(d_mean, m["g"], m["eps"], m["t_air"])
            row[f"mc_{sc}_set_mean"] = {"t_apparent_c": q(ta), "t_surface_c": q(ts)}
            ta, ts = apply(float(s["delta"][i_ref]), m["g"], m["eps"], m["t_air"])
            row[f"mc_{sc}_reference_frame"] = {"frame": "last" if i_ref == -1 else "first",
                                               "t_apparent_c": q(ta), "t_surface_c": q(ts)}
        results[case] = row

    # -- one factor at a time: how far each unknown moves the 95 cm answers (others at nominal) ------------
    def at95(eps=EPS_NOMINAL, t_air_c=cc.T_AIR_C, t_cold_c=cc.T_COLD_C, t_hot_c=t_hot50_c, which="joint"):
        gc_, gh_ = (float(v) for v in gains(eps, t_air_c, t_cold_c, t_hot_c))
        g = {"joint": np.sqrt(gc_ * gh_), "cold": gc_, "hot": gh_}[which]
        out = {}
        for case, i in (("hot_water_95", -1), ("cold_water_95", 0)):
            ta, ts = apply(float(sets[case]["delta"][i]), g, eps, t_air_c)
            out[case] = (float(ta), float(ts))
        return out

    def span(a, b):
        return {c: {"t_apparent_c": [round(a[c][0], 2), round(b[c][0], 2)],
                    "t_surface_c": [round(a[c][1], 2), round(b[c][1], 2)]} for c in a}

    sensitivity = {
        "paper_eps_0.90_vs_0.96": span(at95(eps=0.90), at95(eps=0.96)),
        "hot_reading_58_vs_66_at_hot50_start": span(at95(t_hot_c=58.0), at95(t_hot_c=66.0)),
        "cold_point_only_vs_hot_point_only": span(at95(which="cold"), at95(which="hot")),
        "air_21.5_vs_22.5": span(at95(t_air_c=21.5), at95(t_air_c=22.5)),
        "cold_reading_10.5_vs_11.5": span(at95(t_cold_c=10.5), at95(t_cold_c=11.5)),
        "frames": "hot_water_95 last frame, cold_water_95 first frame",
    }

    # -- Newton extrapolation from the 50 cm sets alone (is "58 C at the end of hot_95" plausible?) -------
    def newton(case: str, target: str):
        s = sets[case]
        _, ts = apply(s["delta"], g_joint, EPS_NOMINAL)
        ex = ts - cc.T_AIR_C
        tt = s["t"] - s["t"][0]
        k, b = np.polyfit(tt, np.log(np.abs(ex)), 1)  # |T - T_air| = exp(b) exp(k t)
        tau = -1 / k
        t_target = sets[target]["t"] - s["t"][0]
        pred = cc.T_AIR_C + np.sign(ex[0]) * np.exp(b + k * t_target)
        return {"tau_s": round(float(tau), 0), "fit_on": case,
                "predicted_surface_c_at_target_first_last": [round(float(pred[0]), 2), round(float(pred[-1]), 2)]}

    extrap = {"hot_water_95": newton("hot_water_50", "hot_water_95"),
              "cold_water_95": newton("cold_water_50", "cold_water_95"),
              "note": "single-exponential cooling fitted to each 50 cm set's own frames (44-56 s), extrapolated "
                      "4-6 min. Evaporation makes hot water cool faster than exponential at the start, so tau "
                      "from the first minute is a lower bound and the hot prediction a lower bound too."}

    # -- blur: what the 95 cm water region loses against the 50 cm one -----------------------------------
    blur = blur_loss(sets)

    # Absolute form per session: c = G L + O.
    offsets = {c: round(float(np.mean(sets[c]["bg"]) - g_joint * l_air), 1) for c in cc.CASES}
    sitf_22 = g_joint * dldt(cc.T_AIR_C)
    sitf_300k = g_joint * float((L(300.05) - L(299.95)) / 0.1)

    report = {
        "timeline_jst": timeline,
        "assumption_hot": "66 C read at the first frame of hot_50 (the earliest hot set), 58 C at the last frame "
                          "of hot_95. hot_50 starts 06:03 min before hot_95 and the water only cools.",
        "regions_px_y0_y1_x0_x1": {c: {k: sets[c]["regions"][k] for k in ("water", "bg_left", "bg_right")}
                                   for c in cc.CASES},
        "calibration_points_50cm": {
            "cold_water_50": {"t_surface_c": cc.T_COLD_C, "delta_counts": round(dc_cold50, 1)},
            "hot_water_50": {"t_surface_c": t_hot50_c, "delta_counts": round(dc_hot50, 1)},
            "board": {"t_c": cc.T_AIR_C, "role": "per-frame offset reference (the third point)"}},
        "model": "c = G * L_B + O(t);  dc = c_cup - c_board = G * eps * (L_B(T_s) - L_B(T_air))",
        "band_radiance_w_m2_sr": {f"{t:g}C": round(float(L(cc.c2k(t))), 4) for t in (11, 22, 58, 66)},
        "eps_paper_nominal": EPS_NOMINAL, "eps_paper_range": EPS_RANGE,
        "gain_counts_per_w_m2_sr": {"joint_geometric_mean": round(g_joint, 3),
                                    "inverse_variance_weighted_not_used": round(float(np.exp(log_g_w)), 3), "cold_point_only": round(g_c, 3),
                                    "hot_point_only": round(g_h, 3),
                                    "hot_over_cold": round(g_h / g_c, 4), "birge_ratio": round(birge, 2)},
        "kelvin_linear_check_counts_per_k": {"cold_point": round(sitf_c, 2), "hot_point": round(sitf_h, 2),
                                             "hot_over_cold": round(sitf_h / sitf_c, 4)},
        "consistency_of_the_three_temperatures": consistent,
        "sitf_counts_per_k": {"at_22c_scene": round(sitf_22, 2), "at_300k": round(sitf_300k, 2)},
        "offset_counts_per_set": offsets,
        "calibration_formula": "T_apparent = L_B^-1( L_B(T_air) + (c - c_board) / G );  "
                               "T_surface = L_B^-1( L_B(T_air) + (c - c_board) / (G eps) );  "
                               f"G = {g_joint:.2f} counts per W m^-2 sr^-1 at eps = {EPS_NOMINAL}, "
                               f"G*eps = {g_joint * EPS_NOMINAL:.2f} (what the data fix)",
        "results": results,
        "sensitivity_one_factor_at_a_time_95cm": sensitivity,
        "newton_extrapolation_from_50cm": extrap,
        "blur_loss_95_vs_50": blur,
    }
    (cc.OUT / "task1_calibration.json").write_text(json.dumps(report, indent=1))

    # Calibrated apparent temperature images (set means) for task 3.
    for c in cc.CASES:
        s = sets[c]
        img = cc.band_l_inv(l_air + (s["mean"] - np.mean(s["bg"])) / g_joint).astype(np.float32)
        assert img.dtype == np.float32
        np.save(cc.OUT / f"apparent_t_{c}.npy", img)

    plot(sets, report, g_joint, g_c, g_h, apply)
    print(json.dumps({k: report[k] for k in ("gain_counts_per_w_m2_sr", "kelvin_linear_check_counts_per_k",
                                             "consistency_of_the_three_temperatures", "sitf_counts_per_k")}, indent=1))
    for c in cc.CASES:
        r = results[c]
        print(c, "joint", r["nominal_joint"], "\n   MC-A ref", r["mc_A_time_order_reference_frame"],
              "\n   MC-B ref", r["mc_B_hot_anywhere_58_66_reference_frame"])
    print(json.dumps(extrap, indent=1))
    print(json.dumps(blur, indent=1))


def blur_loss(sets: dict) -> dict:
    """Contrast the camera's blur takes from the centre of the water region, 95 cm against 50 cm.

    The line-spread function is the derivative of the hot_50 cup's vertical edges (hot_50 minus cold_50, so
    the board's own structure cancels), out to +-100 px so the halo is in it. The 2-D PSF is taken as
    LSF(x) LSF(y). The water region is a rectangle of its 50 %-width x plateau height; above it the dry band
    and below it the table reflection sit at about half the water's contrast, so only half of what spills
    out vertically is lost.
    """
    h = sets["hot_water_50"]
    k = sets["cold_water_50"]
    d = (h["mean"] - np.mean(h["bg"])) - (k["mean"] - np.mean(k["bg"]))
    y0, y1, _, _ = h["regions"]["water"]
    c0, c1 = h["regions"]["half_width_cols"]
    prof = d[y0:y1].mean(0)
    peak = prof[(c0 + c1) // 2]
    esf = prof[c0 - 100:(c0 + c1) // 2] / peak  # rising edge, board -> cup
    lsf = np.clip(np.gradient(esf), 0, None)
    lsf /= lsf.sum()
    x = np.arange(len(lsf)) - int(np.argmax(lsf))

    def frac_inside(width: float) -> float:
        return float(lsf[np.abs(x) <= width / 2].sum())

    out = {}
    for case in ("hot_water_50", "hot_water_95"):
        r = sets[case]["regions"]
        w = r["half_width_cols"][1] - r["half_width_cols"][0] + 1
        hgt = r["plateau_rows"][1] - r["plateau_rows"][0] + 1 + 8  # plateau (>= 90 %) is narrower than the band
        fx, fy = frac_inside(w), frac_inside(hgt)
        kept = fx * (fy + 0.5 * (1 - fy))
        out[case] = {"width_px": int(w), "height_px": int(hgt), "fraction_kept": round(kept, 4)}
    out["relative_loss_95_vs_50"] = round(1 - out["hot_water_95"]["fraction_kept"] / out["hot_water_50"]["fraction_kept"], 4)
    out["lsf_10_90_px"] = int(np.argmax(np.cumsum(lsf) > 0.9) - np.argmax(np.cumsum(lsf) > 0.1))
    return out


def plot(sets, report, g_joint, g_c, g_h, apply) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = cc.band_l
    l_air = float(L(cc.c2k(cc.T_AIR_C)))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    eps = EPS_NOMINAL
    dl = np.linspace(-12, 50, 50)
    for g, lab, ls in ((g_joint, "joint", "-"), (g_c, "cold point only", ":"), (g_h, "hot point only", "--")):
        ax[0].plot(dl, g * dl, ls, color="0.3", lw=1, label=f"G = {g:.1f} ({lab})")
    colors = {"hot_water_50": "#c0392b", "hot_water_95": "#e67e22", "cold_water_50": "#2471a3", "cold_water_95": "#5dade2"}
    for c, t_c in (("cold_water_50", cc.T_COLD_C), ("hot_water_50", cc.T_HOT_RANGE_C[1])):
        x = eps * (float(L(cc.c2k(t_c))) - l_air)
        dc = report["calibration_points_50cm"][c]["delta_counts"]
        ax[0].plot([x], [dc], "o", color=colors[c], ms=8, label=f"{c} (fit), {t_c:g} C")
    for c, t_c in (("cold_water_95", cc.T_COLD_C), ("hot_water_95", cc.T_HOT_RANGE_C[0])):
        x = eps * (float(L(cc.c2k(t_c))) - l_air)
        dc = report["results"][c]["delta_counts_first_last_mean"][1 if c.startswith("hot") else 0]
        ax[0].plot([x], [dc], "s", mfc="none", color=colors[c], ms=8, label=f"{c} (test), {t_c:g} C")
    ax[0].axhline(0, color="0.8", lw=0.8)
    ax[0].axvline(0, color="0.8", lw=0.8)
    ax[0].set_xlabel(r"$\varepsilon\,[L_B(T_s) - L_B(T_{air})]$  (W m$^{-2}$ sr$^{-1}$), $\varepsilon$ = 0.93")
    ax[0].set_ylabel("cup - board (counts)")
    ax[0].set_title("Radiance-linear calibration, board as per-frame offset")
    ax[0].legend(fontsize=7)
    t_ref = min(sets[c]["t"][0] for c in cc.CASES)
    for c in cc.CASES:
        s = sets[c]
        _, ts = apply(s["delta"], g_joint, EPS_NOMINAL)
        ax[1].plot((s["t"] - t_ref) / 60, ts, ".-", color=colors[c], label=c)
    ax[1].axhspan(*cc.T_HOT_RANGE_C, color="#c0392b", alpha=0.08, label="hot reading 58-66 C")
    ax[1].axhline(cc.T_COLD_C, color="#2471a3", lw=0.8, ls=":", label="cold reading 11 C")
    ax[1].set_xlabel("minutes after 07:18:48 JST")
    ax[1].set_ylabel("calibrated surface T (C), eps = 0.93, joint G")
    ax[1].set_title("Time series: the hot water cools, the cold water warms")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(cc.OUT / "task1_calibration.png", dpi=130)


if __name__ == "__main__":
    main()
