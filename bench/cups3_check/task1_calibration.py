#!/usr/bin/env python
"""Task 1: counts -> radiance from the hot cup, the cold cup and the board; the warm cup is the independent test.

    /home/jumasaet/isaac_sim/python.sh bench/cups3_check/task1_calibration.py

Model (docs/physics-model.md §3.2 (b), §9.3: the bolometer is linear in band radiance, not in kelvin):

    c = G * L_app + O(t)

  - board: L_app = L_B(T_air). Closed room, board and everything it reflects at the air temperature, so its
    apparent radiance is L_B(T_air) whatever its emissivity. O(t) is the board plane at the cup (per frame, so
    FFC steps and drift drop out; cups3_common.board_plane).
  - cup: L_app = eps L_B(T_s) + (1 - eps) L_B(T_air), so  dc = c_cup - c_board = G eps (L_B(T_s) - L_B(T_air)).
    The data fix the product G eps. The surface temperature T_s of the same paper does not depend on eps;
    the apparent temperature (eps = 1 reading) does, through G = (G eps) / eps.
  - T_s = T_air + k (T_water - T_air): k < 1 is the drop across the wet paper wall (water-side film + paper
    vs the outside film). Nominal k = 1. Any k common to the three cups is absorbed into G eps to first order,
    so the warm-cup test barely sees it; the Monte Carlo carries it.
  - T_water per frame: linear in time between the first frame (before reading) and the last (after reading)
    -- the stated assumption. The data contradict it (see `in_recording_change`); scenario B drops it.

Calibration: G eps per cup = mean over frames of dc / (L_B(T_s) - L_B(T_air)); joint G eps = inverse-variance
mean of the two (variance from 0.5 K on each reading). Then for the warm cup, per frame:

    T_s,warm   = L_B^-1( L_B(T_air) + dc_warm / (G eps) )            (eps-free; compare with the probe)
    T_app,warm = L_B^-1( L_B(T_air) + dc_warm / G ),  G = G eps / 0.93 (compare with L_B^-1(eps L_B(T_w) + ...))

Writes out/task1_calibration.json, out/task1_calibration.png, out/real_apparent_t_mean.npy (time-mean calibrated
apparent-temperature image, K, float32), out/real_apparent_t_frames.npy (every frame, float32; local only).
"""
from __future__ import annotations

import json

import numpy as np

import cups3_common as c3

EPS_PAPER = 0.93  # bench/cups_check/materials/paper_cup.yaml (ESTIMATED, range 0.90-0.96)
SIGMA_READING_K = 0.5
SIGMA_AIR_K = 0.5
N_MC = 20000


def levels_variants(frames: np.ndarray, regions: dict, coef: np.ndarray) -> dict:
    """dc per cup and frame for three water ROIs: the plateau's central half (nominal), its upper half and its
    lower half. The cups are stratified (hot/warm warmest near the top, cold coldest near the bottom) and the
    probe's depth is unknown, so the ROI is an uncertainty."""
    out = {}
    for name in ("nominal", "upper", "lower"):
        reg = {}
        for cup, r in regions.items():
            y0, y1, x0, x1 = r["water"]
            ym = (y0 + y1) // 2
            w = {"nominal": (y0, y1), "upper": (y0, ym), "lower": (ym, y1)}[name]
            reg[cup] = dict(r, water=(w[0], w[1], x0, x1))
        lv = c3.per_frame_levels(frames, reg, coef)
        out[name] = {cup: lv[cup]["delta"] for cup in c3.CUPS}
    return out


def dl(t_s_c: np.ndarray, t_air_c: float, eps: float = 1.0) -> np.ndarray:
    return eps * (c3.band_l(c3.c2k(t_s_c)) - c3.band_l(c3.c2k(t_air_c)))


def calibrate(dc: dict, tw: dict, t_air: float, k: float = 1.0, t_bg: float | None = None) -> dict:
    """Joint G eps from cold + hot, and the warm cup's surface temperature per frame.

    t_bg: the board's apparent temperature if it is not T_air (the paper still reflects T_air)."""
    t_bg = t_air if t_bg is None else t_bg
    ts = {cup: t_air + k * (tw[cup] - t_air) for cup in c3.CUPS}

    def x(cup, geps_unused=None):
        # dc = G eps [L(T_s) - L(T_air)] + G [L(T_air) - L(T_bg)]; for the board-offset term take eps -> its
        # nominal value (it is a correction of the order of 0.1 K).
        return dl(ts[cup], t_air) + (c3.band_l(c3.c2k(t_air)) - c3.band_l(c3.c2k(t_bg))) / EPS_PAPER

    g = {cup: float(np.mean(dc[cup] / x(cup))) for cup in c3.CALIBRATION_CUPS}
    # inverse-variance weights from SIGMA_READING_K on the water temperature
    w = {}
    for cup in c3.CALIBRATION_CUPS:
        tm = float(np.mean(ts[cup]))
        dldt = float((c3.band_l(c3.c2k(tm) + 0.05) - c3.band_l(c3.c2k(tm) - 0.05)) / 0.1)
        rel = dldt * k * SIGMA_READING_K / abs(float(np.mean(x(cup))))
        w[cup] = 1.0 / rel ** 2
    geps = sum(w[c] * g[c] for c in g) / sum(w.values())
    off = (c3.band_l(c3.c2k(t_air)) - c3.band_l(c3.c2k(t_bg))) / EPS_PAPER
    ts_warm = c3.band_l_inv(c3.band_l(c3.c2k(t_air)) + dc["warm"] / geps - off) - 273.15
    return {"geps": geps, "geps_per_cup": g, "weights": w, "ts": ts, "ts_warm_meas": ts_warm}


def apparent_c(dc: np.ndarray, geps: float, t_air: float, eps: float) -> np.ndarray:
    return c3.band_l_inv(c3.band_l(c3.c2k(t_air)) + dc * eps / geps) - 273.15


def expected_apparent_c(t_s: np.ndarray, t_air: float, eps: float) -> np.ndarray:
    return c3.band_l_inv(eps * c3.band_l(c3.c2k(t_s)) + (1 - eps) * c3.band_l(c3.c2k(t_air))) - 273.15


def main() -> None:
    frames, t, names = c3.load_thermal()
    tr = t - t[0]
    mean = frames.mean(0)
    regions = c3.find_cups(mean)
    mask = c3.board_mask(regions)
    coef = c3.board_plane(frames, mask)
    var = levels_variants(frames, regions, coef)
    dc = var["nominal"]
    tw = {cup: c3.probe_t_c(cup, t) for cup in c3.CUPS}

    # -- nominal ------------------------------------------------------------------------------------------
    nom = calibrate(dc, tw, c3.T_AIR_C)
    geps = nom["geps"]
    err_s = nom["ts_warm_meas"] - nom["ts"]["warm"]
    t_app_w = apparent_c(dc["warm"], geps, c3.T_AIR_C, EPS_PAPER)
    t_app_exp = expected_apparent_c(nom["ts"]["warm"], c3.T_AIR_C, EPS_PAPER)
    err_app = t_app_w - t_app_exp
    g_cold, g_hot = nom["geps_per_cup"]["cold"], nom["geps_per_cup"]["hot"]
    # Birge-like ratio: disagreement of the two single-cup gains over its 1-sigma from the readings
    sig_rel = {c: 1 / np.sqrt(nom["weights"][c]) for c in c3.CALIBRATION_CUPS}
    birge = abs(np.log(g_hot / g_cold)) / np.hypot(sig_rel["cold"], sig_rel["hot"])
    sitf_300 = geps / EPS_PAPER * float((c3.band_l(300.05) - c3.band_l(299.95)) / 0.1)
    # What each cup's temperature does during the recording, read off the calibration itself (T_s, eps-free)
    change = {}
    for cup in c3.CUPS:
        ts_meas = c3.band_l_inv(c3.band_l(c3.c2k(c3.T_AIR_C)) + dc[cup] / geps) - 273.15
        slope = np.polyfit(tr, ts_meas, 1)[0]
        late = tr > 2.0  # after the first ~1.5 s transient (see RESULTADOS.md)
        rate = np.polyfit(tr[late], ts_meas[late], 1)[0] * 60.0
        a, b = c3.T_PROBE_C[cup]
        change[cup] = {"measured_k_over_recording": round(float(slope * tr[-1]), 3),
                       "rate_k_per_min_after_2s": round(float(rate), 3),
                       "minutes_for_the_reading_change_at_that_rate": round(float((b - a) / rate), 1),
                       "assumed_k_over_recording": round(b - a, 2),
                       "measured_mean_c": round(float(ts_meas.mean()), 2),
                       "assumed_mean_c": round(float(tw[cup].mean()), 2)}
    # Each cup's apparent temperature per frame (eps = 0.93 -> G), for task 2
    app = {cup: apparent_c(dc[cup], geps, c3.T_AIR_C, EPS_PAPER) for cup in c3.CUPS}
    app_exp = {cup: expected_apparent_c(nom["ts"][cup], c3.T_AIR_C, EPS_PAPER) for cup in c3.CUPS}

    # -- one factor at a time (warm-cup error in T_s, frame mean) -----------------------------------------
    def warm_err(dc_=dc, tw_=tw, t_air=c3.T_AIR_C, k=1.0, t_bg=None):
        r = calibrate(dc_, tw_, t_air, k, t_bg)
        return float(np.mean(r["ts_warm_meas"] - r["ts"]["warm"]))

    def shifted(cup, dk):
        return {c: (tw[c] + dk if c == cup else tw[c]) for c in c3.CUPS}

    base = warm_err()
    oat = {
        "hot reading +-0.5 K": [warm_err(tw_=shifted("hot", -0.5)), warm_err(tw_=shifted("hot", 0.5))],
        "cold reading +-0.5 K": [warm_err(tw_=shifted("cold", -0.5)), warm_err(tw_=shifted("cold", 0.5))],
        "warm reading +-0.5 K": [warm_err(tw_=shifted("warm", -0.5)), warm_err(tw_=shifted("warm", 0.5))],
        "air 23.4 / 24.4 C": [warm_err(t_air=c3.T_AIR_C - 0.5), warm_err(t_air=c3.T_AIR_C + 0.5)],
        "board apparent T_air -+0.5 K": [warm_err(t_bg=c3.T_AIR_C - 0.5), warm_err(t_bg=c3.T_AIR_C + 0.5)],
        "wall factor k 0.85 / 1.0": [warm_err(k=0.85), base],
        "ROI upper / lower half": [warm_err(dc_=var["upper"]), warm_err(dc_=var["lower"])],
    }
    # Readings at the ends of a longer interval: each cup anywhere between its two readings, constant
    frac = {c: tw[c] * 0 for c in c3.CUPS}
    oat["timing: all at 'before' / all at 'after'"] = [
        warm_err(tw_={c: frac[c] + c3.T_PROBE_C[c][0] for c in c3.CUPS}),
        warm_err(tw_={c: frac[c] + c3.T_PROBE_C[c][1] for c in c3.CUPS})]
    oat = {k: [round(v[0] - base, 3), round(v[1] - base, 3)] for k, v in oat.items()}

    # -- Monte Carlo --------------------------------------------------------------------------------------
    rng = np.random.default_rng(20261028)

    def mc(scenario: str) -> dict:
        es, ea, gs = [], [], []
        roi_names = list(var)
        for _ in range(N_MC):
            t_air = c3.T_AIR_C + rng.normal(0, SIGMA_AIR_K)
            t_bg = t_air + rng.normal(0, 0.3)
            k = rng.uniform(0.85, 1.0)
            eps = rng.uniform(0.90, 0.96)
            dc_ = var[roi_names[rng.integers(3)]]
            tw_ = {}
            for cup in c3.CUPS:
                a, b = (v + rng.normal(0, SIGMA_READING_K) for v in c3.T_PROBE_C[cup])
                if scenario == "A":
                    tw_[cup] = a + (b - a) * tr / tr[-1]
                else:  # B: anywhere between the readings, constant over the 29 s
                    tw_[cup] = np.full_like(tr, a + (b - a) * rng.uniform())
            r = calibrate(dc_, tw_, t_air, k, t_bg)
            es.append(float(np.mean(r["ts_warm_meas"] - r["ts"]["warm"])))
            app_w = apparent_c(dc_["warm"], r["geps"], t_air, eps)
            ea.append(float(np.mean(app_w - expected_apparent_c(r["ts"]["warm"], t_air, eps))))
            gs.append(r["geps"])
        es, ea, gs = map(np.array, (es, ea, gs))
        pc = lambda a: [round(float(x), 3) for x in np.percentile(a, [2.5, 16, 50, 84, 97.5])]  # noqa: E731
        return {"warm_err_surface_k_p2.5_16_50_84_97.5": pc(es), "warm_err_apparent_k_p2.5_16_50_84_97.5": pc(ea),
                "geps_p2.5_16_50_84_97.5": pc(gs), "warm_err_surface_std_k": round(float(es.std()), 3)}

    mc_a, mc_b = mc("A"), mc("B")

    # -- calibrated images ---------------------------------------------------------------------------------
    # One offset per frame for the whole image: the board plane averaged over the board mask (the board's own
    # ~0.4 K left-right gradient is kept as data, not flattened).
    yy, xx = np.nonzero(mask)
    off = c3.plane_at(coef, yy, xx).mean(axis=1)
    g_app = geps / EPS_PAPER
    t_img = (c3.band_l_inv(c3.band_l(c3.c2k(c3.T_AIR_C)) + (frames - off[:, None, None]) / g_app)).astype(np.float32)
    assert t_img.dtype == np.float32
    np.save(c3.OUT / "real_apparent_t_frames.npy", t_img)
    np.save(c3.OUT / "real_apparent_t_mean.npy", t_img.mean(0).astype(np.float32))
    np.savez(c3.OUT / "task1_per_frame.npz", t_unix=t, names=np.array(names), dc_cold=dc["cold"], dc_warm=dc["warm"],
             dc_hot=dc["hot"], **{f"app_{c}": app[c] for c in c3.CUPS}, **{f"app_exp_{c}": app_exp[c] for c in c3.CUPS},
             **{f"tw_{c}": tw[c] for c in c3.CUPS}, board_offset=off)

    rep = {
        "frames": len(t), "duration_s": round(float(tr[-1]), 3), "t_air_c": c3.T_AIR_C, "probe_c": c3.T_PROBE_C,
        "eps_paper_for_apparent": EPS_PAPER, "regions": regions,
        "calibration": {"geps_counts_per_w_m2_sr": round(geps, 3),
                        "geps_cold_only": round(g_cold, 3), "geps_hot_only": round(g_hot, 3),
                        "hot_over_cold_minus_1_pct": round(100 * (g_hot / g_cold - 1), 2),
                        "birge_ratio": round(float(birge), 2),
                        "weights_rel": {c: round(float(v / sum(nom["weights"].values())), 3) for c, v in nom["weights"].items()},
                        "g_with_eps_0.93": round(geps / EPS_PAPER, 3), "sitf_counts_per_k_at_300k": round(sitf_300, 2),
                        "contrast_ratio_hot_over_cold_measured": round(float(np.mean(dc["hot"]) / -np.mean(dc["cold"])), 4),
                        "contrast_ratio_predicted_radiance": round(float(np.mean(dl(tw["hot"], c3.T_AIR_C)) / -np.mean(dl(tw["cold"], c3.T_AIR_C))), 4),
                        "contrast_ratio_predicted_kelvin": round(float(np.mean(tw["hot"] - c3.T_AIR_C) / -np.mean(tw["cold"] - c3.T_AIR_C)), 4)},
        "warm_test": {
            "surface_t_c_mean": round(float(nom["ts_warm_meas"].mean()), 3),
            "probe_c_mean": round(float(nom["ts"]["warm"].mean()), 3),
            "error_surface_k_mean": round(float(err_s.mean()), 3),
            "error_surface_k_first_last_frame": [round(float(err_s[0]), 3), round(float(err_s[-1]), 3)],
            "error_surface_k_frame_std": round(float(err_s.std()), 3),
            "apparent_t_c_mean": round(float(t_app_w.mean()), 3),
            "expected_apparent_t_c_mean": round(float(t_app_exp.mean()), 3),
            "error_apparent_k_mean": round(float(err_app.mean()), 3),
            "error_apparent_k_first_last_frame": [round(float(err_app[0]), 3), round(float(err_app[-1]), 3)],
        },
        "apparent_t_c_mean_per_cup": {c: round(float(app[c].mean()), 3) for c in c3.CUPS},
        "expected_apparent_t_c_mean_per_cup": {c: round(float(app_exp[c].mean()), 3) for c in c3.CUPS},
        "in_recording_change": change,
        "one_at_a_time_shift_of_warm_error_k": oat,
        "mc_A_linear_in_time": mc_a, "mc_B_anywhere_between_readings": mc_b,
        "mc_inputs": {"readings_sigma_k": SIGMA_READING_K, "air_sigma_k": SIGMA_AIR_K, "board_vs_air_sigma_k": 0.3,
                      "wall_factor_k": [0.85, 1.0], "eps": [0.90, 0.96], "roi": list(var), "n": N_MC},
        "board_plane_residual_rms_counts": round(float(np.std(frames[:, yy, xx] - c3.plane_at(coef, yy, xx))), 2),
    }
    (c3.OUT / "task1_calibration.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("calibration", "warm_test", "in_recording_change",
                                          "one_at_a_time_shift_of_warm_error_k", "mc_A_linear_in_time",
                                          "mc_B_anywhere_between_readings", "apparent_t_c_mean_per_cup",
                                          "expected_apparent_t_c_mean_per_cup")}, indent=1))
    plot(tr, dc, tw, nom, t_app_w, t_app_exp, geps, rep)


def plot(tr, dc, tw, nom, t_app_w, t_app_exp, geps, rep) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"cold": "#2166ac", "warm": "#b8860b", "hot": "#b2182b"}
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axs[0]
    xs = np.linspace(-20, 50, 10)
    ax.plot(xs, geps * xs, "k-", lw=0.8, label=f"fit (cold + hot): G·ε = {geps:.2f}")
    for cup in c3.CUPS:
        x = dl(nom["ts"][cup], c3.T_AIR_C)
        ax.plot(x, dc[cup], ".", ms=3, color=col[cup], label=f"{cup}" + (" (test)" if cup == "warm" else ""))
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel("L_B(T_probe) - L_B(T_air)  [W m⁻² sr⁻¹]")
    ax.set_ylabel("counts - board plane")
    ax.set_title("Calibration: hot, cold, board; warm cup held out", fontsize=10)
    ax.legend(fontsize=8)
    ax = axs[1]
    ax.plot(tr, nom["ts"]["warm"], "-", color="k", lw=1, label="probe, linear in time (assumed)")
    ax.plot(tr, nom["ts_warm_meas"], ".", ms=3, color=col["warm"], label="calibrated surface T (ε-free)")
    ax.plot(tr, t_app_exp, "--", color="0.4", lw=1, label="expected apparent (ε 0.93)")
    ax.plot(tr, t_app_w, ".", ms=3, color="#7f3b08", label="calibrated apparent (ε 0.93)")
    w = rep["warm_test"]
    ax.set_title(f"Warm cup (independent): surface error {w['error_surface_k_mean']:+.2f} K, "
                 f"apparent {w['error_apparent_k_mean']:+.2f} K", fontsize=10)
    ax.set_xlabel("time since first frame [s]")
    ax.set_ylabel("°C")
    ax.legend(fontsize=8)
    ax = axs[2]
    for cup in c3.CUPS:
        ts_meas = c3.band_l_inv(c3.band_l(c3.c2k(c3.T_AIR_C)) + dc[cup] / geps) - 273.15
        ax.plot(tr, ts_meas - ts_meas[0], ".", ms=3, color=col[cup], label=f"{cup}: calibrated")
        ax.plot(tr, tw[cup] - tw[cup][0], "-", color=col[cup], lw=1, label=f"{cup}: assumed")
    ax.set_title("Change during the 29 s: data vs linear-in-time assumption", fontsize=10)
    ax.set_xlabel("time since first frame [s]")
    ax.set_ylabel("K relative to the first frame")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(c3.OUT / "task1_calibration.png", dpi=110)


if __name__ == "__main__":
    main()
