#!/usr/bin/env python
"""Task 8: the cups' reflection in the table, added to the simulated frames without Kit (refl_common.py).

    /home/jumasaet/isaac_sim/python.sh bench/cups3_check/task8_reflection.py step0   # ~1.5 min
    /home/jumasaet/isaac_sim/python.sh bench/cups3_check/task8_reflection.py fit     # ~10 min
    /home/jumasaet/isaac_sim/python.sh bench/cups3_check/task8_reflection.py final   # ~1.5 min
    /home/jumasaet/isaac_sim/python.sh bench/cups3_check/task8_reflection.py ruler   # ~1.5 min
    then task8_similarity.py (conda `vision`, as task 3) for SSIM / LPIPS, and `figure` for out/task8_reflection.png.

Needs out/sim/gbuffer.npz (task8_gbuffer.py, Kit, once). Each step adds its section to out/task8_reflection.json.

step0 -- ESTIMATE ONLY, not a model: L_mirror is the simulated scene flipped about each column's first table row
         (the reflection of a camera 3.3 cm over a flat table is nearly that), rho_s with the preliminary
         sigma_h 5.3 um (profile-only fit, no lens blur). Frames in out/refl/step0/.
fit   -- sigma_h through the simulated camera: the ray-cast reflection (refl_common.mirror_raycast) for each
         sigma_h on a grid, the IDEAL chain (no noise), one instant, measured with task 4's rho(d) and compared with
         the real profile (rows d <= refl_common.D_MAX_FIT). Fitted on two cups and tested on the third, in the
         three combinations, then on all three. n = 1.5 throughout (not identifiable at 80-85 deg).
ruler -- hypothesis test for the hot cup's second lobe (task 4): `final` plus the steel ruler of the RGB frames,
         mapped onto the thermal image through the table plane, as a smooth mirror (rho 0.9). Frames in
         out/refl/raycast_ruler/.
final -- the ray-cast reflection at the all-cup sigma_h through the full chain (noise, seed 0 as task2_render)
         at the eight instants. Frames in out/refl/raycast/; the comparison frames without reflection are Kit's
         out/sim/ (the replay reproduces them bit for bit: max |delta| 0.0 K).
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import refl_common as rc

c3 = rc.c3
JSON = c3.OUT / "task8_reflection.json"
SIGMA_GRID_UM = np.round(np.arange(3.0, 7.01, 0.25), 2)


def update(section: str, data: dict) -> None:
    rep = json.loads(JSON.read_text()) if JSON.exists() else {}
    rep[section] = data
    JSON.write_text(json.dumps(rep, indent=1))
    print(json.dumps({section: data}, indent=1), flush=True)


def profiles_summary(sim: dict, real: dict) -> dict:
    out = {}
    for cup in c3.CUPS:
        out[cup] = {
            "rho_near": round(sim[cup]["rho_near"], 3),
            "rho_mirror": round(sim[cup]["rho_mirror"], 3),
            "rms_vs_real": round(rc.profile_rms(sim, real, cup), 4),
            "profile_rho": [round(float(v), 3) for v in sim[cup]["rho"]],
            "d0": int(sim[cup]["d"][0]),
        }
    return out


def setup():
    g = rc.load_gbuffer()
    cfg, lut, q, t0, resp = rc.build_chain()
    le = rc.l_env(cfg, lut, q, t0)
    tg = rc.table_geometry(g)
    return g, cfg, lut, q, t0, resp, le, tg


def step0() -> None:
    g, cfg, lut, q, t0, resp, le, tg = setup()
    l_scene = rc.scene_radiance(g, g["temperature_k"], cfg, lut, q, le)
    lm = rc.mirror_image_space(g, l_scene)
    rho_s = rc.rho_specular(tg["cos"], rc.SIGMA_H_PRELIM_UM, resp)
    add = np.zeros(g["ids"].shape)
    add[tg["mask"]] = rho_s * (lm - le)
    outs = rc.replay(rc.planes_for(g, rc.inject(g, cfg, lut, q, le, add)), cfg, t0)
    rc.save_frames("step0", outs)
    real = rc.real_profiles()
    update(
        "step0_image_space",
        {
            "sigma_h_um": rc.SIGMA_H_PRELIM_UM,
            "n": rc.N_TABLE,
            "rho_s_by_incidence_deg": rc.rho_s_table(rc.SIGMA_H_PRELIM_UM, resp),
            "profiles": profiles_summary(rc.sim_profiles([o.apparent_t for o in outs]), real),
        },
    )


def mirror_field(g, tg, cfg, lut, q, le):
    obj_l = rc.object_radiance(g, cfg, lut, q, le)
    lm, counts = rc.mirror_raycast(g, tg, obj_l, le)
    return lm, counts, obj_l


def fit() -> None:
    g, _cfg, lut, q, t0, resp, le, tg = setup()
    cfg, *_ = rc.build_chain(noise=False)
    lm, counts, _obj = mirror_field(g, tg, cfg, lut, q, le)
    real = rc.real_profiles()
    rms = {cup: [] for cup in c3.CUPS}
    near = {cup: [] for cup in c3.CUPS}
    t_start = time.time()
    for s in SIGMA_GRID_UM:
        add = np.zeros(g["ids"].shape)
        add[tg["mask"]] = rc.rho_specular(tg["cos"], s, resp) * (lm - le)
        cfg, *_ = rc.build_chain(noise=False)  # a fresh chain: its housing node cannot step back in time
        out = rc.replay(rc.planes_for(g, rc.inject(g, cfg, lut, q, le, add)), cfg, t0, all_instants=False)
        sim = rc.sim_profiles([out[0].apparent_t])
        for cup in c3.CUPS:
            rms[cup].append(rc.profile_rms(sim, real, cup))
            near[cup].append(sim[cup]["rho_near"])
        print(
            f"sigma_h {s:.2f} um: rms "
            + " ".join(f"{c} {rms[c][-1]:.4f}" for c in c3.CUPS)
            + f"  ({time.time() - t_start:.0f} s)",
            flush=True,
        )
    grid = SIGMA_GRID_UM
    r = {cup: np.array(v) for cup, v in rms.items()}

    def best(cups):
        tot = np.sqrt(np.mean([r[c] ** 2 for c in cups], axis=0))
        i = int(np.argmin(tot))
        if 0 < i < len(grid) - 1:  # parabola through the three grid points around the minimum
            y0, y1, y2 = tot[i - 1 : i + 2]
            den = y0 - 2 * y1 + y2
            sh = 0.5 * (y0 - y2) / den if den > 0 else 0.0
            return float(grid[i] + sh * (grid[1] - grid[0])), float(y1)
        return float(grid[i]), float(tot[i])

    def rms_at(cup, s):
        return float(np.interp(s, grid, r[cup]))

    loo = {}
    for test in c3.CUPS:
        train = [c for c in c3.CUPS if c != test]
        s_fit, _ = best(train)
        s_own, rms_own = best([test])
        loo[test] = {
            "train": train,
            "sigma_h_fit_um": round(s_fit, 2),
            "test_rms_at_fit": round(rms_at(test, s_fit), 4),
            "test_own_best_sigma_h_um": round(s_own, 2),
            "test_own_best_rms": round(rms_own, 4),
            "test_rms_without_reflection": round(
                float(np.sqrt(np.mean(real[test]["rho"][real[test]["d"] <= rc.D_MAX_FIT[test]] ** 2))), 4
            ),
        }
    s_all, rms_all = best(list(c3.CUPS))
    update(
        "fit_sigma_h",
        {
            "n": rc.N_TABLE,
            "grid_um": grid.tolist(),
            "chain": "ideal (no noise), one instant",
            "d_max_rows": rc.D_MAX_FIT,
            "rms_by_cup": {c: [round(v, 4) for v in r[c]] for c in c3.CUPS},
            "rho_near_by_cup": {c: [round(v, 3) for v in near[c]] for c in c3.CUPS},
            "leave_one_out": loo,
            "all_cups": {"sigma_h_um": round(s_all, 2), "rms": round(rms_all, 4)},
            "mirror_hits_table_pixels": counts,
            "rho_s_by_incidence_deg_at_fit": rc.rho_s_table(s_all, resp),
        },
    )


def final() -> None:
    g, cfg, lut, q, t0, resp, le, tg = setup()
    s = json.loads(JSON.read_text())["fit_sigma_h"]["all_cups"]["sigma_h_um"]
    lm, counts, obj_l = mirror_field(g, tg, cfg, lut, q, le)
    add = np.zeros(g["ids"].shape)
    add[tg["mask"]] = rc.rho_specular(tg["cos"], s, resp) * (lm - le)
    outs = rc.replay(rc.planes_for(g, rc.inject(g, cfg, lut, q, le, add)), cfg, t0)
    rc.save_frames("raycast", outs)
    real = rc.real_profiles()
    base = [np.load(c3.OUT / "sim" / f"apparent_t_{j}.npy") for j in range(8)]
    update(
        "final_raycast",
        {
            "sigma_h_um": s,
            "n": rc.N_TABLE,
            "mirror_hits_table_pixels": counts,
            "object_radiance_w_m2_sr": {k: round(v, 4) for k, v in obj_l.items()},
            "l_env_w_m2_sr": round(le, 4),
            "profiles_with_reflection": profiles_summary(rc.sim_profiles([o.apparent_t for o in outs]), real),
            "profiles_without_reflection": profiles_summary(rc.sim_profiles(base), real),
        },
    )


def ruler() -> None:
    """Hypothesis test for the hot cup's second lobe: the final model plus the steel ruler as a mirror."""
    g, cfg, lut, q, t0, resp, le, tg = setup()
    s = json.loads(JSON.read_text())["fit_sigma_h"]["all_cups"]["sigma_h_um"]
    lm, _counts, _obj = mirror_field(g, tg, cfg, lut, q, le)
    m, info = rc.ruler_mask_thermal(g)
    k = g["ids"].shape[0] // 480
    on_ruler = np.kron(m, np.ones((k, k), bool))[tg["mask"]]
    rho = np.where(on_ruler, rc.RHO_STEEL, rc.rho_specular(tg["cos"], s, resp))
    add = np.zeros(g["ids"].shape)
    add[tg["mask"]] = rho * (lm - le)
    outs = rc.replay(rc.planes_for(g, rc.inject(g, cfg, lut, q, le, add)), cfg, t0)
    rc.save_frames("raycast_ruler", outs)
    np.save(rc.REFL / "raycast_ruler" / "ruler_mask.npy", m)
    update(
        "ruler_hypothesis",
        {
            "sigma_h_um": s,
            "rho_steel": rc.RHO_STEEL,
            "ruler": info,
            "profiles": profiles_summary(rc.sim_profiles([o.apparent_t for o in outs]), rc.real_profiles()),
        },
    )


def figure() -> None:
    """out/task8_reflection.png: real / sim without / sim with the ray-cast reflection (same window), the
    residual real - sim with the ruler outline, and rho(d) per cup against the real profile."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage

    rep = json.loads((c3.OUT / "sim" / "report.json").read_text())
    real_all = np.load(c3.OUT / "real_apparent_t_frames.npy", mmap_mode="r")
    idx = [r["real_frame_index"] for r in rep["renders"]]
    t_real = np.mean([np.asarray(real_all[i], np.float64) for i in idx], 0)
    t_base = np.mean([np.load(c3.OUT / "sim" / f"apparent_t_ref_{j}.npy") for j in range(8)], 0)
    t_refl = np.mean([np.load(rc.REFL / "raycast" / f"apparent_t_ref_{j}.npy") for j in range(8)], 0)
    d = json.loads(JSON.read_text())
    sim_p = d["final_raycast"]["profiles_with_reflection"]
    real = rc.real_profiles()
    ruler = np.load(rc.REFL / "raycast_ruler" / "ruler_mask.npy")
    edge = ruler ^ ndimage.binary_erosion(ruler)
    sl, ext = (slice(170, 390), slice(30, 600)), (30, 600, 390, 170)
    fig = plt.figure(figsize=(16, 9))
    for k, (img, title) in enumerate(
        (
            (t_real, "real (mean of the 8 instants)"),
            (t_base, "simulated, no reflection"),
            (t_refl, f"simulated, ray-cast table reflection (sigma_h {d['final_raycast']['sigma_h_um']} um)"),
        )
    ):
        ax = fig.add_subplot(3, 2, 2 * k + 1)
        ax.imshow(img[sl] - 273.15, cmap="inferno", vmin=20, vmax=40, extent=ext)
        ax.set_title(title + ", apparent T 20-40 C", fontsize=9)
    ax = fig.add_subplot(3, 2, 2)
    res = c3.band_l(t_real) - c3.band_l(t_refl)
    im = ax.imshow(res[sl], cmap="RdBu_r", vmin=-1.5, vmax=1.5, extent=ext)
    yy, xx = np.nonzero(edge[sl])
    ax.plot(xx + 30, yy + 170, "k.", ms=0.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="W m-2 sr-1")
    ax.set_title("real - sim with reflection (band radiance); black: steel ruler mapped from the RGB", fontsize=9)
    ax = fig.add_subplot(3, 2, (4, 6))
    col = {"cold": "#2166ac", "warm": "#b8860b", "hot": "#b2182b"}
    for cup in c3.CUPS:
        ax.plot(real[cup]["d"], real[cup]["rho"], "-", color=col[cup], label=f"{cup} real")
        ds = np.arange(sim_p[cup]["d0"], sim_p[cup]["d0"] + len(sim_p[cup]["profile_rho"]))
        ax.plot(
            ds,
            sim_p[cup]["profile_rho"],
            "--",
            color=col[cup],
            label=f"{cup} sim (rms {sim_p[cup]['rms_vs_real']:.3f})",
        )
    ax.axvline(rc.D_MAX_FIT["hot"], color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("d = rows from the contact line")
    ax.set_ylabel("rho(d), task 4 definition")
    ax.set_title(
        "Table reflection per cup: real (solid) and ray-cast sim (dashed); simulated without reflection: 0", fontsize=9
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(c3.OUT / "task8_reflection.png", dpi=100)


if __name__ == "__main__":
    {"step0": step0, "fit": fit, "final": final, "ruler": ruler, "figure": figure}[sys.argv[1]]()
