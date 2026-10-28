#!/usr/bin/env python
"""Task 4: the cups' reflection in the table -- measured only (not simulated).

    /home/jumasaet/isaac_sim/python.sh bench/cups3_check/task4_reflection.py

DEFINITION (counts are linear in band radiance, so ratios of count differences are radiance ratios; no gain,
no emissivity and no temperature enter):

  For each cup, with b = the cup's bottom row (its contact line with the table, measured on the mean frame,
  out/scene_geometry.json) and the columns of its water ROI (central half of its width):

      rho(d) = [ c(b + d) - c_table(b + d) ] / [ c(b - d) - c_board(b - d) ]

  - c(r): median counts of row r over those columns;
  - c(b + d): the table at d rows BELOW the contact line, where a mirror would show the cup's wall at d rows
    ABOVE it (camera 3.3 cm over a flat table: the reflected image is the cup flipped about its contact row);
  - c_table(r): the table's own level on that row -- median of two strips 20-45 px outside the cup's 50 %
    outline, left and right (the warm cup's right strip is dropped if the steel ruler crosses it, see
    `ruler_check`);
  - c_board(r): the board plane (cups3_common.board_plane) on that row at the cup's centre.

  Reported per cup: the profile rho(d) and

      rho_near   = median of rho(d) for d = 15..27 rows  (the lower wall, ~1.2-2.2 cm up; ~85-86 deg incidence)
      rho_mirror = median of rho(d) for d = 15 .. (b - top of the water ROI)  (the whole wetted wall)

  d starts at 15 rows so that the cup's own blurred bottom edge (Gaussian sigma ~5-6 px) leaks < 1 % into the
  reflection rows. For a smooth, specular table at T_air in an isothermal room rho is the table's specular
  reflectance at that incidence angle (diluted by the blur); 0 means no reflection.

  The same numbers on the simulated frames (out/sim/apparent_t_*.npy, converted to band radiance) show what the
  simulator gives today (rough plywood, environment-only reflections: expect ~0).

Writes out/task4_reflection.json and out/task4_reflection.png.
"""
from __future__ import annotations

import json

import numpy as np

import cups3_common as c3

D_NEAR = (15, 27)
D_MIN = 15
STRIP = (20, 45)


def incidence_deg(d_rows: np.ndarray, x_m: float, h_m: float, f_px: float) -> np.ndarray:
    """Incidence angle at the table of the ray that sees the reflection of the wall point z = d x / f."""
    z = d_rows * x_m / f_px
    return np.degrees(np.arctan2(x_m, h_m + z))


def rho_profile(img: np.ndarray, coef_row, r: dict, sil: dict, ruler_drop: bool) -> dict:
    """img: (H, W) frame in a radiance-linear unit; coef_row(row, col) -> board level."""
    b = int(round(sil["bottom_row"]))
    y0, _y1, x0, x1 = r["water"]
    c0, c1 = r["half_width_cols"]
    ds = np.arange(D_MIN, b - y0 + 1)
    num, den, tab_l, tab_r = [], [], [], []
    xc = 0.5 * (x0 + x1 - 1)
    for d in ds:
        rr, ru = b + d, b - d
        refl = np.median(img[rr, x0:x1])
        left = np.median(img[rr, c0 - STRIP[1]:c0 - STRIP[0]])
        right = np.median(img[rr, c1 + STRIP[0]:c1 + STRIP[1]])
        tab = left if ruler_drop else 0.5 * (left + right)
        num.append(refl - tab)
        den.append(np.median(img[ru, x0:x1]) - coef_row(ru, xc))
        tab_l.append(left)
        tab_r.append(right)
    num, den = np.array(num), np.array(den)
    rho = num / den
    near = (ds >= D_NEAR[0]) & (ds <= D_NEAR[1])
    return {"d": ds, "rho": rho, "rho_near": float(np.median(rho[near])), "rho_mirror": float(np.median(rho)),
            "table_left_minus_right": float(np.median(np.array(tab_l) - np.array(tab_r))), "num": num, "den": den}


def main() -> None:
    frames, t, _ = c3.load_thermal()
    geo = json.loads((c3.OUT / "scene_geometry.json").read_text())
    regions = geo["regions"]
    for r in regions.values():
        for k in ("water", "half_width_cols", "plateau_rows"):
            r[k] = tuple(r[k])
    sil = {cup: geo["silhouettes"][cup]["measured"] for cup in c3.CUPS}
    mask = c3.board_mask(regions)
    coef = c3.board_plane(frames, mask)
    mean = frames.mean(0)
    coef_mean = c3.board_plane(mean[None], mask)
    # Ruler check: the warm cup's two table strips against each other, against the cold cup's (no ruler there)
    tmp = {cup: rho_profile(mean, lambda y, x: float(c3.plane_at(coef_mean, y, x)[0]), regions[cup], sil[cup], False)
           for cup in c3.CUPS}
    lr = {cup: round(tmp[cup]["table_left_minus_right"], 2) for cup in c3.CUPS}
    ruler_drop = {cup: False for cup in c3.CUPS}
    ruler_drop["warm"] = abs(lr["warm"]) > 3 * max(abs(lr["cold"]), 2.0)
    res, per_frame = {}, {cup: {"near": [], "mirror": []} for cup in c3.CUPS}
    for cup in c3.CUPS:
        p = rho_profile(mean, lambda y, x: float(c3.plane_at(coef_mean, y, x)[0]), regions[cup], sil[cup], ruler_drop[cup])
        for i in range(len(frames)):
            q = rho_profile(frames[i], lambda y, x, i=i: float(c3.plane_at(coef[i:i + 1], y, x)[0]), regions[cup], sil[cup],
                            ruler_drop[cup])
            per_frame[cup]["near"].append(q["rho_near"])
            per_frame[cup]["mirror"].append(q["rho_mirror"])
        x_m = geo["lidar"]["cups"][cup]["axis_forward_m"] - c3.CUP_BOTTOM_DIAMETER_M / 2
        h_m = geo["image_pose"]["camera_height_above_table_m"]
        inc = incidence_deg(p["d"], x_m, h_m, c3.K_TH[1, 1])
        near = (p["d"] >= D_NEAR[0]) & (p["d"] <= D_NEAR[1])
        res[cup] = {"bottom_row": sil[cup]["bottom_row"], "water_roi": regions[cup]["water"],
                    "d_range_rows": [int(p["d"][0]), int(p["d"][-1])],
                    "rho_near": round(p["rho_near"], 3), "rho_mirror": round(p["rho_mirror"], 3),
                    "rho_near_frame_std": round(float(np.std(per_frame[cup]["near"])), 3),
                    "rho_mirror_frame_std": round(float(np.std(per_frame[cup]["mirror"])), 3),
                    "rho_max": round(float(p["rho"].max()), 3), "d_at_max": int(p["d"][np.argmax(p["rho"])]),
                    "incidence_deg_near": [round(float(inc[near].max()), 1), round(float(inc[near].min()), 1)],
                    "incidence_deg_mirror": [round(float(inc.max()), 1), round(float(inc.min()), 1)],
                    "cup_contrast_counts_median": round(float(np.median(p["den"])), 1),
                    "table_strips": "left only (ruler)" if ruler_drop[cup] else "left and right",
                    "profile_rho": [round(float(v), 3) for v in p["rho"]]}
    # The simulator's frames: apparent T -> band radiance (linear, like counts)
    sim = {}
    sim_dir = c3.OUT / "sim"
    files = sorted(sim_dir.glob("apparent_t_*.npy"))
    if files:
        simrep = json.loads((sim_dir / "report.json").read_text())
        img = np.mean([c3.band_l(np.load(f).astype(np.float64)) for f in files], axis=0)
        reg_s = c3.find_cups(img)
        sil_s = {cup: simrep["silhouette"][cup] for cup in c3.CUPS}
        coef_s = c3.board_plane(img[None], c3.board_mask(reg_s))
        for cup in c3.CUPS:
            p = rho_profile(img, lambda y, x: float(c3.plane_at(coef_s, y, x)[0]), reg_s[cup], sil_s[cup], False)
            sim[cup] = {"rho_near": round(p["rho_near"], 3), "rho_mirror": round(p["rho_mirror"], 3)}
    rep = {"definition": __doc__.split("DEFINITION", 1)[1].split("Writes", 1)[0].strip(), "d_near_rows": D_NEAR,
           "ruler_check": {"table_left_minus_right_counts": lr, "warm_right_strip_dropped": ruler_drop["warm"]},
           "real": res, "sim": sim}
    (c3.OUT / "task4_reflection.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "definition"}, indent=1))
    plot(mean, res, regions, sil, sim)


def plot(mean, res, regions, sil, sim) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"cold": "#2166ac", "warm": "#b8860b", "hot": "#b2182b"}
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 2, 1)
    lo, hi = 15150, 15950
    ax.imshow(mean[170:380, 30:570], cmap="gray", vmin=lo, vmax=hi, extent=(30, 570, 380, 170))
    for cup in c3.CUPS:
        r = res[cup]
        b = r["bottom_row"]
        y0, _y1, x0, x1 = regions[cup]["water"]
        d0, d1 = r["d_range_rows"]
        ax.add_patch(plt.Rectangle((x0, b + d0), x1 - x0, d1 - d0, fill=False, ec=col[cup], lw=1))
        ax.add_patch(plt.Rectangle((x0, b - d1), x1 - x0, d1 - d0, fill=False, ec=col[cup], lw=1, ls="--"))
        ax.axhline(b, xmin=(x0 - 40 - 30) / 540, xmax=(x1 + 40 - 30) / 540, color="w", lw=0.6)
        c0, c1 = regions[cup]["half_width_cols"]
        for xa, xb in ((c0 - 45, c0 - 20), (c1 + 20, c1 + 45)):
            if xa > c1 and r["table_strips"].startswith("left only"):
                continue
            ax.add_patch(plt.Rectangle((xa, b + d0), xb - xa, d1 - d0, fill=False, ec="w", lw=0.6, ls=":"))
    ax.set_title("Mean real frame (counts, fixed window). Solid: reflection rows; dashed: the wall rows they mirror;\n"
                 "dotted: table reference strips; white line: contact row", fontsize=9)
    ax = fig.add_subplot(1, 2, 2)
    for cup in c3.CUPS:
        r = res[cup]
        d = np.arange(r["d_range_rows"][0], r["d_range_rows"][1] + 1)
        lab = f"{cup}: ρ_near {r['rho_near']:.2f}, ρ_mirror {r['rho_mirror']:.2f}"
        if sim:
            lab += f"  (sim {sim[cup]['rho_near']:.2f} / {sim[cup]['rho_mirror']:.2f})"
        ax.plot(d, r["profile_rho"], "-", color=col[cup], label=lab)
    ax.axvspan(*D_NEAR, color="0.9")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlabel("d = rows from the contact line (reflection at b + d, wall at b − d)")
    ax.set_ylabel("ρ(d) = (refl − table) / (cup − board)")
    ax.set_title("Reflection contrast per cup (grey band: ρ_near rows)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(c3.OUT / "task4_reflection.png", dpi=110)


if __name__ == "__main__":
    main()
