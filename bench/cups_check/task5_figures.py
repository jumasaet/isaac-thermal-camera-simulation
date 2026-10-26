#!/usr/bin/env python
"""Task 5: side-by-side sheets, real calibrated cup against the simulated one, for slides.

    /home/jumasaet/isaac_sim/python.sh bench/cups_check/task5_figures.py     (after task1 and the renders)

Real: the blackbody apparent temperature of ONE frame, the one the temperature reading refers to (first frame of
hot_50, cold_50, cold_95; last frame of hot_95 -- the task-1 time-order assumption), with the task-1 calibration
T_app = L_B^-1(L_B(T_air) + (c - c_board) / G) and that frame's own board reference (docs/physics-model.md §3.2 (b),
§3.3). Not the set mean that task 1 saves as apparent_t_<case>.npy: hot_50 cools 2.3 K during its series.
Simulated: the camera chain's apparent_t of the sim_<case> render (calibrated YAML; PSF and noise included),
which is within 0.1 K of its scene_apparent_t over the water.

Each number is the MEAN apparent temperature over the water region: the task-1 rectangle on the real frame, the
task-3 wetted-wall core (3 px clear of its edges, central half of its width) on the render. RESULTADOS.md quotes
the median of the same regions; the two differ by at most 0.3 K (hot_50, where the region has a gradient).

Both images of a sheet share one fixed temperature window and one palette. The window is per water temperature
(hot / cold), identical for 50 and 95 cm, so the two distances of one water compare directly.

Writes figures/cups_<case>_<palette>.png and cups_summary_<palette>.png for palette in (whitehot, jet).
"""
from __future__ import annotations

import json

import numpy as np

import cups_common as cc

FIG = cc.HERE / "figures"
REF_FRAME = {"hot_water_50": 0, "hot_water_95": -1, "cold_water_50": 0, "cold_water_95": 0}
T_SET = {"hot_water_50": 66.0, "hot_water_95": 58.0, "cold_water_50": 11.0, "cold_water_95": 11.0}
ORDER = ("hot_water_50", "hot_water_95", "cold_water_50", "cold_water_95")
LABEL = {"hot_water_50": "Agua caliente, 50 cm", "hot_water_95": "Agua caliente, 95 cm",
         "cold_water_50": "Agua fría, 50 cm", "cold_water_95": "Agua fría, 95 cm"}
WINDOW_C = {"hot": (15.0, 70.0), "cold": (8.0, 26.0)}
PALETTES = {"whitehot": "gray", "jet": "jet"}
CROP_W, CROP_H = 256, 192  # same pixel box on both images, centred on the real water region


def fmt(x: float, nd: int = 1, sign: bool = False) -> str:
    """Spanish decimal comma."""
    return (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}").replace(".", ",").replace("-", "\u2212")


def real_frame_c(case: str, t1: dict) -> tuple[np.ndarray, tuple, float]:
    """(apparent T image in C, water region, frame time s) of the reference frame."""
    frames, times = cc.load_case(case)
    i = REF_FRAME[case]
    f = frames[i]
    reg = t1["regions_px_y0_y1_x0_x1"][case]
    bg = np.median(np.concatenate([cc.roi(f, reg["bg_left"]).ravel(), cc.roi(f, reg["bg_right"]).ravel()]))
    g = t1["gain_counts_per_w_m2_sr"]["joint_geometric_mean"]
    l_air = float(cc.band_l(cc.c2k(cc.T_AIR_C)))
    t_k = cc.band_l_inv(l_air + (f - bg) / g).astype(np.float32)
    assert t_k.dtype == np.float32
    return t_k - np.float32(273.15), tuple(reg["water"]), float(times[i])


def sim_frame_c(case: str) -> tuple[np.ndarray, np.ndarray]:
    """(camera apparent T image in C, water-core mask) of the sim_<case> render."""
    d = cc.OUT / f"sim_{case}"
    t_k = np.load(d / "apparent_t_k.npy")
    assert t_k.dtype == np.float32
    scene = np.load(d / "scene_apparent_t_k.npy")
    ids = np.load(d / "ids.npy")
    # The instance names were not saved with ids.npy. The wetted wall is the small label furthest from the air
    # temperature (the dry band sits half-way, the board, table and void at the air temperature).
    small = [i for i in np.unique(ids) if 0 < (ids == i).sum() < 20000]
    wet_id = max(small, key=lambda i: abs(np.median(scene[ids == i]) - 273.15 - cc.T_AIR_C))
    wet = ids == wet_id
    core = wet.copy()
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            core &= np.roll(np.roll(wet, dy, 0), dx, 1)
    cols = np.nonzero(wet.any(0))[0]
    c0, c1 = cols.min(), cols.max()
    q = (c1 - c0) // 4
    core[:, :c0 + q] = False
    core[:, c1 - q + 1:] = False
    rep = json.loads((d / "report.json").read_text())
    check = float(np.median(t_k[core])) - 273.15
    assert abs(check - rep["cup_apparent_t_c"]) < 0.01, (case, check, rep["cup_apparent_t_c"])
    return t_k - np.float32(273.15), core


def crop_box(water: tuple) -> tuple[int, int, int, int]:
    y0, y1, x0, x1 = water
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    r0 = int(np.clip(cy - CROP_H // 2, 0, 480 - CROP_H))
    c0 = int(np.clip(cx - CROP_W // 2, 0, 640 - CROP_W))
    return r0, r0 + CROP_H, c0, c0 + CROP_W


def collect() -> dict:
    t1 = json.loads((cc.OUT / "task1_calibration.json").read_text())
    data = {}
    for case in ORDER:
        real, water, t_s = real_frame_c(case, t1)
        sim, core = sim_frame_c(case)
        y0, y1, x0, x1 = water
        data[case] = {"real": real, "sim": sim, "water": water, "core": core, "box": crop_box(water),
                      "real_mean": float(cc.roi(real, water).mean()), "sim_mean": float(sim[core].mean()),
                      "frame": "último" if REF_FRAME[case] == -1 else "primer"}
        data[case]["diff"] = data[case]["real_mean"] - data[case]["sim_mean"]
    return data


def draw_pair(ax_real, ax_sim, d: dict, cmap: str, lo: float, hi: float, size: float = 10.0):
    """Both images on one window, the region the mean is taken over outlined, the mean under each."""
    from matplotlib.patches import Rectangle

    r0, r1, c0, c1 = d["box"]
    ext = (c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5)
    im = None
    for ax, img, mean, name in ((ax_real, d["real"], d["real_mean"], "Real (calibrada)"),
                                (ax_sim, d["sim"], d["sim_mean"], "Simulada")):
        im = ax.imshow(img[r0:r1, c0:c1], cmap=cmap, vmin=lo, vmax=hi, extent=ext, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=size + 1)
        ax.set_xlabel(f"T aparente media del agua: {fmt(mean)} °C", fontsize=size)
    y0, y1, x0, x1 = d["water"]
    for lw, col, ls in ((2.4, "white", "-"), (1.2, "black", (0, (3, 2)))):
        ax_real.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, lw=lw, ec=col, ls=ls))
        ax_sim.contour(np.arange(640), np.arange(480), d["core"].astype(float), levels=[0.5], colors=[col],
                       linewidths=lw, linestyles=[ls])
    ax_sim.set_xlim(ext[0], ext[1])
    ax_sim.set_ylim(ext[2], ext[3])
    return im


def note(case: str) -> str:
    indep = ("calibración ajustada a 50 cm: comparación no independiente" if case.endswith("50")
             else "prueba independiente (la calibración se ajustó a 50 cm)")
    return f"Simulación: superficie a {fmt(T_SET[case], 0)} °C (lectura del termómetro), ε = 0,93 · {indep}"


def sheet(case: str, d: dict, pal: str) -> None:
    import matplotlib.pyplot as plt

    lo, hi = WINDOW_C["hot" if case.startswith("hot") else "cold"]
    fig = plt.figure(figsize=(11.0, 5.4), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], left=0.03, right=0.92, top=0.84, bottom=0.17, wspace=0.06)
    ax_r, ax_s, cax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    im = draw_pair(ax_r, ax_s, d, PALETTES[pal], lo, hi, size=11)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Temperatura aparente (°C), ε = 1", fontsize=10)
    fig.suptitle(f"{LABEL[case]} · diferencia real − simulada: {fmt(d['diff'], 1, True)} K", fontsize=14,
                 fontweight="bold", y=0.96)
    fig.text(0.475, 0.885, f"Real: {d['frame']} frame de la serie, al que se refiere la lectura · ventana fija "
             f"{fmt(lo, 0)}–{fmt(hi, 0)} °C en las dos imágenes", ha="center", fontsize=9.5, color="#444")
    fig.text(0.475, 0.04, note(case) + "\nMedia sobre la región marcada (agua). RESULTADOS.md usa la mediana "
             "(≤ 0,3 K de diferencia).", ha="center", fontsize=8.5, color="#444")
    fig.savefig(FIG / f"cups_{case}_{pal}.png", facecolor="white")
    plt.close(fig)


def summary(data: dict, pal: str) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16.0, 9.0), dpi=150)
    gs = fig.add_gridspec(2, 7, width_ratios=[1, 1, 0.045, 0.16, 1, 1, 0.045], left=0.045, right=0.955,
                          top=0.80, bottom=0.17, wspace=0.08, hspace=0.30)
    cols = {"hot_water_50": 0, "hot_water_95": 1, "cold_water_50": 4, "cold_water_95": 5}
    ims = {}
    for case, c in cols.items():
        d = data[case]
        kind = "hot" if case.startswith("hot") else "cold"
        lo, hi = WINDOW_C[kind]
        ax_r, ax_s = fig.add_subplot(gs[0, c]), fig.add_subplot(gs[1, c])
        ims[kind] = draw_pair(ax_r, ax_s, d, PALETTES[pal], lo, hi, size=10.5)
        ax_r.set_title(f"{LABEL[case]}\nReal (calibrada)", fontsize=12)
        ax_s.set_title(f"Simulada · Δ = {fmt(d['diff'], 1, True)} K", fontsize=12, fontweight="bold")
    for kind, c in (("hot", 2), ("cold", 6)):
        lo, hi = WINDOW_C[kind]
        cb = fig.colorbar(ims[kind], cax=fig.add_subplot(gs[:, c]))
        cb.set_label(f"T aparente (°C) · ventana {fmt(lo, 0)}–{fmt(hi, 0)} °C", fontsize=11)
    row_lbl = {0: "Real", 1: "Simulada"}
    for r, y in ((0, 0.64), (1, 0.31)):
        fig.text(0.012, y, row_lbl[r], rotation=90, va="center", fontsize=14, fontweight="bold")
    fig.suptitle("Vasos de agua: cámara real calibrada frente al simulador (LWIR 8–14 µm)", fontsize=18,
                 fontweight="bold", y=0.965)
    fig.text(0.5, 0.885, "Δ = T aparente media del agua, real − simulada · arriba y abajo, misma ventana y paleta",
             ha="center", fontsize=12, color="#444")
    fig.text(0.5, 0.045,
             "Simulación con la temperatura leída (66 / 58 / 11 / 11 °C) y ε = 0,93. A 50 cm la calibración se "
             "ajustó con esos mismos vasos (no independiente); 95 cm es la prueba.\nA 95 cm el error viene casi "
             "entero de la lectura de entrada: con la temperatura que predice la serie de 50 cm, baja a +0,15 K "
             "(caliente) y +0,5 K (frío). Detalle: bench/cups_check/RESULTADOS.md",
             ha="center", fontsize=10.5, color="#444")
    fig.savefig(FIG / f"cups_summary_{pal}.png", facecolor="white")
    plt.close(fig)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    FIG.mkdir(exist_ok=True)
    data = collect()
    for pal in PALETTES:
        for case in ORDER:
            sheet(case, data[case], pal)
        summary(data, pal)
    rows = {c: {k: round(data[c][k], 2) for k in ("real_mean", "sim_mean", "diff")} for c in ORDER}
    (FIG / "figures.json").write_text(json.dumps({"mean_apparent_t_c_water_region": rows, "window_c": WINDOW_C,
                                                  "reference_frame": REF_FRAME}, indent=1))
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
