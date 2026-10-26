#!/usr/bin/env python
"""Task 5: side-by-side sheets, real calibrated cup against the simulated one, for slides.

    /home/jumasaet/isaac_sim/python.sh bench/cups_check/task5_figures.py [--sim sim6]   (after task1, the renders,
                                                                                    and task6_psf.py on both)

Real: the blackbody apparent temperature of ONE frame, the one the temperature reading refers to (first frame of
hot_50, cold_50, cold_95; last frame of hot_95 -- the task-1 time-order assumption), with the task-1 calibration
T_app = L_B^-1(L_B(T_air) + (c - c_board) / G) and that frame's own board reference (docs/physics-model.md §3.2 (b),
§3.3). Not the set mean that task 1 saves as apparent_t_<case>.npy: hot_50 cools 2.3 K during its series.
Simulated: the camera chain's apparent_t of the <tag>_<case> render (PSF and noise included). The default tag,
sim6, is the CTC-006 camera with the blur measured in task 6 (bench/camera_check/cellplus_ctc006_lwir_cups.yaml);
`--sim sim` redraws the first sheets (diffraction-only optics, cellplus_ctc007_lwir_cups.yaml).
Each sheet also quotes the 10-90 % edge width, real and simulated, measured the same way on both (task6_psf.py,
model-free half-ESF of the cup's side edges).

Each number is the MEAN apparent temperature over the water region, found the same way on both images
(cups_common.find_cup: the plateau >= 90 % of the cup's peak, 2 px in, central half of the 50 % width). Until
task 6 the render used the task-3 wetted-wall core instead; with the measured blur that core reaches into the dry
band's blur, so it no longer measures the same thing as the real rectangle.

Both images of a sheet share one fixed temperature window and one palette. The window is per water temperature
(hot / cold), identical for 50 and 95 cm, so the two distances of one water compare directly.

Writes figures/cups_<case>_<palette>.png and cups_summary_<palette>.png for palette in (whitehot, jet).
`--lang en --out <dir>` writes the same sheets with every label in English (decimal point) to <dir>; the text of
both languages is in TEXT below and nothing else changes between them.
"""
from __future__ import annotations

import json

import numpy as np

import cups_common as cc

FIG = cc.HERE / "figures"
REF_FRAME = {"hot_water_50": 0, "hot_water_95": -1, "cold_water_50": 0, "cold_water_95": 0}
T_SET = {"hot_water_50": 66.0, "hot_water_95": 58.0, "cold_water_50": 11.0, "cold_water_95": 11.0}
ORDER = ("hot_water_50", "hot_water_95", "cold_water_50", "cold_water_95")
WINDOW_C = {"hot": (15.0, 70.0), "cold": (8.0, 26.0)}
PALETTES = {"whitehot": "gray", "jet": "jet"}
CROP_W, CROP_H = 256, 192  # same pixel box on both images, centred on the real water region
SIM_TAG = "sim6"
LANG = "es"
TEXT = {
    "es": {
        "label": {"hot_water_50": "Agua caliente, 50 cm", "hot_water_95": "Agua caliente, 95 cm",
                  "cold_water_50": "Agua fría, 50 cm", "cold_water_95": "Agua fría, 95 cm"},
        "camera": {"sim6": "cámara CTC-006 (14,6 mm F/1,3) con el desenfoque medido: foco ∞ + σ 80 µm",
                   "sim": "óptica solo con difracción (YAML anterior)"},
        "real": "Real (calibrada)", "sim": "Simulada",
        "mean": "T aparente media del agua: {} °C",
        "cbar": "Temperatura aparente (°C), ε = 1",
        "title": "{label} · diferencia real − simulada: {d} K",
        "sub": "Real: {frame} frame de la serie, al que se refiere la lectura · ventana fija {lo}–{hi} °C en las dos "
               "imágenes",
        "first": "primer", "last": "último",
        "edge": "Borde 10–90 %: real {r} px · simulada {s} px",
        "region": "Media sobre la región marcada (meseta del agua, misma definición en las dos imágenes).",
        "note": "Simulación: superficie a {t} °C (lectura del termómetro), ε = 0,93 · {indep}",
        "dep": "calibración ajustada a 50 cm: comparación no independiente",
        "ind": "prueba independiente (la calibración se ajustó a 50 cm)",
        "s_sim": "Simulada · Δ = {d} K", "s_edge": "borde 10–90 %: real {r} · sim {s} px",
        "s_cbar": "T aparente (°C) · ventana {lo}–{hi} °C", "rows": ("Real", "Simulada"),
        "s_title": "Vasos de agua: cámara real calibrada frente al simulador (LWIR 8–14 µm)",
        "s_sub": "Δ = T aparente media del agua, real − simulada · misma ventana y paleta · ",
        "s_foot": "Simulación con la temperatura leída (66 / 58 / 11 / 11 °C) y ε = 0,93. A 50 cm la calibración se "
                  "ajustó con esos mismos vasos (no independiente); 95 cm es la prueba.\nA 95 cm el error viene casi "
                  "entero de la lectura de entrada: con la temperatura que predice la serie de 50 cm, baja a +0,15 K "
                  "(caliente) y +0,5 K (frío). Detalle: bench/cups_check/RESULTADOS.md",
    },
    "en": {
        "label": {"hot_water_50": "Hot water, 50 cm", "hot_water_95": "Hot water, 95 cm",
                  "cold_water_50": "Cold water, 50 cm", "cold_water_95": "Cold water, 95 cm"},
        "camera": {"sim6": "CTC-006 camera (14.6 mm F/1.3) with the measured blur: focus ∞ + σ 80 µm",
                   "sim": "diffraction-only optics (previous YAML)"},
        "real": "Real (calibrated)", "sim": "Simulated",
        "mean": "Mean apparent water temperature: {} °C",
        "cbar": "Apparent temperature (°C), ε = 1",
        "title": "{label} · real − simulated: {d} K",
        "sub": "Real: {frame} frame of the series, the one the thermometer reading refers to · fixed window "
               "{lo}–{hi} °C on both images",
        "first": "first", "last": "last",
        "edge": "10–90 % edge width: real {r} px · simulated {s} px",
        "region": "Mean over the outlined region (water plateau, same definition on both images).",
        "note": "Simulation: surface at {t} °C (thermometer reading), ε = 0.93 · {indep}",
        "dep": "calibration fitted at 50 cm: not an independent comparison",
        "ind": "independent test (the calibration was fitted at 50 cm)",
        "s_sim": "Simulated · Δ = {d} K", "s_edge": "10–90 % edge: real {r} · sim {s} px",
        "s_cbar": "Apparent T (°C) · window {lo}–{hi} °C", "rows": ("Real", "Simulated"),
        "s_title": "Water cups: calibrated real camera vs. simulator (LWIR 8–14 µm)",
        "s_sub": "Δ = mean apparent water temperature, real − simulated · same window and palette · ",
        "s_foot": "Simulated at the thermometer readings (66 / 58 / 11 / 11 °C), ε = 0.93. At 50 cm the calibration "
                  "was fitted on these same cups (not independent); 95 cm is the test.\nAt 95 cm the error comes "
                  "almost entirely from the input reading: fed the temperature the 50 cm series predicts, it drops to "
                  "+0.15 K (hot) and +0.5 K (cold). Details: bench/cups_check/RESULTADOS.md",
    },
}
CAMERA = TEXT["es"]["camera"]  # the --sim choices


def tx(key: str):
    return TEXT[LANG][key]


def fmt(x: float, nd: int = 1, sign: bool = False) -> str:
    """Decimal comma in Spanish, point in English; a true minus sign in both."""
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return (s.replace(".", ",") if LANG == "es" else s).replace("-", "\u2212")


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


def sim_frame_c(case: str, tag: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(camera apparent T image in C, water-core mask) of the <tag>_<case> render."""
    d = cc.OUT / f"{tag or SIM_TAG}_{case}"
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
    # The region the mean is taken over is found exactly as on the real frame (cups_common.find_cup on a quantity
    # linear in radiance): with a blurred camera the wetted-wall core above reaches into the dry band's blur and a
    # differently defined region would compare two different things.
    water = cc.find_cup(cc.band_l(t_k), case.startswith("hot"))["water"]
    region = np.zeros(t_k.shape, bool)
    region[water[0]:water[1], water[2]:water[3]] = True
    return t_k - np.float32(273.15), region


def crop_box(water: tuple) -> tuple[int, int, int, int]:
    y0, y1, x0, x1 = water
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    r0 = int(np.clip(cy - CROP_H // 2, 0, 480 - CROP_H))
    c0 = int(np.clip(cx - CROP_W // 2, 0, 640 - CROP_W))
    return r0, r0 + CROP_H, c0, c0 + CROP_W


def collect() -> dict:
    t1 = json.loads((cc.OUT / "task1_calibration.json").read_text())
    edge_real = json.loads((cc.OUT / "task6_psf.json").read_text())["cases"]
    edge_sim = json.loads((cc.OUT / f"task6_psf_sim_{SIM_TAG}.json").read_text())["cases"]
    data = {}
    for case in ORDER:
        real, water, t_s = real_frame_c(case, t1)
        sim, core = sim_frame_c(case)
        data[case] = {"real": real, "sim": sim, "water": water, "core": core, "box": crop_box(water),
                      "real_mean": float(cc.roi(real, water).mean()), "sim_mean": float(sim[core].mean()),
                      "frame": tx("last") if REF_FRAME[case] == -1 else tx("first"),
                      "edge_real_px": edge_real[case]["core_plus_halo"]["w10_90_px_modelfree"],
                      "edge_sim_px": edge_sim[case]["core_plus_halo"]["w10_90_px_modelfree"]}
        data[case]["diff"] = data[case]["real_mean"] - data[case]["sim_mean"]
        if SIM_TAG != "sim" and (cc.OUT / f"sim_{case}").exists():
            prev, prev_core = sim_frame_c(case, "sim")
            data[case]["sim_mean_previous_camera"] = float(prev[prev_core].mean())
    return data


def draw_pair(ax_real, ax_sim, d: dict, cmap: str, lo: float, hi: float, size: float = 10.0):
    """Both images on one window, the region the mean is taken over outlined, the mean under each."""
    from matplotlib.patches import Rectangle

    r0, r1, c0, c1 = d["box"]
    ext = (c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5)
    im = None
    for ax, img, mean, name in ((ax_real, d["real"], d["real_mean"], tx("real")),
                                (ax_sim, d["sim"], d["sim_mean"], tx("sim"))):
        im = ax.imshow(img[r0:r1, c0:c1], cmap=cmap, vmin=lo, vmax=hi, extent=ext, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=size + 1)
        ax.set_xlabel(tx("mean").format(fmt(mean)), fontsize=size)
    y0, y1, x0, x1 = d["water"]
    for lw, col, ls in ((2.4, "white", "-"), (1.2, "black", (0, (3, 2)))):
        ax_real.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, lw=lw, ec=col, ls=ls))
        ax_sim.contour(np.arange(640), np.arange(480), d["core"].astype(float), levels=[0.5], colors=[col],
                       linewidths=lw, linestyles=[ls], corner_mask=False)
    ax_sim.set_xlim(ext[0], ext[1])
    ax_sim.set_ylim(ext[2], ext[3])
    return im


def note(case: str) -> str:
    indep = tx("dep") if case.endswith("50") else tx("ind")
    return tx("note").format(t=fmt(T_SET[case], 0), indep=indep)


def sheet(case: str, d: dict, pal: str) -> None:
    import matplotlib.pyplot as plt

    lo, hi = WINDOW_C["hot" if case.startswith("hot") else "cold"]
    fig = plt.figure(figsize=(11.0, 5.4), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], left=0.03, right=0.92, top=0.84, bottom=0.17, wspace=0.06)
    ax_r, ax_s, cax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    im = draw_pair(ax_r, ax_s, d, PALETTES[pal], lo, hi, size=11)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(tx("cbar"), fontsize=10)
    fig.suptitle(tx("title").format(label=tx("label")[case], d=fmt(d["diff"], 1, True)), fontsize=14,
                 fontweight="bold", y=0.96)
    fig.text(0.475, 0.885, tx("sub").format(frame=d["frame"], lo=fmt(lo, 0), hi=fmt(hi, 0)), ha="center",
             fontsize=9.5, color="#444")
    fig.text(0.475, 0.025, tx("edge").format(r=fmt(d["edge_real_px"]), s=fmt(d["edge_sim_px"]))
             + f" · {tx('camera')[SIM_TAG]}\n" + note(case) + "\n" + tx("region"), ha="center", fontsize=8.5,
             color="#444")
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
        ax_r.set_title(f"{tx('label')[case]}\n{tx('real')}", fontsize=12)
        ax_s.set_title(tx("s_sim").format(d=fmt(d["diff"], 1, True)), fontsize=12, fontweight="bold")
        ax_s.text(0.5, -0.20, tx("s_edge").format(r=fmt(d["edge_real_px"]), s=fmt(d["edge_sim_px"])),
                  transform=ax_s.transAxes, ha="center", va="top", fontsize=10, color="#444")
    for kind, c in (("hot", 2), ("cold", 6)):
        lo, hi = WINDOW_C[kind]
        cb = fig.colorbar(ims[kind], cax=fig.add_subplot(gs[:, c]))
        cb.set_label(tx("s_cbar").format(lo=fmt(lo, 0), hi=fmt(hi, 0)), fontsize=11)
    row_lbl = dict(enumerate(tx("rows")))
    for r, y in ((0, 0.64), (1, 0.31)):
        fig.text(0.012, y, row_lbl[r], rotation=90, va="center", fontsize=14, fontweight="bold")
    fig.suptitle(tx("s_title"), fontsize=18, fontweight="bold", y=0.965)
    fig.text(0.5, 0.885, tx("s_sub") + tx("camera")[SIM_TAG], ha="center", fontsize=12, color="#444")
    fig.text(0.5, 0.045, tx("s_foot"), ha="center", fontsize=10.5, color="#444")
    fig.savefig(FIG / f"cups_summary_{pal}.png", facecolor="white")
    plt.close(fig)


def main() -> None:
    import argparse

    import matplotlib

    global SIM_TAG, LANG, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=SIM_TAG, choices=sorted(CAMERA))
    ap.add_argument("--lang", default=LANG, choices=sorted(TEXT))
    ap.add_argument("--out", default=None, help="output directory (default bench/cups_check/figures)")
    args = ap.parse_args()
    SIM_TAG, LANG = args.sim, args.lang
    if args.out:
        FIG = cc.pathlib.Path(args.out).resolve()
    matplotlib.use("Agg")
    FIG.mkdir(parents=True, exist_ok=True)
    data = collect()
    for pal in PALETTES:
        for case in ORDER:
            sheet(case, data[case], pal)
        summary(data, pal)
    keys = ("real_mean", "sim_mean", "diff", "sim_mean_previous_camera", "edge_real_px", "edge_sim_px")
    rows = {c: {k: round(data[c][k], 3) for k in keys if k in data[c]} for c in ORDER}
    (FIG / "figures.json").write_text(json.dumps({"sim_tag": SIM_TAG, "camera": tx("camera")[SIM_TAG],
                                                  "mean_apparent_t_c_water_region_and_edges": rows,
                                                  "window_c": WINDOW_C, "reference_frame": REF_FRAME}, indent=1))
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
