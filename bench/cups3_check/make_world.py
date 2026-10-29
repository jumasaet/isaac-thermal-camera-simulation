#!/usr/bin/env python
"""Outside view of the simulated three-cup set-up with both cameras' fields of view drawn (host only, English).

    /home/jumasaet/isaac_sim/python.sh bench/cups3_check/make_world.py [full|none|initial]   (after render_world.py)

The three-cup counterpart of bench/cups_check/presentation/world_<case>.png, drawn with that script's own functions
(make_presentation.fov_border / project / rgb_calibration): each camera's image border -- the TRUE field of view,
undistorted with its calibration (thermal: Brown-Conrady of thermal_specs/cali_board_new; RGB: the RGB intrinsics
and distortion of the same report) -- is cast from its optical centre to the plane through the cups (the mean axis
distance, 0.51 m), and projected into the outside camera.

Cup labels (argument, default all three): full = name and both probe readings -> figures/cups3_world.png;
none -> figures/cups3_world_nolabels.png; initial = only the reading at the start of the recording ->
figures/cups3_world_initial.png.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PRES = HERE.parent / "cups_check" / "presentation"


def load_presentation():
    """make_presentation by path (it imports bench/cups_check's task5_figures, which shares a name with ours)."""
    import sys

    sys.path.insert(0, str(PRES.parent))
    sys.modules.pop("task5_figures", None)
    spec = importlib.util.spec_from_file_location("make_presentation", PRES / "make_presentation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OUTPUT = {"full": "cups3_world.png", "none": "cups3_world_nolabels.png", "initial": "cups3_world_initial.png"}


def main() -> None:
    import sys

    for mode in (sys.argv[1:] or list(OUTPUT)):
        draw(mode)


def draw(mode: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    mp = load_presentation()
    import cups3_common as c3

    d = HERE / "out" / "world"
    cams = json.loads((d / "cams.json").read_text())
    ext = np.asarray(Image.open(d / "ext.png").convert("RGB"))
    e = cams["ext"]
    m_ext = np.array(e["world_from_camera_usd"])
    cups = {c: next(o for o in cams["objects"] if o["name"] == f"{t}Wet")
            for c, t in (("cold", "Cold"), ("warm", "Warm"), ("hot", "Hot"))}
    depth = float(np.mean([-o["centre"][0] for o in cups.values()]))
    k_rgb, dist_rgb = mp.rgb_calibration()
    rigs = (("Thermal camera (Cellplus CTC-006-VGA, 640 × 480)", np.array(cams["thermal"]["world_from_camera_usd"]),
             c3.K_TH, c3.DIST_TH, 640, 480, "#ff5a1f"),
            ("RGB camera (oCam-1CGN, 1280 × 960)", np.array(cams["rgb"]["world_from_camera_usd"]), k_rgb, dist_rgb,
             1280, 960, "#2f7bff"))
    fig, ax = plt.subplots(figsize=(14, 8.9), dpi=130)
    ax.imshow(ext)
    for name, m, k, dist, w, h, col in rigs:
        o = m[:3, 3]
        border, corners = mp.fov_border(m, k, dist, w, h, depth)
        uv = mp.project(m_ext, e["fx_px"], e["w"], e["h"], border)
        uo = mp.project(m_ext, e["fx_px"], e["w"], e["h"], o[None])[0]
        ax.fill(uv[:, 0], uv[:, 1], color=col, alpha=0.12, lw=0)
        ax.plot(np.r_[uv[:, 0], uv[0, 0]], np.r_[uv[:, 1], uv[0, 1]], color=col, lw=1.6, label=name)
        for i in corners:
            ax.plot([uo[0], uv[i, 0]], [uo[1], uv[i, 1]], color=col, lw=1.0, alpha=0.9)
        ax.plot(*uo, "o", color=col, ms=4)
    offsets = {"cold": (-70, 62), "warm": (0, 92), "hot": (70, 62)}  # points; keeps the three labels apart
    for cup, o in cups.items():
        if mode == "none":
            break
        top = np.array([o["centre"][0], o["centre"][1], c3.CUP_HEIGHT_M + 0.012])
        u, v = mp.project(m_ext, e["fx_px"], e["w"], e["h"], top[None])[0]
        a, b = c3.T_PROBE_C[cup]
        text = f"{cup}\n{a:g} / {b:g} °C (mean {c3.T_MEAN_C[cup]:g})" if mode == "full" else f"{c3.T_MEAN_C[cup]:g} °C"
        ax.annotate(text, (u, v), xytext=offsets[cup], textcoords="offset points", ha="center",
                    fontsize=9.5, color="k", arrowprops=dict(arrowstyle="-", color="0.25", lw=0.8),
                    bbox=dict(fc="w", alpha=0.8, lw=0))
    ax.set_xlim(0, e["w"] - 1)
    ax.set_ylim(e["h"] - 1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=11, framealpha=0.85)
    ax.set_title("Simulated world, outside view — three paper cups at 0.5 m", fontsize=15, fontweight="bold")
    b = next(o for o in cams["objects"] if o["name"] == "Background")
    ax.text(0.99, 0.01, f"Fields of view: each camera's image border (true, distorted FOV) drawn on the plane {depth:.2f} m "
            "in front of it, through the cups.\nRGB corners clipped where its distortion model stops being invertible. "
            "Boxes mark the camera centres (not modelled bodies).\n"
            f"Camera {cams['thermal']['position'][2] * 100:.1f} cm above the table, pitched "
            f"{np.degrees(np.arcsin(cams['thermal']['look'][2])):.2f}° up, rolled {cams['thermal']['roll_deg']:.2f}°. "
            f"Board {b['size'][1]:.2f} × {b['size'][2]:.2f} m at {-b['centre'][0] - 0.01:.2f} m (LiDAR).",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="w", bbox=dict(fc="k", alpha=0.45, lw=0))
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / OUTPUT[mode], facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
