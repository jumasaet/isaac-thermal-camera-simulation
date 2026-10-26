#!/usr/bin/env python
"""Task 2: real Cellplus frames next to simulated ones -- a look, not a quantitative comparison.

The scenes differ (a lab, handheld and moving, vs the reference wall scene), so only the general
character is compared: sharpness, amount and kind of noise, row/column banding. Both sides are shown
in COUNTS through a FIXED window, no AGC:

    real   stored counts (16383 - raw, higher = hotter), window 15096-15500: the capture tool's own
           fixed window (bench/camera_check/RESULTADOS.md, test 4)
    sim    dn16, a window of the SAME width (404 counts), placed so the wall sits at the gray level the
           real background has. Same counts per gray level on both sides; whether that is the same
           kelvin per gray level depends on the real counts/K, which is not measured yet.
    sim    the same frames through the task-1 window (15-45 C apparent), for reference

Numbers (noise_metrics.py, the same operators on both sides): temporal noise from a static stack
(real: lab frames 650-663, camera at rest; sim: 12 frames at pose 47), row/column noise, pixel-scale
fixed pattern, isolated pixels, and 10-90 % edge widths of the strongest edges.

    /home/jumasaet/isaac_sim/python.sh bench/camera_preview/compare_real_sim.py
Writes bench/camera_preview/compare/{sheet.png, metrics.json}.
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import noise_metrics as nm  # noqa: E402

REPO = HERE.parent.parent
REAL = sorted(glob.glob(str(REPO / "thermal_specs/lab/thermal/*.npy")))
SIM = HERE / "comparison"
OUT = HERE / "compare"
OUT.mkdir(exist_ok=True)

REAL_FRAMES = [0, 300, 1040]  # positions in the sorted file list (the file prefix can skip lost frames)
REAL_STATIC = (650, 663)
REAL_WIN = (15096.0, 15500.0)
SIM_FRAMES = [int(a) for a in sys.argv[1:]] or [25, 47, 95]
SIM_TASK1_WIN_C = json.loads((SIM / "report.json").read_text())["window_c"]
W, H = 640, 480
PAD = 8


def font(size, mono=False):
    if mono and pathlib.Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf").exists():
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if pathlib.Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def win8(a, lo, hi):
    return np.clip(np.rint(255.0 * (np.asarray(a, np.float64) - lo) / (hi - lo)), 0, 255).astype(np.uint8)


def flattest(img, size=(120, 160), step=20):
    """(rows, cols) slices of the window with the smallest p1-p99 spread: the flattest patch."""
    h, w = size
    best = None
    for r in range(4, img.shape[0] - h - 4, step):
        for c in range(4, img.shape[1] - w - 4, step):
            p = img[r:r + h, c:c + w]
            s = np.percentile(p, 99) - np.percentile(p, 1)
            if best is None or s < best[0]:
                best = (s, r, c)
    return slice(best[1], best[1] + h), slice(best[2], best[2] + w)


def edges(a, n_best=5):
    """Strongest near-vertical edges: per 40-row band, the column of largest mean |dI/dx|; returns
    [(step, col, row0, 10-90 width px)] sorted by step."""
    a = np.asarray(a, np.float64)
    gx = np.abs(np.diff(a, axis=1))
    out = []
    for r0 in range(20, a.shape[0] - 40, 40):
        g = gx[r0:r0 + 40].mean(0)
        c = int(np.argmax(g[12:-12])) + 12
        prof = a[r0:r0 + 40, c - 10:c + 11].mean(0)
        out.append((float(abs(prof[-3:].mean() - prof[:3].mean())), c, r0, nm.edge_width(prof)))
    return sorted(out, reverse=True)[:n_best]


# -- data ------------------------------------------------------------------------------------------
real = {k: np.load(REAL[k]).astype(np.float64) for k in REAL_FRAMES}
real_static = np.stack([np.load(f) for f in REAL[REAL_STATIC[0]:REAL_STATIC[1] + 1]]).astype(np.float64)
sim = {k: np.load(SIM / "stills" / f"frame_{k:05d}_dn16.npy").astype(np.float64) for k in SIM_FRAMES}
sim_t = {k: np.load(SIM / "stills" / f"frame_{k:05d}_apparent_t_k.npy") for k in SIM_FRAMES}
sim_static = np.stack([np.load(f) for f in sorted(glob.glob(str(SIM / "static" / "static_*_dn16.npy")))]).astype(np.float64)
sim_static_t = np.load(sorted(glob.glob(str(SIM / "static" / "static_*_apparent_t_k.npy")))[0])

# Sim window: same width as the real one; the wall (median of the flattest patch of the static
# frame) at the gray level of the real background (median of the real static stretch).
width = REAL_WIN[1] - REAL_WIN[0]
real_bg_frac = (np.median(real_static) - REAL_WIN[0]) / width
roi_sim_flat = flattest(sim_static.mean(0))
wall_dn = float(np.median(sim_static.mean(0)[roi_sim_flat]))
SIM_WIN = (round(wall_dn - real_bg_frac * width), round(wall_dn - real_bg_frac * width) + width)
rep = json.loads((SIM / "report.json").read_text())
dn_per_k = rep["dn_per_k_at_300k"]
wall_c = float(np.median(sim_static_t[roi_sim_flat])) - 273.15
sim_win_c = (wall_c - (wall_dn - SIM_WIN[0]) / dn_per_k, wall_c + (SIM_WIN[1] - wall_dn) / dn_per_k)

# -- metrics ---------------------------------------------------------------------------------------
roi_real = (slice(150, 450), slice(40, 600))  # the static stretch looks at a flat surface almost everywhere
roi_sim = (slice(roi_sim_flat[0].start, roi_sim_flat[0].start + 120), slice(roi_sim_flat[1].start, roi_sim_flat[1].start + 160))
roi_sim_big = flattest(sim_static.mean(0), size=(200, 280), step=20)
metrics = {"real": {"static_frames": list(REAL_STATIC), "roi": [roi_real[0].start, roi_real[0].stop, roi_real[1].start, roi_real[1].stop]},
           "sim": {"static_pose": rep["static_pose"], "roi": [roi_sim_big[0].start, roi_sim_big[0].stop, roi_sim_big[1].start, roi_sim_big[1].stop],
                   "dn_per_k_at_300k": dn_per_k}}
for side, stack, roi in (("real", real_static, roi_real), ("sim", sim_static, roi_sim_big)):
    m = metrics[side]
    m.update(nm.temporal(stack, roi))
    m.update(nm.spatial(stack.mean(0), roi))
    m["pixel_pattern_of_mean"] = nm.pixel_pattern(stack.mean(0), roi)
    m["pixel_pattern_single_frame"] = nm.pixel_pattern(stack[0], roi)
    m["outliers_single_frame_whole"] = nm.outliers(stack[0], (slice(2, -2), slice(2, -2)))
metrics["real"]["edges_10_90_px"] = {k: edges(real[k], 3) for k in REAL_FRAMES}
metrics["sim"]["edges_10_90_px"] = {k: edges(sim[k], 3) for k in SIM_FRAMES}
metrics["windows"] = {"real_counts": REAL_WIN, "sim_dn16": SIM_WIN, "sim_dn16_as_c": [round(v, 2) for v in sim_win_c],
                      "sim_task1_c": SIM_TASK1_WIN_C}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))

# -- sheet -----------------------------------------------------------------------------------------
f_big, f_small, f_mono = font(20), font(15), font(15, mono=True)
ROW_H = 28 + H
ZOOM = 4
CROP = (H // ZOOM, W // ZOOM)  # 120 x 160 detector pixels -> one 640 x 480 tile
TEXT_H = 210
sheet = Image.new("RGB", (3 * W + 4 * PAD, PAD + 4 * (ROW_H + PAD) + TEXT_H), (24, 24, 24))
dr = ImageDraw.Draw(sheet)


def put(img8, col, row, title):
    x, y = PAD + col * (W + PAD), PAD + row * (ROW_H + PAD)
    dr.text((x, y + 4), title, fill=(235, 235, 235), font=f_small)
    im = Image.fromarray(img8) if img8.ndim == 2 else Image.fromarray(img8, "RGB")
    sheet.paste(im.convert("RGB"), (x, y + 28))


for i, k in enumerate(REAL_FRAMES):
    put(win8(real[k], *REAL_WIN), i, 0, f"REAL lab frame {k} | counts {REAL_WIN[0]:.0f}-{REAL_WIN[1]:.0f} (capture tool's window)")
for i, k in enumerate(SIM_FRAMES):
    put(win8(sim[k], *SIM_WIN), i, 1, f"SIM frame {k} | dn16 {SIM_WIN[0]}-{SIM_WIN[1]} (same 404-count width ~ {sim_win_c[0]:.1f}-{sim_win_c[1]:.1f} C)")
    lo, hi = SIM_TASK1_WIN_C
    put(win8(sim_t[k] - 273.15, lo, hi), i, 2, f"SIM frame {k} | apparent T {lo:g}-{hi:g} C (task 1 window)")

# Row 4: x4 nearest-neighbour crops. Flat patches at +-20 counts (noise and banding), then the two edges.
roi_real_flat = flattest(real_static[0])
rmed = np.median(real_static[0][roi_real_flat])
put(np.kron(win8(real_static[0][roi_real_flat], rmed - 20, rmed + 20), np.ones((ZOOM, ZOOM), np.uint8)), 0, 3,
    f"REAL flat patch x{ZOOM}, frame {REAL_STATIC[0]} | median +-20 counts")
smed = np.median(sim_static[0][roi_sim_flat])
put(np.kron(win8(sim_static[0][roi_sim_flat], smed - 20, smed + 20), np.ones((ZOOM, ZOOM), np.uint8)), 1, 3,
    f"SIM flat wall x{ZOOM}, static pose {rep['static_pose']} | median +-20 DN (~ +-{20 / dn_per_k:.2f} K)")
# edge tile: left half the sharpest strong real edge, right half the sharpest strong sim edge, both x4
def edge_crop(a, lo, hi, cands):
    step, c, r0, _w = max(cands[:3], key=lambda e: -e[3])  # among the 3 strongest, the sharpest
    r = slice(max(0, r0 - 40), max(0, r0 - 40) + CROP[0])
    cc = slice(max(0, c - CROP[1] // 4), max(0, c - CROP[1] // 4) + CROP[1] // 2)
    return np.kron(win8(a[r, cc], lo, hi), np.ones((ZOOM, ZOOM), np.uint8)), _w


k_re = min(REAL_FRAMES, key=lambda k: min(e[3] for e in metrics["real"]["edges_10_90_px"][k]))
k_se = min(SIM_FRAMES, key=lambda k: min(e[3] for e in metrics["sim"]["edges_10_90_px"][k]))
er, wr = edge_crop(real[k_re], *REAL_WIN, metrics["real"]["edges_10_90_px"][k_re])
es, ws = edge_crop(sim[k_se], *SIM_WIN, metrics["sim"]["edges_10_90_px"][k_se])
tile = np.zeros((H, W), np.uint8)
tile[:, :er.shape[1]] = er[:H]
tile[:, W // 2:W // 2 + es.shape[1]] = es[:H]
tile[:, W // 2 - 1:W // 2 + 1] = 128
put(tile, 2, 3, f"EDGES x{ZOOM}: real fr {k_re} (10-90 {wr:.1f} px) | sim fr {k_se} ({ws:.1f} px)")

r_, s_ = metrics["real"], metrics["sim"]
lines = [
    "Fixed windows, no AGC.  Real: stored counts.  Sim: dn16 (simulated SITF {:.1f} DN/K at 300 K; real counts/K NOT measured, so 'counts' may not mean the same K).".format(dn_per_k),
    "Static-stack numbers (same operators both sides; counts / DN):      REAL lab 650-663      SIM pose {} ({} fr)".format(rep["static_pose"], s_["n_frames"]),
    "  temporal noise per pixel (sd of cleaned frame differences / sqrt2) {:6.2f}               {:6.2f}".format(r_["sigma_t_sd"], s_["sigma_t_sd"]),
    "  row noise, temporal  /  column noise, temporal                     {:5.2f} / {:5.2f}        {:5.2f} / {:5.2f}".format(r_["sigma_row_t"], r_["sigma_col_t"], s_["sigma_row_t"], s_["sigma_col_t"]),
    "  row pattern, fixed  /  column pattern, fixed (upper bounds)        {:5.2f} / {:5.2f}        {:5.2f} / {:5.2f}".format(r_["sigma_row_pattern"], r_["sigma_col_pattern"], s_["sigma_row_pattern"], s_["sigma_col_pattern"]),
    "  pixel-scale fixed pattern (high-pass of the stack mean)            {:6.2f}               {:6.2f}".format(r_["pixel_pattern_of_mean"], s_["pixel_pattern_of_mean"]),
    "  isolated pixels > 50 counts off their 3x3 median, one frame (/1e6) {:7.0f}             {:8.0f}".format(r_["outliers_single_frame_whole"]["per_million"], s_["outliers_single_frame_whole"]["per_million"]),
    "  sharpest strong edge, 10-90 % (px)                                {:6.1f}               {:6.1f}   (real handheld: includes motion blur)".format(wr, ws),
]
y = PAD + 4 * (ROW_H + PAD)
for ln in lines:
    dr.text((PAD, y), ln, fill=(235, 235, 235), font=f_mono)
    y += 24
sheet.save(OUT / "sheet.png")
print(json.dumps({k: v for k, v in metrics.items()}, indent=1, default=float)[:4000])
print(f"wrote {OUT / 'sheet.png'}")
