#!/usr/bin/env python3
"""One pose at supersample 1, 2 and 4: full frame, two enlarged crops, and the measured frame rate.

    make_sheet.py <whitehot|jet> <out.png> [frame]

Reads the thermal PNGs of outputs/comparison_ref[_k2|_k4][_whitehot]/ and the reports of the runs
that wrote them. The crops are nearest-neighbour enlargements: every block is one detector pixel.

    A  the left edge of HotPlate (90 C), its whole height. The camera is pitched 8 degrees, so this
       vertical edge leans 1.76 degrees in the image at frame 47: 5 px over its 164 rows, one pixel
       column every ~33 rows. Enlarged x8 across and x2 along, or the lean would not show.
    B  HotSpot (150 C, 25 px across): an edge at every orientation. x5 both ways.
    bottom row: |this frame - the k = 4 frame| x4. The noise field is the same draw at every k
       (same seed, same frame index, added on the detector grid), so what is left is the geometry.
"""
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUTS = pathlib.Path("/home/jumasaet/isaac-thermal-camera-simulation/outputs")
REPORTS = pathlib.Path("/home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/results/rebase_2026-10-22/ref_out")
pal = sys.argv[1]
out = sys.argv[2]
frame = int(sys.argv[3]) if len(sys.argv) > 3 else 47
suffix = "_whitehot" if pal == "whitehot" else ""
KS = [(1, "comparison_ref"), (2, "comparison_ref_k2"), (4, "comparison_ref_k4")]
CROPS = [("A", (44, 136, 84, 296), (8, 2)), ("B", (362, 308, 426, 372), (5, 5))]  # (u0, v0, u1, v1), (zoom x, zoom y)
CROP_PX = 320
DIFF_ROWS = (100, 400)  # the rows of the full frame the difference panel shows
DIFF_GAIN = 4
W, H = 640, 512
PAD, TEXT_H = 8, 96
COLORS = {"A": (0, 255, 255), "B": (255, 0, 255)}


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if pathlib.Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def fps_of(k_dir):
    """The timed pass of the jet run and of the white-hot run of this k: (fps list, report)."""
    reps = [json.loads((REPORTS / (k_dir + s) / "report.json").read_text()) for s in ("", "_whitehot")]
    passes = [p for r in reps for p in r["passes"] if p["mode"] == "timed"]
    return passes, reps[0]


col_w = W
DIFF_H = DIFF_ROWS[1] - DIFF_ROWS[0]
sheet = Image.new("RGB", (3 * col_w + 4 * PAD, PAD + 28 + H + PAD + CROP_PX + PAD + 22 + DIFF_H + PAD + TEXT_H), (24, 24, 24))
REF = np.asarray(Image.open(OUTS / (KS[-1][1] + suffix) / "thermal" / f"frame_{frame:05d}.png").convert("RGB")).astype(np.int16)
dr = ImageDraw.Draw(sheet)
f_big, f_small = font(22), font(15)
for i, (k, d) in enumerate(KS):
    x0 = PAD + i * (col_w + PAD)
    im = Image.open(OUTS / (d + suffix) / "thermal" / f"frame_{frame:05d}.png").convert("RGB")
    assert im.size == (W, H)
    passes, rep = fps_of(d)
    rp = rep.get("render_product", [W, H])
    dr.text((x0, PAD), f"k = {k}   render {rp[0]}x{rp[1]} -> detector {W}x{H}   frame {frame}", fill=(235, 235, 235), font=f_small)
    y0 = PAD + 28
    full = im.copy()
    fd = ImageDraw.Draw(full)
    for name, box, _z in CROPS:
        fd.rectangle(box, outline=COLORS[name], width=1)
        fd.text((box[2] + 3, box[1]), name, fill=COLORS[name], font=f_small)
    sheet.paste(full, (x0, y0))
    y1 = y0 + H + PAD
    for j, (name, box, (zx, zy)) in enumerate(CROPS):
        c = im.crop(box).resize(((box[2] - box[0]) * zx, (box[3] - box[1]) * zy), Image.NEAREST)
        assert c.size == (CROP_PX, CROP_PX), c.size
        cx = x0 + j * CROP_PX
        sheet.paste(c, (cx, y1))
        dr.rectangle((cx, y1, cx + CROP_PX - 1, y1 + CROP_PX - 1), outline=COLORS[name], width=2)
        dr.text((cx + 6, y1 + 4), f"{name}  x{zx}" if zx == zy else f"{name}  x{zx} horiz, x{zy} vert", fill=COLORS[name], font=f_small)
    y_d = y1 + CROP_PX + PAD
    diff = np.abs(np.asarray(im).astype(np.int16) - REF).max(axis=-1)
    if k == KS[-1][0]:
        dr.text((x0, y_d), "k = 4: la referencia de las dos diferencias", fill=(160, 160, 160), font=f_small)
    else:
        dr.text((x0, y_d), f"|k = {k} - k = 4| x{DIFF_GAIN}   ({int((diff > 16).sum())} px a mas de 16 codigos; "
                f"{100 * (diff == 0).mean():.0f} % de pixeles identicos)", fill=(235, 235, 235), font=f_small)
        panel = np.clip(diff[DIFF_ROWS[0]:DIFF_ROWS[1]] * DIFF_GAIN, 0, 255).astype(np.uint8)
        sheet.paste(Image.fromarray(panel).convert("RGB"), (x0, y_d + 22))
    y2 = y_d + 22 + DIFF_H + PAD
    fps = [p["fps"]["mean"] for p in passes]
    ms = [p["ms"]["total"]["mean"] for p in passes]
    mean_fps = len(fps) / sum(1.0 / f for f in fps)
    dr.text((x0, y2), f"{mean_fps:.1f} fps", fill=(255, 255, 255), font=f_big)
    dr.text((x0 + 130, y2 + 6), f"({sum(ms) / len(ms):.1f} ms/frame; pasadas: " + ", ".join(f"{f:.1f}" for f in fps) + ")",
            fill=(210, 210, 210), font=f_small)
    dr.text((x0, y2 + 32), f"min {min(p['fps']['min'] for p in passes):.1f}  max {max(p['fps']['max'] for p in passes):.1f}  "
            f"lag 1 en {sum(p['freshness']['lag_hist'].get('1', 0) for p in passes)}/{240 * len(passes)}", fill=(210, 210, 210), font=f_small)
    dr.text((x0, y2 + 54), "headless, ROS2 + suscriptor, RGB + termico, flat field + M9", fill=(160, 160, 160), font=f_small)
sheet.save(out)
print("wrote", out, sheet.size)
