#!/usr/bin/env python
"""Contact sheets for the user's eyes: thermal, RGB as lit by the brief, RGB two stops down.

Plain Python (PIL + NumPy), no Kit. The last row is NOT what was asked for: it is the same
apparent-temperature frames shown over 15-60 C, to make visible what the 15-160 C range hides.
"""
import pathlib, sys
sys.path.insert(0, "/home/jumasaet/isaac-thermal-camera-simulation/src")
import numpy as np
from PIL import Image, ImageDraw
from irsim.isp.palette import palette_table, quantise_display
O = pathlib.Path("/home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/results/rebase_2026-10-22/demo_out")
A, B = O / "deliver_iso100", O / "deliver_iso25"
FRAMES = (0, 20, 40, 60)
W, H, PAD, LAB = 640, 512, 6, 150
pal = palette_table("ironbow")

def bar(lo, hi, h=H):
    """A vertical colour bar with tick labels, hot at the top."""
    im = Image.new("RGB", (LAB, h), (20, 20, 20)); d = ImageDraw.Draw(im)
    col = pal[np.linspace(255, 0, h - 40).round().astype(int)]
    im.paste(Image.fromarray(np.repeat(col[:, None, :], 26, axis=1)), (14, 20))
    for t in np.linspace(lo, hi, 6):
        y = 20 + (h - 41) * (1 - (t - lo) / (hi - lo))
        d.text((48, y - 6), f"{t:.0f} C", fill=(230, 230, 230))
    return im

def alt(traj, j, lo=15.0, hi=60.0):
    t = np.load(A / traj / "stills" / f"apparent_t_k_{j:03d}.npy").astype(np.float64) - 273.15
    return Image.fromarray(pal[quantise_display((t - lo) / (hi - lo))])

for traj in ("linear", "circular"):
    rows = [("termico 15-160 C (lo pedido)", lambda j: Image.open(A / traj / "stills" / f"thermal_{j:03d}.png"), bar(15, 160)),
            ("RGB, luces del encargo, exposicion Kit (filmIso 100)", lambda j: Image.open(A / traj / "stills" / f"rgb_{j:03d}.png"), None),
            ("RGB, mismas luces, exposicion -2 EV (filmIso 25)", lambda j: Image.open(B / traj / "stills" / f"rgb_{j:03d}.png"), None),
            ("NO pedido: mismo termico mostrado en 15-60 C", lambda j: alt(traj, j), bar(15, 60))]
    sheet = Image.new("RGB", (LAB + len(FRAMES) * (W + PAD), len(rows) * (H + 22 + PAD)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    for r, (title, get, cb) in enumerate(rows):
        y = r * (H + 22 + PAD)
        d.text((LAB + 4, y + 4), f"{traj}: {title}   | frames {FRAMES}", fill=(255, 255, 255))
        if cb is not None:
            sheet.paste(cb, (0, y + 22))
        for c, j in enumerate(FRAMES):
            sheet.paste(get(j).convert("RGB"), (LAB + c * (W + PAD), y + 22))
    sheet.save(O / f"contact_{traj}.png")
    sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(O / f"contact_{traj}_half.png")
    print("wrote", O / f"contact_{traj}.png", sheet.size)
