"""Contact sheet: thermal over RGB for a few frames. usage: contact.py <outputs dir> <png> [frames...]"""
import sys
from PIL import Image, ImageDraw
d, out = sys.argv[1], sys.argv[2]
frames = [int(x) for x in sys.argv[3:]] or [0, 20, 40, 60, 75, 90, 105, 119]
s = 2
w, h = 640 // s, 512 // s
sheet = Image.new("RGB", (w * len(frames), h * 2))
dr = ImageDraw.Draw(sheet)
for i, k in enumerate(frames):
    for r, name in enumerate(("thermal", "rgb")):
        im = Image.open(f"{d}/{name}/frame_{k:05d}.png").convert("RGB").resize((w, h), Image.BILINEAR)
        sheet.paste(im, (i * w, r * h))
    dr.text((i * w + 4, 4), f"{k}", fill=(255, 255, 0))
sheet.save(out)
