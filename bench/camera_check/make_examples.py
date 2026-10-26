"""Example frames for viewing, all on ONE fixed scale: stored count 15000 -> 0, 16000 -> 255 (clipped).
Same window for every file, so brightness is comparable across frames and across both data sets.

    /home/jumasaet/isaac_sim/python.sh bench/camera_check/make_examples.py
"""
import glob, os
import numpy as np
from PIL import Image, ImageDraw
ROOT = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(ROOT))
OUT = os.path.join(ROOT, "examples"); os.makedirs(OUT, exist_ok=True)
LO, HI = 15000, 16000
lab = sorted(glob.glob(os.path.join(REPO, "thermal_specs/lab/thermal/*.npy")))
cal = sorted(glob.glob(os.path.join(REPO, "thermal_specs/cali_board/thermal/*.npy")))
def g(f, label):
    a = np.load(f).astype(np.float64)
    im = Image.fromarray(np.clip((a - LO) / (HI - LO) * 255, 0, 255).astype(np.uint8)).convert("RGB")
    ImageDraw.Draw(im).text((4, 4), label, fill=(255, 255, 0)); return np.array(im)
def save(name, tiles):
    Image.fromarray(np.hstack(tiles)).save(os.path.join(OUT, name))
save("lab_0000_desk.png", [g(lab[0], "lab 0")])
save("lab_1040_tubes.png", [g(lab[1040], "lab 1040 ceiling tubes")])
save("lab_ffc_step_0940_vs_0941.png", [g(lab[940], "lab 940"), g(lab[941], "lab 941 (+106 counts, 0.7 s later)")])
save("lab_tube_enters_1022_vs_1023.png", [g(lab[1022], "lab 1022"), g(lab[1023], "lab 1023 tube enters")])
save("cal_05_vs_26.png", [g(cal[5], "cali_board 5"), g(cal[26], "cali_board 26")])
# fixed-scale colour bar
bar = np.tile(np.linspace(0, 255, 640).astype(np.uint8), (24, 1))
im = Image.fromarray(bar).convert("RGB"); d = ImageDraw.Draw(im)
d.text((2, 6), str(LO), fill=(255, 255, 0)); d.text((590, 6), str(HI), fill=(0, 0, 0)); im.save(os.path.join(OUT, "scale_15000_16000.png"))
print(sorted(os.listdir(OUT)))
