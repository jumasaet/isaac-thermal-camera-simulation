#!/usr/bin/env python
"""Numbers and contact sheets from one run_stages.py session. Plain Python (NumPy + PIL), no Kit.

    analyse.py <tag> <traj>:<pose>      e.g.  analyse.py hl circular:54

Everything quoted in the write-up about what an image *looks like* is computed here from the
float planes the runner kept (radiance, signal in DN, the [0, 1] display image), not read off a PNG.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, "/home/jumasaet/isaac-thermal-camera-simulation/src")
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from irsim.config.loader import load_sensor_config
from irsim.isp.palette import palette_table, quantise_display
from irsim.radiometry.lut_files import load_band_lut_for_config

tag, still = sys.argv[1], sys.argv[2]
traj, _, pose = still.partition(":")
pose = int(pose)
ROOT = pathlib.Path("/tmp/irsim_render_cost/stages/out") / tag
D = ROOT / traj
REPORT = json.loads((ROOT / "report.json").read_text())
VARIANTS = ("12", "123", "1236", "12356", "123456", "repo")
NAMES = {"12": "1+2  (lo actual)", "123": "1+2+3  optica", "1236": "1+2+3+6  optica + ISP",
         "12356": "1+2+3+5+6  + ruido", "123456": "1+2+3+4+5+6  completo (config del demo)",
         "repo": "NO pedido: completo como lo configura el repo (flat field + cadena M9)"}
T_LO, T_HI = 273.15 + 15.0, 273.15 + 160.0
DN_PER_K = REPORT["sensor"]["dn_per_k_300"]
SIGMA_TVH = REPORT["sensor"]["sigma_tvh_dn"]

sensor = load_sensor_config("/tmp/irsim_render_cost/demo/out/_sensor_boson_ss1.yaml")
lut = load_band_lut_for_config(sensor, "/home/jumasaet/isaac-thermal-camera-simulation/data/lut")
Q = sensor.sensor.quantity

S = {v: dict(np.load(D / f"still_{v}_{pose:03d}.npz")) for v in VARIANTS}
SLOW = {v: dict(np.load(D / f"slow_{v}_{pose:03d}.npz")) for v in VARIANTS[2:] if (D / f"slow_{v}_{pose:03d}.npz").exists()}
meta = np.load(D / f"meta_{pose:03d}.npz")
IDS = meta["ids"]
LABELS = {int(k): v.split("/")[-1] for k, v in json.loads(str(meta["labels"])).items()}
ID_OF = {v: k for k, v in LABELS.items()}
H, W = IDS.shape


def t_app_c(radiance):
    return np.asarray(lut.apparent_temperature(np.asarray(radiance, np.float32), Q), np.float64) - 273.15


def gray_of(v: str, s: dict) -> np.ndarray:
    """(H, W) uint8 white-hot grey: the fixed 15-160 C map for 12/123, the camera's ISP for the rest."""
    if v in ("12", "123"):
        return quantise_display((t_app_c(s["scene_radiance"]) + 273.15 - T_LO) / (T_HI - T_LO))
    d = s["display8"]
    assert (d[..., 0] == d[..., 1]).all() and (d[..., 1] == d[..., 2]).all(), "isp palette is not grey"
    return d[..., 0]


def linear_plane(v: str, s: dict) -> tuple[np.ndarray, str]:
    """The plane the edge is measured on, before any display mapping."""
    if v in ("12", "123"):
        return s["scene_radiance"].astype(np.float64), "radiancia"
    return s["signal"].astype(np.float64), "senal DN"


GRAY = {v: gray_of(v, S[v]) for v in VARIANTS}

# -- masks ---------------------------------------------------------------------------------------


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            sh = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            out &= sh
    out[:r] = out[-r:] = False
    out[:, :r] = out[:, -r:] = False
    return out


def interior(name: str, r: int = 10) -> np.ndarray:
    return erode(IDS == ID_OF[name], r) if name in ID_OF else np.zeros_like(IDS, bool)


def box_mean(a: np.ndarray, r: int) -> np.ndarray:
    """(2r+1)^2 box mean, wrap at the border (callers stay r pixels inside)."""
    acc = np.zeros_like(a, dtype=np.float64)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            acc += np.roll(np.roll(a, dy, axis=0), dx, axis=1)
    return acc / (2 * r + 1) ** 2


def rsd(x: np.ndarray) -> float:
    """Robust sd (1.4826 MAD): a handful of stuck pixels must not set the number."""
    x = np.asarray(x, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


# -- edges ---------------------------------------------------------------------------------------
HALF = 8


def edge_profiles(plane: np.ndarray, a: str, b: str, side: str) -> np.ndarray:
    """Rows crossing the a|b boundary: (n_rows, 2*HALF) samples, b on the left (index < HALF)."""
    ma, mb = IDS == ID_OF[a], IDS == ID_OF[b]
    rows = np.flatnonzero(ma.any(axis=1))
    out = []
    for r in rows:
        cols = np.flatnonzero(ma[r])
        c = cols[0] if side == "left" else cols[-1] + 1
        lo, hi = c - HALF, c + HALF
        if lo < 0 or hi > W:
            continue
        if side == "left":
            ok = mb[r, lo:c].all() and ma[r, c:hi].all()
            prof = plane[r, lo:hi]
        else:
            ok = ma[r, lo:c].all() and mb[r, c:hi].all()
            prof = plane[r, lo:hi][::-1]
        if ok:
            out.append(prof)
    return np.asarray(out, dtype=np.float64)


def cross(n: np.ndarray, level: float) -> float:
    i = int(np.flatnonzero(n >= level)[0])
    if i == 0:
        return 0.0
    return (i - 1) + (level - n[i - 1]) / (n[i] - n[i - 1])


def edge_stats(profiles: np.ndarray) -> dict | None:
    if len(profiles) < 4:
        return None
    lo = profiles[:, :3].mean(axis=1, keepdims=True)
    hi = profiles[:, -3:].mean(axis=1, keepdims=True)
    n = (profiles - lo) / (hi - lo)
    mean = n.mean(axis=0)  # rows share the pixel phase: the render samples pixel centres
    widths = [cross(row, 0.9) - cross(row, 0.1) for row in n]
    return {"rows": int(len(n)), "w10_90_px_median": float(np.median(widths)),
            "w10_90_px_of_mean_profile": float(cross(mean, 0.9) - cross(mean, 0.1)),
            "overshoot": float(mean.max() - 1.0), "undershoot": float(-mean.min()),
            "mean_profile": [round(float(x), 4) for x in mean]}


EDGES = [("Anomaly", "Floor", "left"), ("Anomaly", "Floor", "right"), ("Anomaly", "Wall", "left"),
         ("Crate_01", "Floor", "left"), ("Crate_01", "Wall", "left"), ("Crate_01", "Wall", "right"),
         ("Crate_02", "Floor", "left"), ("Crate_02", "Wall", "left"), ("Crate_02", "Floor", "right")]
out: dict = {"tag": tag, "still": still, "labels": LABELS, "edges": {}, "texture": {}, "range": {}, "objects": {}}
for a, b, side in EDGES:
    if a not in ID_OF or b not in ID_OF:
        continue
    key = f"{a}|{b}:{side}"
    for v in VARIANTS:
        plane, what = linear_plane(v, S[v])
        st = edge_stats(edge_profiles(plane, a, b, side))
        if st is None:
            continue
        row = {"linear": st, "plane": what}
        if "y" in S[v]:
            row["display"] = edge_stats(edge_profiles(S[v]["y"].astype(np.float64), a, b, side))
        out["edges"].setdefault(key, {})[v] = row

# -- texture on a flat surface ---------------------------------------------------------------------
for name in ("Wall", "Floor"):
    m = interior(name)
    if m.sum() < 2000:
        continue
    mi = erode(m, 3)  # the 5x5 high-pass must not reach across the surface's own boundary
    ys, xs = np.nonzero(m)
    # the largest rectangle is not needed: a 64-row band of the interior is enough for row/column means
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    for v in VARIANTS:
        g = GRAY[v][m].astype(np.float64)
        hp = (GRAY[v].astype(np.float64) - box_mean(GRAY[v].astype(np.float64), 2))[mi]
        row = {"pixels": int(m.sum()), "display_codes_sd": float(g.std()), "display_distinct_codes": int(np.unique(GRAY[v][m]).size),
               "display_code_median": float(np.median(g)),
               "display_local_sd_codes": float(rsd(hp)), "display_local_p99_abs_codes": float(np.percentile(np.abs(hp), 99))}
        if "signal" in S[v]:
            sg = S[v]["signal"].astype(np.float64)
            row["signal_local_sd_mk"] = float(1e3 * rsd((sg - box_mean(sg, 2))[mi]) / DN_PER_K)
        if "signal" in S[v]:
            s = S[v]["signal"].astype(np.float64)
            sub, msub = s[box], m[box]
            full_rows = msub.all(axis=1)
            full_cols = msub.all(axis=0)
            row.update(signal_sd_dn=float(s[m].std()), signal_sd_mk=float(1e3 * s[m].std() / DN_PER_K))
            # stripes: sd of the row / column means over a fully-interior rectangle
            rr = np.flatnonzero(msub.mean(axis=1) > 0.999)
            if rr.size > 16:
                cc = np.flatnonzero(msub[rr].all(axis=0))
                if cc.size > 16:
                    rect = sub[np.ix_(rr, cc)]
                    rect = rect - rect.mean()
                    row.update(rect_shape=[int(rr.size), int(cc.size)],
                               row_means_sd_mk=float(1e3 * rect.mean(axis=1).std() / DN_PER_K),
                               col_means_sd_mk=float(1e3 * rect.mean(axis=0).std() / DN_PER_K),
                               pixel_sd_mk=float(1e3 * rect.std() / DN_PER_K))
        else:
            row["t_app_distinct_values"] = int(np.unique(S[v]["scene_radiance"][m]).size)
            row["t_app_sd_mk"] = float(1e3 * t_app_c(S[v]["scene_radiance"][m]).std())
        out["texture"].setdefault(name, {})[v] = row

# -- grey range ------------------------------------------------------------------------------------
for v in VARIANTS:
    g = GRAY[v]
    out["range"][v] = {
        "distinct_codes": int(np.unique(g).size), "p01": float(np.percentile(g, 1)), "p50": float(np.percentile(g, 50)),
        "p99": float(np.percentile(g, 99)), "min": int(g.min()), "max": int(g.max()),
        "frac_px_in_bottom_10pct": float((g <= 25).mean()), "frac_px_in_top_10pct": float((g >= 230).mean()),
    }
    for ident, name in sorted(LABELS.items()):
        m = interior(name, 3)
        if m.sum() < 50:
            continue
        out["objects"].setdefault(name, {"pixels": int((IDS == ident).sum()),
                                         "t_app_c": float(np.median(t_app_c(S["12"]["scene_radiance"][m])))})[v] = \
            float(np.median(g[m]))

# -- the glow round a hot object: PSF wings or DDE?  Control = same chain with psf=None ---------------
from irsim.detector.quantise import quantise
from irsim.isp.display import run_display_branch
from irsim.materials.library import MaterialLibrary
from irsim.materials.table import MaterialTable
from irsim.optics.stage import apply_optics
from irsim.pipeline.core import PipelineConfig, PipelineState

_cfg = PipelineConfig.from_sensor(sensor, MaterialTable.from_library(MaterialLibrary.load(), sensor.sensor.band.band_id), lut)
_lb_h = float(lut.lookup(PipelineState().housing_temp_k, Q)[()])
_bits = sensor.sensor.fpa.bit_depth


def isp_of(radiance, psf):
    flux = apply_optics(radiance, sensor.sensor, _lb_h, supersample=1, psf=psf)
    return run_display_branch(quantise(_cfg.detector.noiseless_signal_dn(flux), _bits), sensor.sensor.isp, _bits).display8[..., 0]


CONTROL_126 = isp_of(S["12"]["scene_radiance"], None)
REDO_1236 = isp_of(S["12"]["scene_radiance"], _cfg.psf)
out["control"] = {"offline_1236_equals_run_px_differing": int((REDO_1236 != GRAY["1236"]).sum()),
                  "psf_shape": list(_cfg.psf.shape)}
_HALF = HALF
HALF = 28
for a, b, side in (("Anomaly", "Floor", "left"), ("Anomaly", "Floor", "right"), ("HotPipe", "Wall", "left")):
    if a not in ID_OF or b not in ID_OF:
        continue
    pr = {name: edge_profiles(img.astype(np.float64), a, b, side) for name, img in
          (("126_no_psf", CONTROL_126), ("1236", GRAY["1236"]), ("12_fixed_range", GRAY["12"]), ("123_fixed_range", GRAY["123"]))}
    if len(pr["1236"]) < 4:
        continue
    row = {"rows": int(len(pr["1236"]))}
    for name, p_ in pr.items():
        mean = p_.mean(axis=0)[:HALF][::-1]  # index 0 = the pixel touching the hot object, outward
        far = float(mean[20:].mean())
        row[name] = {"far_code": far, "excess_codes_at_px_1_2_3_4_6_8_12_16": [round(float(mean[k - 1] - far), 1) for k in (1, 2, 3, 4, 6, 8, 12, 16)]}
    out.setdefault("halo", {})[f"{a}|{b}:{side}"] = row
HALF = _HALF

# -- the lag: 123456 against 12356, same pose, same frame index, same noise draw ---------------------


def lag_stats(a: dict, b: dict, label: str) -> dict:
    d = a["signal"].astype(np.float64) - b["signal"].astype(np.float64)
    dd = a["display8"][..., 0].astype(np.int16) - b["display8"][..., 0].astype(np.int16)
    return {"which": label, "same_frame_index": bool(int(a["frame_index"]) == int(b["frame_index"])),
            "signal_diff_max_dn": float(np.abs(d).max()), "signal_diff_max_k": float(np.abs(d).max() / DN_PER_K),
            "px_over_3_sigma_tvh": int((np.abs(d) > 3 * SIGMA_TVH).sum()),
            "px_over_1_sigma_tvh": int((np.abs(d) > SIGMA_TVH).sum()),
            "display_px_over_8_codes": int((np.abs(dd) > 8).sum()), "display_diff_max_codes": int(np.abs(dd).max()),
            "display_diff_mean_abs_codes": float(np.abs(dd).mean())}


out["lag"] = [lag_stats(S["123456"], S["12356"], "demo speed (1 pose/frame)")]
if "123456" in SLOW and "12356" in SLOW:
    out["lag"].append(lag_stats(SLOW["123456"], SLOW["12356"], "10x slower camera"))
out["noise_changes_agc"] = {
    "display_diff_mean_abs_codes_1236_vs_12356": float(np.abs(GRAY["1236"].astype(np.int16) - GRAY["12356"].astype(np.int16)).mean())}

# -- AGC frame to frame: the level a constant-temperature surface is shown at, over the sequence ------
seq12 = np.load(D / "display_seq_12.npy", mmap_mode="r")
wall_rgb = None
m = interior("Wall", 3)
if m.any():
    wall_rgb = np.asarray(S["12"]["display8"][..., :3][m][0])
    floor_rgb = np.asarray(S["12"]["display8"][..., :3][interior("Floor", 3)][0]) if interior("Floor", 3).any() else None
    agc = {}
    for v in ("1236", "123456"):
        seq = np.load(D / f"display_seq_{v}.npy", mmap_mode="r")
        lev_w, lev_f, hot = [], [], []
        for j in range(seq.shape[0]):
            f12 = np.asarray(seq12[j])
            mw = (f12 == wall_rgb).all(axis=-1)
            lev_w.append(float(np.median(seq[j][..., 0][mw])) if mw.sum() > 500 else np.nan)
            if floor_rgb is not None:
                mf = (f12 == floor_rgb).all(axis=-1)
                lev_f.append(float(np.median(seq[j][..., 0][mf])) if mf.sum() > 500 else np.nan)
            hot.append(float((f12.astype(np.int32).sum(axis=-1) > 300).mean()))
        lw, lf = np.asarray(lev_w), np.asarray(lev_f)
        agc[v] = {"wall_code_min": float(np.nanmin(lw)), "wall_code_max": float(np.nanmax(lw)),
                  "wall_code_max_step_between_frames": float(np.nanmax(np.abs(np.diff(lw)))),
                  "floor_code_min": float(np.nanmin(lf)), "floor_code_max": float(np.nanmax(lf)),
                  "floor_code_max_step_between_frames": float(np.nanmax(np.abs(np.diff(lf)))),
                  "wall_minus_floor_codes_min": float(np.nanmin(lw - lf)), "wall_minus_floor_codes_max": float(np.nanmax(lw - lf)),
                  "wall_codes": [round(x, 1) for x in lw.tolist()], "floor_codes": [round(x, 1) for x in lf.tolist()]}
    out["agc_over_sequence"] = agc
    out["agc_fixed_range_reference"] = {"wall_code": float(np.median(GRAY["12"][interior("Wall", 3)])),
                                        "floor_code": float(np.median(GRAY["12"][interior("Floor", 3)]))}

(ROOT / f"analysis_{traj}_{pose:03d}.json").write_text(json.dumps(out, indent=1))

# -- contact sheets --------------------------------------------------------------------------------
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except OSError:
    FONT = SMALL = ImageFont.load_default()
PAD, HEAD, BG, FG = 8, 30, (18, 18, 18), (235, 235, 235)
ZOOM = 8
CW, CH = W // ZOOM, H // ZOOM  # crop size that fills one tile at 8x


def crop_box(cy: int, cx: int):
    y0 = int(np.clip(cy - CH // 2, 0, H - CH))
    x0 = int(np.clip(cx - CW // 2, 0, W - CW))
    return y0, x0


def zoom(img: np.ndarray, y0: int, x0: int) -> Image.Image:
    c = img[y0:y0 + CH, x0:x0 + CW]
    return Image.fromarray(c).resize((CW * ZOOM, CH * ZOOM), Image.NEAREST)


def rgb(g: np.ndarray) -> np.ndarray:
    return np.repeat(g[..., None], 3, axis=-1)


def corner_of(name: str, which: str = "tl"):
    m = IDS == ID_OF[name]
    ys, xs = np.nonzero(m)
    if which == "tl":
        return int(ys.min()), int(xs[ys == ys.min()].min())
    return int(ys.max()), int(xs.min())


# crop 1: a hot corner (the anomaly if it is in frame, else the pipe); crop 2: a low-contrast corner
hot = "Anomaly" if "Anomaly" in ID_OF and (IDS == ID_OF["Anomaly"]).sum() > 400 else "HotPipe"
cool = next((n for n in ("Crate_01", "Crate_02", "Ball") if n in ID_OF and (IDS == ID_OF[n]).sum() > 400), hot)
C1 = crop_box(*corner_of(hot))
C2 = crop_box(*corner_of(cool))
out["crops"] = {"hot": [hot, *C1], "cool": [cool, *C2], "size": [CH, CW], "zoom": ZOOM}
(ROOT / f"analysis_{traj}_{pose:03d}.json").write_text(json.dumps(out, indent=1))


def stretch(v: str, y0: int, x0: int) -> Image.Image:
    """The low-contrast crop with its own min-max stretch, so +-1 code is visible on any screen."""
    c = GRAY[v][y0:y0 + CH, x0:x0 + CW].astype(np.float64)
    lo, hi = np.percentile(c, 1), np.percentile(c, 99)
    g = np.clip((c - lo) / max(hi - lo, 1.0), 0, 1)
    return Image.fromarray(rgb((255 * g).astype(np.uint8))).resize((CW * ZOOM, CH * ZOOM), Image.NEAREST)


SUMMARY = json.loads(pathlib.Path("/tmp/irsim_render_cost/stages/out/summary.json").read_text())
SHORT = {"12": "1+2  lo actual", "123": "1+2+3  + optica", "1236": "1+2+3+6  + ISP", "12356": "1+2+3+5+6  + ruido",
         "123456": "1+2+3+4+5+6  completo", "repo": "NO pedido: completo + flat field + cadena M9"}


def sheet(cols, rows, path, title):
    """rows: list of (label, fn(variant) -> PIL image W x H). One label line per row, full width."""
    wide = PAD + len(cols) * (W + PAD)
    top = 3 * HEAD
    tall = top + len(rows) * (H + HEAD + PAD) + PAD
    im = Image.new("RGB", (wide, tall), BG)
    d = ImageDraw.Draw(im)
    d.text((PAD, 6), title, fill=(255, 220, 120), font=FONT)
    for c, v in enumerate(cols):
        x = PAD + c * (W + PAD)
        hl, gui = SUMMARY["hl"]["rows"][v], SUMMARY["gui"]["rows"][v]
        d.text((x + 2, HEAD + 4), SHORT[v], fill=FG, font=FONT)
        d.text((x + 2, 2 * HEAD + 2), f"CPU: {hl['ms']:.1f} ms = {hl['fps']:.0f} fps sin GUI | {gui['ms']:.1f} ms = {gui['fps']:.0f} fps con GUI",
               fill=(150, 220, 150) if hl["ms"] < 33.3 else (255, 140, 140), font=SMALL)
    for r, (label, fn) in enumerate(rows):
        y = top + r * (H + HEAD + PAD)
        d.text((PAD + 2, y + 6), label, fill=(255, 220, 120), font=SMALL)
        for c, v in enumerate(cols):
            im.paste(fn(v).convert("RGB"), (PAD + c * (W + PAD), y + HEAD))
    im.save(path)
    im.resize((im.width // 2, im.height // 2), Image.LANCZOS).save(str(path).replace(".png", "_half.png"))
    print("wrote", path, im.size)


def boxed(v: str) -> Image.Image:
    im = Image.fromarray(rgb(GRAY[v]))
    d = ImageDraw.Draw(im)
    for (y0, x0), col in ((C1, (255, 80, 80)), (C2, (80, 200, 255))):
        d.rectangle([x0, y0, x0 + CW - 1, y0 + CH - 1], outline=col)
    return im


ROWS = [
    ("A. Tal como sale: 1+2 y 1+2+3 por el mapa fijo 15-160 C (ironbow) del demo; desde 1+2+3+6, por el ISP del YAML del Boson (plateau 0.012, DDE 0.35, white-hot, gris)",
     lambda v: Image.fromarray(S[v]["display8"][..., :3])),
    ("B. Todas en gris white-hot, para que solo cambie la etapa y no la paleta (1+2 y 1+2+3: el mismo rango fijo 15-160 C). Recuadro rojo = recorte C, azul = recorte D", boxed),
    (f"C. Recorte de 80x64 px ampliado x{ZOOM} (vecino mas proximo) en la esquina de {hot} (135 C aparentes sobre suelo a 20 C). Mismos grises que B", lambda v: zoom(rgb(GRAY[v]), *C1)),
    (f"D. Recorte x{ZOOM} en la esquina de {cool} (18 C contra pared a 25 C). OJO: contraste re-estirado al p1-p99 de cada recorte, para ver bordes y grano de bajo contraste", lambda v: stretch(v, *C2)),
]
title = (f"{tag}  {traj} pose {pose}  |  Boson 640x512, supersample 1, 60 Hz  |  misma pose, mismo frame index en todas "
         f"(ruido: misma tirada en 12356 y 123456)")
sheet(VARIANTS[:5], ROWS, ROOT / f"contact_{traj}_{pose:03d}.png", title)
sheet(VARIANTS, ROWS, ROOT / f"contact6_{traj}_{pose:03d}.png", title + "  + columna extra")

# lag sheet: 123456 - 12356 (same noise), fast and slow, amplified
if "123456" in SLOW:
    def diff_img(a, b, gain=8.0):
        d = (a["signal"].astype(np.float64) - b["signal"].astype(np.float64)) / SIGMA_TVH
        g = np.clip(127.5 + gain * d * 4, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb(g))
    tiles = [("123456 a la velocidad del demo", Image.fromarray(rgb(GRAY["123456"]))),
             ("diferencia 123456 - 12356 (solo retardo), gris medio = 0, +-4 sigma_TVH = negro/blanco",
              diff_img(S["123456"], S["12356"])),
             ("123456 con la camara 10x mas lenta", Image.fromarray(rgb(SLOW["123456"]["display8"][..., 0]))),
             ("diferencia, camara 10x mas lenta, misma escala", diff_img(SLOW["123456"], SLOW["12356"]))]
    im = Image.new("RGB", (PAD + 2 * (W + PAD), PAD + 2 * (H + HEAD + PAD)), BG)
    d = ImageDraw.Draw(im)
    for k, (label, t) in enumerate(tiles):
        x, y = PAD + (k % 2) * (W + PAD), PAD + (k // 2) * (H + HEAD + PAD)
        d.text((x + 2, y + 6), label, fill=FG, font=SMALL)
        im.paste(t, (x, y + HEAD))
    im.save(ROOT / f"lag_{traj}_{pose:03d}.png")
    print("wrote", ROOT / f"lag_{traj}_{pose:03d}.png", im.size)
(ROOT / f"analysis_{traj}_{pose:03d}.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out.get(k) for k in ("control", "halo", "lag")}, indent=1))
for n_, t_ in out["texture"].items():
    for v_, r_ in t_.items():
        print("TEX", n_, v_, {k: (round(x, 2) if isinstance(x, float) else x) for k, x in r_.items()})
