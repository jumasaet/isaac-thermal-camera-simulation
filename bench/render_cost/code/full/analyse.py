#!/usr/bin/env python
"""What each block does to the picture, in numbers, and the contact sheets. Plain Python, no Kit.

    analyse.py [--sessions k1=k1_hl_vpoff k2=k2_hl_vpoff k4=k4_hl_vpoff]

Everything is computed from the float planes the runner kept at the still poses (flux, DN, the
[0, 1] display image), not read off a PNG.

  gradient   the floor is one prescribed temperature, so any slope of its DN against (1 - RI) is the
             camera's: cos4 acting on (L_scene - L_housing). Reported as corner-minus-centre.
  defects    pixels more than 6 sigma_TVH from the median of their 3x3 neighbours.
  edges      along a slanted boundary between two prims, the sub-pixel position where the noiseless
             flux crosses half-way, column by column; rms about a quadratic = the staircase.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, "/home/jumasaet/isaac-thermal-camera-simulation/src")
sys.path.insert(0, "/tmp/irsim_render_cost/full")
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import common as cm
import jet
from irsim.config.loader import load_sensor_config
from irsim.optics.stage import optics_field

p = argparse.ArgumentParser()
p.add_argument("--sessions", nargs="*", default=["k1=k1_hl_vpoff", "k2=k2_hl_vpoff", "k4=k4_hl_vpoff"])
p.add_argument("--stills", nargs="*", default=["circular:54", "linear:40", "circular:10", "linear:20"])
p.add_argument("--no-video", action="store_true")
args = p.parse_args()
OUT = cm.FULL / "out"
DELIV = OUT / "deliver"
DELIV.mkdir(parents=True, exist_ok=True)
SESS = {}
for item in args.sessions:
    name, _, tag = item.partition("=")
    if (OUT / tag / "report.json").exists():
        SESS[name] = tag
REP = {name: json.loads((OUT / tag / "report.json").read_text()) for name, tag in SESS.items()}
H, W = 512, 640
SIGMA = REP["k1"]["sensor"]["sigma_tvh_dn"]
DN_PER_K = REP["k1"]["sensor"]["dn_per_k_300"]
RI = np.asarray(optics_field(load_sensor_config(cm.sensor_yaml(1)).sensor), np.float64)
JET = jet.jet_table()


def still(name: str, variant: str, traj: str, pose: int) -> dict | None:
    f = OUT / SESS[name] / traj / f"still_{variant}_{pose:03d}.npz"
    return dict(np.load(f)) if f.exists() else None


def meta(name: str, traj: str, pose: int):
    m = np.load(OUT / SESS[name] / traj / f"meta_{pose:03d}.npz")
    k = REP[name]["k"]
    labels = {int(i): v.split("/")[-1] for i, v in json.loads(str(m["labels"])).items()}
    return m["ids"], m["ids"][k // 2::k, k // 2::k], labels


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(r):
        m = np.pad(out, 1, mode="constant")
        out = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    return out


def box(a: np.ndarray, r: int) -> np.ndarray:
    a = a.astype(np.float64)
    c = np.cumsum(np.cumsum(np.pad(a, ((r + 1, r), (r + 1, r)), mode="edge"), axis=0), axis=1)
    n = 2 * r + 1
    return (c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]) / (n * n)


def gradient(plane: np.ndarray, mask: np.ndarray) -> dict:
    """Least squares of `plane` on (1 - RI) inside `mask`; corner-minus-centre is slope * max(1 - RI)."""
    x, y = (1.0 - RI)[mask], plane.astype(np.float64)[mask]
    a = np.stack([np.ones_like(x), x], axis=1)
    (c0, c1), *_ = np.linalg.lstsq(a, y, rcond=None)
    span = float((1.0 - RI).max())
    return {"corner_minus_centre": float(c1 * span), "level_on_axis": float(c0), "px": int(mask.sum()),
            "one_minus_ri_covered": [float(x.min()), float(x.max())]}


def outliers(dn: np.ndarray, inside: np.ndarray) -> int:
    """Pixels more than 6 sigma_TVH from the median of their 8 neighbours, counted only well inside
    a prim (a scene edge is not a defect)."""
    d = dn.astype(np.float64)
    pad = np.pad(d, 1, mode="edge")
    neigh = np.stack([pad[dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1)])
    return int(((np.abs(d - np.median(neigh, axis=0)) > 6.0 * SIGMA) & inside).sum())


def edge(flux: np.ndarray, ids: np.ndarray, above: int, below: int) -> dict | None:
    """Sub-pixel row of the half-level crossing per column, along the boundary `above` over `below`."""
    f = flux.astype(np.float64)
    hit = (ids[:-1] == above) & (ids[1:] == below)
    cols, rows = [], []
    for x in range(W):
        ys = np.flatnonzero(hit[:, x])
        if ys.size != 1:
            continue
        y = int(ys[0])
        if y < 8 or y > H - 10 or not ((ids[y - 7:y + 1, x] == above).all() and (ids[y + 1:y + 9, x] == below).all()):
            continue
        prof = f[y - 7:y + 9, x]
        la, lb = prof[:3].mean(), prof[-3:].mean()
        if abs(la - lb) < 1e-12:
            continue
        n = (prof - la) / (lb - la)
        i = np.flatnonzero((n[:-1] < 0.5) & (n[1:] >= 0.5))
        if i.size != 1:
            continue
        i = int(i[0])
        cols.append(x)
        rows.append(y - 7 + i + (0.5 - n[i]) / (n[i + 1] - n[i]))
    if len(cols) < 60:
        return None
    cols, rows = np.asarray(cols, np.float64), np.asarray(rows)
    # the longest run of consecutive columns
    brk = np.flatnonzero(np.diff(cols) > 1.5)
    starts, ends = np.r_[0, brk + 1], np.r_[brk + 1, cols.size]
    j = int(np.argmax(ends - starts))
    cols, rows = cols[starts[j]:ends[j]], rows[starts[j]:ends[j]]
    if cols.size < 60:
        return None
    keep = np.ones(cols.size, bool)
    for _ in range(4):  # clip crossings another prim or a PSF halo pulled away; the staircase is < 0.5 px
        coef = np.polyfit(cols[keep], rows[keep], 2)
        res = rows - np.polyval(coef, cols)
        keep = np.abs(res) < 0.75
        if keep.sum() < 60:
            return None
    clipped = int((~keep).sum())
    cols, rows, res = cols[keep], rows[keep], res[keep]
    return {"columns": int(cols.size), "clipped": clipped, "slope_px_per_px": float(np.polyval(np.polyder(coef), cols.mean())),
            "rms_px": float(res.std()), "max_abs_px": float(np.abs(res).max()), "x0": int(cols[0]), "x1": int(cols[-1]),
            "row_mid": float(np.polyval(coef, cols.mean()))}


# -- numbers --------------------------------------------------------------------------------------
metrics: dict = {"sessions": SESS, "gradient": {}, "defects": {}, "edges": {}, "housing_k": {}, "codes": {}}
for item in args.stills:
    traj, _, pose = item.partition(":")
    pose = int(pose)
    if "k1" not in SESS or not (OUT / SESS["k1"] / traj / f"meta_{pose:03d}.npz").exists():
        continue
    _ids_ss, ids, labels = meta("k1", traj, pose)
    id_of = {v: k for k, v in labels.items()}
    for variant in ("g", "g_ff", "g_m9", "g_m9_ff"):
        s = still("k1", variant, traj, pose)
        if s is None:
            continue
        key = f"{traj}:{pose}"
        metrics["housing_k"].setdefault(key, {})[variant] = float(s["housing_k"])
        g_rows = {}
        for prim in ("Floor", "Wall"):
            if prim in id_of:
                m = erode(ids == id_of[prim], 6)
                if m.sum() > 5000:
                    gd = gradient(s["display_dn"], m)
                    gy = gradient(255.0 * s["y"], m)
                    g_rows[prim] = {"shown_dn": gd["corner_minus_centre"], "as_mk_at_300k": 1e3 * gd["corner_minus_centre"] / DN_PER_K,
                                    "display_codes": gy["corner_minus_centre"], "px": gd["px"], "one_minus_ri": gd["one_minus_ri_covered"],
                                    "codes_p01_p50_p99": [float(v) for v in np.percentile(box(255.0 * s["y"], 2)[m], (1, 50, 99))]}
        metrics["gradient"].setdefault(key, {})[variant] = g_rows
        inside = np.zeros((H, W), bool)
        for prim in ("Floor", "Wall"):
            if prim in id_of:
                inside |= erode(ids == id_of[prim], 8)
        metrics["defects"].setdefault(key, {})[variant] = {"px_over_6_sigma_dn16": outliers(s["dn16"], inside),
                                                           "px_examined": int(inside.sum())}
    for name in SESS:
        s = still(name, "g", traj, pose)
        if s is None:
            continue
        for a, b in (("Wall", "HotPipe"), ("HotPipe", "Wall"), ("Wall", "Floor"), ("Floor", "Crate_01"), ("Floor", "Crate_02")):
            if a in id_of and b in id_of:
                e = edge(s["flux"], ids, id_of[a], id_of[b])
                if e is not None:
                    metrics["edges"].setdefault(f"{traj}:{pose} {a}/{b}", {})[name] = e
(DELIV / "image_metrics.json").write_text(json.dumps(metrics, indent=1))
print("== housing temperature the frame was rendered with (K) ==")
for k_, v in metrics["housing_k"].items():
    print(" ", k_, v)
print("== floor / wall: corner minus centre (fit on 1 - RI) ==")
for k_, v in metrics["gradient"].items():
    for variant, rows in v.items():
        for prim, r in rows.items():
            print(f"  {k_:12s} {variant:8s} {prim:5s} shown DN {r['shown_dn']:+8.1f} (= {r['as_mk_at_300k']:+7.0f} mK) | display {r['display_codes']:+6.1f} codes | "
                  f"codes p01/p50/p99 {r['codes_p01_p50_p99'][0]:.0f}/{r['codes_p01_p50_p99'][1]:.0f}/{r['codes_p01_p50_p99'][2]:.0f} | 1-RI covered "
                  f"{r['one_minus_ri'][0]:.3f}-{r['one_minus_ri'][1]:.3f}")
print("== pixels > 6 sigma_TVH from their 3x3 median ==")
for k_, v in metrics["defects"].items():
    print(" ", k_, v)
print("== edge staircase: rms of the half-level crossing about a quadratic, native pixels ==")
for k_, v in metrics["edges"].items():
    print(f"  {k_:28s} " + " | ".join(f"{n}: rms {e['rms_px']:.3f} max {e['max_abs_px']:.3f} px ({e['columns']} cols, {e['clipped']} clipped, slope {e['slope_px_per_px']:+.3f})"
                                       for n, e in v.items()))


# -- pictures ---------------------------------------------------------------------------------------
def font(size: int):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if pathlib.Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


F1, F2 = font(17), font(14)


def seq(name: str, variant: str, traj: str):
    f = OUT / SESS[name] / traj / f"display_seq_{variant}.npy"
    return np.load(f, mmap_mode="r") if name in SESS and f.exists() else None


def timing(name: str, variant: str) -> str:
    ps = [x for x in REP[name]["passes"] if x["variant"] == variant and x["mode"] == "timed"]
    if not ps:
        return ""
    ms = float(np.mean([x["ms"]["total"]["mean"] for x in ps]))
    return f"{ms:.1f} ms  {1e3 / ms:.0f} fps"


def panels_for(traj: str, pose: int):
    """(title, subtitle, RGB image) for the sheet; all are the timed pass's own delivered frames."""
    out = []

    def add(title, name, variant, palette=None):
        s = seq(name, variant, traj) if name in SESS else None
        if s is None:
            return
        img = np.asarray(s[pose])
        if palette is not None:
            img = palette[img[..., 0]]
        out.append((title, timing(name, variant), img))

    add("CPU completo (etapas 3-6 en host), k=1", "k1", "cpu")
    add("GPU completo (3,4,5,ADC,6), k=1", "k1", "g")
    add("GPU + flat field + cadena M9, k=1", "k1", "g_m9_ff")
    add("GPU + flat field + M9, supersample 2", "k2", "g_m9_ff")
    add("GPU + flat field + M9, supersample 4", "k4", "g_m9_ff")
    add("white-hot (default): GPU + flat field + M9, k=1", "k1", "g_m9_ff")
    add("jet: el mismo frame, misma tabla de codigos", "k1", "g_m9_ff", JET)
    add("GPU completo (sin flat field ni M9), supersample 4", "k4", "g")
    return out


def crop_centre(traj: str, pose: int):
    e = metrics["edges"].get(f"{traj}:{pose} Wall/Floor", {}).get("k1") or metrics["edges"].get(f"{traj}:{pose} Wall/HotPipe", {}).get("k1")
    if e is None:
        return H // 2, W // 2
    return int(round(e["row_mid"])), (e["x0"] + e["x1"]) // 2


def zoom(img: np.ndarray, cy: int, cx: int, hh: int = 26, hw: int = 40, z: int = 8) -> Image.Image:
    y0, x0 = int(np.clip(cy - hh, 0, H - 2 * hh)), int(np.clip(cx - hw, 0, W - 2 * hw))
    c = img[y0:y0 + 2 * hh, x0:x0 + 2 * hw]
    return Image.fromarray(np.ascontiguousarray(c)).resize((2 * hw * z, 2 * hh * z), Image.NEAREST)


def sheet(traj: str, pose: int):
    ps = panels_for(traj, pose)
    if not ps:
        return
    cy, cx = crop_centre(traj, pose)
    pad, lab, zh = 8, 46, 416
    cols = 4
    rows = (len(ps) + cols - 1) // cols
    cell_h = lab + H + pad + zh + pad
    im = Image.new("RGB", (cols * (W + pad) + pad, rows * cell_h + 40), (18, 18, 18))
    d = ImageDraw.Draw(im)
    d.text((pad, 8), f"{traj} pose {pose}: el frame entregado por la pasada cronometrada de cada configuracion; debajo, recorte x8 "
                     f"(vecino mas proximo) centrado en ({cx}, {cy})", fill=(235, 235, 235), font=F1)
    for i, (title, sub, img) in enumerate(ps):
        x, y = pad + (i % cols) * (W + pad), 40 + (i // cols) * cell_h
        d.text((x, y), title, fill=(255, 255, 255), font=F1)
        d.text((x, y + 22), sub, fill=(170, 200, 255), font=F2)
        im.paste(Image.fromarray(np.ascontiguousarray(img)), (x, y + lab))
        d.rectangle([x + cx - 40, y + lab + cy - 26, x + cx + 40, y + lab + cy + 26], outline=(255, 80, 80))
        im.paste(zoom(img, cy, cx), (x, y + lab + H + pad))
    f = DELIV / f"contact_{traj}_{pose:03d}.png"
    im.save(f)
    im.resize((im.width // 2, im.height // 2), Image.LANCZOS).save(DELIV / f"contact_{traj}_{pose:03d}_half.png")
    print("wrote", f, im.size)


def blocks_sheet(traj: str, pose: int):
    """What each block of task 2 does to the picture: g, +flat field, +M9, both. k = 1."""
    vs = [("g", "GPU completo"), ("g_ff", "+ flat field"), ("g_m9", "+ cadena M9"), ("g_m9_ff", "+ flat field + M9")]
    ims = [(t, timing("k1", v), np.asarray(seq("k1", v, traj)[pose]), v) for v, t in vs if seq("k1", v, traj) is not None]
    if not ims:
        return
    pad, lab = 8, 66
    im = Image.new("RGB", (len(ims) * (W + pad) + pad, lab + H + pad + 30), (18, 18, 18))
    d = ImageDraw.Draw(im)
    d.text((pad, 6), f"{traj} pose {pose}, k=1: que aporta cada bloque", fill=(235, 235, 235), font=F1)
    for i, (title, sub, img, v) in enumerate(ims):
        x = pad + i * (W + pad)
        g = metrics["gradient"].get(f"{traj}:{pose}", {}).get(v, {}).get("Floor")
        hk = metrics["housing_k"].get(f"{traj}:{pose}", {}).get(v)
        extra = "" if g is None else f"suelo, esquina-centro: {g['shown_dn']:+.0f} DN ({g['as_mk_at_300k']:+.0f} mK)"
        d.text((x, 28), title + "   " + sub, fill=(255, 255, 255), font=F2)
        d.text((x, 46), f"carcasa {hk:.2f} K   {extra}", fill=(170, 200, 255), font=F2)
        im.paste(Image.fromarray(np.ascontiguousarray(img)), (x, lab))
    f = DELIV / f"blocks_{traj}_{pose:03d}.png"
    im.save(f)
    im.resize((im.width // 2, im.height // 2), Image.LANCZOS).save(DELIV / f"blocks_{traj}_{pose:03d}_half.png")
    print("wrote", f, im.size)


def mp4(path, frames_iter, w, h, fps=30):
    pr = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14", "-movflags", "+faststart", str(path)],
                          stdin=subprocess.PIPE)
    for fr in frames_iter:
        pr.stdin.write(np.ascontiguousarray(fr).tobytes())
    pr.stdin.close()
    print("wrote", path, "rc", pr.wait())


def videos(traj: str):
    single = [("cpu_completo_k1", "k1", "cpu", None), ("gpu_completo_k1", "k1", "g", None), ("gpu_ff_m9_k1", "k1", "g_m9_ff", None),
              ("gpu_ff_m9_k2", "k2", "g_m9_ff", None), ("gpu_ff_m9_k4", "k4", "g_m9_ff", None),
              ("gpu_ff_m9_k2_jet", "k2", "g_m9_ff", JET), ("gpu_ff_m9_k1_jet", "k1", "g_m9_ff", JET)]
    have = {}
    for name, sess, variant, pal in single:
        s = seq(sess, variant, traj) if sess in SESS else None
        if s is None:
            continue
        have[name] = (s, pal)
        mp4(DELIV / f"{traj}_{name}.mp4", ((pal[np.asarray(f)[..., 0]] if pal is not None else np.asarray(f)) for f in s), W, H)
    order = [n for n in ("cpu_completo_k1", "gpu_completo_k1", "gpu_ff_m9_k1", "gpu_ff_m9_k2", "gpu_ff_m9_k4", "gpu_ff_m9_k2_jet") if n in have]
    if len(order) >= 2:
        n = have[order[0]][0].shape[0]

        def frames():
            for j in range(n):
                tiles = [(pal[np.asarray(s[j])[..., 0]] if pal is not None else np.asarray(s[j])) for s, pal in (have[o] for o in order)]
                while len(tiles) % 3:
                    tiles.append(np.zeros((H, W, 3), np.uint8))
                yield np.concatenate([np.concatenate(tiles[r:r + 3], axis=1) for r in range(0, len(tiles), 3)], axis=0)

        rows = (len(order) + 2) // 3
        mp4(DELIV / f"{traj}_comparacion_{'_'.join(str(i + 1) for i in range(len(order)))}.mp4", frames(), 3 * W, rows * H)
        (DELIV / f"{traj}_comparacion_orden.txt").write_text("izquierda a derecha, arriba a abajo:\n" + "\n".join(order) + "\n")


for item in args.stills[:2]:
    traj, _, pose = item.partition(":")
    sheet(traj, int(pose))
    blocks_sheet(traj, int(pose))
if not args.no_video:
    for traj in ("linear", "circular"):
        videos(traj)
