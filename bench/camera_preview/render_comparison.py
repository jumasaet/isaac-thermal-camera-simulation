#!/usr/bin/env python
"""Task 1: the comparison scene and its reference trajectory, seen by the Cellplus CTC-007 draft.

Same scene (ref_scene.OBJECTS, prescribed temperatures, isothermal 21 C enclosure) and same 240-pose
trajectory as the reference run in outputs/comparison_ref/; only the camera changes: the Boson YAML
is replaced by bench/camera_check/cellplus_ctc007_lwir.yaml, unchanged. See common.py for the path.

Display: NO AGC. The apparent-temperature plane is mapped through one fixed linear window,
white hot, the same for every frame (--window, default 15 45 C).

    /home/jumasaet/isaac_sim/python.sh bench/camera_preview/render_comparison.py [--n 240] [--static 12]

Writes bench/camera_preview/comparison/:
    thermal/frame_NNNNN.png     every frame, fixed window
    rgb/frame_NNNNN.png         the visible companion (same render product, box-reduced)
    thermal.mp4, rgb.mp4        30 fps
    stills/                     a few frames, with the window printed, + their raw dn16 / apparent_t .npy
    static/                     --static frames at one pose (pose 47), raw dn16 + apparent_t: the
                                temporal noise the moving trajectory cannot show (for task 2)
    report.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as cm  # noqa: E402

sys.path.insert(0, str(cm.BENCH / "ref"))
import ref_scene as rs  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=rs.N_FRAMES, help="frames of the trajectory to render (from 0)")
p.add_argument("--static", type=int, default=12, help="extra frames at pose --static-pose, camera at rest")
p.add_argument("--static-pose", type=int, default=47)
p.add_argument("--window", type=float, nargs=2, default=(15.0, 45.0), metavar=("LO_C", "HI_C"))
p.add_argument("--stills", type=int, nargs="*", default=[0, 25, 47, 70, 95, 119])
p.add_argument("--fps", type=int, default=30)
p.add_argument("--out", default=str(cm.HERE / "comparison"))
args = p.parse_args()

import numpy as np  # noqa: E402

OUT = pathlib.Path(args.out)
for sub in ("thermal", "rgb", "stills", "static"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

# Scene YAML generated from the reference table, written HERE (the bench's own copy is left alone).
rs.SCENE_YAML = OUT / "scene_ref.yaml"
scene_yaml = rs.write_scene()
POS, LOOK = rs.trajectory()
ctx = cm.boot(sensor_yaml=cm.CELLPLUS_YAML, scene_yaml=scene_yaml, objects=rs.OBJECTS, materials=rs.materials(),
              first_pose=(POS[0], LOOK[0]))

from PIL import Image, ImageDraw  # noqa: E402

LO, HI = args.window
H, W = ctx["shape"]
clock = cm.Clock(ctx)
spec = ctx["spec"]
print(f"[cellplus] render product {ctx['shape_ss'][1]}x{ctx['shape_ss'][0]} (k {ctx['k']}) -> {W}x{H}, "
      f"HFOV {ctx['hfov_deg']:.2f} VFOV {ctx['vfov_deg']:.2f}, period {clock.period * 1e3:.1f} ms, window {LO:g}-{HI:g} C",
      flush=True)

for _ in range(4):  # warm-up at pose 0: the membrane and the chain start from a settled frame
    clock.frame(cm.render_sync(ctx, POS[0], LOOK[0]))

n = min(args.n, rs.N_FRAMES)
therm = np.zeros((n, H, W), np.uint8)
rgbs = np.zeros((n, H, W, 3), np.uint8)
per_frame = []
t_start = time.perf_counter()
for j in range(n):
    t0 = time.perf_counter()
    out, rgb, _ids = clock.frame(cm.render_sync(ctx, POS[j], LOOK[j]))
    therm[j] = cm.window_gray(out.apparent_t, LO, HI)
    rgbs[j] = rgb
    Image.fromarray(therm[j]).save(OUT / "thermal" / f"frame_{j:05d}.png")
    Image.fromarray(rgb).save(OUT / "rgb" / f"frame_{j:05d}.png")
    ta = out.apparent_t - 273.15
    per_frame.append({"frame": j, "t_app_c_p1_p50_p99": [float(np.percentile(ta, q)) for q in (1, 50, 99)],
                      "dn16_p1_p50_p99": [float(np.percentile(out.dn16, q)) for q in (1, 50, 99)],
                      "clip_lo_frac": float((therm[j] == 0).mean()), "clip_hi_frac": float((therm[j] == 255).mean()),
                      "ffc": None if out.report is None else bool(out.report.ffc.fired)})
    if j in args.stills:
        np.save(OUT / "stills" / f"frame_{j:05d}_dn16.npy", out.dn16)
        np.save(OUT / "stills" / f"frame_{j:05d}_apparent_t_k.npy", out.apparent_t)
        im = Image.fromarray(therm[j]).convert("RGB")
        ImageDraw.Draw(im).text((6, H - 16), f"Cellplus CTC-007 sim | frame {j} | fixed window {LO:g}-{HI:g} C, white hot",
                                fill=(255, 255, 0))
        im.save(OUT / "stills" / f"frame_{j:05d}.png")
        Image.fromarray(rgb).save(OUT / "stills" / f"frame_{j:05d}_rgb.png")
    if j % 20 == 0:
        print(f"[cellplus] frame {j}/{n} {1e3 * (time.perf_counter() - t0):.0f} ms | T_app p1/p50/p99 "
              f"{per_frame[-1]['t_app_c_p1_p50_p99']}", flush=True)
loop_s = time.perf_counter() - t_start

static = []
for i in range(args.static):
    out, _rgb, _ids = clock.frame(cm.render_sync(ctx, POS[args.static_pose], LOOK[args.static_pose]))
    np.save(OUT / "static" / f"static_{i:03d}_dn16.npy", out.dn16)
    np.save(OUT / "static" / f"static_{i:03d}_apparent_t_k.npy", out.apparent_t)
    static.append(out.dn16.astype(np.float64))
if len(static) >= 2:
    d = np.diff(np.stack(static), axis=0)
    sigma_t_dn = float(np.median(np.abs(d - np.median(d))) * 1.4826 / np.sqrt(2))
else:
    sigma_t_dn = None

info = {"mp4": {"thermal": cm.write_mp4(OUT / "thermal.mp4", therm, args.fps),
                "rgb": cm.write_mp4(OUT / "rgb.mp4", rgbs, args.fps)}}
from irsim.isp.radiometric import dn_per_kelvin  # noqa: E402

cfg = ctx["cfg"]
report = {"sensor_yaml": str(cm.CELLPLUS_YAML), "scene_yaml": str(scene_yaml), "n_frames": n,
          "render_product_w_h": [ctx["shape_ss"][1], ctx["shape_ss"][0]], "supersample_k": ctx["k"],
          "resolution_w_h": [W, H], "hfov_deg": ctx["hfov_deg"], "vfov_deg": ctx["vfov_deg"],
          "frame_period_s": clock.period, "window_c": [LO, HI], "loop_s": loop_s,
          "dn_per_k_at_300k": float(dn_per_kelvin(cfg.calibration, ctx["lut"], 300.0)),
          "sigma_signal_dn": float(cfg.detector.sigma_signal_dn),
          "static_pose": args.static_pose, "static_frames": args.static, "static_sigma_temporal_dn_robust": sigma_t_dn,
          "isp": spec.isp.model_dump(mode="json"), "per_frame": per_frame, **info}
(OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))
print(f"[cellplus] {n} frames in {loop_s:.1f} s; dn/K {report['dn_per_k_at_300k']:.2f}; static sigma_t "
      f"{sigma_t_dn} DN; mp4 {info['mp4']}", flush=True)
print(f"wrote {OUT / 'report.json'}", flush=True)
ctx["app"].close()
