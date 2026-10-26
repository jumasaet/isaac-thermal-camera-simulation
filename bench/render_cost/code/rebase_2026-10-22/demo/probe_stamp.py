#!/usr/bin/env python
"""Which per-frame stamp names the pose an AOV read actually shows?

Ground truth is the instance-id plane itself: a synchronous pass renders every pose of a
monotonic sweep and keeps its CRC, so any later read is identified by content. Against that,
this asks whether the `camera_params` annotator's view transform (a) comes from the rendered
frame rather than from USD at read time, and (b) agrees with the content, frame for frame,
in loops loose and tight enough to produce lag 1, lag 2 and repeated reads.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import nave_stage as ns  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--gui", action="store_true")
p.add_argument("--viewport", action="store_true")
p.add_argument("--n", type=int, default=120)
args, kit_args = p.parse_known_args()
ctx = ns.boot(gui=args.gui, viewport=args.viewport, kit_args=kit_args, materials=ns.load_materials())

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import warp as wp  # noqa: E402

pc = time.perf_counter
camera = ctx["camera"].open(settle_frames=16)
settings = ctx["settings"]
settings.set("/rtx/rendermode", "RealTimePathTracing")
RP = camera._reader.render_product_path
W, H = camera.optics.resolution
camera._reader.detach()
print("rendermode", settings.get("/rtx/rendermode"), "res", W, H, flush=True)

names = sorted(rep.AnnotatorRegistry.get_registered_annotators())
print("annotators matching time/frame/camera:",
      [n for n in names if any(s in n.lower() for s in ("time", "frame", "camera", "sync"))], flush=True)


def attach(name, device=None, init=None):
    a = rep.AnnotatorRegistry.get_annotator(name, init_params=init, **({"device": device} if device else {}))
    a.attach(RP)
    return a


annos = {
    "instance": attach("instance_id_segmentation", "cuda", {"colorize": False}),
    "distance": attach("DistanceToCameraSD", "cuda"),
    "rgb": attach("rgb", "cuda"),
    "cam": attach("camera_params"),
}
extra = {}
for cand in ("ReferenceTime", "IsaacReadTimes", "IsaacReadSimulationTime"):
    if cand in names:
        try:
            extra[cand] = attach(cand)
        except Exception as exc:  # noqa: BLE001
            print(f"attach {cand} failed: {exc}", flush=True)
for _ in range(8):
    rep.orchestrator.step(rt_subframes=1)

d = annos["cam"].get_data()
print("camera_params keys:", {k: (np.asarray(v).shape, np.asarray(v).dtype.str) for k, v in d.items()}, flush=True)
print("cameraViewTransform:", np.asarray(d["cameraViewTransform"]).reshape(4, 4).round(4).tolist(), flush=True)
print("cameraProjection:", np.asarray(d["cameraProjection"]).reshape(4, 4).round(5).tolist(), flush=True)
for k, a in extra.items():
    print(k, "->", a.get_data(), flush=True)


def data_of(v):
    return v.get("data", v) if isinstance(v, dict) else v


def crc():
    return zlib.crc32(data_of(annos["instance"].get_data()).numpy().tobytes())


def stamp_pos():
    """Camera world position recovered from the annotator's view transform (row-vector USD)."""
    v = np.asarray(annos["cam"].get_data()["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
    return np.linalg.inv(v)[3, :3]


# A monotonic sweep: every pose distinct, so content names the pose without ambiguity.
N = args.n
POS = np.stack([np.full(N, 2.5), np.linspace(-4.5, 4.5, N), np.full(N, 2.0)], axis=1)
LOOK = np.array([-np.cos(np.radians(12.0)), 0.0, -np.sin(np.radians(12.0))])
ops = ctx["cam_ops"]


def pose(k):
    ops.set_pose(POS[k], LOOK)


ref, ref_stamp = [], []
for k in range(N):
    pose(k)
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=1)
    ref.append(crc())
    ref_stamp.append(stamp_pos())
index = {h: k for k, h in enumerate(ref)}
ref_stamp = np.asarray(ref_stamp)
print(f"reference: {len(set(ref))}/{N} distinct id planes; sync stamp error "
      f"{np.abs(ref_stamp - POS).max():.2e} m", flush=True)


def by_stamp(pos):
    d2 = np.abs(POS - pos[None, :]).max(axis=1)
    j = int(d2.argmin())
    return j if d2[j] < 1e-4 else None


sink = wp.zeros((H, W), dtype=wp.float32, device="cuda:0")
res = {}
for mode, work in (("nowait+sync", "sync"), ("nowait tight", "none"), ("nowait+2ms sleep", "sleep"),
                   ("wait", "wait")):
    pose(0)
    for _ in range(6):
        rep.orchestrator.step(rt_subframes=1)
    rows, times = [], []
    last_h = None
    for k in range(N):
        t0 = pc()
        pose(k)
        rep.orchestrator.step(rt_subframes=1, wait_for_render=(work == "wait"))
        if work == "sync":
            sink.fill_(1.0)
            wp.synchronize()
        elif work == "sleep":
            time.sleep(0.002)
        h = crc()
        j_crc = index.get(h)
        j_st = by_stamp(stamp_pos())
        times.append(pc() - t0)
        rows.append((k, j_crc, j_st, h == last_h))
        last_h = h
    tail = rows[10:]
    lag_c = [k - j for k, j, _, _ in tail if j is not None]
    lag_s = [k - j for k, _, j, _ in tail if j is not None]
    agree = sum(1 for _, a, b, _ in tail if a == b)
    out = {
        "frame_ms": 1e3 * float(np.mean(times[10:])),
        "lag_by_content": {str(v): lag_c.count(v) for v in sorted(set(lag_c))},
        "lag_by_stamp": {str(v): lag_s.count(v) for v in sorted(set(lag_s))},
        "stamp_agrees_with_content": f"{agree}/{len(tail)}",
        "content_unmatched": sum(1 for _, a, _, _ in tail if a is None),
        "stamp_unmatched": sum(1 for _, _, b, _ in tail if b is None),
        "repeated_reads": sum(1 for *_, r in tail if r),
        "disagreements": [(k, a, b) for k, a, b, _ in tail if a != b][:12],
    }
    res[mode] = out
    print(f"[{mode:17s}] {json.dumps(out)}", flush=True)

pathlib.Path(ns.DEMO / "out" / ("probe_stamp_gui.json" if args.gui else "probe_stamp.json")).write_text(
    json.dumps(res, indent=1))
ctx["app"].close()
