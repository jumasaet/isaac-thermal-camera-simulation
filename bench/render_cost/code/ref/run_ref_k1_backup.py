#!/usr/bin/env python
"""Reference run: the 240-frame wall trajectory through the full device camera, saved and published.

/tmp only; nothing here edits the repo. The Kit loop, the device camera and the check against
`run_frame` are the ones of full/run_full.py (variant g_m9_ff: flat field + M9 chain, supersample 1,
RGB attached); what is new is the scene and trajectory (ref_scene.py), ROS 2 publication
(ros_pub.py, with the repo's message builders), the output layout, and a second freshness stamp.

Per frame (one loop iteration):

    pose      camera translate + orient, and the iteration serial (see below)
    render    rep.orchestrator.step(wait_for_render=False)
    get_data  instance ids + distance + rgb (device) + camera_params (host)
    fresh     which iteration does this read show?  -> the frame index it is filed under
    tables    thermal clock, id -> T / id -> material tables, stage-2 terms
    s12       gather + stage 1 + stage 2, launched
    gpu       stages 3-6 + ADC + flat field + M9, launched
    sync      one wp.synchronize()
    download  thermal RGBA8 (and DN16 if it is what is published) and RGB to host memory
    store     into the sequence arrays (disk is written after the timed loop)
    publish   two sensor_msgs/Image (+ CameraInfo) built by irsim_isaac.ros2_bridge, on rclpy

**Freshness, with two stamps.** The trajectory repeats poses by construction (frame k and frame
239 - k are the same pose; 119 and 120 are consecutive AND identical), so the pose stamp of the
earlier sessions cannot say which of two iterations a read shows near the turn. Each iteration
therefore also writes its own serial number into the camera's focusDistance (depth of field is off,
fStop = 0: the attribute does not reach the image -- checked, not assumed, see `serial_check`), and
`camera_params` hands it back with the rendered frame exactly as it hands back the view transform.
The serial gives the lag; the pose stamp must then equal the pose that iteration wrote. A lag
other than 1, a repeated serial, a pose that disagrees and a frame that never arrived are counted.
`--audit-ids` re-renders every pose synchronously and compares instance-id planes exactly.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import shutil
import subprocess
import sys
import time
import zlib

sys.path.insert(0, "/tmp/irsim_render_cost/demo")
sys.path.insert(0, "/tmp/irsim_render_cost/full")
sys.path.insert(0, "/tmp/irsim_render_cost/ref")
import common as cm  # noqa: E402
import nave_stage as ns  # noqa: E402
import ref_scene as rs  # noqa: E402

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("--gui", action="store_true", help="open the Kit window (default: headless)")
p.add_argument("--viewport", action="store_true", help="leave viewport updates on")
p.add_argument("--passes", type=int, default=1, help="timed traversals of the 240 frames")
p.add_argument("--profile", action="store_true", help="one more traversal with a synchronisation after every block")
p.add_argument("--warm", type=int, default=60, help="untimed lead-in iterations, camera at rest on frame 0's pose")
p.add_argument("--palette", default="white_hot",
               help="white_hot (the sensor YAML: gray + white_hot) | jet | ironbow | rainbow | lava | arctic")
p.add_argument("--save", default=None, metavar="DIR", help="write DIR/{rgb,thermal}/frame_%%05d.png and the two MP4s")
p.add_argument("--video-fps", type=int, default=30)
p.add_argument("--ros", action="store_true", help="publish every delivered frame on ROS 2")
p.add_argument("--ros-ir-topic", default="/thermal_sim/ir/image_raw")
p.add_argument("--ros-rgb-topic", default="/thermal_sim/rgb/image_raw")
p.add_argument("--ros-ir-content", choices=("display8", "dn16"), default="display8",
               help="display8: the shown image, rgba8 | dn16: the raw ADC counts, mono16")
p.add_argument("--ros-qos", choices=("reliable", "best_effort"), default="reliable")
p.add_argument("--ros-depth", type=int, default=10)
p.add_argument("--ros-wait-subs", type=int, default=0, help="wait for this many subscriptions per image topic")
p.add_argument("--no-ros-camera-info", action="store_true")
p.add_argument("--no-check", action="store_true", help="skip the comparison with run_frame")
p.add_argument("--audit-ids", action="store_true")
p.add_argument("--check-poses", type=int, nargs="*", default=[8, 40, 62, 100])
p.add_argument("--tag", required=True)
p.add_argument("--out", default="/tmp/irsim_render_cost/ref/out")
args, kit_args = p.parse_known_args()
K = 1
rs.install(ns)
ns.sensor_ss1_yaml = lambda: cm.sensor_yaml(K)
ctx = ns.boot(gui=args.gui, viewport=args.viewport, kit_args=kit_args, materials=rs.materials())

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import warp as wp  # noqa: E402
from pxr import UsdGeom  # noqa: E402

import fullgpu  # noqa: E402
import jet  # noqa: E402
import lite  # noqa: E402
from irsim.detector.quantise import quantise  # noqa: E402
from irsim.isp.display import run_display_branch  # noqa: E402
from irsim.isp.radiometric import dn_per_kelvin  # noqa: E402
from irsim.noise.three_d import Sigmas7  # noqa: E402
from irsim.pipeline.core import PipelineConfig, PipelineState  # noqa: E402
from irsim.pipeline.frame import run_frame  # noqa: E402
from irsim.pipeline.sensor_chain import attach_sensor_chain  # noqa: E402
from irsim_isaac.pipeline import warp_stages  # noqa: E402
from irsim_isaac.pipeline.material_ids import labels_from_payload, labels_to_paths  # noqa: E402
from irsim_isaac.ros2_bridge import ImageMessage, camera_info, publish_frame  # noqa: E402

pc = time.perf_counter
OUT = pathlib.Path(args.out) / args.tag
OUT.mkdir(parents=True, exist_ok=True)
settings, lut, q, band, scene = ctx["settings"], ctx["lut"], ctx["quantity"], ctx["band"], ctx["scene"]
camera = ctx["camera"].open(settle_frames=16)
cfg: PipelineConfig = camera.config
spec = cfg.sensor.sensor
for _ in range(3):
    settings.set("/rtx/rendermode", "RealTimePathTracing")
    camera._reader.step(frames=1, rt_subframes=1)
RP = camera._reader.render_product_path
WS, HS = camera.optics.resolution
H, W = spec.fpa_shape
assert (HS, WS) == (H, W) and cfg.supersample == 1, ((HS, WS), (H, W), cfg.supersample)
BITS = spec.fpa.bit_depth
camera._reader.detach()
DEV = "cuda:0"
N = rs.N_FRAMES
POS, LOOK = rs.trajectory()
CFG_FF = PipelineConfig.from_sensor(ctx["sensor"], ctx["table"], lut, sky=scene.sky_models[band],
                                    atmosphere=scene.layered, flat_field_enabled=True)
DN_PER_K = float(dn_per_kelvin(cfg.calibration, lut, 300.0))
GAIN = float(cfg.detector.transfer.gain_dn_per_w)
SIGMA_TVH = float(cfg.detector.sigma_signal_dn)
PALETTE = None if args.palette == "white_hot" else jet.table_for(args.palette)


def with_chain(base):
    return attach_sensor_chain(base, scene.weather, t0_s=scene.t0_s)


def cpu_config(variant: str):
    return with_chain(CFG_FF) if variant == "g_m9_ff" else cfg


def device_frame(variant: str = "g_m9_ff", profile: bool = False, palette=None) -> fullgpu.FullGpu:
    ff, m9 = variant.endswith("_ff"), "m9" in variant
    base = CFG_FF if ff else cfg
    return fullgpu.FullGpu(base, device=DEV, flat_field=ff, chain=with_chain(base).chain if m9 else None,
                           optics="repo", profile=profile, palette=palette)


def attach(name, device=None, init=None):
    a = rep.AnnotatorRegistry.get_annotator(name, init_params=init, **({"device": device} if device else {}))
    a.attach(RP)
    return a


annos = {
    "instance": attach("instance_id_segmentation", "cuda", {"colorize": False}),
    "distance": attach("DistanceToCameraSD", "cuda"),
    "cam": attach("camera_params"),
    "rgb": attach("rgb", "cuda"),
}
for _ in range(8):
    rep.orchestrator.step(rt_subframes=1)


def data_of(v):
    return v.get("data", v) if isinstance(v, dict) else v


def wrap_f32(h):
    if h.ndim == 2 and h.dtype is wp.float32:
        return h
    return wp.array(ptr=h.ptr, dtype=wp.float32, shape=tuple(h.shape[:2]), device=h.device)


gpu = lite.LiteGpu(camera, (HS, WS), DEV)
out2 = wp.zeros((HS, WS), dtype=wp.float32, device=DEV)
T0 = camera.scene.t0_s
PERIOD = camera.frame_period_s
CLOCK = [0.0]
ops = ctx["cam_ops"]
FOCUS = UsdGeom.Camera(ctx["stage"].GetPrimAtPath(ns.CAMERA_PATH)).GetFocusDistanceAttr()
SER0 = 1000  # serials are whole metres from here: exact in float32 far beyond any run length
SERIAL = [SER0]


def set_pose(k: int) -> int:
    """Write pose k and a fresh serial; returns the serial."""
    SERIAL[0] += 1
    ops.set_pose(POS[k], LOOK[k])
    FOCUS.Set(float(SERIAL[0]))
    return SERIAL[0]


def tables(payload):
    camera.bridge.advance_to(CLOCK[0])
    t_s = T0 + CLOCK[0]
    gpu.refresh(labels_from_payload(payload["instance"]), t_s)
    camera.state.t_s = t_s
    CLOCK[0] += PERIOD
    return warp_stages.atmosphere_terms(cfg, camera.state), t_s


def kernels_s12(payload, terms):
    gpu.run(lite.wrap_ids(data_of(payload["instance"])))
    warp_stages.launch_atmosphere(gpu.out, wrap_f32(data_of(payload["distance"])), gpu.sky, terms, out2, DEV)


def rgb_host(payload):
    return data_of(payload["rgb"]).numpy().reshape(HS, WS, -1)[..., :4]


def read_stamp(payload):
    """(position, forward, serial) of the camera OF THE RENDERED FRAME."""
    cam = payload["cam"]
    v = np.asarray(cam["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
    c2w = np.linalg.inv(v)
    return c2w[3, :3], -c2w[2, :3], float(np.asarray(cam["cameraFocusDistance"]).reshape(-1)[0])


def read_all():
    return {c: a.get_data() for c, a in annos.items()}


POSE_TOL_M = 1e-4  # neighbouring poses are 0.0504 m apart
BLOCKS = ("pose", "render", "get_data", "fresh", "tables", "s12", "gpu", "sync", "download", "store", "publish")

# -- ROS 2 --------------------------------------------------------------------------------------
ROS = None
ros_info: dict = {"enabled": bool(args.ros)}
if args.ros:
    import ros_pub

    ROS = ros_pub.RosNode(reliability=args.ros_qos, depth=args.ros_depth)
    IR_ENC = "rgba8" if args.ros_ir_content == "display8" else "mono16"
    pub_ir, pub_rgb = ROS.publisher(args.ros_ir_topic, IR_ENC), ROS.publisher(args.ros_rgb_topic, "rgba8")
    if not args.no_ros_camera_info:
        ROS.publisher(f"{args.ros_ir_topic.rsplit('/', 1)[0]}/camera_info", "info")
    if args.ros_wait_subs:
        ros_info["subscribers_at_start"] = ROS.wait_for_subscribers([args.ros_ir_topic, args.ros_rgb_topic], args.ros_wait_subs)
        time.sleep(0.5)
    ros_info.update(ir_topic=args.ros_ir_topic, rgb_topic=args.ros_rgb_topic, ir_content=args.ros_ir_content,
                    ir_encoding=IR_ENC, rgb_encoding="rgba8", camera_info=not args.no_ros_camera_info)


def stamp_ns(pass_i: int, slot: int) -> int:
    """Header stamp: sim time from the start of the run, (pass * 240 + frame) frame periods."""
    return int(round((pass_i * N + slot) * PERIOD * 1e9))


def publish(pass_i: int, slot: int, th, dn, rg) -> None:
    ns_ = stamp_ns(pass_i, slot)
    if args.ros_ir_content == "display8":
        ir = ImageMessage(topic=args.ros_ir_topic, encoding="rgba8", height=H, width=W, step=4 * W, data=th,
                          frame_id="ir_camera", stamp_ns=ns_)
    else:
        ir = ImageMessage(topic=args.ros_ir_topic, encoding="mono16", height=H, width=W, step=2 * W, data=dn,
                          frame_id="ir_camera", stamp_ns=ns_)
    if args.no_ros_camera_info:
        ROS.publisher(ir.topic, ir.encoding).publish(ir)
    else:
        publish_frame(ROS, [ir], camera_info(ctx["sensor"], t_s=ns_ * 1e-9, frame_id="ir_camera"))
    ROS.publisher(args.ros_rgb_topic, "rgba8").publish(
        ImageMessage(topic=args.ros_rgb_topic, encoding="rgba8", height=H, width=W, step=4 * W, data=rg,
                     frame_id="rgb_camera", stamp_ns=ns_))


# The sequence buffers, allocated and TOUCHED once, off the clock: a fresh np.zeros is lazily mapped,
# and the first write to each page inside the loop is a page fault (it read 2.8 ms per frame for the
# two RGBA frames against 0.3 ms once the pages exist).
SEQ = {"thermal": np.empty((N, H, W, 4), np.uint8), "rgb": np.empty((N, H, W, 4), np.uint8),
       "dn16": np.empty((N, H, W), np.uint16) if args.ros and args.ros_ir_content == "dn16" else None}
for _buf in SEQ.values():
    if _buf is not None:
        _buf.fill(0)


def stats_ms(xs):
    a = 1e3 * np.asarray(xs, dtype=np.float64)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "min": float(a.min()), "sd": float(a.std(ddof=1))}


def host_state():
    out = {"epp": pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference").read_text().strip()}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,power.draw,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        out["gpu_temp_util_power_mem"] = r.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        out["gpu"] = f"<{type(exc).__name__}>"
    return out


def run_pass(pass_i: int, mode: str = "timed") -> dict:
    """One traversal. Iteration i writes pose clip(i, 0, 239) and expects to read iteration i - 1.

    The camera rests on frame 0's pose through the lead-in and on frame 239's for the one extra
    iteration that reads it. mode: timed | profile (sync per block) | audit (keeps id planes).
    """
    g = device_frame("g_m9_ff", profile=mode == "profile", palette=PALETTE)
    want_dn = ROS is not None and args.ros_ir_content == "dn16"
    therm, rgbs, dn_seq = SEQ["thermal"], SEQ["rgb"], SEQ["dn16"] if want_dn else None
    for buf in (therm, rgbs, dn_seq):
        if buf is not None:
            buf.fill(0)
    ids_seq = np.full((N, H, W), 255, np.uint8) if mode == "audit" else None
    filled = np.zeros(N, dtype=np.int64)
    acc = {k: [] for k in (*BLOCKS, "total")}
    sub_acc = {k: [] for k in fullgpu.BLOCKS}
    lag_hist: collections.Counter = collections.Counter()
    events, ffc_events = [], 0
    serial_of: dict[int, int] = {}  # serial -> iteration i that wrote it
    prev_serial = None
    state0 = host_state()
    first = -args.warm
    serial_of[set_pose(0)] = first - 1
    for _ in range(4):
        rep.orchestrator.step(rt_subframes=1)
    do_pub = ROS is not None and mode != "audit"
    for i in range(first, N + 1):
        k_write = min(max(i, 0), N - 1)
        t0 = pc()
        serial_of[set_pose(k_write)] = i
        ta = pc()
        rep.orchestrator.step(rt_subframes=1, wait_for_render=False)
        t1 = pc()
        payload = read_all()
        t2 = pc()
        # ---- freshness: the iteration this read shows, by its serial; its pose must agree
        pos_s, fwd_s, ser = read_stamp(payload)
        src = serial_of.get(int(round(ser))) if abs(ser - round(ser)) < 1e-3 else None
        lag = None if src is None else i - src
        repeated = prev_serial is not None and ser == prev_serial
        prev_serial = ser
        slot = src if src is not None and 0 <= src < N else None
        k_src = None if src is None else min(max(src, 0), N - 1)
        pose_ok = k_src is not None and float(np.abs(pos_s - POS[k_src]).max()) < POSE_TOL_M \
            and float(fwd_s @ LOOK[k_src]) > 1.0 - 1e-9
        t3 = pc()
        terms, t_s = tables(payload)
        t4 = pc()
        kernels_s12(payload, terms)
        if mode == "profile":
            wp.synchronize()
        t5 = pc()
        tm = dict.fromkeys(fullgpu.BLOCKS, 0.0)
        g.frame(out2, t_s, tm)
        t6 = pc()
        wp.synchronize()
        t7 = pc()
        th = g.rgba.numpy()
        dn = g.dn16.numpy() if want_dn else None
        rg = rgb_host(payload)
        t8 = pc()
        timed = i >= 1
        deliver = timed and slot is not None and pose_ok and filled[slot] == 0
        if deliver:
            therm[slot] = th
            rgbs[slot] = rg
            if dn_seq is not None:
                dn_seq[slot] = dn
            filled[slot] += 1
        t9 = pc()
        if deliver and do_pub:
            publish(pass_i, slot, th, dn, rg)
        t10 = pc()
        if ids_seq is not None and deliver:  # after the clock: the id plane this frame was computed from
            ids_seq[slot] = data_of(payload["instance"]).numpy().reshape(HS, WS).astype(np.uint8)
        ev = g.last_event
        bad = lag != 1 or repeated or not pose_ok
        if timed:
            ts = (t0, ta, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10)
            for k_, a, b in zip(BLOCKS, ts[:-1], ts[1:], strict=True):
                acc[k_].append(b - a)
            acc["total"].append(t10 - t0)
            for k_ in fullgpu.BLOCKS:
                sub_acc[k_].append(tm[k_])
            lag_hist["unmatched" if lag is None else str(lag)] += 1
            ffc_events += int(bool(ev.fired) or bool(ev.frozen))
            if bad:
                events.append({"iter": i, "lag": lag, "filed_as": slot, "repeated": bool(repeated), "pose_ok": bool(pose_ok)})
    missing = np.flatnonzero(filled == 0).tolist()
    total = np.asarray(acc["total"])
    fps_inst = 1.0 / total
    rep_ = {
        "pass": pass_i, "mode": mode, "n_frames": N,
        "fps": {"mean": float(N / total.sum()), "min": float(fps_inst.min()), "max": float(fps_inst.max()),
                "sd": float(fps_inst.std(ddof=1)), "p05": float(np.percentile(fps_inst, 5)),
                "p50": float(np.percentile(fps_inst, 50))},
        "ms": {k_: stats_ms(v) for k_, v in acc.items()},
        "sub_ms": {k_: float(1e3 * np.mean(v)) for k_, v in sub_acc.items() if np.any(v)},
        "frac_over_33ms": float((total > 1 / 30).mean()),
        "freshness": {"lag_hist": dict(lag_hist), "anomalous_reads": len(events), "frames_never_delivered": missing,
                      "events": events[:20]},
        "ffc_fired_or_frozen_frames": ffc_events, "housing_k_last": float(g.housing_k),
        "m9": {"dT_eff_k": float(g.chain.ffc.delta_t_eff_k), "replacement_passes": int(g.replacement_passes),
               "defects": int(g.chain.bad_pixels.count)},
        "host_before": state0, "host_after": host_state(),
    }
    if do_pub:
        rep_["published"] = {"ir": {str(stamp_ns(pass_i, j)): zlib.crc32((dn_seq if want_dn else therm)[j])
                                    for j in range(N) if filled[j]},
                             "rgb": {str(stamp_ns(pass_i, j)): zlib.crc32(rgbs[j]) for j in range(N) if filled[j]}}
    m, f = rep_["ms"], rep_["fps"]
    sub = " ".join(f"{k_}={v:.2f}" for k_, v in rep_["sub_ms"].items())
    print(f"[{args.tag} #{pass_i} {mode:7s}] {f['mean']:6.2f} fps (min {f['min']:.1f} max {f['max']:.1f} sd {f['sd']:.2f}) | "
          f"{m['total']['mean']:6.2f} ms (p95 {m['total']['p95']:.2f} max {m['total']['max']:.2f}) = "
          + " + ".join(f"{k_} {m[k_]['mean']:.2f}" for k_ in BLOCKS)
          + f" | {sub} | lag {dict(lag_hist)} anomalies {len(events)} missing {len(missing)} ffc {ffc_events}", flush=True)
    return {"report": rep_, "thermal": therm, "rgb": rgbs, "ids_seq": ids_seq, "filled": filled}


# -- off the clock ------------------------------------------------------------------------------


def render_sync(k: int):
    ser = set_pose(k)
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=1)
    payload = read_all()
    pos_s, fwd_s, ser_s = read_stamp(payload)
    if np.abs(pos_s - POS[k]).max() > POSE_TOL_M or fwd_s @ LOOK[k] < 1 - 1e-9 or int(round(ser_s)) != ser:
        raise RuntimeError("synchronous render delivered another frame")
    return payload


def serial_check() -> dict:
    """Does the serial (focusDistance) reach the image? The same pose rendered with two serials far
    apart: the id and distance planes must be bit-identical, and RGB no more different than two
    renders with the SAME serial are (the path tracer's own frame-to-frame noise)."""
    out = {"camera_fstop": float(np.asarray(annos["cam"].get_data()["cameraFStop"]).reshape(-1)[0])}
    rows = []
    for k in (40, 100):
        a = render_sync(k)
        ia, da, ra = (data_of(a[c]).numpy().copy() for c in ("instance", "distance", "rgb"))
        b = render_sync(k)
        rb = data_of(b["rgb"]).numpy().copy()
        SERIAL[0] += 50000
        c = render_sync(k)
        ic, dc, rc = (data_of(c[x]).numpy().copy() for x in ("instance", "distance", "rgb"))
        rows.append({"pose": k, "ids_identical": bool(np.array_equal(ia, ic)),
                     "distance_identical": bool(np.array_equal(da.view(np.uint32), dc.view(np.uint32))),
                     "rgb_mean_abs_diff_same_serial": float(np.abs(ra.astype(np.int16) - rb.astype(np.int16)).mean()),
                     "rgb_mean_abs_diff_other_serial": float(np.abs(ra.astype(np.int16) - rc.astype(np.int16)).mean())})
    out["rows"] = rows
    out["ok"] = all(r["ids_identical"] and r["distance_identical"] for r in rows) and out["camera_fstop"] == 0.0
    return out


def gbuffer_planes(payload, t_s):
    ids = data_of(payload["instance"]).numpy().reshape(HS, WS).astype(np.uint32)
    labels = labels_from_payload(payload["instance"])
    _l1, temp, mat, sky = lite.reduced_numpy(ids, labels, camera, t_s)
    dist = data_of(payload["distance"]).numpy().reshape(HS, WS)
    dist_g = np.where(sky, 0.0, np.nan_to_num(dist, posinf=0.0)).astype(np.float32)
    return {"temperature_k": temp, "material_id": mat, "sky_mask": sky, "distance_m": dist_g,
            "sky_view_factor": np.full((HS, WS), 0.5, np.float32)}, ids, labels


def cpu_planes(ref, config) -> dict:
    display_dn = ref.dn16
    flat = config.flat_field
    if config.chain is not None and config.chain.display_nuc is not None:
        flat = config.chain.display_nuc
    if flat is not None:
        display_dn = quantise(flat.apply(ref.dn16), BITS)
    d = run_display_branch(display_dn, spec.isp, BITS)
    if not np.array_equal(d.display8, ref.display8):
        raise RuntimeError("the recomputed display branch is not run_frame's")
    return {"flux": ref.flux, "signal": ref.signal_dn, "dn16": ref.dn16, "display_dn": display_dn, "y": d.y,
            "display8": ref.display8}


OBJ_ROWS: dict[str, dict] = {}


def note_objects(ids, labels, radiance) -> None:
    """Per prim seen in this frame: kinetic T and the apparent T the camera reads (median)."""
    t_app = np.asarray(lut.apparent_temperature(radiance, q), np.float64)
    temps = camera.bridge.facet_temperatures()
    for ident, path in labels_to_paths(labels).items():
        px = ids == ident
        n = int(px.sum())
        if ident == 0 or n == 0 or (path in OBJ_ROWS and OBJ_ROWS[path]["pixels"] >= n):
            continue
        OBJ_ROWS[path] = {"t_kinetic_c": float(temps[path]) - 273.15,
                          "t_apparent_c_median": float(np.median(t_app[px])) - 273.15, "pixels": n}


def equivalence(variant: str, start: int, n: int = 4) -> dict:
    """`run_frame` (all host, from the G-buffer) against the device frame, n consecutive poses."""
    config = cpu_config(variant)
    gb, gc = device_frame(variant), device_frame(variant)
    st = PipelineState(t_s=T0 + CLOCK[0])
    sig7 = Sigmas7.from_ratios(SIGMA_TVH, spec.noise.sigma_ratios())
    bad_map = None if config.chain is None else np.asarray(config.chain.bad_pixels.mask)
    rows, stats = [], []
    for j in range(start, start + n):
        payload = render_sync(j)
        terms, t_s = tables(payload)
        kernels_s12(payload, terms)
        wp.synchronize()
        planes, ids, labels = gbuffer_planes(payload, t_s)
        if j == start:
            note_objects(ids, labels, out2.numpy())
        st.t_s = t_s
        fi = st.frame_index
        ref = run_frame(planes, config, st)
        cpu = cpu_planes(ref, config)
        noise = cm.cpu_noise_field(config, fi)
        rts = None if config.chain is None else np.asarray(config.chain.defect_state.bad, dtype=np.uint8)
        gb.frame(out2, t_s, inject_noise=noise, rts_override=rts)
        b = gb.planes()
        row = cm.compare(cpu, b, DN_PER_K, GAIN)
        row.update(pose=j, frame_index=fi, housing_k_cpu=float(st.housing_temp_k), housing_k_dev=float(gb.housing_k))
        if abs(row["housing_k_cpu"] - row["housing_k_dev"]) > 1e-9:
            row["fails"].append("housing")
        if ref.report is not None:
            ev_c, ev_g = ref.report.ffc, gb.last_event
            if (ev_c.fired, ev_c.frozen) != (ev_g.fired, ev_g.frozen):
                row["fails"].append("ffc_schedule")
        rows.append(row)
        gc.frame(out2, t_s)
        c = gc.planes()
        stats.append(cm.noise_stats(cpu["signal"], c["signal"], sig7, exclude=bad_map))
    fails = sorted({f for r in rows for f in r["fails"]})
    zmax = {k_: float(np.max(np.abs([s[k_] for s in stats]))) for k_ in ("z_px", "z_row", "z_col", "z_mean")}
    if max(zmax.values()) > 4.0:
        fails.append("noise_statistics")
    worst = {k_: max(r[k_] for r in rows) for k_ in rows[0]
             if isinstance(rows[0][k_], (int, float)) and k_ not in ("pose", "frame_index", "n_px")}
    print(f"[{args.tag} check {variant:8s} poses {start}..{start + n - 1}] same noise: flux {worst['flux_max_rel']:.1e} rel | signal max "
          f"{worst['signal_max_abs_dn']:.4f} DN = {worst['signal_max_abs_mk_at_300k']:.2f} mK | dn16 {worst['dn16_px_differing']} px "
          f"max {worst['dn16_max_code']} | display8 {worst['display8_px_differing']} px max {worst['display8_max_code']}; away from a DN "
          f"change {worst['display8_far_px_differing']} px max {worst['display8_far_max_code']} || own noise: max|z| "
          f"{max(zmax.values()):.2f} || {'FAILS ' + str(fails) if fails else 'ok'}", flush=True)
    return {"variant": variant, "start": start, "worst": worst, "noise_max_abs_z": zmax, "fails": fails}


def audit_ids() -> dict:
    """Noise-free freshness audit: one more traversal keeping (after the clock) the instance-id
    plane each delivered frame was computed from, then every pose rendered synchronously and the
    planes compared exactly. Frames whose reference equals a neighbour's cannot tell fresh from
    stale and are counted apart (119/120 are such a pair by construction)."""
    r = run_pass(0, "audit")
    got = r["ids_seq"]
    ref = np.zeros((N, H, W), np.uint8)
    for j in range(N // 2):
        ref[j] = ref[N - 1 - j] = data_of(render_sync(j)["instance"]).numpy().reshape(HS, WS).astype(np.uint8)
    idx = np.arange(N)
    own = np.array([int((got[j] != ref[j]).sum()) for j in idx])
    prev = np.array([int((got[j] != ref[max(j - 1, 0)]).sum()) for j in idx])
    nxt = np.array([int((got[j] != ref[min(j + 1, N - 1)]).sum()) for j in idx])
    amb = np.array([(j > 0 and np.array_equal(ref[j], ref[j - 1])) or (j < N - 1 and np.array_equal(ref[j], ref[j + 1]))
                    for j in idx])
    inner = ~amb & (idx > 0) & (idx < N - 1)
    return {"identical_to_own_pose": int((own == 0).sum()), "n": N, "worst_px_differing_from_own": int(own.max()),
            "identical_to_previous_pose": int(((prev == 0) & inner).sum()),
            "identical_to_next_pose": int(((nxt == 0) & inner).sum()),
            "frames_whose_reference_equals_a_neighbour": np.flatnonzero(amb).tolist(),
            "distinguishable": int((~amb).sum()), "distinguishable_and_identical_to_own": int(((own == 0) & ~amb).sum()),
            "min_px_between_neighbouring_poses": int(min(prev[inner].min(), nxt[inner].min())),
            "lag_hist": r["report"]["freshness"]["lag_hist"], "delivered": int((r["filled"] > 0).sum())}


# -- disk ---------------------------------------------------------------------------------------


def write_png(path, rgba):
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(rgba[..., :3])).save(path)


def write_mp4(path, frames):
    if shutil.which("ffmpeg") is None:
        return "no ffmpeg"
    h, w = frames.shape[1:3]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(args.video_fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14",
           "-movflags", "+faststart", str(path)]
    r = subprocess.run(cmd, input=np.ascontiguousarray(frames[..., :3]).tobytes(), capture_output=True)
    return "ok" if r.returncode == 0 else r.stderr.decode()[-400:]


def save(run: dict, directory: str) -> dict:
    d = pathlib.Path(directory)
    info = {"dir": str(d)}
    for name in ("rgb", "thermal"):
        sub = d / name
        if sub.exists():
            shutil.rmtree(sub)  # only the two folders this run owns; nothing else in DIR is touched
        sub.mkdir(parents=True)
        for j in range(N):
            write_png(sub / f"frame_{j:05d}.png", run[name][j])
        info[f"{name}.mp4"] = write_mp4(d / f"{name}.mp4", run[name])
        info[f"{name}_png"] = len(list(sub.glob("frame_*.png")))
    return info


# -- main ---------------------------------------------------------------------------------------
proj = np.asarray(annos["cam"].get_data()["cameraProjection"], dtype=np.float64).reshape(4, 4)
report: dict = {
    "tag": args.tag, "gui": args.gui, "viewport_flag": args.viewport, "k": K, "resolution": [W, H], "n_frames": N,
    "frame_period_s": PERIOD, "palette": args.palette, "polarity": spec.isp.polarity, "ros": ros_info,
    "rendermode": settings.get("/rtx/rendermode"), "host": host_state(),
    "camera": {"authored_usd": {k_: v for k_, v in camera.authored.items() if not k_.startswith("omni:")},
               "fpa": {"pitch_um": float(spec.fpa.pitch_um), "shape_h_w": list(spec.fpa_shape)},
               "hfov_deg": float(np.degrees(2 * np.arctan(1 / proj[0, 0]))),
               "vfov_deg": float(np.degrees(2 * np.arctan(1 / proj[1, 1])))},
    "trajectory": {"frame_0": POS[0].tolist(), "frame_119": POS[119].tolist(), "frame_120": POS[120].tolist(),
                   "frame_239": POS[239].tolist(), "step_m": float(np.linalg.norm(POS[1] - POS[0])), "look": LOOK[0].tolist()},
    "sensor": {"name": spec.name, "sigma_tvh_dn": SIGMA_TVH, "dn_per_k_300": DN_PER_K, "isp": spec.isp.model_dump(mode="json")},
    "passes": [], "check": [],
}
try:
    from omni.kit.viewport.utility import get_active_viewport

    vp = get_active_viewport()
    if vp is not None and not args.viewport and vp.updates_enabled:
        vp.updates_enabled = False
        report["viewport_forced_off_by_api"] = True
        for _ in range(4):
            rep.orchestrator.step(rt_subframes=1)
    report["viewport_updates_enabled"] = None if vp is None else bool(vp.updates_enabled)
    report["viewport_resolution"] = None if vp is None else list(vp.resolution)
except Exception as exc:  # noqa: BLE001
    report["viewport_updates_enabled"] = f"<{type(exc).__name__}: {exc}>"
print(f"[{args.tag}] gui {args.gui} viewport flag {args.viewport} -> updates_enabled {report['viewport_updates_enabled']} | rendermode "
      f"{report['rendermode']} | epp {report['host']['epp']} | {W}x{H} HFOV {report['camera']['hfov_deg']:.2f} VFOV "
      f"{report['camera']['vfov_deg']:.2f} | authored {report['camera']['authored_usd']} | period {PERIOD * 1e3:.3f} ms | palette {args.palette} | ros {ros_info}", flush=True)


def dump():
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))


def finish(code: int = 0):
    dump()
    print(f"wrote {OUT / 'report.json'}", flush=True)
    if ROS is not None:
        time.sleep(1.0)  # let a reliable subscriber drain the last frames before the node goes
        ROS.close()
    ctx["app"].close()
    sys.exit(code)


report["serial_check"] = serial_check()
print(f"[{args.tag} serial] {json.dumps(report['serial_check'])}", flush=True)
if not report["serial_check"]["ok"]:
    print(f"[{args.tag}] the serial stamp reaches the image: stopping", flush=True)
    finish(4)

if not args.no_check:
    for variant in ("g", "g_m9_ff"):
        for s0 in args.check_poses:
            report["check"].append(equivalence(variant, s0))
    report["check_fails"] = sorted({f"{e['variant']}@{e['start']}: {f}" for e in report["check"] for f in e["fails"]})
    report["objects"] = {path.rsplit("/", 1)[-1]: row for path, row in sorted(OBJ_ROWS.items())}
    dump()
    if report["check_fails"]:
        print(f"[{args.tag}] CHECK FAILED -> {report['check_fails']}; nothing is timed", flush=True)
        finish(3)
    print(f"[{args.tag}] check passed: the device frame is run_frame's within the repo's tolerances", flush=True)
    for name, row in report["objects"].items():
        print(f"[{args.tag} object] {name:10s} kinetic {row['t_kinetic_c']:7.2f} C  apparent {row['t_apparent_c_median']:7.2f} C  "
              f"({row['pixels']} px)", flush=True)

published: dict = {"ir": {}, "rgb": {}}
last = None
for pass_i in range(args.passes):
    r = run_pass(pass_i)
    pub = r["report"].pop("published", None)
    if pub:
        published["ir"].update(pub["ir"])
        published["rgb"].update(pub["rgb"])
    report["passes"].append(r["report"])
    last = r
    dump()
if ROS is not None:
    (OUT / "published.json").write_text(json.dumps(published))
    report["ros"].update(ROS.describe())
if args.save and last is not None:
    miss = last["report"]["freshness"]["frames_never_delivered"]
    if miss:
        print(f"[{args.tag}] frames never delivered: {miss}; NOT filled in, the PNGs of those frames are black", flush=True)
    report["saved"] = save(last, args.save)
    print(f"[{args.tag} saved] {json.dumps(report['saved'])}", flush=True)
del last
if args.profile:
    r = run_pass(args.passes, "profile")
    pub = r["report"].pop("published", None)
    if pub:  # the profiled pass publishes too: it belongs in the table the subscriber is checked against
        published["ir"].update(pub["ir"])
        published["rgb"].update(pub["rgb"])
        (OUT / "published.json").write_text(json.dumps(published))
    report["passes"].append(r["report"])
    del r
if args.audit_ids:
    report["audit_ids"] = audit_ids()
    print(f"[{args.tag} audit-ids] {json.dumps(report['audit_ids'])}", flush=True)
finish(0)
