#!/usr/bin/env python
"""Demo runner for the lite GPU path: the industrial hall, two camera paths, thermal + RGB.

/tmp only; nothing here edits the repo. One process = one renderer configuration (GUI or not,
viewport on or off); it runs every requested trajectory ``--passes`` times, interleaved, and
writes frames, MP4s and a report.

Per frame (one loop iteration):

    pose      write the camera's translate + orient ops
    render    rep.orchestrator.step(wait_for_render=False)       -- one app.update()
    get_data  instance ids + distance + rgb (device) + camera_params (host)
    fresh     which pose does this read show?  -> the slot the frame is stored in
    tables    thermal clock, id -> T / id -> material tables, L_env scalar, stage-2 terms
    kernels   gather + stage 1 + stage 2 + apparent-T palette, then wp.synchronize()
    output    thermal RGBA8 and RGB to host memory (disk is written after the timed loop)

**Freshness is measured, not assumed.** With ``wait_for_render=False`` a read shows the pose
of an earlier iteration -- one back when the loop does not outrun the renderer, two when it
does, and sometimes the same frame twice. The `camera_params` annotator carries the view
transform *of the rendered frame* (probe_stamp.py: it agrees with the id plane's content in
110/110 reads in every loop tried, including the ones that lag 2 and repeat). Each read is
therefore filed under the pose its own stamp names, never under the pose the loop just wrote;
a lag other than 1, a repeated stamp, a stamp that matches no pose and a pose that never
arrived are all counted and reported. ``--audit`` then re-renders every pose synchronously
and compares content, so the stamp itself is checked on each run.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import nave_stage as ns  # noqa: E402

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("--trajectory", choices=("linear", "circular", "both"), default="both")
p.add_argument("--n-frames", type=int, default=120)
p.add_argument("--passes", type=int, default=2)
p.add_argument("--warm", type=int, default=30, help="untimed lead-in iterations per pass")
p.add_argument("--gui", action="store_true", help="open the Kit window (default: headless)")
p.add_argument("--viewport", action="store_true", help="leave viewport updates on")
p.add_argument("--no-rgb", action="store_true")
p.add_argument("--film-iso", type=float, default=None,
               help="RGB exposure: /rtx/post/tonemap/filmIso (Kit default 100). Does not touch the lights")
p.add_argument("--palette", default="ironbow")
p.add_argument("--t-range-c", type=float, nargs=2, default=(15.0, 160.0))
p.add_argument("--set-material", action="append", default=[], metavar="Prim=material")
p.add_argument("--no-audit", action="store_true")
p.add_argument("--no-save", action="store_true", help="timing only: no PNG, no MP4")
p.add_argument("--video-fps", type=int, default=30)
p.add_argument("--stills", type=int, nargs="*", default=[0, 20, 40, 60])
p.add_argument("--inject-sync-step", type=int, default=0, metavar="K",
               help="SELF-TEST of the freshness check: every K-th timed iteration waits for its render, "
                    "so that read shows the pose just written (lag 0), one pose is never delivered and "
                    "the next read repeats a frame. Not for timing")
p.add_argument("--tag", required=True)
p.add_argument("--out", default=str(ns.DEMO / "out"))
args, kit_args = p.parse_known_args()
MATERIALS = ns.load_materials(args.set_material)
ctx = ns.boot(gui=args.gui, viewport=args.viewport, kit_args=kit_args, materials=MATERIALS)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import warp as wp  # noqa: E402

import lite  # noqa: E402  (the verified reduced path, /home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/code/rebase_2026-10-22/lite/lite.py)
from irsim.isp.palette import palette_table, quantise_display  # noqa: E402
from irsim.pipeline.atmosphere import apply_layered_gbuffer  # noqa: E402
from irsim_isaac.pipeline import warp_stages  # noqa: E402
from irsim_isaac.pipeline.material_ids import labels_from_payload, labels_to_paths  # noqa: E402

pc = time.perf_counter
OUT = pathlib.Path(args.out) / args.tag
OUT.mkdir(parents=True, exist_ok=True)
settings, lut, q, band = ctx["settings"], ctx["lut"], ctx["quantity"], ctx["band"]
camera = ctx["camera"].open(settle_frames=16)
cfg = camera.config
for _ in range(3):
    settings.set("/rtx/rendermode", "RealTimePathTracing")
    camera._reader.step(frames=1, rt_subframes=1)
if args.film_iso is not None:
    settings.set("/rtx/post/tonemap/filmIso", float(args.film_iso))
RP = camera._reader.render_product_path
W, H = camera.optics.resolution
camera._reader.detach()
DEV = "cuda:0"
N = args.n_frames
USE_RGB = not args.no_rgb
T_MIN_K, T_MAX_K = (273.15 + t for t in args.t_range_c)


def attach(name, device=None, init=None):
    a = rep.AnnotatorRegistry.get_annotator(name, init_params=init, **({"device": device} if device else {}))
    a.attach(RP)
    return a


annos = {
    "instance": attach("instance_id_segmentation", "cuda", {"colorize": False}),
    "distance": attach("DistanceToCameraSD", "cuda"),
    "cam": attach("camera_params"),
}
if USE_RGB:
    annos["rgb"] = attach("rgb", "cuda")
for _ in range(8):
    rep.orchestrator.step(rt_subframes=1)


def data_of(v):
    return v.get("data", v) if isinstance(v, dict) else v


def wrap_f32(h):
    if h.ndim == 2 and h.dtype is wp.float32:
        return h
    return wp.array(ptr=h.ptr, dtype=wp.float32, shape=tuple(h.shape[:2]), device=h.device)


# -- display: at-aperture band radiance -> apparent temperature -> palette, on the device ------
# One table over radiance, linear between L_B(T_min) and L_B(T_max): each entry is the palette
# colour of the apparent temperature L_B^-1(L) of its radiance (docs/physics-model.md §3.3,
# §11.4). 16384 entries put the table's own step two orders below one display code.
N_TAB = 16384
L_LO = float(lut.lookup(np.float64(T_MIN_K), q)[()])
L_HI = float(lut.lookup(np.float64(T_MAX_K), q)[()])


def display_host(radiance: np.ndarray) -> np.ndarray:
    """The CPU reference of the display: RGBA8 from float32 radiance, through the repo's LUT."""
    t_app = np.asarray(lut.apparent_temperature(np.asarray(radiance, np.float32), q), np.float64)
    dn8 = quantise_display((t_app - T_MIN_K) / (T_MAX_K - T_MIN_K))
    rgb = palette_table(args.palette)[dn8]
    return np.concatenate([rgb, np.full((*dn8.shape, 1), 255, np.uint8)], axis=-1)


_tab_rgba = display_host(np.linspace(L_LO, L_HI, N_TAB, dtype=np.float64).astype(np.float32))
DISP_TAB = wp.array(np.ascontiguousarray(_tab_rgba).view(np.uint32).reshape(-1), dtype=wp.uint32, device=DEV)


@wp.kernel
def _display_kernel(
    radiance: wp.array2d(dtype=wp.float32),
    table: wp.array(dtype=wp.uint32),
    l_lo: wp.float32,
    inv_step: wp.float32,
    n: wp.int32,
    out: wp.array2d(dtype=wp.uint32),
):
    i, j = wp.tid()
    x = (radiance[i, j] - l_lo) * inv_step
    k = wp.clamp(wp.int32(wp.floor(x + 0.5)), 0, n - 1)
    out[i, j] = table[k]


gpu = lite.LiteGpu(camera, (H, W), DEV)
out2 = wp.zeros((H, W), dtype=wp.float32, device=DEV)
disp = wp.zeros((H, W), dtype=wp.uint32, device=DEV)
INV_STEP = np.float32((N_TAB - 1) / (L_HI - L_LO))
T0 = camera.scene.t0_s
PERIOD = camera.frame_period_s
CLOCK = [0.0]  # monotonic across passes: the thermal bridge refuses to rewind
ops = ctx["cam_ops"]


def tables(payload):
    """Host work of one frame; returns the stage-2 terms."""
    camera.bridge.advance_to(CLOCK[0])
    t_s = T0 + CLOCK[0]
    gpu.refresh(labels_from_payload(payload["instance"]), t_s)
    camera.state.t_s = t_s
    CLOCK[0] += PERIOD
    return warp_stages.atmosphere_terms(cfg, camera.state)


def kernels(payload, terms):
    """Device work of one frame; `out2` is at-aperture radiance, `disp` the palette image."""
    gpu.run(lite.wrap_ids(data_of(payload["instance"])))
    warp_stages.launch_atmosphere(gpu.out, wrap_f32(data_of(payload["distance"])), gpu.sky, terms, out2, DEV)
    wp.launch(_display_kernel, dim=(H, W),
              inputs=[out2, DISP_TAB, np.float32(L_LO), INV_STEP, np.int32(N_TAB)], outputs=[disp], device=DEV)
    wp.synchronize()


def thermal_host():
    return disp.numpy().view(np.uint8).reshape(H, W, 4)


def rgb_host(payload):
    return data_of(payload["rgb"]).numpy().reshape(H, W, -1)[..., :4]


def read_stamp(payload):
    """(position, forward) of the camera *of the rendered frame*, in stage axes."""
    v = np.asarray(payload["cam"]["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
    c2w = np.linalg.inv(v)  # USD row vectors: rows are the camera's axes, then its position
    return c2w[3, :3], -c2w[2, :3]


def read_all():
    return {c: a.get_data() for c, a in annos.items()}


POSE_TOL_M = 1e-4  # neighbouring poses are 0.15 m apart on both paths
BLOCKS = ("pose", "render", "get_data", "fresh", "tables", "kernels", "output")


def host_state():
    out = {"epp": pathlib.Path(
        "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference").read_text().strip()}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,power.draw,clocks.gr,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        out["gpu_temp_util_power_clock_mem"] = r.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        out["gpu"] = f"<{type(exc).__name__}>"
    try:
        mhz = [float(l.split(":")[1]) for l in pathlib.Path("/proc/cpuinfo").read_text().splitlines()
               if l.startswith("cpu MHz")]
        out["cpu_mhz_mean"] = round(sum(mhz) / len(mhz))
    except Exception:  # noqa: BLE001
        pass
    return out


def stats_ms(xs):
    a = 1e3 * np.asarray(xs, dtype=np.float64)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "min": float(a.min()), "sd": float(a.std(ddof=1))}


def run_pass(traj: str, pass_i: int) -> dict:
    """One timed traversal. Iteration i writes pose i mod N and expects to read pose (i-1) mod N."""
    POS, LOOK = ns.trajectory(traj, N)
    therm = np.zeros((N, H, W, 4), np.uint8)
    rgbs = np.zeros((N, H, W, 4), np.uint8) if USE_RGB else None
    filled = np.zeros(N, dtype=np.int64)
    acc = {k: [] for k in (*BLOCKS, "total")}
    lag_hist: collections.Counter = collections.Counter()
    events, warm_events = [], 0
    hist: list[int] = []
    prev = None
    state0 = host_state()

    # Park one pose before the lead-in and let the renderer settle there.
    first = -args.warm
    ops.set_pose(POS[(first - 1) % N], LOOK[(first - 1) % N])
    for _ in range(4):
        rep.orchestrator.step(rt_subframes=1)
    hist.append((first - 1) % N)
    wall0 = wall1 = 0.0
    for i in range(first, N + 1):
        pose_i = i % N
        if i == 1:
            wall0 = pc()
        t0 = pc()
        ops.set_pose(POS[pose_i], LOOK[pose_i])
        ta = pc()
        inject = bool(args.inject_sync_step) and i >= 1 and i % args.inject_sync_step == 0
        rep.orchestrator.step(rt_subframes=1, wait_for_render=inject)
        t1 = pc()
        payload = read_all()
        t2 = pc()
        # ---- freshness: the pose this read shows, by its own stamp
        hist.append(pose_i)
        pos_s, fwd_s = read_stamp(payload)
        cand = set(np.flatnonzero(np.abs(POS - pos_s[None, :]).max(axis=1) < POSE_TOL_M).tolist())
        repeated = prev is not None and float(np.abs(pos_s - prev[0]).max()) < 1e-9 \
            and float(np.abs(fwd_s - prev[1]).max()) < 1e-9
        prev = (pos_s, fwd_s)
        lag = slot = None
        if hist[-2] in cand:
            lag, slot = 1, hist[-2]
        else:
            for back in range(min(len(hist), 16)):
                if hist[-1 - back] in cand:
                    lag, slot = back, hist[-1 - back]
                    break
        looks_right = slot is not None and float(fwd_s @ LOOK[slot]) > 1.0 - 1e-9
        t3 = pc()
        terms = tables(payload)
        t4 = pc()
        kernels(payload, terms)
        t5 = pc()
        th = thermal_host()
        rg = rgb_host(payload) if USE_RGB else None
        timed = i >= 1
        if timed and slot is not None and looks_right:
            therm[slot] = th
            if rg is not None:
                rgbs[slot] = rg
            filled[slot] += 1
        t6 = pc()
        if i == N:
            wall1 = t6
        bad = lag != 1 or repeated or not looks_right
        if timed:
            for k, v in zip((*BLOCKS, "total"),
                            (ta - t0, t1 - ta, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t6 - t0), strict=True):
                acc[k].append(v)
            lag_hist["unmatched" if lag is None else str(lag)] += 1
            if bad:
                events.append({"iter": i, "wrote_pose": pose_i, "lag": lag, "stored_as": slot,
                               "repeated": bool(repeated), "orientation_ok": bool(looks_right)})
        elif bad and i > first + 2:
            warm_events += 1

    missing = np.flatnonzero(filled == 0).tolist()
    for j in missing:  # off the clock: a pose that never arrived is rendered synchronously
        ops.set_pose(POS[j], LOOK[j])
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=1)
        payload = read_all()
        kernels(payload, tables(payload))
        therm[j] = thermal_host()
        if USE_RGB:
            rgbs[j] = rgb_host(payload)

    total = np.asarray(acc["total"])
    fps_inst = 1.0 / total
    out = {
        "trajectory": traj, "pass": pass_i, "n_frames": N,
        "fps": {"sequence": float(N / total.sum()), "wall": float(N / (wall1 - wall0)),
                "min": float(fps_inst.min()), "max": float(fps_inst.max()), "sd": float(fps_inst.std(ddof=1)),
                "p05": float(np.percentile(fps_inst, 5)), "p50": float(np.percentile(fps_inst, 50))},
        "ms": {k: stats_ms(v) for k, v in acc.items()},
        "frac_over_33ms": float((total > 1 / 30).mean()),
        "frac_over_16p7ms": float((total > 1 / 60).mean()),
        "freshness": {
            "lag_hist": dict(lag_hist), "anomalous_reads": len(events),
            "repeated_reads": sum(1 for e in events if e["repeated"]),
            "poses_never_delivered": missing, "poses_delivered_twice": np.flatnonzero(filled > 1).tolist(),
            "recovered_synchronously": len(missing), "lead_in_anomalies": warm_events, "events": events[:40],
        },
        "table_uploads": None, "host_before": state0, "host_after": host_state(),
        "thermal_sha1": hashlib.sha1(therm.tobytes()).hexdigest(),
    }
    m, f = out["ms"], out["fps"]
    print(f"[{args.tag} {traj:8s} #{pass_i}] {f['sequence']:6.1f} fps (min {f['min']:5.1f} max {f['max']:5.1f} sd "
          f"{f['sd']:4.1f}) | {m['total']['mean']:6.2f} ms = pose {m['pose']['mean']:.2f} + render "
          f"{m['render']['mean']:.2f} + get {m['get_data']['mean']:.2f} + fresh {m['fresh']['mean']:.2f} + tables "
          f"{m['tables']['mean']:.2f} + kern {m['kernels']['mean']:.2f} + out {m['output']['mean']:.2f} | p95 "
          f"{m['total']['p95']:.2f} max {m['total']['max']:.2f} | lag {dict(lag_hist)} anomalies {len(events)} "
          f"missing {len(missing)}", flush=True)
    return {"report": out, "thermal": therm, "rgb": rgbs}


# -- reference pass and audit (off the clock) ---------------------------------------------------


def reference(traj: str) -> dict:
    """Every pose rendered synchronously (lag 0, checked), plus one CPU-reference comparison."""
    POS, LOOK = ns.trajectory(traj, N)
    therm = np.zeros((N, H, W, 4), np.uint8)
    rgbs = np.zeros((N, H, W, 4), np.uint8) if USE_RGB else None
    floats, checks, stamp_bad = {}, {}, 0
    for j in range(N):
        ops.set_pose(POS[j], LOOK[j])
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=1)
        payload = read_all()
        pos_s, fwd_s = read_stamp(payload)
        stamp_bad += int(np.abs(pos_s - POS[j]).max() > POSE_TOL_M or fwd_s @ LOOK[j] < 1 - 1e-9)
        t_s = T0 + CLOCK[0]
        kernels(payload, tables(payload))
        therm[j] = thermal_host()
        if USE_RGB:
            rgbs[j] = rgb_host(payload)
        if j in args.stills:
            rad = out2.numpy().copy()
            ids = data_of(payload["instance"]).numpy().reshape(H, W).astype(np.uint32)
            labels = labels_from_payload(payload["instance"])
            floats[j] = {"radiance": rad, "ids": ids, "labels": labels_to_paths(labels)}
            if not checks:  # the device chain against the repo's CPU reference, once per trajectory
                l1, _t, _m, sky = lite.reduced_numpy(ids, labels, camera, t_s)
                dist = data_of(payload["distance"]).numpy().reshape(H, W)
                dist_g = np.where(sky, 0.0, np.nan_to_num(dist, posinf=0.0)).astype(np.float32)
                l2 = apply_layered_gbuffer(cfg.atmosphere, band, t_s, l1, dist_g, q, sky_mask=sky)
                ta = np.asarray(lut.apparent_temperature(rad, q), np.float64)
                tb = np.asarray(lut.apparent_temperature(l2.astype(np.float32), q), np.float64)
                dn_dev = therm[j].astype(np.int16)
                dn_cpu = display_host(l2.astype(np.float32)).astype(np.int16)
                checks = {
                    "frame": j,
                    "s1_bit_identical_px": int((lite.bits(gpu.out.numpy()) == lite.bits(l1)).sum()),
                    "n_px": int(rad.size),
                    "s1s2_vs_cpu_max_rel": float((np.abs(rad.astype(np.float64) - l2) / l2).max()),
                    "s1s2_vs_cpu_max_mK": float(np.abs(ta - tb).max() * 1e3),
                    "display_px_differing_from_cpu": int((dn_dev != dn_cpu).any(axis=-1).sum()),
                    "display_max_code_diff": int(np.abs(dn_dev - dn_cpu).max()),
                }
    return {"thermal": therm, "rgb": rgbs, "floats": floats, "checks": checks, "stamp_bad": stamp_bad}


def audit(run: dict, ref: dict) -> dict:
    """Is every delivered frame the picture of the pose it was filed under?"""
    th, rt = run["thermal"], ref["thermal"]
    diff_px = np.array([int((th[j] != rt[j]).any(axis=-1).sum()) for j in range(N)])
    # If the frames were filed one slot late or early they would match these instead:
    shifted = {s: int(sum(np.array_equal(th[j], rt[(j + s) % N]) for j in range(N))) for s in (-2, -1, 1, 2)}
    out = {"thermal_frames_identical": int((diff_px == 0).sum()), "thermal_worst_frame_px": int(diff_px.max()),
           "thermal_identical_if_shifted": shifted}
    if run["rgb"] is not None:
        a, b = run["rgb"][..., :3].astype(np.float32), ref["rgb"][..., :3].astype(np.float32)
        mse = {s: np.array([float(((a[j] - b[(j + s) % N]) ** 2).mean()) for j in range(N)]) for s in (-1, 0, 1)}
        best = np.argmin(np.stack([mse[-1], mse[0], mse[1]]), axis=0) - 1
        out.update({"rgb_best_match_is_own_pose": int((best == 0).sum()),
                    "rgb_best_match_hist": {str(s): int((best == s).sum()) for s in (-1, 0, 1)},
                    "rgb_mse_own": float(mse[0].mean()),
                    "rgb_mse_neighbour": float(0.5 * (mse[-1].mean() + mse[1].mean()))})
    return out


def object_table(fl: dict) -> list[dict]:
    """Per prim: kinetic T, band emissivity, apparent T the camera reads (median over its pixels)."""
    t_app = np.asarray(lut.apparent_temperature(fl["radiance"], q), np.float64)
    temps = camera.bridge.facet_temperatures()
    by_path = {r.path: int(r.material_id) for r in camera.resolutions}
    eps = np.asarray(ctx["table"].emissivity, dtype=np.float64) if hasattr(ctx["table"], "emissivity") else None
    rows = []
    for ident, path in sorted(fl["labels"].items()):
        px = fl["ids"] == ident
        if ident == 0 or not px.any():
            continue
        mid = by_path.get(path)
        rows.append({"prim": path, "material": ctx["table"].names[mid] if mid is not None else None,
                     "eps_band": None if eps is None or mid is None else float(eps[mid]),
                     "t_kinetic_c": float(temps[path]) - 273.15,
                     "t_apparent_c_median": float(np.median(t_app[px])) - 273.15, "pixels": int(px.sum())})
    bg = fl["ids"] == 0
    if bg.any():
        rows.append({"prim": "<background, id 0>", "material": None, "eps_band": None, "t_kinetic_c": None,
                     "t_apparent_c_median": float(np.median(t_app[bg])) - 273.15, "pixels": int(bg.sum())})
    return rows


# -- disk ---------------------------------------------------------------------------------------


def write_png(path, rgba):
    try:
        from PIL import Image
        Image.fromarray(np.ascontiguousarray(rgba[..., :3])).save(path)
    except ImportError:
        from irsim.io.png import write_png as _w
        _w(path, np.ascontiguousarray(rgba[..., :3]))


def write_mp4(path, frames):
    if shutil.which("ffmpeg") is None:
        return "no ffmpeg"
    h, w = frames.shape[1:3]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(args.video_fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14",
           "-movflags", "+faststart", str(path)]
    r = subprocess.run(cmd, input=np.ascontiguousarray(frames[..., :3]).tobytes(), capture_output=True)
    return "ok" if r.returncode == 0 else r.stderr.decode()[-400:]


def save(traj: str, run: dict, ref: dict | None) -> dict:
    d = OUT / traj
    info = {}
    for name, stack in (("thermal", run["thermal"]), ("rgb", run["rgb"])):
        if stack is None:
            continue
        (d / name).mkdir(parents=True, exist_ok=True)
        for j in range(N):
            write_png(d / name / f"frame_{j:03d}.png", stack[j])
        info[f"{name}.mp4"] = write_mp4(d / f"{name}.mp4", stack)
    if run["rgb"] is not None:
        info["side_by_side.mp4"] = write_mp4(d / "side_by_side.mp4",
                                             np.concatenate([run["thermal"], run["rgb"]], axis=2))
    (d / "stills").mkdir(parents=True, exist_ok=True)
    for j in args.stills:
        write_png(d / "stills" / f"thermal_{j:03d}.png", run["thermal"][j])
        if run["rgb"] is not None:
            write_png(d / "stills" / f"rgb_{j:03d}.png", run["rgb"][j])
        if ref is not None and j in ref["floats"]:
            fl = ref["floats"][j]
            np.save(d / "stills" / f"apparent_t_k_{j:03d}.npy",
                    np.asarray(lut.apparent_temperature(fl["radiance"], q), np.float32))
            np.save(d / "stills" / f"radiance_{j:03d}.npy", fl["radiance"])
    return info


# -- main ---------------------------------------------------------------------------------------

proj = np.asarray(annos["cam"].get_data()["cameraProjection"], dtype=np.float64).reshape(4, 4)
report: dict = {
    "tag": args.tag, "gui": args.gui, "viewport_flag": args.viewport, "rgb": USE_RGB, "inject_sync_step": args.inject_sync_step,
    "resolution": [W, H], "n_frames": N, "palette": args.palette, "t_range_c": list(args.t_range_c),
    "materials": MATERIALS, "frame_period_s": PERIOD,
    "camera": {"authored": {k: v for k, v in camera.authored.items() if not k.startswith("omni:")},
               "hfov_deg_from_rendered_projection": float(np.degrees(2 * np.arctan(1 / proj[0, 0]))),
               "vfov_deg_from_rendered_projection": float(np.degrees(2 * np.arctan(1 / proj[1, 1])))},
    "settings": {k: settings.get(k) for k in (
        "/rtx/rendermode", "/rtx/post/aa/op", "/rtx/post/dlss/execMode", "/rtx/post/tonemap/filmIso", "/app/vsync",
        "/app/runLoops/main/rateLimitEnabled", "/app/runLoops/main/rateLimitFrequency",
        "/app/asyncRendering", "/app/window/width", "/app/window/height",
        "/app/renderer/resolution/width", "/app/renderer/resolution/height", "/app/window/hideUi")},
    "passes": [], "audit": {}, "objects": {}, "saved": {},
}
try:
    from omni.kit.viewport.utility import get_active_viewport
    vp = get_active_viewport()
    # With the window open, SimulationApp's `disable_viewport_updates` leaves the viewport
    # rendering (it read back True and cost the same as viewport on), so it is switched off here.
    if vp is not None and not args.viewport and vp.updates_enabled:
        vp.updates_enabled = False
        report["viewport_forced_off_by_api"] = True
        for _ in range(4):
            rep.orchestrator.step(rt_subframes=1)
    report["viewport_updates_enabled"] = None if vp is None else bool(vp.updates_enabled)
    report["viewport_resolution"] = None if vp is None else list(vp.resolution)
except Exception as exc:  # noqa: BLE001
    report["viewport_updates_enabled"] = f"<{type(exc).__name__}: {exc}>"
print(f"[{args.tag}] gui {args.gui} viewport flag {args.viewport} -> updates_enabled "
      f"{report['viewport_updates_enabled']} | rendermode {report['settings']['/rtx/rendermode']} | "
      f"HFOV {report['camera']['hfov_deg_from_rendered_projection']:.3f} VFOV "
      f"{report['camera']['vfov_deg_from_rendered_projection']:.3f}", flush=True)

TRAJS = ("linear", "circular") if args.trajectory == "both" else (args.trajectory,)
runs: dict[tuple[str, int], dict] = {}
for pass_i in range(1, args.passes + 1):
    for traj in TRAJS:
        r = run_pass(traj, pass_i)
        runs[(traj, pass_i)] = r
        report["passes"].append(r["report"])
        (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))

for traj in TRAJS:
    ref = None
    if not args.no_audit:
        ref = reference(traj)
        report["audit"][traj] = {"reference_stamp_mismatches": ref["stamp_bad"], "device_vs_cpu": ref["checks"],
                                 "passes": {str(i): audit(runs[(traj, i)], ref)
                                            for i in range(1, args.passes + 1)}}
        if ref["floats"]:
            report["objects"][traj] = {str(j): object_table(fl) for j, fl in ref["floats"].items()}
        print(f"[{args.tag} {traj} audit] {json.dumps(report['audit'][traj])}", flush=True)
    if not args.no_save:
        report["saved"][traj] = save(traj, runs[(traj, args.passes)], ref)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))

print(f"wrote {OUT / 'report.json'}", flush=True)
ctx["app"].close()
