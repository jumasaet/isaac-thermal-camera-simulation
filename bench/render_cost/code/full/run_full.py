#!/usr/bin/env python
"""The whole thermal frame on the device, in the Kit loop: measured, and held to `run_frame`.

/tmp only; nothing here edits the repo. Same Kit loop as demo/run_demo.py and stages/run_stages.py
(pose, render with wait_for_render=False, read, freshness by pose stamp, tables, stage-1/2
kernels); what is new is that stages 3-6 run on the device too (fullgpu.FullGpu), so one frame is
one RGBA8 download.

Variants (one process runs all of them, interleaved across passes):

    r0        pose, render, read, freshness: the renderer and the annotators alone, no Warp work of ours
    s12       device stages 1+2 and a synchronisation; nothing shown, nothing downloaded
    12        the lite demo as it was: + fixed-range apparent-T palette, downloaded   (k = 1 only)
    cpu       stages 3-6 on the host from the device radiance (the 17 fps frame)      (k = 1 only)
    cpu_repo  the same with flat field + M9 chain                                     (k = 1 only)
    g         device 3, 4, 5, ADC, 6
    g_ff      + display flat field (the new kernel)
    g_m9      + M9 chain: housing/FPA nodes, drift, defects, replacement, NUC residual, FFC
    g_m9_ff   both: the camera as the repo's render scripts configure it

`--check` (default on) comes first and can stop the run: for a few consecutive poses of each
trajectory, rendered synchronously, the repo's own `run_frame` on the host G-buffer is compared
plane by plane with the device frame -- once with the CPU's noise field fed to the device (mode B:
everything must agree) and once with the device's own generator (mode C: the difference must have
the moments of two independent draws). A failure exits before anything is timed.

Passes per variant and trajectory: `--passes` timed (one synchronisation per frame, as it would
run), then one profiled (a synchronisation after every block: the breakdown), then one untimed
that keeps the float planes of the still poses.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

sys.path.insert(0, "/tmp/irsim_render_cost/demo")
sys.path.insert(0, "/tmp/irsim_render_cost/full")
import common as cm  # noqa: E402
import nave_stage as ns  # noqa: E402

ALL = ("r0", "s12", "12", "cpu", "cpu_repo", "g", "g_ff", "g_m9", "g_m9_ff")
p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("--k", type=int, default=1, help="optics.supersample_factor (render product = k x 640x512)")
p.add_argument("--optics", choices=("repo", "fused"), default="repo")
p.add_argument("--variants", nargs="*", default=["s12", "g", "g_ff", "g_m9", "g_m9_ff"], choices=ALL)
p.add_argument("--trajectory", choices=("linear", "circular", "both"), default="both")
p.add_argument("--n-frames", type=int, default=120)
p.add_argument("--passes", type=int, default=2)
p.add_argument("--no-profile", action="store_true")
p.add_argument("--warm", type=int, default=30)
p.add_argument("--gui", action="store_true")
p.add_argument("--viewport", action="store_true", help="leave viewport updates on")
p.add_argument("--no-rgb", action="store_true", help="do not attach or download the rgb annotator")
p.add_argument("--stills", nargs="*", default=["circular:54", "circular:10", "linear:40", "linear:20"])
p.add_argument("--save-seq", nargs="*", default=[], help="variants whose display sequence is written to disk")
p.add_argument("--no-check", action="store_true")
p.add_argument("--audit-ids", nargs="*", default=[], help="variants audited by the id plane each delivered frame was computed from")
p.add_argument("--audit", nargs="*", default=[], help="variants whose delivered frames are compared with a synchronous re-render of every pose")
p.add_argument("--check-only", action="store_true")
p.add_argument("--tag", required=True)
p.add_argument("--out", default="/tmp/irsim_render_cost/full/out")
args, kit_args = p.parse_known_args()
K = args.k
if K != 1:
    args.variants = [v for v in args.variants if v not in ("12", "cpu", "cpu_repo")]
ns.sensor_ss1_yaml = lambda: cm.sensor_yaml(K)  # the demo's boot, at this run's supersample factor
MATERIALS = ns.load_materials([])
ctx = ns.boot(gui=args.gui, viewport=args.viewport, kit_args=kit_args, materials=MATERIALS)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import warp as wp  # noqa: E402

import fullgpu  # noqa: E402
import lite  # noqa: E402
from irsim.detector.quantise import quantise  # noqa: E402
from irsim.isp.display import run_display_branch  # noqa: E402
from irsim.isp.palette import palette_table, quantise_display  # noqa: E402
from irsim.isp.radiometric import dn_per_kelvin  # noqa: E402
from irsim.noise.three_d import Sigmas7  # noqa: E402
from irsim.pipeline.core import PipelineConfig, PipelineState  # noqa: E402
from irsim.pipeline.frame import run_frame  # noqa: E402
from irsim.pipeline.sensor_chain import attach_sensor_chain  # noqa: E402
from irsim_isaac.pipeline import warp_stages  # noqa: E402
from irsim_isaac.pipeline.material_ids import labels_from_payload, labels_to_paths  # noqa: E402

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
WS, HS = camera.optics.resolution  # the render product: k x the detector grid
H, W = spec.fpa_shape
assert (HS, WS) == (H * K, W * K) and cfg.supersample == K, ((HS, WS), (H, W), K, cfg.supersample)
BITS = spec.fpa.bit_depth
camera._reader.detach()
DEV = "cuda:0"
N = args.n_frames
USE_RGB = not args.no_rgb
STILLS = collections.defaultdict(list)
for item in args.stills:
    tr, _, j = item.partition(":")
    STILLS[tr].append(int(j))
CFG_FF = PipelineConfig.from_sensor(ctx["sensor"], ctx["table"], lut, sky=scene.sky_models[band],
                                    atmosphere=scene.layered, flat_field_enabled=True)
DN_PER_K = float(dn_per_kelvin(cfg.calibration, lut, 300.0))
GAIN = float(cfg.detector.transfer.gain_dn_per_w)
SIGMA_TVH = float(cfg.detector.sigma_signal_dn)


def with_chain(base):
    """A fresh chain: its nodes, drift, defect and FFC state belong to one sequence."""
    return attach_sensor_chain(base, scene.weather, t0_s=scene.t0_s)


def cpu_config(variant: str):
    return with_chain(CFG_FF) if variant in ("cpu_repo", "g_m9_ff") else with_chain(cfg) if variant == "g_m9" \
        else CFG_FF if variant == "g_ff" else cfg


def device_frame(variant: str, profile: bool = False) -> fullgpu.FullGpu:
    ff, m9 = variant.endswith("_ff"), "m9" in variant
    base = CFG_FF if ff else cfg
    return fullgpu.FullGpu(base, device=DEV, flat_field=ff, chain=with_chain(base).chain if m9 else None,
                           optics=args.optics, profile=profile)


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


# -- the lite display of variant 12, unchanged from run_demo.py (k = 1) --------------------------
N_TAB = 16384
T_MIN_K, T_MAX_K = 273.15 + 15.0, 273.15 + 160.0
L_LO = float(lut.lookup(np.float64(T_MIN_K), q)[()])
L_HI = float(lut.lookup(np.float64(T_MAX_K), q)[()])


def display_host(radiance: np.ndarray) -> np.ndarray:
    t_app = np.asarray(lut.apparent_temperature(np.asarray(radiance, np.float32), q), np.float64)
    dn8 = quantise_display((t_app - T_MIN_K) / (T_MAX_K - T_MIN_K))
    rgb = palette_table("ironbow")[dn8]
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


gpu = lite.LiteGpu(camera, (HS, WS), DEV)
out2 = wp.zeros((HS, WS), dtype=wp.float32, device=DEV)
disp = wp.zeros((HS, WS), dtype=wp.uint32, device=DEV) if K == 1 else None
INV_STEP = np.float32((N_TAB - 1) / (L_HI - L_LO))
T0 = camera.scene.t0_s
PERIOD = camera.frame_period_s
CLOCK = [0.0]
ops = ctx["cam_ops"]


def tables(payload):
    camera.bridge.advance_to(CLOCK[0])
    t_s = T0 + CLOCK[0]
    gpu.refresh(labels_from_payload(payload["instance"]), t_s)
    camera.state.t_s = t_s
    CLOCK[0] += PERIOD
    return warp_stages.atmosphere_terms(cfg, camera.state), t_s


def kernels_s12(payload, terms):
    """Gather + stage 1 + stage 2, launched; `out2` is at-aperture radiance on the k x grid."""
    gpu.run(lite.wrap_ids(data_of(payload["instance"])))
    warp_stages.launch_atmosphere(gpu.out, wrap_f32(data_of(payload["distance"])), gpu.sky, terms, out2, DEV)


def rgb_host(payload):
    return data_of(payload["rgb"]).numpy().reshape(HS, WS, -1)[..., :4]


def read_stamp(payload):
    v = np.asarray(payload["cam"]["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
    c2w = np.linalg.inv(v)
    return c2w[3, :3], -c2w[2, :3]


def read_all():
    return {c: a.get_data() for c, a in annos.items()}


POSE_TOL_M = 1e-4
BLOCKS = ("pose", "render", "get_data", "fresh", "tables", "kernels", "gpu", "sync", "cpu", "output")


def stats_ms(xs):
    a = 1e3 * np.asarray(xs, dtype=np.float64)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "min": float(a.min()), "sd": float(a.std(ddof=1))}


def run_pass(variant: str, traj: str, pass_i: int, mode: str = "timed") -> dict:
    """One traversal of `traj`. mode: timed | profile (sync per block) | stills (untimed, keeps planes)."""
    POS, LOOK = ns.trajectory(traj, N)
    is_g, is_cpu = variant.startswith("g"), variant.startswith("cpu")
    g = device_frame(variant, profile=mode == "profile") if is_g else None
    config = cpu_config(variant) if is_cpu else None
    st = PipelineState(t_s=T0 + CLOCK[0])
    shown = variant not in ("s12", "r0")
    therm = np.zeros((N, H, W, 4), np.uint8) if shown else None
    filled = np.zeros(N, dtype=np.int64)
    frame_index_of = np.full(N, -1, dtype=np.int64)
    acc = {k: [] for k in (*BLOCKS, "total")}
    sub_acc = {k: [] for k in fullgpu.BLOCKS}
    lag_hist: collections.Counter = collections.Counter()
    events, hist, prev = [], [], None
    floats: dict[int, dict] = {}
    ids_seq = np.full((N, H, W), 255, np.uint8) if mode == "audit" else None
    first = -args.warm
    ops.set_pose(POS[(first - 1) % N], LOOK[(first - 1) % N])
    for _ in range(4):
        rep.orchestrator.step(rt_subframes=1)
    hist.append((first - 1) % N)
    for i in range(first, N + 1):
        pose_i = i % N
        t0 = pc()
        ops.set_pose(POS[pose_i], LOOK[pose_i])
        ta = pc()
        rep.orchestrator.step(rt_subframes=1, wait_for_render=False)
        t1 = pc()
        payload = read_all()
        t2 = pc()
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
        if variant == "r0":
            t_s, t4 = 0.0, t3
        else:
            terms, t_s = tables(payload)
            t4 = pc()
            kernels_s12(payload, terms)
        if variant == "12":
            wp.launch(_display_kernel, dim=(HS, WS),
                      inputs=[out2, DISP_TAB, np.float32(L_LO), INV_STEP, np.int32(N_TAB)], outputs=[disp], device=DEV)
        if (not is_g or mode == "profile") and variant != "r0":
            wp.synchronize()
        t5 = pc()
        tm = dict.fromkeys(fullgpu.BLOCKS, 0.0)
        fi_now = -1
        th = None
        t6 = t7 = t8 = t5
        if is_g:
            fi_now = g.frame_index
            g.frame(out2, t_s, tm)
            t6 = pc()
            wp.synchronize()
            t7 = t8 = pc()
            th = g.rgba.numpy()
        elif is_cpu:
            st.t_s = t_s
            fi_now = st.frame_index
            th = cm.cpu_tail(out2.numpy(), config, st)["display8"]
            t8 = pc()
        elif variant == "12":
            th = disp.numpy().view(np.uint8).reshape(H, W, 4)
        rg = rgb_host(payload) if USE_RGB else None
        timed = i >= 1
        if timed and slot is not None and looks_right:
            if shown:
                therm[slot] = th
            filled[slot] += 1
            frame_index_of[slot] = fi_now
        t9 = pc()
        if mode == "stills" and timed and is_g and slot in STILLS.get(traj, ()) and slot not in floats:
            floats[slot] = {**g.planes(), "frame_index": fi_now, "housing_k": float(g.housing_k)}
            if variant == "g":
                floats[slot]["radiance_ss"] = out2.numpy().copy()
        if ids_seq is not None and timed and slot is not None and looks_right:
            # after the clock: the id plane this iteration's thermal frame was computed from
            ids_seq[slot] = data_of(payload["instance"]).numpy().reshape(HS, WS)[K // 2::K, K // 2::K].astype(np.uint8)
        bad = lag != 1 or repeated or not looks_right
        if timed:
            for k_, v in zip((*BLOCKS, "total"),
                             (ta - t0, t1 - ta, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6, t8 - t7, t9 - t8, t9 - t0),
                             strict=True):
                acc[k_].append(v)
            for k_ in fullgpu.BLOCKS:
                sub_acc[k_].append(tm[k_])
            lag_hist["unmatched" if lag is None else str(lag)] += 1
            if bad:
                events.append({"iter": i, "lag": lag, "stored_as": slot, "repeated": bool(repeated)})
        del rg
    total = np.asarray(acc["total"])
    fps_inst = 1.0 / total
    rep_ = {
        "variant": variant, "trajectory": traj, "pass": pass_i, "mode": mode,
        "fps": {"sequence": float(N / total.sum()), "min": float(fps_inst.min()), "max": float(fps_inst.max()),
                "sd": float(fps_inst.std(ddof=1)), "p05": float(np.percentile(fps_inst, 5)),
                "p50": float(np.percentile(fps_inst, 50))},
        "ms": {k_: stats_ms(v) for k_, v in acc.items()},
        "sub_ms": {k_: float(1e3 * np.mean(v)) for k_, v in sub_acc.items() if np.any(v)},
        "frac_over_33ms": float((total > 1 / 30).mean()), "frac_over_16p7ms": float((total > 1 / 60).mean()),
        "freshness": {"lag_hist": dict(lag_hist), "anomalous_reads": len(events),
                      "poses_never_delivered": np.flatnonzero(filled == 0).tolist(), "events": events[:20]},
    }
    if is_g:
        rep_["housing_k_last"] = float(g.housing_k)
        if g.chain is not None:
            rep_["m9"] = {"dT_eff_k": float(g.chain.ffc.delta_t_eff_k), "replacement_passes": int(g.replacement_passes),
                          "defects": int(g.chain.bad_pixels.count)}
    m = rep_["ms"]
    sub = " ".join(f"{k_}={v:.2f}" for k_, v in rep_["sub_ms"].items())
    print(f"[{args.tag} {variant:8s} {traj:8s} #{pass_i} {mode:7s}] {rep_['fps']['sequence']:6.1f} fps (min {rep_['fps']['min']:.1f} max "
          f"{rep_['fps']['max']:.1f} sd {rep_['fps']['sd']:.1f}) | {m['total']['mean']:6.2f} ms (p95 {m['total']['p95']:.2f} max "
          f"{m['total']['max']:.2f}) = pose {m['pose']['mean']:.2f} + render {m['render']['mean']:.2f} + get {m['get_data']['mean']:.2f} + "
          f"fresh {m['fresh']['mean']:.2f} + tables {m['tables']['mean']:.2f} + s12 {m['kernels']['mean']:.2f} + gpu {m['gpu']['mean']:.2f} + "
          f"sync {m['sync']['mean']:.2f} + cpu {m['cpu']['mean']:.2f} + out {m['output']['mean']:.2f} | {sub} | lag {dict(lag_hist)} "
          f"anomalies {len(events)}", flush=True)
    return {"report": rep_, "thermal": therm, "floats": floats, "frame_index_of": frame_index_of, "ids_seq": ids_seq}


# -- off the clock: synchronous renders and the comparison with run_frame -------------------------


def render_sync(pos, look):
    ops.set_pose(pos, look)
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=1)
    payload = read_all()
    pos_s, fwd_s = read_stamp(payload)
    if np.abs(pos_s - pos).max() > POSE_TOL_M or fwd_s @ look < 1 - 1e-9:
        raise RuntimeError("synchronous render delivered another pose")
    return payload


def gbuffer_planes(payload, t_s):
    """The reduced path's G-buffer as the plane dict `run_frame` takes (host, CPU reference)."""
    ids = data_of(payload["instance"]).numpy().reshape(HS, WS).astype(np.uint32)
    labels = labels_from_payload(payload["instance"])
    _l1, temp, mat, sky = lite.reduced_numpy(ids, labels, camera, t_s)
    dist = data_of(payload["distance"]).numpy().reshape(HS, WS)
    dist_g = np.where(sky, 0.0, np.nan_to_num(dist, posinf=0.0)).astype(np.float32)
    return {"temperature_k": temp, "material_id": mat, "sky_mask": sky, "distance_m": dist_g,
            # the enclosure is isothermal: L_env is the same at any V_s (lite's own premise)
            "sky_view_factor": np.full((HS, WS), 0.5, np.float32)}, ids, labels


def cpu_planes(ref, config) -> dict:
    """`run_frame`'s Outputs as the plane dict `compare` takes. The shown DN and y are not in
    Outputs, so they are recomputed with the same two calls -- and checked against its display8."""
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


def equivalence(variant: str, traj: str, start: int, n: int = 4) -> dict:
    """`run_frame` (all host, from the G-buffer) against the device frame, n consecutive poses."""
    POS, LOOK = ns.trajectory(traj, N)
    config = cpu_config(variant)
    gb, gc = device_frame(variant), device_frame(variant)
    st = PipelineState(t_s=T0 + CLOCK[0])
    sig7 = Sigmas7.from_ratios(SIGMA_TVH, spec.noise.sigma_ratios())
    bad_map = None if config.chain is None else np.asarray(config.chain.bad_pixels.mask)
    rows, stats, keep = [], [], {}
    for j in range(start, start + n):
        payload = render_sync(POS[j % N], LOOK[j % N])
        terms, t_s = tables(payload)
        kernels_s12(payload, terms)
        wp.synchronize()
        planes, _ids, _labels = gbuffer_planes(payload, t_s)
        st.t_s = t_s
        fi = st.frame_index
        ref = run_frame(planes, config, st)
        cpu = cpu_planes(ref, config)
        noise = cm.cpu_noise_field(config, fi)
        rts = None if config.chain is None else np.asarray(config.chain.defect_state.bad, dtype=np.uint8)
        gb.frame(out2, t_s, inject_noise=noise, rts_override=rts)
        b = gb.planes()
        row = cm.compare(cpu, b, DN_PER_K, GAIN)
        row.update(pose=j % N, frame_index=fi, housing_k_cpu=float(st.housing_temp_k), housing_k_dev=float(gb.housing_k))
        if abs(row["housing_k_cpu"] - row["housing_k_dev"]) > 1e-9:
            row["fails"].append("housing")
        if ref.report is not None:
            ev_c, ev_g = ref.report.ffc, gb.last_event
            if (ev_c.fired, ev_c.frozen) != (ev_g.fired, ev_g.frozen):
                row["fails"].append("ffc_schedule")
            row["defects_active_cpu"], row["defects_active_dev"] = int(ref.report.defects_active), int(b["active"].sum())
        rows.append(row)
        gc.frame(out2, t_s)
        c = gc.planes()
        s = cm.noise_stats(cpu["signal"], c["signal"], sig7, exclude=bad_map)
        s.update(pose=j % N, display_mean_code_cpu=float(cpu["display8"][..., 0].mean()),
                 display_mean_code_dev=float(c["display8"][..., 0].mean()),
                 display_p01_p99_cpu=[float(x) for x in np.percentile(cpu["display8"][..., 0], (1, 50, 99))],
                 display_p01_p99_dev=[float(x) for x in np.percentile(c["display8"][..., 0], (1, 50, 99))])
        stats.append(s)
        if j % N in STILLS.get(traj, ()):
            keep = {"cpu_display8": cpu["display8"], "dev_display8_same_noise": b["display8"],
                    "dev_display8_own_noise": c["display8"], "cpu_dn16": cpu["dn16"], "dev_dn16_same_noise": b["dn16"],
                    "pose": j % N}
    fails = sorted({f for r in rows for f in r["fails"]})
    zmax = {k_: float(np.max(np.abs([s[k_] for s in stats]))) for k_ in ("z_px", "z_row", "z_col", "z_mean")}
    if max(zmax.values()) > 4.0:
        fails.append("noise_statistics")
    worst = {k_: max(r[k_] for r in rows) for k_ in rows[0] if isinstance(rows[0][k_], (int, float)) and k_ not in ("pose", "frame_index", "n_px")}
    print(f"[{args.tag} check {variant:8s} {traj:8s} poses {start % N}..{(start + n - 1) % N}] B: flux {worst['flux_max_rel']:.1e} rel | signal max "
          f"{worst['signal_max_abs_dn']:.4f} DN = {worst['signal_max_abs_mk_at_300k']:.2f} mK (mean {worst['signal_mean_abs_dn']:.4f}) | dn16 "
          f"{worst['dn16_px_differing']} px ({100 * worst['dn16_frac_differing']:.2f} %) max {worst['dn16_max_code']} | shown DN "
          f"{worst['display_dn_px_differing']} px max {worst['display_dn_max_code']} | display8 {worst['display8_px_differing']} px max "
          f"{worst['display8_max_code']}; away from a DN change {worst['display8_far_px_differing']} px max {worst['display8_far_max_code']} | "
          f"housing {rows[-1]['housing_k_cpu']:.3f}/{rows[-1]['housing_k_dev']:.3f} K || C: sd {np.mean([s['sd_px'] for s in stats]):.3f} "
          f"(exp {stats[0]['sd_px_expected']:.3f}) rows {np.mean([s['sd_row'] for s in stats]):.3f} ({stats[0]['sd_row_expected']:.3f}) cols "
          f"{np.mean([s['sd_col'] for s in stats]):.3f} ({stats[0]['sd_col_expected']:.3f}) max|z| {max(zmax.values()):.2f} || "
          f"{'FAILS ' + str(fails) if fails else 'ok'}", flush=True)
    return {"variant": variant, "trajectory": traj, "start": start % N, "rows": rows, "stats": stats, "worst": worst,
            "noise_max_abs_z": zmax, "fails": fails, "keep": keep}


def audit(variant: str, traj: str, therm: np.ndarray) -> dict:
    """Is every delivered frame the picture of the pose it was filed under? Checks the stamp itself.

    Every pose is rendered again synchronously (lag 0, by its own stamp) through a fresh device
    camera, in path order so the membrane carries the same motion. The noise draws differ, so the
    test is which re-rendered pose a delivered frame is closest to: its own, or a neighbour.
    """
    POS, LOOK = ns.trajectory(traj, N)
    g = device_frame(variant)
    ref = np.zeros((N, H, W), np.uint8)
    for j in range(-3, N):
        payload = render_sync(POS[j % N], LOOK[j % N])
        terms, t_s = tables(payload)
        kernels_s12(payload, terms)
        g.frame(out2, t_s)
        wp.synchronize()
        if j >= 0:
            ref[j] = g.rgba.numpy()[..., 0]
    a, b = therm[..., 0].astype(np.float32), ref.astype(np.float32)
    shifts = (-2, -1, 0, 1, 2)
    mse = np.stack([[float(((a[j] - b[(j + s) % N]) ** 2).mean()) for j in range(N)] for s in shifts])
    best = np.asarray(shifts)[np.argmin(mse, axis=0)]
    return {"frames_closest_to_own_pose": int((best == 0).sum()), "n": N,
            "closest_hist": {str(s): int((best == s).sum()) for s in shifts},
            "mse_own_pose": float(mse[2].mean()), "mse_neighbour_pose": float(0.5 * (mse[1].mean() + mse[3].mean())),
            "worst_ratio_own_over_best_neighbour": float((mse[2] / np.minimum(mse[1], mse[3])).max())}


def audit_ids(variant: str, traj: str) -> dict:
    """Noise-free freshness audit. One more traversal with the variant's whole frame in the loop,
    keeping (after the clock) the instance-id plane each iteration's frame was computed from, filed
    under the pose its stamp names. Every pose is then rendered synchronously and the id planes are
    compared exactly. A pose whose id plane equals a neighbour's cannot tell fresh from stale and
    is counted apart."""
    POS, LOOK = ns.trajectory(traj, N)
    r = run_pass(variant, traj, 0, "audit")
    got = r["ids_seq"]
    ref = np.zeros((N, H, W), np.uint8)
    for j in range(N):
        payload = render_sync(POS[j], LOOK[j])
        ref[j] = data_of(payload["instance"]).numpy().reshape(HS, WS)[K // 2::K, K // 2::K].astype(np.uint8)
    own = np.array([int((got[j] != ref[j]).sum()) for j in range(N)])
    prev = np.array([int((got[j] != ref[(j - 1) % N]).sum()) for j in range(N)])
    nxt = np.array([int((got[j] != ref[(j + 1) % N]).sum()) for j in range(N)])
    amb = np.array([np.array_equal(ref[j], ref[(j - 1) % N]) or np.array_equal(ref[j], ref[(j + 1) % N]) for j in range(N)])
    out = {"identical_to_own_pose": int((own == 0).sum()), "n": N, "worst_px_differing_from_own": int(own.max()),
           "identical_to_previous_pose": int((prev == 0).sum()), "identical_to_next_pose": int((nxt == 0).sum()),
           "poses_identical_to_a_neighbour_in_the_reference": int(amb.sum()),
           "distinguishable_and_identical_to_own": int(((own == 0) & ~amb).sum()), "distinguishable": int((~amb).sum()),
           "min_px_between_neighbouring_distinguishable_poses": int(min(prev[~amb].min(), nxt[~amb].min())) if (~amb).any() else None,
           "lag_hist": r["report"]["freshness"]["lag_hist"], "ms": r["report"]["ms"]["total"]["mean"]}
    return out


# -- main ---------------------------------------------------------------------------------------
report: dict = {
    "tag": args.tag, "gui": args.gui, "viewport_flag": args.viewport, "rgb": USE_RGB, "k": K, "optics": args.optics,
    "render_product": [WS, HS], "resolution": [W, H], "n_frames": N, "frame_period_s": PERIOD, "variants": args.variants,
    "epp": pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference").read_text().strip(),
    "rendermode": settings.get("/rtx/rendermode"),
    "sensor": {"name": spec.name, "psf_shape": None if cfg.psf is None else list(cfg.psf.shape), "sigma_tvh_dn": SIGMA_TVH,
               "dn_per_k_300": DN_PER_K, "isp": spec.isp.model_dump(mode="json"),
               "t_housing_cal_k": float(cfg.t_housing_cal_k)},
    "tolerances": {**cm.TOL, "signal_mk": cm.MK_TOL, "noise_abs_z": 4.0},
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
      f"{report['rendermode']} | epp {report['epp']} | k {K} render product {WS}x{HS} | optics {args.optics} psf "
      f"{report['sensor']['psf_shape']} | rgb {USE_RGB} | sigma_tvh {SIGMA_TVH:.3f} DN, {DN_PER_K:.1f} DN/K", flush=True)
TRAJS = ("linear", "circular") if args.trajectory == "both" else (args.trajectory,)


def dump():
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))


if not args.no_check:
    keeps = {}
    for traj in TRAJS:
        starts = [j - 3 for j in STILLS.get(traj, [3])]
        for variant in ("g", "g_m9_ff"):
            for s0 in starts:
                e = equivalence(variant, traj, s0)
                kp = e.pop("keep")
                if kp:
                    keeps[(variant, traj, kp["pose"])] = kp
                report["check"].append(e)
    for (variant, traj, pose), kp in keeps.items():
        (OUT / traj).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / traj / f"check_{variant}_{pose:03d}.npz", **{k_: v for k_, v in kp.items() if k_ != "pose"})
    report["check_fails"] = sorted({f"{e['variant']}/{e['trajectory']}@{e['start']}: {f}" for e in report["check"] for f in e["fails"]})
    dump()
    if report["check_fails"]:
        print(f"[{args.tag}] CHECK FAILED -> {report['check_fails']}; nothing is timed", flush=True)
        ctx["app"].close()
        sys.exit(3)
    print(f"[{args.tag}] check passed: the device frame is run_frame's within the repo's tolerances", flush=True)
if args.check_only:
    print(f"wrote {OUT / 'report.json'}", flush=True)
    ctx["app"].close()
    sys.exit(0)

for traj in TRAJS:
    for variant in args.audit_ids:
        a = audit_ids(variant, traj)
        report.setdefault("audit_ids", {})[f"{variant}/{traj}"] = a
        print(f"[{args.tag} audit-ids {variant} {traj}] {json.dumps(a)}", flush=True)
        dump()
runs: dict = {}
for pass_i in range(1, args.passes + 1):
    for traj in TRAJS:
        for variant in args.variants:
            r = run_pass(variant, traj, pass_i)
            report["passes"].append(r["report"])
            if pass_i == args.passes and variant in args.audit:
                a = audit(variant, traj, r["thermal"])
                report.setdefault("audit", {})[f"{variant}/{traj}"] = a
                print(f"[{args.tag} audit {variant} {traj}] {json.dumps(a)}", flush=True)
            if pass_i == args.passes and variant in args.save_seq:
                (OUT / traj).mkdir(parents=True, exist_ok=True)
                np.save(OUT / traj / f"display_seq_{variant}.npy", r["thermal"][..., :3])
                np.save(OUT / traj / f"frame_index_{variant}.npy", r["frame_index_of"])
            del r
            dump()
for traj in TRAJS:
    for variant in args.variants:
        if not variant.startswith("g"):
            continue
        if not args.no_profile:
            report["passes"].append(run_pass(variant, traj, 0, "profile")["report"])
        r = run_pass(variant, traj, 0, "stills")
        (OUT / traj).mkdir(parents=True, exist_ok=True)
        for j, fl in r["floats"].items():
            np.savez_compressed(OUT / traj / f"still_{variant}_{j:03d}.npz", **fl)
        missing = [j for j in STILLS.get(traj, ()) if j not in r["floats"]]
        if missing:
            print(f"[{args.tag}] WARNING no float planes for {variant} {traj}:{missing}", flush=True)
        dump()
    # the id plane of each still pose, for region statistics (native grid: every k-th sample)
    POS, LOOK = ns.trajectory(traj, N)
    for j in STILLS.get(traj, ()):
        payload = render_sync(POS[j], LOOK[j])
        _planes, ids, labels = gbuffer_planes(payload, T0 + CLOCK[0])
        np.savez_compressed(OUT / traj / f"meta_{j:03d}.npz", ids=ids,
                            labels=json.dumps({str(k_): v for k_, v in labels_to_paths(labels).items()}))
dump()
print(f"wrote {OUT / 'report.json'}", flush=True)
ctx["app"].close()
