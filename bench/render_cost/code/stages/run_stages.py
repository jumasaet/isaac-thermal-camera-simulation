#!/usr/bin/env python
"""What do stages 3-6 cost and change when added, on the CPU, to the lite GPU path?

/tmp only; nothing here edits the repo. Same Kit loop as demo/run_demo.py (pose, render with
wait_for_render=False, read, freshness by pose stamp, tables, stage-1/2 kernels); the only
addition is one block, `cpu`, that downloads the stage-2 radiance and runs the repo's own CPU
stage functions on it. One process, every variant, interleaved across passes.

Variants (cumulative; the names are the brief's):

    12       lite as it is: stages 1+2 on the device, fixed-range apparent-T palette on the device
    123      + stage 3   apply_optics (PSF, box, aperture, cos4, housing); shown through the same
                         fixed-range palette, after invert_optics takes the radiometry back out
    1236     + stage 6   display branch on DN16 (AGC, gamma, DDE, polarity, palette). The linear
                         flux -> DN transfer and the ADC are needed to get there and are timed apart
    12356    + noise     sigma_TVH per pixel (the repo keeps it in stage 4, frame_from_signal) and
                         the six correlated NVESD components (stage 5, NoiseStage.apply)
    123456   + lag       the membrane IIR, which is what is left of stage 4 -> this is run_frame
                         from stage 3 on, as the demo's PipelineConfig has it (no chain, no flat field)
    repo     NOT in the brief: the same, configured as the repo's render scripts configure a
                         camera -- flat_field_enabled=True and attach_sensor_chain (M9: housing node,
                         drifting pattern, defects, replacement, NUC residual, FFC)

`--check` (default on) runs the repo's whole `run_frame` on the same G-buffer for a few consecutive
poses and compares it with variant 123456 fed from the device radiance: that is the host/device
seam, and the replication of run_frame's tail in `cpu_chain`, tested together.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

sys.path.insert(0, "/tmp/irsim_render_cost/demo")
import nave_stage as ns  # noqa: E402

ALL = ("12", "123", "1236", "12356", "123456", "repo")
p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("--variants", nargs="*", default=list(ALL), choices=ALL)
p.add_argument("--trajectory", choices=("linear", "circular", "both"), default="both")
p.add_argument("--n-frames", type=int, default=120)
p.add_argument("--passes", type=int, default=2)
p.add_argument("--warm", type=int, default=30)
p.add_argument("--gui", action="store_true")
p.add_argument("--no-rgb", action="store_true")
p.add_argument("--palette", default="ironbow")
p.add_argument("--t-range-c", type=float, nargs=2, default=(15.0, 160.0))
p.add_argument("--stills", nargs="*", default=["circular:54", "circular:10", "linear:40", "linear:20"])
p.add_argument("--slow-factor", type=int, default=10,
               help="off the clock: the still poses again with the camera this many times slower")
p.add_argument("--no-check", action="store_true")
p.add_argument("--tag", required=True)
p.add_argument("--out", default="/tmp/irsim_render_cost/stages/out")
args, kit_args = p.parse_known_args()
MATERIALS = ns.load_materials([])
ctx = ns.boot(gui=args.gui, viewport=False, kit_args=kit_args, materials=MATERIALS)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import warp as wp  # noqa: E402

import lite  # noqa: E402
from irsim.detector.quantise import dn_max_for_bits, quantise  # noqa: E402
from irsim.isp.display import run_display_branch  # noqa: E402
from irsim.isp.palette import palette_table, quantise_display  # noqa: E402
from irsim.isp.radiometric import apparent_temperature, dn_per_kelvin  # noqa: E402
from irsim.optics.stage import apply_optics, invert_optics, shutter_flux  # noqa: E402
from irsim.pipeline.core import PipelineConfig, PipelineState  # noqa: E402
from irsim.pipeline.detector import bolometer_lag, lag_interval_s  # noqa: E402
from irsim.pipeline.frame import _scene_radiance_from_signal, run_frame  # noqa: E402
from irsim.pipeline.sensor_chain import attach_sensor_chain  # noqa: E402
from irsim_isaac.pipeline import warp_stages  # noqa: E402
from irsim_isaac.pipeline.material_ids import labels_from_payload  # noqa: E402

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
W, H = camera.optics.resolution
K = cfg.supersample
BITS = spec.fpa.bit_depth
camera._reader.detach()
DEV = "cuda:0"
N = args.n_frames
USE_RGB = not args.no_rgb
T_MIN_K, T_MAX_K = (273.15 + t for t in args.t_range_c)
STILLS = collections.defaultdict(list)
for item in args.stills:
    tr, _, j = item.partition(":")
    STILLS[tr].append(int(j))

# The repo-configured camera: flat field on, M9 chain attached (scripts/render_aerial_demo.py).
CFG_REPO_BASE = PipelineConfig.from_sensor(ctx["sensor"], ctx["table"], lut, sky=scene.sky_models[band],
                                           atmosphere=scene.layered, flat_field_enabled=True)


def repo_config() -> PipelineConfig:
    """A fresh chain each pass: its drift, defect and FFC state belong to one sequence."""
    return attach_sensor_chain(CFG_REPO_BASE, scene.weather, t0_s=scene.t0_s)


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


# -- the lite display, unchanged from run_demo.py ----------------------------------------------
N_TAB = 16384
L_LO = float(lut.lookup(np.float64(T_MIN_K), q)[()])
L_HI = float(lut.lookup(np.float64(T_MAX_K), q)[()])


def display_host(radiance: np.ndarray) -> np.ndarray:
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
blur_dev = wp.zeros((H, W), dtype=wp.float32, device=DEV)
disp = wp.zeros((H, W), dtype=wp.uint32, device=DEV)
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


def launch_display(src):
    wp.launch(_display_kernel, dim=(H, W),
              inputs=[src, DISP_TAB, np.float32(L_LO), INV_STEP, np.int32(N_TAB)], outputs=[disp], device=DEV)


def kernels(payload, terms, with_display: bool):
    gpu.run(lite.wrap_ids(data_of(payload["instance"])))
    warp_stages.launch_atmosphere(gpu.out, wrap_f32(data_of(payload["distance"])), gpu.sky, terms, out2, DEV)
    if with_display:
        launch_display(out2)
    wp.synchronize()


def thermal_host():
    return disp.numpy().view(np.uint8).reshape(H, W, 4)


def rgb_host(payload):
    return data_of(payload["rgb"]).numpy().reshape(H, W, -1)[..., :4]


def read_stamp(payload):
    v = np.asarray(payload["cam"]["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
    c2w = np.linalg.inv(v)
    return c2w[3, :3], -c2w[2, :3]


def read_all():
    return {c: a.get_data() for c, a in annos.items()}


# -- stages 3-6 on the host, the repo's functions in run_frame's order --------------------------
CPU_BLOCKS = ("download", "s3_optics", "s4_transfer", "s4_lag", "s5_tvh", "s5_correlated", "m9_chain",
              "adc", "s6_flat_field", "s6_display", "fixed_display")


def cpu_chain(variant: str, config: PipelineConfig, st: PipelineState, tm: dict, keep: dict | None = None):
    """`out2` (device, stage-2 radiance) -> RGBA8 (host). Mirrors irsim/pipeline/frame.py:300-395.

    `tm` accumulates seconds per block. `keep`, when given, receives the intermediate planes.
    """
    t = pc()
    rad = out2.numpy()  # float32 (H, W): the only host/device crossing on the way in
    tm["download"] += pc() - t
    chain = config.chain
    t_fpa_k = st.housing_temp_k
    if chain is not None:
        t = pc()
        housing_k, t_fpa_k = chain.begin_frame(st.t_s, st.frame_index)
        st.housing_temp_k = housing_k
        tm["m9_chain"] += pc() - t
    t = pc()
    lb_h = float(lut.lookup(st.housing_temp_k, q)[()])
    flux = apply_optics(rad, spec, lb_h, supersample=K, psf=config.psf)
    tm["s3_optics"] += pc() - t
    if keep is not None:
        keep["flux"] = flux
    if variant == "123":
        # No detector and no ISP yet: show the blurred radiance through the display 12 uses.
        t = pc()
        scene_l = invert_optics(flux, spec, lb_h)
        blur_dev.assign(scene_l)
        launch_display(blur_dev)
        rgba = thermal_host()
        tm["fixed_display"] += pc() - t
        if keep is not None:
            keep["scene_radiance"] = scene_l
        st.advance()
        return rgba
    det = config.detector
    fi = st.frame_index
    t = pc()
    signal = det.noiseless_signal_dn(flux)
    tm["s4_transfer"] += pc() - t
    if variant in ("123456", "repo"):
        t = pc()
        signal = bolometer_lag(signal, config, st, lag_interval_s(config, st))
        tm["s4_lag"] += pc() - t
    if keep is not None:
        keep["signal_noiseless"] = signal
    if variant != "1236":
        t = pc()
        frame = det.frame_from_signal(signal, fi, config.sensor_seed)
        tm["s5_tvh"] += pc() - t
        t = pc()
        fixed = None if chain is None else chain.fixed_pattern
        signal = config.noise.apply(frame.signal_dn, frame.sigma_dn, fi, fixed_override=fixed)
        tm["s5_correlated"] += pc() - t
    if chain is not None:
        t = pc()
        if chain.needs_shutter_frame(fi):
            lb_shutter = float(lut.lookup(t_fpa_k, q)[()])
            chain.record_shutter(det.noiseless_signal_dn(shutter_flux(spec, lb_shutter, lb_h)), config.flat_field)
        signal, _report = chain.finish_frame(signal, dn_max_for_bits(BITS), fi, t_fpa_k)
        tm["m9_chain"] += pc() - t
    t = pc()
    dn16 = quantise(signal, BITS)
    tm["adc"] += pc() - t
    display_dn = dn16
    flat = config.flat_field
    if chain is not None and chain.display_nuc is not None:
        flat = chain.display_nuc
    if flat is not None:
        t = pc()
        display_dn = quantise(flat.apply(dn16), BITS)
        tm["s6_flat_field"] += pc() - t
    t = pc()
    display = run_display_branch(display_dn, spec.isp, BITS)
    tm["s6_display"] += pc() - t
    if keep is not None:
        keep.update(signal=signal, dn16=dn16, display_dn=display_dn, y=display.y)
    st.advance()
    return display.display8


def config_for(variant: str) -> PipelineConfig:
    return repo_config() if variant == "repo" else cfg


POSE_TOL_M = 1e-4
BLOCKS = ("pose", "render", "get_data", "fresh", "tables", "kernels", "cpu", "output")


def stats_ms(xs):
    a = 1e3 * np.asarray(xs, dtype=np.float64)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "min": float(a.min()), "sd": float(a.std(ddof=1))}


def run_pass(variant: str, traj: str, pass_i: int) -> dict:
    """One timed traversal of `traj` with `variant`'s host stages in the loop."""
    POS, LOOK = ns.trajectory(traj, N)
    config = config_for(variant)
    st = PipelineState(t_s=T0 + CLOCK[0])
    therm = np.zeros((N, H, W, 4), np.uint8)
    filled = np.zeros(N, dtype=np.int64)
    frame_index_of = np.full(N, -1, dtype=np.int64)
    acc = {k: [] for k in (*BLOCKS, "total")}
    cpu_acc = {k: [] for k in CPU_BLOCKS}
    lag_hist: collections.Counter = collections.Counter()
    events, hist, prev = [], [], None
    floats: dict[int, dict] = {}
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
        terms, t_s = tables(payload)
        t4 = pc()
        kernels(payload, terms, with_display=variant == "12")
        t5 = pc()
        timed = i >= 1
        tm = dict.fromkeys(CPU_BLOCKS, 0.0)
        keep = {} if (timed and slot in STILLS.get(traj, ()) and pass_i == args.passes) else None
        if variant == "12":
            th = thermal_host()
        else:
            st.t_s = t_s
            if keep is not None:
                keep["frame_index"] = st.frame_index
            fi_now = st.frame_index
            th = cpu_chain(variant, config, st, tm, keep)
        t6 = pc()
        rg = rgb_host(payload) if USE_RGB else None
        if timed and slot is not None and looks_right:
            therm[slot] = th
            filled[slot] += 1
            if variant != "12":
                frame_index_of[slot] = fi_now
        t7 = pc()
        if keep is not None:
            floats[slot] = {k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in keep.items()}
        if timed and variant == "12" and slot in STILLS.get(traj, ()) and pass_i == args.passes:
            floats[slot] = {"scene_radiance": out2.numpy().copy()}
        bad = lag != 1 or repeated or not looks_right
        if timed:
            for k, v in zip((*BLOCKS, "total"),
                            (ta - t0, t1 - ta, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6, t7 - t0), strict=True):
                acc[k].append(v)
            for k in CPU_BLOCKS:
                cpu_acc[k].append(tm[k])
            lag_hist["unmatched" if lag is None else str(lag)] += 1
            if bad:
                events.append({"iter": i, "lag": lag, "stored_as": slot, "repeated": bool(repeated)})
        del rg
    total = np.asarray(acc["total"])
    fps_inst = 1.0 / total
    rep_ = {
        "variant": variant, "trajectory": traj, "pass": pass_i,
        "fps": {"sequence": float(N / total.sum()), "min": float(fps_inst.min()), "p05": float(np.percentile(fps_inst, 5)),
                "p50": float(np.percentile(fps_inst, 50)), "max": float(fps_inst.max())},
        "ms": {k: stats_ms(v) for k, v in acc.items()},
        "cpu_ms": {k: stats_ms(v) for k, v in cpu_acc.items()},
        "frac_over_33ms": float((total > 1 / 30).mean()),
        "freshness": {"lag_hist": dict(lag_hist), "anomalous_reads": len(events),
                      "poses_never_delivered": np.flatnonzero(filled == 0).tolist(), "events": events[:20]},
    }
    m = rep_["ms"]
    cpu_txt = " ".join(f"{k}={v['mean']:.2f}" for k, v in rep_["cpu_ms"].items() if v["mean"] > 0)
    print(f"[{args.tag} {variant:7s} {traj:8s} #{pass_i}] {rep_['fps']['sequence']:6.1f} fps | {m['total']['mean']:7.2f} ms "
          f"(p95 {m['total']['p95']:.2f} max {m['total']['max']:.2f}) = render {m['render']['mean']:.2f} + get "
          f"{m['get_data']['mean']:.2f} + tables {m['tables']['mean']:.2f} + kern {m['kernels']['mean']:.2f} + cpu "
          f"{m['cpu']['mean']:.2f} + out {m['output']['mean']:.2f} | cpu: {cpu_txt} | lag {dict(lag_hist)} "
          f"anomalies {len(events)}", flush=True)
    return {"report": rep_, "thermal": therm, "floats": floats, "frame_index_of": frame_index_of}


# -- off the clock: synchronous renders ---------------------------------------------------------


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
    ids = data_of(payload["instance"]).numpy().reshape(H, W).astype(np.uint32)
    labels = labels_from_payload(payload["instance"])
    _l1, temp, mat, sky = lite.reduced_numpy(ids, labels, camera, t_s)
    dist = data_of(payload["distance"]).numpy().reshape(H, W)
    dist_g = np.where(sky, 0.0, np.nan_to_num(dist, posinf=0.0)).astype(np.float32)
    return {"temperature_k": temp, "material_id": mat, "sky_mask": sky, "distance_m": dist_g,
            # the enclosure is isothermal: L_env is the same at any V_s (lite's own premise)
            "sky_view_factor": np.full((H, W), 0.5, np.float32)}, ids, labels


def seam_check(traj: str, start: int, n: int = 4) -> dict:
    """`run_frame` (all host) against variant 123456 fed from the device, over n consecutive poses."""
    POS, LOOK = ns.trajectory(traj, N)
    st_ref, st_mine = PipelineState(t_s=T0 + CLOCK[0]), PipelineState(t_s=T0 + CLOCK[0])
    rows = []
    for j in range(start, start + n):
        payload = render_sync(POS[j % N], LOOK[j % N])
        terms, t_s = tables(payload)
        kernels(payload, terms, with_display=False)
        planes, _ids, _labels = gbuffer_planes(payload, t_s)
        st_ref.t_s = st_mine.t_s = t_s
        ref = run_frame(planes, cfg, st_ref)
        keep: dict = {}
        mine = cpu_chain("123456", cfg, st_mine, dict.fromkeys(CPU_BLOCKS, 0.0), keep)
        d_flux = np.abs(keep["flux"].astype(np.float64) - ref.flux.astype(np.float64))
        d_sig = np.abs(keep["signal"].astype(np.float64) - ref.signal_dn.astype(np.float64))
        d_dn = np.abs(keep["dn16"].astype(np.int64) - ref.dn16.astype(np.int64))
        d_disp = np.abs(mine.astype(np.int16) - ref.display8.astype(np.int16))
        rows.append({
            "pose": j % N, "frame_index": st_ref.frame_index - 1,
            "flux_max_rel": float((d_flux / ref.flux.astype(np.float64)).max()),
            "signal_max_abs_dn": float(d_sig.max()),
            "sigma_tvh_dn": float(cfg.detector.sigma_signal_dn),
            "dn16_px_differing": int((d_dn > 0).sum()), "dn16_max_diff": int(d_dn.max()),
            "display8_px_differing": int((d_disp > 0).any(axis=-1).sum()), "display8_max_code_diff": int(d_disp.max()),
            "n_px": int(H * W),
        })
    return {"trajectory": traj, "rows": rows}


def slow_stills(traj: str, variants, lead: int = 12) -> dict:
    """The still poses reached with the camera `--slow-factor` times slower (off the clock).

    Pose j of the N-frame path is pose j*f of the N*f-frame path; the `lead` frames before it
    are rendered synchronously and run through each variant so the membrane state is the slow
    camera's. Frame indices are chosen so the still carries the same noise draw as the fast one.
    """
    f = args.slow_factor
    POS, LOOK = ns.trajectory(traj, N * f)
    out: dict = {}
    for j in STILLS.get(traj, ()):
        seq = [(j * f - lead + s) % (N * f) for s in range(lead + 1)]
        rads = []
        for jj in seq:
            payload = render_sync(POS[jj], LOOK[jj])
            terms, t_s = tables(payload)
            kernels(payload, terms, with_display=False)
            rads.append((out2.numpy().copy(), t_s))
        for variant in variants:
            if variant in ("12", "123"):
                continue
            config = config_for(variant)
            st = PipelineState(t_s=rads[0][1])
            st.frame_index = args.warm + j + 1 - lead  # pose j is read one iteration after it is written
            keep: dict = {}
            for s, (rad, t_s) in enumerate(rads):
                out2.assign(rad)
                st.t_s = t_s
                keep = {}
                rgba = cpu_chain(variant, config, st, dict.fromkeys(CPU_BLOCKS, 0.0), keep)
            out[(variant, j)] = {"display8": rgba.copy(), **{k: np.array(v) for k, v in keep.items() if isinstance(v, np.ndarray)},
                                 "frame_index": st.frame_index - 1}
    return out


def ids_for_stills(traj: str) -> dict:
    POS, LOOK = ns.trajectory(traj, N)
    out = {}
    for j in STILLS.get(traj, ()):
        payload = render_sync(POS[j], LOOK[j])
        terms, t_s = tables(payload)
        kernels(payload, terms, with_display=False)
        _planes, ids, labels = gbuffer_planes(payload, t_s)
        from irsim_isaac.pipeline.material_ids import labels_to_paths
        out[j] = {"ids": ids, "labels": labels_to_paths(labels), "radiance_sync": out2.numpy().copy()}
    return out


def radiometric_branch_ms(n: int = 20) -> dict:
    """Stage 6's other half, which run_frame also computes when the outputs ask for it."""
    st = PipelineState(t_s=T0 + CLOCK[0])
    keep: dict = {}
    cpu_chain("123456", cfg, st, dict.fromkeys(CPU_BLOCKS, 0.0), keep)
    lb_cal = float(lut.lookup(cfg.t_housing_cal_k, q)[()])
    ts = []
    for _ in range(n):
        t = pc()
        scene_l = _scene_radiance_from_signal(keep["signal"], cfg, lb_cal)
        apparent_temperature(scene_l, lut, q)
        ts.append(pc() - t)
    return stats_ms(ts)


# -- main ---------------------------------------------------------------------------------------
import pathlib as _pl  # noqa: E402

report: dict = {
    "tag": args.tag, "gui": args.gui, "rgb": USE_RGB, "resolution": [W, H], "n_frames": N, "supersample": K,
    "frame_period_s": PERIOD, "variants": args.variants,
    "epp": _pl.Path("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference").read_text().strip(),
    "rendermode": settings.get("/rtx/rendermode"),
    "sensor": {"name": spec.name, "isp": spec.isp.model_dump(mode="json"),
               "psf_shape": None if cfg.psf is None else list(cfg.psf.shape),
               "sigma_tvh_dn": float(cfg.detector.sigma_signal_dn),
               "dn_per_k_300": float(dn_per_kelvin(cfg.calibration, lut, 300.0)),
               "tau_th_ms": float(spec.fpa.thermal_time_constant_ms), "frame_rate_hz": float(spec.fpa.frame_rate_hz),
               "noise_ratios": list(spec.noise.sigma_ratios()),
               "flat_field_in_demo_config": cfg.flat_field is not None, "chain_in_demo_config": cfg.chain is not None},
    "passes": [], "seam": [], "numpy": np.__version__,
}
try:
    from omni.kit.viewport.utility import get_active_viewport
    vp = get_active_viewport()
    if vp is not None and vp.updates_enabled:
        vp.updates_enabled = False
        for _ in range(4):
            rep.orchestrator.step(rt_subframes=1)
    report["viewport_updates_enabled"] = None if vp is None else bool(vp.updates_enabled)
except Exception as exc:  # noqa: BLE001
    report["viewport_updates_enabled"] = f"<{type(exc).__name__}: {exc}>"
print(f"[{args.tag}] gui {args.gui} viewport {report['viewport_updates_enabled']} rendermode {report['rendermode']} "
      f"epp {report['epp']} k {K} psf {report['sensor']['psf_shape']} sigma_tvh {report['sensor']['sigma_tvh_dn']:.3f} DN "
      f"dn/K {report['sensor']['dn_per_k_300']:.2f}", flush=True)

TRAJS = ("linear", "circular") if args.trajectory == "both" else (args.trajectory,)


def dump():
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))


# The seam first: if the chained host stages do not reproduce run_frame, nothing below means much.
if not args.no_check:
    for traj in TRAJS:
        j0 = (STILLS.get(traj) or [0])[0] - 3
        s = seam_check(traj, j0)
        report["seam"].append(s)
        print(f"[{args.tag} seam {traj}] {json.dumps(s['rows'])}", flush=True)
    dump()

runs: dict = {}
for pass_i in range(1, args.passes + 1):
    for traj in TRAJS:
        for variant in args.variants:
            r = run_pass(variant, traj, pass_i)
            report["passes"].append(r["report"])
            if pass_i == args.passes:
                runs[(variant, traj)] = r
            dump()

report["radiometric_branch_ms"] = radiometric_branch_ms()
print(f"[{args.tag} radiometric branch] {json.dumps(report['radiometric_branch_ms'])}", flush=True)

# -- disk: stills, their float planes, the whole display sequences ------------------------------
for traj in TRAJS:
    d = OUT / traj
    d.mkdir(parents=True, exist_ok=True)
    meta = ids_for_stills(traj)
    slow = slow_stills(traj, args.variants) if args.slow_factor > 1 else {}
    for j, mm in meta.items():
        np.savez_compressed(d / f"meta_{j:03d}.npz", ids=mm["ids"], radiance_sync=mm["radiance_sync"],
                            labels=json.dumps({str(k): v for k, v in mm["labels"].items()}))
    for variant in args.variants:
        r = runs[(variant, traj)]
        np.save(d / f"display_seq_{variant}.npy", r["thermal"][..., :3])
        np.save(d / f"frame_index_{variant}.npy", r["frame_index_of"])
        for j in STILLS.get(traj, ()):
            fl = r["floats"].get(j)
            if fl is None:
                print(f"[{args.tag}] WARNING no float planes for {variant} {traj}:{j}", flush=True)
                continue
            arrs = {k: v for k, v in fl.items() if isinstance(v, np.ndarray)}
            np.savez_compressed(d / f"still_{variant}_{j:03d}.npz", display8=r["thermal"][j],
                                frame_index=np.int64(fl.get("frame_index", -1)), **arrs)
    for (variant, j), fl in slow.items():
        np.savez_compressed(d / f"slow_{variant}_{j:03d}.npz", **{k: v for k, v in fl.items()})
dump()
print(f"wrote {OUT / 'report.json'}", flush=True)
ctx["app"].close()
