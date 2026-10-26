#!/usr/bin/env python
"""TASK 2 -- how long does one frame of the reduced path take, device-resident?

Renderer advanced as in point 1 of the earlier report: rep.orchestrator.step(wait_for_render=False)
(one app.update), viewport off when --disable-viewport is given. Annotators on device="cuda",
their buffers wrapped by Warp without a copy. Nothing in the repo is edited.

Per frame:  step | get_data | refresh (host tables, clock advancing) | kernels (+sync) | readback
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402

parser = common.base_parser()
parser.add_argument("--n", type=int, default=300)
parser.add_argument("--warm", type=int, default=30)
parser.add_argument("--rgb-mode", choices=("none", "pt", "rt"), default="none",
                    help="pt: IrCamera(capture_rgb=True) as the repo does it (PathTracing); "
                         "rt: rgb annotator under the bench's own rendermode")
parser.add_argument("--no-light", action="store_true")
parser.add_argument("--skip-lag", action="store_true")
args, kit_args = parser.parse_known_args()
args.capture_rgb = args.rgb_mode != "none"
args.light = (args.light or args.capture_rgb) and not args.no_light
ctx = common.boot(args, kit_args)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import warp as wp  # noqa: E402

import lite  # noqa: E402
from irsim.pipeline.atmosphere import apply_layered_gbuffer  # noqa: E402
from irsim_isaac.pipeline import warp_stages  # noqa: E402
from irsim_isaac.pipeline.gbuffer_isaac import AOV_INIT_PARAMS, AOV_NAMES  # noqa: E402
from irsim_isaac.pipeline.material_ids import labels_from_payload  # noqa: E402

pc = common.pc
sync = wp.synchronize
camera = ctx["camera"].open(settle_frames=args.settle)
cfg, lut, q, settings = camera.config, ctx["lut"], ctx["quantity"], ctx["settings"]
if args.rgb_mode != "pt":
    for _ in range(3):
        settings.set("/rtx/post/aa/op", 0)
        settings.set("/rtx/rendermode", args.rendermode)
        camera._reader.step(frames=1, rt_subframes=1)
RP = camera._reader.render_product_path
W, H = camera.optics.resolution
camera._reader.detach()
DEV = "cuda:0"

res: dict[str, object] = {
    "scene": args.scene, "disable_viewport": args.disable_viewport, "rgb_mode": args.rgb_mode,
    "resolution": [W, H], "n": args.n, "rendermode": settings.get("/rtx/rendermode"),
    "settings": {k: settings.get(k) for k in (
        "/rtx/pathtracing/spp", "/rtx/pathtracing/totalSpp", "/rtx/pathtracing/maxBounces",
        "/rtx/post/aa/op", "/app/asyncRendering", "/app/renderer/resolution/width",
        "/app/renderer/resolution/height")},
    "governor_epp": pathlib.Path(
        "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference").read_text().strip(),
    "variants": {}, "checks": {},
}
try:
    from omni.kit.viewport.utility import get_active_viewport
    vp = get_active_viewport()
    res["viewport_updates_enabled"] = None if vp is None else bool(vp.updates_enabled)
except Exception as exc:  # noqa: BLE001
    res["viewport_updates_enabled"] = f"<{type(exc).__name__}: {exc}>"


def stats(xs):
    ms = sorted(1e3 * x for x in xs)
    n = len(ms)
    return {"mean": statistics.fmean(ms), "p50": ms[n // 2], "p95": ms[int(0.95 * n)],
            "p99": ms[int(0.99 * n)], "max": ms[-1], "sd": statistics.stdev(ms)}


def attach(channels):
    out = {}
    for c in channels:
        a = rep.AnnotatorRegistry.get_annotator(AOV_NAMES[c][0], init_params=AOV_INIT_PARAMS.get(c),
                                                device="cuda")
        a.attach(RP)
        out[c] = a
    for _ in range(6):
        rep.orchestrator.step(rt_subframes=1)
    return out


def detach(annos):
    for a in annos.values():
        a.detach(RP)


def data_of(v):
    return v.get("data", v) if isinstance(v, dict) else v


def wrap_f32(h):
    if h.ndim == 2 and h.dtype is wp.float32:
        return h
    return wp.array(ptr=h.ptr, dtype=wp.float32, shape=tuple(h.shape[:2]), device=h.device)


gpu = lite.LiteGpu(camera, (H, W), DEV)
out2 = wp.zeros((H, W), dtype=wp.float32, device=DEV)
T0 = camera.scene.t0_s
PERIOD = camera.frame_period_s
CLOCK = [0.0]  # monotonic across variants: the thermal solvers refuse to rewind


def run_variant(name, channels, compute, n):
    """compute: 'none' | 's1' | 's1+s2'. Clock advances one frame period per frame."""
    annos = attach(channels)
    acc = {k: [] for k in ("step", "get_data", "refresh", "kernels", "readback", "rgb_readback", "total")}
    uploads0 = gpu.uploads
    t_rel = CLOCK[0]
    for i in range(n + args.warm):
        t0 = pc()
        rep.orchestrator.step(rt_subframes=1, wait_for_render=False)
        t1 = pc()
        payload = {c: a.get_data() for c, a in annos.items()}
        t2 = pc()
        t3 = t4 = t5 = t2
        if compute != "none":
            camera.bridge.advance_to(t_rel)
            t_s = T0 + t_rel
            gpu.refresh(labels_from_payload(payload["instance"]), t_s)
            ids_dev = lite.wrap_ids(data_of(payload["instance"]))
            terms = None
            if compute == "s1+s2":
                camera.state.t_s = t_s
                terms = warp_stages.atmosphere_terms(cfg, camera.state)
            t3 = pc()
            result = gpu.run(ids_dev)
            if terms is not None:
                result = warp_stages.launch_atmosphere(
                    gpu.out, wrap_f32(data_of(payload["distance"])), gpu.sky, terms, out2, DEV)
            sync()
            t4 = pc()
            host = result.numpy()
            t5 = pc()
        t6 = t5
        if "rgb" in payload:
            rgb_host = data_of(payload["rgb"]).numpy()
            t6 = pc()
        t_rel += PERIOD
        if i >= args.warm:
            for k, v in (("step", t1 - t0), ("get_data", t2 - t1), ("refresh", t3 - t2),
                         ("kernels", t4 - t3), ("readback", t5 - t4), ("rgb_readback", t6 - t5),
                         ("total", t6 - t0)):
                acc[k].append(v)
    CLOCK[0] = t_rel
    out = {k: stats(v) for k, v in acc.items()}
    out["table_uploads"] = gpu.uploads - uploads0
    out["frac_over_33ms"] = sum(1 for v in acc["total"] if v > 1 / 30) / len(acc["total"])
    out["channels"] = list(channels)
    # ---- one-off checks on the last frame of this variant
    if compute != "none":
        ids_h = data_of(payload["instance"])
        chk = {"ids_type": type(ids_h).__name__, "ids_dtype": str(ids_h.dtype),
               "ids_shape": list(ids_h.shape), "ids_device": str(ids_h.device),
               "zero_copy_same_ptr": int(ids_dev.ptr) == int(ids_h.ptr),
               "payload_has_labels": bool(labels_from_payload(payload["instance"]))}
        ids_np = ids_h.numpy().reshape(H, W).astype(np.uint32)
        l_ref, _t, _m, sky_ref = lite.reduced_numpy(ids_np, labels_from_payload(payload["instance"]),
                                                   camera, t_s)
        s1 = gpu.out.numpy()
        d = np.abs(s1.astype(np.float64) - l_ref.astype(np.float64)) / l_ref
        chk["s1_vs_cpu_reference_max_rel"] = float(d.max())
        chk["s1_bit_identical_px"] = int((lite.bits(s1) == lite.bits(l_ref)).sum())
        chk["n_px"] = int(s1.size)
        if compute == "s1+s2":
            dist_np = data_of(payload["distance"]).numpy().reshape(H, W)
            dist_g = np.where(sky_ref, 0.0, np.nan_to_num(dist_np, posinf=0.0)).astype(np.float32)
            l2_ref = apply_layered_gbuffer(cfg.atmosphere, ctx["band"], t_s, l_ref, dist_g, q,
                                           sky_mask=sky_ref)
            d2 = np.abs(host.astype(np.float64) - l2_ref.astype(np.float64)) / l2_ref
            ta = np.asarray(lut.apparent_temperature(host, q), np.float64)
            tb = np.asarray(lut.apparent_temperature(l2_ref.astype(np.float32), q), np.float64)
            chk["s2_vs_cpu_reference_max_rel"] = float(d2.max())
            chk["s2_vs_cpu_reference_max_mK"] = float(np.abs(ta - tb).max() * 1e3)
        if "rgb" in payload:
            r = rgb_host.reshape(H, W, -1)[..., :3].astype(np.float64)
            chk["rgb"] = {"dtype": str(rgb_host.dtype), "shape": list(rgb_host.shape),
                          "mean": float(r.mean()), "max": float(r.max()), "std": float(r.std()),
                          "nonzero_frac": float((r.sum(axis=2) > 0).mean())}
            np.save(args.out + f".{name}.rgb.npy", rgb_host)
        res["checks"][name] = chk
    detach(annos)
    print(f"[{name:22s}] total {out['total']['mean']:6.2f} (p50 {out['total']['p50']:6.2f} p95 "
          f"{out['total']['p95']:6.2f} max {out['total']['max']:6.2f}) = step {out['step']['mean']:6.2f}"
          f" + get {out['get_data']['mean']:5.2f} + refresh {out['refresh']['mean']:5.2f} + kern "
          f"{out['kernels']['mean']:5.2f} + rb {out['readback']['mean']:5.2f} + rgb_rb "
          f"{out['rgb_readback']['mean']:5.2f} | >33ms {100 * out['frac_over_33ms']:.1f}% "
          f"uploads {out['table_uploads']}", flush=True)
    return out


VARIANTS = [
    ("all5_step_only", ("distance", "position", "normal", "instance", "semantic"), "none"),
    ("ids:s1", ("instance",), "s1"),
    ("ids+dist:s1+s2", ("instance", "distance"), "s1+s2"),
]
if args.rgb_mode != "none":
    VARIANTS += [
        ("ids+rgb:s1", ("instance", "rgb"), "s1"),
        ("ids+dist+rgb:s1+s2", ("instance", "distance", "rgb"), "s1+s2"),
    ]

# compile every kernel before anything is timed
_a = attach(("instance", "distance"))
_p = {c: a.get_data() for c, a in _a.items()}
gpu.refresh(labels_from_payload(_p["instance"]), T0)
gpu.run(lite.wrap_ids(data_of(_p["instance"])))
camera.state.t_s = T0
warp_stages.launch_atmosphere(gpu.out, wrap_f32(data_of(_p["distance"])), gpu.sky,
                              warp_stages.atmosphere_terms(cfg, camera.state), out2, DEV)
sync()
detach(_a)

for rep_i in (1, 2):
    for name, channels, compute in VARIANTS:
        key = f"{name}#{rep_i}"
        try:
            res["variants"][key] = run_variant(key, channels, compute, args.n)
        except Exception:  # noqa: BLE001
            import traceback
            res["variants"][key] = {"error": traceback.format_exc()}
            print(f"[{key}] FAILED\n{traceback.format_exc()}", flush=True)
        pathlib.Path(args.out + ".json").write_text(json.dumps(res, indent=1, default=str))


# ---- LAG: how many frames late is the data, for the annotator set the lite path attaches?
import time  # noqa: E402
import zlib  # noqa: E402

import omni.kit.app  # noqa: E402

_kit = omni.kit.app.get_app()


def _adv_wait():
    rep.orchestrator.step(rt_subframes=1)


def _adv_nowait():
    rep.orchestrator.step(rt_subframes=1, wait_for_render=False)


def _adv_nowait_plus1():
    rep.orchestrator.step(rt_subframes=1, wait_for_render=False)
    _kit.update()


LAG_MODES = {"wait": (_adv_wait, None), "nowait": (_adv_nowait, None),
             "nowait+1update": (_adv_nowait_plus1, None), "nowait@30Hz": (_adv_nowait, 1.0 / 30.0)}


def lag_phase(channels, n=150):
    """Tilt the camera 0.1 deg per frame; an id-plane checksum names the pose each read shows.

    Each frame also runs the lite stage-1 compute, so the loop has the runner's own timing.
    """
    annos = attach(channels)
    place = ctx["placement_cls"]

    def pose(k):
        ctx["cam_ops"].set_placement(place(position_enu=(0.0, 0.5, 2.0), tilt_deg=-25.0 + 0.1 * k))

    def crc(payload):
        return zlib.crc32(data_of(payload["instance"]).numpy().tobytes())

    ref = []
    for k in range(n):
        pose(k)
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=1)
        ref.append(crc({c: a.get_data() for c, a in annos.items()}))
    index = {h: k for k, h in enumerate(ref)}
    out = {"ref_unique": len(set(ref)), "n": n}
    for mode, (advance, period) in LAG_MODES.items():
        pose(0)
        for _ in range(6):
            rep.orchestrator.step(rt_subframes=1)
        lags, times, stale, last, unknown = [], [], 0, None, 0
        for k in range(n):
            t0 = pc()
            pose(k)
            advance()
            payload = {c: a.get_data() for c, a in annos.items()}
            gpu.refresh(labels_from_payload(payload["instance"]), T0 + CLOCK[0])
            gpu.run(lite.wrap_ids(data_of(payload["instance"])))
            sync()
            times.append(pc() - t0)
            h = crc(payload)
            if h in index:
                lags.append(k - index[h])
            else:
                unknown += 1
            stale += int(h == last)
            last = h
            if period is not None:
                rest = period - (pc() - t0)
                if rest > 0:
                    time.sleep(rest)
        tail = lags[10:]
        hist = {str(v): tail.count(v) for v in sorted(set(tail))}
        out[mode] = {"lag_hist": hist, "stale": stale, "unmatched": unknown,
                     "frame_ms": stats(times[10:])["mean"], "frame_p95_ms": stats(times[10:])["p95"]}
        print(f"   lag {'+'.join(channels):14s} {mode:15s} hist {hist} stale {stale} unmatched "
              f"{unknown}  frame {out[mode]['frame_ms']:.2f} ms (p95 {out[mode]['frame_p95_ms']:.2f})",
              flush=True)
    pose(0)
    ctx["cam_ops"].set_placement(place(position_enu=(0.0, 0.5, 2.0), tilt_deg=-20.0))
    detach(annos)
    return out


for chans in ([] if args.skip_lag else [("instance",)] + ([("instance", "rgb")] if args.rgb_mode != "none" else [])):
    try:
        res["checks"]["lag:" + "+".join(chans)] = lag_phase(chans)
    except Exception:  # noqa: BLE001
        import traceback
        res["checks"]["lag:" + "+".join(chans)] = traceback.format_exc()
pathlib.Path(args.out + ".json").write_text(json.dumps(res, indent=1, default=str))
print("checks:", json.dumps(res["checks"], default=str), flush=True)
print(f"wrote {args.out}.json", flush=True)
ctx["app"].close()
