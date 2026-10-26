#!/usr/bin/env python
"""Where one frame's host time goes, function by function, for the inspection bench.

Does not edit the repo: everything is wrapped from outside. Four passes in ONE process, so the
passes share the machine state and the thermal/GPU drift between them is visible:

  A0  plain       get_outputs() wall only, nothing wrapped          -- the number to explain
  A1  sync        + reader.step/read wrapped with wp.synchronize()  -- renderer vs host split
  P   cProfile    every Python and C call, dumped to <out>.prof     -- the map (file:line)
  S   spans       wall timers on the functions named in --spans     -- the map, without the
                                                                     profiler's per-call tax
  A2  plain again (wrappers left in place)                          -- drift inside the process

Setup is bench_render_cost.py's, verbatim, so the frame is the same frame.
"""
from __future__ import annotations

import argparse
import cProfile
import importlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

REPO = pathlib.Path("/home/jumasaet/isaac-thermal-camera-simulation")
BENCH = pathlib.Path("/tmp/irsim_render_cost")

parser = argparse.ArgumentParser()
parser.add_argument("--sensor", required=True)
parser.add_argument("--scene", default=str(BENCH / "scene_bench.yaml"))
parser.add_argument("--rendermode", required=True)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--frames", type=int, default=30)
parser.add_argument("--rt-subframes", type=int, default=1)
parser.add_argument("--pt-spp", type=int, default=32)
parser.add_argument("--settle", type=int, default=16)
parser.add_argument("--spans", default=None, help="text file, one module:qualname per line")
parser.add_argument("--lines", default=None, help="text file, one module:qualname per line")
parser.add_argument("--out", required=True)
args = parser.parse_args()

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from irsim_isaac.env import simulation_app_config  # noqa: E402

from isaacsim import SimulationApp  # noqa: E402

# Kit parses its own settings off argv, and /rtx/post/aa/op cannot be forced from carb at runtime
# on this build: set to 0 before open() it reads back 3 after. Pass it as a launch argument so AA
# is off before the render product exists (ADR 0014's half-resolution trap).
for _arg in ("--/rtx/post/aa/op=0", "--/rtx/post/dlss/execMode=0"):
    if _arg not in sys.argv:
        sys.argv.append(_arg)

app = SimulationApp(simulation_app_config())

import carb  # noqa: E402
import numpy as np  # noqa: E402

settings = carb.settings.get_settings()
settings.set("/rtx/rendermode", args.rendermode)
if args.rendermode == "PathTracing":
    settings.set("/rtx/pathtracing/spp", args.pt_spp)
    settings.set("/rtx/pathtracing/totalSpp", args.pt_spp)
settings.set("/rtx/sceneDb/ambientLightIntensity", 0.0)
settings.set("/rtx/post/aa/op", 0)  # DLSS/AA renders internally at reduced res (ADR 0014)

from irsim.config.loader import load_sensor_config  # noqa: E402
from irsim.materials.library import MaterialLibrary  # noqa: E402
from irsim.materials.mapping import MaterialResolver, load_mapping_rules  # noqa: E402
from irsim.materials.table import MaterialTable  # noqa: E402
from irsim.pipeline.core import PipelineConfig  # noqa: E402
from irsim.radiometry.lut_files import (  # noqa: E402
    load_band_lut_for_config,
    load_band_response_for_config,
)
from irsim.scene import Scene  # noqa: E402
from irsim_isaac.pipeline.illumination_isaac import SceneIllumination  # noqa: E402
from irsim_isaac.pipeline.ir_camera import IrCamera  # noqa: E402
from irsim_isaac.pipeline.materials_usd import prim_records  # noqa: E402
from irsim_isaac.pipeline.point_bridge import bindings_from_scene  # noqa: E402

from inspection_bench_stage import author_camera_ops, new_bench_stage  # noqa: E402
from render_inspection_bench_offline import CameraPlacement, stage_scene_config  # noqa: E402
from irsim.atmosphere.skylight import skylight_for_sensor  # noqa: E402

try:
    import warp as wp

    wp.init()

    def sync() -> None:
        wp.synchronize()
except Exception as exc:  # pragma: no cover
    print(f"warp unavailable ({exc}); relying on the device->host copy as the sync", file=sys.stderr)

    def sync() -> None:
        return None


sensor = load_sensor_config(args.sensor)
spec = sensor.sensor
band = spec.band.band_id
quantity = spec.quantity
k = spec.optics.supersample_factor
lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
response = load_band_response_for_config(sensor, REPO / "data")
skylight = skylight_for_sensor(sensor, quantity)

config = stage_scene_config(args.scene)
scene = Scene.from_config(
    config, {band: lut}, quantity=quantity, skylights={band: skylight}, responses={band: response}
)
materials = {s.name: s.material for s in config.scene.thermal.surfaces}
stage, prim_to_target = new_bench_stage(scene, materials)

placement = CameraPlacement(position_enu=(0.0, 0.5, 2.0), tilt_deg=-20.0)
camera_path = "/World/IrCamera"
author_camera_ops(stage, camera_path, placement)

table = MaterialTable.from_library(MaterialLibrary.load(), band)
resolver = MaterialResolver(load_mapping_rules(), list(table.names))
resolutions = resolver.resolve_all(prim_records(root="/World"))

pipeline = PipelineConfig.from_sensor(
    sensor, table, lut, sky=scene.sky_models[band], atmosphere=scene.layered,
    flat_field_enabled=False,
)
camera = IrCamera(
    sensor, scene, pipeline=pipeline, prim_to_target=prim_to_target, resolutions=resolutions,
    camera_path=camera_path, strict_materials=False, strict_thermal_nodes=False,
    strict_patch_coverage=False, surface_fields=bindings_from_scene(scene),
    illumination=SceneIllumination.for_camera(sensor, scene, pipeline.quantity, heading_deg=0.0),
    heading_deg=0.0,
).open(settle_frames=args.settle)


# ---- profiling: wrap, do not edit the repo ---------------------------------------------------
N = args.frames


def plain_pass(label: str) -> list[float]:
    walls = []
    for _ in range(N):
        t0 = time.perf_counter()
        camera.get_outputs(rt_subframes=args.rt_subframes)
        walls.append(time.perf_counter() - t0)
    print(f"[pass {label}] get_outputs {1e3 * statistics.fmean(walls):.1f} ms "
          f"+-{1e3 * statistics.stdev(walls):.1f}", flush=True)
    return walls


# Same renderer state as bench_render_cost.py: the render product's creation resets AA.
for _attempt in range(3):
    settings.set("/rtx/post/aa/op", 0)
    settings.set("/rtx/rendermode", args.rendermode)
    camera._reader.step(frames=1, rt_subframes=1)
    if settings.get("/rtx/post/aa/op") == 0:
        break

for _ in range(args.warmup):
    camera.get_outputs(rt_subframes=args.rt_subframes)

res: dict[str, object] = {"sensor": args.sensor, "scene": args.scene, "frames": N,
                          "warmup": args.warmup,
                          "effective_rendermode": settings.get("/rtx/rendermode"),
                          "aa_op": settings.get("/rtx/post/aa/op")}
res["A0_plain"] = plain_pass("A0 plain")

# -- A1: renderer/host split, the previous bench's convention (sync closes each renderer call)
reader = camera._reader
_step, _read = reader.step, reader.read
rstate = {"step": 0.0, "read": 0.0}


def timed_step(**kw):
    t0 = time.perf_counter()
    _step(**kw)
    sync()
    rstate["step"] += time.perf_counter() - t0


def timed_read():
    t0 = time.perf_counter()
    out = _read()
    sync()
    rstate["read"] += time.perf_counter() - t0
    return out


reader.step, reader.read = timed_step, timed_read
a1, a1_step, a1_read = [], [], []
for _ in range(N):
    rstate["step"] = rstate["read"] = 0.0
    t0 = time.perf_counter()
    camera.get_outputs(rt_subframes=args.rt_subframes)
    a1.append(time.perf_counter() - t0)
    a1_step.append(rstate["step"])
    a1_read.append(rstate["read"])
res["A1_sync"] = a1
res["A1_reader_step"] = a1_step
res["A1_reader_read"] = a1_read
print(f"[pass A1 sync] get_outputs {1e3 * statistics.fmean(a1):.1f} ms, reader.step "
      f"{1e3 * statistics.fmean(a1_step):.1f}, reader.read {1e3 * statistics.fmean(a1_read):.1f}",
      flush=True)

# -- P: cProfile. Enabled only around get_outputs, so the map is of the frame and nothing else.
# The renderer step runs OUTSIDE the profiler: Kit's update drops cProfile's open call stack, so
# a profile that spans it loses get_outputs() and planes() themselves (their callees survive as
# orphans). get_outputs(step=True) is `reader.step()` then exactly get_outputs(step=False), so
# stepping by hand first profiles the same frame minus the one call that is a black box anyway.
prof = cProfile.Profile()
pw, pstep = [], []
for _ in range(N):
    t0 = time.perf_counter()
    reader.step(frames=1, rt_subframes=args.rt_subframes)   # the timed, synced wrapper
    # ... and so does reader.read(): the annotators' get_data() drops the stack the same way.
    # Read by hand, then hand planes() the same RawAovs object through a one-shot stub.
    _aovs = timed_read()
    reader.read = lambda _a=_aovs: _a
    t1 = time.perf_counter()
    prof.enable()
    camera.get_outputs(step=False)
    prof.disable()
    reader.read = timed_read
    pw.append(time.perf_counter() - t0)
    pstep.append(t1 - t0)
prof.dump_stats(args.out + ".prof")
res["P_cprofile_wall"] = pw
res["P_step_and_read_outside_profile"] = pstep
print(f"[pass P cProfile] get_outputs {1e3 * statistics.fmean(pw):.1f} ms under the profiler",
      flush=True)

# -- L: line_profiler on the functions named in --lines, same framing as P (step+read outside).
def resolve(spec: str):
    modname, _, qual = spec.partition(":")
    obj = importlib.import_module(modname)
    for part in qual.split("."):
        raw = obj.__dict__[part] if isinstance(obj, type) else getattr(obj, part)
        obj = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
    return obj


if args.lines:
    sys.path.append(str(BENCH / "pydeps"))   # appended: must not shadow anything Isaac ships
    from line_profiler import LineProfiler
    lp = LineProfiler()
    bad = []
    for spec in pathlib.Path(args.lines).read_text().splitlines():
        spec = spec.strip()
        if not spec or spec.startswith("#"):
            continue
        try:
            lp.add_function(resolve(spec))
        except Exception as exc:
            bad.append(f"{spec}: {type(exc).__name__}: {exc}")
    if bad:
        print("UNRESOLVED LINE TARGETS:\n  " + "\n  ".join(bad), flush=True)
    lw = []
    for _ in range(N):
        t0 = time.perf_counter()
        reader.step(frames=1, rt_subframes=args.rt_subframes)
        _aovs = timed_read()
        reader.read = lambda _a=_aovs: _a
        lp.enable()
        camera.get_outputs(step=False)
        lp.disable()
        reader.read = timed_read
        lw.append(time.perf_counter() - t0)
    res["L_line_wall"] = lw
    print(f"[pass L lines] get_outputs {1e3 * statistics.fmean(lw):.1f} ms under line_profiler",
          flush=True)
    import io
    buf = io.StringIO()
    lp.print_stats(stream=buf, output_unit=1e-3, stripzeros=True)
    pathlib.Path(args.out + ".lines.txt").write_text(buf.getvalue())
    lp.dump_stats(args.out + ".lprof")

# -- S: span timers. Inclusive wall per named function, and exclusive (minus named children).
span_names = []
if args.spans:
    span_names = [l.strip() for l in pathlib.Path(args.spans).read_text().splitlines()
                  if l.strip() and not l.lstrip().startswith("#")]
incl: dict[str, float] = {}
excl: dict[str, float] = {}
ncall: dict[str, int] = {}
stack: list[list] = []  # [name, child_time]
unresolved = []


def make_span(name, fn):
    def wrapper(*a, **kw):
        frame = [name, 0.0]
        stack.append(frame)
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            dt = time.perf_counter() - t0
            stack.pop()
            incl[name] = incl.get(name, 0.0) + dt
            excl[name] = excl.get(name, 0.0) + dt - frame[1]
            ncall[name] = ncall.get(name, 0) + 1
            if stack:
                stack[-1][1] += dt
    wrapper.__wrapped__ = fn
    return wrapper


def install(spec: str) -> None:
    """module:attr[.attr]  -- patches the attribute where it is DEFINED and every already-imported
    module that holds the same object under the same name (the `from x import f` copies).
    A trailing '@obj' form ('@camera.bridge.temperature_plane') patches a live instance."""
    if spec.startswith("@"):
        parts = spec[1:].split(".")
        obj = {"camera": camera, "scene": scene, "pipeline": pipeline}[parts[0]]
        for p in parts[1:-1]:
            obj = getattr(obj, p)
        fn = getattr(obj, parts[-1])
        setattr(obj, parts[-1], make_span(spec, fn))
        return
    modname, _, qual = spec.partition(":")
    mod = importlib.import_module(modname)
    parts = qual.split(".")
    owner = mod
    for p in parts[:-1]:
        owner = getattr(owner, p)
    raw = owner.__dict__[parts[-1]] if isinstance(owner, type) else getattr(owner, parts[-1])
    is_static = isinstance(raw, staticmethod)
    fn = raw.__func__ if is_static else raw
    wrapped = make_span(spec, fn)
    setattr(owner, parts[-1], staticmethod(wrapped) if is_static else wrapped)
    if len(parts) == 1:  # a module-level function: follow the from-imports
        for other in list(sys.modules.values()):
            if other is None or other is mod:
                continue
            d = getattr(other, "__dict__", None)
            if d is not None and d.get(parts[-1]) is fn:
                d[parts[-1]] = wrapped


for spec in span_names:
    try:
        install(spec)
    except Exception as exc:  # report, never silently drop a span
        unresolved.append(f"{spec}: {type(exc).__name__}: {exc}")
if unresolved:
    print("UNRESOLVED SPANS:\n  " + "\n  ".join(unresolved), flush=True)

sw = []
if span_names:
    incl.clear(); excl.clear(); ncall.clear()
    for _ in range(N):
        t0 = time.perf_counter()
        camera.get_outputs(rt_subframes=args.rt_subframes)
        sw.append(time.perf_counter() - t0)
    res["S_span_wall"] = sw
    res["S_spans"] = {k: {"incl_ms": 1e3 * incl[k] / N, "excl_ms": 1e3 * excl[k] / N,
                          "calls_per_frame": ncall[k] / N} for k in incl}
    res["S_unresolved"] = unresolved
    print(f"[pass S spans] get_outputs {1e3 * statistics.fmean(sw):.1f} ms", flush=True)
    for k, v in sorted(res["S_spans"].items(), key=lambda kv: -kv[1]["incl_ms"]):
        print(f"   {v['incl_ms']:9.2f} incl {v['excl_ms']:9.2f} excl "
              f"{v['calls_per_frame']:8.1f} calls  {k}", flush=True)

res["A2_plain_after"] = plain_pass("A2 plain (wrappers in place)")
res["cache_sizes"] = {
    "sky._cache": len(scene.sky_models[band]._cache),
    "sky._path_cache": len(scene.sky_models[band]._path_cache),
}
pathlib.Path(args.out + ".json").write_text(json.dumps(res, indent=1))
print(f"wrote {args.out}.json and {args.out}.prof", flush=True)
app.close()
