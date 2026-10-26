#!/usr/bin/env python
"""What one frame costs in Isaac, split renderer vs host, for the inspection bench.

Phases, each closed after a `wp.synchronize()` (and after the annotators' device->host copy,
which is the real sync for the AOVs):

  (a) renderer      reader.step() + reader.read()      -- the AOVs
  (b) assembly      IrCamera.planes() minus (a)        -- the G-buffer build
  (c) inside (b)    material_id_plane, temperature_plane
      and outside   environment_radiance -- which does NOT run in planes(); it is in the radiance
                    stage, so it is timed inside get_outputs() and reported separately.

Writes JSON to --out. One process per (supersample, rendermode): the render product resolution
changes with the supersample factor, so it cannot be switched in place.
"""
from __future__ import annotations

import argparse
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

# ---- instrumentation: wrap, do not edit the repo -------------------------------------------
timings: dict[str, list[float]] = {kk: [] for kk in
                                   ("renderer", "planes_total", "material_id", "temperature",
                                    "environment_radiance", "sky_clear_lut", "get_outputs_total")}
state = {"renderer": 0.0, "material_id": 0.0, "temperature": 0.0, "env": 0.0, "clear_lut": 0.0}
counts = {"env_calls": 0, "clear_lut_calls": 0}

reader = camera._reader
_step, _read = reader.step, reader.read


def timed_step(**kw):
    t0 = time.perf_counter()
    _step(**kw)
    sync()
    state["renderer"] += time.perf_counter() - t0


def timed_read():
    t0 = time.perf_counter()
    out = _read()          # get_data() + _to_numpy(): the device->host copy
    sync()
    state["renderer"] += time.perf_counter() - t0
    return out


reader.step, reader.read = timed_step, timed_read

import irsim_isaac.pipeline.ir_camera as ir_camera_mod  # noqa: E402

_mid = ir_camera_mod.material_id_plane


def timed_material_id(*a, **kw):
    t0 = time.perf_counter()
    out = _mid(*a, **kw)
    state["material_id"] += time.perf_counter() - t0
    return out


ir_camera_mod.material_id_plane = timed_material_id

_tplane = camera.bridge.temperature_plane


def timed_temperature(*a, **kw):
    t0 = time.perf_counter()
    out = _tplane(*a, **kw)
    state["temperature"] += time.perf_counter() - t0
    return out


camera.bridge.temperature_plane = timed_temperature

# environment_radiance runs in the RADIANCE STAGE, not in planes(). Patch every namespace that
# imported it so whichever path the frame takes is counted.
import irsim.pipeline.environment as env_mod  # noqa: E402

_env = env_mod.environment_radiance


def timed_env(*a, **kw):
    t0 = time.perf_counter()
    out = _env(*a, **kw)
    state["env"] += time.perf_counter() - t0
    counts["env_calls"] += 1
    return out


patched = []
for modname in ("irsim.pipeline.environment", "irsim.pipeline.radiance",
                "irsim_isaac.pipeline.warp_stages", "irsim.validation.aerial"):
    mod = sys.modules.get(modname)
    if mod is not None and hasattr(mod, "environment_radiance"):
        mod.environment_radiance = timed_env
        patched.append(modname)

from irsim.atmosphere.sky import SkyModel  # noqa: E402

_clear_lut = SkyModel._clear_lut


def timed_clear_lut(self, t_s):
    t0 = time.perf_counter()
    out = _clear_lut(self, t_s)
    state["clear_lut"] += time.perf_counter() - t0
    counts["clear_lut_calls"] += 1
    return out


SkyModel._clear_lut = timed_clear_lut

_planes = camera.planes


def timed_planes(**kw):
    t0 = time.perf_counter()
    out = _planes(**kw)
    timings["planes_total"].append(time.perf_counter() - t0)
    return out


camera.planes = timed_planes


def vram_mib() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return float("nan")
    mine = {str(os.getpid())}
    try:  # Kit may render from a child
        tree = subprocess.run(["pgrep", "-P", str(os.getpid())], capture_output=True,
                              text=True, timeout=10).stdout.split()
        mine |= set(tree)
    except Exception:
        pass
    total = 0.0
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] in mine:
            total += float(parts[1])
    if total:
        return total
    vals = [float(p.split(",")[1].strip()) for p in out.strip().splitlines() if "," in p]
    return max(vals) if vals else float("nan")


# The render product's creation resets AA: aa/op set to 0 before open() reads back 3 after it.
# ADR 0014 / geometry_probe.RENDERER_SETTINGS: with AA on, DLSS renders internally at a reduced
# resolution and upscales, so both the cost and the pixel addressing stop being the product's.
for _attempt in range(3):
    settings.set("/rtx/post/aa/op", 0)
    settings.set("/rtx/rendermode", args.rendermode)
    if args.rendermode == "PathTracing":
        settings.set("/rtx/pathtracing/spp", args.pt_spp)
        settings.set("/rtx/pathtracing/totalSpp", args.pt_spp)
    _step(frames=1, rt_subframes=1)  # let the setting take effect on a real frame
    if settings.get("/rtx/post/aa/op") == 0:
        break
aa_final = settings.get("/rtx/post/aa/op")
if aa_final != 0:
    print(f"WARNING: /rtx/post/aa/op is {aa_final}, not 0 -- AA/DLSS is active", file=sys.stderr)

RENDER_KEYS = ("/rtx/rendermode", "/rtx/post/aa/op", "/rtx/post/dlss/execMode",
               "/rtx/pathtracing/spp", "/rtx/pathtracing/totalSpp",
               "/rtx/sceneDb/ambientLightIntensity")
settings_readback = {kk: settings.get(kk) for kk in RENDER_KEYS}
effective_mode = settings.get("/rtx/rendermode")
res = camera.resolutions
print(f"\nrequested rendermode {args.rendermode!r} -> effective {effective_mode!r}")
print(f"supersample x{k}, native {spec.fpa.width}x{spec.fpa.height}, "
      f"render product {spec.fpa.width * k}x{spec.fpa.height * k}")

total = args.warmup + args.frames
vram_samples: list[float] = []
for i in range(total):
    for key in state:
        state[key] = 0.0
    t0 = time.perf_counter()
    camera.get_outputs(rt_subframes=args.rt_subframes)
    wall = time.perf_counter() - t0
    if i < args.warmup:
        continue
    timings["get_outputs_total"].append(wall)
    timings["renderer"].append(state["renderer"])
    timings["material_id"].append(state["material_id"])
    timings["temperature"].append(state["temperature"])
    timings["environment_radiance"].append(state["env"])
    timings["sky_clear_lut"].append(state["clear_lut"])
    if (i - args.warmup) % 5 == 0:
        vram_samples.append(vram_mib())

# planes_total collected every frame including warmup; keep the measured tail only
timings["planes_total"] = timings["planes_total"][args.warmup:]
assembly = [p - r for p, r in zip(timings["planes_total"], timings["renderer"])]
timings["assembly_b"] = assembly


def stats(xs: list[float]) -> dict[str, float]:
    return {
        "mean_ms": 1e3 * statistics.fmean(xs),
        "std_ms": 1e3 * (statistics.stdev(xs) if len(xs) > 1 else 0.0),
        "min_ms": 1e3 * min(xs),
        "max_ms": 1e3 * max(xs),
        "n": len(xs),
    }


result = {
    "sensor": str(args.sensor),
    "scene": str(args.scene),
    "requested_rendermode": args.rendermode,
    "effective_rendermode": effective_mode,
    "supersample_factor": k,
    "native_px": [spec.fpa.width, spec.fpa.height],
    "render_product_px": [spec.fpa.width * k, spec.fpa.height * k],
    "rt_subframes": args.rt_subframes,
    "pt_spp": args.pt_spp if args.rendermode == "PathTracing" else None,
    "warmup": args.warmup,
    "frames": args.frames,
    "env_radiance_patched_in": patched,
    "renderer_settings_readback": settings_readback,
    "env_radiance_calls_per_frame": counts["env_calls"] / max(1, total),
    "sky_clear_lut_calls_per_frame": counts["clear_lut_calls"] / max(1, total),
    "vram_mib_max": max(vram_samples) if vram_samples else None,
    "phases": {name: stats(xs) for name, xs in timings.items() if xs},
}
pathlib.Path(args.out).write_text(json.dumps(result, indent=2))
print(json.dumps(result["phases"], indent=2))
print(f"VRAM max {result['vram_mib_max']} MiB -> {args.out}")
app.close()
