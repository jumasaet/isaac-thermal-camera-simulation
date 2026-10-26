#!/usr/bin/env python
"""Can the texture->host copy behind reader.step/reader.read be avoided on this build?

Does not edit the repo. One process holds BOTH annotator device modes, in ABAB order, on the same
render product, so the cpu/cuda comparison shares machine state. Setup is profile_frame.py's.

Phases (each wrapped: a failure is recorded and the run goes on; JSON is rewritten after each):
  G   camera.get_outputs() with the reader swapped cpu -> cuda        (the numbers in context)
  T   tight loop step/read: cpu, none, cuda, cpu, cuda, none          (the split per mode)
  D   orchestrator.step taken apart: every app.update() inside it timed
  L   ladder: step cost against the number of channels attached, per device
  R   raw app.update(): renderer off / on, no orchestrator
  K   known value + zero copy: a cube face at a known distance read through a wrapped pointer
  LAG how many updates late the data is, per way of advancing the renderer
  V   the default viewport switched off
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time
import traceback

REPO = pathlib.Path("/home/jumasaet/isaac-thermal-camera-simulation")
BENCH = pathlib.Path("/tmp/irsim_render_cost")

parser = argparse.ArgumentParser()
parser.add_argument("--sensor", default=str(BENCH / "sensor_ss1.yaml"))
parser.add_argument("--scene", required=True)
parser.add_argument("--rendermode", default="RaytracedLighting")
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--n-ctx", type=int, default=8)
parser.add_argument("--settle", type=int, default=16)
parser.add_argument("--skip", default="", help="comma list of phases to skip")
parser.add_argument("--disable-viewport", action="store_true")
parser.add_argument("--out", required=True)
args, kit_args = parser.parse_known_args()
SKIP = {s for s in args.skip.split(",") if s}

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from irsim_isaac.env import simulation_app_config  # noqa: E402

from isaacsim import SimulationApp  # noqa: E402

sys.argv = [sys.argv[0], *kit_args]
for _arg in ("--/rtx/post/aa/op=0", "--/rtx/post/dlss/execMode=0"):
    if _arg not in sys.argv:
        sys.argv.append(_arg)

app = SimulationApp(simulation_app_config(
    **({"disable_viewport_updates": True} if args.disable_viewport else {})))

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import warp as wp  # noqa: E402

wp.init()
sync = wp.synchronize
pc = time.perf_counter

settings = carb.settings.get_settings()
settings.set("/rtx/rendermode", args.rendermode)
settings.set("/rtx/sceneDb/ambientLightIntensity", 0.0)
settings.set("/rtx/post/aa/op", 0)

from irsim.atmosphere.skylight import skylight_for_sensor  # noqa: E402
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
from irsim_isaac.pipeline.gbuffer_isaac import AovReader  # noqa: E402
from irsim_isaac.pipeline.illumination_isaac import SceneIllumination  # noqa: E402
from irsim_isaac.pipeline.ir_camera import IrCamera  # noqa: E402
from irsim_isaac.pipeline.materials_usd import prim_records  # noqa: E402
from irsim_isaac.pipeline.point_bridge import bindings_from_scene  # noqa: E402

from inspection_bench_stage import author_camera_ops, new_bench_stage  # noqa: E402
from render_inspection_bench_offline import CameraPlacement, stage_scene_config  # noqa: E402

sensor = load_sensor_config(args.sensor)
spec = sensor.sensor
band = spec.band.band_id
quantity = spec.quantity
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

for _attempt in range(3):
    settings.set("/rtx/post/aa/op", 0)
    settings.set("/rtx/rendermode", args.rendermode)
    camera._reader.step(frames=1, rt_subframes=1)
    if settings.get("/rtx/post/aa/op") == 0:
        break

W, H = camera.optics.resolution
RP = camera._reader.render_product_path
BASE_NAMES = dict(camera._reader.names)
REQ = ("distance", "position", "normal", "instance")
N = args.n

res: dict[str, object] = {
    "scene": args.scene, "n": N, "disable_viewport": args.disable_viewport, "resolution": [W, H], "kit_args": kit_args,
    "effective_rendermode": settings.get("/rtx/rendermode"), "aa_op": settings.get("/rtx/post/aa/op"),
    "camera_reader_device": camera._reader.device,
    "resolved": dict(camera._reader.resolved), "attach_failures": dict(camera._reader.failures),
    "warp": wp.__version__, "phases": {}, "errors": {}, "marks": [],
}
res["settings"] = {k: settings.get(k) for k in (
    "/app/asyncRendering", "/app/asyncRenderingLowLatency", "/omni/replicator/asyncRendering",
    "/rtx/hydra/supportMultiTickRate", "/app/runLoops/main/rateLimitEnabled",
    "/app/runLoops/main/rateLimitFrequency", "/app/hydraEngine/waitIdle", "/app/vsync",
    "/omni/replicator/captureOnPlay", "/omni/replicator/RTSubframes",
    "/exts/omni.syntheticdata/renderVarTextureToBuffer/cudaCopyNoStride",
    "/exts/omni.replicator/optim/useCachedParams", "/renderer/multiGpu/enabled",
    "/renderer/activeGpu", "/app/renderer/resolution/width", "/app/renderer/resolution/height",
    "/log/file",
)}
res["governor"] = {
    "scaling_governor": pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip(),
    "epp": pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference").read_text().strip(),
}

smi = subprocess.Popen(
    ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu", "--format=csv,noheader,nounits", "-lms", "100"],
    stdout=open(args.out + ".smi.csv", "w"), stderr=subprocess.DEVNULL,
)


def cpu_mhz() -> float:
    vals = [float(line.split(":")[1]) for line in open("/proc/cpuinfo") if line.startswith("cpu MHz")]
    return round(statistics.fmean(vals), 0)


def save() -> None:
    pathlib.Path(args.out + ".json").write_text(json.dumps(res, indent=1, default=str))


def mark(name: str) -> None:
    carb.log_warn(f"IRSIM_MARK {name}")
    res["marks"].append({"name": name, "wall": time.time(), "cpu_mhz": cpu_mhz()})
    print(f"=== {name} ===", flush=True)


def stats(xs: list[float]) -> dict[str, float]:
    ms = sorted(1e3 * x for x in xs)
    return {"mean": statistics.fmean(ms), "sd": statistics.stdev(ms) if len(ms) > 1 else 0.0,
            "p50": ms[len(ms) // 2], "min": ms[0], "max": ms[-1], "n": len(ms)}


def phase(name: str):
    def deco(fn):
        if name.split(":")[0] in SKIP:
            return fn
        mark(name)
        try:
            out = fn()
            if out is not None:
                res["phases"][name] = out
                brief = {k: (round(v["mean"], 2) if isinstance(v, dict) and "mean" in v else v)
                         for k, v in out.items()} if isinstance(out, dict) else out
                print(f"[{name}] {json.dumps(brief, default=str)[:900]}", flush=True)
        except Exception:
            res["errors"][name] = traceback.format_exc()
            print(f"[{name}] FAILED\n{traceback.format_exc()}", flush=True)
        save()
        return fn
    return deco


def mk_reader(device: str, channels: tuple[str, ...] | None = None) -> AovReader:
    names = {c: BASE_NAMES[c] for c in (channels or tuple(BASE_NAMES)) if c in BASE_NAMES}
    required = tuple(c for c in REQ if c in names)
    return AovReader(RP, device=device, names=names, required=required,
                     expected_shape=(H, W)).attach(settle_frames=4)


def graph_nodes() -> dict[str, int]:
    """Node types currently in the SyntheticData graphs, counted."""
    from omni.syntheticdata import SyntheticData
    counts: dict[str, int] = {}
    for _path, node in SyntheticData.Get()._graphNodes.items():
        try:
            if hasattr(node, "is_valid") and not node.is_valid():
                continue
            t = node.get_type_name()
        except Exception as exc:  # noqa: BLE001
            t = f"<{type(exc).__name__}>"
        counts[t] = counts.get(t, 0) + 1
    return counts


def handles_of(reader: AovReader) -> dict[str, object]:
    return {c: a.get_data() for c, a in reader._annotators.items()}


def data_of(v: object) -> object:
    return v.get("data", v) if isinstance(v, dict) else v


@wp.kernel
def finite_sum(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    v = a[i]
    if wp.isfinite(v):
        wp.atomic_add(out, 0, wp.float64(v))


def finite_sum_of(h) -> float:
    flat = wp.array(ptr=h.ptr, dtype=wp.float32, shape=(int(h.size),), device=h.device)
    out = wp.zeros(1, dtype=wp.float64, device=h.device)
    wp.launch(finite_sum, dim=flat.shape[0], inputs=[flat, out], device=h.device)
    return float(out.numpy()[0])


# compile + load the kernel now, so no timed loop pays for it
finite_sum_of(wp.zeros(16, dtype=wp.float32, device="cuda:0"))


def gpu_consume(reader: AovReader) -> float:
    """Use the distance plane ON the device: a kernel over the wrapped pointer, 4 bytes read back."""
    return finite_sum_of(data_of(reader._annotators["distance"].get_data()))


def loop(reader: AovReader | None, n: int, read_mode: str) -> dict[str, object]:
    steps, reads = [], []
    for _ in range(n):
        t0 = pc()
        if reader is None:
            rep.orchestrator.step(rt_subframes=1)
        else:
            reader.step(frames=1, rt_subframes=1)
        sync()
        t1 = pc()
        if reader is not None:
            if read_mode == "repo":
                reader.read()
            elif read_mode == "handles":
                handles_of(reader)
            elif read_mode == "gpu":
                gpu_consume(reader)
        sync()
        t2 = pc()
        steps.append(t1 - t0)
        reads.append(t2 - t1)
    return {"step": stats(steps), "read": stats(reads), "read_mode": read_mode,
            "total_mean": 1e3 * (statistics.fmean(steps) + statistics.fmean(reads))}


def warm(reader: AovReader | None, n: int = 10, mode: str = "repo") -> None:
    for _ in range(n):
        if reader is None:
            rep.orchestrator.step(rt_subframes=1)
        else:
            reader.step(frames=1, rt_subframes=1)
            reader.read() if mode == "repo" else handles_of(reader)


# ------------------------------------------------------------------------------- G: in context
cam_reader = camera._reader


def ctx_pass(reader: AovReader, n: int) -> dict[str, object]:
    camera._reader = reader
    _step, _read = reader.step, reader.read
    acc = {"step": 0.0, "read": 0.0}

    def t_step(**kw):
        t0 = pc(); _step(**kw); sync(); acc["step"] += pc() - t0

    def t_read():
        t0 = pc(); out = _read(); sync(); acc["read"] += pc() - t0
        return out

    reader.step, reader.read = t_step, t_read
    walls, s, r = [], [], []
    try:
        for i in range(n + 3):
            acc["step"] = acc["read"] = 0.0
            t0 = pc()
            camera.get_outputs(rt_subframes=1)
            if i >= 3:
                walls.append(pc() - t0); s.append(acc["step"]); r.append(acc["read"])
    finally:
        del reader.step, reader.read
        camera._reader = cam_reader
    return {"get_outputs": stats(walls), "step": stats(s), "read": stats(r)}


@phase("G:ctx_cpu_1")
def _():
    return ctx_pass(cam_reader, args.n_ctx)


cuda_reader_ctx: list[AovReader] = []


@phase("G:ctx_cuda_1")
def _():
    cam_reader.detach()
    r = mk_reader("cuda")
    cuda_reader_ctx.append(r)
    out = ctx_pass(r, args.n_ctx)
    out["nodes"] = graph_nodes()
    return out


@phase("G:ctx_cpu_2")
def _():
    for r in cuda_reader_ctx:
        r.detach()
    r = mk_reader("cpu")
    out = ctx_pass(r, args.n_ctx)
    r.detach()
    return out


@phase("G:ctx_cuda_2")
def _():
    r = mk_reader("cuda")
    out = ctx_pass(r, args.n_ctx)
    r.detach()
    return out


# ------------------------------------------------- X: the alternatives, inside get_outputs()
def ctx_pass_mode(reader: AovReader, n: int, mode: str) -> dict[str, object]:
    """get_outputs() with reader.step swapped for another way of advancing the renderer.

    The scene and camera are static, so a frame that is one update late is the same frame: the
    timing is that of the real pipeline, the pairing of pose and AOV is not exercised here.
    """
    camera._reader = reader
    _step, _read = reader.step, reader.read
    acc = {"step": 0.0, "read": 0.0}
    a = omni.kit.app.get_app()

    def t_step(**kw):
        t0 = pc()
        if mode == "wait":
            _step(**kw)
        else:
            rep.orchestrator.step(rt_subframes=1, wait_for_render=False)
            if mode == "nowait+1":
                a.update()
        sync()
        acc["step"] += pc() - t0

    def t_read():
        t0 = pc(); out = _read(); sync(); acc["read"] += pc() - t0
        return out

    reader.step, reader.read = t_step, t_read
    walls, s_, r_ = [], [], []
    try:
        for i in range(n + 3):
            acc["step"] = acc["read"] = 0.0
            t0 = pc()
            camera.get_outputs(rt_subframes=1)
            if i >= 3:
                walls.append(pc() - t0); s_.append(acc["step"]); r_.append(acc["read"])
    finally:
        del reader.step, reader.read
        camera._reader = cam_reader
    return {"get_outputs": stats(walls), "step": stats(s_), "read": stats(r_)}


@phase("X:alternatives")
def _():
    from omni.kit.viewport.utility import get_active_viewport
    vp = get_active_viewport()
    cam_reader.detach()
    out: dict[str, object] = {}
    for rep_i in (1, 2):
        for vp_on in (True, False):
            vp.updates_enabled = vp_on
            for dev in ("cpu", "cuda"):
                r = mk_reader(dev)
                for mode in ("wait", "nowait", "nowait+1"):
                    o = ctx_pass_mode(r, args.n_ctx, mode)
                    key = f"vp{'on' if vp_on else 'off'}:{dev}:{mode}:{rep_i}"
                    out[key] = {"step": o["step"]["mean"], "step_sd": o["step"]["sd"],
                                "read": o["read"]["mean"], "get_outputs": o["get_outputs"]["mean"]}
                    print(f"   {key:28s} step {o['step']['mean']:6.1f} +-{o['step']['sd']:4.1f}  read "
                          f"{o['read']['mean']:5.2f}  get_outputs {o['get_outputs']['mean']:7.1f}", flush=True)
                r.detach()
    vp.updates_enabled = True
    return out


# make sure nothing is attached before the tight loops
cam_reader.detach()
for _r in cuda_reader_ctx:
    _r.detach()

# ------------------------------------------------------------------------------- T: tight loop
def describe(v: object) -> dict[str, object]:
    d: dict[str, object] = {"type": f"{type(v).__module__}.{type(v).__name__}"}
    if isinstance(v, dict):
        d["keys"] = list(v.keys())
        if isinstance(v.get("info"), dict):
            d["info_keys"] = list(v["info"].keys())
        d["data"] = describe(v.get("data"))
    elif isinstance(v, wp.array):
        d.update(device=str(v.device), dtype=getattr(v.dtype, "__name__", str(v.dtype)),
                 shape=tuple(v.shape), strides=tuple(v.strides), ptr=hex(v.ptr or 0),
                 is_contiguous=bool(v.is_contiguous), has_deleter=v.deleter is not None,
                 cuda_array_interface=bool(getattr(v, "__cuda_array_interface__", None))
                 if str(v.device).startswith("cuda") else False)
    elif isinstance(v, np.ndarray):
        d.update(dtype=str(v.dtype), shape=tuple(v.shape), owndata=bool(v.flags.owndata),
                 ptr=hex(v.ctypes.data))
    return d


def tight(device: str | None, tag: str):
    @phase(f"T:{tag}")
    def _():
        if device is None:
            warm(None)
            out = {"none": loop(None, N, "none"), "nodes": graph_nodes()}
            out["step"] = out["none"]["step"]
            return out
        r = mk_reader(device)
        warm(r)
        out: dict[str, object] = {"nodes": graph_nodes(), "resolved": dict(r.resolved),
                                  "returns": {c: describe(v) for c, v in handles_of(r).items()}}
        out["repo"] = loop(r, N, "repo")
        out["handles"] = loop(r, N, "handles")
        if device.startswith("cuda"):
            out["gpu"] = loop(r, N, "gpu")
        out["repo_again"] = loop(r, N // 2, "repo")
        out["step"] = out["repo"]["step"]
        out["read"] = out["repo"]["read"]
        r.detach()
        return out


for tag, dev in (("cpu_1", "cpu"), ("none_1", None), ("cuda_1", "cuda"), ("cpu_2", "cpu"),
                 ("cuda_2", "cuda"), ("none_2", None)):
    tight(dev, tag)


# ------------------------------------------------------------------- D: inside orchestrator.step
class AppProxy:
    def __init__(self, real):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "times", [])

    def update(self):
        t0 = pc()
        self._real.update()
        self.times.append(pc() - t0)

    def __getattr__(self, k):
        return getattr(self._real, k)


def decompose(reader: AovReader | None, n: int) -> dict[str, object]:
    real_get = omni.kit.app.get_app
    proxy = AppProxy(real_get())
    counts, totals, inside, firsts, rests = [], [], [], [], []
    omni.kit.app.get_app = lambda: proxy
    try:
        for _ in range(n):
            proxy.times.clear()
            t0 = pc()
            rep.orchestrator.step(rt_subframes=1)
            totals.append(pc() - t0)
            counts.append(len(proxy.times))
            inside.append(sum(proxy.times))
            firsts.append(proxy.times[0] if proxy.times else 0.0)
            rests.extend(proxy.times[1:])
    finally:
        omni.kit.app.get_app = real_get
    return {"updates_per_step": {"mean": statistics.fmean(counts), "min": min(counts), "max": max(counts)},
            "step": stats(totals), "sum_of_updates": stats(inside),
            "outside_updates": stats([t - i for t, i in zip(totals, inside)]),
            "first_update": stats(firsts), "later_updates": stats(rests) if rests else None}


for dev in ("cpu", "cuda", None):
    @phase(f"D:{dev or 'none'}")
    def _(dev=dev):
        r = mk_reader(dev) if dev else None
        warm(r)
        out = decompose(r, max(20, N // 2))
        if r:
            r.detach()
        return out


# ---------------------------------------------------------------------------------- L: ladder
LADDER = [c for c in ("distance", "position", "normal", "instance", "semantic", "occlusion")
          if c in res["resolved"]]

def get_viewport():
    try:
        from omni.kit.viewport.utility import get_active_viewport
        return get_active_viewport()
    except Exception:  # noqa: BLE001
        return None


kit = omni.kit.app.get_app()


def raw_updates(n: int) -> dict[str, float]:
    ts = []
    for _ in range(n):
        t0 = pc(); kit.update(); sync(); ts.append(pc() - t0)
    return stats(ts)


def raw_both(n: int) -> dict[str, float]:
    """Mean ms per raw app.update(), default viewport rendering and not."""
    vp = get_viewport()
    was = bool(vp.updates_enabled) if vp is not None else None
    out = {}
    for label, on in (("vp_on", True), ("vp_off", False)):
        if vp is None and not on:
            continue
        if vp is not None:
            vp.updates_enabled = on
        for _ in range(8):
            kit.update()
        st = raw_updates(n)
        out[label] = st["p50"]
        out[label + "_mean"] = st["mean"]
    if vp is not None:
        vp.updates_enabled = was
        for _ in range(4):
            kit.update()
    return out


for dev in ("cpu", "cuda"):
    @phase(f"L:{dev}")
    def _(dev=dev):
        out: dict[str, object] = {"none": raw_both(max(20, N // 2))}
        for k in range(1, len(LADDER) + 1):
            subset = tuple(LADDER[:k])
            r = mk_reader(dev, subset)
            warm(r, 6, "handles")
            out["+".join(subset)] = raw_both(max(20, N // 2))
            r.detach()
        for c in LADDER[1:]:
            r = mk_reader(dev, (c,))
            warm(r, 6, "handles")
            out[f"only:{c}"] = raw_both(max(20, N // 2))
            r.detach()
        return out


# ------------------------------------------------------------------------- R: raw app.update()
@phase("R:raw")
def _():
    out: dict[str, object] = {}
    ht = getattr(camera._render_product, "hydra_texture", None)
    vp = get_viewport()
    out["has_hydra_texture"] = ht is not None
    out["has_viewport"] = vp is not None
    if vp is not None:
        out["viewport_resolution"] = tuple(vp.resolution)
        out["viewport_updates_enabled"] = bool(vp.updates_enabled)
    for _ in range(10):
        kit.update()
    out["update_no_annotators"] = raw_updates(N)
    if ht is not None:
        ht.set_updates_enabled(False)
        for _ in range(10):
            kit.update()
        out["update_render_product_off"] = raw_updates(N)
        if vp is not None:
            vp.updates_enabled = False
            for _ in range(10):
                kit.update()
            out["update_render_product_off_viewport_off"] = raw_updates(N)
            vp.updates_enabled = True
        ht.set_updates_enabled(True)
        for _ in range(10):
            kit.update()
    for dev in ("cpu", "cuda"):
        r = mk_reader(dev)
        warm(r, 6)
        both = raw_both(N)
        out[f"update_{dev}_annotators"] = {"p50": both.get("vp_on"), "mean": both.get("vp_on_mean")}
        out[f"update_{dev}_annotators_viewport_off"] = {"p50": both.get("vp_off"), "mean": both.get("vp_off_mean")}
        r.detach()
    return out


# ----------------------------------------------- H: get_data() cost, per device and copy flag
from irsim_isaac.pipeline.gbuffer_isaac import AOV_INIT_PARAMS  # noqa: E402


class RawSet:
    """Annotators attached directly, so `do_array_copy` (which AovReader does not expose) can vary."""

    def __init__(self, device: str, do_copy: bool, channels: tuple[str, ...] | None = None):
        self.annos = {}
        for c in channels or tuple(LADDER):
            a = rep.AnnotatorRegistry.get_annotator(
                res["resolved"][c], init_params=AOV_INIT_PARAMS.get(c), device=device,
                do_array_copy=do_copy)
            a.attach(RP)
            self.annos[c] = a
        for _ in range(4):
            rep.orchestrator.step(rt_subframes=1)

    def get(self) -> dict[str, object]:
        return {c: a.get_data() for c, a in self.annos.items()}

    def detach(self) -> None:
        for a in self.annos.values():
            a.detach(RP)
        self.annos.clear()


@phase("H:get_data")
def _():
    out: dict[str, object] = {}
    for dev in ("cpu", "cuda"):
        for do_copy in (True, False):
            rs = RawSet(dev, do_copy)
            for _ in range(6):
                rep.orchestrator.step(rt_subframes=1)
                rs.get()
            steps, gets, hosts = [], [], []
            for _ in range(max(20, N // 2)):
                t0 = pc(); rep.orchestrator.step(rt_subframes=1); sync()
                t1 = pc(); hs = rs.get(); sync()
                t2 = pc()
                for v in hs.values():
                    v = data_of(v)
                    v.numpy() if hasattr(v, "numpy") else np.asarray(v)
                sync()
                t3 = pc()
                steps.append(t1 - t0); gets.append(t2 - t1); hosts.append(t3 - t2)
            key = f"{dev}_copy{int(do_copy)}"
            per = {c: [] for c in rs.annos}
            for _ in range(20):
                rep.orchestrator.step(rt_subframes=1)
                for c, a in rs.annos.items():
                    t0 = pc(); a.get_data(); sync(); per[c].append(pc() - t0)
            out[key] = {"step": stats(steps)["mean"], "get_data_all": stats(gets)["mean"],
                        "then_to_host": stats(hosts)["mean"],
                        "get_data_per_channel": {c: stats(v)["mean"] for c, v in per.items()},
                        "distance": describe(data_of(rs.get()["distance"]))}
            rs.detach()
    return out


# -------------------------------------------------------------- K: known value, zero-copy wrap
from pxr import Gf, Usd, UsdGeom  # noqa: E402

cam_prim = stage.GetPrimAtPath(camera_path)
CAM_M = UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
CAM_P = CAM_M.ExtractTranslation()
CAM_F = CAM_M.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()
HALF = 0.1
cube = UsdGeom.Cube.Define(stage, "/World/ProbeCube")
cube.GetSizeAttr().Set(2 * HALF)
cube.GetExtentAttr().Set([(-HALF, -HALF, -HALF), (HALF, HALF, HALF)])
_xf = UsdGeom.Xformable(cube)
_xf.ClearXformOpOrder()
_mop = _xf.AddTransformOp()
CY, CX = H // 2, W // 2
OFFSET = [None]  # measured distance minus authored distance (K:known_value fills it)


def place(d: float) -> None:
    """Cube face towards the camera sits exactly d metres along the optical axis."""
    m = Gf.Matrix4d(CAM_M)
    m.SetTranslateOnly(CAM_P + CAM_F * (d + HALF))
    _mop.Set(m)


def centre(arr) -> float:
    arr = arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)
    a = arr.reshape(H, W, -1)[CY - 1:CY + 1, CX - 1:CX + 1, 0]
    return float(a.mean())


def wrap(h):
    return wp.array(ptr=h.ptr, dtype=h.dtype, shape=h.shape, strides=h.strides, device=h.device)


@phase("K:known_value")
def _():
    out: dict[str, object] = {}
    clip = UsdGeom.Camera(cam_prim).GetClippingRangeAttr()
    out["clipping_range"] = tuple(clip.Get())
    place(0.700)
    rc = RawSet("cuda", True, ("distance", "position"))   # what AovReader(device="cuda") attaches
    rh = RawSet("cpu", True, ("distance",))              # an independent host-side reading
    for _ in range(4):
        rep.orchestrator.step(rt_subframes=1)
    h = data_of(rc.get()["distance"])
    host = np.array(data_of(rh.get()["distance"]), copy=True)
    out["authored_face_distance_m"] = 0.700
    out["cuda_copy1_type"] = describe(h)
    out["cpu_type"] = describe(data_of(rh.get()["distance"]))
    out["cpu_centre_m"] = centre(host)
    pos_full = data_of(rc.get()["position"]).numpy().reshape(H, W, -1)[..., :3].astype(np.float64)
    pos = pos_full[CY - 1:CY + 1, CX - 1:CX + 1].reshape(-1, 3).mean(0)
    out["position_aov_centre_xyz"] = [float(x) for x in pos]
    out["position_aov_centre_norm"] = float(np.linalg.norm(pos))
    # |position| - distance over the frame: constant, or near/cos(theta)?
    rng = np.linalg.norm(pos_full, axis=-1)
    ok = np.isfinite(host) & np.isfinite(rng) & (rng > 0)
    diff = np.where(ok, rng - host, np.nan)
    cos_t = np.where(ok, -pos_full[..., 2] / np.where(rng > 0, rng, 1.0), np.nan)
    near = float(clip.Get()[0])
    out["range_minus_distance"] = {
        "centre": float(diff[CY, CX]), "min": float(np.nanmin(diff)), "max": float(np.nanmax(diff)),
        "cos_theta_min": float(np.nanmin(cos_t)),
        "max_abs_err_vs_near_over_cos": float(np.nanmax(np.abs(diff - near / cos_t))),
        "max_abs_err_vs_constant_near": float(np.nanmax(np.abs(diff - near))),
    }

    host_sum = float(host[np.isfinite(host)].astype(np.float64).sum())
    out["host_finite_sum"] = host_sum

    def wrap_test(hh) -> dict[str, object]:
        t0 = pc()
        w = wrap(hh)
        o: dict[str, object] = {"wrap_ms": 1e3 * (pc() - t0), "same_ptr": w.ptr == hh.ptr,
                                "wrapper_owns_memory": w.deleter is not None}
        wn = w.numpy()
        o["centre_m"] = centre(wn)
        o["bit_identical_to_cpu_annotator"] = bool(np.array_equal(wn.reshape(host.shape), host))
        o["gpu_kernel_finite_sum"] = finite_sum_of(hh)
        o["gpu_kernel_sum_rel_err"] = abs(o["gpu_kernel_finite_sum"] - host_sum) / host_sum
        return o

    out["wrap_copy1"] = wrap_test(h)
    OFFSET[0] = out["wrap_copy1"]["centre_m"] - 0.700

    # aliasing, no step in between. (a) write through a wrapper of the do_array_copy=True handle
    wrap(h).fill_(777.0)
    sync()
    h_again = data_of(rc.get()["distance"])
    out["alias_copy1"] = {"fresh_ptr_same": h_again.ptr == h.ptr, "fresh_centre": centre(h_again),
                          "held_handle_centre": centre(h),
                          "cpu_annotator_centre": centre(data_of(rh.get()["distance"]))}
    seq1 = {"authored": [], "held_wrapper": [], "fresh": [], "ptr": [hex(h.ptr)]}
    w1 = wrap(h)
    for d in (0.800, 0.900, 1.000):
        place(d)
        rep.orchestrator.step(rt_subframes=1)
        a = data_of(rc.get()["distance"])
        seq1["authored"].append(d); seq1["held_wrapper"].append(centre(w1))
        seq1["fresh"].append(centre(a)); seq1["ptr"].append(hex(a.ptr))
    out["sequence_copy1"] = seq1
    rc.detach()

    # (b) the same AOV with do_array_copy=False, on a node of its own (a shared node keeps the
    # first annotator's flag, so it cannot be attached beside the copy=True one)
    place(0.700)
    rn = RawSet("cuda", False, ("distance",))
    for _ in range(4):
        rep.orchestrator.step(rt_subframes=1)
    hn = data_of(rn.get()["distance"])
    host = np.array(data_of(rh.get()["distance"]), copy=True)
    host_sum = float(host[np.isfinite(host)].astype(np.float64).sum())
    out["cuda_copy0_type"] = describe(hn)
    out["wrap_copy0"] = wrap_test(hn)
    wn0 = wrap(hn)
    wn0.fill_(555.0)
    sync()
    hn_again = data_of(rn.get()["distance"])
    out["alias_copy0"] = {"fresh_ptr_same": hn_again.ptr == hn.ptr, "fresh_centre": centre(hn_again),
                          "cpu_annotator_centre": centre(data_of(rh.get()["distance"]))}

    # lifetime: move the cube, step, read through the OLD copy0 wrapper without asking again
    seq = {"authored": [], "old_copy0_wrapper": [], "fresh_copy0": [], "cpu": [],
           "ptr_copy0": [hex(hn.ptr)]}
    for d in (0.800, 0.900, 1.000, 1.100, 1.200, 1.300):
        place(d)
        rep.orchestrator.step(rt_subframes=1)
        seq["authored"].append(d)
        seq["old_copy0_wrapper"].append(centre(wn0))
        a = data_of(rn.get()["distance"])
        seq["ptr_copy0"].append(hex(a.ptr))
        seq["fresh_copy0"].append(centre(a))
        seq["cpu"].append(centre(data_of(rh.get()["distance"])))
    out["sequence_copy0"] = seq
    rc = rn  # the near-clip test below reads through `rc`

    # where the constant offset comes from: does the measured distance follow the near clip?
    near_test = {}
    place(1.000)
    for near in (0.05, 0.20, 0.01, 0.05):
        clip.Set(Gf.Vec2f(near, 1.0e4))
        for _ in range(4):
            rep.orchestrator.step(rt_subframes=1)
        near_test[f"near={near}#{len(near_test)}"] = centre(data_of(rc.get()["distance"]))
    out["near_clip_test_authored_1.000"] = near_test
    rn.detach(); rh.detach()
    return out


# ------------------------------------------------------------------------------------ LAG
D0, DSTEP = 0.600, 0.005


class Counting:
    """Count and time every app.update() made through `omni.kit.app.get_app()`."""

    def __enter__(self):
        self._real_get = omni.kit.app.get_app
        self.proxy = AppProxy(self._real_get())
        omni.kit.app.get_app = lambda: self.proxy
        return self.proxy

    def __exit__(self, *exc):
        omni.kit.app.get_app = self._real_get


def lag_run(anno, advance, n: int = 60) -> dict[str, object]:
    off = OFFSET[0] if OFFSET[0] is not None else 0.0
    lags, ts, ups, stale, last = [], [], [], 0, None
    with Counting() as proxy:
        for k in range(n):
            place(D0 + DSTEP * k)
            proxy.times.clear()
            t0 = pc()
            advance()
            sync()
            ts.append(pc() - t0)
            ups.append(len(proxy.times))
            v = centre(data_of(anno.get_data()))
            lags.append(k - round((v - off - D0) / DSTEP))
            stale += int(last is not None and abs(v - last) < 1e-6)
            last = v
    tail = lags[10:]
    return {"advance": stats(ts[10:]), "updates_per_advance": statistics.fmean(ups[10:]),
            "lag_frames_mode": max(set(tail), key=tail.count), "lag_min": min(tail),
            "lag_max": max(tail), "stale_reads": stale}


def upd() -> None:
    omni.kit.app.get_app().update()


def step_nowait(**kw) -> None:
    rep.orchestrator.step(rt_subframes=1, wait_for_render=False, **kw)


ADVANCES = {
    "orch_wait": lambda: rep.orchestrator.step(rt_subframes=1),
    "orch_wait_dt": lambda: rep.orchestrator.step(rt_subframes=1, delta_time=1.0 / 30.0),
    "orch_nowait": step_nowait,
    "orch_nowait_dt": lambda: step_nowait(delta_time=1.0 / 30.0),
    "orch_nowait_then_50ms_idle": lambda: (step_nowait(), time.sleep(0.05)),
    "orch_nowait_plus1_update": lambda: (step_nowait(), upd()),
    "orch_nowait_plus2_updates": lambda: (step_nowait(), upd(), upd()),
    "raw_update": upd,
    "raw_2_updates": lambda: (upd(), upd()),
    "raw_3_updates": lambda: (upd(), upd(), upd()),
    "orch_wait_again": lambda: rep.orchestrator.step(rt_subframes=1),
}

for dev in ("cuda", "cpu"):
    @phase(f"LAG:{dev}")
    def _(dev=dev):
        out: dict[str, object] = {"offset_m": OFFSET[0]}
        r = mk_reader(dev)
        warm(r, 6)
        anno = r._annotators["distance"]
        for name, adv in ADVANCES.items():
            out[name] = lag_run(anno, adv)
            o = out[name]
            print(f"   {dev:5s} {name:28s} {o['advance']['mean']:7.2f} ms  updates "
                  f"{o['updates_per_advance']:.2f}  lag {o['lag_frames_mode']} "
                  f"[{o['lag_min']},{o['lag_max']}] stale {o['stale_reads']}", flush=True)
        r.detach()
        return out


# ----------------------------------------------------------------------------------- V: viewport
@phase("V:viewport_off")
def _():
    out: dict[str, object] = {}
    vp = get_viewport()
    out["has_viewport"] = vp is not None
    if vp is None:
        return out
    for dev in ("cpu", "cuda"):
        r = mk_reader(dev)
        warm(r, 6)
        out[f"{dev}_on"] = loop(r, max(20, N // 2), "repo")
        vp.updates_enabled = False
        warm(r, 6)
        out[f"{dev}_off"] = loop(r, max(20, N // 2), "repo")
        for name in ("orch_wait", "orch_nowait", "orch_nowait_plus1_update",
                     "orch_nowait_plus2_updates", "raw_update"):
            o = lag_run(r._annotators["distance"], ADVANCES[name])
            out[f"{dev}_off:{name}"] = o
            print(f"   vp-off {dev:5s} {name:28s} {o['advance']['mean']:7.2f} ms  updates "
                  f"{o['updates_per_advance']:.2f}  lag {o['lag_frames_mode']} "
                  f"[{o['lag_min']},{o['lag_max']}] stale {o['stale_reads']}", flush=True)
        vp.updates_enabled = True
        warm(r, 6)
        r.detach()
    return out


# ------------------------------------------------ P: timeline playing (last: it is not undone)
import omni.timeline  # noqa: E402


@phase("P:playing")
def _():
    out: dict[str, object] = {}
    r = mk_reader("cuda")
    warm(r, 6)
    anno = r._annotators["distance"]
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        kit.update()
    out["is_playing"] = bool(timeline.is_playing())
    out["raw_update_playing"] = lag_run(anno, upd)
    out["raw_2_updates_playing"] = lag_run(anno, lambda: (upd(), upd()))
    return out


mark("END")
smi.terminate()
save()
print(f"wrote {args.out}.json", flush=True)
app.close()
