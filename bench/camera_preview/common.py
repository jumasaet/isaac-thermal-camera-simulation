"""Shared by render_comparison.py (task 1) and two_cups/render_two_cups.py (task 3).

The path is the one the reference run uses as its oracle (bench/render_cost/code/rebase_2026-10-22/
ref/run_ref.py, `equivalence()`): Kit renders only the G-buffer (instance ids + distance + RGB),
synchronously, and the whole camera runs on the host in `irsim.pipeline.frame.run_frame` with the
same chain as the reference run (factory flat field + M9 sensor chain: FPN, bad pixels, FFC, drift).
No device kernels of ours, so nothing here depends on the GPU bench's freshness bookkeeping beyond
one check: every read must carry the pose and serial this script wrote (`render_sync`).

Stage: +Z up, metres. The stage author and the camera ops are the bench's (`nave_stage`), fed our
object table; boot is nave_stage.boot with the sensor YAML, scene YAML and LUT dir as arguments.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import numpy as np

REPO = pathlib.Path("/home/jumasaet/isaac-thermal-camera-simulation")
BENCH = REPO / "bench" / "render_cost" / "code" / "rebase_2026-10-22"
HERE = REPO / "bench" / "camera_preview"
CELLPLUS_YAML = REPO / "bench" / "camera_check" / "cellplus_ctc007_lwir.yaml"
LUT_DIR = HERE / "lut"  # scripts/generate_luts.py --configs <CELLPLUS_YAML> --out bench/camera_preview/lut
ENV_ISOTHERMAL = BENCH / "env_isothermal.yaml"
for p in (REPO / "src", BENCH / "demo", BENCH / "lite"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import nave_stage as ns  # noqa: E402

POSE_TOL_M = 1e-4


def boot(*, sensor_yaml: pathlib.Path, scene_yaml: pathlib.Path, objects: list[dict], materials: dict[str, str],
         first_pose: tuple[np.ndarray, np.ndarray], gui: bool = False, lights: bool = True) -> dict:
    """SimulationApp + stage + opened IrCamera with our own annotators. Returns a context dict."""
    sys.argv = [sys.argv[0], "--/rtx/post/aa/op=0", "--/rtx/post/dlss/execMode=0"]
    from irsim_isaac.env import simulation_app_config

    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=not gui, disable_viewport_updates=True))
    import carb
    import warp as wp

    wp.init()
    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "RealTimePathTracing")
    settings.set("/rtx/post/aa/op", 0)

    from irsim.atmosphere.skylight import skylight_for_sensor
    from irsim.config.loader import load_sensor_config
    from irsim.config.scene import load_scene_config
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import MaterialResolver, load_mapping_rules
    from irsim.materials.table import MaterialTable
    from irsim.pipeline.core import PipelineConfig
    from irsim.pipeline.sensor_chain import attach_sensor_chain
    from irsim.radiometry.lut_files import load_band_lut_for_config, load_band_response_for_config
    from irsim.scene import Scene
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.materials_usd import prim_records

    sensor = load_sensor_config(sensor_yaml)
    spec = sensor.sensor
    band, quantity = spec.band.band_id, spec.quantity
    lut = load_band_lut_for_config(sensor, LUT_DIR)
    response = load_band_response_for_config(sensor, REPO / "data")
    skylight = skylight_for_sensor(sensor, quantity)
    scene = Scene.from_config(load_scene_config(scene_yaml), {band: lut}, quantity=quantity,
                              skylights={band: skylight}, responses={band: response})

    ns.OBJECTS = objects  # author_stage walks this table
    stage, prim_to_target = ns.author_stage(materials, lights=lights)
    cam_ops = ns.CameraOps(stage, ns.CAMERA_PATH)
    cam_ops.set_pose(*first_pose)

    table = MaterialTable.from_library(MaterialLibrary.load(), band)
    resolver = MaterialResolver(load_mapping_rules(), list(table.names))
    resolutions = resolver.resolve_all(prim_records(root="/World"))
    pipeline = PipelineConfig.from_sensor(sensor, table, lut, sky=scene.sky_models[band], atmosphere=scene.layered,
                                          flat_field_enabled=False)
    camera = IrCamera(sensor, scene, pipeline=pipeline, prim_to_target=prim_to_target, resolutions=resolutions,
                      camera_path=ns.CAMERA_PATH, strict_materials=False, strict_thermal_nodes=False,
                      strict_patch_coverage=False, heading_deg=0.0, capture_rgb=False).open(settle_frames=16)
    # The camera chain run_frame uses: the reference run's g_m9_ff variant.
    cfg_ff = PipelineConfig.from_sensor(sensor, table, lut, sky=scene.sky_models[band], atmosphere=scene.layered,
                                        flat_field_enabled=True)
    cfg_run = attach_sensor_chain(cfg_ff, scene.weather, t0_s=scene.t0_s)

    import omni.replicator.core as rep
    from pxr import UsdGeom

    for _ in range(3):
        settings.set("/rtx/rendermode", "RealTimePathTracing")
        camera._reader.step(frames=1, rt_subframes=1)
    rp = camera._reader.render_product_path
    camera._reader.detach()

    def attach(name, device=None, init=None):
        a = rep.AnnotatorRegistry.get_annotator(name, init_params=init, **({"device": device} if device else {}))
        a.attach(rp)
        return a

    annos = {"instance": attach("instance_id_segmentation", "cuda", {"colorize": False}),
             "distance": attach("DistanceToCameraSD", "cuda"), "cam": attach("camera_params"),
             "rgb": attach("rgb", "cuda")}
    for _ in range(8):
        rep.orchestrator.step(rt_subframes=1)
    ws, hs = camera.optics.resolution
    h, w = spec.fpa_shape
    focus = UsdGeom.Camera(stage.GetPrimAtPath(ns.CAMERA_PATH)).GetFocusDistanceAttr()
    proj = np.asarray(annos["cam"].get_data()["cameraProjection"], dtype=np.float64).reshape(4, 4)
    return dict(app=app, rep=rep, settings=settings, sensor=sensor, spec=spec, scene=scene, lut=lut, quantity=quantity,
                camera=camera, cfg=cfg_run, cam_ops=cam_ops, annos=annos, focus=focus, serial=[1000],
                shape=(h, w), shape_ss=(hs, ws), k=hs // h,
                hfov_deg=float(np.degrees(2 * np.arctan(1 / proj[0, 0]))),
                vfov_deg=float(np.degrees(2 * np.arctan(1 / proj[1, 1]))))


def _data(v):
    return v.get("data", v) if isinstance(v, dict) else v


def render_sync(ctx: dict, position, look) -> dict:
    """Write the pose and a fresh serial (in focusDistance, fStop 0), step, read, and refuse any read
    whose camera_params stamp is not this pose and this serial."""
    ctx["serial"][0] += 1
    ser = ctx["serial"][0]
    ctx["cam_ops"].set_pose(position, look)
    ctx["focus"].Set(float(ser))
    for _ in range(3):
        ctx["rep"].orchestrator.step(rt_subframes=1)
    payload = {c: a.get_data() for c, a in ctx["annos"].items()}
    cam = payload["cam"]
    c2w = np.linalg.inv(np.asarray(cam["cameraViewTransform"], dtype=np.float64).reshape(4, 4))
    pos_s, fwd_s = c2w[3, :3], -c2w[2, :3]
    ser_s = float(np.asarray(cam["cameraFocusDistance"]).reshape(-1)[0])
    look = np.asarray(look, float) / np.linalg.norm(look)
    if np.abs(pos_s - np.asarray(position)).max() > POSE_TOL_M or fwd_s @ look < 1 - 1e-9 or int(round(ser_s)) != ser:
        raise RuntimeError(f"synchronous render delivered another frame (serial {ser_s} != {ser})")
    return payload


class Clock:
    """The thermal clock and the camera state across a sequence: one frame period per call."""

    def __init__(self, ctx: dict, min_distance_m: float | None = None) -> None:
        from irsim.pipeline.core import PipelineState

        self.ctx = ctx
        #: Floor on the distance plane the camera chain sees (None = off, the default). A bench workaround for
        #: `layered` defocus: its eight bins are equal-width in W020 over the frame's whole range, so a table
        #: reaching to 3 cm under the lens takes seven of them and leaves the 0.5 m cup sharing the 1 m board's
        #: kernel (bench/cups_check/RESULTADOS.md, task 6). Only geometry nearer than the floor changes.
        self.min_distance_m = min_distance_m
        self.t0 = ctx["camera"].scene.t0_s
        self.period = ctx["camera"].frame_period_s
        self.rel = 0.0
        self.state = PipelineState(t_s=self.t0)

    def frame(self, payload: dict):
        """G-buffer planes from the payload (as run_ref.gbuffer_planes), then run_frame. -> (Outputs, rgb, ids)."""
        import lite
        from irsim.pipeline.frame import run_frame
        from irsim_isaac.pipeline.material_ids import labels_from_payload

        ctx, camera = self.ctx, self.ctx["camera"]
        hs, ws = ctx["shape_ss"]
        camera.bridge.advance_to(self.rel)
        t_s = self.t0 + self.rel
        camera.state.t_s = t_s
        self.rel += self.period
        ids = _data(payload["instance"]).numpy().reshape(hs, ws).astype(np.uint32)
        labels = labels_from_payload(payload["instance"])
        _l, temp, mat, sky = lite.reduced_numpy(ids, labels, camera, t_s)
        dist = _data(payload["distance"]).numpy().reshape(hs, ws)
        dist = np.where(sky, 0.0, np.nan_to_num(dist, posinf=0.0)).astype(np.float32)
        if self.min_distance_m:
            dist = np.where(sky, 0.0, np.maximum(dist, np.float32(self.min_distance_m))).astype(np.float32)
        planes = {"temperature_k": temp, "material_id": mat, "sky_mask": sky, "distance_m": dist,
                  "sky_view_factor": np.full((hs, ws), 0.5, np.float32)}
        self.state.t_s = t_s
        out = run_frame(planes, ctx["cfg"], self.state)
        for name in ("apparent_t", "radiance"):
            a = getattr(out, name)
            assert a is not None and a.dtype == np.float32, (name, None if a is None else a.dtype)
        k = ctx["k"]
        rgb = _data(payload["rgb"]).numpy().reshape(hs, ws, -1)[..., :3].astype(np.float32)
        h, w = ctx["shape"]
        rgb = np.rint(rgb.reshape(h, k, w, k, 3).mean(axis=(1, 3))).astype(np.uint8)
        return out, rgb, ids


def window_gray(t_k: np.ndarray, lo_c: float, hi_c: float) -> np.ndarray:
    """Apparent temperature (K) -> 8-bit gray, white hot, one FIXED linear window (no AGC)."""
    x = (np.asarray(t_k, np.float64) - 273.15 - lo_c) / (hi_c - lo_c)
    return np.clip(np.rint(255.0 * x), 0, 255).astype(np.uint8)


def write_mp4(path: pathlib.Path, frames: np.ndarray, fps: int) -> str:
    """frames (N, H, W) gray or (N, H, W, 3) uint8 -> H.264 MP4."""
    if shutil.which("ffmpeg") is None:
        return "no ffmpeg"
    if frames.ndim == 3:
        frames = np.repeat(frames[..., None], 3, axis=3)
    n, h, w, _ = frames.shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14", "-movflags", "+faststart", str(path)]
    r = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
    return "ok" if r.returncode == 0 else r.stderr.decode()[-400:]
