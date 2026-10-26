"""Shared set-up for the lite-path checks. /tmp only; nothing here edits the repo.

Boots Kit, authors the inspection bench, and builds the IrCamera exactly as bench_x.py does --
except that the temperatures come from ``targets: solver: prescribed`` (one value per prim), which
is the premise of the lite path. The geometry is still authored from the patched scene YAML,
because that is where the boxes' extents live.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile
import time

import yaml

REPO = pathlib.Path("/home/jumasaet/isaac-thermal-camera-simulation")
BENCH = pathlib.Path("/tmp/irsim_render_cost")

#: One temperature per prim, all away from the 300 K enclosure so (1 - eps) L_env is not hidden.
PRESCRIBED_K = {"wall": 303.15, "hot_pipe": 353.15, "cold_floor": 291.15}

pc = time.perf_counter


def base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--sensor", default=str(BENCH / "sensor_ss1.yaml"))
    p.add_argument("--scene", required=True)
    p.add_argument("--temps", choices=("prescribed", "surfaces"), default="prescribed")
    p.add_argument("--rendermode", default="RaytracedLighting")
    p.add_argument("--disable-viewport", action="store_true")
    p.add_argument("--capture-rgb", action="store_true")
    p.add_argument("--light", action="store_true", help="author a dome light (for RGB content)")
    p.add_argument("--settle", type=int, default=16)
    p.add_argument("--out", required=True)
    return p


def prescribed_yaml(scene_path: str) -> str:
    """The scene with its thermal.surfaces swapped for prescribed per-prim targets."""
    raw = yaml.safe_load(pathlib.Path(scene_path).read_text(encoding="utf-8"))
    spec = raw["scene"]
    names = [s["name"] for s in spec["thermal"]["surfaces"]]
    del spec["thermal"]
    spec["targets"] = [
        {"name": n, "solver": "prescribed", "schedule_s": [0.0, 86400.0],
         "schedule_k": [PRESCRIBED_K[n], PRESCRIBED_K[n]]}
        for n in names
    ]
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=str(BENCH / "lite"))
    yaml.safe_dump(raw, f)
    f.close()
    return f.name


def boot(args, kit_args):
    """SimulationApp + bench stage + IrCamera (not opened). Returns a namespace-like dict."""
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "scripts"))
    from irsim_isaac.env import simulation_app_config

    from isaacsim import SimulationApp

    sys.argv = [sys.argv[0], *kit_args]
    for a in ("--/rtx/post/aa/op=0", "--/rtx/post/dlss/execMode=0"):
        if a not in sys.argv:
            sys.argv.append(a)
    app = SimulationApp(simulation_app_config(
        **({"disable_viewport_updates": True} if args.disable_viewport else {})))

    import carb
    import warp as wp

    wp.init()
    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", args.rendermode)
    settings.set("/rtx/sceneDb/ambientLightIntensity", 0.0)
    settings.set("/rtx/post/aa/op", 0)

    from irsim.atmosphere.skylight import skylight_for_sensor
    from irsim.config.loader import load_sensor_config
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import MaterialResolver, load_mapping_rules
    from irsim.materials.table import MaterialTable
    from irsim.pipeline.core import PipelineConfig
    from irsim.radiometry.lut_files import load_band_lut_for_config, load_band_response_for_config
    from irsim.scene import Scene
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.materials_usd import prim_records
    from irsim_isaac.pipeline.point_bridge import bindings_from_scene

    from inspection_bench_stage import author_camera_ops, new_bench_stage
    from render_inspection_bench_offline import CameraPlacement, stage_scene_config

    sensor = load_sensor_config(args.sensor)
    spec = sensor.sensor
    band = spec.band.band_id
    quantity = spec.quantity
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    response = load_band_response_for_config(sensor, REPO / "data")
    skylight = skylight_for_sensor(sensor, quantity)

    def build(cfg):
        return Scene.from_config(cfg, {band: lut}, quantity=quantity,
                                 skylights={band: skylight}, responses={band: response})

    geo_cfg = stage_scene_config(args.scene)
    geo_scene = build(geo_cfg)
    materials = {s.name: s.material for s in geo_cfg.scene.thermal.surfaces}
    stage, prim_to_target = new_bench_stage(geo_scene, materials)
    if args.temps == "prescribed":
        scene = build(stage_scene_config(prescribed_yaml(args.scene)))
        fields = ()
    else:
        scene = geo_scene
        fields = bindings_from_scene(scene)

    if args.light:
        from pxr import UsdLux
        dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
        dome.CreateIntensityAttr(1000.0)

    placement = CameraPlacement(position_enu=(0.0, 0.5, 2.0), tilt_deg=-20.0)
    camera_path = "/World/IrCamera"
    cam_ops = author_camera_ops(stage, camera_path, placement)
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
        strict_patch_coverage=False, surface_fields=fields, heading_deg=0.0,
        capture_rgb=args.capture_rgb,
    )
    return dict(app=app, wp=wp, settings=settings, sensor=sensor, scene=scene, lut=lut,
                quantity=quantity, band=band, table=table, pipeline=pipeline, camera=camera,
                cam_ops=cam_ops, placement_cls=CameraPlacement, stage=stage,
                prim_to_target=prim_to_target, resolutions=resolutions)
