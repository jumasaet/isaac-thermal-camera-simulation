"""The industrial-hall comparison scene as an Isaac stage, its camera paths, and the Kit boot.

/tmp only; nothing here edits the repo. The stage is **+Z up, metres**, as the scene it copies:
positions, sizes and light rotations are the original's numbers, unrotated. `IrCamera.planes()`
needs a +Y-up stage (it measures sky azimuth about the up axis), but the lite path never calls
it: it reads the instance-id and distance annotators directly, and its environment is one
scalar, so no direction on this stage enters the radiometry.

    OBJECTS       geometry + visible colour, one row per prim   (temperatures: scene_nave.yaml)
    trajectory()  the two camera paths, as (position, look direction) per frame
    boot()        SimulationApp + stage + IrCamera (not opened)
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import yaml

REPO = pathlib.Path("/home/jumasaet/isaac-thermal-camera-simulation")
BENCH = pathlib.Path("/tmp/irsim_render_cost")
DEMO = BENCH / "demo"
SENSOR_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
SCENE_YAML = DEMO / "scene_nave.yaml"
MATERIALS_YAML = DEMO / "nave_materials.yaml"

#: name, kind, centre (m), size (box: x,y,z | cyl: radius,height,axis | sphere: radius), sRGB.
#: The colours are this script's choice -- the original's are not in the brief.
OBJECTS: list[dict] = [
    dict(name="Floor", kind="box", centre=(0.0, 0.0, -0.05), size=(20.0, 20.0, 0.1), rgb=(0.45, 0.45, 0.44)),
    dict(name="Wall", kind="box", centre=(-4.0, 0.0, 2.0), size=(0.3, 12.0, 4.0), rgb=(0.62, 0.60, 0.56)),
    dict(name="HotPipe", kind="cyl", centre=(-3.5, 0.0, 1.6), size=(0.15, 8.0, "Y"), rgb=(0.42, 0.24, 0.16)),
    dict(name="Anomaly", kind="box", centre=(-1.5, -1.0, 0.175), size=(0.35, 0.35, 0.35), rgb=(0.25, 0.27, 0.30)),
    dict(name="Crate_01", kind="box", centre=(-2.2, 1.6, 0.3), size=(0.6, 0.6, 0.6), rgb=(0.20, 0.35, 0.62)),
    dict(name="Crate_02", kind="box", centre=(-0.8, 2.3, 0.3), size=(0.6, 0.6, 0.6), rgb=(0.70, 0.55, 0.20)),
    dict(name="Ball", kind="sphere", centre=(0.3, -2.2, 0.3), size=(0.3,), rgb=(0.75, 0.20, 0.15)),
]

#: Lights of the original: a bluish dome at 600 and a distant light at 2500, 1 degree, rotateXYZ.
DOME_INTENSITY = 600.0
DOME_RGB = (0.75, 0.85, 1.0)  # "azulada": the brief gives no numbers, this is a guess
DISTANT_INTENSITY = 2500.0
DISTANT_ANGLE_DEG = 1.0
DISTANT_ROTATE_XYZ = (40.0, 0.0, 30.0)

CAMERA_PATH = "/World/IrCamera"
WORLD_UP = (0.0, 0.0, 1.0)


def target_name(prim_name: str) -> str:
    """Prim name -> target name in scene_nave.yaml (HotPipe -> hot_pipe, Crate_01 -> crate_01)."""
    import re

    return re.sub(r"(?<=[a-z])(?=[A-Z])", "_", prim_name).lower()


def load_materials(overrides: list[str] | None = None) -> dict[str, str]:
    mats = dict(yaml.safe_load(MATERIALS_YAML.read_text(encoding="utf-8")))
    for item in overrides or []:
        prim, _, mat = item.partition("=")
        if prim not in mats or not mat:
            raise SystemExit(f"--set-material {item!r}: expected one of {sorted(mats)}=<material>")
        mats[prim] = mat
    missing = {o["name"] for o in OBJECTS} ^ set(mats)
    if missing:
        raise SystemExit(f"nave_materials.yaml and OBJECTS disagree on: {sorted(missing)}")
    return mats


def sensor_ss1_yaml() -> pathlib.Path:
    """The repo's Boson YAML with ONE added line: ``supersample_factor: 1``.

    The repo file does not set it, and the schema default is 4 (a 2560x2048 render product).
    The 640x512 timings of the earlier sessions were all taken with this override.
    """
    text = SENSOR_YAML.read_text(encoding="utf-8")
    anchor = "    f_number: 1.0\n"
    if text.count(anchor) != 1:
        raise SystemExit("flir_boson_640_lwir.yaml changed shape: cannot place supersample_factor")
    out = DEMO / "out" / "_sensor_boson_ss1.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text.replace(anchor, anchor + "    supersample_factor: 1   # demo override (/tmp)\n"))
    return out


# -- camera paths ---------------------------------------------------------------------------


def triangle(n: int) -> np.ndarray:
    """s[k] in [0, 1]: 0 at k = 0, 1 at k = n/2, back to 2/n at k = n-1 (loops seamlessly).

    No two consecutive frames share a pose, which is what lets a repeated read be told from a
    fresh one by its pose stamp alone.
    """
    k = np.arange(n, dtype=np.float64)
    return 1.0 - np.abs(1.0 - 2.0 * k / n)


def trajectory(name: str, n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """``(positions (n, 3), look directions (n, 3))`` in stage axes (+Z up, metres)."""
    s = triangle(n)
    if name == "linear":
        pos = np.stack([np.full(n, 2.5), -4.5 + 9.0 * s, np.full(n, 2.0)], axis=1)
        t = math.radians(-12.0)
        look = np.tile(np.array([-math.cos(t), 0.0, math.sin(t)]), (n, 1))
    elif name == "circular":
        centre = np.array([-2.0, 0.0, 0.8])
        th = np.radians(-45.0 + 90.0 * s)
        pos = np.stack([centre[0] + 6.0 * np.cos(th), centre[1] + 6.0 * np.sin(th), np.full(n, 2.2)], axis=1)
        look = centre[None, :] - pos
        look /= np.linalg.norm(look, axis=1, keepdims=True)
    else:
        raise ValueError(name)
    return pos, look


class CameraOps:
    """Translate + orient ops of the camera prim; wings level about stage +Z."""

    def __init__(self, stage, path: str) -> None:
        from pxr import Gf, UsdGeom

        cam = UsdGeom.Camera.Define(stage, path)
        xf = UsdGeom.Xformable(cam)
        xf.ClearXformOpOrder()
        self._t = xf.AddTranslateOp()
        self._q = xf.AddOrientOp()
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1.0e4))

    def set_pose(self, position, look) -> None:
        from pxr import Gf

        from irsim_isaac.aircraft_pass import look_at_quaternion

        q = look_at_quaternion(look, world_up=WORLD_UP)
        self._t.Set(Gf.Vec3d(*(float(c) for c in position)))
        self._q.Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*(float(v) for v in q[1:]))))


# -- stage ----------------------------------------------------------------------------------


def author_stage(materials: dict[str, str], lights: bool = True):
    """A fresh +Z-up stage with the seven prims and the two lights. Returns (stage, prim_to_target)."""
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")
    prim_to_target: dict[str, str] = {}
    for o in OBJECTS:
        path = f"/World/{o['name']}"
        if o["kind"] == "box":
            g = UsdGeom.Cube.Define(stage, path)
            g.CreateSizeAttr(1.0)
            scale = Gf.Vec3f(*o["size"])
        elif o["kind"] == "cyl":
            radius, height, axis = o["size"]
            g = UsdGeom.Cylinder.Define(stage, path)
            g.CreateRadiusAttr(radius)
            g.CreateHeightAttr(height)
            g.CreateAxisAttr(axis)
            scale = None
        else:
            g = UsdGeom.Sphere.Define(stage, path)
            g.CreateRadiusAttr(o["size"][0])
            scale = None
        xf = UsdGeom.Xformable(g)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*o["centre"]))
        if scale is not None:
            xf.AddScaleOp().Set(scale)
        g.CreateDisplayColorAttr([Gf.Vec3f(*o["rgb"])])
        g.GetPrim().CreateAttribute("thermal:material", Sdf.ValueTypeNames.String).Set(materials[o["name"]])
        prim_to_target[path] = target_name(o["name"])
    if lights:
        # Outside /World, so the material resolver's stage walk does not meet them.
        stage.DefinePrim("/Lights", "Xform")
        dome = UsdLux.DomeLight.Define(stage, "/Lights/Dome")
        dome.CreateIntensityAttr(DOME_INTENSITY)
        dome.CreateColorAttr(Gf.Vec3f(*DOME_RGB))
        sun = UsdLux.DistantLight.Define(stage, "/Lights/Distant")
        sun.CreateIntensityAttr(DISTANT_INTENSITY)
        sun.CreateAngleAttr(DISTANT_ANGLE_DEG)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(*DISTANT_ROTATE_XYZ))
    return stage, prim_to_target


def boot(*, gui: bool, viewport: bool, kit_args: list[str], materials: dict[str, str],
         capture_rgb: bool = True, aa: bool = False):
    """SimulationApp + stage + IrCamera (not opened), as the verified lite benches boot it."""
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(BENCH / "lite"))
    from irsim_isaac.env import simulation_app_config

    from isaacsim import SimulationApp

    sys.argv = [sys.argv[0], *kit_args]
    if not aa:
        for a in ("--/rtx/post/aa/op=0", "--/rtx/post/dlss/execMode=0"):
            if a not in sys.argv:
                sys.argv.append(a)
    over: dict = {"headless": not gui}
    if not viewport:
        over["disable_viewport_updates"] = True
    app = SimulationApp(simulation_app_config(**over))

    import carb
    import warp as wp

    wp.init()
    settings = carb.settings.get_settings()
    # RaytracedLighting reads back as RealTimePathTracing on this build; ask for it by name.
    settings.set("/rtx/rendermode", "RealTimePathTracing")
    if not aa:
        settings.set("/rtx/post/aa/op", 0)

    from irsim.atmosphere.skylight import skylight_for_sensor
    from irsim.config.loader import load_sensor_config
    from irsim.config.scene import load_scene_config
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import MaterialResolver, load_mapping_rules
    from irsim.materials.table import MaterialTable
    from irsim.pipeline.core import PipelineConfig
    from irsim.radiometry.lut_files import load_band_lut_for_config, load_band_response_for_config
    from irsim.scene import Scene
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.materials_usd import prim_records

    sensor = load_sensor_config(sensor_ss1_yaml())
    spec = sensor.sensor
    band, quantity = spec.band.band_id, spec.quantity
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    response = load_band_response_for_config(sensor, REPO / "data")
    skylight = skylight_for_sensor(sensor, quantity)
    scene = Scene.from_config(load_scene_config(SCENE_YAML), {band: lut}, quantity=quantity,
                              skylights={band: skylight}, responses={band: response})

    stage, prim_to_target = author_stage(materials)
    cam_ops = CameraOps(stage, CAMERA_PATH)
    pos, look = trajectory("linear")
    cam_ops.set_pose(pos[0], look[0])

    table = MaterialTable.from_library(MaterialLibrary.load(), band)
    resolver = MaterialResolver(load_mapping_rules(), list(table.names))
    resolutions = resolver.resolve_all(prim_records(root="/World"))
    pipeline = PipelineConfig.from_sensor(sensor, table, lut, sky=scene.sky_models[band],
                                          atmosphere=scene.layered, flat_field_enabled=False)
    camera = IrCamera(
        sensor, scene, pipeline=pipeline, prim_to_target=prim_to_target, resolutions=resolutions,
        camera_path=CAMERA_PATH, strict_materials=False, strict_thermal_nodes=False,
        strict_patch_coverage=False, heading_deg=0.0, capture_rgb=False,
    )
    return dict(app=app, wp=wp, settings=settings, sensor=sensor, scene=scene, lut=lut,
                quantity=quantity, band=band, table=table, pipeline=pipeline, camera=camera,
                cam_ops=cam_ops, stage=stage, prim_to_target=prim_to_target,
                resolutions=resolutions, capture_rgb=capture_rgb)
