#!/usr/bin/env python
"""Outside view of the simulated three-cup set-up (Kit): the scene, proxies at both camera centres, a raised camera.

    env -u LD_LIBRARY_PATH -u PYTHONPATH -u AMENT_PREFIX_PATH -u ROS_DISTRO -u ROS_VERSION -u ROS_PYTHON_VERSION \
        /home/jumasaet/isaac_sim/python.sh bench/cups3_check/render_world.py

The three-cup counterpart of bench/cups_check/presentation/render_rgb_world.py (outside view only). The scene is
task2_render.py's, built by its own functions: three paper cups at the LiDAR ranges and image bearings, the fitted
pose with roll, the board, the table, the colours and the light scale of the RGB sheet. On top of it:
  - the RGB camera's pose from the RGB-thermal calibration (P_thermal = T_th_rgb P_rgb, OpenCV axes; in USD axes
    world-from-RGB = M_w,th F T_th_rgb F, F = diag(1, -1, -1)) -- placed, not rendered;
  - two small proxy boxes at the optical centres (orange thermal, blue RGB): markers, not modelled bodies;
  - an outside camera, diagonal and raised, rendered with 48 subframes.

Writes out/world/: ext.png and cams.json (world-from-camera poses in USD axes, column-vector convention, and the
outside camera's intrinsics read back from camera_params). make_world.py draws the fields of view on it.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

import task2_render as t2  # noqa: E402  (puts bench/camera_preview on sys.path; reads only JSON at import)

cm, c3 = t2.cm, t2.c3
OUT = c3.OUT / "world"
OUT.mkdir(parents=True, exist_ok=True)
EXT_W, EXT_H, EXT_F = 1600, 1000, 900.0
EXT_EYE = np.array([0.20, 0.62, 0.30])
EXT_TARGET = np.array([-0.56, -0.04, 0.07])
SUBFRAMES = 48
F = np.diag([1.0, -1.0, -1.0, 1.0])


# -- copied from bench/cups_check/presentation/render_rgb_world.py (that script reads argv at import) ------------


def world_from_prim(stage, path: str) -> np.ndarray:
    from pxr import Usd, UsdGeom

    m = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(m, dtype=np.float64).T


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World-from-camera, USD camera axes (looks down -Z, +Y up)."""
    f = target - eye
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, -f, eye
    return m


def define_camera(stage, path: str, m_wc: np.ndarray, f_px: float, w: int, h: int):
    from pxr import Gf, UsdGeom

    p = 0.003
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(float(f_px * p))
    cam.CreateHorizontalApertureAttr(float(w * p))
    cam.CreateVerticalApertureAttr(float(h * p))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.005, 100.0))
    cam.CreateFStopAttr(0.0)
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(m_wc.T.tolist()))
    return cam


def proxy_box(stage, path: str, m_wc: np.ndarray, edge: float, rgb) -> None:
    """A cube of side `edge`, aligned with the camera, its front face on the optical centre."""
    from pxr import Gf, UsdGeom

    back = np.eye(4)
    back[2, 3] = edge / 2
    m = m_wc @ back
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(float(edge))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    xf = UsdGeom.Xformable(cube)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(m.T.tolist()))


def main() -> None:
    t2.patch_library()
    t2.patch_stage()
    t2.patch_roll(t2.POSE["camera_roll_deg"])
    cm.ns.DOME_INTENSITY *= t2.RGB_LIGHT_SCALE
    cm.ns.DISTANT_INTENSITY *= t2.RGB_LIGHT_SCALE
    objs, thermal = t2.objects()
    scene_yaml = t2.write_scene(thermal, t2.write_weather())
    th = math.radians(t2.POSE["camera_pitch_up_deg"])
    pos = np.array([0.0, 0.0, t2.POSE["camera_height_above_table_m"]])
    look = np.array([-math.cos(th), 0.0, math.sin(th)])
    ctx = cm.boot(sensor_yaml=t2.SENSOR, scene_yaml=scene_yaml, objects=objs,
                  materials={n: m for n, (m, _t) in thermal.items()}, first_pose=(pos, look))
    cm.render_sync(ctx, pos, look)  # the thermal camera's pose is now on the stage

    import omni.usd
    from PIL import Image

    rep = ctx["rep"]
    stage = omni.usd.get_context().get_stage()
    m_wth = world_from_prim(stage, cm.ns.CAMERA_PATH)
    m_wrgb = m_wth @ F @ c3.cc.T_TH_RGB @ F
    stage.DefinePrim("/Props", "Xform")
    proxy_box(stage, "/Props/ThermalCamBody", m_wth, 0.030, (0.85, 0.25, 0.10))
    proxy_box(stage, "/Props/RgbCamBody", m_wrgb, 0.024, (0.10, 0.35, 0.85))
    stage.DefinePrim("/Rig", "Xform")
    m_wext = look_at(EXT_EYE, EXT_TARGET)
    define_camera(stage, "/Rig/ExtCam", m_wext, EXT_F, EXT_W, EXT_H)
    rpe = rep.create.render_product("/Rig/ExtCam", (EXT_W, EXT_H))
    annos = {}
    for name, key in (("rgb", "rgb"), ("camera_params", "cam")):
        a = rep.AnnotatorRegistry.get_annotator(name)
        a.attach(rpe)
        annos[key] = a
    for _ in range(2):
        rep.orchestrator.step(rt_subframes=SUBFRAMES)
    got = {k: a.get_data() for k, a in annos.items()}
    ext = np.asarray(cm._data(got["rgb"]))[..., :3].astype(np.uint8)
    Image.fromarray(ext).save(OUT / "ext.png")
    proj = np.asarray(got["cam"]["cameraProjection"], dtype=np.float64).reshape(4, 4)
    view = np.asarray(got["cam"]["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
    cams = {
        "sensor_yaml": str(t2.SENSOR.relative_to(c3.REPO)),
        "thermal": {"world_from_camera_usd": m_wth.tolist(), "position": pos.tolist(), "look": look.tolist(),
                    "roll_deg": t2.POSE["camera_roll_deg"]},
        "rgb": {"world_from_camera_usd": m_wrgb.tolist()},
        "ext": {"eye": EXT_EYE.tolist(), "target": EXT_TARGET.tolist(), "w": EXT_W, "h": EXT_H, "f_px_set": EXT_F,
                "fx_px": proj[0, 0] * EXT_W / 2, "fy_px": proj[1, 1] * EXT_H / 2,
                "world_from_camera_usd": np.linalg.inv(view).T.tolist()},
        "objects": objs, "light_scale": t2.RGB_LIGHT_SCALE,
    }
    (OUT / "cams.json").write_text(json.dumps(cams, indent=1, default=float))
    print("DONE", json.dumps({"ext_fx": cams["ext"]["fx_px"]}), flush=True)
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
