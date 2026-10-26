#!/usr/bin/env python
"""Presentation renders for one case: the rig's RGB camera, and an outside view of the simulated world.

    /home/jumasaet/isaac_sim/python.sh bench/cups_check/presentation/render_rgb_world.py <case>
        case = hot_water_50 | cold_water_95   (any of the four works)

The scene is task3_render_cups.py's, built by its own functions (one paper cup, the fitted pose, the CTC-006
camera of task 6 with the 0.3 m defocus floor). On top of it:

* an RGB camera placed with the RGB-thermal calibration (thermal_specs/cali_board_new/
  RGBT_calibration_result_matrix.txt, ~2 px stereo error): P_thermal = T_th_rgb P_rgb in OpenCV axes, so in
  USD axes (F = diag(1, -1, -1)) the world-from-RGB pose is  M_w,th F T_th_rgb F.  Isaac's camera is a pinhole,
  so the RGB view is rendered as a pinhole with the calibrated fx on a canvas wide enough to hold what the real
  lens's barrel distortion pulls in (2304 x 1728 at f = 992.6 px); make_presentation.py applies the calibrated
  Brown-Conrady distortion afterwards, so the comparison happens in the real image's own pixels;
* small proxy boxes at the two camera centres (added after the RGB capture, so they never block a view), and an
  outside camera, diagonal and raised.

The lights are dimmed for these captures only (the bench's are set for the thermal camera's RGB preview and
saturate): nothing thermal reads them.

Writes presentation/render/<case>/: rgb_pinhole.png, rgb_ids.npy + rgb_labels.json (instance mask of the RGB
view), ext.png, cams.json (every pose world-from-camera, USD axes, column-vector convention, with the
intrinsics each product actually rendered with, read back from camera_params), thermal_apparent_t_k.npy (this
session's thermal frame, a consistency check against out/sim6_<case>/).
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
CUPS = HERE.parent
REPO = CUPS.parents[1]
CASE = sys.argv[1]
SENSOR = REPO / "bench/camera_check/cellplus_ctc006_lwir_cups.yaml"
# task3_render_cups reads its case, YAML and output directory from argv at import time.
sys.argv = [str(CUPS / "task3_render_cups.py"), CASE, str(SENSOR), f"../presentation/render/{CASE}"]
sys.path.insert(0, str(CUPS))
import task3_render_cups as t3  # noqa: E402

cm, cc = t3.cm, t3.cc
OUT = t3.OUT

RGB_W, RGB_H = 2304, 1728          # pinhole canvas, centred principal point
RGB_F = 992.57287085               # calibrated fx (fy = 993.23 differs by 0.07 %)
EXT_W, EXT_H, EXT_F = 1600, 1000, 900.0
EXT_EYE = np.array([0.42, 0.78, 0.40])
EXT_TARGET = np.array([-0.62, -0.12, 0.16])
LIGHT_SCALE = 0.12
SUBFRAMES = 48
F = np.diag([1.0, -1.0, -1.0, 1.0])
T_TH_RGB = cc.T_TH_RGB


def world_from_prim(stage, path: str) -> np.ndarray:
    from pxr import Usd, UsdGeom

    m = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(m, dtype=np.float64).T          # Gf is row-vector; this is the column-vector matrix


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
    """A pinhole USD camera with square pixels and a centred principal point (apertures in 'mm' of a 3 um pixel)."""
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
    back[2, 3] = edge / 2                          # +Z is behind a USD camera
    m = m_wc @ back
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(float(edge))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    xf = UsdGeom.Xformable(cube)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(m.T.tolist()))


def intrinsics_from_params(params: dict, w: int, h: int) -> dict:
    proj = np.asarray(params["cameraProjection"], dtype=np.float64).reshape(4, 4)
    view = np.asarray(params["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
    return {"fx_px": proj[0, 0] * w / 2, "fy_px": proj[1, 1] * h / 2,
            "proj_2_0": proj[2, 0], "proj_2_1": proj[2, 1],
            "world_from_camera_usd": np.linalg.inv(view).T.tolist()}


def capture(rep, annos: dict, subframes: int) -> dict:
    for _ in range(2):
        rep.orchestrator.step(rt_subframes=subframes)
    return {k: a.get_data() for k, a in annos.items()}


def main() -> None:
    t3.patch_library()
    t3.patch_stage()
    objs, thermal = t3.objects()
    scene_yaml = t3.write_scene(thermal, t3.write_weather())
    th = math.radians(t3.POSE["camera_pitch_up_deg"])
    pos = np.array([0.0, 0.0, t3.POSE["camera_height_above_table_m"]])
    look = np.array([-math.cos(th), 0.0, math.sin(th)])
    ctx = cm.boot(sensor_yaml=SENSOR, scene_yaml=scene_yaml, objects=objs,
                  materials={n: m for n, (m, _t) in thermal.items()}, first_pose=(pos, look))
    clock = cm.Clock(ctx, min_distance_m=0.3)
    for _ in range(4):
        clock.frame(cm.render_sync(ctx, pos, look))
    out, _rgb, _ids = clock.frame(cm.render_sync(ctx, pos, look))
    assert out.apparent_t.dtype == np.float32
    np.save(OUT / "thermal_apparent_t_k.npy", out.apparent_t)

    import omni.usd
    from PIL import Image

    from irsim_isaac.pipeline.material_ids import labels_from_payload, labels_to_paths

    rep = ctx["rep"]
    stage = omni.usd.get_context().get_stage()
    for p in ("/Lights/Dome", "/Lights/Distant"):
        a = stage.GetPrimAtPath(p).GetAttribute("inputs:intensity")
        a.Set(float(a.Get()) * LIGHT_SCALE)

    m_wth = world_from_prim(stage, cm.ns.CAMERA_PATH)
    m_wrgb = m_wth @ F @ T_TH_RGB @ F
    stage.DefinePrim("/Rig", "Xform")
    define_camera(stage, "/Rig/RgbCam", m_wrgb, RGB_F, RGB_W, RGB_H)
    rp = rep.create.render_product("/Rig/RgbCam", (RGB_W, RGB_H))
    annos = {}
    for name, key, init in (("rgb", "rgb", None), ("instance_id_segmentation", "ids", {"colorize": False}),
                            ("camera_params", "cam", None)):
        a = rep.AnnotatorRegistry.get_annotator(name, init_params=init)
        a.attach(rp)
        annos[key] = a
    got = capture(rep, annos, SUBFRAMES)
    rgb = np.asarray(cm._data(got["rgb"]))[..., :3].astype(np.uint8)
    Image.fromarray(rgb).save(OUT / "rgb_pinhole.png")
    ids = np.asarray(cm._data(got["ids"])).reshape(RGB_H, RGB_W).astype(np.uint32)
    np.save(OUT / "rgb_ids.npy", ids)
    paths = labels_to_paths(labels_from_payload(got["ids"]))
    (OUT / "rgb_labels.json").write_text(json.dumps({str(k): v for k, v in paths.items()}, indent=1))
    rgb_meta = intrinsics_from_params(got["cam"], RGB_W, RGB_H)
    for a in annos.values():
        a.detach()

    # Outside view: proxies at the two optical centres, then a diagonal raised camera.
    stage.DefinePrim("/Props", "Xform")
    proxy_box(stage, "/Props/ThermalCamBody", m_wth, 0.030, (0.85, 0.25, 0.10))
    proxy_box(stage, "/Props/RgbCamBody", m_wrgb, 0.024, (0.10, 0.35, 0.85))
    m_wext = look_at(EXT_EYE, EXT_TARGET)
    define_camera(stage, "/Rig/ExtCam", m_wext, EXT_F, EXT_W, EXT_H)
    rpe = rep.create.render_product("/Rig/ExtCam", (EXT_W, EXT_H))
    ext_annos = {}
    for name, key in (("rgb", "rgb"), ("camera_params", "cam")):
        a = rep.AnnotatorRegistry.get_annotator(name)
        a.attach(rpe)
        ext_annos[key] = a
    got = capture(rep, ext_annos, SUBFRAMES)
    ext = np.asarray(cm._data(got["rgb"]))[..., :3].astype(np.uint8)
    Image.fromarray(ext).save(OUT / "ext.png")
    ext_meta = intrinsics_from_params(got["cam"], EXT_W, EXT_H)

    cams = {
        "case": CASE, "sensor_yaml": str(SENSOR.relative_to(REPO)),
        "thermal": {"world_from_camera_usd": m_wth.tolist(), "position": pos.tolist(), "look": look.tolist()},
        "rgb": {"world_from_camera_usd_set": m_wrgb.tolist(), "w": RGB_W, "h": RGB_H, "f_px_set": RGB_F, **rgb_meta},
        "ext": {"world_from_camera_usd_set": m_wext.tolist(), "w": EXT_W, "h": EXT_H, "f_px_set": EXT_F,
                "eye": EXT_EYE.tolist(), "target": EXT_TARGET.tolist(), **ext_meta},
        "objects": objs, "light_scale": LIGHT_SCALE, "defocus_min_distance_m": 0.3,
    }
    (OUT / "cams.json").write_text(json.dumps(cams, indent=1, default=float))
    print("DONE", CASE, json.dumps({"rgb_fx": rgb_meta["fx_px"], "rgb_p20": rgb_meta["proj_2_0"],
                                    "ext_fx": ext_meta["fx_px"]}), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
