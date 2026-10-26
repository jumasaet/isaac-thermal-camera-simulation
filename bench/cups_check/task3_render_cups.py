#!/usr/bin/env python
"""Task 3: one paper cup per case through the simulator, at the measured temperatures and the measured pose.

    /home/jumasaet/isaac_sim/python.sh bench/cups_check/task3_render_cups.py <case> [sensor_yaml] [out_subdir]

    case         hot_water_50 | hot_water_95 | cold_water_50 | cold_water_95
    sensor_yaml  default bench/camera_check/cellplus_ctc007_lwir_cups.yaml (task 4: measured calibration);
                 the original cellplus_ctc007_lwir.yaml gives the simulator's own geometry (task 2)
    out_subdir   default sim_<case>, under bench/cups_check/out/

Built on bench/camera_preview/two_cups/render_two_cups.py and the same render path (common.py: Kit G-buffer,
the whole camera on the host in irsim.pipeline.frame.run_frame), changed to the experiment:
  - ONE cup, a FRUSTUM (7 cm top, 5 cm bottom, 6 cm tall; the template has cylinders) of PAPER
    (materials/paper_cup.yaml, LWIR eps 0.93, added to the library at run time -- configs/ untouched);
  - the cup in two parts: the wetted wall up to the water line (3.8 cm, read off the 50 cm frames: the
    steep step 64 % of the way up) at the measured surface temperature, and the dry band above it at
    the mean of that and the air (ESTIMATED; it only feeds the blur around the rim, not the comparison);
  - pose from task 2 (out/lidar_geometry.json): camera 3.23 cm above the table, pitched 1.21 deg up (fitted
    to the cup rows; the LiDAR pose is ~1.4 deg off), cup axis and board at the LiDAR distances;
  - temperatures: cold 11 C (both), hot 66 C at 50 cm and 58 C at 95 cm (the time-order assumption of
    task 1); air, board and table 22 C; isothermal room, one WeatherSeries for air and enclosure (rule 6).

Writes out/<subdir>/: frame.png (fixed window), frame_rgb.png, scene_apparent_t_k.npy, apparent_t_k.npy,
ids.npy (native grid), report.json -- the cup's scene_apparent_t and apparent_t over its water region, and
its silhouette in pixels off the instance mask (exact, no blur) for task 2.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "camera_preview"))
sys.path.insert(0, str(HERE))
import common as cm  # noqa: E402

CASE = sys.argv[1]
SENSOR = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else cm.REPO / "bench/camera_check/cellplus_ctc007_lwir_cups.yaml"
OUT = HERE / "out" / (sys.argv[3] if len(sys.argv) > 3 else f"sim_{CASE}")
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402

import cups_common as cc  # noqa: E402

T_SURFACE_C = {"hot_water_50": 66.0, "hot_water_95": 58.0, "cold_water_50": 11.0, "cold_water_95": 11.0}
WATER_LINE_M = 0.038
PAPER = HERE / "materials" / "paper_cup.yaml"
GEO = json.loads((cc.OUT / "lidar_geometry.json").read_text())
POSE = GEO["image_pose"]
G = GEO["cases"][CASE]


def r_at(z: float) -> float:
    return cc.CUP_BOTTOM_DIAMETER_M / 2 + (cc.CUP_TOP_DIAMETER_M - cc.CUP_BOTTOM_DIAMETER_M) / 2 * z / cc.CUP_HEIGHT_M


def objects() -> tuple[list[dict], dict]:
    t_s = T_SURFACE_C[CASE]
    t_air = cc.T_AIR_C
    d_cup, lat = G["cup_axis_horizontal_distance_m"], -G["cup_axis_lateral_m"]  # template: -Y = left in image
    d_board = G["board_distance_horizontal_m"]
    objs = [
        dict(name="Background", kind="box", centre=(-d_board - 0.01, 0.0, 0.45 - 0.05), size=(0.02, 1.4, 0.9),
             rgb=(0.85, 0.85, 0.85)),
        dict(name="Table", kind="box", centre=((0.3 - d_board) / 2, 0.0, -0.01), size=(0.3 + d_board, 1.6, 0.02),
             rgb=(0.75, 0.62, 0.45)),
        dict(name="CupWet", kind="frustum", centre=(-d_cup, lat, 0.0), z=(0.0, WATER_LINE_M),
             rgb=(0.9, 0.9, 0.88)),
        dict(name="CupDry", kind="frustum", centre=(-d_cup, lat, 0.0), z=(WATER_LINE_M, cc.CUP_HEIGHT_M),
             rgb=(0.95, 0.95, 0.93)),
    ]
    thermal = {"Background": ("concrete_white", t_air), "Table": ("plywood", t_air),
               "CupWet": ("paper_cup", t_s), "CupDry": ("paper_cup", 0.5 * (t_s + t_air))}
    return objs, thermal


def patch_library() -> None:
    """Add paper_cup to every MaterialLibrary.load() of this process (configs/materials/ is not touched)."""
    from irsim.materials import library as lib

    orig = lib.MaterialLibrary.load.__func__

    def load(cls, material_dir=None, data_dir=None):
        base = orig(cls, material_dir, data_dir)
        mats = dict(base.items())
        mats["paper_cup"] = lib.load_material(PAPER, data_dir)
        return cls(mats)

    lib.MaterialLibrary.load = classmethod(load)


def patch_stage() -> None:
    """nave_stage.author_stage knows box / cyl / sphere; author our frustum parts as closed meshes."""
    ns = cm.ns
    orig = ns.author_stage

    def author_stage(materials, lights=True):
        all_objs = ns.OBJECTS
        ns.OBJECTS = [o for o in all_objs if o["kind"] != "frustum"]
        stage, p2t = orig(materials, lights=lights)
        ns.OBJECTS = all_objs
        from pxr import Gf, Sdf, UsdGeom, Vt

        for o in all_objs:
            if o["kind"] != "frustum":
                continue
            path = f"/World/{o['name']}"
            z0, z1 = o["z"]
            n = 192
            ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
            pts = [Gf.Vec3f(float(r_at(z) * math.cos(a)), float(r_at(z) * math.sin(a)), float(z)) for z in (z0, z1) for a in ang]
            pts += [Gf.Vec3f(0, 0, float(z0)), Gf.Vec3f(0, 0, float(z1))]
            counts, idx = [], []
            for i in range(n):
                j = (i + 1) % n
                counts.append(4)
                idx += [i, j, n + j, n + i]  # side, outward winding
                counts.append(3)
                idx += [2 * n, j, i]  # bottom cap
                counts.append(3)
                idx += [2 * n + 1, n + i, n + j]  # top cap
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
            xf = UsdGeom.Xformable(mesh)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*o["centre"]))
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*o["rgb"])])
            mesh.GetPrim().CreateAttribute("thermal:material", Sdf.ValueTypeNames.String).Set(materials[o["name"]])
            p2t[path] = ns.target_name(o["name"])
        return stage, p2t

    ns.author_stage = author_stage


def write_weather() -> pathlib.Path:
    t = cc.T_AIR_C + 273.15
    rows = [f"2024-06-{d}T{h:02d}:00:00Z,{t:.2f},0.5,0.1,1.0,0.0,0.0,23000.0,0.0" for d in (21, 22) for h in range(24)]
    path = OUT / "weather_constant.csv"
    path.write_text("# irsim weather v1\n# SYNTHETIC closed room: constant air temperature, fully covered, no sun "
                    "(GENERATED by task3_render_cups.py)\n"
                    "time_utc,t_air_k,rh_fraction,wind_speed_m_s,cloud_fraction,dni_w_m2,dhi_w_m2,visibility_m,precip_mm_h\n"
                    + "\n".join(rows) + "\n")
    return path


def write_scene(thermal: dict, weather: pathlib.Path) -> pathlib.Path:
    rows = "\n".join(f"    - {{name: {cm.ns.target_name(n)}, solver: prescribed, schedule_s: [0.0, 86400.0], "
                     f"schedule_k: [{t + 273.15:.2f}, {t + 273.15:.2f}]}}  # {t:g} C, {m}" for n, (m, t) in thermal.items())
    path = OUT / "scene.yaml"
    path.write_text(f"""# One paper cup, {CASE}. GENERATED by bench/cups_check/task3_render_cups.py.
schema_version: 17
scene:
  name: cups_{CASE}
  description: "Paper cup before a flat matte board, prescribed temperatures, isothermal room at {cc.T_AIR_C:g} C"
  weather_file: {weather}
  atmosphere_preset: us_standard_clear
  environment_preset: {cm.ENV_ISOTHERMAL}
  site: {{latitude_deg: 37.5, longitude_deg: 127.0, altitude_m: 50.0}}
  start_utc: "2024-06-21T02:00:00Z"

  targets:
{rows}
""")
    return path


def silhouette(mask_ss: np.ndarray, k: int) -> dict:
    """Top/bottom rows and widths 10 % in from each, in native pixels, off a supersampled mask."""
    rows = np.nonzero(mask_ss.any(1))[0]
    top, bot = rows.min() / k, (rows.max() + 1) / k
    hgt = bot - top

    def width(row_native: float) -> float:
        r = int(round(row_native * k))
        cols = np.nonzero(mask_ss[r])[0]
        return (cols.max() + 1 - cols.min()) / k

    cols = np.nonzero(mask_ss.any(0))[0]
    return {"top_row": round(float(top), 2), "bottom_row": round(float(bot), 2), "height_px": round(float(hgt), 2),
            "width_top10_px": round(float(width(top + 0.1 * hgt)), 2),
            "width_bottom10_px": round(float(width(bot - 0.1 * hgt)), 2),
            "centre_col": round(float((cols.min() + cols.max() + 1) / 2 / k), 2)}


def main() -> None:
    patch_library()
    patch_stage()
    objs, thermal = objects()
    scene_yaml = write_scene(thermal, write_weather())
    th = math.radians(POSE["camera_pitch_up_deg"])
    pos = np.array([0.0, 0.0, POSE["camera_height_above_table_m"]])
    look = np.array([-math.cos(th), 0.0, math.sin(th)])
    ctx = cm.boot(sensor_yaml=SENSOR, scene_yaml=scene_yaml, objects=objs,
                  materials={n: m for n, (m, _t) in thermal.items()}, first_pose=(pos, look))
    from PIL import Image

    from irsim_isaac.pipeline.material_ids import labels_from_payload, labels_to_paths

    clock = cm.Clock(ctx)
    for _ in range(4):
        clock.frame(cm.render_sync(ctx, pos, look))
    payload = cm.render_sync(ctx, pos, look)
    out, rgb, ids_ss = clock.frame(payload)
    assert out.scene_apparent_t is not None and out.scene_apparent_t.dtype == np.float32
    k = ctx["k"]
    h, w = ctx["shape"]
    ids = ids_ss[k // 2::k, k // 2::k][:h, :w]
    paths = labels_to_paths(labels_from_payload(payload["instance"]))
    by_name = {p.rsplit("/", 1)[-1]: i for i, p in paths.items()}
    np.save(OUT / "scene_apparent_t_k.npy", out.scene_apparent_t)
    np.save(OUT / "apparent_t_k.npy", out.apparent_t)
    np.save(OUT / "ids.npy", ids)
    np.save(OUT / "dn16.npy", out.dn16)
    Image.fromarray(cm.window_gray(out.apparent_t, 0.0, 70.0)).save(OUT / "frame.png")
    Image.fromarray(cm.window_gray(out.scene_apparent_t, 0.0, 70.0)).save(OUT / "frame_scene.png")
    Image.fromarray(rgb).save(OUT / "frame_rgb.png")

    # Water region: the wetted part, 3 px clear of its edges, central half of its width.
    wet = ids == by_name["CupWet"]
    core = wet.copy()
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            core &= np.roll(np.roll(wet, dy, 0), dx, 1)
    cols = np.nonzero(wet.any(0))[0]
    c0, c1 = cols.min(), cols.max()
    q = (c1 - c0) // 4
    core[:, :c0 + q] = False
    core[:, c1 - q + 1:] = False
    cup_ss = (ids_ss == by_name["CupWet"]) | (ids_ss == by_name["CupDry"])
    bg = ids == by_name["Background"]
    rep = {
        "case": CASE, "sensor_yaml": str(SENSOR), "t_surface_c": T_SURFACE_C[CASE], "t_air_c": cc.T_AIR_C,
        "paper_eps_lwir": 0.93, "camera_position": pos.tolist(), "camera_look": look.tolist(),
        "cup_axis_m": objs[2]["centre"], "board_distance_m": G["board_distance_horizontal_m"],
        "hfov_deg": ctx["hfov_deg"], "vfov_deg": ctx["vfov_deg"], "render_product_w_h": [ctx["shape_ss"][1], ctx["shape_ss"][0]],
        "water_roi_pixels": int(core.sum()),
        "cup_scene_apparent_t_c": round(float(np.median(out.scene_apparent_t[core])) - 273.15, 3),
        "cup_apparent_t_c": round(float(np.median(out.apparent_t[core])) - 273.15, 3),
        "board_scene_apparent_t_c": round(float(np.median(out.scene_apparent_t[bg])) - 273.15, 3),
        "board_apparent_t_c": round(float(np.median(out.apparent_t[bg])) - 273.15, 3),
        "silhouette": silhouette(cup_ss, k),
    }
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)
    ctx["app"].close()


if __name__ == "__main__":
    main()
