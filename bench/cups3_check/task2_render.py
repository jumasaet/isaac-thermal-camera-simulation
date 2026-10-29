#!/usr/bin/env python
"""Task 2 (render): the three paper cups through the simulator at eight instants of the recording.

    env -u LD_LIBRARY_PATH -u PYTHONPATH -u AMENT_PREFIX_PATH -u ROS_DISTRO -u ROS_VERSION -u ROS_PYTHON_VERSION \
        CUPS_MIN_DISTANCE_M=0.3 /home/jumasaet/isaac_sim/python.sh bench/cups3_check/task2_render.py

Built on bench/cups_check/task3_render_cups.py (same render path: Kit G-buffer, the whole camera on the host in
irsim.pipeline.frame.run_frame; same frustum author, same paper material), changed to this experiment:
  - THREE cups, 7 / 5 / 6 cm frustums of paper (eps 0.93), at the LiDAR ranges and the image bearings of
    out/scene_geometry.json (task0_scene.py); camera 3.29 cm above the table (LiDAR), pitched 0.97 deg up and
    rolled 1.30 deg (fitted to the silhouettes) -- roll composed onto nave_stage's wings-level look-at;
  - each cup in two parts: the wetted wall up to the water line read off the thermal profiles (12 mm under the
    rim -> 4.8 cm) at the probe temperature, and the dry band above at T_air + f (T_water - T_air), f measured
    on the same profiles (cold 0.21, warm 0.96, hot 0.67);
  - temperatures prescribed per target, constant at the mean of the 'before' and 'after' probe readings
    (the readings carry no time; the same as task 1); the frames rendered are 8 of the
    94 real ones, at their own times;
  - board: the LiDAR's (1.008 m, 0.94 m wide, 0.52 m tall, its 6 deg yaw left out -- the room is isothermal, so
    the yaw changes nothing but the edge position); a back wall at 2.5 m; table (plywood) to the board;
    air, board, table and wall at 23.9 C; one WeatherSeries for air and enclosure (CLAUDE.md rule 6);
  - camera: bench/camera_check/cellplus_ctc006_lwir_cups.yaml (30 Hz, calibrated projection, measured blur).

Writes out/sim/: apparent_t_<i>.npy, scene_apparent_t_<i>.npy (K, float32), ids.npy, frame_rgb.png,
frame_<i>.png, scene.yaml, weather_constant.csv, report.json -- per instant and cup the median scene_apparent_t
and apparent_t over the wetted wall (3 px clear of its outline, central half of its width).
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "camera_preview"))
sys.path.insert(0, str(HERE))
import common as cm  # noqa: E402

import numpy as np  # noqa: E402

import cups3_common as c3  # noqa: E402

SENSOR = c3.SENSOR_YAML
OUT = c3.OUT / "sim"
OUT.mkdir(parents=True, exist_ok=True)
GEO = json.loads((c3.OUT / "scene_geometry.json").read_text())
POSE = GEO["image_pose"]
WATER_LINE_M = round(c3.CUP_HEIGHT_M - GEO["water_line_below_rim_mm_used"] / 1000, 4)
DRY_FRACTION = {cup: GEO["water"][cup]["dry_level_fraction_of_wet"] for cup in c3.CUPS}
PER_FRAME = np.load(c3.OUT / "task1_per_frame.npz")
T_UNIX = PER_FRAME["t_unix"]
TR = T_UNIX - T_UNIX[0]
RENDER_IDX = np.linspace(0, len(TR) - 1, 8).round().astype(int)
S0 = 1.0  # scene time of the first real frame; the warm-up frames before it see the 'before' temperatures
PRIM = {"cold": "Cold", "warm": "Warm", "hot": "Hot"}
PAPER = HERE.parent / "cups_check" / "materials" / "paper_cup.yaml"
# Visible (RGB) look only: displayColor per part and the two lights' intensity. Nothing here reaches the thermal
# camera (it reads instance ids, distance and the prescribed temperatures), which task2_render's thermal planes
# confirm bit for bit across a colour change (RESULTADOS.md). nave_stage's lights (dome 600, distant 2500) are sized
# for the hangar bench and wash this small white scene out to one cream; RGB_LIGHT_SCALE brings it into range.
RGB = {"Background": (0.90, 0.90, 0.90), "BackWall": (0.80, 0.80, 0.78), "Table": (0.62, 0.45, 0.28),
       "cold": (0.45, 0.62, 0.90), "warm": (0.92, 0.82, 0.42), "hot": (0.90, 0.45, 0.42)}
RGB_LIGHT_SCALE = 0.3


# -- copied from bench/cups_check/task3_render_cups.py (that script reads sys.argv at import) ---------------


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
    """nave_stage.author_stage knows box / cyl / sphere; author the frustum parts as closed meshes."""
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
            pts = [Gf.Vec3f(float(c3.r_at(z) * math.cos(a)), float(c3.r_at(z) * math.sin(a)), float(z))
                   for z in (z0, z1) for a in ang]
            pts += [Gf.Vec3f(0, 0, float(z0)), Gf.Vec3f(0, 0, float(z1))]
            counts, idx = [], []
            for i in range(n):
                j = (i + 1) % n
                counts += [4, 3, 3]
                idx += [i, j, n + j, n + i]  # side, outward winding
                idx += [2 * n, j, i]  # bottom cap
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


def silhouette(mask_ss: np.ndarray, k: int) -> dict:
    """Top/bottom rows and widths 10 % in from each, in native pixels, off a supersampled mask."""
    rows = np.nonzero(mask_ss.any(1))[0]
    top, bot = rows.min() / k, (rows.max() + 1) / k
    hgt = bot - top

    def width(row_native: float) -> float:
        cols = np.nonzero(mask_ss[int(round(row_native * k))])[0]
        return (cols.max() + 1 - cols.min()) / k

    cols = np.nonzero(mask_ss.any(0))[0]
    return {"top_row": round(float(top), 2), "bottom_row": round(float(bot), 2), "height_px": round(float(hgt), 2),
            "width_top10_px": round(float(width(top + 0.1 * hgt)), 2),
            "width_bottom10_px": round(float(width(bot - 0.1 * hgt)), 2),
            "centre_col": round(float((cols.min() + cols.max() + 1) / 2 / k), 2)}


# -- this scene ---------------------------------------------------------------------------------------------


def objects() -> tuple[list[dict], dict]:
    b = GEO["lidar"]["board"]
    d_board = b["distance_m"]
    lat_c = 0.5 * (b["left_edge_m"] + b["right_edge_m"])
    width = b["left_edge_m"] - b["right_edge_m"]
    objs = [
        dict(name="BackWall", kind="box", centre=(-2.5, 0.0, 0.9), size=(0.02, 6.0, 2.4), rgb=RGB["BackWall"]),
        dict(name="Background", kind="box", centre=(-d_board - 0.01, -lat_c, b["top_m"] / 2), size=(0.02, width, b["top_m"]),
             rgb=RGB["Background"]),
        dict(name="Table", kind="box", centre=((0.3 - d_board) / 2, 0.0, -0.01), size=(0.3 + d_board, 1.8, 0.02),
             rgb=RGB["Table"]),
    ]
    thermal = {"BackWall": ("concrete_white", None), "Background": ("concrete_white", None), "Table": ("plywood", None)}
    for cup in c3.CUPS:
        g = GEO["lidar"]["cups"][cup]
        centre = (-g["axis_forward_m"], -POSE["axis_left_m"][cup], 0.0)  # stage: -X forward, +Y right
        objs.append(dict(name=f"{PRIM[cup]}Wet", kind="frustum", centre=centre, z=(0.0, WATER_LINE_M), rgb=RGB[cup]))
        objs.append(dict(name=f"{PRIM[cup]}Dry", kind="frustum", centre=centre, z=(WATER_LINE_M, c3.CUP_HEIGHT_M),
                         rgb=RGB[cup]))
        thermal[f"{PRIM[cup]}Wet"] = ("paper_cup", cup)
        thermal[f"{PRIM[cup]}Dry"] = ("paper_cup", cup + ":dry")
    return objs, thermal


def schedule_c(tag) -> tuple[float, float]:
    """(T at the first frame, T at the last frame), C: equal, the mean of the two probe readings (their times
    are unknown, so there is no ramp)."""
    if tag is None:
        return c3.T_AIR_C, c3.T_AIR_C
    cup, _, part = tag.partition(":")
    a = b = c3.T_MEAN_C[cup]
    if part == "dry":
        f = DRY_FRACTION[cup]
        return c3.T_AIR_C + f * (a - c3.T_AIR_C), c3.T_AIR_C + f * (b - c3.T_AIR_C)
    return a, b


def write_weather() -> pathlib.Path:
    t = c3.T_AIR_C + 273.15
    rows = [f"2024-06-{d}T{h:02d}:00:00Z,{t:.2f},0.5,0.1,1.0,0.0,0.0,23000.0,0.0" for d in (21, 22) for h in range(24)]
    path = OUT / "weather_constant.csv"
    path.write_text("# irsim weather v1\n# SYNTHETIC closed room: constant air temperature, fully covered, no sun "
                    "(GENERATED by bench/cups3_check/task2_render.py)\n"
                    "time_utc,t_air_k,rh_fraction,wind_speed_m_s,cloud_fraction,dni_w_m2,dhi_w_m2,visibility_m,precip_mm_h\n"
                    + "\n".join(rows) + "\n")
    return path


def write_scene(thermal: dict, weather: pathlib.Path) -> pathlib.Path:
    s1 = S0 + float(TR[-1])
    rows = []
    for n, (m, tag) in thermal.items():
        a, b = schedule_c(tag)
        # flat before the first frame (warm-up) and after the last: a prescribed schedule does not extrapolate
        rows.append(f"    - {{name: {cm.ns.target_name(n)}, solver: prescribed, schedule_s: [0.0, {S0}, {s1:.3f}, {s1 + 60:.3f}], "
                    f"schedule_k: [{a + 273.15:.3f}, {a + 273.15:.3f}, {b + 273.15:.3f}, {b + 273.15:.3f}]}}"
                    f"  # {a:.2f} -> {b:.2f} C, {m}")
    path = OUT / "scene.yaml"
    path.write_text(f"""# Three paper cups, bench/cups3_check. GENERATED by bench/cups3_check/task2_render.py.
schema_version: 17
scene:
  name: cups3
  description: "Three paper cups before a flat matte board, prescribed temperatures, isothermal room at {c3.T_AIR_C:g} C"
  weather_file: {weather}
  atmosphere_preset: us_standard_clear
  environment_preset: {cm.ENV_ISOTHERMAL}
  site: {{latitude_deg: 37.5, longitude_deg: 127.0, altitude_m: 50.0}}
  start_utc: "2024-06-21T02:00:00Z"

  targets:
""" + "\n".join(rows) + "\n")
    return path


def patch_roll(roll_deg: float) -> None:
    """Compose a roll about the camera's optical axis onto nave_stage's wings-level look-at.

    USD cameras look down local -Z; a positive rotation about local +Z turns the camera's right toward its up
    (counter-clockwise seen from behind), the opposite of task0_scene's positive roll -> rotate by -roll."""
    ops = cm.ns.CameraOps
    orig = ops.set_pose

    def set_pose(self, position, look):
        from pxr import Gf

        orig(self, position, look)
        q = self._q.Get()
        qd = Gf.Quatd(float(q.GetReal()), Gf.Vec3d(*(float(v) for v in q.GetImaginary())))
        qr = Gf.Rotation(Gf.Vec3d(0, 0, 1), -roll_deg).GetQuat()
        qn = qd * qr
        self._q.Set(Gf.Quatf(float(qn.GetReal()), Gf.Vec3f(*(float(v) for v in qn.GetImaginary()))))

    ops.set_pose = set_pose


def wet_core(ids: np.ndarray, pid: int) -> np.ndarray:
    wet = ids == pid
    core = wet.copy()
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            core &= np.roll(np.roll(wet, dy, 0), dx, 1)
    cols = np.nonzero(wet.any(0))[0]
    c0, c1 = cols.min(), cols.max()
    q = (c1 - c0) // 4
    core[:, :c0 + q] = False
    core[:, c1 - q + 1:] = False
    return core


def main() -> None:
    patch_library()
    patch_stage()
    patch_roll(POSE["camera_roll_deg"])
    cm.ns.DOME_INTENSITY *= RGB_LIGHT_SCALE
    cm.ns.DISTANT_INTENSITY *= RGB_LIGHT_SCALE
    objs, thermal = objects()
    scene_yaml = write_scene(thermal, write_weather())
    th = math.radians(POSE["camera_pitch_up_deg"])
    pos = np.array([0.0, 0.0, POSE["camera_height_above_table_m"]])
    look = np.array([-math.cos(th), 0.0, math.sin(th)])
    ctx = cm.boot(sensor_yaml=SENSOR, scene_yaml=scene_yaml, objects=objs,
                  materials={n: m for n, (m, _t) in thermal.items()}, first_pose=(pos, look))
    from PIL import Image

    from irsim_isaac.pipeline.material_ids import labels_from_payload, labels_to_paths

    min_d = float(os.environ.get("CUPS_MIN_DISTANCE_M", "0")) or None
    clock = cm.Clock(ctx, min_distance_m=min_d)
    for _ in range(4):
        clock.frame(cm.render_sync(ctx, pos, look))
    k = ctx["k"]
    h, w = ctx["shape"]
    per = []
    ids = ids_ss = by_name = None
    for j, i in enumerate(RENDER_IDX):
        clock.rel = S0 + float(TR[i])
        payload = cm.render_sync(ctx, pos, look)
        out, rgb, ids_ss = clock.frame(payload)
        assert out.scene_apparent_t is not None and out.scene_apparent_t.dtype == np.float32
        assert out.apparent_t.dtype == np.float32
        if ids is None:
            ids = ids_ss[k // 2::k, k // 2::k][:h, :w]
            paths = labels_to_paths(labels_from_payload(payload["instance"]))
            by_name = {p.rsplit("/", 1)[-1]: q for q, p in paths.items()}
            np.save(OUT / "ids.npy", ids)
            Image.fromarray(rgb).save(OUT / "frame_rgb.png")
        np.save(OUT / f"apparent_t_{j}.npy", out.apparent_t)
        np.save(OUT / f"scene_apparent_t_{j}.npy", out.scene_apparent_t)
        Image.fromarray(cm.window_gray(out.apparent_t, 0.0, 70.0)).save(OUT / f"frame_{j}.png")
        cups = {}
        for cup in c3.CUPS:
            core = wet_core(ids, by_name[f"{PRIM[cup]}Wet"])
            a, b = schedule_c(cup)
            cups[cup] = {"t_surface_c": round(a + (b - a) * float(TR[i] / TR[-1]), 3), "roi_pixels": int(core.sum()),
                         "scene_apparent_t_c": round(float(np.median(out.scene_apparent_t[core])) - 273.15, 3),
                         "apparent_t_c": round(float(np.median(out.apparent_t[core])) - 273.15, 3)}
        bg = ids == by_name["Background"]
        per.append({"render": j, "real_frame_index": int(i), "t_since_first_s": round(float(TR[i]), 3), "cups": cups,
                    "board_scene_apparent_t_c": round(float(np.median(out.scene_apparent_t[bg])) - 273.15, 3),
                    "board_apparent_t_c": round(float(np.median(out.apparent_t[bg])) - 273.15, 3)})
        print(json.dumps(per[-1]), flush=True)
    sil = {cup: silhouette((ids_ss == by_name[f"{PRIM[cup]}Wet"]) | (ids_ss == by_name[f"{PRIM[cup]}Dry"]), k)
           for cup in c3.CUPS}
    rep = {"sensor_yaml": str(SENSOR), "frame_rate_hz": round(1.0 / ctx["camera"].frame_period_s, 3), "t_air_c": c3.T_AIR_C,
           "paper_eps_lwir": 0.93, "water_line_m": WATER_LINE_M, "dry_fraction": DRY_FRACTION,
           "camera_position": pos.tolist(), "camera_look": look.tolist(), "camera_roll_deg": POSE["camera_roll_deg"],
           "hfov_deg": ctx["hfov_deg"], "vfov_deg": ctx["vfov_deg"],
           "render_product_w_h": [ctx["shape_ss"][1], ctx["shape_ss"][0]], "defocus_min_distance_m": min_d,
           "objects": objs, "renders": per, "silhouette": sil}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"silhouette": sil}, indent=1), flush=True)
    ctx["app"].close()


if __name__ == "__main__":
    main()
