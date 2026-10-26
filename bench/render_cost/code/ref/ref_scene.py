"""The reference scene and the reference trajectory, as data. /tmp only; nothing here edits the repo.

One table (`OBJECTS`) is the single source for the stage geometry, the visible colour, the thermal
material and the prescribed temperature of every prim: the scene YAML the repo's loader reads, the
material map and the table printed in the report are all generated from it, so they cannot drift.

Stage: +Z up, metres. The wall is the box of the earlier hall scene, unchanged (centre X = -4.0,
0.3 thick): its front face, the surface the camera actually looks at, is at X = -3.85. The camera
runs along X = -1.0 at Z = 1.0 looking toward -X, 8 degrees down: 2.85 m from that face.

    OBJECTS            name, kind, centre, size, T, thermal material, sRGB
    trajectory()       the 240 poses, exactly as specified (out 0..119, back 120..239 mirrored)
    write_scene()      scene YAML (prescribed temperatures) + material map, into this directory
    install(ns)        point the earlier session's stage/boot module at this scene
    visibility()       per object, the frames whose image its bounding box projects into
    table_markdown()   the replication table
"""
from __future__ import annotations

import math
import pathlib

import numpy as np

REF = pathlib.Path("/tmp/irsim_render_cost/ref")
SCENE_YAML = REF / "scene_ref.yaml"

N_FRAMES = 240
CAM_X, CAM_Z = -1.0, 1.0
Y0, Y_SPAN = -3.0, 6.0
TILT_DEG = -8.0
WALL_FACE_X = -3.85

#: name, kind, centre (m), size (box: x,y,z | cyl: radius,height,axis | sphere: radius), T (C),
#: thermal material (configs/materials/), sRGB for the visible camera, what it is for.
OBJECTS: list[dict] = [
    dict(name="Floor", kind="box", centre=(0.0, 0.0, -0.05), size=(20.0, 20.0, 0.1), t_c=20.0,
         material="concrete", rgb=(0.45, 0.45, 0.44), why="suelo, como en la escena anterior"),
    dict(name="Wall", kind="box", centre=(-4.0, 0.0, 2.0), size=(0.3, 12.0, 4.0), t_c=25.0,
         material="concrete", rgb=(0.62, 0.60, 0.56), why="pared, como en la escena anterior; fondo de todo el recorrido"),
    dict(name="ColdTank", kind="cyl", centre=(-3.55, -2.90, 0.40), size=(0.20, 0.80, "Z"), t_c=5.0,
         material="car_paint_white", rgb=(0.80, 0.86, 0.92), why="frio: deposito refrigerado, contorno curvo"),
    dict(name="WarmCrate", kind="box", centre=(-3.55, -2.05, 0.25), size=(0.50, 0.50, 0.50), t_c=35.0,
         material="abs_plastic_white", rgb=(0.20, 0.35, 0.62), why="intermedio: caja con aristas rectas y cara superior visible"),
    dict(name="HotPlate", kind="box", centre=(-3.83, -1.05, 0.70), size=(0.04, 0.40, 0.40), t_c=90.0,
         material="rusted_steel", rgb=(0.42, 0.24, 0.16), why="caliente: placa en la pared, escalon de 65 K en 4 bordes rectos"),
    dict(name="HotSpot", kind="sphere", centre=(-3.82, -0.45, 0.40), size=(0.03,), t_c=150.0,
         material="rusted_steel", rgb=(0.85, 0.30, 0.10), why="muy caliente y pequeno (25 px): PSF, saturacion, reaccion del AGC"),
    dict(name="Bar_C1", kind="box", centre=(-3.84, 0.00, 0.55), size=(0.02, 0.04, 0.30), t_c=60.0,
         material="abs_plastic_white", rgb=(0.90, 0.80, 0.20), why="barras gruesas: 4 cm, periodo 8 cm (34 px)"),
    dict(name="Bar_C2", kind="box", centre=(-3.84, 0.08, 0.55), size=(0.02, 0.04, 0.30), t_c=60.0,
         material="abs_plastic_white", rgb=(0.90, 0.80, 0.20), why="id."),
    dict(name="Bar_C3", kind="box", centre=(-3.84, 0.16, 0.55), size=(0.02, 0.04, 0.30), t_c=60.0,
         material="abs_plastic_white", rgb=(0.90, 0.80, 0.20), why="id."),
    dict(name="Bar_F1", kind="box", centre=(-3.84, 0.30, 0.55), size=(0.02, 0.01, 0.30), t_c=60.0,
         material="abs_plastic_white", rgb=(0.90, 0.80, 0.20), why="barras finas: 1 cm, periodo 2 cm (8.4 px)"),
    dict(name="Bar_F2", kind="box", centre=(-3.84, 0.32, 0.55), size=(0.02, 0.01, 0.30), t_c=60.0,
         material="abs_plastic_white", rgb=(0.90, 0.80, 0.20), why="id."),
    dict(name="Bar_F3", kind="box", centre=(-3.84, 0.34, 0.55), size=(0.02, 0.01, 0.30), t_c=60.0,
         material="abs_plastic_white", rgb=(0.90, 0.80, 0.20), why="id."),
    dict(name="Bar_F4", kind="box", centre=(-3.84, 0.36, 0.55), size=(0.02, 0.01, 0.30), t_c=60.0,
         material="abs_plastic_white", rgb=(0.90, 0.80, 0.20), why="id."),
    dict(name="HotPipe", kind="cyl", centre=(-3.75, 2.20, 0.95), size=(0.05, 3.20, "Y"), t_c=80.0,
         material="rusted_steel", rgb=(0.55, 0.20, 0.12), why="caliente: tuberia, borde horizontal largo (referencia de tilt)"),
    dict(name="ColdPipe", kind="cyl", centre=(-3.75, 2.20, 0.78), size=(0.05, 3.20, "Y"), t_c=8.0,
         material="car_paint_white", rgb=(0.20, 0.45, 0.80), why="frio: tuberia a 17 cm de la caliente, maximo contraste (72 K)"),
    dict(name="Post", kind="box", centre=(-3.20, 1.50, 0.60), size=(0.08, 0.08, 1.20), t_c=20.0,
         material="abs_plastic_white", rgb=(0.15, 0.15, 0.15), why="poste a 0.65 m de la pared: paralaje, oculta las tuberias al pasar"),
    dict(name="AluPanel", kind="box", centre=(-3.84, 2.55, 0.40), size=(0.02, 0.45, 0.45), t_c=90.0,
         material="bare_aluminium", rgb=(0.78, 0.79, 0.80), why="misma T que HotPlate con eps 0.09: discrimina emisividad"),
    dict(name="WarmBall", kind="sphere", centre=(-3.55, 3.30, 0.20), size=(0.20,), t_c=50.0,
         material="rubber_tyre", rgb=(0.75, 0.20, 0.15), why="intermedio: esfera, cierre del recorrido"),
]


def target_name(prim_name: str) -> str:
    import re

    return re.sub(r"(?<=[a-z])(?=[A-Z])", "_", prim_name).lower()


def materials() -> dict[str, str]:
    return {o["name"]: o["material"] for o in OBJECTS}


def look_direction() -> np.ndarray:
    t = math.radians(TILT_DEG)
    return np.array([-math.cos(t), 0.0, math.sin(t)])


def trajectory(_name: str = "ref", n: int = N_FRAMES) -> tuple[np.ndarray, np.ndarray]:
    """``(positions (240, 3), look directions (240, 3))``, stage axes.

    k <= 119: s = k / 119, Y = -3 + 6 s. k >= 120: the pose of frame 239 - k, taken from the same
    array, so the mirror is bit-exact. Frames 119 and 120 therefore share one pose.
    """
    if n != N_FRAMES:
        raise ValueError("the reference trajectory has exactly 240 frames")
    half = n // 2
    s = np.arange(half, dtype=np.float64) / (half - 1)
    out = np.stack([np.full(half, CAM_X), Y0 + Y_SPAN * s, np.full(half, CAM_Z)], axis=1)
    pos = np.concatenate([out, out[::-1]], axis=0)
    assert all(np.array_equal(pos[k], pos[n - 1 - k]) for k in range(half, n))
    return pos, np.tile(look_direction(), (n, 1))


def write_scene(weather_csv: str = "/tmp/irsim_render_cost/demo/nave_weather_48h.csv",
                environment: str = "/tmp/irsim_render_cost/env_isothermal.yaml") -> pathlib.Path:
    """The scene YAML: one prescribed temperature per prim; enclosure and air at 21 C as before."""
    rows = "\n".join(
        f"    - {{name: {target_name(o['name'])}, solver: prescribed, schedule_s: [0.0, 86400.0], "
        f"schedule_k: [{o['t_c'] + 273.15:.2f}, {o['t_c'] + 273.15:.2f}]}}  # {o['t_c']:g} C" for o in OBJECTS)
    SCENE_YAML.write_text(f"""# Reference scene for the two-simulator comparison. GENERATED by ref_scene.py from OBJECTS.
schema_version: 17
scene:
  name: reference_wall
  description: "Compact wall scene for the 240-frame reference run, prescribed temperatures"
  weather_file: {weather_csv}
  atmosphere_preset: us_standard_clear
  environment_preset: {environment}
  site: {{latitude_deg: 37.5, longitude_deg: 127.0, altitude_m: 50.0}}
  start_utc: "2024-06-21T02:00:00Z"

  targets:
{rows}
""")
    return SCENE_YAML


def install(ns) -> None:
    """Make the earlier session's `nave_stage` author and boot THIS scene (its code, our data)."""
    write_scene()
    ns.OBJECTS = OBJECTS
    ns.SCENE_YAML = SCENE_YAML
    ns.trajectory = trajectory
    ns.target_name = target_name


# -- geometry of the view (no engine): which frames see which object ---------------------------

FOCAL_MM, PITCH_UM, RES = 14.0, 12.0, (640, 512)


def _corners(o) -> np.ndarray:
    c = np.asarray(o["centre"], float)
    if o["kind"] == "box":
        h = 0.5 * np.asarray(o["size"], float)
    elif o["kind"] == "sphere":
        h = np.full(3, o["size"][0])
    else:
        r, length, axis = o["size"]
        h = np.full(3, r)
        h["XYZ".index(axis)] = 0.5 * length
    sg = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    return c + sg * h


def project(points: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stage points -> (u, v) in pixels (pixel-edge convention, origin top-left) and depth."""
    f = look_direction()
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    d = points - pos
    z = d @ f
    fpx = FOCAL_MM * 1000.0 / PITCH_UM
    return RES[0] / 2 + fpx * (d @ right) / z, RES[1] / 2 - fpx * (d @ up) / z, z


def visibility() -> dict[str, dict]:
    """Per object: outbound frames (0..119) in which its bounding box overlaps the image at all,
    and those in which it is wholly inside. Occlusion is ignored. The return leg mirrors it."""
    pos, _ = trajectory()
    out = {}
    for o in OBJECTS[2:]:
        cs = _corners(o)
        part, full = [], []
        for k in range(N_FRAMES // 2):
            u, v, z = project(cs, pos[k])
            if (z <= 0).any():
                continue
            if u.max() > 0 and u.min() < RES[0] and v.max() > 0 and v.min() < RES[1]:
                part.append(k)
                if u.min() >= 0 and u.max() <= RES[0] and v.min() >= 0 and v.max() <= RES[1]:
                    full.append(k)
        out[o["name"]] = {"any": (part[0], part[-1]) if part else None, "whole": (full[0], full[-1]) if full else None,
                          "n_any": len(part)}
    return out


def size_text(o) -> str:
    if o["kind"] == "box":
        return "caja {:g} x {:g} x {:g} (X, Y, Z)".format(*o["size"])
    if o["kind"] == "cyl":
        return "cilindro r {:g}, largo {:g}, eje {}".format(*o["size"])
    return f"esfera r {o['size'][0]:g}"


def table_markdown() -> str:
    lines = ["| Nombre | Tipo y tamaño (m) | Centro X, Y, Z (m) | T (°C) | Material (ε LWIR) |", "|---|---|---|---|---|"]
    eps = {"concrete": 0.92, "car_paint_white": 0.90, "abs_plastic_white": 0.95, "rusted_steel": 0.85,
           "bare_aluminium": 0.09, "rubber_tyre": 0.95}
    for o in OBJECTS:
        lines.append("| {} | {} | {:g}, {:g}, {:g} | {:g} | {} ({:.2f}) |".format(
            o["name"], size_text(o), *o["centre"], o["t_c"], o["material"], eps[o["material"]]))
    return "\n".join(lines)


if __name__ == "__main__":
    pos, look = trajectory()
    print("frames 0, 1, 119, 120, 239:", pos[[0, 1, 119, 120, 239]].tolist())
    print("step m/frame:", float(np.linalg.norm(pos[1] - pos[0])), "look:", look[0].tolist())
    f = look_direction()
    for name, x in (("wall face", WALL_FACE_X), ("X=-3.55", -3.55), ("X=-3.2", -3.2)):
        d = CAM_X - x
        hv = math.atan(RES[1] / 2 * PITCH_UM / (FOCAL_MM * 1000.0))
        hh = math.atan(RES[0] / 2 * PITCH_UM / (FOCAL_MM * 1000.0))
        t = math.radians(TILT_DEG)
        print(f"{name}: Z from {CAM_Z + d * math.tan(t - hv):.3f} to {CAM_Z + d * math.tan(t + hv):.3f}, "
              f"Y half-width on the axis row {d / math.cos(t) * math.tan(hh):.3f}")
    for k, v in visibility().items():
        print(f"{k:10s} any {v['any']}  whole {v['whole']}  ({v['n_any']} frames)")
    covered = np.zeros(N_FRAMES // 2, int)
    for v in visibility().values():
        if v["any"]:
            covered[v["any"][0]:v["any"][1] + 1] += 1
    print("objects in view per outbound frame: min", covered.min(), "max", covered.max(),
          "| frames with none:", np.flatnonzero(covered == 0).tolist())
    print(table_markdown())
