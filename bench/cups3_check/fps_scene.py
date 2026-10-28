"""The three-cup scene as a drop-in for the GPU bench's `ref_scene` (bench/render_cost/code/rebase_2026-10-22/ref).

Read by fps_run.py, which puts this module in sys.modules["ref_scene"] before it runs that bench's run_ref.py, so
the bench's loop, device camera and timers run unchanged on this scene. Everything here comes from task2_render.py's
render (out/sim/report.json -> objects, colours, camera pose; out/sim/scene.yaml -> prescribed temperatures, the
same one WeatherSeries), so the two cannot drift:

    OBJECTS / materials()   the nine prims and their thermal materials (paper_cup added to the library at run time)
    trajectory()            240 poses: the task-2 pose at frame 0, then a +-1 cm lateral sway (sin, one period),
                            so the renderer never sees a static camera; frame 0 is the pose of the comparison images
    install(ns)             points nave_stage at this scene: frustum author, camera roll, the RGB light scale
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SIM = HERE / "out" / "sim"
REPORT = json.loads((SIM / "report.json").read_text())
SCENE_YAML = SIM / "scene.yaml"
PAPER = HERE.parent / "cups_check" / "materials" / "paper_cup.yaml"
RGB_LIGHT_SCALE = 0.3  # task2_render.RGB_LIGHT_SCALE
N_FRAMES = 240
SWAY_M = 0.01

OBJECTS: list[dict] = [dict(o, centre=tuple(o["centre"]), size=tuple(o["size"]) if "size" in o else None)
                       for o in REPORT["objects"]]
_MAT = {"BackWall": "concrete_white", "Background": "concrete_white", "Table": "plywood"}


def materials() -> dict[str, str]:
    return {o["name"]: _MAT.get(o["name"], "paper_cup") for o in OBJECTS}


def target_name(prim_name: str) -> str:
    import re

    return re.sub(r"(?<=[a-z])(?=[A-Z])", "_", prim_name).lower()


def trajectory(_name: str = "ref", n: int = N_FRAMES) -> tuple[np.ndarray, np.ndarray]:
    pos0 = np.asarray(REPORT["camera_position"], float)
    look = np.asarray(REPORT["camera_look"], float)
    k = np.arange(n)
    pos = np.tile(pos0, (n, 1))
    pos[:, 1] += SWAY_M * np.sin(2 * np.pi * k / n)
    return pos, np.tile(look, (n, 1))


def _r_at(z: float) -> float:
    return 0.025 + 0.010 * z / 0.060  # 7 / 5 / 6 cm frustum (cups3_common.r_at)


def _patch_library() -> None:
    from irsim.materials import library as lib

    orig = lib.MaterialLibrary.load.__func__

    def load(cls, material_dir=None, data_dir=None):
        mats = dict(orig(cls, material_dir, data_dir).items())
        mats["paper_cup"] = lib.load_material(PAPER, data_dir)
        return cls(mats)

    lib.MaterialLibrary.load = classmethod(load)


def install(ns) -> None:
    """Make nave_stage author and boot THIS scene (its code, our data); task2_render.py's frustums and roll."""
    ns.OBJECTS = OBJECTS
    ns.SCENE_YAML = SCENE_YAML
    ns.trajectory = trajectory
    ns.target_name = target_name
    ns.DOME_INTENSITY *= RGB_LIGHT_SCALE
    ns.DISTANT_INTENSITY *= RGB_LIGHT_SCALE
    _patch_library()
    orig_author = ns.author_stage

    def author_stage(mats, lights=True):
        all_objs = ns.OBJECTS
        ns.OBJECTS = [o for o in all_objs if o["kind"] != "frustum"]
        stage, p2t = orig_author(mats, lights=lights)
        ns.OBJECTS = all_objs
        from pxr import Gf, Sdf, UsdGeom, Vt

        for o in all_objs:
            if o["kind"] != "frustum":
                continue
            path = f"/World/{o['name']}"
            z0, z1 = o["z"]
            n = 192
            ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
            pts = [Gf.Vec3f(float(_r_at(z) * math.cos(a)), float(_r_at(z) * math.sin(a)), float(z)) for z in (z0, z1) for a in ang]
            pts += [Gf.Vec3f(0, 0, float(z0)), Gf.Vec3f(0, 0, float(z1))]
            counts, idx = [], []
            for i in range(n):
                j = (i + 1) % n
                counts += [4, 3, 3]
                idx += [i, j, n + j, n + i, 2 * n, j, i, 2 * n + 1, n + i, n + j]
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
            mesh.GetPrim().CreateAttribute("thermal:material", Sdf.ValueTypeNames.String).Set(mats[o["name"]])
            p2t[path] = ns.target_name(o["name"])
        return stage, p2t

    ns.author_stage = author_stage
    roll = float(REPORT["camera_roll_deg"])
    orig_pose = ns.CameraOps.set_pose

    def set_pose(self, position, look):
        from pxr import Gf

        orig_pose(self, position, look)
        q = self._q.Get()
        qd = Gf.Quatd(float(q.GetReal()), Gf.Vec3d(*(float(v) for v in q.GetImaginary())))
        qn = qd * Gf.Rotation(Gf.Vec3d(0, 0, 1), -roll).GetQuat()  # task2_render.patch_roll
        self._q.Set(Gf.Quatf(float(qn.GetReal()), Gf.Vec3f(*(float(v) for v in qn.GetImaginary()))))

    ns.CameraOps.set_pose = set_pose
