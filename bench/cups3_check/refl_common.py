"""Shared by the task-8 scripts: the three cups' table reflection, added to the simulated scene WITHOUT Kit.

The simulator reflects only a uniform environment (docs/physics-model.md §4.3; ADR 0045, 0067), and the real
table mirrors the cups at grazing incidence (task4_reflection.py: rho ~0.4 next to the base). This module is the
bench-side prototype of the specular ray -- nothing in src/ changes:

  * replay: the G-buffer task8_gbuffer.py dumped once from Kit goes through the same camera chain task2_render.py
    runs (PipelineConfig.from_sensor + attach_sensor_chain, built as bench/camera_preview/common.boot builds them),
    at the same instants. The camera is still and every temperature constant, so one G-buffer stands for all eight.
  * table model, docs/physics-model.md §4.1-§4.3 (the split is this bench's, not ADR 0067's):

        L_table = eps L_B(T) + rho L_env + rho_s(theta) (L_mirror - L_env)
        rho_s(theta) = <R_F(theta; n) exp(-(4 pi sigma_h cos(theta) / lambda)^2)>_band

    R_F is the unpolarised Fresnel reflectance of a smooth dielectric (irsim.materials.fresnel), the exponential
    the coherent (mirror) share of a surface of RMS height sigma_h (Davies 1954, the Rayleigh criterion that §4.3
    quotes as sigma << lambda/8, with its cos(theta)), and <.>_band the average over the camera's response weighted
    by dB/dT at 300 K. The rest of the reflectance stays diffuse and sees L_env; in this isothermal room that part
    and the table's own emission are both L_B(T_air), so only the mirror term carries contrast.
  * injection without touching src/: stage 1 computes eps L_B(T) + rho L_env from temperature_k, so a table pixel
    gets the EQUIVALENT temperature T' with eps L_B(T') + rho L_env = L_table above (exact in band radiance, through
    the band LUT's inverse). The camera chain after stage 1 -- atmosphere, blur, vignetting, detector, noise, ISP --
    is untouched, which is the point: the reflection reaches the image through the simulated lens.
  * L_mirror, two ways: `mirror_image_space` (step 0: the scene flipped about each column's table edge, an estimate
    only) and `mirror_raycast` (step 3: the reflected ray of every table pixel against the scene's analytic
    primitives -- the cups as cones, board and back wall as boxes).
  * measurement: task4_reflection.rho_profile, the definition the real reflection was measured with.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("CUPS3_SIM_DIR", "sim")
import task2_render as t2  # noqa: E402  (Kit-free at import: Kit is only booted in its main())

c3, cm = t2.c3, t2.cm
GB = c3.OUT / "sim" / "gbuffer.npz"
REFL = c3.OUT / "refl"
MIN_DISTANCE_M = 0.3  # the defocus workaround run_all.sh applies (CUPS_MIN_DISTANCE_M); re-applied at replay
N_TABLE = 1.5  # varnish; at 80-85 deg R_F moves 0.60 -> 0.62 for n 1.4 -> 1.6, and the data cannot pin it
SIGMA_H_PRELIM_UM = 5.3  # the profile-only fit (no lens blur) of the first proposal; step 0 uses it
#: Kit's DistanceToCameraSD is measured from the camera's NEAR CLIPPING PLANE, not its centre: against the
#: board plane (x = -1.0076 m) the dumped distance is short by 50.0 mm at every angle off axis (50.00 on axis,
#: 50.79 at cos 0.8-0.9 -- a constant, not 0.05/cos), and nave_stage sets clippingRange (0.05, 1e4). The geometry
#: here adds it back. The camera chain is replayed with the distance as Kit gave it, so that the replay stays
#: bit-identical to task2_render (the 5 cm also reaches the bench's defocus and atmosphere: a separate issue).
NEAR_CLIP_M = 0.05


# -- the G-buffer and the camera chain ---------------------------------------------------------------------


def load_gbuffer() -> dict:
    g = dict(np.load(GB))
    assert g["temperature_k"].dtype == np.float32 and g["distance_m"].dtype == np.float32
    g["by_name"] = {str(n): int(q) for n, q in zip(g["names"], g["name_ids"], strict=True)}
    return g


def build_chain(noise: bool | None = None, seed: int = 0):
    """(cfg, lut, quantity, t0_s, response) as bench/camera_preview/common.boot builds them, without Kit.

    noise None = the YAML's fidelity setting and seed 0, exactly what boot passes (replay is then bit-identical
    to Kit's out/sim frames, check_replay); False = the ideal chain, for fitting."""
    from irsim.atmosphere.skylight import skylight_for_sensor
    from irsim.config.loader import load_sensor_config
    from irsim.config.scene import load_scene_config
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.table import MaterialTable
    from irsim.pipeline.core import PipelineConfig
    from irsim.pipeline.sensor_chain import attach_sensor_chain
    from irsim.radiometry.lut_files import load_band_lut_for_config, load_band_response_for_config
    from irsim.scene import Scene

    t2.patch_library()
    sensor = load_sensor_config(t2.SENSOR)
    spec = sensor.sensor
    band, quantity = spec.band.band_id, spec.quantity
    lut = load_band_lut_for_config(sensor, cm.LUT_DIR)
    response = load_band_response_for_config(sensor, c3.REPO / "data")
    scene = Scene.from_config(
        load_scene_config(c3.OUT / "sim" / "scene.yaml"),
        {band: lut},
        quantity=quantity,
        skylights={band: skylight_for_sensor(sensor, quantity)},
        responses={band: response},
    )
    table = MaterialTable.from_library(MaterialLibrary.load(), band)
    cfg = PipelineConfig.from_sensor(
        sensor,
        table,
        lut,
        sky=scene.sky_models[band],
        atmosphere=scene.layered,
        flat_field_enabled=True,
        noise_enabled=noise,
        sensor_seed=seed,
    )
    cfg = attach_sensor_chain(cfg, scene.weather, t0_s=scene.t0_s)
    return cfg, lut, quantity, scene.t0_s, response


def planes_for(g: dict, temperature_k: np.ndarray) -> dict:
    """The planes Clock.frame hands run_frame, with the 0.3 m distance floor of run_all.sh."""
    sky = g["sky_mask"]
    dist = np.where(sky, 0.0, np.maximum(g["distance_m"], np.float32(MIN_DISTANCE_M))).astype(np.float32)
    assert temperature_k.dtype == np.float32
    return {
        "temperature_k": temperature_k,
        "material_id": g["material_id"],
        "sky_mask": sky,
        "distance_m": dist,
        "sky_view_factor": np.full(sky.shape, 0.5, np.float32),
    }


def replay(planes: dict, cfg, t0_s: float, all_instants: bool = True) -> list:
    """The task2_render sequence: 4 warm-up frames one period apart from t0, then the 8 instants (or only the
    first). Returns the Outputs of the rendered instants."""
    from irsim.pipeline.core import PipelineState
    from irsim.pipeline.frame import run_frame

    period = 1.0 / cfg.sensor.sensor.fpa.frame_rate_hz
    state = PipelineState(t_s=t0_s)
    for i in range(4):
        state.t_s = t0_s + i * period
        run_frame(planes, cfg, state)
    outs = []
    idx = t2.RENDER_IDX if all_instants else t2.RENDER_IDX[:1]
    for i in idx:
        state.t_s = t0_s + t2.S0 + float(t2.TR[i])
        out = run_frame(planes, cfg, state)
        assert out.apparent_t.dtype == np.float32
        outs.append(out)
    return outs


# -- radiometry of the scene ---------------------------------------------------------------------------------


def l_env(cfg, lut, quantity, t_s: float) -> float:
    from irsim.pipeline.environment import environment_radiance

    return float(environment_radiance(cfg.sky, lut, t_s, np.array(0.5), quantity))


def scene_radiance(g: dict, temperature_k: np.ndarray, cfg, lut, quantity, le: float) -> np.ndarray:
    """eps L_B(T) + rho L_env per pixel: stage 1 for an opaque scene (float64, for the arithmetic only)."""
    eps, rho, _tau = cfg.materials.properties_for(g["material_id"], g["sky_mask"])
    return eps * lut.lookup(temperature_k, quantity).astype(np.float64) + rho * le


def object_radiance(g: dict, cfg, lut, quantity, le: float) -> dict:
    """Prim name -> its stage-1 radiance (each object is one material at one temperature)."""
    out = {}
    for name, q in g["by_name"].items():
        m = g["ids"] == q
        if not m.any():
            continue
        mid = np.unique(g["material_id"][m])
        assert len(mid) == 1, (name, mid)
        t = np.float32(np.median(g["temperature_k"][m]))
        eps, rho, _tau = cfg.materials.properties_for(mid[:1], np.zeros(1, bool))
        out[name] = float(eps[0] * lut.lookup(np.array([t]), quantity)[0] + rho[0] * le)
    return out


def rho_specular(
    cos_theta: np.ndarray, sigma_h_um: float, response, n: float = N_TABLE, t_weight_k: float = 300.0
) -> np.ndarray:
    """rho_s(theta) = <R_F(theta) exp(-(4 pi sigma_h cos theta / lambda)^2)>_band, weights R(lambda) dB/dT."""
    from irsim.materials.fresnel import fresnel_reflectance
    from irsim.radiometry.planck import d_spectral_radiance_dT

    lam = np.linspace(*response.support_um, 241)
    w = response.resampled(lam) * d_spectral_radiance_dT(lam, np.full_like(lam, t_weight_k))
    w = w / w.sum()
    c = np.clip(np.asarray(cos_theta, np.float64), 0.0, 1.0)
    rf = fresnel_reflectance(n, 0.0, c)[0]
    g = np.exp(-((4 * np.pi * sigma_h_um * c[..., None] / lam) ** 2)) @ w
    return rf * g


def inject(g: dict, cfg, lut, quantity, le: float, add: np.ndarray) -> np.ndarray:
    """temperature_k with every TABLE pixel at the equivalent temperature of (its radiance + add)."""
    table = g["ids"] == g["by_name"]["Table"]
    eps, rho, _tau = cfg.materials.properties_for(g["material_id"], g["sky_mask"])
    l_base = scene_radiance(g, g["temperature_k"], cfg, lut, quantity, le)
    l_new = l_base + np.where(table, add, 0.0)
    lb = (l_new - rho * le) / np.maximum(eps, 1e-6)
    t = g["temperature_k"].copy()
    t[table] = lut.apparent_temperature(lb[table].astype(np.float32), quantity)
    assert t.dtype == np.float32
    return t


# -- geometry -------------------------------------------------------------------------------------------------


def camera_rays(g: dict) -> tuple[np.ndarray, np.ndarray]:
    """(origin (3,), unit world directions (H, W, 3)) of the render product's pixel centres.

    The render carries the calibrated Brown-Conrady distortion (IrCamera writes it into the USD lens schema), so a
    supersampled pixel is mapped to native coordinates and undistorted with the same calibration
    (bench/cups_check/cups_common.undistort_points); the camera axes come from camera_params' view transform.
    check_geometry() holds the result against the scene: table points at z = 0, cup points on their cones."""
    hs, ws = g["distance_m"].shape
    k = hs // 480
    c2w = np.linalg.inv(g["view"])
    right, up, fwd = c2w[0, :3], c2w[1, :3], -c2w[2, :3]
    u = (np.arange(ws) + 0.5) / k - 0.5
    v = (np.arange(hs) + 0.5) / k - 0.5
    uu, vv = np.meshgrid(u, v)
    x, y = c3.undistort_points(uu.ravel(), vv.ravel())
    d = fwd[None] + x[:, None] * right[None] - y[:, None] * up[None]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return c2w[3, :3].copy(), d.reshape(hs, ws, 3)


def primitives() -> tuple[list[dict], list[dict]]:
    """(cones, boxes) of the scene task2_render authored, the table left out (it is the mirror)."""
    objs, _thermal = t2.objects()
    cones, boxes = {}, []
    for o in objs:
        if o["kind"] == "frustum":
            base = o["name"].removesuffix("Wet").removesuffix("Dry")
            cone = cones.setdefault(base, {"centre": np.array(o["centre"], float), "parts": []})
            cone["parts"].append((o["z"][0], o["z"][1], o["name"]))
        elif o["name"] != "Table":
            c, s = np.array(o["centre"], float), np.array(o["size"], float)
            boxes.append({"name": o["name"], "lo": c - s / 2, "hi": c + s / 2})
    return list(cones.values()), boxes


def _hit_cone(p, d, cone) -> tuple[np.ndarray, np.ndarray]:
    """First t > 0 on the cup's side wall (r = r0 + s z, z in [0, H]); (t, part index) or inf / -1."""
    r0 = c3.CUP_BOTTOM_DIAMETER_M / 2
    s = (c3.CUP_TOP_DIAMETER_M - c3.CUP_BOTTOM_DIAMETER_M) / 2 / c3.CUP_HEIGHT_M
    q = p - cone["centre"]
    a = d[:, 0] ** 2 + d[:, 1] ** 2 - (s * d[:, 2]) ** 2
    rq = r0 + s * q[:, 2]
    b = 2 * (q[:, 0] * d[:, 0] + q[:, 1] * d[:, 1] - s * d[:, 2] * rq)
    c = q[:, 0] ** 2 + q[:, 1] ** 2 - rq**2
    disc = b * b - 4 * a * c
    ok = disc >= 0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t_best = np.full(len(p), np.inf)
    for t in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)):
        z = q[:, 2] + t * d[:, 2]
        good = ok & (t > 1e-6) & (z >= 0) & (z <= c3.CUP_HEIGHT_M) & (t < t_best)
        t_best = np.where(good, t, t_best)
    z = q[:, 2] + t_best * d[:, 2]
    part = np.full(len(p), -1)
    for i, (z0, z1, _n) in enumerate(cone["parts"]):
        part = np.where(np.isfinite(t_best) & (z >= z0) & (z <= z1 + 1e-9), i, part)
    return t_best, part


def _hit_box(p, d, box) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (box["lo"] - p) / d
        t1 = (box["hi"] - p) / d
    near = np.nanmax(np.minimum(t0, t1), axis=1)
    far = np.nanmin(np.maximum(t0, t1), axis=1)
    hit = far >= np.maximum(near, 1e-6)
    return np.where(hit, np.maximum(near, 1e-6), np.inf)


def first_hit(p: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(t, name) of the first primitive along each ray; name '' where nothing is hit."""
    cones, boxes = primitives()
    t_best = np.full(len(p), np.inf)
    name = np.full(len(p), "", dtype=object)
    for cone in cones:
        t, part = _hit_cone(p, d, cone)
        better = t < t_best
        t_best = np.where(better, t, t_best)
        for i, (_z0, _z1, n) in enumerate(cone["parts"]):
            name[better & (part == i)] = n
    for box in boxes:
        t = _hit_box(p, d, box)
        better = t < t_best
        t_best = np.where(better, t, t_best)
        name[better] = box["name"]
    return t_best, name


def table_geometry(g: dict) -> dict:
    """Per table pixel: world point, incoming direction, cos(incidence) (the table normal is +Z)."""
    o, rays = camera_rays(g)
    table = g["ids"] == g["by_name"]["Table"]
    d = rays[table]
    p = o[None] + (g["distance_m"][table][:, None].astype(np.float64) + NEAR_CLIP_M) * d
    return {"mask": table, "p": p, "d": d, "cos": np.abs(d[:, 2]), "origin": o, "rays": rays}


def check_geometry(g: dict) -> dict:
    """The reconstructed rays against the authored scene: table z, and each cup point's distance to its cone."""
    o, rays = camera_rays(g)
    tg = table_geometry(g)
    out = {"table_z_mm_p50_p99_abs": [round(float(np.percentile(np.abs(tg["p"][:, 2]), q)) * 1e3, 3) for q in (50, 99)]}
    cones, _ = primitives()
    r0 = c3.CUP_BOTTOM_DIAMETER_M / 2
    s = (c3.CUP_TOP_DIAMETER_M - c3.CUP_BOTTOM_DIAMETER_M) / 2 / c3.CUP_HEIGHT_M
    for cone in cones:
        for _z0, _z1, n in cone["parts"]:
            m = g["ids"] == g["by_name"][n]
            p = o[None] + (g["distance_m"][m][:, None].astype(np.float64) + NEAR_CLIP_M) * rays[m]
            q = p - cone["centre"]
            res = np.hypot(q[:, 0], q[:, 1]) - (r0 + s * q[:, 2])
            out[f"{n}_radial_mm_p50_p99_abs"] = [round(float(np.percentile(np.abs(res), v)) * 1e3, 3) for v in (50, 99)]
    # and the forward model of the primary ray: the analytic first hit of each camera ray against the depth plane
    sub = (slice(None, None, 16), slice(None, None, 16))
    r = rays[sub].reshape(-1, 3)
    t, _ = first_hit(np.repeat(o[None], len(r), 0), r)
    dd = g["distance_m"][sub].ravel().astype(np.float64) + NEAR_CLIP_M
    tab = (g["ids"][sub] == g["by_name"]["Table"]).ravel()
    t_tab = -o[2] / np.where(r[:, 2] < 0, r[:, 2], -1e-12)  # the plane z = 0
    t_all = np.where(tab, t_tab, t)
    ok = np.isfinite(t_all) & (dd > 0)
    out["primary_ray_depth_mm_p50_p99_abs"] = [
        round(float(np.percentile(np.abs(t_all[ok] - dd[ok]), q)) * 1e3, 3) for q in (50, 99)
    ]
    return out


def mirror_raycast(g: dict, tg: dict, obj_l: dict, le: float) -> tuple[np.ndarray, dict]:
    """L_mirror per table pixel: the radiance of the first object along the reflected ray (L_env if none)."""
    d = tg["d"] * np.array([1.0, 1.0, -1.0])
    _t, name = first_hit(tg["p"] + 1e-7 * d, d)
    lm = np.array([obj_l.get(n, le) if n else le for n in name], np.float64)
    counts = {str(n) or "(none)": int(c) for n, c in zip(*np.unique(name.astype(str), return_counts=True), strict=True)}
    return lm, counts


def mirror_image_space(g: dict, l_scene: np.ndarray) -> np.ndarray:
    """Step 0 only: L_mirror(r, c) = L_scene(2 e(c) - r, c), e(c) = the first table row of column c."""
    table = g["ids"] == g["by_name"]["Table"]
    hs, _ws = table.shape
    first = np.where(table.any(0), table.argmax(0), hs)
    rows = np.arange(hs)[:, None]
    src = np.clip(2 * first[None, :] - 1 - rows, 0, hs - 1)
    lm = np.take_along_axis(l_scene, src, axis=0)
    return lm[table]


# -- measurement ----------------------------------------------------------------------------------------------


def real_profiles() -> dict:
    d = json.loads((c3.OUT / "task4_reflection.json").read_text())["real"]
    return {
        cup: {"d": np.arange(v["d_range_rows"][0], v["d_range_rows"][1] + 1), "rho": np.array(v["profile_rho"])}
        for cup, v in d.items()
    }


def sim_profiles(apparent_t: list[np.ndarray]) -> dict:
    """task4_reflection.rho_profile on the mean simulated frame (band radiance), as task4 measures the sim."""
    import task4_reflection as t4

    img = np.mean([c3.band_l(np.asarray(a, np.float64)) for a in apparent_t], axis=0)
    reg = c3.find_cups(img)
    sil = json.loads((c3.OUT / "sim" / "report.json").read_text())["silhouette"]
    coef = c3.board_plane(img[None], c3.board_mask(reg))
    out = {}
    for cup in c3.CUPS:
        p = t4.rho_profile(img, lambda y, x: float(c3.plane_at(coef, y, x)[0]), reg[cup], sil[cup], False)
        out[cup] = {"d": p["d"], "rho": p["rho"], "rho_near": p["rho_near"], "rho_mirror": p["rho_mirror"]}
    return out


#: d rows used to compare profiles. The hot cup stops at 45: beyond it the real profile rises again (the second
#: lobe, task 4), which no model here contains; fixed before any fit was run.
D_MAX_FIT = {"cold": 54, "warm": 64, "hot": 45}


def profile_rms(sim: dict, real: dict, cup: str) -> float:
    dr, rr = real[cup]["d"], real[cup]["rho"]
    ds, rs = sim[cup]["d"], sim[cup]["rho"]
    common = np.intersect1d(dr[dr <= D_MAX_FIT[cup]], ds)
    a = rr[np.searchsorted(dr, common)]
    b = rs[np.searchsorted(ds, common)]
    return float(np.sqrt(np.mean((a - b) ** 2)))


def board_referenced(apparent_t: list[np.ndarray]) -> list[np.ndarray]:
    """task2_compare's treatment: one radiance offset per frame puts the board's mean at L_B(T_air)."""
    l_app = np.stack([c3.band_l(np.asarray(a, np.float64)) for a in apparent_t])
    reg = c3.find_cups(l_app.mean(0))
    mask = c3.board_mask(reg)
    l_air = c3.band_l(c3.T_AIR_C + 273.15)
    out = []
    for j in range(len(l_app)):
        off = l_app[j][mask].mean() - l_air
        t_ref = c3.band_l_inv(l_app[j] - off).astype(np.float32)
        assert t_ref.dtype == np.float32
        out.append(t_ref)
    return out


def rho_s_table(sigma_h_um: float, response) -> dict:
    """rho_s at a few angles, for the report."""
    ang = np.array([70.0, 75.0, 80.0, 82.5, 85.0, 87.5])
    v = rho_specular(np.cos(np.radians(ang)), sigma_h_um, response)
    return {f"{a:g}": round(float(x), 4) for a, x in zip(ang, v, strict=True)}


def save_frames(name: str, outs: list) -> pathlib.Path:
    d = REFL / name
    d.mkdir(parents=True, exist_ok=True)
    ref = board_referenced([o.apparent_t for o in outs])
    for j, (o, r) in enumerate(zip(outs, ref, strict=True)):
        np.save(d / f"apparent_t_{j}.npy", o.apparent_t)
        np.save(d / f"apparent_t_ref_{j}.npy", r)
    return d


__all__ = [name for name in dir() if not name.startswith("_") and name != "annotations"]


# -- the steel ruler (hypothesis test for the hot cup's second lobe) -------------------------------------------

#: RGB camera, thermal_specs/cali_board_new/RGBT_calibration_result_matrix.txt (26 views, 0.05 px).
K_RGB = np.array([[992.57287085, 0.0, 633.59511582], [0.0, 993.22643183, 475.61896147], [0.0, 0.0, 1.0]])
DIST_RGB = np.array([-0.4448657494, 0.2603371017, 0.0011241232, -0.0017513185, -0.1007693884])
#: Polished steel in LWIR reflects ~0.9 and keeps it to grazing; smooth at 10 um (no Rayleigh loss). An
#: assumption of the test, not a measurement.
RHO_STEEL = 0.9


def _undistort_rgb(u, v):
    xd = (u - K_RGB[0, 2]) / K_RGB[0, 0]
    yd = (v - K_RGB[1, 2]) / K_RGB[1, 1]
    k1, k2, p1, p2, k3 = DIST_RGB
    x, y = xd.copy(), yd.copy()
    for _ in range(40):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        x = (xd - (2 * p1 * x * y + p2 * (r2 + 2 * x * x))) / rad
        y = (yd - (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)) / rad
    return x, y


def _rgb_to_thermal_on_table(g, u, v):
    """RGB pixels -> thermal pixels of the table points they see (the table plane from the fitted thermal pose)."""
    c2w = np.linalg.inv(g["view"])
    rcw = np.stack([c2w[0, :3], -c2w[1, :3], -c2w[2, :3]])  # world -> OpenCV camera axes
    n_cam, h = rcw @ np.array([0.0, 0.0, 1.0]), c2w[3, 2]
    r, t = c3.cc.T_TH_RGB[:3, :3], c3.cc.T_TH_RGB[:3, 3]
    x, y = _undistort_rgb(np.asarray(u, float), np.asarray(v, float))
    d = (r @ np.stack([x, y, np.ones_like(x)])).T
    s = (-h - t @ n_cam) / (d @ n_cam)
    p = t[None] + s[:, None] * d
    ut, vt = c3.cc.project_thermal(p)
    return ut, vt, s


def ruler_mask_thermal(g: dict) -> tuple[np.ndarray, dict]:
    """The steel ruler of the RGB frames (grey on the yellow table), mapped onto the thermal image through the
    table plane, then shifted by the offset the same mapping shows on the cold and warm cups' contact points
    (thermal silhouettes, task 0). (480, 640) bool and the bookkeeping."""
    import glob

    from PIL import Image
    from scipy import ndimage

    fs = sorted(glob.glob(str(c3.DATA / "rgb" / "*.png")))
    rgb = np.mean([np.asarray(Image.open(f), np.float64)[..., :3] for f in fs[::10]], axis=0)
    hi, lo = rgb.max(2), rgb.min(2)
    sat = (hi - lo) / np.maximum(hi, 1)
    yy, xx = np.mgrid[0 : rgb.shape[0], 0 : rgb.shape[1]]
    lab, n = ndimage.label((sat < 0.22) & (yy > 575) & (xx > 700))
    ruler = lab == 1 + int(np.argmax(ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))))
    geo = json.loads((c3.OUT / "scene_geometry.json").read_text())["silhouettes"]
    white = lo > 170
    off = []
    for cup, (ca, cb) in {"cold": (370, 480), "warm": (640, 750)}.items():
        sub = white[440:620, ca:cb]
        rows = np.nonzero(sub.sum(1) > 0.5 * sub.sum(1).max())[0]
        cols = np.nonzero(sub[rows.max()])[0]
        ut, vt, _s = _rgb_to_thermal_on_table(g, [ca + 0.5 * (cols.min() + cols.max())], [440 + rows.max() + 0.5])
        m = geo[cup]["measured"]
        off.append((m["centre_col"] - ut[0], m["bottom_row"] - vt[0]))
    du, dv = np.mean(off, axis=0)
    v, u = np.nonzero(ruler)
    ut, vt, s = _rgb_to_thermal_on_table(g, u, v)
    ut, vt = ut + du, vt + dv
    ok = (s > 0) & (ut >= 0) & (ut < 640) & (vt >= 0) & (vt < 480)
    m = np.zeros((480, 640), bool)
    m[vt[ok].astype(int), ut[ok].astype(int)] = True
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=3))
    return m, {
        "rgb_pixels": int(ruler.sum()),
        "offset_cols_rows_from_cold_warm": [round(float(du), 1), round(float(dv), 1)],
        "per_cup_offsets": [[round(float(a), 1), round(float(b), 1)] for a, b in off],
    }
