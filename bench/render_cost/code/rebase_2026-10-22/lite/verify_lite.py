#!/usr/bin/env python
"""TASK 1 -- is stage-1 radiance the same by the full path and by the reduced one? Byte for byte.

Full:    IrCamera.planes() as it stands, then band_radiance(...) with stage_illumination(...),
         the exact call run_frame makes.
Reduced: instance ids -> (material, temperature, background mask), scalar L_env. Nothing else.

No runner is built; nothing in the repo is edited.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402

parser = common.base_parser()
args, kit_args = parser.parse_known_args()
ctx = common.boot(args, kit_args)

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

import lite  # noqa: E402
from irsim.atmosphere.layered import LayeredAtmosphere  # noqa: E402
from irsim.pipeline.atmosphere import apply_layered_gbuffer  # noqa: E402
from irsim.pipeline.frame import run_frame  # noqa: E402
from irsim.pipeline.radiance import band_radiance, stage_illumination  # noqa: E402
from irsim_isaac.pipeline import warp_stages  # noqa: E402
from irsim_isaac.pipeline.material_ids import labels_to_paths  # noqa: E402

camera = ctx["camera"].open(settle_frames=args.settle)
cfg, lut, q = camera.config, ctx["lut"], ctx["quantity"]
settings = ctx["settings"]
for _ in range(3):
    settings.set("/rtx/post/aa/op", 0)
    settings.set("/rtx/rendermode", args.rendermode)
    camera._reader.step(frames=1, rt_subframes=1)

res: dict[str, object] = {
    "scene": args.scene, "temps": args.temps,
    "rendermode": settings.get("/rtx/rendermode"),
    "resolved": dict(camera._reader.resolved),
    "angle_lut_packed": cfg.materials.angle_lut is not None,
    "regime": ctx["sensor"].sensor.band.regime,
    "atmosphere": type(cfg.atmosphere).__name__,
    "poses": {},
}


def mk(l_a, l_b):
    """Apparent-temperature difference in mK between two radiance planes."""
    ta = np.asarray(lut.apparent_temperature(np.asarray(l_a, np.float32), q), np.float64)
    tb = np.asarray(lut.apparent_temperature(np.asarray(l_b, np.float32), q), np.float64)
    return 1e3 * (ta - tb)


def cmp(name, a, b, groups):
    """Bit comparison of two float32 planes, broken down by pixel group."""
    ne = lite.bits(a) != lite.bits(b)
    out = {"n_diff": int(ne.sum()), "n": int(ne.size)}
    if ne.any():
        d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
        out["max_abs"] = float(np.abs(d).max())
        out["max_rel"] = float((np.abs(d) / np.abs(np.asarray(b, np.float64))).max())
        out["max_abs_mK"] = float(np.abs(mk(a, b)).max())
        out["by_group"] = {
            g: {"n_diff": int((ne & m).sum()), "n": int(m.sum()),
                "max_abs_mK": float(np.abs(mk(a, b))[ne & m].max()) if (ne & m).any() else 0.0,
                "signed_mean_mK": float(mk(a, b)[m].mean()) if m.any() else 0.0}
            for g, m in groups.items()
        }
    print(f"   {name:46s} diff {out['n_diff']:7d}/{out['n']}"
          + (f"  max {out['max_abs_mK']:.4f} mK  rel {out['max_rel']:.2e}" if ne.any() else "  IDENTICAL"),
          flush=True)
    return out


POSES = {
    "bench_default": ((0.0, 0.5, 2.0), -20.0),      # the pose every earlier bench used
    "pulled_back": ((0.0, -3.0, 2.0), 0.0),         # background above AND below the horizon
}

dump: dict[str, np.ndarray] = {}
for pose, (pos, tilt) in POSES.items():
    print(f"\n=== pose {pose}: position_enu={pos} tilt={tilt} ===", flush=True)
    ctx["cam_ops"].set_placement(ctx["placement_cls"](position_enu=pos, tilt_deg=tilt))
    camera.refresh_pose()
    camera._reader.step(frames=6, rt_subframes=1)
    planes = camera.planes(step=True)
    last = camera._last
    ids, labels = last.instance_id, last.labels
    t_s = camera.scene.t0_s + camera._t_rel_s
    camera.state.t_s = t_s
    paths = labels_to_paths(labels)

    # ---- full path, stage 1: the call run_frame makes
    illum = stage_illumination(planes, cfg, camera.state)
    l_full = band_radiance(
        planes["temperature_k"], planes["material_id"], cfg.materials, lut, q,
        sky_mask=planes.get("sky_mask"), l_behind=planes.get("radiance_behind"),
        illumination=illum, normal_dot_view=planes.get("normal_dot_view"),
    )
    # ---- reduced path
    l_red, t_red, m_red, sky_red = lite.reduced_numpy(ids, labels, camera, t_s)

    sky_full = np.asarray(planes["sky_mask"], bool)
    elev = np.asarray(last.elevation_rad, np.float64)
    groups = {"bg_above_horizon": sky_full & (elev >= 0.0), "bg_below_horizon": sky_full & (elev < 0.0)}
    for ident, path in sorted(paths.items()):
        if ident != 0 and (ids == ident).any():
            groups[path] = (ids == ident) & ~sky_full

    p: dict[str, object] = {"shape": list(ids.shape), "t_s": t_s,
                            "pixels": {g: int(m.sum()) for g, m in groups.items()},
                            "labels": {str(k): v for k, v in paths.items()}}
    # inputs, one at a time
    l_env_plane = np.asarray(illum.l_env)
    scalar = lite.scalar_l_env(camera, t_s)
    v_s = np.asarray(planes["sky_view_factor"])
    t_air = float(camera.scene.weather_at(camera._t_rel_s).t_air_k)
    l_b_air = np.float32(lut.lookup(np.float64(t_air), q)[()])
    p["inputs"] = {
        "illumination_terms_live": [n for n in ("l_env", "l_sun", "l_night") if getattr(illum, n) is not None],
        "planes_keys": sorted(planes),
        "sky_mask_mismatch": int((sky_full != sky_red).sum()),
        "nonfinite_distance_with_id": int((sky_full & (ids != 0)).sum()),
        "material_id_mismatch": int((np.asarray(planes["material_id"]) != m_red).sum()),
        "material_id_mismatch_on_geometry": int(((np.asarray(planes["material_id"]) != m_red) & ~sky_full).sum()),
        "v_s_min": float(v_s.min()), "v_s_max": float(v_s.max()), "v_s_unique": int(np.unique(v_s).size),
        "l_env_unique_bit_patterns": int(np.unique(lite.bits(l_env_plane.astype(np.float32))).size),
        "l_env_min": float(l_env_plane.min()), "l_env_max": float(l_env_plane.max()),
        "l_env_scalar": float(scalar),
        "l_env_plane_equals_scalar_bits": bool((lite.bits(l_env_plane.astype(np.float32)) == lite.bits(scalar)).all()),
        "t_air_k": t_air, "l_b_t_air": float(l_b_air),
        "scalar_equals_l_b_t_air_bits": bool(lite.bits(scalar) == lite.bits(l_b_air)),
        "l_env_spread_mK": float(np.ptp(lut.apparent_temperature(l_env_plane.astype(np.float32), q)) * 1e3),
    }
    print("   inputs:", json.dumps(p["inputs"])[:1200], flush=True)
    t_full = np.asarray(planes["temperature_k"], np.float32)
    p["temperature"] = cmp("temperature_k full vs reduced", t_full, t_red, groups)
    if p["temperature"]["n_diff"]:
        d = (t_full.astype(np.float64) - t_red.astype(np.float64)) * 1e3
        p["temperature"]["kelvin_diff_mK_by_group"] = {
            g: {"min": float(d[m].min()), "max": float(d[m].max()), "n_diff": int((d[m] != 0).sum())}
            for g, m in groups.items() if m.any()}
        print("   T diff (mK) by group:", json.dumps(p["temperature"]["kelvin_diff_mK_by_group"]), flush=True)
    for g in ("bg_above_horizon", "bg_below_horizon"):
        if groups[g].any():
            tb = t_full[groups[g]].astype(np.float64)
            p[f"full_T_{g}"] = {"min": float(tb.min()), "max": float(tb.max()),
                                "unique": int(np.unique(tb).size)}
            print(f"   full T on {g}: {p[f'full_T_{g}']}", flush=True)

    # ---- THE comparison
    p["radiance_full_vs_reduced"] = cmp("stage-1 radiance: full vs reduced (numpy)", l_full, l_red, groups)
    # same, with the background temperature copied from the full path: isolates the geometry terms
    t_iso = np.where(sky_full, t_full, t_red).astype(np.float32)
    l_red_bg = band_radiance(t_iso, m_red, cfg.materials, lut, q, sky_mask=sky_red,
                             l_env=np.float64(scalar))
    p["radiance_full_vs_reduced_fullbgT"] = cmp(
        "  ... reduced, background T from full path", l_full, l_red_bg, groups)
    # reduced with the full path's whole temperature plane (meaningful with --temps surfaces)
    l_red_t = band_radiance(t_full, m_red, cfg.materials, lut, q, sky_mask=sky_red,
                            l_env=np.float64(scalar))
    p["radiance_full_vs_reduced_fullT"] = cmp(
        "  ... reduced, whole T plane from full path", l_full, l_red_t, groups)

    # ---- on the device: the repo's kernel on the full inputs, and the lite gather + same kernel
    mat_full = np.asarray(planes["material_id"])
    w_full = None
    if not (mat_full[~sky_full] == 0).any():
        w_full = warp_stages.band_radiance_warp(
            t_full, mat_full, cfg.materials, lut, q, sky_mask=sky_full,
            l_env=l_env_plane.astype(np.float32), device="cuda:0")
    gpu = lite.LiteGpu(camera, ids.shape)
    gpu.refresh(labels, t_s)
    ids_dev = wp.array(np.ascontiguousarray(ids, dtype=np.uint32), dtype=wp.uint32, device="cuda:0")
    w_red = gpu.run(ids_dev).numpy()
    if w_full is not None:
        p["warp_full_vs_warp_reduced"] = cmp("stage-1 radiance: full vs reduced (warp kernel)", w_full, w_red, groups)
        p["warp_full_vs_numpy_full"] = cmp("  kernel vs CPU reference, same full inputs", w_full, l_full, groups)
    p["warp_reduced_vs_numpy_reduced"] = cmp("  kernel vs CPU reference, same reduced inputs", w_red, l_red, groups)

    # ---- what the skipped stages would have done
    if isinstance(cfg.atmosphere, LayeredAtmosphere):
        l_atm = apply_layered_gbuffer(
            cfg.atmosphere, ctx["band"], t_s, l_full, np.asarray(planes["distance_m"]), q,
            sky_mask=planes.get("sky_mask"))
        p["stage2_effect"] = cmp("stage 2 (atmosphere) applied vs not", l_atm, l_full, groups)
        dist = np.asarray(planes["distance_m"])
        p["distance_m"] = {g: [float(dist[m].min()), float(dist[m].max())] for g, m in groups.items() if m.any()}
    outputs = run_frame(dict(planes), cfg, camera.state)
    if outputs.radiance is not None and outputs.radiance.shape == l_red.shape:
        d = mk(outputs.radiance, l_red)
        p["full_chain_radiance_vs_reduced_mK"] = {
            "rms": float(np.sqrt((d ** 2).mean())), "max_abs": float(np.abs(d).max()),
            "mean": float(d.mean()), "noise_enabled": bool(cfg.noise_enabled)}
        print("   run_frame().radiance vs reduced stage 1 (mK):", p["full_chain_radiance_vs_reduced_mK"], flush=True)

    res["poses"][pose] = p
    for k, v in (("ids", ids), ("l_full", l_full), ("l_red", l_red), ("t_full", t_full),
                 ("t_red", t_red), ("v_s", v_s), ("l_env", l_env_plane), ("elev", elev),
                 ("sky_full", sky_full), ("w_red", w_red)):
        dump[f"{pose}:{k}"] = np.asarray(v)

# ---- the background temperature the full path assigns, swept over every elevation
el = np.radians(np.linspace(-90.0, 90.0, 3601))
t_bg = np.asarray(camera.bridge.background_temperature_k(el), np.float64)
t_s = camera.scene.t0_s + camera._t_rel_s
t_encl = lite.enclosure_temperature_k(camera, t_s)
res["background_sweep"] = {
    "enclosure_k (ground_temperature_k)": t_encl,
    "below_horizon_min_max": [float(t_bg[el < 0].min()), float(t_bg[el < 0].max())],
    "above_horizon_min_max": [float(t_bg[el >= 0].min()), float(t_bg[el >= 0].max())],
    "max_abs_dev_mK": float(np.abs(t_bg - t_encl).max() * 1e3),
    "cloud_base_m": float(cfg.sky.cloud_base_m(t_s)) if hasattr(cfg.sky, "cloud_base_m") else None,
    "t_cloud_base_k": float(cfg.sky.cloud_base_temperature_k(t_s)) if hasattr(cfg.sky, "cloud_base_temperature_k") else None,
}
print("\nbackground sweep:", json.dumps(res["background_sweep"]), flush=True)

pathlib.Path(args.out + ".json").write_text(json.dumps(res, indent=1, default=str))
np.savez_compressed(args.out + ".npz", **dump)
print(f"wrote {args.out}.json", flush=True)
ctx["app"].close()
