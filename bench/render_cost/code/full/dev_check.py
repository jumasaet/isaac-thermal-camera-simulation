#!/usr/bin/env python
"""Is the composed device frame the CPU frame? Bare Isaac interpreter, no Kit, no renderer.

/tmp only. Input: the stage-2 radiance stills the stage measurement saved (640x512), cycled so
consecutive frames show different scenes and the membrane IIR has something to lag. For k > 1
the still is upsampled with sub-pixel structure added, which is not a render but is a k x plane
the PSF and the box have real work on.

For each camera (plain, +flat field, +M9, +M9+flat field, and a camera with a 6-frame FFC
schedule) and each optics implementation, N consecutive frames in two modes:

    B  the device frame is given the CPU's noise field and RTS state -> everything else must
       agree plane by plane (flux, signal, dn16, shown DN, display)
    C  the device draws its own noise -> the difference must have the moments two independent
       draws of the same model have

    dev_check.py [--k 1 2 4] [--frames 8] [--timing]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

sys.path.insert(0, "/home/jumasaet/isaac-thermal-camera-simulation/src")
sys.path.insert(0, "/tmp/irsim_render_cost/full")
import numpy as np

from irsim_isaac.env import ensure_warp_on_path

ensure_warp_on_path()
import warp as wp

wp.config.quiet = True
wp.init()

import common as cm
import fullgpu
from irsim.atmosphere.skylight import skylight_for_sensor
from irsim.config.loader import load_sensor_config
from irsim.config.scene import load_scene_config
from irsim.materials.library import MaterialLibrary
from irsim.materials.table import MaterialTable
from irsim.pipeline.core import PipelineConfig, PipelineState
from irsim.pipeline.sensor_chain import attach_sensor_chain
from irsim.radiometry.lut_files import load_band_lut_for_config, load_band_response_for_config
from irsim.scene import Scene
from irsim.isp.radiometric import dn_per_kelvin
from irsim.noise.three_d import Sigmas7

p = argparse.ArgumentParser()
p.add_argument("--k", type=int, nargs="*", default=[1])
p.add_argument("--frames", type=int, default=8)
p.add_argument("--timing", action="store_true")
p.add_argument("--out", default=str(cm.FULL / "out" / "dev_check.json"))
args = p.parse_args()
DEV = "cuda:0"
pc = time.perf_counter
STILLS = [("linear", 40), ("linear", 20), ("circular", 54), ("circular", 10)]
ROOT = cm.FULL.parent / "stages" / "out" / "hl"
RAD1 = [np.load(ROOT / t / f"still_12_{j:03d}.npz")["scene_radiance"].astype(np.float32) for t, j in STILLS]


def spec_ratios(cfg):
    return cfg.sensor.sensor.noise.sigma_ratios()


def upsample(rad: np.ndarray, k: int, seed: int) -> np.ndarray:
    if k == 1:
        return rad
    big = np.kron(rad, np.ones((k, k), np.float32))
    rng = np.random.default_rng(seed)
    return (big * (1.0 + 0.02 * rng.standard_normal(big.shape).astype(np.float32))).astype(np.float32)


def build(k: int, ffc=None):
    sensor = load_sensor_config(cm.sensor_yaml(k, *(ffc or (None, None))))
    spec = sensor.sensor
    band, q = spec.band.band_id, spec.quantity
    lut = load_band_lut_for_config(sensor, cm.REPO / "data" / "lut")
    response = load_band_response_for_config(sensor, cm.REPO / "data")
    scene = Scene.from_config(load_scene_config("/tmp/irsim_render_cost/demo/scene_nave.yaml"), {band: lut}, quantity=q,
                              skylights={band: skylight_for_sensor(sensor, q)}, responses={band: response})
    table = MaterialTable.from_library(MaterialLibrary.load(), band)
    kw = dict(sky=scene.sky_models[band], atmosphere=scene.layered)
    plain = PipelineConfig.from_sensor(sensor, table, lut, flat_field_enabled=False, **kw)
    flat = PipelineConfig.from_sensor(sensor, table, lut, flat_field_enabled=True, **kw)
    return {"plain": plain, "flat": flat, "scene": scene}


def camera(cfgs, name):
    """(config for the CPU, config + kwargs for the device). Chains are never shared: each side
    advances its own housing node, FFC schedule and residual."""
    scene = cfgs["scene"]
    base = cfgs["flat"] if "ff" in name else cfgs["plain"]
    if "m9" in name:
        return attach_sensor_chain(base, scene.weather, t0_s=scene.t0_s), \
            lambda: attach_sensor_chain(base, scene.weather, t0_s=scene.t0_s).chain, base
    return base, lambda: None, base


def run(k: int, name: str, optics: str, n: int, ffc=None) -> dict:
    cfgs = build(k, ffc)
    cfg_cpu, chain_for_dev, base = camera(cfgs, name)
    scene = cfgs["scene"]
    global DN_PER_K
    DN_PER_K = float(dn_per_kelvin(base.calibration, base.lut, 300.0))
    gb = fullgpu.FullGpu(base, device=DEV, flat_field="ff" in name, chain=chain_for_dev(), optics=optics)
    gc = fullgpu.FullGpu(base, device=DEV, flat_field="ff" in name, chain=chain_for_dev(), optics=optics)
    st = PipelineState(t_s=scene.t0_s)
    rows, stats, events = [], [], []
    bad_map = None if cfg_cpu.chain is None else np.asarray(cfg_cpu.chain.bad_pixels.mask)
    for i in range(n):
        rad = upsample(RAD1[i % len(RAD1)], k, i)
        t_s = scene.t0_s + i / 60.0
        st.t_s = t_s
        cpu = cm.cpu_tail(rad, cfg_cpu, st, want_noise=True)
        rad_dev = wp.array(rad, dtype=wp.float32, device=DEV)
        gb.frame(rad_dev, t_s, inject_noise=cpu["noise"], rts_override=cpu.get("rts"))
        b = gb.planes()
        row = cm.compare(cpu, b, DN_PER_K, float(base.detector.transfer.gain_dn_per_w))
        row.update(frame=i, housing_k_cpu=cpu["housing_k"], housing_k_dev=float(gb.housing_k))
        if cpu["report"] is not None:
            ev_c, ev_g = cpu["report"].ffc, gb.last_event
            row["ffc_cpu"] = [bool(ev_c.fired), bool(ev_c.frozen), int(ev_c.frames_held)]
            row["ffc_dev"] = [bool(ev_g.fired), bool(ev_g.frozen), int(ev_g.frames_held)]
            row["dT_eff_cpu"], row["dT_eff_dev"] = float(cpu["report"].delta_t_fpa_k), float(gb.chain.ffc.delta_t_eff_k)
            row["active_cpu"] = int(cpu["report"].defects_active)
            if row["ffc_cpu"] != row["ffc_dev"] or abs(row["housing_k_cpu"] - row["housing_k_dev"]) > 1e-9:
                row["fails"].append("m9_schedule")
        rows.append(row)
        gc.frame(rad_dev, t_s)
        c = gc.planes()
        frozen = cpu["report"] is not None and cpu["report"].ffc.frozen
        if not frozen:
            s = cm.noise_stats(cpu["signal"], c["signal"], Sigmas7.from_ratios(float(base.detector.sigma_signal_dn), spec_ratios(base)),
                               exclude=bad_map)
            s["frame"] = i
            s["display_mean_code_cpu"] = float(cpu["display8"][..., 0].mean())
            s["display_mean_code_dev"] = float(c["display8"][..., 0].mean())
            stats.append(s)
    worst = {key: max(r[key] for r in rows) for key in rows[0] if isinstance(rows[0][key], (int, float)) and key not in ("frame", "n_px")}
    fails = sorted({f for r in rows for f in r["fails"]})
    zs = {key: float(np.max(np.abs([s[key] for s in stats]))) for key in ("z_px", "z_row", "z_col", "z_mean")} if stats else {}
    pooled = {key: float(np.mean([s[key] for s in stats])) for key in ("sd_px", "sd_px_expected", "sd_row", "sd_row_expected", "sd_col", "sd_col_expected")} if stats else {}
    out = {"k": k, "camera": name, "optics": optics, "frames": n, "worst": worst, "fails": fails,
           "noise_max_abs_z": zs, "noise_pooled": pooled, "rows": rows, "stats": stats}
    print(f"k={k} {name:8s} {optics:5s} B: flux {worst['flux_max_rel']:.1e} rel (= {worst['flux_max_abs_as_dn']:.4f} DN) | signal max "
          f"{worst['signal_max_abs_dn']:.4f} DN = {worst['signal_max_abs_mk_at_300k']:.2f} mK, mean {worst['signal_mean_abs_dn']:.4f} | "
          f"dn16 {worst['dn16_px_differing']} px ({100 * worst['dn16_frac_differing']:.2f} %), max {worst['dn16_max_code']} | shown DN {worst['display_dn_px_differing']} px, max "
          f"{worst['display_dn_max_code']} | display8 {worst['display8_px_differing']} px max {worst['display8_max_code']}, far from a DN "
          f"change {worst['display8_far_px_differing']} px max {worst['display8_far_max_code']} | FAILS {fails or 'none'}", flush=True)
    if stats:
        print(f"          C: sd(dev-cpu) {pooled['sd_px']:.3f} (exp {pooled['sd_px_expected']:.3f}) rows {pooled['sd_row']:.3f} "
              f"(exp {pooled['sd_row_expected']:.3f}) cols {pooled['sd_col']:.3f} (exp {pooled['sd_col_expected']:.3f}) DN | max |z| "
              f"{ {k_: round(v, 2) for k_, v in zs.items()} }", flush=True)
    return out


def timing(k: int, name: str, optics: str, reps: int = 200) -> dict:
    cfgs = build(k)
    _cfg_cpu, chain_for_dev, base = camera(cfgs, name)
    scene = cfgs["scene"]
    out = {}
    rad_dev = wp.array(upsample(RAD1[0], k, 0), dtype=wp.float32, device=DEV)
    for profile in (False, True):
        g = fullgpu.FullGpu(base, device=DEV, flat_field="ff" in name, chain=chain_for_dev(), optics=optics, profile=profile)
        tm = dict.fromkeys(fullgpu.BLOCKS, 0.0)
        tot = []
        for i in range(reps + 20):
            if i == 20:
                tm = dict.fromkeys(fullgpu.BLOCKS, 0.0)
            t = pc()
            g.frame(rad_dev, scene.t0_s + i / 60.0, tm)
            wp.synchronize_device(DEV)
            if i >= 20:
                tot.append(pc() - t)
        a = 1e3 * np.asarray(tot)
        out["profile" if profile else "plain"] = {"total_mean": float(a.mean()), "total_p95": float(np.percentile(a, 95)),
                                                  "blocks": {b: 1e3 * v / reps for b, v in tm.items() if v > 0}}
    pl, pr = out["plain"], out["profile"]
    print(f"k={k} {name:8s} {optics:5s} alone on the GPU: {pl['total_mean']:.3f} ms/frame (p95 {pl['total_p95']:.3f}); synchronised per block "
          f"{pr['total_mean']:.3f} = " + " ".join(f"{b}={v:.3f}" for b, v in pr["blocks"].items()), flush=True)
    return out


res: dict = {"equivalence": [], "timing": {}}
for k in args.k:
    impls = ("repo",) if k == 1 else ("repo", "fused")
    for optics in impls:
        for name in ("plain", "ff", "m9", "m9_ff"):
            res["equivalence"].append(run(k, name, optics, args.frames))
    # the shutter: a 6-frame schedule with a 3-frame freeze, so the fire, the hold, the resets
    # of xi and of the pattern, and the refreshed display offset all happen inside the run
    r = run(k, "m9_ff", impls[-1], 20, ffc=(0.1, 50))
    r["camera"] = "m9_ff, FFC every 6 frames"
    print("          FFC (fired, frozen, held) cpu|dev:", [(tuple(x["ffc_cpu"]), tuple(x["ffc_dev"])) for x in r["rows"][4:14]])
    res["equivalence"].append(r)
    if args.timing:
        for optics in impls:
            for name in ("plain", "ff", "m9", "m9_ff"):
                res["timing"][f"k{k}_{name}_{optics}"] = timing(k, name, optics)
all_fails = sorted({f"{r['camera']}/k{r['k']}/{r['optics']}: {f}" for r in res["equivalence"] for f in r["fails"]})
res["fails"] = all_fails
json.dump(res, open(args.out, "w"), indent=1, default=float)
print("\nVERDICT:", "all planes within tolerance" if not all_fails else f"FAILS -> {all_fails}")
