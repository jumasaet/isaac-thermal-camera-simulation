#!/usr/bin/env python
"""What does each Warp twin of stages 3-6 cost at 640x512 on cuda:0, launched ALONE?

/tmp only. Bare Isaac Sim interpreter, no Kit, no renderer sharing the GPU. This composes
nothing: every twin is launched on its own, on the host plane the CPU oracle produced for the
still pose (uploaded once, outside the timer), and synchronised before the clock stops. The sum
of the rows is therefore an estimate of a composed frame, not a measurement of one.

    time_twins.py <tag> <traj>:<pose>
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
import time

REPO = pathlib.Path("/home/jumasaet/isaac-thermal-camera-simulation")
sys.path.insert(0, str(REPO / "src"))
import numpy as np

from irsim_isaac.env import ensure_warp_on_path

ensure_warp_on_path()
import warp as wp

wp.config.quiet = True
wp.init()
DEV = "cuda:0"

from irsim.atmosphere.skylight import skylight_for_sensor
from irsim.config.loader import load_sensor_config
from irsim.config.scene import load_scene_config
from irsim.detector.quantise import dn_max_for_bits, quantise
from irsim.isp.display import run_display_branch
from irsim.materials.library import MaterialLibrary
from irsim.materials.table import MaterialTable
from irsim.optics.stage import apply_optics
from irsim.pipeline.core import PipelineConfig, PipelineState
from irsim.pipeline.sensor_chain import attach_sensor_chain
from irsim.radiometry.lut_files import load_band_lut_for_config, load_band_response_for_config
from irsim.scene import Scene
from irsim_isaac.pipeline import warp_stages as ws

tag, still = sys.argv[1], sys.argv[2]
traj, _, pose = still.partition(":")
ROOT = pathlib.Path("/tmp/irsim_render_cost/stages/out") / tag
S = {v: dict(np.load(ROOT / traj / f"still_{v}_{int(pose):03d}.npz")) for v in ("12", "123", "1236", "12356", "repo")}

sensor = load_sensor_config("/tmp/irsim_render_cost/demo/out/_sensor_boson_ss1.yaml")
spec = sensor.sensor
band, q = spec.band.band_id, spec.quantity
lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
response = load_band_response_for_config(sensor, REPO / "data")
scene = Scene.from_config(load_scene_config("/tmp/irsim_render_cost/demo/scene_nave.yaml"), {band: lut}, quantity=q,
                          skylights={band: skylight_for_sensor(sensor, q)}, responses={band: response})
table = MaterialTable.from_library(MaterialLibrary.load(), band)
cfg = PipelineConfig.from_sensor(sensor, table, lut, sky=scene.sky_models[band], atmosphere=scene.layered)
cfg_repo = attach_sensor_chain(
    PipelineConfig.from_sensor(sensor, table, lut, sky=scene.sky_models[band], atmosphere=scene.layered,
                               flat_field_enabled=True), scene.weather, t0_s=scene.t0_s)
H, W = spec.fpa_shape
BITS = spec.fpa.bit_depth
pc = time.perf_counter
REPS = 300


def timed(fn, reps=REPS, warm=20):
    for _ in range(warm):
        fn()
    wp.synchronize_device(DEV)
    ts = []
    for _ in range(reps):
        t = pc()
        fn()
        wp.synchronize_device(DEV)
        ts.append(pc() - t)
    a = 1e3 * np.asarray(ts)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def up(x, dtype):
    return wp.array(np.ascontiguousarray(x), dtype=dtype, device=DEV)


res: dict = {"device": DEV, "shape": [H, W], "reps": REPS, "rows": {}, "checks": {}}
state = PipelineState()
lb_h = float(lut.lookup(state.housing_temp_k, q)[()])

# -- stage 3 ------------------------------------------------------------------------------------
rad = S["12"]["scene_radiance"].astype(np.float32)
rad_dev = up(rad, wp.float32)
flux_dev = wp.zeros((H, W), dtype=wp.float32, device=DEV)
terms3 = ws.optics_terms(spec, lb_h, cfg.supersample, cfg.psf)
res["rows"]["s3 launch_optics, as written (allocs `blurred`, uploads psf + cos4 each call)"] = timed(
    lambda: ws.launch_optics(rad_dev, terms3, flux_dev, DEV))
flux_cpu = apply_optics(rad, spec, lb_h, supersample=cfg.supersample, psf=cfg.psf)
got = flux_dev.numpy()
res["checks"]["s3 flux vs CPU oracle, max rel"] = float(np.abs(got.astype(np.float64) - flux_cpu) .max() / flux_cpu.max())
terms3_dev = dataclasses.replace(terms3, cos4=up(terms3.cos4, wp.float32), psf=up(terms3.psf, wp.float32))
res["rows"]["s3 launch_optics, psf + cos4 already on device (still allocs `blurred`)"] = timed(
    lambda: ws.launch_optics(rad_dev, terms3_dev, flux_dev, DEV))
res["psf_taps"] = int(terms3.psf.size)

# -- stage 4 ------------------------------------------------------------------------------------
flux_o = up(flux_cpu, wp.float32)
terms4 = ws.detector_terms(cfg)
dstate = ws.WarpPipelineState(DEV)
sig_dev = wp.zeros((H, W), dtype=wp.float32, device=DEV)
lag_dev = wp.zeros((H, W), dtype=wp.float32, device=DEV)
res["rows"]["s4 launch_detector (transfer + membrane IIR, state on device)"] = timed(
    lambda: ws.launch_detector(flux_o, terms4, dstate, sig_dev, lag_dev))
res["checks"]["s4 signal vs CPU oracle, max abs DN"] = float(
    np.abs(sig_dev.numpy().astype(np.float64) - cfg.detector.noiseless_signal_dn(flux_cpu)).max())
res["alpha_twin"] = float(terms4.alpha)

# -- stage 5 ------------------------------------------------------------------------------------
sig_o = up(cfg.detector.noiseless_signal_dn(flux_cpu), wp.float32)
sigma_tvh = float(cfg.detector.sigma_signal_dn)
terms5 = ws.noise_terms(cfg, sigma_tvh)
fixed = dstate.fixed_pattern((H, W), cfg.noise.unit_fixed)
noisy_dev = wp.zeros((H, W), dtype=wp.float32, device=DEV)
fi = [0]


def _noise():
    fi[0] += 1
    ws.launch_noise(sig_o, fixed, terms5, fi[0], cfg.sensor_seed, noisy_dev, DEV)


res["rows"]["s5 launch_noise (all seven NVESD components, TVH included)"] = timed(_noise)
d = noisy_dev.numpy().astype(np.float64) - sig_o.numpy()
res["checks"]["s5 added noise sd, DN (CPU sigma_TVH = %.3f, total with ratios = %.3f)" % (
    sigma_tvh, sigma_tvh * float(np.sqrt(sum(r * r for r in spec.noise.sigma_ratios()))))] = float(d.std())

# -- ADC ----------------------------------------------------------------------------------------
dn_dev = wp.zeros((H, W), dtype=wp.uint16, device=DEV)
res["rows"]["ADC _quantise_kernel"] = timed(lambda: wp.launch(
    ws._quantise_kernel, dim=(H, W), inputs=[noisy_dev, np.int32(dn_max_for_bits(BITS))], outputs=[dn_dev], device=DEV))

# -- stage 6 ------------------------------------------------------------------------------------
dn_cpu = S["12356"]["dn16"]
dn_o = up(dn_cpu, wp.uint16)
res["rows"]["s6 launch_display as written (histogram, AGC table via host, DDE, palette)"] = timed(
    lambda: ws.launch_display(dn_o, spec.isp, BITS, DEV))
res["rows"]["s6   of which agc_lut_warp (3 transfers of 65536 entries + syncs)"] = timed(
    lambda: ws.agc_lut_warp(dn_o, spec.isp, BITS, DEV))
y_dev, rgba_dev = ws.launch_display(dn_o, spec.isp, BITS, DEV)
res["rows"]["s6   download of the RGBA8 frame to host"] = timed(lambda: rgba_dev.numpy())
cpu_disp = run_display_branch(dn_cpu, spec.isp, BITS).display8
dd = np.abs(rgba_dev.numpy().astype(np.int16) - cpu_disp.astype(np.int16))
res["checks"]["s6 display8 vs CPU oracle"] = {"px_differing": int((dd > 0).any(axis=-1).sum()), "max_code_diff": int(dd.max())}

# -- M9 on device (the repo-configured camera) ----------------------------------------------------
chain = cfg_repo.chain
dterms = ws.defect_terms(spec, dn_max_for_bits(BITS))
kind, _, _ = dstate.defect_buffers(chain.bad_pixels, np.asarray(chain.defect_state.bad, dtype=np.uint8))
on_map = dstate.factory_map_buffer(chain.bad_pixels)
defective = wp.zeros((H, W), dtype=wp.float32, device=DEV)
active = wp.zeros((H, W), dtype=wp.uint8, device=DEV)
replaced = wp.zeros((H, W), dtype=wp.float32, device=DEV)
held_out = wp.zeros((H, W), dtype=wp.float32, device=DEV)
passes = [0]


def _defects():
    fi[0] += 1
    rts = ws.launch_rts_step(dstate, dterms, fi[0], cfg.sensor_seed, DEV)
    ws.launch_defects(noisy_dev, kind, rts, dterms, defective, active, DEV, on_map=on_map)


def _replace():
    passes[0] = ws.launch_replacement(defective, active, replaced, DEV)


res["rows"]["m9 launch_rts_step + launch_defects"] = timed(_defects)
res["rows"]["m9 launch_replacement as written (5 allocs, one sync + readback per pass)"] = timed(_replace)
res["replacement_passes"] = int(passes[0])
res["defects"] = {"count": int(chain.bad_pixels.count), "active_now": int(active.numpy().sum())}
res["rows"]["m9 launch_ffc_hold (not freezing: two device copies)"] = timed(
    lambda: ws.launch_ffc_hold(replaced, dstate, False, held_out, DEV))
unit = chain.pattern.drift.sigmas
res["rows"]["m9 launch_ou_drift (V, H, VH fields, one step)"] = timed(
    lambda: ws.launch_ou_drift(fixed, (unit.v, unit.h, unit.vh), 1 / 60, spec.noise.fpn_drift_tau_s, fi[0], cfg.sensor_seed, DEV))
resid = chain.residual
res_out = wp.zeros((H, W), dtype=wp.float32, device=DEV)
gx, ox = np.asarray(resid.gain_field, np.float32), np.asarray(resid.offset_field, np.float32)
res["rows"]["m9 residual_fields (re-uploads both xi planes every call)"] = timed(lambda: dstate.residual_fields(gx, ox))
xi = dstate.residual_fields(gx, ox)
ref_dev = wp.zeros((H, W), dtype=wp.float32, device=DEV)
res["rows"]["m9 launch_nuc_residual (reference plane given)"] = timed(lambda: ws.launch_nuc_residual(
    replaced, xi, spec.nuc.residual_gain_ppm_per_k, spec.nuc.residual_offset_mk_per_k, 178.0, 0.1, res_out, DEV, reference=ref_dev))

# what has no twin at all, on the host, for scale
flat = cfg_repo.flat_field
t = []
for _ in range(30):
    t0 = pc()
    quantise(flat.apply(dn_cpu), BITS)
    t.append(pc() - t0)
res["no_twin_cpu_ms"] = {"display flat field (TwoPointNuc.apply + quantise)": float(1e3 * np.mean(t))}

(ROOT / "twins.json").write_text(json.dumps(res, indent=1))
for k, v in res["rows"].items():
    print(f"{v['p50']:7.3f} ms p50  {v['mean']:7.3f} mean  {v['p95']:7.3f} p95  {v['max']:7.3f} max | {k}")
print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
