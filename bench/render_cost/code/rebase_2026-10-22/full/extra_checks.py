#!/usr/bin/env python
"""Three small things, bare interpreter: (1) the palette kernel with the jet table equals
jet[code of the white-hot frame], so jet frames can be made from delivered frames; (2) what an FFC
event frame costs on the device path; (3) display noise in codes with and without flat field."""
import json, sys, time
sys.path.insert(0, "/home/jumasaet/isaac-thermal-camera-simulation/src"); sys.path.insert(0, "/home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/code/rebase_2026-10-22/full")
import numpy as np
from irsim_isaac.env import ensure_warp_on_path
ensure_warp_on_path()
import warp as wp
wp.config.quiet = True; wp.init()
import common as cm, fullgpu, jet
from irsim.atmosphere.skylight import skylight_for_sensor
from irsim.config.loader import load_sensor_config
from irsim.config.scene import load_scene_config
from irsim.materials.library import MaterialLibrary
from irsim.materials.table import MaterialTable
from irsim.pipeline.core import PipelineConfig
from irsim.pipeline.sensor_chain import attach_sensor_chain
from irsim.radiometry.lut_files import load_band_lut_for_config, load_band_response_for_config
from irsim.scene import Scene
DEV = "cuda:0"; pc = time.perf_counter
def build(ffc=None):
    sensor = load_sensor_config(cm.sensor_yaml(1, *(ffc or (None, None)))); spec = sensor.sensor
    band, q = spec.band.band_id, spec.quantity
    lut = load_band_lut_for_config(sensor, cm.REPO / "data" / "lut")
    scene = Scene.from_config(load_scene_config("/home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/code/rebase_2026-10-22/demo/scene_nave.yaml"), {band: lut}, quantity=q,
                              skylights={band: skylight_for_sensor(sensor, q)}, responses={band: load_band_response_for_config(sensor, cm.REPO / "data")})
    table = MaterialTable.from_library(MaterialLibrary.load(), band)
    return PipelineConfig.from_sensor(sensor, table, lut, sky=scene.sky_models[band], atmosphere=scene.layered, flat_field_enabled=True), scene
rad = wp.array(np.load(cm.FULL.parent / "stages/out/hl/circular/still_12_054.npz")["scene_radiance"].astype(np.float32), dtype=wp.float32, device=DEV)
out = {}
# (1)
cfg, scene = build()
g = fullgpu.FullGpu(cfg, device=DEV, flat_field=True, chain=attach_sensor_chain(cfg, scene.weather, t0_s=scene.t0_s).chain, extra_palette=jet.jet_table())
for i in range(3):
    g.frame(rad, scene.t0_s + i / 60, extra_palette=True)
white, jet_dev = g.rgba.numpy(), g.rgba2.numpy()
out["jet_kernel_equals_table_of_white_hot_codes"] = bool(np.array_equal(jet_dev[..., :3], jet.jet_table()[white[..., 0]]))
out["white_hot_is_grey"] = bool((white[..., 0] == white[..., 1]).all() and (white[..., 1] == white[..., 2]).all())
# (2)
cfg_f, scene = build((0.1, 50))
g = fullgpu.FullGpu(cfg_f, device=DEV, flat_field=True, chain=attach_sensor_chain(cfg_f, scene.weather, t0_s=scene.t0_s).chain, profile=True)
rows = []
for i in range(260):
    tm = dict.fromkeys(fullgpu.BLOCKS, 0.0); t = pc()
    g.frame(rad, scene.t0_s + i / 60, tm); wp.synchronize_device(DEV)
    rows.append((pc() - t, g.last_event.fired, g.last_event.frozen, tm["m9_shutter"], tm["m9_ffc"]))
rows = rows[20:]
a = np.array([(r[0], r[3], r[4]) for r in rows]) * 1e3
fired = np.array([r[1] for r in rows]); frozen = np.array([r[2] for r in rows])
out["ffc_event_frame_ms"] = {"n_events": int(fired.sum()), "total": float(a[fired, 0].mean()), "of_which_shutter_record_host": float(a[fired, 1].mean()),
                             "of_which_resets_and_hold": float(a[fired, 2].mean())}
out["frozen_not_fired_frame_ms"] = float(a[frozen & ~fired, 0].mean())
out["live_frame_ms"] = float(a[~frozen, 0].mean())
# (3)
S = cm.FULL / "out" / "k1_hl_vpoff" / "circular"
m = np.load(S / "meta_054.npz"); ids = m["ids"]; lab = {v.split("/")[-1]: int(k) for k, v in json.loads(str(m["labels"])).items()}
def interior(name, r=10):
    mk = ids == lab[name]
    for _ in range(r):
        p = np.pad(mk, 1); mk = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return mk
def hp(a):  # pixel minus the mean of its 4 neighbours: kills gradients, keeps white noise (x sqrt(5/4))
    a = a.astype(np.float64); p = np.pad(a, 1, mode="edge")
    return (a - 0.25 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])) / np.sqrt(1.25)
for v in ("g", "g_ff", "g_m9", "g_m9_ff"):
    s = np.load(S / f"still_{v}_054.npz")
    out.setdefault("pixel_noise", {})[v] = {p: {"dn16_sd": float(hp(s["dn16"])[interior(p)].std()), "display_codes_sd": float(hp(255.0 * s["y"])[interior(p)].std())}
                                            for p in ("Wall", "Floor")}
print(json.dumps(out, indent=1))
json.dump(out, open(cm.FULL / "out" / "deliver" / "extra_checks.json", "w"), indent=1)
