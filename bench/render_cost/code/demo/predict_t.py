#!/usr/bin/env python3
"""Engine-free: apparent temperature of each prim at zero range, T_app = L^-1(eps L(T) + (1-eps) L(T_env))."""
import sys, pathlib
sys.path.insert(0, "/tmp/irsim_render_cost/demo"); sys.path.insert(0, "/home/jumasaet/isaac-thermal-camera-simulation/src")
import numpy as np, yaml
import nave_stage as ns
from irsim.config.loader import load_sensor_config
from irsim.materials.library import MaterialLibrary
from irsim.materials.table import MaterialTable
from irsim.radiometry.lut_files import load_band_lut_for_config
sensor = load_sensor_config(ns.sensor_ss1_yaml()); q = sensor.sensor.quantity; band = sensor.sensor.band.band_id
lut = load_band_lut_for_config(sensor, ns.REPO / "data" / "lut")
table = MaterialTable.from_library(MaterialLibrary.load(), band)
eps = {n: float(e) for n, e in zip(table.names, table.emissivity)}
T_ENV = 294.15
L = lambda t: float(lut.lookup(np.float64(t), q)[()])
def t_app(t_k, e):
    return float(lut.apparent_temperature(np.float32(e * L(t_k) + (1 - e) * L(T_ENV)), q)) - 273.15
spec = yaml.safe_load(ns.SCENE_YAML.read_text())["scene"]["targets"]
tk = {t["name"]: t["schedule_k"][0] for t in spec}
mats = ns.load_materials()
print(f"band {band}  quantity {q}  T_env {T_ENV-273.15:.2f} C   (eps = band-averaged, normal incidence)")
print(f"{'prim':10s} {'material':18s} {'eps':>5s} {'T_kin':>7s} {'T_app':>7s} | as bare_aluminium (eps {eps['bare_aluminium']:.3f}) | as aluminium_polished (eps {eps['aluminium_polished']:.3f})")
for o in ns.OBJECTS:
    n = o["name"]; t = tk[ns.target_name(n)]; e = eps[mats[n]]
    print(f"{n:10s} {mats[n]:18s} {e:5.3f} {t-273.15:7.1f} {t_app(t, e):7.2f} | {t_app(t, eps['bare_aluminium']):28.2f} | {t_app(t, eps['aluminium_polished']):10.2f}")
