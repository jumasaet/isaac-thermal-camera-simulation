"""Is L_env a scalar under the isothermal-enclosure preset? Engine-free, no Isaac.

Sweeps the sky-view factor over [0, 1] and reports the spread of L_env. A closed isothermal
enclosure must give one value at every V_s; anything else means the preset still has an angular
sky in it.
"""
from __future__ import annotations
import pathlib, sys, time
import numpy as np

REPO = pathlib.Path("/home/jumasaet/isaac-thermal-camera-simulation")
sys.path.insert(0, str(REPO / "src"))

from irsim.config.scene import load_scene_config          # noqa: E402
from irsim.config.loader import load_sensor_config          # noqa: E402
from irsim.scene import Scene                              # noqa: E402
from irsim.pipeline.environment import environment_radiance, ground_temperature_k  # noqa: E402
from irsim.radiometry.lut_files import (                    # noqa: E402
    load_band_lut_for_config,
    load_band_response_for_config,
)

SENSOR = REPO / "configs/sensors/flir_boson_640_lwir.yaml"

def report(scene_path: pathlib.Path, band: str | None = None) -> None:
    sensor = load_sensor_config(SENSOR)
    band = band or sensor.sensor.band.band_id
    quantity = sensor.sensor.quantity
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    response = load_band_response_for_config(sensor, REPO / "data")
    spec = load_scene_config(scene_path)
    scene = Scene.from_config(
        spec, {band: lut}, quantity=quantity, responses={band: response}
    )
    sky = scene.sky_models[band]
    t_s = scene.t0_s
    v = np.linspace(0.0, 1.0, 1001)
    l_env = environment_radiance(sky, lut, t_s, v, quantity)
    t_air = sky.weather.at(t_s).t_air_k
    print(f"\n--- {scene_path.name} [{band}] ---")
    print(f"  T_air            = {t_air:.4f} K")
    print(f"  T_ground         = {ground_temperature_k(sky, t_s):.4f} K")
    print(f"  cloud_fraction   = {sky.weather.at(t_s).cloud_fraction}")
    print(f"  cloud base       = {sky.cloud_base_m(t_s):.6f} m")
    print(f"  T_cloud_base     = {sky.cloud_base_temperature_k(t_s):.4f} K")
    print(f"  L_env  min       = {l_env.min():.9g} W/m2/sr")
    print(f"  L_env  max       = {l_env.max():.9g}")
    print(f"  L_env  spread    = {l_env.max() - l_env.min():.3e}  "
          f"({100*(l_env.max()-l_env.min())/l_env.mean():.4f} % of mean)")
    l_wall = float(lut.lookup(np.float64(t_air), quantity)[()])
    print(f"  L_B(T_air)       = {l_wall:.9g}")
    print(f"  max |L_env - L_B(T_air)| = {np.abs(l_env - l_wall).max():.3e}")
    # apparent-temperature spread is the number that matters against a 50 mK NETD
    t_app = lut.apparent_temperature(l_env, quantity)
    print(f"  T_app spread     = {float(np.ptp(t_app))*1e3:.3f} mK   (NETD is ~50 mK)")
    # cost: is there a scalar fast path?
    for n in (640*512, 2560*2048):
        vv = np.full(n, 0.5)
        t0 = time.perf_counter()
        for _ in range(10):
            environment_radiance(sky, lut, t_s, vv, quantity)
        dt = (time.perf_counter() - t0) / 10
        print(f"  environment_radiance over {n:>9d} px: {dt*1e3:8.3f} ms")

for p in sys.argv[1:]:
    report(pathlib.Path(p))
