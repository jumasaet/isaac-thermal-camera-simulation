#!/usr/bin/env python
"""Task 7: where the simulated camera's +0.28 K on the board comes from, engine-free.

    /home/jumasaet/isaac_sim/python.sh bench/cups3_check/task7_housing.py          (seconds, no Kit)

docs/physics-model.md §8.2 (optics self-emission), §12.2 `optics.housing_*`; ADR 0016, ADR 0021.

A uniform eps = 1 scene at the room's 23.9 C goes through the camera chain task2_render.py uses (factory flat field +
M9 sensor chain, bench/camera_preview/common.py), noiseless, at 1 m. A perfect camera reads 0 K everywhere. Each
row changes ONE thing in the sensor YAML. What the rows show: the offset is not any one Boson number, it is the
MISMATCH between the housing temperature the camera runs at (`coupled`: T_air + housing_self_heating_k = 301.05 K)
and the one its radiometric inversion assumes (`PipelineConfig.t_housing_cal_k`, 300 K when `housing_temp_k` is
unset). The housing term grows off-axis faster than the scene's cos^4, so the same mismatch is also the
vignetting bowl. Whenever the two agree the offset is 0 whatever the self-heating or transmittance.

The real camera's header temperatures (board, shutter) would set both properly; they are not in the recording
(RESULTADOS.md, task 7). The Kit check of the proposed YAML is task2_render.py + task2_compare.py with
CUPS3_SENSOR_YAML and CUPS3_SIM_DIR (see RESULTADOS.md).

Writes out/task7_housing.json.
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

import numpy as np
import yaml

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from irsim.config.sensor import SensorConfig  # noqa: E402
from irsim.materials.table import MaterialTable  # noqa: E402
from irsim.pipeline import (  # noqa: E402
    PipelineConfig,
    PipelineState,
    attach_sensor_chain,
    run_frame,
)
from irsim.radiometry.lut_files import load_band_lut_for_config  # noqa: E402

OLD_YAML = REPO / "bench" / "camera_check" / "cellplus_ctc006_lwir_cups.yaml"
NEW_YAML = REPO / "bench" / "camera_check" / "cellplus_ctc006_lwir_cups3.yaml"
LUT_DIR = REPO / "bench" / "camera_preview" / "lut"
T_AIR_K = 23.9 + 273.15


def offset(label: str, path: pathlib.Path, optics: dict | None = None, cal_k: float | None = None) -> dict:
    """Mean apparent_t - T_air of the last of 6 noiseless frames (the housing has settled; t0 = steady state)."""
    d = yaml.safe_load(path.read_text())
    d = copy.deepcopy(d)
    d["sensor"]["optics"].update(optics or {})
    sensor = SensorConfig.model_validate(d)
    lut = load_band_lut_for_config(sensor, LUT_DIR)
    k = sensor.sensor.optics.supersample_factor
    h, w = sensor.sensor.fpa_shape
    cfg = PipelineConfig.from_sensor(sensor, MaterialTable.constant(1.0), lut=lut, noise_enabled=False,
                                     flat_field_enabled=True, t_housing_cal_k=cal_k, sensor_seed=7)
    cfg = attach_sensor_chain(cfg, ambient_provider=lambda t: T_AIR_K)
    planes = {"temperature_k": np.full((h * k, w * k), T_AIR_K, np.float32),
              "material_id": np.ones((h * k, w * k), np.int32),
              "distance_m": np.full((h * k, w * k), 1.0, np.float32)}
    state = PipelineState()
    for i in range(6):
        state.t_s = i / 30.0
        out = run_frame(planes, cfg, state)
    assert out.apparent_t is not None and out.apparent_t.dtype == np.float32
    a = np.asarray(out.apparent_t, np.float64) - T_AIR_K
    row = {"label": label, "yaml": path.name, "t_housing_k": round(state.housing_temp_k, 3),
           "t_housing_cal_k": round(cfg.t_housing_cal_k, 3), "offset_median_k": round(float(np.median(a)), 3),
           "offset_centre_k": round(float(a[h // 2 - 20:h // 2 + 20, w // 2 - 20:w // 2 + 20].mean()), 3),
           "offset_corner_k": round(float(a[:20, :20].mean()), 3), "bowl_peak_to_peak_k": round(float(np.ptp(a)), 3)}
    print(json.dumps(row), flush=True)
    return row


def main() -> None:
    rows = [
        offset("as cellplus_ctc006_lwir_cups.yaml (Boson housing numbers)", OLD_YAML),
        offset("housing_self_heating_k 0 (housing = air)", OLD_YAML, {"housing_self_heating_k": 0.0}),
        offset("transmittance 1.0 (no lens emission)", OLD_YAML, {"transmittance": 1.0}),
        offset("calibration at the steady housing (T_air + 4 K)", OLD_YAML, cal_k=T_AIR_K + 4.0),
        offset("self-heating 0 + calibration at T_air", OLD_YAML, {"housing_self_heating_k": 0.0}, cal_k=T_AIR_K),
        offset("fixed housing at 300 K", OLD_YAML, {"housing_temp_mode": "fixed", "housing_temp_k": 300.0}),
        offset("PROPOSED cellplus_ctc006_lwir_cups3.yaml (housing_temp_k 301.05)", NEW_YAML),
    ]
    rep = {"scene": "uniform eps = 1 at 23.9 C, 1 m, noiseless, factory flat field + M9 chain", "t_air_k": T_AIR_K,
           "rows": rows,
           "apparent_k_per_k_of_housing_mismatch": round((rows[0]["offset_median_k"] - rows[1]["offset_median_k"]) / 4.0, 4)}
    (HERE / "out" / "task7_housing.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
