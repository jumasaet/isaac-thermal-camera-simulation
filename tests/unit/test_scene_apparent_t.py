"""The scene's apparent temperature before the camera: the radiometric reference plane.

``apparent_t`` is the camera's reading -- stage 6 inverting a signal that carries the PSF, cos⁴,
the housing, noise, the NUC and the calibration. It cannot be the reference a real camera's
radiometric frame is compared against, because every one of those is a thing the comparison is
trying to measure. ``scene_apparent_t`` is L_B⁻¹ of the stage 1+2 radiance, taken before the
optics (docs/physics-model.md §3.3, §12.2 ``outputs``).

What would go wrong unnoticed, one test each: the plane averaged in kelvin rather than radiance
(a pixel integrates power, and at an edge the two differ by kelvins), taken after something the
camera does (so noise or the PSF leaks into the reference), a hash that moves for every camera
that never asked for it, and a sidecar that says "unspecified" or a writer that drops to a
narrower dtype.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from irsim.config.loader import config_hash, load_sensor_config
from irsim.config.sensor import FULL_FIDELITY, SensorConfig
from irsim.io.dataset import PLANE_UNITS, write_frame
from irsim.materials import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, run_frame

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
N = 32  # detector grid edge
LUT_MK = 1e-3  # the LUT's own round trip (tests/unit/test_planck.py): 1 mK


def _boson(*, k: int = 1, plane: bool | None = True, **fidelity: bool) -> SensorConfig:
    """The Boson at N x N, supersample ``k``; ``plane=None`` leaves the flag unwritten."""
    d = load_sensor_config(BOSON_YAML).model_dump(mode="json")
    d["sensor"]["fpa"].update(width=N, height=N)
    d["sensor"]["optics"]["supersample_factor"] = k
    if plane is None:
        d["sensor"]["outputs"].pop("scene_apparent_temperature", None)
    else:
        d["sensor"]["outputs"]["scene_apparent_temperature"] = plane
    if fidelity:
        d["sensor"]["fidelity"] = {**FULL_FIDELITY.model_dump(), **fidelity}
    return SensorConfig.model_validate(d)


def _planes(t: np.ndarray) -> dict[str, np.ndarray]:
    """A blackbody (ε = 1, so ρ = 0 and nothing is reflected) at ``t``, no atmosphere."""
    return {
        "temperature_k": t.astype(np.float32),
        "material_id": np.ones(t.shape, np.int32),
        "distance_m": np.full(t.shape, 50.0, np.float32),
    }


def _patches(k: int = 1) -> np.ndarray:
    t = np.full((N * k, N * k), 295.0)
    t[8 * k : 24 * k, 8 * k : 24 * k] = 320.0
    return t


def _config(sensor: SensorConfig, lut, seed: int = 77) -> PipelineConfig:  # type: ignore[no-untyped-def]
    return PipelineConfig.from_sensor(
        sensor, MaterialTable.constant(1.0), lut=lut, sensor_seed=seed
    )


def test_off_is_the_default_and_off_is_not_in_the_hash(boson_lut) -> None:  # type: ignore[no-untyped-def]
    """Every camera written before the flag existed keeps its hash and gets no plane."""
    unwritten, off, on = _boson(plane=None), _boson(plane=False), _boson(plane=True)
    assert unwritten.sensor.outputs.scene_apparent_temperature is False
    assert config_hash(unwritten) == config_hash(off)
    assert config_hash(on) != config_hash(off), "a frame must say whether it carries the plane"
    out = run_frame(_planes(_patches()), _config(off, boson_lut), PipelineState())
    assert out.scene_apparent_t is None


def test_a_blackbody_reads_its_own_temperature_before_the_camera(boson_lut) -> None:  # type: ignore[no-untyped-def]
    """ε = 1 and no path: the reference is the kinetic temperature to the LUT's round trip,
    while the camera's own reading of the same scene is off by its noise and optics."""
    t = _patches()
    out = run_frame(_planes(t), _config(_boson(), boson_lut), PipelineState())
    ref = out.scene_apparent_t
    assert ref is not None and ref.dtype == np.float32 and ref.shape == (N, N)
    assert np.max(np.abs(ref - t)) < LUT_MK
    # The camera's reading is a different thing -- or this test could not tell the two apart.
    assert out.apparent_t is not None
    assert np.max(np.abs(out.apparent_t - t)) > 20 * LUT_MK


def test_a_pixel_averages_radiance_not_kelvin(boson_lut) -> None:  # type: ignore[no-untyped-def]
    """Each 2x2 block holds 250 K and 350 K. A pixel collects the mean *power*, whose apparent
    temperature sits well above the 300 K mean of the two -- Planck is convex in T in LWIR."""
    k = 2
    t = np.where((np.indices((N * k, N * k)).sum(axis=0) % 2) == 0, 250.0, 350.0)
    out = run_frame(_planes(t), _config(_boson(k=k), boson_lut), PipelineState())
    l_mean = 0.5 * (boson_lut.lookup(250.0, "lb") + boson_lut.lookup(350.0, "lb"))
    expected = float(boson_lut.apparent_temperature(np.float32(l_mean), "lb"))
    assert expected - 300.0 > 1.0, "the case must discriminate radiance from kelvin averaging"
    assert out.scene_apparent_t is not None
    assert np.max(np.abs(out.scene_apparent_t - expected)) < LUT_MK


def test_nothing_the_camera_does_reaches_the_reference(boson_lut) -> None:  # type: ignore[no-untyped-def]
    """Full fidelity vs every switch off, two seeds, three frames: the reference is bit-identical
    throughout; the camera's reading is not (noise, PSF, defects and the NUC all move it)."""
    planes = _planes(_patches())
    full = _config(_boson(), boson_lut, seed=1)
    bare = _config(
        _boson(noise=False, optical_psf=False, bad_pixels=False, nuc_residual=False),
        boson_lut,
        seed=2,
    )
    refs, readings = [], []
    for config in (full, bare):
        state = PipelineState()
        for _ in range(3):
            out = run_frame(planes, config, state)
            refs.append(out.scene_apparent_t)
            readings.append(out.apparent_t)
    assert all(np.array_equal(refs[0], r) for r in refs[1:])
    assert not all(np.array_equal(readings[0], r) for r in readings[1:])


@pytest.mark.parametrize("plane", [True, False])
def test_the_writer_declares_the_unit_and_keeps_float32(
    boson_lut,  # type: ignore[no-untyped-def]
    tmp_path: pathlib.Path,
    plane: bool,
) -> None:
    """The sidecar names the unit (never "unspecified"), the file round-trips bit for bit, and a
    plane that was not asked for is absent rather than zero."""
    out = run_frame(_planes(_patches()), _config(_boson(plane=plane), boson_lut), PipelineState())
    record = write_frame(tmp_path, out)
    doc = json.loads(record.sidecar.read_text())
    if not plane:
        assert "scene_apparent_t" not in doc["planes"]
        return
    entry = doc["planes"]["scene_apparent_t"]
    assert entry["unit"] == PLANE_UNITS["scene_apparent_t"] != "unspecified"
    assert entry["unit"].startswith("K")
    assert entry["dtype"] == "float32"
    back = np.load(tmp_path / entry["file"])
    assert back.dtype == np.float32 and np.array_equal(back, out.scene_apparent_t)
