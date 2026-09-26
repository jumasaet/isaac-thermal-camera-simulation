"""ADR 0123 — a prescribed temperature on a patched surface: ground truth for a thermal anomaly.

A steam line, a hot bearing, a failed insulation panel: a dataset for anomaly detection needs the
anomaly at a temperature *it* chooses, not at whatever the weather made of it. `prescribed:` holds
the cells of a region (or a whole patch) at a fixed kelvin value, inside the same solve as the
cells around it -- a Dirichlet boundary, not a paint-over.

What would be wrong, and which test catches it:

* the held cells drift with the weather -> exactly 340.0 K, float32, at t0, between ticks and an
  hour later, under a clear summer noon and an overcast night alike;
* the pin is applied after the solve instead of inside it -> the free neighbours would not feel
  it: a cell next to a hot region must read warmer than the same cell with no region, already at
  t0 (the spin-up held it too), and less so further away;
* the pin leaks into the balance of cells it does not hold -> with lateral conduction off, every
  free cell is bit-identical to the unprescribed surface;
* the pin touches the optics -> the cells' emissivity is the material's, unchanged.

docs/physics-model.md §6.1, §6.4, §6.6; ADR 0087, ADR 0094, ADR 0123
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from irsim.config.scene import PrescribedSpec, SurfaceSpec, load_scene_config
from irsim.scene import Scene
from irsim.thermal.conduction import ConductionOperator

#: Two weathers as different as the library has: a clear summer noon and an overcast night.
WEATHERS = {
    "clear_noon": (
        "weather/clear_midlat_summer_48h.csv",
        "clear_dry",
        "2024-06-21T12:00:00Z",
    ),
    "overcast_night": (
        "weather/overcast_still_48h.csv",
        "overcast",
        "2024-06-22T01:00:00Z",
    ),
}

SCENE = """
schema_version: 16
scene:
  name: prescribed_test
  weather_file: {weather}
  atmosphere_preset: us_standard_clear
  environment_preset: {environment}
  site: {{latitude_deg: 45.0, longitude_deg: 30.0, altitude_m: 120.0}}
  start_utc: "{start}"
  thermal:
    spin_up_hours: 12.0
    tick_s: 60.0
    surfaces:
      - name: pipe
        material: rusted_steel
        tilt_deg: 0.0
        lateral_conduction: {lateral}
        patch:
          origin_m: [0.0, 0.0, 0.0]
          u_axis: [1.0, 0.0, 0.0]
          v_axis: [0.0, 1.0, 0.0]
          n_u: 12
          n_v: 4
          du_m: 0.05
          dv_m: 0.05
          thickness_m: 0.01
          frame: world
          prim_path: /World/pipe
{prescribed}
      - name: wall
        material: concrete
        tilt_deg: 90.0
"""

#: The cells whose centre is in u ∈ [0.20, 0.30] m: columns 4 and 5 of 12, every row.
HOT_SPOT = "        prescribed: [{temperature_k: 360.0, region_m: [0.20, 0.30, 0.0, 0.20]}]"


def _scene(
    tmp_path: pathlib.Path,
    weather: str = "clear_noon",
    prescribed: str = "",
    lateral: bool = True,
) -> Scene:
    path_weather, environment, start = WEATHERS[weather]
    text = SCENE.format(
        weather=path_weather,
        environment=environment,
        start=start,
        lateral=str(lateral).lower(),
        prescribed=prescribed,
    )
    path = tmp_path / f"{weather}_{bool(prescribed)}_{lateral}.yaml"
    path.write_text(text)
    return Scene.from_file(path)


def _times(scene: Scene) -> list[float]:
    """t0, a third of the way through the first tick, and an hour and a bit later."""
    t0 = scene.t0_s
    return [t0, t0 + 20.0, t0 + 3600.0 + 17.5]


# --- the whole surface ---------------------------------------------------------------------------


@pytest.mark.parametrize("weather", sorted(WEATHERS))
def test_a_prescribed_surface_reads_its_temperature_in_every_cell_at_every_time(
    tmp_path: pathlib.Path, weather: str
) -> None:
    scene = _scene(tmp_path, weather, "        prescribed: [{temperature_k: 340.0}]")
    fld = scene.surface_fields["pipe"]
    for t in _times(scene):
        fld.advance_to(t)
        cells = fld.temperature_at(t)
        assert cells.dtype == np.float32  # CLAUDE.md #2
        assert np.all(cells == np.float32(340.0)), (t, cells.min(), cells.max())
    # The per-prim value the render bridge carries is held too, so a pixel outside the patch
    # cannot show the balance temperature beside a prescribed one.
    for t in _times(scene):
        scene.thermal.advance_to(t)
        assert scene.surface_temperature_k("pipe", t) == 340.0
    np.testing.assert_array_equal(fld.pinned_k, np.full(48, 340.0))


def test_the_weather_really_differed_so_the_hold_is_not_a_coincidence(
    tmp_path: pathlib.Path,
) -> None:
    """The unprescribed surface in the same scenes is tens of kelvin apart between the two."""
    noon = _scene(tmp_path, "clear_noon", "        prescribed: [{temperature_k: 340.0}]")
    night = _scene(tmp_path, "overcast_night", "        prescribed: [{temperature_k: 340.0}]")
    free_noon = noon.surface_temperature_k("wall", noon.t0_s)
    free_night = night.surface_temperature_k("wall", night.t0_s)
    assert abs(free_noon - free_night) > 5.0, (free_noon, free_night)


# --- a region: the anomaly and its halo ----------------------------------------------------------


def test_a_region_holds_only_its_own_cells(tmp_path: pathlib.Path) -> None:
    scene = _scene(tmp_path, prescribed=HOT_SPOT)
    fld = scene.surface_fields["pipe"]
    held = np.isfinite(fld.pinned_k)
    image = held.reshape(fld.patch.shape)
    assert image[:, 4:6].all() and not image[:, :4].any() and not image[:, 6:].any()
    for t in _times(scene):
        fld.advance_to(t)
        cells = fld.temperature_at(t)
        assert np.all(cells[held] == np.float32(360.0))
        assert np.all(cells[~held] < np.float32(360.0))
    # A region is part of the surface, so the per-prim value stays the solved one.
    assert scene.surface_temperature_k("pipe", scene.t0_s) != 360.0


def test_the_neighbours_feel_the_hot_spot_through_the_solve_already_at_t0(
    tmp_path: pathlib.Path,
) -> None:
    """Steel conducts: the column next to the region is warmed, and less so further out.

    Fails if the pin were a paint-over after the solve (no halo), or if the spin-up had not
    held it (no halo at t0).
    """
    plain = _scene(tmp_path).surface_fields["pipe"]
    hot = _scene(tmp_path, prescribed=HOT_SPOT).surface_fields["pipe"]
    t0 = plain.t0_s
    gap = (hot.temperature_image(t0) - plain.temperature_image(t0)).astype(np.float64)
    row = gap.mean(axis=0)
    assert row[3] > 1.0 and row[6] > 1.0, row  # the adjacent columns, either side
    assert row[3] > row[2] > row[1] > row[0] > 0.0, row
    assert row[6] > row[7] > row[8] > row[9] > 0.0, row


def test_without_conduction_the_free_cells_are_untouched_to_the_bit(
    tmp_path: pathlib.Path,
) -> None:
    plain = _scene(tmp_path, lateral=False).surface_fields["pipe"]
    hot = _scene(tmp_path, prescribed=HOT_SPOT, lateral=False).surface_fields["pipe"]
    free = ~np.isfinite(hot.pinned_k)
    for t in _times(hot):
        plain.advance_to(t)
        hot.advance_to(t)
        assert np.array_equal(hot.temperature_at(t)[free], plain.temperature_at(t)[free])


def test_the_optics_are_the_materials_and_not_the_prescriptions(tmp_path: pathlib.Path) -> None:
    plain = _scene(tmp_path).surface_fields["pipe"]
    hot = _scene(tmp_path, prescribed=HOT_SPOT).surface_fields["pipe"]
    for name in ("emissivity", "solar_absorptivity", "heat_capacity_j_m2_k"):
        np.testing.assert_array_equal(
            getattr(hot.field.properties, name), getattr(plain.field.properties, name)
        )


def test_an_unprescribed_surface_has_no_pin(tmp_path: pathlib.Path) -> None:
    assert _scene(tmp_path).surface_fields["pipe"].pinned_k is None


# --- the Dirichlet row ---------------------------------------------------------------------------


def test_a_fixed_row_returns_its_right_hand_side_and_couples_its_neighbour_to_it() -> None:
    """Two cells, one link, cell 0 fixed: the solve hands back rhs[0], and cell 1 is the
    backward-Euler blend of its own rhs with *that* value -- not with one the step has moved."""
    k = 10.0
    op = ConductionOperator(np.array([[0.0, k], [k, 0.0]]), np.array([1.0, 1.0]))
    c, dt = np.array([1000.0, 1000.0]), 60.0
    lu = spla.splu(op.implicit_matrix(c, dt, fixed=[True, False]))
    rhs = np.array([400.0, 300.0])
    out = lu.solve(rhs)
    assert out[0] == 400.0
    a = dt * k / (c[1] * 1.0)
    assert out[1] == pytest.approx((rhs[1] + a * 400.0) / (1.0 + a), abs=1e-12)


# --- the schema ----------------------------------------------------------------------------------


def test_the_schema_refuses_what_would_give_a_cell_two_temperatures() -> None:
    patch = {
        "origin_m": [0.0, 0.0, 0.0],
        "u_axis": [1.0, 0.0, 0.0],
        "v_axis": [0.0, 1.0, 0.0],
        "n_u": 4,
        "n_v": 4,
        "du_m": 0.1,
        "dv_m": 0.1,
    }
    ok = SurfaceSpec(
        name="p", material="rusted_steel", patch=patch, prescribed=[{"temperature_k": 350.0}]
    )
    assert ok.prescribed == [PrescribedSpec(temperature_k=350.0)]
    with pytest.raises(ValueError, match="needs a `patch:`"):
        SurfaceSpec(name="p", material="rusted_steel", prescribed=[{"temperature_k": 350.0}])
    with pytest.raises(ValueError, match="layered"):
        SurfaceSpec(
            name="p",
            material="rusted_steel",
            patch=patch,
            layers=3,
            prescribed=[{"temperature_k": 350.0}],
        )
    with pytest.raises(ValueError, match="two authorities"):
        SurfaceSpec(
            name="p",
            material="asphalt_dry",
            patch=patch,
            water={"depth_mm": 5.0},
            prescribed=[{"temperature_k": 350.0}],
        )
    with pytest.raises(ValueError, match="whole surface"):
        SurfaceSpec(
            name="p",
            material="rusted_steel",
            patch=patch,
            prescribed=[
                {"temperature_k": 350.0},
                {"temperature_k": 360.0, "region_m": [0.0, 0.1, 0.0, 0.1]},
            ],
        )
    with pytest.raises(ValueError, match="overlap"):
        SurfaceSpec(
            name="p",
            material="rusted_steel",
            patch=patch,
            prescribed=[
                {"temperature_k": 350.0, "region_m": [0.0, 0.2, 0.0, 0.2]},
                {"temperature_k": 360.0, "region_m": [0.1, 0.3, 0.1, 0.3]},
            ],
        )
    with pytest.raises(ValueError, match="u1 > u0"):
        PrescribedSpec(temperature_k=350.0, region_m=(0.2, 0.1, 0.0, 0.1))
    with pytest.raises(ValueError):
        PrescribedSpec(temperature_k=0.0)
    # Two regions that only touch along an edge are two anomalies, not one conflict.
    SurfaceSpec(
        name="p",
        material="rusted_steel",
        patch=patch,
        prescribed=[
            {"temperature_k": 350.0, "region_m": [0.0, 0.1, 0.0, 0.1]},
            {"temperature_k": 360.0, "region_m": [0.1, 0.2, 0.0, 0.1]},
        ],
    )


def test_a_region_smaller_than_a_cell_is_refused_rather_than_lost(tmp_path: pathlib.Path) -> None:
    """[0.201, 0.224] m lies between the cell centres at 0.175 and 0.225 m along u."""
    tiny = "        prescribed: [{temperature_k: 360.0, region_m: [0.201, 0.224, 0.0, 0.2]}]"
    with pytest.raises(ValueError, match="no cell centre"):
        _scene(tmp_path, prescribed=tiny)


def test_a_v15_scene_still_loads_and_prescribes_nothing(tmp_path: pathlib.Path) -> None:
    """v16 only adds an optional field, so the readable floor does not move."""
    path_weather, environment, start = WEATHERS["clear_noon"]
    text = SCENE.format(
        weather=path_weather, environment=environment, start=start, lateral="true", prescribed=""
    ).replace("schema_version: 16", "schema_version: 15")
    path = tmp_path / "v15.yaml"
    path.write_text(text)
    config = load_scene_config(path)
    assert config.schema_version == 15
    assert config.scene.thermal is not None
    assert all(not s.prescribed for s in config.scene.thermal.surfaces)
