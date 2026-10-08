"""SkyModel (MS.2): overcast limit to 1e-9, coldest at zenith and the horizon toward T_air, the
tilt LUT against scipy quad, the elevation LUT within 0.5 K of the layered model, the cos^q form
measured (not authored), broadband delegation, and the object-only constructor.

Also the one place the model skips work: under an opaque overcast the tilt blend does not build
the clear-sky LUT whose weight is exactly zero. That is asserted on the bytes of the result, and
the long path is asserted to still run one ulp short of total cover and under a total cover that
is not opaque."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from irsim.atmosphere import LayeredAtmosphere, load_atmosphere_preset
from irsim.atmosphere.sky import ELEVATION_GRID_DEG, SkyModel, azimuth_kernel
from irsim.config.environment import load_environment_preset
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import load_spectral_response
from irsim.thermal import WeatherSample, WeatherSeries
from irsim.thermal.longwave import longwave_down_from_sample

T_AIR = 288.15


def _weather(cloud: float, rh: float = 0.3) -> WeatherSeries:
    return WeatherSeries.constant(
        WeatherSample(T_AIR, rh, 1.0, cloud, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )


@pytest.fixture(scope="module")
def lwir(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    p = tmp_path_factory.mktemp("sky") / "tophat_7p5_14.csv"
    p.write_text("# top-hat\n7.5,1.0\n14.0,1.0\n")
    r = load_spectral_response(p)
    return r, BandLUT.build(r)


def _sky(  # type: ignore[no-untyped-def]
    lwir,
    cloud: float = 0.0,
    preset: str = "us_standard_clear",
    rh: float = 0.3,
    env: str = "clear_dry",
) -> SkyModel:
    r, lut = lwir
    atm = LayeredAtmosphere(
        load_atmosphere_preset(preset), _weather(cloud, rh), {"lwir": lut}, {"lwir": r}
    )
    return SkyModel(atm, load_environment_preset(env), "lwir", lut)


@pytest.mark.slow  # GT.1: over a second on its own
def test_overcast_limit_and_cloud_blend(lwir) -> None:  # type: ignore[no-untyped-def]
    """RH = 1: the cloud base (LCL) is at the surface, so thick overcast reads T_air exactly."""
    _, lut = lwir
    sky = _sky(lwir, cloud=1.0, rh=1.0)
    lb_air = float(lut.lookup(np.float64(T_AIR))[()])
    for tilt in (0.0, 45.0, 90.0, 135.0, 180.0):
        assert float(sky.effective_radiance(0.0, math.radians(tilt))) == pytest.approx(
            lb_air, rel=1e-9
        )
    for el in (0.0, 15.0, 90.0):
        assert float(sky.radiance(0.0, math.radians(el))) == pytest.approx(lb_air, rel=1e-9)
        assert float(sky.apparent_temperature_k(0.0, math.radians(el))) == pytest.approx(
            T_AIR, abs=1e-3
        )
    half = _sky(lwir, cloud=0.5, rh=1.0)
    clear = _sky(lwir, cloud=0.0, rh=1.0)
    el = math.radians(30.0)
    expect = 0.5 * float(clear.radiance(0.0, el)) + 0.5 * lb_air
    assert float(half.radiance(0.0, el)) == pytest.approx(expect, rel=1e-12)


TILTS_RAD = np.radians([[0.0, 30.0, 90.0], [135.0, 179.5, 180.0]])


def _count_clear_lut_calls(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Every time a SkyModel asks for the clear-sky LUT, by the time it asked for."""
    calls: list[float] = []
    inner = SkyModel._clear_lut

    def counted(self: SkyModel, t_s: float):  # type: ignore[no-untyped-def]
        calls.append(float(t_s))
        return inner(self, t_s)

    monkeypatch.setattr(SkyModel, "_clear_lut", counted)
    return calls


@pytest.mark.slow  # GT.1: over a second on its own
def test_opaque_overcast_skips_the_clear_sky_bit_for_bit(lwir, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """c ε = 1: the clear term's weight is exactly 0.0, so ``effective_radiance`` never builds the
    clear-sky LUT -- and returns what the long path returns **bit for bit**, not within a
    tolerance: it is the same number with ``0.0 * L_clear,eff`` not added to it.

    RH = 0.3 puts the base aloft, so L_B(T_base) is not L_B(T_air): a short path that returned
    the wrong cloud radiance, or the right one in the wrong shape or dtype, fails the comparison.
    """
    _, lut = lwir
    sky = _sky(lwir, cloud=1.0)
    calls = _count_clear_lut_calls(monkeypatch)

    short = sky.effective_radiance(0.0, TILTS_RAD)
    short_scalar = sky.effective_radiance(0.0, 0.0)
    short_view = sky.effective_radiance_from_sky_view(0.0, np.array([0.0, 0.25, 1.0]))
    assert calls == [], "the clear-sky LUT was built for a term whose weight is exactly zero"

    c, l_cloud = sky._cloud(0.0)
    assert c == 1.0

    def long_path(tilt_rad: object) -> np.ndarray:
        return np.asarray(
            (1.0 - c) * sky.effective_clear_radiance(0.0, tilt_rad) + c * l_cloud, dtype=np.float64
        )

    for got, tilt in (
        (short, TILTS_RAD),
        (short_scalar, 0.0),
        (short_view, np.arccos(2.0 * np.array([0.0, 0.25, 1.0]) - 1.0)),
    ):
        want = long_path(tilt)
        assert got.dtype == want.dtype == np.float64
        assert got.shape == want.shape
        assert got.tobytes() == want.tobytes()
    assert calls, "the reference never ran the long path, so the comparison proved nothing"

    assert np.all(short < float(lut.lookup(np.float64(T_AIR))[()])), "the base is aloft, not T_air"
    # what the long path did with a bad tilt, the short one still does
    with pytest.raises(ValueError, match="tilt"):
        sky.effective_radiance(0.0, math.radians(190.0))
    assert np.isnan(sky.effective_radiance(0.0, np.nan))


@pytest.mark.slow  # GT.1: over a second on its own
@pytest.mark.parametrize(
    ("cloud", "env", "clear_sky_shows"),
    [
        # one ulp short of total cover: the weight is 1.1e-16, which is small and not zero
        (float(np.nextafter(1.0, 0.0)), "clear_dry", False),
        # total cover by a cloud that is not opaque (eps = 1 - exp(-0.79 x 12)): the clear sky
        # still leaks through, so "cloud_fraction == 1" alone must not take the short path
        (1.0, "scattered_cumulus", True),
        (0.5, "clear_dry", True),
    ],
)
def test_any_clear_weight_above_zero_still_integrates_the_clear_sky(  # type: ignore[no-untyped-def]
    lwir, monkeypatch, cloud: float, env: str, clear_sky_shows: bool
) -> None:
    sky = _sky(lwir, cloud=cloud, env=env)
    calls = _count_clear_lut_calls(monkeypatch)
    got = sky.effective_radiance(0.0, TILTS_RAD)
    assert calls == [0.0], "the clear term has weight > 0 and was not evaluated"

    c, l_cloud = sky._cloud(0.0)
    assert 1.0 - c > 0.0
    want = np.asarray(
        (1.0 - c) * sky.effective_clear_radiance(0.0, TILTS_RAD) + c * l_cloud, dtype=np.float64
    )
    assert got.tobytes() == want.tobytes()
    if clear_sky_shows:
        # the clear LWIR sky is colder than the cloud base at every tilt, so any of it in the
        # blend pulls the result below the cloud's own radiance
        assert np.all(got < l_cloud)


def test_coldest_at_zenith_and_horizon_toward_t_air(lwir) -> None:  # type: ignore[no-untyped-def]
    sky = _sky(lwir)
    els = np.radians([0.0, 1.0, 5.0, 15.0, 45.0, 90.0])
    t = sky.apparent_temperature_k(0.0, els)
    assert np.all(np.diff(t) < 0.0), t
    assert t[-1] < T_AIR - 40.0 and abs(t[0] - T_AIR) < 8.0
    assert sky.weather is sky.atmosphere.weather and sky.band == "lwir"


def test_tilt_lut_matches_scipy_quad_and_sky_view_indexing(lwir) -> None:  # type: ignore[no-untyped-def]
    sky = _sky(lwir)
    theta_grid = np.radians(ELEVATION_GRID_DEG)
    clear = sky.clear_radiance(0.0, theta_grid)

    def kernels_at(beta: float):  # type: ignore[no-untyped-def]
        def k(th: float) -> float:
            a = np.array(math.sin(beta) * math.cos(th))
            b = np.array(math.cos(beta) * math.sin(th))
            return float(azimuth_kernel(a, b)) * math.cos(th)

        def num(th: float) -> float:
            return k(th) * float(np.interp(th, theta_grid, clear))

        return k, num

    for beta_deg in (0.0, 30.0, 60.0, 90.0, 120.0, 170.0):
        beta = math.radians(beta_deg)
        kern, num = kernels_at(beta)
        # breakpoints: the LUT knots (piecewise-linear radiance) and the kernel's kink
        pts = sorted(
            {
                *theta_grid[1:-1].tolist(),
                *[k for k in (beta, math.pi - beta) if 0 < k < math.pi / 2],
            }
        )
        n_val = quad(num, 0.0, math.pi / 2, points=pts, limit=2000)[0]
        d_val = quad(kern, 0.0, math.pi / 2, points=pts, limit=2000)[0]
        if d_val > 0.0:
            assert float(sky.effective_clear_radiance(0.0, beta)) == pytest.approx(
                n_val / d_val, rel=1e-6
            )
    # facing up: the cosine-weighted hemispheric mean; vertical: half the sky, warmer (more horizon)
    up = float(sky.effective_clear_radiance(0.0, 0.0))
    wall = float(sky.effective_clear_radiance(0.0, math.pi / 2))
    assert wall > up
    assert float(sky.effective_radiance_from_sky_view(0.0, 1.0)) == pytest.approx(up, rel=1e-12)
    assert float(sky.effective_radiance_from_sky_view(0.0, 0.5)) == pytest.approx(wall, rel=1e-12)
    # analytic azimuth kernel: full circle when b >= a, none when b <= -a, half circle at b = 0
    assert float(azimuth_kernel(np.array(0.3), np.array(0.5))) == pytest.approx(2 * math.pi * 0.5)
    assert float(azimuth_kernel(np.array(0.3), np.array(-0.5))) == 0.0
    assert float(azimuth_kernel(np.array(1.0), np.array(0.0))) == pytest.approx(2.0)


@pytest.mark.parametrize("preset", ["us_standard_clear", "midlat_summer_humid", "tropical", "haze"])
def test_elevation_lut_fast_path_within_half_a_kelvin(lwir, preset: str) -> None:  # type: ignore[no-untyped-def]
    _, lut = lwir
    sky = _sky(lwir, preset=preset, rh=0.5)
    for deg in (5.0, 7.3, 12.0, 20.5, 33.0, 47.0, 61.0, 80.0, 89.0):
        direct = sky.atmosphere.apparent_sky_temperature_k("lwir", 0.0, math.radians(deg))
        fast = float(sky.apparent_temperature_k(0.0, math.radians(deg)))
        assert abs(fast - direct) < 0.5, (preset, deg, fast, direct)


def test_cos_q_form_is_derived_and_its_error_recorded(lwir) -> None:  # type: ignore[no-untyped-def]
    """ADR 0044: the §5.3(a) power law cannot follow the column model over 5-90 deg; the fit
    is derived, its error measured, and the elevation LUT is the fast path instead."""
    sky = _sky(lwir, rh=0.2)
    fit = sky.fit_cos_q(0.0)
    assert 40.0 < fit.delta_t_k < 120.0 and 0.05 < fit.q < 1.0
    assert fit.max_error_k > 0.5, "the spec's form is not within 0.5 K -- recorded, not hidden"
    assert fit.max_error_k < 15.0 and fit.rms_error_k < fit.max_error_k
    # the authored preset ranges (55-70 K clear) are the zenith depression the form implies
    assert fit.q < 0.5, "the column saturates toward the horizon faster than cos^0.5"


def test_broadband_delegation_and_constructor_guards(lwir) -> None:  # type: ignore[no-untyped-def]
    r, lut = lwir
    sky = _sky(lwir, cloud=0.2)
    expect = float(longwave_down_from_sample(sky.weather.at(0.0), 0.8, 290.0))
    assert float(sky.broadband_downwelling(0.0, 0.8, 290.0)) == expect
    assert expect > 250.0
    env = load_environment_preset("clear_dry")
    with pytest.raises(TypeError, match="never a scalar"):
        SkyModel(288.15, env, "lwir", lut)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="never a scalar"):
        SkyModel(_weather(0.0), env, "lwir", lut)  # type: ignore[arg-type]
    atm = LayeredAtmosphere(
        load_atmosphere_preset("haze"), _weather(0.0), {"lwir": lut}, {"lwir": r}
    )
    with pytest.raises(TypeError, match="EnvironmentSpec"):
        SkyModel(atm, {"sky": {}}, "lwir", lut)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="elevation"):
        sky.radiance(0.0, math.radians(95.0))
    with pytest.raises(ValueError, match="tilt"):
        sky.effective_radiance(0.0, math.radians(190.0))
