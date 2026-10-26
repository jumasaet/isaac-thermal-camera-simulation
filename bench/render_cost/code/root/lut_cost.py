#!/usr/bin/env python
"""What the cold clear-sky LUT costs and where, engine-free (no Kit): the bench's own sky model.

  lut_cost.py time     cold _clear_lut wall, N distinct times
  lut_cost.py lines    the same under line_profiler
"""
from __future__ import annotations

import pathlib
import statistics
import sys
import time

REPO = pathlib.Path("/home/jumasaet/isaac-thermal-camera-simulation")
BENCH = pathlib.Path("/tmp/irsim_render_cost")
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from irsim.atmosphere import LayeredAtmosphere, load_atmosphere_preset  # noqa: E402
from irsim.atmosphere.layered import ExponentialSum, column_length  # noqa: E402
from irsim.atmosphere.sky import SkyModel  # noqa: E402
from irsim.config.environment import load_environment_preset  # noqa: E402
from irsim.config.loader import load_sensor_config  # noqa: E402
from irsim.radiometry.lut_files import (  # noqa: E402
    load_band_lut_for_config,
    load_band_response_for_config,
)
from irsim.thermal.weather_io import load_weather_csv  # noqa: E402


def build() -> tuple[SkyModel, float]:
    sensor = load_sensor_config(BENCH / "sensor_ss1.yaml")
    band = sensor.sensor.band.band_id
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    response = load_band_response_for_config(sensor, REPO / "data")
    weather = load_weather_csv(REPO / "data" / "weather" / "clear_midlat_summer_48h.csv")
    atm = LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), weather, {band: lut}, {band: response}
    )
    sky = SkyModel(atm, load_environment_preset("clear_dry"), band, lut, sensor.sensor.quantity)
    return sky, 26.0 * 3600.0  # the bench starts 2024-06-21T02:00Z; any in-range time will do


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "time"
if __name__ == "__main__" and mode in ("time", "lines"):
    sky, t0 = build()
    sky._clear_lut(t0 - 1.0)  # warm imports, class weights, LUT tables
    if mode == "time":
        walls = []
        for i in range(12):
            a = time.perf_counter()
            sky._clear_lut(t0 + i / 30.0)
            walls.append(time.perf_counter() - a)
        print(f"cold _clear_lut: {1e3 * statistics.fmean(walls):.1f} ms "
              f"+-{1e3 * statistics.stdev(walls):.1f} (min {1e3 * min(walls):.1f}), "
              f"n_terms={sky.atmosphere.exponential_sum(sky.band, t0).n_terms}")
    else:
        sys.path.append(str(BENCH / "pydeps"))
        from line_profiler import LineProfiler

        lp = LineProfiler()
        for fn in (SkyModel._clear_lut, SkyModel._tilt_lut.__func__ if hasattr(SkyModel._tilt_lut, "__func__") else SkyModel._tilt_lut,
                   LayeredAtmosphere.path_radiance, LayeredAtmosphere.exponential_sum,
                   ExponentialSum.path_radiance_per_class, ExponentialSum.optical_depths,
                   column_length):
            lp.add_function(fn)
        lp.enable()
        for i in range(6):
            sky._clear_lut(t0 + i / 30.0)
        lp.disable()
        lp.print_stats(output_unit=1e-3, stripzeros=True)


# ---- prototypes: exact rewrites of the 181 scalar calls, compared bytewise --------------------
import math  # noqa: E402

from irsim.atmosphere.layered import _scaled  # noqa: E402
from irsim.atmosphere.sky import ELEVATION_FLOOR_RAD, ELEVATION_GRID_DEG  # noqa: E402
from irsim.radiometry.band_integration import simpson  # noqa: E402

ELS = [max(math.radians(d), ELEVATION_FLOOR_RAD) for d in ELEVATION_GRID_DEG]


def scalar(sky, t):
    return np.array([sky._atm.sky_radiance(sky._band, t, el, sky._q) for el in ELS])


def v_hoist(sky, t, n_steps=4000, u_max=40.0, s_max_m=300e3):
    """Hoist only: one exponential_sum, one s_dense. Everything else as the scalar path."""
    atm, band, q = sky._atm, sky._band, sky._q
    es = atm.exponential_sum(band, t)
    lbh = atm._lb_of_height(band, t, q)
    s_dense = np.concatenate([[0.0], np.logspace(-3, math.log10(s_max_m), 20000)])
    n = n_steps if n_steps % 2 == 0 else n_steps + 1
    res = np.empty(len(ELS))
    for i, el in enumerate(ELS):
        st = math.sin(el)
        out = np.zeros(es.n_terms)
        od_dense = es.optical_depths(s_dense, el)
        for k in range(es.n_terms):
            od_end = float(od_dense[k, -1])
            if od_end <= 0.0:
                continue
            u = np.linspace(0.0, min(od_end, u_max), n + 1)
            s_of_u = np.interp(u, od_dense[k], s_dense)
            lb = lbh(s_of_u * st)
            out[k] = float(simpson(lb * np.exp(-u), float(u[1] - u[0])))
        res[i] = float(np.dot(es.weights, out))
    return res


def od_dedup(es, s_dense, el):
    """optical_depths(s_dense, el) for a scalar el: sin once, one column per distinct height."""
    cols: dict[float, np.ndarray] = {}

    def col(h: float) -> np.ndarray:
        if h not in cols:
            cols[h] = np.asarray(column_length(s_dense, el, h), dtype=np.float64)
        return cols[h]

    aer = _scaled(es.gamma_aerosol, col(es.aerosol_scale_height_m))
    out = np.empty((es.n_terms, s_dense.size))
    for k in range(es.n_terms):
        out[k] = _scaled(float(es.gamma_0[k]), col(float(es.scale_heights_m[k]))) + aer
    return out


def v_dedup(sky, t, n_steps=4000, u_max=40.0, s_max_m=300e3, share_u=False, batch_lb=False):
    atm, band, q = sky._atm, sky._band, sky._q
    es = atm.exponential_sum(band, t)
    lbh = atm._lb_of_height(band, t, q)
    s_dense = np.concatenate([[0.0], np.logspace(-3, math.log10(s_max_m), 20000)])
    n = n_steps if n_steps % 2 == 0 else n_steps + 1
    ucache: dict[float, tuple] = {}
    res = np.empty(len(ELS))
    for i, el in enumerate(ELS):
        st = math.sin(el)
        out = np.zeros(es.n_terms)
        od_dense = od_dedup(es, s_dense, el)
        rows = []
        for k in range(es.n_terms):
            od_end = float(od_dense[k, -1])
            if od_end <= 0.0:
                continue
            u_top = min(od_end, u_max)
            if share_u:
                hit = ucache.get(u_top)
                if hit is None:
                    u = np.linspace(0.0, u_top, n + 1)
                    hit = ucache[u_top] = (u, np.exp(-u), float(u[1] - u[0]))
                u, e, du = hit
            else:
                u = np.linspace(0.0, u_top, n + 1)
                e, du = np.exp(-u), float(u[1] - u[0])
            s_of_u = np.interp(u, od_dense[k], s_dense)
            if batch_lb:
                rows.append((k, s_of_u * st, e, du))
            else:
                out[k] = float(simpson(lbh(s_of_u * st) * e, du))
        if batch_lb and rows:
            lb = lbh(np.stack([r[1] for r in rows]))
            for j, (k, _, e, du) in enumerate(rows):
                out[k] = float(simpson(lb[j] * e, du))
        res[i] = float(np.dot(es.weights, out))
    return res



def v_grid(sky, t, n_steps=4000, u_max=40.0, s_max_m=300e3, chunk=None, share_u=True):
    """hoist + dedup columns + ONE lb_of_height lookup for every (elevation, class) row."""
    atm, band, q = sky._atm, sky._band, sky._q
    es = atm.exponential_sum(band, t)
    lbh = atm._lb_of_height(band, t, q)
    s_dense = np.concatenate([[0.0], np.logspace(-3, math.log10(s_max_m), 20000)])
    n = n_steps if n_steps % 2 == 0 else n_steps + 1
    ucache: dict[float, tuple] = {}
    per = np.zeros((len(ELS), es.n_terms))
    heights = np.empty((len(ELS) * es.n_terms, n + 1))
    rows = []
    for i, el in enumerate(ELS):
        st = math.sin(el)
        od_dense = od_dedup(es, s_dense, el)
        for k in range(es.n_terms):
            od_end = float(od_dense[k, -1])
            if od_end <= 0.0:
                continue
            u_top = min(od_end, u_max)
            hit = ucache.get(u_top) if share_u else None
            if hit is None:
                u = np.linspace(0.0, u_top, n + 1)
                hit = (u, np.exp(-u), float(u[1] - u[0]))
                if share_u:
                    ucache[u_top] = hit
            u, e, du = hit
            heights[len(rows)] = np.interp(u, od_dense[k], s_dense) * st
            rows.append((i, k, e, du))
    m = len(rows)
    step = m if chunk is None else chunk
    for a in range(0, m, step):
        lb = lbh(heights[a:a + step])
        for j, (i, k, e, du) in enumerate(rows[a:a + step]):
            per[i, k] = float(simpson(lb[j] * e, du))
    return np.array([float(np.dot(es.weights, per[i])) for i in range(len(ELS))])


def v_naive_batch(sky, t, n_steps=4000, u_max=40.0, s_max_m=300e3):
    """All 181 elevations through ONE optical_depths call (the literal vectorisation)."""
    atm, band, q = sky._atm, sky._band, sky._q
    es = atm.exponential_sum(band, t)
    lbh = atm._lb_of_height(band, t, q)
    s_dense = np.concatenate([[0.0], np.logspace(-3, math.log10(s_max_m), 20000)])
    n = n_steps if n_steps % 2 == 0 else n_steps + 1
    els = np.asarray(ELS)
    od_all = es.optical_depths(s_dense[None, :], els[:, None])  # (K, N, 20001)
    res = np.empty(len(ELS))
    for i, el in enumerate(ELS):
        st = math.sin(el)
        out = np.zeros(es.n_terms)
        for k in range(es.n_terms):
            od_end = float(od_all[k, i, -1])
            if od_end <= 0.0:
                continue
            u = np.linspace(0.0, min(od_end, u_max), n + 1)
            s_of_u = np.interp(u, od_all[k, i], s_dense)
            out[k] = float(simpson(lbh(s_of_u * st) * np.exp(-u), float(u[1] - u[0])))
        res[i] = float(np.dot(es.weights, out))
    return res


def proto() -> None:
    sky, t0 = build()
    ref = scalar(sky, t0)
    variants = {
        "scalar (181 sky_radiance calls)": scalar,
        "hoist es + s_dense": v_hoist,
        "naive batch optical_depths": v_naive_batch,
        "hoist + dedup columns": v_dedup,
        "  + shared u/exp(-u)": lambda s, t: v_dedup(s, t, share_u=True),
        "  + one lb lookup per elevation": lambda s, t: v_dedup(s, t, share_u=True, batch_lb=True),
        "  + one lb lookup for the grid": v_grid,
        "  + lb lookup in chunks of 64 rows": lambda s, t: v_grid(s, t, chunk=64),
        "  grid, no shared u": lambda s, t: v_grid(s, t, share_u=False),
    }
    for name, fn in variants.items():
        same = True
        walls = []
        for i in range(5):
            t = t0 + 600.0 * i
            want = scalar(sky, t) if name != "scalar (181 sky_radiance calls)" else None
            a = time.perf_counter()
            got = fn(sky, t)
            walls.append(time.perf_counter() - a)
            if want is not None:
                same &= got.tobytes() == want.tobytes()
        print(f"{1e3 * statistics.fmean(walls):8.1f} ms (min {1e3 * min(walls):6.1f})  "
              f"bytes==scalar: {same}   {name}")
    a = time.perf_counter()
    tilt = SkyModel._tilt_lut(ref)
    print(f"{1e3 * (time.perf_counter() - a):8.1f} ms  _tilt_lut (not part of the 181 calls)")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "proto":
    proto()


def rest() -> None:
    """Where the remaining ~160 ms sits, and what a hoisted weather sample would add."""
    sky, t0 = build()
    atm, band, q = sky._atm, sky._band, sky._q
    a = time.perf_counter()
    for _ in range(200):
        atm._weather.at(t0)
    print(f"weather.at: {1e6 * (time.perf_counter() - a) / 200:.0f} us per call")
    a = time.perf_counter()
    for _ in range(50):
        atm.exponential_sum(band, t0)
    print(f"exponential_sum: {1e6 * (time.perf_counter() - a) / 50:.0f} us per call")
    lut = atm._luts[band]
    p = atm._preset.profile
    real = atm._lb_of_height

    def hoisted(band_, t_s, quantity):
        t_air = atm._weather.at(t_s).t_air_k

        def lb(height_m):
            h = np.minimum(np.asarray(height_m, dtype=np.float64), p.tropopause_m)
            return np.asarray(lut.lookup(np.asarray(t_air - p.lapse_rate_k_per_m * h, dtype=np.float64), quantity), dtype=np.float64)

        return lb

    for label, fn in (("per-elevation lb, stock closure", real), ("per-elevation lb, weather hoisted", hoisted)):
        atm._lb_of_height = fn
        walls, same = [], True
        for i in range(5):
            t = t0 + 600.0 * i
            a = time.perf_counter()
            got = v_dedup(sky, t, batch_lb=True)
            walls.append(time.perf_counter() - a)
            atm._lb_of_height = real
            same &= got.tobytes() == scalar(sky, t).tobytes()
            atm._lb_of_height = fn
        print(f"{1e3 * statistics.fmean(walls):8.1f} ms  bytes==scalar: {same}  {label}")
    atm._lb_of_height = real
    sys.path.append(str(BENCH / "pydeps"))
    from line_profiler import LineProfiler

    lp = LineProfiler()
    lp.add_function(v_dedup)
    lp.add_function(od_dedup)
    lp.add_function(column_length)
    lp.enable()
    for i in range(5):
        v_dedup(sky, t0 + 600.0 * i, batch_lb=True)
    lp.disable()
    import io, re
    buf = io.StringIO()
    lp.print_stats(stream=buf, output_unit=1e-3, stripzeros=True)
    for line in buf.getvalue().splitlines():
        m = re.match(r"\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.*)", line)
        if line.startswith(("Function", "Total")):
            print(line)
        elif m and float(m.group(3)) / 5 >= 2.0:
            print(f"  L{m.group(1):>4} {float(m.group(3)) / 5:7.1f} ms  {m.group(6).strip()[:100]}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "rest":
    rest()
