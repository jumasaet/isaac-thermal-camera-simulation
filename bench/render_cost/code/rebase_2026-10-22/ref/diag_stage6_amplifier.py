"""Where agc_information turns ~2000 one-DN flips into tens of display codes, and whether option
(b)'s host stage 6 is run_frame's when both get the same display_dn.

1. chained: the four poses of each window, fed run_frame's display_dn, carrying the host branch's
   own damper state from a fresh start (what FullGpu does) -> must equal run_frame's display8.
2. split: on the first pose of each window (no carried state, so the deviation is all input),
   rebuild agc_information's y = T[idx] + g * T'[idx] * high on both sides and attribute
   y_dev - y_cpu (in display codes, x255) to
       transfer   T_dev vs T_cpu at the cpu's idx and high (base and detail term)
       index      floor(low) moving a bin, at T_dev
       high       the high-pass itself, at T_dev and idx_dev
   Reports, at the worst pixel and over all pixels, which term carries the difference, and the
   |high| and slope at the worst pixel.

Usage: python.sh diag_stage6_amplifier.py <dump dir> [out.json]
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

from irsim.config.loader import load_sensor_config
from irsim.isp.damping import TransferDamper, damping_over
from irsim.isp.display import run_display_branch
from irsim.isp.information import bilateral_split

SENSOR = pathlib.Path("/home/jumasaet/isaac-thermal-camera-simulation/configs/sensors/flir_boson_640_lwir.yaml")
spec = load_sensor_config(SENSOR).sensor
ISP, BITS, FR = spec.isp, spec.fpa.bit_depth, float(spec.fpa.frame_rate_hz)
G = 1.0 + ISP.dde_gain


def transfer(dn, sigma):
    got = {}

    def hook(lut):
        got["t"] = np.asarray(lut, np.float64)
        return got["t"]

    d = run_display_branch(dn, ISP.model_copy(update=dict(agc_damping=0.0)), BITS, noise_sigma_dn=sigma, lut_hook=hook)
    return got["t"], d.display8


def y_of(t, idx, high):
    return np.clip(t[idx] + G * np.gradient(t)[idx] * high, 0.0, 1.0) * 255.0


def main(d: pathlib.Path, out: pathlib.Path | None) -> None:
    files = sorted(d.glob("*.npz"))
    windows = defaultdict(list)
    for f in files:
        v, s, j = f.stem.rsplit("_", 2)
        windows[(v, int(s))].append((int(j), f))
    chained, split = [], []
    for (v, s), lst in sorted(windows.items()):
        lst.sort()
        prev = t_prev = None
        eq = []
        for j, f in lst:
            z = np.load(f)
            t_s = float(z["t_s"])
            damper = None
            if ISP.agc_damping > 0.0:
                dt = 1.0 / FR if t_prev is None or t_s - t_prev <= 0.0 else t_s - t_prev
                damper = TransferDamper(damping_over(ISP.agc_damping, FR, dt), prev)
            disp = run_display_branch(z["cpu_display_dn"], ISP, BITS, noise_sigma_dn=float(z["dev_sigma"]), lut_hook=damper)
            prev, t_prev = damper.previous, t_s
            eq.append(bool(np.array_equal(disp.display8, z["cpu_display8"])))
        chained.append({"window": f"{v}@{s}", "bit_equal_per_frame": eq})
        print(f"[chained {v}@{s}] host branch fed run_frame's display_dn == run_frame display8: {eq}", flush=True)

        z = np.load(lst[0][1])
        sigma = float(z["cpu_sigma_tvh"])
        cdn, ddn = z["cpu_display_dn"], z["dev_display_dn"]
        tc, c8 = transfer(cdn, sigma)
        td, d8 = transfer(ddn, sigma)
        lc, hc = bilateral_split(cdn.astype(np.float64), ISP.smoothing_sigma_dn)
        ld, hd = bilateral_split(ddn.astype(np.float64), ISP.smoothing_sigma_dn)
        top = float(2**BITS - 1)
        ic = np.floor(np.clip(lc, 0.0, top)).astype(np.int64)
        idd = np.floor(np.clip(ld, 0.0, top)).astype(np.int64)
        y_c = y_of(tc, ic, hc)
        y_t = y_of(td, ic, hc)       # only the transfer changed
        y_ti = y_of(td, idd, hc)     # + the bin index
        y_d = y_of(td, idd, hd)      # + the high-pass = the device side
        total = y_d - y_c
        parts = {"transfer": y_t - y_c, "index": y_ti - y_t, "high": y_d - y_ti}
        w = np.unravel_index(np.argmax(np.abs(total)), total.shape)
        obs = np.abs(d8[..., 0].astype(np.int16) - c8[..., 0].astype(np.int16))
        changed = cdn != ddn
        near = np.zeros_like(changed)
        for dy in (-2, -1, 0, 1, 2):  # the bilateral's 5x5
            for dx in (-2, -1, 0, 1, 2):
                near |= np.roll(np.roll(changed, dy, 0), dx, 1)
        slope_c = np.gradient(tc) * 255.0
        row = {"window": f"{v}@{s}", "pose": lst[0][0],
               "rebuild_matches_display8_max_code": float(np.abs(np.round(y_c) - c8[..., 0]).max()),
               "observed_max_code": int(obs.max()), "y_total_max_codes": float(np.abs(total).max()),
               "max_codes_by_term": {k: float(np.abs(p).max()) for k, p in parts.items()},
               "sum_abs_codes_by_term": {k: float(np.abs(p).sum()) for k, p in parts.items()},
               "worst_px": [int(w[0]), int(w[1])],
               "worst_px_by_term": {k: float(p[w]) for k, p in parts.items()},
               "worst_px_dn_cpu_dev": [int(cdn[w]), int(ddn[w])], "worst_px_high_cpu": float(hc[w]),
               "worst_px_idx_cpu_dev": [int(ic[w]), int(idd[w])],
               "worst_px_slope_codes_per_dn_cpu_dev": [float(slope_c[ic[w]]), float(np.gradient(td)[idd[w]] * 255.0)],
               "transfer_max_abs_diff_codes": float(np.abs(td - tc).max() * 255.0),
               "transfer_slope_max_abs_diff_codes_per_dn": float(np.abs(np.gradient(td) - np.gradient(tc)).max() * 255.0),
               "px_idx_moved": int((ic != idd).sum()),
               "px_off_by_gt1_code_outside_5x5_of_a_dn_change": int(((obs > 1) & ~near).sum()),
               "px_off_by_gt1_code": int((obs > 1).sum()),
               "slope_codes_per_dn_p50_p99_max_over_occupied_bins": [
                   float(np.percentile(np.abs(slope_c[np.unique(ic)]), q)) for q in (50, 99, 100)],
               "abs_high_p50_p99_max_dn": [float(np.percentile(np.abs(hc), q)) for q in (50, 99, 100)]}
        split.append(row)
        print(json.dumps(row), flush=True)
    if out is not None:
        out.write_text(json.dumps({"chained": chained, "split": split}, indent=1))
        print("wrote", out)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None)
