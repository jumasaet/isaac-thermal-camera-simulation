"""Why option (b)'s display8 is not run_frame's with the Boson's information_based AGC (run B).

Reads the stage-6 inputs `run_ref.py` dumps under DIAG_DUMP (both sides, every checked pose) and
re-runs `irsim.isp.display.run_display_branch` with the inputs swapped one at a time:

    sanity      each side's own inputs reproduce that side's display8 (else the dump is wrong)
    sigma       the noise sigma each side passed (run_frame: PipelineState.sigma_tvh_dn)
    damper      the transfer each side carried in from its previous frame
    input only  device display_dn with run_frame's carried transfer and sigma: the 1-DN plane
                differences alone, within one frame
    state only  run_frame's display_dn with the device's carried transfer
    undamped    both display_dn planes with no damper, then the same with the AGC's detail gate
                off, the gain cap off, and both off: which part of the operator amplifies 1 DN

Usage: python.sh diag_stage6.py <dump dir> [out.json]
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

from irsim.config.loader import load_sensor_config
from irsim.isp.damping import TransferDamper, damping_over
from irsim.isp.display import run_display_branch

SENSOR = pathlib.Path("/home/jumasaet/isaac-thermal-camera-simulation/configs/sensors/flir_boson_640_lwir.yaml")
spec = load_sensor_config(SENSOR).sensor
ISP, BITS, FR = spec.isp, spec.fpa.bit_depth, float(spec.fpa.frame_rate_hz)


class Capture:
    """A lut_hook that damps like TransferDamper (or not) and keeps the transfer it returned."""

    def __init__(self, prev, t_prev, t_s):
        self.damper = None
        if ISP.agc_damping > 0.0 and prev is not None:
            dt = 1.0 / FR
            if t_prev is not None and t_s - t_prev > 0.0:
                dt = t_s - t_prev
            self.damper = TransferDamper(damping_over(ISP.agc_damping, FR, dt), prev)
        elif ISP.agc_damping > 0.0:
            self.damper = TransferDamper(damping_over(ISP.agc_damping, FR, 1.0 / FR), None)
        self.out = None

    def __call__(self, lut):
        self.out = np.asarray(lut if self.damper is None else self.damper(lut), np.float64)
        return self.out


def show(dn, prev, t_prev, t_s, sigma, isp=ISP, damp=True):
    hook = Capture(prev, t_prev, t_s) if damp else None
    if not damp:
        isp = isp.model_copy(update=dict(agc_damping=0.0))
    d = run_display_branch(dn, isp, BITS, noise_sigma_dn=sigma, lut_hook=hook)
    return d.display8, (None if hook is None else hook.out)


def diff(a, b):
    a, b = a[..., :3].astype(np.int16), b[..., :3].astype(np.int16)
    e = np.abs(a - b).max(axis=-1)
    return {"max_code": int(e.max()), "px": int((e > 0).sum()), "px_gt1": int((e > 1).sum())}


def main(d: pathlib.Path, out: pathlib.Path | None) -> None:
    rows = []
    for f in sorted(d.glob("*.npz")):
        z = np.load(f)
        opt = lambda k: None if z[k].size == 0 else z[k]  # noqa: E731
        nanf = lambda k: None if np.isnan(float(z[k])) else float(z[k])  # noqa: E731
        t_s, cs, ds = float(z["t_s"]), float(z["cpu_sigma_tvh"]), float(z["dev_sigma"])
        cdn, ddn = z["cpu_display_dn"], z["dev_display_dn"]
        cp, cpt, dp, dpt = opt("cpu_prev_transfer"), nanf("cpu_prev_t"), opt("dev_prev_transfer"), nanf("dev_prev_t")
        c8, _ = show(cdn, cp, cpt, t_s, cs)
        g8, _ = show(ddn, dp, dpt, t_s, ds)
        r = {"file": f.name, "frame_index": int(z["frame_index"]),
             "sanity_cpu_reproduces": bool(np.array_equal(c8, z["cpu_display8"])),
             "sanity_dev_reproduces": bool(np.array_equal(g8, z["dev_display8"])),
             "sigma_cpu": cs, "sigma_dev": ds, "sigma_equal": cs == ds,
             "dt_prev_cpu": None if cpt is None else t_s - cpt, "dt_prev_dev": None if dpt is None else t_s - dpt,
             "damper_alpha": float(damping_over(ISP.agc_damping, FR, (t_s - cpt) if cpt is not None and t_s > cpt else 1.0 / FR)),
             "carried_transfer_max_abs_diff_codes": (None if cp is None or dp is None else float(np.abs(cp - dp).max() * 255)),
             "display_dn_px_differing": int((cdn != ddn).sum()),
             "display_dn_max_diff": int(np.abs(cdn.astype(np.int32) - ddn.astype(np.int32)).max()),
             "observed_dev_vs_cpu": diff(z["dev_display8"], z["cpu_display8"])}
        a8, ta = show(ddn, cp, cpt, t_s, cs)
        r["input_only"] = diff(a8, c8)
        b8, _ = show(cdn, dp, dpt, t_s, cs)
        r["state_only"] = diff(b8, c8)
        r["sigma_only"] = diff(show(cdn, cp, cpt, t_s, ds)[0], c8)
        und = {}
        for name, isp in (("as_configured", ISP),
                          ("no_detail_gate", ISP.model_copy(update=dict(detail_threshold_sigma=0.0))),
                          ("no_gain_cap", ISP.model_copy(update=dict(max_gain=0.0))),
                          ("no_gate_no_cap", ISP.model_copy(update=dict(detail_threshold_sigma=0.0, max_gain=0.0)))):
            hc, hd = Capture(None, None, t_s), Capture(None, None, t_s)
            isp0 = isp.model_copy(update=dict(agc_damping=0.0))
            x = run_display_branch(cdn, isp0, BITS, noise_sigma_dn=cs, lut_hook=hc).display8
            y = run_display_branch(ddn, isp0, BITS, noise_sigma_dn=cs, lut_hook=hd).display8
            und[name] = diff(y, x) | {"transfer_max_abs_diff_codes": float(np.abs(hc.out - hd.out).max() * 255)}
        r["undamped"] = und
        rows.append(r)
        print(json.dumps({k: r[k] for k in ("file", "sanity_cpu_reproduces", "sanity_dev_reproduces", "sigma_equal",
                                             "carried_transfer_max_abs_diff_codes", "display_dn_px_differing")}
                         | {"obs": r["observed_dev_vs_cpu"]["max_code"], "input_only": r["input_only"]["max_code"],
                            "state_only": r["state_only"]["max_code"],
                            "undamped": {k: v["max_code"] for k, v in und.items()}}), flush=True)
    if out is not None:
        out.write_text(json.dumps({"isp": ISP.model_dump(),
                                   "frame_rate_hz": FR, "rows": rows}, indent=1, default=str))
        print("wrote", out)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None)
