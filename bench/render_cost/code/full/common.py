"""Shared by the bare-interpreter check and the Kit runner. /tmp only; nothing here edits the repo.

    sensor_yaml(k, ...)   the repo's Boson YAML with supersample_factor (and, for the FFC exercise,
                          a short shutter schedule) written in
    cpu_tail(...)         run_frame from stage 3 on, out of the repo's own functions, in its order
    cpu_noise_field(...)  the noise the CPU adds to frame `fi` (TVH + the six correlated terms)
    compare(...)          CPU planes against device planes, with the tolerances and the verdict
"""
from __future__ import annotations

import pathlib

import numpy as np

REPO = pathlib.Path("/home/jumasaet/isaac-thermal-camera-simulation")
FULL = pathlib.Path("/tmp/irsim_render_cost/full")
SENSOR_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"

#: Tolerances -- the repo's own, not new ones:
#:   flux      1e-5 relative: tests/integration/test_kernels_vs_reference.py holds the optics twin to it
#:   signal    5 mK: the same harness's MK_TOL, here at 300 K through dDN/dT (178 DN/K -> 0.89 DN)
#:   dn16      one code (a floor either side of an integer), on no more pixels than the signal
#:             difference explains: P(flip) = |d signal| for d << 1 DN
#:   display   test_warp_display_stage.py's bar is +/-1 code on >= 99.9 % of pixels FOR THE SAME DN.
#:             Here the DN planes are not bit-equal, so the bar is applied where it means the same
#:             thing: every pixel with no differing DN in its 3x3 (DDE's box) within one code.
#: For scale, the stage measurement's seam (CPU optics on both sides) read flux 3e-7, signal
#: 0.012 DN, ~300 px of dn16 one code apart, display <= 4 codes on < 1000 px.
MK_TOL = 5.0
TOL = {
    "flux_max_rel": 1e-5,
    "dn16_max_code": 1,
    "display_far_max_code": 1,
}


def sensor_yaml(k: int, ffc_interval_s: float | None = None, ffc_freeze_ms: float | None = None) -> pathlib.Path:
    """The repo's Boson YAML plus `supersample_factor: k` (the repo file leaves the default, 4)."""
    text = SENSOR_YAML.read_text(encoding="utf-8")
    anchor = "    f_number: 1.0\n"
    if text.count(anchor) != 1:
        raise SystemExit("flir_boson_640_lwir.yaml changed shape: cannot place supersample_factor")
    text = text.replace(anchor, anchor + f"    supersample_factor: {int(k)}   # /tmp override\n")
    name = f"_sensor_boson_ss{int(k)}"
    if ffc_interval_s is not None:
        import re

        text, n1 = re.subn(r"ffc_interval_s: 300\b", f"ffc_interval_s: {ffc_interval_s}", text)
        text, n2 = re.subn(r"ffc_freeze_ms: 700\b", f"ffc_freeze_ms: {ffc_freeze_ms}", text)
        if (n1, n2) != (1, 1):
            raise SystemExit("could not patch the FFC schedule")
        name += f"_ffc{ffc_interval_s}"
    out = FULL / "out" / f"{name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def cpu_tail(rad_ss, config, st, want_noise: bool = False) -> dict:
    """`run_frame` from stage 3 on: irsim/pipeline/frame.py:300-395, the same calls in the same order.

    `rad_ss` is the stage-2 radiance on the k x grid. Advances `st.frame_index`. With
    `want_noise` the dict also carries the noise field this frame received and the RTS state it
    was rendered with, which is what `FullGpu.frame(inject_noise=, rts_override=)` takes.
    """
    from irsim.detector.quantise import dn_max_for_bits, quantise
    from irsim.isp.display import run_display_branch
    from irsim.optics.stage import apply_optics, shutter_flux
    from irsim.pipeline.frame import _detector_signal

    spec, lut, q, k = config.sensor.sensor, config.lut, config.quantity, config.supersample
    bits = spec.fpa.bit_depth
    chain, fi = config.chain, st.frame_index
    t_fpa_k = st.housing_temp_k
    if chain is not None:
        housing_k, t_fpa_k = chain.begin_frame(st.t_s, fi)
        st.housing_temp_k = housing_k
    lb_h = float(lut.lookup(st.housing_temp_k, q)[()])
    flux = apply_optics(rad_ss, spec, lb_h, supersample=k, psf=config.psf)
    out: dict = {"flux": flux, "housing_k": float(st.housing_temp_k)}
    if want_noise:
        out["noise"] = cpu_noise_field(config, fi)
        if chain is not None:
            out["rts"] = np.asarray(chain.defect_state.bad, dtype=np.uint8).copy()
    signal = _detector_signal(flux, config, st)
    report = None
    if chain is not None and chain.needs_shutter_frame(fi):
        lb_sh = float(lut.lookup(t_fpa_k, q)[()])
        chain.record_shutter(config.detector.noiseless_signal_dn(shutter_flux(spec, lb_sh, lb_h)), config.flat_field)
    if chain is not None:
        signal, report = chain.finish_frame(signal, dn_max_for_bits(bits), fi, t_fpa_k)
    dn16 = quantise(signal, bits)
    display_dn = dn16
    flat = config.flat_field
    if chain is not None and chain.display_nuc is not None:
        flat = chain.display_nuc
    if flat is not None:
        display_dn = quantise(flat.apply(dn16), bits)
    display = run_display_branch(display_dn, spec.isp, bits)
    st.advance()
    out.update(signal=signal, dn16=dn16, display_dn=display_dn, y=display.y, display8=display.display8,
               report=report)
    return out


def cpu_noise_field(config, frame_index: int) -> np.ndarray:
    """What `_detector_signal` adds at `frame_index`: sigma_TVH (frame_from_signal) plus the six
    correlated terms (NoiseStage.apply), with the chain's breathing pattern when there is one.

    The bolometer's noise is additive in DN and independent of the signal, so running the two
    repo calls on a zero plane returns the field itself. Call it after `chain.begin_frame`.
    """
    shape = config.sensor.sensor.fpa_shape
    if not config.noise_enabled:
        return np.zeros(shape, np.float32)
    fr = config.detector.frame_from_signal(np.zeros(shape, np.float32), frame_index, config.sensor_seed)
    fixed = None if config.chain is None else config.chain.fixed_pattern
    return np.asarray(config.noise.apply(fr.signal_dn, fr.sigma_dn, frame_index, fixed_override=fixed), np.float32)


def _dilate3(mask: np.ndarray) -> np.ndarray:
    m = np.pad(mask, 1, mode="constant")
    out = np.zeros_like(mask)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out |= m[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return out


def compare(cpu: dict, gpu: dict, dn_per_k: float, gain_dn_per_w: float) -> dict:
    """Plane by plane. `display8` is split by cause: a pixel whose shown DN differs (or has such a
    neighbour, through DDE's 3x3 box) may differ by the AGC slope times one DN; any other pixel
    can differ only through the AGC table moving, which is far below one code."""
    f_c, f_g = cpu["flux"].astype(np.float64), gpu["flux"].astype(np.float64)
    d_sig = np.abs(cpu["signal"].astype(np.float64) - gpu["signal"].astype(np.float64))
    d_dn = np.abs(cpu["dn16"].astype(np.int64) - gpu["dn16"].astype(np.int64))
    d_dd = np.abs(cpu["display_dn"].astype(np.int64) - gpu["display_dn"].astype(np.int64))
    d_y = np.abs(cpu["y"].astype(np.float64) - gpu["y"].astype(np.float64))
    d_d8 = np.abs(cpu["display8"].astype(np.int16) - gpu["display8"].astype(np.int16)).max(axis=-1)
    near = _dilate3(d_dd > 0)
    n = d_dn.size
    sig_tol_dn = MK_TOL * 1e-3 * dn_per_k
    row = {
        "flux_max_rel": float((np.abs(f_c - f_g) / f_c).max()),
        "flux_max_abs_as_dn": float(gain_dn_per_w * np.abs(f_c - f_g).max()),
        "signal_max_abs_dn": float(d_sig.max()), "signal_mean_abs_dn": float(d_sig.mean()),
        "signal_max_abs_mk_at_300k": float(1e3 * d_sig.max() / dn_per_k),
        "dn16_px_differing": int((d_dn > 0).sum()), "dn16_max_code": int(d_dn.max()),
        "dn16_frac_differing": float((d_dn > 0).mean()),
        "display_dn_px_differing": int((d_dd > 0).sum()), "display_dn_max_code": int(d_dd.max()),
        "y_max_abs_codes_far": float(255.0 * d_y[~near].max()) if (~near).any() else 0.0,
        "display8_px_differing": int((d_d8 > 0).sum()), "display8_max_code": int(d_d8.max()),
        "display8_frac_within_1": float((d_d8 <= 1).mean()),
        "display8_far_px_differing": int((d_d8[~near] > 0).sum()),
        "display8_far_max_code": int(d_d8[~near].max()) if (~near).any() else 0,
        "n_px": int(n),
    }
    fails = []
    if row["flux_max_rel"] > TOL["flux_max_rel"]:
        fails.append("flux")
    if row["signal_max_abs_dn"] > sig_tol_dn:
        fails.append("signal")
    if row["dn16_max_code"] > TOL["dn16_max_code"] or row["dn16_frac_differing"] > 2.0 * row["signal_mean_abs_dn"] + 1e-4:
        fails.append("dn16")
    if row["display8_far_max_code"] > TOL["display_far_max_code"]:
        fails.append("display8")
    row["fails"] = fails
    return row


def noise_stats(cpu_signal: np.ndarray, gpu_signal: np.ndarray, sig, exclude: np.ndarray | None = None) -> dict:
    """Two generators, one distribution? `d = gpu - cpu` carries only the per-frame draws (the
    fixed pattern is the same realisation on both sides), so its moments are known in advance:

        sd(d)            = sqrt(2) sqrt(s_tvh^2 + s_t^2 + s_tv^2 + s_th^2)
        sd(row means d)  = sqrt(2) sqrt(s_tv^2 + (s_tvh^2 + s_th^2) / W)
        sd(col means d)  = sqrt(2) sqrt(s_th^2 + (s_tvh^2 + s_tv^2) / H)
        mean(d)          ~ N(0, 2 s_t^2 + ...)           one draw per frame per side

    `sig` is the full Sigmas7 for this frame's sigma_TVH in DN, TVH slot included (NOT
    `NoiseStage.sigmas`, whose TVH slot is zero because the detector owns that term). Returns measured, expected
    and z-scores (the sampling error of an sd estimate from n samples is sd / sqrt(2 n))."""
    d = gpu_signal.astype(np.float64) - cpu_signal.astype(np.float64)
    if exclude is not None:
        d = np.where(exclude, np.nan, d)
    h, w = d.shape
    n = int(np.isfinite(d).sum())
    e_px = np.sqrt(2.0) * np.sqrt(sig.tvh**2 + sig.t**2 + sig.tv**2 + sig.th**2)
    e_row = np.sqrt(2.0) * np.sqrt(sig.tv**2 + (sig.tvh**2 + sig.th**2) / w)
    e_col = np.sqrt(2.0) * np.sqrt(sig.th**2 + (sig.tvh**2 + sig.tv**2) / h)
    e_mean = np.sqrt(2.0) * np.sqrt(sig.t**2 + sig.tv**2 / h + sig.th**2 / w + sig.tvh**2 / n)
    m_px, m_row, m_col = float(np.nanstd(d)), float(np.nanstd(np.nanmean(d, axis=1))), float(np.nanstd(np.nanmean(d, axis=0)))
    return {
        "sd_px": m_px, "sd_px_expected": float(e_px), "z_px": float((m_px - e_px) / (e_px / np.sqrt(2 * n))),
        "sd_row": m_row, "sd_row_expected": float(e_row), "z_row": float((m_row - e_row) / (e_row / np.sqrt(2 * h))),
        "sd_col": m_col, "sd_col_expected": float(e_col), "z_col": float((m_col - e_col) / (e_col / np.sqrt(2 * w))),
        "mean": float(np.nanmean(d)), "mean_sd_expected": float(e_mean), "z_mean": float(np.nanmean(d) / e_mean),
    }
