"""Smoke test: one engine-free frame through irsim's pipeline with the draft Cellplus config.

Uniform 295 K room with a 310 K patch, at full 640x480. Proves the draft builds end to end (band LUT
for 8-14 um, bolometer transfer, noise, NUC, ISP) and reports the simulated DN per kelvin and noise in
DN, which is what the recordings' 2.1-count sigma will be compared against once counts/K is measured.

    /home/jumasaet/isaac_sim/python.sh bench/camera_check/smoke_frame.py
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(ROOT))
sys.path.insert(0, os.path.join(REPO, "src"))
from irsim.config.loader import load_sensor_config
from irsim.materials import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.radiometry.encoding import encode_temperature
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import load_spectral_response

cfg = load_sensor_config(os.path.join(ROOT, "cellplus_ctc007_lwir.yaml"))
s = cfg.sensor
lut = BandLUT.build(load_spectral_response(s.band.spectral_response))
pc = PipelineConfig.from_sensor(cfg, MaterialTable.constant(1.0), lut=lut, sensor_seed=1)
H, W = s.fpa_shape
S = s.optics.supersample_factor  # the G-buffer is rendered at S x the detector grid (ADR 0059)
def planes(t0):
    t = np.full((H, W), t0, np.float32); t[200:280, 280:360] = t0 + 15.0
    t = np.kron(t, np.ones((S, S), np.float32)); hs, ws = t.shape
    return {"temperature_k": t, "encoded_t": encode_temperature(t), "normal_dot_view": np.ones((hs, ws), np.float32),
            "distance_m": np.full((hs, ws), 5.0, np.float32), "material_id": np.ones((hs, ws), np.int32),
            "sky_view_factor": np.zeros((hs, ws), np.float32)}
outs = {}
for t0 in (295.0, 296.0):
    st = PipelineState()
    outs[t0] = run_frame(planes(t0), pc, st)
o = outs[295.0]
names = [k for k in dir(o) if not k.startswith("_")]
print("frame fields:", [k for k in names if getattr(o, k, None) is not None and isinstance(getattr(o, k), np.ndarray)])
dn = o.dn16.astype(np.float64); dn1 = outs[296.0].dn16.astype(np.float64)
bg = np.ones((H, W), bool); bg[180:300, 260:380] = False
print(f"dn_16 dtype {o.dn16.dtype} shape {o.dn16.shape}; background mean {dn[bg].mean():.1f}, "
      f"patch-bg {dn[200:280, 280:360].mean() - dn[bg].mean():.1f} DN for 15 K")
print(f"SITF ~ {(dn1[bg].mean() - dn[bg].mean()):.1f} DN/K at 295 K; background std {dn[bg].std():.2f} DN")
print(f"apparent T background mean {o.apparent_t[bg].mean():.2f} K")

# Radial falloff: simulated (uniform 295 K scene) vs the real camera's mean over the whole moving
# lab sequence, where the scene averages out and only fixed structure (vignetting, shading) remains.
import glob
yy, xx = np.mgrid[:H, :W]; rr = np.hypot(yy - (H - 1) / 2, xx - (W - 1) / 2) / np.hypot(H / 2, W / 2)
def prof(img, sitf):
    c = img[(rr >= 0.15) & (rr < 0.25)].mean()  # just outside the centre patch
    return [(img[(rr >= a) & (rr < a + 0.1)].mean() - c) / sitf for a in (0.3, 0.6, 0.9)]
print("simulated ring offset from r/rmax 0.15-0.25, K at r/rmax = 0.3, 0.6, 0.9:",
      np.round(prof(dn, dn1[bg].mean() - dn[bg].mean()), 2))
fs = sorted(glob.glob(os.path.join(REPO, "thermal_specs/lab/thermal/*.npy")))
mean = np.mean([np.load(f).astype(np.float64) for f in fs], axis=0)
print("real mean-frame ring offset from r/rmax 0.15-0.25, COUNTS at r/rmax = 0.3, 0.6, 0.9:",
      np.round(prof(mean, 1.0), 1), " (sigma_tvh = 2.1 counts)")
