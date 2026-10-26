"""Stage 6 on the host alone (no Kit): run_display_branch on the real 640x512 display_dn planes the
check dumped, the Boson's ISP (information_based, damped) and the pre-rebase one (plateau + DDE 0.35).
Also the information_based pieces timed separately. Usage: python.sh bare_stage6_cost.py <dump dir> <out.json>"""
import json, pathlib, sys, time
import numpy as np
from irsim.config.loader import load_sensor_config
from irsim.config.sensor import IspSpec
from irsim.isp.damping import TransferDamper, damping_over
from irsim.isp.display import run_display_branch
from irsim.isp import information as inf

spec = load_sensor_config(pathlib.Path("/home/jumasaet/isaac-thermal-camera-simulation/configs/sensors/flir_boson_640_lwir.yaml")).sensor
BOSON = spec.isp
OLD = IspSpec(agc="plateau_equalization", plateau=0.012, clip_percentiles=[0.005, 0.995], gamma=1.0, dde_gain=0.35,
              polarity="white_hot", palette="gray")
frames = [np.load(f)["dev_display_dn"] for f in sorted(pathlib.Path(sys.argv[1]).glob("g_m9_ff_*.npz"))]
sigma = float(np.load(sorted(pathlib.Path(sys.argv[1]).glob("*.npz"))[0])["dev_sigma"])
alpha = damping_over(BOSON.agc_damping, 60.0, 1 / 60.0)

def run(isp, reps=3):
    ts, prev = [], None
    for _ in range(reps):
        for dn in frames:
            d = TransferDamper(alpha, prev) if isp.agc_damping > 0 else None
            t = time.perf_counter(); run_display_branch(dn, isp, 16, noise_sigma_dn=sigma, lut_hook=d); ts.append(time.perf_counter() - t)
            prev = None if d is None else d.previous
    ts = np.array(ts[len(frames):]) * 1e3  # first lap is warm-up
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)), "n": int(ts.size)}

out = {"shape": list(frames[0].shape), "numpy": np.__version__, "boson_information_based": run(BOSON), "old_plateau_dde": run(OLD)}
dn = frames[0].astype(np.float64)
t = time.perf_counter()
for _ in range(10): low, high = inf.bilateral_split(dn, BOSON.smoothing_sigma_dn)
out["boson_bilateral_split_5x5_ms"] = (time.perf_counter() - t) * 100
print(json.dumps(out, indent=1)); pathlib.Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
