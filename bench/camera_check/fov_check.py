"""Load the draft Cellplus config through the simulator's own loader and report its field of view.

    /home/jumasaet/isaac_sim/python.sh bench/camera_check/fov_check.py
"""
import math, os, sys
ROOT = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(ROOT))
sys.path.insert(0, os.path.join(REPO, "src"))
from irsim.config.loader import load_sensor_config
from irsim.optics.defocus import hyperfocal_distance_m

for name in ("cellplus_ctc007_lwir.yaml",):
    s = load_sensor_config(os.path.join(ROOT, name)).sensor
    vfov = math.degrees(2 * math.atan(s.fpa.height * s.fpa.pitch_um * 1e-3 / 2 / s.optics.focal_length_mm))
    dfov = math.degrees(2 * math.atan(math.hypot(s.fpa.width, s.fpa.height) * s.fpa.pitch_um * 1e-3 / 2
                                      / s.optics.focal_length_mm))
    print(f"{s.name}: band {s.band.band_id} {s.band.lambda_min_um}-{s.band.lambda_max_um} um, "
          f"{s.fpa.width}x{s.fpa.height} @ {s.fpa.pitch_um} um, f={s.optics.focal_length_mm} mm F/{s.optics.f_number}")
    print(f"  HFOV (SensorSpec.hfov_deg) = {s.hfov_deg:.2f} deg   VFOV = {vfov:.2f} deg   DFOV = {dfov:.2f} deg")
    print(f"  IFOV = {s.fpa.pitch_um / s.optics.focal_length_mm:.3f} mrad   Nyquist {s.nyquist_cyc_per_mm:.1f} cyc/mm"
          f"   diffraction cut-off {s.cutoff_cyc_per_mm:.1f} cyc/mm (lambda {s.reference_wavelength_um} um)")
    print(f"  hyperfocal (c = 1 pitch) = {hyperfocal_distance_m(s.optics.focal_length_mm, s.optics.f_number, s.fpa.pitch_um):.1f} m")
    nu = s.nyquist_cyc_per_mm / s.cutoff_cyc_per_mm
    print(f"  diffraction MTF at Nyquist = {2/math.pi*(math.acos(nu)-nu*math.sqrt(1-nu*nu)):.3f}")
