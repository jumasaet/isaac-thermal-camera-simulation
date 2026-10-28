#!/usr/bin/env python
"""Frame rate of the GPU path with the CellPlus camera on the three-cup scene, at one supersample factor.

    env -u LD_LIBRARY_PATH -u PYTHONPATH -u AMENT_PREFIX_PATH -u ROS_DISTRO -u ROS_VERSION -u ROS_PYTHON_VERSION \
        /home/jumasaet/isaac_sim/python.sh bench/cups3_check/fps_run.py --tag <tag> --k <1|2|4> [run_ref.py args]

Runs bench/render_cost/code/rebase_2026-10-22/ref/run_ref.py (the GPU bench, see its NOTES.md) UNCHANGED, with three
things swapped in before it starts:
  - the scene: fps_scene.py in place of ref_scene (three cups, task 2's pose, colours and temperatures);
  - the sensor: bench/camera_check/cellplus_ctc006_lwir_cups.yaml with `optics.supersample_factor: k` written in
    (out/fps/_sensor_cellplus_ss<k>.yaml), in place of the bench's Boson YAML;
  - the LUT directory: bench/camera_preview/lut (it holds this band's LUT; the bench's own dir holds the Boson's).
Headless (run_ref's default), thermal + RGB annotators, RealTimePathTracing.

The GPU path has NO defocus: its stage 3 (full/fullgpu.py) convolves with `PipelineConfig.psf`, the in-focus
kernel (diffraction + the YAML's 80 um aberration Gaussian); the YAML's `defocus_model: hopkins` /
`defocus_apply: layered` exist only on the host path (irsim.pipeline.frame -> irsim.optics.layered). Nothing
replaces them here. run_ref's equivalence check against run_frame would therefore differ by the defocus, so the
timed runs pass --no-check.
"""
from __future__ import annotations

import pathlib
import re
import runpy
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
BENCH = REPO / "bench" / "render_cost" / "code" / "rebase_2026-10-22"
CELLPLUS = REPO / "bench" / "camera_check" / "cellplus_ctc006_lwir_cups.yaml"
LUT_DIR = REPO / "bench" / "camera_preview" / "lut"
OUT = HERE / "out" / "fps"

for p in (BENCH / "ref", BENCH / "full", BENCH / "demo", BENCH / "lite", REPO / "src", HERE):
    sys.path.insert(0, str(p))

import common  # noqa: E402  (bench full/common.py: run_ref imports it as `cm`)
import fps_scene  # noqa: E402


def sensor_yaml(k: int, *_a, **_kw) -> pathlib.Path:
    text = CELLPLUS.read_text(encoding="utf-8")
    anchor = "    f_number: 1.3"
    if text.count(anchor) != 1:
        raise SystemExit("cellplus YAML changed shape: cannot place supersample_factor")
    text = text.replace(anchor, f"    supersample_factor: {int(k)}   # bench/cups3_check/fps_run.py\n" + anchor, 1)
    text = re.sub(r"(?m)^  name: cellplus_ctc006_lwir_cups$", f"  name: cellplus_ctc006_lwir_cups_ss{int(k)}", text)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"_sensor_cellplus_ss{int(k)}.yaml"
    out.write_text(text)
    return out


common.sensor_yaml = sensor_yaml
sys.modules["ref_scene"] = fps_scene

from irsim.radiometry import lut_files  # noqa: E402

_orig_lut = lut_files.load_band_lut_for_config
lut_files.load_band_lut_for_config = lambda sensor, _dir, *a, **kw: _orig_lut(sensor, LUT_DIR, *a, **kw)

sys.argv[0] = str(BENCH / "ref" / "run_ref.py")
if not any(a.startswith("--out") for a in sys.argv):
    sys.argv += ["--out", str(OUT)]
runpy.run_path(str(BENCH / "ref" / "run_ref.py"), run_name="__main__")
