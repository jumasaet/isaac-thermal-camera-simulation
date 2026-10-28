#!/usr/bin/env bash
# run_all.sh -- the three-cup experiment end to end: tasks 0 -> 5, in order, one log per task.
#
#     bash bench/cups3_check/run_all.sh            # everything (about 3 min)
#     FROM=3 bash bench/cups3_check/run_all.sh     # start at task 3 (the earlier outputs must exist)
#
# Reads thermal_specs/three_cups_experiments/sync_cups/ and writes bench/cups3_check/out/ and figures/
# (bench/GUIA.md, section 3, lists what each task produces). Stops at the first task that fails and names
# its log. Task 3 needs torch (conda env `vision`) plus lpips and scikit-image, installed once into $PYLIBS
# with `pip --target` -- nothing in the conda env changes (task3_similarity.py's docstring).
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PY=${ISAAC_PY:-/home/jumasaet/isaac_sim/python.sh}
VISION_PY=${VISION_PY:-$HOME/miniconda3/envs/vision/bin/python}
PYLIBS=${PYLIBS:-$HOME/.cache/irsim_pylibs}
FROM=${FROM:-0}
D=bench/cups3_check
L=$D/out/logs
mkdir -p "$L" "$D/figures"

# Kit aborts if the system ROS 2 variables leak into it (task2_render.py's docstring); the analysis scripts
# do not care, so every Isaac step runs with them removed.
ISAAC=(env -u LD_LIBRARY_PATH -u PYTHONPATH -u AMENT_PREFIX_PATH -u ROS_DISTRO -u ROS_VERSION -u ROS_PYTHON_VERSION
       CUPS_MIN_DISTANCE_M=0.3 "$PY")

step() {  # step <n> <log name> <command...>
  local n=$1 log=$L/$2.log; shift 2
  [ "$n" -lt "$FROM" ] && return 0
  local t0=$SECONDS
  echo "[$(date +%T)] task $n: $*"
  if ! timeout 1800 "$@" > "$log" 2>&1; then
    echo "  FAILED -- see $log (last lines below)"; tail -n 15 "$log"; exit 1
  fi
  echo "  ok in $((SECONDS - t0)) s -> $log"
}

if [ "$FROM" -le 3 ] && ! PYTHONPATH=$PYLIBS "$VISION_PY" -c "import lpips, skimage" 2>/dev/null; then
  echo "installing lpips 0.1.4 + scikit-image 0.24 into $PYLIBS (once)"
  "$VISION_PY" -m pip install -q --target "$PYLIBS" --no-deps \
      lpips==0.1.4 scikit-image==0.24.0 imageio lazy_loader tifffile networkx || exit 1
fi

# The band LUT the render reads (bench/camera_preview/common.py, LUT_DIR) is gitignored: build it if absent.
LUT=bench/camera_preview/lut
if [ "$FROM" -le 2 ] && ! ls $LUT/*_lut.json > /dev/null 2>&1; then
  echo "building the band LUT into $LUT"
  "${ISAAC[@]}" scripts/generate_luts.py --configs bench/camera_check/cellplus_ctc006_lwir_cups.yaml --out $LUT \
      --data data > "$L/generate_luts.log" 2>&1 || { echo "  FAILED -- see $L/generate_luts.log"; exit 1; }
fi

step 0 task0_scene      "${ISAAC[@]}" $D/task0_scene.py
step 1 task1_calibration "${ISAAC[@]}" $D/task1_calibration.py
step 2 task2_render      "${ISAAC[@]}" $D/task2_render.py
step 2 task2_compare     "${ISAAC[@]}" $D/task2_compare.py
step 3 task3_similarity  env PYTHONPATH="$PYLIBS" "$VISION_PY" $D/task3_similarity.py
step 4 task4_reflection  "${ISAAC[@]}" $D/task4_reflection.py
step 5 task5_figures     "${ISAAC[@]}" $D/task5_figures.py
echo "done: results in $D/out/ (json, png), presentation sheets in $D/figures/"
