#!/usr/bin/env bash
# Renders every task-3 case (measured calibration, cellplus_ctc007_lwir_cups.yaml) and every case with the
# original draft YAML (the simulator's own 22.2 mm geometry, for task 2). One Kit session per render, the
# system ROS variables removed for that process (bench/render_cost/.../ref/run_one.sh explains why).
set -u
cd /home/jumasaet/isaac-thermal-camera-simulation
PY=/home/jumasaet/isaac_sim/python.sh
L=bench/cups_check/out/logs
mkdir -p $L
run() {  # case yaml subdir
  timeout 900 env -u LD_LIBRARY_PATH -u PYTHONPATH -u AMENT_PREFIX_PATH -u ROS_DISTRO -u ROS_VERSION -u ROS_PYTHON_VERSION \
    $PY bench/cups_check/task3_render_cups.py "$1" "$2" "$3" > "$L/$3.log" 2>&1
  echo "$3 rc=$?"
}
for c in ${CASES:-hot_water_50 hot_water_95 cold_water_50 cold_water_95}; do
  [ -f bench/cups_check/out/sim_$c/report.json ] || run $c bench/camera_check/cellplus_ctc007_lwir_cups.yaml sim_$c
done
for c in ${CASES:-hot_water_50 hot_water_95 cold_water_50 cold_water_95}; do
  [ -f bench/cups_check/out/simorig_$c/report.json ] || run $c bench/camera_check/cellplus_ctc007_lwir.yaml simorig_$c
done
echo ALL_DONE
