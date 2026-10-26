#!/usr/bin/env bash
# Every (supersample, rendermode) combination, one process each. Logs and JSON into /tmp.
set -u
PY=/home/jumasaet/isaac_sim/python.sh
B=/tmp/irsim_render_cost
cd /home/jumasaet/isaac-thermal-camera-simulation
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

run() {  # run <tag> <sensor_ss> <rendermode> <scene>
  local tag=$1 k=$2 mode=$3 scene=$4
  echo "=================== $tag ==================="
  timeout 2400 $PY $B/bench_render_cost.py \
      --sensor "$B/sensor_ss${k}.yaml" --scene "$scene" --rendermode "$mode" \
      --warmup 10 --frames 30 --out "$B/result_${tag}.json" > "$B/log_${tag}.txt" 2>&1
  local rc=$?
  # Kit swallows a python traceback and still exits 0, so the JSON is the real success test.
  if [ ! -f "$B/result_${tag}.json" ]; then rc=99; fi
  echo "exit=$rc"
  if [ $rc -ne 0 ]; then grep -E "Error|Traceback|ModuleNotFound|No module|raise |assert " "$B/log_${tag}.txt" | tail -20; else
    grep -E "requested rendermode|supersample x|VRAM max" "$B/log_${tag}.txt"
  fi
}

run ss4_rt 4 RaytracedLighting  "$B/scene_bench.yaml"
run ss1_rt 1 RaytracedLighting  "$B/scene_bench.yaml"
run ss4_pt 4 PathTracing        "$B/scene_bench.yaml"
run ss1_pt 1 PathTracing        "$B/scene_bench.yaml"
# The isothermal enclosure: same geometry, L_env collapses to one scalar.
run ss4_rt_iso 4 RaytracedLighting "$B/scene_iso.yaml"
run ss1_rt_iso 1 RaytracedLighting "$B/scene_iso.yaml"
echo "=== all done ==="
ls -la $B/result_*.json 2>/dev/null
