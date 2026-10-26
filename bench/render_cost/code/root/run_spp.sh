#!/usr/bin/env bash
# Is the renderer's fixed cost spp-bound or readback-bound? Sweep spp at the native resolution.
set -u
PY=/home/jumasaet/isaac_sim/python.sh
B=/tmp/irsim_render_cost
cd /home/jumasaet/isaac-thermal-camera-simulation
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
for SPP in 1 4 8 16; do
  echo "=================== ss1_pt_spp${SPP} ==================="
  timeout 1800 $PY $B/bench_render_cost.py --sensor "$B/sensor_ss1.yaml" \
      --scene "$B/scene_bench.yaml" --rendermode PathTracing --pt-spp $SPP \
      --warmup 10 --frames 30 --out "$B/result_ss1_pt_spp${SPP}.json" \
      > "$B/log_ss1_pt_spp${SPP}.txt" 2>&1
  rc=$?; [ -f "$B/result_ss1_pt_spp${SPP}.json" ] || rc=99
  echo "exit=$rc"
done
# And the smallest render product the sensor schema allows, at spp 1: the floor's floor.
echo "=== spp sweep done ==="
