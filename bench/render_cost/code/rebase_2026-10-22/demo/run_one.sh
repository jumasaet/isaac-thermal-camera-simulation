#!/usr/bin/env bash
# run_one.sh <tag> [run_demo.py args]   -- one Kit session, log beside its report
set -u
PY=/home/jumasaet/isaac_sim/python.sh
D=/home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/code/rebase_2026-10-22/demo
O=/home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/results/rebase_2026-10-22/demo_out
cd /home/jumasaet/isaac-thermal-camera-simulation
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
tag=$1; shift
mkdir -p "$O/$tag"; rm -f "$O/$tag/report.json"
timeout 1500 $PY $D/run_demo.py --tag "$tag" "$@" > "$O/$tag/run.log" 2>&1
rc=$?; grep -q "^wrote " "$O/$tag/run.log" || rc=99
echo "$tag exit=$rc"
grep -E "^\[$tag" "$O/$tag/run.log" | cut -c1-2200
[ $rc -ne 0 ] && grep -E "Traceback|Error\]|File \"/tmp|raise |assert " "$O/$tag/run.log" | tail -25
exit 0
