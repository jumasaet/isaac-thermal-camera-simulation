#!/usr/bin/env bash
# run_time.sh <tag> <iso|clear> [extra args]
set -u
PY=/home/jumasaet/isaac_sim/python.sh
B=/home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/code/rebase_2026-10-22
O=$B/lite/out; mkdir -p "$O"
cd /home/jumasaet/isaac-thermal-camera-simulation
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
tag=$1; which=$2; shift 2
scene=$B/scene_bench.yaml; [ "$which" = iso ] && scene=$B/scene_iso.yaml
rm -f "$O/$tag.json"
timeout 1200 $PY $B/lite/time_lite.py --scene "$scene" --out "$O/$tag" "$@" > "$O/$tag.log" 2>&1
rc=$?; [ -f "$O/$tag.json" ] || rc=99
echo "$tag exit=$rc"
grep -E "^\[[^0-9]|^   lag|FAILED" "$O/$tag.log" | cut -c1-2500
[ $rc -ne 0 ] && grep -E "Error|Traceback|ModuleNotFound|No module|raise |assert |File \"/tmp" "$O/$tag.log" | tail -30
exit 0
