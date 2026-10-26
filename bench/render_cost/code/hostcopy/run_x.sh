#!/usr/bin/env bash
# run_one.sh <tag> <iso|clear> [extra args...]   -- one process, JSON + log into hostcopy/out/
set -u
PY=/home/jumasaet/isaac_sim/python.sh
B=/tmp/irsim_render_cost
O=$B/hostcopy/out
mkdir -p "$O"
cd /home/jumasaet/isaac-thermal-camera-simulation
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
tag=$1; which=$2; shift 2
scene=$B/scene_bench.yaml; [ "$which" = iso ] && scene=$B/scene_iso.yaml
rm -f "$O/$tag.json"
timeout 2400 $PY $B/hostcopy/bench_x.py --scene "$scene" --out "$O/$tag" "$@" > "$O/$tag.log" 2>&1
rc=$?; [ -f "$O/$tag.json" ] || rc=99
echo "$tag exit=$rc"
grep -E "^\[[A-Z]+:|FAILED|^=== " "$O/$tag.log" | cut -c1-600
[ $rc -ne 0 ] && grep -E "Error|Traceback|ModuleNotFound|No module|raise |assert " "$O/$tag.log" | tail -20
exit 0
