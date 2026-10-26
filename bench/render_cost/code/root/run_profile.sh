#!/usr/bin/env bash
# run_profile.sh <tag> <iso|clear> [spans-file]   -- one process, JSON + .prof into prof/
set -u
PY=/home/jumasaet/isaac_sim/python.sh
B=/tmp/irsim_render_cost
cd /home/jumasaet/isaac-thermal-camera-simulation
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
tag=$1; which=$2; spans=${3:-}; lines=${4:-}
scene=$B/scene_bench.yaml; [ "$which" = iso ] && scene=$B/scene_iso.yaml
extra=(); [ -n "$spans" ] && extra+=(--spans "$spans"); [ -n "$lines" ] && extra+=(--lines "$lines")
rm -f "$B/prof/$tag.json"
timeout 1800 $PY $B/profile_frame.py --sensor $B/sensor_ss1.yaml --scene "$scene" \
    --rendermode RaytracedLighting --warmup 10 --frames 30 "${extra[@]}" \
    --out "$B/prof/$tag" > "$B/prof/$tag.log" 2>&1
rc=$?; [ -f "$B/prof/$tag.json" ] || rc=99
echo "$tag exit=$rc"
grep -E "^\[pass|UNRESOLVED|^   +[0-9.]+ incl" "$B/prof/$tag.log"
[ $rc -ne 0 ] && grep -E "Error|Traceback|ModuleNotFound|No module|raise |assert " "$B/prof/$tag.log" | tail -20
exit 0
