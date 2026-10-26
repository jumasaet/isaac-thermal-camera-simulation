#!/usr/bin/env bash
# run_one.sh <tag> [run_demo.py args]   -- one Kit session, log beside its report
set -u
PY=/home/jumasaet/isaac_sim/python.sh
D=/tmp/irsim_render_cost/demo
cd /home/jumasaet/isaac-thermal-camera-simulation
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
tag=$1; shift
mkdir -p "$D/out/$tag"; rm -f "$D/out/$tag/report.json"
timeout 1500 $PY $D/run_demo.py --tag "$tag" "$@" > "$D/out/$tag/run.log" 2>&1
rc=$?; grep -q "^wrote " "$D/out/$tag/run.log" || rc=99
echo "$tag exit=$rc"
grep -E "^\[$tag" "$D/out/$tag/run.log" | cut -c1-2200
[ $rc -ne 0 ] && grep -E "Traceback|Error\]|File \"/tmp|raise |assert " "$D/out/$tag/run.log" | tail -25
exit 0
