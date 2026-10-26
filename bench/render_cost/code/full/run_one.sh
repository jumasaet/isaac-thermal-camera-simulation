#!/usr/bin/env bash
# run_one.sh <tag> [run_full.py args]   -- one Kit session, log beside its report
set -u
PY=/home/jumasaet/isaac_sim/python.sh
D=/tmp/irsim_render_cost/full
cd /home/jumasaet/isaac-thermal-camera-simulation
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
tag=$1; shift
mkdir -p "$D/out/$tag"; rm -f "$D/out/$tag/report.json"
timeout 3000 $PY $D/run_full.py --tag "$tag" "$@" > "$D/out/$tag/run.log" 2>&1
rc=$?; grep -q "^wrote " "$D/out/$tag/run.log" || rc=99
echo "$tag exit=$rc epp=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)"
grep -E "^\[$tag" "$D/out/$tag/run.log" | cut -c1-1400
[ $rc -ne 0 ] && grep -E "Traceback|Error\]|File \"/tmp|raise |assert |Error:" "$D/out/$tag/run.log" | tail -25
exit 0
