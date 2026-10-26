#!/usr/bin/env bash
# run_one.sh <tag> [sub] [run_ref.py args]   -- one Kit session, log beside its report.
#
# "sub" as the second word starts the system-ROS subscriber (ros_sub.py, python3.10) first, waits for
# it in the runner (--ros-wait-subs 1) and compares what it received with what was published.
#
# THE ENVIRONMENT. The Kit process must NOT see the system ROS 2: /opt/ros/humble/lib holds the
# python3.10 builds of the message libraries (lib*__rosidl_generator_py.so), and with them first on
# LD_LIBRARY_PATH Isaac's python3.12 aborts on the first publisher (assertion in
# rcl_interfaces ... convert_from_py). So the six variables a sourced ROS shell exports are removed
# for that one process; python.sh's own setup_ros_env.sh then puts Isaac's bundled Humble libraries
# on LD_LIBRARY_PATH and selects rmw_fastrtps_cpp. ROS_DOMAIN_ID is kept, so both sides share a domain.
# The subscriber runs in the normal sourced shell.
set -u
PY=/home/jumasaet/isaac_sim/python.sh
D=/home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/code/rebase_2026-10-22/ref
O=/home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/results/rebase_2026-10-22/ref_out
cd /home/jumasaet/isaac-thermal-camera-simulation
tag=$1; shift
sub=0; if [ "${1:-}" = "sub" ]; then sub=1; shift; fi
mkdir -p "$O/$tag"; rm -f "$O/$tag/report.json" "$O/$tag/sub.json" "$O/$tag/published.json"
extra=()
if [ $sub -eq 1 ]; then
  /usr/bin/python3 $D/ros_sub.py --out "$O/$tag/sub.json" --idle 45 --max-wait 900 ${SUB_ARGS:-} > "$O/$tag/sub.log" 2>&1 &
  subpid=$!
  extra=(--ros --ros-wait-subs 1)
fi
timeout 1800 env -u LD_LIBRARY_PATH -u PYTHONPATH -u AMENT_PREFIX_PATH -u ROS_DISTRO -u ROS_VERSION -u ROS_PYTHON_VERSION \
  $PY $D/run_ref.py --tag "$tag" "${extra[@]}" "$@" > "$O/$tag/run.log" 2>&1
rc=$?; grep -q "^wrote " "$O/$tag/run.log" || rc=99
echo "$tag exit=$rc epp=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)"
grep -E "^\[$tag" "$O/$tag/run.log" | cut -c1-1500
[ $rc -ne 0 ] && grep -E "Traceback|Error\]|File \"/tmp|raise |assert |Error:|Assertion" "$O/$tag/run.log" | tail -25
if [ $sub -eq 1 ]; then
  wait $subpid
  /usr/bin/python3 $D/compare_sub.py "$O/$tag/published.json" "$O/$tag/sub.json" \
    ir=${IR_TOPIC:-/thermal_sim/ir/image_raw} rgb=${RGB_TOPIC:-/thermal_sim/rgb/image_raw} | tee "$O/$tag/reception.txt"
fi
exit 0
