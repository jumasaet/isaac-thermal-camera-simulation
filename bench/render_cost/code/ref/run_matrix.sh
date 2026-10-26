#!/usr/bin/env bash
# The timing matrix: one Kit session per line, 3 timed passes of the 240 frames + 1 profiled.
# Two rounds, interleaved, so a slow drift does not land on one configuration.
D=/tmp/irsim_render_cost/ref
powerprofilesctl set performance; echo "profile: $(powerprofilesctl get)  epp: $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)"
A="--passes 3 --profile --no-check"
for r in 1 2; do
  $D/run_one.sh m_hl_noros_r$r        $A
  $D/run_one.sh m_hl_ros_nosub_r$r    $A --ros
  $D/run_one.sh m_hl_ros_sub_r$r  sub $A
  $D/run_one.sh m_gui_noros_r$r       $A --gui --viewport
  $D/run_one.sh m_gui_ros_nosub_r$r   $A --gui --viewport --ros
  $D/run_one.sh m_gui_ros_sub_r$r sub $A --gui --viewport
done
# the two options, each exercised once
$D/run_one.sh opt_jet --passes 1 --no-check --palette jet --save $D/out/opt_jet/outputs
SUB_ARGS="" $D/run_one.sh opt_dn16 sub --passes 1 --no-check --ros-ir-content dn16
echo "=== matrix done ==="
