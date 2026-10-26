#!/usr/bin/env bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference
env REVERIFY_OLD_ISP=1 bash /home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/code/rebase_2026-10-22/ref/run_one.sh m_A_gpu_oldisp --audit-ids --optics fused --passes 2
env HOST_DISPLAY=1 bash /home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/code/rebase_2026-10-22/ref/run_one.sh m_B_host_newisp --audit-ids --optics fused --passes 2
env HOST_DISPLAY=1 REVERIFY_OLD_ISP=1 bash /home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/code/rebase_2026-10-22/ref/run_one.sh m_C_host_oldisp --audit-ids --optics fused --passes 2
echo ALLDONE
