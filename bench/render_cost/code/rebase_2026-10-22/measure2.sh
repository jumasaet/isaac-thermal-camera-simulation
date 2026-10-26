#!/usr/bin/env bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
uptime
env REVERIFY_OLD_ISP=1 bash /home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/code/rebase_2026-10-22/ref/run_one.sh m2_A_gpu_oldisp --optics fused --passes 2 --no-check
uptime
env HOST_DISPLAY=1 bash /home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/code/rebase_2026-10-22/ref/run_one.sh m2_B_host_newisp --optics fused --passes 2 --no-check
uptime
env HOST_DISPLAY=1 REVERIFY_OLD_ISP=1 bash /home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/code/rebase_2026-10-22/ref/run_one.sh m2_C_host_oldisp --optics fused --passes 2 --no-check
uptime
echo ALLDONE
