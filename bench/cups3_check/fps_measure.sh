#!/usr/bin/env bash
# fps_measure.sh -- GPU-path frame rate, CellPlus camera, three-cup scene, supersample 1 / 2 / 4.
# Two Kit sessions per k, interleaved (1 2 4 1 2 4) so drift between sessions shows. Each: 60 warm-up
# iterations, 2 timed passes x 240 frames, 1 profile pass (a synchronisation after every block). Headless,
# thermal + RGB, --optics fused (src's warp_stages.launch_optics is broken, RESULTADOS.md), --no-check (the GPU
# path has no defocus; run_frame's layered defocus would differ by design). Set A also saves the frames.
set -u
cd /home/jumasaet/isaac-thermal-camera-simulation
O=bench/cups3_check/out/fps
mkdir -p $O
stamp() { echo "== $1 $(date -Is) | $(uptime | sed 's/.*load/load/') | epp $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference) | gpu $(nvidia-smi --query-gpu=utilization.gpu,clocks.sm,temperature.gpu --format=csv,noheader)"; }
run() {  # run <tag> <k> [extra]
  stamp "$1"
  timeout 1800 env -u LD_LIBRARY_PATH -u PYTHONPATH -u AMENT_PREFIX_PATH -u ROS_DISTRO -u ROS_VERSION -u ROS_PYTHON_VERSION \
    /home/jumasaet/isaac_sim/python.sh bench/cups3_check/fps_run.py --tag "$1" --k "$2" --optics fused --no-check \
    --passes 2 --profile "${@:3}" > "$O/$1.log" 2>&1
  echo "$1 rc=$?"; grep -E "^\[$1 #" "$O/$1.log" | cut -c1-260
}
for k in 1 2 4; do run "A_k$k" $k --save "/tmp/claude-1000/-home-jumasaet-isaac-thermal-camera-simulation/5b90ac53-c576-49ce-b21c-9370c45d52c6/scratchpad/fps_frames_k$k"; done
for k in 1 2 4; do run "B_k$k" $k; done
stamp end
echo ALLDONE
