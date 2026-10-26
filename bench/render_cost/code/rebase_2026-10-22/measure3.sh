#!/usr/bin/env bash
# measure3.sh -- option (b) against all-GPU on an idle machine, 2026-10-23. Everything lands in $OUT.
#   A  all GPU, the pre-rebase ISP (plateau + DDE): the only stage 6 the device has
#   B  option (b): stages 1-5 on the GPU, stage 6 = run_frame's display branch on the host, the Boson's ISP
#   C  the host stage 6 with A's ISP: the cost of moving stage 6 alone, same operator as A
# B runs first WITH the check (the same-input stage-6 verdict), then everything --no-check, interleaved
# A B C A B C so drift between Kit sessions shows. Headless (run_ref.py's default), 2 timed passes x 240 frames.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
B=/home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost
RUN=$B/code/rebase_2026-10-22/ref/run_one.sh
OUT=$B/results/rebase_2026-10-22/fps_option_b_2026-10-23_idle
mkdir -p "$OUT"
stamp() { echo "== $1 $(date -Is) | $(uptime | sed 's/.*load/load/') | epp $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference) | gpu $(nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader)"; }
one() {  # one <tag> <env...> -- <args>
  local tag=$1; shift
  stamp "$tag"
  env "$@" bash $RUN "$tag" --optics fused --passes 2 ${EXTRA:-}
  cp $B/results/rebase_2026-10-22/ref_out/$tag/report.json "$OUT/$tag.report.json" 2>/dev/null
  cp $B/results/rebase_2026-10-22/ref_out/$tag/run.log "$OUT/$tag.run.log" 2>/dev/null
}
EXTRA=""          one m3_B_host_newisp_check HOST_DISPLAY=1
EXTRA="--no-check" one m3_A1_gpu_oldisp  REVERIFY_OLD_ISP=1
EXTRA="--no-check" one m3_B1_host_newisp HOST_DISPLAY=1
EXTRA="--no-check" one m3_C1_host_oldisp HOST_DISPLAY=1 REVERIFY_OLD_ISP=1
EXTRA="--no-check" one m3_A2_gpu_oldisp  REVERIFY_OLD_ISP=1
EXTRA="--no-check" one m3_B2_host_newisp HOST_DISPLAY=1
EXTRA="--no-check" one m3_C2_host_oldisp HOST_DISPLAY=1 REVERIFY_OLD_ISP=1
stamp end
echo ALLDONE
