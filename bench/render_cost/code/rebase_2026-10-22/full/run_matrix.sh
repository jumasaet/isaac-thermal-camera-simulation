#!/usr/bin/env bash
# The measurement matrix. One Kit session per line; two timed passes per (variant, trajectory),
# then one profiled and one for the stills. The CPU/GPU comparison runs once per (k, optics).
D=/home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/code/rebase_2026-10-22/full
powerprofilesctl set performance; echo "profile: $(powerprofilesctl get)  epp: $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)"
V1="s12 12 g g_ff g_m9 g_m9_ff"
V2="r0 s12 g g_ff g_m9 g_m9_ff"
# --- k = 1, RGB attached and downloaded as in every earlier measurement (the 68 fps baseline)
$D/run_one.sh k1_hl_vpoff   --k 1 --variants $V1 cpu cpu_repo --save-seq cpu cpu_repo g g_ff g_m9 g_m9_ff --audit g g_m9_ff
$D/run_one.sh k1_hl_vpon    --k 1 --variants $V1 --no-check --viewport
$D/run_one.sh k1_gui_vpoff  --k 1 --variants $V1 --no-check --gui
$D/run_one.sh k1_gui_vpon   --k 1 --variants $V1 --no-check --gui --viewport --audit g_m9_ff
# --- k = 2, fused optics, thermal only
$D/run_one.sh k2_hl_vpoff   --k 2 --optics fused --no-rgb --variants $V2 --save-seq g g_m9_ff --audit g_m9_ff
$D/run_one.sh k2_hl_vpon    --k 2 --optics fused --no-rgb --variants $V2 --no-check --viewport
$D/run_one.sh k2_gui_vpoff  --k 2 --optics fused --no-rgb --variants $V2 --no-check --gui
$D/run_one.sh k2_gui_vpon   --k 2 --optics fused --no-rgb --variants $V2 --no-check --gui --viewport --audit g_m9_ff
# --- k = 1 thermal only, and k = 2 with RGB: what the RGB annotator costs at each
$D/run_one.sh k1_hl_vpoff_norgb  --k 1 --no-rgb --variants r0 s12 g g_m9_ff --no-check
$D/run_one.sh k1_gui_vpon_norgb  --k 1 --no-rgb --variants r0 s12 g g_m9_ff --no-check --gui --viewport
$D/run_one.sh k2_hl_vpoff_rgb    --k 2 --optics fused --variants s12 g g_m9_ff --no-check
$D/run_one.sh k2_hl_vpon_rgb     --k 2 --optics fused --variants s12 g g_m9_ff --no-check --viewport
$D/run_one.sh k2_gui_vpoff_rgb   --k 2 --optics fused --variants s12 g g_m9_ff --no-check --gui
$D/run_one.sh k2_gui_vpon_rgb    --k 2 --optics fused --variants s12 g g_m9_ff --no-check --gui --viewport
# --- k = 2 with the repo's two-pass optics, for what the fused kernel is worth there
$D/run_one.sh k2_hl_vpoff_repo   --k 2 --optics repo --no-rgb --variants s12 g g_m9_ff
# --- k = 4 (the repo default)
$D/run_one.sh k4_hl_vpoff        --k 4 --optics fused --no-rgb --variants r0 s12 g g_m9_ff --save-seq g g_m9_ff
$D/run_one.sh k4_gui_vpon        --k 4 --optics fused --no-rgb --variants r0 s12 g g_m9_ff --no-check --gui --viewport
$D/run_one.sh k4_hl_vpoff_repo   --k 4 --optics repo --no-rgb --variants s12 g --no-check --passes 1
echo "=== matrix done ==="
