#!/usr/bin/env bash
# The pair, interleaved, in one sitting: iso, clear, iso, clear -- renderer as in point 1.
L=/home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/code/rebase_2026-10-22/lite
for i in 1 2; do
  $L/run_time.sh pair_iso_$i   iso   --disable-viewport
  $L/run_time.sh pair_clear_$i clear --disable-viewport
done
$L/run_time.sh vpon_iso iso
echo "=== pair done ==="
