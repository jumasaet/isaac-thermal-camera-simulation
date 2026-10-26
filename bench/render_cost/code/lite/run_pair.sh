#!/usr/bin/env bash
# The pair, interleaved, in one sitting: iso, clear, iso, clear -- renderer as in point 1.
L=/tmp/irsim_render_cost/lite
for i in 1 2; do
  $L/run_time.sh pair_iso_$i   iso   --disable-viewport
  $L/run_time.sh pair_clear_$i clear --disable-viewport
done
$L/run_time.sh vpon_iso iso
echo "=== pair done ==="
