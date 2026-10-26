#!/usr/bin/env bash
# The pair, interleaved, in one sitting: iso, clear, iso, clear. Drift shows as run 1 vs run 2.
B=/tmp/irsim_render_cost
for i in 1 2; do
  $B/run_profile.sh pair_iso_$i   iso   $B/spans.txt $B/lines.txt
  $B/run_profile.sh pair_clear_$i clear $B/spans.txt $B/lines.txt
done
echo "=== pair done ==="
