#!/usr/bin/env bash
# The pair, interleaved, in one sitting (drift shows as run 1 vs run 2); then the variants.
H=/tmp/irsim_render_cost/hostcopy
for i in 1 2; do
  $H/run_one.sh pair_iso_$i   iso   --n 100 --n-ctx 8
  $H/run_one.sh pair_clear_$i clear --n 100 --n-ctx 8
done
echo "=== pair done ==="
$H/run_one.sh vpoff_iso   iso   --n 100 --n-ctx 8 --disable-viewport --skip L,H,K,P
$H/run_one.sh vpoff_clear clear --n 100 --n-ctx 8 --disable-viewport --skip L,H,K,P
$H/run_one.sh waitidle_iso iso --n 60 --skip G,L,H,P,R --/app/hydraEngine/waitIdle=true --/app/updateOrder/checkForHydraRenderComplete=1000
$H/run_one.sh notick_iso   iso --n 60 --skip G,L,H,P,R --/rtx/hydra/supportMultiTickRate=false
echo "=== all done ==="
