#!/usr/bin/env bash
# The four renderer configurations, both trajectories, two passes per session, two sessions each.
# Sessions are interleaved (r1: all four, r2: all four) so a slow drift does not land on one config.
D=/tmp/irsim_render_cost/demo
powerprofilesctl get
for r in 1 2; do
  $D/run_one.sh m_hl_vpoff_r$r  --no-save
  $D/run_one.sh m_hl_vpon_r$r   --no-save --viewport
  $D/run_one.sh m_gui_vpoff_r$r --no-save --gui
  $D/run_one.sh m_gui_vpon_r$r  --no-save --gui --viewport
done
# Deliverables: frames + MP4, lights as in the brief; once at Kit's exposure, once 2 stops down.
$D/run_one.sh deliver_iso100 --passes 1
$D/run_one.sh deliver_iso25  --passes 1 --film-iso 25
echo "=== matrix done ==="
