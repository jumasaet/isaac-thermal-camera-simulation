# Option (b) vs all-GPU, idle machine, 2026-10-23 11:47-11:54 (measure3.sh)

Headless, 640x512, k 1, fused optics, RealTimePathTracing, camera moving (240-frame path), lag 1 on 240/240
frames in every pass, EPP performance (governor string still reads powersave; intel_pstate active).
No other test or Kit process; load average 0.5 before, 2-6 during (Kit's own threads).
Two timed passes per session, sessions interleaved A B C A B C.

| run | stage 6 | ISP | fps (4 passes) | ms/frame | stage-6 cost in loop |
|---|---|---|---|---|---|
| A all GPU | device | pre-rebase plateau + DDE | 51.7-53.2 | 18.8-19.4 | 1.5 ms (agc_table 1.25 + lut_dde 0.17 + palette 0.07) |
| C host stage 6 | host | pre-rebase plateau + DDE | 34.8-35.4 | 28.2-28.8 | 10.6-11.1 ms + 0.2 download |
| B option (b) | host | Boson information_based | 10.4-10.6 | 94.5-96.1 | 76.8-78.3 ms + 0.2 download |

The loop is serial: host stage 6 adds to the frame, it does not overlap the render (render 9.5 ms in all three).
Bare interpreter, same frames (bare_stage6_cost.json): Boson 104 ms, of which bilateral_split 78 ms; old ISP 15.6 ms.

Check (m3_B_host_newisp_check): passed; stage 6 fed run_frame's display_dn is bit-equal 32/32
(8 windows x 4 poses). Cause of yesterday's 76-code display8 deviation: ../diag_2026-10-23/ and NOTES.md.
