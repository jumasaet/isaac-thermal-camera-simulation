# rebase_2026-10-22 — the bench as it ran after the rebase onto origin/main 15015cd2

Copied 2026-10-23 from the session scratchpad
`/tmp/claude-1000/-home-jumasaet-isaac-thermal-camera-simulation/79b950a6-68e8-41eb-bfb4-83fa28176cf5/scratchpad/irsim_render_cost/`
before a reboot. The tree layout is kept (`full/`, `ref/`, `demo/`, `lite/`), because the scripts refer to each
other by those relative positions. Run outputs are in `../../results/rebase_2026-10-22/`.

**Paths.** 33 files still name the scratchpad above (the LUT dir, the scene YAMLs, `run_one.sh`). That directory
will be gone after the reboot: rewrite those paths before running anything.

## What differs from the 2026-10-11 copies (`../full`, `../ref`, …)

- **Re-verified against the new `run_frame`**: `OpticsTerms.phi_housing` → `housing_base` / `housing_per_ri`
  in `full/fullgpu.py` (SC.34), LUTs read from a regenerated dir, `full/common.py` gains `REVERIFY_OLD_ISP=1`
  (diagnostic only: swaps in the pre-rebase `isp` block so stages 1-5 can be compared on their own).
- **Option (b), stage 6 on the host**: `HOST_DISPLAY=1` makes `full/fullgpu.py` download dn16 and run
  `irsim.isp.display.run_display_branch` with a `TransferDamper`. `full/fullgpu.py.orig` and `ref/run_ref.py.orig`
  are the files just before that change.
- `measure.sh` / `measure2.sh`: the three runs A (all GPU, old ISP), B (host stage 6, Boson's ISP), C (host
  stage 6, old ISP). `measure2.sh` was written but never run.
- `gate_sharded.sh`: the gate on tracked files only, pytest split into 12 processes (no xdist in Isaac's Python).

## What the runs showed (`../../results/rebase_2026-10-22/fps_option_b_2026-10-23_loaded/`)

**The fps numbers are not usable**: they ran while a 12-process pytest saturated the CPU (all-GPU 24-26 fps,
against 49 the day before). Two qualitative findings stand:

1. B fails the check: display8 is up to 76 codes off `run_frame` with the Boson's `information_based` AGC, while
   C (old ISP) matches (max 4-14). The damper state or the σ passed to `run_display_branch` is not run_frame's.
2. Host stage 6 is not cheap: "gpu" goes from ~7 ms to 28-36 ms per frame in C. To be re-measured on an idle
   machine before deciding whether option (b) stands.

## 2026-10-23 (session after the reboot)

**Paths.** The 33 files were rewritten: `<scratchpad>/irsim_render_cost/<x>` -> this folder's `<x>`;
the `out/` dirs -> `../../results/rebase_2026-10-22/{ref,full,demo}_out`; `<scratchpad>/lut` ->
`bench/render_cost/lut` (regenerated with `scripts/generate_luts.py --out bench/render_cost/lut`, 8 bundles).
The three `run_one.sh` write their logs to the same results dirs (`$O`). The six old `run.log`s under
`ref_out/` still name the scratchpad: they are history and were left as they were.

**Why B's display8 is not run_frame's** (`../../results/rebase_2026-10-22/diag_2026-10-23/`).
`DIAG_DUMP=<dir>` makes `ref/run_ref.py`'s check dump both sides' stage-6 inputs per pose;
`ref/diag_stage6.py` and `ref/diag_stage6_amplifier.py` analyse the dump offline. On all 32 checked poses:
- the sigma is the same (run_frame's `PipelineState.sigma_tvh_dn` == the bench's), and on the first pose
  of every window there is no carried damper state at all, yet the deviation is already whole (76 codes
  at pose 40): neither the damping nor the sigma is the cause;
- fed run_frame's display_dn and carrying its own state from a fresh start, the host branch is
  run_frame's display8 bit for bit, 32/32: option (b)'s stage 6 is right;
- the cause is the operator: dn16 differs on ~2000 px by one DN (float32 in stages 3-5, inside the
  repo's tolerance, identical in A, B and C). A neighbour's flip moves the 5x5 bilateral's floor(low)
  one bin at an edge pixel; the equalised transfer's per-bin slope T' jumps between adjacent bins
  (0.19 -> 0.33 codes/DN) and agc_information's y = T[idx] + T'[idx] * high, with |high| = 552 DN there,
  jumps ~78 codes (observed 76). The old ISP has no such term, hence C's <= 1 code away from a DN change.

**Check changed accordingly**: with `HOST_DISPLAY=1`, `equivalence()` runs a shadow `FullGpu`'s
`host_stage6()` (split out of `frame()`) on run_frame's display_dn and demands bit equality
(`stage6_same_input`); with information_based the propagated display8 difference is kept in the row
but no longer fails the check. Stages 1-5 keep their own tolerances.

**Stage 6 alone on the host** (`ref/bare_stage6_cost.py`, real 640x512 frames, idle machine):
Boson information_based 104 ms/frame median (the bilateral split alone 78 ms), old plateau + DDE 15.6 ms.

**fps, idle** (`measure3.sh` -> `../../results/rebase_2026-10-22/fps_option_b_2026-10-23_idle/SUMMARY.md`;
it supersedes `measure2.sh`, whose three runs it contains, with the check on for B and a repeat):
A all-GPU (old ISP) 52 fps / 19 ms; C host stage 6 (old ISP) 35 fps / 28.5 ms; B option (b) (Boson ISP)
10.5 fps / 95 ms, of which 77 ms is the host display branch. Option (b) as built does not hold at 60 Hz.
