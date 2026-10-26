# DRAFT — not filed

Status: draft written 2026-10-23, not opened. File it only after the maintainer approves (issue-log skill).

---

**Title:** `scene_apparent_t` is not published on ROS 2 nor emitted by the GPU path

**Labels:** `pipeline`, `isaac`, follow-up

## Context

Commit `4c5517b4` (`feat(radiometry): a frame can carry the scene's apparent temperature before the camera`)
added the optional plane `scene_apparent_t`: L_B⁻¹ of the stage 1+2 radiance, box-averaged in radiance to the
native grid, before the gain ceiling, optics, detector and ISP (docs/physics-model.md §3.3, §12.2 `outputs`).
It is switched on by `outputs.scene_apparent_temperature: true` in the sensor YAML, written by `write_frame` as
float32 with its unit in `PLANE_UNITS`, and verified in `tests/unit/test_scene_apparent_t.py`.

It is the radiometric reference a real camera's frame will be compared against, so every path that produces
frames should be able to carry it.

## What is missing

1. **ROS 2.** `src/irsim_isaac/ros2_bridge.py` publishes from a fixed plane table (`"apparent_t": ("apparent_t",
   "32FC1")`, …). `scene_apparent_t` is not in it, so a live consumer never sees the plane even when the config
   asks for it.
2. **GPU path.** The device pipeline (to be integrated in task 1, today in `bench/render_cost/code/…/fullgpu.py`)
   reproduces `run_frame` stage by stage but does not emit this plane. The stage-2 radiance is already on the
   device there, so the cost should be one box-average and one LUT inversion.

## Done when

- The ROS 2 bridge publishes `scene_apparent_t` as `32FC1` whenever the frame carries it, and does not publish
  zeros when it does not.
- The GPU path emits the plane when the flag is on, and the GPU-vs-CPU comparison
  (`tests/integration/test_kernels_vs_reference.py`, marked `gpu`) checks it against `run_frame`'s plane in mK.
