# Layered defocus: a surface reaching under the lens takes every W020 bin and the subject shares the background's kernel

**Where:** `src/irsim/optics/layered.py` → `depth_layers` (OC.6 / OC.13, ADR 0129).

**What happens.** `depth_layers` splits the frame into at most `DEFAULT_MAX_LAYERS = 8` bins of equal width in
W020, over the frame's whole W020 range. In a low camera over a table (bench/cups_check, camera 3.2 cm above
the table, CTC-006 14.6 mm F/1.3 focused at infinity) the table reaches 3.2 cm from the lens, where the blur
circle is ~80 px. That near strip takes seven of the eight bins. One layer spans 0.14–1.29 m and carries the
1.01 m kernel, so a cup at 0.50 m (blur circle 13.6 px) renders with the board's kernel (6.8 px).

**Measured** (bench/cups_check/task6_psf.py --sim): the cup's edge at 50 cm came out identical to the one at
95 cm (Gaussian σ 3.8 px at both), where the lens model gives 4.8 and 3.8 px. With the distance plane floored
at 0.3 m (bench workaround `common.Clock(min_distance_m=0.3)`), the 50 cm edge becomes σ 4.6 px, as expected.

**Why it matters.** Any scene with ground or a floor running under the camera — a vehicle-mounted camera
looking along a road, a robot near the floor — has the same shape: the nearest few centimetres to metres of
ground set the W020 range, and everything of interest lands in one bin.

**Options.**
1. Bins equal-width in W020 but over a clipped range (e.g. W020 percentiles 1–99, with the tails clamped into
   the end bins). This keeps OC.13's rationale, and the near strip still blurs hardest.
2. Make `max_layers` a sensor/fidelity field (schema), so a scene can pay for more layers.
3. Bins equal-width in log W020. Relative blur error stays bounded everywhere, but the error bound OC.6 states
   becomes relative instead of absolute.

**Test that would catch it:** a G-buffer fixture with a ground plane reaching 0.03 m, a target at 0.5 m and a
backdrop at 1 m. Assert the target's edge width matches the single-kernel defocus at 0.5 m within a quarter
pixel.
