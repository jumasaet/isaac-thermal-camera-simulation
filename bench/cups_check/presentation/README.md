# Cups experiment: weekly presentation

All figures here were generated on 2026-10-26 from `bench/cups_check/` (details, in Spanish: `../RESULTADOS.md`).

| Figure | What it shows |
|---|---|
| `figures_en/cups_<case>_{whitehot,jet}.png` | Real calibrated vs. simulated thermal image, one sheet per case: same fixed window and palette, °C colour bar, mean apparent water temperature and the difference |
| `figures_en/cups_summary_{whitehot,jet}.png` | The four cases on one slide |
| `rgb_thermal_{hot_water_50,cold_water_95}_{whitehot,jet}.png` | Four panels: real RGB, simulated RGB, real calibrated thermal, simulated thermal (thermal pair on one window) |
| `rgb_alignment_{hot_water_50,cold_water_95}.png` | Zoom on the cup: real (magenta) vs. simulated (cyan) position in the RGB image |
| `world_{hot_water_50,cold_water_95}.png` | Outside view of the simulated world, with both cameras and their fields of view |
| `results_table.{png,md}` | The four cases: real and simulated temperature, difference, independence, 10–90 % edge widths |

## Cup misalignment, real vs. simulated RGB

Measured in real-image pixels (1280 × 960), as simulated − real. The simulated RGB camera uses the RGB-thermal
calibration (intrinsics, distortion, and RGB→thermal extrinsics with 2.2 px stereo error in thermal pixels,
≈ 3.5 RGB px). It hangs off the thermal camera's pose, which was fitted to the thermal cup rows (task 2).

| Case | Axis (x) | Bottom edge | Top edge | Foot width | Height |
|---|---:|---:|---:|---:|---:|
| Hot water, 50 cm | +9.3 px | −18.6 px | −10.0 px | +0.5 px | −8.6 px (−7 %) |
| Cold water, 95 cm | +7.0 px | −1.0 px\* | +4.0 px | +2.0 px | −5.0 px (−8 %) |

- The simulated cup sits 7–9 px to the right at both distances.
- At 50 cm it also sits 10–19 px higher, about 1° of pitch at f = 993 px. That is the size of the ~1.4° pitch
  error already seen in the LiDAR → RGB → thermal chain (task 2), now showing on the RGB side.
- The simulated cup is 5–9 px (7–8 %) shorter.

\* At 95 cm the cup's contact shadow is not resolved, so the real bottom edge there is the printed band plus the
paper strip, scaled from its 50 cm measurement (±2 px).

**How the real cup is measured:** foot width and axis from the printed brown band, which stops 1–3 px short
of the foot; bottom edge from the contact shadow; top edge from the shadow under the rim lip.

**How the simulated RGB is built:** a pinhole render with the calibrated fx, mapped through the calibrated
Brown-Conrady distortion into the real image's pixels. The black corners are where that polynomial stops
being invertible (outside the calibration's data). Colours are flat on purpose: only geometry and framing are
compared.

## Reproduce

```bash
PY=/home/jumasaet/isaac_sim/python.sh     # system ROS variables unset, as in run_all_renders.sh
$PY bench/cups_check/task5_figures.py --lang en --out bench/cups_check/presentation/figures_en
$PY bench/cups_check/presentation/render_rgb_world.py hot_water_50    # Kit, ~75 s each
$PY bench/cups_check/presentation/render_rgb_world.py cold_water_95
$PY bench/cups_check/presentation/make_presentation.py                # host only
```

`render/<case>/` holds the raw renders: the pinhole RGB canvas, the outside view, and `cams.json` with every
camera pose and intrinsic read back from Isaac. `presentation.json` holds the measured numbers.
