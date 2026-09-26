# ADR 0123 — A prescribed surface temperature is a Dirichlet cell inside the solve

**Status:** Accepted
**Date:** 2026-09-26
Physics: docs/physics-model.md §6.1 (the balance), §6.4 (the integrator), §6.6 ("scripted, not
predicted"). Builds on [ADR 0087](0087-point-wise-surface-temperature.md) (patches),
[ADR 0094](0094-imex-conduction-and-the-explicit-guard.md) (implicit conduction) and
[ADR 0108](0108-still-water-as-a-region-of-a-patch.md) (a region of a patch).

## Context

Synthetic data for thermal anomaly detection (GITCC/EMPLOI WP4: a quadruped inspecting an
industrial site) needs the anomaly at a temperature the **dataset** chooses: a steam line at
420 K, a bearing 25 K over its housing, a failed insulation panel. Every surface in a scene's
`thermal:` block was solved from the weather, so an author could only get a hot spot by tuning a
material or a heat source until the balance happened to land near the number wanted. That makes
the label a by-product of the forcing and not a controlled variable, and a sweep over anomaly
magnitude impossible.

§6.6 already names the right idea for vehicle heat sources, "scripted, not predicted", and
`PrescribedSolver` implements it for a target node. Neither reached a patched surface's cells.

## Options considered

1. **A uniform `prescribed_k` on `SurfaceSpec`.** Simplest, but a hot spot is then a separate
   surface with its own prim, so an anomaly's position means re-authoring geometry, and its
   neighbours never feel it.
2. **An `anomaly: {region_m, temperature_k}` block.** Localised, but names a dataset label in the
   physics schema: a steam line at its design temperature is not an anomaly.
3. **A `prescribed:` list of `{temperature_k, region_m?}`, the cells held as a Dirichlet boundary
   inside the `FacetSolver` step.** Covers the uniform case (no `region_m`) and the localised one;
   several hot spots per surface; the cell selection is `water:`'s and `film:`'s (cell centre in
   `[u0, u1, v0, v1]`).
4. **Paint the prescribed value over the solved field after each tick.** Rejected: the
   neighbours would exchange heat with a cell the solve believes is somewhere else, so a hot spot
   would have no halo, which is the most recognisable thing about a real one.

## Decision

Option 3. `SurfaceSpec.prescribed` (schema v16) lists `PrescribedSpec` entries; `FacetSolver`
takes `pinned_k` (kelvin per facet, NaN where free). A pinned facet:

* is set to its value in the initial state, in the midpoint state and in the explicit update, so
  the balance never moves it and any evaporation from it is evaluated at its real temperature;
* is an **identity row** in the backward-Euler conduction matrix (`ConductionOperator.
  implicit_matrix(..., fixed=)`), so the free neighbours are coupled to exactly the prescribed
  value inside the implicit step, then set again to remove the LU's last-bit rounding;
* is excluded from §6.4's explicit bound, since it is never stepped.

The pin holds **through the spin-up** as well: a steam line that has been hot for days has
already warmed its surroundings, so the halo is present at t₀. A surface prescribed whole is also
pinned in the per-prim solve, so the value the render bridge carries for pixels outside the patch
agrees with its cells. A surface prescribed on a region keeps its solved per-prim value.

Emissivity, solar absorptivity and heat capacity stay the material's (ADR 0043): only the
temperature is imposed, so the camera renders a prescribed cell through exactly the same
radiometry as a solved one. `PlanarThermalField.pinned_k` exposes the held cells, which is the
ground-truth mask a dataset exporter keys on.

`PrescribedSolver` is not reused: it is a scalar node with its own clock behind the
`TemperatureSolver` protocol, and a cell is a row of an array solve. A time-varying prescription
(a bearing warming over a shift) would reuse its `np.interp` schedule when it is needed.

Refused by the schema, rather than silently half-supported: `prescribed:` without a patch (a
mesh or per-prim surface), on a layered surface, beside `water:`, on a cabin panel, overlapping
regions, and an unregioned entry beside another. A region that holds no cell centre raises at
build time: an anomaly smaller than a cell would otherwise vanish from the ground truth.

## Consequences

* A scene can place an anomaly of chosen size, position and temperature on any patched surface
  from YAML, and its surroundings respond physically.
* A prescribed cell is an infinite reservoir: it is not in the field's energy budget, and a
  conservation check over a field with pins has to exclude them.
* A surface with a pin is spun up per cell (like one under occluders), which costs the per-cell
  spin-up once per scene build. The spin-up cache key includes the pins only when present, so
  every existing key is unchanged.
* Every scene without `prescribed:` is bit-identical: `_pin` is the identity without pins, and no
  per-cell spin-up is triggered.

## Revisit when

* An anomaly must change over time (a warm-up, an intermittent fault): add a schedule to
  `PrescribedSpec`, evaluated per tick.
* Anomalies are needed on curved geometry: extend the same pin to `TriangleMeshField` (WM.7).
* A layered surface needs one: the pin would hold the top layer and let the substrate respond.
