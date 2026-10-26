#!/usr/bin/env python
"""Timing table from run_stages.py reports: mean over every pass (2 trajectories x 2 passes)."""
import json, sys
import numpy as np
V = ("12", "123", "1236", "12356", "123456", "repo")
CPU = ("download", "s3_optics", "s4_transfer", "s4_lag", "s5_tvh", "s5_correlated", "m9_chain", "adc", "s6_flat_field", "s6_display", "fixed_display")
out = {}
for tag in sys.argv[1:]:
    r = json.load(open(f"/tmp/irsim_render_cost/stages/out/{tag}/report.json"))
    rows = {}
    for v in V:
        ps = [p for p in r["passes"] if p["variant"] == v]
        tot = np.array([p["ms"]["total"]["mean"] for p in ps])
        rows[v] = {
            "n_passes": len(ps), "ms": float(tot.mean()), "ms_spread": float(tot.max() - tot.min()),
            "fps": float(np.mean([p["fps"]["sequence"] for p in ps])),
            "p95_ms": float(np.max([p["ms"]["total"]["p95"] for p in ps])), "max_ms": float(np.max([p["ms"]["total"]["max"] for p in ps])),
            "over_33ms": float(np.mean([p["frac_over_33ms"] for p in ps])),
            "render": float(np.mean([p["ms"]["render"]["mean"] for p in ps])),
            "loop_other": float(np.mean([sum(p["ms"][k]["mean"] for k in ("pose", "get_data", "fresh", "tables", "kernels", "output")) for p in ps])),
            "cpu_block": float(np.mean([p["ms"]["cpu"]["mean"] for p in ps])),
            "cpu": {k: float(np.mean([p["cpu_ms"][k]["mean"] for p in ps])) for k in CPU},
            "lag": {k: sum(p["freshness"]["lag_hist"].get(k, 0) for p in ps) for k in ("0", "1", "2", "unmatched")},
            "anomalies": sum(p["freshness"]["anomalous_reads"] for p in ps),
        }
    out[tag] = {"rows": rows, "radiometric_branch_ms": r["radiometric_branch_ms"]["mean"], "seam": r.get("seam")}
    print(f"\n== {tag}  (gui={r['gui']}, epp={r['epp']}, {r['rendermode']}, viewport updates {r['viewport_updates_enabled']}) ==")
    print("variant   ms/frame  fps   p95    max   spread | render other cpu-block | " + " ".join(CPU))
    for v, x in rows.items():
        print(f"{v:8s} {x['ms']:7.2f} {x['fps']:6.1f} {x['p95_ms']:6.1f} {x['max_ms']:6.1f} {x['ms_spread']:5.2f}  | {x['render']:5.2f} {x['loop_other']:5.2f} {x['cpu_block']:6.2f} | "
              + " ".join(f"{x['cpu'][k]:5.2f}" for k in CPU) + f" | lag {x['lag']} anomalies {x['anomalies']}")
    print("radiometric branch (not in any row):", round(out[tag]["radiometric_branch_ms"], 2), "ms")
json.dump(out, open("/tmp/irsim_render_cost/stages/out/summary.json", "w"), indent=1)
