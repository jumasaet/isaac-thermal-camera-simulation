#!/usr/bin/env python
"""Timing tables from run_full.py reports. Plain Python, no Kit.

    summarise.py [tag ...]        default: every session under out/ that has a report

Per session and variant, over the timed passes (2 trajectories x 2 passes unless noted):
fps mean / min / max / sd, ms per frame, the loop blocks, and -- from the profiled pass, where
every block is synchronised -- the device sub-blocks. Lag is the freshness histogram summed.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

OUT = pathlib.Path("/home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/results/rebase_2026-10-22/full_out")
tags = sys.argv[1:] or sorted(p.parent.name for p in OUT.glob("*/report.json") if not p.parent.name.startswith("smoke"))
LOOP = ("pose", "render", "get_data", "fresh", "tables", "kernels", "gpu", "sync", "cpu", "output")
summary: dict = {}
for tag in tags:
    r = json.loads((OUT / tag / "report.json").read_text())
    rows = {}
    for v in r["variants"]:
        ps = [p for p in r["passes"] if p["variant"] == v and p["mode"] == "timed"]
        pr = [p for p in r["passes"] if p["variant"] == v and p["mode"] == "profile"]
        if not ps:
            continue
        tot = np.array([p["ms"]["total"]["mean"] for p in ps])
        row = {
            "n_passes": len(ps), "ms": float(tot.mean()), "ms_pass_spread": float(tot.max() - tot.min()),
            "fps_mean": float(np.mean([p["fps"]["sequence"] for p in ps])),
            "fps_min": float(np.min([p["fps"]["min"] for p in ps])), "fps_max": float(np.max([p["fps"]["max"] for p in ps])),
            "fps_sd": float(np.sqrt(np.mean([p["fps"]["sd"] ** 2 for p in ps]))),
            "fps_p05": float(np.min([p["fps"]["p05"] for p in ps])),
            "ms_p95": float(np.max([p["ms"]["total"]["p95"] for p in ps])), "ms_max": float(np.max([p["ms"]["total"]["max"] for p in ps])),
            "over_33ms": float(np.mean([p["frac_over_33ms"] for p in ps])), "over_16p7ms": float(np.mean([p["frac_over_16p7ms"] for p in ps])),
            "loop": {k: float(np.mean([p["ms"][k]["mean"] for p in ps])) for k in LOOP},
            "lag": {k: sum(p["freshness"]["lag_hist"].get(k, 0) for p in ps) for k in ("0", "1", "2", "unmatched")},
            "anomalies": sum(p["freshness"]["anomalous_reads"] for p in ps),
            "never_delivered": sum(len(p["freshness"]["poses_never_delivered"]) for p in ps),
        }
        if pr:
            keys = sorted({k for p in pr for k in p["sub_ms"]}, key=lambda k: list(pr[0]["sub_ms"]).index(k) if k in pr[0]["sub_ms"] else 99)
            row["profile_sub"] = {k: float(np.mean([p["sub_ms"].get(k, 0.0) for p in pr])) for k in keys}
            row["profile_ms"] = float(np.mean([p["ms"]["total"]["mean"] for p in pr]))
            row["profile_s12"] = float(np.mean([p["ms"]["kernels"]["mean"] for p in pr]))
            row["profile_gpu"] = float(np.mean([p["ms"]["gpu"]["mean"] for p in pr]))
        rows[v] = row
    summary[tag] = {"gui": r["gui"], "viewport": r["viewport_updates_enabled"], "rgb": r["rgb"], "k": r["k"], "optics": r["optics"],
                    "epp": r["epp"], "rendermode": r["rendermode"], "rows": rows, "check_fails": r.get("check_fails"),
                    "audit": r.get("audit"), "checked": bool(r.get("check"))}
    print(f"\n== {tag}: k={r['k']} ({r['render_product'][0]}x{r['render_product'][1]}) optics={r['optics']} gui={r['gui']} "
          f"viewport_updates={r['viewport_updates_enabled']} rgb={r['rgb']} epp={r['epp']} {r['rendermode']} | check: "
          f"{'not run' if not r.get('check') else ('FAILS ' + str(r['check_fails']) if r['check_fails'] else 'passed')} ==")
    print("variant    fps mean   min   max    sd |  ms    p95    max  >33ms | pose render  get fresh tables  s12   gpu  sync   cpu  out | lag(0/1/2/unm) anom")
    for v, x in rows.items():
        lp = x["loop"]
        print(f"{v:9s} {x['fps_mean']:7.1f} {x['fps_min']:6.1f} {x['fps_max']:5.1f} {x['fps_sd']:5.1f} | {x['ms']:5.2f} {x['ms_p95']:6.2f} {x['ms_max']:6.2f} "
              f"{100 * x['over_33ms']:5.1f}% | " + " ".join(f"{lp[k]:5.2f}" for k in LOOP)
              + f" | {x['lag']['0']}/{x['lag']['1']}/{x['lag']['2']}/{x['lag']['unmatched']} {x['anomalies']}")
    for v, x in rows.items():
        if "profile_sub" in x:
            print(f"   {v:8s} profiled {x['profile_ms']:.2f} ms: s12 {x['profile_s12']:.2f} | "
                  + " ".join(f"{k}={val:.2f}" for k, val in x["profile_sub"].items()))
    if r.get("audit"):
        for k, a in r["audit"].items():
            print(f"   audit {k}: {a['frames_closest_to_own_pose']}/{a['n']} delivered frames closest to their own pose; hist {a['closest_hist']}; "
                  f"mse own {a['mse_own_pose']:.1f} vs neighbour {a['mse_neighbour_pose']:.1f}")
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
