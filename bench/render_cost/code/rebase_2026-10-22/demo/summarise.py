#!/usr/bin/env python3
"""Table of the measurement matrix from out/m_*/report.json. Plain Python, no Kit."""
import json, pathlib, collections
O = pathlib.Path("/home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/results/rebase_2026-10-22/demo_out")
CFG = [("hl_vpoff", "sin GUI, viewport off"), ("hl_vpon", "sin GUI, viewport on"),
       ("gui_vpoff", "GUI, viewport off"), ("gui_vpon", "GUI, viewport on")]
rows = collections.defaultdict(list)
sha = collections.defaultdict(set)
tot = collections.Counter()
for key, _ in CFG:
    for f in sorted(O.glob(f"m_{key}_r*/report.json")):
        d = json.loads(f.read_text())
        for p in d["passes"]:
            rows[(key, p["trajectory"])].append((f.parent.name[-2:], p))
            sha[p["trajectory"]].add(p["thermal_sha1"])
            fr = p["freshness"]
            tot["frames"] += p["n_frames"]; tot["lag1"] += fr["lag_hist"].get("1", 0)
            tot["anomalous"] += fr["anomalous_reads"]; tot["repeated"] += fr["repeated_reads"]
            tot["missing"] += len(fr["poses_never_delivered"]); tot["lead_in"] += fr["lead_in_anomalies"]
        for t, a in d["audit"].items():
            for i, x in a["passes"].items():
                tot["audit_passes"] += 1
                tot["thermal_identical_frames"] += x["thermal_frames_identical"]
                tot["rgb_own_pose"] += x.get("rgb_best_match_is_own_pose", 0)
                tot["audit_frames"] += d["n_frames"]
            tot["ref_stamp_bad"] += a["reference_stamp_mismatches"]
print("PER PASS  (session.pass: fps | ms mean / p95 / max | >16.7ms | anomalies)")
for (key, name) in CFG:
    for traj in ("linear", "circular"):
        for sess, p in rows[(key, traj)]:
            m = p["ms"]["total"]; f = p["fps"]
            print(f"  {name:22s} {traj:8s} {sess}.{p['pass']}  {f['sequence']:5.1f} fps (min {f['min']:5.1f} max {f['max']:5.1f} sd {f['sd']:4.1f})"
                  f" | {m['mean']:5.2f} / {m['p95']:5.2f} / {m['max']:6.2f} ms | {100*p['frac_over_16p7ms']:5.1f}% | {p['freshness']['anomalous_reads']}"
                  f" | gpu {p['host_before'].get('gpu_temp_util_power_clock_mem','?')}")
print("\nSUMMARY  (4 passes each: 2 sessions x 2 passes)")
print(f"  {'config':22s} {'traj':8s} {'fps mean':>8s} {'fps range (pass means)':>24s} {'frame fps min..max':>20s} {'sd':>5s} | pose render get fresh tables kern out = total ms")
for (key, name) in CFG:
    for traj in ("linear", "circular"):
        ps = [p for _, p in rows[(key, traj)]]
        if not ps: continue
        f = [p["fps"]["sequence"] for p in ps]
        blk = {b: sum(p["ms"][b]["mean"] for p in ps) / len(ps) for b in ("pose", "render", "get_data", "fresh", "tables", "kernels", "output", "total")}
        print(f"  {name:22s} {traj:8s} {sum(f)/len(f):8.1f} {min(f):11.1f} .. {max(f):5.1f}      {min(p['fps']['min'] for p in ps):6.1f} .. {max(p['fps']['max'] for p in ps):5.1f}"
              f" {sum(p['fps']['sd'] for p in ps)/len(ps):5.1f} | {blk['pose']:.2f} {blk['render']:.2f} {blk['get_data']:.2f} {blk['fresh']:.2f} {blk['tables']:.2f} {blk['kernels']:.2f} {blk['output']:.2f} = {blk['total']:.2f}")
print("\nFRESHNESS", dict(tot))
print("thermal sha1 distinct per trajectory across every pass and config:", {k: len(v) for k, v in sha.items()})
