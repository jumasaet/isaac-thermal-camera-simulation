"""Tables from the host-copy bench. Usage: report.py [tag ...]  (default: the pair)"""
import csv
import datetime as dt
import json
import pathlib
import statistics as st
import sys

O = pathlib.Path("/tmp/irsim_render_cost/hostcopy/out")
tags = sys.argv[1:] or ["pair_iso_1", "pair_clear_1", "pair_iso_2", "pair_clear_2"]
runs = {t: json.loads((O / f"{t}.json").read_text()) for t in tags if (O / f"{t}.json").exists()}
W = 16


def row(label, vals, fmt="{:.1f}"):
    print(f"{label:<46}" + "".join(
        f"{(fmt.format(v) if isinstance(v, (int, float)) else str(v)):>{W}}" for v in vals))


def g(r, *path, default=float("nan")):
    cur = r["phases"]
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


print(f"{'':<46}" + "".join(f"{t:>{W}}" for t in runs))
for t, r in runs.items():
    if r["errors"]:
        print(f"!! {t} errors: {list(r['errors'])}")
row("governor / EPP", [f"{r['governor']['scaling_governor']}/{r['governor']['epp'][:4]}" for r in runs.values()])
row("mean cpu MHz at marks", [st.fmean(m["cpu_mhz"] for m in r["marks"]) for r in runs.values()], "{:.0f}")

print("\n-- G: inside camera.get_outputs() (ms / frame)")
for ph in ("G:ctx_cpu_1", "G:ctx_cuda_1", "G:ctx_cpu_2", "G:ctx_cuda_2"):
    for k in ("get_outputs", "step", "read"):
        row(f"{ph} {k}", [g(r, ph, k, "mean") for r in runs.values()])

print("\n-- T: tight loop, reader.step / reader.read (ms, mean)")
for ph in ("T:cpu_1", "T:cuda_1", "T:cpu_2", "T:cuda_2"):
    row(f"{ph} step", [g(r, ph, "repo", "step", "mean") for r in runs.values()])
    row(f"{ph} step sd", [g(r, ph, "repo", "step", "sd") for r in runs.values()])
    row(f"{ph} read (repo: to host)", [g(r, ph, "repo", "read", "mean") for r in runs.values()], "{:.2f}")
    row(f"{ph} read (handles only)", [g(r, ph, "handles", "read", "mean") for r in runs.values()], "{:.2f}")
    if "cuda" in ph:
        row(f"{ph} read (1 plane, GPU kernel)", [g(r, ph, "gpu", "read", "mean") for r in runs.values()], "{:.2f}")
for ph in ("T:none_1", "T:none_2"):
    row(f"{ph} step, nothing attached", [g(r, ph, "none", "step", "mean") for r in runs.values()])
row("ToHost nodes in cpu mode", [g(r, "T:cpu_1", "nodes", "omni.syntheticdata.SdPostRenderVarToHost", default=0) for r in runs.values()], "{}")
row("ToHost nodes in cuda mode", [g(r, "T:cuda_1", "nodes", "omni.syntheticdata.SdPostRenderVarToHost", default=0) for r in runs.values()], "{}")
row("TextureToBuffer nodes in cpu mode", [g(r, "T:cpu_1", "nodes", "omni.syntheticdata.SdPostRenderVarTextureToBuffer", default=0) for r in runs.values()], "{}")
row("TextureToBuffer nodes in cuda mode", [g(r, "T:cuda_1", "nodes", "omni.syntheticdata.SdPostRenderVarTextureToBuffer", default=0) for r in runs.values()], "{}")

print("\n-- D: orchestrator.step taken apart")
for ph in ("D:cpu", "D:cuda", "D:none"):
    row(f"{ph} app.update() per step", [g(r, ph, "updates_per_step", "mean") for r in runs.values()], "{:.2f}")
    row(f"{ph} step ms", [g(r, ph, "step", "mean") for r in runs.values()])
    row(f"{ph} ms per update (first)", [g(r, ph, "first_update", "mean") for r in runs.values()], "{:.2f}")
    row(f"{ph} ms per update (later)", [g(r, ph, "later_updates", "mean") for r in runs.values()], "{:.2f}")
    row(f"{ph} ms outside the updates", [g(r, ph, "outside_updates", "mean") for r in runs.values()], "{:.2f}")

print("\n-- R: one raw app.update() (ms)")
for k in ("update_render_product_off_viewport_off", "update_render_product_off", "update_no_annotators"):
    row(k + " p50", [g(r, "R:raw", k, "p50") for r in runs.values()], "{:.2f}")
for k in ("update_cuda_annotators", "update_cpu_annotators", "update_cuda_annotators_viewport_off",
          "update_cpu_annotators_viewport_off"):
    row(k + " p50", [g(r, "R:raw", k, "p50") for r in runs.values()], "{:.2f}")
row("viewport", [str(g(r, "R:raw", "viewport_resolution", default="-")) for r in runs.values()], "{}")

print("\n-- L: raw update p50 ms against channels attached (viewport on | off)")
for dev in ("cpu", "cuda"):
    keys = list(next(iter(runs.values()))["phases"].get(f"L:{dev}", {}))
    for k in keys:
        row(f"{dev} {k[:38]}", [f"{g(r, f'L:{dev}', k, 'vp_on'):.1f} | {g(r, f'L:{dev}', k, 'vp_off'):.1f}"
                                for r in runs.values()], "{}")

print("\n-- H: get_data() for all channels (ms)")
for k in ("cpu_copy1", "cpu_copy0", "cuda_copy1", "cuda_copy0"):
    row(f"{k} get_data all", [g(r, "H:get_data", k, "get_data_all") for r in runs.values()], "{:.2f}")
    row(f"{k} then .numpy()", [g(r, "H:get_data", k, "then_to_host") for r in runs.values()], "{:.2f}")
    row(f"{k} step", [g(r, "H:get_data", k, "step") for r in runs.values()])

print("\n-- LAG: ms per advance | app.update() count | frames late [min,max]")
for dev in ("cuda", "cpu"):
    keys = [k for k in next(iter(runs.values()))["phases"].get(f"LAG:{dev}", {}) if k != "offset_m"]
    for k in keys:
        def cell(r):
            o = g(r, f"LAG:{dev}", k, default=None)
            if not isinstance(o, dict):
                return "-"
            return f"{o['advance']['mean']:.1f}|{o['updates_per_advance']:.1f}|{o['lag_frames_mode']}[{o['lag_min']},{o['lag_max']}]"
        row(f"{dev} {k}", [cell(r) for r in runs.values()], "{}")

print("\n-- V: default viewport on / off (repo read)")
for k in ("cpu_on", "cpu_off", "cuda_on", "cuda_off"):
    row(f"{k} step", [g(r, "V:viewport_off", k, "step", "mean") for r in runs.values()])
for dev in ("cuda", "cpu"):
    for name in ("orch_wait", "orch_nowait", "orch_nowait_plus1_update", "orch_nowait_plus2_updates", "raw_update"):
        def cell(r):
            o = g(r, "V:viewport_off", f"{dev}_off:{name}", default=None)
            if not isinstance(o, dict):
                return "-"
            return f"{o['advance']['mean']:.1f}|{o['updates_per_advance']:.1f}|{o['lag_frames_mode']}[{o['lag_min']},{o['lag_max']}]"
        row(f"vp off, {dev} {name}", [cell(r) for r in runs.values()], "{}")

print("\n-- P: timeline playing")
for k in ("raw_update_playing", "raw_2_updates_playing"):
    def cell(r):
        o = g(r, "P:playing", k, default=None)
        if not isinstance(o, dict):
            return "-"
        return f"{o['advance']['mean']:.1f}|{o['updates_per_advance']:.1f}|{o['lag_frames_mode']}[{o['lag_min']},{o['lag_max']}]"
    row(k, [cell(r) for r in runs.values()], "{}")

print("\n-- GPU utilisation % (nvidia-smi, mean over the phase)")
util = {}
for t, r in runs.items():
    f = O / f"{t}.smi.csv"
    samples = []
    if f.exists():
        for line in csv.reader(f.open()):
            try:
                ts = dt.datetime.strptime(line[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                samples.append((ts, float(line[1])))
            except Exception:
                pass
    marks = r["marks"]
    util[t] = {}
    for a, b in zip(marks, marks[1:]):
        vals = [u for ts, u in samples if a["wall"] + 0.5 <= ts <= b["wall"]]
        if vals:
            util[t][a["name"]] = st.fmean(vals)
names = []
for t in util:
    for n in util[t]:
        if n not in names:
            names.append(n)
for n in names:
    row(n, [util[t].get(n, float("nan")) for t in runs], "{:.0f}")
