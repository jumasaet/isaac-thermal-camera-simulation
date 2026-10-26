#!/usr/bin/env python3
"""The timing matrix as tables. usage: summarise.py [out dir]

Per configuration, over every timed pass of both rounds (240 frames each):
    mean  frames / total time (so a slow pass weighs what it took)
    min, max  of the instantaneous rate 1 / frame time, over all frames
    sd    of the instantaneous rate within a pass, pooled over passes (root mean square)
    passes  the range of the per-pass means -- the run-to-run spread
"""
import json
import math
import pathlib
import sys

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/irsim_render_cost/ref/out")
BLOCKS = ("pose", "render", "get_data", "fresh", "tables", "s12", "gpu", "sync", "download", "store", "publish")
CONFIGS = [("headless, sin ROS", "m_hl_noros"), ("headless, ROS sin suscriptor", "m_hl_ros_nosub"),
           ("headless, ROS + suscriptor", "m_hl_ros_sub"), ("GUI, sin ROS", "m_gui_noros"),
           ("GUI, ROS sin suscriptor", "m_gui_ros_nosub"), ("GUI, ROS + suscriptor", "m_gui_ros_sub")]


def load(prefix):
    timed, prof, meta = [], [], []
    for d in sorted(OUT.glob(prefix + "_r*")):
        f = d / "report.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        meta.append(r)
        for p in r["passes"]:
            (timed if p["mode"] == "timed" else prof).append(p)
    return timed, prof, meta


rows = {}
print("| Configuración | fps medio | mín | máx | σ | medias por pasada | ms/frame | frescura (lag 1 / leídos) | recepción |")
print("|---|---|---|---|---|---|---|---|---|")
for label, prefix in CONFIGS:
    timed, prof, meta = load(prefix)
    if not timed:
        continue
    n = sum(p["n_frames"] for p in timed)
    t = sum(p["n_frames"] / p["fps"]["mean"] for p in timed)
    means = [p["fps"]["mean"] for p in timed]
    sd = math.sqrt(sum(p["fps"]["sd"] ** 2 for p in timed) / len(timed))
    lag1 = sum(p["freshness"]["lag_hist"].get("1", 0) for p in timed)
    anomalies = sum(p["freshness"]["anomalous_reads"] for p in timed)
    missing = sum(len(p["freshness"]["frames_never_delivered"]) for p in timed)
    rec = []
    for d in sorted(OUT.glob(prefix + "_r*")):
        f = d / "reception.txt"
        if f.exists():
            lines = [json.loads(x) for x in f.read_text().splitlines() if x.startswith("{")]
            rec.append("+".join(f"{x['received']}/{x['sent']}" for x in lines) + (" ok" if "RECEPTION OK" in f.read_text() else " MAL"))
    rows[prefix] = {"timed": timed, "prof": prof, "fps": n / t}
    print(f"| {label} | {n / t:.1f} | {min(p['fps']['min'] for p in timed):.1f} | {max(p['fps']['max'] for p in timed):.1f} | "
          f"{sd:.1f} | {min(means):.1f} – {max(means):.1f} ({len(timed)}) | {1e3 * t / n:.2f} | {lag1}/{n}"
          f"{'' if not anomalies and not missing else f' ANOMALIAS {anomalies} FALTAN {missing}'} | {'; '.join(rec) or '—'} |")

print()
print("| Bloque (ms, media) | " + " | ".join(lbl for lbl, pre in CONFIGS if pre in rows) + " |")
print("|---|" + "---|" * len(rows))
for b in (*BLOCKS, "total"):
    cells = []
    for _, prefix in CONFIGS:
        if prefix in rows:
            ps = rows[prefix]["timed"]
            cells.append(f"{sum(p['ms'][b]['mean'] for p in ps) / len(ps):.2f}")
    print(f"| {b} | " + " | ".join(cells) + " |")

print()
print("Pasada perfilada (una sincronizacion tras cada bloque), ms medios por kernel:")
for label, prefix in CONFIGS:
    if prefix in rows and rows[prefix]["prof"]:
        ps = rows[prefix]["prof"]
        keys = list(ps[0]["sub_ms"])
        sub = {k: sum(p["sub_ms"].get(k, 0.0) for p in ps) / len(ps) for k in keys}
        s12 = sum(p["ms"]["s12"]["mean"] for p in ps) / len(ps)
        pub = sum(p["ms"]["publish"]["mean"] for p in ps) / len(ps)
        tot = sum(p["ms"]["total"]["mean"] for p in ps) / len(ps)
        print(f"  {label}: total {tot:.2f} | s12 {s12:.2f} | " + " ".join(f"{k}={v:.2f}" for k, v in sub.items()) + f" | publish {pub:.2f}")
for label, prefix in CONFIGS:
    timed, _, meta = load(prefix)
    for m in meta:
        print(f"  {m['tag']}: viewport_updates {m.get('viewport_updates_enabled')} rendermode {m['rendermode']} epp {m['host']['epp']} "
              f"gpu {m['host'].get('gpu_temp_util_power_mem')} ros {m['ros'].get('topics')}")
