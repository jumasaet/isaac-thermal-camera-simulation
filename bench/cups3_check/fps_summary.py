#!/usr/bin/env python
"""Summary of fps_measure.sh: the table and the side-by-side sheet (English).

    /home/jumasaet/isaac_sim/python.sh bench/cups3_check/fps_summary.py <frames_dir_prefix>

Reads out/fps/{A,B}_k{1,2,4}/report.json (two Kit sessions per k) and the frames set A saved
(<prefix>_k<k>/{thermal,rgb}/frame_00000.png: frame 0 is the task-2 pose exactly, the same for every k).

Per frame, run_ref's loop blocks are grouped as:
    render     = pose + render       (write pose and serial; rep.orchestrator.step(wait_for_render=False))
    read       = get_data + fresh + download + store   (annotators -> device arrays, stamp check, device -> host)
    processing = tables + s12 + gpu + sync              (thermal clock and tables, stages 1-2, 3-6 + ADC + flat
                                                          field + M9, one wp.synchronize)
In the timed passes nothing synchronises between blocks, so time spent waiting for the renderer lands in whichever
block first needs its output (mostly `render`/`get_data` and `sync`); the profile pass synchronises after every
block and shows where the device time actually goes.

Writes out/fps/fps_summary.json, out/fps/FPS.md (the table), out/fps/frames_k<k>_{thermal,rgb}.png and
figures/cups3_fps_supersample.png.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
FPS_OUT = HERE / "out" / "fps"
FIG = HERE / "figures"
KS = (1, 2, 4)
GROUPS = {"render": ("pose", "render"), "read": ("get_data", "fresh", "download", "store"),
          "processing": ("tables", "s12", "gpu", "sync")}


def load(tag: str) -> dict:
    return json.loads((FPS_OUT / tag / "report.json").read_text())


def main() -> None:
    prefix = sys.argv[1]
    out = {}
    for k in KS:
        reps = [load(f"{s}_k{k}") for s in ("A", "B")]
        timed = [p for r in reps for p in r["passes"] if p["mode"] == "timed"]
        prof = [p for r in reps for p in r["passes"] if p["mode"] == "profile"]
        n = sum(p["n_frames"] for p in timed)
        tot = sum(p["n_frames"] * p["ms"]["total"]["mean"] for p in timed) / n
        ms = {b: float(np.mean([p["ms"][b]["mean"] for p in timed])) for b in timed[0]["ms"]}
        msp = {b: float(np.mean([p["ms"][b]["mean"] for p in prof])) for b in prof[0]["ms"]}
        sub = {b: float(np.mean([p["sub_ms"].get(b, 0.0) for p in prof])) for b in prof[0]["sub_ms"]}
        out[k] = {
            "render_product": reps[0]["render_product"], "sessions": [r["tag"] for r in reps], "timed_frames": n,
            "fps_mean": 1000.0 / tot, "fps_min": min(p["fps"]["min"] for p in timed),
            "fps_max": max(p["fps"]["max"] for p in timed), "fps_p05": float(np.mean([p["fps"]["p05"] for p in timed])),
            "fps_mean_per_pass": [round(p["fps"]["mean"], 2) for p in timed],
            "ms_total": tot, "ms_total_p95": float(np.mean([p["ms"]["total"]["p95"] for p in timed])),
            "ms_total_max": max(p["ms"]["total"]["max"] for p in timed),
            "ms_groups_timed": {g: sum(ms[b] for b in bs) for g, bs in GROUPS.items()},
            "ms_groups_profile": {g: sum(msp[b] for b in bs) for g, bs in GROUPS.items()},
            "ms_profile_total": msp["total"], "ms_blocks_timed": ms, "ms_blocks_profile": msp, "sub_ms_profile": sub,
            "lag_hist": [p["freshness"]["lag_hist"] for p in timed],
            "anomalous_reads": sum(p["freshness"]["anomalous_reads"] for p in timed),
            "frames_never_delivered": sum(len(p["freshness"]["frames_never_delivered"]) for p in timed),
            "epp": [r["host"]["epp"] for r in reps], "rgb_reduction_check": reps[0].get("rgb_reduction_check"),
        }
        for kind in ("thermal", "rgb"):
            shutil.copy(pathlib.Path(f"{prefix}_k{k}") / kind / "frame_00000.png", FPS_OUT / f"frames_k{k}_{kind}.png")
    # Edge effect in numbers: 8-bit codes against k = 4, whole frame and the zoom window of the sheet
    from PIL import Image

    zy, zx = slice(185, 245), slice(470, 530)
    ref = {kind: np.asarray(Image.open(FPS_OUT / f"frames_k4_{kind}.png").convert("RGB"), np.int16) for kind in ("thermal", "rgb")}
    for k in KS:
        out[k]["diff_vs_k4"] = {}
        for kind in ("thermal", "rgb"):
            d = np.abs(np.asarray(Image.open(FPS_OUT / f"frames_k{k}_{kind}.png").convert("RGB"), np.int16) - ref[kind])
            out[k]["diff_vs_k4"][kind] = {"frame_mean": round(float(d.mean()), 3), "frame_max": int(d.max()),
                                          "zoom_mean": round(float(d[zy, zx].mean()), 3), "zoom_max": int(d[zy, zx].max())}
    (FPS_OUT / "fps_summary.json").write_text(json.dumps(out, indent=1))
    lines = ["| supersample | render product | FPS mean | FPS min | FPS max | ms/frame (p95) | render | read | processing "
             "| profile: render / read / processing (total) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in out.items():
        g, gp = r["ms_groups_timed"], r["ms_groups_profile"]
        lines.append(f"| {k} | {r['render_product'][0]}×{r['render_product'][1]} | {r['fps_mean']:.1f} | {r['fps_min']:.1f} | "
                     f"{r['fps_max']:.1f} | {r['ms_total']:.2f} ({r['ms_total_p95']:.2f}) | {g['render']:.2f} | {g['read']:.2f} | "
                     f"{g['processing']:.2f} | {gp['render']:.2f} / {gp['read']:.2f} / {gp['processing']:.2f} ({r['ms_profile_total']:.2f}) |")
    (FPS_OUT / "FPS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for k, r in out.items():
        print(k, "per pass", r["fps_mean_per_pass"], "lag", r["lag_hist"], "anom", r["anomalous_reads"], "missing",
              r["frames_never_delivered"], "blocks", {b: round(v, 2) for b, v in r["ms_blocks_timed"].items()},
              "profile", {b: round(v, 2) for b, v in r["ms_blocks_profile"].items()}, "rgbchk", r["rgb_reduction_check"],
              "diff_vs_k4", r["diff_vs_k4"])
    sheet(out)


def sheet(out: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    imgs = {(k, kind): np.asarray(Image.open(FPS_OUT / f"frames_k{k}_{kind}.png").convert("RGB")) for k in KS for kind in ("thermal", "rgb")}
    # Zoom: the hot cup's right edge and its rim (thermal), the same window in RGB; nearest neighbour, x4.
    zy, zx = slice(185, 245), slice(470, 530)
    fig, axs = plt.subplots(4, 3, figsize=(15, 17.5), gridspec_kw={"height_ratios": [1, 1, 0.75, 0.75]})
    for j, k in enumerate(KS):
        r = out[k]
        sub = f"k = {k} ({r['render_product'][0]}×{r['render_product'][1]}): {r['fps_mean']:.1f} fps, {r['ms_total']:.1f} ms"
        axs[0, j].imshow(imgs[(k, "thermal")])
        axs[0, j].set_title("Thermal 8-bit, " + sub, fontsize=10)
        axs[1, j].imshow(imgs[(k, "rgb")])
        axs[1, j].set_title("RGB, " + sub, fontsize=10)
        for row, kind in ((2, "thermal"), (3, "rgb")):
            axs[row, j].imshow(imgs[(k, kind)][zy, zx], interpolation="nearest")
            d = out[k]["diff_vs_k4"][kind]
            axs[row, j].set_title(f"{kind}, hot cup rim/edge ×4 — vs k=4: mean |Δ| {d['zoom_mean']:.2f}, max {d['zoom_max']} codes",
                                  fontsize=8.5)
        for row in (0, 1):
            axs[row, j].add_patch(plt.Rectangle((zx.start, zy.start), zx.stop - zx.start, zy.stop - zy.start, fill=False,
                                                ec="c", lw=1))
    for a in axs.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("GPU path, Cellplus CTC-006 (640×480), three-cup scene, same pose — effect of the supersample factor\n"
                 "No defocus: the GPU path applies only the in-focus PSF (diffraction + 80 µm aberration); "
                 "layered defocus exists only on the host path", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "cups3_fps_supersample.png", dpi=95)


if __name__ == "__main__":
    main()
