#!/usr/bin/env python
"""Task 5: the presentation sheet -- RGB and thermal, real and simulated, one temperature bar, and the results table.

    /home/jumasaet/isaac_sim/python.sh bench/cups3_check/task5_figures.py

Reads the outputs of tasks 0-4 (out/*.json, out/real_apparent_t_frames.npy, out/sim/) and writes, in English:
  figures/cups3_sheet_<cmap>.png    RGB real | RGB sim / thermal real | thermal sim (same window, one colour bar)
                                    / real - sim (K) | results table; cmap = inferno and gray (white-hot)
  figures/cups3_results_table.png   the table alone
  figures/figures.json              what each figure shows and which instant
The instant is the middle rendered one (render 4 = real frame 53, 13.2 s into the recording). The thermal
images are apparent temperature (eps = 1 reading) in a fixed window of 5-70 C, the window task 3 uses.
"""
from __future__ import annotations

import json

import numpy as np

import cups3_common as c3

WIN_C = (5.0, 70.0)
DIFF_K = 3.0


def load_json(name: str) -> dict:
    return json.loads((c3.OUT / name).read_text())


def results_rows(t1: dict, t2: dict, t3: dict, t4: dict) -> tuple[list[str], list[list[str]], list[str], list[list[str]]]:
    """Two tables: per cup, and whole-image."""
    cups = t2["cups"]
    head = ["Per cup", "Cold (left)", "Warm (centre)", "Hot (right)"]
    mean = lambda v: float(np.mean(v))  # noqa: E731
    loc = t3["alignment"]["local_per_cup_middle_pair"]
    sil = t3["alignment"]["silhouette_real_minus_sim_px"]
    w = t1["warm_test"]
    a = t1["mc_A_linear_in_time"]["warm_err_surface_k_p2.5_16_50_84_97.5"]
    rows = [
        ["Role in the calibration"] + ["reference", "INDEPENDENT test", "reference"],
        ["Probe water T, mean of 8 instants (°C)"] + [f"{mean(cups[c]['probe_c']):.2f}" for c in c3.CUPS],
        ["Real apparent T, calibrated (°C)"] + [f"{mean(cups[c]['real_c']):.2f}" for c in c3.CUPS],
        ["Simulated apparent T (°C)"] + [f"{mean(cups[c]['sim_board_referenced_c']):.2f}" for c in c3.CUPS],
        ["Error real − sim (K)"] + [f"{cups[c]['error_real_minus_sim_k_mean']:+.2f}" for c in c3.CUPS],
        ["   at first / last frame (K)"] + [f"{cups[c]['error_real_minus_sim_k_first_last'][0]:+.2f} / "
                                              f"{cups[c]['error_real_minus_sim_k_first_last'][1]:+.2f}" for c in c3.CUPS],
        ["Task 1: surface T − probe (K), 68 %", "", f"{w['error_surface_k_mean']:+.2f} [{a[1]:+.2f}, {a[3]:+.2f}]", ""],
        ["Table reflection ρ_near / ρ_mirror"] + [f"{t4['real'][c]['rho_near']:.2f} / {t4['real'][c]['rho_mirror']:.2f}"
                                                  for c in c3.CUPS],
        ["SSIM / LPIPS, cup + reflection window"] + [f"{loc[c]['ssim_as_rendered']:.3f} / {loc[c]['lpips_as_rendered']:.3f}"
                                                            for c in c3.CUPS],
        ["Silhouette real − sim, rim / bottom (px)"] + [f"{sil[c]['top_row']:+.1f} / {sil[c]['bottom_row']:+.1f}" for c in c3.CUPS],
    ]
    m = t3["metrics"]["primary_5_70C"]
    ar, rg = m["as_rendered_mean_min_max"], m["registered_mean_min_max"]
    g = t3["shift_grid_middle_pair"]
    sh = t3["alignment"]["global_shift_rows_cols_px"]
    head2 = ["Whole image, 5–70 °C, 8 pairs", "As rendered", f"Shifted {sh[0]:+.0f}, {sh[1]:+.0f} px"]
    rows2 = [
        ["SSIM, full frame", f"{ar['ssim_full'][0]:.3f}", f"{rg['ssim_full'][0]:.3f}"],
        ["SSIM, cups crop", f"{ar['ssim_crop'][0]:.3f}", f"{rg['ssim_crop'][0]:.3f}"],
        ["LPIPS, full frame", f"{ar['lpips_full'][0]:.3f}", f"{rg['lpips_full'][0]:.3f}"],
        ["LPIPS, cups crop", f"{ar['lpips_crop'][0]:.3f}", f"{rg['lpips_crop'][0]:.3f}"],
        ["SSIM change, crop, 2 / 4 px shift", f"{g['ssim_change_at_2px']:+.3f} / {g['ssim_change_at_4px']:+.3f}", ""],
        ["LPIPS change, crop, 2 / 4 px shift", f"{g['lpips_change_at_2px']:+.3f} / {g['lpips_change_at_4px']:+.3f}", ""],
        ["Spearman (cold, warm, hot, board)", "1.00 (8 of 8)" if t3["rank_correlation"]["spearman_all"] == [1.0]
         else ", ".join(f"{v:.2f}" for v in t3["rank_correlation"]["spearman_all"]),
         " < ".join(t3["rank_correlation"]["order_real"])],
    ]
    return head, rows, head2, rows2


def draw_table(ax, head, rows, widths, fs=8.5, highlight=()) -> None:
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=head, loc="upper center", cellLoc="center", colLoc="center", colWidths=widths)
    tb.auto_set_font_size(False)
    tb.set_fontsize(fs)
    tb.scale(1, 1.45)
    for (r, cidx), cell in tb.get_celld().items():
        cell.set_edgecolor("0.75")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("0.92")
        if cidx == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.03
        if (r, cidx) in highlight:
            cell.set_facecolor("#fff3cd")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    c3.FIG.mkdir(exist_ok=True)
    t1, t2, t3, t4 = (load_json(n) for n in ("task1_calibration.json", "task2_comparison.json",
                                             "task3_similarity.json", "task4_reflection.json"))
    simrep = json.loads((c3.OUT / "sim" / "report.json").read_text())
    j = len(simrep["renders"]) // 2
    i = simrep["renders"][j]["real_frame_index"]
    t_s = simrep["renders"][j]["t_since_first_s"]
    real = np.asarray(np.load(c3.OUT / "real_apparent_t_frames.npy", mmap_mode="r")[i], np.float64) - 273.15
    sim = np.load(c3.OUT / "sim" / f"apparent_t_ref_{j}.npy").astype(np.float64) - 273.15  # board-referenced
    shift = t3["alignment"]["global_shift_rows_cols_px"]
    from scipy import ndimage

    sim_reg = ndimage.shift(sim, shift, order=1, mode="nearest")
    rgb_files = c3.frame_files("rgb")
    name = c3.frame_files("thermal")[i]
    stem = name.rsplit("/", 1)[1].rsplit(".", 1)[0]
    rgb_path = next((f for f in rgb_files if stem in f), None)
    if rgb_path is None:  # this set has no RGB (93 RGB for 94 thermal): take the nearest in time
        tt = c3.frame_time_s(name)
        rgb_path = min(rgb_files, key=lambda f: abs(c3.frame_time_s(f) - tt))
    rgb_real = np.asarray(Image.open(rgb_path).convert("RGB"))
    rgb_sim = np.asarray(Image.open(c3.OUT / "sim" / "frame_rgb.png").convert("RGB"))
    head, rows, head2, rows2 = results_rows(t1, t2, t3, t4)
    info = {}
    hl = {(5, 2), (7, 2)}  # the independent (warm) results
    crop = (slice(170, 370), slice(40, 560))
    for cmap in ("inferno", "gray"):
        fig = plt.figure(figsize=(14, 21))
        gs = fig.add_gridspec(4, 3, width_ratios=[1, 1, 0.04], height_ratios=[1, 1, 1, 0.95], hspace=0.17, wspace=0.05)
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(rgb_real)
        ax.set_title("Real RGB (oCam-1CGN, 1280 × 960)", fontsize=11)
        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(rgb_sim)
        ax.set_title("Simulated RGB (Isaac Sim, thermal camera pose)", fontsize=11)
        ax = fig.add_subplot(gs[1, 0])
        im = ax.imshow(real, cmap=cmap, vmin=WIN_C[0], vmax=WIN_C[1])
        ax.set_title(f"Real thermal, calibrated (frame {i}, t = {t_s:.1f} s)", fontsize=11)
        ax = fig.add_subplot(gs[1, 1])
        ax.imshow(sim, cmap=cmap, vmin=WIN_C[0], vmax=WIN_C[1])
        ax.set_title("Simulated thermal (CTC-006 camera model, same instant)", fontsize=11)
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[1, 2]))
        cb.set_label("Apparent temperature (°C, ε = 1)", fontsize=10)
        ax = fig.add_subplot(gs[2, 0])
        ax.imshow(real - sim_reg, cmap="RdBu_r", vmin=-DIFF_K, vmax=DIFF_K)
        ax.set_title(f"Real − simulated, K (sim shifted {shift[0]:+.0f}, {shift[1]:+.0f} px)", fontsize=11)
        ax = fig.add_subplot(gs[2, 1])
        det = np.vstack([real[crop], np.full((6, real[crop].shape[1]), np.nan), sim[crop]])
        dim = ax.imshow(det, cmap=cmap, vmin=20, vmax=40)
        ax.set_title("Detail, window 20–40 °C: real (top) / simulated (bottom)", fontsize=11)
        cax = ax.inset_axes([0.2, -0.06, 0.6, 0.03])
        cbd = fig.colorbar(dim, cax=cax, orientation="horizontal")
        cbd.ax.tick_params(labelsize=8)
        dm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(-DIFF_K, DIFF_K), cmap="RdBu_r")
        cb2 = fig.colorbar(dm, cax=fig.add_subplot(gs[2, 2]))
        cb2.set_label("Real − sim (K)", fontsize=10)
        for a_ in fig.axes:
            if a_ not in (cb.ax, cb2.ax, cbd.ax):
                a_.set_xticks([])
                a_.set_yticks([])
        sub = gs[3, :].subgridspec(1, 2, width_ratios=[1.5, 1], wspace=0.03)
        draw_table(fig.add_subplot(sub[0]), head, rows, [0.42, 0.17, 0.24, 0.17], fs=8, highlight=hl)
        ax2 = fig.add_subplot(sub[1])
        draw_table(ax2, head2, rows2, [0.44, 0.22, 0.34], fs=8)
        ax2.text(0.0, 0.12, "Calibration: hot cup, cold cup and the board at the air temperature (23.9 °C);\n"
                 "the warm cup is the independent test. Probe temperatures interpolated linearly\n"
                 "between the first and the last frame (the data show the cups changed less).\n"
                 "Real and simulated thermal images are both board-referenced apparent temperature.",
                 transform=ax2.transAxes, fontsize=8.5, color="0.3", va="bottom")
        fig.suptitle("Three paper cups (cold, warm, hot) at 0.5 m — real vs simulated LWIR, Cellplus CTC-006-VGA",
                     fontsize=14, y=0.905)
        out = c3.FIG / f"cups3_sheet_{cmap}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
        info[out.name] = {"real_frame_index": int(i), "render": j, "t_since_first_s": t_s, "window_c": WIN_C,
                          "rgb_real": rgb_path.rsplit("/", 1)[1], "colormap": cmap}
    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1], hspace=0.05)
    draw_table(fig.add_subplot(gs[0]), head, rows, [0.42, 0.17, 0.24, 0.17], fs=10, highlight=hl)
    draw_table(fig.add_subplot(gs[1]), head2, rows2, [0.44, 0.22, 0.34], fs=10)
    fig.suptitle("Three cups — results (real vs simulated, Cellplus CTC-006-VGA)", fontsize=12)
    fig.savefig(c3.FIG / "cups3_results_table.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    info["cups3_results_table.png"] = {"rows": [r[0] for r in rows] + [r[0] for r in rows2]}
    (c3.FIG / "figures.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
