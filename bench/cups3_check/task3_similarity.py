#!/usr/bin/env python
"""Task 3: image similarity between the calibrated real frames and the simulated ones.

    PYTHONPATH=<dir with lpips 0.1.4 + scikit-image 0.24> ~/miniconda3/envs/vision/bin/python \
        bench/cups3_check/task3_similarity.py

(The Isaac interpreter has no torch; the `vision` conda env has torch 2.5.1 and numpy 2.2. lpips and
scikit-image were pip-installed with --target into a scratch directory -- `pip install --target <dir> --no-deps
lpips==0.1.4 scikit-image==0.24.0 imageio lazy_loader tifffile networkx`; nothing in the conda env changed. The
AlexNet weights come from torchvision's model zoo on first use.) cups3_common is imported by path below; it needs
only numpy here -- no irsim import.

Pairs: the 8 simulated instants (out/sim/report.json) against the real frame taken at the same time (task 1's
calibrated apparent temperature, eps 0.93 -> G, out/real_apparent_t_frames.npy). The simulated frame is the
camera's apparent_t board-referenced the same way as the real image (out/sim/apparent_t_ref_<j>.npy, written by
task2_compare.py): one radiance offset per frame puts the board at L_B(T_air) in both. Both are apparent
temperatures in K, so one window maps both to [0, 1]:

    x = clip((T_app - T_lo) / (T_hi - T_lo), 0, 1),   window [5, 70] C (the cups' range; primary)
                                                      and [20, 40] C (board/table detail; secondary)

  - SSIM: Wang et al. 2004 with skimage defaults for that paper (Gaussian window sigma 1.5, K1 0.01, K2 0.03,
    population covariance, data_range 1).
  - LPIPS: Zhang et al. 2018, AlexNet, linear layers v0.1, the gray image repeated to 3 channels, scaled to [-1, 1].
  - Both on the full 640 x 480 frame and on a crop around the cups and their reflections (rows 160-380,
    cols 30-570).
  - Rank correlation (Spearman, and Kendall tau) of the brightness of the three cups' water ROIs and the board,
    real against simulated (n = 4).

Misalignment:
  (1) residual position of each cup: its measured silhouette (rim row, bottom row, centre column; task0_scene)
      minus the rendered one (out/sim/report.json) -- edges, not blob centroids (the real cups' table
      reflection drags a centroid down by 3-5 px);
  (2) global registration: the whole-pixel shift of the simulated frame, within +-4 px, that maximises SSIM on
      the crop of the middle pair; every pair re-scored with it ("registered");
  (3) local registration: the same per cup, on a window around each cup and its reflection;
  (4) SSIM and LPIPS on a grid of imposed shifts +-8 px about the as-rendered position: the cost per pixel.
  Reference levels: a real frame against the next real frame (noise and plume only), and against itself
  shifted by 1, 2 and 4 px.

Writes out/task3_similarity.json and out/task3_similarity.png.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
WINDOWS_C = {"primary_5_70C": (5.0, 70.0), "secondary_20_40C": (20.0, 40.0)}
CROP = (slice(160, 380), slice(30, 570))
SHIFTS = np.arange(-8, 9, 2)


def load_common():
    """cups3_common by path. It and bench/cups_check/cups_common import only numpy at module level; irsim is
    imported lazily by the radiometry functions, which are not used here."""
    spec = importlib.util.spec_from_file_location("cups3_common", HERE / "cups3_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def norm(t_k: np.ndarray, win: tuple[float, float]) -> np.ndarray:
    lo, hi = win
    return np.clip((t_k.astype(np.float64) - 273.15 - lo) / (hi - lo), 0.0, 1.0)


def main() -> None:
    import lpips
    import torch
    from scipy import ndimage
    from scipy.stats import kendalltau, spearmanr
    from skimage.metrics import structural_similarity

    c3 = load_common()
    rep_sim = json.loads((OUT / "sim" / "report.json").read_text())
    real_all = np.load(OUT / "real_apparent_t_frames.npy", mmap_mode="r")
    assert real_all.dtype == np.float32
    net = lpips.LPIPS(net="alex", version="0.1", verbose=False).eval()

    def ssim(a, b):
        return float(structural_similarity(a, b, data_range=1.0, gaussian_weights=True, sigma=1.5,
                                           use_sample_covariance=False))

    def lp(a, b):
        ta = torch.from_numpy(np.repeat((a * 2 - 1)[None, None], 3, 1)).float()
        tb = torch.from_numpy(np.repeat((b * 2 - 1)[None, None], 3, 1)).float()
        with torch.no_grad():
            return float(net(ta, tb).item())

    def metrics(a, b):
        return {"ssim_full": ssim(a, b), "lpips_full": lp(a, b), "ssim_crop": ssim(a[CROP], b[CROP]),
                "lpips_crop": lp(a[CROP], b[CROP])}

    pairs = []
    for r in rep_sim["renders"]:
        i, j = r["real_frame_index"], r["render"]
        real = np.asarray(real_all[i], np.float32)
        sim = np.load(OUT / "sim" / f"apparent_t_ref_{j}.npy")  # board-referenced like the real image (task2_compare)
        assert sim.dtype == np.float32 and sim.shape == real.shape
        pairs.append((i, j, real, sim))

    # -- alignment --------------------------------------------------------------------------------------------
    win0 = WINDOWS_C["primary_5_70C"]
    real_m = np.mean([p[2] for p in pairs], axis=0)
    sim_m = np.mean([p[3] for p in pairs], axis=0)
    reg_r, reg_s = c3.find_cups(real_m.astype(np.float64)), c3.find_cups(sim_m.astype(np.float64))

    geo = json.loads((OUT / "scene_geometry.json").read_text())["silhouettes"]
    sil_s = rep_sim["silhouette"]
    keys_sil = ("top_row", "bottom_row", "centre_col", "height_px", "width_top10_px", "width_bottom10_px")
    cup_offsets = {cup: {k: round(geo[cup]["measured"][k] - sil_s[cup][k], 2) for k in keys_sil}
                   for cup in ("cold", "warm", "hot")}

    def shifted(img, dy, dx):
        return ndimage.shift(img.astype(np.float64), (dy, dx), order=1, mode="nearest")

    def best_shift(a, sim, win_sl, rng=4):
        best = (-1.0, (0, 0))
        for dy in range(-rng, rng + 1):
            for dx in range(-rng, rng + 1):
                v = ssim(a[win_sl], norm(shifted(sim, dy, dx), win0)[win_sl])
                best = max(best, (v, (dy, dx)))
        return best[1]

    _i_m, _j_m, real_mid, sim_mid = pairs[len(pairs) // 2]
    a_mid = norm(real_mid, win0)
    shift = [float(v) for v in best_shift(a_mid, sim_mid, CROP)]  # (rows, cols) applied to sim
    local = {}
    for cup in ("cold", "warm", "hot"):
        m = geo[cup]["measured"]
        wsl = (slice(int(m["top_row"]) - 30, int(m["bottom_row"]) + 70),
               slice(int(m["centre_col"]) - 75, int(m["centre_col"]) + 75))
        b0 = norm(sim_mid, win0)
        sh = best_shift(a_mid, sim_mid, wsl)
        b1 = norm(shifted(sim_mid, *sh), win0)
        local[cup] = {"window_rows_cols": [[wsl[0].start, wsl[0].stop], [wsl[1].start, wsl[1].stop]],
                      "best_shift_rows_cols_px": list(sh),
                      "ssim_as_rendered": round(ssim(a_mid[wsl], b0[wsl]), 4), "ssim_registered": round(ssim(a_mid[wsl], b1[wsl]), 4),
                      "lpips_as_rendered": round(lp(a_mid[wsl], b0[wsl]), 4), "lpips_registered": round(lp(a_mid[wsl], b1[wsl]), 4)}

    # -- metrics per pair and window ------------------------------------------------------------------------
    res = {}
    for wname, win in WINDOWS_C.items():
        rows, rows_reg = [], []
        for i, j, real, sim in pairs:
            a = norm(real, win)
            rows.append(metrics(a, norm(sim, win)))
            rows_reg.append(metrics(a, norm(shifted(sim, *shift), win)))
        agg = lambda rs: {k: [round(float(np.mean([r[k] for r in rs])), 4), round(float(np.min([r[k] for r in rs])), 4),  # noqa: E731
                              round(float(np.max([r[k] for r in rs])), 4)] for k in rs[0]}
        res[wname] = {"as_rendered_mean_min_max": agg(rows), "registered_mean_min_max": agg(rows_reg),
                      "per_pair": [{"real_frame": p[0], "render": p[1], **{k: round(v, 4) for k, v in r.items()}}
                                   for p, r in zip(pairs, rows)]}

    # reference levels (primary window)
    refs = {"real_vs_next_real_frame": [], "real_vs_itself_shifted_1px": [], "real_vs_itself_shifted_2px": [],
            "real_vs_itself_shifted_4px": []}
    for i, _j, real, _sim in pairs:
        a = norm(real, win0)
        i2 = min(i + 1, len(real_all) - 1) if i + 1 < len(real_all) else i - 1
        refs["real_vs_next_real_frame"].append(metrics(a, norm(np.asarray(real_all[i2]), win0)))
        for s in (1, 2, 4):
            refs[f"real_vs_itself_shifted_{s}px"].append(metrics(a, norm(shifted(real, s / np.sqrt(2), s / np.sqrt(2)), win0)))
    refs = {k: {m: round(float(np.mean([r[m] for r in v])), 4) for m in v[0]} for k, v in refs.items()}

    # imposed-shift grid on the middle pair (primary window)
    i, j, real, sim = pairs[len(pairs) // 2]
    a = norm(real, win0)
    grid = {"shifts_px": SHIFTS.tolist(), "ssim_crop": [], "lpips_crop": []}
    for dy in SHIFTS:
        rs, rl = [], []
        for dx in SHIFTS:
            b = norm(shifted(sim, dy, dx), win0)
            rs.append(round(ssim(a[CROP], b[CROP]), 4))
            rl.append(round(lp(a[CROP], b[CROP]), 4))
        grid["ssim_crop"].append(rs)
        grid["lpips_crop"].append(rl)
    gs, gl = np.array(grid["ssim_crop"]), np.array(grid["lpips_crop"])
    k0 = len(SHIFTS) // 2
    for n in (1, 2):  # +-2 px and +-4 px, averaged over the four directions
        ring = [gs[k0 + n, k0], gs[k0 - n, k0], gs[k0, k0 + n], gs[k0, k0 - n]]
        ringl = [gl[k0 + n, k0], gl[k0 - n, k0], gl[k0, k0 + n], gl[k0, k0 - n]]
        grid[f"ssim_change_at_{2 * n}px"] = round(float(np.mean(ring) - gs[k0, k0]), 4)
        grid[f"lpips_change_at_{2 * n}px"] = round(float(np.mean(ringl) - gl[k0, k0]), 4)

    # -- rank correlation ----------------------------------------------------------------------------------
    def levels(img, reg):
        out = {}
        for cup in ("cold", "warm", "hot"):
            out[cup] = float(np.median(c3.roi(img, reg[cup]["water"]))) - 273.15
        out["board"] = float(np.median(img[100:180, 150:420])) - 273.15
        return out

    rank = []
    for i, j, real, sim in pairs:
        lr, ls = levels(real, reg_r), levels(sim, reg_s)
        keys = list(lr)
        rank.append({"real_c": {k: round(lr[k], 2) for k in keys}, "sim_c": {k: round(ls[k], 2) for k in keys},
                     "spearman": round(float(spearmanr([lr[k] for k in keys], [ls[k] for k in keys])[0]), 3),
                     "kendall": round(float(kendalltau([lr[k] for k in keys], [ls[k] for k in keys])[0]), 3)})

    rep = {"pairs": [(p[0], p[1]) for p in pairs], "windows_c": WINDOWS_C, "crop_rows_cols": [[160, 380], [30, 570]],
           "alignment": {"global_shift_rows_cols_px": shift, "silhouette_real_minus_sim_px": cup_offsets,
                         "local_per_cup_middle_pair": local},
           "metrics": res, "reference_levels_primary_window": refs, "shift_grid_middle_pair": grid,
           "rank_correlation": {"per_pair": rank, "spearman_all": sorted({r["spearman"] for r in rank}),
                                "order_real": sorted(rank[0]["real_c"], key=rank[0]["real_c"].get),
                                "order_sim": sorted(rank[0]["sim_c"], key=rank[0]["sim_c"].get)},
           "versions": {"lpips": lpips.__version__ if hasattr(lpips, "__version__") else "0.1.4",
                        "torch": torch.__version__, "skimage": __import__("skimage").__version__}}
    (OUT / "task3_similarity.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("alignment", "reference_levels_primary_window")}, indent=1))
    for w in res:
        print(w, json.dumps(res[w]["as_rendered_mean_min_max"]), json.dumps(res[w]["registered_mean_min_max"]))
    print(json.dumps({k: grid[k] for k in grid if "change" in k}), json.dumps(rep["rank_correlation"]["per_pair"][0]))
    plot(pairs, shift, grid, rep)


def plot(pairs, shift, grid, rep) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage

    win = WINDOWS_C["primary_5_70C"]
    i, j, real, sim = pairs[len(pairs) // 2]
    a, b = norm(real, win), norm(sim, win)
    br = norm(ndimage.shift(sim.astype(np.float64), shift, order=1, mode="nearest"), win)
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.9))
    axs[0].imshow((a - b)[CROP], cmap="RdBu_r", vmin=-0.15, vmax=0.15)
    axs[0].set_title(f"real − sim, as rendered (window {win[0]:g}–{win[1]:g} °C)", fontsize=9)
    axs[1].imshow((a - br)[CROP], cmap="RdBu_r", vmin=-0.15, vmax=0.15)
    axs[1].set_title(f"real − sim, sim shifted by ({shift[0]:+.0f}, {shift[1]:+.0f}) px (row, col)", fontsize=9)
    s = np.array(grid["shifts_px"])
    for ax, key, lab in ((axs[2], "ssim_crop", "SSIM (crop)"), (axs[3], "lpips_crop", "LPIPS (crop)")):
        im = ax.imshow(np.array(grid[key]), extent=(s[0] - 1, s[-1] + 1, s[-1] + 1, s[0] - 1), cmap="viridis")
        ax.set_xlabel("extra column shift [px]")
        ax.set_ylabel("extra row shift [px]")
        ax.set_title(f"{lab} vs shift of the simulated frame", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axs[:2]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "task3_similarity.png", dpi=110)


if __name__ == "__main__":
    main()
