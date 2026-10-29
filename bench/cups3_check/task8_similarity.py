#!/usr/bin/env python
"""Task 8 (similarity): SSIM and LPIPS of the simulated frames with and without the table reflection.

    PYTHONPATH=~/.cache/irsim_pylibs ~/miniconda3/envs/vision/bin/python bench/cups3_check/task8_similarity.py \
        sim refl/step0 refl/raycast

Each argument is a directory under out/ holding apparent_t_ref_<j>.npy (board-referenced, as task2_compare
writes them). The metrics are task 3's, imported from task3_similarity.py where they are module-level and
restated where they are local to its main(): the same pairs (out/sim/report.json), the same windows, the same
crop, SSIM (Gaussian sigma 1.5) and LPIPS (AlexNet v0.1), and the same per-cup window around each cup and its
reflection (rows top - 30 .. bottom + 70, cols centre +- 75, measured silhouettes), on the middle pair, AS
RENDERED (no registration). Adds the section "similarity" to out/task8_reflection.json.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import task3_similarity as t3

OUT = t3.OUT


def main(dirs: list[str]) -> None:
    import lpips
    import torch
    from skimage.metrics import structural_similarity

    net = lpips.LPIPS(net="alex", version="0.1", verbose=False).eval()

    def ssim(a, b):
        return float(
            structural_similarity(a, b, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        )

    def lp(a, b):
        ta = torch.from_numpy(np.repeat((a * 2 - 1)[None, None], 3, 1)).float()
        tb = torch.from_numpy(np.repeat((b * 2 - 1)[None, None], 3, 1)).float()
        with torch.no_grad():
            return float(net(ta, tb).item())

    rep_sim = json.loads((OUT / "sim" / "report.json").read_text())
    real_all = np.load(OUT / "real_apparent_t_frames.npy", mmap_mode="r")
    geo = json.loads((OUT / "scene_geometry.json").read_text())["silhouettes"]
    renders = rep_sim["renders"]
    mid = len(renders) // 2
    res = {}
    for d in dirs:
        r = {}
        for wname, win in t3.WINDOWS_C.items():
            vals = {"ssim_full": [], "lpips_full": [], "ssim_crop": [], "lpips_crop": []}
            local = {}
            for k, rr in enumerate(renders):
                a = t3.norm(np.asarray(real_all[rr["real_frame_index"]], np.float32), win)
                sim = np.load(OUT / d / f"apparent_t_ref_{rr['render']}.npy")
                assert sim.dtype == np.float32
                b = t3.norm(sim, win)
                vals["ssim_full"].append(ssim(a, b))
                vals["lpips_full"].append(lp(a, b))
                vals["ssim_crop"].append(ssim(a[t3.CROP], b[t3.CROP]))
                vals["lpips_crop"].append(lp(a[t3.CROP], b[t3.CROP]))
                if k == mid:
                    for cup in ("cold", "warm", "hot"):
                        m = geo[cup]["measured"]
                        wsl = (
                            slice(int(m["top_row"]) - 30, int(m["bottom_row"]) + 70),
                            slice(int(m["centre_col"]) - 75, int(m["centre_col"]) + 75),
                        )
                        local[cup] = {"ssim": round(ssim(a[wsl], b[wsl]), 4), "lpips": round(lp(a[wsl], b[wsl]), 4)}
            r[wname] = {
                "mean_8_pairs": {k: round(float(np.mean(v)), 4) for k, v in vals.items()},
                "local_middle_pair_as_rendered": local,
            }
        res[d] = r
        print(d, json.dumps(r), flush=True)
    path = OUT / "task8_reflection.json"
    rep = json.loads(path.read_text()) if path.exists() else {}
    rep.setdefault("similarity", {}).update(res)
    path.write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
