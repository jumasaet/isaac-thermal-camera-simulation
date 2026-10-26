"""Noise and banding numbers computed the same way on real counts and simulated DN (task 2).

All inputs are 2-D count planes (real: stored counts, higher = hotter; sim: dn16). A static stack
(camera at rest) separates temporal noise from scene: frame DIFFERENCES cancel the scene.

    temporal(stack, roi)  -> sigma_t (robust, per pixel), row and column temporal noise
    spatial(mean, roi)    -> row / column fixed pattern of the stack mean, after removing the slow
                             trend with a 31-sample running median (an upper bound: scene structure
                             at those scales also counts)
    pixel_pattern(img, roi) -> robust sd of (img - median of its 8 neighbours): pixel-scale structure.
                             On a static stack MEAN it is the fixed pattern (temporal noise averaged
                             down by sqrt(n)); on one frame it is fixed pattern + temporal noise
    outliers(img, roi)    -> isolated pixels more than `thr` counts from their 3x3 median (bad / stuck
                             pixels), an ABSOLUTE threshold so both sides are judged alike
    clean(img, thr)       -> the same pixels replaced by that median; temporal() and spatial() run on
                             cleaned frames, or a few railed pixels (0 / 16383) become "row noise"
    edge_width(profile)   -> 10-90 % rise in pixels of a 1-D edge profile (linear interpolation)
"""
from __future__ import annotations

import numpy as np


def _robust_sd(x: np.ndarray) -> float:
    x = np.asarray(x, np.float64).ravel()
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def temporal(stack: np.ndarray, roi: tuple[slice, slice], thr: float = 50.0) -> dict:
    s = np.stack([clean(f, thr) for f in np.asarray(stack, np.float64)])[(slice(None), *roi)]
    d = np.diff(s, axis=0) / np.sqrt(2.0)  # one frame's noise, scene cancelled
    d -= np.median(d, axis=(1, 2), keepdims=True)  # a whole-frame level step (drift, FFC) is not pixel noise
    rows = d.mean(axis=2)  # (n-1, H): row means of each difference
    cols = d.mean(axis=1)
    rows -= rows.mean(axis=1, keepdims=True)  # a whole-frame offset step (drift, FFC) is not row noise
    cols -= cols.mean(axis=1, keepdims=True)
    # sigma_t: robust, but the MAD of integer differences is quantised (MAD = 2 counts -> 2.097);
    # sigma_t_sd: plain sd of the cleaned differences, not quantised
    return {"sigma_t": _robust_sd(d), "sigma_t_sd": float(d.std()), "sigma_row_t": float(rows.std()), "sigma_col_t": float(cols.std()),
            "n_frames": int(s.shape[0])}


def _running_median(x: np.ndarray, n: int = 31) -> np.ndarray:
    p = n // 2
    xp = np.pad(x, p, mode="edge")
    return np.array([np.median(xp[i:i + n]) for i in range(x.size)])


def spatial(mean: np.ndarray, roi: tuple[slice, slice], thr: float = 50.0) -> dict:
    m = clean(mean, thr)[roi]
    r, c = m.mean(axis=1), m.mean(axis=0)
    return {"sigma_row_pattern": float(np.std(r - _running_median(r))),
            "sigma_col_pattern": float(np.std(c - _running_median(c)))}


def _median8(a: np.ndarray) -> np.ndarray:
    pad = np.pad(a, 1, mode="edge")
    nb = np.stack([pad[1 + dy:1 + dy + a.shape[0], 1 + dx:1 + dx + a.shape[1]]
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
    return np.median(nb, axis=0)


def _highpass(img: np.ndarray, roi: tuple[slice, slice]) -> np.ndarray:
    a = np.asarray(img, np.float64)[roi]
    return a - _median8(a)


def clean(img: np.ndarray, thr: float = 50.0) -> np.ndarray:
    a = np.asarray(img, np.float64)
    med = _median8(a)
    return np.where(np.abs(a - med) > thr, med, a)


def pixel_pattern(img: np.ndarray, roi: tuple[slice, slice]) -> float:
    return _robust_sd(_highpass(img, roi))


def outliers(img: np.ndarray, roi: tuple[slice, slice], thr: float = 50.0) -> dict:
    dev = _highpass(img, roi)
    n = int((np.abs(dev) > thr).sum())
    return {"isolated_px": n, "per_million": round(1e6 * n / dev.size, 1), "threshold": thr}


def edge_width(profile: np.ndarray) -> float:
    """10-90 % rise of a monotone-ish edge profile, pixels."""
    p = np.asarray(profile, np.float64)
    if p[0] > p[-1]:
        p = p[::-1]
    lo, hi = np.median(p[:4]), np.median(p[-4:])
    y = (p - lo) / (hi - lo)

    def cross(level: float) -> float:
        i = int(np.argmax(y >= level))
        return i - 1 + (level - y[i - 1]) / (y[i] - y[i - 1]) if i > 0 else 0.0

    return float(cross(0.9) - cross(0.1))
