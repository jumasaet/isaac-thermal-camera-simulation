"""The reduced ("lite") stage-1 path under test. /tmp only.

    instance id -> material id, instance id -> temperature, background mask, stage-1 kernel

Everything per-pixel is a gather off the instance-id plane; L_env is one scalar. Two forms:
``reduced_numpy`` calls the repo's CPU reference `band_radiance`, so a difference against the full
path is a difference in *inputs* and not in arithmetic; ``LiteGpu`` is the same thing on the
device, feeding the repo's own `_band_radiance_kernel`.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from irsim.pipeline.environment import environment_radiance, ground_temperature_k
from irsim.pipeline.radiance import band_radiance
from irsim_isaac.pipeline import warp_stages
from irsim_isaac.pipeline.material_ids import BACKGROUND_INSTANCE_ID, labels_to_paths


def instance_tables(labels, camera, t_bg_k: float, n_min: int = 1):
    """(temperature[id] float32, material[id] int32). Row 0 is the background."""
    paths = labels_to_paths(labels)
    by_path = {r.path: int(r.material_id) for r in camera.resolutions}
    facet = camera.bridge.facet_table(labels, strict=False)
    n = max(int(facet.size), (max(paths) if paths else 0) + 1, int(n_min))
    t_tab = np.zeros(n, dtype=np.float32)
    t_tab[: facet.size] = facet
    m_tab = np.zeros(n, dtype=np.int32)
    for ident, path in paths.items():
        if ident != BACKGROUND_INSTANCE_ID:
            m_tab[ident] = by_path.get(path, 0)
    t_tab[BACKGROUND_INSTANCE_ID] = np.float32(t_bg_k)
    return t_tab, m_tab


def scalar_l_env(camera, t_s: float) -> np.float32:
    """L_env evaluated once, at an arbitrary V_s -- the hypothesis is that V_s does not matter."""
    cfg = camera.config
    return np.float32(environment_radiance(cfg.sky, cfg.lut, t_s, np.float64(0.5), cfg.quantity))


def enclosure_temperature_k(camera, t_s: float) -> float:
    return float(ground_temperature_k(camera.config.sky, t_s))


def reduced_numpy(ids, labels, camera, t_s: float, t_bg_k: float | None = None):
    """The reduced path through the CPU reference kernel. Returns (radiance, T, mat, sky)."""
    cfg = camera.config
    ids = np.asarray(ids)
    t_bg = enclosure_temperature_k(camera, t_s) if t_bg_k is None else t_bg_k
    t_tab, m_tab = instance_tables(labels, camera, t_bg, int(ids.max()) + 1)
    sky = ids == BACKGROUND_INSTANCE_ID
    temp = t_tab[ids]
    mat = m_tab[ids]
    # float64 0-d, as `Illumination.total_incident()` hands the plane to the kernel in float64
    l_env = np.float64(scalar_l_env(camera, t_s))
    out = band_radiance(temp, mat, cfg.materials, cfg.lut, cfg.quantity, sky_mask=sky, l_env=l_env)
    return out, temp, mat, sky


@wp.kernel
def _gather_kernel(
    ids: wp.array2d(dtype=wp.uint32),
    t_tab: wp.array(dtype=wp.float32),
    m_tab: wp.array(dtype=wp.int32),
    n: wp.int32,
    temperature_k: wp.array2d(dtype=wp.float32),
    material_id: wp.array2d(dtype=wp.int32),
    sky_mask: wp.array2d(dtype=wp.uint8),
):
    i, j = wp.tid()
    k = wp.min(wp.int32(ids[i, j]), n - 1)
    temperature_k[i, j] = t_tab[k]
    material_id[i, j] = m_tab[k]
    if k == 0:
        sky_mask[i, j] = wp.uint8(1)
    else:
        sky_mask[i, j] = wp.uint8(0)


class LiteGpu:
    """Device-resident reduced path. Tables re-upload only when idToLabels changes."""

    def __init__(self, camera, shape, device: str = "cuda:0") -> None:
        self.camera = camera
        self.device = device
        cfg = camera.config
        self.tables = warp_stages.device_tables(cfg.lut, cfg.materials, cfg.quantity, device)
        h, w = shape
        self.temp = wp.zeros((h, w), dtype=wp.float32, device=device)
        self.mat = wp.zeros((h, w), dtype=wp.int32, device=device)
        self.sky = wp.zeros((h, w), dtype=wp.uint8, device=device)
        self.env = wp.zeros((h, w), dtype=wp.float32, device=device)
        self.out = wp.zeros((h, w), dtype=wp.float32, device=device)
        self._labels = None
        self._t_s = None
        self.t_tab = self.m_tab = None
        self.n = 0
        self.uploads = 0

    def refresh(self, labels, t_s: float) -> None:
        """Per-frame host work: nothing unless the label table or the clock moved."""
        if labels == self._labels and t_s == self._t_s:
            return
        t_bg = enclosure_temperature_k(self.camera, t_s)
        t_tab, m_tab = instance_tables(labels, self.camera, t_bg)
        self.t_tab = wp.array(t_tab, dtype=wp.float32, device=self.device)
        self.m_tab = wp.array(m_tab, dtype=wp.int32, device=self.device)
        self.n = int(t_tab.size)
        self.env.fill_(float(scalar_l_env(self.camera, t_s)))
        self._labels, self._t_s = dict(labels), t_s
        self.uploads += 1

    def run(self, ids_dev):
        """ids_dev: wp.array2d(uint32) on the device (the annotator's own buffer, wrapped)."""
        wp.launch(_gather_kernel, dim=ids_dev.shape,
                  inputs=[ids_dev, self.t_tab, self.m_tab, np.int32(self.n)],
                  outputs=[self.temp, self.mat, self.sky], device=self.device)
        warp_stages.launch_band_radiance(self.temp, self.mat, self.sky, self.env, self.tables,
                                         self.out)
        return self.out


def wrap_ids(h):
    """The annotator's device buffer as a uint32 (H, W) Warp array, no copy."""
    if h.dtype is wp.uint32 and h.ndim == 2:
        return h
    return wp.array(ptr=h.ptr, dtype=wp.uint32, shape=tuple(h.shape[:2]), device=h.device)


def bits(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
