"""
irsim_lite/kernel.py — núcleo GPU del "irsim ligero".

Un kernel Warp fusionado, un hilo por píxel:
    ids + normales + rayo de cámara -> θ -> ε(θ)
    L   = ε·L_B(T_obj) + (1-ε)·L_env
    L_env = s·L_spec(r) + (1-s)·[V_s·L_sky_dif + (1-V_s)·L_ground]
    T_ap = Planck⁻¹(L)        (inversa en banda, LUT uniforme en L)

Todo lo que depende del repo irsim (Planck en banda, ε(θ) por material, cielo) se
precalcula en host como tablas y entra como arrays: el kernel solo interpola.
Módulo puro: NumPy + Warp, sin pxr/omni/carb.

Convenciones
- Mundo: +Z arriba (stage de Isaac). Normales del AOV en mundo (VERIFICAR, ver notas).
- Cámara USD: mira a -Z local, +Y arriba. Filas de imagen crecen hacia abajo.
- R_wc: rotación cámara->mundo, convención columna (v_w = R_wc @ v_c).
- Radiancias en banda, en las unidades de la LUT del repo (da igual cuáles, si son
  consistentes entre L_B, cielo y suelo).
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np
import warp as wp

RAD2DEG = wp.constant(57.29577951308232)


# --------------------------------------------------------------------------------------
# Funciones de dispositivo
# --------------------------------------------------------------------------------------
@wp.func
def lerp_1d(tab: wp.array(dtype=wp.float32), f_in: float) -> float:
    """Interpolación lineal con índice fraccional f_in (saturado a los extremos)."""
    n = tab.shape[0]
    f = wp.clamp(f_in, 0.0, float(n - 1))
    i = wp.min(int(f), n - 2)
    t = f - float(i)
    return tab[i] * (1.0 - t) + tab[i + 1] * t


@wp.func
def eps_at(tab: wp.array2d(dtype=wp.float32), m: int, theta_deg: float) -> float:
    """ε del material m al ángulo θ (grados, 0 = normal). Tabla uniforme 0..90°."""
    n = tab.shape[1]
    f = wp.clamp(theta_deg / 90.0 * float(n - 1), 0.0, float(n - 1))
    i = wp.min(int(f), n - 2)
    t = f - float(i)
    return tab[m, i] * (1.0 - t) + tab[m, i + 1] * t


@wp.func
def env_dir(d: wp.vec3, sky: wp.array(dtype=wp.float32), L_ground: float) -> float:
    """Radiancia del entorno en la dirección d (mundo): cielo por elevación o suelo."""
    if d[2] <= 0.0:
        return L_ground
    e = wp.asin(wp.min(d[2], 1.0)) * RAD2DEG
    return lerp_1d(sky, e / 90.0 * float(sky.shape[0] - 1))


@wp.kernel
def irsim_lite_kernel(
    seg: wp.array2d(dtype=wp.uint32),          # (H, W) instance id
    normals: wp.array3d(dtype=wp.float32),     # (H, W, >=3) normal en mundo
    id_to_obj: wp.array(dtype=wp.int32),       # instance id -> índice de objeto, -1 = fondo
    obj_LB: wp.array(dtype=wp.float32),        # L_B(T_obj) en banda, por objeto
    obj_mat: wp.array(dtype=wp.int32),         # objeto -> índice de material
    mat_eps: wp.array2d(dtype=wp.float32),     # (n_mat, n_ang) ε(θ), θ uniforme 0..90°
    mat_spec: wp.array(dtype=wp.float32),      # fracción especular del reflejo, 0..1
    sky: wp.array(dtype=wp.float32),           # L_sky(elevación), uniforme 0..90°
    L_sky_dif: float,                          # cielo integrado con coseno (superficie horizontal)
    L_ground: float,
    R_wc: wp.mat33,
    fx: float, fy: float, cx: float, cy: float,
    inv_T: wp.array(dtype=wp.float32),         # T (°C) sobre rejilla uniforme de L
    L0: float, inv_dL: float,
    out_T: wp.array2d(dtype=wp.float32),       # T aparente, °C
    out_L: wp.array2d(dtype=wp.float32),       # radiancia en banda (para depurar/comparar)
):
    v, u = wp.tid()

    # Rayo de visión en cámara (USD: -Z adelante, +Y arriba) -> mundo.
    d_c = wp.vec3((float(u) + 0.5 - cx) / fx, -(float(v) + 0.5 - cy) / fy, -1.0)
    d = wp.normalize(R_wc * d_c)

    sid = int(seg[v, u])
    o = int(-1)
    if sid < id_to_obj.shape[0]:
        o = id_to_obj[sid]

    L = float(0.0)
    if o < 0:
        # Fondo: el rayo ve el entorno directamente.
        L = env_dir(d, sky, L_ground)
    else:
        n = wp.normalize(wp.vec3(normals[v, u, 0], normals[v, u, 1], normals[v, u, 2]))
        c = float(-wp.dot(d, n))
        if c < 0.0:            # cara trasera / normal invertida: orientar hacia la cámara
            n = -n
            c = -c
        theta = wp.acos(wp.min(c, 1.0)) * RAD2DEG

        m = obj_mat[o]
        eps = eps_at(mat_eps, m, theta)
        s = mat_spec[m]

        # Reflejo difuso: factor de vista de cielo de un plano inclinado.
        Vs = 0.5 * (1.0 + n[2])
        L_dif = Vs * L_sky_dif + (1.0 - Vs) * L_ground
        # Reflejo especular: dirección espejo (solo cielo/suelo, sin interreflexiones).
        r = d - 2.0 * wp.dot(d, n) * n
        L_spec = env_dir(r, sky, L_ground)

        L_env = s * L_spec + (1.0 - s) * L_dif
        L = eps * obj_LB[o] + (1.0 - eps) * L_env

    out_L[v, u] = L
    out_T[v, u] = lerp_1d(inv_T, (L - L0) * inv_dL)


# --------------------------------------------------------------------------------------
# Host: construcción de tablas (puro NumPy)
# --------------------------------------------------------------------------------------
def build_inverse_lut(T_C: np.ndarray, L: np.ndarray, n: int = 8192):
    """De la LUT directa del repo (T -> L, monótona) a una inversa uniforme en L.
    Devuelve (tabla T °C, L0, 1/dL)."""
    T_C = np.asarray(T_C, np.float64)
    L = np.asarray(L, np.float64)
    if not np.all(np.diff(L) > 0):
        raise ValueError("La LUT T->L debe ser estrictamente creciente")
    Lg = np.linspace(L[0], L[-1], n)
    Tg = np.interp(Lg, L, T_C)
    return Tg.astype(np.float32), float(Lg[0]), float(1.0 / (Lg[1] - Lg[0]))


def sky_cosine_weighted(sky_by_elev: np.ndarray) -> float:
    """Radiancia de cielo equivalente para una superficie horizontal lambertiana:
    (1/π)∫ L cosθz dΩ = 2∫₀^{π/2} L(e) sin e cos e de."""
    e = np.radians(np.linspace(0.0, 90.0, len(sky_by_elev)))
    return float(np.trapezoid(2.0 * sky_by_elev * np.sin(e) * np.cos(e), e))


def intrinsics(width: int, height: int, focal_mm: float, h_aperture_mm: float):
    """Pinhole de Isaac: píxel cuadrado, apertura vertical ajustada al aspecto."""
    fx = width * focal_mm / h_aperture_mm
    return fx, fx, width / 2.0, height / 2.0


def build_id_map(id_to_labels: dict, obj_paths: list[str]) -> np.ndarray:
    """idToLabels del anotador (str(id) -> prim path) -> array id -> índice de objeto.
    Un prim hijo (/World/Tank/Mesh) hereda el objeto del padre (/World/Tank)."""
    ids = [int(k) for k in id_to_labels] or [0]
    table = np.full(max(ids) + 1, -1, np.int32)
    for k, path in id_to_labels.items():
        path = path if isinstance(path, str) else str(path)
        for j, root in enumerate(obj_paths):
            if path == root or path.startswith(root + "/"):
                table[int(k)] = j
                break
    return table


class PoseDelay:
    """Las AOVs llegan `lag` app.update() después: el frame que se procesa ahora
    corresponde a la pose de cámara de hace `lag` updates."""

    def __init__(self, lag: int = 3):
        self.buf = collections.deque(maxlen=lag + 1)

    def push(self, R_wc: np.ndarray) -> np.ndarray:
        self.buf.append(np.asarray(R_wc, np.float32))
        return self.buf[0]


# --------------------------------------------------------------------------------------
# Host: wrapper
# --------------------------------------------------------------------------------------
@dataclass
class Tables:
    obj_LB: np.ndarray        # (n_obj,)
    obj_mat: np.ndarray       # (n_obj,) int32
    mat_eps: np.ndarray       # (n_mat, n_ang)
    mat_spec: np.ndarray      # (n_mat,)
    sky: np.ndarray           # (n_elev,)
    L_ground: float
    inv_T: np.ndarray
    L0: float
    inv_dL: float


class IrsimLite:
    def __init__(self, tables: Tables, width: int, height: int,
                 focal_mm: float, h_aperture_mm: float, device: str = "cuda:0"):
        self.dev = device
        self.W, self.H = width, height
        self.fx, self.fy, self.cx, self.cy = intrinsics(width, height, focal_mm, h_aperture_mm)
        f32 = wp.float32
        t = tables
        self.obj_LB = wp.array(t.obj_LB.astype(np.float32), dtype=f32, device=device)
        self.obj_mat = wp.array(t.obj_mat.astype(np.int32), dtype=wp.int32, device=device)
        self.mat_eps = wp.array(t.mat_eps.astype(np.float32), dtype=f32, device=device)
        self.mat_spec = wp.array(t.mat_spec.astype(np.float32), dtype=f32, device=device)
        self.sky = wp.array(t.sky.astype(np.float32), dtype=f32, device=device)
        self.L_sky_dif = sky_cosine_weighted(t.sky)
        self.L_ground = float(t.L_ground)
        self.inv_T = wp.array(t.inv_T, dtype=f32, device=device)
        self.L0, self.inv_dL = t.L0, t.inv_dL
        self.id_to_obj = wp.full(1, -1, dtype=wp.int32, device=device)
        self.out_T = wp.zeros((height, width), dtype=f32, device=device)
        self.out_L = wp.zeros((height, width), dtype=f32, device=device)

    def set_id_map(self, table: np.ndarray):
        """Llamar cuando cambie idToLabels (no cada frame)."""
        self.id_to_obj = wp.array(table, dtype=wp.int32, device=self.dev)

    def run(self, seg: wp.array, normals: wp.array, R_wc: np.ndarray) -> wp.array:
        wp.launch(
            irsim_lite_kernel, dim=(self.H, self.W), device=self.dev,
            inputs=[seg, normals, self.id_to_obj, self.obj_LB, self.obj_mat,
                    self.mat_eps, self.mat_spec, self.sky, self.L_sky_dif, self.L_ground,
                    wp.mat33(*np.asarray(R_wc, np.float32).ravel()),
                    self.fx, self.fy, self.cx, self.cy,
                    self.inv_T, self.L0, self.inv_dL, self.out_T, self.out_L],
        )
        return self.out_T
