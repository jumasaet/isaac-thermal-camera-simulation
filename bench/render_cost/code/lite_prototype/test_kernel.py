"""Tests del kernel en CPU (wp device "cpu"): no necesitan Isaac ni GPU.
Planck en banda aquí es un sustituto numérico 8–14 µm; en el código real
la LUT T->L y ε(θ) salen del repo irsim."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import warp as wp
from kernel import IrsimLite, Tables, build_inverse_lut, build_id_map, sky_cosine_weighted, PoseDelay

wp.init()
DEV = "cpu"
C1, C2 = 1.191042e8, 1.4387752e4      # W·µm⁴/m²/sr, µm·K

def band_L(T_C, lo=8.0, hi=14.0, n=600):
    lam = np.linspace(lo, hi, n)
    T = np.asarray(T_C, float)[..., None] + 273.15
    return np.trapezoid(C1 / lam**5 / np.expm1(C2 / (lam * T)), lam, axis=-1)

T_GRID = np.linspace(-80.0, 400.0, 4801)
L_GRID = band_L(T_GRID)
INV = build_inverse_lut(T_GRID, L_GRID)
SKY_T = np.linspace(5.0, -30.0, 91)           # horizonte templado, cénit frío
SKY = band_L(SKY_T)
T_GROUND = 20.0
# Cámara mirando a -X mundo, horizontal: columnas = ejes de cámara en mundo.
R_WC = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], np.float32)
H, W = 48, 64

def make(eps_by_mat, spec_by_mat, T_obj, mat_of_obj):
    n_ang = 91
    mat_eps = np.array([np.full(n_ang, e) for e in eps_by_mat], np.float32)
    t = Tables(obj_LB=band_L(np.array(T_obj)).astype(np.float32),
               obj_mat=np.array(mat_of_obj, np.int32), mat_eps=mat_eps,
               mat_spec=np.array(spec_by_mat, np.float32), sky=SKY.astype(np.float32),
               L_ground=float(band_L(T_GROUND)), inv_T=INV[0], L0=INV[1], inv_dL=INV[2])
    return IrsimLite(t, W, H, focal_mm=18.0, h_aperture_mm=20.955, device=DEV)

def frame(normal, sid_lower=1):
    seg = np.zeros((H, W), np.uint32); seg[H // 2:, :] = sid_lower    # arriba fondo, abajo objeto
    nrm = np.zeros((H, W, 4), np.float32); nrm[..., :3] = normal
    return wp.array(seg, dtype=wp.uint32, device=DEV), wp.array(nrm, dtype=wp.float32, device=DEV)

def T_obj_pixels(sim, normal):
    sim.set_id_map(np.array([-1, 0], np.int32))
    seg, nrm = frame(normal)
    return sim.run(seg, nrm, R_WC).numpy()

def test_inverse_lut_error():
    T_back = np.interp(L_GRID, np.linspace(INV[1], INV[1] + (len(INV[0]) - 1) / INV[2], len(INV[0])), INV[0])
    assert np.abs(T_back - T_GRID).max() < 0.05

def test_blackbody_is_exact():
    T = T_obj_pixels(make([1.0], [0.0], [100.0], [0]), [1, 0, 0])
    assert np.abs(T[H // 2:] - 100.0).max() < 0.05

def test_hot_aluminum_looks_cold():
    T = T_obj_pixels(make([0.09], [1.0], [100.0], [0]), [1, 0, 0])
    print("Al 100 °C, ε=0.09, especular, pared vertical:", T[H // 2:].mean())
    assert T[H // 2:].mean() < 50.0

def test_concrete_close_to_true():
    T = T_obj_pixels(make([0.92], [0.0], [100.0], [0]), [1, 0, 0])
    print("Concreto 100 °C, ε=0.92:", T[H // 2:].mean())
    assert 90.0 < T[H // 2:].mean() < 100.0

def test_background_is_sky_and_ground():
    T = T_obj_pixels(make([1.0], [0.0], [100.0], [0]), [1, 0, 0])
    assert T[0].mean() < T[H // 2 - 1].mean() < 5.5      # más frío hacia arriba

def test_backface_normal_is_flipped():
    a = T_obj_pixels(make([0.09], [0.0], [100.0], [0]), [1, 0, 0])
    b = T_obj_pixels(make([0.09], [0.0], [100.0], [0]), [-1, 0, 0])
    assert np.allclose(a, b, atol=1e-3)

def test_id_map_children_inherit():
    m = build_id_map({"0": "BACKGROUND", "3": "/World/Tank/Mesh", "5": "/World/Tank2"},
                     ["/World/Tank", "/World/Tank2"])
    assert list(m) == [-1, -1, -1, 0, -1, 1]

def test_sky_cosine_weight_of_uniform_sky():
    assert abs(sky_cosine_weighted(np.full(91, 7.0)) - 7.0) < 1e-3

def test_pose_delay():
    p = PoseDelay(lag=3)
    out = [p.push(np.eye(3) * k)[0, 0] for k in range(6)]
    assert out == [0, 0, 0, 0, 1, 2]
