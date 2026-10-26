# camera_preview: el simulador con la Cellplus CTC-007

Fecha: 2026-10-23. Nada en `configs/` ni en `src/`. El YAML `bench/camera_check/cellplus_ctc007_lwir.yaml`
se usa **sin cambios**. Solo se generó su LUT de 8–14 µm en `lut/`, con
`scripts/generate_luts.py --configs bench/camera_check/cellplus_ctc007_lwir.yaml --out bench/camera_preview/lut`.

**Camino de render** (`common.py`):
- Kit renderiza solo el G-buffer (ids, distancia y RGB), en modo síncrono, y se comprueba la pose y el
  número de serie de cada lectura.
- La cámara entera corre en CPU con `irsim.pipeline.frame.run_frame`, con la misma cadena que la corrida
  de referencia (flat field + M9: FPN, píxeles malos, FFC, deriva).
- Supersample 4 (el valor por defecto del esquema): producto de render de 2560×1920 → 640×480.
  Tarda ~2,9 s/frame.

Todo se ejecuta con `/home/jumasaet/isaac_sim/python.sh` y sin las variables de ROS del sistema (ver
`bench/render_cost/code/rebase_2026-10-22/ref/run_one.sh`).

| Tarea | Script | Salida |
|---|---|---|
| 1. Escena y trayectoria de referencia | `render_comparison.py` | `comparison/` |
| 2. Comparación real vs. simulada | `compare_real_sim.py` (+ `noise_metrics.py`) | `compare/sheet.png`, `compare/metrics.json` |
| 3. Plantilla de los dos vasos | `two_cups/render_two_cups.py`, parámetros en `two_cups/params.yaml` | `two_cups/out/` |

**Tarea 1.**
- Escena: `ref_scene.OBJECTS`, los mismos 240 poses que `outputs/comparison_ref/`.
- Ventana fija de temperatura aparente de 15–45 °C, blanco caliente, sin AGC.
- En `comparison/static/` hay 12 frames con la cámara quieta en la pose 47, para el ruido temporal.

**Tarea 2.**
- Real: cuentas almacenadas con la ventana fija del software de captura, 15096–15500.
- Simulada: dn16 con una ventana del mismo ancho (404 cuentas), 2701–3105 (≈ 22,5–31,7 °C con la SITF
  simulada de 44,1 DN/K).
- La ganancia en cuentas/K de la cámara real no está medida. Por eso «misma anchura en cuentas» no
  garantiza la misma anchura en kelvin.
