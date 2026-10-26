# Cellplus CTC-007/008: comprobación de los datos y borrador de la cámara

Fecha: 2026-10-23. Scripts y salidas en este directorio; nada en `configs/` ni en `src/`.
Se ejecutan con `/home/jumasaet/isaac_sim/python.sh bench/camera_check/<script>.py`, en este orden:
`agc_check.py` → `hot_entry_check.py` → `agc_hypotheses.py` → `noise_prelim.py` → `make_examples.py`;
`fov_check.py` y `smoke_frame.py` para el YAML.

## Qué son los .npy

| | lab | cali_board |
|---|---|---|
| frames | 1146 (más 18 perdidos por USB) | 27, de dos sesiones separadas ~55 min |
| dtype, forma | int16, (480, 640) | int16, (480, 640) |
| rango | 14 969 – 16 963 | 14 750 – 16 433 |
| cadencia | 10,0 Hz (dt mediano 0,100 s) | — |

- **No son vídeo de 8 bits.** Son cuentas crudas de 14 bits. `thermal_info.json` lo dice
  (`"16383 - raw sensor counts (higher = hotter)"`, `invert_raw: true`), y los datos lo confirman: las
  columnas pares e impares tienen la misma media (con bytes YUY2 reinterpretados no la tendrían) y los
  píxeles vecinos varían de forma suave.
- **Forma:** 480 filas × 640 columnas, y la escena sale derecha. Las grabaciones están en apaisado,
  640 de ancho. La hoja de datos da la lectura nativa en vertical (480 × 640, FOV H×V 30×40), así que
  el visor o el software de captura la gira. Hay un indicio en los datos: el patrón fijo por filas es
  ~2× el de columnas, lo que encaja con columnas de lectura giradas 90°.
- Núcleo: Obsidian Sensors MirVG, firmware 4.6.6.3, `denoise: 40` activo durante la captura.
- `thermal_color/*.png` (lab) los generó el software de captura a partir del .npy.

## Tarea 1: ¿mapeo fijo o AGC?

**Conclusión: en los .npy el mapeo es fijo, sin AGC.** El offset sí varía con el tiempo (FFC y deriva).
Lo que no se puede saber con estos datos es si el vídeo de 8 bits de la propia cámara lleva AGC,
porque no se grabó.

Pruebas, todas en cuentas almacenadas:

1. **No hay estiramiento.** El rango por frame va de 149 a 1824 cuentas (mediana 576, el 3,5 % de los
   14 bits) y sigue a la escena. La mediana por frame se queda en 15 114 – 15 283. Un AGC
   llevaría todos los frames al mismo rango.
2. **Los tubos fluorescentes que entran o salen no mueven el fondo.** En 36 pares consecutivos
   (0,1 s) con ≥ 1000 px cruzando el umbral de caliente:
   - desplazamiento medido del fondo: |0,47| cuentas de media;
   - lo que habría producido un AGC que mantiene la media: |3,1| cuentas;
   - correlación entre medido y predicho: −0,26.
   En los 5 eventos en que el tubo cambia el rango del frame un 28–41 %, la dispersión del fondo cambia
   un 4–8 %, sin relación con eso (correlación 0,17). Un AGC min–max la habría escalado lo mismo que
   el rango. (`out/agc_hypotheses.json`)
3. **Estabilidad con la escena quieta.** Hay dos tramos quietos (frames 377–389 y 650–663). La ROI
   central de 80×80 cambia 0,4 y 1,5 cuentas entre frames. El ruido temporal por píxel es
   σ = 2,1 cuentas (robusto, tras el denoise). El segundo tramo deriva 16 cuentas en 1,4 s.
4. **El PNG del software de captura también es fijo.** Es una ventana lineal donde la cuenta
   15 096 pasa a 0 y la 15 500 a 255 (0,631 códigos/cuenta). Coincide al 0,1 % en los 1146 frames y deja
   saturar hasta el 8,7 % de los píxeles a 255 y el 15 % a 0, cosa que un AGC no haría.
5. **El nivel no es constante en el tiempo.**
   - Frame 940 → 941: salto de +106 cuentas en todo el frame (p1, p50 y p99 por igual, mismo
     contraste) durante una pausa de 0,70 s sin índices perdidos. Probablemente es un FFC del
     obturador: la cabecera de frame trae la temperatura del obturador.
   - Entre FFC, el nivel deriva hasta ~1 cuenta/frame.
   - Entre las dos sesiones de cali_board hay ~70 cuentas de diferencia.
   - Para leer temperatura hará falta modelar el offset con el tiempo y las temperaturas de la
     cabecera.
6. **No sirve como prueba:** la correlación de 0,84 entre la "pared" y la fracción fría en
   cali_board. Al acercarse el tablero, la moda del histograma pasa a ser el sustrato del tablero, y
   además la distancia va con el tiempo.

### Qué habría que grabar para cerrar lo que queda

- **AGC del vídeo de 8 bits:** grabar a la vez el flujo UVC de 8 bits y el crudo, con la cámara en
  trípode y una escena quieta durante 10 s. Después meter y sacar algo caliente en una esquina (mano,
  taza de agua caliente o soldador) durante 5 s, y repetirlo dos veces. Si el fondo de 8 bits se
  mueve y el crudo no, el AGC está en el vídeo.
- **Cuentas/K:** un cuerpo negro (o una placa de emisividad alta con termopar) a 2–3 temperaturas
  entre 20 y 60 °C, llenando el campo.
- **FFC y deriva:** 15 min en trípode ante una escena quieta, registrando de la cabecera el contador,
  la temperatura de placa y la del obturador, y anotando cuándo suena el obturador.
- **Ruido 3D:** la tapa de la lente o el cuerpo negro durante 30 s o más, con `denoise` a 0.

## Tarea 2: borrador del YAML

`cellplus_ctc007_lwir.yaml`, con la respuesta espectral provisional
`cellplus_ctc007_response_ESTIMATED.csv`.

- El loader del simulador lo acepta.
- **FOV: 38,17° (H) × 29,09° (V)** en 640×480; en la orientación nativa vertical son 29 × 38.
  IFOV 1,08 mrad, DFOV 46,8°.
- `smoke_frame.py` renderiza un frame engine-free completo de 640×480 con él (LUT de 8–14 µm,
  bolómetro, ruido, NUC, ISP). La temperatura aparente de un fondo de 295 K sale 295,00 K.

Rellenado desde la hoja de datos:
- banda de 8 a 14 µm;
- F/1.3 y focal de 22,2 mm;
- 640×480 y pitch de 24 µm;
- 9 Hz;
- NETD de 50 mK;
- `bad_pixel_fraction` 0,005 (operabilidad del 99,5 %).

Rellenado desde `thermal_info.json` o los datos:
- `bit_depth` 14;
- `nuc.mode: shuttered`;
- `ffc_freeze_ms` 700.

### Lo que se queda del Boson (PENDIENTE)

- `transmittance` (0,92).
- `housing_tau_s` (900 s) y `housing_self_heating_k` (4 K). **Sospechoso:** este núcleo consume
  1,2 W, unas 2,4 veces el Boson.
- `fill_factor` (0,90).
- `thermal_time_constant_ms` (8 ms). La "latencia 1 ms" de la hoja es otra cosa.
- `g_th_w_per_k` (2e-8) y `tcr_per_k` (−0,021, VOx). No se sabe el material, y el píxel es de
  24 µm, no de 12 µm.
- Los 7 `ratios_3d`. **No encajan:** se midió tv ≈ 0,25 y th ≈ 0,30, frente a 0,05.
- `fpn_drift_tau_s`, el clúster y los fallos tardíos de píxeles malos.
- `ffc_interval_s` (300 s) y los dos residuales de NUC.
- Distorsión a cero y `vignetting_cos4`.

### Lo del Boson que no tiene sentido aquí, y se cambió

- `aberration_sigma_um` 1,654: salía del 42 % de MTF del datasheet de FLIR. Puesto a 0, solo
  difracción, que da 0,63 en Nyquist y es una cota superior.
- `bit_depth` 16 y `frame_rate_hz` 60.
- Banda 7,5–13,5 µm y `boson_vox.csv`.
- El AGC `information_based` con los ajustes de fábrica de FLIR. Los datos son crudos: hay que comparar
  `dn16`, no `display8`. El esquema no tiene un modo de ventana fija como el del software de captura,
  así que se deja `linear` provisional.

## Lo que falta para replicar la cámara

1. **Unidad y cadencia.** Los datos van a 10,0 Hz y no cuadran ni con 9 Hz (CTC-007) ni con 30 Hz
   (CTC-008).
2. **Cuentas/K y radiometría.**
   - Cuerpo negro a varias temperaturas → `calibration.radiometric.sitf_dn_per_k` y `netd_k`.
   - Significado de `t_list` [0,006, 4, 0,35, 55000, 0]: preguntarlo a Cellplus/Obsidian.
3. **Ruido 3D, mapas de FPN y píxeles malos.** Tapa de la lente o cuerpo negro, con denoise apagado.
   Además, decidir cómo modelar `denoise: 40`, que el simulador no tiene.
4. **Geometría.** Calibrar con las 27 vistas del tablero de círculos (6×4) → `calibration.geometric`.
5. **MTF y enfoque.**
   - Medir un borde inclinado → `aberration_sigma_um`.
   - Averiguar a qué distancia está enfocada la lente. La hiperfocal es 15,8 m, así que enfocada a
     infinito se ve blanda a 1–5 m, como en el lab.
6. **FFC y deriva.** Intervalo, disparadores y salto típico; offset en función de la temperatura del
   FPA, sacado de las temperaturas de la cabecera.
7. **Respuesta espectral real** (8–14 µm es solo el rango nominal) y transmitancia de la lente.
8. **Viñeteo y sombreado.** Con un blanco uniforme, decidir si dejar `cos4` o poner un `vignetting_map`.
9. **Ruta de vídeo de 8 bits** (si se va a usar): grabarla según el protocolo de arriba.
10. **Térmica de la carcasa:** τ y autocalentamiento con 1,2 W.
