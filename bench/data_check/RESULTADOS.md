# Revisión de los datos de `thermal_specs/`: inventario, la inconsistencia 96 / 94 / 93 y qué se puede borrar

Fecha: 2026-10-29. Script: `inventory.py` (solo lee; no mueve ni borra nada). Salida: `out/inventory.json`.

    python3 -I bench/data_check/inventory.py thermal_specs bench/data_check/out/inventory.json     (~5 min, 18 GB)

Qué comprueba en cada archivo:
- `.npy`: que cargue, que sea `int16` de 480 × 640 y que no sea constante;
- `.png`: la firma y el bloque final `IEND` (si falta, el archivo está cortado);
- `.pcd`: que el tamaño sea la cabecera más `POINTS` × 26 bytes;
- `.pgm`: la firma `P5` y 1 228 816 bytes;
- duplicados: archivos idénticos byte a byte (MD5) dentro de una misma carpeta.

La hora de cada archivo es la de su nombre (`<índice>_<hora unix>`). Con ella se calculan la duración y la
frecuencia.

---

## 1. Inventario

Horas en UTC. «Frecuencia» es la de los archivos guardados (n − 1 entre la duración), no la de la cámara.

### `three_cups_experiments/` (27-oct.)

| carpeta | modalidad | archivos | de → a (UTC) | duración | frecuencia | problemas |
|---|---|---|---|---|---|---|
| `sync_cups/lidar` | LiDAR `.pcd` | **96** | 07:46:59 → 07:47:28 | 28,8 s | 3,3 Hz | ninguno (el set 90 trae solo 96 puntos: un barrido partido, archivo válido) |
| `sync_cups/thermal` | térmica `.npy` | **94** | ídem | 28,8 s | 3,2 Hz | faltan los sets 59 y 71; **los sets 80 = 81 y 88 = 89 son el mismo frame** |
| `sync_cups/thermal_color` | vista PNG | 94 | ídem | | | los mismos huecos y duplicados que `thermal` |
| `sync_cups/rgb` | RGB `.png` 1280 × 960 | **93** | ídem | | 3,2 Hz | faltan los sets 59, 71 y 91; **88 = 89** |
| `sync_cups.zip` | | 382 entradas | | | | **idéntico a `sync_cups/`** (CRC y tamaño de las 382 entradas). **Borrado el 29-oct.** |
| `cups/lidar` | LiDAR | 300 | 07:50:28 → 07:50:58 | 29,9 s | 10,0 Hz | ninguno |
| `cups/thermal` | térmica | 7454 | 07:50:53 → 08:00:47 | 594 s | 12,5 Hz | la cámara manda 30 Hz: 7743 frames perdidos por USB según su log, y 2119 recibidos que no se escribieron (huecos en el índice) |
| `cups/thermal_color` | vista PNG | 630 | 07:50:53 → 08:00:03 | | 1,1 Hz | solo 1 de cada 12. **`008641_1793088003.068578481.png` estaba cortado** (56 KiB, sin `IEND`; borrado el 29-oct., quedan 629) |
| `cups/rgb` | RGB `.png` | 633 | 07:50:40 → 07:51:43 | 63 s | 10,0 Hz | — |
| `cups/rgb_raw` | RGB Bayer crudo `.pgm` | 4384 | 07:51:43 → 08:00:47 | 544 s | 8,1 Hz | continúa el índice del RGB (1737 → 17192): a partir de ahí la grabadora guardó el crudo en lugar del PNG |
| `covered_thermal/thermal` | térmica, **cámara tapada** | 10 533 | 08:01:59 → 08:07:52 | 354 s | **29,8 Hz** | ninguno; 72 frames perdidos por USB según su log, ni un hueco en el índice |
| `covered_thermal/thermal_color` | vista PNG | 1126 | 08:01:59 → 08:07:33 | | 3,4 Hz | solo 1 de cada 9 |

### Sesiones anteriores

| carpeta | archivos | de → a (UTC) | frecuencia | problemas |
|---|---|---|---|---|
| `lab/thermal` (+ `thermal_color`, 1146 PNG) | 1146 `.npy` | 21-oct. 10:39:58 → 10:41:58 (120 s) | 9,5 Hz (intervalo mediano 100 ms) | 18 índices sin archivo, exactamente los «18 frame(s) dropped (shifted)» de su log. `denoise: 40` |
| `cali_board/thermal` (+ `thermal_selec`, 27 PNG) | 27 | 21-oct. 10:15 → 11:12 | elegidos a mano (índices 107…1104) | ninguno |
| `cali_board_new/cold_water_50` | 15 + 15 + 15 + 15 | 23-oct. 22:48:36 (56 s) | | ninguno |
| `cali_board_new/cold_water_95` | 12 × 4 | 23-oct. 22:52:28 (56 s) | | ninguno |
| `cali_board_new/hot_water_50` | 19 × 4 | 23-oct. 22:18:48 (44 s) | | **los frames térmicos 6, 7 y 8 son el mismo** (tres sets en 56 ms con un solo frame de cámara; `thermal_gray` igual). `bench/cups_check/task4_noise.py` ya descartaba además los 14 y 15 por contenido corrupto |
| `cali_board_new/hot_water_95` | 17 × 4 | 23-oct. 22:24:51 (42 s) | | ninguno |

En cada caso de `cali_board_new`, `thermal`, `thermal_gray`, `rgb` y `lidar` tienen el mismo número de archivos.

**Ningún archivo vacío, ni `.npy`, `.pcd` o `.pgm` dañado, en las 18 GB.** El único archivo corrupto es el PNG de
vista cortado de `cups/thermal_color`.

### La cámara va a 30 Hz

- `covered_thermal`: 10 533 frames en 353,65 s, más 72 perdidos según su log, son **29,99 Hz**. El intervalo
  mediano es de 32 ms.
- `cups/thermal`: 9573 frames recibidos (el índice) más 7743 perdidos por USB, más los «dropped (shifted)» de las
  reaperturas, en 594 s: ~29–30 Hz.
- Los 3,3 Hz de `sync_cups` y los ~0,2–0,4 Hz de `cali_board_new` son los sets guardados, no la cámara.
- `lab` (21-oct.) se grabó a 10 Hz: es una configuración anterior de la herramienta de captura.

---

## 2. La inconsistencia: 96 LiDAR, 94 térmicos, 93 RGB

### Lo primero: no la produce la sincronización

La sincronización (`sync_cups/fps_summary.json`) guarda un set por barrido de LiDAR y lo descarta **entero** si una
cámara se desvía más de 33 ms. Lo que descarta no deja rastro en las carpetas: no hay sets a medias por eso.
Las cuentas son estas:
- `sync_index.csv` lista **96 sets**;
- **los 96 tienen hora de RGB y de térmica, con desfases dentro de ±33 ms** (el mayor, 32,97 ms);
- los sets 59, 71 y 91 también: 59 tiene +6,0 / +1,0 ms, 71 tiene +14,4 / −3,0 ms y 91 tiene −9,1 / −3,4 ms
  (RGB / térmica).

La sincronización **aceptó** esos sets y eligió sus imágenes. Lo que falta es el archivo escrito.

### Lo que hay de verdad, set por set

| | sets | LiDAR | térmica | RGB |
|---|---|---|---|---|
| en el índice | 96 | 96 | 96 | 96 |
| archivo que falta | | — | 59, 71 | 59, 71, 91 |
| **archivos** | | **96** | **94** | **93** |
| repetidos (mismo frame de cámara en dos sets) | | — | 81 (= 80), 89 (= 88) | 89 (= 88) |
| **frames distintos** | | 96 nubes | **92** | **92** |
| sets con las tres modalidades | **93** | | | |
| … y distintos en el tiempo | **91** | | | |

**Los archivos que faltan (59 y 71 sin térmica ni RGB; 91 sin RGB)** se perdieron al escribir, no al sincronizar:
- **El zip original tiene los mismos huecos**, así que no es una copia incompleta.
- **El set 59 es el último antes del mayor hueco de la grabación.** Entre los sets 59 y 60 pasan 4,73 s (barridos
  165 a 207 sin set), justo cuando el nodo térmico avisa de que no le llegaron 63 y luego 35 frames en 5 s
  («CPU busy or USB loss», `thermal/thermal_sync_events.csv`). La máquina estaba saturada.
- **Las RGB se escribieron entre 30 y 77 s después de capturarse, y desordenadas.** Según las horas de modificación
  (hora local de la máquina, conservadas en el zip), los `.npy` y los `.pcd` acaban a las 16:47:29, al ritmo de la
  captura. Los PNG de `rgb/` van de 16:47:59 a 16:48:45, a ~2 por segundo, y el 061 se escribió antes que el 056.
  El escritor de RGB llevaba una cola y se retrasó.
- Lo más probable, por tanto: el escritor de cámara descartó o no completó esas imágenes.
- `write_queue_drops: 0` solo cuenta las nubes de LiDAR; **la grabadora no registra las pérdidas de escritura de
  las cámaras**.
- Su código no está en esta máquina: la sesión se grabó en `/home/gitcc/ros2_ws/...`, en otro equipo. **Es una
  inferencia, no está comprobada.**

**Los repetidos** (80 = 81, 88 = 89) sí vienen de la sincronización, pero no por descartar frames sino por usar uno dos
veces:
- los dos LiDAR de cada par llegan con 16 y 14 ms de diferencia (barridos partidos: el 80 tiene 4224 puntos y el 81
  tiene 19 104);
- la sincronización empareja cada uno con la cámara más cercana, y en los dos casos es la misma: el mismo
  `thermal_stamp` en el índice, y en 88/89 también el mismo `rgb_stamp`;
- en `cali_board_new/hot_water_50` pasa lo mismo con tres sets (6, 7 y 8).

**Las propias cuentas de `fps_summary.json` no cierran**:
- de 298 barridos se guardaron 96, así que 202 no se guardaron;
- los motivos suman 136 «esperando a las cámaras» + 83 «térmica > 33 ms» + 116 «RGB > 33 ms» = 335;
- un barrido cuenta, por tanto, en más de un motivo, y sin el código no se puede repartir;
- además, el último set es el barrido 288, así que los barridos 289 a 297 (el último segundo) no tienen set.

**Efecto en lo ya hecho**:
- `bench/cups3_check` tareas 1 a 5 usaron los 94 térmicos, con los dos repetidos incluidos. Eso pesa el doble dos
  instantes de 94: ningún número publicado cambia apreciablemente (es una media por frame);
- la tarea 6 (ruido) los quita, porque un par idéntico daría ruido 0;
- quien use `sync_cups` en adelante debe quitar los repetidos por contenido o por `thermal_stamp`, no por nombre.

---

## 3. Otras cosas que no cuadraban con la guía

- **`cups/` no es «la grabación completa sin sincronizar» de `sync_cups`.** Es otra grabación posterior:
  - empieza 3 min 25 s después de que acabe `sync_cups` (07:50:28 frente a 07:47:28), en otra sesión de la
    grabadora (`2026-10-27_16-50-11` frente a `16-46-25`);
  - sus modalidades apenas se solapan: el LiDAR dura 30 s (07:50:28–58), la térmica empieza a las 07:50:53 y dura
    10 min, y el RGB pasa a crudo `.pgm` al minuto;
  - **las tres juntas solo coinciden 4,4 s**;
  - no sirve para recuperar los frames que faltan en `sync_cups`.
- **`covered_thermal/` es la grabación con la cámara tapada**: 10 533 frames a 30 Hz durante 354 s, con
  `denoise: 80`, 71 s después de acabar `cups/`. Son los datos «con tapa» de la tarea de ruido. No se han usado:
  la tarea 2 de esta sesión pedía solo la grabación sincronizada.
- **Ningún archivo trae temperaturas internas de la cámara válidas.** La cabecera de cada frame lleva la temperatura
  de placa y de obturador, pero la grabadora no la guarda:
  - `thermal_info.json` solo tiene puerto, firmware, `t_list`, `denoise` y formato;
  - `thermal_sync_events.csv` de `sync_cups` solo tiene avisos de pérdida;
  - los valores «board / shutter» que aparecen en `lab`, `cups` y `covered_thermal` (4,3 a 16,9 °C, en una sala a
    23,9 °C) salen de **cabeceras desalineadas**, y son **cuentas de píxel**. Por ejemplo,
    `counter 28180910` = `0x01AE01AE`, que son dos píxeles de 430 pegados, y su «placa 4,31 °C» es otro píxel de
    431 / 100. En `covered_thermal`, la mediana de `16383 − cuentas` es 436.

---

## 4. Qué borraría y qué conservaría (propuesta; no se ha borrado nada)

### Borrar sin perder información — **hecho el 29-oct.**

Los dos archivos de esta tabla se borraron el 29-oct., con la confirmación del usuario.

| qué | tamaño | motivo |
|---|---|---|
| `three_cups_experiments/sync_cups.zip` | 185 MB | copia exacta de `sync_cups/` (CRC de las 382 entradas). Si se quiere guardar el original tal como llegó, mejor moverlo a `~/archivo_irsim/` que tenerlo dos veces aquí |
| `three_cups_experiments/cups/thermal_color/008641_1793088003.068578481.png` | 56 KB | cortado (sin `IEND`), no se abre |

### Borrar solo si hace falta espacio (se pierde algo que hoy no usa nadie)

| qué | tamaño | motivo |
|---|---|---|
| `cups/thermal_color/` y `covered_thermal/thermal_color/` | 146 + 341 MB | vistas de 8 bits de la herramienta de captura, de 1 de cada 12 y 1 de cada 9 frames, con ventana fija. Se rehacen de los `.npy` |
| `cups/rgb_raw/` | 5,1 GB | Bayer crudo de 07:51:43 a 08:00:47, sin térmica sincronizada ni LiDAR en ese tramo. Ningún script lo lee. Es el único RGB de esos 9 minutos |

### Conservar

| qué | motivo |
|---|---|
| `sync_cups/` entero, **incluidos los repetidos** | lo usan `bench/cups3_check` y la tarea 6. Borrar 81 u 89 abriría huecos nuevos en el índice; los scripts los filtran |
| `cali_board_new/` entero, con los `*_matrix.txt` | `bench/cups_check` y las matrices de calibración de los tres vasos |
| `lab/` (térmica, vistas, `normalization.json`, `thermal_info.json`) | `bench/camera_check` (con las vistas se dedujo la ventana fija) |
| `cali_board/` | calibración geométrica pendiente (`distortion` del YAML) |
| `covered_thermal/thermal/` | **la grabación con tapa**: lo que falta para medir el ruido y el patrón fijo sin escena, y las FFC (354 s seguidos a 30 Hz) |
| `cups/thermal/`, `cups/lidar/`, `cups/rgb/` | la única secuencia larga con escena (10 min). Sirve para medir la deriva del patrón fijo (`fpn_drift_tau_s`, que la tarea 6 no puede fijar en 29 s) y el intervalo de FFC (`ffc_interval_s`, PENDIENTE) |
| `thermal_camera_specs.md` | la hoja de datos |

Lo que más ayudaría a la próxima grabación: que la grabadora **guarde la temperatura de placa y de obturador** de
cada cabecera buena (ya las lee para validarla) y **registre las imágenes que no llega a escribir**.
