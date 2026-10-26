# Vasos de agua fría y caliente: calibración, geometría, real frente a simulado y ruido

Fecha: 2026-10-26. Datos: `thermal_specs/cali_board_new/` (4 casos, 63 frames térmicos, LiDAR, RGB).
Todo está en este directorio. Lo único que se escribió fuera es el YAML nuevo
`bench/camera_check/cellplus_ctc007_lwir_cups.yaml`; el original no se tocó, y nada en `configs/` ni en `src/`.

Se ejecuta con `/home/jumasaet/isaac_sim/python.sh`, en este orden:

| Paso | Script | Salida (`out/`) |
|---|---|---|
| 1 | `task1_calibration.py` | `task1_calibration.{json,png}`, `apparent_t_<caso>.npy` |
| 2 | `task2_geometry.py` | `task2_geometry.{json,png}`, `lidar_geometry.json` (pose para el render) |
| 4 | `task4_noise.py` | `task4_noise.{json,png}` y el YAML nuevo |
| 3a | `run_all_renders.sh` (llama a `task3_render_cups.py`) | `sim_<caso>/` (YAML nuevo) y `simorig_<caso>/` (YAML original) |
| 3b | `task3_compare.py` | `task3_comparison.{json,png}` |
| 5 | `task5_figures.py` | `figures/`: una hoja real/simulado por caso y una resumen, en white-hot y jet |
| 6a | `task6_psf.py` | `task6_psf.{json,png}`: el borde real medido (ESF, LSF); con `--sim <tag>`, el mismo en los renders |
| 6b | `task6_focus_fit.py` | `task6_focus_fit.{json,png}`: foco y aberración que reproducen los bordes |
| 6c | `task3_render_cups.py <caso> …ctc006…yaml sim6_<caso>` | `sim6_<caso>/`: los cuatro casos con la cámara nueva |

`cups_common.py` reúne lo compartido: lectura de los datos, regiones del vaso y del fondo, radiancia de
banda y LiDAR. `materials/paper_cup.yaml` es el material de papel, que se añade a la librería solo
durante el render.

---

## Tarea 1: calibración de cuentas a temperatura

### Qué temperatura del agua caliente va con cada caso

Orden temporal de los archivos (hora de Corea, 2026-10-24):

| caso | inicio | fin | frames | pausa desde el anterior |
|---|---|---|---|---|
| hot_water_50 | 07:18:48 | 07:19:32 | 19 | — |
| hot_water_95 | 07:24:51 | 07:25:33 | 17 | 5 min 19 s |
| cold_water_50 | 07:48:36 | 07:49:32 | 15 | 23 min |
| cold_water_95 | 07:52:28 | 07:53:24 | 12 | 2 min 56 s |

**Supuesto:** 66 °C se leyó al empezar hot_water_50 y 58 °C al terminar hot_water_95, es decir, las dos
lecturas acotan la sesión caliente.

**Por qué:** hot_50 es la primera serie caliente y el agua solo se enfría. Además, los propios datos
muestran que se enfría deprisa: dentro de hot_50 el contraste cae de 2782 a 2637 cuentas en 44 s, unos
2,3 K. Para el vaso frío tomo 11 °C al inicio de cold_water_50. cold_95 se grabó 4 minutos después, y
dentro de cada serie fría el agua se calienta, así que es de esperar que cold_95 esté algo por encima de
11 °C.

### Modelo y ajuste (solo 50 cm)

El bolómetro es lineal en la radiancia de banda (docs/physics-model.md §3.2 b, §9.3):
`c = G·L_B + O(t)`. El offset O(t) se mueve:
- saltos de FFC de ~15 cuentas dentro de una serie (cold_50, entre los frames 2 y 3);
- ~50 cuentas entre la sesión caliente y la fría.

Por eso el **fondo es el tercer punto como referencia en cada frame**. Está en una sala cerrada a 22 °C,
así que su radiancia aparente es L_B(22 °C) sea cual sea su emisividad. Restando el fondo en cada frame:

    Δc = c_vaso − c_fondo = G·ε·(L_B(T_s) − L_B(T_aire))

Los datos fijan el producto G·ε, no cada factor:
- La emisividad del papel solo reparte ese producto entre la ganancia de la cámara y la temperatura
  aparente.
- La **temperatura de superficie** de un papel igual a 95 cm, `L_B⁻¹(L_B(22) + Δc/(G·ε))`, no depende de
  ε.

| | valor |
|---|---|
| G·ε (lo que fijan los datos) | **58,7 cuentas por W m⁻² sr⁻¹** |
| G con ε = 0,93 | 63,2 cuentas por W m⁻² sr⁻¹ |
| SITF | 50,6 cuentas/K a 22 °C; **52,9 cuentas/K a 300 K** |
| offset absoluto O | 12 033 / 12 011 cuentas (hot_50 / hot_95) y 11 980 (las dos frías), con G = 63,2 |

**Fórmula:** `T_ap = L_B⁻¹(L_B(T_aire) + (c − c_fondo)/G)`, con L_B integrada sobre la respuesta del YAML
(8–14 µm, forma ESTIMADA).

**El ajuste no cierra, y eso es el resultado principal de la tarea.** Con dos puntos y una incógnita
queda un grado de libertad de comprobación:

| | caliente / frío |
|---|---|
| cociente de contrastes medido | **6,29** |
| predicción lineal en radiancia (66 / 22 / 11 °C) | 5,11 |
| predicción lineal en kelvin | 4,00 |

- La ganancia sacada solo del punto frío (56,9) y solo del caliente (70,1) difieren un **23 %**. Con las
  incertidumbres de lectura supuestas (±0,5 °C), la razón de Birge es 3,1.
- En kelvin difieren un 57 % (40,2 frente a 63,2 cuentas/K). El modelo en radiancia es claramente mejor,
  pero las tres lecturas no son coherentes entre sí: haría falta el vaso frío a **13,2 °C**, o el caliente
  a **74,5 °C**, o el fondo a **20,3 °C**.
- No se puede saber cuál de las tres falla. No sabemos con qué instrumento, en qué punto ni en qué
  momento se midieron.
- Por eso la G nominal es la media geométrica de las dos, y la incertidumbre recorre todo el intervalo
  entre ellas.

### Comprobación independiente a 95 cm

Se aplica la calibración (sin tocarla) al frame al que se refiere cada lectura. Valores nominales; entre
corchetes, el intervalo del 68 % del Monte Carlo.

| caso | T aparente (ε = 1) | T superficie (ε = 0,93) | lectura | diferencia |
|---|---|---|---|---|
| hot_water_95, último frame | **52,1 °C** [49,6; 54,1] | **54,1 °C** [51,5; 56,2] | 58 °C | −3,9 K |
| cold_water_95, primer frame | **13,8 °C** [13,2; 14,7] | **13,2 °C** [12,5; 14,1] | 11 °C | +2,2 K |

Hay que separar dos cosas:
- **Las lecturas no se reproducen a 95 cm.** Pero a 50 cm la calibración tampoco las reproduce del todo,
  por el desacuerdo de arriba: con la G conjunta, cold_50 da 12,2 °C de superficie y hot_50 da 70,1 °C.
- **La transferencia de 50 a 95 cm sí se sostiene.** Ajusté una exponencial de enfriamiento a los frames
  de cada serie de 50 cm (solo datos de 50 cm) y la extrapolé al momento de la serie de 95 cm:

  | | predicho desde 50 cm | medido a 95 cm |
  |---|---|---|
  | caliente (inicio → fin) | 55,3 → 53,9 °C | 55,0 → 54,1 °C |
  | frío | 12,6 °C | 13,2 °C |

  El agua caliente se enfrió más deprisa de lo que admite el rango 58–66 °C. Con 66 °C al inicio, a los
  6 minutos estaba a ~54 °C, no a 58.
- **El desenfoque** se come un 3,3 % del contraste del agua a 95 cm frente a 50 cm (el vaso es más
  pequeño que el halo de la lente). Son ≈ −1 K en el caliente y +0,3 K en el frío. No está corregido;
  explica parte del +0,6 K del frío.

### Incertidumbre

Monte Carlo con 20 000 muestras (`task1_calibration.json` → `results`, `mc_*`):
- ε del papel uniforme entre 0,90 y 0,96;
- T_aire y lectura fría con σ = 0,5 °C;
- lectura caliente de 66 °C menos un retraso uniforme de 0 a 1,5 K;
- G en cualquier punto entre la del punto frío y la del caliente.

Escenario B: sin el supuesto del orden temporal, el primer frame caliente en cualquier punto entre 58 y
66 °C. Para hot_95 da 52,4 °C [49,6; 55,7] de superficie; para cold_95, 13,6 °C [12,7; 14,7].

Desglose, un factor cada vez (hot_95 / cold_95, en T aparente):

| factor | hot_95 | cold_95 | ¿afecta a T superficie? |
|---|---|---|---|
| desacuerdo de lecturas (G del punto frío frente al caliente) | **5,6 K** | 1,8 K | sí |
| rango del agua caliente, 58 frente a 66 °C | **3,0 K** | 0,9 K | sí |
| aire, 21,5 frente a 22,5 °C | 1,8 K | 0,8 K | sí |
| ε del papel, 0,90 frente a 0,96 | 1,7 K | 0,55 K | **no** (se cancela) |
| lectura fría, 10,5 frente a 11,5 °C | 1,2 K | 0,4 K | sí |

---

## Tarea 2: tamaño del vaso en píxeles

**Medición** sobre el frame medio de cada serie:
- anchos por fila con el cruce del 50 % entre el fondo y el nivel propio del vaso en esa fila;
- borde superior: el primer escalón significativo desde arriba (el borde del vaso, no la línea de agua);
- borde inferior: el escalón hacia la mesa.

**Predicción:** un tronco de cono de 7 / 5 / 6 cm apoyado en la mesa. Su posición sale del LiDAR: plano de
la mesa y frente del vaso, llevados al marco térmico con las dos extrínsecas.
- Eje del vaso a **0,503 / 0,951 m** (caliente) y **0,502 / 0,958 m** (frío).
- Fondo a 1,01–1,03 m.
- Así que «50 cm» y «95 cm» son distancias al eje.

Errores (medido / predicho − 1), con la distancia del LiDAR:

| caso | ancho arriba med. | intrínseca | simulador | ancho abajo med. | intrínseca | simulador | alto med. | intrínseca | simulador |
|---|---|---|---|---|---|---|---|---|---|
| hot_50 | 79,4 px | 84,6 (**−6 %**) | 127,5 (**−38 %**) | 63,6 | 63,7 (−0 %) | 96,3 (−34 %) | 80,0 | 78,5 (+2 %) | 118,0 (−32 %) |
| hot_95 | 43,6 | 44,9 (−3 %) | 66,9 (−35 %) | 36,0 | 34,0 (+6 %) | 50,8 (−29 %) | 40,8 | 40,7 (+0 %) | 60,9 (−33 %) |
| cold_50 | 78,4 | 84,4 (−7 %) | 128,1 (−39 %) | 63,3 | 63,7 (−1 %) | 96,7 (−35 %) | 73,0 | 78,3 (−7 %) | 117,8 (−38 %) |
| cold_95 | 42,8 | 44,5 (−4 %) | 66,6 (−36 %) | 36,5 | 33,8 (+8 %) | 50,6 (−28 %) | 37,4 | 40,1 (−7 %) | 60,1 (−38 %) |

- «intrínseca» es la calibración RGB-térmica: fx = 626,9, fy = 620,6 y Brown-Conrady.
- «simulador» es el pinhole del YAML original: 22,2 mm / 24 µm = **925 px**, sin distorsión.

**Intrínseca: −7 % a +8 %** (≤ 6 px). Lo sistemático:
- el ancho de arriba sale 3–7 % corto (¿el borde enrollado del vaso mide menos de 7 cm por fuera?);
- el de abajo a 95 cm sale 6–8 % largo (son 2–3 px, del orden del error de 2 px de la calibración).

**Simulador: −28 % a −39 %.** Dibuja el vaso ~1,5 veces más grande. Es el cociente 925/627: **la lente no
es de 22,2 mm**. La calibración implica 15,0 mm con pitch de 24 µm, y un HFOV de 54° en lugar de 38°.

Con la distancia nominal (0,50 / 0,95 m) los números cambian menos de 1 punto. El cociente de tamaños entre
50 y 95 cm (1,74–1,96) encaja con el de distancias del LiDAR (1,89–1,91).

Los renders lo confirman (`task3_comparison.json` → `geometry_from_renders`):
- con el YAML original, el vaso simulado mide lo que predice el pinhole de 925 px;
- con el YAML nuevo, lo que predice la intrínseca.

| caso | render, YAML original (HFOV 38,2°): error | render, YAML nuevo (HFOV 54,1°): error |
|---|---|---|
| hot_50 | −38 / −34 / −32 % | −6 / −0 / +3 % |
| hot_95 | −35 / −29 / −32 % | −3 / +6 / +1 % |
| cold_50 | −39 / −35 / −38 % | −7 / −0 / −6 % |
| cold_95 | −36 / −28 / −38 % | −3 / +8 / −7 % |

(ancho arriba / ancho abajo / alto; medido / render − 1, silueta exacta de la máscara de instancia)

**Posición:** con la pose del LiDAR, el vaso modelado queda 15–23 px más abajo que el real. La pose que se
ajusta a las filas de los cuatro vasos es 3,23 cm de altura y 1,21° de inclinación hacia arriba (RMS
2,2 px), frente a 3,3–3,7 cm y 2,4–2,8° según el LiDAR. Hay ~1,4° de error de cabeceo en la cadena
LiDAR→RGB→térmica. El render usa la pose ajustada.

---

## Tarea 3: real frente a simulado

**Escena:** la plantilla de dos vasos (`bench/camera_preview/two_cups`), cambiada así:
- **un solo vaso de papel por caso**: tronco de cono, ε = 0,93;
- parte mojada hasta la línea de agua (3,8 cm) a la temperatura medida;
- franja seca a la media entre esa temperatura y el aire (ESTIMADO);
- fondo y mesa a 22 °C, sala isoterma, un solo `WeatherSeries`;
- pose y distancias de la Tarea 2;
- YAML nuevo con la geometría calibrada.

**Temperaturas de entrada:** frío 11 °C en los dos casos; caliente 66 °C a 50 cm y 58 °C a 95 cm.

**Qué se compara:** la temperatura aparente del agua, real calibrada (Tarea 1) frente a la
`scene_apparent_t` del simulador en la misma región. El render coincide con
`L_B⁻¹(ε·L_B(T_s) + (1−ε)·L_B(22))` en menos de 0,03 K.

| caso | real (ε = 1) | simulado | **error real − sim** | Monte Carlo, 16–50–84 % | ¿independiente? |
|---|---|---|---|---|---|
| hot_50 | 67,2 °C | 63,3 °C | **+3,9 K** | +0,4 / +3,0 / +6,6 | no (se ajustó aquí) |
| cold_50 | 12,9 °C | 11,8 °C | **+1,1 K** | +0,4 / +1,1 / +1,9 | no |
| hot_95 | 52,1 °C | 55,8 °C | **−3,7 K** | −6,1 / −4,3 / −1,7 | sí |
| cold_95 | 13,8 °C | 11,8 °C | **+2,0 K** | +1,4 / +2,1 / +2,8 | sí |

- **A 50 cm** el error es solo el desacuerdo de las tres lecturas que una única G no puede absorber.
- **A 95 cm** el error viene casi entero de la temperatura de entrada, no del simulador. Si al simulador se
  le da la temperatura que predice la propia serie de 50 cm (53,9 °C y 12,6 °C), el error baja a
  **+0,15 K** (caliente) y **+0,5 K** (frío).
- La `apparent_t` de la cadena de cámara simulada difiere de su `scene_apparent_t` en menos de 0,1 K. Su
  PSF es solo difracción y el vaso es grande.

Lo que el simulador **no** reproduce y se ve en los datos:
- el desenfoque real (10–90 % ≈ 14 px), con su halo;
- el reflejo especular del vaso caliente en la mesa: ~40 % de su contraste debajo del vaso, mientras que el
  contrachapado del simulador es rugoso;
- la columna de vapor caliente que sube del vaso caliente y se mueve entre frames
  (`look/hot95_diff01.png`).

---

## Tarea 4: ruido medido y YAML propuesto

**Región:** filas 10–180 × columnas 330–530, sobre pizarra y pared, lejos de los vasos y de la columna de
vapor. Encima del vaso caliente esa columna se leía como σ_th = 1,3 cuentas. Los pares temporales se
toman dentro de un mismo periodo FFC; hay un FFC en cold_50, entre los frames 2 y 3, con +15 cuentas.

**El `denoise: 40` de la cámara tiene memoria.** El ruido de un par de frames crece con el tiempo entre
ellos y se estabiliza a partir de ~1,5 s:

| tiempo entre frames | σ del par (cuentas) | correlación |
|---|---|---|
| 0,1 s | 1,00 | 0,49 |
| 0,2–0,5 s | 1,18 | 0,28 |
| 1,5–5 s | 1,32 | 0,10 |
| > 5 s | 1,40 | 0 |

Por eso las componentes temporales se miden con pares a ≥ 1,5 s, que son frames independientes.

| componente (cuentas) | medido | cociente propuesto | antes (Boson) |
|---|---|---|---|
| σ_tvh | **1,39** (1,30–1,45 según el conjunto) | 1,00 | 1,00 |
| σ_vh (fijo, dentro de un periodo FFC) | 1,43 (de ello, 0,86 sobrevive de un conjunto a otro) | **1,03** | 0,30 |
| σ_v (filas fijas) | 0,58 | **0,41** | 0,08 |
| σ_h (columnas fijas) | 0,28 | **0,20** | 0,15 |
| σ_tv (filas temporales) | 0,30 | **0,22** | 0,05 |
| σ_th (columnas temporales) | 0,39 | **0,28** | 0,05 |
| σ_t (rebote del frame) | 0,10 (pares ≤ 0,35 s, cota inferior) | **0,07** | 0,02 |

- **NETD de salida:** 1,39 / 52,9 = **26 mK** (24–29 mK según la SITF), por debajo de los ≤ 50 mK de la
  hoja de datos. Es el ruido *después* del denoise.
- **Filas fijas ≈ 2 × columnas fijas**, como en el set del lab (la lectura está girada 90°).
- **Colas:**
  - más allá de 5σ hay 4·10⁻⁴ de las muestras (la gaussiana daría 6·10⁻⁷);
  - curtosis en exceso 2,4;
  - no las producen unos pocos píxeles ruidosos: solo el 0,09 % tiene más del doble de la σ mediana, y
    ninguno más del triple.
- **El ruido temporal está correlacionado en el espacio:** r = 0,37 con el vecino horizontal y 0,30 con
  el vertical. Es el denoise; el `tvh` del simulador es blanco.

**Píxeles malos.** La cámara los sustituye y su denoise borra la huella. No encontré:
- ningún píxel que copie a un vecino en los 63 frames (salvo 4 pares en el borde inferior derecho, un
  artefacto);
- ninguno con la varianza baja y la correlación alta que dejaría una interpolación;
- ningún defecto visible sin sustituir (> 8 σ_vh en los cuatro conjuntos).

Así que la población sustituida no se puede contar con estos datos. Se queda el 0,5 % de la hoja de
datos, y `bad_pixel_late_fraction: 0` (0 vistos de ~1500 implica < 0,2 % al 95 %).

**YAML nuevo:** `bench/camera_check/cellplus_ctc007_lwir_cups.yaml`, generado por `task4_noise.py`; el
loader lo acepta. Contiene:
- el bloque `noise` de arriba;
- `calibration.geometric` (la intrínseca medida, que sustituye a los 22,2 mm del diseño). No era parte de
  la tarea, pero sin ella el simulador dibuja el vaso un 50 % más grande;
- `calibration.radiometric.netd_k: 0.0263` y `sitf_dn_per_k: 52.91`.

**No representable en el esquema actual:** la memoria temporal del denoise, su correlación espacial y las
colas. Haría falta una extensión del esquema.

---

## Tarea 6: el desenfoque real, medido y puesto en la cámara

Fecha: 2026-10-26. La cámara confirmada es la **CTC-006-VGA**: 14,6 mm, F/1,3, 30 Hz. YAML nuevo:
`bench/camera_check/cellplus_ctc006_lwir_cups.yaml`. Los dos YAML anteriores no se tocaron.

### Focal y FOV

| | focal | FOV H × V |
|---|---|---|
| hoja de datos CTC-006 | 14,6 mm | 60° × 50° |
| pinhole 14,6 mm, 640 × 480 a 24 µm | — | 55,5° × 43,1° |
| calibración RGB-térmica (fx, fy) | 15,05 / 14,89 mm | pinhole 54,2° × 42,0° |
| calibración con su distorsión (bordes del sensor) | — | **61,3° × 44,7°** |

- La focal calibrada está un 2–3 % por encima de la nominal, dentro de la tolerancia habitual.
- El FOV horizontal coincide con la hoja de datos (61,3° frente a 60°). El vertical, no: 44,7° frente a 50°.
- La proyección del simulador sigue en `calibration.geometric`. `focal_length_mm` solo entra en el
  desenfoque y en el viñeteo cos⁴, así que el vaso no cambia de tamaño en el render.

### Medida: borde → ESF → LSF (`task6_psf.py`)

**Método:**
- Se usan los dos bordes laterales de cada vaso.
- La pared está inclinada 9,5°, así que cada fila muestrea el borde con otra fase subpíxel (slant edge).
- Los counts se usan restando la pizarra de cada frame y dividiendo por el contraste del agua de ese
  frame. Son lineales en radiancia, y eso elimina los saltos de FFC y el enfriamiento.
- Se promedian todos los frames de la serie (12–19).
- Modelo: un rectángulo convolucionado con la LSF, ajustado a todos los píxeles de las filas del agua.

| caso | frames | distancia | gaussiano σ (rms) | núcleo σ | halo σ | fracción en el halo (rms) | 10–90 % sin modelo |
|---|---|---|---|---|---|---|---|
| caliente 50 | 19 | 0,503 m | 5,14 px (0,013) | 4,75 | 14,9 | 9 % (0,012) | **13,5 px** |
| frío 50 | 15 | 0,502 m | 5,09 px (0,018) | 4,68 | 11,2 | 11 % (0,018) | **13,25 px** |
| caliente 95 | 17 | 0,951 m | 3,56 px (0,025) | 2,64 | 17,4 | 28 % (0,009) | **12,0 px** |
| frío 95 | 12 | 0,958 m | 3,84 px (0,032) | 2,73 | 19,4 | 33 % (0,017) | **13,75 px** |

- **A 50 cm un gaussiano encaja**: el halo apenas baja el residuo.
- **A 95 cm no encaja**:
  - hace falta un núcleo estrecho (σ ≈ 2,7 px) más un halo ancho (σ ≈ 18 px) con el 30 % de la energía;
  - el gaussiano solo deja 2–3 veces más residuo;
  - el halo sale igual en el vaso frío que en el caliente, así que no es la columna de vapor.
- El núcleo se estrecha con la distancia, así que el desenfoque es real. El halo relativo crece a 95 cm.
  Eso es lo típico de una lente con aberración esférica a un lado del foco, pero estos datos no lo
  demuestran.

### Ajuste con la óptica del simulador (`task6_focus_fit.py`)

**Método:**
- Para cada par (vergencia de foco x = 1/s_f, σ de aberración), la LSF es la OTF de Hopkins de `irsim`
  (§8.3, ADR 0129) por el gaussiano de aberración y por la caja del píxel, a 11 µm.
- Círculo de desenfoque `c = f²|1/s − x| / (F(1 − f x))`, con f = 14,6 mm.
- Se ajusta a los cuatro casos a la vez. Cada caso conserva su nivel, contraste, bordes e inclinación.

| modelo | foco | σ aberración | rms conjunto |
|---|---|---|---|
| M0: sin desenfoque | — | 104 µm (4,3 px) | 0,0257 |
| **M1: foco ≥ f + aberración (lo que admite el esquema)** | **infinito** | **80 µm (3,3 px)** | **0,0230** |
| M2: foco libre, también «más allá del infinito» | −0,2 dpt | 70 µm | 0,0229 |

Perfil del error frente al foco, con la aberración reoptimizada:

| vergencia | −0,4 | −0,2 | 0 (∞) | +0,15 (hiperfocal, 6,8 m) | +0,3 (3,3 m) | +0,5 (2 m) | +1 (1 m) |
|---|---|---|---|---|---|---|---|
| rms | 0,0230 | 0,0229 | 0,0230 | 0,0231 | 0,0233 | 0,0236 | 0,0245 |

- **Enfoque implícito: lejos, a ≳ 3 m.**
  - Los datos no distinguen el infinito de la hiperfocal ni de un ligero «más allá del infinito».
  - A 2 m o 1 m el ajuste es claramente peor.
- La lente está enfocada para distancias de vehículo, así que a 0,5 m desenfoca un círculo de 13,6 px.
  Eso explica la mayor parte del borde a 50 cm.
- Los 80 µm independientes de la distancia mezclan dos cosas que el esquema no separa:
  - las aberraciones de la lente;
  - el denoise espacial del núcleo (r = 0,37 con el vecino, tarea 4).
- **El halo de 95 cm no es representable en el esquema** (hueco de esquema).

### La cámara nueva en el render

Se renderizaron los cuatro casos en `sim6_<caso>/` (focus `infinity`, `hopkins`, `aberration_sigma_um: 80`,
`defocus_apply: layered`) y se midieron con el mismo script (`task6_psf.py --sim sim6`):

| caso | 10–90 % real | 10–90 % simulado antes (solo difracción) | **10–90 % simulado ahora** | σ gaussiano real / sim |
|---|---|---|---|---|
| caliente 50 | 13,5 px | 1,2 px | **12,0 px** | 5,14 / 4,58 |
| frío 50 | 13,25 px | 1,2 px | **12,0 px** | 5,09 / 4,54 |
| caliente 95 | 12,0 px | 1,2 px | **10,25 px** | 3,56 / 3,81 |
| frío 95 | 13,75 px | 1,2 px | **10,25 px** | 3,84 / 3,81 |

(Antes, el 10–90 % sale del ajuste gaussiano; la medida sin modelo tiene un suelo de ~1,7 px por el
suavizado de 5 bins.)

- **A 50 cm** el borde simulado queda un 10 % más fino que el real.
- **A 95 cm** coincide el gaussiano equivalente, pero el 10–90 % real lo marca el halo, que falta.

**Un problema encontrado y cómo se sorteó.**
- Con `layered` tal cual, el vaso a 50 cm salía con el mismo borde que a 95 cm (σ 3,8 px).
- **Causa:** `irsim.optics.layered.depth_layers` usa 8 bins de anchura igual en W020 sobre todo el rango del
  frame. La mesa llega a 3,2 cm debajo de la lente y se queda con 7 bins. Una sola capa va de 0,14 a 1,29 m,
  con el núcleo de 1,01 m, así que el vaso recibía el desenfoque de la pizarra.
- **Remedio, solo en el banco:** `common.Clock(min_distance_m=…)` mediante `CUPS_MIN_DISTANCE_M=0.3`
  (`report.json` → `defocus_min_distance_m`). Solo cambia la mesa a menos de 0,3 m, fuera del recorte de las
  figuras.
- El arreglo de verdad va en `src/`: más capas o bins acotados. Está en `ISSUE_DRAFT_layered_bins.md`.

### ¿Cambia la temperatura aparente de los vasos?

Media sobre la meseta del agua, con la misma definición (`find_cup`) en real y simulado:

| caso | real | sim, cámara anterior | **sim, CTC-006** | cambio del sim | real − sim antes → ahora |
|---|---|---|---|---|---|
| caliente 50 | 66,94 °C | 63,27 | **62,99** | −0,28 K | +3,67 → **+3,95 K** |
| caliente 95 | 51,96 °C | 55,69 | **55,27** | −0,42 K | −3,74 → **−3,32 K** |
| frío 50 | 12,87 °C | 11,71 | **11,74** | +0,03 K | +1,16 → **+1,12 K** |
| frío 95 | 13,85 °C | 11,73 | **11,88** | +0,15 K | +2,12 → **+1,96 K** |

- **Cambia poco: menos de 0,5 K.** Los vasos miden 40–80 px y el desenfoque solo les quita el borde.
- El cambio es mayor a 95 cm, donde el vaso es más pequeño: el caliente pierde un 1,2 % de su contraste
  de 34 K.
- En los datos reales, la pérdida de 50 a 95 cm era del 3,3 % (tarea 1). La diferencia es el halo que
  falta.
- Los errores real − sim siguen dominados por las lecturas de temperatura (tareas 1 y 3), no por la óptica.

### Nota aparte: el reflejo del vaso en la mesa

**Sí está en los datos.** En el eje del vaso, justo debajo del borde inferior, hay una meseta de unas
12 filas:

| caso | fracción del contraste del vaso |
|---|---|
| caliente 50 | 0,42 |
| caliente 95 | 0,49–0,55 |
| frío 50 | 0,47–0,52 |
| frío 95 | 0,49–0,57 |

Después decae hacia la mesa. A dos anchos de vaso hacia un lado, la mesa da 0 ± 0,03.

**Es un reflejo especular, no conducción:**
- es igual de fuerte con el vaso frío que con el caliente;
- tiene la forma del vaso;
- empieza en el mismo borde.

**Cuadra con la física:** la cámara está a 3,3 cm sobre la mesa y mira el tablero a ~86° de incidencia.
A ese ángulo, la reflectancia de Fresnel de una superficie barnizada (n ≈ 1,5) ronda 0,5.

**El simulador no lo reproduce hoy:**
- El término reflejado es el entorno (cielo más un suelo uniforme, ADR 0045).
- El lóbulo GGX (ADR 0067) solo muestrea ese entorno.
- Los reflejos entre objetos de la escena no están modelados.

**Formas de conseguirlo**, de la más barata a la más completa (skill `isaac-sim-spg`, «The reflection
problem»):
1. Un rayo Warp en la dirección especular, solo para los materiales marcados como especulares (la mesa),
   contra el campo de temperaturas.
2. Una segunda pasada de render con rebotes, codificando radiancia.

Además, la mesa tendría que tener un material con Fresnel especular a ángulo rasante. El contrachapado
rugoso actual no lo tiene. No está implementado.

---

## Limitaciones de estos datos

1. **Las lecturas de temperatura son el eslabón débil.**
   - Hay una sola lectura fría para dos series.
   - El rango caliente de 8 °C no viene asignado a ningún momento.
   - No sabemos el instrumento, el punto medido ni la hora.
   - Las tres lecturas son incoherentes entre sí en ~2 K para una cámara lineal en radiancia. Ese
     desacuerdo domina la incertidumbre (±3 K en el caliente a 95 cm).
   - Un termopar en la pared del vaso, registrado con hora, lo resolvería.
2. **No hay referencia de cuerpo negro.** Calibrar con vasos de papel (ε 0,90–0,96, no medida) y con un
   fondo que se supone a la temperatura del aire acopla la ganancia a la emisividad. Esa ε no afecta a la
   temperatura de superficie transferida, pero sí a la aparente.
3. **El agua cambia durante cada serie**: −2,3 K en 44 s el caliente a 50 cm. Las series duran 42–56 s y
   el contraste no es estacionario.
4. **Son capturas sueltas, no vídeo**, con `denoise: 40` activo y FFC sin registrar.
   - El ruido medido es el de la salida filtrada.
   - Separar el núcleo de su filtro necesita 30 s de vídeo con denoise 0 ante la tapa de la lente.
5. **Desenfoque y halo de la lente** (14 px de 10–90 %): pierden un 3 % de contraste a 95 cm. El
   simulador solo tiene difracción.
6. **Geometría.**
   - La calibración RGB-térmica tiene 2 px de error estéreo.
   - La cadena LiDAR→térmica tiene ~1,4° de error de cabeceo.
   - Las medidas del vaso son de usuario: 7 / 5 / 6 cm.
7. **La respuesta espectral 8–14 µm es ESTIMADA.** La forma de L_B(T) entra en la prueba de linealidad.
   Con bandas planas de 7,5–13,5, 8–12, 10–14 o 7–15 µm, el cociente previsto va de 4,9 a 5,2: no llega
   a 6,29, así que el desacuerdo no es un problema de banda.
8. **Escena.** Ni el reflejo especular en la mesa ni la columna de vapor están en la escena simulada; la
   franja seca del vaso se puso a una temperatura estimada.
