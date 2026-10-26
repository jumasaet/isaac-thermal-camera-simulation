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
