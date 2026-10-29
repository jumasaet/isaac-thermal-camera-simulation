# Tres vasos (frío, templado, caliente): calibración con prueba independiente, real frente a simulado, similitud y reflejo

Fecha: 2026-10-28; **rehecho el 2026-10-29 con la temperatura constante** (ver «Cambio del 29-oct.» abajo). Datos: `thermal_specs/three_cups_experiments/sync_cups/`. Todo está en este directorio, salvo un
comentario corregido en `bench/camera_check/cellplus_ctc006_lwir_cups.yaml` (la cadencia, abajo). Nada en `src/` ni en `configs/`.

| Paso | Script | Intérprete | Salida (`out/`, `figures/`) |
|---|---|---|---|
| 0 | `task0_scene.py` | Isaac `python.sh` | `scene_geometry.{json,png}`: LiDAR, pose, siluetas, nivel del agua |
| 1 | `task1_calibration.py` | Isaac | `task1_calibration.{json,png}`, `task1_per_frame.npz`, `real_apparent_t_*.npy` |
| 2a | `task2_render.py` (Kit, ver su docstring para el `env -u …`) | Isaac | `sim/`: 8 instantes, `report.json` |
| 2b | `task2_compare.py` | Isaac | `task2_comparison.{json,png}`, `sim/apparent_t_ref_*.npy` |
| 3 | `task3_similarity.py` | conda `vision` + lpips 0.1.4 y scikit-image 0.24 (`pip install --target`, ver docstring) | `task3_similarity.{json,png}` |
| 4 | `task4_reflection.py` | Isaac | `task4_reflection.{json,png}` |
| 5 | `task5_figures.py` | Isaac | `figures/cups3_sheet_{inferno,gray}.png`, `figures/cups3_results_table.png` |
| 6 | `task6_noise.py` (29-oct.) | Isaac | `task6_noise.{json,png}`: ruido 3D a `denoise: 80` |
| 7 | `task7_housing.py` (29-oct.; sin Kit) + 2a/2b con `CUPS3_SENSOR_YAML`, `CUPS3_SIM_DIR` | Isaac | `task7_housing.json`, `sim_cups3yaml/`, `task2_comparison_sim_cups3yaml.{json,png}` |

`cups3_common.py` reúne lo compartido: lectura de los datos, horas, temperaturas por frame, regiones de los vasos,
superficie de referencia del fondo y radiancia de banda. La radiometría y la geometría LiDAR→térmica vienen de
`bench/cups_check/cups_common.py` (mismo rig, mismos ficheros de calibración). Los `.npy` no se versionan
(`bench/.gitignore`); se regeneran con los scripts.

---

## Cambio del 29-oct.: temperatura constante en lugar de rampa

La primera versión interpolaba cada vaso en línea recta entre la lectura «antes» (en el primer frame) y la «después»
(en el último), como si las lecturas se hubieran tomado al empezar y acabar la grabación. Los datos lo contradicen
(sección «Lo que los datos dicen…»: el caliente bajó 1,15 K en los 29 s, no 7 K), y **las lecturas no tienen hora**.
Como no se sabe cuándo se tomó cada una, ahora cada vaso va a **temperatura constante, la media de sus dos lecturas**:
caliente 65,8 °C, templado 35,45 °C, frío 7,35 °C. Es en la calibración (tarea 1), en el render (tarea 2) y en todo lo
que cuelga de ellos. El resto (geometría, tarea 0, reflejo de la tarea 4) no depende de las temperaturas y no cambia.

| | antes (rampa) | **ahora (media)** |
|---|---|---|
| **error del templado, prueba independiente**, superficie (media de 94 frames) | −0,50 K | **−0,49 K** |
| ídem, aparente (ε 0,93) | −0,47 K | **−0,46 K** |
| ídem, primer / último frame (superficie) | −0,96 / +0,16 K | **−0,38 / −0,56 K** |
| ídem, Monte Carlo A, aparente [68 %] | [−0,99; +0,02] K | [−0,99; +0,02] K |
| G·ε (cuentas por W m⁻² sr⁻¹) | 53,23 | **53,57** |
| frío solo / caliente solo (difieren) | 52,01 / 53,54 (2,9 %) | 52,16 / 53,92 (3,4 %) |
| **error real − sim, frío** (media de 8 instantes) | +0,35 K | **+0,40 K** |
| **error real − sim, templado** (independiente) | −0,43 K | **−0,43 K** |
| **error real − sim, caliente** | +0,16 K | **+0,28 K** |
| deriva del error del caliente, primer → último de los 8 instantes | −2,23 → +3,25 K (rango 5,5 K) | **+0,87 → −0,23 K (rango 1,1 K)** |
| ídem, desviación típica del error del caliente | 2,03 K | **0,40 K** |
| rango del error en el frío / templado | 0,74 / 1,04 K | 0,45 / 0,18 K |
| SSIM, frame completo / recorte (5–70 °C) | 0,986 / 0,970 | 0,986 / 0,970 |
| LPIPS, frame completo / recorte (5–70 °C) | 0,058 / 0,101 | 0,057 / 0,100 |
| SSIM / LPIPS en la ventana 20–40 °C, recorte | 0,872 / 0,243 | 0,873 / 0,242 |

**Cómo leerlo.**
- **El error del templado no cambia** (−0,43 K frente a −0,43 K; −0,46 frente a −0,47 en aparente): lo que sí se
  mueve es su deriva dentro de la grabación, que se reduce de 1,1 K a 0,2 K. La media ya cuadraba, porque la
  grabación cae hacia el centro del intervalo entre lecturas. La incertidumbre sigue siendo la misma: el −0,5 K
  está a ~0,9 σ de cero.
- **Los errores del frío y del caliente suben unas décimas** (+0,05 y +0,12 K). Son residuos de la calibración, no
  pruebas: con la media, la calibración usa 65,8 en lugar de la rampa, y G·ε cambia un 0,6 %.
- **La deriva del caliente era un artefacto de la rampa.** Ahora el simulador está quieto y el vaso real baja 1,1 K
  (de 64,07 °C aparentes en los 3 primeros frames a 62,98 en los 3 últimos), así que el error real − sim pasa de
  +0,87 a −0,23 K. Esa pendiente es **la bajada real del vaso, no un fallo del simulador**: no se reproduce porque
  sin horas de lectura no se puede prescribir. Ya no hay un ±3 K que no existe en los datos.
- **SSIM y LPIPS casi no se mueven** (≤ 0,001 y ≤ 0,001): están dominadas por el contenido que falta (reflejo,
  ruido, bordes) y no por 1–2 K en un vaso, en una ventana de 65 K.
- **La correlación de rango sigue en 1,00 en los 8 pares.**

Los valores «antes» de las secciones de abajo se han sustituido por los nuevos; solo esta tabla guarda los antiguos.

---

## Datos

| | |
|---|---|
| térmica | 94 frames `int16` (480, 640), cuentas crudas de 14 bits, más alto = más caliente, `invert_raw: true`. **Solo 92 son distintos**: los sets 80/81 y 88/89 son el mismo frame de cámara, y faltan los sets 59 y 71 (`bench/data_check/RESULTADOS.md`) |
| RGB | 93 PNG 1280 × 960, ya con balance de blancos (faltan las de los sets 59, 71 y 91; 88 = 89) |
| LiDAR | 96 nubes Livox |
| duración | 28,80 s entre el primer y el último frame; intervalos de 0,014 a 5,59 s |
| hora | 2026-10-27 07:46:59 UTC (16:46 en Corea) |

- **Horas:** las del nombre de archivo (la marca del LiDAR de cada set), como pediste. La marca propia del frame
  térmico (`sync_index.csv`) difiere menos de 33 ms, que no cuenta para una temperatura del agua.
- **`thermal_info.json`:** el mismo núcleo y firmware (Obsidian MirVG, 4.6.6.3) y el mismo `t_list`. Lo que cambia es
  **`denoise: 80`**; en las sesiones anteriores era 40. El ruido medido entonces (`ratios_3d` del YAML) es el de denoise 40.
- **`thermal_sync_events.csv`:** 19 avisos de frames perdidos por USB entre las horas 223,7 y 323,9. El nodo térmico
  siguió unos 100 s, más allá de los 29 s sincronizados. En un aviso faltan 63 frames en 5,0 s, más de los 50 que
  mandaría una cámara a 10 Hz. **Es una prueba de que el núcleo va a más de 12,6 Hz**, compatible con los 30 Hz.

### Cámara: 30 Hz

El YAML de la CTC-006 ya tenía `frame_rate_hz: 30.0`, pero su comentario atribuía los 10 Hz a la herramienta de
captura. Lo he corregido: los 10 Hz vienen de sincronizar con el LiDAR de 10 Hz, que guarda un set por barrido y lo
descarta si una cámara se desvía más de 33 ms (`fps_summary.json`). El render confirma que la cámara simulada va a
30,0 Hz (`sim/report.json`).

---

## Geometría de la escena (`task0_scene.py`)

**LiDAR** (96 nubes, llevadas al marco térmico con las extrínsecas del rig):

| | valor |
|---|---|
| cámara sobre la mesa | 3,29 cm |
| cabeceo / alabeo según el LiDAR | 3,02° / 1,40° |
| fondo | a 1,008 m, guiñada 6,1°, bordes a +0,51 / −0,43 m del eje, 0,52 m de alto |
| eje de cada vaso (adelante) | frío 0,494 m, templado 0,511 m, caliente 0,525 m |
| punto más cercano de la base | **0,500 / 0,487 / 0,513 m**: los ~50 cm que diste |
| separación entre ejes | 14,7 y 14,9 cm (con el rumbo de la imagen): los ~15 cm que diste |

**Pose ajustada a las siluetas térmicas** (borde superior, borde inferior y columna central de los tres vasos):

- La distancia viene del LiDAR y el rumbo de la imagen. La cámara fija el ángulo a < 1 px (0,8 mm a 0,5 m), mientras
  que la mediana del frente del vaso en el LiDAR se dispersa ~1 cm. Corrección lateral sobre el LiDAR:
  +3,2 / +8,1 / +0,2 mm.
- Con los tres vasos a la misma distancia, la altura de la cámara y el cabeceo son degenerados: un ajuste libre se
  iba a h = −1 mm. La altura se queda en la del LiDAR.
- **Resultado: cabeceo 0,97°, alabeo 1,30°, RMS 1,3 px.** Con la pose del LiDAR el RMS era 18,2 px.
- El alabeo coincide con el del LiDAR (1,40°). El cabeceo difiere 2,05°: es el mismo error de la cadena
  LiDAR→RGB→térmica que vio `bench/cups_check` (~1,4°).
- El alabeo es lo que pone el vaso frío ~10 px más bajo que el caliente. La cámara del simulador mantiene las alas
  niveladas, así que `task2_render.py` compone el alabeo sobre su orientación. Las siluetas renderizadas caen a
  < 0,15 px de las del modelo.

**Tamaño frente al tronco de cono de 7 / 5 / 6 cm:**
- ancho del borde superior: entre −5,5 y −6,8 % (como en la sesión anterior: el borde enrollado mide menos de 7 cm
  visto desde aquí);
- ancho de la base: entre +1,4 y +2,6 %;
- alto: +0,7 % en el templado y +5 % en el frío y el caliente. En estos dos, el reflejo de debajo alarga el borde
  inferior medido.

**Nivel del agua** (`out/scene_geometry.png`). Modelo de la columna central: fondo | franja seca | pared mojada,
con desenfoque gaussiano.

| vaso | línea de agua bajo el borde | nivel de la franja seca (fracción del contraste mojado) |
|---|---|---|
| frío | **11,3 mm** | 0,24 |
| caliente | **12,1 mm** | 0,67 |
| templado | no se resuelve | 0,95 con la línea fijada a 11,7 mm; libre da 5 mm y 0,82, igual de bueno |

- El agua está a **~12 mm del borde: 4,8 cm de 6**, unos 125 ml. Es «casi hasta el borde», pero no del todo.
- En el templado, la franja seca está casi a la temperatura del agua, así que no tiene línea visible.
- El simulador usa la línea a 4,83 cm en los tres vasos y estas fracciones para la franja seca.

---

## Tarea 1: calibración con prueba independiente

### Modelo

`c = G·L_app + O(t)`, lineal en radiancia de banda (docs/physics-model.md §3.2 b, §9.3):

- **Fondo:** `L_app = L_B(T_aire)`. Sala cerrada e isoterma, así que vale para cualquier emisividad. `O(t)` es la
  superficie cuadrática ajustada al fondo, evaluada en el centro de cada ROI, en cada frame. Así salen los saltos de
  FFC, la deriva y el gradiente del fondo (~20 cuentas de izquierda a derecha).
- **Vaso:** `Δc = c_vaso − c_fondo = G·ε·(L_B(T_s) − L_B(T_aire))`. Los datos fijan el producto **G·ε**.
  - La temperatura de superficie `T_s` sale **sin depender de ε**.
  - La aparente (lectura con ε = 1) usa `G = G·ε / 0,93`.
- **ROI del agua:** la meseta ≥ 90 % del pico en la columna central, y la mitad central del ancho.
- **Temperatura del agua por frame:** constante, la media de las lecturas «antes» y «después» (las lecturas no
  tienen hora). `T_s = T_agua` (factor de pared k = 1).

### Calibración (caliente + frío + fondo)

| | valor |
|---|---|
| **G·ε** | **53,57 cuentas por W m⁻² sr⁻¹** |
| solo el frío / solo el caliente | 52,16 / 53,92: **difieren un 3,4 %** (razón de Birge 1,07, con ±0,5 K por lectura) |
| G con ε = 0,93 | 57,60; SITF **48,3 cuentas/K a 300 K** (la sesión anterior: 52,9) |
| cociente de contrastes caliente/frío, medido | **3,393** |
| predicho, lineal en radiancia | 3,283 |
| predicho, lineal en kelvin | 2,532 |

- **Esta vez las tres lecturas sí son coherentes** entre sí, a diferencia de `bench/cups_check`, donde diferían un 23 %.
- El modelo en radiancia es claramente el bueno: el lineal en kelvin falla un 25 % en el cociente.
- El caliente pesa el 80 % en la media ponderada, porque su contraste de 42 K hace su ganancia menos sensible a la
  lectura.

### Prueba independiente: el vaso templado

| | calibrado | sonda | **error** |
|---|---|---|---|
| T de superficie (sin ε), media de 94 frames | 34,96 °C | 35,45 °C | **−0,49 K** |
| T aparente (ε = 0,93) frente a `L_B⁻¹(ε L_B(T_w) + (1−ε) L_B(T_aire))` | 34,22 °C | 34,68 °C | **−0,46 K** |
| error en el primer / último frame (superficie) | | | −0,38 / −0,56 K |

### Incertidumbre

Monte Carlo, 20 000 muestras (`task1_calibration.json`, `mc_*`):
- cada lectura de sonda y el aire con σ = 0,5 K;
- el fondo frente al aire con σ = 0,3 K;
- el factor de pared k entre 0,85 y 1;
- ε entre 0,90 y 0,96;
- la ROI al azar entre completa, mitad superior y mitad inferior (los vasos están estratificados y no sabemos a qué
  profundidad midió la sonda).

| escenario | error de superficie, mediana [68 %] [95 %] | error aparente [68 %] |
|---|---|---|
| **A: constante, la media de las lecturas (el supuesto)** | **−0,52 K [−1,11; +0,10] [−1,68; +0,69]** | −0,48 K [−0,99; +0,02] |
| B: cada vaso en cualquier punto entre sus dos lecturas, constante | −0,51 K [−1,35; +0,34] [−2,14; +1,18] | −0,47 K [−1,22; +0,28] |

El −0,5 K está a ~0,9 σ de cero. **La prueba pasa dentro de la incertidumbre.**

Desglose, un factor cada vez (desplazamiento del error del templado):

| factor | desplazamiento |
|---|---|
| lectura del templado ±0,5 K | ∓0,50 K |
| fondo frente al aire ±0,5 K | ±0,47 K |
| aire ±0,5 K | ±0,44 K |
| todas las lecturas en «antes» / en «después» | +0,24 / −0,23 K |
| lectura del caliente ±0,5 K | ±0,12 K |
| lectura del frío ±0,5 K | ∓0,06 K |
| ROI mitad superior / inferior | +0,04 / −0,06 K |
| factor de pared 0,85 | −0,03 K (un k común se absorbe en G·ε) |

Dominan la propia lectura del templado y la temperatura que se supone en el fondo y el aire.

### Lo que los datos dicen de la rampa entre lecturas (por eso se usa la media)

**Los datos contradicen la rampa de la primera versión.** Cambio de cada vaso durante los 29 s, leído de la propia calibración (superficie, sin ε):

| vaso | medido | que daría la rampa (descartada) |
|---|---|---|
| frío | +0,28 K | +1,1 K |
| templado | −0,18 K | −1,3 K |
| caliente | −1,14 K | **−7,0 K** |

- Las lecturas se tomaron con minutos de separación, no en los 29 s de la grabación. Al ritmo medido después del
  transitorio inicial (frío +0,29, templado −0,30, caliente −2,3 K/min), el cambio entre las dos lecturas lleva
  **3,8 / 4,3 / 3,0 min**. Los tres vasos dan lo mismo: las lecturas se tomaron con **~3–4 min** de separación, y
  probablemente a la vez en los tres.
- Las medias sí cuadran (la propia calibración da hoy caliente 66,04 °C frente a 65,8 °C prescritos; templado 34,96
  frente a 35,45; frío 7,82 frente a 7,35), porque la grabación cae hacia el centro del intervalo. Por eso **G·ε apenas
  se ve afectado** (escenario B) y **se usa la media**.
- Lo que producía la rampa era una deriva falsa: la del error del templado durante la grabación (−0,96 → +0,16 K,
  ahora −0,38 → −0,56 K) y la del caliente en la tarea 2.
- En los primeros ~1,5 s el frío sube 0,35 K, el caliente baja 0,25 K y el templado baja 0,1 K; después se quedan
  casi quietos. No es un cambio común de ganancia: en proporción es 1,9 / 0,8 / 0,3 %. **Hipótesis (no
  comprobada):** el agua se acababa de remover con la sonda y la pared se relaja al pararse la convección.

---

## Tarea 2: real frente a simulado

**Escena** (`task2_render.py`):
- tres troncos de cono de papel (ε 0,93) con la geometría de la tarea 0;
- pared mojada hasta 4,83 cm a la temperatura constante de la media de las dos lecturas (el mismo supuesto);
- franja seca con las fracciones medidas;
- fondo con las medidas del LiDAR, sin la guiñada (la sala es isoterma, así que la guiñada solo movería sus bordes);
- pared trasera a 2,5 m;
- mesa de contrachapado;
- todo a 23,9 °C, con un solo `WeatherSeries`.

Se renderizan **8 instantes**, los de los frames reales 0, 13, 27, 40, 53, 66, 80 y 93, cada uno a su hora.

**Misma definición en los dos lados:**
- ROI: `find_cups` sobre la imagen media de cada lado;
- referencia: la superficie cuadrática del fondo;
- simulado: la `apparent_t` de la cámara simulada, referida al fondo igual que la real. La cámara simulada lee el
  fondo **+0,28 K** por encima de la escena (autoemisión de carcasa y óptica, con parámetros que siguen siendo los
  del Boson en el YAML, PENDIENTE), y además añade un cuenco de viñeteo de 0,1 a 0,8 W m⁻² sr⁻¹. La calibración real
  quita el offset de la cámara real del mismo modo.

| vaso | ¿independiente? | sonda (media) | real calibrado | simulado | **error real − sim** | primer / último frame |
|---|---|---|---|---|---|---|
| frío | no (calibra) | 7,35 °C | 9,02 °C | 8,63 °C | **+0,40 K** | +0,08 / +0,53 |
| **templado** | **sí** | 35,45 °C | 34,22 °C | 34,65 °C | **−0,43 K** | −0,33 / −0,49 |
| caliente | no (calibra) | 65,80 °C | 63,49 °C | 63,21 °C | **+0,28 K** | +0,87 / −0,23 |

- **El único error independiente es el del templado.** El frío y el caliente fijaron la calibración real con las
  mismas temperaturas que recibe el simulador, así que su error es el residuo de la calibración, no una prueba.
- La incertidumbre del templado es la de la tarea 1 en aparente: **[−0,99; +0,02] K al 68 %**.
- **El simulador no aporta error propio:**
  - su escena coincide con `L_B⁻¹(ε L_B(T_s) + (1−ε) L_B(23,9))` en ≤ 0,021 K;
  - su cámara, referida al fondo, cambia los vasos en +0,01 / −0,02 / −0,04 K: el desenfoque les quita un poco de
    contraste;
  - el error del templado (−0,43 K) es la prueba de la tarea 1 (−0,47 K) vista a través del simulador.
- Sin referir el simulado al fondo, los errores serían −0,10 / −0,51 / −0,02 K (valores de la primera versión; no se han rehecho).
- **La deriva del caliente** es ahora +0,87 → −0,23 K (desviación típica 0,40 K; con la rampa era −2,2 → +3,3 K,
  desviación 2,03 K). El simulador está quieto y el vaso real baja 1,1 K en los 29 s: esa pendiente es la bajada
  real, que no se puede prescribir sin hora de lectura.
- **Un intento descartado:** con un plano como referencia del fondo, en lugar de la superficie cuadrática, quedaba la
  curvatura del cuenco de la cámara simulada (−0,16 K en el templado, que está centrado).

---

## Tarea 3: similitud de imagen

**Pares:** los 8 instantes simulados frente al frame real de la misma hora. Las dos imágenes son temperatura aparente
referida al fondo, en la **misma ventana**: `x = clip((T − 5 °C) / 65 K, 0, 1)`, el rango de los vasos.

- **SSIM:** Wang 2004, gaussiana σ 1,5, K1 0,01, K2 0,03, `data_range` 1.
- **LPIPS:** AlexNet, v0.1, gris repetido en 3 canales, escalado a [−1, 1].
- Se calculan en el frame completo y en el recorte de los vasos y sus reflejos (filas 160–380, columnas 30–570).

| ventana 5–70 °C, 8 pares (rango) | SSIM | LPIPS |
|---|---|---|
| frame completo | **0,986** (0,986–0,987) | **0,057** (0,056–0,059) |
| recorte de los vasos | **0,970** (0,970–0,971) | **0,100** (0,098–0,101) |
| con la ventana 20–40 °C (fondo y mesa): completo / recorte | 0,916 / 0,873 | 0,194 / 0,242 |

**Correlación de rango** del brillo de los tres vasos y el fondo: **Spearman = Kendall = 1,00 en los 8 pares**. El
orden es el mismo en real y simulado: frío < fondo < templado < caliente.

### Cuánto pesa la desalineación

Posición residual por bordes (silueta real − renderizada):
- columna central: −0,16 / −0,09 / −0,03 px;
- borde superior: −2,2 / +0,1 / −1,9 px;
- borde inferior: +1,8 / +0,4 / +1,8 px.

Los vasos están centrados al píxel. El ±2 px vertical es de **tamaño**, no de posición: los vasos reales se ven ~4 px
más altos por el reflejo y ~5 px más estrechos en el borde que el modelo de 7/5/6 cm.

| | SSIM recorte | LPIPS recorte |
|---|---|---|
| tal cual | 0,970 | 0,100 |
| registrado (mejor desplazamiento global en ±4 px: +1 fila) | 0,971 | 0,098 |
| desplazar 2 px / 4 px (rejilla en el par central) | −0,002 / −0,007 | +0,002 / +0,009 |
| referencia: real contra el frame real siguiente | 0,9997 | 0,0002 |
| referencia: real contra sí mismo desplazado 1 / 2 / 4 px | 0,998 / 0,995 / 0,986 | 0,003 / 0,007 / 0,018 |

Ventana alrededor de cada vaso y su reflejo (par central), SSIM / LPIPS tal cual → registrado localmente:

| vaso | desplazamiento local | SSIM | LPIPS |
|---|---|---|---|
| frío | (3, 0) px | 0,975 → 0,978 | 0,063 → 0,056 |
| templado | (2, −1) px | 0,990 → 0,991 | 0,049 → 0,047 |
| caliente | (0, −1) px | 0,925 → 0,926 | 0,159 → 0,160 |

**Conclusión: la desalineación pesa poco.**
- Registrar mejora la SSIM en < 0,001 y el LPIPS en ~0,002 (en local, como mucho 0,002 y 0,006).
- Lo que falta hasta 1 (0,030 de SSIM y 0,10 de LPIPS en el recorte) es contenido que el simulado no tiene:
  - **el reflejo en la mesa**, por eso el caliente tiene la SSIM local más baja (0,925);
  - los bordes del fondo y la sala detrás;
  - el ruido y la textura;
  - el cuenco de viñeteo del modelo.
- Una desalineación de 4 px costaría 0,007 de SSIM, menos de una cuarta parte de esa diferencia.

---

## Tarea 4: reflejo en la mesa (solo medido)

### Definición

Las cuentas son lineales en radiancia, así que el cociente no necesita ni ganancia, ni ε, ni temperaturas. Para cada
vaso, con `b` la fila de contacto con la mesa (su borde inferior medido) y las columnas de su ROI de agua:

    ρ(d) = [ c(b + d) − c_mesa(b + d) ] / [ c(b − d) − c_fondo(b − d) ]

- `c(r)` es la mediana de la fila r en esas columnas.
- Con la cámara a 3,3 cm sobre una mesa plana, la imagen reflejada es el vaso volteado sobre su fila de contacto:
  la fila b + d muestra la pared que está d filas por encima.
- `c_mesa` es la mediana de dos franjas de mesa, a 20–45 px fuera del contorno del vaso, a izquierda y derecha, en
  la misma fila.
- `c_fondo` es la superficie del fondo en esa fila.
- `d` empieza en 15 filas, para que el borde desenfocado del propio vaso aporte < 1 %.
- **ρ_near** es la mediana para d = 15–27 (pared baja, a 84–85° de incidencia). **ρ_mirror** es la mediana para toda
  la pared mojada (hasta 80–81°).
- Para una mesa especular en una sala isoterma, ρ es su reflectancia especular a ese ángulo (diluida por el
  desenfoque); 0 sería que no refleja nada.

### Resultado (frame medio; dispersión entre frames ≤ 0,003)

| vaso | ρ_near | ρ_mirror | ρ en d = 15 | incidencia |
|---|---|---|---|---|
| frío | **0,35** | 0,23 | 0,42 | 84,6° → 81,1° |
| templado | **0,40** | 0,23 | 0,43 | 84,8° → 80,3° |
| caliente | **0,40** | 0,29 | 0,45 | 84,9° → 81,0° |
| simulado (los tres) | 0,00 | 0,00 | | |

- **La mesa refleja lo mismo en los tres vasos**, ~0,4 junto a la base, y cae hacia 0,15 al bajar a 81°, como
  Fresnel lejos del rasante.
- **El reflejo del caliente se ve mucho más solo porque su contraste es 4 veces el del templado** (2220 cuentas
  frente a 510), no porque refleje más.
- **Incertidumbre por la referencia de mesa:** las dos franjas difieren −20 / −11 / −4 cuentas (gradiente de la
  mesa), es decir ±0,015 / ±0,011 / ±0,001 en ρ.
- **El ρ_mirror del caliente** sale más alto por un segundo lóbulo a d > 45 filas. Ahí ρ vuelve a subir a 0,33, con
  un borde diagonal visible. No es la imagen especular del vaso: esas filas reflejan la franja seca, que está más
  fría. Es algo de la superficie de la mesa en ese punto, sin identificar.
- La regla metálica cruza la mesa entre el templado y el caliente, fuera de las franjas de referencia.
- El simulador da 0 porque su contrachapado es rugoso y solo refleja el entorno (ADR 0045, 0067). Los reflejos entre
  objetos no están modelados; las opciones ya están en `bench/cups_check/RESULTADOS.md`, tarea 6. No se implementa.

---

## Tarea 5: figuras (en inglés)

- `figures/cups3_sheet_inferno.png` y `figures/cups3_sheet_gray.png`. Contienen:
  - RGB real y simulada;
  - térmica real calibrada y simulada en la misma ventana de 5–70 °C, con una sola barra de temperatura;
  - el mapa real − simulado en K;
  - el detalle en 20–40 °C;
  - dos tablas de resultados (por vaso y global).
  - Instante: frame 53, a 13,2 s. Las temperaturas de sonda de las tablas son las medias de las dos lecturas.
- `figures/cups3_results_table.png`: las tablas solas.
- `out/task1_calibration.png`, `out/task2_comparison.png`, `out/task3_similarity.png`, `out/task4_reflection.png` y
  `out/scene_geometry.png`: las figuras de diagnóstico de cada tarea.

---

## Limitaciones

1. **Hora de las lecturas.** No tienen hora, y las lecturas distan ~3–4 min (el caliente bajó 1,15 K en los 29 s,
   no los 7 K entre lecturas). Por eso cada vaso va a la media de sus dos lecturas. El error de cada vaso lleva
   la incertidumbre de ese valor (hasta ±3,5 K en el caliente, ±0,65 K en el templado, según dónde cayera la
   grabación) y la bajada real del caliente (1,1 K) no se puede reproducir. Una sonda registrada con hora, o dejada
   dentro del agua, lo resolvería.
2. **Fondo y aire.** Que el fondo esté a 23,9 °C pesa ±0,47 K por cada 0,5 K en el resultado del templado. No hay
   cuerpo negro de referencia.
3. **ε del papel no medida.** No afecta a la temperatura de superficie, pero sí a la aparente.
4. **`denoise: 80`.** El ruido del YAML es de denoise 40. **Medido el 29-oct. (tarea 6)**: cambia poco, y los valores
   están en `bench/camera_check/cellplus_ctc006_lwir_cups3.yaml`, que usan las tareas 2–5 desde el 29-oct. (abajo, «YAML nuevo por defecto»).
5. **El simulador no tiene** el reflejo en la mesa, la sala detrás del fondo, la columna de vapor ni la regla. Con el
   YAML anterior, su cámara tiene un offset de carcasa (+0,28 K) y un cuenco de viñeteo. **Los dos salen de un mismo
   desacuerdo de temperaturas de carcasa y desaparecen con el YAML nuevo (tarea 7)**; los valores de la Boson siguen
   PENDIENTES.
6. **Geometría.** El cabeceo de la cadena LiDAR→térmica falla ~2°, y se corrige con las siluetas. La altura de la
   cámara depende solo del LiDAR (3,29 cm). La guiñada de 6° del fondo no está en la escena.

---

# Ampliación 2026-10-28 (tarde): colores del RGB simulado y FPS del camino GPU

## Colores del RGB simulado

`task2_render.py` (diccionario `RGB` y `RGB_LIGHT_SCALE`). Solo cambia lo visible:
- mesa: madera clara (0,62, 0,45, 0,28);
- vaso caliente: rojo suave;
- vaso templado: amarillo suave;
- vaso frío: azul suave;
- fondo: blanco mate (0,90).

**Luces.** Con solo el `displayColor` no basta: las luces de `nave_stage` (domo 600, distante 2500) están dimensionadas
para la nave del banco y saturan esta escena pequeña a un único crema. La mesa ya tenía color y salía blanca. Las
dos luces se multiplican por 0,3. La cámara térmica no las ve: lee ids de instancia, distancia y las temperaturas
prescritas.

**La térmica simulada no cambia: diferencia 0.** Re-render completo comparado con el anterior:
- `apparent_t_0..7` y `scene_apparent_t_0..7`: idénticos bit a bit (máx. |Δ| = 0,0);
- `apparent_t_ref_0..7`, rehechos por `task2_compare.py`: idénticos;
- `ids.npy`: cambia la numeración de las instancias entre sesiones de Kit, pero es una permutación uno a uno de
  las mismas 9 instancias, es decir, la misma geometría;
- los resultados de las tareas 2 a 4 salen iguales.

Hojas rehechas: `figures/cups3_sheet_{inferno,gray}.png`.

## FPS del camino GPU con la CellPlus

**Montaje.**
- `fps_run.py` ejecuta **sin cambios** `bench/render_cost/code/rebase_2026-10-22/ref/run_ref.py` (ver su
  `NOTES.md`), con tres sustituciones:
  - la escena de tres vasos (`fps_scene.py`, sacada de `out/sim/report.json` y `scene.yaml`);
  - `cellplus_ctc006_lwir_cups.yaml` con `supersample_factor: k`;
  - la LUT de esta banda.
- Headless, térmica + RGB, RealTimePathTracing.
- Trayectoria de 240 poses: el frame 0 es la pose de la tarea 2 y después hay un vaivén lateral de ±1 cm, para
  que la cámara nunca esté quieta.
- `fps_measure.sh`: dos sesiones de Kit por factor, intercaladas (1, 2, 4, 1, 2, 4). Cada una con 60 iteraciones
  de calentamiento, 2 pasadas cronometradas de 240 frames y una pasada de perfil (sincronización tras cada bloque).
- `fps_summary.py`: tabla e imágenes.

**Antes de medir:**
- Sin procesos de Isaac, pytest ni cómputo en la GPU (4 %). Había una sesión de Claude registrada
  (`airix-ros2-workshop`), pero inactiva y sin procesos.
- Escritorio (navegador, VS Code), carga media 0,4–4 durante las medidas.
- `powerprofilesctl set performance`: EPP `performance`; el governor sigue diciendo `powersave`, como es normal con
  `intel_pstate`.

**Desenfoque por capas: NO existe en el camino GPU.**
- La etapa 3 de `full/fullgpu.py` convoluciona con `PipelineConfig.psf`, el núcleo enfocado (difracción + gaussiana
  de aberración de 80 µm).
- El banco de desenfoque (`defocus_model: hopkins`, `defocus_apply: layered`, y también `global`) solo lo usa el
  camino host: `irsim.pipeline.frame` → `irsim.optics.layered`.
- **Medido sin desenfoque y sin sustituirlo por nada.**
- Por eso las corridas van con `--no-check`: la comprobación contra `run_frame` difiere justo en el desenfoque.

**Fallo encontrado en `src/`.**
- `irsim_isaac.pipeline.warp_stages.launch_optics` pasa 9 argumentos a `_optics_kernel`, que desde SC.34 (7efb966d)
  recibe 8: sobra `terms.lb_housing`. La etapa 3 Warp del repositorio falla al lanzarse.
- No hay test sin GPU que la ejercite.
- Aquí se midió con el kernel fusionado del propio banco (`--optics fused`, el de `measure3.sh`, verificado contra
  `run_frame` el 23-10). No está arreglado en `src/`.

**Resultados.** 240 × 2 pasadas × 2 sesiones por factor:
- en todos los frames, desfase de lectura 1;
- 0 lecturas anómalas y 0 frames perdidos;
- la reducción del RGB en el dispositivo es exactamente la media en caja del host (k = 2 y 4).

| supersample | render | FPS medio | mín | máx | ms/frame (p95) | render | lectura | procesado |
|---|---|---|---|---|---|---|---|---|
| 1 | 640×480 | **53,3** | 37,1 | 59,0 | 18,8 (20,1) | 10,5 | 2,6 | 5,7 |
| 2 | 1280×960 | **34,4** | 28,2 | 36,4 | 29,1 (30,5) | 15,7 | 3,2 | 10,3 |
| 4 | 2560×1920 | **11,9** | 7,0 | 12,2 | 84,2 (87,2) | 37,0 | 5,5 | 41,6 |

**Cómo se agrupan los bloques:**
- **render** = escribir la pose + `orchestrator.step`;
- **lectura** = anotadores + sello + descarga al host;
- **procesado** = tablas + etapas 1–2 + etapas 3–6/ADC/M9 + `synchronize`.

**Lo que no hay que leer de más en el desglose.** El render corre en la misma GPU y de forma asíncrona, así que la
espera al render cae en el primer bloque que necesita su salida. El total por frame sí es fiable. En la pasada de
perfil (sincronizando tras cada bloque):
- a k = 4, «etapas 1–2» sube a 26 ms y «etapas 3–6» baja a 15 ms;
- a k = 2, 4,7 y 6,2 ms;
- a k = 1, 0,7 y 4,0 ms.

**Valores por sesión** (estables, < 1 % entre sesiones):
- k = 1: 53,7 / 52,5 / 54,3 / 52,7 fps;
- k = 2: 34,4 / 34,2 / 34,5 / 34,4 fps;
- k = 4: 11,9 / 11,9 / 11,9 / 11,9 fps.

Los mínimos de 37 y 7 fps son frames sueltos; el p05 está en 51 / 33 / 11,5 fps.

**Efecto en los bordes** (`figures/cups3_fps_supersample.png`). Misma pose (frame 0), códigos de 8 bits frente a
k = 4:

| | térmica: media en el frame / ventana del borde (máx) | RGB: media en el frame / ventana del borde (máx) |
|---|---|---|
| k = 1 | 0,025 / 0,43 (6) | 0,72 / 1,64 (31) |
| k = 2 | 0,007 / 0,08 (2) | 0,43 / 0,93 (10) |

- **En la térmica el supersample casi no se ve.** La PSF enfocada (σ ≈ 3,3 px) ya difumina más que el escalonado
  de k = 1, y k = 2 queda a < 0,1 código de k = 4.
- **En el RGB sí:** k = 1 deja escalones en el borde inclinado del vaso, y k = 2 los quita casi todos.
- **Para esta cámara, k = 2 da casi la calidad de k = 4 a 34 fps.** Por encima de los 30 Hz de la CTC-006, pero con
  poco margen. k = 1 llega a 53 fps.
- Ninguno incluye desenfoque. Con él, que solo existe en el host, la térmica se difuminaría aún más; la
  sensibilidad al supersample quedaría todavía menor.

Archivos: `out/fps/FPS.md`, `out/fps/fps_summary.json`, `out/fps/{A,B}_k{1,2,4}/report.json` y sus logs,
`out/fps/frames_k*_{thermal,rgb}.png`, `out/fps/measure.log` (estado de la máquina antes de cada sesión).

## Vista exterior del montaje simulado

`figures/cups3_world.png`: el equivalente de `bench/cups_check/presentation/world_cold_water_95.png` para los
tres vasos.
- `render_world.py` (Kit) construye la escena de `task2_render.py`, con la pose, el alabeo y los colores de la hoja
  RGB. Coloca la cámara RGB con la calibración RGB-térmica, pone cajas en los dos centros ópticos (marcadores, no
  cuerpos modelados) y renderiza una cámara exterior diagonal y elevada.
- `make_world.py` dibuja con las funciones de `make_presentation.py` el campo de visión real (con distorsión) de
  cada cámara, en el plano de los vasos (0,51 m), y etiqueta cada vaso con sus lecturas de sonda.

---

# Ampliación 2026-10-29: ruido a `denoise: 80` y desfase de carcasa

El YAML nuevo es **`bench/camera_check/cellplus_ctc006_lwir_cups3.yaml`**: la copia de `cellplus_ctc006_lwir_cups.yaml`
con tres cambios, marcados `[DATA cups3]`. El anterior no se ha tocado. Desde la tarde del 29-oct. es el YAML por defecto de
las tareas 2–5 (sección siguiente).

| campo | antes (`..._cups.yaml`) | **ahora (`..._cups3.yaml`)** | de dónde |
|---|---|---|---|
| `noise.ratios_3d` vh / v / h | 1,027 / 0,413 / 0,201 | **1,058 / 0,523 / 0,228** | tarea 6 |
| `noise.ratios_3d` tv / th / t | 0,215 / 0,280 / 0,073 | **0,201 / 0,280 / 0,082** | tarea 6 |
| `calibration.radiometric.netd_k` | 0,0263 K (denoise 40) | **0,0262 K** (denoise 80) | tarea 6 + SITF de la tarea 1 |
| `calibration.radiometric.sitf_dn_per_k` | 52,91 (`bench/cups_check`) | **48,25** (tarea 1 de aquí) | tarea 1 |
| `optics.housing_temp_k` | (sin poner → calibración a 300 K) | **301,05 K** | tarea 7 |
| `extensions.capture_denoise` | 40 | **80** | `thermal_info.json` |

## Tarea 6: ruido 3D a `denoise: 80` (`task6_noise.py`)

**Datos y método.** Los mismos estimadores que `bench/cups_check/task4_noise.py` (importados de allí), para que los
números se comparen uno a uno con los de denoise 40. Cambian tres cosas, por cómo es esta grabación:
- **92 frames distintos**: se quitan los dos repetidos, porque un par idéntico daría ruido 0;
- **región**: filas 10–150 × columnas 160–420, el fondo por encima de los vasos. Fuera quedan:
  - el penacho del vaso caliente (columnas ≥ 440, donde la σ temporal sube a 4–7 cuentas);
  - el borde izquierdo, donde ya se ve la sala tras el fondo;
  - al menos 50 filas alrededor de cada borde de vaso;
- **sin FFC** en los 29 s (ningún salto de nivel > 8 cuentas; el mayor es 2,4), así que hay un solo patrón fijo.

**Dos correcciones al método de denoise 40**, porque esta grabación es continua y aquella eran sets sueltos:
1. **El ruido temporal sale de pares de 1,5 a 5 s.** La σ de un par sigue creciendo con la separación:

   | separación | 0–0,2 s | 0,2–0,5 | 0,5–1,5 | 1,5–3 | 3–5 | 5–8 | 8–12 | 12–18 | 18–24 | 24–30 s |
   |---|---|---|---|---|---|---|---|---|---|---|
   | σ_tvh del par (cuentas) | 0,95 | 1,12 | 1,21 | 1,25 | 1,27 | 1,30 | 1,33 | 1,36 | 1,40 | 1,41 |

   - Hasta ~1,5 s es la memoria del denoise.
   - Después, la σ crece igual al principio, en medio y al final de la grabación (1,27 / 1,25 / 1,29 en tramos de
     10 s). Por tanto no es que el ruido cambie: **es el patrón fijo que deriva**. El simulador lo modela aparte
     (`fpn_drift_tau_s`).
   - Con todos los pares ≥ 1,5 s, como en denoise 40, sale 1,341: es el valor comparable uno a uno.
2. **El residuo temporal de la imagen media se mide**, como la mitad de la diferencia entre las medias de las dos
   mitades de la grabación. La regla de denoise 40 («grupos separados > 1,5 s») da aquí 3 grupos y restaría unas
   6 veces de más: con ella, vh saldría 1,13 en lugar de 1,34.

**Resultado** (cuentas):

| | tvh | tv | th | t | vh | v | h | NETD |
|---|---|---|---|---|---|---|---|---|
| denoise 40 (`bench/cups_check`) | 1,391 | 0,299 | 0,390 | 0,101 | 1,429 | 0,575 | 0,280 | 26,3 mK |
| **denoise 80 (aquí)** | **1,263** | 0,254 | 0,354 | 0,103 | **1,337** | 0,661 | 0,288 | **26,2 mK** |
| cociente | 0,91 (0,96 con el estimador de denoise 40) | 0,85 | 0,91 | 1,02 | 0,94 | 1,15 | 1,03 | |

- **Pasar de denoise 40 a 80 apenas cambia el ruido de salida.** El temporal baja un 4–9 %, y el patrón fijo queda
  igual dentro de un ±15 %.
- **La memoria y la correlación espacial son las mismas.** Correlación entre frames: r = 0,44 por debajo de 0,2 s y
  0,22 entre 0,2 y 0,5 s (antes 0,485 y 0,28). Correlación con el vecino: 0,38 en horizontal y 0,30 en vertical
  (antes 0,374 y 0,297). El filtro no se nota más fuerte.
- **Las colas siguen igual de pesadas.** Más allá de 5 σ hay 3,2·10⁻⁴ de las muestras (una gaussiana daría
  5,7·10⁻⁷); el exceso de curtosis es 2,0. El 0,44 % de los píxeles tiene más del doble de la σ temporal mediana.
- **NETD 26,2 mK** con el SITF de esta calibración (48,25 cuentas/K). Con el SITF de la sesión anterior (52,91) serían
  23,9 mK: la mayor parte de la diferencia de SITF es de calibración, no de ruido.
- **Deriva del patrón.** La varianza de un par crece 0,0176 cuentas²/s hasta 30 s, y el patrón de las dos mitades
  (separadas ~14 s) se correlaciona r = 0,887. Si derivase todo el vh, sería τ ≈ 100 s, compatible con los 120 s
  del YAML. En 29 s no se puede fijar, así que `fpn_drift_tau_s` se queda como estaba.
- **Control en el simulador** (`sim_control` en `task6_noise.py`, sin Kit, escena uniforme a 23,9 °C):
  - con el YAML nuevo, la cámara simulada da σ temporal de 1,252 cuentas, frente a 1,263 en la real (−1 %);
  - con el anterior daba 1,259, frente a 1,391 a denoise 40 (−9 %).
  - El simulador convierte el NETD a cuentas con su propia ganancia (~47,8 cuentas/K), que se parece más al SITF de
    los tres vasos.

**No representable** (como en denoise 40, hueco del esquema): la memoria en el tiempo, la correlación espacial y las
colas. El `tvh` simulado es blanco en el tiempo y en el espacio.

**Datos con tapa.** `thermal_specs/three_cups_experiments/covered_thermal/` es una grabación con la cámara tapada:
354 s a 30 Hz, `denoise: 80`, sin escena. No se ha usado, porque esta tarea pedía solo la grabación sincronizada. Con
ella se separarían el patrón fijo y la textura del fondo, y se medirían la deriva y las FFC.

## Tarea 7: el desfase de carcasa (`task7_housing.py`)

**Lo primero que se pidió: ¿traen los datos temperaturas internas de la cámara?** No:
- `sync_cups/thermal/thermal_info.json` solo tiene puerto, firmware, `t_list`, `denoise` y formato;
- su `thermal_sync_events.csv` solo trae 19 avisos de frames perdidos.
- La cabecera de cada frame lleva temperatura de placa y de obturador (la grabadora la lee para validar el frame),
  pero no la guarda.
- Los únicos valores «board / shutter» de los logs (en `lab`, `cups` y `covered_thermal`) salen de cabeceras
  desalineadas y son cuentas de píxel / 100. La prueba está en `bench/data_check/RESULTADOS.md`, §3.

**No hay telemetría con la que corregir los parámetros de la Boson**, y no se ha inventado ninguna.

**De dónde sale el +0,28 K.** Escena uniforme de ε = 1 a 23,9 °C, a 1 m, sin ruido, por la misma cadena de cámara
que `task2_render.py`. Una cámara perfecta leería 0 K. Cada fila cambia una sola cosa:

| variante | T carcasa | T calibración | desfase mediano | centro | esquina | cuenco (pico a pico) |
|---|---|---|---|---|---|---|
| YAML anterior (Boson) | 301,05 K | 300 K | **+0,425 K** | +0,096 | +1,228 | 1,226 K |
| sin autocalentamiento (carcasa = aire) | 297,05 | 300 | −1,182 | −0,265 | −3,463 | 3,469 |
| lente sin emisión (τ = 1) | 301,05 | 300 | +0,305 | +0,002 | +1,044 | 1,129 |
| calibración a la carcasa estable | 301,05 | 301,05 | **0,000** | 0,000 | 0,000 | 0,016 |
| sin autocalentamiento y calibración al aire | 297,05 | 297,05 | 0,000 | 0,000 | 0,000 | 0,000 |
| carcasa fija a 300 K | 300 | 300 | 0,000 | 0,000 | 0,000 | 0,011 |
| **YAML nuevo** (`housing_temp_k: 301.05`) | 301,05 | 301,05 | **0,000** | 0,000 | 0,000 | 0,016 |

- **El desfase no es ningún número de la Boson por sí solo.** Es el desacuerdo entre la temperatura a la que trabaja
  la carcasa y la que supone la inversión radiométrica:
  - en modo `coupled` la carcasa se asienta en T_aire + `housing_self_heating_k` = 23,9 + 4,0 °C = 301,05 K;
  - sin `housing_temp_k`, `PipelineConfig` calibra a 300 K;
  - esos 1,05 K dan +0,40 K de desfase mediano por kelvin de desacuerdo.
- **El mismo desacuerdo es el cuenco de viñeteo** (0,1–0,8 W m⁻² sr⁻¹, tarea 2): el término de carcasa crece hacia
  las esquinas más deprisa que el cos⁴ de la escena.
- Cuando las dos temperaturas coinciden, el desfase es 0 con cualquier autocalentamiento y cualquier transmitancia.
- **Por qué 301,05 K y no otro número.** La tarea 1 calibró la cámara real en su sitio, en esta sala y a la
  temperatura que tuviera su carcasa. `housing_temp_k: 301.05` le dice lo mismo a la simulada.
  - En modo `coupled` ese campo solo fija la temperatura de calibración (`PipelineConfig.from_sensor`); la carcasa
    sigue el nodo acoplado.
  - Está **ligado a los 23,9 °C de esta sala**. Otra sala necesita su valor, o mejor, las temperaturas de cabecera
    registradas.
  - `housing_self_heating_k`, `housing_tau_s` y `transmittance` siguen PENDIENTES (Boson). Con la calibración en su
    sitio, solo afectarían a la deriva, que en 29 s y en una sala isoterma es nula.

**Comprobación en Kit**: tareas 2a y 2b re-renderizadas con el YAML nuevo, en `out/sim_cups3yaml/` (no pisa
`out/sim/`). El YAML nuevo también lleva el ruido de la tarea 6, que se re-muestrea en cada corrida.

```bash
CUPS3_SENSOR_YAML=$PWD/bench/camera_check/cellplus_ctc006_lwir_cups3.yaml CUPS3_SIM_DIR=sim_cups3yaml \
  env -u LD_LIBRARY_PATH -u PYTHONPATH -u AMENT_PREFIX_PATH -u ROS_DISTRO -u ROS_VERSION -u ROS_PYTHON_VERSION \
  CUPS_MIN_DISTANCE_M=0.3 /home/jumasaet/isaac_sim/python.sh bench/cups3_check/task2_render.py
# y lo mismo con task2_compare.py
```

| | YAML anterior | **YAML nuevo** |
|---|---|---|
| fondo: cámara simulada − escena | **+0,282 K** | **−0,000 K** (−0,006 a +0,006 en los 8 instantes) |
| **error real − sim del templado** (independiente), referido al fondo | −0,428 K | **−0,417 K** |
| ídem, sin referir al fondo | −0,513 K | −0,418 K |
| ídem, primer / último instante | −0,326 / −0,491 | −0,316 / −0,481 |
| error del frío / del caliente (residuos de calibración) | +0,398 / +0,282 K | +0,388 / +0,278 K |
| cámara − escena en el templado (desenfoque) | −0,024 K | −0,034 K |

- **El desfase desaparece y el error del templado no empeora**: mejora 0,011 K, muy por debajo de su incertidumbre
  de ±0,5 K.
- **Referir al fondo deja de hacer falta.** Sin referir, el error del templado pasa de −0,513 a −0,418 K y coincide
  con el referido: ya no hay cuenco que quitar.
- El −0,42 K que queda es el de la prueba de la tarea 1 (−0,46 K en aparente), no de la cámara simulada.

## YAML nuevo por defecto (29-oct., tarde)

`SENSOR_YAML` en `cups3_common.py` apunta ahora a `cellplus_ctc006_lwir_cups3.yaml`, y se ha reejecutado `FROM=2`
(tareas 2–7; `out/sim/` y las hojas de `figures/` son las del YAML nuevo). La banda y la respuesta espectral son las
mismas, así que la LUT y las tareas 0–1 no cambian. El ruido se re-muestrea en cada corrida, así que las cifras de
tercera decimal se mueven algo entre corridas.

| | YAML anterior | **YAML nuevo (por defecto)** |
|---|---|---|
| error real − sim, frío (residuo de calibración) | +0,398 K | **+0,388 K** |
| **error real − sim, templado (independiente)** | −0,428 K | **−0,417 K** |
| error real − sim, caliente (residuo de calibración) | +0,282 K | **+0,278 K** |
| ídem sin referir al fondo, frío / templado / caliente | −0,050 / −0,513 / +0,102 K | **+0,386 / −0,418 / +0,278 K** |
| fondo: cámara simulada − escena | +0,282 K | **−0,000 K** |
| SSIM, frame completo / recorte (5–70 °C) | 0,9864 / 0,9702 | **0,9869 / 0,9704** |
| LPIPS, frame completo / recorte (5–70 °C) | 0,0574 / 0,1000 | **0,0569 / 0,0998** |
| SSIM / LPIPS recorte, ventana 20–40 °C | 0,873 / 0,242 | **0,876 / 0,239** |
| SSIM local del vaso caliente (par central) | 0,925 | 0,925 |
| ρ de la mesa en el simulado (tarea 4) | 0,00 | 0,00 |

- Coincide con la comprobación de la tarea 7 (mismas cifras de `sim_cups3yaml/`): el desfase desaparece, los errores
  por vaso se mueven ≤ 0,011 K y ya no hace falta referir al fondo.
- SSIM y LPIPS mejoran en la tercera o cuarta decimal; la mejora mayor es en la ventana 20–40 °C (fondo y mesa),
  donde se ve el cuenco de viñeteo que ya no está. Lo que falta hasta 1 sigue siendo el reflejo en la mesa, la sala y
  la textura del ruido.

---

# Ampliación 2026-10-29 (noche): tarea 8, el reflejo de los vasos en la mesa

Prototipo **en el banco**, sin tocar `src/` ni escribir ADR: se quería ver primero si funciona. Scripts:
`task8_gbuffer.py` (Kit, una vez), `refl_common.py`, `task8_reflection.py` (`step0`, `fit`, `final`, `ruler`, `figure`),
`task8_similarity.py` (conda `vision`, como la tarea 3). Salidas: `out/task8_reflection.{json,png}`,
`out/refl/{step0,raycast,raycast_ruler}/`.

## Cómo se hace sin Kit y sin tocar `src/`

- **Un solo G-buffer.** La cámara está quieta y las temperaturas son constantes, así que `task8_gbuffer.py` guarda
  una vez los planos que recibe `run_frame` (2560 × 1920, supersample 4) y la cámara. `refl_common.replay` repite la
  cadena de cámara de `task2_render.py` sobre ellos, en los mismos instantes: **reproduce los 8 frames de Kit bit a
  bit** (máx. |Δ| = 0,0 K), a ~5,6 s por frame.
- **Modelo de la mesa** (docs/physics-model.md §4.1–4.3): `L = ε L_B(T) + ρ L_env + ρ_s(θ) (L_espejo − L_env)`, con
  `ρ_s(θ) = ⟨R_F(θ; n) · exp(−(4π σ_h cos θ / λ)²)⟩_banda`: Fresnel de un dieléctrico liso (n = 1,5) por la parte
  coherente de una superficie de rugosidad σ_h (Davies; el criterio de Rayleigh que la §4.3 cita como σ ≪ λ/8, con
  su cos θ), promediado en la banda con R(λ)·∂B/∂T a 300 K. En esta sala isoterma la parte difusa y la emisión de la
  mesa valen L_B(T_aire), así que solo el término especular da contraste.
- **Inyección:** cada píxel de mesa recibe la **temperatura equivalente** que da esa radiancia en la etapa 1
  (exacta en radiancia de banda, por la inversa de la LUT). Todo lo que viene después (desenfoque, viñeteo, detector,
  ruido, ISP) es la cadena de siempre: el reflejo pasa por la lente simulada como en la cámara real.
- **Geometría verificada:** los rayos de cada píxel salen de la matriz de vista y de la calibración con distorsión.
  Puntos de mesa a 0,12 mm de z = 0 (mediana; p99 0,75 mm), puntos de los vasos a 0,2 mm de sus conos, profundidad del
  rayo primario a 0,4 mm.

**Hallazgo aparte: `DistanceToCameraSD` llega 50 mm corta.** Kit la mide desde el plano de recorte cercano de la
cámara (`nave_stage` pone `clippingRange` (0,05, 1e4)), no desde su centro: contra el plano del fondo falta 50,0 mm a
cualquier ángulo (50,00 en el eje, 50,79 a cos 0,8–0,9; constante, no 0,05/cos). `refl_common` los suma para la
geometría. La cadena de cámara del banco ha recibido siempre la distancia corta, lo que toca al desenfoque por capas
(a 0,5 m, W020 ∝ 1/s cambia ~11 %) y a la atmósfera (despreciable a 1 m). **No se ha corregido**: cambiaría todos los
renders anteriores y merece su propia comprobación.

## Paso 0: prototipo en espacio de imagen (solo para estimar)

La escena simulada volteada sobre la primera fila de mesa de cada columna, con σ_h = 5,3 µm (el ajuste sin lente de
la propuesta). No es la solución: solo sirve para estimar la ganancia.

| | sin reflejo | **paso 0** |
|---|---|---|
| **recorte del vaso caliente (par central): SSIM / LPIPS** | 0,925 / 0,159 | **0,952 / 0,095** |
| recorte de los vasos (8 pares): SSIM / LPIPS | 0,970 / 0,100 | 0,980 / 0,057 |
| ρ_near frío / templado / caliente (real 0,35 / 0,40 / 0,40) | 0 / 0 / 0 | 0,38 / 0,38 / 0,39 |

## Paso 2: σ_h ajustada pasando el render por la cámara simulada

Para cada σ_h de una rejilla de 3 a 7 µm (paso 0,25), el reflejo del trazado de rayos (paso 3) entra por la cadena ideal
(sin ruido) y se mide con la definición de la tarea 4. Se compara con el perfil real en d ≤ 54 / 64 / 45 filas. El
caliente se corta en 45 por el segundo lóbulo, y ese corte se fijó antes de ajustar nada. n = 1,5 fijo (no se puede
identificar a 80–85°).

| vaso de prueba | ajustado con | σ_h ajustada | rms del de prueba | su propio mejor σ_h (rms) | rms sin reflejo |
|---|---|---|---|---|---|
| frío | templado + caliente | 5,42 µm | **0,038** | 6,05 µm (0,013) | 0,263 |
| templado | frío + caliente | 5,73 µm | **0,023** | 5,40 µm (0,015) | 0,270 |
| caliente | frío + templado | 5,73 µm | **0,022** | 5,43 µm (0,014) | 0,326 |
| **los tres** | | **5,62 µm** | 0,023 | | |

- **La validación cruzada pasa**: cada vaso, predicho con σ_h ajustada en los otros dos, queda a 0,022–0,038 de su
  perfil. Sin reflejo el rms es 0,26–0,33. El rango de σ_h entre combinaciones es 5,4–5,7 µm.
- **El frío es el que peor encaja**: quiere 6,05 µm, porque su perfil real cae algo más deprisa. Su referencia de mesa
  tiene el mayor desacuerdo entre franjas (−20 cuentas, ±0,015 en ρ, tarea 4).
- **La lente se llevaba ~0,3 µm**: sin pasar por la cámara, el ajuste daba 5,1–5,5 µm, y a través de ella da 5,62.
- ρ_s con 5,62 µm: 0,002 a 70°, 0,016 a 75°, **0,10 a 80°, 0,22 a 82,5°, 0,43 a 85°**, 0,71 a 87,5°. La mesa solo es
  espejo a ángulos rasantes.

## Paso 3: rayo especular en NumPy contra las primitivas de la escena

Para cada píxel de mesa (2,26 M a supersample 4): punto en la mesa → rayo reflejado → primer impacto contra los conos
de los vasos (parte mojada y franja seca), el fondo y la pared trasera (cajas). La mesa se excluye, porque es el espejo.
`L_espejo` es la radiancia de etapa 1 del objeto alcanzado. Resultado: 1,56 M rayos al fondo, 0,44 M a la pared y
0,26 M a los vasos. El trazado tarda 1,6 s; la reconstrucción de rayos (6 s) y ρ_s (5,7 s) se calculan una vez para
una escena quieta. Se usa σ_h = 5,62 µm, y la cadena completa con ruido y semilla 0, como `task2_render`.

| reflejo, definición de la tarea 4 | real | sin reflejo | **trazado de rayos** |
|---|---|---|---|
| ρ_near frío / templado / caliente | 0,35 / 0,40 / 0,40 | 0 / 0 / 0 | **0,37 / 0,36 / 0,37** |
| ρ_mirror frío / templado / caliente | 0,23 / 0,23 / 0,29 | 0 / 0 / 0 | **0,27 / 0,22 / 0,27** |
| rms del perfil frente a lo real | | 0,26 / 0,27 / 0,33 | **0,027 / 0,019 / 0,017** |

| SSIM / LPIPS (ventana 5–70 °C salvo donde se dice) | sin reflejo | paso 0 | **trazado de rayos** |
|---|---|---|---|
| frame completo, 8 pares | 0,987 / 0,057 | 0,991 / 0,041 | **0,992 / 0,038** |
| recorte de los vasos, 8 pares | 0,970 / 0,100 | 0,980 / 0,057 | **0,984 / 0,048** |
| **recorte del vaso caliente**, par central | 0,925 / 0,159 | 0,952 / 0,095 | **0,965 / 0,079** |
| recorte del frío / templado | 0,976 / 0,063 · 0,990 / 0,049 | 0,989 / 0,024 · 0,994 / 0,028 | **0,991 / 0,016 · 0,995 / 0,021** |
| recorte de los vasos, ventana 20–40 °C | 0,876 / 0,239 | 0,927 / 0,129 | **0,940 / 0,107** |

- **Funciona.** El LPIPS del recorte baja a menos de la mitad (0,100 → 0,048) y el del vaso caliente a la mitad
  (0,159 → 0,079). Lo que la tarea 3 decía que faltaba era sobre todo esto.
- Las temperaturas de los vasos no cambian: el reflejo solo toca píxeles de mesa.
- **Lo que queda en el caliente es casi todo el segundo lóbulo** (`out/task8_reflection.png`, panel del residuo).

## El segundo lóbulo bajo el vaso caliente: acotado, no explicado

Lo que se ha podido establecer:
1. **Es un reflejo del vaso caliente.** En los 29 s, su contraste dividido por el del vaso se queda en
   0,285–0,287, y baja un 4,6 % cuando el vaso baja un 3,2 % (el reflejo cercano, −3,4 %). Una mancha caliente en la
   mesa (por ejemplo agua derramada) se enfriaría a su propio ritmo.
2. **No es la referencia de mesa.** Las dos franjas de la tarea 4 están planas (15 377 cuentas a izquierda y derecha,
   a todas las alturas): la regla es invisible en una sala isoterma.
3. **Está en una cuña concreta**, filas ~320–355 delante del caliente, a ~0,2 m de la cámara sobre la mesa. Tiene un
   borde inferior izquierdo recto, paralelo a la regla de acero que se ve en la RGB. Su brillo pide ρ ≈ 0,8–0,9, el de
   un metal, 4–5 veces el del barniz a ese ángulo.
4. **La regla, donde la pone la calibración, no lo reproduce.** La regla se segmenta en la RGB y se lleva a la
   térmica por el plano de la mesa, corregida en +11,6 / +1,3 px con los contactos de los vasos frío y templado (que
   cuadran a ±4 px). Tratada como espejo de acero (ρ 0,9, modo `ruler`), no cambia nada el perfil del caliente: en las
   filas del lóbulo no llega a sus columnas. **Borda la cuña, no la contiene.**
5. Una regla de acero **desplazada** unos 25 px perpendicularmente a sí misma sí explica la cuña: el rms del residuo
   en la zona baja de 4,17 a 2,56 W m⁻² sr⁻¹, con ρ efectivo de 0,79–0,91. Pero ese desplazamiento rígido sacaría su
   arranque de la base del templado, y la proyección está comprobada a ±4 px en los vasos.

**Conclusión:** lo más probable sigue teniendo que ver con la regla (borde recto paralelo, reflectancia de metal),
por ejemplo un bisel, una tira mojada o una zona más pulida a lo largo de ella, o la regla algo levantada. Pero con
estos datos no se puede demostrar. **Lo resolvería** grabar unos segundos con la regla retirada, o una foto cenital
del montaje.

## Limitaciones

- σ_h es un parámetro efectivo de esta mesa a 80–85°. n = 1,5 se supone, y la forma exp(−(4πσ cos θ/λ)²) es la de una
  superficie con alturas gaussianas, sin medir.
- **El reparto especular contradice el ADR 0067**: su w_s = (1 − r)² no depende del ángulo. Hará falta un ADR antes de
  llevar esto a `src/`.
- Primitivas analíticas: sirven para esta escena (conos y cajas). Para una escena USD general hace falta el trazado en
  Warp (`mesh_query_ray`).
- El paso 0 y el paso 3 usan la temperatura equivalente: es un truco del banco, no la interfaz que tendría `src/`.
