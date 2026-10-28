# Tres vasos (frío, templado, caliente): calibración con prueba independiente, real frente a simulado, similitud y reflejo

Fecha: 2026-10-28. Datos: `thermal_specs/three_cups_experiments/sync_cups/`. Todo está en este directorio, salvo un
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

`cups3_common.py` reúne lo compartido: lectura de los datos, horas, temperaturas por frame, regiones de los vasos,
superficie de referencia del fondo y radiancia de banda. La radiometría y la geometría LiDAR→térmica vienen de
`bench/cups_check/cups_common.py` (mismo rig, mismos ficheros de calibración). Los `.npy` no se versionan
(`bench/.gitignore`); se regeneran con los scripts.

---

## Datos

| | |
|---|---|
| térmica | 94 frames `int16` (480, 640), cuentas crudas de 14 bits, más alto = más caliente, `invert_raw: true` |
| RGB | 93 PNG 1280 × 960, ya con balance de blancos (falta la del set 000091) |
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
- **Temperatura del agua por frame:** lineal entre la lectura «antes» en el primer frame y la «después» en el
  último, como pediste. `T_s = T_agua` (factor de pared k = 1).

### Calibración (caliente + frío + fondo)

| | valor |
|---|---|
| **G·ε** | **53,23 cuentas por W m⁻² sr⁻¹** |
| solo el frío / solo el caliente | 52,01 / 53,54: **difieren un 2,9 %** (razón de Birge 0,93, con ±0,5 K por lectura) |
| G con ε = 0,93 | 57,24; SITF **47,9 cuentas/K a 300 K** (la sesión anterior: 52,9) |
| cociente de contrastes caliente/frío, medido | **3,393** |
| predicho, lineal en radiancia | 3,308 |
| predicho, lineal en kelvin | 2,545 |

- **Esta vez las tres lecturas sí son coherentes** entre sí, a diferencia de `bench/cups_check`, donde diferían un 23 %.
- El modelo en radiancia es claramente el bueno: el lineal en kelvin falla un 25 % en el cociente.
- El caliente pesa el 80 % en la media ponderada, porque su contraste de 42 K hace su ganancia menos sensible a la
  lectura.

### Prueba independiente: el vaso templado

| | calibrado | sonda | **error** |
|---|---|---|---|
| T de superficie (sin ε), media de 94 frames | 35,02 °C | 35,52 °C | **−0,50 K** |
| T aparente (ε = 0,93) frente a `L_B⁻¹(ε L_B(T_w) + (1−ε) L_B(T_aire))` | 34,28 °C | 34,74 °C | **−0,47 K** |
| error en el primer / último frame (superficie) | | | −0,96 / +0,16 K |

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
| **A: lineal en el tiempo (el supuesto)** | **−0,52 K [−1,12; +0,09] [−1,69; +0,69]** | −0,48 K [−0,99; +0,02] |
| B: cada vaso en cualquier punto entre sus dos lecturas, constante | −0,51 K [−1,35; +0,34] [−2,14; +1,18] | −0,47 K [−1,22; +0,28] |

El −0,5 K está a ~0,9 σ de cero. **La prueba pasa dentro de la incertidumbre.**

Desglose, un factor cada vez (desplazamiento del error del templado):

| factor | desplazamiento |
|---|---|
| lectura del templado ±0,5 K | ∓0,50 K |
| fondo frente al aire ±0,5 K | ±0,47 K |
| aire ±0,5 K | ±0,44 K |
| todas las lecturas en «antes» / en «después» | +0,24 / −0,22 K |
| lectura del caliente ±0,5 K | ±0,12 K |
| lectura del frío ±0,5 K | ∓0,06 K |
| ROI mitad superior / inferior | +0,04 / −0,06 K |
| factor de pared 0,85 | −0,03 K (un k común se absorbe en G·ε) |

Dominan la propia lectura del templado y la temperatura que se supone en el fondo y el aire.

### Lo que los datos dicen del supuesto lineal en el tiempo

**Los datos lo contradicen.** Cambio de cada vaso durante los 29 s, leído de la propia calibración (superficie, sin ε):

| vaso | medido | supuesto |
|---|---|---|
| frío | +0,29 K | +1,1 K |
| templado | −0,18 K | −1,3 K |
| caliente | −1,15 K | **−7,0 K** |

- Las lecturas se tomaron con minutos de separación, no en los 29 s de la grabación. Al ritmo medido después del
  transitorio inicial (frío +0,29, templado −0,30, caliente −2,3 K/min), el cambio entre las dos lecturas lleva
  **3,8 / 4,3 / 3,0 min**. Los tres vasos dan lo mismo: las lecturas se tomaron con **~3–4 min** de separación, y
  probablemente a la vez en los tres.
- Las medias sí cuadran (caliente 66,26 °C frente a 66,17 °C supuesto), porque la grabación cae hacia el centro del
  intervalo. Por eso **G·ε apenas se ve afectado** (escenario B).
- Lo que sí produce el supuesto es la deriva del error del templado durante la grabación (−0,96 → +0,16 K) y la del
  caliente en la tarea 2.
- En los primeros ~1,5 s el frío sube 0,35 K, el caliente baja 0,25 K y el templado baja 0,1 K; después se quedan
  casi quietos. No es un cambio común de ganancia: en proporción es 1,9 / 0,8 / 0,3 %. **Hipótesis (no
  comprobada):** el agua se acababa de remover con la sonda y la pared se relaja al pararse la convección.

---

## Tarea 2: real frente a simulado

**Escena** (`task2_render.py`):
- tres troncos de cono de papel (ε 0,93) con la geometría de la tarea 0;
- pared mojada hasta 4,83 cm a la temperatura de la sonda, en rampa lineal (el mismo supuesto);
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
| frío | no (calibra) | 7,30 °C | 8,92 °C | 8,58 °C | **+0,35 K** | +0,48 / −0,07 |
| **templado** | **sí** | 35,51 °C | 34,28 °C | 34,71 °C | **−0,43 K** | −0,87 / +0,17 |
| caliente | no (calibra) | 66,15 °C | 63,70 °C | 63,54 °C | **+0,16 K** | −2,23 / +3,25 |

- **El único error independiente es el del templado.** El frío y el caliente fijaron la calibración real con las
  mismas temperaturas que recibe el simulador, así que su error es el residuo de la calibración, no una prueba.
- La incertidumbre del templado es la de la tarea 1 en aparente: **[−0,99; +0,02] K al 68 %**.
- **El simulador no aporta error propio:**
  - su escena coincide con `L_B⁻¹(ε L_B(T_s) + (1−ε) L_B(23,9))` en ≤ 0,021 K;
  - su cámara, referida al fondo, cambia los vasos en +0,01 / −0,02 / −0,04 K: el desenfoque les quita un poco de
    contraste;
  - el error del templado (−0,43 K) es la prueba de la tarea 1 (−0,47 K) vista a través del simulador.
- Sin referir el simulado al fondo, los errores serían −0,10 / −0,51 / −0,02 K.
- **La deriva del caliente** (−2,2 → +3,3 K) es entera del supuesto de 7 K en 29 s. El simulador sigue la rampa
  supuesta y el vaso real baja 1,1 K.
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
| frame completo | **0,986** (0,986–0,987) | **0,058** (0,056–0,060) |
| recorte de los vasos | **0,970** (0,969–0,971) | **0,101** (0,097–0,103) |
| con la ventana 20–40 °C (fondo y mesa): completo / recorte | 0,916 / 0,872 | 0,195 / 0,243 |

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
| tal cual | 0,970 | 0,101 |
| registrado (mejor desplazamiento global en ±4 px: +1 fila) | 0,970 | 0,099 |
| desplazar 2 px / 4 px (rejilla en el par central) | −0,002 / −0,007 | +0,002 / +0,009 |
| referencia: real contra el frame real siguiente | 0,9997 | 0,0002 |
| referencia: real contra sí mismo desplazado 1 / 2 / 4 px | 0,998 / 0,995 / 0,986 | 0,003 / 0,007 / 0,018 |

Ventana alrededor de cada vaso y su reflejo (par central), SSIM / LPIPS tal cual → registrado localmente:

| vaso | desplazamiento local | SSIM | LPIPS |
|---|---|---|---|
| frío | (3, 0) px | 0,975 → 0,977 | 0,063 → 0,057 |
| templado | (2, −1) px | 0,990 → 0,991 | 0,050 → 0,048 |
| caliente | (0, −1) px | 0,925 → 0,925 | 0,159 → 0,160 |

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
  - Instante: frame 53, a 13,2 s.
- `figures/cups3_results_table.png`: las tablas solas.
- `out/task1_calibration.png`, `out/task2_comparison.png`, `out/task3_similarity.png`, `out/task4_reflection.png` y
  `out/scene_geometry.png`: las figuras de diagnóstico de cada tarea.

---

## Limitaciones

1. **Hora de las lecturas.** El supuesto lineal en 29 s contradice los datos (−1,15 K medido frente a −7 K supuesto
   en el caliente; las lecturas distan ~3–4 min). No mueve la calibración, porque las medias cuadran, pero sí las derivas por frame. Una sonda
   registrada con hora, o dejada dentro del agua, lo resolvería.
2. **Fondo y aire.** Que el fondo esté a 23,9 °C pesa ±0,47 K por cada 0,5 K en el resultado del templado. No hay
   cuerpo negro de referencia.
3. **ε del papel no medida.** No afecta a la temperatura de superficie, pero sí a la aparente.
4. **`denoise: 80`.** El ruido del YAML es de denoise 40. No se ha vuelto a medir; no hacía falta para estas tareas.
5. **El simulador no tiene** el reflejo en la mesa, la sala detrás del fondo, la columna de vapor ni la regla. Su
   cámara tiene un offset de carcasa (+0,28 K) y un cuenco de viñeteo con parámetros PENDIENTES (del Boson).
6. **Geometría.** El cabeceo de la cadena LiDAR→térmica falla ~2°, y se corrige con las siluetas. La altura de la
   cámara depende solo del LiDAR (3,29 cm). La guiñada de 6° del fondo no está en la escena.
