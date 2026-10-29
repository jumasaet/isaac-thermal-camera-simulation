# Guía del banco de validación (`bench/`)

Para quien quiere entender y repetir el trabajo de validación de octubre de 2026 sin conocer el código del
simulador. Explica cómo se genera un frame térmico, qué hay en cada carpeta, cómo se repite el experimento de los
tres vasos, cómo se cambia la escena y qué problemas conocidos hay.

Los detalles y los números de cada experimento están en el `RESULTADOS.md` de su carpeta. Esta guía no los repite.

---

## 1. Cómo se genera un frame térmico

Un **frame** es una imagen de la secuencia. Aquí, una imagen térmica de 640 × 480 píxeles.

Intervienen tres piezas, una detrás de otra:

```
 ┌──────────────────────┐     ┌─────────────────────────────┐     ┌──────────────────────────────┐
 │ 1. ISAAC SIM         │     │ 2. EL SIMULADOR (irsim)     │     │ 3. LA CÁMARA SIMULADA        │
 │    (motor 3D)        │     │    (la física)              │     │    (el aparato)              │
 │                      │     │                             │     │                              │
 │ escena 3D + cámara   │     │ por cada píxel:             │     │ lente: desenfoca, viñetea    │
 │        │             │     │  qué objeto → material      │     │ detector: radiancia → señal  │
 │        ▼             │ ──► │  y temperatura del objeto   │ ──► │ ruido: temporal y fijo       │
 │ por cada píxel:      │     │        │                    │     │ conversor: señal → cuentas   │
 │  - qué objeto se ve  │     │        ▼                    │     │ procesado (ISP): cuentas →   │
 │  - a qué distancia   │     │  radiancia que sale de ahí  │     │  temperatura aparente e      │
 │  - color RGB         │     │  (emitida + reflejada)      │     │  imagen de 8 bits            │
 └──────────────────────┘     └─────────────────────────────┘     └──────────────────────────────┘
        G-buffer                    radiancia de escena              apparent_t, dn16, display8
```

**1. Isaac Sim** es el motor 3D de NVIDIA. Aquí no calcula nada térmico: solo dibuja la escena desde la cámara. Por
cada píxel entrega el **G-buffer** (las «capas geométricas» de la imagen):
- qué objeto se ve en ese píxel (un número de identificación, el *instance id*);
- a qué distancia está;
- su color visible (la imagen RGB, que solo sirve para mirar y para la hoja de presentación).

Isaac corre dentro de **Kit**, su programa base. Por eso los scripts que lo usan arrancan con
`/home/jumasaet/isaac_sim/python.sh`, el Python que trae Isaac.

**2. El simulador** (`src/irsim/`, en Python puro) hace la física. Con el número de objeto de cada píxel busca el
material de ese objeto y su temperatura. La temperatura de cada objeto la fija la escena: en los vasos está
**prescrita**, es decir, se la damos nosotros con la lectura de la sonda; no se calcula. Con eso obtiene la
**radiancia**: la potencia infrarroja que sale de ese punto hacia la cámara, en W/(m²·sr). Tiene dos partes:
- **emitida**, que depende de la temperatura (ley de Planck) y de la **emisividad** ε, un número entre 0 y 1 que
  dice cuánto emite un material comparado con un emisor perfecto («cuerpo negro»). El papel tiene ε ≈ 0,93;
- **reflejada**: lo que el objeto refleja del entorno, en proporción 1 − ε.

Solo cuenta la radiancia de la **banda** de la cámara (8–14 µm, el infrarrojo térmico, LWIR). El paso de temperatura a
radiancia en esa banda se precalcula en una tabla, la **LUT** (*look-up table*).

**3. La cámara simulada** (`irsim.pipeline.frame.run_frame`) convierte esa radiancia en lo que daría una cámara real.
Su descripción completa está en un archivo YAML (ver §4):
- **lente:** desenfoca la imagen (la **PSF**, la mancha en que se convierte un punto) y oscurece las esquinas
  (**viñeteo**). La propia lente y la carcasa, que están a su temperatura, añaden algo de radiancia;
- **detector:** un **microbolómetro**, una matriz de diminutos termómetros. Convierte la radiancia en una señal
  eléctrica;
- **ruido:** **temporal** (cambia en cada frame) y **de patrón fijo** (FPN: cada píxel tiene su pequeño desvío
  propio). También hay píxeles defectuosos y la **FFC** (*flat field correction*): la cámara cierra un obturador de
  vez en cuando para recalibrarse, y eso produce saltos;
- **conversor:** pasa la señal a números enteros, las **cuentas** (`dn16`). Son lo que guardan los `.npy` de la
  cámara real;
- **ISP** (el procesado interno): de las cuentas saca la **temperatura aparente** (`apparent_t`) y una imagen de
  8 bits para ver (`display8`).

**Temperatura aparente** es la temperatura que tendría un cuerpo negro (ε = 1) que diese la misma radiancia.
Un vaso de papel a 35 °C se lee algo más frío de lo que está, porque emite solo el 93 %.

Los scripts guardan también `scene_apparent_t`: la temperatura aparente **antes** de la cámara, sin desenfoque ni
ruido. Sirve para separar los errores de la física de la escena de los errores del modelo de cámara.

Para el GPU hay un segundo camino más rápido (`bench/render_cost/`), que sirve para medir velocidad (fps). El
experimento de los vasos usa el camino de CPU de arriba, que es la referencia.

---

## 2. Mapa de carpetas

### Qué es nuestro y qué es del repo original

El repo original es `reza-shahriari/isaac-thermal-camera-simulation` (remoto `origin`). Nuestra copia es
`jumasaet/...` (remoto `fork`), y trabajamos en la rama `wip/validation-bench`.

| | de quién | qué es |
|---|---|---|
| `src/`, `tests/`, `configs/`, `docs/`, `scripts/`, `data/`, `site/` … | **repo original** | el simulador. En esta rama solo lo tocan 3 commits nuestros (6 archivos de `src/` y sus tests): dos de rendimiento de la atmósfera y uno que añade `scene_apparent_t` |
| **`bench/`** | **nuestro** | todo el trabajo de validación con la cámara real |
| **`thermal_specs/`** | **nuestro** (datos) | las grabaciones de la cámara real, LiDAR y RGB. Está fuera de git (18 GB) salvo unos pocos archivos pequeños |

Para ver exactamente qué hemos añadido: `git diff --stat origin/main...HEAD`.

### `bench/` (por orden de fecha)

| carpeta | experimento | qué contiene |
|---|---|---|
| `camera_check/` | 23-oct. Primer vistazo a los datos de la cámara Cellplus (CTC-007) | Comprobaciones: ¿son cuentas crudas? (sí), ¿hay AGC? (no), saltos de FFC, ruido, campo de visión. **Aquí están los YAML de la cámara**: `cellplus_ctc007_lwir.yaml` (borrador), `cellplus_ctc007_lwir_cups.yaml` (calibrado con los dos vasos) y **`cellplus_ctc006_lwir_cups.yaml`** (el que usan los tres vasos), además de la respuesta espectral estimada `cellplus_ctc007_response_ESTIMATED.csv`. **`cellplus_ctc006_lwir_cups3.yaml`** (29-oct.) es la propuesta nueva: ruido a `denoise: 80` y sin desfase de carcasa (tareas 6 y 7 de `cups3_check/`) |
| `camera_preview/` | 23-oct. Primera vez que el simulador corre con la cámara Cellplus | **`common.py` es el camino de render que usan todos los experimentos de vasos** (arranca Kit, lee el G-buffer y llama a `run_frame`). `lut/` guarda la LUT de la banda (fuera de git, se regenera). Lo demás es la comparación de la trayectoria de referencia y la plantilla de dos vasos |
| `cups_check/` | 26-oct. Dos vasos (agua fría y caliente, 4 casos; datos `thermal_specs/cali_board_new/`) | Calibración, geometría, comparación real-simulado y ruido, más la medida del desenfoque de la lente, que dio el YAML CTC-006. **Los tres vasos reutilizan** de aquí `cups_common.py` (radiometría, LiDAR), `task2_geometry.py` (siluetas) y `materials/paper_cup.yaml` (el material del papel). `presentation/` son las figuras de la presentación semanal |
| `cups3_check/` | 28-oct. **Tres vasos** (frío, templado y caliente; datos `thermal_specs/three_cups_experiments/sync_cups/`) | El experimento de la §3. `run_all.sh` lo corre entero. También están la medida de fps del camino GPU (`fps_*`), la vista exterior del montaje (`make_world.py`, `render_world.py`) y, del 29-oct., el ruido a `denoise: 80` (`task6_noise.py`) y el desfase de carcasa (`task7_housing.py`) |
| `data_check/` | 29-oct. Revisión de `thermal_specs/` | `inventory.py` (solo lee) y `RESULTADOS.md`: cuántos archivos hay de cada modalidad, frecuencia real, archivos dañados o repetidos, por qué hay 96 / 94 / 93, y qué se podría borrar (propuesta, nada borrado) |
| `render_cost/` | 8 al 23-oct. Cuánto cuesta renderizar (fps) | Código copiado de sesiones antiguas. **`code/rebase_2026-10-22/` es la versión viva**: `camera_preview/common.py` usa de ahí `demo/nave_stage.py` (construye la escena en Isaac), `lite/lite.py` y `env_isothermal.yaml`. Las demás subcarpetas de `code/` son copias del 11-oct, que solo valen como historia; `code/lite_prototype/` es el primer kernel GPU, descartado |

Cada carpeta tiene un `RESULTADOS.md` o un `README.md` con lo que se hizo y lo que salió. Dentro, `out/` son
salidas (JSON, PNG y logs), `figures/` son las figuras finales y `logs/` los registros de cada corrida. Los arrays
`.npy`/`.npz`, las LUT y los vídeos no están en git (`bench/.gitignore`): se regeneran. Los frames y vídeos en bruto
de las medidas de coste de render del 11-oct (7,6 GB) y las secuencias de la trayectoria de referencia están
archivados fuera del repo, en `~/archivo_irsim/outputs_prev/` y `~/archivo_irsim/outputs/`.

**Cuidado al borrar:** los tres vasos dependen de `camera_check/`, `camera_preview/`, `cups_check/` y
`render_cost/code/rebase_2026-10-22/`. Ninguna de esas carpetas es prescindible.

### `thermal_specs/` (los datos reales)

| carpeta | qué es | la usa |
|---|---|---|
| `thermal_camera_specs.md` | hoja de datos de la cámara, en español | referencia |
| `lab/` | 1146 frames del laboratorio (cuentas en `thermal/*.npy`, colores del software de captura en `thermal_color/`) | `camera_check/` |
| `cali_board/` | 27 frames de la placa de calibración | `camera_check/` |
| `cali_board_new/` | los 4 casos de dos vasos, cada uno con `thermal/`, `rgb/` y `lidar/`, y las matrices de calibración entre cámaras (`*_matrix.txt`) | `cups_check/`, `cups3_check/` (las matrices) |
| `three_cups_experiments/sync_cups/` | **los tres vasos**: 96 sets sincronizados con 96 nubes LiDAR, 94 frames térmicos y 93 RGB, más `thermal_info.json` (ajustes de la cámara). Solo **92 frames térmicos son distintos**: dos sets repiten el frame de cámara del anterior (80/81, 88/89), y en otros tres se perdió al escribir la térmica o la RGB, no al sincronizar (`data_check/RESULTADOS.md`) | `cups3_check/` |
| `three_cups_experiments/cups/` | **otra grabación**, empezada 3,5 min después de `sync_cups`, **no su versión sin sincronizar**: 10 min de térmica, 30 s de LiDAR y RGB en PNG y luego en crudo `.pgm` (11 GB). Las tres modalidades solo coinciden 4,4 s | todavía nada |
| `three_cups_experiments/covered_thermal/` | la cámara **tapada**: 354 s a 30 Hz, `denoise: 80` (6,4 GB) | todavía nada |

Los frames térmicos son `int16` de 480 × 640 en **cuentas crudas de 14 bits**: un número más alto es más caliente.
No son imágenes, así que no se abren con un visor. Los nombres son `<índice>_<hora unix>.npy`.

---

## 3. El experimento de los tres vasos, de principio a fin

**Qué se hace.** Tres vasos de papel con agua fría (~7 °C), templada (~35 °C) y caliente (~66 °C) a 50 cm de la
cámara. Con el frío, el caliente y el fondo se **calibra** la cámara real (cuentas → temperatura). El templado
queda fuera de la calibración y sirve de **prueba independiente**. Después se reproduce la escena en el simulador y se
comparan las dos imágenes.

### Un solo comando

Desde la raíz del repo:

```bash
bash bench/cups3_check/run_all.sh
```

Tarda unos 3 minutos; casi todo es el render de la tarea 2. Va diciendo qué tarea corre y cuánto tardó. Si algo
falla, se para y enseña el final del log.

**La primera vez**, el script instala `lpips` y `scikit-image` en `~/.cache/irsim_pylibs` (necesita internet una
vez; el entorno conda `vision` no se modifica). También genera la LUT si falta.

Para repetir solo una parte: `FROM=3 bash bench/cups3_check/run_all.sh` empieza en la tarea 3. Las salidas de las
anteriores tienen que existir ya.

### Qué hace cada paso

Todas las salidas quedan en `bench/cups3_check/out/`, y los logs en `out/logs/<tarea>.log`.

| orden | script | qué hace | produce |
|---|---|---|---|
| — | (`generate_luts.py`) | crea la LUT de la banda si falta | `bench/camera_preview/lut/` |
| 0 | `task0_scene.py` | Mide la escena: dónde están los vasos (LiDAR), la pose de la cámara (ajustada a las siluetas térmicas) y el nivel del agua | `scene_geometry.json` (lo lee el render), `scene_geometry.png` |
| 1 | `task1_calibration.py` | Calibra la cámara real con el frío, el caliente y el fondo, y comprueba el templado. Incluye la incertidumbre (Monte Carlo) | `task1_calibration.{json,png}`, `task1_per_frame.npz`, `real_apparent_t_*.npy` (las imágenes reales calibradas) |
| 2a | `task2_render.py` | Construye la escena en Isaac y renderiza 8 instantes con la cámara simulada (abre Kit, ~2 min) | `sim/`: `apparent_t_<i>.npy`, `frame_<i>.png`, `frame_rgb.png`, `scene.yaml`, `report.json` |
| 2b | `task2_compare.py` | Temperatura de cada vaso, real frente a simulado, con la misma definición en los dos lados | `task2_comparison.{json,png}` |
| 3 | `task3_similarity.py` | Parecido de imagen: SSIM y LPIPS (abajo), orden de brillo y efecto de la desalineación | `task3_similarity.{json,png}` |
| 4 | `task4_reflection.py` | Mide cuánto refleja la mesa a los vasos (solo en lo real; el simulador no lo modela) | `task4_reflection.{json,png}` |
| 5 | `task5_figures.py` | Hojas de presentación, en inglés | `../figures/cups3_sheet_inferno.png`, `cups3_sheet_gray.png`, `cups3_results_table.png` |
| 6 | `task6_noise.py` | Ruido 3D de la cámara real a `denoise: 80`, con los estimadores de `cups_check` | `task6_noise.{json,png}` |
| 7 | `task7_housing.py` | De dónde sale el desfase de carcasa de la cámara simulada (sin Kit). La comprobación en Kit con el YAML nuevo es aparte (`RESULTADOS.md`, tarea 7) | `task7_housing.json` |

**SSIM** (índice de similitud estructural) compara dos imágenes por brillo, contraste y estructura: 1 significa
idénticas. **LPIPS** es una distancia «perceptual», calculada con una red neuronal: 0 significa idénticas, y valores
pequeños quieren decir que una persona apenas notaría la diferencia.

**Las temperaturas de los vasos: las lecturas no tienen hora.** De cada vaso hay dos lecturas de la sonda, una
antes y otra después de grabar (caliente 69,3 → 62,3 °C, templado 36,1 → 34,8, frío 6,8 → 7,9). **No se anotó a qué hora
se tomó ninguna.** La primera versión las trató como si fueran el primer y el último frame y unió ambas con una recta,
pero los datos lo desmienten: el caliente bajó **1,15 K** en los 29 s de grabación, no 7 K. Las lecturas distan unos
3–4 min entre sí, y la grabación cae en algún punto del medio. Como no se sabe dónde, cada vaso se pone a **temperatura
constante, igual a la media de sus dos lecturas**: caliente 65,8 °C, templado 35,45 °C, frío 7,35 °C (`T_MEAN_C` en
`cups3_common.py`). Se usa igual en la calibración (tarea 1) y en el render (tarea 2), para que el simulador reciba lo
mismo que la calibración real. La media es el punto central del intervalo, el que menos se equivoca si no hay otro
dato, y la calibración confirma que la grabación cae cerca de él (ver `RESULTADOS.md`). Consecuencia: la bajada real del
caliente (1,1 K) se ve como una deriva suave del error real − sim; no es un fallo del simulador. Una sonda con hora, o
dejada dentro del agua, lo resolvería.

**Qué mirar primero:**
1. `figures/cups3_sheet_inferno.png`: lo real y lo simulado, lado a lado, con la tabla de resultados.
2. `out/task2_comparison.json` → `cups.warm.error_real_minus_sim_k_mean`: el error del vaso independiente. Hoy
   sale −0,43 K.
3. `RESULTADOS.md`: la explicación de cada número.

**Fuera de `run_all.sh`** (son aparte y más lentos):
- `fps_measure.sh` → `fps_summary.py`: la velocidad del camino GPU, ~8 minutos, con la máquina en reposo;
- `render_world.py` (Kit, ~1 min) → `make_world.py`: la vista exterior del montaje con el campo de visión de las dos
  cámaras, en `figures/cups3_world*.png`.

---

## 4. Cómo cambiar la escena

La escena de los tres vasos no se escribe a mano: **sale de las medidas**. La tarea 0 mide posiciones y pose y las
deja en `out/scene_geometry.json`, y la tarea 2 (`task2_render.py`) construye con eso la escena en Isaac. Para cambiar
algo hay que tocar uno de estos sitios y volver a correr desde la tarea que lo lee.

| quiero cambiar | dónde | después correr |
|---|---|---|
| **temperaturas** del agua (lecturas antes y después; se usa su media, `T_MEAN_C`) y del aire | `cups3_check/cups3_common.py`: `T_PROBE_C` y `T_AIR_C` | `FROM=1` (cambian la calibración y el render) |
| tamaño del vaso | `cups3_common.py`: `CUP_TOP_DIAMETER_M`, `CUP_BOTTOM_DIAMETER_M`, `CUP_HEIGHT_M` | `FROM=0` |
| **objetos**: añadir o quitar, tamaño y posición de fondo, mesa y pared | `task2_render.py`, función `objects()`. Cada objeto es un `dict` con `name`, `kind` (`box`, `frustum` = tronco de cono…), `centre` (x, y, z en metros), `size` o `z`, y su color `rgb` | `FROM=2` |
| **material** térmico de cada objeto | `task2_render.py`, `objects()`, diccionario `thermal`: `"Table": ("plywood", None)`. Los materiales están en `configs/materials/`; el papel, en `bench/cups_check/materials/paper_cup.yaml` (ahí está su emisividad) | `FROM=2` |
| **distancias** de los vasos | Vienen del LiDAR (`scene_geometry.json` → `lidar.cups.<vaso>.axis_forward_m`). Para forzar otra, cambia `centre` en `objects()`. En la escena, −X es «hacia delante» desde la cámara y +Z es arriba | `FROM=2` |
| **pose de la cámara** (altura, cabeceo, alabeo) | `scene_geometry.json` → `image_pose`, que lee `main()` de `task2_render.py` (variables `pos` y `look`). Si editas el JSON a mano, la próxima tarea 0 lo sobrescribe | `FROM=2` |
| **la cámara** en sí (lente, resolución, ruido, frecuencia, paleta…) | el YAML `bench/camera_check/cellplus_ctc006_lwir_cups.yaml` (para probar otro sin tocar `cups3_common.py`: `CUPS3_SENSOR_YAML=<yaml> CUPS3_SIM_DIR=<carpeta>` en las tareas 2a/2b, que escriben en `out/<carpeta>/` y `task2_comparison_<carpeta>.*`). Los valores con `[DS]` vienen de la hoja de datos, los `[DATA]` están medidos y los `PENDIENTE` están copiados de otra cámara (Boson) a falta de medirlos. Si cambias la banda o la respuesta espectral, hay que rehacer la LUT: borra `bench/camera_preview/lut/` y `run_all.sh` la regenera | `FROM=2` |
| qué instantes se renderizan | `task2_render.py`: `RENDER_IDX` (hoy, 8 frames repartidos) | `FROM=2` |
| colores del RGB simulado y luces | `task2_render.py`: `RGB` y `RGB_LIGHT_SCALE`. No afectan a la térmica | `FROM=2` |

Otra forma de escribir escenas, la del repo original, son los YAML de `configs/scenes/`, que lanzan
los `scripts/render_*.py`. Los experimentos de vasos no la usan, porque necesitan la geometría medida.

---

## 5. Problemas conocidos y cómo reconocerlos

| síntoma | causa | qué hacer |
|---|---|---|
| `Killed` en una tarea, o `Out of memory: Killed process` en `journalctl -k` | Falta RAM. Hasta el 28-oct la tarea 0 llegaba a 15 GB por una SVD completa innecesaria (corregido: ahora 0,5 GB; el mismo arreglo en `bench/cups_check/task2_geometry.py` la baja de 0,56 a 0,27 GB) | Cerrar otras sesiones de Isaac y repetir |
| Kit aborta al arrancar con `PyUnicode_IS_READY` o `Could not import rosidl_typesupport_c` | Las variables del ROS 2 del sistema (`/opt/ros/humble`) se cuelan en el Python de Isaac | Lanzar con `env -u LD_LIBRARY_PATH -u PYTHONPATH -u AMENT_PREFIX_PATH -u ROS_DISTRO -u ROS_VERSION -u ROS_PYTHON_VERSION`. `run_all.sh` ya lo hace |
| `pytest` muere enseguida con `No module named 'lark'` | pytest carga los plugins de ROS | `export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` |
| `make check` falla en lint o en algunos tests que nombran `scripts/*inspection_bench*`, `configs/scenes/inspection_bench.yaml`… | Archivos sin versionar de la carpeta de trabajo | No son de la rama. Se comprueba con `git status` |
| `FileNotFoundError: no LUT bundle …` o `StaleLUTError` al renderizar | Falta la LUT de la banda, o se hizo con otra banda o respuesta espectral | Borrar `bench/camera_preview/lut/` y correr `run_all.sh` |
| `ModuleNotFoundError: lpips` en la tarea 3 | El Python de Isaac no tiene torch; la tarea 3 usa el conda `vision` más `~/.cache/irsim_pylibs` | `run_all.sh` lo instala. Si se borró `~/.cache`, lo reinstala solo |
| Un script de `bench/render_cost/code/` no encuentra `/tmp/...` | 33 archivos de esa copia apuntan al scratchpad de una sesión que ya no existe (`code/rebase_2026-10-22/NOTES.md`) | Reescribir esas rutas antes de usarlos. Los tres vasos no usan esas rutas |
| La etapa 3 Warp de `src/` falla al lanzarse (número de argumentos) | `irsim_isaac.pipeline.warp_stages.launch_optics` pasa 9 argumentos y el kernel espera 8 (fallo de `src/`, sin arreglar) | El banco de fps usa su propio kernel (`--optics fused`) |
| El vaso real se ve más alto (~4 px) y con «sombra» clara debajo | El reflejo en la mesa, que el simulador no modela | Es esperado (tarea 4); baja la SSIM del vaso caliente |
| El vaso cercano y el fondo salen con el mismo desenfoque | El desenfoque por capas reparte sus 8 capas entre la mesa a 3 cm y el fondo a 1 m | `CUPS_MIN_DISTANCE_M=0.3` (ya en `run_all.sh`) pone un suelo a la distancia |
| El camino GPU (fps) no coincide con el de CPU | El camino GPU no tiene desenfoque por capas | Es esperado; por eso se mide con `--no-check` |
| Los números de objeto de `out/sim/ids.npy` cambian entre corridas | Kit numera las instancias de otra forma en cada sesión | Es la misma geometría; los scripts usan nombres, no números |
| El error real − sim del vaso caliente baja ~1,1 K entre el primer y el último frame | El simulador lo mantiene a la media de las dos lecturas (no tienen hora) y el vaso real se enfría de verdad | Limitación de los datos, no del simulador. Antes (rampa entre lecturas) era una deriva falsa de −2 → +3 K. Ver §3 y `RESULTADOS.md` |
| La cámara simulada lee el fondo +0,3 K de más, con un cuenco hacia las esquinas | La carcasa trabaja a aire + 4 K (301,05 K) y la inversión radiométrica supone 300 K: un desacuerdo, no un número de la Boson en sí | `cellplus_ctc006_lwir_cups3.yaml` calibra a la carcasa en su sitio (`housing_temp_k: 301.05`) y el desfase pasa a 0,000 K. No sale de telemetría: la cámara no la guarda (`cups3_check/RESULTADOS.md`, tarea 7) |
| El ruido simulado no se parece al real | El YAML tiene el ruido medido con `denoise: 40`, y los tres vasos se grabaron con `denoise: 80` | Medido (tarea 6): cambia poco (temporal −4 a −9 %). Los valores nuevos están en `cellplus_ctc006_lwir_cups3.yaml` |
| Dos frames térmicos de `sync_cups` son idénticos al anterior | Dos barridos de LiDAR a 14–16 ms se emparejaron con el mismo frame de cámara (sets 80/81, 88/89) | Quitar los repetidos por contenido antes de medir ruido (lo hace `task6_noise.py`) |
| Una medida de fps sale más baja de lo normal | Otra carga en la máquina (pytest, otra sesión de Isaac) o la CPU en modo ahorro | Máquina en reposo y `powerprofilesctl set performance` |
