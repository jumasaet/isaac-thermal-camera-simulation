# lite_prototype — el primer kernel del «irsim ligero» (descartado)

Fecha: 2026-10-07. Estaba en la raíz del repo, sin versionar; se movió aquí el 2026-10-28 para no perderlo.

- `kernel.py`: un kernel Warp fusionado, un hilo por píxel (ids + normales → ε(θ) → radiancia → T aparente), con
  Planck en banda y el cielo precalculados en host como tablas.
- `test_kernel.py`: sus tests en CPU, con un Planck 8–14 µm sustituto.

Se descartó a favor de `../lite/lite.py`, que usa el kernel de radiancia del propio repo. Nada lo importa. Necesita
`warp`, que el `python.sh` de Isaac no carga fuera de Kit, así que los tests no corren tal cual.
