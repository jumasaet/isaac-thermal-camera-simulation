"""After the step: cold _clear_lut against the 181 scalar calls it replaced, interleaved, bytes."""
import statistics, sys, time
sys.argv = [sys.argv[0], "none"]
sys.path.insert(0, "/tmp/irsim_render_cost")
import numpy as np
import lut_cost as L

sky, t0 = L.build()
sky._clear_lut(t0 - 1.0)
new, old, lut_new, same = [], [], [], True
for i in range(12):
    t = t0 + 600.0 * i
    a = time.perf_counter(); got = sky._atm.sky_radiance_grid(sky._band, t, L.ELS, sky._q); b = time.perf_counter()
    want = L.scalar(sky, t); c = time.perf_counter()
    new.append(b - a); old.append(c - b)
    same &= got.tobytes() == want.tobytes() and got.dtype == want.dtype
    a = time.perf_counter(); sky._clear_lut(t + 1.0); lut_new.append(time.perf_counter() - a)
f = lambda xs: f"{1e3 * statistics.fmean(xs):7.1f} ms +-{1e3 * statistics.stdev(xs):4.1f} (min {1e3 * min(xs):6.1f})"
print("181 scalar sky_radiance calls :", f(old))
print("sky_radiance_grid             :", f(new), " bytes equal:", same)
print("cold _clear_lut (grid + tilt) :", f(lut_new))
