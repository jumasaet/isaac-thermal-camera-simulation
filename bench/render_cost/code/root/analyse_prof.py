"""Callee tree of get_outputs from a cProfile dump, per frame, with file:line.

Edge times are cProfile's own caller->callee cumulative times, so a function called from two
places is split between them rather than double counted.  usage: analyse_prof.py <prof> <frames>
[min_ms=5] [max_depth=12]
"""
import pstats, sys, collections
path, frames = sys.argv[1], int(sys.argv[2])
min_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
max_depth = int(sys.argv[4]) if len(sys.argv) > 4 else 12
st = pstats.Stats(path).stats            # func -> (cc, nc, tt, ct, callers{caller: (cc,nc,tt,ct)})
callees = collections.defaultdict(dict)  # caller -> {callee: (nc, tt, ct)}
for func, (cc, nc, tt, ct, callers) in st.items():
    for caller, (ecc, enc, ett, ect) in callers.items():
        callees[caller][func] = (enc, ett, ect)

REPO = "/home/jumasaet/isaac-thermal-camera-simulation/"
def name(f):
    file, line, fn = f
    if file == "~":
        return fn
    short = file.replace(REPO, "")
    for marker in ("site-packages/", "/exts/", "/extscache/", "/kit/"):
        if marker in short:
            short = "…/" + short.split(marker)[-1]
    return f"{fn}  [{short}:{line}]"

root = next(f for f in st if f[2] == "get_outputs" and "ir_camera" in f[0])
total = st[root][3] / frames * 1e3
print(f"{path}: get_outputs {total:.1f} ms/frame over {frames} frames (cProfile)\n")
print(f"{'incl ms':>9} {'self ms':>9} {'calls/f':>8}  function")

def walk(func, nc, tt, ct, depth, seen):
    incl = ct / frames * 1e3
    # self time of this node, scaled to this edge's share of the function's total
    full_ct = st[func][3] or 1e-12
    share = ct / full_ct
    self_ms = st[func][2] * share / frames * 1e3
    print(f"{incl:9.2f} {self_ms:9.2f} {nc / frames:8.1f}  {'  ' * depth}{name(func)}")
    if depth >= max_depth or func in seen:
        return
    kids = sorted(callees.get(func, {}).items(), key=lambda kv: -kv[1][2])
    shown = 0.0
    for kid, (knc, ktt, kct) in kids:
        k_incl = kct * share / frames * 1e3
        if k_incl >= min_ms:
            walk(kid, knc * share, ktt * share, kct * share, depth + 1, seen | {func})
            shown += k_incl
    rest = incl - self_ms - shown
    if kids and rest >= min_ms:
        n = sum(1 for kid, v in kids if v[2] * share / frames * 1e3 < min_ms)
        print(f"{rest:9.2f} {'':9} {'':8}  {'  ' * (depth + 1)}(+{n} callees each < {min_ms:g} ms)")

walk(root, st[root][1], st[root][2], st[root][3], 0, frozenset())
