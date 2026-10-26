"""The table, the fixed/per-pixel split, and the achievable frame rate."""
import json, pathlib, itertools
B = pathlib.Path("/tmp/irsim_render_cost")

def load(tag):
    p = B / f"result_{tag}.json"
    return json.load(open(p)) if p.exists() else None

MAIN = [("ss4_rt", "4"), ("ss1_rt", "1"), ("ss4_pt", "4"), ("ss1_pt", "1")]
rows = []
for tag, _ in MAIN:
    d = load(tag)
    if d is None: continue
    p = d["phases"]; r = d["renderer_settings_readback"]
    rows.append(dict(tag=tag, mode=d["effective_rendermode"], ss=d["supersample_factor"],
                     px=d["render_product_px"][0]*d["render_product_px"][1],
                     product="x".join(map(str, d["render_product_px"])),
                     spp=r["/rtx/pathtracing/spp"], vram=d["vram_mib_max"],
                     **{k: (p[k]["mean_ms"], p[k]["std_ms"]) for k in
                        ("renderer","assembly_b","material_id","temperature",
                         "environment_radiance","sky_clear_lut","get_outputs_total")}))

W = 118
print("="*W)
print("RENDER COST, inspection bench, 30 frames after 10 discarded, RTX 3090")
print("="*W)
hdr = f"{'mode':<22}{'ss':>3}{'product':>12}{'spp':>5}{'(a) renderer':>18}{'(b) assembly':>18}{'total frame':>18}{'VRAM':>8}"
print(hdr); print("-"*W)
for r in rows:
    print(f"{r['mode']:<22}{r['ss']:>3}{r['product']:>12}{r['spp']:>5}"
          f"{r['renderer'][0]:>11.1f} ±{r['renderer'][1]:<5.1f}"
          f"{r['assembly_b'][0]:>11.1f} ±{r['assembly_b'][1]:<5.1f}"
          f"{r['get_outputs_total'][0]:>11.1f} ±{r['get_outputs_total'][1]:<5.1f}"
          f"{r['vram']:>7.0f}M")
print()
print("(c) inside (b), and the radiance stage beside it:")
print(f"{'mode':<22}{'ss':>3}{'material_id':>14}{'temperature':>14}{'env_radiance':>16}{'of which sky LUT':>18}")
print("-"*W)
for r in rows:
    print(f"{r['mode']:<22}{r['ss']:>3}"
          f"{r['material_id'][0]:>9.1f} ±{r['material_id'][1]:<4.1f}"
          f"{r['temperature'][0]:>9.1f} ±{r['temperature'][1]:<4.1f}"
          f"{r['environment_radiance'][0]:>11.1f} ±{r['environment_radiance'][1]:<4.1f}"
          f"{r['sky_clear_lut'][0]:>13.1f} ±{r['sky_clear_lut'][1]:<4.1f}")

# fixed vs per-pixel, per renderer mode
print("\n" + "="*W)
print("RENDERER COST MODEL:  t = a + b*px   (two resolutions per mode)")
print("="*W)
for mode in sorted({r["mode"] for r in rows}):
    g = sorted([r for r in rows if r["mode"] == mode], key=lambda r: r["px"])
    if len(g) < 2: continue
    lo, hi = g[0], g[-1]
    b = (hi["renderer"][0]-lo["renderer"][0])/(hi["px"]-lo["px"])
    a = lo["renderer"][0] - b*lo["px"]
    print(f"{mode} (spp {lo['spp']}): fixed a = {a:6.1f} ms,  per-pixel b = {b*1e6:5.3f} ms/Mpx")
    print(f"   -> ceiling from the fixed term alone: {1000/a:5.1f} fps, at ANY resolution")
    need = 1000/30
    if a >= need:
        print(f"   -> 30 fps needs {need:.1f} ms: BELOW the fixed term. Unreachable at every resolution.")
    else:
        px = (need-a)/b
        print(f"   -> 30 fps budget {need:.1f} ms allows {px/1e6:.2f} Mpx "
              f"(~{int((px/1.25)**0.5)}x{int((px/1.25)**0.5*1.25*0.8)} at 5:4)")
    # and with host processing free, what the measured resolutions give
    for r in g:
        print(f"      measured {r['product']:>11}: renderer {r['renderer'][0]:6.1f} ms -> "
              f"{1000/r['renderer'][0]:5.1f} fps   (full frame {r['get_outputs_total'][0]/1000:5.2f} s -> "
              f"{1000/r['get_outputs_total'][0]:5.2f} fps)")

# spp sweep, if present
sweep = [(s, load(f"ss1_pt_spp{s}")) for s in (1,4,8,16)]
sweep = [(s,d) for s,d in sweep if d]
if sweep:
    print("\n" + "="*W)
    print("SPP SWEEP at 640x512 PathTracing: is the floor spp-bound or readback-bound?")
    print("="*W)
    base = load("ss1_pt")
    allp = sweep + ([(32, base)] if base else [])
    for s, d in allp:
        p = d["phases"]
        print(f"  spp {s:>3}: renderer {p['renderer']['mean_ms']:7.2f} ±{p['renderer']['std_ms']:5.2f} ms"
              f"  -> {1000/p['renderer']['mean_ms']:6.2f} fps   (total frame {p['get_outputs_total']['mean_ms']/1000:5.2f} s)")
    if len(allp) >= 2:
        lo, hi = allp[0], allp[-1]
        c = (hi[1]["phases"]["renderer"]["mean_ms"]-lo[1]["phases"]["renderer"]["mean_ms"])/(hi[0]-lo[0])
        a0 = lo[1]["phases"]["renderer"]["mean_ms"] - c*lo[0]
        print(f"\n  spp-independent remainder at 640x512: {a0:.2f} ms  ({c:.3f} ms per spp)")
        print(f"  -> that remainder alone caps the rate at {1000/a0:.1f} fps")

# isothermal enclosure
print("\n" + "="*W)
print("ISOTHERMAL ENCLOSURE: does a scalar L_env cost less?")
print("="*W)
for ss in (4, 1):
    a_, b_ = load(f"ss{ss}_rt"), load(f"ss{ss}_rt_iso")
    if not (a_ and b_): continue
    pa, pb = a_["phases"], b_["phases"]
    for k in ("environment_radiance", "sky_clear_lut"):
        d = pb[k]["mean_ms"] - pa[k]["mean_ms"]
        print(f"  ss={ss} {k:22s} {pa[k]['mean_ms']:8.2f} -> {pb[k]['mean_ms']:8.2f} ms  "
              f"({100*d/pa[k]['mean_ms']:+6.1f} %)")
