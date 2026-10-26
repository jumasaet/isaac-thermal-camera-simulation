"""The pair, side by side: every pass's wall, the span table, and the cProfile tree's top level."""
import json, statistics as st, pathlib
P = pathlib.Path("/tmp/irsim_render_cost/prof")
runs = {t: json.load(open(P / f"{t}.json")) for t in ("pair_iso_1", "pair_clear_1", "pair_iso_2", "pair_clear_2")}
ms = lambda xs: (1e3 * st.fmean(xs), 1e3 * st.stdev(xs))
print(f"{'pass':<34}" + "".join(f"{t:>20}" for t in runs))
for key, label in (("A0_plain", "A0 plain, nothing wrapped"), ("A1_sync", "A1 + renderer sync"),
                   ("A1_reader_step", "   reader.step"), ("A1_reader_read", "   reader.read"),
                   ("P_cprofile_wall", "P  under cProfile"), ("L_line_wall", "L  under line_profiler"),
                   ("S_span_wall", "S  under span timers"), ("A2_plain_after", "A2 plain again")):
    print(f"{label:<34}" + "".join("{:>13.1f} ±{:<5.1f}".format(*ms(r[key])) for r in runs.values()))
print("\ncaches after the run:", {t: r["cache_sizes"] for t, r in runs.items()})
names = sorted({k for r in runs.values() for k in r["S_spans"]},
               key=lambda k: -runs["pair_iso_1"]["S_spans"].get(k, {"incl_ms": 0})["incl_ms"])
print(f"\n{'span (inclusive ms / frame)':<62}" + "".join(f"{t[5:]:>10}" for t in runs))
for k in names:
    print(f"{k:<62}" + "".join(f"{r['S_spans'].get(k, {}).get('incl_ms', float('nan')):>10.1f}" for r in runs.values()))
