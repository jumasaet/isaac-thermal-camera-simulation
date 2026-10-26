"""Applied to run_full.py after the matrix: a noise-free content audit of freshness (--audit-ids)."""
import pathlib
p = pathlib.Path("/home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/code/rebase_2026-10-22/full/run_full.py"); s = p.read_text()
rep = [
 ('p.add_argument("--no-check", action="store_true")\n',
  'p.add_argument("--no-check", action="store_true")\np.add_argument("--audit-ids", nargs="*", default=[], help="variants audited by the id plane each delivered frame was computed from")\n'),
 ('''    floats: dict[int, dict] = {}
    first = -args.warm''', '''    floats: dict[int, dict] = {}
    ids_seq = np.full((N, H, W), 255, np.uint8) if mode == "audit" else None
    first = -args.warm'''),
 ('''        bad = lag != 1 or repeated or not looks_right
        if timed:''', '''        if ids_seq is not None and timed and slot is not None and looks_right:
            # after the clock: the id plane this iteration's thermal frame was computed from
            ids_seq[slot] = data_of(payload["instance"]).numpy().reshape(HS, WS)[K // 2::K, K // 2::K].astype(np.uint8)
        bad = lag != 1 or repeated or not looks_right
        if timed:'''),
 ('''    return {"report": rep_, "thermal": therm, "floats": floats, "frame_index_of": frame_index_of}''',
  '''    return {"report": rep_, "thermal": therm, "floats": floats, "frame_index_of": frame_index_of, "ids_seq": ids_seq}'''),
 ('''# -- main ---------------------------------------------------------------------------------------
report: dict = {''', '''def audit_ids(variant: str, traj: str) -> dict:
    """Noise-free freshness audit. One more traversal with the variant's whole frame in the loop,
    keeping (after the clock) the instance-id plane each iteration's frame was computed from, filed
    under the pose its stamp names. Every pose is then rendered synchronously and the id planes are
    compared exactly. A pose whose id plane equals a neighbour's cannot tell fresh from stale and
    is counted apart."""
    POS, LOOK = ns.trajectory(traj, N)
    r = run_pass(variant, traj, 0, "audit")
    got = r["ids_seq"]
    ref = np.zeros((N, H, W), np.uint8)
    for j in range(N):
        payload = render_sync(POS[j], LOOK[j])
        ref[j] = data_of(payload["instance"]).numpy().reshape(HS, WS)[K // 2::K, K // 2::K].astype(np.uint8)
    own = np.array([int((got[j] != ref[j]).sum()) for j in range(N)])
    prev = np.array([int((got[j] != ref[(j - 1) % N]).sum()) for j in range(N)])
    nxt = np.array([int((got[j] != ref[(j + 1) % N]).sum()) for j in range(N)])
    amb = np.array([np.array_equal(ref[j], ref[(j - 1) % N]) or np.array_equal(ref[j], ref[(j + 1) % N]) for j in range(N)])
    out = {"identical_to_own_pose": int((own == 0).sum()), "n": N, "worst_px_differing_from_own": int(own.max()),
           "identical_to_previous_pose": int((prev == 0).sum()), "identical_to_next_pose": int((nxt == 0).sum()),
           "poses_identical_to_a_neighbour_in_the_reference": int(amb.sum()),
           "distinguishable_and_identical_to_own": int(((own == 0) & ~amb).sum()), "distinguishable": int((~amb).sum()),
           "min_px_between_neighbouring_distinguishable_poses": int(min(prev[~amb].min(), nxt[~amb].min())) if (~amb).any() else None,
           "lag_hist": r["report"]["freshness"]["lag_hist"], "ms": r["report"]["ms"]["total"]["mean"]}
    return out


# -- main ---------------------------------------------------------------------------------------
report: dict = {'''),
 ('''runs: dict = {}
for pass_i in range(1, args.passes + 1):''', '''for traj in TRAJS:
    for variant in args.audit_ids:
        a = audit_ids(variant, traj)
        report.setdefault("audit_ids", {})[f"{variant}/{traj}"] = a
        print(f"[{args.tag} audit-ids {variant} {traj}] {json.dumps(a)}", flush=True)
        dump()
runs: dict = {}
for pass_i in range(1, args.passes + 1):'''),
]
for a, b in rep:
    assert a in s, a[:60]
    s = s.replace(a, b, 1)
p.write_text(s)
print("patched")
