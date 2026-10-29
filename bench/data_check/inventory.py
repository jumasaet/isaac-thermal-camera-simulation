#!/usr/bin/env python
"""Inventory of thermal_specs/: per folder, count, time span, real rate, corrupt / empty / duplicate files.

    python3 -I bench/data_check/inventory.py thermal_specs bench/data_check/out/inventory.json     (~5 min, reads 18 GB)

Every data file is named <index>_<unix time>.<ext>; the time is the one the rate and gaps are computed from.
Checks, per extension: .npy loads, is int16 (480, 640) and not constant; .png has its signature and ends in IEND;
.pcd's size is header + POINTS x 26 bytes (x y z intensity tag line timestamp); .pgm is P5 and 1228816 bytes.
Duplicates are byte-identical files (MD5) inside one folder. Read-only: nothing is moved or deleted.
Findings and the delete / keep proposal: RESULTADOS.md next to this script.
"""
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = sys.argv[1]
STAMP = re.compile(r"^(\d+)_(\d+\.\d+)\.(npy|png|pcd|pgm)$")
out = {}


def check_npy(p):
    try:
        a = np.load(p, allow_pickle=False)
    except Exception as e:  # noqa: BLE001
        return "load_error:" + type(e).__name__, None
    if a.shape != (480, 640) or a.dtype != np.int16:
        return f"shape_dtype:{a.shape}:{a.dtype}", None
    if a.min() == a.max():
        return f"constant:{int(a.min())}", a
    return None, a


def check_png(p):
    with open(p, "rb") as f:
        b = f.read()
    if len(b) == 0:
        return "empty"
    if b[:8] != b"\x89PNG\r\n\x1a\n":
        return "bad_signature"
    if b[-12:-8] != b"\x00\x00\x00\x00" or b[-8:-4] != b"IEND":
        return "truncated(no IEND)"
    return None


def check_pcd(p):
    with open(p, "rb") as f:
        head = f.read(1024)
    if not head:
        return "empty", 0
    m = re.search(rb"POINTS (\d+)", head)
    n = int(m.group(1)) if m else -1
    hdr_end = head.find(b"DATA binary\n") + len(b"DATA binary\n")
    size = os.path.getsize(p)
    expect = hdr_end + n * 26
    if size != expect:
        return f"size {size} != {expect}", n
    return None, n


def check_pgm(p):
    size = os.path.getsize(p)
    with open(p, "rb") as f:
        head = f.read(64)
    if not head.startswith(b"P5"):
        return "bad_signature"
    return None if size == 1228816 else f"size {size}"


for dirpath, dirs, files in sorted(os.walk(ROOT)):
    dirs.sort()
    stamped = [f for f in files if STAMP.match(f)]
    if not stamped:
        continue
    rel = os.path.relpath(dirpath, ROOT)
    idx, ts, problems = [], [], []
    hashes = defaultdict(list)
    consec_equal = 0
    prev = None
    sizes = []
    pts = []
    for f in sorted(stamped, key=lambda s: (float(STAMP.match(s).group(2)), s)):
        m = STAMP.match(f)
        idx.append(int(m.group(1)))
        ts.append(float(m.group(2)))
        p = os.path.join(dirpath, f)
        sizes.append(os.path.getsize(p))
        ext = m.group(3)
        if os.path.getsize(p) == 0:
            problems.append((f, "empty"))
            continue
        with open(p, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        hashes[h].append(f)
        if ext == "npy":
            err, a = check_npy(p)
            if a is not None:
                if prev is not None and np.array_equal(a, prev):
                    consec_equal += 1
                prev = a
        elif ext == "png":
            err = check_png(p)
        elif ext == "pcd":
            err, n = check_pcd(p)
            pts.append(n)
        else:
            err = check_pgm(p)
        if err:
            problems.append((f, err))
    ts = np.array(ts)
    dt = np.diff(ts)
    idx = np.array(idx)
    di = np.diff(idx)
    dup = [v for v in hashes.values() if len(v) > 1]
    rec = {
        "n": len(stamped),
        "other_files": sorted(set(files) - set(stamped)),
        "t_first": ts[0],
        "t_last": ts[-1],
        "span_s": float(ts[-1] - ts[0]),
        "rate_hz": float((len(ts) - 1) / (ts[-1] - ts[0])) if len(ts) > 1 else None,
        "dt_median_ms": float(np.median(dt) * 1e3) if len(dt) else None,
        "dt_min_ms": float(dt.min() * 1e3) if len(dt) else None,
        "dt_max_ms": float(dt.max() * 1e3) if len(dt) else None,
        "n_gaps_gt_3x_median": int((dt > 3 * np.median(dt)).sum()) if len(dt) else 0,
        "index_first": int(idx[0]),
        "index_last": int(idx[-1]),
        "index_span": int(idx[-1] - idx[0] + 1),
        "index_missing": int((idx[-1] - idx[0] + 1) - len(set(idx))),
        "index_duplicated": int(len(idx) - len(set(idx))),
        "index_nonmonotonic": int((di <= 0).sum()),
        "problems": problems,
        "identical_content_groups": dup[:10],
        "n_identical_groups": len(dup),
        "consecutive_identical_npy": consec_equal,
        "size_min": min(sizes),
        "size_max": max(sizes),
        "bytes_total": int(sum(sizes)),
    }
    if pts:
        rec["pcd_points_min"] = int(min(pts))
        rec["pcd_points_median"] = int(np.median(pts))
    out[rel] = rec
    print(rel, json.dumps({k: v for k, v in rec.items() if k not in ("identical_content_groups",)}, default=str), flush=True)

with open(sys.argv[2], "w") as f:
    json.dump(out, f, indent=1, default=str)
