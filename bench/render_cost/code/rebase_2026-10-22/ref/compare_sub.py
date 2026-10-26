#!/usr/bin/env python3
"""What the subscriber received against what the publisher sent: counts, order, duplicates, CRCs.

    compare_sub.py <published.json: {name: {stamp_ns: crc}}> <sub.json> name=topic [name=topic ...]
"""
import json
import sys

pub = json.load(open(sys.argv[1]))
sub = json.load(open(sys.argv[2]))["topics"]
ok = True
for item in sys.argv[3:]:
    name, _, topic = item.partition("=")
    sent = {int(k): v for k, v in pub[name].items()}
    got = sub.get(topic, {"count": 0, "stamps_ns": [], "crc32": []})
    stamps = got["stamps_ns"]
    bad_crc = sum(1 for s, c in zip(stamps, got["crc32"]) if sent.get(s) != c)
    missing = sorted(set(sent) - set(stamps))
    res = {"topic": topic, "sent": len(sent), "received": got["count"], "missing": len(missing),
           "duplicates": len(stamps) - len(set(stamps)), "in_order": stamps == sorted(stamps),
           "payload_crc_mismatches": bad_crc, "encoding": got.get("encoding"), "frame_id": got.get("frame_id"),
           "width_height_step_len": got.get("width_height_step_len")}
    if got["count"] > 1:
        span = got["arrival_s"][-1] - got["arrival_s"][0]
        res["arrival_rate_hz"] = round((got["count"] - 1) / span, 2) if span > 0 else None
    ok &= res["missing"] == 0 and bad_crc == 0 and res["duplicates"] == 0 and res["in_order"]
    print(json.dumps(res))
print("RECEPTION OK" if ok else "RECEPTION NOT CLEAN")
