#!/usr/bin/env python3
"""The receiving end, in the SYSTEM ROS 2 Humble (python3.10): proof that frames arrive, and which.

    /usr/bin/python3 ros_sub.py --out sub.json [--topics /thermal_sim/ir/image_raw /thermal_sim/rgb/image_raw]
                                [--qos reliable|best_effort] [--depth 10] [--idle 6]

One subscription per topic. Per message it keeps the header stamp, a CRC-32 of the payload, the
encoding and the arrival time; nothing else is done with the image. It writes the JSON and exits
when no message has arrived for `--idle` seconds after the first one (or on SIGINT/SIGTERM).
The runner compares the stamps and CRCs with what it published.
"""
import argparse
import json
import signal
import time
import zlib

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

p = argparse.ArgumentParser()
p.add_argument("--topics", nargs="+", default=["/thermal_sim/ir/image_raw", "/thermal_sim/rgb/image_raw"])
p.add_argument("--qos", choices=("reliable", "best_effort"), default="reliable")
p.add_argument("--depth", type=int, default=10)
p.add_argument("--idle", type=float, default=6.0)
p.add_argument("--max-wait", type=float, default=600.0, help="give up if nothing ever arrives")
p.add_argument("--out", required=True)
args = p.parse_args()

rclpy.init()
node = rclpy.create_node("irsim_reference_check")
rel = ReliabilityPolicy.RELIABLE if args.qos == "reliable" else ReliabilityPolicy.BEST_EFFORT
qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=args.depth, reliability=rel)
got = {t: [] for t in args.topics}
last = [None]


def make(topic):
    def cb(msg):
        now = time.monotonic()
        last[0] = now
        got[topic].append((msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec, zlib.crc32(msg.data),
                           msg.encoding, msg.width, msg.height, msg.step, len(msg.data), msg.header.frame_id, now))
    return cb


subs = [node.create_subscription(Image, t, make(t), qos) for t in args.topics]
stop = [False]
for s in (signal.SIGINT, signal.SIGTERM):
    signal.signal(s, lambda *_: stop.__setitem__(0, True))
ex = SingleThreadedExecutor()
ex.add_node(node)
t_start = time.monotonic()
print(f"subscribed to {args.topics} ({args.qos}, depth {args.depth}), rmw {rclpy.get_rmw_implementation_identifier()}", flush=True)
while not stop[0]:
    ex.spin_once(timeout_sec=0.2)
    now = time.monotonic()
    if last[0] is not None and now - last[0] > args.idle:
        break
    if last[0] is None and now - t_start > args.max_wait:
        break

out = {"qos": args.qos, "depth": args.depth, "topics": {}}
for t, rows in got.items():
    out["topics"][t] = {
        "count": len(rows),
        "stamps_ns": [r[0] for r in rows], "crc32": [r[1] for r in rows],
        "encoding": sorted({r[2] for r in rows}), "width_height_step_len": sorted({r[3:7] for r in rows}),
        "frame_id": sorted({r[7] for r in rows}),
        "arrival_s": [round(r[8] - rows[0][8], 6) for r in rows],
    }
    print(f"{t}: {len(rows)} messages", flush=True)
with open(args.out, "w") as f:
    json.dump(out, f)
node.destroy_node()
rclpy.shutdown()
