"""No Kit: can Isaac's interpreter publish the repo's messages, how long does one publish take,
and does a system-ROS subscriber receive exactly those bytes? Run with python.sh."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/jumasaet/isaac-thermal-camera-simulation/src")
sys.path.insert(0, "/home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/code/rebase_2026-10-22/ref")
import ros_pub  # noqa: E402

from irsim_isaac.ros2_bridge import ImageMessage, image_message  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 240
wait = sys.argv[2] != "nosub" if len(sys.argv) > 2 else True
rate = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
qos = sys.argv[4] if len(sys.argv) > 4 else "reliable"
node = ros_pub.RosNode(reliability=qos)
ir = node.publisher("/thermal_sim/ir/image_raw", "rgba8")
rgb = node.publisher("/thermal_sim/rgb/image_raw", "rgba8")
ir.keep_crc = rgb.keep_crc = True
print(node.describe(), flush=True)
if wait:
    print("subscribers:", node.wait_for_subscribers([ir.topic, rgb.topic]), flush=True)
    time.sleep(0.5)
rng = np.random.default_rng(0)
frames = rng.integers(0, 256, (8, 512, 640, 4), dtype=np.uint8)
dn = rng.integers(0, 16384, (512, 640), dtype=np.uint16)
m16 = image_message("dn16", dn, t_s=0.0, namespace="/thermal_sim/ir")
print("repo builder for dn16 ->", m16.topic, m16.encoding, m16.step)
dt = []
for k in range(n):
    t0 = time.perf_counter()
    stamp = int(round(k / 60.0 * 1e9))
    a = ImageMessage(topic=ir.topic, encoding="rgba8", height=512, width=640, step=2560, data=frames[k % 8],
                     frame_id="ir_camera", stamp_ns=stamp)
    b = ImageMessage(topic=rgb.topic, encoding="rgba8", height=512, width=640, step=2560, data=frames[(k + 3) % 8],
                     frame_id="rgb_camera", stamp_ns=stamp)
    ir.publish(a)
    rgb.publish(b)
    t1 = time.perf_counter()
    dt.append(t1 - t0)
    rest = 1.0 / rate - (t1 - t0)
    if rest > 0:
        time.sleep(rest)
dt = 1e3 * np.asarray(dt)
print(f"publish of both images: mean {dt.mean():.3f} ms p50 {np.percentile(dt, 50):.3f} p95 {np.percentile(dt, 95):.3f} "
      f"max {dt.max():.3f} (n {n}, {rate:g} Hz, subscriber {wait})", flush=True)
json.dump({"ir": {str(k): v for k, v in ir.crcs.items()}, "rgb": {str(k): v for k, v in rgb.crcs.items()}},
          open("/home/jumasaet/isaac-thermal-camera-simulation/bench/render_cost/results/rebase_2026-10-22/ref_out/bare_pub.json", "w"))
time.sleep(1.0)
node.close()
