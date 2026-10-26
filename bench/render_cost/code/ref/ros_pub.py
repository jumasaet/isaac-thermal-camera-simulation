"""The live half of ROS 2 publication for the reference runner. /tmp only; nothing here edits the repo.

The repo (`irsim_isaac.ros2_bridge`) builds the messages as plain data and checks them without a
graph; its `publish_frame(node, messages, info)` needs a `node` with a `publisher(topic, encoding)`
factory whose objects take those plain messages. `RosNode` is that node, on rclpy.

**Which rclpy.** Isaac Sim's interpreter is CPython 3.12; the system ROS 2 Humble is built for 3.10,
so its rclpy cannot be imported here (its C extension is cp310). Isaac ships its own Humble rclpy
for cp312 under `exts/isaacsim.ros2.core/humble/rclpy`; that directory goes first on `sys.path`.
The shared libraries it loads (rcl, rmw_fastrtps, the type supports) come from `LD_LIBRARY_PATH`:
the system's when a ROS shell is sourced (python.sh then adds nothing), Isaac's own copy when it is
not (python.sh's setup_ros_env.sh adds it). Both are Humble + Fast DDS and talk to each other.

`msg._data` is assigned directly: the public setter walks a non-`array.array` value element by
element in Python (1.3 M ints for a 640x512 RGBA frame). The generated C converter takes anything
with the buffer protocol.
"""
from __future__ import annotations

import array
import sys
import zlib

import numpy as np

ISAAC_RCLPY = "/home/jumasaet/isaac_sim/exts/isaacsim.ros2.core/humble/rclpy"


def import_rclpy():
    """Isaac's cp312 rclpy, whatever the shell exported.

    A sourced ROS shell puts the system's python3.10 packages on PYTHONPATH. Half of them are pure
    Python and import fine here, then fail at the first type-support load (`rcl_interfaces` from
    /opt/ros with Isaac's rclpy: "Could not import rosidl_typesupport_c"). So every /opt/ros entry
    leaves `sys.path`, anything already imported from there is dropped, and Isaac's copy goes first.
    """
    sys.path[:] = [p for p in sys.path if not p.startswith("/opt/ros/")]
    for name, mod in list(sys.modules.items()):
        if (getattr(mod, "__file__", None) or "").startswith("/opt/ros/"):
            del sys.modules[name]
    if ISAAC_RCLPY not in sys.path:
        sys.path.insert(0, ISAAC_RCLPY)
    import rclpy

    if not rclpy.__file__.startswith(ISAAC_RCLPY):
        raise RuntimeError(f"rclpy came from {rclpy.__file__}, not from Isaac's cp312 build")
    return rclpy


class _ImagePublisher:
    def __init__(self, node, topic: str, qos, msg_cls) -> None:
        self._pub = node.create_publisher(msg_cls, topic, qos)
        self._msg = msg_cls()
        self.topic, self.count, self.crcs = topic, 0, {}
        self.keep_crc = False

    def publish(self, m) -> None:
        """`m`: irsim_isaac.ros2_bridge.ImageMessage."""
        msg = self._msg
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(int(m.stamp_ns), 1_000_000_000)
        msg.header.frame_id = m.frame_id
        msg.height, msg.width, msg.step = int(m.height), int(m.width), int(m.step)
        msg.encoding, msg.is_bigendian = m.encoding, int(m.is_bigendian)
        buf = np.ascontiguousarray(m.data).view(np.uint8).reshape(-1)
        a = array.array("B")
        a.frombytes(buf)
        msg._data = a
        self._pub.publish(msg)
        self.count += 1
        if self.keep_crc:
            self.crcs[int(m.stamp_ns)] = zlib.crc32(buf)

    @property
    def subscribers(self) -> int:
        return self._pub.get_subscription_count()


class _InfoPublisher:
    def __init__(self, node, topic: str, qos, msg_cls) -> None:
        self._pub = node.create_publisher(msg_cls, topic, qos)
        self._cls = msg_cls
        self.topic, self.count = topic, 0

    def publish(self, m) -> None:
        """`m`: irsim_isaac.ros2_bridge.CameraInfoMessage."""
        msg = self._cls()
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(int(m.stamp_ns), 1_000_000_000)
        msg.header.frame_id = m.frame_id
        msg.width, msg.height = int(m.width), int(m.height)
        # the repo's model names are its own; a ROS consumer expects plumb_bob for (k1, k2, p1, p2, k3)
        msg.distortion_model = str(m.distortion_model)
        msg.d = [float(c) for c in m.d]
        msg.k = [float(c) for c in m.k]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        fx, _, cx, _, fy, cy = m.k[:6]
        msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self._pub.publish(msg)
        self.count += 1

    @property
    def subscribers(self) -> int:
        return self._pub.get_subscription_count()


class RosNode:
    """`publisher(topic, encoding)` as `irsim_isaac.ros2_bridge.publish_frame` expects it."""

    def __init__(self, name: str = "irsim_reference", reliability: str = "reliable", depth: int = 10) -> None:
        self.rclpy = rclpy = import_rclpy()
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, Image

        self._image_cls, self._info_cls = Image, CameraInfo
        if not rclpy.ok():
            rclpy.init(args=[])
        self.node = rclpy.create_node(name)
        rel = {"reliable": ReliabilityPolicy.RELIABLE, "best_effort": ReliabilityPolicy.BEST_EFFORT}[reliability]
        self.qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=int(depth), reliability=rel)
        self.reliability, self.depth = reliability, int(depth)
        self._pubs: dict[str, object] = {}

    def publisher(self, topic: str, encoding: str):
        if topic not in self._pubs:
            cls = _InfoPublisher if encoding == "info" else _ImagePublisher
            self._pubs[topic] = cls(self.node, topic, self.qos, self._info_cls if encoding == "info" else self._image_cls)
        return self._pubs[topic]

    def wait_for_subscribers(self, topics, n: int = 1, timeout_s: float = 20.0) -> dict[str, int]:
        import time

        t_end = time.perf_counter() + timeout_s
        while time.perf_counter() < t_end:
            if all(self._pubs[t].subscribers >= n for t in topics):
                break
            time.sleep(0.05)
        return {t: self._pubs[t].subscribers for t in topics}

    def describe(self) -> dict:
        import os

        import rclpy

        return {"rclpy": rclpy.__file__, "rmw": self.rclpy.get_rmw_implementation_identifier(),
                "domain_id": os.environ.get("ROS_DOMAIN_ID"), "reliability": self.reliability, "depth": self.depth,
                "topics": {t: {"published": p.count, "subscribers": p.subscribers} for t, p in self._pubs.items()}}

    def close(self) -> None:
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()
