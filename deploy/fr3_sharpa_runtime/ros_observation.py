"""One fresh, causal ROS observation snapshot. No publishers or SDK calls."""

from __future__ import annotations

import base64
import json
import sys
import time

import numpy as np

from .protocol import SHARPA_JOINT_NAMES

TOPICS = {
    "left_arm": "/left/franka/joint_states",
    "right_arm": "/right/franka/joint_states",
    "left_hand": "/sharpa/left/joint_states",
    "right_hand": "/sharpa/right/joint_states",
    "cam0": "/cam0/color/image_raw",
    "cam1": "/cam1/color/image_raw",
    "cam2": "/cam2/color/image_raw",
}
MAX_AGE_NS = 200_000_000
MAX_SPREAD_NS = 100_000_000


def ordered_state(message, side: str, hand: bool) -> list[float]:
    names = list(message.name)
    values = list(message.position)
    if len(names) != len(values) or len(names) != len(set(names)):
        raise ValueError("Invalid joint state names/values")
    data = dict(zip(names, values, strict=True))
    if hand:
        required = [f"{side}_{name}" for name in SHARPA_JOINT_NAMES]
    else:
        alternatives = [
            [f"{side}_fr3_joint{i}", f"{side}_fr3v2_joint{i}"]
            for i in range(1, 8)
        ]
        required = []
        for group in alternatives:
            present = [name for name in group if name in data]
            if len(present) != 1:
                raise ValueError(f"Missing/ambiguous FR3 state joint: {group}")
            required.append(present[0])
    if any(name not in data for name in required):
        raise ValueError(f"Missing {side} {'hand' if hand else 'arm'} state joint")
    result = np.asarray([data[name] for name in required], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite measured joint state")
    return result.tolist()


def rgb_image(message) -> dict:
    channels = {"rgb8": 3, "bgr8": 3}
    if message.encoding not in channels:
        raise ValueError(f"Unsupported non-RGB camera encoding: {message.encoding}")
    height, width, step = int(message.height), int(message.width), int(message.step)
    count = channels[message.encoding]
    if height <= 0 or width <= 0 or step < width * count:
        raise ValueError("Invalid camera image dimensions")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size != height * step:
        raise ValueError("Invalid camera byte count")
    pixels = raw.reshape(height, step)[:, :width * count].reshape(height, width, count)
    if message.encoding.startswith("bgr"):
        pixels = pixels[:, :, [2, 1, 0]]
    else:
        pixels = pixels[:, :, :3]
    return {"shape": [height, width, 3], "rgb_b64": base64.b64encode(pixels.tobytes()).decode("ascii")}


def stamp_ns(message) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class ObservationProvider:
    """One ROS context/node and subscriptions for the runtime lifetime."""
    def __init__(self):
        import rclpy
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                               ReliabilityPolicy, qos_profile_sensor_data)
        from sensor_msgs.msg import Image, JointState
        self.rclpy = rclpy
        rclpy.init()
        self.node = rclpy.create_node("fr3_sharpa_observation")
        self.latest = {}
        self.subscriptions = []
        self.closed = False
        camera_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5,
                                reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.VOLATILE)
        try:
            for key, topic in TOPICS.items():
                self.subscriptions.append(self.node.create_subscription(
                    Image if key.startswith("cam") else JointState, topic,
                    lambda message, selected=key: self.latest.__setitem__(selected, (message, time.monotonic_ns())),
                    camera_qos if key.startswith("cam") else qos_profile_sensor_data))
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.node.destroy_node()
            finally:
                self.rclpy.shutdown()

    def snapshot(self, timeout_s: float = 5.0) -> dict:
        if self.closed:
            raise RuntimeError("Observation provider is closed")
        rclpy, node, latest = self.rclpy, self.node, self.latest
        deadline = time.monotonic() + timeout_s
        timing_failure = "No complete snapshot received"
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())))
            if len(latest) != len(TOPICS):
                continue
            # Keep receiving real updates until the existing timing checks pass.
            # No timestamps or frames are synthesized, shifted or duplicated.
            now_mono = time.monotonic_ns()
            now_ros = node.get_clock().now().nanoseconds
            stamps = []
            issues = []
            for key, (message, received) in latest.items():
                stamp = stamp_ns(message)
                if not 0 <= now_mono - received <= MAX_AGE_NS:
                    issues.append(f"Stale received observation: {key}")
                if not 0 <= now_ros - stamp <= MAX_AGE_NS:
                    issues.append(f"Stale/future ROS timestamp: {key} (age={(now_ros-stamp)/1e6:.3f} ms)")
                stamps.append(stamp)
            span = max(stamps) - min(stamps)
            if span > MAX_SPREAD_NS:
                issues.append(f"Multimodal observation timestamps are not synchronized (span={span/1e6:.3f} ms, limit={MAX_SPREAD_NS/1e6:.3f} ms)")
            if not issues:
                break
            timing_failure = "; ".join(issues)
        else:
            missing = sorted(set(TOPICS) - set(latest))
            if missing:
                raise TimeoutError(f"Missing ROS observations: {missing}")
            raise TimeoutError(f"No valid ROS observation snapshot within {timeout_s:g}s: {timing_failure}")
        state = sum((ordered_state(latest[f"{side}_{kind}"][0], side, kind == "hand")
                     for side, kind in (("left", "arm"), ("left", "hand"), ("right", "arm"), ("right", "hand"))), [])
        serialization_start = time.monotonic_ns()
        result = {
            "state": state,
            "images": {key: rgb_image(latest[key][0]) for key in ("cam0", "cam1", "cam2")},
            "topic_stamp_ns": {key: stamp_ns(value[0]) for key, value in latest.items()},
            "captured_monotonic_ns": now_mono,
        }
        result["timing"] = {"snapshot_ready_monotonic_ns": now_mono,
                            "observation_encoding_ms": (time.monotonic_ns()-serialization_start)/1e6}
        return result


def snapshot(timeout_s: float = 5.0) -> dict:
    """Standalone diagnostic compatibility; runtime uses ObservationProvider."""
    provider = ObservationProvider()
    try:
        return provider.snapshot(timeout_s)
    finally:
        provider.close()


def write_snapshot(result):
    start = time.monotonic_ns()
    payload = json.dumps(result, allow_nan=False)
    timing = {"json_serialization_ms": (time.monotonic_ns()-start)/1e6,
              "ros_payload_ready_monotonic_ns": time.monotonic_ns()}
    print(payload[:-1] + ', "serialization_timing": ' + json.dumps(timing) + '}', flush=True)


def persistent_main():
    """Read-only request/reply worker; idle time continues to spin subscriptions."""
    import select
    provider = ObservationProvider()
    try:
        while True:
            if not select.select([sys.stdin], [], [], 0)[0]:
                provider.rclpy.spin_once(provider.node, timeout_sec=0.01)
                continue
            line = sys.stdin.readline()
            if not line:
                break
            try:
                request = json.loads(line)
                if request.get("command") == "close":
                    break
                write_snapshot(provider.snapshot(float(request.get("timeout_s", 5.0))))
            except Exception as error:
                print(json.dumps({"error": str(error)}), flush=True)
    finally:
        provider.close()


if __name__ == "__main__":
    if "--persistent" in sys.argv:
        persistent_main()
    else:
        try:
            write_snapshot(snapshot())
        except Exception as error:
            print(f"observation snapshot failed: {error}", file=sys.stderr)
            raise
