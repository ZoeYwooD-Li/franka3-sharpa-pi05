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


def snapshot(timeout_s: float = 5.0) -> dict:
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, JointState

    rclpy.init()
    node = rclpy.create_node("fr3_sharpa_observation")
    latest = {}
    subscriptions = []
    try:
        for key, topic in TOPICS.items():
            kind = Image if key.startswith("cam") else JointState
            subscriptions.append(node.create_subscription(
                kind, topic,
                lambda message, selected=key: latest.__setitem__(selected, (message, time.monotonic_ns())),
                qos_profile_sensor_data,
            ))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if len(latest) == len(TOPICS):
                break
        if len(latest) != len(TOPICS):
            raise TimeoutError(f"Missing ROS observations: {sorted(set(TOPICS)-set(latest))}")
        # Snapshot time is after receipt of all modalities: no future observations.
        now_mono = time.monotonic_ns()
        now_ros = node.get_clock().now().nanoseconds
        stamps = []
        for key, (message, received) in latest.items():
            stamp = stamp_ns(message)
            if not 0 <= now_mono - received <= MAX_AGE_NS:
                raise ValueError(f"Stale received observation: {key}")
            if not 0 <= now_ros - stamp <= MAX_AGE_NS:
                raise ValueError(f"Stale/future ROS timestamp: {key}")
            stamps.append(stamp)
        if max(stamps) - min(stamps) > MAX_SPREAD_NS:
            raise ValueError("Multimodal observation timestamps are not synchronized")
        state = sum((ordered_state(latest[f"{side}_{kind}"][0], side, kind == "hand")
                     for side, kind in (("left", "arm"), ("left", "hand"), ("right", "arm"), ("right", "hand"))), [])
        return {
            "state": state,
            "images": {key: rgb_image(latest[key][0]) for key in ("cam0", "cam1", "cam2")},
            "topic_stamp_ns": {key: stamp_ns(value[0]) for key, value in latest.items()},
            "captured_monotonic_ns": now_mono,
        }
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    try:
        print(json.dumps(snapshot(), allow_nan=False))
    except Exception as error:
        print(f"observation snapshot failed: {error}", file=sys.stderr)
        raise
