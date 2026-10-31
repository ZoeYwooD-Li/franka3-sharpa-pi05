"""Capture one real T-Rex observation using subscriptions only. No SDK/publishers."""

import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np

from ..ros_observation import MAX_SPREAD_NS, ObservationProvider
from .ros_tactile import RosTactileBuffer
from .tactile import ROUTES


def capture(output, *, timeout_s=15.0):
    from geometry_msgs.msg import WrenchStamped
    from sensor_msgs.msg import Image
    from rclpy.qos import qos_profile_sensor_data

    provider = ObservationProvider(startup_stable_s=2.0)
    tactile = RosTactileBuffer()
    node = provider.node
    try:
        for route in ROUTES:
            for kind, cls in (("wrench", WrenchStamped), ("deformation", Image)):
                def callback(message, r=route, k=kind):
                    tactile.remember(r, k, message,
                                     now_ros_ns=node.get_clock().now().nanoseconds,
                                     now_monotonic_ns=time.monotonic_ns())
                provider.subscriptions.append(node.create_subscription(
                    cls, f"/sharpa/{route[0]}/tactile/{route[1]}/{kind}",
                    callback, qos_profile_sensor_data))
        deadline = time.monotonic() + timeout_s
        last_error = "Waiting for observations"
        while time.monotonic() < deadline:
            try:
                obs = provider.snapshot(timeout_s=min(1.0, deadline - time.monotonic()))
                data = tactile.snapshot(now_ros_ns=node.get_clock().now().nanoseconds,
                                        now_monotonic_ns=time.monotonic_ns())
                stamps = list(obs["topic_stamp_ns"].values()) + data["tactile_stamp_ns"][-1].tolist()
                if max(stamps) - min(stamps) > MAX_SPREAD_NS:
                    raise ValueError("Camera/state/tactile timestamps are not synchronized")
                break
            except (ValueError, TimeoutError) as error:
                last_error = str(error)
        else:
            raise TimeoutError(f"No complete T-Rex observation: {last_error}")
        data["state"] = np.asarray(obs["state"], dtype=np.float32)
        for camera, image in obs["images"].items():
            data[camera] = np.frombuffer(base64.b64decode(image["rgb_b64"]), dtype=np.uint8).reshape(image["shape"]).copy()
        data["metadata_json"] = np.asarray(json.dumps({
            "source": "live_ros", "published": False,
            "timestamp_basis": "host_acquisition_ros_header_not_sdk_frame_id",
            "routes": ROUTES, "topic_stamp_ns": obs["topic_stamp_ns"],
            "captured_monotonic_ns": obs["captured_monotonic_ns"],
            "alignment": obs["alignment"],
        }))
        # Exclusive creation prevents replacing an earlier diagnostic recording.
        with Path(output).open("xb") as stream:
            np.savez_compressed(stream, **data)
        return {"output": str(Path(output).resolve()), "published": False,
                "f6_shape": list(data["tactile_f6"].shape),
                "tactile_source_span_ms": float(np.ptp(data["tactile_stamp_ns"][-1]) / 1e6)}
    finally:
        provider.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args(argv)
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    print(json.dumps(capture(args.output, timeout_s=args.timeout)))


if __name__ == "__main__":
    main()
