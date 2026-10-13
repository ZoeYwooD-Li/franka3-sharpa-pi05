"""Capture a manually positioned 58D task pose from live ROS feedback only."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from .feedback import FEEDBACK_KEYS, FEEDBACK_TOPICS, ROSFeedbackProvider
from .prepare import GROUP_NAMES, PrepareConfig


def make_config(feedback, template, *, pose_name, ros_now_ns, monotonic_now_ns):
    feedback.validate_freshness(now_ros_ns=ros_now_ns, now_monotonic_ns=monotonic_now_ns)
    document = json.loads(json.dumps(template))
    document.update(pose_name=pose_name, backend="verify_only", unit="rad",
                    representation="absolute_joint_position")
    groups, sources = {}, {}
    offset = 0
    for (group, names), key, stamp, received in zip(
            GROUP_NAMES.items(), FEEDBACK_KEYS, feedback.ros_stamps_ns,
            feedback.received_monotonic_ns, strict=True):
        groups[group] = {"joint_names": list(names),
                         "positions": feedback.state[offset:offset+len(names)].tolist()}
        sources[group] = {"topic": FEEDBACK_TOPICS[key], "message_type": "sensor_msgs/msg/JointState",
                          "header_stamp_ns": stamp, "received_monotonic_ns": received}
        offset += len(names)
    document["groups"] = groups
    document["capture"] = {
        "source": "live_ros_feedback", "protocol": "left_fr3_7,left_sharpa_22,right_fr3_7,right_sharpa_22",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "captured_ros_timestamp_ns": ros_now_ns, "captured_monotonic_timestamp_ns": monotonic_now_ns,
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
        "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION", ""), "sources": sources,
        "timestamp_span_ns": max(feedback.ros_stamps_ns)-min(feedback.ros_stamps_ns),
        "joint_name_mapping": "FR3 state names fr3_joint1..7 or fr3v2_joint1..7 are read by name and saved as fr3v2 command names; Sharpa uses the frozen Wave 22D order",
    }
    return document


def save_config(document, output, *, overwrite=False):
    # Validate the fully captured document before writing the requested file.
    output.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    descriptor, candidate = tempfile.mkstemp(prefix=".captured_pose-", suffix=".json", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
        PrepareConfig.load(Path(candidate))
        if overwrite:
            os.replace(candidate, output)
        else:
            # Atomic create without overwriting an existing confirmed pose.
            os.link(candidate, output)
    finally:
        Path(candidate).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-name", required=True, help="Operator-defined name for the manually reached task start pose")
    parser.add_argument("--output", type=Path, default=Path.home()/".config/fr3_sharpa_runtime/prepare_config.json")
    parser.add_argument("--template", type=Path, default=Path(__file__).with_name("prepare_config.example.json"))
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing captured pose")
    args = parser.parse_args()
    if not args.pose_name.strip() or args.timeout <= 0:
        parser.error("Nonempty pose name and positive timeout required")
    if args.output.exists() and not args.overwrite:
        parser.error(f"Output already exists: {args.output}; choose another file or --overwrite")
    template = json.loads(args.template.read_text())
    import rclpy
    rclpy.init()
    node = rclpy.create_node("fr3_sharpa_capture_prepare_pose")
    provider = ROSFeedbackProvider(node)
    try:
        feedback = provider.wait(timeout_s=args.timeout)
        document = make_config(feedback, template, pose_name=args.pose_name,
                               ros_now_ns=node.get_clock().now().nanoseconds,
                               monotonic_now_ns=time.monotonic_ns())
        save_config(document, args.output, overwrite=args.overwrite)
        print(json.dumps({"status": "PASS", "output": str(args.output.resolve()),
                          "dimensions": [7,22,7,22], "backend": "verify_only",
                          "capture": document["capture"], "published": False}, indent=2))
    finally:
        provider.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
