"""Live ROS inspection only. No command publishers, SDK calls, or permit changes."""

from __future__ import annotations

import os
import time

from . import config
from .feedback import FEEDBACK_KEYS, ROSFeedbackProvider
from .preflight import execute_preflight
from .ros_observation import TOPICS


def inspect_live_preflight(node, actions=None, session_id="preflight", *, discovery_timeout_s=3.0) -> dict:
    import rclpy
    from .ros_transport import (
        _require_parameter, prepare_messages, validate_action_ranges,
        validate_gateway_graph, validate_guarded_graph,
    )

    provider = ROSFeedbackProvider(node)
    try:
        # The same 3 s discovery window already used by execute.
        deadline = time.monotonic() + discovery_timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())))
        types = dict(node.get_topic_names_and_types())
        expected = {
            topic: "sensor_msgs/msg/Image" if key.startswith("cam") else "sensor_msgs/msg/JointState"
            for key, topic in TOPICS.items()
        }
        expected.update({config.FR3_TOPIC: "teleop_interfaces/msg/ArmCommand",
                         config.SHARPA_LEFT_TOPIC: "sensor_msgs/msg/JointState",
                         config.SHARPA_RIGHT_TOPIC: "sensor_msgs/msg/JointState"})
        # Downstream entries are evidence, not new gating rules.
        inventory_topics = list(expected) + [
            "/target_robot/joint_commands", "/left/gello/joint_states", "/right/gello/joint_states",
            "/sharpa/left/command", "/sharpa/right/command", "/teleop/sharpa/supervisor/state",
        ]
        inventory = {
            topic: {
                "expected_type": expected.get(topic), "observed_types": types.get(topic, []),
                "publishers": [item.node_name for item in node.get_publishers_info_by_topic(topic)],
                "subscribers": [item.node_name for item in node.get_subscriptions_info_by_topic(topic)],
            }
            for topic in inventory_topics
        }
        checks = {}
        for label, validate, remote, parameter, required in (
            ("gateway", validate_gateway_graph, "teleop_safety_gateway", "allowed_sources", [config.SOURCE]),
            ("guarded_runtime", validate_guarded_graph, "sharpa_driver", "supervised_validation_guard", True),
        ):
            try:
                validate(node)
                interface_topics = ([config.FR3_TOPIC] if label == "gateway" else
                                    [config.SHARPA_LEFT_TOPIC, config.SHARPA_RIGHT_TOPIC])
                for topic in interface_topics:
                    if expected[topic] not in types.get(topic, []):
                        raise RuntimeError(f"Message type mismatch on {topic}: expected {expected[topic]}, observed {types.get(topic, [])}")
                _require_parameter(node, remote, parameter, required)
                checks[label] = {"status": "PASS", "node": remote, "parameter": parameter}
            except Exception as error:
                checks[label] = {"status": "FAIL", "reason": str(error)}

        camera_failures = []
        for key in ("cam0", "cam1", "cam2"):
            entry = inventory[TOPICS[key]]
            if entry["expected_type"] not in entry["observed_types"] or not entry["publishers"]:
                camera_failures.append(f"{TOPICS[key]}: no RGB Image publisher discovered")
        checks["camera_interfaces"] = {
            "status": "FAIL" if camera_failures else "PASS", "issues": camera_failures,
            "scope": "graph interfaces; actual image/state snapshot is checked by the policy CLI",
        }

        feedback = None
        try:
            feedback = provider.wait()
            now_ros = node.get_clock().now().nanoseconds
            now_mono = time.monotonic_ns()
            feedback.validate_freshness(now_ros_ns=now_ros, now_monotonic_ns=now_mono)
            checks["feedback"] = {
                "status": "PASS", "state_shape": list(feedback.state.shape),
                "dimensions": [7, 22, 7, 22],
                "ros_age_ms": {key: (now_ros - stamp) / 1e6 for key, stamp in zip(FEEDBACK_KEYS, feedback.ros_stamps_ns, strict=True)},
                "received_age_ms": {key: (now_mono - stamp) / 1e6 for key, stamp in zip(FEEDBACK_KEYS, feedback.received_monotonic_ns, strict=True)},
            }
        except Exception as error:
            checks["feedback"] = {"status": "FAIL", "reason": str(error)}

        if actions is None:
            checks["action_valid"] = {"status": "NOT_CHECKED", "reason": "waiting for real observation and policy inference"}
        else:
            try:
                validate_action_ranges(actions)
                if feedback is None:
                    raise ValueError("No fresh live feedback; cannot validate the first action")
                # Reuse execute's graph and first-target checks at the current time.
                validate_gateway_graph(node)
                validate_guarded_graph(node)
                check = execute_preflight(
                    actions, feedback, now_ros_ns=node.get_clock().now().nanoseconds,
                    now_monotonic_ns=time.monotonic_ns(),
                )
                chunk, messages = prepare_messages(actions, session_id, node.get_clock().now().nanoseconds)
                checks["action_valid"] = {"status": "PASS", "shape": [50, 58], "finite": True,
                                          "execute_preflight": check, "message_frames": len(messages),
                                          "timestamp_step_ns": 50_000_000, "unit": chunk.unit}
            except Exception as error:
                checks["action_valid"] = {"status": "FAIL", "reason": str(error)}

        readiness_pass = all(checks[key]["status"] == "PASS" for key in ("gateway", "guarded_runtime", "camera_interfaces", "feedback"))
        failed = {key: value for key, value in checks.items() if value["status"] == "FAIL"}
        return {
            "mode": "preflight-only", "status": "FAIL" if failed else "PENDING" if actions is None else "PASS",
            "readiness_pass": readiness_pass, "published": False, "command_publishers_created": 0,
            "checks": checks, "failures": failed,
            "ros_environment": {key: os.environ.get(key, "<unset>") for key in (
                "ROS_DOMAIN_ID", "ROS_LOCALHOST_ONLY", "RMW_IMPLEMENTATION", "CYCLONEDDS_URI")},
            "nodes": node.get_node_names(), "topic_inventory": inventory,
        }
    finally:
        provider.close()
