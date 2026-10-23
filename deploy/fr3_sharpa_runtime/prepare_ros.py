"""Read-only task-pose verification entry; guarded motion backend is pending."""
import argparse
import json
from pathlib import Path
import time

from .feedback import ROSFeedbackProvider
from .prepare import PrepareConfig, prepare_pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    settings = PrepareConfig.load(args.config)
    if settings.backend == "guarded_motion":
        raise RuntimeError("Guarded prepare motion backend is not connected. Existing FR3 reset bypasses Safety Gateway; no Sharpa task-pose interface exists. Execute blocked.")
    import rclpy
    from .ros_transport import validate_execution_graph, _require_parameter
    from . import config
    rclpy.init()
    node = rclpy.create_node("fr3_sharpa_prepare_verification")
    feedback = ROSFeedbackProvider(node)
    class VerifyOnlyBackend:
        def check_interfaces(self):
            deadline = time.monotonic()+3
            while time.monotonic()<deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
            validate_execution_graph(node)
            _require_parameter(node, "teleop_safety_gateway", "allowed_sources", [config.SOURCE])
            _require_parameter(node, "sharpa_driver", "supervised_validation_guard", True)
        def read_feedback(self):
            # Settle sampling uses the existing 30 Hz timeline.
            rclpy.spin_once(node, timeout_sec=1/config.ACTION_HZ)
            return feedback.wait()
        def now_monotonic_ns(self): return time.monotonic_ns()
        def now_ros_ns(self): return node.get_clock().now().nanoseconds
        def move_to_pose(self, configuration, initial):
            raise RuntimeError("verify_only cannot move the robot; confirmed task pose is not reached. Execute blocked.")
    try:
        report = prepare_pose(settings, VerifyOnlyBackend())
        report["published"] = False
        print(json.dumps(report))
    finally:
        feedback.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
