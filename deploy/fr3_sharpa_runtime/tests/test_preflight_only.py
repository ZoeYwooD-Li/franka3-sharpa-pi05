import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime import config
from deploy.fr3_sharpa_runtime.feedback import LiveFeedback
from deploy.fr3_sharpa_runtime.live_preflight import inspect_live_preflight
from deploy.fr3_sharpa_runtime.protocol import as_finite_action_chunk
from deploy.fr3_sharpa_runtime.ros_observation import TOPICS


class ReadOnlyGraph:
    def get_node_names(self):
        return ["teleop_safety_gateway", "sharpa_driver"]

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=time.time_ns()))

    def get_topic_names_and_types(self):
        topics = [(topic, ["sensor_msgs/msg/Image" if key.startswith("cam") else "sensor_msgs/msg/JointState"])
                  for key, topic in TOPICS.items()]
        return topics + [(config.FR3_TOPIC, ["teleop_interfaces/msg/ArmCommand"]),
                         (config.SHARPA_LEFT_TOPIC, ["sensor_msgs/msg/JointState"]),
                         (config.SHARPA_RIGHT_TOPIC, ["sensor_msgs/msg/JointState"])]

    def get_subscriptions_info_by_topic(self, topic):
        if topic == config.FR3_TOPIC:
            return [SimpleNamespace(node_name="teleop_safety_gateway")]
        if topic in (config.SHARPA_LEFT_TOPIC, config.SHARPA_RIGHT_TOPIC):
            return [SimpleNamespace(node_name="sharpa_driver")]
        return []

    def get_publishers_info_by_topic(self, topic):
        if topic in TOPICS.values():
            return [SimpleNamespace(node_name="state_or_camera_driver")]
        if topic in ("/sharpa/left/command", "/sharpa/right/command"):
            return [SimpleNamespace(node_name="sharpa_driver")]
        return []

    def create_publisher(self, *args, **kwargs):
        raise AssertionError("preflight-only must not create command publishers")


@pytest.fixture
def readonly_probe(monkeypatch):
    import deploy.fr3_sharpa_runtime.ros_transport as transport
    import deploy.fr3_sharpa_runtime.live_preflight as probe

    monkeypatch.setitem(sys.modules, "rclpy", SimpleNamespace(spin_once=lambda *a, **kw: None))
    parameter_checks = []
    monkeypatch.setattr(transport, "_require_parameter", lambda *args: parameter_checks.append(args[1:]))
    monkeypatch.setattr(transport, "validate_action_ranges", as_finite_action_chunk)
    monkeypatch.setattr(transport, "prepare_messages", lambda *args: (SimpleNamespace(unit="rad"), [None] * 50))
    state = np.zeros(58, dtype=np.float32)
    closed = []

    class Feedback:
        def __init__(self, node):
            pass

        def wait(self):
            return LiveFeedback(state, (time.time_ns(),) * 4, (time.monotonic_ns(),) * 4)

        def close(self):
            closed.append(True)

    monkeypatch.setattr(probe, "ROSFeedbackProvider", Feedback)
    return state, parameter_checks, closed


def test_preflight_only_pass_never_creates_command_publishers(readonly_probe):
    state, parameter_checks, closed = readonly_probe
    result = inspect_live_preflight(ReadOnlyGraph(), np.tile(state, (50, 1)), discovery_timeout_s=0)
    assert result["status"] == "PASS"
    assert all(result["checks"][key]["status"] == "PASS" for key in ("feedback", "gateway", "guarded_runtime", "action_valid"))
    assert result["published"] is False and result["command_publishers_created"] == 0
    assert len(parameter_checks) == 2 and closed == [True]


def test_preflight_only_reports_missing_live_nodes(readonly_probe):
    class EmptyGraph(ReadOnlyGraph):
        def get_node_names(self):
            return []

    result = inspect_live_preflight(EmptyGraph(), discovery_timeout_s=0)
    assert result["status"] == "FAIL" and result["readiness_pass"] is False
    assert "teleop_safety_gateway" in result["checks"]["gateway"]["reason"]
    assert "sharpa_driver" in result["checks"]["guarded_runtime"]["reason"]
    assert result["checks"]["action_valid"]["status"] == "NOT_CHECKED"
    assert result["command_publishers_created"] == 0


def test_preflight_only_reports_excessive_live_initial_delta(readonly_probe):
    state, _, _ = readonly_probe
    actions = np.tile(state, (50, 1))
    actions[0, 7] = 0.2
    result = inspect_live_preflight(ReadOnlyGraph(), actions, discovery_timeout_s=0)
    assert result["status"] == "FAIL"
    assert "first-target delta" in result["checks"]["action_valid"]["reason"]
    assert result["published"] is False


def test_preflight_only_reports_incompatible_arm_message_type(readonly_probe):
    class WrongTypeGraph(ReadOnlyGraph):
        def get_topic_names_and_types(self):
            return [(topic, ["sensor_msgs/msg/JointState"] if topic == config.FR3_TOPIC else types)
                    for topic, types in super().get_topic_names_and_types()]

    result = inspect_live_preflight(WrongTypeGraph(), discovery_timeout_s=0)
    assert result["checks"]["gateway"]["status"] == "FAIL"
    assert "Message type mismatch" in result["checks"]["gateway"]["reason"]
    assert result["command_publishers_created"] == 0


def test_cli_missing_graph_does_not_infer_or_load_episode(monkeypatch, capsys):
    from deploy.fr3_sharpa_runtime import run

    report = {"status": "FAIL", "readiness_pass": False, "checks": {}, "failures": {"gateway": "absent"},
              "published": False, "command_publishers_created": 0}
    monkeypatch.setattr(run, "ros_boundary", lambda *args: report)
    monkeypatch.setattr(run, "load_sample", lambda *args: pytest.fail("recorded samples are forbidden"))
    monkeypatch.setattr(run, "SharpaRuntime", lambda *args: pytest.fail("policy must wait for real observations"))
    monkeypatch.setattr(sys, "argv", ["run", "--preflight-only", "--ros-overlay", "/unused-overlay"])
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["published"] is False


@pytest.mark.parametrize("forbidden", [["--enable-hardware"], ["--episode-parquet", "/episode.parquet"], ["--execute"]])
def test_cli_preflight_only_forbids_hardware_or_recorded_input(monkeypatch, forbidden):
    from deploy.fr3_sharpa_runtime import run

    monkeypatch.setattr(run, "ros_boundary", lambda *args: pytest.fail("must reject before ROS boundary"))
    monkeypatch.setattr(sys, "argv", ["run", "--preflight-only", "--ros-overlay", "/unused-overlay", *forbidden])
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2
