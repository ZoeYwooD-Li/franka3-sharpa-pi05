"""Hardware-free integration of prepare, observations, policy and stream playback."""
import json
import sys

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.action_adapter import SharpaActionAdapter
from deploy.fr3_sharpa_runtime.execution_trace import ExecutionTraceWriter
from deploy.fr3_sharpa_runtime.feedback import LiveFeedback
from deploy.fr3_sharpa_runtime.policy_prepare import SlowApproach, prepare_and_run_chunks
from deploy.fr3_sharpa_runtime.sdk_direct import chunk_command_frames, play_command_frames


class SimulatedRobot:
    def __init__(self):
        self.now = 1.0
        self.state = np.zeros(58, dtype=np.float32)
        self.operations = []
        self.sent = []

    def clock(self): return self.now
    def sleep(self, duration): self.now += duration
    def feedback(self):
        stamp = int(self.now * 1e9)
        return LiveFeedback(self.state.copy(), (stamp,) * 4, (stamp,) * 4)

    def observe(self, report=None):
        return {"observation/state": self.state.copy(), "prompt": "tomato",
                "_captured_monotonic_ns": int(self.now * 1e9)}

    def request(self, payload):
        self.operations.append(payload["operation"])
        if payload["operation"] == "prepare":
            planner = SlowApproach(payload["target"], self.state,
                                   speed_rad_s=payload["speed_rad_s"])
            for _ in range(10000):
                stamp = int(self.now * 1e9)
                self.state, settled, _ = planner.step(self.feedback(), now_ros_ns=stamp,
                    now_monotonic_ns=stamp, enforce_execute_delta=False)
                self.now += 0.05
                if settled:
                    break
            else:
                pytest.fail("prepare did not settle")
        else:
            frames, _ = chunk_command_frames(payload["actions"], self.state,
                speed_scale=payload["execute_speed_scale"],
                approach_speed_rad_s=payload["approach_speed_rad_s"])
            def send(frame):
                self.state = frame.copy()
                self.sent.append(frame.copy())
            play_command_frames(frames, send, self.feedback, clock=self.clock, sleep=self.sleep)
        return {"status": "PASS", "published": True,
                "completed_ros_ns": int(self.now * 1e9),
                "completed_monotonic_ns": int(self.now * 1e9)}


def policy(obs):
    actions = np.repeat(obs["observation/state"][None, :], 50, axis=0)
    actions += np.linspace(0.005, 0.03, 50, dtype=np.float32)[:, None]
    chunk = SharpaActionAdapter().adapt_chunk(actions, start_timestamp_ns=1_000_000_000,
                                             session_id="test", source="openpi")
    return {"checkpoint": "test", "norm_stats": "test"}, actions, chunk


def test_prepare_then_five_complete_chunks_reobserve_actual_end_state(tmp_path):
    robot = SimulatedRobot()
    traces = ExecutionTraceWriter(tmp_path)
    observations = []
    def infer(obs):
        observations.append(obs["observation/state"].copy())
        return policy(obs)
    report = prepare_and_run_chunks(infer, robot.observe, robot, robot.observe(),
        num_chunks=5, execute_speed_scale=0.1, trace_writer=traces)
    assert robot.operations == ["prepare"] + ["chunk"] * 5
    assert len(observations) == 6
    assert report["num_chunks"] == 5 and report["status"] == "COMPLETE"
    assert np.allclose(observations[1], 0.005)  # First inference only sets prepare.
    assert np.allclose(observations[-1], 0.125)
    assert np.allclose(robot.state, 0.155)
    assert len(traces.paths) == 5
    for index, path in enumerate(traces.paths):
        record = json.loads(open(path).read())
        assert record["status"] == "COMPLETE"
        assert record["chunk_index"] == index
        assert record["feedback_after"] is not None
        assert len(record["policy_output_unnormalized"]) == 50
    assert json.loads(traces.run_path.read_text())["num_chunks"] == 5


def test_continuous_mode_runs_beyond_two_chunks_and_records_interrupt(tmp_path):
    robot = SimulatedRobot()
    traces = ExecutionTraceWriter(tmp_path)
    def infer(obs):
        if robot.operations.count("chunk") == 3:
            raise KeyboardInterrupt
        return policy(obs)
    with pytest.raises(KeyboardInterrupt):
        prepare_and_run_chunks(infer, robot.observe, robot, robot.observe(),
                               num_chunks=0, trace_writer=traces)
    report = json.loads(traces.run_path.read_text())
    assert report["status"] == "INTERRUPTED" and report["num_chunks"] == 3
    assert robot.operations == ["prepare", "chunk", "chunk", "chunk"]
    # A failure before the next chunk must not overwrite the last completed trace.
    assert all(json.loads(open(path).read())["status"] == "COMPLETE" for path in traces.paths)


def test_motion_failure_records_reason_and_stops_next_inference(tmp_path):
    robot = SimulatedRobot()
    traces = ExecutionTraceWriter(tmp_path)
    robot_request = robot.request
    # Preserve the real prepare implementation while failing the chunk IPC.
    class BrokenTransport:
        def request(self, payload):
            if payload["operation"] == "chunk":
                raise RuntimeError("feedback link lost")
            return robot_request(payload)
    with pytest.raises(RuntimeError, match="feedback link lost"):
        prepare_and_run_chunks(policy, robot.observe, BrokenTransport(), robot.observe(),
                               num_chunks=5, trace_writer=traces)
    assert len(traces.paths) == 1
    record = json.loads(open(traces.paths[0]).read())
    assert record["status"] == "FAILED" and record["failure_stage"] == "execute"
    assert json.loads(traces.run_path.read_text())["num_chunks"] == 0


def test_execute_defaults_to_direct_policy_prepare_and_accepts_continuous(monkeypatch, capsys):
    from deploy.fr3_sharpa_runtime import run
    captured = {}
    def execute(args):
        captured.update(vars(args))
        return {"status": "test"}
    monkeypatch.setattr(run, "execute_policy_prepare", execute)
    monkeypatch.setattr(sys, "argv", ["run", "--execute", "--enable-hardware",
        "--ros-overlay", "/test", "--num-chunks", "0"])
    run.main()
    assert captured["sharpa_backend"] == "sdk_direct"
    assert captured["prepare_policy_first"] is True
    assert captured["num_chunks"] == 0


def test_initial_inference_failure_releases_sdk_worker(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from deploy.fr3_sharpa_runtime import run, motion_worker
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "setup.bash").write_text("")
    closed = []
    class Worker:
        def __init__(self, *args, **kwargs): pass
        def request(self, request): return {"operation": "ready", "status": "PASS"}
        def close(self): closed.append(True)
    class FailedPolicy:
        def __init__(self, *args): pass
        def infer_chunk(self, obs): raise RuntimeError("policy unavailable")
    monkeypatch.setattr(motion_worker, "MotionWorker", Worker)
    monkeypatch.setattr(run, "_start_observation_worker", lambda *args: None)
    monkeypatch.setattr(run, "live_observation", lambda *args: {})
    monkeypatch.setattr(run, "SharpaRuntime", FailedPolicy)
    monkeypatch.setattr(run.atexit, "register", lambda *args: None)
    args = SimpleNamespace(ros_overlay=overlay, sharpa_backend="sdk_direct",
                           execution_trace_dir=tmp_path, policy_url="test", prompt="tomato")
    with pytest.raises(RuntimeError, match="policy unavailable"):
        run.execute_policy_prepare(args)
    assert closed == [True]
    report = json.loads(next(tmp_path.glob("*-run.json")).read_text())
    assert report["stage"] == "startup" and report["status"] == "FAILED"


def test_post_chunk_sensor_failure_does_not_hide_transmission_completion(tmp_path):
    robot = SimulatedRobot()
    traces = ExecutionTraceWriter(tmp_path)
    def observe(report):
        if robot.operations.count("chunk"):
            raise TimeoutError("camera stopped")
        return robot.observe(report)
    with pytest.raises(TimeoutError, match="camera stopped"):
        prepare_and_run_chunks(policy, observe, robot, robot.observe(),
                               num_chunks=5, trace_writer=traces)
    record = json.loads(open(traces.paths[0]).read())
    assert record["status"] == "COMPLETE"
    assert record["failure_stage"] == "post_chunk_observation"
    report = json.loads(traces.run_path.read_text())
    assert report["num_chunks"] == 1 and report["status"] == "FAILED"
