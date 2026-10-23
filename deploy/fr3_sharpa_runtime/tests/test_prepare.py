from deploy.fr3_sharpa_runtime.protocol import DT_NS, FPS
import sys
from types import SimpleNamespace
import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.feedback import LiveFeedback
from deploy.fr3_sharpa_runtime.prepare import PrepareConfig, prepare_then_observe


class FakeGuardedBackend:
    def __init__(self, *, fail=False, moving=False):
        self.clock = 1_000_000_000
        self.state = np.ones(58, dtype=np.float32)
        self.events = []
        self.fail = fail
        self.moving = moving
    def check_interfaces(self): self.events.append("guarded_interfaces")
    def now_monotonic_ns(self): return self.clock
    def now_ros_ns(self): return self.clock+10_000_000_000
    def read_feedback(self):
        self.clock += DT_NS
        self.events.append("feedback")
        if self.moving:
            self.state[0] = 0.03 if self.state[0] == 0 else 0
        return LiveFeedback(self.state.copy(), (self.now_ros_ns(),)*4, (self.clock,)*4)
    def move_to_pose(self, configuration, initial):
        self.events.append("guarded_move_request")
        if self.fail: raise RuntimeError("Existing guard rejected prepare")
        self.state = configuration.target.copy()


def settings():
    return PrepareConfig("test_pose", np.zeros(58, dtype=np.float32), "guarded_motion", 0.01, 0.005, 0.1, 0.5)


def test_success_requires_settling_then_reacquires_observation():
    backend = FakeGuardedBackend()
    def acquire(report):
        backend.events.append("new_observation")
        return {"_captured_monotonic_ns": backend.clock+1}
    report, obs = prepare_then_observe(settings(), backend, acquire)
    assert report["status"] == "PASS"
    assert backend.events[-1] == "new_observation"
    assert backend.events.index("guarded_move_request") < len(backend.events)-1
    assert backend.events.count("feedback") >= 4


@pytest.mark.parametrize("backend", [FakeGuardedBackend(fail=True), FakeGuardedBackend(moving=True)])
def test_failed_prepare_does_not_acquire_observation_or_execute(backend):
    with pytest.raises((RuntimeError, TimeoutError)):
        prepare_then_observe(settings(), backend, lambda report: pytest.fail("must not enter observation/policy/execute"))


def test_cli_execute_cannot_continue_after_prepare_failure(monkeypatch):
    from deploy.fr3_sharpa_runtime import run
    monkeypatch.setattr(sys, "argv", ["run", "--execute", "--enable-hardware", "--sharpa-backend", "litchibot", "--prepare-config", "/pose.json", "--ros-overlay", "/overlay"])
    def reject(*args): raise RuntimeError("prepare rejected")
    monkeypatch.setattr(run, "prepare_before_observation", reject)
    monkeypatch.setattr(run, "SharpaRuntime", lambda *args: pytest.fail("policy must not run"))
    monkeypatch.setattr(run, "ros_boundary", lambda *args: pytest.fail("execute must not run"))
    with pytest.raises(RuntimeError, match="prepare rejected"):
        run.main()


def test_template_has_no_guessed_angles():
    from pathlib import Path
    template = Path(__file__).resolve().parents[1] / "prepare_config.example.json"
    with pytest.raises(ValueError, match="Task initial pose not configured"):
        PrepareConfig.load(template)


def test_runtime_rejects_cached_observation_after_settling(monkeypatch, tmp_path):
    import json
    from deploy.fr3_sharpa_runtime import run
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "setup.bash").write_text("")
    args = SimpleNamespace(prepare_config=tmp_path / "pose.json", ros_overlay=overlay, prompt="tomato")
    monkeypatch.setattr(PrepareConfig, "load", lambda path: settings())
    completed = {"status": "PASS", "completed_ros_ns": 100, "completed_monotonic_ns": 200}
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps(completed)))
    observations = [
        {"_topic_stamp_ns": dict.fromkeys(range(7), 99), "_captured_monotonic_ns": 201},
        {"_topic_stamp_ns": dict.fromkeys(range(7), 101), "_captured_monotonic_ns": 202},
    ]
    calls = []
    def acquire(*args):
        calls.append(True)
        return observations.pop(0)
    monkeypatch.setattr(run, "live_observation", acquire)
    report, observation = run.prepare_before_observation(args)
    assert len(calls) == 2
    assert min(observation["_topic_stamp_ns"].values()) == 101
