"""Task-pose preparation contract; no guessed targets or hardware backend."""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol
import numpy as np

from .feedback import LiveFeedback
from .protocol import FR3_COMMAND_NAMES, LEFT_SHARPA_NAMES, RIGHT_SHARPA_NAMES

GROUP_NAMES = {"left_fr3": FR3_COMMAND_NAMES[:7], "left_sharpa": LEFT_SHARPA_NAMES,
               "right_fr3": FR3_COMMAND_NAMES[7:], "right_sharpa": RIGHT_SHARPA_NAMES}


@dataclass(frozen=True)
class PrepareConfig:
    pose_name: str
    target: np.ndarray
    backend: str
    tolerance_rad: float
    stability_delta_rad: float
    stable_duration_s: float
    timeout_s: float

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text())
        if data.get("unit") != "rad" or data.get("representation") != "absolute_joint_position":
            raise ValueError("Prepare pose must use absolute joint positions in rad")
        positions = []
        for group, names in GROUP_NAMES.items():
            entry = data["groups"][group]
            if entry.get("joint_names") != list(names):
                raise ValueError(f"Prepare joint order mismatch: {group}")
            values = entry.get("positions")
            if not isinstance(values, list) or len(values) != len(names):
                raise ValueError(f"Task initial pose not configured: {group}; no angles are inferred")
            positions.extend(values)
        target = np.asarray(positions, dtype=np.float32)
        if target.shape != (58,) or not np.isfinite(target).all():
            raise ValueError("Prepare pose requires 58 finite explicitly configured positions")
        numeric = [float(data[k]) for k in ("tolerance_rad", "stability_delta_rad", "stable_duration_s", "timeout_s")]
        if not all(np.isfinite(v) and v > 0 for v in numeric) or numeric[2] >= numeric[3]:
            raise ValueError("Invalid prepare settling configuration")
        if not data.get("pose_name") or data.get("backend") not in ("verify_only", "guarded_motion"):
            raise ValueError("Prepare requires named pose and explicit backend")
        return cls(data["pose_name"], target, data["backend"], *numeric)


class PrepareBackend(Protocol):
    """Future motion implementation MUST use the existing two guarded paths.

    FR3: /teleop/arm_commands -> Safety Gateway.
    Sharpa: /teleop/sharpa/{left,right}/openpi_target -> guarded runtime.
    Backend must retain authorization, feedback, timestamp and native limits.
    Raw controller/reset topics and /sharpa/*/command are not valid backends.
    """
    def check_interfaces(self) -> None: ...
    def read_feedback(self) -> LiveFeedback: ...
    def now_monotonic_ns(self) -> int: ...
    def now_ros_ns(self) -> int: ...
    def move_to_pose(self, configuration: PrepareConfig, initial: LiveFeedback) -> None: ...


def prepare_pose(configuration: PrepareConfig, backend: PrepareBackend) -> dict:
    backend.check_interfaces()
    def read():
        feedback = backend.read_feedback()
        feedback.validate_freshness(now_ros_ns=backend.now_ros_ns(),
                                    now_monotonic_ns=backend.now_monotonic_ns())
        return feedback
    start = backend.now_monotonic_ns()
    initial = read()
    if np.max(np.abs(initial.state-configuration.target)) > configuration.tolerance_rad:
        backend.move_to_pose(configuration, initial)
    stable_since = None
    previous = initial.state
    while backend.now_monotonic_ns()-start < configuration.timeout_s*1e9:
        current = read().state
        now = backend.now_monotonic_ns()
        error = float(np.max(np.abs(current-configuration.target)))
        delta = float(np.max(np.abs(current-previous)))
        if error <= configuration.tolerance_rad and delta <= configuration.stability_delta_rad:
            stable_since = now if stable_since is None else stable_since
            if now-stable_since >= configuration.stable_duration_s*1e9:
                return {"status": "PASS", "pose_name": configuration.pose_name,
                        "backend": configuration.backend, "maximum_pose_error_rad": error,
                        "completed_monotonic_ns": now, "completed_ros_ns": backend.now_ros_ns()}
        else:
            stable_since = None
        previous = current
    raise TimeoutError("Prepare failed: task pose did not settle; execute is blocked")


def prepare_then_observe(configuration, backend, acquire_observation):
    result = prepare_pose(configuration, backend)
    observation = acquire_observation(result)
    if observation["_captured_monotonic_ns"] < result["completed_monotonic_ns"]:
        raise ValueError("Prepare requires a newly acquired observation after settling")
    return result, observation
