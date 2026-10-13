"""Slow initialization planner. Targets are explicit policy positions in rad."""
from dataclasses import dataclass
import numpy as np
import time

from .feedback import LiveFeedback
from .preflight import execute_preflight
from .protocol import (
    DT_NS, LEFT_FR3_SLICE, LEFT_SHARPA_SLICE, RIGHT_FR3_SLICE,
    RIGHT_SHARPA_SLICE, as_finite_action_chunk,
)


ERROR_GROUPS = {
    "left_fr3": LEFT_FR3_SLICE,
    "left_sharpa": LEFT_SHARPA_SLICE,
    "right_fr3": RIGHT_FR3_SLICE,
    "right_sharpa": RIGHT_SHARPA_SLICE,
}


def target_chunk(target):
    target = np.asarray(target, dtype=np.float32)
    if target.shape != (58,) or not np.isfinite(target).all():
        raise ValueError("Prepare requires a finite 58D absolute target in rad")
    return as_finite_action_chunk(np.repeat(target[None, :], 50, axis=0))


def check_validation_excursion(target, current, *, mode, excursion_rad, source='openpi'):
    """Reuse an existing guarded-mode bound, not a replacement authorization."""
    from .protocol import LEFT_SHARPA_SLICE, RIGHT_SHARPA_SLICE
    # Match validation_safety.py: OpenPI task prepare is exempt from the
    # acceptance baseline bound; acquisition/continuity/slew remain enforced.
    if mode == 'supervised_hardware_validation' and source != 'openpi':
        for side, part in [('left', LEFT_SHARPA_SLICE), ('right', RIGHT_SHARPA_SLICE)]:
            delta = float(np.max(np.abs(target_chunk(target)[0, part]-target_chunk(current)[0, part])))
            if delta > excursion_rad:
                raise ValueError(f"Existing supervised validation total excursion {excursion_rad} rad blocks {side} prepare displacement {delta:.6f} rad; slow steps do not bypass authorization")


@dataclass
class SlowApproach:
    target: np.ndarray
    previous: np.ndarray
    speed_rad_s: float = 0.03
    tolerance_rad: float = 0.01
    stability_delta_rad: float = 0.005
    stable_duration_s: float = 0.5
    timeout_s: float = 60.0

    def __post_init__(self):
        self.target = target_chunk(self.target)[0].copy()
        self.previous = target_chunk(self.previous)[0].copy()
        values = (self.speed_rad_s, self.tolerance_rad, self.stability_delta_rad,
                  self.stable_duration_s, self.timeout_s)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("Prepare motion parameters must be finite and positive")
        self.started_ns = None
        self.stable_since_ns = None
        self.last_feedback = self.previous.copy()
        travel_time = float(np.max(np.abs(self.target - self.previous))) / self.speed_rad_s
        self.effective_timeout_s = (
            max(self.timeout_s, 120.0, travel_time * 2.0 + self.stable_duration_s + 5.0)
            if self.timeout_s == 60.0 else self.timeout_s
        )

    def step(self, feedback: LiveFeedback, *, now_ros_ns: int, now_monotonic_ns: int,
             enforce_execute_delta: bool = True):
        feedback.validate_freshness(now_ros_ns=now_ros_ns, now_monotonic_ns=now_monotonic_ns)
        if self.started_ns is None:
            self.started_ns = now_monotonic_ns
        if now_monotonic_ns-self.started_ns >= self.effective_timeout_s*1e9:
            errors = {
                name: float(np.max(np.abs(self.last_feedback[part] - self.target[part])))
                for name, part in ERROR_GROUPS.items()
            }
            remaining = max(errors.values())
            raise TimeoutError(
                "Prepare did not reach and settle at policy first target; "
                f"chunks blocked (remaining_error_rad={remaining:.6f}, "
                f"timeout_s={self.effective_timeout_s:.1f}, "
                f"error_by_group={errors})"
            )
        # A straight joint-space trajectory; never alter the policy target.
        # Progress only one nominal 20Hz interval even if scheduling is delayed.
        error = self.target-self.previous
        maximum = float(np.max(np.abs(error)))
        alpha = min(1.0, self.speed_rad_s*(DT_NS/1e9)/maximum) if maximum else 1.0
        command = self.previous+alpha*error
        # Prepare is deliberately allowed to bridge a large current-to-policy
        # displacement.  It still requires fresh feedback; execute's first
        # target delta gate is applied only after preparation completes.
        if enforce_execute_delta:
            execute_preflight(target_chunk(command), feedback, now_ros_ns=now_ros_ns,
                              now_monotonic_ns=now_monotonic_ns)
        measured_error = float(np.max(np.abs(feedback.state-self.target)))
        stable = float(np.max(np.abs(feedback.state-self.last_feedback))) <= self.stability_delta_rad
        at_command_target = np.max(np.abs(command-self.target)) <= 1e-6
        if at_command_target and measured_error <= self.tolerance_rad and stable:
            if self.stable_since_ns is None:
                self.stable_since_ns = now_monotonic_ns
        else:
            self.stable_since_ns = None
        settled = (self.stable_since_ns is not None
                   and now_monotonic_ns-self.stable_since_ns >= self.stable_duration_s*1e9)
        self.previous = command.copy()
        self.last_feedback = feedback.state.copy()
        return command, settled, measured_error


def prepare_and_run_chunks(infer, acquire_after, transport, initial_observation,
                           *, num_chunks=1, speed_rad_s=0.03, trace_writer=None):
    """Old prediction is used only as a prepare target, never as the first chunk."""
    if num_chunks < 1:
        raise ValueError("num_chunks must be positive")
    metadata, initial_actions, _ = infer(initial_observation)
    initial_actions = as_finite_action_chunk(initial_actions)
    prepared = transport.request({"operation": "prepare", "target": initial_actions[0].tolist(),
                                  "speed_rad_s": speed_rad_s})
    if prepared.get('status') != 'PASS':
        raise RuntimeError("Prepare did not pass; chunks blocked")
    results = []
    completed = prepared
    next_observation = None
    for chunk_index in range(num_chunks):
        if trace_writer is None:
            # Preserve the original control flow for legacy and untraced
            # callers: acquire exactly once before each chunk inference.
            observation = acquire_after(completed)
        elif next_observation is None:
            observation = acquire_after(completed)
        else:
            observation = next_observation
            next_observation = None
        metadata, actions, adapted_chunk = infer(observation)
        trace_path = None
        if trace_writer is not None:
            trace_path = trace_writer.begin_chunk(
                chunk_index=chunk_index, metadata=metadata,
                observation_before=observation, actions=actions,
                adapted_chunk=adapted_chunk)
        command_started = time.monotonic()
        completed = transport.request({"operation": "chunk", "actions": actions.tolist()})
        if completed.get('status') != 'PASS':
            raise RuntimeError("Guarded chunk failed; further inference stopped")
        command_duration = time.monotonic() - command_started
        if trace_writer is not None:
            after = acquire_after(completed)
            next_observation = after
            trace_writer.finish_chunk(
                trace_path,
                observation_after=after,
                command_duration_s=command_duration,
                end_to_end_duration_s=time.monotonic() - command_started,
            )
        results.append(completed)
    return {"mode": "execute", "prepare": prepared, "chunks": results,
            "checkpoint": metadata["checkpoint"], "norm_stats": metadata["norm_stats"],
            "action_shape": [50,58], "action_hz":20, "num_chunks":len(results),
            "published": any(r.get("published", False) for r in [prepared, *results])}
