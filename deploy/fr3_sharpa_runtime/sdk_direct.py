"""Direct Sharpa Wave SDK backend.

This module intentionally has no dependency on the historical litchibot
validation, permit, acquisition, or handoff modules.  The SDK import is lazy
so dry-run and unit tests never touch hardware.
"""

from __future__ import annotations

import os
import time
import json
import math
from dataclasses import dataclass
from typing import Callable, Any

import numpy as np

from .protocol import DT_NS, LEFT_SHARPA_SLICE, RIGHT_SHARPA_SLICE, as_finite_action_chunk

SDK_SLEW_LIMIT_RAD = 0.10
WATCHDOG_TIMEOUT_S = 0.25


def scaled_action_frames(actions: np.ndarray, speed_scale: float):
    """Time-stretch an absolute chunk while retaining 30 Hz output."""
    chunk = as_finite_action_chunk(actions)
    if not np.isfinite(speed_scale) or not 0.0 < speed_scale <= 1.0:
        raise ValueError("execute speed scale must be finite and in (0, 1]")
    steps = max(1, int(math.ceil(1.0 / float(speed_scale))))
    previous = chunk[0].copy()
    yield previous
    for target in chunk[1:]:
        delta = target - previous
        for index in range(1, steps + 1):
            yield previous + delta * (index / steps)
        previous = target.copy()


def chunk_command_frames(actions, start_state, *, speed_scale=1.0, approach_speed_rad_s=0.03):
    """Join measured state to a new chunk, preserving every policy target.

    The bridge is joint-space interpolation, not another policy inference.
    Check slew on the emitted stream (including its first transition), rather
    than rejecting raw nodes that become executable through time stretching.
    Joint limits on the policy targets and start state are checked by ROS.
    """
    chunk = as_finite_action_chunk(actions)
    start = np.asarray(start_state, dtype=np.float32)
    if start.shape != (58,) or not np.isfinite(start).all():
        raise ValueError("Chunk bridge requires finite measured 58D state")
    if not np.isfinite(approach_speed_rad_s) or approach_speed_rad_s <= 0:
        raise ValueError("Chunk bridge speed must be finite and positive")
    delta = chunk[0] - start
    distance = float(np.max(np.abs(delta)))
    bridge_steps = int(math.ceil(distance / (approach_speed_rad_s * DT_NS / 1e9)))
    bridge = [start + delta * (i / bridge_steps) for i in range(1, bridge_steps + 1)]
    frames = np.asarray([*bridge, *scaled_action_frames(chunk, speed_scale)], dtype=np.float32)
    changes = np.diff(np.vstack([start, frames]), axis=0)
    for part in (LEFT_SHARPA_SLICE, RIGHT_SHARPA_SLICE):
        if np.max(np.abs(changes[:, part])) >= SDK_SLEW_LIMIT_RAD:
            raise ValueError("Interpolated Sharpa stream exceeds slew limit; reduce execute speed scale")
    return frames, bridge_steps


def play_command_frames(frames, send, feedback, *, clock=time.monotonic, sleep=time.sleep):
    """Schedule 30Hz against an absolute clock, without accumulating send time.

    Never burst stale frames to catch up. Real feedback is checked before every
    command; the native SDK watchdog and FR3 gateway remain authoritative.
    """
    period = DT_NS / 1e9
    started = clock()
    for index, frame in enumerate(frames):
        due = started + index * period
        remaining = due - clock()
        if remaining > 0:
            sleep(remaining)
        if clock() - due > period:
            raise RuntimeError("Chunk 30Hz deadline missed; stopped")
        feedback()
        if clock() - due > period:
            raise RuntimeError("Chunk feedback missed 30Hz deadline; stopped")
        send(frame)
    remaining = started + len(frames) * period - clock()
    if remaining > 0:
        sleep(remaining)


def validate_direct_hand_actions(actions: np.ndarray, *, check_slew: bool = True) -> None:
    """Validate finite positions, native Wave limits and per-frame slew."""
    chunk = as_finite_action_chunk(actions)
    try:
        from litchi_hardware.hardware.sharpa.constants import SHARPA_JOINT_LIMITS
    except ImportError:
        schema_path = os.getenv(
            "SHARPA_SCHEMA_PATH", "/home/user/gello_upper_body_teleop/config/litchibot/sharpa_schema.json")
        with open(schema_path, encoding="utf-8") as stream:
            limits = json.load(stream)["joint_limits"]
        SHARPA_JOINT_LIMITS = np.asarray(limits, dtype=np.float64)
    for side, part in (("left", LEFT_SHARPA_SLICE), ("right", RIGHT_SHARPA_SLICE)):
        hand = chunk[:, part]
        limits = np.asarray(SHARPA_JOINT_LIMITS, dtype=np.float64)
        if np.any(hand < limits[:, 0]) or np.any(hand > limits[:, 1]):
            raise ValueError(f"{side} Sharpa SDK command exceeds Wave joint limits")
        if check_slew and len(hand) > 1 and np.max(np.abs(np.diff(hand, axis=0))) >= SDK_SLEW_LIMIT_RAD:
            raise ValueError(f"{side} Sharpa SDK command exceeds slew limit")


@dataclass
class DirectSdkConfig:
    left_serial: str | None = None
    right_serial: str | None = None
    watchdog_timeout_s: float = WATCHDOG_TIMEOUT_S


class SharpaSdkDirectExecutor:
    """Two-hand executor around litchi_hardware SharpaSdkHand.

    ``hand_factory`` is injectable for unit tests.  Real SDK objects are only
    constructed by ``connect`` and only in the explicit hardware path.
    """

    def __init__(self, cfg: DirectSdkConfig | None = None,
                 hand_factory: Callable[..., Any] | None = None) -> None:
        self.cfg = cfg or DirectSdkConfig(
            left_serial=os.getenv("SHARPA_LEFT_SERIAL"),
            right_serial=os.getenv("SHARPA_RIGHT_SERIAL"),
        )
        self._factory = hand_factory
        self.left = self.right = None
        self._last_send = None

    def connect(self) -> None:
        sdk_factory = self._factory is None
        if self._factory is None:
            from litchi_hardware.hardware.sharpa import create_hand
            from litchi_hardware.core.types import HandSide
            from litchi_hardware.hardware.sharpa.config import SharpaConfig
            def factory(side, serial):
                # Some installed litchi_hardware wheels expose ``HandSide``
                # as a plain string for compatibility.  Normalize at this
                # boundary so the SDK config always receives the enum it
                # expects, while avoiding a fragile ``side.value`` access.
                resolved_side = side if isinstance(side, HandSide) else HandSide(str(side))
                return create_hand(SharpaConfig(
                    name=f"sharpa-{resolved_side.value}", side=resolved_side, serial=serial))
            self._factory = factory
        if not sdk_factory:
            # Test factories may accept opaque side values and must not need
            # the litchi_hardware package installed.
            left_side, right_side = "left", "right"
        else:
            from litchi_hardware.core.types import HandSide
            left_side, right_side = HandSide.LEFT, HandSide.RIGHT
        self.left = self._factory(left_side, self.cfg.left_serial)
        self.right = self._factory(right_side, self.cfg.right_serial)
        self.left.connect()
        try:
            self.right.connect()
        except Exception:
            self.left.disconnect()
            self.left = None
            raise
        # The transport may reserve a short synchronized start delay after
        # connect; watchdog starts with the first command, not connection.
        self._last_send = None

    def send(self, left: Any, right: Any) -> None:
        if self.left is None or self.right is None:
            raise RuntimeError("Sharpa SDK backend is not connected")
        now = time.monotonic()
        if self._last_send is not None and now - self._last_send > self.cfg.watchdog_timeout_s:
            self.stop()
            raise RuntimeError("Sharpa SDK watchdog expired")
        self.left.command_joint_position(tuple(float(x) for x in left))
        self.right.command_joint_position(tuple(float(x) for x in right))
        self._last_send = now

    def resume_stream(self) -> None:
        """Start a new authorized command phase without reconnecting.

        Policy inference and live observation happen between 30 Hz command
        phases.  That gap is intentional, so the next phase must establish a
        fresh watchdog epoch; once its first frame is sent, the normal
        watchdog continues to protect every subsequent frame.
        """
        if self.left is None or self.right is None:
            raise RuntimeError("Sharpa SDK backend is not connected")
        self._last_send = None

    def stop(self) -> list[str]:
        """Best-effort stop and disconnect for both hands.

        A transport failure on one hand must not prevent the other hand from
        receiving its stop/disable attempt.  Errors are returned so callers
        can surface a cleanup warning without aborting the already completed
        motion result.
        """
        errors = []
        for side, hand in (("left", self.left), ("right", self.right)):
            if hand is None:
                continue
            try:
                hand.stop()
            except Exception as error:
                errors.append(f"{side}.stop: {error}")
            try:
                hand.disconnect()
            except Exception as error:
                errors.append(f"{side}.disconnect: {error}")
        self.left = self.right = None
        self._last_send = None
        return errors
