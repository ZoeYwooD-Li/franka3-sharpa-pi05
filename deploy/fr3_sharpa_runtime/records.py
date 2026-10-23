"""Pure Python records emitted by the offline Sharpa action adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .protocol import FPS, DT_NS


def _vector(value: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains NaN or Inf")
    return result.copy()


@dataclass(frozen=True)
class OfflineArmCommand:
    timestamp_ns: int
    sequence: int
    source: str
    session_id: str
    active_sides: Tuple[str, str]
    joint_names: Tuple[str, ...]
    positions: np.ndarray

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0 or self.sequence < 0:
            raise ValueError("timestamp_ns and sequence must be non-negative")
        if self.active_sides != ("left", "right"):
            raise ValueError("Arm command must contain both left and right sides")
        if len(self.joint_names) != 14 or len(set(self.joint_names)) != 14:
            raise ValueError("Arm command requires 14 unique joint names")
        if not self.source or not self.session_id:
            raise ValueError("source and session_id must be non-empty")
        object.__setattr__(self, "positions", _vector(self.positions, (14,), "arm positions"))


@dataclass(frozen=True)
class OfflineSharpaCommand:
    timestamp_ns: int
    sequence: int
    session_id: str
    side: str
    joint_names: Tuple[str, ...]
    position: np.ndarray

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0 or self.sequence < 0:
            raise ValueError("timestamp_ns and sequence must be non-negative")
        if self.side not in {"left", "right"}:
            raise ValueError("Sharpa side must be left or right")
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        if len(self.joint_names) != 22 or len(set(self.joint_names)) != 22:
            raise ValueError("Sharpa command requires 22 unique joint names")
        object.__setattr__(self, "position", _vector(self.position, (22,), "Sharpa position"))


@dataclass(frozen=True)
class OfflineFrame:
    timestamp_ns: int
    sequence: int
    fr3: OfflineArmCommand
    left_sharpa: OfflineSharpaCommand
    right_sharpa: OfflineSharpaCommand

    def __post_init__(self) -> None:
        records = (self.fr3, self.left_sharpa, self.right_sharpa)
        if any(record.timestamp_ns != self.timestamp_ns or record.sequence != self.sequence for record in records):
            raise ValueError("All records in a frame must share timestamp and sequence")


@dataclass(frozen=True)
class OfflineChunk:
    frames: Tuple[OfflineFrame, ...]
    start_timestamp_ns: int
    fps: int
    causal: bool
    action_representation: str
    unit: str
    session_id: str
    source: str

    def __post_init__(self) -> None:
        if len(self.frames) != 50:
            raise ValueError("OfflineChunk must contain 50 frames")
        if self.fps != FPS or not self.causal:
            raise ValueError(f"Sharpa offline chunk requires {FPS}Hz causal metadata")
        if self.start_timestamp_ns < 0 or not self.session_id or not self.source:
            raise ValueError("Invalid chunk metadata")
        expected = self.start_timestamp_ns
        for sequence, frame in enumerate(self.frames):
            if frame.sequence != sequence or frame.timestamp_ns != expected:
                raise ValueError("Chunk timestamps or sequences are not contiguous")
            expected += DT_NS
