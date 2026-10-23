"""Frozen, hardware-facing names and slices for the 58D Sharpa protocol.

This module contains data only.  It deliberately has no ROS or hardware imports.
"""

from __future__ import annotations

import numpy as np

ACTION_DIM = 58
ACTION_HORIZON = 50
FPS = 30
DT_NS = 1_000_000_000 // FPS
UNIT = "rad"
ACTION_REPRESENTATION = "absolute_joint_position"

LEFT_FR3_SLICE = slice(0, 7)
LEFT_SHARPA_SLICE = slice(7, 29)
RIGHT_FR3_SLICE = slice(29, 36)
RIGHT_SHARPA_SLICE = slice(36, 58)

FR3_POLICY_NAMES = tuple(
    [f"left_fr3_joint{i}" for i in range(1, 8)]
    + [f"right_fr3_joint{i}" for i in range(1, 8)]
)
FR3_COMMAND_NAMES = tuple(
    [f"left_fr3v2_joint{i}" for i in range(1, 8)]
    + [f"right_fr3v2_joint{i}" for i in range(1, 8)]
)
FR3_POLICY_TO_COMMAND = dict(zip(FR3_POLICY_NAMES, FR3_COMMAND_NAMES, strict=True))

SHARPA_JOINT_NAMES = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)
LEFT_SHARPA_NAMES = tuple(f"left_{name}" for name in SHARPA_JOINT_NAMES)
RIGHT_SHARPA_NAMES = tuple(f"right_{name}" for name in SHARPA_JOINT_NAMES)


def as_finite_action_chunk(actions: np.ndarray) -> np.ndarray:
    """Validate and copy one complete policy action chunk."""
    value = np.asarray(actions, dtype=np.float32)
    if value.shape != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(f"Expected action shape {(ACTION_HORIZON, ACTION_DIM)}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Action chunk contains NaN or Inf")
    return value.copy()
