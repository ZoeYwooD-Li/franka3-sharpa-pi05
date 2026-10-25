"""Decode T-Rex relative link8 poses into the existing 58-joint order."""

from __future__ import annotations

import numpy as np

from .contract import ACTION_DIM, ACTION_HORIZON, STATE_DIM, URDF_PATH
from .kinematics import ArmKinematics, IKError

LEFT_ARM = slice(0, 9)
LEFT_HAND = slice(9, 31)
RIGHT_ARM = slice(31, 40)
RIGHT_HAND = slice(40, 62)


def rotation_from_columns(values) -> np.ndarray:
    """Map the policy's first two rotation columns to SO(3)."""
    columns = np.asarray(values, dtype=np.float64)
    if columns.shape != (6,) or not np.isfinite(columns).all():
        raise ValueError("Rotation representation must be six finite values")
    first = columns[:3]
    norm = np.linalg.norm(first)
    if norm < 1e-6:
        raise ValueError("Degenerate first rotation column")
    first = first / norm
    second = columns[3:] - first * np.dot(first, columns[3:])
    norm = np.linalg.norm(second)
    if norm < 1e-6:
        raise ValueError("Degenerate second rotation column")
    second = second / norm
    return np.column_stack((first, second, np.cross(first, second)))


def absolute_link8_pose(base_pose, arm_action) -> np.ndarray:
    """All rows use the measured chunk-start flange pose, never the prior row."""
    base = np.asarray(base_pose, dtype=np.float64)
    action = np.asarray(arm_action, dtype=np.float64)
    if base.shape != (4, 4) or action.shape != (9,) or not np.isfinite(action).all():
        raise ValueError("Expected a 4x4 base pose and finite 9D arm action")
    target = np.eye(4, dtype=np.float64)
    target[:3, 3] = base[:3, 3] + base[:3, :3] @ action[:3]
    target[:3, :3] = base[:3, :3] @ rotation_from_columns(action[3:])
    return target


class TrexActionDecoder:
    def __init__(self, urdf=URDF_PATH, *, max_joint_jump_rad=0.2):
        self.left = ArmKinematics(urdf, "left")
        self.right = ArmKinematics(urdf, "right")
        if max_joint_jump_rad <= 0:
            raise ValueError("Maximum decoded joint jump must be positive")
        self.max_joint_jump_rad = float(max_joint_jump_rad)

    def decode(self, actions, measured_state) -> np.ndarray:
        """Return 16x58 absolute targets; reject any row that fails IK/sanity."""
        policy = np.asarray(actions, dtype=np.float64)
        measured = np.asarray(measured_state, dtype=np.float64)
        if policy.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(policy).all():
            raise ValueError("T-Rex actions must be finite 16x62")
        if measured.shape != (STATE_DIM,) or not np.isfinite(measured).all():
            raise ValueError("Measured robot state must be finite 58D")
        q_left, q_right = measured[:7].copy(), measured[29:36].copy()
        base_left, base_right = self.left.forward(q_left), self.right.forward(q_right)
        decoded = np.empty((ACTION_HORIZON, STATE_DIM), dtype=np.float32)
        for index, row in enumerate(policy):
            left_target = absolute_link8_pose(base_left, row[LEFT_ARM])
            right_target = absolute_link8_pose(base_right, row[RIGHT_ARM])
            next_left = self.left.inverse(left_target, q_left)
            next_right = self.right.inverse(right_target, q_right)
            if (np.max(np.abs(next_left - q_left)) > self.max_joint_jump_rad
                    or np.max(np.abs(next_right - q_right)) > self.max_joint_jump_rad):
                raise IKError(f"Decoded FR3 joint jump too large at action {index}")
            decoded[index, :7] = next_left
            decoded[index, 7:29] = row[LEFT_HAND]  # Absolute hand angles in rad.
            decoded[index, 29:36] = next_right
            decoded[index, 36:58] = row[RIGHT_HAND]
            q_left, q_right = next_left, next_right
        return decoded
