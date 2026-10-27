"""Geometry and shape checks for the separate T-Rex action path."""

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.trex.actions import TrexActionDecoder, absolute_link8_pose
from deploy.fr3_sharpa_runtime.trex.contract import URDF_PATH
from deploy.fr3_sharpa_runtime.trex.kinematics import ArmKinematics, IKError


def _policy_arm(base, target):
    delta = base[:3, :3].T @ (target[:3, 3] - base[:3, 3])
    relative = base[:3, :3].T @ target[:3, :3]
    return np.r_[delta, relative[:, 0], relative[:, 1]]


def test_link8_fk_and_relative_action_round_trip():
    model = ArmKinematics(URDF_PATH, "left")
    q0 = np.array([0., -0.35, 0., -1.8, 0., 2., 0.])
    q1 = q0 + np.array([0.01, -0.008, 0.004, 0.006, 0., -0.005, 0.01])
    base, target = model.forward(q0), model.forward(q1)
    restored = absolute_link8_pose(base, _policy_arm(base, target))
    np.testing.assert_allclose(restored, target, atol=1e-10)
    solved = model.inverse(restored, q0)
    np.testing.assert_allclose(model.forward(solved), target, atol=2e-3)
    assert np.linalg.norm(model.forward(q0)[:3, 3] - model.forward(solved)[:3, 3]) > 1e-4


def test_bimanual_chunk_uses_one_measured_base_and_keeps_hands_absolute():
    decoder = TrexActionDecoder()
    left_q = np.array([0., -0.35, 0., -1.8, 0., 2., 0.])
    right_q = np.array([0., 0.35, 0., -1.8, 0., 2., 0.])
    measured = np.r_[left_q, np.zeros(22), right_q, np.zeros(22)]
    left_base, right_base = decoder.left.forward(left_q), decoder.right.forward(right_q)
    actions = np.zeros((16, 62))
    for index in range(16):
        fraction = (index + 1) / 16
        left_goal = decoder.left.forward(left_q + fraction * np.array([0.03, 0., 0., 0., 0., 0., 0.]))
        right_goal = decoder.right.forward(right_q + fraction * np.array([0., 0.03, 0., 0., 0., 0., 0.]))
        actions[index, :9] = _policy_arm(left_base, left_goal)
        actions[index, 9:31] = 0.1 + fraction * 0.01
        actions[index, 31:40] = _policy_arm(right_base, right_goal)
        actions[index, 40:62] = 0.2 + fraction * 0.01
    result = decoder.decode(actions, measured)
    assert result.shape == (16, 58)
    np.testing.assert_allclose(result[:, 7:29], actions[:, 9:31])
    np.testing.assert_allclose(result[:, 36:58], actions[:, 40:62])
    for index in (0, 7, 15):
        np.testing.assert_allclose(decoder.left.forward(result[index, :7]),
                                   absolute_link8_pose(left_base, actions[index, :9]), atol=2e-3)
        np.testing.assert_allclose(decoder.right.forward(result[index, 29:36]),
                                   absolute_link8_pose(right_base, actions[index, 31:40]), atol=2e-3)


def test_degenerate_rotation_and_unreachable_target_fail_closed():
    decoder = TrexActionDecoder()
    measured = np.r_[np.array([0., -0.35, 0., -1.8, 0., 2., 0.]),
                     np.zeros(22), np.array([0., 0.35, 0., -1.8, 0., 2., 0.]), np.zeros(22)]
    actions = np.zeros((16, 62))
    with pytest.raises(ValueError, match="Degenerate"):
        decoder.decode(actions, measured)
    actions[:, 3:9] = [1, 0, 0, 0, 1, 0]
    actions[:, 34:40] = [1, 0, 0, 0, 1, 0]
    actions[0, 0] = 10
    with pytest.raises(IKError, match="IK failed"):
        decoder.decode(actions, measured)


def test_action_shape_and_live_joint_validation():
    decoder = TrexActionDecoder()
    with pytest.raises(ValueError, match="16x62"):
        decoder.decode(np.zeros((50, 58)), np.zeros(58))
    with pytest.raises(IKError, match="limits"):
        decoder.left.forward(np.zeros(7))  # FR3 joint 6 has a positive lower limit.
