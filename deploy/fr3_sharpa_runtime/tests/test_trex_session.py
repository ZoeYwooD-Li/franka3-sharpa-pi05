import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.trex.session import TrexChunkSession


def _measured():
    return np.r_[np.array([0., -0.35, 0., -1.8, 0., 2., 0.]), np.zeros(22),
                 np.array([0., 0.35, 0., -1.8, 0., 2., 0.]), np.zeros(22)]


def _actions(hand_target):
    actions = np.zeros((16, 62), dtype=np.float32)
    actions[:, 3:9] = [1, 0, 0, 0, 1, 0]
    actions[:, 34:40] = [1, 0, 0, 0, 1, 0]
    actions[:, 9:31] = hand_target
    actions[:, 40:62] = hand_target
    return actions


def test_fast_update_only_replaces_future_targets():
    session = TrexChunkSession(4, _measured(), _actions(0.1), start_monotonic_ns=1000)
    assert session.next_action()[0] == 0
    first_joint_row = session.joint_actions[0].copy()
    offset = session.refine(4, _actions(0.2), sampled_monotonic_ns=1100, now_monotonic_ns=1200)
    assert offset == 1
    np.testing.assert_array_equal(session.joint_actions[0], first_joint_row)
    np.testing.assert_allclose(session.next_action()[1][7:29], 0.2)


def test_fast_update_rejects_wrong_chunk_or_old_tactile():
    session = TrexChunkSession(4, _measured(), _actions(0.1), start_monotonic_ns=1000)
    with pytest.raises(ValueError, match="another"):
        session.refine(5, _actions(0.2), sampled_monotonic_ns=1100, now_monotonic_ns=1200)
    with pytest.raises(ValueError, match="stale"):
        session.refine(4, _actions(0.2), sampled_monotonic_ns=1001, now_monotonic_ns=200_001_002)
    with pytest.raises(ValueError, match="predates"):
        session.refine(4, _actions(0.2), sampled_monotonic_ns=999, now_monotonic_ns=1200)
