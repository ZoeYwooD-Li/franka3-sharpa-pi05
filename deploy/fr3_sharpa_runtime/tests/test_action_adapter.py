from deploy.fr3_sharpa_runtime.protocol import DT_NS, FPS
import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.action_adapter import SharpaActionAdapter
from deploy.fr3_sharpa_runtime.protocol import (
    ACTION_DIM,
    ACTION_HORIZON,
    FR3_COMMAND_NAMES,
    LEFT_SHARPA_NAMES,
    RIGHT_SHARPA_NAMES,
)


def make_actions():
    return np.arange(ACTION_HORIZON * ACTION_DIM, dtype=np.float32).reshape(ACTION_HORIZON, ACTION_DIM)


def test_shape_and_finite_validation():
    adapter = SharpaActionAdapter()
    result = adapter.adapt_chunk(make_actions(), start_timestamp_ns=1_000_000_000)
    assert len(result.frames) == 50
    with pytest.raises(ValueError, match="Expected action shape"):
        adapter.adapt_chunk(np.zeros((49, 58), dtype=np.float32))
    bad = make_actions()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        adapter.adapt_chunk(bad)


def test_joint_names_and_slices():
    frame = SharpaActionAdapter().adapt_chunk(make_actions()).frames[0]
    assert frame.fr3.joint_names == FR3_COMMAND_NAMES
    assert frame.left_sharpa.joint_names == LEFT_SHARPA_NAMES
    assert frame.right_sharpa.joint_names == RIGHT_SHARPA_NAMES
    np.testing.assert_array_equal(frame.fr3.positions, np.r_[make_actions()[0, :7], make_actions()[0, 29:36]])
    np.testing.assert_array_equal(frame.left_sharpa.position, make_actions()[0, 7:29])
    np.testing.assert_array_equal(frame.right_sharpa.position, make_actions()[0, 36:58])


def test_timestamp_interval_and_sequence_continuity():
    result = SharpaActionAdapter().adapt_chunk(make_actions(), start_timestamp_ns=123)
    timestamps = [frame.timestamp_ns for frame in result.frames]
    sequences = [frame.sequence for frame in result.frames]
    assert sequences == list(range(50))
    assert [b - a for a, b in zip(timestamps, timestamps[1:])] == [DT_NS] * 49


def test_frame_records_share_metadata():
    result = SharpaActionAdapter().adapt_chunk(make_actions(), session_id="s")
    for frame in result.frames:
        assert frame.fr3.session_id == frame.left_sharpa.session_id == frame.right_sharpa.session_id == "s"
        assert frame.fr3.source == "openpi"
        assert frame.left_sharpa.side == "left"
        assert frame.right_sharpa.side == "right"
