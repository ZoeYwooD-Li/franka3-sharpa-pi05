from types import SimpleNamespace

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.trex.tactile import ROUTES, TactileHistory


def _frames(index):
    frames = {}
    for physical, (side, finger) in enumerate(ROUTES):
        image = SimpleNamespace(width=240, height=240, encoding="mono8",
                                data=bytes([physical]) * (240 * 240))
        frames[(side, finger)] = SimpleNamespace(
            side=side, finger=finger, frame_index=index,
            timestamp_s=1000.0 + index / 30 + physical * 0.0001,
            wrench=[physical + value / 10 for value in range(6)],
            deformation=image)
    return frames


def test_history_preserves_physical_finger_order_and_real_density():
    history = TactileHistory()
    start = 1_000_000_000
    for index in range(16):
        history.append(_frames(index), received_monotonic_ns=start + index * 33_333_333)
    f6, deformation, stamp = history.snapshot(now_monotonic_ns=start + 15 * 33_333_333)
    assert f6.shape == (16, 10, 6)
    assert deformation.shape == (10, 240, 240)
    np.testing.assert_array_equal(f6[-1, :, 0], np.arange(10))
    np.testing.assert_array_equal(deformation[:, 0, 0], np.arange(10))
    assert stamp == start + 15 * 33_333_333


def test_missing_finger_or_nonadvancing_frame_invalidates_history():
    history = TactileHistory()
    history.append(_frames(1), received_monotonic_ns=1_000)
    missing = _frames(2)
    del missing[("right", "pinky")]
    with pytest.raises(ValueError, match="exactly ten"):
        history.append(missing, received_monotonic_ns=2_000)
    assert len(history.samples) == 0
    history.append(_frames(2), received_monotonic_ns=2_000)
    with pytest.raises(ValueError, match="Non-advancing"):
        history.append(_frames(2), received_monotonic_ns=3_000)
    assert len(history.samples) == 0


def test_gap_and_wrong_physical_identity_are_rejected():
    history = TactileHistory()
    history.append(_frames(1), received_monotonic_ns=1_000)
    wrong = _frames(2)
    wrong[("left", "thumb")].finger = "pinky"
    with pytest.raises(ValueError, match="identity"):
        history.append(wrong, received_monotonic_ns=2_000)
    history.append(_frames(3), received_monotonic_ns=3_000)
    with pytest.raises(ValueError, match="gap"):
        history.append(_frames(4), received_monotonic_ns=100_000_000)
    assert len(history.samples) == 0


def test_sdk_reads_named_fingers_without_reimplementing_channel_numbers():
    class Hand:
        tactile_available = True

        def __init__(self, side):
            self.side = side
            self.calls = []

        def read_tactile(self, finger, timeout_s):
            self.calls.append((finger, timeout_s))
            return _frames(1)[(self.side, finger)]

    left, right = Hand("left"), Hand("right")
    sample = TactileHistory().read_from_hands(left, right, received_monotonic_ns=1000)
    assert sample.f6.shape == (10, 6)
    assert [finger for finger, _ in left.calls] == ["thumb", "index", "middle", "ring", "pinky"]
    assert [finger for finger, _ in right.calls] == ["thumb", "index", "middle", "ring", "pinky"]
