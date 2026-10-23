import numpy as np
import pytest

from deploy.fr3_sharpa_runtime import sdk_direct


@pytest.fixture
def limits(monkeypatch):
    # Non-binary-exact bounds also exercise float32 rounding at SDK boundaries.
    bounds = np.tile([0.0, 1.7453], (22, 1))
    monkeypatch.setattr(sdk_direct, "hand_joint_limits", lambda: bounds)
    return bounds


def test_clip_recent_negative_pinky_prediction_and_audit_without_mutating_original(limits):
    actions = np.zeros((50, 58), dtype=np.float32)
    actions[1, 27] = -0.016828244552016258  # left pinky_PIP
    actions[:, :7] = -99  # Arm targets are untouched by hand clipping.
    actions[15:, 27] = -1  # Discarded future is untouched, too.
    original = actions.copy()
    corrected, report = sdk_direct.clip_hand_predictions(actions, execute_steps=15)
    assert corrected[1, 27] == 0
    np.testing.assert_array_equal(corrected[:, :7], original[:, :7])
    np.testing.assert_array_equal(corrected[15:], original[15:])
    np.testing.assert_array_equal(actions, original)
    assert report["corrected_values"] == 1
    assert report["corrections"][0]["joint"] == "pinky_PIP"
    assert report["corrections"][0]["action_index"] == 1
    assert report["max_applied_correction_rad"] == pytest.approx(0.016828244552016258)


@pytest.mark.parametrize(("cap", "value"), [(0.05, -0.051), (0.0, -0.001)])
def test_large_correction_or_disabled_clipping_rejects(limits, cap, value):
    actions = np.zeros((50, 58), dtype=np.float32)
    actions[0, 7] = value
    with pytest.raises(ValueError, match="requires correction"):
        sdk_direct.clip_hand_predictions(actions, max_correction_rad=cap)


def test_both_hand_upper_bounds_round_inward_for_driver(limits):
    actions = np.zeros((50, 58), dtype=np.float32)
    actions[:15, 7:29] = 1.7553
    actions[:15, 36:] = 1.7553
    corrected, report = sdk_direct.clip_hand_predictions(actions)
    assert report["corrected_values"] == 15 * 44
    for part in (slice(7, 29), slice(36, 58)):
        assert np.all(corrected[:15, part].astype(np.float64) <= limits[:, 1])


def test_corrected_stream_still_checks_slew_after_interpolation(limits):
    actions = np.zeros((50, 58), dtype=np.float32)
    actions[:15, 27] = -0.01
    actions[1:15, 7] = 0.5
    corrected, _ = sdk_direct.clip_hand_predictions(actions)
    with pytest.raises(ValueError, match="slew"):
        sdk_direct.chunk_command_frames(corrected, np.zeros(58), execute_steps=15, speed_scale=1)
    frames, _ = sdk_direct.chunk_command_frames(corrected, np.zeros(58), execute_steps=15, speed_scale=0.1)
    assert np.max(np.abs(np.diff(frames[:, 7:29], axis=0))) < sdk_direct.SDK_SLEW_LIMIT_RAD


@pytest.mark.parametrize("cap", [-0.01, float("nan"), float("inf")])
def test_invalid_allowance_rejects(limits, cap):
    with pytest.raises(ValueError, match="nonnegative"):
        sdk_direct.clip_hand_predictions(np.zeros((50, 58)), max_correction_rad=cap)
