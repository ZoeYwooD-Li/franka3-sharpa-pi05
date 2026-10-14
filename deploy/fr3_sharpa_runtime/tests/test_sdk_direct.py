import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.protocol import LEFT_SHARPA_SLICE, RIGHT_SHARPA_SLICE
from deploy.fr3_sharpa_runtime.sdk_direct import (
    DirectSdkConfig, SharpaSdkDirectExecutor, validate_direct_hand_actions,
    scaled_action_frames,
)


def chunk(value=0.0):
    return np.full((50, 58), value, dtype=np.float32)


def test_direct_validator_rejects_slew():
    actions = chunk()
    actions[1, LEFT_SHARPA_SLICE.start] = 0.11
    with pytest.raises(ValueError, match="slew"):
        validate_direct_hand_actions(actions)


def test_scaled_action_frames_preserves_targets_and_stretches_time():
    actions = chunk()
    actions[1, 0] = 0.1
    actions[2, 0] = 0.2
    frames = np.asarray(list(scaled_action_frames(actions, 0.1)))
    assert frames.shape == (491, 58)
    assert np.allclose(frames[0], actions[0])
    assert np.allclose(frames[10], actions[1])
    assert np.allclose(frames[20], actions[2])
    assert np.allclose(frames[-1], actions[-1])


def test_scaled_action_frames_rejects_invalid_scale():
    with pytest.raises(ValueError, match="speed scale"):
        list(scaled_action_frames(chunk(), 0.0))


class FakeHand:
    def __init__(self):
        self.connected = False
        self.sent = []
        self.stopped = False

    def connect(self): self.connected = True
    def command_joint_position(self, values): self.sent.append(tuple(values))
    def stop(self): self.stopped = True
    def disconnect(self): self.connected = False


def test_direct_executor_uses_injected_hands_without_sdk_import():
    hands = []
    def factory(side, serial):
        hand = FakeHand(); hands.append(hand); return hand
    executor = SharpaSdkDirectExecutor(DirectSdkConfig(watchdog_timeout_s=1.0), factory)
    executor.connect()
    executor.send([0.0] * 22, [0.0] * 22)
    assert len(hands) == 2 and all(hand.sent for hand in hands)
    executor.stop()
    assert all(hand.stopped for hand in hands)


def test_resume_stream_starts_a_new_watchdog_epoch():
    executor = SharpaSdkDirectExecutor(DirectSdkConfig(watchdog_timeout_s=1.0),
                                       lambda side, serial: FakeHand())
    executor.connect()
    executor.send([0.0] * 22, [0.0] * 22)
    assert executor._last_send is not None
    executor.resume_stream()
    assert executor._last_send is None
    executor.send([0.0] * 22, [0.0] * 22)
    assert executor._last_send is not None
    executor.stop()


def test_stop_attempts_both_hands_when_one_stop_fails():
    class FailingStopHand(FakeHand):
        def stop(self):
            self.stopped = True
            raise RuntimeError("link lost")

    hands = []
    def factory(side, serial):
        hand = FailingStopHand() if side == "left" else FakeHand()
        hands.append(hand)
        return hand
    executor = SharpaSdkDirectExecutor(DirectSdkConfig(), factory)
    executor.connect()
    errors = executor.stop()
    assert any(error.startswith("left.stop:") for error in errors)
    assert all(hand.stopped for hand in hands)
    assert all(not hand.connected for hand in hands)
