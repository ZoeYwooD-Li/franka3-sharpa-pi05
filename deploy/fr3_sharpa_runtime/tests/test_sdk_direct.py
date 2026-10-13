import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.protocol import LEFT_SHARPA_SLICE, RIGHT_SHARPA_SLICE
from deploy.fr3_sharpa_runtime.sdk_direct import (
    DirectSdkConfig, SharpaSdkDirectExecutor, validate_direct_hand_actions,
)


def chunk(value=0.0):
    return np.full((50, 58), value, dtype=np.float32)


def test_direct_validator_rejects_slew():
    actions = chunk()
    actions[1, LEFT_SHARPA_SLICE.start] = 0.11
    with pytest.raises(ValueError, match="slew"):
        validate_direct_hand_actions(actions)


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
