from types import SimpleNamespace

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.ros_observation import ordered_state, rgb_image
from deploy.fr3_sharpa_runtime.protocol import SHARPA_JOINT_NAMES


def test_state_order_matches_training_protocol():
    arm = SimpleNamespace(
        name=[f"left_fr3v2_joint{i}" for i in range(7, 0, -1)],
        position=list(range(7, 0, -1)),
    )
    hand = SimpleNamespace(
        name=[f"left_{joint}" for joint in reversed(SHARPA_JOINT_NAMES)],
        position=list(range(21, -1, -1)),
    )
    assert ordered_state(arm, "left", False) == list(range(1, 8))
    assert ordered_state(hand, "left", True) == list(range(22))
    with pytest.raises(ValueError, match="Missing"):
        ordered_state(SimpleNamespace(name=arm.name[:-1], position=arm.position[:-1]), "left", False)


def test_camera_bgr_to_rgb_and_invalid_shape():
    message = SimpleNamespace(encoding="bgr8", height=1, width=1, step=3, data=bytes([1, 2, 3]))
    import base64

    result = rgb_image(message)
    np.testing.assert_array_equal(np.frombuffer(base64.b64decode(result["rgb_b64"]), dtype=np.uint8), [3, 2, 1])
    with pytest.raises(ValueError, match="Invalid camera byte count"):
        rgb_image(SimpleNamespace(encoding="rgb8", height=2, width=1, step=3, data=b"123"))
