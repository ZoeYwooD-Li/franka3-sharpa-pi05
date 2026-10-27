import io
import pickle

import numpy as np
import pytest
from PIL import Image

from deploy.fr3_sharpa_runtime.trex.client import TrexClient, slow_payload, tactile_payload
from deploy.fr3_sharpa_runtime.trex.contract import IMAGE_SIZE, URDF_SHA256


def test_tactile_order_and_image_routing_are_explicit():
    head = np.full((2, 3, 3), 1, dtype=np.uint8)
    right = np.full((2, 3, 3), 2, dtype=np.uint8)
    left = np.full((2, 3, 3), 3, dtype=np.uint8)
    f6 = np.arange(16 * 10 * 6, dtype=np.float32).reshape(16, 10, 6)
    deform = np.full((10, 240, 240), 255, dtype=np.uint8)
    payload = slow_payload(head, right, left, "pick tomato", f6, deform)
    assert payload["mode"] == "slow_and_fast"
    for key, value in (("image_head", 1), ("image_wrist_right", 2), ("image_wrist_left", 3)):
        np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(payload[key]))), value)
    np.testing.assert_array_equal(payload["tactile_f6"], f6)
    assert payload["tactile_deform"].shape == (10, 240, 240)
    assert np.all(payload["tactile_deform"] == 1)


def test_tactile_history_cannot_be_synthesized_by_padding():
    with pytest.raises(ValueError, match="16,10,6"):
        tactile_payload(np.zeros((1, 10, 6)), np.zeros((10, 240, 240)))
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        tactile_payload(np.zeros((16, 10, 6)), np.full((10, 240, 240), 4., dtype=np.float32))


class FakeSocket:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.sent = []

    def send(self, data):
        self.sent.append(pickle.loads(data))

    def recv(self):
        return pickle.dumps(next(self.replies))


def test_server_handshake_and_chunk_identity():
    client = TrexClient()
    client._socket = FakeSocket([
        {"status": "success", "checkpoint": "/weights", "action_shape": [16, 62],
         "urdf_sha256": URDF_SHA256, "warmed_up": True, "image_size": list(IMAGE_SIZE)},
        {"status": "success", "chunk_id": 7, "actions": np.zeros((16, 62))},
        {"status": "success", "chunk_id": 8, "actions": np.zeros((16, 62))},
    ])
    assert client.hello("/weights")["warmed_up"]
    chunk_id, actions = client.fast(np.zeros((16, 10, 6)), np.zeros((10, 240, 240)), 7)
    assert chunk_id == 7 and actions.shape == (16, 62)
    with pytest.raises(ValueError, match="another action chunk"):
        client.fast(np.zeros((16, 10, 6)), np.zeros((10, 240, 240)), 7)
    assert client._socket.sent[0] == {"mode": "hello"}
    assert client._socket.sent[1]["mode"] == "fast"


def test_server_handshake_rejects_wrong_image_size():
    client = TrexClient()
    client._socket = FakeSocket([{
        "status": "success", "checkpoint": "/weights", "action_shape": [16, 62],
        "urdf_sha256": URDF_SHA256, "warmed_up": True, "image_size": [224, 224],
    }])
    with pytest.raises(ValueError, match="image size"):
        client.hello("/weights")


def test_remote_pickle_endpoint_is_rejected():
    with pytest.raises(ValueError, match="loopback"):
        TrexClient("tcp://192.168.1.4:5555")
