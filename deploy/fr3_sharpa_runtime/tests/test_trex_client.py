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


def test_wire_contains_no_numpy_pickle_globals_and_preserves_arrays():
    from deploy.fr3_sharpa_runtime.trex.wire import encode, decode
    arrays={'f6':np.zeros((16,10,6),dtype=np.float32),
            'deform':np.ones((10,240,240),dtype=np.float32),
            'actions':np.arange(16*62,dtype=np.float32).reshape(16,62)}
    serialized=pickle.dumps(encode(arrays),protocol=5)
    assert b'numpy' not in serialized
    restored=decode(pickle.loads(serialized))
    for key in arrays:
        np.testing.assert_array_equal(restored[key],arrays[key])
        assert restored[key].dtype==arrays[key].dtype


def test_wire_rejects_wrong_buffer_length():
    from deploy.fr3_sharpa_runtime.trex.wire import decode, TAG
    with pytest.raises(ValueError,match='Malformed'):
        decode({TAG:True,'dtype':'<f4','shape':[16,62],'data':b'bad'})


@pytest.mark.parametrize('stage',['send','receive'])
def test_timeout_explains_endpoint_and_discards_req_without_retry(stage):
    import zmq
    from deploy.fr3_sharpa_runtime.trex.client import TrexRequestTimeout
    class TimeoutSocket:
        def __init__(self): self.sent=[];self.closed=False
        def send(self,payload):
            self.sent.append(pickle.loads(payload))
            if stage=='send': raise zmq.Again()
        def recv(self): raise zmq.Again()
        def close(self,linger=0): self.closed=True
    client=TrexClient(timeout_ms=30);sock=TimeoutSocket();client._socket=sock
    client._session_token='existing-session'
    with pytest.raises(TrexRequestTimeout) as caught:
        client._request({'mode':'fast'})
    assert caught.value.stage==stage and caught.value.mode=='fast'
    assert caught.value.endpoint=='tcp://127.0.0.1:5555'
    assert '30 ms' in str(caught.value) and 'not retried' in str(caught.value)
    assert sock.closed and client._socket is None
    client.release()
    assert len(sock.sent)==1 and client._session_token is None


def test_real_zmq_no_reply_reports_timeout_without_efsm():
    import zmq
    from deploy.fr3_sharpa_runtime.trex.client import TrexRequestTimeout
    context=zmq.Context();server=context.socket(zmq.REP)
    server.setsockopt(zmq.LINGER,0)
    port=server.bind_to_random_port('tcp://127.0.0.1')
    try:
        with TrexClient(f'tcp://127.0.0.1:{port}',timeout_ms=50) as client:
            with pytest.raises(TrexRequestTimeout,match='hello receive timed out'):
                client.hello()
            # Even a delayed reply cannot make a lost stateful session reusable.
            assert server.poll(1000)
            assert pickle.loads(server.recv())=={'mode':'hello'}
            server.send(pickle.dumps({'status':'success'}))
            assert client._socket is None
    finally:
        server.close();context.term()


def test_worker_unavailable_server_fails_before_decoder_and_joins(monkeypatch):
    import time
    import zmq
    from deploy.fr3_sharpa_runtime.trex import inference_worker as module
    context=zmq.Context();server=context.socket(zmq.REP)
    server.setsockopt(zmq.LINGER,0)
    port=server.bind_to_random_port('tcp://127.0.0.1')
    monkeypatch.setattr(module,'TrexActionDecoder',lambda:pytest.fail('Decoder must follow successful hello/claim'))
    worker=module.InferenceWorker('/weights',f'tcp://127.0.0.1:{port}',np.full(58,-5),np.full(58,5),timeout_ms=50)
    try:
        deadline=time.monotonic()+2
        while worker.failure is None and time.monotonic()<deadline:time.sleep(.005)
        assert not worker.ready.is_set()
        with pytest.raises(RuntimeError,match=r'TrexRequestTimeout.*hello receive timed out'):
            worker.poll()
    finally:
        worker.close();server.close();context.term()
    assert not worker.thread.is_alive()
