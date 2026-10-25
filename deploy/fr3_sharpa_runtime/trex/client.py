"""Loopback-only client for the stateful T-Rex slow/fast policy service."""

from __future__ import annotations

import io
import pickle

import numpy as np
from PIL import Image

from .contract import ACTION_DIM, ACTION_HORIZON, URDF_SHA256


def _image_bytes(value, label: str) -> bytes:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"{label} must be HxWx3 uint8 RGB")
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def tactile_payload(f6, deformation) -> dict:
    """Require a real dense F6 history and ten ordered deformation images."""
    history = np.asarray(f6, dtype=np.float32)
    deform = np.asarray(deformation)
    if history.shape != (16, 10, 6) or not np.isfinite(history).all():
        raise ValueError("T-Rex requires finite F6 history [16,10,6]")
    if deform.shape != (10, 240, 240) or not np.issubdtype(deform.dtype, np.number):
        raise ValueError("T-Rex requires ten 240x240 deformation images")
    if not np.isfinite(deform).all():
        raise ValueError("T-Rex deformation image contains NaN or Inf")
    if deform.dtype == np.uint8:
        normalized = deform.astype(np.float32) / 255.0
    else:
        normalized = deform.astype(np.float32)
        if (normalized < 0).any() or (normalized > 1).any():
            raise ValueError("Float deformation images must already be in [0,1]")
    return {"tactile_f6": history.copy(), "tactile_deform": normalized.copy()}


def slow_payload(head, right_wrist, left_wrist, prompt, f6, deformation) -> dict:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("T-Rex task prompt is required")
    return {"mode": "slow_and_fast", "image_head": _image_bytes(head, "head"),
            "image_wrist_right": _image_bytes(right_wrist, "right wrist"),
            "image_wrist_left": _image_bytes(left_wrist, "left wrist"),
            "task_description": prompt, **tactile_payload(f6, deformation)}


class TrexClient:
    def __init__(self, endpoint="tcp://127.0.0.1:5555", timeout_ms=120_000):
        if not (endpoint.startswith("tcp://127.0.0.1:") or endpoint.startswith("tcp://localhost:")):
            raise ValueError("T-Rex pickle protocol is only permitted on loopback")
        self.endpoint = endpoint
        self.timeout_ms = int(timeout_ms)
        self._context = None
        self._socket = None

    def __enter__(self):
        import zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.connect(self.endpoint)
        return self

    def __exit__(self, *_):
        if self._socket is not None:
            self._socket.close()
        if self._context is not None:
            self._context.term()
        self._socket = self._context = None

    def _request(self, payload):
        if self._socket is None:
            raise RuntimeError("T-Rex client must be used within a context manager")
        self._socket.send(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        response = pickle.loads(self._socket.recv())  # Trusted loopback server only.
        if not isinstance(response, dict) or response.get("status") != "success":
            raise RuntimeError(f"T-Rex server rejected request: {response!r}")
        return response

    def hello(self, expected_checkpoint=None):
        response = self._request({"mode": "hello"})
        if response.get("action_shape") != [ACTION_HORIZON, ACTION_DIM]:
            raise ValueError("T-Rex server action shape does not match 16x62")
        if response.get("urdf_sha256") != URDF_SHA256 or response.get("warmed_up") is not True:
            raise ValueError("T-Rex server geometry or warmup differs from deployment")
        if expected_checkpoint is not None:
            from pathlib import Path
            if response.get("checkpoint") != str(Path(expected_checkpoint).expanduser().resolve()):
                raise ValueError("T-Rex server loaded another checkpoint")
        return response

    def slow_and_fast(self, head, right_wrist, left_wrist, prompt, f6, deformation):
        response = self._request(slow_payload(head, right_wrist, left_wrist, prompt, f6, deformation))
        return self._actions(response)

    def fast(self, f6, deformation, expected_chunk_id):
        response = self._request({"mode": "fast", **tactile_payload(f6, deformation)})
        if response.get("chunk_id") != expected_chunk_id:
            raise ValueError("T-Rex fast response belongs to another action chunk")
        return self._actions(response)

    @staticmethod
    def _actions(response):
        actions = np.asarray(response.get("actions"), dtype=np.float32)
        if actions.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
            raise ValueError("T-Rex server returned invalid 16x62 actions")
        if not isinstance(response.get("chunk_id"), int):
            raise ValueError("T-Rex server omitted chunk_id")
        return response["chunk_id"], actions
