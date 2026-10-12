"""Policy client and one-chunk manager. No ROS or SDK imports."""

from __future__ import annotations

import time
import uuid

import numpy as np

from . import config
from .action_adapter import SharpaActionAdapter
from .protocol import as_finite_action_chunk


def validate_observation(observation: dict) -> dict:
    required = (
        "observation/state",
        "observation/image",
        "observation/left_wrist_image",
        "observation/right_wrist_image",
        "prompt",
    )
    missing = [key for key in required if key not in observation]
    if missing:
        raise ValueError(f"Missing policy observations: {missing}")
    state = np.asarray(observation["observation/state"], dtype=np.float32)
    if state.shape != (config.STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError("Expected finite 58D measured state")
    clean = {"observation/state": state}
    for key in required[1:4]:
        image = np.asarray(observation[key])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{key} must be an HxWx3 uint8 RGB image")
        clean[key] = image
    prompt = observation["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Task prompt is required")
    clean["prompt"] = prompt
    return clean


class SharpaRuntime:
    """Request one 50-step prediction and prepare one independent session."""

    def __init__(self, policy_url: str = config.POLICY_URL):
        if not policy_url.startswith("ws://127.0.0.1:") and not policy_url.startswith("ws://localhost:"):
            raise ValueError("First runtime only accepts a loopback policy server")
        self.policy_url = policy_url
        self.adapter = SharpaActionAdapter()

    def infer_chunk(self, observation: dict):
        # Import here so the pure adapter remains importable in Pixi ROS2.
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        clean = validate_observation(observation)
        with connect(self.policy_url, compression=None, max_size=None, open_timeout=10, close_timeout=5) as ws:
            metadata = msgpack_numpy.unpackb(ws.recv(timeout=10))
            if metadata.get("config_name") != config.CONFIG_NAME:
                raise ValueError("Policy server config does not match Sharpa runtime")
            if metadata.get("checkpoint") != str(config.CHECKPOINT):
                raise ValueError("Policy server checkpoint does not match frozen runtime")
            if metadata.get("norm_stats") != str(config.NORM_STATS):
                raise ValueError("Policy server norm_stats does not match frozen runtime")
            if (
                metadata.get("action_dim") != config.ACTION_DIM
                or metadata.get("action_horizon") != config.ACTION_HORIZON
                or metadata.get("action_hz") != config.ACTION_HZ
            ):
                raise ValueError("Policy server action contract does not match Sharpa runtime")
            ws.send(msgpack_numpy.Packer().pack(clean))
            response = ws.recv(timeout=120)
        if isinstance(response, str):
            raise RuntimeError(f"Policy server error: {response}")
        result = msgpack_numpy.unpackb(response)
        actions = as_finite_action_chunk(result["actions"])
        # Fresh session for this single chunk; this version never splices chunks.
        session_id = f"openpi-{uuid.uuid4().hex}"
        chunk = self.adapter.adapt_chunk(
            actions, start_timestamp_ns=time.time_ns(), session_id=session_id, source=config.SOURCE
        )
        return metadata, actions, chunk
