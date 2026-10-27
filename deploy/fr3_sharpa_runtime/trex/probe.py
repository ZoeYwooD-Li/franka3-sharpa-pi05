"""Synthetic T-Rex ZMQ inference probe; never reads or commands hardware."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .client import TrexClient
from .contract import IMAGE_SIZE


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be in [1024,65535]")
    black = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
    f6 = np.zeros((16, 10, 6), dtype=np.float32)
    deform = np.zeros((10, 240, 240), dtype=np.uint8)
    with TrexClient(f"tcp://127.0.0.1:{args.port}") as client:
        hello = client.hello(args.checkpoint)
        chunk_id, actions = client.slow_and_fast(
            black, black, black, "pick ripe tomato", f6, deform)
    print(json.dumps({
        "checkpoint_kind": hello["checkpoint_kind"],
        "chunk_id": chunk_id,
        "action_shape": list(actions.shape),
        "finite": bool(np.isfinite(actions).all()),
    }, indent=2))


if __name__ == "__main__":
    main()
