"""Local-only policy server using the frozen checkpoint and external statistics."""

from __future__ import annotations

import argparse

from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import normalize
from openpi.training import config as training_config

from . import config
from .timing import TimedPolicy, timed_server_class


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    if not config.CHECKPOINT.is_dir() or not config.NORM_STATS.is_file():
        raise FileNotFoundError("Frozen Sharpa checkpoint or norm_stats is missing")
    stats = normalize.load(config.NORM_STATS.parent)
    policy = policy_config.create_trained_policy(
        training_config.get_config(config.CONFIG_NAME), config.CHECKPOINT, norm_stats=stats
    )
    metadata = {
        "config_name": config.CONFIG_NAME,
        "checkpoint": str(config.CHECKPOINT),
        "norm_stats": str(config.NORM_STATS),
        "action_dim": config.ACTION_DIM,
        "action_horizon": config.ACTION_HORIZON,
        "action_hz": config.ACTION_HZ,
    }
    timed_server_class(WebsocketPolicyServer)(TimedPolicy(policy), host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
