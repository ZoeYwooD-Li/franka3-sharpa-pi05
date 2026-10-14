"""Frozen defaults for the first FR3 + Sharpa deployment runtime."""

from pathlib import Path

from .protocol import ACTION_DIM, ACTION_HORIZON, FPS

CONFIG_NAME = "pi05_fr3_sharpa_tier_a_b1_20hz"
CHECKPOINT = Path("/home/user/checkpoints/new-train/pi05_fr3_sharpa_tier_a_b1_20hz/sharpa_camfix_a70_b130_20hz_30k_wandb/29999")
NORM_STATS = Path("/home/user/sharpa_training_assets/fr3_sharpa/tier_a_b1_20hz/norm_stats.json")
POLICY_URL = "ws://127.0.0.1:18765"
ACTION_HZ = FPS
SOURCE = "openpi"
FR3_TOPIC = "/teleop/arm_commands"
SHARPA_LEFT_TOPIC = "/teleop/sharpa/left/openpi_target"
SHARPA_RIGHT_TOPIC = "/teleop/sharpa/right/openpi_target"
STATE_DIM = ACTION_DIM
CHUNK_LENGTH = ACTION_HORIZON
EXECUTION_TRACE_DIR = Path("/home/user/.local/state/fr3_sharpa_runtime/execution_traces")
