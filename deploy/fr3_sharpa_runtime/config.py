"""30 Hz FR3 + Sharpa deployment contract shared by client and server."""

import os
from pathlib import Path

from .protocol import ACTION_DIM, ACTION_HORIZON, FPS

CONFIG_NAME = "pi05_fr3_sharpa_ab4_normal_rgb_fullft_absolute_30hz_deploy"
# User-provided full-finetune checkpoint. An override must name the complete
# step directory; the matching stats are selected from its assets/.
CHECKPOINT = Path(os.environ.get("SHARPA_CHECKPOINT",
    "/media/user/8C0F29AFFD6CD874/sharpa_tomato_ckpt_4000")).expanduser().resolve()
def _norm_stats_path():
    if os.environ.get("SHARPA_NORM_STATS"):
        return Path(os.environ["SHARPA_NORM_STATS"]).expanduser().resolve()
    if CHECKPOINT is not None:
        candidates = sorted((CHECKPOINT / "assets").rglob("norm_stats.json"))
        if len(candidates) == 1:
            return candidates[0].resolve()
    return None


# Prefer checkpoint-owned statistics. Never reuse another experiment's stats
# just because its data was sampled at the same rate.
NORM_STATS = _norm_stats_path()
POLICY_URL = "ws://127.0.0.1:18765"
ACTION_HZ = FPS
SOURCE = "openpi"
FR3_TOPIC = "/teleop/arm_commands"
SHARPA_LEFT_TOPIC = "/teleop/sharpa/left/openpi_target"
SHARPA_RIGHT_TOPIC = "/teleop/sharpa/right/openpi_target"
STATE_DIM = ACTION_DIM
CHUNK_LENGTH = ACTION_HORIZON
EXECUTION_TRACE_DIR = Path("/home/user/.local/state/fr3_sharpa_runtime/execution_traces")


def require_checkpoint() -> Path:
    if CHECKPOINT is None:
        raise ValueError("Set SHARPA_CHECKPOINT to the complete trained 30 Hz step directory in both terminals")
    if NORM_STATS is None:
        raise ValueError("Set SHARPA_NORM_STATS to matching norm_stats.json, or include exactly one in checkpoint assets/")
    return CHECKPOINT


def validate_training_contract(training):
    if training.data.action_hz != ACTION_HZ:
        raise ValueError(f"Training action_hz={training.data.action_hz} does not match deployment {ACTION_HZ} Hz")
    if (training.model.paligemma_variant != "gemma_2b"
            or training.model.action_expert_variant != "gemma_300m"
            or training.data.extra_delta_transform):
        raise ValueError("Deployment requires full-finetune Gemma variants and absolute actions (no delta transform)")
    if training.model.action_dim != ACTION_DIM or training.model.action_horizon != ACTION_HORIZON:
        raise ValueError("Training action shape does not match the 50x58 deployment contract")


def validate_policy_metadata(metadata):
    expected = {"config_name": CONFIG_NAME, "checkpoint": str(require_checkpoint()),
                "norm_stats": str(NORM_STATS), "action_dim": ACTION_DIM,
                "action_horizon": ACTION_HORIZON, "action_hz": ACTION_HZ,
                "model_finetune_mode": "full", "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "action_representation": "absolute_joint_position", "warmed_up": True}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Policy server {key}={metadata.get(key)!r} does not match deployment {value!r}")


DEFAULT_PROMPT = (
    "Pick up a tomato truss with the right hand, then pick a cherry tomato "
    "with the left hand and place it in the left basket."
)
