"""Validate the merged T-Rex checkpoint and its physical action contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ACTION_DIM = 62
ACTION_HORIZON = 16
STATE_DIM = 58
IMAGE_SIZE = (384, 288)
URDF_SHA256 = "6be9900731b19bfb3edd56e08d63feb9112de3045957f84088b0fe6c7c873939"
URDF_PATH = Path(__file__).parent / "assets/dual_fr3_kinematics.urdf"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_checkpoint(path: str | Path) -> dict:
    """Fail closed if the files or dimensions differ from the trained policy."""
    root = Path(path).expanduser().resolve()
    required = [root / name for name in (
        "model.pt", "config.json", "stats_data.json", "training_args.json", "adapter_metadata.json",
        "robot/left.urdf", "robot/right.urdf",
    )]
    missing = [str(file) for file in required if not file.is_file()]
    if missing:
        raise ValueError(f"Incomplete merged T-Rex checkpoint: {missing}")
    if not (root / "processor").is_dir():
        raise ValueError("Merged T-Rex checkpoint lacks processor/")
    for name in ("tokenizer.json", "preprocessor_config.json"):
        if not (root / "processor" / name).is_file():
            raise ValueError(f"T-Rex processor lacks {name}")
    if (root / "model.pt").stat().st_size <= 1_000_000_000:
        raise ValueError("model.pt is too small for a full or merged checkpoint")
    if _sha256(URDF_PATH) != URDF_SHA256:
        raise ValueError("Vendored nominal FR3 URDF changed")
    metadata = json.loads((root / "adapter_metadata.json").read_text())
    training = json.loads((root / "training_args.json").read_text())
    rank = training.get("lora_rank")
    if type(rank) is not int or rank < 0:
        raise ValueError("T-Rex training_args.lora_rank must be a nonnegative integer")
    origin_file = root / "lora_origin.json"
    if rank == 0:
        if origin_file.exists():
            raise ValueError("Full-training checkpoint has unexpected lora_origin.json")
        kind = "full_finetune"
    else:
        if not origin_file.is_file():
            raise ValueError("LoRA adapter must be merged with its midtrain base before deployment")
        origin = json.loads(origin_file.read_text())
        if origin.get("rank") != rank or origin.get("alpha") != training.get("lora_alpha"):
            raise ValueError("Merged LoRA provenance disagrees with training_args.json")
        if not all(isinstance(origin.get(key), str) and origin[key]
                   for key in ("base_checkpoint", "adapter_checkpoint")):
            raise ValueError("Merged LoRA provenance is incomplete")
        kind = "merged_lora"
    if (metadata.get("action_dim"), metadata.get("action_chunk")) != (ACTION_DIM, ACTION_HORIZON):
        raise ValueError("T-Rex action contract must be 16x62")
    if (training.get("action_dim"), training.get("action_chunk")) != (ACTION_DIM, ACTION_HORIZON):
        raise ValueError("T-Rex training action shape differs from adapter metadata")
    if training.get("use_robot_state") != 0 or training.get("use_tactile_vec") != 1 or training.get("use_tactile_deform") != 1:
        raise ValueError("Unexpected T-Rex observation modalities")
    if metadata.get("action_mode") != "delta-base from chunk-start measured FK; absolute hand targets":
        raise ValueError("Unexpected T-Rex action representation")
    for side in ("left", "right"):
        robot = metadata.get("robot_models", {}).get(side, {})
        if (robot.get("base_link"), robot.get("tip_link")) != (
            f"{side}_fr3v2_link0", f"{side}_fr3v2_link8"
        ):
            raise ValueError(f"{side} arm must use the link8 flange")
        if robot.get("joint_names") != [f"{side}_fr3v2_joint{i}" for i in range(1, 8)]:
            raise ValueError(f"{side} FR3 joint order differs from deployment")
        if robot.get("sha256") != URDF_SHA256 or _sha256(root / robot["urdf"]) != URDF_SHA256:
            raise ValueError(f"{side} checkpoint URDF differs from the nominal model")
    if metadata.get("cameras") != {"head": "cam0", "wrist_right": "cam1", "wrist_left": "cam2"}:
        raise ValueError("T-Rex camera roles differ from deployment")
    from ..protocol import SHARPA_JOINT_NAMES
    if metadata.get("hand_order") != list(SHARPA_JOINT_NAMES):
        raise ValueError("T-Rex hand joint order differs from deployment")
    if metadata.get("hand_scale") != 1.0 or metadata.get("wrench_scale") != [1] * 6:
        raise ValueError("Unexpected hand or tactile scaling")
    stats = json.loads((root / "stats_data.json").read_text())
    if len(stats) != 1:
        raise ValueError("T-Rex stats must contain exactly one dataset")
    dataset_name, dataset_stats = next(iter(stats.items()))
    action = dataset_stats.get("action", {})
    tactile = dataset_stats.get("tactile_f6", {})
    if not isinstance(action.get("mask"), list) or len(action["mask"]) != ACTION_DIM:
        raise ValueError("T-Rex action mask must be 62D")
    for key in ("q01", "q99"):
        values = action.get(key)
        if not isinstance(values, list) or len(values) != ACTION_HORIZON or any(
            not isinstance(row, list) or len(row) != ACTION_DIM for row in values
        ):
            raise ValueError(f"T-Rex action/{key} statistics must be [16,62]")
    for key in ("mask", "q01", "q99"):
        values = tactile.get(key)
        if not isinstance(values, list) or len(values) != 60:
            raise ValueError(f"T-Rex tactile_f6/{key} statistics must be 60D")
    return {"checkpoint": str(root), "action_shape": [ACTION_HORIZON, ACTION_DIM],
            "urdf_sha256": URDF_SHA256, "dataset_name": dataset_name, "training": training,
            "checkpoint_kind": kind}
