"""Validate the merged T-Rex checkpoint and its physical action contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ACTION_DIM = 62
ACTION_HORIZON = 16
STATE_DIM = 58
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
    if _sha256(URDF_PATH) != URDF_SHA256:
        raise ValueError("Vendored nominal FR3 URDF changed")
    metadata = json.loads((root / "adapter_metadata.json").read_text())
    training = json.loads((root / "training_args.json").read_text())
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
    for field, width in (("action", ACTION_DIM), ("tactile_f6", 60)):
        block = dataset_stats.get(field, {})
        for key in ("mask", "q01", "q99"):
            values = block.get(key)
            if not isinstance(values, list) or len(values) == 0:
                raise ValueError(f"Missing T-Rex {field}/{key} statistics")
            if field == "action":
                values = values[0] if isinstance(values[0], list) else values
            if len(values) != width:
                raise ValueError(f"T-Rex {field}/{key} statistics width differs from {width}")
    return {"checkpoint": str(root), "action_shape": [ACTION_HORIZON, ACTION_DIM],
            "urdf_sha256": URDF_SHA256, "dataset_name": dataset_name, "training": training}
