"""Named arm and hand Home; optional hand targets share the same prepare phase."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml

from .protocol import SHARPA_JOINT_NAMES


def load_home(path: Path, task: str, *, hand_path: Path | None = None):
    content = path.read_bytes()
    data = yaml.safe_load(content)
    home = data["tasks"][task]["home"]
    arms = {}
    for side in ("left", "right"):
        entry = home[side]
        expected = [f"{side}_fr3_joint{i}" for i in range(1, 8)]
        names = entry["joint_names"]
        if len(names) != 7 or len(set(names)) != 7 or set(names) != set(expected):
            raise ValueError(f"Invalid {side} Home joint names")
        values = np.asarray(entry["positions"], dtype=np.float32)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {side} Home positions")
        arms[side] = values[[names.index(name) for name in expected]]
    metadata = {
        "kind": "fixed_arm_home",
        "task": task,
        "file": str(path.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "updated_utc": data.get("updated_utc"),
        "unit": "rad",
        "hands": "hold_initial_measured_positions",
    }
    provenance_path = path.with_suffix(".source.json")
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        if provenance["sha256"] != metadata["sha256"]:
            raise ValueError("Home snapshot differs from recorded source; resync the pose file")
        metadata["source"] = provenance["source"]
    if hand_path is not None:
        hand_content = hand_path.read_bytes()
        hand_data = json.loads(hand_content)
        if hand_data.get("unit") != "rad":
            raise ValueError("Hand Home must use radians")
        for side in ("left", "right"):
            entry = hand_data["tasks"][task][side]
            expected = [f"{side}_{name}" for name in SHARPA_JOINT_NAMES]
            names = entry["joint_names"]
            if len(names) != 22 or len(set(names)) != 22 or set(names) != set(expected):
                raise ValueError(f"Invalid {side} hand Home joint names")
            values = np.asarray(entry["positions"], dtype=np.float32)
            if values.shape != (22,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {side} hand Home positions")
            arms[f"{side}_hand"] = values[[names.index(name) for name in expected]]
        metadata.update(
            kind="fixed_arm_and_hand_home",
            hands="fixed_named_hand_home",
            hand_home_file=str(hand_path.resolve()),
            hand_home_sha256=hashlib.sha256(hand_content).hexdigest(),
            hand_home_source=hand_data.get("source"),
        )
    return arms, metadata


def home_target(arms, measured):
    target = np.asarray(measured, dtype=np.float32).copy()
    if target.shape != (58,) or not np.isfinite(target).all():
        raise ValueError("Home requires a finite measured 58D state")
    target[:7] = arms["left"]
    target[29:36] = arms["right"]
    if "left_hand" in arms or "right_hand" in arms:
        for side, part in (("left", slice(7, 29)), ("right", slice(36, 58))):
            values = np.asarray(arms[f"{side}_hand"], dtype=np.float32)
            if values.shape != (22,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {side} hand Home target")
            target[part] = values
    return target


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Snapshot operator Home without changing source permissions")
    parser.add_argument("--source", type=Path, default=Path.home() / "franka_teleop_data/operator_gui/poses.yaml")
    parser.add_argument("--output", type=Path, default=root / ".deployment/operator_poses.yaml")
    parser.add_argument("--task", default="powder_weighing")
    parser.add_argument("--via-docker", action="store_true", help="Read root-owned source using a read-only mount")
    args = parser.parse_args()
    if args.via_docker:
        content = subprocess.check_output(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--entrypoint",
                "cat",
                "--mount",
                f"type=bind,src={args.source.resolve()},dst=/poses.yaml,readonly",
                "franka-upper-body-teleop:latest",
                "/poses.yaml",
            ]
        )
    else:
        content = args.source.read_bytes()
    # Validate before replacing an existing deployment snapshot.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".yaml.tmp")
    try:
        temporary.write_bytes(content)
        load_home(temporary, args.task)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    provenance = {"source": str(args.source.resolve()), "sha256": hashlib.sha256(content).hexdigest()}
    args.output.with_suffix(".source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({**provenance, "snapshot": str(args.output), "task": args.task}))


if __name__ == "__main__":
    main()
