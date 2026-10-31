"""Replay a captured observation through T-Rex and nominal IK. Never publishes."""

import argparse
import json
from pathlib import Path

import numpy as np

from .actions import TrexActionDecoder
from .client import TrexClient, tactile_payload
from .contract import validate_checkpoint


def load_observation(path):
    with np.load(path, allow_pickle=False) as saved:
        result = {key: saved[key].copy() for key in
                  ("state", "cam0", "cam1", "cam2", "tactile_f6", "tactile_deform")}
        metadata = json.loads(str(saved["metadata_json"].item()))
    if result["state"].shape != (58,) or not np.isfinite(result["state"]).all():
        raise ValueError("Expected a finite measured 58D joint state")
    for key in ("cam0", "cam1", "cam2"):
        image = result[key]
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Invalid RGB image: {key}")
    tactile_payload(result["tactile_f6"], result["tactile_deform"])
    return result, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be in [1024,65535]")
    contract = validate_checkpoint(args.checkpoint)
    obs, metadata = load_observation(args.observation)
    with TrexClient(f"tcp://127.0.0.1:{args.port}") as client:
        hello = client.hello(args.checkpoint)
        chunk_id, raw = client.slow_and_fast(obs["cam0"], obs["cam1"], obs["cam2"],
                                           args.prompt, obs["tactile_f6"], obs["tactile_deform"])
    report = {"checkpoint": contract["checkpoint"], "checkpoint_kind": hello["checkpoint_kind"],
              "chunk_id": chunk_id, "published": False, "mode": "offline_shadow",
              "observation_metadata": metadata, "action_shape": list(raw.shape),
              "execute_ready": False}
    arrays = {"raw_actions": raw, "measured_state": obs["state"]}
    try:
        arrays["joint_actions"] = TrexActionDecoder().decode(raw, obs["state"])
        report["ik"] = {"status": "PASS", "shape": list(arrays["joint_actions"].shape)}
    except ValueError as error:
        report["ik"] = {"status": "FAIL", "reason": str(error)}
    arrays["report_json"] = np.asarray(json.dumps(report))
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    print(json.dumps(report, indent=2))
    if report["ik"]["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
