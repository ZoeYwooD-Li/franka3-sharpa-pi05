"""Prepare local paths and a dedicated gateway config; never start devices."""

import argparse
from pathlib import Path
import shlex

import yaml


def gateway_config(source: dict) -> dict:
    import copy

    result = copy.deepcopy(source)
    parameters = result["teleop_safety_gateway"]["ros__parameters"]
    allowed = parameters["allowed_sources"]
    if "openpi" not in allowed:
        allowed.append("openpi")
    return result


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path.home() / "桌面/sharpa_tomato_ckpt_4000")
    parser.add_argument("--gello", type=Path, default=Path.home() / "ZH/gello_upper_body_teleop")
    parser.add_argument("--python", type=Path, default=root.parent / "franka3-sharpa-pi05/.venv/bin/python")
    args = parser.parse_args()
    checkpoint, gello = args.checkpoint.expanduser().resolve(), args.gello.expanduser().resolve()
    # Preserve the venv path: resolving its Python symlink would select the base interpreter.
    interpreter = args.python.expanduser().absolute()
    required = [
        checkpoint / "params/manifest.ocdbt",
        gello / "docker/compose.yaml",
        gello / "docker/.env",
        gello / "config/modes/teleop_control.yaml",
        gello / "config/litchibot/sharpa_schema.json",
        gello / "portable_deps/litchi_hardware/pixi.toml",
        interpreter,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    stats = list((checkpoint / "assets").rglob("norm_stats.json"))
    if len(stats) != 1:
        raise ValueError("Checkpoint must contain exactly one matching norm_stats.json")
    deployment = root / ".deployment"
    deployment.mkdir(exist_ok=True)
    gateway = deployment / "teleop_openpi.yaml"
    gateway.write_text(
        yaml.safe_dump(
            gateway_config(yaml.safe_load((gello / "config/modes/teleop_control.yaml").read_text())), sort_keys=False
        )
    )
    paths = {
        "SHARPA_CHECKPOINT": checkpoint,
        "SHARPA_GELLO_REPO": gello,
        "SHARPA_PYTHON": interpreter,
        "SHARPA_SCHEMA_PATH": gello / "config/litchibot/sharpa_schema.json",
        "SHARPA_GATEWAY_CONFIG": gateway,
        "SHARPA_EXECUTION_TRACE_DIR": Path.home() / ".local/state/fr3_sharpa_runtime/execution_traces",
    }
    lines = ["# Generated local defaults. Caller environment takes precedence."]
    for name, path in paths.items():
        lines.append(f'if [[ -z "${{{name}:-}}" ]]; then export {name}={shlex.quote(str(path))}; fi')
    (deployment / "local.env").write_text("\n".join(lines) + "\n")
    print(f"Prepared {deployment / 'local.env'} and {gateway}; no devices started")


if __name__ == "__main__":
    main()
