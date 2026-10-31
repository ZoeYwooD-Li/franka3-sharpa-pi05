"""Serve the merged dual-FR3 T-Rex checkpoint using the official model code.

This entry point fixes the official demo's five-finger warmup for our ten-
finger checkpoint and binds its pickle/ZMQ protocol to loopback only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np

from .contract import ACTION_DIM, ACTION_HORIZON, IMAGE_SIZE, validate_checkpoint

VENDORED_TREX = Path(__file__).resolve().parent / "vendor/trex"
SOURCE_MANIFEST = Path(__file__).resolve().parent / "vendor/source_manifest.json"


def verify_model_source(trex_repo: Path) -> None:
    """Reject a model checkout that differs from the reviewed training code."""
    expected = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))["sha256"]
    for relative, digest in expected.items():
        source = trex_repo / relative
        if not source.is_file():
            raise ValueError(f"T-Rex model source is missing {relative}: {trex_repo}")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"T-Rex model source differs from reviewed version: {relative}")


def _load_official_server(trex_repo: Path):
    verify_model_source(trex_repo)
    script = trex_repo / "scripts/test.py"
    if not script.is_file():
        raise ValueError(f"Official T-Rex scripts/test.py not found: {script}")
    for package in ("qwen_vla", "tactile_vqvae"):
        loaded = sys.modules.get(package)
        location = getattr(loaded, "__file__", None)
        if location is not None and not Path(location).resolve().is_relative_to(trex_repo.resolve()):
            raise RuntimeError(f"{package} is already imported from another T-Rex checkout: {location}")
    sys.path.insert(0, str(trex_repo))
    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location("trex_official_inference", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _png_black() -> bytes:
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", IMAGE_SIZE, "black").save(buffer, format="PNG")
    return buffer.getvalue()


def build_server(trex_repo, checkpoint, cuda):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("T-Rex serving requires a CUDA GPU")
    contract = validate_checkpoint(checkpoint)
    official = _load_official_server(Path(trex_repo).resolve())
    training = contract["training"]
    args = argparse.Namespace(**training)
    args.checkpoint_path = contract["checkpoint"]
    args.base_model_path = ""
    args.stats_path = str(Path(checkpoint).resolve() / "stats_data.json")
    args.dataset_name = contract["dataset_name"]
    args.action_dim = ACTION_DIM
    args.action_chunk = ACTION_HORIZON
    args.cuda = str(cuda)
    args.image_size = IMAGE_SIZE
    args.disable_tactile = 0
    model, processor, statistics = official.model_load(args)
    server = official.CascadedServer(args, model, processor, statistics)
    black = _png_black()
    warmup = {"image_head": black, "image_wrist_right": black,
              "image_wrist_left": black, "task_description": "warmup",
              "tactile_f6": np.zeros((16, 10, 6), dtype=np.float32),
              "tactile_deform": np.zeros((10, 240, 240), dtype=np.float32)}
    result = server.predict("slow_and_fast", warmup)
    actions = np.asarray(result.get("actions"), dtype=np.float32)
    if actions.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
        raise RuntimeError(f"T-Rex warmup returned invalid action shape {actions.shape}")
    metadata = {"status": "success", "checkpoint": contract["checkpoint"],
                "action_shape": [ACTION_HORIZON, ACTION_DIM],
                "urdf_sha256": contract["urdf_sha256"], "warmed_up": True,
                "checkpoint_kind": contract["checkpoint_kind"],
                "image_size": list(IMAGE_SIZE)}
    return server, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description="Loopback T-Rex model server; no robot commands")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--trex-repo", type=Path, default=VENDORED_TREX,
                        help="Reviewed T-Rex source; defaults to the bundled copy")
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--check-only", action="store_true", help="Validate checkpoint without loading CUDA")
    parser.add_argument("--warmup-only", action="store_true", help="Run one real model inference, then exit")
    args = parser.parse_args(argv)
    contract = validate_checkpoint(args.checkpoint)
    if args.check_only:
        print(json.dumps({key: contract[key] for key in ("checkpoint", "checkpoint_kind", "action_shape", "urdf_sha256")}, indent=2))
        return
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be in [1024,65535]")
    server, metadata = build_server(args.trex_repo, args.checkpoint, args.cuda)
    from .wire import encode, decode
    from .execution_core import ExclusiveLease
    lease = ExclusiveLease()
    metadata['exclusive_sessions'] = True
    print(f"T-Rex warmup OK: {metadata['action_shape']}; checkpoint={metadata['checkpoint']}", flush=True)
    if args.warmup_only:
        return
    import zmq
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://127.0.0.1:{args.port}")
    print(f"T-Rex server listening on 127.0.0.1:{args.port}", flush=True)
    try:
        while True:
            try:
                request = decode(pickle.loads(socket.recv()))  # Trusted loopback client only.
                if not isinstance(request, dict):
                    raise ValueError("T-Rex request must be a dictionary")
                mode = request.get("mode")
                if mode == 'claim':
                    lease.claim(request.get('token'))
                    response = metadata
                elif mode == 'release':
                    lease.release(request.get('_session'))
                    response = metadata
                elif mode == 'hello':
                    response = metadata
                elif mode == 'keepalive':
                    lease.check(request.get('_session'))
                    response = metadata
                else:
                    lease.check(request.get('_session'))
                    response = server.predict(mode, request)
            except Exception as error:
                traceback.print_exc()
                response = {"status": "error", "message": str(error)}
            socket.send(pickle.dumps(encode(response), protocol=pickle.HIGHEST_PROTOCOL))
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
