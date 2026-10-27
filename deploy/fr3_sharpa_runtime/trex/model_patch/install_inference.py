"""Apply the Sharpa inference fixes to a pinned upstream T-Rex checkout.

The installer is intentionally narrow: an unknown upstream layout fails closed.
Run install_adapter.py and install_lora.py first for the training model changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

UPSTREAM_COMMIT = "ba4047ce3e9f3d2013d60de600e136897e90ccb8"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if new in source:
        return source
    if source.count(old) != 1:
        raise RuntimeError(f"Unexpected upstream {label}; inspect scripts/test.py")
    return source.replace(old, new, 1)


def patch_source(source: str) -> str:
    source = replace_once(
        source,
        '    sd = torch.load(ckpt_file, map_location="cpu")',
        '    sd = torch.load(ckpt_file, map_location="cpu", weights_only=True, mmap=True)',
        "checkpoint loading",
    )
    source = replace_once(
        source,
        '    print(f"Checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}")\n'
        '    if missing:\n'
        '        print(f"  missing (first 10): {missing[:10]}")',
        '    print(f"Checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}")\n'
        '    if missing or unexpected:\n'
        '        raise RuntimeError("Checkpoint architecture mismatch: "\n'
        '                           f"missing={missing[:10]}, unexpected={unexpected[:10]}")',
        "checkpoint architecture check",
    )
    source = replace_once(
        source,
        '    dummy_f6    = np.zeros((5, 6), dtype=np.float32) if args.use_tactile_vec else None\n'
        '    dummy_deform = np.zeros((5, 240, 240), dtype=np.float32) if args.use_tactile_deform else None',
        '    n_fingers = 10 if args.action_dim == 62 else 5\n'
        '    dummy_f6    = np.zeros((n_fingers, 6), dtype=np.float32) if args.use_tactile_vec else None\n'
        '    dummy_deform = np.zeros((n_fingers, 240, 240), dtype=np.float32) if args.use_tactile_deform else None',
        "dual-hand tactile warm-up",
    )
    source = replace_once(
        source,
        '    socket.bind(f"tcp://0.0.0.0:{args.port}")',
        '    socket.bind(f"tcp://{args.bind_host}:{args.port}")',
        "inference bind address",
    )
    source = replace_once(
        source,
        '    parser.add_argument("--port", type=int, default=5555)',
        '    parser.add_argument("--port", type=int, default=5555)\n'
        '    parser.add_argument("--bind_host", default="127.0.0.1",\n'
        '                        help="ZMQ bind address; expose only on a trusted network")',
        "bind address option",
    )
    return source


def install(repo: Path) -> bool:
    target = repo.resolve() / "scripts" / "test.py"
    original = target.read_text(encoding="utf-8")
    patched = patch_source(original)
    if patched == original:
        return False
    target.write_text(patched, encoding="utf-8")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    print("Inference patch installed" if install(args.repo) else "Inference patch already installed")
