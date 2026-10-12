#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec python3 "$ROOT/deploy/fr3_sharpa_runtime/hardware_stack.py" "$@"
