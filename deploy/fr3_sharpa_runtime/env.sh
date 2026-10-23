#!/usr/bin/env bash
# Source from launchers. Explicit caller environment takes precedence over local defaults.
SHARPA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "$SHARPA_ROOT/.deployment/local.env" ]]; then
  source "$SHARPA_ROOT/.deployment/local.env"
fi
export PYTHONPATH="$SHARPA_ROOT/src:$SHARPA_ROOT/packages/openpi-client/src:$SHARPA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

sharpa_python() {
  local interpreter="${SHARPA_PYTHON:-$SHARPA_ROOT/.venv/bin/python}"
  [[ -x "$interpreter" ]] || {
    echo 'Run setup_local.py first, or set SHARPA_PYTHON to an existing OpenPI interpreter.' >&2
    return 1
  }
  exec "$interpreter" "$@"
}
