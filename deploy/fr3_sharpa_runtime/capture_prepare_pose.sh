#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
GELLO="${SHARPA_GELLO_REPO:-/home/user/gello_upper_body_teleop}"
DOMAIN="$(awk -F= '$1 == "TELEOP_ROS_DOMAIN_ID" {print $2; exit}' "$GELLO/docker/.env")"
[[ "$DOMAIN" =~ ^[0-9]+$ ]] || { echo "Missing TELEOP_ROS_DOMAIN_ID" >&2; exit 1; }
export ROS_DOMAIN_ID="$DOMAIN" ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$GELLO/config/cyclonedds.xml"
export SHARPA_CAPTURE_ROOT="$ROOT"
cd "$GELLO/portable_deps/litchi_hardware"
exec pixi run -e ros2 bash -lc '
  export PYTHONPATH="$SHARPA_CAPTURE_ROOT:${PYTHONPATH:-}"
  exec python -m deploy.fr3_sharpa_runtime.capture_prepare_pose "$@"
' capture-prepare "$@"
