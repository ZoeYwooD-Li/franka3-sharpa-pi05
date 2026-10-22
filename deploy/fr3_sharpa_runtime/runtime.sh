#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
GELLO="${SHARPA_GELLO_REPO:-/home/user/gello_upper_body_teleop}"
DOMAIN="$(awk -F= '$1 == "TELEOP_ROS_DOMAIN_ID" {print $2; exit}' "$GELLO/docker/.env")"
[[ "$DOMAIN" =~ ^[0-9]+$ ]] || { echo "Missing TELEOP_ROS_DOMAIN_ID in docker/.env" >&2; exit 1; }
export ROS_DOMAIN_ID="$DOMAIN" ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$GELLO/config/cyclonedds.xml"
# Persist the Pixi-compatible message build across reboots; /tmp is ephemeral.
ROS_MESSAGES_OVERLAY="${SHARPA_ROS_MESSAGES_OVERLAY:-$ROOT/.deployment/sharpa_ros_messages/install}"
if [[ ! -f "$ROS_MESSAGES_OVERLAY/setup.bash" && -z "${SHARPA_ROS_MESSAGES_OVERLAY:-}" ]]; then
  bash "$ROOT/deploy/fr3_sharpa_runtime/build_messages.sh"
fi
[[ -f "$ROS_MESSAGES_OVERLAY/setup.bash" ]] || {
  echo "Missing teleop_interfaces overlay: $ROS_MESSAGES_OVERLAY (check SHARPA_ROS_MESSAGES_OVERLAY)" >&2
  exit 1
}
cd "$ROOT"
exec uv run --no-sync python -m deploy.fr3_sharpa_runtime.run \
  --ros-overlay "$ROS_MESSAGES_OVERLAY" "$@"
