#!/usr/bin/env bash
# Operate only the existing camera service; never start robot controllers.
set -euo pipefail
TREX_CAMERA_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PI05_ROOT="/home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test"
source "$PI05_ROOT/deploy/fr3_sharpa_runtime/env.sh"
export TREX_CAMERA_CONFIG_DIR="$TREX_CAMERA_DIR/cameras"
COMPOSE=(docker compose --profile openpi-hardware --project-directory "$SHARPA_GELLO_REPO/docker"
  --env-file "$SHARPA_GELLO_REPO/docker/.env"
  -f "$SHARPA_GELLO_REPO/docker/compose.yaml"
  -f "$PI05_ROOT/deploy/fr3_sharpa_runtime/compose.hardware.yaml")
case "${1:-check}" in
  check) "${COMPOSE[@]}" -f "$TREX_CAMERA_DIR/compose.cameras.yaml" config --quiet ;;
  up) "${COMPOSE[@]}" -f "$TREX_CAMERA_DIR/compose.cameras.yaml" up -d --no-deps openpi-cameras ;;
  restore) "${COMPOSE[@]}" up -d --no-deps openpi-cameras ;;
  *) echo 'Usage: cameras.sh [check|up|restore]' >&2; exit 2 ;;
esac
