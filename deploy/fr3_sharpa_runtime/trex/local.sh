#!/usr/bin/env bash
# execute requires both --execute and --enable-hardware; all other modes are non-motion.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PYTHON="${TREX_PYTHON:-$ROOT/.venv-trex/bin/python}"
CKPT="${TREX_CHECKPOINT:-/home/descfly/桌面/fanghao_Trex_all_6250}"
case "${1:-help}" in
  health)
    shift
    exec "$PYTHON" -m deploy.fr3_sharpa_runtime.trex.execute --checkpoint "$CKPT" --check-server "$@"
    ;;
  replay)
    shift
    exec "$PYTHON" -m deploy.fr3_sharpa_runtime.trex.replay "$@"
    ;;
  execute|doctor)
    MODE="$1"
    shift
    ROS_REPO="${TREX_ROS_REPO:-/home/descfly/ZH/gello_upper_body_teleop}"
    ROS_PYTHON="${TREX_ROS_PYTHON:-$ROS_REPO/portable_deps/litchi_hardware/.pixi/envs/ros2/bin/python}"
    OVERLAY="${TREX_ROS_OVERLAY:-/home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test/.deployment/sharpa_ros_messages/install/setup.bash}"
    [[ -f "$OVERLAY" ]] || { echo "Missing existing ROS message overlay: $OVERLAY" >&2; exit 1; }
    # setup.bash sources ROS environment hooks, which can overwrite caller
    # discovery settings (notably ROS_LOCALHOST_ONLY=1 -> 0). Preserve the
    # workcell/caller settings before loading message types, then restore them.
    TREX_DOMAIN="${ROS_DOMAIN_ID:-0}"
    TREX_LOCALHOST="${ROS_LOCALHOST_ONLY:-1}"
    TREX_RMW="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
    TREX_DDS="${CYCLONEDDS_URI:-file://$ROS_REPO/config/cyclonedds.xml}"
    set +u
    source "$OVERLAY"
    set -u
    export PYTHONPATH="$ROOT:$ROS_REPO:$ROS_REPO/ros_ws/src/teleop_core:$ROS_REPO/portable_deps/litchi_hardware/src${PYTHONPATH:+:$PYTHONPATH}"
    export ROS_DOMAIN_ID="$TREX_DOMAIN"
    export ROS_LOCALHOST_ONLY="$TREX_LOCALHOST"
    export RMW_IMPLEMENTATION="$TREX_RMW"
    export CYCLONEDDS_URI="$TREX_DDS"
    # Per-process CPU placement follows the existing workcell; no host tuning.
    CPUSET="$(awk -F= '$1=="HOUSEKEEPING_CPUSET" {gsub(/\r/, "", $2); print $2; exit}' "$ROS_REPO/docker/.env")"
    [[ -n "$CPUSET" ]] || { echo 'Missing existing HOUSEKEEPING_CPUSET' >&2; exit 1; }
    exec taskset -c "$CPUSET" "$ROS_PYTHON" -m "deploy.fr3_sharpa_runtime.trex.$MODE" --checkpoint "$CKPT" "$@"
    ;;
  check|warmup|serve)
    MODE="$1"
    shift
    EXTRA=()
    if [[ "$MODE" == check ]]; then EXTRA+=(--check-only); fi
    if [[ "$MODE" == warmup ]]; then EXTRA+=(--warmup-only); fi
    exec "$PYTHON" -m deploy.fr3_sharpa_runtime.trex.serve --checkpoint "$CKPT" "${EXTRA[@]}" "$@"
    ;;
  probe|shadow)
    MODE="$1"
    shift
    exec "$PYTHON" -m "deploy.fr3_sharpa_runtime.trex.$MODE" --checkpoint "$CKPT" "$@"
    ;;
  capture)
    shift
    ROS_REPO="${TREX_ROS_REPO:-/home/descfly/ZH/gello_upper_body_teleop}"
    ROS_PYTHON="${TREX_ROS_PYTHON:-$ROS_REPO/portable_deps/litchi_hardware/.pixi/envs/ros2/bin/python}"
    # Standard ROS messages are already installed in this interpreter. No
    # workspace overlay, Pixi solve/install, or changes to an existing env.
    export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    exec "$ROS_PYTHON" -m deploy.fr3_sharpa_runtime.trex.capture "$@"
    ;;
  *)
    echo 'Usage: bash deploy/fr3_sharpa_runtime/trex/local.sh {health|doctor|check|warmup|serve|probe|capture|shadow|execute|replay} [arguments]'
    ;;
esac
