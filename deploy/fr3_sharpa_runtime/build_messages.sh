#!/usr/bin/env bash
# Build message types only. Does not start ROS nodes or connect to hardware.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/deploy/fr3_sharpa_runtime/env.sh"
GELLO="${SHARPA_GELLO_REPO:-/home/user/gello_upper_body_teleop}"
BUILD_ROOT="$ROOT/.deployment/sharpa_ros_messages"
SOURCE="$GELLO/ros_ws/src/teleop_interfaces"
[[ -f "$SOURCE/package.xml" ]] || { echo "Missing message sources: $SOURCE" >&2; exit 1; }
pixi run --locked --manifest-path "$GELLO/portable_deps/litchi_hardware/pixi.toml" -e ros2 \
  bash -c '
    set -eo pipefail
    colcon --log-base "$2/log" build --base-paths "$1" \
      --packages-select teleop_interfaces --build-base "$2/build" --install-base "$2/install" \
      --cmake-args -DPython3_EXECUTABLE="$(command -v python)" -DPYTHON_EXECUTABLE="$(command -v python)"
    source "$2/install/setup.bash"
    python -c "from teleop_interfaces.msg import ArmCommand; ArmCommand(); print(\"ArmCommand import: OK\")"
  ' -- "$SOURCE" "$BUILD_ROOT"
echo "ROS message overlay ready: $BUILD_ROOT/install"
