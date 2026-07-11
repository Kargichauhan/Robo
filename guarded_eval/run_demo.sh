#!/usr/bin/env bash
# Launches the guarded_eval demo end to end. By default picks a genuinely
# free rosbridge port instead of hardcoding one -- this machine is shared,
# and any fixed port can be taken by someone else's process at any time.
#
# Usage:
#   bash run_demo.sh            # scan 19090-19099, use the first free one
#   bash run_demo.sh 19090      # use exactly this port, or fail loudly if
#                                # it's actually taken (a real bind check,
#                                # not silently falling back to the scan) --
#                                # for when you want a fixed port so your
#                                # SSH tunnel command never has to change.
#
# Requires: conda env "ros_env" (RoboStack ROS 2 Jazzy) already set up as
# described in src/guarded_eval/README.md, and this script run from a shell
# where that env is either already active or activatable via the path below.

set -eo pipefail
# (deliberately not `-u`/nounset: colcon's and ROS 2's own generated
# setup.bash scripts reference variables like COLCON_TRACE without
# initializing them first, so sourcing them under `set -u` breaks with
# "unbound variable" -- verified empirically, not a hypothetical.)

PORT_RANGE_START=19090
PORT_RANGE_END=19099
ROS_DOMAIN_ID_VALUE=77
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== guarded_eval launcher =="

# 1. Kill only MY OWN orphaned guarded_eval processes from previous runs --
#    scoped to the current user (-u "$(whoami)"), never touches anyone
#    else's processes on this shared machine. `|| true` since pkill exits
#    non-zero when nothing matches, which isn't a real error here.
#    -9 (SIGKILL) specifically: verified empirically that plain SIGTERM does
#    NOT reliably kill rosbridge_websocket (an rclpy process; rclpy installs
#    its own signal handling that appears to only act on SIGINT, not
#    SIGTERM), so a plain `pkill` here would silently leave it running and
#    still holding its port. SIGKILL can't be caught or ignored.
echo "Cleaning up any of my own orphaned rosbridge/selection_node/sim_node processes..."
pkill -9 -u "$(whoami)" -f "rosbridge_websocket|selection_node|sim_node" 2>/dev/null || true
sleep 1

is_port_free() {
  python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(('0.0.0.0', $1))
    s.close()
    sys.exit(0)
except OSError:
    sys.exit(1)
"
}

CHOSEN_PORT=""
if [ -n "${1:-}" ]; then
  # 2 (fixed-port mode): the caller wants predictability (e.g. a tunnel
  # command that never has to change) over resilience -- verify with a
  # real bind test and fail loudly rather than silently picking something
  # else, since silently substituting a different port would defeat the
  # entire point of asking for a fixed one.
  echo "Checking requested port $1..."
  if is_port_free "$1"; then
    CHOSEN_PORT="$1"
    echo "Port $1 is free, using it."
  else
    echo "ERROR: port $1 was requested but is already in use. Not falling back automatically -- pick a different port, or run with no argument to scan ${PORT_RANGE_START}-${PORT_RANGE_END} for a free one instead." >&2
    exit 1
  fi
else
  # 2 (scan mode): find a genuinely free port with a real bind test --
  # ss/lsof can miss another user's process depending on permissions, but
  # actually trying to bind the port cannot lie.
  echo "Scanning ports ${PORT_RANGE_START}-${PORT_RANGE_END} for a free one..."
  for port in $(seq "$PORT_RANGE_START" "$PORT_RANGE_END"); do
    if is_port_free "$port"; then
      CHOSEN_PORT="$port"
      break
    fi
  done
  if [ -z "$CHOSEN_PORT" ]; then
    echo "ERROR: no free port found in ${PORT_RANGE_START}-${PORT_RANGE_END}. All of them are taken -- widen the range in this script." >&2
    exit 1
  fi
  echo "Chosen port: $CHOSEN_PORT (verified free via a real bind test)"
fi

# 3. Activate ros_env if it isn't already (harmless if it's already active).
if [ "${CONDA_DEFAULT_ENV:-}" != "ros_env" ]; then
  source /home/kchauha3/miniconda3/etc/profile.d/conda.sh
  conda activate ros_env
fi

export ROS_DOMAIN_ID="$ROS_DOMAIN_ID_VALUE"
cd "$WORKSPACE_DIR"
source install/setup.bash

echo ""
echo "=================================================================="
echo "DASHBOARD: set WS_PORT to ${CHOSEN_PORT} and open dashboard.html"
echo "  -> dashboard.html?port=${CHOSEN_PORT}"
echo "  (serve it, e.g. \`python3 -m http.server 8080\` from"
echo "   src/guarded_eval/dashboard/, then open that URL through it)"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID_VALUE}"
echo "=================================================================="
echo ""

exec ros2 launch guarded_eval demo.launch.py port:="$CHOSEN_PORT"
