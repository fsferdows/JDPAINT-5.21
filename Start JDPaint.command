#!/usr/bin/env bash
# Double-click to launch JDPaint 5.21 (Mac Edition) + OpenCascade CAD Kernel Bridge
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
BRIDGE_PID=""

cleanup() {
  if [ -n "$BRIDGE_PID" ] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
    kill "$BRIDGE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "=========================================================="
echo " Starting JDPaint 5.21 (Mac Edition) + OpenCascade CAD..."
echo "=========================================================="

if [ -x "$DIR/cad-kernel/.venv/bin/python" ] && [ -f "$DIR/cad-kernel/cad_bridge_server.py" ]; then
  "$DIR/cad-kernel/.venv/bin/python" "$DIR/cad-kernel/cad_bridge_server.py" &
  BRIDGE_PID=$!
fi

cd "$DIR/jdpaint-mac"
npm run dev -- --open
