#!/usr/bin/env bash
# Double-click to launch JDPaint 5.21 (Mac Edition)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR/jdpaint-mac"

echo "=========================================================="
echo " Starting JDPaint 5.21 (Mac Edition)..."
echo "=========================================================="

npm run dev -- --open
