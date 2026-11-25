#!/usr/bin/env bash
# JDPaint 5.21 Mac Edition Launcher
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo " Starting JDPaint 5.21 (Mac Edition) CAD/CAM Studio..."
echo "=========================================================="

# Check if node is available
if ! command -v npm &> /dev/null; then
    echo "Error: Node.js / npm is required to run JDPaint Mac Edition."
    exit 1
fi

# Run vite dev server and open in browser
npm run dev -- --open
