#!/usr/bin/env bash
set -euo pipefail

echo "=== REA Environment Preflight ==="

# 1. Check REA MCP schemas
MCP_DIR="${HOME}/.gemini/antigravity/mcp/rea"
if [[ -d "${MCP_DIR}" ]]; then
  SCHEMA_COUNT=$(ls -1 "${MCP_DIR}"/*.json 2>/dev/null | wc -l | tr -d ' ')
  echo "[OK] REA MCP schemas: ${SCHEMA_COUNT} tools in ${MCP_DIR}"
else
  echo "[WARN] REA MCP schema directory not found at ${MCP_DIR}"
fi

# 2. Check REA CLI
if command -v rea >/dev/null 2>&1; then
  echo "[OK] rea CLI on PATH: $(command -v rea) ($(rea --version 2>/dev/null || echo 'version unknown'))"
elif [[ -x "${HOME}/.hermes/node/bin/rea" ]]; then
  echo "[OK] rea CLI at ${HOME}/.hermes/node/bin/rea ($("${HOME}/.hermes/node/bin/rea" --version 2>/dev/null || echo 'version unknown'))"
else
  echo "[WARN] rea CLI not found on PATH or ~/.hermes/node/bin/rea"
fi

# 3. Check local browser executables for capture_browser_scenario launch mode
for BIN in \
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" \
  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"; do
  if [[ -x "${BIN}" ]]; then
    echo "[OK] Browser executable: ${BIN}"
  fi
done

# 4. Check Chrome remote debugging CDP endpoint (passive connect mode)
if curl -fsS --max-time 1 "http://127.0.0.1:9222/json/version" >/dev/null 2>&1; then
  echo "[OK] Loopback CDP active on http://127.0.0.1:9222"
else
  echo "[INFO] No active Chrome CDP listener on http://127.0.0.1:9222 (enable chrome://inspect/#remote-debugging for connect mode, or use launch mode)"
fi
