"""Localhost HTTP Bridge exposing build123d / OpenCascade B-Rep CAD operations to JDPaint."""

from __future__ import annotations

import json
import signal
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import agentcad
import build123d as b3d

from brep_builder import build_parametric_preset, build_solid_from_vectors

HOST = "127.0.0.1"
PORT = 5174
MAX_BODY_BYTES = 8 * 1024 * 1024


class CadBridgeHandler(BaseHTTPRequestHandler):
    server_version = "JDPaint-OpenCascade/1.0"

    def _send_json(self, status_code: int, obj: dict[str, Any]) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {
                "ok": True,
                "kernel": "OpenCascade (build123d + agentcad)",
                "build123d_version": getattr(b3d, "__version__", "0.10.0"),
                "agentcad_version": getattr(agentcad, "__version__", "0.8.1"),
            })
            return
        self._send_json(404, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send_json(400, {"ok": False, "error": f"Invalid Content-Length: {length}"})
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))

            if self.path == "/api/cad/from-vectors":
                res = build_solid_from_vectors(payload)
                self._send_json(200, {"ok": True, **res})
                return

            if self.path == "/api/cad/parametric-preset":
                preset_id = str(payload.get("preset", "bracket"))
                params = payload.get("params") or {}
                res = build_parametric_preset(preset_id, params)
                self._send_json(200, {"ok": True, "preset": preset_id, **res})
                return

            self._send_json(404, {"ok": False, "error": f"Unknown route: {self.path}"})
        except Exception as exc:
            traceback.print_exc()
            self._send_json(422, {"ok": False, "error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        return


class ReusableThreadingServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    server = ReusableThreadingServer((HOST, PORT), CadBridgeHandler)
    signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))
    print(f"OpenCascade CAD Kernel Bridge listening on http://{HOST}:{PORT} (build123d {b3d.__version__})", flush=True)
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
