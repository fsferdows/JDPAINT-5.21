"""Localhost HTTP Bridge exposing build123d / OpenCascade B-Rep CAD operations to JDPaint."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import agentcad
import build123d as b3d

from brep_builder import build_parametric_preset, build_solid_from_vectors

HOST = "127.0.0.1"
PORT = 5174
MAX_BODY_BYTES = 32 * 1024 * 1024

_HERE = os.path.dirname(os.path.abspath(__file__))
_VISION_SCRIPT = os.path.join(_HERE, "vision_worker.py")
_CNC_PYTHON = "/Users/user/Desktop/CNC/.venv/bin/python"
_VISION_PYTHON = _CNC_PYTHON if os.path.isfile(_CNC_PYTHON) else sys.executable
_VISION_AVAILABLE = os.path.isfile(_VISION_SCRIPT) and os.path.isfile(_CNC_PYTHON)


class _VisionWorkerClient:
    """Lazy persistent subprocess client for OpenCV + VTracer + Depth-Anything-V2."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None

    def _ensure_proc(self) -> subprocess.Popen[str]:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [_VISION_PYTHON, "-u", _VISION_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                text=True,
                bufsize=1,
            )
        return self._proc

    def call(self, op: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            proc = self._ensure_proc()
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps({"op": op, "payload": payload}) + "\n")
            proc.stdin.flush()
            resp_line = proc.stdout.readline()
            if not resp_line:
                self.close()
                proc = self._ensure_proc()
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(json.dumps({"op": op, "payload": payload}) + "\n")
                proc.stdin.flush()
                resp_line = proc.stdout.readline()
            if not resp_line:
                raise RuntimeError("Vision worker terminated without output")
            data = json.loads(resp_line)
            if not data.get("ok"):
                raise RuntimeError(data.get("error", f"Vision op {op} failed"))
            return data

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()
        self._proc = None


_VISION_WORKER = _VisionWorkerClient()


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
                "vision_ai": _VISION_AVAILABLE,
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

            if self.path == "/api/vision/vectorize":
                if not _VISION_AVAILABLE:
                    self._send_json(503, {"ok": False, "error": "Local Vision worker unavailable"})
                    return
                res = _VISION_WORKER.call("vectorize", payload)
                self._send_json(200, res)
                return

            if self.path == "/api/vision/reconstruct-3d":
                if not _VISION_AVAILABLE:
                    self._send_json(503, {"ok": False, "error": "Local Vision worker unavailable"})
                    return
                res = _VISION_WORKER.call("reconstruct_3d", payload)
                self._send_json(200, res)
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
        _VISION_WORKER.close()
        server.server_close()


if __name__ == "__main__":
    main()
