#!/usr/bin/env python3
"""Deterministic Multi-Layer Verification Gate for JDPaint 2D+3D Self-Healing Cycles."""

from __future__ import annotations

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_CNC_PY = "/Users/user/Desktop/CNC/.venv/bin/python"
_DIAG = os.path.join(_HERE, "diagnose_2d3d.py")


def _run(cmd: list[str], cwd: str | None = None, env: dict[str, str] | None = None) -> None:
    print(f"\n>>> RUN: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=cwd, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"Validator failed (exit {res.returncode}): {' '.join(cmd)}")


def main() -> int:
    try:
        # 1. Gravitas installation integrity
        _run([sys.executable, os.path.join(_REPO_ROOT, ".agents/skills/gravitas/scripts/doctor.py")])

        # 2. Rust core tests + optimized release build
        rust_dir = os.path.join(_REPO_ROOT, "cad-kernel", "jdpaint_rust_core")
        _run(["cargo", "test"], cwd=rust_dir)
        _run(["cargo", "build", "--release"], cwd=rust_dir)

        # 3. Python syntax & 2D/3D self-healing pixel/vector/depth audit
        py_bin = _CNC_PY if os.path.isfile(_CNC_PY) else sys.executable
        _run([
            py_bin, "-m", "py_compile",
            os.path.join(_REPO_ROOT, "cad-kernel/vision_worker.py"),
            os.path.join(_REPO_ROOT, "cad-kernel/brep_builder.py"),
            os.path.join(_REPO_ROOT, "cad-kernel/rust_verify_report.py"),
            _DIAG,
        ])
        _run([py_bin, os.path.join(_REPO_ROOT, "cad-kernel/rust_verify_report.py")])
        _run([py_bin, _DIAG, "--all-fixtures", "--assert-clean", "--out-dir", "/tmp/jdpaint_diag_gate"])

        # 4. Zero-CJK guard + Vite production build
        cjk_check = (
            "import glob, re, sys\n"
            f"files = glob.glob('{_REPO_ROOT}/jdpaint-mac/src/**/*.js', recursive=True)\n"
            "bad = [f for f in files if re.search(r'[\\u4e00-\\u9fff]', open(f, encoding='utf-8').read())]\n"
            "assert not bad, f'CJK strings found: {bad}'\n"
            "print('Zero-CJK check: CLEAN')\n"
        )
        _run([sys.executable, "-c", cjk_check])
        _run(["npm", "run", "build"], cwd=os.path.join(_REPO_ROOT, "jdpaint-mac"))

        print("\nVERDICT: PASS", flush=True)
        return 0
    except Exception as exc:
        print(f"\nVERDICT: FAIL -- {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
