#!/usr/bin/env python3
"""Render live JDPaint Chrome Split-View (Zoomed 2D + Standard PBR + Solid Clay) proof PNGs for any image."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_BRIDGE_PY = os.path.join(_REPO_ROOT, "cad-kernel", ".venv", "bin", "python")
_BRIDGE_SRV = os.path.join(_REPO_ROOT, "cad-kernel", "cad_bridge_server.py")


def _bridge_ok() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:5174/health", timeout=1) as r:
            return b'"ok": true' in r.read()
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture live JDPaint browser 2D+3D renders for visual inspection.")
    parser.add_argument("--image", required=True, help="Input image path to trace and preview.")
    parser.add_argument("--out-dir", default="/tmp/jdpaint_diag", help="Output directory for PNG screenshots.")
    parser.add_argument("--url", default="http://localhost:5173", help="Running JDPaint frontend URL.")
    parser.add_argument("--zoom-steps", type=int, default=12, help="Wheel zoom steps over 2D canvas center.")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    proc = None
    if not _bridge_ok():
        proc = subprocess.Popen([_BRIDGE_PY, _BRIDGE_SRV], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for _ in range(40):
            if _bridge_ok():
                break
            time.sleep(0.25)

    errors: list[str] = []
    console_errs: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                headless=True,
                args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"],
            )
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.on("pageerror", lambda err: errors.append(str(err)))
            page.on("console", lambda msg: console_errs.append(msg.text) if msg.type == "error" else None)

            page.goto(args.url, wait_until="networkidle")
            page.locator("#input-ref-image").set_input_files(args.image)
            page.wait_for_timeout(350)
            page.click("#btn-auto-trace-2d")
            page.wait_for_function(
                "() => (document.querySelector('#status-mode')?.textContent || '').includes('Auto-Trace complete')",
                timeout=120000,
            )
            page.wait_for_timeout(400)
            std_path = os.path.join(args.out_dir, f"live_{stem}_standard.png")
            page.screenshot(path=std_path)

            canvas = page.locator("#cad-canvas")
            box = canvas.bounding_box()
            if box and args.zoom_steps > 0:
                page.mouse.move(box["x"] + box["width"] * 0.50, box["y"] + box["height"] * 0.48)
                for _ in range(args.zoom_steps):
                    page.mouse.wheel(0, -180)
                    page.wait_for_timeout(35)
            page.locator('[data-shade="solid"]').click()
            page.wait_for_timeout(400)
            solid_path = os.path.join(args.out_dir, f"live_{stem}_zoom_solid.png")
            page.screenshot(path=solid_path)

            browser.close()
        print(f"Saved: {std_path} & {solid_path} (PageErrors={len(errors)}, ConsoleErrors={len(console_errs)})")
        if errors or console_errs:
            print("Errors:", errors, console_errs)
            return 1
        return 0
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=3)


if __name__ == "__main__":
    raise SystemExit(main())
