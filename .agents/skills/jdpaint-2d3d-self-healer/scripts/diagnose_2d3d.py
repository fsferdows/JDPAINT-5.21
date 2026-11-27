#!/usr/bin/env python3
"""Autonomous 2D CAD Vector & 3D Relief Diagnostic Analyzer for JDPaint.

Requires: OpenCV (cv2), NumPy, SciPy/skimage inside `/Users/user/Desktop/CNC/.venv/bin/python`
(or an active environment with JDPaint `cad-kernel` dependencies).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections import Counter
from typing import Any

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_CAD_KERNEL = os.path.join(_REPO_ROOT, "cad-kernel")
if _CAD_KERNEL not in sys.path:
    sys.path.insert(0, _CAD_KERNEL)


def _sample_cubic_ring(pts: list[dict[str, Any]], closed: bool, ox: float, oy: float, sx: float, sy: float, sc_x: float, sc_y: float) -> np.ndarray:
    """Evaluate Cubic Bezier anchors and handles cleanly into polyline pixels."""
    out: list[np.ndarray] = []
    n = len(pts)
    cnt = n if closed else max(0, n - 1)
    def mm2px(xm: float, ym: float) -> np.ndarray:
        return np.array([(xm - ox) / sx * sc_x, (ym - oy) / sy * sc_y], dtype=np.float32)
    for i in range(cnt):
        a = pts[i]
        b = pts[(i + 1) % n]
        p0 = mm2px(a["x"], a["y"])
        p3 = mm2px(b["x"], b["y"])
        c1 = mm2px(a["hOut"]["x"], a["hOut"]["y"]) if a.get("hOut") else p0
        c2 = mm2px(b["hIn"]["x"], b["hIn"]["y"]) if b.get("hIn") else p3
        steps = 1 if (not a.get("hOut") and not b.get("hIn")) else 8
        for s in range(steps):
            t = s / steps
            it = 1.0 - t
            out.append((it ** 3) * p0 + 3 * (it ** 2) * t * c1 + 3 * it * (t ** 2) * c2 + (t ** 3) * p3)
    if not closed and n > 0:
        out.append(mm2px(pts[-1]["x"], pts[-1]["y"]))
    if not out:
        return np.zeros((0, 2), dtype=np.int32)
    return np.round(np.array(out)).astype(np.int32)


def _patch_span(depth_u16: np.ndarray, y0f: float, y1f: float, x0f: float, x1f: float) -> tuple[float, float]:
    """Compute [p05..p95] depth relief span and p96 peak inside normalized ROI."""
    dh, dw = depth_u16.shape[:2]
    y0, y1 = max(0, int(dh * y0f)), min(dh, int(dh * y1f))
    x0, x1 = max(0, int(dw * x0f)), min(dw, int(dw * x1f))
    patch = (depth_u16[y0:y1, x0:x1].astype(np.float32) / 65535.0).ravel()
    if patch.size < 8:
        return 0.0, 0.0
    p05, p95, p96 = np.percentile(patch, [5, 95, 96])
    return float(p95 - p05), float(p96)


def analyze_image(image_path: str, out_dir: str, detail: float = 0.70, corners: float = 0.65, resolution: int = 512) -> dict[str, Any]:
    """Run full 2D watermark/grain/curve and 3D depth relief audit + render 4-panel proof."""
    import vision_worker as vw

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"OpenCV failed to decode image: {image_path}")
    ih, iw = bgr.shape[:2]

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    # 1. Watermark audit (detect interior false positives vs margin banners)
    roi = vw.detect_furniture_roi(bgr)
    subject_wm_px = 0
    total_wm_px = 0
    if roi is not None:
        x0, y0, x1, y1 = roi["x0"], roi["y0"], roi["x1"], roi["y1"]
        sub = bgr[y0 : y1 + 1, x0 : x1 + 1].copy()
        _, wm = vw._inpaint_watermarks(
            sub, roi["dc"][y0 : y1 + 1, x0 : x1 + 1], roi["L"][y0 : y1 + 1, x0 : x1 + 1], roi["l_med"]
        )
    else:
        L_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32) * (100.0 / 255.0)
        dc_full = np.abs(L_full - cv2.GaussianBlur(L_full, (0, 0), sigmaX=max(12.0, min(iw, ih) * 0.025)))
        _, wm = vw._inpaint_watermarks(bgr.copy(), dc_full, L_full, float(np.median(L_full)))
    wh_m, ww_m = wm.shape[:2]
    total_wm_px = int((wm > 0).sum())
    subject_wm_px = int((wm[int(wh_m * 0.20) : int(wh_m * 0.78), int(ww_m * 0.25) : int(ww_m * 0.75)] > 0).sum())

    # 2. Capture intermediate masks & 2D vectorize output
    saved_masks: dict[str, np.ndarray] = {}
    orig_trace = vw._trace_field_rust

    def _spy_trace(mask_u8: np.ndarray, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        saved_masks[str(kwargs.get("layer_group", "mask"))] = mask_u8.copy()
        return orig_trace(mask_u8, *args, **kwargs)

    vw._trace_field_rust = _spy_trace
    try:
        v_res = vw.handle_vectorize({
            "image_b64": b64,
            "trace_quality": 3840,
            "detail": detail,
            "corners": corners,
            "noise_px": 10,
            "width_mm": 100.0,
            "height_mm": 100.0,
        })
    finally:
        vw._trace_field_rust = orig_trace

    ents = v_res.get("entities") or []
    conv = v_res.get("convergence") or {}
    layers = dict(Counter(e.get("layerGroup", "Unknown") for e in ents))

    # Measure curve smoothness, mega-blob fusion, active 3D layer sync & floating short vertical grain lines
    motif_nodes, motif_cubic, max_ring_pts = 0, 0, 0
    active_3d_cnt = 0
    short_unanchored_v = 0
    xs_all = [p["x"] for e in ents for p in (e.get("points") or [{"x": e.get("x1", 50)}, {"x": e.get("x2", 50)}])]
    ys_all = [p["y"] for e in ents for p in (e.get("points") or [{"y": e.get("y1", 50)}, {"y": e.get("y2", 50)}])]
    x_min_d = min(xs_all) if xs_all else 0.0
    x_max_d = max(xs_all) if xs_all else 100.0
    w_d = max(1e-3, x_max_d - x_min_d)
    margin_grain_cnt = 0

    for e in ents:
        if (e.get("relief") or {}).get("enabled", False):
            active_3d_cnt += 1
        ps = e.get("points") or ([{"x": e["x1"], "y": e["y1"]}, {"x": e["x2"], "y": e["y2"]}] if e.get("type") == "line" else [])
        max_ring_pts = max(max_ring_pts, len(ps))
        if e.get("layerGroup") == "Raised Relief Motifs":
            for p in ps:
                motif_nodes += 1
                if p.get("hIn") or p.get("hOut"):
                    motif_cubic += 1
        if ps:
            exs = [p["x"] for p in ps]
            eys = [p["y"] for p in ps]
            ex0, ex1, ey0, ey1 = min(exs), max(exs), min(eys), max(eys)
            if ((ex1 < x_min_d + w_d * 0.042) or (ex0 > x_max_d - w_d * 0.042)) and (ex1 - ex0) < w_d * 0.040 and (ey1 - ey0) < 25.0:
                margin_grain_cnt += 1
            if e.get("type") == "line" and abs(e["x2"] - e["x1"]) < 0.15 and abs(e["y2"] - e["y1"]) < 5.0:
                short_unanchored_v += 1

    cubic_ratio = round(motif_cubic / max(1, motif_nodes), 3) if motif_nodes > 0 else 1.0
    active_3d_ratio = round(active_3d_cnt / max(1, len(ents)), 3)

    # 3. 3D Relief heightmap audit
    r3d = vw.handle_reconstruct_3d({"image_b64": b64, "resolution": resolution, "remove_bg": True, "delight": True})
    dh, dw = int(r3d["height"]), int(r3d["width"])
    u16 = np.frombuffer(base64.b64decode(r3d["depth_u16_b64"]), dtype=np.uint16).reshape((dh, dw))
    overall_span, overall_peak = _patch_span(u16, 0.05, 0.95, 0.05, 0.95)
    top_span, top_peak = _patch_span(u16, 0.08, 0.26, 0.15, 0.85)
    mid_span, mid_peak = _patch_span(u16, 0.28, 0.58, 0.22, 0.78)
    bot_span, bot_peak = _patch_span(u16, 0.64, 0.86, 0.22, 0.78)

    # Classify defects
    defects: list[str] = []
    if subject_wm_px > 200:
        defects.append(f"WM-ERASE: watermark detector matched {subject_wm_px} interior subject pixels")
    if not conv.get("converged", False) or conv.get("coverage_pct", 0.0) < 96.0:
        defects.append(f"LOW-COVERAGE: convergence={conv}")
    if motif_nodes > 0 and cubic_ratio < 0.75:
        defects.append(f"SQUARE-POCKET/ZIGZAG-FACETS: motif_cubic_ratio={cubic_ratio} < 0.75")
    if max_ring_pts > 190:
        defects.append(f"MERGED-BLOBS: max_ring_pts={max_ring_pts} exceeds 190")
    if margin_grain_cnt > 4:
        defects.append(f"GRAIN-TICKS: margin_grain_cnt={margin_grain_cnt} > 4")
    if ents and active_3d_ratio < 0.85:
        defects.append(f"DEAD-3D-LAYERS: active_3d_ratio={active_3d_ratio} < 0.85")
    if r3d.get("is_planar_panel") and (mid_span < 0.25 or bot_span < 0.24 or bot_peak < 0.42):
        defects.append(f"CRATER-3D: low planar relief span mid={mid_span:.3f}, bot={bot_span:.3f}, peak={bot_peak:.3f}")

    # 4. Render 4-panel visual diagnostic strip
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    pane_h = 720
    pane_w = max(240, int(round(pane_h * iw / max(1, ih))))
    sx_mm = 100.0 / max(iw, ih)
    ox = (100.0 - iw * sx_mm) * 0.5
    oy = (100.0 - ih * sx_mm) * 0.5

    p1 = cv2.resize((bgr.astype(np.float32) * 0.24).astype(np.uint8), (pane_w, pane_h))
    scx, scy = pane_w / max(1, iw), pane_h / max(1, ih)
    for e in ents:
        lg = e.get("layerGroup", "")
        col = (238, 211, 34) if lg == "Architectural Frame" else ((94, 197, 34) if "Raised" in lg else (248, 189, 56))
        if e.get("type") == "line":
            pt1 = (int(round((e["x1"] - ox) / sx_mm * scx)), int(round((e["y1"] - oy) / sx_mm * scy)))
            pt2 = (int(round((e["x2"] - ox) / sx_mm * scx)), int(round((e["y2"] - oy) / sx_mm * scy)))
            cv2.line(p1, pt1, pt2, col, 1, cv2.LINE_AA)
        else:
            for ring in ([e.get("points") or []] + (e.get("holes") or [])):
                arr = _sample_cubic_ring(ring, bool(e.get("isClosed", True)), ox, oy, sx_mm, sx_mm, scx, scy)
                if len(arr) >= 2:
                    cv2.polylines(p1, [arr], bool(e.get("isClosed", True)), col, 1, cv2.LINE_AA)

    L_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    k_th = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(9, min(iw, ih) // 32 | 1),) * 2)
    wh_vis = np.clip(cv2.morphologyEx(L_u8, cv2.MORPH_TOPHAT, k_th).astype(np.float32) * 3.0, 0, 255).astype(np.uint8)
    p2 = cv2.resize(cv2.cvtColor(wh_vis, cv2.COLOR_GRAY2BGR), (pane_w, pane_h))

    p3 = np.zeros((pane_h, pane_w, 3), dtype=np.uint8)
    rm = saved_masks.get("Raised Relief Motifs")
    gm = saved_masks.get("Carved Grooves & Panels")
    if rm is not None:
        p3[:, :, 1] = cv2.resize(rm, (pane_w, pane_h), interpolation=cv2.INTER_NEAREST)
    if gm is not None:
        p3[:, :, 0] = cv2.resize(gm, (pane_w, pane_h), interpolation=cv2.INTER_NEAREST)

    depth_u8 = np.clip(u16.astype(np.float32) / 65535.0 * 255.0, 0, 255).astype(np.uint8)
    p4 = cv2.resize(cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO), (pane_w, pane_h))

    panels = [("1:2D Vectors", p1), ("2:WhiteTopHat", p2), ("3:LayerMasks", p3), ("4:3D Depth", p4)]
    for label, pimg in panels:
        cv2.putText(pimg, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    strip_path = os.path.join(out_dir, f"diag_{stem}.png")
    cv2.imwrite(strip_path, np.hstack([p for _, p in panels]))

    return {
        "image": image_path,
        "size": [iw, ih],
        "watermark": {"total_px": total_wm_px, "interior_subject_px": subject_wm_px},
        "vectors": {
            "total": len(ents),
            "layers": layers,
            "convergence": conv,
            "cubic_ratio": cubic_ratio,
            "max_ring_pts": max_ring_pts,
            "margin_grain_cnt": margin_grain_cnt,
            "short_vertical_cnt": short_unanchored_v,
            "active_3d_ratio": active_3d_ratio,
        },
        "relief_3d": {
            "is_planar_panel": bool(r3d.get("is_planar_panel")),
            "grid": [dw, dh],
            "overall_span": round(overall_span, 3),
            "top_span": round(top_span, 3),
            "mid_span": round(mid_span, 3),
            "bot_span": round(bot_span, 3),
            "bot_peak": round(bot_peak, 3),
        },
        "defect_flags": defects,
        "proof_image": strip_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose 2D vector and 3D relief fidelity on any input image.")
    parser.add_argument("--image", help="Path to a single image to diagnose.")
    parser.add_argument("--all-fixtures", action="store_true", help="Run diagnostic audit on default desktop test fixtures.")
    parser.add_argument("--out-dir", default="/tmp/jdpaint_diag", help="Output folder for 4-panel visual proof strips.")
    parser.add_argument("--assert-clean", action="store_true", help="Exit non-zero if any DEFECT_FLAG triggers.")
    args = parser.parse_args()

    targets: list[str] = []
    if args.image:
        targets.append(args.image)
    if args.all_fixtures or not targets:
        defaults = [
            "/Users/user/Desktop/Group-122819.jpg",
            "/Users/user/Desktop/sagwan-wood-designer-single-door.jpg",
            "/Users/user/Desktop/sagwan-wood-doors-500x500.webp",
            "/Users/user/Desktop/cropped-Unique-Door-Design.jpg",
            "/Users/user/Desktop/id/IMG_2508.jpg",
        ]
        targets.extend([p for p in defaults if os.path.isfile(p)])

    if not targets:
        print(json.dumps({"ok": False, "error": "No valid input image found."}))
        return 1

    reports = [analyze_image(t, args.out_dir) for t in targets]
    print(json.dumps({"ok": True, "reports": reports}, indent=2))
    failed = [r for r in reports if r["defect_flags"]]
    if args.assert_clean and failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
