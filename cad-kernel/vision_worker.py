"""Local AI + OpenCV + VTracer Vision Worker for JDPaint 5.21.

Reads newline-delimited JSON commands from stdin and writes JSON responses to stdout.
Uses cached local Apple M1 MPS Depth-Anything-V2-Small-hf, OpenCV denoising/GrabCut,
VTracer hierarchical spline vectorization, and junction-split skeletonization.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import sys
import traceback
from typing import Any

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
import vtracer

_DEPTH_PROC = None
_DEPTH_MODEL = None
_DEPTH_DEVICE = "cpu"


def _ensure_depth_model():
    global _DEPTH_PROC, _DEPTH_MODEL, _DEPTH_DEVICE
    if _DEPTH_MODEL is not None:
        return _DEPTH_PROC, _DEPTH_MODEL, _DEPTH_DEVICE
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    _DEPTH_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    _DEPTH_PROC = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
    _DEPTH_MODEL = AutoModelForDepthEstimation.from_pretrained(
        model_id, local_files_only=True
    ).to(_DEPTH_DEVICE).eval()
    return _DEPTH_PROC, _DEPTH_MODEL, _DEPTH_DEVICE


def _decode_image_b64(b64_str: str) -> np.ndarray:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to decode input image")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=2)
    return img


def detect_furniture_roi(bgr: np.ndarray) -> dict[str, Any] | None:
    """Isolate wood/carved door & architectural panels from rooms, mirrors, walls & floors."""
    h, w = bgr.shape[:2]
    if h < 64 or w < 64:
        return None
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)
    A = lab[:, :, 1] - 128.0
    B = lab[:, :, 2] - 128.0

    cx0, cx1 = int(w * 0.34), int(w * 0.66)
    cy0, cy1 = int(h * 0.20), int(h * 0.72)
    a_med = float(np.median(A[cy0:cy1, cx0:cx1]))
    b_med = float(np.median(B[cy0:cy1, cx0:cx1]))
    dc = np.hypot(A - a_med, B - b_med)
    in_core = L[cy0:cy1, cx0:cx1][dc[cy0:cy1, cx0:cx1] <= 20.0]
    l_med = float(np.median(in_core)) if in_core.size > 20 else 50.0
    chroma_med = float(math.hypot(a_med, b_med))
    if chroma_med < 9.5:
        return None

    tol_c = np.where(L < 28.0, 28.0, 17.5)
    panel = ((dc <= tol_c) & (np.abs(L - l_med) <= 32.0)).astype(np.float32)

    y0_b, y1_b = int(h * 0.18), int(h * 0.76)
    col_frac = panel[y0_b:y1_b, :].mean(axis=0)
    k_col = max(3, int(round(w * 0.014)) | 1)
    sm_col = np.convolve(col_frac, np.ones(k_col) / k_col, mode="same")

    runs = []
    in_run = False
    start = 0
    for x in range(w):
        if sm_col[x] >= 0.30 and not in_run:
            in_run = True
            start = x
        elif sm_col[x] < 0.25 and in_run:
            runs.append((start, x - 1))
            in_run = False
    if in_run:
        runs.append((start, w - 1))

    merged: list[tuple[int, int]] = []
    max_gap_x = max(3, int(w * 0.015))
    for r0, r1 in runs:
        if (
            merged
            and r0 - merged[-1][1] <= max_gap_x
            and float(sm_col[merged[-1][1] : r0 + 1].min()) >= 0.16
        ):
            merged[-1] = (merged[-1][0], r1)
        else:
            merged.append((r0, r1))

    best_x = None
    best_sx = -1.0
    xmid = w * 0.5
    for r0, r1 in merged:
        rw = r1 - r0 + 1
        if rw < w * 0.24:
            continue
        ov = max(0.0, min(float(r1), w * 0.72) - max(float(r0), w * 0.28))
        sc = ov * 3.0 + rw - abs((r0 + r1) * 0.5 - xmid) * 0.8
        if sc > best_sx:
            best_sx = sc
            best_x = (r0, r1)
    if not best_x:
        return None
    rx0, rx1 = best_x

    row_frac = panel[:, rx0 : rx1 + 1].mean(axis=1)
    k_row = max(3, int(round(h * 0.012)) | 1)
    sm_row = np.convolve(row_frac, np.ones(k_row) / k_row, mode="same")

    rruns = []
    in_run = False
    start = 0
    for y in range(h):
        if sm_row[y] >= 0.34 and not in_run:
            in_run = True
            start = y
        elif sm_row[y] < 0.28 and in_run:
            rruns.append((start, y - 1))
            in_run = False
    if in_run:
        rruns.append((start, h - 1))

    rmerged: list[tuple[int, int]] = []
    max_gap_y = max(3, int(h * 0.02))
    for r0, r1 in rruns:
        if (
            rmerged
            and r0 - rmerged[-1][1] <= max_gap_y
            and float(sm_row[rmerged[-1][1] : r0 + 1].min()) >= 0.16
        ):
            rmerged[-1] = (rmerged[-1][0], r1)
        else:
            rmerged.append((r0, r1))

    best_y = None
    best_sy = -1.0
    ymid = h * 0.45
    for r0, r1 in rmerged:
        rh = r1 - r0 + 1
        if rh < h * 0.35:
            continue
        ov = max(0.0, min(float(r1), h * 0.72) - max(float(r0), h * 0.18))
        sc = ov * 3.0 + rh - abs((r0 + r1) * 0.5 - ymid) * 0.6
        if sc > best_sy:
            best_sy = sc
            best_y = (r0, r1)
    if not best_y:
        return None
    ry0, ry1 = best_y

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(gray[ry0 : ry1 + 1, :], cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
    gy = np.abs(cv2.Sobel(gray[:, rx0 : rx1 + 1], cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
    px, py = max(4, int(w * 0.035)), max(4, int(h * 0.035))
    if rx0 > px:
        rx0 = max(0, rx0 - px) + int(np.argmax(gx[max(0, rx0 - px) : min(w, rx0 + px)]))
    if rx1 < w - px:
        rx1 = max(0, rx1 - px) + int(np.argmax(gx[max(0, rx1 - px) : min(w, rx1 + px)]))
    if ry0 > py:
        ry0 = max(0, ry0 - py) + int(np.argmax(gy[max(0, ry0 - py) : min(h, ry0 + py)]))
    if ry1 < h - py:
        # Door bottom shadow crease across [68%..ry1+py] catches wood floor continuity
        y_lo = max(ry0 + int((ry1 - ry0) * 0.68), ry1 - int(h * 0.18))
        y_hi = min(h, ry1 + py)
        row_dark = (L[y_lo:y_hi, rx0 : rx1 + 1] < 24.0).mean(axis=1)
        row_edge = gy[y_lo:y_hi]
        score = row_edge + row_dark * 55.0
        # Reward clear dark gap spanning >45% of panel width near bottom
        ry1 = y_lo + int(np.argmax(score))

    rw, rh = rx1 - rx0 + 1, ry1 - ry0 + 1
    if rw < w * 0.24 or rh < h * 0.35:
        return None
    has_bg = (
        rx0 > w * 0.025 or rx1 < w * (1.0 - 0.025) or ry0 > h * 0.025 or ry1 < h * (1.0 - 0.025)
    )
    panel_sub = panel[ry0 : ry1 + 1, rx0 : rx1 + 1]
    density = float(panel_sub.mean()) if panel_sub.size > 0 else 0.0
    if density < 0.45:
        return None
    return {
        "x0": int(rx0),
        "y0": int(ry0),
        "x1": int(rx1),
        "y1": int(ry1),
        "has_bg": bool(has_bg),
        "chroma": chroma_med,
        "l_med": l_med,
        "dc": dc,
        "L": L,
    }


def _inpaint_watermarks(sub_bgr: np.ndarray, dc_sub: np.ndarray, L_sub: np.ndarray, l_med: float) -> tuple[np.ndarray, np.ndarray]:
    """Suppress neutral bright overlays, anti-aliased fringe ink text & chromatic stamps."""
    b, g, r = cv2.split(sub_bgr)
    rgb_sat = np.maximum(np.maximum(r, g), b).astype(np.int16) - np.minimum(np.minimum(r, g), b).astype(np.int16)
    luma_raw = (0.2126 * r + 0.7152 * g + 0.0722 * b)

    is_bright_wm = (L_sub > l_med + 12.0) & (rgb_sat < 26) & (dc_sub > 16.0)
    is_neutral_ink = (luma_raw < 35.0) & (rgb_sat < 14) & (dc_sub > 15.0)
    is_logo_accent = (r.astype(np.int16) - g.astype(np.int16) > 36) & (dc_sub > 18.0)

    cand = is_bright_wm | is_neutral_ink | is_logo_accent
    tot_px = sub_bgr.shape[0] * sub_bgr.shape[1]
    wm_u8 = np.zeros((sub_bgr.shape[0], sub_bgr.shape[1]), dtype=np.uint8)
    if 0 < int(cand.sum()) < int(tot_px * 0.045):
        n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8) * 255)
        max_comp = int(tot_px * 0.015)
        for ci in range(1, n_lbl):
            if 4 <= int(stats[ci, cv2.CC_STAT_AREA]) <= max_comp:
                wm_u8[lbls == ci] = 255
    if int(wm_u8.sum()) > 0:
        wm_u8 = cv2.dilate(wm_u8, np.ones((3, 3), np.uint8), iterations=1)
        cleaned = cv2.inpaint(sub_bgr, wm_u8, 4, cv2.INPAINT_TELEA)
        return cleaned, wm_u8
    return sub_bgr, wm_u8


def _classify_graphic_logo(bgra: np.ndarray) -> bool:
    """Detect sharp bimodal 2D graphics, logos, typography, or transparent cutouts."""
    h, w = bgra.shape[:2]
    alpha = bgra[:, :, 3]
    if float((alpha < 220).mean()) > 0.05:
        return True
    gray = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2GRAY)
    th, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo = gray[gray <= th]
    hi = gray[gray > th]
    if lo.size < 20 or hi.size < 20:
        return False
    m1, m2 = float(lo.mean()), float(hi.mean())
    if abs(m2 - m1) < 58.0:
        return False
    tot = gray.size
    w1, w2 = lo.size / tot, hi.size / tot
    var_b = w1 * w2 * (m1 - m2) ** 2
    var_t = float(gray.var()) + 1e-6
    separability = var_b / var_t
    extreme = float(((gray < 48) | (gray > 208)).mean())
    # Flat color logos vs textured wood plaques: logos have uniform interiors (< variance inside modes)
    mode_std = 0.5 * (float(lo.std()) + float(hi.std()))
    return separability >= 0.74 and extreme >= 0.65 and mode_std < 34.0


def _skeleton_to_open_polylines(skel: np.ndarray, min_pts: int = 16) -> list[list[list[float]]]:
    H, W = skel.shape
    kern = np.ones((3, 3), np.uint8)
    kern[1, 1] = 0
    nb_cnt = cv2.filter2D(skel, cv2.CV_8U, kern)
    clean = skel.copy()
    clean[(skel > 0) & (nb_cnt >= 3)] = 0

    vis = np.zeros_like(clean)
    chains = []
    nbs = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    deg = cv2.filter2D(clean, cv2.CV_8U, kern)

    for pass_end in (True, False):
        ys, xs = np.where((clean > 0) & ((deg == 1) if pass_end else (deg >= 2)))
        for sy, sx in zip(ys, xs):
            if vis[sy, sx]:
                continue
            pts = []
            cy, cx = int(sy), int(sx)
            while True:
                vis[cy, cx] = 1
                pts.append((float(cx), float(cy)))
                nxt = None
                for dx, dy in nbs:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < W and 0 <= ny < H and clean[ny, nx] and not vis[ny, nx]:
                        nxt = (nx, ny)
                        break
                if nxt is None:
                    break
                cx, cy = nxt
            if len(pts) >= min_pts:
                arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
                approx = cv2.approxPolyDP(arr, 1.45, False).reshape(-1, 2)
                if len(approx) >= 2:
                    chains.append([[float(p[0]), float(p[1])] for p in approx])
    return chains


_CMD_RE = re.compile(r"[MmLlHhVvCcQqZz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _parse_svg_paths_to_entities(
    svg_str: str,
    offset_x: float,
    offset_y: float,
    scale_x: float,
    scale_y: float,
    layer_group: str = "Vector Trace",
) -> list[dict[str, Any]]:
    """Convert VTracer hierarchical SVG `<path>` elements into JDPaint compound CAD entities."""
    entities: list[dict[str, Any]] = []
    tags = re.findall(r"<path\b([^>]*)>", svg_str)
    for attrs in tags:
        d_m = re.search(r'\bd="([^"]+)"', attrs)
        if not d_m:
            continue
        fill_m = re.search(r'\bfill="([^"]+)"', attrs)
        fill_col = fill_m.group(1) if fill_m else "#38bdf8"
        tr_m = re.search(r'transform="translate\(([^,)]+),([^)]+)\)"', attrs)
        tx = float(tr_m.group(1)) if tr_m else 0.0
        ty = float(tr_m.group(2)) if tr_m else 0.0

        tokens = _CMD_RE.findall(d_m.group(1))
        rings: list[list[dict[str, Any]]] = []
        cur_ring: list[dict[str, Any]] = []
        cx, cy = 0.0, 0.0
        sx0, sy0 = 0.0, 0.0
        idx = 0
        cmd = ""

        def to_mm(px: float, py: float) -> tuple[float, float]:
            return (
                round(offset_x + (px + tx) * scale_x, 4),
                round(offset_y + (py + ty) * scale_y, 4),
            )

        def push_anchor(px: float, py: float, corner: bool = True, h_in=None, h_out=None):
            mx, my = to_mm(px, py)
            node: dict[str, Any] = {"x": mx, "y": my, "corner": corner, "hIn": None, "hOut": None}
            if h_in is not None:
                node["hIn"] = {"x": to_mm(h_in[0], h_in[1])[0], "y": to_mm(h_in[0], h_in[1])[1]}
            if h_out is not None:
                node["hOut"] = {"x": to_mm(h_out[0], h_out[1])[0], "y": to_mm(h_out[0], h_out[1])[1]}
            cur_ring.append(node)

        def finish_ring():
            nonlocal cur_ring
            if len(cur_ring) >= 3:
                first = cur_ring[0]
                last = cur_ring[-1]
                if math.hypot(last["x"] - first["x"], last["y"] - first["y"]) < 1e-3:
                    if last.get("hIn") is not None:
                        first["hIn"] = last["hIn"]
                    cur_ring.pop()
                if len(cur_ring) >= 3:
                    rings.append(cur_ring)
            cur_ring = []

        while idx < len(tokens):
            tok = tokens[idx]
            if tok.isalpha():
                cmd = tok
                idx += 1
                if cmd in ("Z", "z"):
                    cx, cy = sx0, sy0
                    finish_ring()
                continue

            if cmd in ("M", "m"):
                x = float(tokens[idx]); y = float(tokens[idx + 1]); idx += 2
                if cmd == "m":
                    x += cx; y += cy
                finish_ring()
                cx, cy = x, y
                sx0, sy0 = x, y
                push_anchor(cx, cy, corner=True)
                cmd = "L" if cmd == "M" else "l"
            elif cmd in ("L", "l"):
                x = float(tokens[idx]); y = float(tokens[idx + 1]); idx += 2
                if cmd == "l":
                    x += cx; y += cy
                cx, cy = x, y
                push_anchor(cx, cy, corner=True)
            elif cmd in ("H", "h"):
                x = float(tokens[idx]); idx += 1
                if cmd == "h":
                    x += cx
                cx = x
                push_anchor(cx, cy, corner=True)
            elif cmd in ("V", "v"):
                y = float(tokens[idx]); idx += 1
                if cmd == "v":
                    y += cy
                cy = y
                push_anchor(cx, cy, corner=True)
            elif cmd in ("C", "c"):
                x1, y1, x2, y2, x, y = (float(tokens[idx + k]) for k in range(6))
                idx += 6
                if cmd == "c":
                    x1 += cx; y1 += cy; x2 += cx; y2 += cy; x += cx; y += cy
                if cur_ring:
                    cur_ring[-1]["hOut"] = {
                        "x": to_mm(x1, y1)[0],
                        "y": to_mm(x1, y1)[1],
                    }
                    cur_ring[-1]["corner"] = False
                cx, cy = x, y
                push_anchor(cx, cy, corner=False, h_in=(x2, y2))
            elif cmd in ("Q", "q"):
                qx, qy, x, y = (float(tokens[idx + k]) for k in range(4))
                idx += 4
                if cmd == "q":
                    qx += cx; qy += cy; x += cx; y += cy
                c1x = cx + (2.0 / 3.0) * (qx - cx)
                c1y = cy + (2.0 / 3.0) * (qy - cy)
                c2x = x + (2.0 / 3.0) * (qx - x)
                c2y = y + (2.0 / 3.0) * (qy - y)
                if cur_ring:
                    cur_ring[-1]["hOut"] = {
                        "x": to_mm(c1x, c1y)[0],
                        "y": to_mm(c1x, c1y)[1],
                    }
                    cur_ring[-1]["corner"] = False
                cx, cy = x, y
                push_anchor(cx, cy, corner=False, h_in=(c2x, c2y))
            else:
                idx += 1

        finish_ring()
        if not rings:
            continue

        # Sort rings by signed area magnitude: largest is outer boundary, enclosed opposite-orientation rings are holes
        def ring_area(r):
            a = 0.0
            for i in range(len(r)):
                p = r[i]; q = r[(i + 1) % len(r)]
                a += p["x"] * q["y"] - q["x"] * p["y"]
            return abs(a) * 0.5

        rings.sort(key=ring_area, reverse=True)
        outer = rings[0]
        if ring_area(outer) < 0.04:
            continue
        holes = [r for r in rings[1:] if ring_area(r) >= 0.025]

        entities.append({
            "type": "curve",
            "points": outer,
            "holes": holes,
            "isClosed": True,
            "color": "#38bdf8",
            "fillColor": fill_col,
            "strokeColor": "#38bdf8",
            "width": 1.3,
            "name": f"{layer_group} #{len(entities) + 1}",
            "layerGroup": layer_group,
            "toneLevel": 2,
            "toneTotal": 2,
            "relief": {
                "enabled": True,
                "shape": "smooth",
                "heightMm": 3.5,
                "radiusMm": 2.4,
                "mode": "max",
            },
        })
    return entities


def _extract_wood_relief_vectors(
    sub_bgr: np.ndarray,
    off_x: float,
    off_y: float,
    sx_mm: float,
    sy_mm: float,
    detail: float,
    corners: float,
) -> list[dict[str, Any]]:
    """De-grain carved relief wood surfaces and extract smooth closed VTracer Cubic Bezier outlines."""
    vh, vw = sub_bgr.shape[:2]
    den = cv2.fastNlMeansDenoisingColored(sub_bgr, None, 6, 6, 7, 21)
    epf = cv2.edgePreservingFilter(den, flags=cv2.NORMCONV_FILTER, sigma_s=18, sigma_r=0.20)
    lab = cv2.cvtColor(epf, cv2.COLOR_BGR2LAB)
    L_u8 = lab[:, :, 0]
    L_sm = cv2.bilateralFilter(L_u8, d=7, sigmaColor=22, sigmaSpace=16).astype(np.float32)

    r_f = max(5, int(round(min(vw, vh) * 0.0065)) | 1)
    r_m = max(11, int(round(min(vw, vh) * 0.018)) | 1)
    bh_f = cv2.morphologyEx(L_sm, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_f, r_f)))
    bh_m = cv2.morphologyEx(L_sm, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_m, r_m)))
    bh = bh_f * 0.62 + bh_m * 0.58

    gx = cv2.Scharr(L_sm, cv2.CV_32F, 1, 0) / 16.0
    gy = cv2.Scharr(L_sm, cv2.CV_32F, 0, 1) / 16.0
    gmag = np.hypot(gx, gy)

    detail_bias = (0.70 - float(np.clip(detail, 0.15, 1.0))) * 14.0
    bh_med = float(np.median(bh))
    bh_mad = float(np.median(np.abs(bh - bh_med)))
    bh_floor = max(float(np.percentile(bh, 76)), bh_med + 3.6 * max(bh_mad, 1.25), 11.5) + detail_bias * 0.25

    g_med = float(np.median(gmag))
    g_mad = float(np.median(np.abs(gmag - g_med)))
    g_floor = max(float(np.percentile(gmag, 74)), g_med + 3.1 * max(g_mad, 1.25), 10.5) + detail_bias * 0.25

    bh_score = np.clip((bh - bh_floor) * 3.1, 0, 255)
    g_score = np.clip((gmag - g_floor) * 2.9, 0, 255) * 0.92
    comb = np.maximum(bh_score, g_score).astype(np.uint8)

    _, mask = cv2.threshold(comb, max(16, int(round(26 + detail_bias * 0.5))), 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Measure local carving density to filter flat bare-stile wood grain speckles while keeping strong flutes & rails
    k_act = max(21, int(round(min(vw, vh) * 0.030)) | 1)
    act_map = cv2.GaussianBlur((comb > 46).astype(np.float32), (k_act, k_act), 0)

    num_l, lbls, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean_mask = np.zeros_like(mask)
    min_area = max(30, int(vw * vh * 0.000030))
    max_grain_w = max(16, int(vw * 0.014))
    for i in range(1, num_l):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        bx_i = int(stats[i, cv2.CC_STAT_LEFT])
        by_i = int(stats[i, cv2.CC_STAT_TOP])
        bw_i = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_i = int(stats[i, cv2.CC_STAT_HEIGHT])
        comp_m = lbls[by_i : by_i + bh_i, bx_i : bx_i + bw_i] == i
        sub_c = comb[by_i : by_i + bh_i, bx_i : bx_i + bw_i][comp_m]
        p75 = float(np.percentile(sub_c, 75))
        mn = float(sub_c.mean())
        pk = int(sub_c.max())
        act_mn = float(act_map[by_i : by_i + bh_i, bx_i : bx_i + bw_i][comp_m].mean())
        if p75 < 56.0 and area < min_area * 12:
            continue
        if act_mn < 0.045 and pk < 102 and area < min_area * 12:
            continue
        if bh_i >= bw_i * 4.0 and bw_i <= max_grain_w and mn < 74.0 and p75 < 92.0 and bh_i < vh * 0.22:
            continue
        if area < min_area * 4 and pk < 88 and mn < 48:
            continue
        clean_mask[lbls == i] = 255

    bw_img = 255 - clean_mask
    ok_b, pb = cv2.imencode(".png", cv2.cvtColor(bw_img, cv2.COLOR_GRAY2BGR))
    if not ok_b:
        return []
    svg_wood = vtracer.convert_raw_image_to_svg(
        bytes(pb),
        img_format="png",
        colormode="binary",
        mode="spline",
        filter_speckle=max(5, int(round(12 - detail * 6))),
        corner_threshold=int(round(36 + corners * 16)),
        length_threshold=4.0,
        splice_threshold=45,
        path_precision=4,
    )
    ents = _parse_svg_paths_to_entities(svg_wood, off_x, off_y, sx_mm, sy_mm, "Carved Grooves & Panels")
    span_w = vw * sx_mm
    span_h = vh * sy_mm
    cleaned_ents: list[dict[str, Any]] = []
    for ent in ents:
        pts = ent.get("points") or []
        if not pts:
            continue
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        if (max(xs) - min(xs)) >= span_w * 0.94 and (max(ys) - min(ys)) >= span_h * 0.94:
            continue
        ent["fillColor"] = None
        ent["strokeColor"] = "#38bdf8"
        ent["color"] = "#38bdf8"
        ent["relief"] = {"enabled": False, "shape": "smooth", "heightMm": 2.0, "radiusMm": 1.2, "mode": "max"}
        cleaned_ents.append(ent)
    return cleaned_ents


def handle_vectorize(req: dict[str, Any]) -> dict[str, Any]:
    bgra = _decode_image_b64(req["image_b64"])
    bgr = bgra[:, :, :3]
    ih, iw = bgr.shape[:2]
    trace_mode = str(req.get("trace_mode", "auto"))
    detail = float(req.get("detail", 0.7))
    corners = float(req.get("corners", 0.65))
    color_count = int(req.get("color_count", 6))
    ignore_bg = bool(req.get("ignore_bg", True))
    width_mm = float(req.get("width_mm", 100.0))
    height_mm = float(req.get("height_mm", 100.0))

    # Compute exact aspect-fit workspace bounds matching ImageVectorizer.getImageFitBounds
    img_aspect = iw / max(1, ih)
    box_aspect = width_mm / max(1e-6, height_mm)
    if img_aspect > box_aspect:
        bw = width_mm
        bh = width_mm / img_aspect
        bx = 0.0
        by = (height_mm - bh) * 0.5
    else:
        bh = height_mm
        bw = height_mm * img_aspect
        bx = (width_mm - bw) * 0.5
        by = 0.0

    is_logo = _classify_graphic_logo(bgra)
    roi = None if (is_logo or trace_mode in ("bw-logo", "lineart", "color") or not ignore_bg) else detect_furniture_roi(bgr)

    if roi is not None:
        x0, y0, x1, y1 = roi["x0"], roi["y0"], roi["x1"], roi["y1"]
        sub = bgr[y0 : y1 + 1, x0 : x1 + 1].copy()
        sub, _ = _inpaint_watermarks(
            sub, roi["dc"][y0 : y1 + 1, x0 : x1 + 1], roi["L"][y0 : y1 + 1, x0 : x1 + 1], roi["l_med"]
        )
    else:
        x0, y0, x1, y1 = 0, 0, iw - 1, ih - 1
        sub = bgr.copy()
        if float((bgra[:, :, 3] < 220).mean()) > 0.05:
            # Composite transparent logo cleanly onto white background for binary VTracer
            alpha_f = bgra[:, :, 3:4].astype(np.float32) / 255.0
            sub = np.clip(sub.astype(np.float32) * alpha_f + 255.0 * (1.0 - alpha_f), 0, 255).astype(np.uint8)

    sh0, sw0 = sub.shape[:2]
    if roi is not None:
        if min(sw0, sh0) < 1200:
            sc = min(3.2, 1200.0 / max(1, min(sw0, sh0)))
            if max(sw0, sh0) * sc > 2800:
                sc = 2800.0 / max(1, max(sw0, sh0))
            if sc > 1.02:
                sub = cv2.resize(sub, (int(round(sw0 * sc)), int(round(sh0 * sc))), interpolation=cv2.INTER_LANCZOS4)
    elif max(sw0, sh0) < 1100:
        sc = min(2.4, 1280.0 / max(1, max(sw0, sh0)))
        sub = cv2.resize(sub, (int(round(sw0 * sc)), int(round(sh0 * sc))), interpolation=cv2.INTER_LANCZOS4)

    vh, vw = sub.shape[:2]
    off_x = bx + (x0 / iw) * bw
    off_y = by + (y0 / ih) * bh
    span_w = ((x1 - x0 + 1) / iw) * bw
    span_h = ((y1 - y0 + 1) / ih) * bh
    sx_mm = span_w / max(1, vw)
    sy_mm = span_h / max(1, vh)

    entities: list[dict[str, Any]] = []
    # Check if denoised wood ROI is a high-contrast 2-tone engraved wood plaque (e.g. Khoj24) vs continuous 3D relief
    is_engraved_wood_plaque = False
    if roi is not None and trace_mode in ("auto", "wood"):
        den_plaque = cv2.fastNlMeansDenoisingColored(sub, None, 5, 5, 7, 21)
        for _ in range(3):
            den_plaque = cv2.bilateralFilter(den_plaque, d=7, sigmaColor=20, sigmaSpace=20)
        lab_p = cv2.cvtColor(den_plaque, cv2.COLOR_BGR2LAB)
        lab_p[:, :, 0] = cv2.createCLAHE(clipLimit=1.55, tileGridSize=(8, 8)).apply(lab_p[:, :, 0])
        gray_r = cv2.cvtColor(cv2.cvtColor(lab_p, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY)
        otsu_t, _ = cv2.threshold(gray_r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lo_r = gray_r[gray_r <= otsu_t]
        hi_r = gray_r[gray_r > otsu_t]
        if lo_r.size > 40 and hi_r.size > 40:
            m1_r, m2_r = float(lo_r.mean()), float(hi_r.mean())
            w1_r, w2_r = lo_r.size / gray_r.size, hi_r.size / gray_r.size
            sep_r = (w1_r * w2_r * (m1_r - m2_r) ** 2) / (float(gray_r.var()) + 1e-6)
            if sep_r >= 0.78 and abs(m2_r - m1_r) >= 62.0:
                is_engraved_wood_plaque = True
                dark_design = hi_r.size > lo_r.size
                gray_raw = cv2.cvtColor(den_plaque, cv2.COLOR_BGR2GRAY)
                dw_s, dh_s = max(32, vw // 4), max(32, vh // 4)
                sm_g = cv2.resize(gray_raw, (dw_s, dh_s), interpolation=cv2.INTER_AREA)
                bg_small = cv2.medianBlur(sm_g, max(15, int(round(min(dw_s, dh_s) * 0.24)) | 1))
                bg_est = cv2.resize(bg_small, (vw, vh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                diff_f = np.clip((bg_est - gray_raw.astype(np.float32)) if dark_design else (gray_raw.astype(np.float32) - bg_est), 0, 255).astype(np.uint8)
                d_th, _ = cv2.threshold(diff_f, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                otsu_raw, _ = cv2.threshold(gray_raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                bw_diff = diff_f >= max(24, int(d_th * 0.78))
                bw_core = (gray_raw <= int(otsu_raw * 0.94)) if dark_design else (gray_raw >= int(otsu_raw * 1.06))
                bw_mask = np.where(bw_diff | bw_core, 255, 0).astype(np.uint8)
                bw_mask = cv2.morphologyEx(bw_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                bw_mask = cv2.morphologyEx(bw_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                bw_eng = 255 - bw_mask
                ok_e, pb_e = cv2.imencode(".png", cv2.cvtColor(bw_eng, cv2.COLOR_GRAY2BGR))
                if ok_e:
                    svg_eng = vtracer.convert_raw_image_to_svg(
                        bytes(pb_e),
                        img_format="png",
                        colormode="binary",
                        mode="spline",
                        filter_speckle=max(10, int(24 - detail * 10)),
                        corner_threshold=int(34 + corners * 30),
                        length_threshold=6.5,
                        splice_threshold=55,
                        path_precision=4,
                    )
                    entities.extend(
                        _parse_svg_paths_to_entities(svg_eng, off_x, off_y, sx_mm, sy_mm, "Engraved Wood Design")
                    )

    if not is_engraved_wood_plaque and (is_logo or trace_mode in ("bw-logo", "lineart")):
        den_logo = cv2.fastNlMeansDenoisingColored(sub, None, 4, 4, 7, 21)
        gray = cv2.cvtColor(den_logo, cv2.COLOR_BGR2GRAY)
        if np.median(np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])) < 120:
            gray = 255 - gray
        ok2, pb2 = cv2.imencode(".png", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        svg_bin = vtracer.convert_raw_image_to_svg(
            bytes(pb2),
            img_format="png",
            colormode="binary",
            mode="spline",
            filter_speckle=max(4, int(14 - detail * 10)),
            corner_threshold=int(32 + corners * 32),
            length_threshold=4.0,
            splice_threshold=45,
            path_precision=4,
        )
        entities.extend(
            _parse_svg_paths_to_entities(svg_bin, off_x, off_y, sx_mm, sy_mm, "BW Vector")
        )
    elif roi is not None and not is_engraved_wood_plaque:
        wood_ents = _extract_wood_relief_vectors(sub, off_x, off_y, sx_mm, sy_mm, detail, corners)
        entities.extend(wood_ents)
        if len(entities) < 10:
            den_c = cv2.fastNlMeansDenoisingColored(sub, None, 5, 5, 7, 21)
            ok_c, png_c = cv2.imencode(".png", den_c)
            if ok_c:
                svg_col = vtracer.convert_raw_image_to_svg(
                    bytes(png_c),
                    img_format="png",
                    colormode="color",
                    hierarchical="stacked",
                    mode="spline",
                    filter_speckle=max(12, int(36 - detail * 18)),
                    color_precision=min(7, max(4, int(round(color_count * 0.7)))),
                    layer_difference=12,
                    corner_threshold=int(32 + corners * 32),
                    length_threshold=5.0,
                    splice_threshold=45,
                    path_precision=4,
                )
                for ent in _parse_svg_paths_to_entities(svg_col, off_x, off_y, sx_mm, sy_mm, "Vector Layer"):
                    pts = ent.get("points") or []
                    if pts:
                        xs = [p["x"] for p in pts]
                        ys = [p["y"] for p in pts]
                        if (max(xs) - min(xs)) >= span_w * 0.94 and (max(ys) - min(ys)) >= span_h * 0.94:
                            continue
                    entities.append(ent)

        if roi["has_bg"]:
            entities.insert(0, {
                "type": "rect",
                "x": round(off_x, 4),
                "y": round(off_y, 4),
                "w": round(span_w, 4),
                "h": round(span_h, 4),
                "cornerRadius": 0,
                "isClosed": True,
                "color": "#22d3ee",
                "fillColor": None,
                "strokeColor": "#22d3ee",
                "width": 1.4,
                "name": "Panel Perimeter Frame",
                "layerGroup": "Architectural Frame",
                "toneLevel": 1,
                "toneTotal": 2,
                "relief": {"enabled": False, "shape": "flat", "heightMm": 0.4, "radiusMm": 1.0, "mode": "max"},
            })
    elif not is_engraved_wood_plaque:
        den_c = cv2.fastNlMeansDenoisingColored(sub, None, 5, 5, 7, 21)
        ok_c, png_c = cv2.imencode(".png", den_c)
        if not ok_c:
            raise RuntimeError("Failed encoding denoised PNG for VTracer")
        svg_col = vtracer.convert_raw_image_to_svg(
            bytes(png_c),
            img_format="png",
            colormode="color",
            hierarchical="stacked",
            mode="spline",
            filter_speckle=max(12, int(36 - detail * 18)),
            color_precision=min(7, max(4, int(round(color_count * 0.7)))),
            layer_difference=max(5, int(18 - color_count)),
            corner_threshold=int(32 + corners * 32),
            length_threshold=5.0,
            splice_threshold=45,
            path_precision=4,
        )
        raw_ents = _parse_svg_paths_to_entities(svg_col, off_x, off_y, sx_mm, sy_mm, "Vector Layer")
        for ent in raw_ents:
            if ignore_bg and ent["points"]:
                xs = [p["x"] for p in ent["points"]]
                ys = [p["y"] for p in ent["points"]]
                if (max(xs) - min(xs)) >= span_w * 0.94 and (max(ys) - min(ys)) >= span_h * 0.94:
                    continue
            entities.append(ent)

    return {
        "ok": True,
        "engine": "opencv-vtracer-vision",
        "entities": entities[:1100],
        "roi": None if roi is None else {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "has_bg": roi["has_bg"]},
    }


def _solve_photometric_sfs_fft(bgr: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """Frankot-Chellappa directional gradient SFS recovery with specular glare clamping."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    fg = fg_mask > 0
    if not fg.any():
        return np.zeros((h, w), dtype=np.float32)

    p92 = float(np.percentile(gray[fg], 92))
    p06 = float(np.percentile(gray[fg], 6))
    clamped = np.clip((gray - p06) / max(0.08, p92 - p06), 0.0, 1.0)
    clamped = cv2.bilateralFilter(clamped.astype(np.float32), d=5, sigmaColor=0.14, sigmaSpace=8.0)

    gx = cv2.Sobel(clamped, cv2.CV_32F, 1, 0, ksize=3) * 0.25
    gy = cv2.Sobel(clamped, cv2.CV_32F, 0, 1, ksize=3) * 0.25
    # Estimate dominant directional illumination axis from high-contrast dipoles
    gmag = np.hypot(gx, gy)
    strong = fg & (gmag > float(np.percentile(gmag[fg], 78)))
    lx, ly = 1.0, 1.0
    if int(strong.sum()) > 16:
        lx_est = float(gx[strong].mean())
        ly_est = float(gy[strong].mean())
        if abs(lx_est) + abs(ly_est) > 1e-4:
            norm_l = math.hypot(lx_est, ly_est)
            lx, ly = lx_est / norm_l, ly_est / norm_l

    # Integrate oriented slope gradient along directional light + gradient crest symmetry
    # In FFT domain: F(z) = (-j*wx*F(p) - j*wy*F(q)) / (wx^2 + wy^2)
    wx = np.fft.fftfreq(w) * 2.0 * math.pi
    wy = np.fft.fftfreq(h) * 2.0 * math.pi
    WX, WY = np.meshgrid(wx, wy)
    denom = WX ** 2 + WY ** 2 + 1e-4
    FP = np.fft.fft2(gx * fg.astype(np.float32))
    FQ = np.fft.fft2(gy * fg.astype(np.float32))
    FZ = (-1j * WX * FP - 1j * WY * FQ) / denom
    z_sfs = np.real(np.fft.ifft2(FZ)).astype(np.float32)

    # Combine oriented SFS surface with signed gradient-dipole crest energy for symmetric faceted relief
    grad_energy = cv2.GaussianBlur(gmag, (5, 5), 1.4)
    vals = z_sfs[fg]
    lo, hi = float(np.percentile(vals, 3)), float(np.percentile(vals, 97))
    z_norm = np.clip((z_sfs - lo) / max(1e-4, hi - lo), 0.0, 1.0)
    ge_hi = max(1e-4, float(np.percentile(grad_energy[fg], 96)))
    crest = np.clip(grad_energy / ge_hi, 0.0, 1.0)
    out = np.clip(z_norm * 0.58 + crest * 0.42, 0.0, 1.0) * fg.astype(np.float32)
    return out


def handle_reconstruct_3d(req: dict[str, Any]) -> dict[str, Any]:
    import torch

    bgra = _decode_image_b64(req["image_b64"])
    bgr = bgra[:, :, :3]
    ih, iw = bgr.shape[:2]
    remove_bg = bool(req.get("remove_bg", True))
    is_logo = _classify_graphic_logo(bgra)
    roi = None if (is_logo or not remove_bg) else detect_furniture_roi(bgr)
    is_planar = bool((not remove_bg) or (roi is not None))

    if roi is not None:
        x0, y0, x1, y1 = roi["x0"], roi["y0"], roi["x1"], roi["y1"]
        crop_bgra = bgra[y0 : y1 + 1, x0 : x1 + 1].copy()
        cleaned_bgr, _ = _inpaint_watermarks(
            crop_bgra[:, :, :3],
            roi["dc"][y0 : y1 + 1, x0 : x1 + 1],
            roi["L"][y0 : y1 + 1, x0 : x1 + 1],
            roi["l_med"],
        )
        crop_bgra[:, :, :3] = cleaned_bgr
    else:
        x0, y0, x1, y1 = 0, 0, iw - 1, ih - 1
        crop_bgra = bgra.copy()

    # Low-res NLM denoise + Lanczos clean restoration for sharp 4K PBR texture
    ch, cw = crop_bgra.shape[:2]
    if max(cw, ch) < 1024 and not is_logo:
        sc = min(2.2, 1280.0 / max(cw, ch))
        nw, nh = int(round(cw * sc)), int(round(ch * sc))
        rgb_up = cv2.resize(crop_bgra[:, :, :3], (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        rgb_up = cv2.fastNlMeansDenoisingColored(rgb_up, None, 4, 4, 7, 21)
        a_up = cv2.resize(crop_bgra[:, :, 3], (nw, nh), interpolation=cv2.INTER_LINEAR)
        crop_bgra = np.dstack([rgb_up, a_up])
        ch, cw = crop_bgra.shape[:2]

    # Run cached Depth-Anything-V2-Small-hf on M1 GPU
    proc, model, device = _ensure_depth_model()
    pil_in = Image.fromarray(cv2.cvtColor(crop_bgra[:, :, :3], cv2.COLOR_BGR2RGB))
    inputs = proc(images=pil_in, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        pred = model(**inputs).predicted_depth
    depth_t = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(ch, cw), mode="bicubic", align_corners=False
    ).squeeze()
    d_np = depth_t.detach().float().cpu().numpy()
    d_lo, d_hi = float(np.percentile(d_np, 2)), float(np.percentile(d_np, 98))
    depth_norm = np.clip((d_np - d_lo) / max(1e-5, d_hi - d_lo), 0.0, 1.0).astype(np.float32)

    # Foreground Mask Extraction tailored strictly per class
    if is_planar:
        fg_mask = np.full((ch, cw), 255, dtype=np.uint8)
        sfs = _solve_photometric_sfs_fft(crop_bgra[:, :, :3], fg_mask)
        depth_norm = np.clip(depth_norm * 0.48 + sfs * 0.52, 0.0, 1.0).astype(np.float32)
    elif is_logo:
        alpha = crop_bgra[:, :, 3]
        if float((alpha < 220).mean()) > 0.05:
            fg_mask = np.where(alpha >= 110, 255, 0).astype(np.uint8)
        else:
            # Bimodal solid backdrop flood-fill + enclosed cutout pocket punch (preserves thin letters!)
            lab_c = cv2.cvtColor(crop_bgra[:, :, :3], cv2.COLOR_BGR2LAB).astype(np.float32)
            border_px = np.concatenate([lab_c[0, :], lab_c[-1, :], lab_c[:, 0], lab_c[:, -1]], axis=0)
            bg_med = np.median(border_px, axis=0)
            de = np.linalg.norm(lab_c - bg_med, axis=2)
            bg_map = (de <= 24.0).astype(np.uint8)
            # Only remove pockets smaller than 22% area so letters O/D cut holes while large solids stay intact
            num_b, blbl, bstats, _ = cv2.connectedComponentsWithStats(bg_map)
            fg_mask = np.full((ch, cw), 255, dtype=np.uint8)
            for bid in range(1, num_b):
                x_s, y_s, w_s, h_s, area = bstats[bid]
                touches_edge = x_s == 0 or y_s == 0 or (x_s + w_s >= cw) or (y_s + h_s >= ch)
                if touches_edge or area < ch * cw * 0.20:
                    fg_mask[blbl == bid] = 0
    else:
        # Real 3D photographed object: fuse Depth + Center-Border Contrast + OpenCV GrabCut
        fg_mask = np.full((ch, cw), 255, dtype=np.uint8)
        if remove_bg:
            gc_mask = np.full((ch, cw), cv2.GC_PR_FGD, dtype=np.uint8)
            pad_x, pad_y = max(2, int(cw * 0.03)), max(2, int(ch * 0.03))
            gc_mask[:pad_y, :] = cv2.GC_PR_BGD
            gc_mask[-pad_y:, :] = cv2.GC_PR_BGD
            gc_mask[:, :pad_x] = cv2.GC_PR_BGD
            gc_mask[:, -pad_x:] = cv2.GC_PR_BGD
            # Deep regions are strong foreground; farthest border depth pixels are sure background
            gc_mask[depth_norm >= float(np.percentile(depth_norm, 62))] = cv2.GC_PR_FGD
            border_d = np.concatenate([depth_norm[0, :], depth_norm[-1, :], depth_norm[:, 0], depth_norm[:, -1]])
            if float(border_d.std()) < 0.16 and float(border_d.mean()) < 0.35:
                gc_mask[depth_norm <= float(border_d.mean()) + 0.06] = cv2.GC_BGD
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(crop_bgra[:, :, :3], gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
                seg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                fg_frac = float((seg > 0).mean())
                if 0.05 < fg_frac < 0.96:
                    fg_mask = seg
            except Exception:
                pass

    # Compute downsample transport buffer (max edge 512px) for browser WebGL mesh construction
    max_t = 512
    sc_t = min(1.0, max_t / max(cw, ch))
    tw, th = max(32, int(round(cw * sc_t))), max(32, int(round(ch * sc_t)))
    d_small = cv2.resize(depth_norm, (tw, th), interpolation=cv2.INTER_AREA)
    fg_small = cv2.resize(fg_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    _, cl_png = cv2.imencode(".png", crop_bgra)
    d_u16 = np.clip(d_small * 65535.0, 0, 65535).astype(np.uint16)

    return {
        "ok": True,
        "engine": f"m1-depth-anything-v2-grabcut ({_DEPTH_DEVICE})",
        "width": tw,
        "height": th,
        "aspect": round(cw / max(1, ch), 6),
        "is_planar_panel": is_planar,
        "is_graphic_logo": is_logo,
        "roi_box": {"x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1, "img_w": iw, "img_h": ih},
        "depth_u16_b64": base64.b64encode(d_u16.tobytes()).decode("ascii"),
        "fg_u8_b64": base64.b64encode(fg_small.tobytes()).decode("ascii"),
        "cleaned_png_b64": base64.b64encode(bytes(cl_png)).decode("ascii"),
    }


def main():
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            op = msg.get("op")
            if op == "ping":
                out = {"ok": True}
            elif op == "vectorize":
                out = handle_vectorize(msg["payload"])
            elif op == "reconstruct_3d":
                out = handle_reconstruct_3d(msg["payload"])
            else:
                out = {"ok": False, "error": f"Unknown op: {op}"}
        except Exception as exc:
            out = {"ok": False, "error": f"{exc}\n{traceback.format_exc()[-600:]}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
