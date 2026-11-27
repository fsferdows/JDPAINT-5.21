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

    # Illumination-normalized chromaticity preserves wood hue/saturation inside dark low-light shadows
    cx0, cx1 = int(w * 0.34), int(w * 0.66)
    cy0, cy1 = int(h * 0.20), int(h * 0.72)
    l_scale = np.clip(48.0 / np.maximum(L, 12.0), 0.75, 2.5)
    A_eq = A * l_scale
    B_eq = B * l_scale
    a_med = float(np.median(A_eq[cy0:cy1, cx0:cx1]))
    b_med = float(np.median(B_eq[cy0:cy1, cx0:cx1]))
    dc_eq = np.hypot(A_eq - a_med, B_eq - b_med)
    dc = np.hypot(A - float(np.median(A[cy0:cy1, cx0:cx1])), B - float(np.median(B[cy0:cy1, cx0:cx1])))
    in_core = L[cy0:cy1, cx0:cx1][dc_eq[cy0:cy1, cx0:cx1] <= 22.0]
    l_med = float(np.median(in_core)) if in_core.size > 20 else 50.0
    chroma_med = float(math.hypot(float(np.median(A[cy0:cy1, cx0:cx1])), float(np.median(B[cy0:cy1, cx0:cx1]))))
    chroma_eq_med = float(math.hypot(a_med, b_med))

    # Support BOTH warm wood furniture (doors/beds) AND bordered neutral/white/gray carved architectural panels
    is_neutral_panel = False
    if chroma_eq_med < 9.0 or chroma_med < 4.2:
        gray_p = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g_sm = cv2.bilateralFilter(gray_p, 5, 18, 12)
        gx_p = np.abs(cv2.Sobel(g_sm, cv2.CV_32F, 1, 0, ksize=3))
        gy_p = np.abs(cv2.Sobel(g_sm, cv2.CV_32F, 0, 1, ksize=3))
        col_str = (gx_p > 22.0).mean(axis=0)
        row_str = (gy_p > 22.0).mean(axis=1)
        has_lr = float(col_str[: int(w * 0.35)].max()) >= 0.34 and float(col_str[int(w * 0.65) :].max()) >= 0.34
        has_tb = float(row_str[: int(h * 0.35)].max()) >= 0.34 and float(row_str[int(h * 0.65) :].max()) >= 0.34
        interior_tex = float(np.hypot(gx_p[cy0:cy1, cx0:cx1], gy_p[cy0:cy1, cx0:cx1]).mean())
        if not (has_lr and has_tb and interior_tex >= 6.5):
            return None
        is_neutral_panel = True

    if is_neutral_panel:
        # Tighter luminance contrast isolates neutral painted doors/beds inside bright room walls
        l_var = float(np.std(L[cy0:cy1, cx0:cx1]))
        l_tol = max(12.0, min(32.0, l_var * 2.2))
        panel = (np.abs(L - l_med) <= l_tol).astype(np.float32)
    else:
        tol_c = np.where(L < 28.0, 28.0, 18.5)
        panel = ((dc_eq <= tol_c) & (np.abs(L - l_med) <= 46.0)).astype(np.float32)

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

    min_h_frac = 0.24 if (w > h * 1.15) else 0.35
    best_y = None
    best_sy = -1.0
    ymid = h * 0.45
    for r0, r1 in rmerged:
        rh = r1 - r0 + 1
        if rh < h * min_h_frac:
            continue
        ov = max(0.0, min(float(r1), h * 0.76) - max(float(r0), h * 0.16))
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
    if rw < w * 0.24 or rh < h * min_h_frac:
        return None
    has_bg = (
        rx0 > w * 0.025 or rx1 < w * (1.0 - 0.025) or ry0 > h * 0.025 or ry1 < h * (1.0 - 0.025)
    )
    panel_sub = panel[ry0 : ry1 + 1, rx0 : rx1 + 1]
    density = float(panel_sub.mean()) if panel_sub.size > 0 else 0.0
    if density < 0.40:
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
    luma_raw = 0.2126 * r + 0.7152 * g + 0.0722 * b

    is_bright_wm = (L_sub > l_med + 14.0) & (rgb_sat < 24) & (dc_sub > 17.0)
    is_neutral_ink = (luma_raw < 34.0) & (rgb_sat < 14) & (dc_sub > 16.0)
    is_red_accent = (r.astype(np.int16) - g.astype(np.int16) > 95) & (dc_sub > 32.0)
    is_blue_accent = (b.astype(np.int16) - r.astype(np.int16) > 28) & (dc_sub > 18.0)

    cand = is_bright_wm | is_neutral_ink | is_red_accent | is_blue_accent
    tot_px = sub_bgr.shape[0] * sub_bgr.shape[1]
    wm_u8 = np.zeros((sub_bgr.shape[0], sub_bgr.shape[1]), dtype=np.uint8)
    if 0 < int(cand.sum()) < int(tot_px * 0.045):
        n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8) * 255)
        max_comp = int(tot_px * 0.015)
        for ci in range(1, n_lbl):
            if 4 <= int(stats[ci, cv2.CC_STAT_AREA]) <= max_comp:
                wm_u8[lbls == ci] = 255
    if int(wm_u8.sum()) > 0:
        wm_u8 = cv2.dilate(wm_u8, np.ones((4, 4), np.uint8), iterations=1)
        cleaned = cv2.inpaint(sub_bgr, wm_u8, 5, cv2.INPAINT_TELEA)
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


def _equalize_illumination_retinex(L_f: np.ndarray, strength: float = 0.58) -> np.ndarray:
    """Multi-scale Retinex reflectance normalization to equalize shadows/low-light without amplifying flat grain."""
    L_pos = np.maximum(L_f.astype(np.float32), 6.0)
    m_dim = min(L_pos.shape[:2])
    sig_s = max(12.0, m_dim * 0.032)
    sig_l = max(34.0, m_dim * 0.115)
    illum_s = cv2.GaussianBlur(L_pos, (0, 0), sigmaX=sig_s)
    illum_l = cv2.GaussianBlur(L_pos, (0, 0), sigmaX=sig_l)
    illum = np.maximum(illum_s * 0.45 + illum_l * 0.55, 10.0)
    # Center local reflectance around local median luminance so shadowed carvings match lit SNR
    base_mid = float(np.clip(np.median(L_pos), 75.0, 155.0))
    refl = np.clip((L_pos / illum) * base_mid, 0.0, 255.0)
    dark_w = np.clip((105.0 - L_pos) / 75.0, 0.0, 1.0)
    mix_w = np.clip(strength * 0.55 + dark_w * strength * 0.45, 0.0, 0.85)
    eq = L_f.astype(np.float32) * (1.0 - mix_w) + refl * mix_w
    return np.clip(eq, 0.0, 255.0).astype(np.float32)


def _solve_planar_relief_heightmap(sub_bgr: np.ndarray) -> np.ndarray:
    """Compute a volumetric de-tilted bas-relief heightmap fusing AI Depth-Anything-V2, SFS & geodesic relief volumes."""
    import torch

    ch, cw = sub_bgr.shape[:2]
    m_dim = min(cw, ch)
    fg_mask = np.full((ch, cw), 255, dtype=np.uint8)
    den = cv2.fastNlMeansDenoisingColored(sub_bgr, None, 5, 5, 7, 21)
    epf_c = cv2.edgePreservingFilter(den, flags=cv2.NORMCONV_FILTER, sigma_s=20, sigma_r=0.20)
    sfs = _solve_photometric_sfs_fft(epf_c, fg_mask)

    # AI Depth-Anything-V2 macro/meso sculptural relief field (immune to plank grain)
    ai_flat = None
    try:
        proc, model, device = _ensure_depth_model()
        pil_in = Image.fromarray(cv2.cvtColor(den, cv2.COLOR_BGR2RGB))
        inputs = {k: v.to(device) for k, v in proc(images=pil_in, return_tensors="pt").items()}
        with torch.no_grad():
            pred = model(**inputs).predicted_depth
        dt_t = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(ch, cw), mode="bicubic", align_corners=False).squeeze()
        ai_raw = dt_t.detach().float().cpu().numpy()
        ai_lo, ai_hi = float(np.percentile(ai_raw, 2)), float(np.percentile(ai_raw, 98))
        ai_norm = np.clip((ai_raw - ai_lo) / max(1e-4, ai_hi - ai_lo), 0.0, 1.0)
        ai_tilt = cv2.GaussianBlur(ai_norm, (0, 0), sigmaX=max(16.0, m_dim * 0.14))
        ai_flat = np.clip(ai_norm * 0.28 + (0.5 + (ai_norm - ai_tilt) * 1.55) * 0.72, 0.0, 1.0)
    except Exception:
        ai_flat = None

    Lc_raw = cv2.bilateralFilter(cv2.cvtColor(epf_c, cv2.COLOR_BGR2LAB)[:, :, 0], 7, 22, 16).astype(np.float32)
    Lc = _equalize_illumination_retinex(Lc_raw, strength=0.62) / 255.0
    k_bg = max(31, int(round(m_dim * 0.14)) | 1)
    bg_l = cv2.GaussianBlur(Lc, (k_bg, k_bg), 0)
    elev = Lc - bg_l
    delight = np.clip(elev * 1.60 + 0.5, 0.0, 1.0)

    # De-tilt broad FFT pillow bow on planar panels
    sfs_bow = cv2.GaussianBlur(sfs, (0, 0), sigmaX=max(18.0, m_dim * 0.11))
    sfs_flat = np.clip(0.5 + (sfs - sfs_bow) * 1.45, 0.0, 1.0)

    # Gate out bare flat wood grain while keeping genuine carved motifs solid
    grad_e = np.hypot(cv2.Scharr(delight, cv2.CV_32F, 1, 0), cv2.Scharr(delight, cv2.CV_32F, 0, 1)) / 16.0
    act_e = cv2.GaussianBlur(np.maximum(grad_e, np.abs(elev) * 2.2), (0, 0), sigmaX=max(6.0, m_dim * 0.015))
    act_floor = max(0.035, float(np.percentile(act_e, 48)))
    grain_gate = np.clip((act_e - act_floor * 0.72) / max(1e-4, act_floor * 1.45), 0.0, 1.0)

    # Geodesic rounded bas-relief dome volume for raised petals & smooth recessed grooves (zero 1px noise needles!)
    pos_bin = (((elev * 1.8) * (0.35 + 0.65 * grain_gate)) > 0.065).astype(np.uint8)
    neg_bin = (((-elev * 1.8) * (0.35 + 0.65 * grain_gate)) > 0.068).astype(np.uint8)
    ker_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pos_bin = cv2.morphologyEx(pos_bin, cv2.MORPH_OPEN, ker_v)
    neg_bin = cv2.morphologyEx(neg_bin, cv2.MORPH_OPEN, ker_v)

    ref_r = max(4.0, m_dim * 0.022)
    dt_pos = cv2.distanceTransform(pos_bin, cv2.DIST_L2, 3)
    dt_neg = cv2.distanceTransform(neg_bin, cv2.DIST_L2, 3)
    dome_pos = np.power(np.clip(dt_pos / ref_r, 0.0, 1.0), 0.58)
    pocket_neg = np.power(np.clip(dt_neg / ref_r, 0.0, 1.0), 0.58)
    vol_raw = 0.5 + dome_pos * 0.44 - pocket_neg * 0.42 + np.clip(elev * grain_gate * 1.15, -0.25, 0.25)
    vol_relief = cv2.GaussianBlur(np.clip(vol_raw, 0.0, 1.0).astype(np.float32), (0, 0), sigmaX=max(0.85, m_dim * 0.0022))

    if ai_flat is not None:
        rel = ai_flat * 0.24 + sfs_flat * 0.26 + vol_relief * 0.50
    else:
        rel = sfs_flat * 0.34 + vol_relief * 0.66

    # Light edge-preserving polish removes micro plank ripple while preserving carved shoulders
    rel = cv2.bilateralFilter(rel.astype(np.float32), d=5, sigmaColor=0.09, sigmaSpace=4.5)
    lo, hi = float(np.percentile(rel, 2)), float(np.percentile(rel, 98))
    return np.clip((rel - lo) / max(1e-4, hi - lo), 0.05, 0.96).astype(np.float32)


def _extract_ai_model_recovery_vectors(
    sub_bgr: np.ndarray,
    off_x: float,
    off_y: float,
    sx_mm: float,
    sy_mm: float,
    corners: float,
) -> list[dict[str, Any]]:
    """AI Depth-Anything + Photometric SFS fallback vectorizer when primary contrast tracking is sparse."""
    import torch

    vh, vw = sub_bgr.shape[:2]
    relief = _solve_planar_relief_heightmap(sub_bgr)
    try:
        proc, model, device = _ensure_depth_model()
        pil_in = Image.fromarray(cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2RGB))
        inputs = {k: v.to(device) for k, v in proc(images=pil_in, return_tensors="pt").items()}
        with torch.no_grad():
            pred = model(**inputs).predicted_depth
        d_t = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(vh, vw), mode="bicubic", align_corners=False).squeeze()
        d_np = d_t.detach().float().cpu().numpy()
        d_lo, d_hi = float(np.percentile(d_np, 3)), float(np.percentile(d_np, 97))
        d_norm = np.clip((d_np - d_lo) / max(1e-4, d_hi - d_lo), 0.0, 1.0)
        relief = np.clip(relief * 0.68 + d_norm * 0.32, 0.0, 1.0)
    except Exception:
        pass

    # Extract balanced multi-level carved silhouette boundaries + adaptive edges
    rel_u8 = np.clip(relief * 255.0, 0, 255).astype(np.uint8)
    edge_c = cv2.Canny(rel_u8, 28, 78)
    t_hi = np.where(rel_u8 >= int(np.percentile(rel_u8, 68)), 255, 0).astype(np.uint8)
    t_lo = np.where(rel_u8 <= int(np.percentile(rel_u8, 26)), 255, 0).astype(np.uint8)
    comb_m = cv2.bitwise_or(cv2.bitwise_xor(t_hi, t_lo), cv2.dilate(edge_c, np.ones((2, 2), np.uint8)))
    comb_m = cv2.morphologyEx(comb_m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    bw_img = 255 - comb_m
    ok_r, pb_r = cv2.imencode(".png", cv2.cvtColor(bw_img, cv2.COLOR_GRAY2BGR))
    if not ok_r:
        return []
    svg_rec = vtracer.convert_raw_image_to_svg(
        bytes(pb_r),
        img_format="png",
        colormode="binary",
        mode="spline",
        filter_speckle=8,
        corner_threshold=int(round(36 + corners * 16)),
        length_threshold=4.2,
        splice_threshold=45,
        path_precision=4,
    )
    ents = _parse_svg_paths_to_entities(svg_rec, off_x, off_y, sx_mm, sy_mm, "Carved Grooves & Panels")
    span_w, span_h = vw * sx_mm, vh * sy_mm
    out: list[dict[str, Any]] = []
    for ent in ents:
        pts = ent.get("points") or []
        if not pts:
            continue
        if not ent.get("holes") and len(pts) <= 6:
            xs = [p["x"] for p in pts]; ys = [p["y"] for p in pts]
            if (max(xs) - min(xs)) >= span_w * 0.94 and (max(ys) - min(ys)) >= span_h * 0.94:
                continue
        ent["fillColor"] = None
        ent["strokeColor"] = "#38bdf8"
        ent["color"] = "#38bdf8"
        ent["name"] = f"AI Relief Contour #{len(out) + 1}"
        ent["relief"] = {"enabled": False, "shape": "smooth", "heightMm": 2.2, "radiusMm": 1.4, "mode": "max"}
        out.append(ent)
    return out


def _pt_line_dist(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-6:
        return math.hypot(px - x1, py - y1)
    return abs((px - x1) * dy - (py - y1) * dx) / seg_len


def _refine_ring_cad(ring: list[dict[str, Any]], is_closed: bool, span_w: float, span_h: float, snap_axis: bool = True) -> list[dict[str, Any]]:
    """Collapse zigzag rails into dead-straight CAD lines and beautify curves with corner-pinned C1 smoothing."""
    if not ring or len(ring) < 2:
        return ring
    min_span = max(1.0, min(span_w, span_h))
    min_pt_d = max(0.05, min_span * 0.0005)
    straight_tol = min(0.24, max(0.065, min_span * 0.0022))
    axis_sin = math.sin(math.radians(1.35))

    # 1. Remove redundant micro-duplicate vertices
    dedup: list[dict[str, Any]] = [{
        "x": float(ring[0]["x"]),
        "y": float(ring[0]["y"]),
        "corner": bool(ring[0].get("corner", True)),
        "hIn": dict(ring[0]["hIn"]) if ring[0].get("hIn") else None,
        "hOut": dict(ring[0]["hOut"]) if ring[0].get("hOut") else None,
    }]
    for i in range(1, len(ring)):
        p = ring[i]
        prev = dedup[-1]
        if math.hypot(p["x"] - prev["x"], p["y"] - prev["y"]) < min_pt_d:
            prev["hOut"] = dict(p["hOut"]) if p.get("hOut") else prev["hOut"]
            continue
        dedup.append({
            "x": float(p["x"]),
            "y": float(p["y"]),
            "corner": bool(p.get("corner", True)),
            "hIn": dict(p["hIn"]) if p.get("hIn") else None,
            "hOut": dict(p["hOut"]) if p.get("hOut") else None,
        })
    if is_closed and len(dedup) > 3:
        if math.hypot(dedup[-1]["x"] - dedup[0]["x"], dedup[-1]["y"] - dedup[0]["y"]) < min_pt_d:
            dedup[0]["hIn"] = dedup[-1]["hIn"]
            dedup.pop()
    if len(dedup) < 2:
        return dedup

    # 2. Straighten individual collinear Cubic segments first
    n = len(dedup)
    for i in range(n if is_closed else n - 1):
        a = dedup[i]
        b = dedup[(i + 1) % n]
        L = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        if L < 1e-5:
            continue
        h1, h2 = a.get("hOut"), b.get("hIn")
        d1 = _pt_line_dist(h1["x"], h1["y"], a["x"], a["y"], b["x"], b["y"]) if h1 else 0.0
        d2 = _pt_line_dist(h2["x"], h2["y"], a["x"], a["y"], b["x"], b["y"]) if h2 else 0.0
        tol_seg = max(straight_tol, min(0.28, L * 0.025))
        if (not h1 and not h2) or (d1 <= tol_seg and d2 <= tol_seg):
            a["hOut"] = None
            b["hIn"] = None

    # 3. Multi-node greedy linear chain simplification (turns zigzag/stepped door rails into single crisp 2-pt lines!)
    n = len(dedup)
    if n > (3 if is_closed else 2):
        simplified: list[dict[str, Any]] = []
        idx = 0
        cos_turn_lim = math.cos(math.radians(28.0))
        while idx < n:
            a = dedup[idx]
            simplified.append(a)
            best_j = idx + 1
            arc_len = 0.0
            limit_j = n if not is_closed else min(n, idx + n - 1)
            for j in range(idx + 1, limit_j):
                p_prev = dedup[(j - 1) % n]
                p_cur = dedup[j % n]
                seg_l = math.hypot(p_cur["x"] - p_prev["x"], p_cur["y"] - p_prev["y"])
                arc_len += seg_l
                if j == idx + 1:
                    continue
                b = dedup[j % n]
                chord_l = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
                if chord_l < 0.1 or chord_l > min_span * 0.72:
                    break
                if arc_len / chord_l > 1.0045:
                    break
                # Stop before crossing a sharp corner turn
                p_last = dedup[(j - 2) % n]
                v1x, v1y = p_prev["x"] - p_last["x"], p_prev["y"] - p_last["y"]
                v2x, v2y = p_cur["x"] - p_prev["x"], p_cur["y"] - p_prev["y"]
                l1, l2 = math.hypot(v1x, v1y), math.hypot(v2x, v2y)
                if l1 > 1e-4 and l2 > 1e-4 and ((v1x * v2x + v1y * v2y) / (l1 * l2)) < cos_turn_lim:
                    break
                # Verify all intermediate nodes & control handles fit cleanly within straight_tol
                max_d = 0.0
                for k in range(idx, j + 1):
                    pk = dedup[k % n]
                    if k > idx and k < j:
                        max_d = max(max_d, _pt_line_dist(pk["x"], pk["y"], a["x"], a["y"], b["x"], b["y"]))
                    if k < j and pk.get("hOut"):
                        max_d = max(max_d, _pt_line_dist(pk["hOut"]["x"], pk["hOut"]["y"], a["x"], a["y"], b["x"], b["y"]))
                    if k > idx and pk.get("hIn"):
                        max_d = max(max_d, _pt_line_dist(pk["hIn"]["x"], pk["hIn"]["y"], a["x"], a["y"], b["x"], b["y"]))
                    if max_d > straight_tol:
                        break
                if max_d <= straight_tol:
                    best_j = j
                else:
                    break
            if best_j > idx + 1:
                a["hOut"] = None
                dedup[best_j % n]["hIn"] = None
            idx = best_j if best_j > idx else idx + 1
        if len(simplified) >= (3 if is_closed else 2):
            dedup = simplified

    # 4. Orthogonal snapping on straight architectural edges >= 1.5% span
    n = len(dedup)
    if snap_axis and n >= 3:
        for i in range(n if is_closed else n - 1):
            a = dedup[i]
            b = dedup[(i + 1) % n]
            if a.get("hOut") is not None or b.get("hIn") is not None:
                continue
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            L = math.hypot(dx, dy)
            if L < min_span * 0.015:
                continue
            if abs(dx) <= L * axis_sin and a.get("corner", True):
                mid_x = round((a["x"] + b["x"]) * 0.5, 4)
                a["x"] = mid_x; b["x"] = mid_x
            elif abs(dy) <= L * axis_sin and a.get("corner", True):
                mid_y = round((a["y"] + b["y"]) * 0.5, 4)
                a["y"] = mid_y; b["y"] = mid_y

    # 5. Corner-pinned micro-fairing on curved segments + clean C1 collinear handles
    n = len(dedup)
    cos_corner = math.cos(math.radians(30.0))
    is_pinned = [True] * n
    for i in range(n):
        pt = dedup[i]
        has_prev = is_closed or i > 0
        has_next = is_closed or i < n - 1
        h_in, h_out = pt.get("hIn"), pt.get("hOut")
        prev_straight = (dedup[(i - 1) % n].get("hOut") is None and h_in is None) if has_prev else True
        next_straight = (h_out is None and dedup[(i + 1) % n].get("hIn") is None) if has_next else True
        if prev_straight or next_straight:
            pt["corner"] = True
            continue
        p_prev = dedup[(i - 1) % n] if has_prev else pt
        p_next = dedup[(i + 1) % n] if has_next else pt
        vx_in = pt["x"] - (h_in["x"] if h_in else p_prev["x"])
        vy_in = pt["y"] - (h_in["y"] if h_in else p_prev["y"])
        vx_out = (h_out["x"] if h_out else p_next["x"]) - pt["x"]
        vy_out = (h_out["y"] if h_out else p_next["y"]) - pt["y"]
        l_in, l_out = math.hypot(vx_in, vy_in), math.hypot(vx_out, vy_out)
        if l_in < 1e-5 or l_out < 1e-5 or ((vx_in * vx_out + vy_in * vy_out) / (l_in * l_out)) < cos_corner:
            pt["corner"] = True
        else:
            pt["corner"] = False
            is_pinned[i] = False

    if n >= 4:
        sm_pts = [(p["x"], p["y"]) for p in dedup]
        for i in range(n):
            if is_pinned[i]:
                continue
            pp = dedup[(i - 1) % n]
            pn = dedup[(i + 1) % n]
            tx = sm_pts[i][0] * 0.76 + (pp["x"] + pn["x"]) * 0.12
            ty = sm_pts[i][1] * 0.76 + (pp["y"] + pn["y"]) * 0.12
            if math.hypot(tx - sm_pts[i][0], ty - sm_pts[i][1]) <= 0.15:
                dedup[i]["x"] = round(tx, 4)
                dedup[i]["y"] = round(ty, 4)

    for i in range(n):
        if is_pinned[i]:
            continue
        pt = dedup[i]
        p_prev = dedup[(i - 1) % n]
        p_next = dedup[(i + 1) % n]
        dx, dy = p_next["x"] - p_prev["x"], p_next["y"] - p_prev["y"]
        dl = math.hypot(dx, dy)
        if dl > 1e-5 and pt.get("hIn") is not None and pt.get("hOut") is not None:
            ux, uy = dx / dl, dy / dl
            s_in = math.hypot(pt["x"] - p_prev["x"], pt["y"] - p_prev["y"])
            s_out = math.hypot(p_next["x"] - pt["x"], p_next["y"] - pt["y"])
            ci = max(0.05, min(s_in * 0.36, math.hypot(pt["x"] - pt["hIn"]["x"], pt["y"] - pt["hIn"]["y"]) * 0.5 + s_in * 0.15))
            co = max(0.05, min(s_out * 0.36, math.hypot(pt["hOut"]["x"] - pt["x"], pt["hOut"]["y"] - pt["y"]) * 0.5 + s_out * 0.15))
            pt["hIn"] = {"x": round(pt["x"] - ux * ci, 4), "y": round(pt["y"] - uy * ci, 4)}
            pt["hOut"] = {"x": round(pt["x"] + ux * co, 4), "y": round(pt["y"] + uy * co, 4)}
    return dedup


def _refine_cad_anchors(entities: list[dict[str, Any]], span_w: float, span_h: float, snap_axis: bool = True) -> list[dict[str, Any]]:
    """Optimize VTracer splines into manual-CAD-grade straight lines and C1 smooth curves."""
    for ent in entities:
        if ent.get("type") not in ("curve", "polyline"):
            continue
        is_closed = bool(ent.get("isClosed", True))
        if ent.get("points"):
            ent["points"] = _refine_ring_cad(ent["points"], is_closed, span_w, span_h, snap_axis)
        if ent.get("holes"):
            ent["holes"] = [
                _refine_ring_cad(h, True, span_w, span_h, snap_axis)
                for h in ent["holes"]
                if len(h) >= 3
            ]
    return entities


def _sample_ring_px(ring: list[dict[str, Any]], is_closed: bool, off_x: float, off_y: float, sx_px: float, sy_px: float) -> np.ndarray:
    pts: list[list[int]] = []
    n = len(ring)
    seg_n = n if is_closed else n - 1
    for i in range(seg_n):
        a = ring[i]
        b = ring[(i + 1) % n]
        h1, h2 = a.get("hOut"), b.get("hIn")
        if h1 or h2:
            c1x, c1y = (h1["x"], h1["y"]) if h1 else (a["x"], a["y"])
            c2x, c2y = (h2["x"], h2["y"]) if h2 else (b["x"], b["y"])
            for s in range(4):
                t = s / 4.0
                mt = 1.0 - t
                x = mt**3 * a["x"] + 3 * mt**2 * t * c1x + 3 * mt * t**2 * c2x + t**3 * b["x"]
                y = mt**3 * a["y"] + 3 * mt**2 * t * c1y + 3 * mt * t**2 * c2y + t**3 * b["y"]
                pts.append([int(round((x - off_x) * sx_px)), int(round((y - off_y) * sy_px))])
        else:
            pts.append([int(round((a["x"] - off_x) * sx_px)), int(round((a["y"] - off_y) * sy_px))])
    if not is_closed and n > 0:
        pts.append([int(round((ring[-1]["x"] - off_x) * sx_px)), int(round((ring[-1]["y"] - off_y) * sy_px))])
    return np.array(pts, dtype=np.int32)


def _measure_vector_coverage(
    entities: list[dict[str, Any]],
    target_mask: np.ndarray,
    off_x: float,
    off_y: float,
    span_w: float,
    span_h: float,
) -> tuple[float, float, np.ndarray]:
    """Closed-loop render-vs-design recall & uncovered mask across 4x4 grid."""
    th, tw = target_mask.shape[:2]
    max_eval = 560
    sc = min(1.0, max_eval / max(tw, th))
    ew, eh = max(32, int(round(tw * sc))), max(32, int(round(th * sc)))
    tgt_small = cv2.resize(target_mask, (ew, eh), interpolation=cv2.INTER_AREA) > 96
    # Use thin boundary/skeleton pixels of target_mask so interior of wide grooves doesn't dominate distance check
    tgt_u8 = (tgt_small.astype(np.uint8)) * 255
    tgt_edges = cv2.bitwise_xor(tgt_u8, cv2.erode(tgt_u8, np.ones((3, 3), np.uint8))) > 0
    tgt_eval = tgt_edges if int(tgt_edges.sum()) >= 64 else tgt_small
    tot_tgt = int(tgt_eval.sum())
    if tot_tgt == 0:
        return 1.0, 1.0, np.zeros((th, tw), dtype=np.uint8)

    sx_px = ew / max(1e-6, span_w)
    sy_px = eh / max(1e-6, span_h)
    vec_canvas = np.zeros((eh, ew), dtype=np.uint8)
    for ent in entities:
        et = ent.get("type")
        if et == "rect":
            x0 = int(round((ent["x"] - off_x) * sx_px))
            y0 = int(round((ent["y"] - off_y) * sy_px))
            x1 = int(round((ent["x"] + ent["w"] - off_x) * sx_px))
            y1 = int(round((ent["y"] + ent["h"] - off_y) * sy_px))
            cv2.rectangle(vec_canvas, (x0, y0), (x1, y1), 255, 1)
        elif et == "line":
            p1 = (int(round((ent["x1"] - off_x) * sx_px)), int(round((ent["y1"] - off_y) * sy_px)))
            p2 = (int(round((ent["x2"] - off_x) * sx_px)), int(round((ent["y2"] - off_y) * sy_px)))
            cv2.line(vec_canvas, p1, p2, 255, 1)
        elif et == "circle":
            c = (int(round((ent["cx"] - off_x) * sx_px)), int(round((ent["cy"] - off_y) * sy_px)))
            cv2.circle(vec_canvas, c, max(1, int(round(ent["r"] * sx_px))), 255, 1)
        elif ent.get("points"):
            is_c = bool(ent.get("isClosed", True))
            poly = _sample_ring_px(ent["points"], is_c, off_x, off_y, sx_px, sy_px)
            if len(poly) >= 2:
                cv2.polylines(vec_canvas, [poly], is_c, 255, 1)
            for h in ent.get("holes", []):
                hpoly = _sample_ring_px(h, True, off_x, off_y, sx_px, sy_px)
                if len(hpoly) >= 2:
                    cv2.polylines(vec_canvas, [hpoly], True, 255, 1)

    if not vec_canvas.any():
        return 0.0, 0.0, (target_mask > 0).astype(np.uint8) * 255

    dt = cv2.distanceTransform(255 - vec_canvas, cv2.DIST_L2, 3)
    tol_px = max(4.5, min(ew, eh) * 0.012)
    covered = tgt_eval & (dt <= tol_px)
    global_rec = float(covered.sum()) / tot_tgt

    tile_recs: list[float] = []
    min_tile_px = max(14, int(ew * eh * 0.00025))
    for gy in range(4):
        y_a, y_b = (gy * eh) // 4, ((gy + 1) * eh) // 4
        for gx in range(4):
            x_a, x_b = (gx * ew) // 4, ((gx + 1) * ew) // 4
            sub_t = tgt_eval[y_a:y_b, x_a:x_b]
            cnt = int(sub_t.sum())
            if cnt >= min_tile_px:
                tile_recs.append(float(covered[y_a:y_b, x_a:x_b].sum()) / cnt)
    min_tile_rec = min(tile_recs) if tile_recs else global_rec

    uncovered_small = ((tgt_small & (dt > tol_px)).astype(np.uint8)) * 255
    uncovered_small = cv2.dilate(uncovered_small, np.ones((7, 7), np.uint8))
    uncovered_full = cv2.resize(uncovered_small, (tw, th), interpolation=cv2.INTER_NEAREST)
    return global_rec, min_tile_rec, uncovered_full


def _trace_mask_vtracer(
    mask_u8: np.ndarray,
    off_x: float,
    off_y: float,
    sx_mm: float,
    sy_mm: float,
    filter_speckle: int,
    corner_thresh: int,
    length_thresh: float,
    layer_group: str = "Carved Grooves & Panels",
    smooth_mask: bool = True,
) -> list[dict[str, Any]]:
    vh, vw = mask_u8.shape[:2]
    if smooth_mask:
        # Subpixel Gaussian mask smoothing before VTracer eliminates jagged staircase pixel teeth
        sm = cv2.GaussianBlur(mask_u8, (3, 3), 0.72)
        work_m = np.where(sm >= 118, 255, 0).astype(np.uint8)
    else:
        work_m = mask_u8
    bw_img = 255 - work_m
    ok_b, pb = cv2.imencode(".png", cv2.cvtColor(bw_img, cv2.COLOR_GRAY2BGR))
    if not ok_b:
        return []
    svg_s = vtracer.convert_raw_image_to_svg(
        bytes(pb),
        img_format="png",
        colormode="binary",
        mode="spline",
        filter_speckle=filter_speckle,
        corner_threshold=corner_thresh,
        length_threshold=length_thresh,
        splice_threshold=45,
        path_precision=4,
    )
    ents = _parse_svg_paths_to_entities(svg_s, off_x, off_y, sx_mm, sy_mm, layer_group)
    span_w, span_h = vw * sx_mm, vh * sy_mm
    cleaned: list[dict[str, Any]] = []
    for ent in ents:
        pts = ent.get("points") or []
        if not pts:
            continue
        # Never discard compound paths carrying inner motif holes (e.g. door lattices & rose rings)
        if not ent.get("holes") and len(pts) <= 6:
            xs = [p["x"] for p in pts]; ys = [p["y"] for p in pts]
            if (max(xs) - min(xs)) >= span_w * 0.94 and (max(ys) - min(ys)) >= span_h * 0.94:
                continue
        ent["fillColor"] = None
        ent["strokeColor"] = "#22c55e" if layer_group == "Raised Relief Motifs" else "#38bdf8"
        ent["color"] = ent["strokeColor"]
        ent["relief"] = {"enabled": False, "shape": "smooth", "heightMm": 2.0, "radiusMm": 1.2, "mode": "max"}
        cleaned.append(ent)
    return _refine_cad_anchors(cleaned, span_w, span_h, snap_axis=True)


def _extract_wood_relief_vectors(
    sub_bgr: np.ndarray,
    off_x: float,
    off_y: float,
    sx_mm: float,
    sy_mm: float,
    detail: float,
    corners: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Multi-layer clean closed-region relief vectorization + Retinex shadow equalization + convergence loop."""
    vh, vw = sub_bgr.shape[:2]
    span_w, span_h = vw * sx_mm, vh * sy_mm
    den = cv2.fastNlMeansDenoisingColored(sub_bgr, None, 6, 6, 7, 21)
    epf_c = cv2.edgePreservingFilter(den, flags=cv2.NORMCONV_FILTER, sigma_s=22, sigma_r=0.22)
    epf_f = cv2.edgePreservingFilter(den, flags=cv2.NORMCONV_FILTER, sigma_s=12, sigma_r=0.16)

    Lc_raw = cv2.bilateralFilter(cv2.cvtColor(epf_c, cv2.COLOR_BGR2LAB)[:, :, 0], d=7, sigmaColor=24, sigmaSpace=16).astype(np.float32)
    Lf_raw = cv2.bilateralFilter(cv2.cvtColor(epf_f, cv2.COLOR_BGR2LAB)[:, :, 0], d=7, sigmaColor=20, sigmaSpace=14).astype(np.float32)
    # Apply local Multi-Scale Retinex equalization so low-light / shadowed carvings match sunlit SNR cleanly
    Lc = _equalize_illumination_retinex(Lc_raw, strength=0.62)
    Lf = _equalize_illumination_retinex(Lf_raw, strength=0.60)
    Lf_u8 = np.clip(Lf, 0, 255).astype(np.uint8)

    r_f = max(5, int(round(min(vw, vh) * 0.006)) | 1)
    r_m = max(13, int(round(min(vw, vh) * 0.018)) | 1)
    ker_f = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_f, r_f))
    ker_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_m, r_m))

    # Coarse carving gate: balance directional X and Y gradients so vertical panel stiles aren't dropped by overhead light!
    bh_c = cv2.morphologyEx(Lc, cv2.MORPH_BLACKHAT, ker_f) * 0.58 + cv2.morphologyEx(Lc, cv2.MORPH_BLACKHAT, ker_m) * 0.62
    gx_c = np.abs(cv2.Scharr(Lc, cv2.CV_32F, 1, 0) / 16.0)
    gy_c = np.abs(cv2.Scharr(Lc, cv2.CV_32F, 0, 1) / 16.0)
    gx_sc = max(12.0, float(np.percentile(gx_c, 84)))
    gy_sc = max(12.0, float(np.percentile(gy_c, 84)))
    g_c = np.maximum(gx_c * (22.0 / gx_sc), gy_c * (22.0 / gy_sc))

    detail_bias = (0.70 - float(np.clip(detail, 0.15, 1.0))) * 14.0
    c_med = float(np.median(bh_c))
    c_mad = float(np.median(np.abs(bh_c - c_med)))
    gc_med = float(np.median(g_c))
    gc_mad = float(np.median(np.abs(g_c - gc_med)))

    c_zone = (
        (bh_c >= max(float(np.percentile(bh_c, 77)), c_med + 3.8 * max(c_mad, 1.3), 17.5) + detail_bias * 0.35)
        | (g_c >= max(float(np.percentile(g_c, 76)), gc_med + 3.4 * max(gc_mad, 1.3), 15.5) + detail_bias * 0.35)
    ).astype(np.uint8)
    k_dil = max(11, int(round(min(vw, vh) * 0.015)) | 1)
    zone_dil = cv2.dilate(c_zone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dil, k_dil)))
    k_rail = max(25, int(round(min(vw, vh) * 0.032)) | 1)
    zone_closed = cv2.morphologyEx(zone_dil, cv2.MORPH_CLOSE, np.ones((k_rail, 3), np.uint8))
    zone_closed = cv2.morphologyEx(zone_closed, cv2.MORPH_CLOSE, np.ones((3, k_rail), np.uint8))

    # Multi-Layer Decomposition: crisp raised carving volumes (strictly interior petals/motifs) vs closed structural grooves
    bg_surf = cv2.GaussianBlur(Lf, (0, 0), sigmaX=max(14.0, min(vw, vh) * 0.042))
    elev = Lf - bg_surf
    pos_floor = max(11.5, float(np.percentile(elev[zone_dil > 0] if zone_dil.any() else elev, 72)) * 0.78) + detail_bias * 0.30
    raised_raw = np.where((zone_dil > 0) & (elev >= pos_floor), 255, 0).astype(np.uint8)
    raised_raw = cv2.morphologyEx(raised_raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    raised_raw = cv2.morphologyEx(raised_raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Recessed valleys, vertical stiles & horizontal rails normalized per-axis so frame rectangles close completely!
    bh_f = cv2.morphologyEx(Lf, cv2.MORPH_BLACKHAT, ker_f) * 0.65 + cv2.morphologyEx(Lf, cv2.MORPH_BLACKHAT, ker_m) * 0.55
    gx_f = np.abs(cv2.Scharr(Lf, cv2.CV_32F, 1, 0) / 16.0)
    gy_f = np.abs(cv2.Scharr(Lf, cv2.CV_32F, 0, 1) / 16.0)
    gxf_sc = max(10.0, float(np.percentile(gx_f, 84)))
    gyf_sc = max(10.0, float(np.percentile(gy_f, 84)))
    g_f = np.maximum(gx_f * (20.0 / gxf_sc), gy_f * (20.0 / gyf_sc))

    bh_floor_f = max(float(np.percentile(bh_f, 70)), float(np.median(bh_f)) + 2.6 * max(float(np.median(np.abs(bh_f - np.median(bh_f)))), 1.1), 8.2) + detail_bias * 0.25
    g_floor_f = max(float(np.percentile(g_f, 71)), float(np.median(g_f)) + 2.5 * max(float(np.median(np.abs(g_f - np.median(g_f)))), 1.1), 8.0) + detail_bias * 0.25

    comb = np.maximum(np.clip((bh_f - bh_floor_f) * 3.15, 0, 255), np.clip((g_f - g_floor_f) * 2.90, 0, 255) * 0.92)
    comb_masked = np.where((zone_closed > 0) | (bh_c > c_med + 5.2 * max(c_mad, 1.3)), comb, 0.0).astype(np.uint8)

    # Directional rail healing bridges faint overhead-lit vertical panel borders & horizontal frame bars cleanly
    v_rail = cv2.morphologyEx(np.where((gx_f * (20.0 / gxf_sc) >= g_floor_f * 0.85) & (zone_closed > 0), 255, 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((21, 1), np.uint8))
    h_rail = cv2.morphologyEx(np.where((gy_f * (20.0 / gyf_sc) >= g_floor_f * 0.85) & (zone_closed > 0), 255, 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 21), np.uint8))

    med_v = float(np.median(Lf_u8))
    canny_edges = cv2.Canny(Lf_u8, max(14, int(med_v * 0.28)), max(40, int(med_v * 0.78)))
    canny_zone = cv2.bitwise_and(canny_edges, cv2.dilate(zone_closed, np.ones((9, 9), np.uint8)) * 255)
    canny_dil = cv2.dilate(canny_zone, np.ones((2, 2), np.uint8))

    _, groove_mask = cv2.threshold(comb_masked, max(15, int(round(22 + detail_bias * 0.4))), 255, cv2.THRESH_BINARY)
    groove_mask = cv2.bitwise_or(groove_mask, canny_dil)
    groove_mask = cv2.bitwise_or(groove_mask, cv2.bitwise_or(v_rail, h_rail))
    groove_mask = cv2.morphologyEx(groove_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Filter out bare plank speckles/grain on both masks cleanly
    k_act = max(21, int(round(min(vw, vh) * 0.028)) | 1)
    act_map = cv2.GaussianBlur((comb_masked > 42).astype(np.float32), (k_act, k_act), 0)
    min_area = max(28, int(vw * vh * 0.000024))
    max_grain_w = max(16, int(vw * 0.014))

    def filter_mask_components(src_m: np.ndarray, min_a: int) -> np.ndarray:
        nl, lbl, st, _ = cv2.connectedComponentsWithStats(src_m)
        out_m = np.zeros_like(src_m)
        for i in range(1, nl):
            area = int(st[i, cv2.CC_STAT_AREA])
            if area < min_a:
                continue
            bxi, byi, bwi, bhi = int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3])
            cm = lbl[byi : byi + bhi, bxi : bxi + bwi] == i
            sc = comb_masked[byi : byi + bhi, bxi : bxi + bwi][cm]
            pk, mn = int(sc.max()), float(sc.mean())
            act_mn = float(act_map[byi : byi + bhi, bxi : bxi + bwi][cm].mean())
            if act_mn < 0.036 and pk < 94 and area < min_a * 12:
                continue
            if bhi >= bwi * 4.2 and bwi <= max_grain_w and ((mn < 78.0 and pk < 108 and bhi < vh * 0.24) or (act_mn < 0.06 and bhi < vh * 0.42 and mn < 95.0)):
                continue
            if area < min_a * 4 and pk < 76:
                continue
            out_m[lbl == i] = 255
        return out_m

    clean_grooves = filter_mask_components(groove_mask, min_area)
    clean_raised = filter_mask_components(raised_raw, max(44, int(min_area * 1.5)))

    # Keep outer bevel frame borders intact while separating inner thick relief bodies cleanly
    raised_frac = float((clean_raised > 0).sum()) / max(1, vw * vh)
    groove_unique = clean_grooves
    if raised_frac >= 0.015:
        groove_unique = cv2.bitwise_and(clean_grooves, 255 - cv2.erode(clean_raised, np.ones((5, 5), np.uint8)))

    target_mask = cv2.bitwise_or(clean_grooves, clean_raised)
    target_mask = cv2.bitwise_or(target_mask, cv2.bitwise_and(canny_zone, ((comb_masked > 10).astype(np.uint8)) * 255))

    corner_t = int(round(40 + corners * 18))
    speckle_base = max(6, int(round(14 - detail * 7)))

    # Closed-Loop Convergence Iter 1: Clean Layered Regions (Raised Relief Motifs + Carved Grooves & Panels)
    all_ents: list[dict[str, Any]] = []
    if raised_frac >= 0.012:
        all_ents.extend(
            _trace_mask_vtracer(
                clean_raised, off_x, off_y, sx_mm, sy_mm,
                max(8, speckle_base + 2), corner_t, 4.6,
                layer_group="Raised Relief Motifs", smooth_mask=True,
            )
        )
    all_ents.extend(
        _trace_mask_vtracer(
            groove_unique, off_x, off_y, sx_mm, sy_mm,
            speckle_base, corner_t, 4.2,
            layer_group="Carved Grooves & Panels", smooth_mask=True,
        )
    )
    cov, min_tile, uncovered = _measure_vector_coverage(all_ents, target_mask, off_x, off_y, span_w, span_h)
    iters = 1

    # Closed-Loop Iter 2..4: Targeted Residual Recovery ONLY inside uncovered motifs/shadow tiles
    def merge_non_overlapping(base: list[dict[str, Any]], extra: list[dict[str, Any]]):
        if not extra:
            return
        boxes = []
        for e in base:
            ps = e.get("points") or []
            if len(ps) < 2:
                continue
            xs = [p["x"] for p in ps]; ys = [p["y"] for p in ps]
            boxes.append((min(xs), min(ys), max(xs), max(ys), len(ps)))
        for e in extra:
            ps = e.get("points") or []
            if len(ps) < 2:
                continue
            xs = [p["x"] for p in ps]; ys = [p["y"] for p in ps]
            x0e, y0e, x1e, y1e = min(xs), min(ys), max(xs), max(ys)
            dup = False
            for (bx0, by0, bx1, by1, _) in boxes:
                ix = max(0.0, min(x1e, bx1) - max(x0e, bx0))
                iy = max(0.0, min(y1e, by1) - max(y0e, by0))
                inter = ix * iy
                ea = max(1e-4, (x1e - x0e) * (y1e - y0e))
                ba = max(1e-4, (bx1 - bx0) * (by1 - by0))
                if inter / min(ea, ba) > 0.78 and abs(ea - ba) / max(ea, ba) < 0.35:
                    dup = True
                    break
            if not dup:
                base.append(e)
                boxes.append((x0e, y0e, x1e, y1e, len(ps)))

    while (cov < 0.945 or min_tile < 0.84) and iters < 4:
        iters += 1
        if iters == 2:
            dark_boost = cv2.bitwise_and(cv2.dilate(((comb_masked > 10).astype(np.uint8)) * 255, np.ones((3, 3), np.uint8)), uncovered)
            dark_boost = cv2.bitwise_or(dark_boost, cv2.bitwise_and(canny_dil, uncovered))
            dark_boost = cv2.morphologyEx(dark_boost, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            if int((dark_boost > 0).sum()) > 32:
                pass2 = _trace_mask_vtracer(dark_boost, off_x, off_y, sx_mm, sy_mm, max(4, speckle_base - 2), corner_t, 3.6, "Carved Grooves & Panels", True)
                merge_non_overlapping(all_ents, pass2)
        elif iters == 3:
            ai_ents = _extract_ai_model_recovery_vectors(sub_bgr, off_x, off_y, sx_mm, sy_mm, corners)
            if ai_ents:
                ai_filtered = []
                for e in _refine_cad_anchors(ai_ents, span_w, span_h, snap_axis=True):
                    ps = e.get("points") or []
                    if not ps:
                        continue
                    cx_mm = sum(p["x"] for p in ps) / len(ps)
                    cy_mm = sum(p["y"] for p in ps) / len(ps)
                    px_i = int(np.clip(round((cx_mm - off_x) / max(1e-6, sx_mm)), 0, vw - 1))
                    py_i = int(np.clip(round((cy_mm - off_y) / max(1e-6, sy_mm)), 0, vh - 1))
                    xs = [p["x"] for p in ps]; ys = [p["y"] for p in ps]
                    x0i = int(np.clip(math.floor((min(xs) - off_x) / max(1e-6, sx_mm)), 0, vw - 1))
                    y0i = int(np.clip(math.floor((min(ys) - off_y) / max(1e-6, sy_mm)), 0, vh - 1))
                    x1i = int(np.clip(math.ceil((max(xs) - off_x) / max(1e-6, sx_mm)), 0, vw - 1))
                    y1i = int(np.clip(math.ceil((max(ys) - off_y) / max(1e-6, sy_mm)), 0, vh - 1))
                    if uncovered[py_i, px_i] > 0 or int((uncovered[y0i : y1i + 1, x0i : x1i + 1] > 0).sum()) > 18:
                        ai_filtered.append(e)
                merge_non_overlapping(all_ents, ai_filtered)
        elif iters == 4:
            res_bin = cv2.bitwise_and(cv2.dilate(((g_f > g_floor_f * 0.72).astype(np.uint8)) * 255, np.ones((2, 2), np.uint8)), uncovered)
            res_bin = cv2.bitwise_and(res_bin, cv2.dilate(zone_dil, np.ones((11, 11), np.uint8)) * 255)
            if int((res_bin > 0).sum()) > 28:
                pass4 = _trace_mask_vtracer(res_bin, off_x, off_y, sx_mm, sy_mm, 4, corner_t, 3.2, "Carved Grooves & Panels", True)
                merge_non_overlapping(all_ents, pass4)

        cov, min_tile, uncovered = _measure_vector_coverage(all_ents, target_mask, off_x, off_y, span_w, span_h)

    conv = {
        "coverage_pct": round(cov * 100.0, 1),
        "min_tile_pct": round(min_tile * 100.0, 1),
        "iterations": iters,
        "converged": bool(cov >= 0.945 and min_tile >= 0.84),
    }
    return all_ents, conv


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
    if roi is not None or not is_logo:
        if min(sw0, sh0) < 1200:
            sc = min(3.2, 1200.0 / max(1, min(sw0, sh0)))
            if max(sw0, sh0) * sc > 2800:
                sc = 2800.0 / max(1, max(sw0, sh0))
            if sc > 1.02:
                sub = cv2.resize(sub, (int(round(sw0 * sc)), int(round(sh0 * sc))), interpolation=cv2.INTER_LANCZOS4)
                if sc > 1.25:
                    sub = cv2.fastNlMeansDenoisingColored(sub, None, 4, 4, 7, 21)
                    sub = cv2.addWeighted(sub, 1.15, cv2.GaussianBlur(sub, (0, 0), 1.05), -0.15, 0)
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
    convergence: dict[str, Any] = {"coverage_pct": 100.0, "min_tile_pct": 100.0, "iterations": 1, "converged": True}
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
                gray_raw = _equalize_illumination_retinex(cv2.cvtColor(den_plaque, cv2.COLOR_BGR2GRAY).astype(np.float32), strength=0.42).astype(np.uint8)
                dw_s, dh_s = max(32, vw // 4), max(32, vh // 4)
                sm_g = cv2.resize(gray_raw, (dw_s, dh_s), interpolation=cv2.INTER_AREA)
                bg_small = cv2.medianBlur(sm_g, max(15, int(round(min(dw_s, dh_s) * 0.24)) | 1))
                bg_est = cv2.resize(bg_small, (vw, vh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                diff_f = np.clip((bg_est - gray_raw.astype(np.float32)) if dark_design else (gray_raw.astype(np.float32) - bg_est), 0, 255).astype(np.uint8)
                d_th, _ = cv2.threshold(diff_f, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                otsu_raw, _ = cv2.threshold(gray_raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                bw_diff = diff_f >= max(22, int(d_th * 0.76))
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
                    plaque_ents = _refine_cad_anchors(
                        _parse_svg_paths_to_entities(svg_eng, off_x, off_y, sx_mm, sy_mm, "Engraved Wood Design"),
                        span_w,
                        span_h,
                        snap_axis=True,
                    )
                    entities.extend(plaque_ents)
                    cov_p, min_t_p, _ = _measure_vector_coverage(entities, bw_mask, off_x, off_y, span_w, span_h)
                    convergence = {
                        "coverage_pct": round(cov_p * 100.0, 1),
                        "min_tile_pct": round(min_t_p * 100.0, 1),
                        "iterations": 1,
                        "converged": bool(cov_p >= 0.92),
                    }

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
            _refine_cad_anchors(
                _parse_svg_paths_to_entities(svg_bin, off_x, off_y, sx_mm, sy_mm, "BW Vector"),
                span_w,
                span_h,
                snap_axis=False,
            )
        )
    elif roi is not None and not is_engraved_wood_plaque:
        wood_ents, convergence = _extract_wood_relief_vectors(sub, off_x, off_y, sx_mm, sy_mm, detail, corners)
        entities.extend(wood_ents)
        total_rings = sum(1 + len(e.get("holes", [])) for e in entities)
        if total_rings < 12:
            ai_ents = _refine_cad_anchors(
                _extract_ai_model_recovery_vectors(sub, off_x, off_y, sx_mm, sy_mm, corners),
                span_w,
                span_h,
                snap_axis=True,
            )
            entities.extend(ai_ents)
        if sum(1 + len(e.get("holes", [])) for e in entities) < 10:
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
                    if pts and not ent.get("holes") and len(pts) <= 6:
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
    # Edge-preserving color vectorization (clean smooth flat zones instead of noisy JPEG artifacts)
    elif not is_engraved_wood_plaque:
        den_c = cv2.fastNlMeansDenoisingColored(sub, None, 6, 6, 7, 21)
        den_c = cv2.bilateralFilter(den_c, d=7, sigmaColor=22, sigmaSpace=14)
        den_c = cv2.bilateralFilter(den_c, d=5, sigmaColor=16, sigmaSpace=10)
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
            if ignore_bg and ent["points"] and not ent.get("holes") and len(ent["points"]) <= 6:
                xs = [p["x"] for p in ent["points"]]
                ys = [p["y"] for p in ent["points"]]
                if (max(xs) - min(xs)) >= span_w * 0.94 and (max(ys) - min(ys)) >= span_h * 0.94:
                    continue
            entities.append(ent)
        entities = _refine_cad_anchors(entities, span_w, span_h, snap_axis=not is_logo)
        if trace_mode in ("auto", "wood") and sum(1 + len(e.get("holes", [])) for e in entities) < 12:
            rel_ents, convergence = _extract_wood_relief_vectors(sub, off_x, off_y, sx_mm, sy_mm, detail, corners)
            entities.extend(rel_ents)
            if sum(1 + len(e.get("holes", [])) for e in entities) < 12:
                entities.extend(_refine_cad_anchors(
                    _extract_ai_model_recovery_vectors(sub, off_x, off_y, sx_mm, sy_mm, corners),
                    span_w,
                    span_h,
                    snap_axis=True,
                ))

    return {
        "ok": True,
        "engine": "opencv-vtracer-vision",
        "entities": entities[:1100],
        "convergence": convergence,
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

    # Pure Poisson integrated Frankot-Chellappa surface (NO unsigned gradient crater-rim spikes!)
    vals = z_sfs[fg]
    lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    z_norm = np.clip((z_sfs - lo) / max(1e-4, hi - lo), 0.0, 1.0)
    z_smooth = cv2.bilateralFilter(z_norm.astype(np.float32), d=5, sigmaColor=0.14, sigmaSpace=5.0)
    return z_smooth * fg.astype(np.float32)


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

    # Low-res NLM denoise + Lanczos + crisp unsharp restoration for sharp 4K PBR texture
    ch, cw = crop_bgra.shape[:2]
    if max(cw, ch) < 1024 and not is_logo:
        sc = min(2.4, 1280.0 / max(cw, ch))
        nw, nh = int(round(cw * sc)), int(round(ch * sc))
        rgb_up = cv2.resize(crop_bgra[:, :, :3], (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        rgb_up = cv2.fastNlMeansDenoisingColored(rgb_up, None, 4, 4, 7, 21)
        rgb_up = cv2.addWeighted(rgb_up, 1.16, cv2.GaussianBlur(rgb_up, (0, 0), 1.0), -0.16, 0)
        a_up = cv2.resize(crop_bgra[:, :, 3], (nw, nh), interpolation=cv2.INTER_LINEAR)
        crop_bgra = np.dstack([rgb_up, a_up])
        ch, cw = crop_bgra.shape[:2]

    # Run cached Depth-Anything-V2-Small-hf on M1 GPU only for non-planar 3D objects/scenes
    depth_norm = None
    if not is_planar:
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
        depth_norm = _solve_planar_relief_heightmap(crop_bgra[:, :, :3])
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

    # Compute crisp up-to-1280px 16-bit depth transport buffer matching user topology preset (4K/2K/Flagship)
    req_res = int(req.get("resolution", 384))
    max_t = min(1280, max(640, req_res * 2))
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
