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
import struct
import subprocess
import sys
import threading
import traceback
from typing import Any

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
import vtracer

_RUST_CORE_BIN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "jdpaint_rust_core",
    "target",
    "release",
    "jdpaint_rust_core",
)
_RUST_PROC: subprocess.Popen | None = None
_RUST_LOCK = threading.Lock()


def _read_exact(pipe, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = pipe.read(n - len(buf))
        if not chunk:
            raise EOFError("Rust worker closed pipe unexpectedly")
        buf.extend(chunk)
    return bytes(buf)


def _rust_rpc(payload: bytes, expect_magic: bytes, poisson_bytes: int = 0) -> bytes | None:
    """Zero-fork persistent binary pipe RPC to native multi-core Rust core."""
    global _RUST_PROC
    if not os.path.isfile(_RUST_CORE_BIN):
        return None
    with _RUST_LOCK:
        for attempt in range(2):
            if _RUST_PROC is None or _RUST_PROC.poll() is not None:
                _RUST_PROC = subprocess.Popen(
                    [_RUST_CORE_BIN, "serve"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,
                )
            try:
                assert _RUST_PROC.stdin is not None and _RUST_PROC.stdout is not None
                _RUST_PROC.stdin.write(payload)
                _RUST_PROC.stdin.flush()
                if expect_magic == b"JDR2":
                    hdr = _read_exact(_RUST_PROC.stdout, 8)
                    if hdr[:4] != b"JDR2":
                        raise IOError("Bad JDR2 magic")
                    body_len = struct.unpack_from("<I", hdr, 4)[0]
                    return _read_exact(_RUST_PROC.stdout, body_len)
                hdr = _read_exact(_RUST_PROC.stdout, 4)
                if hdr != b"JDP2":
                    raise IOError("Bad JDP2 magic")
                return _read_exact(_RUST_PROC.stdout, poisson_bytes)
            except Exception:
                try:
                    _RUST_PROC.kill()
                except Exception:
                    pass
                _RUST_PROC = None
                if attempt == 1:
                    return None
    return None

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
        # Door bottom shadow crease across [68%..ry1+py] catches wood floor continuity, ignoring bright white footer margins
        y_lo = max(ry0 + int((ry1 - ry0) * 0.68), ry1 - int(h * 0.18))
        y_hi = min(h, ry1 + py)
        row_dark = (L[y_lo:y_hi, rx0 : rx1 + 1] < 24.0).mean(axis=1)
        row_white = ((L[y_lo:y_hi, rx0 : rx1 + 1] > 188.0) & (dc[y_lo:y_hi, rx0 : rx1 + 1] < 16.0)).mean(axis=1)
        row_edge = gy[y_lo:y_hi]
        score = row_edge + row_dark * 55.0 - row_white * 140.0
        ry1 = y_lo + int(np.argmax(score))

    # Trim bottom/top pure white catalog margin strips (e.g. Doors Fair footer) cleanly before crop
    sub_w_check = ((L[ry0 : ry1 + 1, rx0 : rx1 + 1] > 192.0) & (dc[ry0 : ry1 + 1, rx0 : rx1 + 1] < 18.0)).mean(axis=1)
    while ry1 - ry0 > int(h * 0.35) and len(sub_w_check) > 8 and sub_w_check[-1] > 0.58:
        ry1 -= 1
        sub_w_check = sub_w_check[:-1]

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
    """Suppress neutral bright overlays, chromatic stamps, and adjacent dark footer typography."""
    vh_w, vw_w = sub_bgr.shape[:2]
    b, g, r = cv2.split(sub_bgr)
    rgb_sat = np.maximum(np.maximum(r, g), b).astype(np.int16) - np.minimum(np.minimum(r, g), b).astype(np.int16)
    luma_raw = 0.2126 * r + 0.7152 * g + 0.0722 * b

    is_bright_wm = (L_sub > l_med + 14.0) & (rgb_sat < 22) & (dc_sub > 16.0)
    is_neutral_ink = (l_med > 82.0) & (luma_raw < 74.0) & (rgb_sat < 14) & ((L_sub < l_med * 0.72) | (dc_sub > 14.0))
    # Pure crimson/red stamps sit noticeably above natural mahogany/teak grain highlights
    is_red_accent = (r.astype(np.int16) - g.astype(np.int16) > 58) & (r.astype(np.int16) - b.astype(np.int16) > 36) & (dc_sub > 20.0)
    is_blue_accent = (b.astype(np.int16) - r.astype(np.int16) > 28) & (dc_sub > 18.0)

    cand = is_bright_wm | is_neutral_ink | is_red_accent | is_blue_accent
    tot_px = vh_w * vw_w
    wm_u8 = np.zeros((vh_w, vw_w), dtype=np.uint8)
    if 0 < int(cand.sum()) < int(tot_px * 0.055):
        n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8) * 255)
        max_comp = int(tot_px * 0.015)
        for ci in range(1, n_lbl):
            if 4 <= int(stats[ci, cv2.CC_STAT_AREA]) <= max_comp:
                wm_u8[lbls == ci] = 255

    # Catch dark/grey sub-tagline typography & TM stamps directly inside a detected watermark banner band
    if int(wm_u8.sum()) > 0:
        band_ky = max(21, int(round(vh_w * 0.045)) | 1)
        band_kx = max(41, int(round(vw_w * 0.14)) | 1)
        wm_band = cv2.dilate(wm_u8, np.ones((band_ky, band_kx), np.uint8))
        # Restrict small neutral tagline pickup strictly to outer header/footer banner zones (<14% top or >86% bottom)
        outer_margin = np.zeros((vh_w, vw_w), dtype=bool)
        outer_margin[: max(4, int(vh_w * 0.14)), :] = True
        outer_margin[min(vh_w - 4, int(vh_w * 0.86)) :, :] = True
        sub_ink = (wm_band > 0) & outer_margin & (luma_raw < 92.0) & (rgb_sat < 28) & ((dc_sub > 9.0) | (L_sub < l_med * 0.72))
        if int(sub_ink.sum()) > 0:
            n_s, lbl_s, st_s, _ = cv2.connectedComponentsWithStats(sub_ink.astype(np.uint8) * 255)
            max_sub_h = max(14, int(round(vh_w * 0.038)))
            for si in range(1, n_s):
                sw_i, sh_i, sa_i = int(st_s[si, 2]), int(st_s[si, 3]), int(st_s[si, 4])
                if 2 <= sa_i <= int(tot_px * 0.008) and sh_i <= max_sub_h and sw_i < vw_w * 0.45:
                    wm_u8[lbl_s == si] = 255

    if int(wm_u8.sum()) > 0:
        wm_u8 = cv2.dilate(wm_u8, np.ones((7, 7), np.uint8), iterations=1)
        cleaned = cv2.inpaint(sub_bgr, wm_u8, 7, cv2.INPAINT_TELEA)
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


def _run_rust_poisson(
    px: np.ndarray,
    qy: np.ndarray,
    z_macro: np.ndarray,
    grain_gate: np.ndarray,
    lam: float = 0.0022,
    iters: int = 16,
    grad_knee: float = 0.036,
    grad_scale: float = 1.15,
) -> np.ndarray | None:
    """Solve logarithmic gradient-compressed Screened Poisson bas-relief in native multi-core Rust."""
    h, w = z_macro.shape[:2]
    header = struct.pack("<4sIIfIff", b"JDP2", w, h, float(lam), int(iters), float(grad_knee), float(grad_scale))
    payload = (
        header
        + np.ascontiguousarray(px, dtype="<f4").tobytes()
        + np.ascontiguousarray(qy, dtype="<f4").tobytes()
        + np.ascontiguousarray(z_macro, dtype="<f4").tobytes()
        + np.ascontiguousarray(grain_gate, dtype="<f4").tobytes()
    )
    out = _rust_rpc(payload, b"JDP2", poisson_bytes=w * h * 4)
    if out is None or len(out) != w * h * 4:
        return None
    return np.frombuffer(out, dtype="<f4").reshape((h, w)).copy()


def _solve_planar_relief_heightmap(sub_bgr: np.ndarray, wm_mask: np.ndarray | None = None) -> np.ndarray:
    """Compute a volumetric de-tilted bas-relief heightmap fusing AI Depth-Anything-V2, SFS & Rust Screened Poisson."""
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
    k_bg = max(31, int(round(m_dim * 0.12)) | 1)
    bg_l = cv2.GaussianBlur(Lc, (k_bg, k_bg), 0)
    elev = Lc - bg_l
    delight = np.clip(elev * 1.55 + 0.5, 0.0, 1.0)

    # De-tilt broad FFT pillow bow on planar panels
    sfs_bow = cv2.GaussianBlur(sfs, (0, 0), sigmaX=max(18.0, m_dim * 0.11))
    sfs_flat = np.clip(0.5 + (sfs - sfs_bow) * 1.45, 0.0, 1.0)

    # Gate out bare flat wood grain while keeping genuine carved motifs solid
    grad_e = np.hypot(cv2.Scharr(delight, cv2.CV_32F, 1, 0), cv2.Scharr(delight, cv2.CV_32F, 0, 1)) / 16.0
    act_e = cv2.GaussianBlur(np.maximum(grad_e, np.abs(elev) * 2.1), (0, 0), sigmaX=max(6.0, m_dim * 0.014))
    act_floor = max(0.032, float(np.percentile(act_e, 46)))
    grain_gate = np.clip((act_e - act_floor * 0.70) / max(1e-4, act_floor * 1.40), 0.0, 1.0).astype(np.float32)

    # De-spike 1-3px glossy specular dots so thin bright ribs don't become needle stalagmites inside dark motifs
    Lc_u8 = np.clip(Lc * 255.0, 0, 255).astype(np.uint8)
    L_med5 = cv2.medianBlur(Lc_u8, 5).astype(np.float32) / 255.0
    Lc_despiked = np.minimum(Lc, L_med5 + 0.030).astype(np.float32)
    Ld_u8 = np.clip(Lc_despiked * 255.0, 0, 255).astype(np.uint8)

    # Smooth architectural foundation bounded so stained/dark floral cavities stay a recessed floor, never a black crater
    z_macro_raw = (ai_flat * 0.42 + sfs_flat * 0.58) if ai_flat is not None else sfs_flat.copy()
    z_macro = np.clip(cv2.GaussianBlur(z_macro_raw.astype(np.float32), (0, 0), sigmaX=max(2.4, m_dim * 0.0045)), 0.28, 0.86)

    # Multi-Scale Morphological White TopHat makes EVERY carved petal/leaf rise proudly (+Z) above its cavity floor
    r_p1 = max(5, int(round(m_dim * 0.012)) | 1)
    r_p2 = max(11, int(round(m_dim * 0.028)) | 1)
    r_p3 = max(21, int(round(m_dim * 0.052)) | 1)
    wh_1 = cv2.morphologyEx(Ld_u8, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_p1, r_p1))).astype(np.float32) / 255.0
    wh_2 = cv2.morphologyEx(Ld_u8, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_p2, r_p2))).astype(np.float32) / 255.0
    wh_3 = cv2.morphologyEx(Ld_u8, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_p3, r_p3))).astype(np.float32) / 255.0
    petal_domes = np.clip(wh_1 * 0.45 + wh_2 * 0.55 + wh_3 * 0.35, 0.0, 0.62)

    r_bh = max(7, int(round(m_dim * 0.014)) | 1)
    ker_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_bh, r_bh))
    bh_r = cv2.morphologyEx(Ld_u8, cv2.MORPH_BLACKHAT, ker_bh).astype(np.float32) / 255.0
    val_carve = np.clip(bh_r * 0.72, 0.0, 0.28)
    sculpt_raw = (petal_domes * 0.90 - val_carve * 0.80) * grain_gate
    if wm_mask is not None and wm_mask.shape[:2] == sculpt_raw.shape[:2] and wm_mask.any():
        wm_d = cv2.dilate((wm_mask > 0).astype(np.uint8) * 255, np.ones((7, 7), np.uint8)) > 0
        sculpt_raw[wm_d] = 0.0
    sculpt_field = cv2.GaussianBlur(
        sculpt_raw.astype(np.float32),
        (0, 0),
        sigmaX=max(1.2, m_dim * 0.0022),
    )

    guide_surf = np.clip(z_macro * 0.48 + (0.36 + sculpt_field * 0.95) * 0.52, 0.02, 0.99).astype(np.float32)
    px = cv2.Scharr(guide_surf, cv2.CV_32F, 1, 0) / 32.0
    qy = cv2.Scharr(guide_surf, cv2.CV_32F, 0, 1) / 32.0

    rel = _run_rust_poisson(
        px,
        qy,
        guide_surf,
        np.maximum(grain_gate, 0.14).astype(np.float32),
        lam=0.0055,
        iters=20,
        grad_knee=0.032,
        grad_scale=1.10,
    )
    if rel is None:
        rel = guide_surf

    # Crisp edge-preserving bilateral finish retains sharp petal ridges without Gaussian mush
    rel = cv2.bilateralFilter(rel.astype(np.float32), d=5, sigmaColor=0.070, sigmaSpace=2.6)
    lo, hi = float(np.percentile(rel, 1.5)), float(np.percentile(rel, 98.5))
    return np.clip((rel - lo) / max(1e-4, hi - lo), 0.04, 0.96).astype(np.float32)


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


def _trace_field_rust(
    field_u8: np.ndarray,
    off_x: float,
    off_y: float,
    sx_mm: float,
    sy_mm: float,
    detail: float = 0.72,
    corners: float = 0.65,
    min_area_px: float = 20.0,
    straight_tol_px: float = 1.15,
    layer_group: str = "Carved Grooves & Panels",
    smooth_sigma: float = 0.68,
) -> list[dict[str, Any]]:
    """Run native multi-core Rust subpixel Marching Squares + Schneider Least-Squares Cubic Bézier fitter."""
    if not os.path.isfile(_RUST_CORE_BIN):
        return []
    vh, vw = field_u8.shape[:2]
    if smooth_sigma > 0.05:
        k_sm = 5 if smooth_sigma >= 0.80 else 3
        field_in = cv2.GaussianBlur(field_u8, (k_sm, k_sm), smooth_sigma)
    else:
        field_in = np.ascontiguousarray(field_u8, dtype=np.uint8)

    header = struct.pack(
        "<4sIIfffffffff",
        b"JDV2",
        vw,
        vh,
        127.5,
        float(np.clip(detail, 0.1, 1.0)),
        float(np.clip(corners, 0.0, 1.0)),
        float(max(4.0, min_area_px)),
        float(max(0.35, straight_tol_px)),
        float(off_x + 0.5 * sx_mm),
        float(off_y + 0.5 * sy_mm),
        float(sx_mm),
        float(sy_mm),
    )
    body = _rust_rpc(
        header + np.ascontiguousarray(field_in, dtype=np.uint8).tobytes(),
        b"JDR2",
    )
    if body is None or len(body) < 4:
        return []

    buf = memoryview(body)
    ring_count = struct.unpack_from("<I", buf, 0)[0]
    pos = 4
    node_dt = np.dtype("<6f4,u4")
    outer_pts_by_idx: dict[int, list[dict[str, Any]]] = {}
    holes_by_parent: dict[int, list[list[dict[str, Any]]]] = {}
    outer_order: list[int] = []

    stroke = (
        "#22c55e"
        if layer_group == "Raised Relief Motifs"
        else ("#f59e0b" if layer_group == "Engraved Wood Design" else "#38bdf8")
    )

    for r_i in range(ring_count):
        if pos + 9 > len(buf):
            break
        is_hole, parent_idx, pt_count = struct.unpack_from("<BiI", buf, pos)
        pos += 9
        byte_len = pt_count * 28
        if pos + byte_len > len(buf):
            break
        arr = np.frombuffer(buf[pos : pos + byte_len], dtype=node_dt)
        pos += byte_len
        coords = arr["f0"].round(4).tolist()
        flags_l = arr["f1"].tolist()
        pts: list[dict[str, Any]] = [
            {
                "x": c[0],
                "y": c[1],
                "corner": bool(fl & 1),
                "hIn": {"x": c[2], "y": c[3]} if (fl & 2) else None,
                "hOut": {"x": c[4], "y": c[5]} if (fl & 4) else None,
            }
            for c, fl in zip(coords, flags_l)
        ]
        if len(pts) < 3:
            continue
        if is_hole and parent_idx >= 0:
            holes_by_parent.setdefault(parent_idx, []).append(pts)
        else:
            outer_pts_by_idx[r_i] = pts
            outer_order.append(r_i)

    span_w, span_h = vw * sx_mm, vh * sy_mm
    entities: list[dict[str, Any]] = []
    for r_i in outer_order:
        pts = outer_pts_by_idx[r_i]
        holes = holes_by_parent.get(r_i, [])
        if not holes and len(pts) <= 6:
            xs = [p["x"] for p in pts]
            ys = [p["y"] for p in pts]
            if (max(xs) - min(xs)) >= span_w * 0.94 and (max(ys) - min(ys)) >= span_h * 0.94:
                continue
        entities.append({
            "type": "curve",
            "points": pts,
            "holes": holes,
            "isClosed": True,
            "color": stroke,
            "fillColor": None,
            "strokeColor": stroke,
            "width": 1.25,
            "layerGroup": layer_group,
            "relief": {"enabled": False, "shape": "smooth", "heightMm": 2.0, "radiusMm": 1.2, "mode": "max"},
        })
    return entities


def _extract_straight_cad_lines(
    gray_eq_u8: np.ndarray,
    zone_closed: np.ndarray,
    wm_sup: np.ndarray | None,
    off_x: float,
    off_y: float,
    sx_mm: float,
    sy_mm: float,
    detail: float,
    ridge_gate: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Detect clean straight architectural rails/stiles/panels via OpenCV LSD + centerline double-wall fusion."""
    vh, vw = gray_eq_u8.shape[:2]
    m_dim = min(vw, vh)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    raw = lsd.detect(gray_eq_u8)[0]
    arch_mask = np.zeros((vh, vw), dtype=np.uint8)
    if raw is None or len(raw) == 0:
        return [], arch_mask

    min_len = max(28.0, m_dim * (0.062 - float(np.clip(detail, 0.1, 1.0)) * 0.015))
    zone_d = cv2.dilate(zone_closed, np.ones((7, 7), np.uint8)) > 0
    ridge_d = cv2.dilate(ridge_gate, np.ones((5, 5), np.uint8)) > 0 if ridge_gate is not None else zone_d
    near_margin = max(12.0, vw * 0.105)

    kept: list[list[float]] = []
    for seg in raw.reshape(-1, 4):
        x1, y1, x2, y2 = float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3])
        dx, dy = x2 - x1, y2 - y1
        ln = math.hypot(dx, dy)
        if ln < min_len:
            continue
        ang = abs(math.degrees(math.atan2(dy, dx))) % 180.0
        is_h = ang <= 2.2 or ang >= 177.8
        is_v = 87.8 <= ang <= 92.2
        is_miter = (36.0 <= ang <= 54.0) or (126.0 <= ang <= 144.0)
        if not (is_h or is_v or is_miter or ln >= min_len * 2.0):
            continue

        # Verify segment genuinely sits on a real carved edge/groove (never bare flat panel wood grain)
        xs_s = np.clip(np.round(np.linspace(x1, x2, 16)).astype(np.int32), 0, vw - 1)
        ys_s = np.clip(np.round(np.linspace(y1, y2, 16)).astype(np.int32), 0, vh - 1)
        if float(zone_d[ys_s, xs_s].mean()) < 0.70:
            continue
        ridge_cov = float(ridge_d[ys_s, xs_s].mean())
        if ridge_cov < (0.56 if (is_v and ln < vh * 0.16) else 0.42):
            continue
        if wm_sup is not None and float((wm_sup[ys_s, xs_s] > 0).mean()) > 0.25:
            continue
        if is_v and ln < vh * 0.42 and (max(x1, x2) < near_margin or min(x1, x2) > vw - near_margin):
            continue
        if is_h and min(y1, y2) > vh * 0.905 and ln < vw * 0.28:
            continue

        if is_h:
            ym = 0.5 * (y1 + y2)
            y1 = y2 = ym
        elif is_v:
            xm = 0.5 * (x1 + x2)
            x1 = x2 = xm
        kept.append([x1, y1, x2, y2, ln])

    if not kept:
        return [], arch_mask

    # Sort longest first and fuse parallel double-wall LSD detections + supported collinear gaps
    kept.sort(key=lambda s: s[4], reverse=True)
    perp_merge_px = max(4.6, m_dim * 0.0048)
    gap_bridge_px = max(20.0, m_dim * 0.024)
    merged: list[list[float]] = []
    for x1, y1, x2, y2, ln in kept:
        dx, dy = x2 - x1, y2 - y1
        ux, uy = dx / ln, dy / ln
        placed = False
        for m in merged:
            mx1, my1, mx2, my2, mln = m
            mdx, mdy = mx2 - mx1, my2 - my1
            mux, muy = mdx / mln, mdy / mln
            dot = ux * mux + uy * muy
            if abs(dot) < 0.996:
                continue
            midx, midy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            perp = abs((midx - mx1) * muy - (midy - my1) * mux)
            if perp > perp_merge_px:
                continue
            p1 = (x1 - mx1) * mux + (y1 - my1) * muy
            p2 = (x2 - mx1) * mux + (y2 - my1) * muy
            lo_p, hi_p = min(p1, p2), max(p1, p2)
            ov = min(hi_p, mln) - max(lo_p, 0.0)
            gap = max(0.0, max(lo_p - mln, -hi_p))
            if gap <= gap_bridge_px and (ov >= -gap_bridge_px):
                # Require image carving support along bridged gap so separate columns never fuse across empty panels
                if gap > 4.0:
                    g_a = mln if lo_p > mln else hi_p
                    g_b = lo_p if lo_p > mln else 0.0
                    gx_s = np.clip(np.round(np.linspace(mx1 + mux * g_a, mx1 + mux * g_b, 8)).astype(np.int32), 0, vw - 1)
                    gy_s = np.clip(np.round(np.linspace(my1 + muy * g_a, my1 + muy * g_b, 8)).astype(np.int32), 0, vh - 1)
                    if float(zone_d[gy_s, gx_s].mean()) < 0.35:
                        continue
                if ov > 0.45 * min(ln, mln) and perp >= 1.0:
                    w_n = ln / (mln + ln)
                    nx = -muy
                    ny = mux
                    signed_perp = ((midx - mx1) * nx + (midy - my1) * ny)
                    mx1 += nx * signed_perp * w_n
                    my1 += ny * signed_perp * w_n
                n_lo = min(0.0, lo_p)
                n_hi = max(mln, hi_p)
                m[0] = mx1 + mux * n_lo
                m[1] = my1 + muy * n_lo
                m[2] = mx1 + mux * n_hi
                m[3] = my1 + muy * n_hi
                m[4] = n_hi - n_lo
                placed = True
                break
        if not placed:
            merged.append([x1, y1, x2, y2, ln])

    # Snap crisp 90-degree orthogonal corner intersections so rectangular panel borders close cleanly without gaps
    corner_snap = max(14.0, m_dim * 0.018)
    h_idxs = [i for i, m in enumerate(merged) if abs(m[3] - m[1]) <= 0.04 * m[4]]
    v_idxs = [i for i, m in enumerate(merged) if abs(m[2] - m[0]) <= 0.04 * m[4]]
    for hi in h_idxs:
        hx1, hy1, hx2, hy2, _ = merged[hi]
        hy = 0.5 * (hy1 + hy2)
        hlo, hhi = min(hx1, hx2), max(hx1, hx2)
        for vi in v_idxs:
            vx1, vy1, vx2, vy2, _ = merged[vi]
            vx = 0.5 * (vx1 + vx2)
            vlo, vhi = min(vy1, vy2), max(vy1, vy2)
            near_h = (abs(vx - hx1) <= corner_snap) or (abs(vx - hx2) <= corner_snap) or (hlo - 2.0 <= vx <= hhi + 2.0)
            near_v = (abs(hy - vy1) <= corner_snap) or (abs(hy - vy2) <= corner_snap) or (vlo - 2.0 <= hy <= vhi + 2.0)
            if not (near_h and near_v):
                continue
            if abs(vx - merged[hi][0]) <= corner_snap:
                merged[hi][0] = vx
            elif abs(vx - merged[hi][2]) <= corner_snap:
                merged[hi][2] = vx
            if abs(hy - merged[vi][1]) <= corner_snap:
                merged[vi][1] = hy
            elif abs(hy - merged[vi][3]) <= corner_snap:
                merged[vi][3] = hy
        merged[hi][4] = math.hypot(merged[hi][2] - merged[hi][0], merged[hi][3] - merged[hi][1])
    for vi in v_idxs:
        merged[vi][4] = math.hypot(merged[vi][2] - merged[vi][0], merged[vi][3] - merged[vi][1])

    # Prune floating unanchored vertical wood-grain ticks while keeping ALL real frame rails, corner boxes, & pilaster flutes
    h_segs = [merged[i] for i in h_idxs if merged[i][4] >= min_len]
    v_set = set(v_idxs)
    anchor_tol = max(16.0, m_dim * 0.022)
    flute_dx = max(14.0, m_dim * 0.024)
    final_segs: list[list[float]] = []
    for idx_m, m in enumerate(merged):
        x1, y1, x2, y2, ln = m
        if ln < min_len:
            continue
        if idx_m in v_set and ln < vh * 0.15:
            vx = 0.5 * (x1 + x2)
            vy_lo, vy_hi = min(y1, y2), max(y1, y2)
            top_anc = any(
                (min(h[0], h[2]) - anchor_tol <= vx <= max(h[0], h[2]) + anchor_tol)
                and (abs(0.5 * (h[1] + h[3]) - vy_lo) <= anchor_tol)
                for h in h_segs
            )
            bot_anc = any(
                (min(h[0], h[2]) - anchor_tol <= vx <= max(h[0], h[2]) + anchor_tol)
                and (abs(0.5 * (h[1] + h[3]) - vy_hi) <= anchor_tol)
                for h in h_segs
            )
            flute_peers = 0
            for j in v_idxs:
                if j == idx_m or merged[j][4] < min_len:
                    continue
                ox = 0.5 * (merged[j][0] + merged[j][2])
                oy_lo, oy_hi = min(merged[j][1], merged[j][3]), max(merged[j][1], merged[j][3])
                ov_y = min(vy_hi, oy_hi) - max(vy_lo, oy_lo)
                if abs(ox - vx) <= flute_dx and ov_y >= 0.72 * max(ln, merged[j][4]):
                    flute_peers += 1
            xs_c = np.clip(np.round(np.linspace(x1, x2, 16)).astype(np.int32), 0, vw - 1)
            ys_c = np.clip(np.round(np.linspace(y1, y2, 16)).astype(np.int32), 0, vh - 1)
            r_cov = float(ridge_d[ys_c, xs_c].mean())
            if not ((top_anc and bot_anc) or (flute_peers >= 2 and (top_anc or bot_anc or r_cov >= 0.60)) or (ln >= vh * 0.115 and r_cov >= 0.78)):
                continue
        final_segs.append(m)

    entities: list[dict[str, Any]] = []
    draw_th = max(3, int(round(m_dim * 0.0042)))
    for x1, y1, x2, y2, ln in final_segs:
        p1i = (int(round(x1)), int(round(y1)))
        p2i = (int(round(x2)), int(round(y2)))
        cv2.line(arch_mask, p1i, p2i, 255, draw_th)
        entities.append({
            "type": "line",
            "x1": round(off_x + (x1 + 0.5) * sx_mm, 4),
            "y1": round(off_y + (y1 + 0.5) * sy_mm, 4),
            "x2": round(off_x + (x2 + 0.5) * sx_mm, 4),
            "y2": round(off_y + (y2 + 0.5) * sy_mm, 4),
            "isClosed": False,
            "color": "#22d3ee",
            "fillColor": None,
            "strokeColor": "#22d3ee",
            "width": 1.35,
            "layerGroup": "Architectural Frame",
            "relief": {"enabled": False, "shape": "flat", "heightMm": 0.4, "radiusMm": 1.0, "mode": "max"},
        })
    return entities, arch_mask


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
    detail_est = float(np.clip((20.0 - filter_speckle) / 14.0, 0.4, 0.95))
    corners_est = float(np.clip((corner_thresh - 32.0) / 32.0, 0.2, 0.95))
    rust_ents = _trace_field_rust(
        mask_u8,
        off_x,
        off_y,
        sx_mm,
        sy_mm,
        detail=detail_est,
        corners=corners_est,
        min_area_px=max(16.0, filter_speckle * 2.2),
        straight_tol_px=1.15,
        layer_group=layer_group,
        smooth_sigma=0.72 if smooth_mask else 0.0,
    )
    if rust_ents:
        return rust_ents

    if smooth_mask:
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


def _fill_tiny_holes(mask_u8: np.ndarray, min_hole_px: int) -> np.ndarray:
    """Close sub-millimeter enclosed pinholes inside relief blobs before VTracer traces them."""
    if min_hole_px <= 4 or not mask_u8.any():
        return mask_u8
    inv = 255 - mask_u8
    nl, lbl, st, _ = cv2.connectedComponentsWithStats(inv)
    vh, vw = mask_u8.shape[:2]
    out = mask_u8.copy()
    for i in range(1, nl):
        bx, by, bw_i, bh_i, area = (int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3]), int(st[i, 4]))
        if bx == 0 or by == 0 or bx + bw_i >= vw or by + bh_i >= vh:
            continue
        if area < min_hole_px:
            out[lbl == i] = 255
    return out


def _extract_wood_relief_vectors(
    sub_bgr: np.ndarray,
    off_x: float,
    off_y: float,
    sx_mm: float,
    sy_mm: float,
    detail: float,
    corners: float,
    wm_mask: np.ndarray | None = None,
    noise_px: float = 10.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Multi-layer clean Architectural CAD Lines + crevice-split relief petals + Retinex + Rust Bézier convergence."""
    vh, vw = sub_bgr.shape[:2]
    m_dim = min(vw, vh)
    span_w, span_h = vw * sx_mm, vh * sy_mm
    # Fast resolution-bounded NLM denoise (avoids slow 10s stalls on 4K inputs while preserving clean grain rejection)
    if max(vw, vh) > 1650:
        sc_d = 1650.0 / max(vw, vh)
        dw, dh = max(64, int(round(vw * sc_d))), max(64, int(round(vh * sc_d)))
        den_s = cv2.fastNlMeansDenoisingColored(cv2.resize(sub_bgr, (dw, dh), interpolation=cv2.INTER_AREA), None, 6, 6, 7, 21)
        den = cv2.resize(den_s, (vw, vh), interpolation=cv2.INTER_CUBIC)
    else:
        den = cv2.fastNlMeansDenoisingColored(sub_bgr, None, 6, 6, 7, 21)
    epf_c = cv2.edgePreservingFilter(den, flags=cv2.NORMCONV_FILTER, sigma_s=20, sigma_r=0.22)
    epf_f = cv2.edgePreservingFilter(den, flags=cv2.NORMCONV_FILTER, sigma_s=11, sigma_r=0.16)

    Lc_raw = cv2.bilateralFilter(cv2.cvtColor(epf_c, cv2.COLOR_BGR2LAB)[:, :, 0], d=7, sigmaColor=24, sigmaSpace=16).astype(np.float32)
    Lf_raw = cv2.bilateralFilter(cv2.cvtColor(epf_f, cv2.COLOR_BGR2LAB)[:, :, 0], d=7, sigmaColor=20, sigmaSpace=14).astype(np.float32)
    # Multi-Scale Retinex equalization so low-light / shadowed carvings match sunlit SNR cleanly
    Lc = _equalize_illumination_retinex(Lc_raw, strength=0.62)
    Lf = _equalize_illumination_retinex(Lf_raw, strength=0.60)
    Lf_u8 = np.clip(Lf, 0, 255).astype(np.uint8)

    r_f = max(5, int(round(m_dim * 0.006)) | 1)
    r_m = max(13, int(round(m_dim * 0.018)) | 1)
    ker_f = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_f, r_f))
    ker_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_m, r_m))

    # Coarse carving gate using capped axis equalization so faint flat-wood vertical grain isn't over-amplified
    bh_c = cv2.morphologyEx(Lc, cv2.MORPH_BLACKHAT, ker_f) * 0.58 + cv2.morphologyEx(Lc, cv2.MORPH_BLACKHAT, ker_m) * 0.62
    gx_c = np.abs(cv2.Scharr(Lc, cv2.CV_32F, 1, 0) / 16.0)
    gy_c = np.abs(cv2.Scharr(Lc, cv2.CV_32F, 0, 1) / 16.0)
    gx_sc = max(15.0, float(np.percentile(gx_c, 84)))
    gy_sc = max(15.0, float(np.percentile(gy_c, 84)))
    g_c = np.maximum(gx_c * min(1.32, 22.0 / gx_sc), gy_c * min(1.32, 22.0 / gy_sc))

    detail_bias = (0.70 - float(np.clip(detail, 0.15, 1.0))) * 14.0
    c_med = float(np.median(bh_c))
    c_mad = float(np.median(np.abs(bh_c - c_med)))
    gc_med = float(np.median(g_c))
    gc_mad = float(np.median(np.abs(g_c - gc_med)))

    c_zone = (
        (bh_c >= max(float(np.percentile(bh_c, 77)), c_med + 3.8 * max(c_mad, 1.3), 17.5) + detail_bias * 0.35)
        | (g_c >= max(float(np.percentile(g_c, 77)), gc_med + 3.6 * max(gc_mad, 1.3), 16.5) + detail_bias * 0.35)
    ).astype(np.uint8)
    k_dil = max(11, int(round(m_dim * 0.015)) | 1)
    zone_dil = cv2.dilate(c_zone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dil, k_dil)))
    k_rail = max(25, int(round(m_dim * 0.032)) | 1)
    zone_closed = cv2.morphologyEx(zone_dil, cv2.MORPH_CLOSE, np.ones((k_rail, 3), np.uint8))
    zone_closed = cv2.morphologyEx(zone_closed, cv2.MORPH_CLOSE, np.ones((3, k_rail), np.uint8))

    wm_sup = None
    if wm_mask is not None and wm_mask.shape[:2] == (vh, vw) and wm_mask.any():
        wm_sup = cv2.dilate((wm_mask > 0).astype(np.uint8) * 255, np.ones((11, 11), np.uint8))

    # Stage 2 prep first so straight CAD lines verify against true carved BlackHat ridges (blocking panel grain ticks)
    bh_f = cv2.morphologyEx(Lf, cv2.MORPH_BLACKHAT, ker_f) * 0.65 + cv2.morphologyEx(Lf, cv2.MORPH_BLACKHAT, ker_m) * 0.55
    gx_f = np.abs(cv2.Scharr(Lf, cv2.CV_32F, 1, 0) / 16.0)
    gy_f = np.abs(cv2.Scharr(Lf, cv2.CV_32F, 0, 1) / 16.0)
    gxf_sc = max(13.0, float(np.percentile(gx_f, 84)))
    gyf_sc = max(13.0, float(np.percentile(gy_f, 84)))
    g_f = np.maximum(gx_f * min(1.30, 20.0 / gxf_sc), gy_f * min(1.30, 20.0 / gyf_sc))

    bh_floor_f = max(float(np.percentile(bh_f, 71)), float(np.median(bh_f)) + 2.7 * max(float(np.median(np.abs(bh_f - np.median(bh_f)))), 1.1), 8.6) + detail_bias * 0.25
    g_floor_f = max(float(np.percentile(g_f, 72)), float(np.median(g_f)) + 2.6 * max(float(np.median(np.abs(g_f - np.median(g_f)))), 1.1), 8.5) + detail_bias * 0.25

    comb = np.maximum(np.clip((bh_f - bh_floor_f) * 3.15, 0, 255), np.clip((g_f - g_floor_f) * 2.90, 0, 255) * 0.92)
    comb_masked = np.where((zone_closed > 0) | (bh_c > c_med + 5.2 * max(c_mad, 1.3)), comb, 0.0).astype(np.uint8)
    ridge_gate = ((comb_masked > 24) | (bh_f >= bh_floor_f * 1.05)).astype(np.uint8)

    # Stage 1: Architectural Straight CAD Lines verified against genuine ridge depth (zero flat-panel wood grain!)
    arch_lines, arch_line_mask = _extract_straight_cad_lines(
        Lf_u8, zone_closed, wm_sup, off_x, off_y, sx_mm, sy_mm, detail, ridge_gate=ridge_gate
    )

    # Whole-Petal Multi-Scale Morphological White TopHat separates FULL carved leaves/petals cleanly to thin fine crevices
    r_l = max(21, int(round(m_dim * 0.036)) | 1)
    ker_l = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_l, r_l))
    wh_petal = (
        cv2.morphologyEx(Lf_u8, cv2.MORPH_TOPHAT, ker_f).astype(np.float32) * 0.48
        + cv2.morphologyEx(Lf_u8, cv2.MORPH_TOPHAT, ker_m).astype(np.float32) * 0.56
        + cv2.morphologyEx(Lf_u8, cv2.MORPH_TOPHAT, ker_l).astype(np.float32) * 0.34
    )
    bh_fine_crev = cv2.morphologyEx(Lf, cv2.MORPH_BLACKHAT, ker_f)
    bh_fine_fl = max(
        float(np.percentile(bh_fine_crev, 76)),
        float(np.median(bh_fine_crev)) + 2.8 * max(float(np.median(np.abs(bh_fine_crev - np.median(bh_fine_crev)))), 1.0),
        8.2,
    )
    crevice_cut = (bh_fine_crev >= bh_fine_fl * 1.15) & (zone_closed > 0)
    pos_vals = wh_petal[zone_dil > 0] if zone_dil.any() else wh_petal
    pos_floor_t = max(8.5, float(np.percentile(pos_vals, 52)) * 0.68) + detail_bias * 0.22
    raised_raw = np.where((zone_dil > 0) & (wh_petal >= pos_floor_t) & (~crevice_cut), 255, 0).astype(np.uint8)
    raised_raw = cv2.morphologyEx(raised_raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    raised_raw = cv2.morphologyEx(raised_raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Smooth morphological groove mask without jagged Canny pixel worms
    med_v = float(np.median(Lf_u8))
    canny_edges = cv2.Canny(Lf_u8, max(16, int(med_v * 0.30)), max(44, int(med_v * 0.82)))
    canny_zone = cv2.bitwise_and(canny_edges, cv2.dilate(zone_closed, np.ones((7, 7), np.uint8)) * 255)
    canny_dil = cv2.dilate(canny_zone, np.ones((2, 2), np.uint8))

    _, groove_mask = cv2.threshold(comb_masked, max(18, int(round(26 + detail_bias * 0.4))), 255, cv2.THRESH_BINARY)
    groove_mask = cv2.morphologyEx(groove_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    groove_mask = cv2.morphologyEx(groove_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Noise/speckle threshold dynamically driven by user noise_px & 4K canvas scale
    noise_scale = max(0.35, float(noise_px) / 10.0)
    k_act = max(21, int(round(m_dim * 0.028)) | 1)
    act_map = cv2.GaussianBlur((comb_masked > 42).astype(np.float32), (k_act, k_act), 0)
    min_area = max(44, int(vw * vh * 0.000044 * noise_scale))
    min_hole_px = max(32, int(vw * vh * 0.000030 * noise_scale))
    max_grain_w = max(22, int(vw * 0.026))

    # Derive structural carving frame bounds from long Architectural Frame lines so empty outer plank margins stay spotless
    frame_y_bot = vh * 0.91
    h_rails = [e for e in arch_lines if abs(e["y2"] - e["y1"]) <= 0.5 and abs(e["x2"] - e["x1"]) >= span_w * 0.25]
    if h_rails:
        max_ry = max(0.5 * (e["y1"] + e["y2"]) for e in h_rails)
        max_ry_px = (max_ry - off_y) / max(1e-6, sy_mm)
        if max_ry_px >= vh * 0.72:
            frame_y_bot = min(vh - 4.0, max_ry_px + vh * 0.015)

    arch_corridor = cv2.dilate(arch_line_mask, np.ones((7, 7), np.uint8)) if arch_line_mask.any() else None

    def filter_mask_components(src_m: np.ndarray, min_a: int, keep_rails: bool = True, skip_arch: bool = True) -> np.ndarray:
        nl, lbl, st, _ = cv2.connectedComponentsWithStats(src_m)
        out_m = np.zeros_like(src_m)
        for i in range(1, nl):
            area = int(st[i, cv2.CC_STAT_AREA])
            if area < min_a:
                continue
            bxi, byi, bwi, bhi = int(st[i, 0]), int(st[i, 1]), int(st[i, 2]), int(st[i, 3])
            cm = lbl[byi : byi + bhi, bxi : bxi + bwi] == i
            if wm_sup is not None and float((wm_sup[byi : byi + bhi, bxi : bxi + bwi][cm] > 0).mean()) > 0.40:
                continue
            # Suppress bottom catalog footer text glyphs sitting below lowest architectural rail (e.g. Doors Fair banner)
            if byi > frame_y_bot and bhi < vh * 0.065 and bwi < vw * 0.35:
                continue
            if (byi + bhi) < vh * 0.035 and bhi < vh * 0.025 and bwi < vw * 0.16:
                continue
            # Suppress outer bare-stile vertical wood-grain streaks inside outermost 10.5% margins
            in_outer_margin = ((bxi + bwi) <= vw * 0.105) or (bxi >= vw * 0.895)
            if in_outer_margin and bwi <= max_grain_w * 1.45 and bhi < vh * 0.55:
                continue
            if skip_arch and arch_corridor is not None and area < vw * vh * 0.04:
                arch_ov = float((arch_corridor[byi : byi + bhi, bxi : bxi + bwi][cm] > 0).mean())
                if arch_ov > 0.68:
                    continue
            sc = comb_masked[byi : byi + bhi, bxi : bxi + bwi][cm]
            pk, mn = int(sc.max()), float(sc.mean())
            act_mn = float(act_map[byi : byi + bhi, bxi : bxi + bwi][cm].mean())
            if act_mn < 0.045 and pk < 96 and area < min_a * 10:
                continue
            if bhi >= bwi * 3.5 and bwi <= max_grain_w:
                mean_w = area / max(1, bhi)
                if (bhi >= bwi * 4.5 and bhi < vh * 0.22 and mean_w < max(8.5, vw * 0.0115)) or (bhi >= bwi * 3.9 and bhi < vh * 0.16 and mean_w < max(6.0, vw * 0.0088)) or (mn < 88.0 and pk < 116 and bhi < vh * 0.32) or (act_mn < 0.075 and bhi < vh * 0.50 and mn < 102.0):
                    continue
            if not keep_rails and area < min_a * 1.8 and pk < 92:
                continue
            if area < min_a * 2.8 and pk < 82:
                continue
            out_m[lbl == i] = 255
        if wm_sup is not None:
            out_m[wm_sup > 0] = 0
        return _fill_tiny_holes(out_m, min_hole_px)

    clean_grooves = filter_mask_components(groove_mask, min_area, keep_rails=True, skip_arch=True)
    clean_raised_raw = filter_mask_components(raised_raw, max(56, int(min_area * 1.35)), keep_rails=False, skip_arch=True)
    if arch_line_mask.any():
        clean_raised = _fill_tiny_holes(cv2.bitwise_and(clean_raised_raw, 255 - arch_line_mask), min_hole_px)
    else:
        clean_raised = clean_raised_raw

    # Clean Manual-CAD Layer Separation: NEVER slice/tear green petals to extract cyan grooves!
    # Keep raised motifs 100% intact, and extract only genuinely disjoint grooves outside a 5x5 halo around motifs & straight lines.
    exclusion_m = np.zeros_like(clean_grooves)
    if arch_corridor is not None:
        exclusion_m = cv2.bitwise_or(exclusion_m, arch_corridor)
    raised_frac = float((clean_raised > 0).sum()) / max(1, vw * vh)
    if raised_frac >= 0.012:
        exclusion_m = cv2.bitwise_or(exclusion_m, cv2.dilate(clean_raised, np.ones((5, 5), np.uint8)))
    groove_sub = cv2.bitwise_and(clean_grooves, 255 - exclusion_m) if exclusion_m.any() else clean_grooves
    # Close tiny gaps along curved architectural borders (e.g. top arch) cleanly without touching central floral motifs
    groove_sub = cv2.morphologyEx(groove_sub, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    groove_unique = filter_mask_components(groove_sub, max(44, int(min_area * 1.25)), keep_rails=True, skip_arch=True)

    target_mask = cv2.bitwise_or(arch_line_mask, cv2.bitwise_or(clean_grooves, clean_raised))
    target_mask = cv2.bitwise_or(target_mask, cv2.bitwise_and(canny_zone, ((comb_masked > 10).astype(np.uint8)) * 255))
    if wm_sup is not None:
        target_mask[wm_sup > 0] = 0
    if frame_y_bot < vh - 4.0:
        # Exclude bottom margin catalog banner from coverage target mask
        bot_i = int(round(frame_y_bot))
        if arch_line_mask.any():
            bot_keep = cv2.dilate(arch_line_mask[bot_i:, :], np.ones((5, 5), np.uint8))
            target_mask[bot_i:, :] = cv2.bitwise_and(target_mask[bot_i:, :], bot_keep)
        else:
            target_mask[bot_i:, :] = 0

    speckle_px = max(24.0, float(min_area) * 0.90)
    straight_tol = max(0.88, 1.32 - detail * 0.38)
    organic_corners = max(0.25, corners * 0.85)

    # Stage 3: Rust Multi-Core Subpixel Vectorization + Closed-Loop Convergence Verification
    all_ents: list[dict[str, Any]] = list(arch_lines)
    if raised_frac >= 0.012:
        all_ents.extend(
            _trace_field_rust(
                clean_raised, off_x, off_y, sx_mm, sy_mm,
                detail=detail, corners=organic_corners, min_area_px=speckle_px * 1.15,
                straight_tol_px=straight_tol, layer_group="Raised Relief Motifs", smooth_sigma=0.84,
            ) or _trace_mask_vtracer(
                clean_raised, off_x, off_y, sx_mm, sy_mm,
                max(10, int(round(18 - detail * 8))), int(round(40 + corners * 18)), 5.0,
                layer_group="Raised Relief Motifs", smooth_mask=True,
            )
        )
    all_ents.extend(
        _trace_field_rust(
            groove_unique, off_x, off_y, sx_mm, sy_mm,
            detail=detail, corners=organic_corners, min_area_px=speckle_px,
            straight_tol_px=straight_tol, layer_group="Carved Grooves & Panels", smooth_sigma=0.82,
        ) or _trace_mask_vtracer(
            groove_unique, off_x, off_y, sx_mm, sy_mm,
            max(8, int(round(16 - detail * 8))), int(round(40 + corners * 18)), 4.6,
            layer_group="Carved Grooves & Panels", smooth_mask=True,
        )
    )
    cov, min_tile, uncovered = _measure_vector_coverage(all_ents, target_mask, off_x, off_y, span_w, span_h)
    iters = 1

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
                if inter / min(ea, ba) > 0.76 and abs(ea - ba) / max(ea, ba) < 0.35:
                    dup = True
                    break
            if not dup:
                base.append(e)
                boxes.append((x0e, y0e, x1e, y1e, len(ps)))

    while (cov < 0.960 or min_tile < 0.86) and iters < 4:
        iters += 1
        if iters == 2:
            dark_boost = cv2.bitwise_and(cv2.dilate(((comb_masked > 16).astype(np.uint8)) * 255, np.ones((3, 3), np.uint8)), uncovered)
            if exclusion_m.any():
                dark_boost = cv2.bitwise_and(dark_boost, 255 - exclusion_m)
            dark_boost = cv2.morphologyEx(dark_boost, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            dark_boost = filter_mask_components(dark_boost, max(40, int(min_area * 1.0)), keep_rails=True, skip_arch=True)
            if int((dark_boost > 0).sum()) > 32:
                pass2 = _trace_field_rust(
                    dark_boost, off_x, off_y, sx_mm, sy_mm,
                    detail=detail, corners=corners, min_area_px=max(24.0, speckle_px * 0.92),
                    straight_tol_px=straight_tol, layer_group="Carved Grooves & Panels", smooth_sigma=0.82,
                )
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
            if exclusion_m.any():
                res_bin = cv2.bitwise_and(res_bin, 255 - exclusion_m)
            res_bin = filter_mask_components(res_bin, max(36, int(min_area * 0.9)), keep_rails=True, skip_arch=True)
            if int((res_bin > 0).sum()) > 28:
                pass4 = _trace_field_rust(
                    res_bin, off_x, off_y, sx_mm, sy_mm,
                    detail=detail, corners=corners, min_area_px=max(20.0, speckle_px * 0.80),
                    straight_tol_px=straight_tol, layer_group="Carved Grooves & Panels", smooth_sigma=0.60,
                )
                merge_non_overlapping(all_ents, pass4)

        cov, min_tile, uncovered = _measure_vector_coverage(all_ents, target_mask, off_x, off_y, span_w, span_h)

    conv = {
        "coverage_pct": round(cov * 100.0, 1),
        "min_tile_pct": round(min_tile * 100.0, 1),
        "iterations": iters,
        "converged": bool(cov >= 0.960 and min_tile >= 0.86),
    }

    group_order = {"Architectural Frame": 0, "Raised Relief Motifs": 1, "Carved Grooves & Panels": 2}
    all_ents.sort(key=lambda e: (
        group_order.get(e.get("layerGroup", ""), 3),
        min((p["y"] for p in (e.get("points") or [{"y": e.get("y1", 0.0)}])), default=e.get("y1", 0.0)),
        min((p["x"] for p in (e.get("points") or [{"x": e.get("x1", 0.0)}])), default=e.get("x1", 0.0)),
    ))
    g_cnt: dict[str, int] = {}
    for e in all_ents:
        lg = e.get("layerGroup", "Carved Grooves & Panels")
        g_cnt[lg] = g_cnt.get(lg, 0) + 1
        prefix = "Carved Motif" if lg == "Raised Relief Motifs" else ("Frame Line" if e.get("type") == "line" else ("Frame Part" if lg == "Architectural Frame" else "Carved Panel"))
        e["name"] = f"{prefix} #{g_cnt[lg]}"

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
    trace_quality = int(np.clip(req.get("trace_quality", 3840), 1024, 3840))
    noise_px = float(np.clip(req.get("noise_px", 10.0), 1.0, 64.0))
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

    wm_mask = None
    if roi is not None:
        x0, y0, x1, y1 = roi["x0"], roi["y0"], roi["x1"], roi["y1"]
        sub = bgr[y0 : y1 + 1, x0 : x1 + 1].copy()
        sub, wm_mask = _inpaint_watermarks(
            sub, roi["dc"][y0 : y1 + 1, x0 : x1 + 1], roi["L"][y0 : y1 + 1, x0 : x1 + 1], roi["l_med"]
        )
    else:
        x0, y0, x1, y1 = 0, 0, iw - 1, ih - 1
        sub = bgr.copy()
        if float((bgra[:, :, 3] < 220).mean()) > 0.05:
            # Composite transparent logo cleanly onto white background for binary VTracer
            alpha_f = bgra[:, :, 3:4].astype(np.float32) / 255.0
            sub = np.clip(sub.astype(np.float32) * alpha_f + 255.0 * (1.0 - alpha_f), 0, 255).astype(np.uint8)
        elif not is_logo:
            L_full = cv2.cvtColor(sub, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
            l_med_f = float(np.median(L_full))
            dc_full = np.abs(L_full - cv2.GaussianBlur(L_full, (0, 0), sigmaX=max(12.0, min(iw, ih) * 0.025)))
            sub, wm_mask = _inpaint_watermarks(sub, dc_full, L_full, l_med_f)

    sh0, sw0 = sub.shape[:2]
    target_long = min(3840, trace_quality)
    if roi is not None or not is_logo:
        proc_cap = min(target_long, 2600 if trace_quality >= 3840 else target_long)
        if max(sw0, sh0) < min(2048, proc_cap):
            sc = min(3.4, min(2048.0, float(proc_cap)) / max(1, max(sw0, sh0)))
            if sc > 1.02:
                if max(sw0, sh0) <= 1400:
                    sub = cv2.fastNlMeansDenoisingColored(sub, None, 4, 4, 7, 21)
                nw_s, nh_s = int(round(sw0 * sc)), int(round(sh0 * sc))
                sub = cv2.resize(sub, (nw_s, nh_s), interpolation=cv2.INTER_LANCZOS4)
                if wm_mask is not None:
                    wm_mask = cv2.resize(wm_mask, (nw_s, nh_s), interpolation=cv2.INTER_NEAREST)
                sub = cv2.addWeighted(sub, 1.15, cv2.GaussianBlur(sub, (0, 0), 1.05), -0.15, 0)
    elif max(sw0, sh0) < min(1600, target_long):
        sc = min(2.8, min(1600.0, float(target_long)) / max(1, max(sw0, sh0)))
        nw_s, nh_s = int(round(sw0 * sc)), int(round(sh0 * sc))
        sub = cv2.resize(sub, (nw_s, nh_s), interpolation=cv2.INTER_LANCZOS4)
        if wm_mask is not None:
            wm_mask = cv2.resize(wm_mask, (nw_s, nh_s), interpolation=cv2.INTER_NEAREST)

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
        wood_ents, convergence = _extract_wood_relief_vectors(
            sub, off_x, off_y, sx_mm, sy_mm, detail, corners, wm_mask=wm_mask, noise_px=noise_px
        )
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
        "engine": "opencv-rust-4k-vision",
        "entities": entities[:1400],
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

    wm_mask_3d = None
    if roi is not None:
        x0, y0, x1, y1 = roi["x0"], roi["y0"], roi["x1"], roi["y1"]
        crop_bgra = bgra[y0 : y1 + 1, x0 : x1 + 1].copy()
        cleaned_bgr, wm_mask_3d = _inpaint_watermarks(
            crop_bgra[:, :, :3],
            roi["dc"][y0 : y1 + 1, x0 : x1 + 1],
            roi["L"][y0 : y1 + 1, x0 : x1 + 1],
            roi["l_med"],
        )
        crop_bgra[:, :, :3] = cleaned_bgr
    else:
        x0, y0, x1, y1 = 0, 0, iw - 1, ih - 1
        crop_bgra = bgra.copy()
        if not is_logo:
            L_f3 = cv2.cvtColor(crop_bgra[:, :, :3], cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
            l_m3 = float(np.median(L_f3))
            dc_3 = np.abs(L_f3 - cv2.GaussianBlur(L_f3, (0, 0), sigmaX=max(12.0, min(iw, ih) * 0.025)))
            cleaned_bgr, wm_mask_3d = _inpaint_watermarks(crop_bgra[:, :, :3], dc_3, L_f3, l_m3)
            crop_bgra[:, :, :3] = cleaned_bgr

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
        if wm_mask_3d is not None:
            wm_mask_3d = cv2.resize(wm_mask_3d, (nw, nh), interpolation=cv2.INTER_NEAREST)
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
        depth_norm = _solve_planar_relief_heightmap(crop_bgra[:, :, :3], wm_mask=wm_mask_3d)
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
