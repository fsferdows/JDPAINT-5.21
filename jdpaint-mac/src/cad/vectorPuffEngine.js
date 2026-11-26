// High-Speed 2D Vector to 3D Relief Puffing Engine (JDPaint Domain Forming & Stroke Emboss)
// Optimized with shared offscreen canvas, compound evenodd holes, & adaptive Bounding-Box EDT

import { CadCanvas } from './cadCanvas.js';

export class VectorPuffEngine {
  static _sharedCanvas = null;

  static getSharedCanvas(N) {
    if (typeof document === 'undefined') return null;
    if (!VectorPuffEngine._sharedCanvas) {
      VectorPuffEngine._sharedCanvas = document.createElement('canvas');
    }
    if (VectorPuffEngine._sharedCanvas.width !== N || VectorPuffEngine._sharedCanvas.height !== N) {
      VectorPuffEngine._sharedCanvas.width = N;
      VectorPuffEngine._sharedCanvas.height = N;
    }
    return VectorPuffEngine._sharedCanvas;
  }

  static isPuffable(ent) {
    if (!ent) return false;
    if (ent.type === 'rect' || ent.type === 'circle' || ent.type === 'text' || ent.type === 'line') return true;
    if (ent.type === 'polyline' || ent.type === 'curve') {
      return ent.points && ent.points.length >= 2;
    }
    return false;
  }

  static shouldStrokeEntity(ent, forceCloseOpen = false) {
    if (ent.type === 'line') return true;
    if (ent.type === 'curve' || ent.type === 'polyline') {
      if (ent.relief && ent.relief.openAsStroke === false && ent.points.length >= 3) {
        return false;
      }
      if (!ent.isClosed && !forceCloseOpen) {
        const pts = ent.points;
        if (pts.length >= 3) {
          if (Math.hypot(pts[0].x - pts[pts.length - 1].x, pts[0].y - pts[pts.length - 1].y) <= 0.5) return false;
        }
        return true;
      }
      return ent.points.length < 3;
    }
    return false;
  }

  /**
   * Compute tight pixel bounding box for an entity so we only read & run EDT on that exact sub-tile!
   */
  static getEntityPixelBox(ent, scaleX, scaleY, N, padPx = 4) {
    let minX = 0, minY = 0, maxX = N - 1, maxY = N - 1;

    if (ent.type === 'rect') {
      const x1 = ent.x * scaleX, x2 = (ent.x + ent.w) * scaleX;
      const y1 = ent.y * scaleY, y2 = (ent.y + ent.h) * scaleY;
      minX = Math.min(x1, x2); maxX = Math.max(x1, x2);
      minY = Math.min(y1, y2); maxY = Math.max(y1, y2);
    } else if (ent.type === 'circle') {
      minX = (ent.cx - ent.r) * scaleX; maxX = (ent.cx + ent.r) * scaleX;
      minY = (ent.cy - ent.r) * scaleY; maxY = (ent.cy + ent.r) * scaleY;
    } else if (ent.type === 'line') {
      minX = Math.min(ent.x1, ent.x2) * scaleX; maxX = Math.max(ent.x1, ent.x2) * scaleX;
      minY = Math.min(ent.y1, ent.y2) * scaleY; maxY = Math.max(ent.y1, ent.y2) * scaleY;
    } else if ((ent.type === 'polyline' || ent.type === 'curve') && ent.points?.length > 0) {
      let lx = Infinity, rx = -Infinity, ty = Infinity, by = -Infinity;
      const expandRing = (ring) => {
        if (!ring) return;
        for (const p of ring) {
          if (p.x < lx) lx = p.x;
          if (p.x > rx) rx = p.x;
          if (p.y < ty) ty = p.y;
          if (p.y > by) by = p.y;
          if (p.hIn) {
            if (p.hIn.x < lx) lx = p.hIn.x;
            if (p.hIn.x > rx) rx = p.hIn.x;
            if (p.hIn.y < ty) ty = p.hIn.y;
            if (p.hIn.y > by) by = p.hIn.y;
          }
          if (p.hOut) {
            if (p.hOut.x < lx) lx = p.hOut.x;
            if (p.hOut.x > rx) rx = p.hOut.x;
            if (p.hOut.y < ty) ty = p.hOut.y;
            if (p.hOut.y > by) by = p.hOut.y;
          }
        }
      };
      expandRing(ent.points);
      if (ent.holes && ent.holes.length > 0) {
        for (const hole of ent.holes) expandRing(hole);
      }
      minX = lx * scaleX; maxX = rx * scaleX;
      minY = ty * scaleY; maxY = by * scaleY;
    } else if (ent.type === 'text') {
      const fontSizeMm = (ent.size || 16) * 0.4;
      const estW = Math.max(12, (ent.text?.length || 1) * fontSizeMm * 0.72 * scaleX);
      const estH = Math.max(12, fontSizeMm * 1.15 * scaleY);
      minX = ent.x * scaleX - 4; maxX = ent.x * scaleX + estW;
      minY = ent.y * scaleY - estH; maxY = ent.y * scaleY + 8;
    }

    const x0 = Math.max(0, Math.floor(minX - padPx));
    const y0 = Math.max(0, Math.floor(minY - padPx));
    const x1 = Math.min(N - 1, Math.ceil(maxX + padPx));
    const y1 = Math.min(N - 1, Math.ceil(maxY + padPx));
    return { x0, y0, bw: Math.max(1, x1 - x0 + 1), bh: Math.max(1, y1 - y0 + 1) };
  }

  /**
   * Rebuild full 3D relief cleanly from Base Photo Heightmap + All Enabled CAD Vector Layers.
   * When clipBaseToEntities=true, trims base photo to stay strictly inside active visible 2D vector outlines!
   */
  static rebuildFromLayers(entities, reliefEngine, options = {}) {
    const {
      baseBuffer = null,
      baseEnabled = true,
      baseWeight = 1.0,
      clipBaseToEntities = false,
      defaultPuff = {},
    } = options;

    const N = reliefEngine.resolution;
    const len = N * N;
    reliefEngine.data.fill(0);

    if (baseEnabled && baseBuffer && baseBuffer.length === len) {
      const closedVisible = entities.filter(
        (ent) => ent.visible !== false && !VectorPuffEngine.shouldStrokeEntity(ent, false) && ent.type !== 'text'
      );

      if (clipBaseToEntities && closedVisible.length > 0) {
        const clipMask = VectorPuffEngine.buildUnionPolygonMask(closedVisible, N, reliefEngine.width, reliefEngine.height);
        for (let i = 0; i < len; i++) {
          if (clipMask[i]) {
            reliefEngine.data[i] = Math.min(1.0, Math.max(0.0, baseBuffer[i] * baseWeight));
          }
        }
      } else {
        for (let i = 0; i < len; i++) {
          reliefEngine.data[i] = Math.min(1.0, Math.max(0.0, baseBuffer[i] * baseWeight));
        }
      }
    }

    let appliedCount = 0;
    for (const ent of entities) {
      if (ent.visible === false) continue;
      if (ent.relief && ent.relief.enabled === false) continue;
      if (!VectorPuffEngine.isPuffable(ent)) continue;

      const cfg = {
        shape: ent.relief?.shape || defaultPuff.shape || 'dome',
        heightMm: ent.relief?.heightMm ?? defaultPuff.heightMm ?? 4.0,
        radiusMm: ent.relief?.radiusMm ?? defaultPuff.radiusMm ?? 12.0,
        adaptiveRadius: ent.relief?.adaptiveRadius ?? defaultPuff.adaptiveRadius ?? true,
        mode: ent.relief?.mode || defaultPuff.mode || 'max',
        forceCloseOpen: false,
      };

      VectorPuffEngine.applyEntityToRelief(ent, reliefEngine, cfg);
      appliedCount++;
    }

    // Light bilateral smoothing strictly on active non-hole interior pixels so overlapping shapes blend smoothly without ragged step edges
    if (appliedCount > 0 && N >= 64) {
      const src = new Float32Array(reliefEngine.data);
      for (let y = 1; y < N - 1; y++) {
        const row = y * N;
        for (let x = 1; x < N - 1; x++) {
          const i = row + x;
          const zc = src[i];
          if (zc <= 0.002) continue;
          const zL = src[i - 1], zR = src[i + 1], zU = src[i - N], zD = src[i + N];
          if (zL <= 0.001 || zR <= 0.001 || zU <= 0.001 || zD <= 0.001) continue;
          let sum = zc * 2.5, wSum = 2.5;
          for (const zn of [zL, zR, zU, zD]) {
            const dz = zn - zc;
            const wt = Math.exp(-(dz * dz) * 32.0);
            sum += zn * wt;
            wSum += wt;
          }
          reliefEngine.data[i] = sum / wSum;
        }
      }
    }

    reliefEngine.notify();
    return appliedCount;
  }

  static traceSmoothPath(ctx, pts, isClosed, scaleX, scaleY) {
    CadCanvas.traceSmoothCanvasPath(ctx, pts, isClosed, (x) => x * scaleX, (y) => y * scaleY);
  }

  /**
   * Trace closed outer contour + all compound holes into current Canvas2D path for evenodd fill.
   */
  static traceClosedEntitySubpaths(ctx, ent, scaleX, scaleY) {
    if (ent.type === 'rect') {
      const rr = (ent.cornerRadius || 0) * scaleX;
      if (rr > 0.5 && typeof ctx.roundRect === 'function') {
        ctx.roundRect(ent.x * scaleX, ent.y * scaleY, ent.w * scaleX, ent.h * scaleY, rr);
      } else {
        ctx.rect(ent.x * scaleX, ent.y * scaleY, ent.w * scaleX, ent.h * scaleY);
      }
      return;
    }
    if (ent.type === 'circle') {
      ctx.arc(ent.cx * scaleX, ent.cy * scaleY, Math.max(0.1, ent.r) * scaleX, 0, Math.PI * 2);
      return;
    }
    const tracePolyRing = (ring) => {
      if (!ring || ring.length < 3) return;
      ctx.moveTo(ring[0].x * scaleX, ring[0].y * scaleY);
      for (let i = 1; i < ring.length; i++) ctx.lineTo(ring[i].x * scaleX, ring[i].y * scaleY);
      ctx.closePath();
    };
    if (ent.type === 'curve' && ent.points?.length >= 2) {
      VectorPuffEngine.traceSmoothPath(ctx, ent.points, true, scaleX, scaleY);
      if (ent.holes && ent.holes.length > 0) {
        for (const hole of ent.holes) {
          if (hole.length >= 2) VectorPuffEngine.traceSmoothPath(ctx, hole, true, scaleX, scaleY);
        }
      }
    } else if (ent.type === 'polyline' && ent.points?.length >= 3) {
      tracePolyRing(ent.points);
      if (ent.holes && ent.holes.length > 0) {
        for (const hole of ent.holes) tracePolyRing(hole);
      }
    }
  }

  /**
   * Fast union raster mask of all visible closed CAD polygons for 2D-Outline 3D clipping
   */
  static buildUnionPolygonMask(entities, N, wMm, hMm) {
    const maskCanvas = VectorPuffEngine.getSharedCanvas(N);
    const ctx = maskCanvas.getContext('2d', { willReadFrequently: true });
    ctx.clearRect(0, 0, N, N);
    ctx.fillStyle = '#ffffff';

    const scaleX = N / wMm;
    const scaleY = N / hMm;

    for (const ent of entities) {
      ctx.beginPath();
      VectorPuffEngine.traceClosedEntitySubpaths(ctx, ent, scaleX, scaleY);
      ctx.fill('evenodd');
    }

    const data = ctx.getImageData(0, 0, N, N).data;
    const out = new Uint8Array(N * N);
    for (let i = 0; i < N * N; i++) {
      out[i] = data[i * 4 + 3] > 64 ? 1 : 0;
    }
    return out;
  }

  static puffEntities(entities, reliefEngine, options = {}) {
    const {
      shape = 'dome',
      heightMm = 4.0,
      radiusMm = 12.0,
      adaptiveRadius = true,
      mode = 'max',
      clearBefore = false,
      useLayerSettings = false,
      forceCloseOpen = false,
    } = options;

    const targetEntities = entities.filter((ent) => ent.visible !== false && VectorPuffEngine.isPuffable(ent));
    if (targetEntities.length === 0) return 0;

    if (clearBefore) {
      reliefEngine.data.fill(0);
    }

    for (const ent of targetEntities) {
      const cfg = useLayerSettings && ent.relief
        ? {
            shape: ent.relief.shape || shape,
            heightMm: ent.relief.heightMm ?? heightMm,
            radiusMm: ent.relief.radiusMm ?? radiusMm,
            adaptiveRadius: ent.relief.adaptiveRadius ?? adaptiveRadius,
            mode: ent.relief.mode || mode,
            forceCloseOpen,
          }
        : { shape, heightMm, radiusMm, adaptiveRadius, mode, forceCloseOpen };

      VectorPuffEngine.applyEntityToRelief(ent, reliefEngine, cfg);
    }

    reliefEngine.notify();
    return targetEntities.length;
  }

  /**
   * High-Speed Bounding-Box Scoped Entity Rasterization & Adaptive EDT Puffing
   */
  static applyEntityToRelief(ent, reliefEngine, cfg) {
    const {
      shape = 'dome',
      heightMm = 4.0,
      radiusMm = 12.0,
      adaptiveRadius = false,
      mode = 'max',
      forceCloseOpen = false,
    } = cfg;

    const N = reliefEngine.resolution;
    const wMm = reliefEngine.width;
    const hMm = reliefEngine.height;
    const maxDepth = reliefEngine.maxDepth || 8.0;

    const maskCanvas = VectorPuffEngine.getSharedCanvas(N);
    const ctx = maskCanvas.getContext('2d', { willReadFrequently: true });

    const scaleX = N / wMm;
    const scaleY = N / hMm;
    const mmPerPixel = (wMm + hMm) / (2 * N);

    const isStroke = VectorPuffEngine.shouldStrokeEntity(ent, forceCloseOpen);
    const strokeHalfWidthMm = Math.max(1.0, Math.min(radiusMm, 25.0));
    const strokeWidthPx = Math.max(2, strokeHalfWidthMm * 2 * Math.min(scaleX, scaleY));
    const padPx = isStroke ? Math.ceil(strokeWidthPx / 2 + 4) : 4;

    const box = VectorPuffEngine.getEntityPixelBox(ent, scaleX, scaleY, N, padPx);
    if (box.bw <= 0 || box.bh <= 0) return;

    // Clear ONLY the bounding-box tile on shared canvas
    ctx.clearRect(box.x0, box.y0, box.bw, box.bh);

    if (ent.type === 'text') {
      ctx.fillStyle = '#ffffff';
      const fontSizePx = (ent.size || 16) * 0.4 * scaleY;
      ctx.font = `${ent.fontWeight || 'bold'} ${Math.max(8, fontSizePx)}px ${ent.fontFamily || 'sans-serif'}`;
      ctx.fillText(ent.text || '', ent.x * scaleX, ent.y * scaleY);
    } else if (isStroke) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = strokeWidthPx;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();

      if (ent.type === 'line') {
        ctx.moveTo(ent.x1 * scaleX, ent.y1 * scaleY);
        ctx.lineTo(ent.x2 * scaleX, ent.y2 * scaleY);
      } else if (ent.type === 'curve') {
        VectorPuffEngine.traceSmoothPath(ctx, ent.points, !!ent.isClosed, scaleX, scaleY);
      } else {
        const pts = ent.points;
        ctx.moveTo(pts[0].x * scaleX, pts[0].y * scaleY);
        for (let i = 1; i < pts.length; i++) {
          ctx.lineTo(pts[i].x * scaleX, pts[i].y * scaleY);
        }
      }
      ctx.stroke();
    } else {
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      VectorPuffEngine.traceClosedEntitySubpaths(ctx, ent, scaleX, scaleY);
      ctx.fill('evenodd');
    }

    // Read ONLY the bounding-box pixels with sub-pixel coverage anti-aliasing
    const imgData = ctx.getImageData(box.x0, box.y0, box.bw, box.bh);
    const tileLen = box.bw * box.bh;
    const mask = new Uint8Array(tileLen);
    const rawAlpha = new Float32Array(tileLen);
    const alphaCov = new Float32Array(tileLen);
    let anyPixel = false;
    for (let i = 0; i < tileLen; i++) {
      const a = imgData.data[i * 4 + 3];
      if (a > 10) {
        const af = a / 255.0;
        rawAlpha[i] = af;
        // Hermite S-curve subpixel coverage removes 1px diagonal sawtooth stairs along angled vectors & curved arcs
        const ramp = Math.max(0.0, Math.min(1.0, (af - 0.04) / 0.84));
        alphaCov[i] = ramp * ramp * (3.0 - 2.0 * ramp);
        if (a >= 96) mask[i] = 1;
        anyPixel = true;
      }
    }
    if (!anyPixel) return;

    // Run Exact Euclidean Distance Transform ONLY on the bw x bh sub-tile!
    const distField = VectorPuffEngine.computeInteriorDistanceRect(mask, box.bw, box.bh);
    let actualMaxDistPx = 0;
    for (let i = 0; i < tileLen; i++) {
      if (mask[i] === 1 && distField[i] > actualMaxDistPx) actualMaxDistPx = distField[i];
    }

    const effectiveRadiusMm = isStroke ? strokeHalfWidthMm : radiusMm;
    const rawTargetPx = Math.max(1.0, effectiveRadiusMm / mmPerPixel);
    const isAdaptive = !!(adaptiveRadius || ent.relief?.adaptiveRadius);
    const targetRadiusPixels = isAdaptive && actualMaxDistPx > 0
      ? Math.min(rawTargetPx, Math.max(1.2, actualMaxDistPx))
      : rawTargetPx;
    // Prevent only ultra-thin (<0.8mm wide) micro noise from towering as spikes while keeping normal 1mm+ carved petals lush & bold
    const widthTaper = isAdaptive && shape !== 'flat'
      ? Math.min(1.0, Math.max(0.45, (actualMaxDistPx * mmPerPixel) / 0.42))
      : 1.0;
    const normHeight = Math.min(1.0, Math.max(0.0, Math.abs(heightMm) / maxDepth)) * widthTaper;
    const flatBevelPx = Math.max(1.5, Math.min(rawTargetPx, Math.max(2.0, actualMaxDistPx * 0.35)));

    for (let ly = 0; ly < box.bh; ly++) {
      const globalRow = (box.y0 + ly) * N;
      const localRow = ly * box.bw;
      for (let lx = 0; lx < box.bw; lx++) {
        const lIdx = localRow + lx;
        const cov = alphaCov[lIdx];
        if (cov > 0.005) {
          const baseD = mask[lIdx] === 1 ? distField[lIdx] : 0.0;
          const af = rawAlpha[lIdx];
          const d = af < 0.98 ? Math.max(0.08, baseD + (af - 0.5) * 0.78) : baseD;
          const t = Math.min(1.0, d / targetRadiusPixels);

          let profileZ = 0;
          if (shape === 'dome') {
            // Blend smooth S-base with elliptic dome so perimeter rises gracefully without black staircase edge cliffs!
            const circ = Math.pow(Math.max(0, 1.0 - Math.pow(1.0 - t, 2)), 0.72);
            const sinBell = 0.5 * (1.0 - Math.cos(t * Math.PI));
            profileZ = circ * 0.55 + sinBell * 0.45;
          } else if (shape === 'cone') {
            profileZ = t;
          } else if (shape === 'smooth') {
            profileZ = 0.5 * (1.0 - Math.cos(t * Math.PI));
          } else if (shape === 'ogee') {
            // Classical furniture Roman Ogee molding curve
            profileZ = t < 0.5 ? 2 * t * t : 1.0 - Math.pow(-2 * t + 2, 2) / 2;
          } else if (shape === 'vcarve') {
            // Crisp V-Bit chiseled ridge profile
            profileZ = Math.pow(t, 0.72);
          } else if (shape === 'scoop') {
            // Concave bowl/dish profile (carves inward cleanly)
            profileZ = 1.0 - Math.sqrt(Math.max(0, 1.0 - t * t));
          } else if (shape === 'flat') {
            // Crisp beveled plateau for architectural signage & plaques
            const tb = Math.min(1.0, d / flatBevelPx);
            profileZ = tb * tb * (3.0 - 2.0 * tb);
          }

          const zVal = profileZ * normHeight * cov;
          const gIdx = globalRow + (box.x0 + lx);

          if (mode === 'add') {
            reliefEngine.data[gIdx] = Math.min(1.0, reliefEngine.data[gIdx] + zVal);
          } else if (mode === 'max') {
            reliefEngine.data[gIdx] = Math.max(reliefEngine.data[gIdx], zVal);
          } else if (mode === 'sub') {
            reliefEngine.data[gIdx] = Math.max(0.0, reliefEngine.data[gIdx] - zVal);
          }
        }
      }
    }
  }

  /**
   * Exact 2-Pass Squared Euclidean Distance Transform (Felzenszwalb-Huttenlocher)
   * Eliminates 8-sided chamfer ridges so 3D domes, petals, and curved moldings carve smoothly!
   */
  static computeInteriorDistanceRect(mask, w, h) {
    const INF = 1e12;
    const len = w * h;
    const colDistSq = new Float64Array(len);

    // Pass 1: Exact 1D distance along columns
    for (let x = 0; x < w; x++) {
      let lastZero = -INF;
      for (let y = 0; y < h; y++) {
        const idx = y * w + x;
        if (mask[idx] === 0) {
          lastZero = y;
          colDistSq[idx] = 0;
        } else {
          const d = y - lastZero;
          colDistSq[idx] = lastZero < -1e9 ? INF : d * d;
        }
      }
      lastZero = INF;
      for (let y = h - 1; y >= 0; y--) {
        const idx = y * w + x;
        if (mask[idx] === 0) {
          lastZero = y;
        } else if (lastZero < 1e9) {
          const d = lastZero - y;
          const d2 = d * d;
          if (d2 < colDistSq[idx]) colDistSq[idx] = d2;
        }
      }
    }

    // Pass 2: 1D lower parabola envelope along rows
    const dist = new Float32Array(len);
    const v = new Int32Array(w);
    const z = new Float64Array(w + 1);

    for (let y = 0; y < h; y++) {
      const row = y * w;
      let k = 0;
      v[0] = 0;
      z[0] = -Infinity;
      z[1] = Infinity;

      for (let q = 1; q < w; q++) {
        const fq = colDistSq[row + q];
        let s;
        while (true) {
          const r = v[k];
          s = ((fq + q * q) - (colDistSq[row + r] + r * r)) / (2 * (q - r));
          if (s <= z[k]) {
            k--;
            if (k < 0) { k = 0; break; }
          } else {
            break;
          }
        }
        k++;
        v[k] = q;
        z[k] = s;
        z[k + 1] = Infinity;
      }

      k = 0;
      for (let q = 0; q < w; q++) {
        while (z[k + 1] < q) k++;
        const r = v[k];
        const dx = q - r;
        const d2 = dx * dx + colDistSq[row + r];
        if (mask[row + q] === 0) {
          dist[row + q] = 0;
        } else {
          dist[row + q] = d2 >= INF * 0.5 ? Math.max(w, h) : Math.max(0.25, Math.sqrt(d2) - 0.5);
        }
      }
    }

    return dist;
  }

  static computeInteriorDistance(mask, N) {
    return VectorPuffEngine.computeInteriorDistanceRect(mask, N, N);
  }
}
