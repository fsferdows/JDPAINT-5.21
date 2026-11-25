// High-Accuracy Image Vectorizer & CNC Wood Design Tracer for JDPaint Mac
// Features: Summed-Area Adaptive Local Thresholding, Wood-Grain Gaussian Denoise,
// Sobel Edge Contour Mode, Otsu Bimodality Auto-Detection, Moore-Neighbor Tracing, and Corner-Preserving Curve Smoothing

export class ImageVectorizer {
  /**
   * Compute aspect-ratio-preserving workpiece bounds (in mm) for an image.
   */
  static getImageFitBounds(img, widthMm = 100, heightMm = 100) {
    const iw = img.naturalWidth || img.width || 1;
    const ih = img.naturalHeight || img.height || 1;
    const imgAspect = iw / ih;
    const wpAspect = widthMm / (heightMm || 1);

    let dw = widthMm;
    let dh = heightMm;
    let dx = 0;
    let dy = 0;

    if (imgAspect > wpAspect) {
      dh = widthMm / imgAspect;
      dy = (heightMm - dh) / 2;
    } else {
      dw = heightMm * imgAspect;
      dx = (widthMm - dw) / 2;
    }

    return { x: dx, y: dy, w: dw, h: dh };
  }

  /**
   * Auto-detect whether foreground is bright-on-dark (returns true) or dark-on-light (returns false)
   */
  static detectLightOnDarkBackground(gray, w, h) {
    let sum = 0;
    let count = 0;
    const pad = Math.min(4, Math.floor(Math.min(w, h) / 4));
    for (let x = 0; x < w; x++) {
      sum += gray[pad * w + x] + gray[(h - 1 - pad) * w + x];
      count += 2;
    }
    for (let y = 0; y < h; y++) {
      sum += gray[y * w + pad] + gray[y * w + (w - 1 - pad)];
      count += 2;
    }
    const borderMean = count > 0 ? sum / count : 1.0;
    return borderMean < 0.42;
  }

  /**
   * Fast 3x3 Gaussian blur to suppress wood grain & JPEG compression speckles while preserving edges
   */
  static gaussianBlur3x3(gray, w, h) {
    const out = new Float32Array(w * h);
    out.set(gray);
    for (let y = 1; y < h - 1; y++) {
      const row = y * w;
      for (let x = 1; x < w - 1; x++) {
        const i = row + x;
        out[i] = (
          gray[i - w - 1] + 2 * gray[i - w] + gray[i - w + 1] +
          2 * gray[i - 1] + 4 * gray[i] + 2 * gray[i + 1] +
          gray[i + w - 1] + 2 * gray[i + w] + gray[i + w + 1]
        ) / 16.0;
      }
    }
    return out;
  }

  /**
   * Compute Integral Image (Summed-Area Table) in O(W*H) for instant local neighborhood averages
   */
  static buildIntegralImage(gray, w, h) {
    const sat = new Float64Array((w + 1) * (h + 1));
    const stride = w + 1;
    for (let y = 1; y <= h; y++) {
      let rowSum = 0;
      const gRow = (y - 1) * w;
      const sRow = y * stride;
      const prevRow = (y - 1) * stride;
      for (let x = 1; x <= w; x++) {
        rowSum += gray[gRow + x - 1];
        sat[sRow + x] = sat[prevRow + x] + rowSum;
      }
    }
    return sat;
  }

  static getLocalMean(sat, w, h, x, y, radius) {
    const x0 = Math.max(0, x - radius);
    const y0 = Math.max(0, y - radius);
    const x1 = Math.min(w - 1, x + radius);
    const y1 = Math.min(h - 1, y + radius);
    const stride = w + 1;
    const sum = sat[(y1 + 1) * stride + (x1 + 1)] - sat[y0 * stride + (x1 + 1)] - sat[(y1 + 1) * stride + x0] + sat[y0 * stride + x0];
    const area = (x1 - x0 + 1) * (y1 - y0 + 1);
    return sum / area;
  }

  /**
   * Main Image-to-2D Vector Tracing Engine
   * Features: Border RGB Chroma Distance + Multi-Channel Gradients + Dual-Tone Local Adaptive Thresholding
   * Ensures wood carvings, colored designs, AND monochrome sketches trace faithfully!
   * @param {Object} options - traceMode: 'auto'|'wood'|'lineart'|'edge', detail: 0.1..0.9
   */
  static traceImageToVectors(img, widthMm = 100, heightMm = 100, options = {}) {
    const {
      threshold = -1,         // -1 = Auto
      invert = false,
      autoPolarity = true,
      traceMode = 'auto',     // 'auto', 'wood', 'lineart', 'edge'
      detail = 0.5,           // 0.1 (clean simple) to 0.9 (ultra-fine details)
      simplification = 0.22,
      smoothCurves = true,    // Corner-preserving smoothing for CNC spindle quality
      maxDimension = 720,
      minPoints = 6,
      minSpanMm = 1.4,
      maxContours = 150,
    } = typeof options === 'number' ? { threshold: options } : options;

    const bounds = ImageVectorizer.getImageFitBounds(img, widthMm, heightMm);

    const iw = img.naturalWidth || img.width || 1;
    const ih = img.naturalHeight || img.height || 1;
    let scaleW = maxDimension;
    let scaleH = Math.round((ih / iw) * maxDimension);
    if (scaleH > maxDimension) {
      scaleH = maxDimension;
      scaleW = Math.round((iw / ih) * maxDimension);
    }
    scaleW = Math.max(120, Math.min(960, scaleW));
    scaleH = Math.max(120, Math.min(960, scaleH));

    const canvas = document.createElement('canvas');
    canvas.width = scaleW;
    canvas.height = scaleH;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, scaleW, scaleH);
    ctx.drawImage(img, 0, 0, scaleW, scaleH);

    const pixels = ctx.getImageData(0, 0, scaleW, scaleH).data;
    const N = scaleW * scaleH;
    const rawGray = new Float32Array(N);
    const rCh = new Float32Array(N);
    const gCh = new Float32Array(N);
    const bCh = new Float32Array(N);
    const histogram = new Uint32Array(256);

    for (let i = 0; i < N; i++) {
      const r = pixels[i * 4];
      const g = pixels[i * 4 + 1];
      const b = pixels[i * 4 + 2];
      rCh[i] = r / 255.0;
      gCh[i] = g / 255.0;
      bCh[i] = b / 255.0;
      const l = Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b);
      rawGray[i] = l / 255.0;
      histogram[l]++;
    }

    // Border RGB + Luminance statistics (for chroma separation & auto polarity)
    let bR = 0, bG = 0, bB = 0, bCnt = 0;
    const pad = Math.min(3, Math.floor(Math.min(scaleW, scaleH) / 4));
    for (let x = 0; x < scaleW; x++) {
      const i1 = pad * scaleW + x, i2 = (scaleH - 1 - pad) * scaleW + x;
      bR += rCh[i1] + rCh[i2]; bG += gCh[i1] + gCh[i2]; bB += bCh[i1] + bCh[i2];
      bCnt += 2;
    }
    for (let y = 0; y < scaleH; y++) {
      const i1 = y * scaleW + pad, i2 = y * scaleW + (scaleW - 1 - pad);
      bR += rCh[i1] + rCh[i2]; bG += gCh[i1] + gCh[i2]; bB += bCh[i1] + bCh[i2];
      bCnt += 2;
    }
    bR /= bCnt; bG /= bCnt; bB /= bCnt;

    // Color distance from border RGB (isolates colored floral/wood patterns even when luma is equal!)
    const colorDiff = new Float32Array(N);
    const chromaHist = new Uint32Array(256);
    let maxChroma = 0;
    for (let i = 0; i < N; i++) {
      const d = Math.hypot(rCh[i] - bR, gCh[i] - bG, bCh[i] - bB) / 1.732;
      colorDiff[i] = d;
      if (d > maxChroma) maxChroma = d;
      chromaHist[Math.min(255, Math.round(d * 255))]++;
    }

    // Gentle blur suppresses wood grain & JPEG blocks cleanly
    const smoothed = ImageVectorizer.gaussianBlur3x3(rawGray, scaleW, scaleH);
    const gray = traceMode === 'wood' ? ImageVectorizer.gaussianBlur3x3(smoothed, scaleW, scaleH) : smoothed;

    // Multi-channel RGB Sobel Gradient Magnitude
    const edgeMag = new Float32Array(N);
    for (let y = 1; y < scaleH - 1; y++) {
      for (let x = 1; x < scaleW - 1; x++) {
        const i = y * scaleW + x;
        const gxL = -gray[i - scaleW - 1] + gray[i - scaleW + 1] - 2 * gray[i - 1] + 2 * gray[i + 1] - gray[i + scaleW - 1] + gray[i + scaleW + 1];
        const gyL = -gray[i - scaleW - 1] - 2 * gray[i - scaleW] - gray[i - scaleW + 1] + gray[i + scaleW - 1] + 2 * gray[i + scaleW] + gray[i + scaleW + 1];
        const gL = Math.hypot(gxL, gyL) * 0.5;
        const gxR = -rCh[i - 1] + rCh[i + 1], gyR = -rCh[i - scaleW] + rCh[i + scaleW];
        const gxG = -gCh[i - 1] + gCh[i + 1], gyG = -gCh[i - scaleW] + gCh[i + scaleW];
        const gxB = -bCh[i - 1] + bCh[i + 1], gyB = -bCh[i - scaleW] + bCh[i + scaleW];
        const gColor = Math.max(Math.hypot(gxR, gyR), Math.hypot(gxG, gyG), Math.hypot(gxB, gyB));
        edgeMag[i] = Math.max(gL, gColor * 0.85);
      }
    }

    const otsuStats = ImageVectorizer.computeOtsuStats(histogram, N);
    const chromaOtsu = ImageVectorizer.computeOtsuStats(chromaHist, N);
    const otsuThreshold = threshold > 0 ? threshold : otsuStats.threshold / 255.0;
    const isBimodalLineArt = otsuStats.separability > 0.85;
    const hasUniformColoredBg = chromaOtsu.separability > 0.78 && maxChroma > 0.25 && !isBimodalLineArt;

    let effectiveMode = traceMode;
    if (effectiveMode === 'auto') {
      effectiveMode = isBimodalLineArt ? 'lineart' : 'wood';
    }

    // Scale thresholds cleanly by `detail` (0.1..0.9 -> default 0.5)
    const detailFactor = Math.max(0.1, Math.min(0.9, detail));
    const binary = new Uint8Array(N);

    if (effectiveMode === 'edge') {
      const edgeThres = threshold > 0 ? threshold * 0.4 : (0.24 - detailFactor * 0.14);
      for (let i = 0; i < N; i++) {
        binary[i] = edgeMag[i] >= edgeThres ? 1 : 0;
      }
    } else if (effectiveMode === 'wood') {
      // Wood Relief + Textured Photo Mode:
      // Captures BOTH dark carved valleys (via local integral mean) AND lit carved bevel edges (Sobel gradient),
      // plus uniform colored backgrounds when chroma separation is clear!
      const sat = ImageVectorizer.buildIntegralImage(gray, scaleW, scaleH);
      const localRadius = Math.max(10, Math.round(Math.min(scaleW, scaleH) * 0.04));
      const baseSens = threshold > 0 ? Math.max(0.02, (1.0 - threshold) * 0.14) : (0.095 - detailFactor * 0.065);
      const edgeGate = Math.max(0.10, 0.30 - detailFactor * 0.18);
      const chromaGate = Math.max(0.08, (chromaOtsu.threshold / 255.0) * (1.25 - detailFactor * 0.45));

      const isLightOnDark = autoPolarity ? ImageVectorizer.detectLightOnDarkBackground(gray, scaleW, scaleH) : false;
      const flip = invert ? !isLightOnDark : isLightOnDark;

      for (let y = 1; y < scaleH - 1; y++) {
        for (let x = 1; x < scaleW - 1; x++) {
          const idx = y * scaleW + x;
          const localMean = ImageVectorizer.getLocalMean(sat, scaleW, scaleH, x, y, localRadius);
          const localFeature = flip
            ? (gray[idx] > localMean * (1.0 + baseSens) || gray[idx] > otsuThreshold + 0.14)
            : (gray[idx] < localMean * (1.0 - baseSens) || gray[idx] < otsuThreshold - 0.14);
          const chromaFeature = hasUniformColoredBg && (colorDiff[idx] > chromaGate);
          const edgeFeature = edgeMag[idx] > edgeGate;

          binary[idx] = (localFeature || chromaFeature || edgeFeature) ? 1 : 0;
        }
      }
    } else {
      // Crisp Line Art / Ink / B&W Scan Mode
      const isLightOnDark = autoPolarity ? ImageVectorizer.detectLightOnDarkBackground(gray, scaleW, scaleH) : false;
      const flip = invert ? !isLightOnDark : isLightOnDark;
      const adjThresh = threshold > 0 ? threshold : Math.max(0.08, Math.min(0.92, otsuThreshold + (detailFactor - 0.5) * 0.15));
      for (let i = 0; i < N; i++) {
        binary[i] = (flip ? gray[i] > adjThresh : gray[i] < adjThresh) ? 1 : 0;
      }
    }

    // Morphological Close (connect tiny 1px breaks) then Clean noise
    const closed = ImageVectorizer.morphCloseAndClean(binary, scaleW, scaleH);

    // Check foreground density: if > 78% of image is foreground, polarity was likely inverted -> flip safely!
    let fgCount = 0;
    for (let i = 0; i < closed.length; i++) if (closed[i]) fgCount++;
    if (fgCount > closed.length * 0.78 && effectiveMode !== 'edge') {
      for (let i = 0; i < closed.length; i++) closed[i] = closed[i] ? 0 : 1;
    }

    return ImageVectorizer.extractContoursFromBinary(closed, scaleW, scaleH, bounds, {
      simplification,
      smoothCurves,
      minPoints,
      minSpanMm,
      maxContours,
      source: 'auto-trace',
      namePrefix: 'Design Layer',
      rejectFullFrame: true,
    });
  }

  /**
   * 3x3 Morphological Closing (bridges 1px hairline breaks) + isolated noise cleanup
   */
  static morphCloseAndClean(binary, w, h) {
    // 1. Light 4-neighbor dilation to bridge 1px hairline gaps
    const dilated = new Uint8Array(w * h);
    for (let y = 1; y < h - 1; y++) {
      const row = y * w;
      for (let x = 1; x < w - 1; x++) {
        const i = row + x;
        if (binary[i]) {
          // Keep pixel only if it has at least 1 neighbor (kills single isolated grain dots immediately!)
          if (binary[i - 1] || binary[i + 1] || binary[i - w] || binary[i + w] ||
              binary[i - w - 1] || binary[i - w + 1] || binary[i + w - 1] || binary[i + w + 1]) {
            dilated[i] = 1;
          }
        } else {
          // Fill tiny 1px pinholes inside lines/shapes
          const nCount = binary[i - 1] + binary[i + 1] + binary[i - w] + binary[i + w];
          if (nCount >= 3) dilated[i] = 1;
        }
      }
    }
    return dilated;
  }

  /**
   * Convert 3D Relief Heightmap into separate editable 2D Vector Contour Layers
   */
  static traceReliefToVectors(reliefEngine, options = {}) {
    const {
      levels = [0.15, 0.45, 0.75],
      simplification = 0.25,
      smoothCurves = true,
      minPoints = 6,
      minSpanMm = 1.5,
      maxContours = 120,
    } = options;

    const N = reliefEngine.resolution;
    const bounds = { x: 0, y: 0, w: reliefEngine.width, h: reliefEngine.height };
    const allEntities = [];
    const colors = ['#38bdf8', '#34d399', '#f472b6', '#fbbf24'];

    levels.forEach((lvl, idx) => {
      const binary = new Uint8Array(N * N);
      for (let i = 0; i < N * N; i++) {
        binary[i] = reliefEngine.data[i] >= lvl ? 1 : 0;
      }

      const contours = ImageVectorizer.extractContoursFromBinary(binary, N, N, bounds, {
        simplification,
        smoothCurves,
        minPoints,
        minSpanMm,
        maxContours: Math.floor(maxContours / levels.length),
        source: '3d-extract',
        namePrefix: `3D Slice (${Math.round(lvl * 100)}%)`,
        color: colors[idx % colors.length],
        defaultHeightMm: Number((lvl * (reliefEngine.maxDepth || 8)).toFixed(1)),
        rejectFullFrame: false,
      });
      allEntities.push(...contours);
    });

    return allEntities;
  }

  /**
   * Binary mask -> Moore-Neighbor Contours -> RDP -> Corner-Preserving Curve Smoothing
   */
  static extractContoursFromBinary(binary, w, h, bounds, opts = {}) {
    const {
      simplification = 0.25,
      smoothCurves = true,
      minPoints = 6,
      minSpanMm = 1.5,
      maxContours = 140,
      source = 'auto-trace',
      namePrefix = 'Contour',
      color = null,
      defaultHeightMm = 4.0,
      rejectFullFrame = true,
    } = opts;

    const layerPalette = ['#38bdf8', '#34d399', '#fbbf24', '#f472b6', '#a78bfa', '#22d3ee', '#fb923c'];

    const visitedBorders = new Uint8Array(w * h);
    const rawContours = [];
    const dxTable = [1, 1, 0, -1, -1, -1, 0, 1];
    const dyTable = [0, 1, 1, 1, 0, -1, -1, -1];

    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = y * w + x;
        if (binary[idx] === 1 && !visitedBorders[idx]) {
          const hasEmptyNeighbor =
            binary[idx - 1] === 0 || binary[idx + 1] === 0 ||
            binary[idx - w] === 0 || binary[idx + w] === 0;

          if (hasEmptyNeighbor) {
            const contour = ImageVectorizer.traceMooreContour(binary, visitedBorders, w, h, x, y, dxTable, dyTable);
            if (contour && contour.length >= minPoints) {
              rawContours.push(contour);
            }
          }
        }
      }
    }

    const candidates = [];
    const scaleFactorX = bounds.w / w;
    const scaleFactorY = bounds.h / h;

    for (const rawContour of rawContours) {
      const mmPoints = rawContour.map((p) => ({
        x: Number((bounds.x + p.x * scaleFactorX).toFixed(3)),
        y: Number((bounds.y + p.y * scaleFactorY).toFixed(3)),
      }));

      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const p of mmPoints) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
      }
      const spanW = maxX - minX;
      const spanH = maxY - minY;
      if (spanW < minSpanMm && spanH < minSpanMm) continue;
      if (rejectFullFrame && spanW > bounds.w * 0.94 && spanH > bounds.h * 0.94) continue;

      let simplified = ImageVectorizer.simplifyRDP(mmPoints, Math.max(0.14, simplification));

      // Corner-preserving smooth pass removes jagged pixel staircase steps for clean CNC arcs
      if (smoothCurves && simplified.length >= 6) {
        simplified = ImageVectorizer.cornerPreservingSmooth(simplified, true);
      }

      if (simplified.length >= 3) {
        const first = simplified[0];
        const last = simplified[simplified.length - 1];
        if (Math.hypot(first.x - last.x, first.y - last.y) > 0.05) {
          simplified.push({ x: first.x, y: first.y });
        }

        const areaEstimate = spanW * spanH;
        candidates.push({
          areaEstimate,
          entity: {
            type: 'curve',
            points: simplified,
            isClosed: true,
            width: 1.5,
            source,
            visible: true,
            locked: false,
            relief: {
              enabled: true,
              shape: 'dome',
              heightMm: defaultHeightMm,
              radiusMm: Math.max(2.0, Math.min(12.0, Math.min(spanW, spanH) * 0.45)),
              mode: 'max',
              openAsStroke: false,
            },
          },
        });
      }
    }

    candidates.sort((a, b) => b.areaEstimate - a.areaEstimate);

    const limited = candidates.slice(0, maxContours);
    return limited.map((c, i) => {
      c.entity.name = `${namePrefix} #${i + 1}`;
      c.entity.color = color || layerPalette[i % layerPalette.length];
      return c.entity;
    });
  }

  /**
   * Corner-preserving Laplacian smoothing: keeps sharp corners (<110 deg) crisp while smoothing stairsteps
   */
  static cornerPreservingSmooth(pts, isClosed = true) {
    const n = pts.length;
    if (n < 4) return pts;
    const out = [];
    for (let i = 0; i < n; i++) {
      const prev = pts[(i - 1 + n) % n];
      const curr = pts[i];
      const next = pts[(i + 1) % n];

      if (!isClosed && (i === 0 || i === n - 1)) {
        out.push({ ...curr });
        continue;
      }

      const v1x = prev.x - curr.x, v1y = prev.y - curr.y;
      const v2x = next.x - curr.x, v2y = next.y - curr.y;
      const len1 = Math.hypot(v1x, v1y);
      const len2 = Math.hypot(v2x, v2y);

      if (len1 > 0.001 && len2 > 0.001) {
        const cosAng = Math.max(-1, Math.min(1, (v1x * v2x + v1y * v2y) / (len1 * len2)));
        const angleDeg = Math.acos(cosAng) * (180 / Math.PI);
        // If sharp corner (< 115 deg), preserve vertex intact
        if (angleDeg < 115) {
          out.push({ ...curr });
          continue;
        }
      }

      out.push({
        x: Number((prev.x * 0.2 + curr.x * 0.6 + next.x * 0.2).toFixed(3)),
        y: Number((prev.y * 0.2 + curr.y * 0.6 + next.y * 0.2).toFixed(3)),
      });
    }
    return out;
  }

  /**
   * Moore-Neighbor Boundary Tracing Algorithm (supports intricate wooden door frames up to 80k steps)
   */
  static traceMooreContour(matrix, visited, w, h, startX, startY, dxTable, dyTable) {
    const contour = [];
    let currX = startX;
    let currY = startY;
    let backtrackDir = 4;

    contour.push({ x: currX, y: currY });
    visited[currY * w + currX] = 1;

    let steps = 0;
    const maxSteps = 80000;

    while (steps < maxSteps) {
      steps++;
      let foundNext = false;
      const startCheck = (backtrackDir + 1) % 8;

      for (let offset = 0; offset < 8; offset++) {
        const dir = (startCheck + offset) % 8;
        const nx = currX + dxTable[dir];
        const ny = currY + dyTable[dir];

        if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
          if (matrix[ny * w + nx] === 1) {
            currX = nx;
            currY = ny;
            visited[currY * w + currX] = 1;
            contour.push({ x: currX, y: currY });
            backtrackDir = (dir + 4) % 8;
            foundNext = true;
            break;
          }
        }
      }

      if (!foundNext) break;
      if (currX === startX && currY === startY && contour.length > 3) {
        break;
      }
    }

    return contour;
  }

  /**
   * Otsu threshold + Histogram Separability Metric (detects line art vs textured photo)
   */
  static computeOtsuStats(histogram, totalPixels) {
    let sum = 0;
    let sumSq = 0;
    for (let t = 0; t < 256; t++) {
      sum += t * histogram[t];
      sumSq += t * t * histogram[t];
    }

    const totalMean = sum / (totalPixels || 1);
    const totalVariance = Math.max(1e-5, (sumSq / (totalPixels || 1)) - totalMean * totalMean);

    let sumB = 0;
    let wB = 0;
    let maxVariance = 0;
    let threshold = 128;

    for (let t = 0; t < 256; t++) {
      wB += histogram[t];
      if (wB === 0) continue;
      const wF = totalPixels - wB;
      if (wF === 0) break;

      sumB += t * histogram[t];
      const mB = sumB / wB;
      const mF = (sum - sumB) / wF;

      const betweenClassVariance = (wB / totalPixels) * (wF / totalPixels) * (mB - mF) * (mB - mF);
      if (betweenClassVariance > maxVariance) {
        maxVariance = betweenClassVariance;
        threshold = t;
      }
    }

    return {
      threshold,
      separability: Math.min(1.0, maxVariance / totalVariance),
    };
  }

  static computeOtsuThreshold(histogram, totalPixels) {
    return ImageVectorizer.computeOtsuStats(histogram, totalPixels).threshold;
  }

  static simplifyRDP(points, epsilon) {
    if (points.length <= 2) return points;

    let maxDist = 0;
    let index = 0;
    const end = points.length - 1;

    for (let i = 1; i < end; i++) {
      const d = ImageVectorizer.perpendicularDistance(points[i], points[0], points[end]);
      if (d > maxDist) {
        maxDist = d;
        index = i;
      }
    }

    if (maxDist > epsilon) {
      const rec1 = ImageVectorizer.simplifyRDP(points.slice(0, index + 1), epsilon);
      const rec2 = ImageVectorizer.simplifyRDP(points.slice(index), epsilon);
      return rec1.slice(0, rec1.length - 1).concat(rec2);
    } else {
      return [points[0], points[end]];
    }
  }

  static perpendicularDistance(pt, lineStart, lineEnd) {
    const dx = lineEnd.x - lineStart.x;
    const dy = lineEnd.y - lineStart.y;
    const mag = Math.hypot(dx, dy);
    if (mag === 0) return Math.hypot(pt.x - lineStart.x, pt.y - lineStart.y);
    return Math.abs(dy * pt.x - dx * pt.y + lineEnd.x * lineStart.y - lineEnd.y * lineStart.x) / mag;
  }
}
