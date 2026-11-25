// Adobe Illustrator-Grade Vector Image Trace & Potrace/CIELAB Layer Engine for JDPaint 5.21
// Clean-room synthesis from Potrace (DP optimal polygon, subpixel adjustment, alpha-corner & opticurve)
// and multi-band CIELAB quantization (stacked relief layers + abutting color cutout layers).

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

  // ============================================================
  // 1. O(W*H) Image Filtering & Perceptual Color Space Helpers
  // ============================================================

  /**
   * Separable O(W*H) Sliding Box Mean using Float64 prefix sums.
   */
  static boxMean(src, w, h, radius) {
    const r = Math.max(1, Math.round(radius));
    const n = w * h;
    const horiz = new Float32Array(n);
    const out = new Float32Array(n);
    const pref = new Float64Array(w + 1);

    for (let y = 0; y < h; y++) {
      const row = y * w;
      pref[0] = 0;
      for (let x = 0; x < w; x++) {
        pref[x + 1] = pref[x] + src[row + x];
      }
      for (let x = 0; x < w; x++) {
        const x0 = x > r ? x - r : 0;
        const x1 = x + r + 1 < w ? x + r + 1 : w;
        horiz[row + x] = (pref[x1] - pref[x0]) / (x1 - x0);
      }
    }

    const colSum = new Float64Array(w);
    const initEnd = Math.min(r, h - 1);
    for (let y = 0; y <= initEnd; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) colSum[x] += horiz[row + x];
    }

    for (let y = 0; y < h; y++) {
      const y0 = y > r ? y - r : 0;
      const y1 = y + r + 1 < h ? y + r + 1 : h;
      const inv = 1.0 / (y1 - y0);
      const row = y * w;
      for (let x = 0; x < w; x++) {
        out[row + x] = colSum[x] * inv;
      }
      const addY = y + r + 1;
      if (addY < h) {
        const aRow = addY * w;
        for (let x = 0; x < w; x++) colSum[x] += horiz[aRow + x];
      }
      const subY = y - r;
      if (subY >= 0) {
        const sRow = subY * w;
        for (let x = 0; x < w; x++) colSum[x] -= horiz[sRow + x];
      }
    }
    return out;
  }

  /**
   * Edge-preserving Fast Guided Filter: suppresses wood grain & JPEG blocks while preserving crisp boundaries.
   */
  static guidedFilter(gray, w, h, radius = 3, eps = 180) {
    const n = w * h;
    const sq = new Float32Array(n);
    for (let i = 0; i < n; i++) sq[i] = gray[i] * gray[i];
    const meanI = ImageVectorizer.boxMean(gray, w, h, radius);
    const corrII = ImageVectorizer.boxMean(sq, w, h, radius);
    const a = new Float32Array(n);
    const b = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const variance = Math.max(0, corrII[i] - meanI[i] * meanI[i]);
      const ai = variance / (variance + eps);
      a[i] = ai;
      b[i] = meanI[i] * (1 - ai);
    }
    const meanA = ImageVectorizer.boxMean(a, w, h, radius);
    const meanB = ImageVectorizer.boxMean(b, w, h, radius);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      out[i] = meanA[i] * gray[i] + meanB[i];
    }
    return out;
  }

  /**
   * Guided RGB denoising per channel (sharp color edges, smooth wood/photo noise).
   */
  static guidedFilterRGB(rArr, gArr, bArr, lumaArr, w, h, radius = 2, eps = 160) {
    const n = w * h;
    const meanI = ImageVectorizer.boxMean(lumaArr, w, h, radius);
    const ii = new Float32Array(n);
    for (let i = 0; i < n; i++) ii[i] = lumaArr[i] * lumaArr[i];
    const varI = ImageVectorizer.boxMean(ii, w, h, radius);
    for (let i = 0; i < n; i++) varI[i] = Math.max(0, varI[i] - meanI[i] * meanI[i]);

    const filterChannel = (pArr) => {
      const ip = new Float32Array(n);
      for (let i = 0; i < n; i++) ip[i] = lumaArr[i] * pArr[i];
      const meanP = ImageVectorizer.boxMean(pArr, w, h, radius);
      const corrIP = ImageVectorizer.boxMean(ip, w, h, radius);
      const a = new Float32Array(n);
      const b = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const cov = corrIP[i] - meanI[i] * meanP[i];
        const ai = cov / (varI[i] + eps);
        a[i] = ai;
        b[i] = meanP[i] - ai * meanI[i];
      }
      const mA = ImageVectorizer.boxMean(a, w, h, radius);
      const mB = ImageVectorizer.boxMean(b, w, h, radius);
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        out[i] = Math.max(0, Math.min(255, mA[i] * lumaArr[i] + mB[i]));
      }
      return out;
    };

    return {
      r: filterChannel(rArr),
      g: filterChannel(gArr),
      b: filterChannel(bArr),
    };
  }

  /**
   * Clamped iterative Retinex local lighting equalization (cancels uneven photo shadows on wood/reliefs).
   */
  static correctLocalLighting(gray, w, h, radius) {
    const n = w * h;
    let bg = ImageVectorizer.boxMean(gray, w, h, radius);
    for (let it = 0; it < 2; it++) {
      const clamped = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        clamped[i] = Math.max(bg[i] - 26, Math.min(bg[i] + 26, gray[i]));
      }
      bg = ImageVectorizer.boxMean(clamped, w, h, radius);
    }
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      out[i] = Math.max(0, Math.min(255, 128 + (gray[i] - bg[i]) * 1.45));
    }
    return out;
  }

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
    return sum / ((x1 - x0 + 1) * (y1 - y0 + 1));
  }

  static rgbToLab(r, g, b) {
    let sr = r / 255.0;
    let sg = g / 255.0;
    let sb = b / 255.0;
    sr = sr > 0.04045 ? Math.pow((sr + 0.055) / 1.055, 2.4) : sr / 12.92;
    sg = sg > 0.04045 ? Math.pow((sg + 0.055) / 1.055, 2.4) : sg / 12.92;
    sb = sb > 0.04045 ? Math.pow((sb + 0.055) / 1.055, 2.4) : sb / 12.92;

    let X = (sr * 0.4124 + sg * 0.3576 + sb * 0.1805) / 0.95047;
    let Y = (sr * 0.2126 + sg * 0.7152 + sb * 0.0722);
    let Z = (sr * 0.0193 + sg * 0.1192 + sb * 0.9505) / 1.08883;

    const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
    X = f(X); Y = f(Y); Z = f(Z);
    return [116 * Y - 16, 500 * (X - Y), 200 * (Y - Z)];
  }

  static rgbToHex(r, g, b) {
    const c = (v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0');
    return `#${c(r)}${c(g)}${c(b)}`;
  }

  /**
   * Deterministic CIELAB K-Means++ Color Quantization (analogous to Illustrator Color Image Trace).
   */
  static quantizeColorsLAB(rArr, gArr, bArr, w, h, kCount = 6) {
    const n = w * h;
    const K = Math.max(2, Math.min(24, Math.round(kCount)));
    const maxSamples = 12000;
    const step = Math.max(1, Math.floor(n / maxSamples));

    const samples = [];
    for (let i = 0; i < n; i += step) {
      const r = rArr[i], g = gArr[i], b = bArr[i];
      const [L, a, lb] = ImageVectorizer.rgbToLab(r, g, b);
      samples.push({ L, a, lb, r, g, b });
    }
    if (samples.length === 0) return { labels: new Uint8Array(n), palette: [] };

    // Deterministic Farthest-Point Initialization (reproducible across runs!)
    let seedIdx = 0;
    let maxDistCenter = -1;
    for (let i = 0; i < samples.length; i++) {
      const s = samples[i];
      const d = s.L * s.L + s.a * s.a + s.lb * s.lb;
      if (d > maxDistCenter) { maxDistCenter = d; seedIdx = i; }
    }

    const centroids = [{ ...samples[seedIdx] }];
    const minDistSq = new Float64Array(samples.length).fill(Infinity);

    for (let c = 1; c < K; c++) {
      const last = centroids[c - 1];
      let bestIdx = 0;
      let bestD = -1;
      for (let i = 0; i < samples.length; i++) {
        const s = samples[i];
        const d2 = (s.L - last.L) ** 2 + (s.a - last.a) ** 2 + (s.lb - last.lb) ** 2;
        if (d2 < minDistSq[i]) minDistSq[i] = d2;
        if (minDistSq[i] > bestD) { bestD = minDistSq[i]; bestIdx = i; }
      }
      centroids.push({ ...samples[bestIdx] });
    }

    // 10 Lloyd iterations on CIELAB samples
    const sLabels = new Int32Array(samples.length);
    for (let iter = 0; iter < 10; iter++) {
      const sumL = new Float64Array(K), sumA = new Float64Array(K), sumB = new Float64Array(K);
      const sumR = new Float64Array(K), sumG = new Float64Array(K), sumBl = new Float64Array(K);
      const cnt = new Int32Array(K);

      for (let i = 0; i < samples.length; i++) {
        const s = samples[i];
        let bestK = 0, bestD = Infinity;
        for (let c = 0; c < K; c++) {
          const ct = centroids[c];
          const d = (s.L - ct.L) ** 2 + (s.a - ct.a) ** 2 + (s.lb - ct.lb) ** 2;
          if (d < bestD) { bestD = d; bestK = c; }
        }
        sLabels[i] = bestK;
        sumL[bestK] += s.L; sumA[bestK] += s.a; sumB[bestK] += s.lb;
        sumR[bestK] += s.r; sumG[bestK] += s.g; sumBl[bestK] += s.b;
        cnt[bestK]++;
      }

      for (let c = 0; c < K; c++) {
        if (cnt[c] > 0) {
          centroids[c].L = sumL[c] / cnt[c];
          centroids[c].a = sumA[c] / cnt[c];
          centroids[c].lb = sumB[c] / cnt[c];
          centroids[c].r = sumR[c] / cnt[c];
          centroids[c].g = sumG[c] / cnt[c];
          centroids[c].b = sumBl[c] / cnt[c];
        }
      }
    }

    // Merge perceptually indistinguishable clusters (CIELAB ΔE < 11) before pixel assignment
    const remap = new Int32Array(K);
    for (let c = 0; c < K; c++) remap[c] = c;
    for (let a = 0; a < K; a++) {
      if (remap[a] !== a) continue;
      for (let b = a + 1; b < K; b++) {
        if (remap[b] !== b) continue;
        const ca = centroids[a], cb = centroids[b];
        const dE2 = (ca.L - cb.L) ** 2 + (ca.a - cb.a) ** 2 + (ca.lb - cb.lb) ** 2;
        if (dE2 < 120) remap[b] = a;
      }
    }

    // Assign full image pixels in perceptual Lab lookup over merged centroids
    const rawLabels = new Uint8Array(n);
    const counts = new Uint32Array(K);
    for (let i = 0; i < n; i++) {
      const r = rArr[i], g = gArr[i], b = bArr[i];
      const [L, a, lb] = ImageVectorizer.rgbToLab(r, g, b);
      let bestK = 0, bestD = Infinity;
      for (let c = 0; c < K; c++) {
        if (remap[c] !== c) continue;
        const ct = centroids[c];
        const d = (L - ct.L) ** 2 + (a - ct.a) ** 2 + (lb - ct.lb) ** 2;
        if (d < bestD) { bestD = d; bestK = c; }
      }
      rawLabels[i] = bestK;
      counts[bestK]++;
    }

    // 3x3 modal filter kills anti-alias fringe specks between adjacent color zones
    const labels = new Uint8Array(rawLabels);
    const hist = new Uint8Array(K);
    for (let y = 1; y < h - 1; y++) {
      const row = y * w;
      for (let x = 1; x < w - 1; x++) {
        const idx = row + x;
        const c0 = rawLabels[idx];
        let sameCount = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const nRow = (y + dy) * w + x;
          if (rawLabels[nRow - 1] === c0) sameCount++;
          if (rawLabels[nRow] === c0) sameCount++;
          if (rawLabels[nRow + 1] === c0) sameCount++;
        }
        if (sameCount <= 2) {
          hist.fill(0);
          let bestK = c0, bestCnt = 0;
          for (let dy = -1; dy <= 1; dy++) {
            const nRow = (y + dy) * w + x;
            for (let dx = -1; dx <= 1; dx++) {
              const lb = rawLabels[nRow + dx];
              const v = ++hist[lb];
              if (v > bestCnt) { bestCnt = v; bestK = lb; }
            }
          }
          labels[idx] = bestK;
        }
      }
    }

    // Recompute exact palette from final labels + border coverage
    const fR = new Float64Array(K), fG = new Float64Array(K), fB = new Float64Array(K);
    const fCnt = new Uint32Array(K), borderCnt = new Uint32Array(K);
    for (let i = 0; i < n; i++) {
      const k = labels[i];
      fR[k] += rArr[i]; fG[k] += gArr[i]; fB[k] += bArr[i];
      fCnt[k]++;
    }
    const pad = 2;
    for (let x = 0; x < w; x++) {
      borderCnt[labels[pad * w + x]]++;
      borderCnt[labels[(h - 1 - pad) * w + x]]++;
    }
    for (let y = 0; y < h; y++) {
      borderCnt[labels[y * w + pad]]++;
      borderCnt[labels[y * w + (w - 1 - pad)]]++;
    }
    const totalBorder = 2 * w + 2 * h;

    const palette = centroids.map((ct, idx) => {
      const count = fCnt[idx];
      const r = count > 0 ? fR[idx] / count : ct.r;
      const g = count > 0 ? fG[idx] / count : ct.g;
      const b = count > 0 ? fB[idx] / count : ct.b;
      const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      const chroma = Math.max(Math.abs(r - g), Math.abs(g - b), Math.abs(r - b));
      return {
        index: idx,
        r, g, b,
        hex: ImageVectorizer.rgbToHex(r, g, b),
        luma,
        count,
        borderFraction: totalBorder > 0 ? borderCnt[idx] / totalBorder : 0,
        isWhitePaper: luma > 236 && chroma < 22,
      };
    });

    return { labels, palette };
  }

  // ============================================================
  // 2. Main Unified Adobe Illustrator Image Trace Entry Points
  // ============================================================

  /**
   * Trace HTMLImageElement / Canvas into exact compound multi-layer 2D CAD vector entities.
   */
  static traceImageToVectors(img, widthMm = 100, heightMm = 100, options = {}) {
    const iw = img.naturalWidth || img.width || 1;
    const ih = img.naturalHeight || img.height || 1;
    const maxDimension = (typeof options === 'object' && options.maxDimension) || 900;

    let scaleW = maxDimension;
    let scaleH = Math.round((ih / iw) * maxDimension);
    if (scaleH > maxDimension) {
      scaleH = maxDimension;
      scaleW = Math.round((iw / ih) * maxDimension);
    }
    scaleW = Math.max(120, Math.min(1024, scaleW));
    scaleH = Math.max(120, Math.min(1024, scaleH));

    const canvas = document.createElement('canvas');
    canvas.width = scaleW;
    canvas.height = scaleH;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingQuality = 'high';
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, scaleW, scaleH);
    ctx.drawImage(img, 0, 0, scaleW, scaleH);

    const pixels = ctx.getImageData(0, 0, scaleW, scaleH).data;
    const bounds = ImageVectorizer.getImageFitBounds(img, widthMm, heightMm);
    return ImageVectorizer.traceFromImageData(pixels, scaleW, scaleH, bounds, options);
  }

  /**
   * Pure-JS Illustrator Image Trace from raw RGBA buffer (runs seamlessly in browser AND Node.js tests!).
   */
  static traceFromImageData(pixels, scaleW, scaleH, bounds, options = {}) {
    const opts = typeof options === 'number' ? { threshold: options } : options;
    const {
      threshold = -1,            // -1 = Auto Otsu/Adaptive
      invert = false,
      autoPolarity = true,
      traceMode = 'auto',        // 'auto' | 'color' | 'grayscale' | 'bw-logo' | 'lineart' | 'wood' | 'edge'
      detail = 0.55,             // 0.1..0.95 (Path tightness / detail retention)
      corners = 0.60,            // 0.1..0.95 (Corner sharpness threshold)
      colorCount = 6,            // 2..16 for 'color' and 'grayscale' modes
      stackMethod = 'stacked',   // 'stacked' (overlapping relief-safe) | 'abutting' (cutout mosaic)
      ignoreBg = true,           // Exclude dominant outer background
      localLighting = true,      // Compensate shadows on textured photos / wood
      noisePx = 12,              // Ignore tiny dust speckles < noisePx px^2
      minSpanMm = 0.9,
      maxContours = 240,
      maxDepthMm = 8.0,
    } = opts;

    const N = scaleW * scaleH;
    const rArr = new Float32Array(N);
    const gArr = new Float32Array(N);
    const bArr = new Float32Array(N);
    const lumaRaw = new Float32Array(N);
    const histogram = new Uint32Array(256);

    let sumChroma = 0;
    let sumRg = 0, sumGb = 0, sumRgSq = 0, sumGbSq = 0;
    let extremeCount = 0;
    for (let i = 0; i < N; i++) {
      const r = pixels[i * 4];
      const g = pixels[i * 4 + 1];
      const b = pixels[i * 4 + 2];
      rArr[i] = r;
      gArr[i] = g;
      bArr[i] = b;
      const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      lumaRaw[i] = l;
      const li = Math.max(0, Math.min(255, Math.round(l)));
      histogram[li]++;
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      sumChroma += mx - mn;
      const rg = (r - g) / 255.0;
      const gb = (g - b) / 255.0;
      sumRg += rg; sumGb += gb;
      sumRgSq += rg * rg; sumGbSq += gb * gb;
      if (l < 42 || l > 213) extremeCount++;
    }

    const otsu = ImageVectorizer.computeOtsuStats(histogram, N);
    const invN = 1.0 / Math.max(1, N);
    const meanChroma = sumChroma * invN / 255.0;
    const varRg = Math.max(0, sumRgSq * invN - (sumRg * invN) ** 2);
    const varGb = Math.max(0, sumGbSq * invN - (sumGb * invN) ** 2);
    const chromaSpread = Math.sqrt(varRg + varGb);
    const extremeRatio = extremeCount * invN;

    // Resolve smart auto preset just like Adobe Illustrator + Monochrome Wood Detection
    let effectiveMode = traceMode;
    if (effectiveMode === 'auto') {
      if (otsu.separability >= 0.78 && extremeRatio > 0.72 && chromaSpread < 0.12) {
        effectiveMode = 'bw-logo';
      } else if (chromaSpread >= 0.16) {
        effectiveMode = 'color';
      } else if (meanChroma >= 0.14) {
        effectiveMode = 'wood';
      } else if (otsu.separability >= 0.76) {
        effectiveMode = 'bw-logo';
      } else {
        effectiveMode = 'grayscale';
      }
    }
    if (effectiveMode === 'lineart') effectiveMode = 'bw-logo';

    const detailFactor = Math.max(0.1, Math.min(0.95, detail));
    const scaleRatio = Math.max(1.0, Math.min(2.4, Math.max(scaleW, scaleH) / 420.0));
    const pxPerMm = Math.max(scaleW / Math.max(1, bounds.w), scaleH / Math.max(1, bounds.h));

    // Smooth edge-preserving Guided Denoise
    const filterRadius = Math.max(1, Math.round((1.15 - detailFactor) * 3.5));
    const denoisedLuma = ImageVectorizer.guidedFilter(lumaRaw, scaleW, scaleH, filterRadius, effectiveMode === 'bw-logo' ? 85 : 185);

    const commonFitOpts = {
      detailFactor,
      corners: Math.max(0.1, Math.min(0.95, corners)),
      minAreaPx: Math.max(4, noisePx * (1.25 - detailFactor * 0.65)),
      minSpanMm,
      maxContours,
      scaleRatio,
      pxPerMm,
      bounds,
      scaleW,
      scaleH,
      maxDepthMm,
    };

    // Dispatch to dedicated Illustrator Preset Pipeline
    if (effectiveMode === 'color') {
      return ImageVectorizer.traceColorLayers(rArr, gArr, bArr, lumaRaw, denoisedLuma, scaleW, scaleH, {
        ...commonFitOpts,
        colorCount,
        stackMethod,
        ignoreBg,
        invert,
      });
    }

    if (effectiveMode === 'grayscale') {
      return ImageVectorizer.traceGrayscaleLayers(lumaRaw, denoisedLuma, scaleW, scaleH, {
        ...commonFitOpts,
        toneLevels: Math.max(2, Math.min(12, colorCount)),
        ignoreBg,
        invert,
        localLighting,
      });
    }

    if (effectiveMode === 'edge') {
      return ImageVectorizer.traceEdgeGrooves(rArr, gArr, bArr, lumaRaw, denoisedLuma, scaleW, scaleH, {
        ...commonFitOpts,
        threshold,
      });
    }

    if (effectiveMode === 'wood') {
      return ImageVectorizer.traceWoodReliefLayers(rArr, gArr, bArr, lumaRaw, denoisedLuma, otsu, scaleW, scaleH, {
        ...commonFitOpts,
        threshold,
        invert,
        autoPolarity,
        ignoreBg,
        localLighting,
        toneLevels: Math.max(2, Math.min(8, Math.round(colorCount))),
      });
    }

    // Default / 'bw-logo': Sub-pixel Accurate Exact Corners + Smooth Cubic Bézier Logo/Vector Trace
    return ImageVectorizer.traceBinaryExact(lumaRaw, denoisedLuma, otsu, scaleW, scaleH, {
      ...commonFitOpts,
      threshold,
      invert,
      autoPolarity,
      localLighting: extremeRatio < 0.78 && localLighting,
    });
  }

  // ============================================================
  // 3. Illustrator Trace Presets (B&W, Color, Grayscale, Wood, Edge)
  // ============================================================

  /**
   * Preset A: Crisp Black & White Logo / Drawing with Sub-Pixel Edge Refinement & Compound Holes.
   */
  static traceBinaryExact(lumaRaw, denoisedLuma, otsu, w, h, opts) {
    const n = w * h;
    let field = denoisedLuma;
    if (opts.localLighting) {
      const illumR = Math.max(10, Math.round(Math.min(w, h) * 0.14));
      field = ImageVectorizer.correctLocalLighting(denoisedLuma, w, h, illumR);
    }

    const baseThresh = opts.threshold > 0
      ? opts.threshold * 255
      : Math.max(18, Math.min(238, otsu.threshold + (opts.detailFactor - 0.5) * 28));

    let darkCount = 0;
    for (let i = 0; i < n; i++) if (field[i] < baseThresh) darkCount++;
    const borderMean = ImageVectorizer.computeBorderMean(field, w, h);
    let traceLight = opts.autoPolarity ? (borderMean < baseThresh) : false;
    if (opts.autoPolarity && !traceLight && darkCount > n * 0.72) traceLight = true;
    if (opts.invert) traceLight = !traceLight;

    const scalar = new Float32Array(n);
    for (let i = 0; i < n; i++) scalar[i] = traceLight ? -field[i] : field[i];
    const th = traceLight ? -baseThresh : baseThresh;

    let mask = new Uint8Array(n);
    for (let i = 0; i < n; i++) mask[i] = scalar[i] < th ? 1 : 0;
    mask = ImageVectorizer.morphClose(mask, w, h, opts.detailFactor < 0.4 ? 1 : 0);

    const loops = ImageVectorizer.marchSquares(mask, w, h, scalar, th);
    return ImageVectorizer.vectorizeMaskLoops(loops, lumaRaw, w, h, traceLight ? -1 : 1, {
      ...opts,
      fillColor: traceLight ? '#f8fafc' : '#1e293b',
      strokeColor: '#38bdf8',
      layerGroup: 'BW Vector',
      namePrefix: 'Vector Shape',
      heightMm: Number((opts.maxDepthMm * 0.55).toFixed(2)),
      shape: 'dome',
      adaptiveRadius: true,
    });
  }

  /**
   * Preset B: Illustrator Perceptual Color Image Trace (K-Means++ CIELAB Multi-Color Layers).
   */
  static traceColorLayers(rArr, gArr, bArr, lumaRaw, denoisedLuma, w, h, opts) {
    const filt = ImageVectorizer.guidedFilterRGB(rArr, gArr, bArr, denoisedLuma, w, h, 2, 140);
    const { labels, palette } = ImageVectorizer.quantizeColorsLAB(filt.r, filt.g, filt.b, w, h, opts.colorCount);
    const n = w * h;

    // Identify background cluster (highest border dominance, or near-white border cluster)
    let bgIndex = -1;
    if (opts.ignoreBg && palette.length > 2) {
      const byBorder = [...palette].sort((a, b) => b.borderFraction - a.borderFraction);
      if (byBorder[0].borderFraction > 0.42 || (byBorder[0].isWhitePaper && byBorder[0].borderFraction > 0.20)) {
        bgIndex = byBorder[0].index;
      }
    }

    // Count true interior pixels (4-neighbors identical) per cluster so thin 1px anti-alias halos are suppressed
    const interiorCnt = new Uint32Array(palette.length);
    for (let y = 1; y < h - 1; y++) {
      const row = y * w;
      for (let x = 1; x < w - 1; x++) {
        const i = row + x;
        const c = labels[i];
        if (labels[i - 1] === c && labels[i + 1] === c && labels[i - w] === c && labels[i + w] === c) {
          interiorCnt[c]++;
        }
      }
    }

    // Order active clusters from broadest/lightest background to finest foreground detail
    const activeClusters = palette
      .filter((p) => p.index !== bgIndex && p.count > opts.minAreaPx * 2 && interiorCnt[p.index] >= Math.max(4, opts.minAreaPx * 0.4))
      .sort((a, b) => (opts.invert ? a.luma - b.luma : b.luma - a.luma));

    if (activeClusters.length === 0) {
      return ImageVectorizer.traceBinaryExact(lumaRaw, denoisedLuma, { threshold: 128 }, w, h, opts);
    }

    const allEntities = [];
    const maxPerLayer = Math.max(12, Math.floor(opts.maxContours / Math.max(1, activeClusters.length)));
    const bfsQueue = new Int32Array(n);

    // Compute distance matrix for continuous subpixel scalar interpolation on each cluster!
    for (let li = 0; li < activeClusters.length; li++) {
      const cl = activeClusters[li];
      const includeHigher = opts.stackMethod === 'stacked';
      const includedSet = new Set([cl.index]);
      if (includeHigher) {
        for (let k = li; k < activeClusters.length; k++) includedSet.add(activeClusters[k].index);
      }

      const mask = new Uint8Array(n);
      if (includeHigher && li < activeClusters.length - 1) {
        // Connected-component stacking: fill nested child colors ONLY inside components actually connected to `cl.index`
        const allowed = new Uint8Array(n);
        let qHead = 0, qTail = 0;
        for (let i = 0; i < n; i++) {
          if (includedSet.has(labels[i])) allowed[i] = 1;
          if (labels[i] === cl.index) {
            mask[i] = 1;
            bfsQueue[qTail++] = i;
          }
        }
        while (qHead < qTail) {
          const cur = bfsQueue[qHead++];
          const cx = cur % w;
          const cy = (cur - cx) / w;
          for (let dy = -1; dy <= 1; dy++) {
            const ny = cy + dy;
            if (ny < 0 || ny >= h) continue;
            const nRow = ny * w;
            for (let dx = -1; dx <= 1; dx++) {
              const nx = cx + dx;
              if (nx < 0 || nx >= w) continue;
              const nb = nRow + nx;
              if (allowed[nb] && !mask[nb]) {
                mask[nb] = 1;
                bfsQueue[qTail++] = nb;
              }
            }
          }
        }
      } else {
        for (let i = 0; i < n; i++) {
          mask[i] = labels[i] === cl.index ? 1 : 0;
        }
      }

      const scalar = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        if (mask[i]) {
          scalar[i] = -1.0;
        } else {
          const dr = filt.r[i] - cl.r, dg = filt.g[i] - cl.g, db = filt.b[i] - cl.b;
          scalar[i] = Math.min(255, Math.hypot(dr, dg, db) * 0.1 + 1.0);
        }
      }

      const cleanedMask = ImageVectorizer.morphClose(mask, w, h, 1);
      // Smooth continuous scalar slightly so Marching Squares edges follow smooth subpixel boundaries
      const smoothScalar = ImageVectorizer.gaussianBlur3x3(scalar, w, h);
      const rawLoops = ImageVectorizer.marchSquares(cleanedMask, w, h, smoothScalar, 0.0);

      // Height increases smoothly up the layer stack for accurate 3D stepped/sculpted relief!
      const heightFrac = (li + 1) / activeClusters.length;
      const layerHeightMm = Number((0.8 + heightFrac * (opts.maxDepthMm - 0.8)).toFixed(2));

      const layerEnts = ImageVectorizer.vectorizeMaskLoops(rawLoops, lumaRaw, w, h, 0, {
        ...opts,
        refineEdges: false,
        maxContours: maxPerLayer,
        fillColor: cl.hex,
        strokeColor: cl.hex,
        layerGroup: `Color #${li + 1} (${cl.hex})`,
        namePrefix: `Color ${li + 1} (${cl.hex})`,
        toneLevel: li + 1,
        toneTotal: activeClusters.length,
        heightMm: layerHeightMm,
        shape: li === 0 && activeClusters.length > 2 ? 'flat' : 'dome',
        mode: li === 0 || opts.stackMethod === 'stacked' ? 'max' : 'max',
        adaptiveRadius: true,
        rejectFullFrame: li === 0 && bgIndex === -1,
      });

      allEntities.push(...layerEnts);
    }

    return allEntities.slice(0, opts.maxContours);
  }

  /**
   * Preset C: Adobe Illustrator "Shades of Gray" / Stacked Multi-Tone 3D Depth Layers.
   * Produces N cumulative elevation contours that match 1:1 with multi-layer 3D carving reliefs!
   */
  static traceGrayscaleLayers(lumaRaw, denoisedLuma, w, h, opts) {
    const n = w * h;
    const field = denoisedLuma;

    const borderMean = ImageVectorizer.computeBorderMean(field, w, h);
    const isLightBg = borderMean > 132;
    const brightUp = opts.invert ? isLightBg : !isLightBg;

    // Normalize intensity so elevation rises cleanly from background (0) to peak detail (255)
    const elevField = new Float32Array(n);
    const fgHist = new Uint32Array(256);
    let fgSamples = 0;

    for (let i = 0; i < n; i++) {
      const e = brightUp ? field[i] : (255 - field[i]);
      elevField[i] = e;
      if (!opts.ignoreBg || e > 24) {
        fgHist[Math.max(0, Math.min(255, Math.round(e)))]++;
        fgSamples++;
      }
    }

    const levels = Math.max(2, Math.min(12, Math.round(opts.toneLevels || 5)));
    // 1D K-Means luminance clustering across foreground intensities -> crisp tonal step boundaries
    let centroids = [];
    for (let lv = 0; lv < levels; lv++) centroids.push(28 + ((lv + 0.5) / levels) * 218);
    if (fgSamples > 50) {
      let minVal = 255, maxVal = 0;
      for (let b = 0; b < 256; b++) if (fgHist[b] > 0) { minVal = Math.min(minVal, b); maxVal = Math.max(maxVal, b); }
      const span = Math.max(12, maxVal - minVal);
      centroids = [];
      for (let lv = 0; lv < levels; lv++) centroids.push(minVal + ((lv + 0.5) / levels) * span);
      for (let it = 0; it < 12; it++) {
        const s = new Float64Array(levels), c = new Float64Array(levels);
        for (let b = 0; b < 256; b++) {
          const wgt = fgHist[b];
          if (wgt === 0) continue;
          let bk = 0, bd = 1e9;
          for (let k = 0; k < levels; k++) {
            const d = Math.abs(b - centroids[k]);
            if (d < bd) { bd = d; bk = k; }
          }
          s[bk] += b * wgt; c[bk] += wgt;
        }
        for (let k = 0; k < levels; k++) if (c[k] > 0) centroids[k] = s[k] / c[k];
      }
      centroids.sort((a, b) => a - b);
      // Merge duplicate/empty clusters closer than 12 gray levels
      const uniqueC = [];
      for (const ct of centroids) {
        if (uniqueC.length === 0 || ct - uniqueC[uniqueC.length - 1] >= 12) uniqueC.push(ct);
      }
      centroids = uniqueC;
    }

    // Threshold at half of lowest foreground cluster (to separate bg), then midpoints between tonal clusters
    const thresholds = [];
    if (centroids.length > 0) {
      thresholds.push(Math.max(18, centroids[0] * 0.5));
      for (let k = 1; k < centroids.length; k++) {
        thresholds.push((centroids[k - 1] + centroids[k]) * 0.5);
      }
    } else {
      thresholds.push(40, 120);
    }

    const effLevels = thresholds.length;
    const allEntities = [];
    const maxPerLevel = Math.max(8, Math.floor(opts.maxContours / Math.max(1, effLevels)));
    const palette = ['#334155', '#475569', '#64748b', '#94a3b8', '#cbd5e1', '#e2e8f0', '#f1f5f9', '#f8fafc'];

    for (let lv = 0; lv < effLevels; lv++) {
      const th = thresholds[lv];
      // Trace cumulative regions where elevation >= th (inverted sign for marchSquares `< th`)
      const scalar = new Float32Array(n);
      const mask = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        scalar[i] = -elevField[i];
        mask[i] = elevField[i] >= th ? 1 : 0;
      }

      const cleaned = ImageVectorizer.morphClose(mask, w, h, 1);
      const rawLoops = ImageVectorizer.marchSquares(cleaned, w, h, scalar, -th);
      const grayVal = Math.round(45 + ((lv + 1) / effLevels) * 195);
      const toneHex = ImageVectorizer.rgbToHex(grayVal, grayVal, grayVal);
      const zMm = Number((((lv + 1) / effLevels) * opts.maxDepthMm).toFixed(2));

      const ents = ImageVectorizer.vectorizeMaskLoops(rawLoops, elevField, w, h, -1, {
        ...opts,
        maxContours: maxPerLevel,
        fillColor: toneHex,
        strokeColor: palette[lv % palette.length],
        layerGroup: `Tone Layer ${lv + 1}/${effLevels}`,
        namePrefix: `Tone L${lv + 1} (${zMm}mm)`,
        toneLevel: lv + 1,
        toneTotal: effLevels,
        heightMm: zMm,
        shape: lv === 0 ? 'flat' : 'smooth',
        mode: 'max',
        adaptiveRadius: true,
        rejectFullFrame: lv === 0,
      });

      allEntities.push(...ents);
    }

    return allEntities.slice(0, opts.maxContours);
  }

  /**
   * Preset D: Specialized Carved Wood Design / Intricate Relief Contour Extraction.
   */
  static traceWoodReliefLayers(rArr, gArr, bArr, lumaRaw, denoisedLuma, otsu, w, h, opts) {
    const n = w * h;
    // Remove lighting glare across wood planks & combine luminance + chroma separation + edge magnitude
    const filt = ImageVectorizer.guidedFilterRGB(rArr, gArr, bArr, denoisedLuma, w, h, 2, 160);
    let illumLuma = denoisedLuma;
    if (opts.localLighting) {
      const illumR = Math.max(12, Math.round(Math.min(w, h) * 0.12));
      illumLuma = ImageVectorizer.correctLocalLighting(denoisedLuma, w, h, illumR);
    }

    // Border chroma reference
    let bR = 0, bG = 0, bB = 0, bCnt = 0;
    for (let x = 0; x < w; x++) {
      bR += filt.r[x] + filt.r[(h - 1) * w + x];
      bG += filt.g[x] + filt.g[(h - 1) * w + x];
      bB += filt.b[x] + filt.b[(h - 1) * w + x];
      bCnt += 2;
    }
    bR /= bCnt; bG /= bCnt; bB /= bCnt;

    // Sobel gradient & foreground saliency field
    const featureScore = new Float32Array(n);
    const borderMean = ImageVectorizer.computeBorderMean(illumLuma, w, h);
    const flip = opts.invert ? (borderMean <= 128) : (borderMean > 128);

    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        const gx = -illumLuma[i - w - 1] + illumLuma[i - w + 1] - 2 * illumLuma[i - 1] + 2 * illumLuma[i + 1] - illumLuma[i + w - 1] + illumLuma[i + w + 1];
        const gy = -illumLuma[i - w - 1] - 2 * illumLuma[i - w] - illumLuma[i - w + 1] + illumLuma[i + w - 1] + 2 * illumLuma[i + w] + illumLuma[i + w + 1];
        const grad = Math.hypot(gx, gy) * 0.25;
        const lDiff = flip ? Math.max(0, 132 - illumLuma[i]) : Math.max(0, illumLuma[i] - 124);
        const cDiff = Math.hypot(filt.r[i] - bR, filt.g[i] - bG, filt.b[i] - bB) * 0.45;
        featureScore[i] = Math.min(255, Math.max(lDiff * 1.15, cDiff, grad * 0.9));
      }
    }

    const smoothedScore = ImageVectorizer.guidedFilter(featureScore, w, h, 2, 120);
    // Use clean 2-3 layered depth extraction OR crisp binary based on toneLevels
    const levels = Math.min(5, Math.max(2, Math.round(opts.toneLevels || 3)));
    const allEntities = [];
    const perLevel = Math.max(15, Math.floor(opts.maxContours / levels));
    const woodPalette = ['#d97706', '#f59e0b', '#fbbf24', '#fef08a', '#38bdf8'];

    for (let lv = 0; lv < levels; lv++) {
      const th = opts.threshold > 0
        ? (opts.threshold * 180) + lv * 28
        : Math.max(22, (72 - opts.detailFactor * 42) + lv * 34);
      const scalar = new Float32Array(n);
      const mask = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        scalar[i] = -smoothedScore[i];
        mask[i] = smoothedScore[i] >= th ? 1 : 0;
      }
      const cleaned = ImageVectorizer.morphClose(mask, w, h, 1);
      const rawLoops = ImageVectorizer.marchSquares(cleaned, w, h, scalar, -th);
      const zMm = Number((((lv + 1) / levels) * opts.maxDepthMm).toFixed(2));

      const ents = ImageVectorizer.vectorizeMaskLoops(rawLoops, smoothedScore, w, h, -1, {
        ...opts,
        maxContours: perLevel,
        fillColor: woodPalette[lv % woodPalette.length],
        strokeColor: '#38bdf8',
        layerGroup: `Wood Relief L${lv + 1}/${levels}`,
        namePrefix: `Carving L${lv + 1} (${zMm}mm)`,
        toneLevel: lv + 1,
        toneTotal: levels,
        heightMm: zMm,
        shape: lv === 0 ? 'ogee' : 'dome',
        mode: 'max',
        adaptiveRadius: true,
        rejectFullFrame: true,
      });
      allEntities.push(...ents);
    }

    return allEntities.slice(0, opts.maxContours);
  }

  /**
   * Preset E: Edge / Carved Groove Ridge Tracer (Multi-Channel Scharr Gradient).
   */
  static traceEdgeGrooves(rArr, gArr, bArr, lumaRaw, denoisedLuma, w, h, opts) {
    const n = w * h;
    const edgeMag = new Float32Array(n);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        const gxL = -denoisedLuma[i - w - 1] + denoisedLuma[i - w + 1] - 2 * denoisedLuma[i - 1] + 2 * denoisedLuma[i + 1] - denoisedLuma[i + w - 1] + denoisedLuma[i + w + 1];
        const gyL = -denoisedLuma[i - w - 1] - 2 * denoisedLuma[i - w] - denoisedLuma[i - w + 1] + denoisedLuma[i + w - 1] + 2 * denoisedLuma[i + w] + denoisedLuma[i + w + 1];
        const gxR = rArr[i + 1] - rArr[i - 1], gyR = rArr[i + w] - rArr[i - w];
        const gxG = gArr[i + 1] - gArr[i - 1], gyG = gArr[i + w] - gArr[i - w];
        const gxB = bArr[i + 1] - bArr[i - 1], gyB = bArr[i + w] - bArr[i - w];
        const gColor = Math.max(Math.hypot(gxR, gyR), Math.hypot(gxG, gyG), Math.hypot(gxB, gyB)) * 1.4;
        edgeMag[i] = Math.min(255, Math.max(Math.hypot(gxL, gyL) * 0.25, gColor));
      }
    }

    const th = opts.threshold > 0 ? opts.threshold * 140 : Math.max(18, 65 - opts.detailFactor * 48);
    const scalar = new Float32Array(n);
    const mask = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      scalar[i] = -edgeMag[i];
      mask[i] = edgeMag[i] >= th ? 1 : 0;
    }
    const cleaned = ImageVectorizer.morphClose(mask, w, h, 1);
    const rawLoops = ImageVectorizer.marchSquares(cleaned, w, h, scalar, -th);

    return ImageVectorizer.vectorizeMaskLoops(rawLoops, edgeMag, w, h, -1, {
      ...opts,
      refineEdges: false,
      fillColor: '#38bdf8',
      strokeColor: '#38bdf8',
      layerGroup: 'Carved Edges',
      namePrefix: 'Edge Groove',
      heightMm: Number((opts.maxDepthMm * 0.4).toFixed(2)),
      shape: 'vcarve',
      mode: 'max',
      adaptiveRadius: true,
      rejectFullFrame: true,
    });
  }

  static computeBorderMean(field, w, h) {
    let sum = 0, cnt = 0;
    const pad = Math.min(3, Math.floor(Math.min(w, h) / 4));
    for (let x = 0; x < w; x++) {
      sum += field[pad * w + x] + field[(h - 1 - pad) * w + x];
      cnt += 2;
    }
    for (let y = 0; y < h; y++) {
      sum += field[y * w + pad] + field[y * w + (w - 1 - pad)];
      cnt += 2;
    }
    return cnt > 0 ? sum / cnt : 255;
  }

  static morphClose(binary, w, h, passes = 1) {
    if (passes <= 0) {
      // At least remove isolated 1px grain dots
      const clean = new Uint8Array(binary);
      for (let y = 1; y < h - 1; y++) {
        const row = y * w;
        for (let x = 1; x < w - 1; x++) {
          const i = row + x;
          if (clean[i] && !(binary[i - 1] || binary[i + 1] || binary[i - w] || binary[i + w])) {
            clean[i] = 0;
          }
        }
      }
      return clean;
    }
    let cur = new Uint8Array(binary);
    for (let p = 0; p < passes; p++) {
      const dil = new Uint8Array(w * h);
      for (let y = 1; y < h - 1; y++) {
        const r = y * w;
        for (let x = 1; x < w - 1; x++) {
          const i = r + x;
          if (cur[i] || cur[i - 1] || cur[i + 1] || cur[i - w] || cur[i + w]) dil[i] = 1;
        }
      }
      const ero = new Uint8Array(w * h);
      for (let y = 1; y < h - 1; y++) {
        const r = y * w;
        for (let x = 1; x < w - 1; x++) {
          const i = r + x;
          if (dil[i] && dil[i - 1] && dil[i + 1] && dil[i - w] && dil[i + w]) ero[i] = 1;
        }
      }
      cur = ero;
    }
    return cur;
  }

  // ============================================================
  // 4. Sub-Pixel Marching Squares (Flat Int32 Adjacency Array)
  // ============================================================

  static marchSquares(binary, w, h, scalarField = null, threshold = 0) {
    const PW = w + 2;
    const PH = h + 2;
    const padMask = new Uint8Array(PW * PH);
    const padScalar = scalarField ? new Float32Array(PW * PH).fill(threshold + 60) : null;

    for (let y = 0; y < h; y++) {
      const srcRow = y * w;
      const dstRow = (y + 1) * PW + 1;
      for (let x = 0; x < w; x++) {
        padMask[dstRow + x] = binary[srcRow + x];
        if (padScalar) padScalar[dstRow + x] = scalarField[srcRow + x];
      }
    }

    const K = 2 * PW + 3;
    const KS = K * (2 * PH + 3);
    const adj = new Int32Array(KS * 2).fill(-1);
    const link = (a, b) => {
      if (adj[2 * a] < 0) adj[2 * a] = b;
      else adj[2 * a + 1] = b;
    };

    for (let y = 0; y < PH - 1; y++) {
      const r0 = y * PW;
      const r1 = (y + 1) * PW;
      for (let x = 0; x < PW - 1; x++) {
        const tl = padMask[r0 + x];
        const tr = padMask[r0 + x + 1];
        const br = padMask[r1 + x + 1];
        const bl = padMask[r1 + x];
        const c = (tl << 3) | (tr << 2) | (br << 1) | bl;
        if (c === 0 || c === 15) continue;

        const T = 2 * y * K + 2 * x + 1;
        const R = (2 * y + 1) * K + 2 * x + 2;
        const B = (2 * y + 2) * K + 2 * x + 1;
        const L = (2 * y + 1) * K + 2 * x;

        let e0 = -1, e1 = -1, e2 = -1, e3 = -1;
        switch (c) {
          case 1: case 14: e0 = L; e1 = B; break;
          case 2: case 13: e0 = B; e1 = R; break;
          case 3: case 12: e0 = L; e1 = R; break;
          case 4: case 11: e0 = T; e1 = R; break;
          case 5: e0 = T; e1 = L; e2 = B; e3 = R; break;
          case 6: case 9:  e0 = T; e1 = B; break;
          case 7: case 8:  e0 = T; e1 = L; break;
          case 10: e0 = T; e1 = R; e2 = L; e3 = B; break;
        }
        if (e0 >= 0) { link(e0, e1); link(e1, e0); }
        if (e2 >= 0) { link(e2, e3); link(e3, e2); }
      }
    }

    const interp = (k) => {
      const X = k % K;
      const Y = (k - X) / K;
      if (!padScalar) {
        return [
          Math.max(0, Math.min(w, X / 2 - 0.5)),
          Math.max(0, Math.min(h, Y / 2 - 0.5)),
        ];
      }
      let idx0, idx1;
      if (X & 1) {
        idx0 = (Y >> 1) * PW + ((X - 1) >> 1);
        idx1 = idx0 + 1;
      } else {
        idx0 = ((Y - 1) >> 1) * PW + (X >> 1);
        idx1 = idx0 + PW;
      }
      const v0 = padScalar[idx0];
      const v1 = padScalar[idx1];
      let t = 0.5;
      if ((v0 - threshold) * (v1 - threshold) <= 0 && v0 !== v1) {
        t = (threshold - v0) / (v1 - v0);
        if (t < 0) t = 0; else if (t > 1) t = 1;
      }
      const px = X & 1 ? ((X - 1) >> 1) + t - 0.5 : (X >> 1) - 0.5;
      const py = X & 1 ? (Y >> 1) - 0.5 : ((Y - 1) >> 1) + t - 0.5;
      return [Math.max(0, Math.min(w, px)), Math.max(0, Math.min(h, py))];
    };

    const visited = new Uint8Array(KS);
    const loops = [];
    for (let s0 = 0; s0 < KS; s0++) {
      if (adj[2 * s0] < 0 || visited[s0]) continue;
      const loop = [];
      let prev = -1;
      let cur = s0;
      let guard = 0;
      while (guard++ < KS) {
        visited[cur] = 1;
        loop.push(interp(cur));
        const a0 = adj[2 * cur];
        const a1 = adj[2 * cur + 1];
        const next = a0 === prev ? a1 : a0;
        if (next < 0 || next === s0 || visited[next]) break;
        prev = cur;
        cur = next;
      }
      if (loop.length >= 4) loops.push(loop);
    }

    return loops;
  }

  // ============================================================
  // 5. Potrace-Style Corner-Exact + Cubic Bézier Curve Fitting
  // ============================================================

  static loopSignedArea(L) {
    let s = 0;
    const n = L.length;
    for (let i = 0; i < n; i++) {
      const a = L[i], b = L[(i + 1) % n];
      s += a[0] * b[1] - b[0] * a[1];
    }
    return s * 0.5;
  }

  static pointInLoop(px, py, L) {
    let inside = false;
    for (let i = 0, j = L.length - 1; i < L.length; j = i++) {
      const ax = L[i][0], ay = L[i][1];
      const bx = L[j][0], by = L[j][1];
      if ((ay > py) !== (by > py) && px < ((bx - ax) * (py - ay)) / (by - ay) + ax) {
        inside = !inside;
      }
    }
    return inside;
  }

  /**
   * Slide open or closed subpixel contour along normals to the exact 50% intensity crossing on original raw luma.
   */
  static refineContourToField(pts, closed, rawField, w, h, sign, scaleRatio) {
    const n = pts.length;
    if (!rawField || n < (closed ? 12 : 5)) return pts;

    const searchPx = Math.min(8, Math.max(2.5, Math.round(scaleRatio * 1.4 + 1)));
    const step = 0.5;
    const steps = Math.round(searchPx / step);

    const sampleField = (x, y) => {
      const cx = Math.max(0, Math.min(w - 1.001, x));
      const cy = Math.max(0, Math.min(h - 1.001, y));
      const i = cx | 0, j = cy | 0;
      const fx = cx - i, fy = cy - j;
      const a = j * w + i;
      return sign * ((rawField[a] * (1 - fx) + rawField[a + 1] * fx) * (1 - fy) + (rawField[a + w] * (1 - fx) + rawField[a + w + 1] * fx) * fy);
    };

    const idxWrap = closed ? (i) => ((i % n) + n) % n : (i) => Math.max(0, Math.min(n - 1, i));
    const shift = new Float32Array(n);
    const nxArr = new Float32Array(n);
    const nyArr = new Float32Array(n);
    const iStart = closed ? 0 : 1;
    const iEnd = closed ? n : n - 1;

    for (let i = iStart; i < iEnd; i++) {
      const a = pts[idxWrap(i - 2)];
      const b = pts[idxWrap(i + 2)];
      const p = pts[i];
      let nx = a[1] - b[1];
      let ny = b[0] - a[0];
      const len = Math.hypot(nx, ny);
      if (len < 1e-6) continue;
      nx /= len; ny /= len;
      nxArr[i] = nx; nyArr[i] = ny;

      const vals = [];
      for (let j = 0; j <= 2 * steps; j++) {
        const t = (j - steps) * step;
        vals.push(sampleField(p[0] + nx * t, p[1] + ny * t));
      }

      let ma = 0, mb = 0;
      for (let j = 0; j < steps - 1; j++) { ma += vals[j]; mb += vals[2 * steps - j]; }
      const negLeft = ma < mb;
      const leftVals = [], rightVals = [];
      for (let j = 0; j <= 2 * steps; j++) {
        if (Math.abs(j - steps) < 1) continue;
        ((j < steps) === negLeft ? leftVals : rightVals).push(vals[j]);
      }
      if (leftVals.length === 0 || rightVals.length === 0) continue;
      rightVals.sort((u, v) => u - v);
      const lo = Math.min(...leftVals);
      const hi = rightVals[Math.floor(rightVals.length * 0.5)];
      if (hi - lo < 18) continue;
      const mid = (lo + hi) * 0.5;

      let bestShift = 1e9;
      for (let j = 0; j < 2 * steps; j++) {
        const u = vals[j] - mid;
        const v = vals[j + 1] - mid;
        if (u * v <= 0 && u !== v) {
          const t = (j - steps) * step + step * (u / (u - v));
          if (Math.abs(t) < Math.abs(bestShift)) bestShift = t;
        }
      }
      if (bestShift < 1e8 && Math.abs(bestShift) <= searchPx * 0.65) {
        shift[i] = bestShift;
      }
    }

    // 5-tap median + mean smooth of normal shifts prevents jitter
    const med = new Float32Array(n);
    for (let i = iStart; i < iEnd; i++) {
      const win = [];
      for (let k = -2; k <= 2; k++) win.push(shift[idxWrap(i + k)]);
      win.sort((a, b) => a - b);
      med[i] = win[2];
    }

    return pts.map((p, i) => {
      if (!closed && (i === 0 || i === n - 1)) return p;
      const t = (med[idxWrap(i - 1)] + med[i] + med[idxWrap(i + 1)]) / 3.0;
      return [p[0] + nxArr[i] * t, p[1] + nyArr[i] * t];
    });
  }

  /**
   * Total Least Squares (TLS) line fit -> { c: [mx, my], d: [ux, uy], dev: maxPerpDist }
   */
  static fitLineTLS(pts) {
    const n = pts.length;
    if (n < 2) return null;
    let mx = 0, my = 0;
    for (const p of pts) { mx += p[0]; my += p[1]; }
    mx /= n; my /= n;
    let a = 0, b = 0, c = 0;
    for (const p of pts) {
      const u = p[0] - mx, v = p[1] - my;
      a += u * u; b += u * v; c += v * v;
    }
    const phi = 0.5 * Math.atan2(2 * b, a - c);
    let dx = Math.cos(phi), dy = Math.sin(phi);
    // Orient direction consistently from pts[0] toward pts[n-1]
    const spanDx = pts[n - 1][0] - pts[0][0];
    const spanDy = pts[n - 1][1] - pts[0][1];
    if (dx * spanDx + dy * spanDy < 0) { dx = -dx; dy = -dy; }
    let dev = 0;
    for (const p of pts) {
      dev = Math.max(dev, Math.abs(-(p[0] - mx) * dy + (p[1] - my) * dx));
    }
    return { c: [mx, my], d: [dx, dy], dev };
  }

  static intersectLines(A, B) {
    if (!A || !B) return null;
    const cross = A.d[0] * B.d[1] - A.d[1] * B.d[0];
    if (Math.abs(cross) < 0.16) return null;
    const dx = B.c[0] - A.c[0], dy = B.c[1] - A.c[1];
    const t = (dx * B.d[1] - dy * B.d[0]) / cross;
    return [A.c[0] + t * A.d[0], A.c[1] + t * A.d[1]];
  }

  /**
   * Volume-preserving open-span Taubin smoothing (zero shrinkage!)
   */
  static taubinSmoothOpen(pts, passes = 2) {
    const n = pts.length;
    if (n < 4 || passes < 1) return pts;
    let Q = pts.map((p) => [p[0], p[1]]);
    for (let it = 0; it < passes * 2; it++) {
      const k = (it & 1) ? -0.53 : 0.50;
      const R = Q.map((p) => [p[0], p[1]]);
      for (let i = 1; i < n - 1; i++) {
        R[i][0] = Q[i][0] + k * ((Q[i - 1][0] + Q[i + 1][0]) * 0.5 - Q[i][0]);
        R[i][1] = Q[i][1] + k * ((Q[i - 1][1] + Q[i + 1][1]) * 0.5 - Q[i][1]);
      }
      Q = R;
    }
    return Q;
  }

  /**
   * Algebraic circle test: detects true geometric circles -> 4 exact Cubic Bezier quadrants (k = 0.55228475).
   */
  static detectExactCircle(pts) {
    const n = pts.length;
    if (n < 12) return null;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of pts) {
      if (p[0] < minX) minX = p[0]; if (p[0] > maxX) maxX = p[0];
      if (p[1] < minY) minY = p[1]; if (p[1] > maxY) maxY = p[1];
    }
    const w = maxX - minX, h = maxY - minY;
    const r = (w + h) * 0.25;
    const cx = (minX + maxX) * 0.5, cy = (minY + maxY) * 0.5;
    if (r < 1.5 || Math.abs(w - h) > Math.max(w, h) * 0.03) return null;

    let maxErr = 0;
    for (const p of pts) {
      maxErr = Math.max(maxErr, Math.abs(Math.hypot(p[0] - cx, p[1] - cy) - r));
    }
    // 0.028 * r is strictly tighter than any polygon (octagon error = 0.082 * r)
    if (maxErr > Math.max(0.45, r * 0.028)) return null;

    const dir = ImageVectorizer.loopSignedArea(pts) >= 0 ? 1 : -1;
    const k = 0.55228475 * r;
    const anchors = [];
    for (let q = 0; q < 4; q++) {
      const ang = dir * q * (Math.PI / 2);
      const cos = Math.cos(ang), sin = Math.sin(ang);
      const px = cx + r * cos, py = cy + r * sin;
      // Tangent along increasing dir*angle is (-dir*sin, dir*cos)
      const tx = -dir * sin, ty = dir * cos;
      anchors.push({
        x: px,
        y: py,
        corner: false,
        hIn: { x: px - tx * k, y: py - ty * k },
        hOut: { x: px + tx * k, y: py + ty * k },
      });
    }
    return anchors;
  }

  // ============================================================
  // 6. Schneider Recursive Cubic Bezier Fitter (Illustrator Core)
  // ============================================================

  static evalCubic(b0, b1, b2, b3, t) {
    const mt = 1 - t;
    const mt2 = mt * mt, t2 = t * t;
    return [
      mt2 * mt * b0[0] + 3 * mt2 * t * b1[0] + 3 * mt * t2 * b2[0] + t2 * t * b3[0],
      mt2 * mt * b0[1] + 3 * mt2 * t * b1[1] + 3 * mt * t2 * b2[1] + t2 * t * b3[1],
    ];
  }

  static generateCubicBezier(pts, first, last, uPrime, tHat1, tHat2) {
    const p0 = pts[first];
    const p3 = pts[last];
    const nPts = last - first + 1;
    let c00 = 0, c01 = 0, c11 = 0, x0 = 0, x1 = 0;

    for (let i = 0; i < nPts; i++) {
      const u = uPrime[i];
      const mu = 1 - u;
      const b0 = mu * mu * mu;
      const b1 = 3 * u * mu * mu;
      const b2 = 3 * u * u * mu;
      const b3 = u * u * u;

      const a1x = tHat1[0] * b1, a1y = tHat1[1] * b1;
      const a2x = tHat2[0] * b2, a2y = tHat2[1] * b2;
      c00 += a1x * a1x + a1y * a1y;
      c01 += a1x * a2x + a1y * a2y;
      c11 += a2x * a2x + a2y * a2y;

      const rx = pts[first + i][0] - (p0[0] * (b0 + b1) + p3[0] * (b2 + b3));
      const ry = pts[first + i][1] - (p0[1] * (b0 + b1) + p3[1] * (b2 + b3));
      x0 += a1x * rx + a1y * ry;
      x1 += a2x * rx + a2y * ry;
    }

    const detC0C1 = c00 * c11 - c01 * c01;
    const chord = Math.hypot(p3[0] - p0[0], p3[1] - p0[1]);
    const fallbackAlpha = chord / 3.0;
    let alphaL = fallbackAlpha;
    let alphaR = fallbackAlpha;

    if (Math.abs(detC0C1) > 1e-10) {
      alphaL = (c11 * x0 - c01 * x1) / detC0C1;
      alphaR = (c00 * x1 - c01 * x0) / detC0C1;
    }

    const eps = 1e-6 * chord;
    if (alphaL < eps || alphaR < eps || alphaL > chord * 1.6 || alphaR > chord * 1.6) {
      alphaL = fallbackAlpha;
      alphaR = fallbackAlpha;
    }

    return [
      p0,
      [p0[0] + tHat1[0] * alphaL, p0[1] + tHat1[1] * alphaL],
      [p3[0] + tHat2[0] * alphaR, p3[1] + tHat2[1] * alphaR],
      p3,
    ];
  }

  static computeMaxError(pts, first, last, bez, u) {
    let maxDistSq = 0;
    let split = (first + last) >> 1;
    for (let i = first + 1; i < last; i++) {
      const q = ImageVectorizer.evalCubic(bez[0], bez[1], bez[2], bez[3], u[i - first]);
      const d2 = (q[0] - pts[i][0]) ** 2 + (q[1] - pts[i][1]) ** 2;
      if (d2 >= maxDistSq) {
        maxDistSq = d2;
        split = i;
      }
    }
    return { maxDistSq, split };
  }

  static reparameterize(pts, first, last, u, bez) {
    const uOut = new Float64Array(last - first + 1);
    const q1 = [3 * (bez[1][0] - bez[0][0]), 3 * (bez[1][1] - bez[0][1])];
    const q2 = [3 * (bez[2][0] - bez[1][0]), 3 * (bez[2][1] - bez[1][1])];
    const q3 = [3 * (bez[3][0] - bez[2][0]), 3 * (bez[3][1] - bez[2][1])];
    const p1 = [2 * (q2[0] - q1[0]), 2 * (q2[1] - q1[1])];
    const p2 = [2 * (q3[0] - q2[0]), 2 * (q3[1] - q2[1])];

    for (let i = first; i <= last; i++) {
      let t = u[i - first];
      const mt = 1 - t;
      const qt = ImageVectorizer.evalCubic(bez[0], bez[1], bez[2], bez[3], t);
      const q1t = [
        mt * mt * q1[0] + 2 * mt * t * q2[0] + t * t * q3[0],
        mt * mt * q1[1] + 2 * mt * t * q2[1] + t * t * q3[1],
      ];
      const q2t = [mt * p1[0] + t * p2[0], mt * p1[1] + t * p2[1]];
      const rx = qt[0] - pts[i][0], ry = qt[1] - pts[i][1];
      const num = rx * q1t[0] + ry * q1t[1];
      const den = q1t[0] * q1t[0] + q1t[1] * q1t[1] + rx * q2t[0] + ry * q2t[1];
      if (Math.abs(den) > 1e-10) t -= num / den;
      uOut[i - first] = Math.max(0, Math.min(1, t));
    }
    return uOut;
  }

  static fitCubicSpan(pts, first, last, tHat1, tHat2, errorSq, depth = 0) {
    const nPts = last - first + 1;
    if (nPts <= 2) {
      const p0 = pts[first], p3 = pts[last];
      return [[p0, null, null, p3]];
    }
    if (nPts === 3) {
      // Exact quadratic midpoint fit converted cleanly to cubic
      const p0 = pts[first], p1 = pts[first + 1], p2 = pts[last];
      const qx = 2 * p1[0] - 0.5 * (p0[0] + p2[0]);
      const qy = 2 * p1[1] - 0.5 * (p0[1] + p2[1]);
      const c1 = [p0[0] / 3 + (2 / 3) * qx, p0[1] / 3 + (2 / 3) * qy];
      const c2 = [p2[0] / 3 + (2 / 3) * qx, p2[1] / 3 + (2 / 3) * qy];
      return [[p0, c1, c2, p2]];
    }

    // Chord-length parameterization
    const u = new Float64Array(nPts);
    for (let i = first + 1; i <= last; i++) {
      u[i - first] = u[i - first - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
    }
    const totalLen = u[nPts - 1] || 1;
    for (let i = 1; i < nPts; i++) u[i] /= totalLen;

    let bez = ImageVectorizer.generateCubicBezier(pts, first, last, u, tHat1, tHat2);
    let { maxDistSq, split } = ImageVectorizer.computeMaxError(pts, first, last, bez, u);
    if (maxDistSq <= errorSq) return [bez];

    if (maxDistSq < errorSq * 4.0) {
      let curU = u;
      for (let it = 0; it < 4; it++) {
        curU = ImageVectorizer.reparameterize(pts, first, last, curU, bez);
        bez = ImageVectorizer.generateCubicBezier(pts, first, last, curU, tHat1, tHat2);
        const err = ImageVectorizer.computeMaxError(pts, first, last, bez, curU);
        maxDistSq = err.maxDistSq;
        split = err.split;
        if (maxDistSq <= errorSq) return [bez];
      }
    }

    if (depth >= 12 || split <= first || split >= last) {
      return [bez];
    }

    const v1x = pts[split - 1][0] - pts[split + 1][0];
    const v1y = pts[split - 1][1] - pts[split + 1][1];
    const vLen = Math.hypot(v1x, v1y) || 1;
    const tHatCenter = [v1x / vLen, v1y / vLen];
    const negCenter = [-tHatCenter[0], -tHatCenter[1]];

    const leftBezs = ImageVectorizer.fitCubicSpan(pts, first, split, tHat1, tHatCenter, errorSq, depth + 1);
    const rightBezs = ImageVectorizer.fitCubicSpan(pts, split, last, negCenter, tHat2, errorSq, depth + 1);
    return leftBezs.concat(rightBezs);
  }

  /**
   * Corner-Exact + Optimal Cubic Bezier Vector Fitter (returns Adobe Illustrator Anchor Points!).
   */
  static fitIllustratorCurve(rawLoop, rawField, w, h, refSign, opts) {
    const n = rawLoop.length;
    if (n < 4) return null;

    const scaleRatio = opts.scaleRatio || 1;
    const doRefine = opts.refineEdges !== false;

    // Step 1: Test exact circle before vertex discretization
    const refinedWhole = doRefine ? ImageVectorizer.refineContourToField(rawLoop, true, rawField, w, h, refSign, scaleRatio) : rawLoop;
    const circleAnchors = ImageVectorizer.detectExactCircle(refinedWhole);
    if (circleAnchors) return circleAnchors;

    // Step 2: Cumulative arc length along contour
    const s = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) {
      const a = rawLoop[i], b = rawLoop[(i + 1) % n];
      s[i + 1] = s[i] + Math.hypot(b[0] - a[0], b[1] - a[1]);
    }
    const perimeter = s[n];
    if (perimeter < 4.0) return null;

    const wChord = Math.max(1.8, Math.min(perimeter * 0.14, 1.0 * scaleRatio));
    const sampleArc = (uTarget) => {
      const u = ((uTarget % perimeter) + perimeter) % perimeter;
      let lo = 0, hi = n;
      while (hi - lo > 1) {
        const m = (lo + hi) >> 1;
        if (s[m] <= u) lo = m; else hi = m;
      }
      const a = rawLoop[lo], b = rawLoop[(lo + 1) % n];
      const seg = s[lo + 1] - s[lo] || 1;
      const t = (u - s[lo]) / seg;
      return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    };

    // Corner sensitivity mapped cleanly: higher corners -> catch shallower angles (~28 deg); lower -> sharper corners (~62 deg)
    const minCornerRad = (1.15 - (opts.corners || 0.6) * 0.72); // ~0.46 to 1.08 rad
    const turns = new Float32Array(n);
    const cand = [];
    for (let i = 0; i < n; i++) {
      const p = rawLoop[i];
      const a = sampleArc(s[i] - wChord);
      const b = sampleArc(s[i] + wChord);
      const ux = p[0] - a[0], uy = p[1] - a[1];
      const vx = b[0] - p[0], vy = b[1] - p[1];
      const den = Math.hypot(ux, uy) * Math.hypot(vx, vy);
      const ang = den > 1e-9 ? Math.acos(Math.max(-1, Math.min(1, (ux * vx + uy * vy) / den))) : 0;
      turns[i] = ang;
      if (ang >= minCornerRad) cand.push(i);
    }

    // Non-maximum suppression along arc length
    cand.sort((a, b) => turns[b] - turns[a] || a - b);
    let cornerIdx = [];
    for (const idx of cand) {
      let ok = true;
      for (const kept of cornerIdx) {
        const d = Math.abs(s[idx] - s[kept]);
        if (Math.min(d, perimeter - d) < wChord * 0.7) { ok = false; break; }
      }
      if (ok) cornerIdx.push(idx);
    }
    cornerIdx.sort((a, b) => a - b);

    // Zero corners (purely smooth closed loop, e.g. oval/petal) -> split cleanly at two antipodal extremes
    if (cornerIdx.length === 0) {
      let pMin = 0;
      for (let i = 1; i < n; i++) {
        if (refinedWhole[i][1] < refinedWhole[pMin][1] || (refinedWhole[i][1] === refinedWhole[pMin][1] && refinedWhole[i][0] < refinedWhole[pMin][0])) {
          pMin = i;
        }
      }
      let pFar = pMin, maxD = -1;
      for (let i = 0; i < n; i++) {
        const d = Math.hypot(refinedWhole[i][0] - refinedWhole[pMin][0], refinedWhole[i][1] - refinedWhole[pMin][1]);
        if (d > maxD) { maxD = d; pFar = i; }
      }
      const splitPair = pMin < pFar ? [pMin, pFar] : [pFar, pMin];
      if (splitPair[1] === splitPair[0]) return null;

      const smoothed = ImageVectorizer.taubinSmoothOpen(
        refinedWhole.slice(splitPair[0]).concat(refinedWhole.slice(0, splitPair[0] + 1)),
        2
      );
      const midInSmoothed = (splitPair[1] - splitPair[0]);
      const errSq = Math.pow(Math.max(0.25, (1.1 - opts.detailFactor) * 1.4), 2);

      const fitHalf = (subPts) => {
        const len = subPts.length;
        const t1 = ImageVectorizer.normVec([subPts[Math.min(len - 1, 2)][0] - subPts[0][0], subPts[Math.min(len - 1, 2)][1] - subPts[0][1]]);
        const t2 = ImageVectorizer.normVec([subPts[Math.max(0, len - 3)][0] - subPts[len - 1][0], subPts[Math.max(0, len - 3)][1] - subPts[len - 1][1]]);
        return ImageVectorizer.fitCubicSpan(subPts, 0, len - 1, t1, t2, errSq);
      };

      const cub1 = fitHalf(smoothed.slice(0, midInSmoothed + 1));
      const cub2 = fitHalf(smoothed.slice(midInSmoothed));
      return ImageVectorizer.cubicsToAnchors(
        [{ cubics: cub1, cornerStart: false }, { cubics: cub2, cornerStart: false }]
      );
    }

    // Extract spans between crisp corners and refine edges + fit TLS lines / Cubic Beziers
    const m = cornerIdx.length;
    const rawSpans = [];
    for (let j = 0; j < m; j++) {
      const a = cornerIdx[j];
      const b = cornerIdx[(j + 1) % m];
      const spanPts = [rawLoop[a]];
      let idx = a;
      let guard = 0;
      do {
        idx = (idx + 1) % n;
        spanPts.push(rawLoop[idx]);
      } while (idx !== b && guard++ < n);
      rawSpans.push(doRefine ? ImageVectorizer.refineContourToField(spanPts, false, rawField, w, h, refSign, scaleRatio) : spanPts);
    }

    const lineTol = Math.max(0.35, (1.05 - opts.detailFactor) * 0.9 + 0.15);
    const guardArc = Math.min(2.2, wChord);
    const spanInfos = rawSpans.map((P) => {
      const q = P.length;
      const sl = new Float64Array(q);
      for (let i = 1; i < q; i++) sl[i] = sl[i - 1] + Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]);
      const Ls = sl[q - 1];
      const info = { P, sl, Ls, isLine: false, A: null, B: null };
      const mid = [];
      for (let i = 0; i < q; i++) if (sl[i] >= guardArc && sl[i] <= Ls - guardArc) mid.push(P[i]);

      if (Ls < guardArc * 2.2 || mid.length < 3) {
        info.isLine = true;
        const d = Math.hypot(P[q - 1][0] - P[0][0], P[q - 1][1] - P[0][1]) || 1;
        info.A = info.B = { c: P[0], d: [(P[q - 1][0] - P[0][0]) / d, (P[q - 1][1] - P[0][1]) / d] };
        return info;
      }

      const tlsAll = ImageVectorizer.fitLineTLS(mid);
      if (tlsAll && tlsAll.dev <= lineTol) {
        let maxAll = 0;
        for (const p of P) {
          maxAll = Math.max(maxAll, Math.abs(-(p[0] - tlsAll.c[0]) * tlsAll.d[1] + (p[1] - tlsAll.c[1]) * tlsAll.d[0]));
        }
        if (maxAll <= lineTol * 1.65) {
          info.isLine = true;
          info.A = info.B = tlsAll;
          return info;
        }
      }

      const tanLen = Math.min(wChord * 2.4, Ls * 0.42);
      const startSeg = [], endSeg = [];
      for (let i = 0; i < q; i++) {
        if (sl[i] >= guardArc * 0.5 && sl[i] <= guardArc * 0.5 + tanLen) startSeg.push(P[i]);
        if (sl[i] <= Ls - guardArc * 0.5 && sl[i] >= Ls - guardArc * 0.5 - tanLen) endSeg.push(P[i]);
      }
      if (startSeg.length >= 3) info.A = ImageVectorizer.fitLineTLS(startSeg);
      if (endSeg.length >= 3) info.B = ImageVectorizer.fitLineTLS(endSeg);
      return info;
    });

    // Snap corners precisely to line-line intersections!
    const dMaxSnap = Math.max(4.5, wChord * 2.5);
    const snappedCorners = [];
    for (let j = 0; j < m; j++) {
      const pc = rawSpans[j][0];
      const prevInfo = spanInfos[(j + m - 1) % m];
      const nextInfo = spanInfos[j];
      let X = null;
      if (prevInfo.B && nextInfo.A) {
        X = ImageVectorizer.intersectLines(prevInfo.B, nextInfo.A);
        if (X && Math.hypot(X[0] - pc[0], X[1] - pc[1]) > dMaxSnap) X = null;
      }
      snappedCorners.push(X || pc);
    }

    const fitErrSq = Math.pow(Math.max(0.28, (1.05 - opts.detailFactor) * 1.45), 2);
    const segmentsList = [];

    for (let j = 0; j < m; j++) {
      const info = spanInfos[j];
      const ca = snappedCorners[j];
      const cb = snappedCorners[(j + 1) % m];

      if (info.isLine) {
        segmentsList.push({
          cubics: [[ca, null, null, cb]],
          cornerStart: true,
        });
        continue;
      }

      const P = info.P, q = P.length;
      const da = [ca[0] - P[0][0], ca[1] - P[0][1]];
      const db = [cb[0] - P[q - 1][0], cb[1] - P[q - 1][1]];
      const blendSpan = Math.min(info.Ls * 0.45, wChord * 2.5);
      const pinned = P.map((p, idx) => {
        const wa = blendSpan > 0 ? Math.max(0, 1 - info.sl[idx] / blendSpan) : 0;
        const wb = blendSpan > 0 ? Math.max(0, 1 - (info.Ls - info.sl[idx]) / blendSpan) : 0;
        return [p[0] + da[0] * wa + db[0] * wb, p[1] + da[1] * wa + db[1] * wb];
      });
      pinned[0] = ca;
      pinned[q - 1] = cb;

      const smPts = ImageVectorizer.taubinSmoothOpen(pinned, 2);
      const kTan = Math.min(q - 1, 3);
      const tHat1 = ImageVectorizer.normVec([smPts[kTan][0] - smPts[0][0], smPts[kTan][1] - smPts[0][1]]);
      const tHat2 = ImageVectorizer.normVec([smPts[q - 1 - kTan][0] - smPts[q - 1][0], smPts[q - 1 - kTan][1] - smPts[q - 1][1]]);
      const cubics = ImageVectorizer.fitCubicSpan(smPts, 0, smPts.length - 1, tHat1, tHat2, fitErrSq);
      segmentsList.push({ cubics, cornerStart: true });
    }

    return ImageVectorizer.cubicsToAnchors(segmentsList);
  }

  static normVec(v) {
    const l = Math.hypot(v[0], v[1]) || 1;
    return [v[0] / l, v[1] / l];
  }

  /**
   * Flatten span chains of cubic Beziers / lines cleanly into Closed Illustrator Anchor Nodes
   * `{ x, y, corner, hIn, hOut }` in pixel space.
   */
  static cubicsToAnchors(segmentsList) {
    const anchors = [];
    for (let sIdx = 0; sIdx < segmentsList.length; sIdx++) {
      const { cubics, cornerStart } = segmentsList[sIdx];
      for (let cIdx = 0; cIdx < cubics.length; cIdx++) {
        const [p0, cp1, cp2, p3] = cubics[cIdx];
        if (anchors.length === 0) {
          anchors.push({
            x: p0[0],
            y: p0[1],
            corner: !!cornerStart,
            hIn: null,
            hOut: cp1 ? { x: cp1[0], y: cp1[1] } : null,
          });
        } else {
          const prevAnchor = anchors[anchors.length - 1];
          prevAnchor.hOut = cp1 ? { x: cp1[0], y: cp1[1] } : null;
        }

        const isLastOfWholeLoop = (sIdx === segmentsList.length - 1) && (cIdx === cubics.length - 1);
        if (isLastOfWholeLoop) {
          anchors[0].hIn = cp2 ? { x: cp2[0], y: cp2[1] } : null;
        } else {
          const isNextCorner = (cIdx === cubics.length - 1) && (segmentsList[sIdx + 1]?.cornerStart);
          anchors.push({
            x: p3[0],
            y: p3[1],
            corner: !!isNextCorner,
            hIn: cp2 ? { x: cp2[0], y: cp2[1] } : null,
            hOut: null,
          });
        }
      }
    }
    return anchors.length >= 2 ? anchors : null;
  }

  /**
   * Transform pixel-space anchors to millimeter CAD coordinates & compute inscribed radius for exact 3D height.
   */
  static anchorsToMm(anchors, bounds, scaleW, scaleH) {
    const sx = bounds.w / scaleW;
    const sy = bounds.h / scaleH;
    const mapPt = (p) => ({
      x: Number((bounds.x + p.x * sx).toFixed(3)),
      y: Number((bounds.y + p.y * sy).toFixed(3)),
    });
    return anchors.map((a) => {
      const base = mapPt(a);
      return {
        x: base.x,
        y: base.y,
        corner: !!a.corner,
        hIn: a.hIn ? mapPt(a.hIn) : null,
        hOut: a.hOut ? mapPt(a.hOut) : null,
      };
    });
  }

  /**
   * Topology Hierarchy + Potrace/Bezier Fitter + Compound Solid+Holes CAD Entity Builder!
   */
  static vectorizeMaskLoops(rawLoops, rawField, w, h, refSign, opts) {
    if (!rawLoops || rawLoops.length === 0) return [];
    const {
      minAreaPx = 10,
      minSpanMm = 0.9,
      maxContours = 180,
      bounds,
      rejectFullFrame = true,
      fillColor = '#38bdf8',
      strokeColor = '#38bdf8',
      layerGroup = 'Auto-Trace',
      namePrefix = 'Design Layer',
      toneLevel = 1,
      toneTotal = 1,
      heightMm = 4.0,
      shape = 'dome',
      mode = 'max',
      adaptiveRadius = true,
    } = opts;

    const sx = bounds.w / w;
    const sy = bounds.h / h;

    // 1. Filter noise & compute tight bounds + area for topological hole nesting
    const filtered = [];
    for (const L of rawLoops) {
      const a = Math.abs(ImageVectorizer.loopSignedArea(L));
      if (a < minAreaPx) continue;
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const p of L) {
        if (p[0] < minX) minX = p[0]; if (p[0] > maxX) maxX = p[0];
        if (p[1] < minY) minY = p[1]; if (p[1] > maxY) maxY = p[1];
      }
      const spanW = (maxX - minX) * sx;
      const spanH = (maxY - minY) * sy;
      if (spanW < minSpanMm && spanH < minSpanMm) continue;
      if (rejectFullFrame && spanW > bounds.w * 0.96 && spanH > bounds.h * 0.96) continue;
      filtered.push({ L, area: a, bb: [minX, minY, maxX, maxY], spanW, spanH });
    }
    if (filtered.length === 0) return [];

    // Sort descending by area so parents ALWAYS precede children
    filtered.sort((a, b) => b.area - a.area);

    // 2. Exact Point-in-Polygon containment hierarchy (even depth = solid, odd depth = hole)
    const nF = filtered.length;
    const parentIdx = new Int32Array(nF).fill(-1);
    const depth = new Int32Array(nF);

    for (let i = 0; i < nF; i++) {
      const pt = filtered[i].L[0];
      let d = 0;
      let directParent = -1;
      for (let j = i - 1; j >= 0; j--) {
        const qb = filtered[j].bb;
        if (pt[0] < qb[0] || pt[0] > qb[2] || pt[1] < qb[1] || pt[1] > qb[3]) continue;
        if (ImageVectorizer.pointInLoop(pt[0], pt[1], filtered[j].L)) {
          d++;
          if (directParent === -1) directParent = j;
        }
      }
      depth[i] = d;
      parentIdx[i] = directParent;
    }

    // 3. Fit exact Illustrator Cubic Bezier anchors per loop & attach odd-depth holes to parent solids!
    const layerPalette = ['#38bdf8', '#34d399', '#fbbf24', '#f472b6', '#a78bfa', '#22d3ee', '#fb923c'];
    const solidMap = new Map();
    const solids = [];

    for (let i = 0; i < nF; i++) {
      const item = filtered[i];
      const anchorsPx = ImageVectorizer.fitIllustratorCurve(item.L, rawField, w, h, refSign, opts);
      if (!anchorsPx || anchorsPx.length < 2) continue;
      const mmAnchors = ImageVectorizer.anchorsToMm(anchorsPx, bounds, w, h);

      // Estimate true feature half-thickness (2 * Area / Perimeter) in mm for responsive 3D bevel radius
      let perimPx = 0;
      for (let k = 0; k < item.L.length; k++) {
        const p0 = item.L[k], p1 = item.L[(k + 1) % item.L.length];
        perimPx += Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
      }
      const featureRadiusMm = Math.max(0.8, Math.min(18.0, (2 * item.area * sx * sy) / (Math.max(1, perimPx) * Math.min(sx, sy))));

      if ((depth[i] & 1) === 1 && parentIdx[i] !== -1 && solidMap.has(parentIdx[i])) {
        // Inner Hole: attach directly onto parent solid entity so evenodd fill & 3D EDT punch it cleanly!
        const parentEnt = solidMap.get(parentIdx[i]);
        parentEnt.holes.push(mmAnchors);
        continue;
      }

      if (solids.length >= maxContours) continue;

      const idx = solids.length;
      const usePaletteColor = fillColor === '#38bdf8' && layerGroup === 'BW Vector' ? layerPalette[idx % layerPalette.length] : strokeColor;
      const ent = {
        type: 'curve',
        points: mmAnchors,
        holes: [],
        isClosed: true,
        width: 1.5,
        color: usePaletteColor,
        fillColor: fillColor || usePaletteColor,
        layerGroup,
        toneLevel,
        toneTotal,
        name: `${namePrefix} #${idx + 1}`,
        source: 'auto-trace',
        visible: true,
        locked: false,
        relief: {
          enabled: true,
          shape,
          heightMm,
          radiusMm: Number(featureRadiusMm.toFixed(2)),
          adaptiveRadius,
          mode,
          openAsStroke: false,
        },
      };

      solidMap.set(i, ent);
      solids.push(ent);
    }

    return solids;
  }

  /**
   * Backwards-compatible helper: 3D Relief Heightmap -> Separate editable 2D Vector Contour Layers
   */
  static traceReliefToVectors(reliefEngine, options = {}) {
    const {
      levels = [0.15, 0.45, 0.75],
      maxContours = 120,
    } = options;

    const N = reliefEngine.resolution;
    const bounds = { x: 0, y: 0, w: reliefEngine.width, h: reliefEngine.height };
    const allEntities = [];
    const colors = ['#38bdf8', '#34d399', '#f472b6', '#fbbf24'];

    const scalar = new Float32Array(N * N);
    for (let i = 0; i < N * N; i++) scalar[i] = -reliefEngine.data[i];

    levels.forEach((lvl, idx) => {
      const binary = new Uint8Array(N * N);
      for (let i = 0; i < N * N; i++) {
        binary[i] = reliefEngine.data[i] >= lvl ? 1 : 0;
      }

      const rawLoops = ImageVectorizer.marchSquares(binary, N, N, scalar, -lvl);
      const contours = ImageVectorizer.vectorizeMaskLoops(rawLoops, scalar, N, N, -1, {
        detailFactor: 0.65,
        corners: 0.55,
        refineEdges: false,
        minAreaPx: 8,
        minSpanMm: 1.4,
        maxContours: Math.floor(maxContours / levels.length),
        scaleRatio: N / Math.max(reliefEngine.width, reliefEngine.height, 1),
        bounds,
        rejectFullFrame: false,
        fillColor: colors[idx % colors.length],
        strokeColor: colors[idx % colors.length],
        layerGroup: `3D Slice (${Math.round(lvl * 100)}%)`,
        namePrefix: `3D Slice (${Math.round(lvl * 100)}%)`,
        toneLevel: idx + 1,
        toneTotal: levels.length,
        heightMm: Number((lvl * (reliefEngine.maxDepth || 8)).toFixed(2)),
        shape: 'smooth',
        adaptiveRadius: true,
      });
      contours.forEach((e) => { e.source = '3d-extract'; });
      allEntities.push(...contours);
    });

    return allEntities;
  }

  /**
   * Otsu threshold + Histogram Separability Metric (used by ReliefEngine too).
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
}
