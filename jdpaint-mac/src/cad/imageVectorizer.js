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
    const detailHint = typeof options === 'object' && typeof options.detail === 'number' ? options.detail : 0.7;
    const maxDimension = (typeof options === 'object' && options.maxDimension) || Math.min(1152, Math.round(820 + detailHint * 332));

    let scaleW = maxDimension;
    let scaleH = Math.round((ih / iw) * maxDimension);
    if (scaleH > maxDimension) {
      scaleH = maxDimension;
      scaleW = Math.round((iw / ih) * maxDimension);
    }
    scaleW = Math.max(120, Math.min(1152, scaleW));
    scaleH = Math.max(120, Math.min(1152, scaleH));

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
   * Step-by-step asynchronous trace for responsive browser UI with live phase updates.
   * Stages: 1=Prepare ROI, 2=Separate Relief/Color Channels, 3=Extract Contours & Holes,
   * 4=Fit CAD Lines & Smooth Beziers, 5=Finalize Layered Entities.
   */
  static async traceImageToVectorsAsync(img, widthMm = 100, heightMm = 100, options = {}, onProgress = null) {
    const yieldFrame = () => new Promise((resolve) => setTimeout(resolve, 16));
    const report = async (step, total, label, pct = null) => {
      const resolvedPct = pct ?? Math.round((step / total) * 100);
      if (typeof onProgress === 'function') onProgress({ step, total, pct: resolvedPct, label });
      await yieldFrame();
    };

    await report(1, 5, 'Step 1/5 (12%): High-Res Supersampling & Illumination Normalization...', 12);
    const iw = img.naturalWidth || img.width || 1;
    const ih = img.naturalHeight || img.height || 1;
    const detailHint = typeof options === 'object' && typeof options.detail === 'number' ? options.detail : 0.7;
    const maxDimension = (typeof options === 'object' && options.maxDimension) || Math.min(1408, Math.round(920 + detailHint * 488));

    let scaleW = maxDimension;
    let scaleH = Math.round((ih / iw) * maxDimension);
    if (scaleH > maxDimension) {
      scaleH = maxDimension;
      scaleW = Math.round((iw / ih) * maxDimension);
    }
    scaleW = Math.max(120, Math.min(1408, scaleW));
    scaleH = Math.max(120, Math.min(1408, scaleH));

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

    await report(2, 5, 'Step 2/5 (34%): Edge-Preserving Denoise & Relief Channel Separation...', 34);
    const vectors = ImageVectorizer.traceFromImageData(pixels, scaleW, scaleH, bounds, options);

    await report(3, 5, `Step 3/5 (62%): Subpixel Contour Extraction & Nested Hole Hierarchy (${vectors.length} regions)...`, 62);
    await report(4, 5, 'Step 4/5 (84%): Fitting Corner-Exact Cubic Beziers & Eliminating Self-Intersections...', 84);

    // Sort clean architectural layout order: larger borders/grooves before interior sculpted petals
    const sorted = [...vectors].sort((a, b) => {
      const la = a.toneLevel || 1;
      const lb = b.toneLevel || 1;
      if (la !== lb) return la - lb;
      return (b.points?.length || 0) - (a.points?.length || 0);
    });

    await report(5, 5, `Step 5/5 (100%): Finalized ${sorted.length} precision 2D CAD vector layer(s)!`, 100);
    return sorted;
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
      maxContours = 480,
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

    // Resolve smart auto preset just like Adobe Illustrator + Monochrome/Architectural Door Wood Detection
    let effectiveMode = traceMode;
    if (effectiveMode === 'auto') {
      const autoRoi = ignoreBg && Math.min(scaleW, scaleH) >= 96
        ? ImageVectorizer.detectFurnitureROI(rArr, gArr, bArr, lumaRaw, scaleW, scaleH)
        : null;
      const isArchitecturalPanel = !!(autoRoi && autoRoi.hasExternalBg && autoRoi.chromaMed >= 10 && autoRoi.w >= scaleW * 0.25 && autoRoi.h >= scaleH * 0.40);
      if (otsu.separability >= 0.78 && extremeRatio > 0.72 && chromaSpread < 0.12) {
        effectiveMode = 'bw-logo';
      } else if (isArchitecturalPanel || (meanChroma >= 0.14 && chromaSpread < 0.16)) {
        effectiveMode = 'wood';
      } else if (chromaSpread >= 0.16) {
        effectiveMode = 'color';
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

    // Smooth edge-preserving Guided Denoise (radius=1 at normal/high detail to keep tight petals crisp!)
    const filterRadius = detailFactor >= 0.45 ? 1 : 2;
    const denoisedLuma = ImageVectorizer.guidedFilter(lumaRaw, scaleW, scaleH, filterRadius, effectiveMode === 'bw-logo' ? 85 : 145);

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
      ImageVectorizer.removeCheckerboardPinches(cleanedMask, w, h, true);
      const { phi } = ImageVectorizer.computeSignedDistancePhi(cleanedMask, w, h, 0.85);
      const smoothScalar = ImageVectorizer.gaussianBlur3x3(scalar, w, h);
      const blendedScalar = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        // Blend continuous color distance with smooth Felzenszwalb L2 signed distance (0-crossing inside boundary)
        blendedScalar[i] = smoothScalar[i] * 0.45 - phi[i] * 0.85;
      }
      const rawLoops = ImageVectorizer.marchSquares(cleanedMask, w, h, blendedScalar, 0.0);

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
      ImageVectorizer.removeCheckerboardPinches(cleaned, w, h, false);
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
   * Automatically isolate the furniture/panel body from exterior walls, studio margins, floors, bricks, plants & corner logos.
   */
  static detectFurnitureROI(rArr, gArr, bArr, lumaRaw, w, h) {
    const n = w * h;
    const L = new Float32Array(n);
    const A = new Float32Array(n);
    const B = new Float32Array(n);

    // Sample interior central zone [28..72% W, 18..78% H] to learn panel surface tone
    const cx0 = Math.floor(w * 0.28), cx1 = Math.ceil(w * 0.72);
    const cy0 = Math.floor(h * 0.18), cy1 = Math.ceil(h * 0.78);
    const aVals = [], bVals = [];
    for (let i = 0; i < n; i++) {
      const [lv, av, bv] = ImageVectorizer.rgbToLab(rArr[i], gArr[i], bArr[i]);
      L[i] = lv; A[i] = av; B[i] = bv;
      const x = i % w, y = (i - x) / w;
      if (x >= cx0 && x < cx1 && y >= cy0 && y < cy1 && (i & 1) === 0) {
        aVals.push(av); bVals.push(bv);
      }
    }
    aVals.sort((u, v) => u - v);
    bVals.sort((u, v) => u - v);
    const aMed = aVals[Math.floor(aVals.length * 0.5)] || 0;
    const bMed = bVals[Math.floor(bVals.length * 0.5)] || 0;

    const lVals = [];
    for (let y = cy0; y < cy1; y += 2) {
      for (let x = cx0; x < cx1; x += 2) {
        const i = y * w + x;
        if (Math.hypot(A[i] - aMed, B[i] - bMed) <= 22) lVals.push(L[i]);
      }
    }
    lVals.sort((u, v) => u - v);
    const lMed = lVals.length > 0 ? lVals[Math.floor(lVals.length * 0.5)] : 50;

    // Classify pixels consistent with furniture panel tone & carved undercuts
    const panelMask = new Uint8Array(n);
    const chromaDiff = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const dc = Math.hypot(A[i] - aMed, B[i] - bMed);
      chromaDiff[i] = dc;
      const dl = Math.abs(L[i] - lMed);
      const tolC = L[i] < 28 ? 34 : 20;
      if (dc <= tolC && dl <= 36) panelMask[i] = 1;
    }

    // Smooth column profile across middle vertical band [15..85% H] and walk outward from center
    const colFrac = new Float64Array(w);
    const yStart = Math.floor(h * 0.15), yEnd = Math.ceil(h * 0.85);
    const ySpan = Math.max(1, yEnd - yStart);
    for (let x = 0; x < w; x++) {
      let c = 0;
      for (let y = yStart; y < yEnd; y++) if (panelMask[y * w + x]) c++;
      colFrac[x] = c / ySpan;
    }
    const smoothCol = new Float64Array(w);
    for (let x = 0; x < w; x++) {
      let s = 0, cnt = 0;
      for (let dx = -3; dx <= 3; dx++) {
        const xx = x + dx;
        if (xx >= 0 && xx < w) { s += colFrac[xx]; cnt++; }
      }
      smoothCol[x] = s / cnt;
    }
    const midX = Math.floor(w * 0.5);
    let rx0 = midX, rx1 = midX;
    while (rx0 > 0 && smoothCol[rx0 - 1] >= 0.35) rx0--;
    while (rx1 < w - 1 && smoothCol[rx1 + 1] >= 0.35) rx1++;

    // Fallback if center band is mostly dark carving: check widest contiguous band >= 0.32
    if (rx1 - rx0 < w * 0.28) {
      rx0 = 0; rx1 = w - 1;
      for (let x = 0; x < midX; x++) { if (smoothCol[x] >= 0.30) { rx0 = x; break; } }
      for (let x = w - 1; x > midX; x--) { if (smoothCol[x] >= 0.30) { rx1 = x; break; } }
    }

    // Compute row profile strictly within [rx0..rx1] and walk outward from center Y
    const rowFrac = new Float64Array(h);
    const xSpan = Math.max(1, rx1 - rx0 + 1);
    for (let y = 0; y < h; y++) {
      let c = 0;
      const row = y * w;
      for (let x = rx0; x <= rx1; x++) if (panelMask[row + x]) c++;
      rowFrac[y] = c / xSpan;
    }
    const smoothRow = new Float64Array(h);
    for (let y = 0; y < h; y++) {
      let s = 0, cnt = 0;
      for (let dy = -3; dy <= 3; dy++) {
        const yy = y + dy;
        if (yy >= 0 && yy < h) { s += rowFrac[yy]; cnt++; }
      }
      smoothRow[y] = s / cnt;
    }
    const midY = Math.floor(h * 0.5);
    let ry0 = midY, ry1 = midY;
    while (ry0 > 0 && smoothRow[ry0 - 1] >= 0.35) ry0--;
    while (ry1 < h - 1 && smoothRow[ry1 + 1] >= 0.35) ry1++;

    if (ry1 - ry0 < h * 0.28) {
      ry0 = 0; ry1 = h - 1;
      for (let y = 0; y < midY; y++) { if (smoothRow[y] >= 0.30) { ry0 = y; break; } }
      for (let y = h - 1; y > midY; y--) { if (smoothRow[y] >= 0.30) { ry1 = y; break; } }
    }

    // Inset minimally ONLY on sides that had detected exterior walls/margins so outer door rails stay intact!
    const hasLeftBg = rx0 > Math.max(4, w * 0.03);
    const hasRightBg = rx1 < w - Math.max(5, w * 0.03);
    const hasTopBg = ry0 > Math.max(4, h * 0.03);
    const hasBottomBg = ry1 < h - Math.max(5, h * 0.03);
    const edgeGuardX = Math.max(4, Math.round(w * 0.010));
    const edgeGuardY = Math.max(4, Math.round(h * 0.010));
    const insetX = Math.max(4, Math.round((rx1 - rx0) * 0.022));
    const insetY = Math.max(4, Math.round((ry1 - ry0) * 0.016));

    const ix0 = hasLeftBg ? Math.min(rx1 - 10, rx0 + insetX) : edgeGuardX;
    const ix1 = hasRightBg ? Math.max(ix0 + 10, rx1 - insetX) : w - 1 - edgeGuardX;
    const iy0 = hasTopBg ? Math.min(ry1 - 10, ry0 + insetY) : edgeGuardY;
    const iy1 = hasBottomBg ? Math.max(iy0 + 10, ry1 - insetY) : h - 1 - edgeGuardY;

    return {
      x0: ix0, y0: iy0, x1: ix1, y1: iy1,
      w: ix1 - ix0 + 1, h: iy1 - iy0 + 1,
      hasExternalBg: hasLeftBg || hasRightBg || hasTopBg || hasBottomBg,
      lMed,
      chromaMed: Math.hypot(aMed, bMed),
      L, chromaDiff, panelMask,
    };
  }

  /**
   * Separable masked normalized box average (only averages pixels inside mask[i] === 1).
   */
  static maskedBoxMean(src, mask, w, h, radius) {
    const n = w * h;
    const valH = new Float32Array(n);
    const cntH = new Float32Array(n);
    const prefV = new Float64Array(w + 1);
    const prefC = new Int32Array(w + 1);

    for (let y = 0; y < h; y++) {
      const row = y * w;
      prefV[0] = 0; prefC[0] = 0;
      for (let x = 0; x < w; x++) {
        const idx = row + x;
        const m = mask[idx];
        prefV[x + 1] = prefV[x] + (m ? src[idx] : 0);
        prefC[x + 1] = prefC[x] + m;
      }
      for (let x = 0; x < w; x++) {
        const l = Math.max(0, x - radius);
        const r = Math.min(w - 1, x + radius);
        valH[row + x] = prefV[r + 1] - prefV[l];
        cntH[row + x] = prefC[r + 1] - prefC[l];
      }
    }

    const out = new Float32Array(n);
    const colV = new Float64Array(h + 1);
    const colC = new Float64Array(h + 1);
    for (let x = 0; x < w; x++) {
      colV[0] = 0; colC[0] = 0;
      for (let y = 0; y < h; y++) {
        const idx = y * w + x;
        colV[y + 1] = colV[y] + valH[idx];
        colC[y + 1] = colC[y] + cntH[idx];
      }
      for (let y = 0; y < h; y++) {
        const t = Math.max(0, y - radius);
        const b = Math.min(h - 1, y + radius);
        const c = colC[b + 1] - colC[t];
        out[y * w + x] = c > 0 ? (colV[b + 1] - colV[t]) / c : src[y * w + x];
      }
    }
    return out;
  }

  static separableExtrema(src, w, h, radius, isMax = true) {
    const n = w * h;
    const temp = new Float32Array(n);
    const out = new Float32Array(n);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        let best = src[row + x];
        const l = Math.max(0, x - radius), r = Math.min(w - 1, x + radius);
        if (isMax) {
          for (let xx = l; xx <= r; xx++) if (src[row + xx] > best) best = src[row + xx];
        } else {
          for (let xx = l; xx <= r; xx++) if (src[row + xx] < best) best = src[row + xx];
        }
        temp[row + x] = best;
      }
    }
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        let best = temp[y * w + x];
        const t = Math.max(0, y - radius), b = Math.min(h - 1, y + radius);
        if (isMax) {
          for (let yy = t; yy <= b; yy++) {
            const v = temp[yy * w + x];
            if (v > best) best = v;
          }
        } else {
          for (let yy = t; yy <= b; yy++) {
            const v = temp[yy * w + x];
            if (v < best) best = v;
          }
        }
        out[y * w + x] = best;
      }
    }
    return out;
  }

  /**
   * Preset D: Multi-Scale Morphological White-Hat / Black-Hat Wood Relief + High-Contrast Engraved Isolator.
   * Detects 2-tone burnt/engraved wood logos directly as sharp solid shapes AND separates 3D bas-relief
   * frames, stems, petals & louvers cleanly without square block artifacts.
   */
  static traceWoodReliefLayers(rArr, gArr, bArr, lumaRaw, denoisedLuma, otsu, w, h, opts) {
    const n = w * h;
    const roi = ImageVectorizer.detectFurnitureROI(rArr, gArr, bArr, lumaRaw, w, h);

    // 1. Suppress exterior background & light/dark overlay watermarks + inpaint before morphology
    const rawRoiMask = new Uint8Array(n);
    const chromaMask = new Float32Array(n);
    const watermarkTol = roi.chromaMed > 14 ? 15 : 22;
    const darkInkCand = new Uint8Array(n);
    let darkInkCnt = 0;
    let roiPixCnt = 0;
    let suppressedCount = 0;
    for (let y = roi.y0; y <= roi.y1; y++) {
      const row = y * w;
      for (let x = roi.x0; x <= roi.x1; x++) {
        const i = row + x;
        roiPixCnt++;
        const dc = roi.chromaDiff[i];
        const rgbSat = Math.max(rArr[i], gArr[i], bArr[i]) - Math.min(rArr[i], gArr[i], bArr[i]);
        const isNeutralOverlay = roi.chromaMed > 14 && roi.L[i] > 84 && rgbSat < 18;
        if (isNeutralOverlay || (dc > watermarkTol && (roi.L[i] > roi.lMed + 8 || dc > 23))) {
          chromaMask[i] = (isNeutralOverlay || dc > watermarkTol + 4) ? 0 : Math.max(0, 1.0 - (dc - watermarkTol) / 4.0);
          rawRoiMask[i] = chromaMask[i] > 0.25 ? 1 : 0;
          if (!rawRoiMask[i]) suppressedCount++;
        } else {
          if (roi.chromaMed >= 14) {
            const isNeutralBlackInk = lumaRaw[i] < 40 && Math.abs(rArr[i] - bArr[i]) < 15 && dc > 11;
            const isLogoAccent = (rArr[i] - gArr[i] > 42 && bArr[i] > gArr[i] + 10 && dc > 18)
              || (rArr[i] - gArr[i] > 58 && rArr[i] - bArr[i] > 48 && dc > 22);
            if (isNeutralBlackInk || isLogoAccent) {
              darkInkCand[i] = 1;
              darkInkCnt++;
            }
          }
          rawRoiMask[i] = 1;
          chromaMask[i] = 1.0;
        }
      }
    }
    if (darkInkCnt > 0 && darkInkCnt < Math.max(24, roiPixCnt * 0.045)) {
      for (let i = 0; i < n; i++) {
        if (darkInkCand[i]) {
          rawRoiMask[i] = 0;
          chromaMask[i] = 0;
          suppressedCount++;
        }
      }
    }
    // 2px dilation around watermark patches so anti-aliased edges never trace false rectangular rings
    const roiMask = new Uint8Array(rawRoiMask);
    if (suppressedCount > 0) {
      for (let y = roi.y0 + 2; y <= roi.y1 - 2; y++) {
        const row = y * w;
        for (let x = roi.x0 + 2; x <= roi.x1 - 2; x++) {
          const i = row + x;
          if (!rawRoiMask[i]) continue;
          if (!rawRoiMask[i - 1] || !rawRoiMask[i + 1] || !rawRoiMask[i - w] || !rawRoiMask[i + w]
            || !rawRoiMask[i - 2] || !rawRoiMask[i + 2] || !rawRoiMask[i - 2 * w] || !rawRoiMask[i + 2 * w]) {
            roiMask[i] = 0;
            chromaMask[i] = 0;
          }
        }
      }
    }

    const sx = opts.bounds.w / w;
    const sy = opts.bounds.h / h;
    const roiMaxSpanW = roi.w * sx * 0.94;
    const roiMaxSpanH = roi.h * sy * 0.94;
    const roiEdgeLeftMm = opts.bounds.x + (roi.x0 + 4) * sx;
    const roiEdgeRightMm = opts.bounds.x + (roi.x1 - 4) * sx;
    const roiEdgeTopMm = opts.bounds.y + (roi.y0 + 4) * sy;
    const roiEdgeBotMm = opts.bounds.y + (roi.y1 - 4) * sy;
    const minWoodSpanMm = Math.max(opts.minSpanMm || 0.9, 1.15);
    const minFeatureAreaPx = Math.max(opts.minAreaPx || 14, Math.round(roi.w * roi.h * 0.00032));

    const filterWoodEnts = (ents, { filterGrainStreaks = false } = {}) => ents.filter((ent) => {
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const p of ent.points) {
        if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
      }
      const ew = maxX - minX, eh = maxY - minY;
      if (ew > roiMaxSpanW && eh > roiMaxSpanH) return false;
      if ((minX <= roiEdgeLeftMm || maxX >= roiEdgeRightMm) && ew < roi.w * sx * 0.04 && eh > roi.h * sy * 0.35) return false;
      if ((minY <= roiEdgeTopMm || maxY >= roiEdgeBotMm) && eh < roi.h * sy * 0.04 && ew > roi.w * sx * 0.35) return false;
      if (filterGrainStreaks && ew >= eh * 3.8 && eh < roi.h * sy * 0.045 && (minX <= opts.bounds.x + 20 * sx || maxX >= opts.bounds.x + (w - 20) * sx)) return false;
      if (filterGrainStreaks && (minX <= opts.bounds.x + (roi.x0 + 48) * sx || maxX >= opts.bounds.x + (roi.x1 - 48) * sx) && ew * eh < opts.bounds.w * opts.bounds.h * 0.0025 && (!ent.holes || ent.holes.length === 0)) return false;
      const aspectRatio = Math.max(ew, eh) / Math.max(0.25, Math.min(ew, eh));
      if (!ent.holes?.length && aspectRatio >= 5.2 && Math.min(ew, eh) < Math.min(roi.w * sx, roi.h * sy) * 0.032) return false;
      const isOuterSpeck = (minX <= opts.bounds.x + (roi.x0 + 18) * sx || maxX >= opts.bounds.x + (roi.x1 - 18) * sx || minY <= opts.bounds.y + (roi.y0 + 18) * sy || maxY >= opts.bounds.y + (roi.y1 - 18) * sy)
        && ew * eh < roi.w * sx * roi.h * sy * 0.0022 && (!ent.holes || ent.holes.length === 0);
      if (isOuterSpeck) return false;
      return true;
    });

    // 2. Replace watermark cutouts with neutral local panel tone BEFORE morphology, then crisp radius-1 denoise
    let preMorphLuma = denoisedLuma;
    if (suppressedCount > 0) {
      const inpaintMean = ImageVectorizer.maskedBoxMean(denoisedLuma, roiMask, w, h, Math.max(8, Math.round(roi.w * 0.032)));
      preMorphLuma = new Float32Array(denoisedLuma);
      for (let y = roi.y0; y <= roi.y1; y++) {
        const row = y * w;
        for (let x = roi.x0; x <= roi.x1; x++) {
          const i = row + x;
          if (!roiMask[i]) preMorphLuma[i] = inpaintMean[i];
        }
      }
    }
    const smoothLuma = ImageVectorizer.guidedFilter(preMorphLuma, w, h, 1, 75);
    const innerPad = 4;
    const xMin = Math.max(1, roi.x0 + innerPad), xMax = Math.min(w - 2, roi.x1 - innerPad);
    const yMin = Math.max(1, roi.y0 + innerPad), yMax = Math.min(h - 2, roi.y1 - innerPad);

    // Check if wood ROI is a high-contrast 2-tone engraved design (e.g. laser/burnt wood sign) vs 3D bas-relief carving
    const roiHist = new Uint32Array(256);
    let roiCnt = 0;
    for (let y = yMin; y <= yMax; y += 2) {
      const row = y * w;
      for (let x = xMin; x <= xMax; x += 2) {
        const i = row + x;
        if (!roiMask[i]) continue;
        roiHist[Math.max(0, Math.min(255, Math.round(smoothLuma[i])))]++;
        roiCnt++;
      }
    }
    const roiOtsu = roiCnt > 60 ? ImageVectorizer.computeOtsuStats(roiHist, roiCnt) : { threshold: 128, separability: 0, m1: 100, m2: 120 };
    if (opts.threshold <= 0 && roiOtsu.separability >= 0.80 && Math.abs(roiOtsu.m2 - roiOtsu.m1) >= 68) {
      // High-contrast engraved wood plaque: use near-full frame with subtle margin so circular badges never clip top/bottom
      const ex0 = 2, ex1 = w - 3;
      const ey0 = 2, ey1 = h - 3;
      const planeR = Math.max(14, Math.round(Math.min(roi.w, roi.h) * 0.08));
      const bgSeedMask = new Uint8Array(n);
      const isDarkDesign = roiOtsu.m2 > roiOtsu.m1;
      for (let y = ey0; y <= ey1; y++) {
        const row = y * w;
        for (let x = ex0; x <= ex1; x++) {
          const i = row + x;
          if (roi.chromaDiff[i] > 24) continue;
          const isBgTone = isDarkDesign ? (smoothLuma[i] >= roiOtsu.threshold) : (smoothLuma[i] <= roiOtsu.threshold);
          if (isBgTone) bgSeedMask[i] = 1;
        }
      }
      const bgPlane = ImageVectorizer.maskedBoxMean(smoothLuma, bgSeedMask, w, h, planeR);
      const diffField = new Float32Array(n);
      const diffHist = new Uint32Array(256);
      let dCnt = 0;
      for (let y = ey0; y <= ey1; y++) {
        const row = y * w;
        for (let x = ex0; x <= ex1; x++) {
          const i = row + x;
          if (roi.chromaDiff[i] > 24) continue;
          const dVal = isDarkDesign
            ? Math.max(0, bgPlane[i] - smoothLuma[i])
            : Math.max(0, smoothLuma[i] - bgPlane[i]);
          diffField[i] = dVal;
          diffHist[Math.max(0, Math.min(255, Math.round(dVal)))]++;
          dCnt++;
        }
      }
      const dOtsu = dCnt > 40 ? ImageVectorizer.computeOtsuStats(diffHist, dCnt) : { threshold: 38 };
      const dTh = Math.max(38, Math.min(82, dOtsu.threshold * 1.04));
      const engMask = new Uint8Array(n);
      const engScalar = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const solid = diffField[i] >= dTh;
        engMask[i] = solid ? 1 : 0;
        engScalar[i] = -diffField[i];
      }
      const cleanEng = ImageVectorizer.morphClose(engMask, w, h, 0);
      const eLoops = ImageVectorizer.marchSquares(cleanEng, w, h, engScalar, -dTh);
      const engEnts = filterWoodEnts(ImageVectorizer.vectorizeMaskLoops(eLoops, diffField, w, h, -1, {
        ...opts,
        refineEdges: false,
        minAreaPx: Math.max(8, Math.round(roi.w * roi.h * 0.000065)),
        minSpanMm: Math.max(0.45, (opts.minSpanMm || 0.9) * 0.55),
        fillColor: '#fbbf24',
        strokeColor: '#38bdf8',
        layerGroup: 'Engraved Wood Design',
        namePrefix: 'Engraved Motif',
        heightMm: Number((opts.maxDepthMm * 0.55).toFixed(2)),
        shape: 'flat',
        mode: 'max',
        adaptiveRadius: false,
        rejectFullFrame: false,
      }), { filterGrainStreaks: true });
      engEnts.forEach((e) => { if (e.relief) e.relief.radiusMm = 1.6; });
      if (engEnts.length >= 2) return engEnts.slice(0, opts.maxContours);
    }

    // 3. 3D Bas-Relief Furniture Wood Carving Mode: Multi-Scale Gaussian-Smoothed White-Hat + Black-Hat
    const planeR = Math.max(7, Math.round(Math.min(roi.w, roi.h) * 0.042));
    const localPlane = ImageVectorizer.maskedBoxMean(smoothLuma, roiMask, w, h, planeR);

    const minRoi = Math.min(roi.w, roi.h);
    const rFine = Math.max(2, Math.min(3, Math.round(minRoi * 0.0065)));
    const rMed = Math.max(5, Math.min(7, Math.round(minRoi * 0.015)));

    const closedFine = ImageVectorizer.separableExtrema(
      ImageVectorizer.separableExtrema(smoothLuma, w, h, rFine, true), w, h, rFine, false
    );
    const openedFine = ImageVectorizer.separableExtrema(
      ImageVectorizer.separableExtrema(smoothLuma, w, h, rFine, false), w, h, rFine, true
    );
    const closedMed = ImageVectorizer.separableExtrema(
      ImageVectorizer.separableExtrema(smoothLuma, w, h, rMed, true), w, h, rMed, false
    );
    const openedMed = ImageVectorizer.separableExtrema(
      ImageVectorizer.separableExtrema(smoothLuma, w, h, rMed, false), w, h, rMed, true
    );

    const rawWh = new Float32Array(n);
    const rawBh = new Float32Array(n);
    for (let y = yMin; y <= yMax; y++) {
      const row = y * w;
      for (let x = xMin; x <= xMax; x++) {
        const i = row + x;
        if (!roiMask[i]) continue;
        const whF = Math.max(0, smoothLuma[i] - openedFine[i]);
        const whM = Math.max(0, smoothLuma[i] - openedMed[i]);
        const posDiff = Math.max(0, smoothLuma[i] - localPlane[i]);

        const bhF = Math.max(0, closedFine[i] - smoothLuma[i]);
        const bhM = Math.max(0, closedMed[i] - smoothLuma[i]);
        const negDiff = Math.max(0, localPlane[i] - smoothLuma[i]);

        // Keep raised petals crisp against local V-crevices (minimal background mound posDiff!) while grooves preserve channels
        rawWh[i] = Math.min(255, ((whF * 0.68 + whM * 0.32) * 1.62 + posDiff * 0.05) * chromaMask[i]);
        rawBh[i] = Math.min(255, ((bhF * 0.68 + bhM * 0.32) * 1.56 + negDiff * 0.16) * chromaMask[i]);
      }
    }

    // Smooth square morphological box corners slightly into natural rounded organic contours
    const raisedScore = ImageVectorizer.gaussianBlur3x3(rawWh, w, h);
    const grooveScore = ImageVectorizer.gaussianBlur3x3(rawBh, w, h);

    const rSamples = [], gSamples = [];
    for (let y = yMin; y <= yMax; y++) {
      const row = y * w;
      for (let x = xMin; x <= xMax; x++) {
        const i = row + x;
        if (!roiMask[i] || (i & 3) !== 0) continue;
        rSamples.push(raisedScore[i]);
        gSamples.push(grooveScore[i]);
      }
    }

    rSamples.sort((a, b) => a - b);
    gSamples.sort((a, b) => a - b);
    const rFloor = rSamples.length > 20 ? rSamples[Math.floor(rSamples.length * 0.56)] : 4;
    const gFloor = gSamples.length > 20 ? gSamples[Math.floor(gSamples.length * 0.56)] : 4;
    const rCap = rSamples.length > 20 ? Math.max(30, rSamples[Math.floor(rSamples.length * 0.90)]) : 92;
    const gCap = gSamples.length > 20 ? Math.max(30, gSamples[Math.floor(gSamples.length * 0.90)]) : 92;

    const cleanRaised = new Float32Array(n);
    const cleanGroove = new Float32Array(n);
    const rHist = new Uint32Array(256), gHist = new Uint32Array(256);
    let rCnt = 0, gCnt = 0;

    for (let y = yMin; y <= yMax; y++) {
      const row = y * w;
      for (let x = xMin; x <= xMax; x++) {
        const i = row + x;
        const rv = Math.min(255, Math.max(0, (raisedScore[i] - rFloor * 1.24) * 1.8));
        const gv = Math.min(255, Math.max(0, (grooveScore[i] - gFloor * 1.20) * 1.8));
        cleanRaised[i] = rv;
        cleanGroove[i] = gv;
        const rh = Math.min(255, Math.round(Math.min(rCap * 1.8, rv)));
        const gh = Math.min(255, Math.round(Math.min(gCap * 1.8, gv)));
        if (rh > 4) { rHist[rh]++; rCnt++; }
        if (gh > 4) { gHist[gh]++; gCnt++; }
      }
    }

    const rOtsu = rCnt > 40 ? ImageVectorizer.computeOtsuThreshold(rHist, rCnt) : 26;
    const gOtsu = gCnt > 40 ? ImageVectorizer.computeOtsuThreshold(gHist, gCnt) : 26;
    const detailAdj = (0.5 - opts.detailFactor) * 24;
    const rTh = opts.threshold > 0 ? Math.max(10, opts.threshold * 145) : Math.max(14, Math.min(54, rOtsu * 0.62 + detailAdj));
    const gTh = opts.threshold > 0 ? Math.max(10, opts.threshold * 145) : Math.max(14, Math.min(54, gOtsu * 0.62 + detailAdj));

    // Gently smooth scalar channels once before marching contours (eliminates pixel staircases while keeping dark V-crevice separations crisp!)
    const smoothRaised = ImageVectorizer.gaussianBlur3x3(cleanRaised, w, h);
    const smoothGroove = ImageVectorizer.gaussianBlur3x3(cleanGroove, w, h);

    const raisedMask = new Uint8Array(n);
    const raisedScalar = new Float32Array(n);
    const grooveMask = new Uint8Array(n);
    const grooveScalar = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      const isCreviceCut = cleanGroove[i] >= gTh * 0.78 && cleanGroove[i] > cleanRaised[i] * 0.90;
      const rVal = isCreviceCut ? 0 : smoothRaised[i];
      const rSolid = roiMask[i] && rVal >= rTh && !isCreviceCut;
      raisedMask[i] = rSolid ? 1 : 0;
      raisedScalar[i] = isCreviceCut ? 12 : -rVal;

      const gSolid = roiMask[i] && smoothGroove[i] >= gTh && smoothGroove[i] >= smoothRaised[i] * 0.92;
      grooveMask[i] = gSolid ? 1 : 0;
      grooveScalar[i] = -smoothGroove[i];
    }

    // Prune isolated single-pixel wood grain nibs, run Meshy pinch cleanup & L2 SDF smoothing before marching contours
    const cleanR = ImageVectorizer.morphClose(raisedMask, w, h, 0);
    for (let y = yMin; y < yMax; y++) {
      const row = y * w;
      for (let x = xMin; x < xMax; x++) {
        const i = row + x;
        if (!cleanR[i]) continue;
        const n4 = cleanR[i - 1] + cleanR[i + 1] + cleanR[i - w] + cleanR[i + w];
        if (n4 <= 1) cleanR[i] = 0;
      }
    }
    ImageVectorizer.removeCheckerboardPinches(cleanR, w, h, true);
    const { phi: phiR } = ImageVectorizer.computeSignedDistancePhi(cleanR, w, h, 0.85);
    const sdfRaisedScalar = new Float32Array(n);
    for (let i = 0; i < n; i++) sdfRaisedScalar[i] = raisedScalar[i] * 0.65 - phiR[i] * 24.0;
    const rLoops = ImageVectorizer.marchSquares(cleanR, w, h, sdfRaisedScalar, -rTh * 0.65);
    const raisedEnts = filterWoodEnts(ImageVectorizer.vectorizeMaskLoops(rLoops, smoothRaised, w, h, -1, {
      ...opts,
      refineEdges: false,
      minAreaPx: minFeatureAreaPx,
      minSpanMm: minWoodSpanMm,
      maxContours: Math.floor(Math.max(400, opts.maxContours) * 0.68),
      fillColor: '#d9a45b',
      strokeColor: '#38bdf8',
      layerGroup: 'Raised Petals & Motifs',
      namePrefix: 'Carved Motif',
      toneLevel: 2,
      toneTotal: 2,
      heightMm: Number((opts.maxDepthMm * 0.88).toFixed(2)),
      shape: 'dome',
      mode: 'max',
      adaptiveRadius: true,
      rejectFullFrame: true,
    }));

    // Build structural groove mask: bridge 1-2px collinear gaps along rails/louvers & filter ONLY chaotic 3px inter-petal valleys
    const bridgedG = new Uint8Array(grooveMask);
    const trapR = 3;
    for (let y = yMin; y <= yMax; y++) {
      const row = y * w;
      for (let x = xMin; x <= xMax; x++) {
        const i = row + x;
        if (!bridgedG[i] && roiMask[i] && smoothRaised[i] < rTh * 0.72 && smoothGroove[i] >= gTh * 0.36) {
          const hGap = (grooveMask[i - 1] && grooveMask[i + 1]) || (x > 2 && x < w - 3 && grooveMask[i - 2] && grooveMask[i + 2]);
          const vGap = (grooveMask[i - w] && grooveMask[i + w]) || (y > 2 && y < h - 3 && grooveMask[i - 2 * w] && grooveMask[i + 2 * w]);
          if (hGap || vGap) bridgedG[i] = 1;
        }
        if (bridgedG[i]) {
          // Keep clean linear horizontal louvers & vertical panel grooves intact; suppress only tight 3px 4-way pocket blotches
          const isLinearRun = (x > 3 && x < w - 4 && grooveMask[i - 2] && grooveMask[i + 2])
            || (y > 3 && y < h - 4 && grooveMask[i - 2 * w] && grooveMask[i + 2 * w]);
          if (!isLinearRun) {
            let hitL = false, hitR = false, hitU = false, hitD = false;
            for (let d = 1; d <= trapR; d++) {
              if (x - d >= 0 && cleanR[row + x - d]) hitL = true;
              if (x + d < w && cleanR[row + x + d]) hitR = true;
              if (y - d >= 0 && cleanR[(y - d) * w + x]) hitU = true;
              if (y + d < h && cleanR[(y + d) * w + x]) hitD = true;
            }
            if ((hitL && hitR) && (hitU && hitD)) bridgedG[i] = 0;
          }
        }
      }
    }
    const cleanG = ImageVectorizer.morphClose(bridgedG, w, h, 0);
    ImageVectorizer.removeCheckerboardPinches(cleanG, w, h, true);
    const { phi: phiG } = ImageVectorizer.computeSignedDistancePhi(cleanG, w, h, 0.85);
    const sdfGrooveScalar = new Float32Array(n);
    for (let i = 0; i < n; i++) sdfGrooveScalar[i] = grooveScalar[i] * 0.65 - phiG[i] * 24.0;
    const gLoops = ImageVectorizer.marchSquares(cleanG, w, h, sdfGrooveScalar, -gTh * 0.65);
    const grooveEnts = filterWoodEnts(ImageVectorizer.vectorizeMaskLoops(gLoops, smoothGroove, w, h, -1, {
      ...opts,
      refineEdges: false,
      minAreaPx: Math.max(8, Math.round(minFeatureAreaPx * 0.65)),
      minSpanMm: Math.max(0.85, minWoodSpanMm * 0.85),
      maxContours: Math.floor(Math.max(400, opts.maxContours) * 0.45),
      fillColor: '#8c5a2b',
      strokeColor: '#0ea5e9',
      layerGroup: 'Carved Grooves & Panels',
      namePrefix: 'Molding / Groove',
      toneLevel: 1,
      toneTotal: 2,
      heightMm: Number((opts.maxDepthMm * 0.30).toFixed(2)),
      shape: 'smooth',
      mode: 'max',
      adaptiveRadius: true,
      rejectFullFrame: true,
    }));

    // Include true sampled RGB fill tint & clean Panel Perimeter Frame when external room background was stripped
    const combined = [...grooveEnts, ...raisedEnts];
    for (const ent of combined) {
      const pts = ent.points;
      if (!pts || pts.length < 2) continue;
      let cx = 0, cy = 0;
      for (const p of pts) { cx += p.x; cy += p.y; }
      cx /= pts.length; cy /= pts.length;
      const px = Math.max(0, Math.min(w - 1, Math.round((cx - opts.bounds.x) / sx)));
      const py = Math.max(0, Math.min(h - 1, Math.round((cy - opts.bounds.y) / sy)));
      const pi = py * w + px;
      const rC = Math.max(0, Math.min(255, Math.round(rArr[pi])));
      const gC = Math.max(0, Math.min(255, Math.round(gArr[pi])));
      const bC = Math.max(0, Math.min(255, Math.round(bArr[pi])));
      ent.fillColor = `#${((1 << 24) + (rC << 16) + (gC << 8) + bC).toString(16).slice(1)}`;
    }

    // Fallback architectural edge rails for sleek paneled doors with few small carvings (e.g. modern walnut stile doors)
    let archEnts = [];
    if (combined.length < 32) {
      const edgeOnly = ImageVectorizer.traceEdgeGrooves(rArr, gArr, bArr, lumaRaw, preMorphLuma, w, h, {
        ...opts,
        threshold: 0.22,
        maxContours: 90,
      }).filter((e) => {
        if (!e.points || e.points.length === 0) return false;
        let inRoi = true;
        for (const p of e.points) {
          if (p.x < roiEdgeLeftMm - 1 || p.x > roiEdgeRightMm + 1 || p.y < roiEdgeTopMm - 1 || p.y > roiEdgeBotMm + 1) {
            inRoi = false;
            break;
          }
        }
        return inRoi;
      });
      archEnts = edgeOnly;
    }

    if (roi.hasExternalBg) {
      archEnts.unshift({
        type: 'rect',
        x: Number((opts.bounds.x + roi.x0 * sx).toFixed(3)),
        y: Number((opts.bounds.y + roi.y0 * sy).toFixed(3)),
        w: Number((roi.w * sx).toFixed(3)),
        h: Number((roi.h * sy).toFixed(3)),
        cornerRadius: 0,
        isClosed: true,
        color: '#22d3ee',
        fillColor: '#c29258',
        strokeColor: '#22d3ee',
        width: 1.4,
        name: 'Panel Perimeter Frame',
        layerGroup: 'Architectural Frame',
        toneLevel: 1,
        toneTotal: 2,
        relief: { enabled: false, shape: 'flat', heightMm: 0.4, radiusMm: 1.0, mode: 'max' },
      });
    }

    return [...archEnts, ...combined].slice(0, Math.max(opts.maxContours, 520));
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
   * Iterative Ramer-Douglas-Peucker polyline simplification for `{x, y}` anchor arrays.
   */
  static simplifyRDP(pts, epsilon = 0.4) {
    if (!pts || pts.length <= 2) return pts ? pts.map((p) => ({ ...p })) : [];
    const n = pts.length;
    const keep = new Uint8Array(n);
    keep[0] = 1;
    keep[n - 1] = 1;
    const stack = [[0, n - 1]];
    const sqEps = epsilon * epsilon;

    while (stack.length > 0) {
      const [start, end] = stack.pop();
      if (end - start <= 1) continue;
      const a = pts[start], b = pts[end];
      const dx = b.x - a.x, dy = b.y - a.y;
      const lenSq = dx * dx + dy * dy;
      let maxDistSq = -1;
      let maxIdx = -1;

      for (let i = start + 1; i < end; i++) {
        const p = pts[i];
        let dSq;
        if (lenSq < 1e-12) {
          dSq = (p.x - a.x) ** 2 + (p.y - a.y) ** 2;
        } else {
          const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / lenSq));
          const px = a.x + t * dx, py = a.y + t * dy;
          dSq = (p.x - px) ** 2 + (p.y - py) ** 2;
        }
        if (dSq > maxDistSq) {
          maxDistSq = dSq;
          maxIdx = i;
        }
      }

      if (maxIdx !== -1 && maxDistSq > sqEps) {
        keep[maxIdx] = 1;
        stack.push([start, maxIdx], [maxIdx, end]);
      }
    }

    const out = [];
    for (let i = 0; i < n; i++) {
      if (keep[i]) out.push({ ...pts[i] });
    }
    return out;
  }

  /**
   * Corner-preserving open/closed path smoother built on Taubin open-span smoothing.
   */
  static cornerPreservingSmooth(pts, closed = false) {
    if (!pts || pts.length < 4) return pts ? pts.map((p) => ({ ...p })) : [];
    const n = pts.length;
    const isCorner = new Uint8Array(n);
    isCorner[0] = closed ? 0 : 1;
    isCorner[n - 1] = closed ? 0 : 1;

    for (let i = 0; i < n; i++) {
      if (!closed && (i === 0 || i === n - 1)) continue;
      const prev = pts[(i - 1 + n) % n];
      const curr = pts[i];
      const next = pts[(i + 1) % n];
      const v1x = curr.x - prev.x, v1y = curr.y - prev.y;
      const v2x = next.x - curr.x, v2y = next.y - curr.y;
      const d1 = Math.hypot(v1x, v1y), d2 = Math.hypot(v2x, v2y);
      if (d1 < 1e-6 || d2 < 1e-6) continue;
      const dot = (v1x * v2x + v1y * v2y) / (d1 * d2);
      if (dot < 0.65) isCorner[i] = 1;
    }

    const out = [];
    let spanStart = 0;
    for (let i = 1; i < n; i++) {
      if (isCorner[i] || i === n - 1) {
        const span = pts.slice(spanStart, i + 1).map((p) => [p.x, p.y]);
        const smoothed = span.length >= 4 ? ImageVectorizer.taubinSmoothOpen(span, 2) : span;
        for (let k = (out.length === 0 ? 0 : 1); k < smoothed.length; k++) {
          const isEndCorner = k === smoothed.length - 1 && isCorner[i];
          out.push({ x: smoothed[k][0], y: smoothed[k][1], corner: isEndCorner });
        }
        spanStart = i;
      }
    }
    return out;
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
    const nRaw = rawLoop.length;
    if (nRaw < 4) return null;

    const scaleRatio = opts.scaleRatio || 1;
    const doRefine = opts.refineEdges !== false;

    // Step 1: Subpixel edge refinement first so staircase steps don't create false 45° corners!
    const workLoop = doRefine ? ImageVectorizer.refineContourToField(rawLoop, true, rawField, w, h, refSign, scaleRatio) : rawLoop;
    const circleAnchors = ImageVectorizer.detectExactCircle(workLoop);
    if (circleAnchors) return circleAnchors;

    const n = workLoop.length;
    // Periodic 3-pass Taubin-cleaned probe loop so Marching Squares / grain stairsteps never trigger fake corners!
    let probeLoop = workLoop;
    if (n >= 10) {
      const wrap = workLoop.concat([workLoop[0], workLoop[1]]);
      const sm = ImageVectorizer.taubinSmoothOpen(wrap, 3);
      probeLoop = sm.slice(0, n);
    }

    // Step 2: Cumulative arc length along subpixel contour
    const s = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) {
      const a = workLoop[i], b = workLoop[(i + 1) % n];
      s[i + 1] = s[i] + Math.hypot(b[0] - a[0], b[1] - a[1]);
    }
    const perimeter = s[n];
    if (perimeter < 4.0) return null;

    // Sharp corner probe chord (large enough to ignore pixel grid & wood ripples, sharp on genuine geometric tips)
    const wChord = perimeter >= 24
      ? Math.max(4.2, Math.min(perimeter * 0.075, 3.6 * scaleRatio))
      : Math.max(2.0, perimeter * 0.14);
    const sampleArc = (loopArr, uTarget) => {
      const u = ((uTarget % perimeter) + perimeter) % perimeter;
      let lo = 0, hi = n;
      while (hi - lo > 1) {
        const m = (lo + hi) >> 1;
        if (s[m] <= u) lo = m; else hi = m;
      }
      const a = loopArr[lo], b = loopArr[(lo + 1) % n];
      const seg = s[lo + 1] - s[lo] || 1;
      const t = (u - s[lo]) / seg;
      return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    };

    const minCornerRad = Math.max(0.72, 1.32 - (opts.corners || 0.65) * 0.56);
    const turns = new Float32Array(n);
    const cand = [];
    for (let i = 0; i < n; i++) {
      const p = probeLoop[i];
      const a = sampleArc(probeLoop, s[i] - wChord);
      const b = sampleArc(probeLoop, s[i] + wChord);
      const ux = p[0] - a[0], uy = p[1] - a[1];
      const vx = b[0] - p[0], vy = b[1] - p[1];
      const den = Math.hypot(ux, uy) * Math.hypot(vx, vy);
      turns[i] = den > 1e-9 ? Math.acos(Math.max(-1, Math.min(1, (ux * vx + uy * vy) / den))) : 0;
    }

    for (let i = 0; i < n; i++) {
      const ang = turns[i];
      if (ang < minCornerRad) continue;
      // Distinguish genuine sharp vertex from uniform curved arc via local curvature prominence!
      const stepIdx = Math.max(1, Math.round((wChord * 1.2) / Math.max(0.5, perimeter / n)));
      const avgNb = 0.5 * (turns[(i - stepIdx + n) % n] + turns[(i + stepIdx) % n]);
      if (ang >= 1.26 || ang - avgNb >= 0.36) {
        cand.push(i);
      }
    }

    cand.sort((a, b) => turns[b] - turns[a] || a - b);
    const cornerIdx = [];
    for (const idx of cand) {
      let ok = true;
      for (const kept of cornerIdx) {
        const d = Math.abs(s[idx] - s[kept]);
        if (Math.min(d, perimeter - d) < wChord * 0.85) { ok = false; break; }
      }
      if (ok) cornerIdx.push(idx);
    }
    cornerIdx.sort((a, b) => a - b);

    // Zero corners -> fit smooth closed Cubic Bezier loop (e.g. smooth wave, petal, oval)
    if (cornerIdx.length === 0) {
      let pMin = 0;
      for (let i = 1; i < n; i++) {
        if (workLoop[i][1] < workLoop[pMin][1] || (workLoop[i][1] === workLoop[pMin][1] && workLoop[i][0] < workLoop[pMin][0])) {
          pMin = i;
        }
      }
      let pFar = pMin, maxD = -1;
      for (let i = 0; i < n; i++) {
        const d = Math.hypot(workLoop[i][0] - workLoop[pMin][0], workLoop[i][1] - workLoop[pMin][1]);
        if (d > maxD) { maxD = d; pFar = i; }
      }
      const splitPair = pMin < pFar ? [pMin, pFar] : [pFar, pMin];
      if (splitPair[1] === splitPair[0]) return null;

      const smoothed = ImageVectorizer.taubinSmoothOpen(
        workLoop.slice(splitPair[0]).concat(workLoop.slice(0, splitPair[0] + 1)),
        4
      );
      const midInSmoothed = (splitPair[1] - splitPair[0]);
      const errSq = Math.pow(Math.max(0.36, (1.08 - opts.detailFactor) * 1.15 * scaleRatio), 2);

      const fitHalf = (subPts) => {
        const len = subPts.length;
        const k = Math.min(len - 1, 3);
        const t1 = ImageVectorizer.normVec([subPts[k][0] - subPts[0][0], subPts[k][1] - subPts[0][1]]);
        const t2 = ImageVectorizer.normVec([subPts[len - 1 - k][0] - subPts[len - 1][0], subPts[len - 1 - k][1] - subPts[len - 1][1]]);
        return ImageVectorizer.fitCubicSpan(subPts, 0, len - 1, t1, t2, errSq);
      };

      const cub1 = fitHalf(smoothed.slice(0, midInSmoothed + 1));
      const cub2 = fitHalf(smoothed.slice(midInSmoothed));
      return ImageVectorizer.cubicsToAnchors(
        [{ cubics: cub1, cornerStart: false }, { cubics: cub2, cornerStart: false }]
      );
    }

    // Extract spans between true corners
    const m = cornerIdx.length;
    const rawSpans = [];
    for (let j = 0; j < m; j++) {
      const a = cornerIdx[j];
      const b = cornerIdx[(j + 1) % m];
      const spanPts = [workLoop[a]];
      let idx = a;
      let guard = 0;
      do {
        idx = (idx + 1) % n;
        spanPts.push(workLoop[idx]);
      } while (idx !== b && guard++ < n);
      rawSpans.push(spanPts);
    }

    const lineTol = Math.max(0.36, (1.02 - opts.detailFactor) * 0.52 * scaleRatio);
    const guardArc = Math.min(2.0, wChord * 0.40);
    const snapAxisLine = (tls, Ls) => {
      if (!tls || Ls < 12) return tls;
      const [dx, dy] = tls.d;
      if (Math.abs(dy) < 0.032) return { c: tls.c, d: [dx >= 0 ? 1 : -1, 0], dev: tls.dev };
      if (Math.abs(dx) < 0.032) return { c: tls.c, d: [0, dy >= 0 ? 1 : -1], dev: tls.dev };
      return tls;
    };

    const classifySpan = (P) => {
      const q = P.length;
      const sl = new Float64Array(q);
      for (let i = 1; i < q; i++) sl[i] = sl[i - 1] + Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]);
      const Ls = sl[q - 1];
      const info = { P, sl, Ls, isLine: false, A: null, B: null };
      const mid = [];
      for (let i = 0; i < q; i++) if (sl[i] >= guardArc && sl[i] <= Ls - guardArc) mid.push(P[i]);

      // Only classify as straight CAD line if ALL points genuinely adhere to a straight line (never force curved short arcs!)
      const lineSrc = mid.length >= 3 ? mid : P;
      const tlsAll = q >= 3 ? ImageVectorizer.fitLineTLS(lineSrc) : null;
      const straightLim = Ls >= 14 ? Math.min(1.08, Math.max(lineTol * 1.15, Ls * 0.018)) : lineTol * 0.92;
      if (tlsAll && tlsAll.dev <= straightLim) {
        let maxAll = 0;
        for (const p of P) {
          maxAll = Math.max(maxAll, Math.abs(-(p[0] - tlsAll.c[0]) * tlsAll.d[1] + (p[1] - tlsAll.c[1]) * tlsAll.d[0]));
        }
        if (maxAll <= straightLim * 1.4) {
          info.isLine = true;
          info.A = info.B = snapAxisLine(tlsAll, Ls);
          return info;
        }
      }

      if (q <= 2 || Ls < 2.2) {
        info.isLine = true;
        const d = Math.hypot(P[q - 1][0] - P[0][0], P[q - 1][1] - P[0][1]) || 1;
        info.A = info.B = snapAxisLine({ c: P[0], d: [(P[q - 1][0] - P[0][0]) / d, (P[q - 1][1] - P[0][1]) / d], dev: 0 }, Ls);
        return info;
      }

      const tanLen = Math.min(wChord * 2.2, Ls * 0.44);
      const startSeg = [], endSeg = [];
      for (let i = 0; i < q; i++) {
        if (sl[i] >= guardArc * 0.3 && sl[i] <= guardArc * 0.3 + tanLen) startSeg.push(P[i]);
        if (sl[i] <= Ls - guardArc * 0.3 && sl[i] >= Ls - guardArc * 0.3 - tanLen) endSeg.push(P[i]);
      }
      if (startSeg.length >= 3) info.A = ImageVectorizer.fitLineTLS(startSeg);
      if (endSeg.length >= 3) info.B = ImageVectorizer.fitLineTLS(endSeg);
      return info;
    };

    let spanInfos = rawSpans.map(classifySpan);

    // Coalesce consecutive nearly-collinear straight spans into single crisp CAD lines
    let merged = true;
    while (merged && spanInfos.length > 3) {
      merged = false;
      for (let j = 0; j < spanInfos.length; j++) {
        const nextJ = (j + 1) % spanInfos.length;
        const s1 = spanInfos[j], s2 = spanInfos[nextJ];
        if (!s1.isLine || !s2.isLine || !s1.A || !s2.A) continue;
        const dot = s1.A.d[0] * s2.A.d[0] + s1.A.d[1] * s2.A.d[1];
        if (dot < 0.95) continue;
        const combinedP = s1.P.slice(0, -1).concat(s2.P);
        const tlsComb = ImageVectorizer.fitLineTLS(combinedP);
        const combLs = s1.Ls + s2.Ls;
        const combLim = combLs >= 16 ? Math.min(1.15, Math.max(lineTol * 1.25, combLs * 0.02)) : lineTol * 1.15;
        if (tlsComb && tlsComb.dev <= combLim) {
          const combInfo = classifySpan(combinedP);
          combInfo.isLine = true;
          combInfo.A = combInfo.B = snapAxisLine(tlsComb, combInfo.Ls);
          if (nextJ === 0) {
            spanInfos[0] = combInfo;
            spanInfos.pop();
          } else {
            spanInfos.splice(j, 2, combInfo);
          }
          merged = true;
          break;
        }
      }
    }

    const mFinal = spanInfos.length;
    const dMaxSnap = Math.max(4.5, wChord * 2.2);
    const snappedCorners = [];
    for (let j = 0; j < mFinal; j++) {
      const pc = spanInfos[j].P[0];
      const prevInfo = spanInfos[(j + mFinal - 1) % mFinal];
      const nextInfo = spanInfos[j];
      let X = null;
      if (prevInfo.B && nextInfo.A) {
        X = ImageVectorizer.intersectLines(prevInfo.B, nextInfo.A);
        if (X && Math.hypot(X[0] - pc[0], X[1] - pc[1]) > dMaxSnap) X = null;
      }
      snappedCorners.push(X || pc);
    }

    const fitErrSq = Math.pow(Math.max(0.36, (1.08 - opts.detailFactor) * 1.15 * scaleRatio), 2);
    const segmentsList = [];

    for (let j = 0; j < mFinal; j++) {
      const info = spanInfos[j];
      const ca = snappedCorners[j];
      const cb = snappedCorners[(j + 1) % mFinal];

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
      const blendSpan = Math.min(info.Ls * 0.45, wChord * 2.2);
      const pinned = P.map((p, idx) => {
        const wa = blendSpan > 0 ? Math.max(0, 1 - info.sl[idx] / blendSpan) : 0;
        const wb = blendSpan > 0 ? Math.max(0, 1 - (info.Ls - info.sl[idx]) / blendSpan) : 0;
        return [p[0] + da[0] * wa + db[0] * wb, p[1] + da[1] * wa + db[1] * wb];
      });
      pinned[0] = ca;
      pinned[q - 1] = cb;

      const smPts = ImageVectorizer.taubinSmoothOpen(pinned, 4);
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
   * Prevent tight Bezier handle overshoot loops that create self-intersections along thin petals or tight curves.
   */
  static sanitizeBezierAnchors(anchors) {
    if (!anchors || anchors.length < 2) return anchors;
    const n = anchors.length;
    for (let i = 0; i < n; i++) {
      const a = anchors[i];
      const b = anchors[(i + 1) % n];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const chord = Math.hypot(dx, dy);
      if (chord < 1e-4) {
        a.hOut = null;
        b.hIn = null;
        continue;
      }
      const maxHandle = chord * 0.45;
      if (a.hOut) {
        const hx = a.hOut.x - a.x;
        const hy = a.hOut.y - a.y;
        const hLen = Math.hypot(hx, hy);
        const proj = (hx * dx + hy * dy) / chord;
        if (hLen > maxHandle || proj < -0.15 * chord) {
          const clampLen = Math.min(hLen, maxHandle);
          const ux = proj < 0 ? dx / chord : hx / (hLen || 1);
          const uy = proj < 0 ? dy / chord : hy / (hLen || 1);
          a.hOut = {
            x: Number((a.x + ux * clampLen).toFixed(3)),
            y: Number((a.y + uy * clampLen).toFixed(3)),
          };
        }
      }
      if (b.hIn) {
        const hx = b.hIn.x - b.x;
        const hy = b.hIn.y - b.y;
        const hLen = Math.hypot(hx, hy);
        const proj = (hx * -dx + hy * -dy) / chord;
        if (hLen > maxHandle || proj < -0.15 * chord) {
          const clampLen = Math.min(hLen, maxHandle);
          const ux = proj < 0 ? -dx / chord : hx / (hLen || 1);
          const uy = proj < 0 ? -dy / chord : hy / (hLen || 1);
          b.hIn = {
            x: Number((b.x + ux * clampLen).toFixed(3)),
            y: Number((b.y + uy * clampLen).toFixed(3)),
          };
        }
      }
    }
    return anchors;
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
      const mmAnchors = ImageVectorizer.sanitizeBezierAnchors(
        ImageVectorizer.anchorsToMm(anchorsPx, bounds, w, h)
      );

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
    let bestM1 = totalMean;
    let bestM2 = totalMean;

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
        bestM1 = mB;
        bestM2 = mF;
      }
    }

    return {
      threshold,
      separability: Math.min(1.0, maxVariance / totalVariance),
      m1: bestM1,
      m2: bestM2,
    };
  }

  static computeOtsuThreshold(histogram, totalPixels) {
    return ImageVectorizer.computeOtsuStats(histogram, totalPixels).threshold;
  }

  /**
   * 1D Felzenszwalb-Huttenlocher exact squared Euclidean distance envelope along a scanline.
   */
  static _edt1d(f, d, v, z, n) {
    let k = 0;
    v[0] = 0;
    z[0] = -1e20;
    z[1] = 1e20;
    for (let q = 1; q < n; q++) {
      let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
      while (s <= z[k]) {
        k--;
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
      }
      k++;
      v[k] = q;
      z[k] = s;
      z[k + 1] = 1e20;
    }
    k = 0;
    for (let q = 0; q < n; q++) {
      while (z[k + 1] < q) k++;
      const dx = q - v[k];
      d[q] = dx * dx + f[v[k]];
    }
  }

  /**
   * Exact O(W*H) 2D Euclidean L2 Distance Transform (distance from pixels where mask == 0 into foreground mask == 1).
   */
  static computeL2Distance2D(mask, w, h) {
    const n = w * h;
    const dist = new Float32Array(n);
    for (let i = 0; i < n; i++) dist[i] = mask[i] ? 1e20 : 0.0;
    const maxDim = Math.max(w, h);
    const f = new Float32Array(maxDim);
    const d = new Float32Array(maxDim);
    const v = new Int32Array(maxDim);
    const z = new Float32Array(maxDim + 1);

    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) f[y] = dist[y * w + x];
      ImageVectorizer._edt1d(f, d, v, z, h);
      for (let y = 0; y < h; y++) dist[y * w + x] = d[y];
    }
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) f[x] = dist[row + x];
      ImageVectorizer._edt1d(f, d, v, z, w);
      for (let x = 0; x < w; x++) dist[row + x] = Math.sqrt(d[x]);
    }
    return dist;
  }

  /**
   * Kovesi 3-pass separable sliding-window Gaussian smoothing in O(W*H).
   */
  static gaussianSmooth2D(field, w, h, sigma = 1.0) {
    if (sigma <= 0 || w < 2 || h < 2) return field;
    const wIdeal = Math.sqrt((12 * sigma * sigma) / 3 + 1);
    let wl = Math.floor(wIdeal);
    if (wl % 2 === 0) wl--;
    const wu = wl + 2;
    const m = Math.round((12 * sigma * sigma - 3 * wl * wl - 12 * wl - 9) / (-4 * wl - 4));
    const tmp = new Float32Array(field.length);

    for (let pass = 0; pass < 3; pass++) {
      const boxW = pass < m ? wl : wu;
      const r = (boxW - 1) >> 1;
      if (r < 1) continue;
      const scale = 1.0 / (r + r + 1);

      for (let y = 0; y < h; y++) {
        const row = y * w;
        let leftIdx = row;
        let rightIdx = row + r;
        const first = field[row];
        const last = field[row + w - 1];
        let val = (r + 1) * first;
        for (let j = 0; j < r; j++) val += field[row + Math.min(j, w - 1)];
        for (let j = 0; j <= r; j++) {
          val += (rightIdx < row + w ? field[rightIdx] : last) - first;
          tmp[row + j] = val * scale;
          rightIdx++;
        }
        for (let j = r + 1; j < w - r; j++) {
          val += field[rightIdx++] - field[leftIdx++];
          tmp[row + j] = val * scale;
        }
        for (let j = Math.max(r + 1, w - r); j < w; j++) {
          val += last - field[leftIdx++];
          tmp[row + j] = val * scale;
        }
      }

      for (let x = 0; x < w; x++) {
        let topIdx = x;
        let botIdx = x + r * w;
        const first = tmp[x];
        const last = tmp[x + (h - 1) * w];
        let val = (r + 1) * first;
        for (let j = 0; j < r; j++) val += tmp[x + Math.min(j, h - 1) * w];
        let outIdx = x;
        for (let j = 0; j <= r; j++) {
          val += (botIdx <= x + (h - 1) * w ? tmp[botIdx] : last) - first;
          field[outIdx] = val * scale;
          botIdx += w;
          outIdx += w;
        }
        for (let j = r + 1; j < h - r; j++) {
          val += tmp[botIdx] - tmp[topIdx];
          field[outIdx] = val * scale;
          botIdx += w;
          topIdx += w;
          outIdx += w;
        }
        for (let j = Math.max(r + 1, h - r); j < h; j++) {
          val += last - tmp[topIdx];
          field[outIdx] = val * scale;
          topIdx += w;
          outIdx += w;
        }
      }
    }
    return field;
  }

  /**
   * Eliminate diagonal checkerboard corner pinches so 2D contours & 3D silhouettes remain manifold-clean.
   */
  static removeCheckerboardPinches(mask, w, h, bridge = false) {
    const iters = bridge ? 6 : 1;
    for (let it = 0; it < iters; it++) {
      let changed = false;
      for (let y = 0; y < h - 1; y++) {
        const row0 = y * w;
        const row1 = (y + 1) * w;
        for (let x = 0; x < w - 1; x++) {
          const a = mask[row0 + x];
          const b = mask[row0 + x + 1];
          const c = mask[row1 + x];
          const d = mask[row1 + x + 1];
          if (a && d && !b && !c) {
            if (bridge) mask[row0 + x + 1] = 1;
            else mask[row1 + x + 1] = 0;
            changed = true;
          } else if (b && c && !a && !d) {
            if (bridge) mask[row0 + x] = 1;
            else mask[row1 + x] = 0;
            changed = true;
          }
        }
      }
      if (!changed) break;
    }
    return mask;
  }

  /**
   * Exact L2 Signed Distance Field (phi > 0 inside mask, phi < 0 outside, smoothed via Kovesi Gaussian).
   */
  static computeSignedDistancePhi(mask, w, h, sigma = 1.0) {
    const invMask = new Uint8Array(mask.length);
    for (let i = 0; i < mask.length; i++) invMask[i] = mask[i] ? 0 : 1;
    const dIn = ImageVectorizer.computeL2Distance2D(mask, w, h);
    const dOut = ImageVectorizer.computeL2Distance2D(invMask, w, h);
    const phi = new Float32Array(mask.length);
    for (let i = 0; i < mask.length; i++) {
      phi[i] = mask[i] ? dIn[i] : -dOut[i];
    }
    ImageVectorizer.gaussianSmooth2D(phi, w, h, sigma);
    return { phi, dIn, dOut };
  }
}
