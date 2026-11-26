import { ImageVectorizer } from '../cad/imageVectorizer.js';
import { VectorPuffEngine } from '../cad/vectorPuffEngine.js';

/**
 * Meshy-Style Image-to-3D Volumetric Asset Generator & PBR Map Synthesizer.
 * Builds smooth-rimmed watertight 3D meshes ('double-sided', 'inflated', 'lathe', 'relief')
 * with multi-scale sculpted surface relief, perimeter vertex relaxation (zero Lego stairs),
 * split rim normals, seam-free Lathe normals, and 1024px PBR BaseColor/Normal/ORM maps.
 */
export class Asset3DEngine {
  /**
   * Extract clean foreground silhouette, multi-scale sculpted depth field, and PBR channel maps.
   */
  static analyzeImage(imgElement, options = {}) {
    const {
      resolution = 160,
      texSize = 1024,
      removeBg = true,
      delight = true,
      invert = false,
      edgeStrength = 0.38,
    } = options;

    const N = Math.max(48, Math.min(320, Math.round(resolution)));
    const canvas = document.createElement('canvas');
    canvas.width = N;
    canvas.height = N;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.clearRect(0, 0, N, N);

    const iw = imgElement.naturalWidth || imgElement.width || 1;
    const ih = imgElement.naturalHeight || imgElement.height || 1;
    const aspect = iw / ih;
    let dw = N;
    let dh = N;
    let dx = 0;
    let dy = 0;
    if (aspect > 1) {
      dh = N / aspect;
      dy = (N - dh) * 0.5;
    } else {
      dw = N * aspect;
      dx = (N - dw) * 0.5;
    }
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(imgElement, dx, dy, dw, dh);
    const rgba = ctx.getImageData(0, 0, N, N).data;

    const tSize = Math.max(256, Math.min(1024, Math.round(texSize)));
    const texCanvas = document.createElement('canvas');
    texCanvas.width = tSize;
    texCanvas.height = tSize;
    const texCtx = texCanvas.getContext('2d', { willReadFrequently: true });
    texCtx.clearRect(0, 0, tSize, tSize);
    const texScale = tSize / N;
    texCtx.imageSmoothingQuality = 'high';
    texCtx.drawImage(imgElement, dx * texScale, dy * texScale, dw * texScale, dh * texScale);
    const texRgba = texCtx.getImageData(0, 0, tSize, tSize).data;

    const total = N * N;
    const luma = new Float32Array(total);
    const chromaL = new Float32Array(total);
    const chromaA = new Float32Array(total);
    const chromaB = new Float32Array(total);
    const alpha = new Uint8Array(total);

    let hasTransparency = false;
    for (let i = 0; i < total; i++) {
      const r = rgba[i * 4];
      const g = rgba[i * 4 + 1];
      const b = rgba[i * 4 + 2];
      const a = rgba[i * 4 + 3];
      alpha[i] = a;
      if (a < 230) hasTransparency = true;
      luma[i] = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
      const [lv, av, bv] = ImageVectorizer.rgbToLab(r, g, b);
      chromaL[i] = lv;
      chromaA[i] = av;
      chromaB[i] = bv;
    }

    const fgMask = Asset3DEngine.extractBorderConnectedForeground(
      rgba,
      alpha,
      chromaL,
      chromaA,
      chromaB,
      N,
      { dx, dy, dw, dh, removeBg, hasTransparency }
    );

    const dist = VectorPuffEngine.computeInteriorDistance(fgMask, N);
    let maxDist = 1.0;
    for (let i = 0; i < total; i++) {
      if (dist[i] > maxDist) maxDist = dist[i];
    }

    // Multi-scale Morphological White-Hat + Black-Hat + De-lighted Shading Relief
    // Ensures true geometric ridges/crevices/carvings are sculpted into the 3D vertices!
    const lumaSmooth = ImageVectorizer.guidedFilter(luma, N, N, 1, 0.008);
    const rMed = Math.max(3, Math.min(8, Math.round(N * 0.032)));
    const openedMed = ImageVectorizer.separableExtrema(
      ImageVectorizer.separableExtrema(lumaSmooth, N, N, rMed, false),
      N, N, rMed, true
    );
    const closedMed = ImageVectorizer.separableExtrema(
      ImageVectorizer.separableExtrema(lumaSmooth, N, N, rMed, true),
      N, N, rMed, false
    );

    const sat = ImageVectorizer.buildIntegralImage(lumaSmooth, N, N);
    const illumR = Math.max(8, Math.round(N * 0.14));

    const sculptSignal = new Float32Array(total);
    let sMin = 1e9;
    let sMax = -1e9;
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const i = y * N + x;
        if (!fgMask[i]) continue;
        const localMean = ImageVectorizer.getLocalMean(sat, N, N, x, y, illumR);
        const wh = Math.max(0, lumaSmooth[i] - openedMed[i]);
        const bh = Math.max(0, closedMed[i] - lumaSmooth[i]);
        let tone = delight
          ? lumaSmooth[i] * 0.55 + (lumaSmooth[i] - localMean + 0.5) * 0.45
          : lumaSmooth[i];
        if (invert) tone = 1.0 - tone;
        const sig = tone * 0.52 + (wh * 1.45 - bh * 1.05) * (invert ? -1 : 1) * 0.48;
        sculptSignal[i] = sig;
        if (sig < sMin) sMin = sig;
        if (sig > sMax) sMax = sig;
      }
    }

    const sSpan = Math.max(0.08, sMax - sMin);
    const combinedDepth = new Float32Array(total);
    for (let y = 1; y < N - 1; y++) {
      for (let x = 1; x < N - 1; x++) {
        const i = y * N + x;
        if (!fgMask[i]) continue;
        const t = Math.min(1.0, dist[i] / Math.max(3.0, maxDist * 0.88));
        const macroDome = Math.sqrt(Math.max(0.0, 1.0 - (1.0 - t) * (1.0 - t)));
        const normSculpt = Math.max(0.0, Math.min(1.0, (sculptSignal[i] - sMin) / sSpan));
        const gx = lumaSmooth[i + 1] - lumaSmooth[i - 1];
        const gy = lumaSmooth[i + N] - lumaSmooth[i - N];
        const edgeCrest = Math.min(0.24, Math.hypot(gx, gy) * edgeStrength * 0.85);
        // Balance broad 3D volume (48%) + deep sculpted geometry detail (52%)
        combinedDepth[i] = Math.max(
          0.08,
          Math.min(1.0, macroDome * 0.48 + normSculpt * 0.48 + edgeCrest * 0.16)
        );
      }
    }

    // Light edge-preserving smooth inside foreground ONLY so silhouette roll-off never blurs with Z=0 backdrop
    const denoisedDepth = new Float32Array(combinedDepth);
    for (let y = 1; y < N - 1; y++) {
      const row = y * N;
      for (let x = 1; x < N - 1; x++) {
        const i = row + x;
        if (!fgMask[i]) continue;
        let wSum = 2.0;
        let vSum = combinedDepth[i] * 2.0;
        const nbs = [i - 1, i + 1, i - N, i + N];
        for (const ni of nbs) {
          if (!fgMask[ni]) continue;
          const d = Math.abs(combinedDepth[ni] - combinedDepth[i]);
          const w = d < 0.14 ? 1.0 : 0.25;
          vSum += combinedDepth[ni] * w;
          wSum += w;
        }
        denoisedDepth[i] = Math.max(0.08, Math.min(1.0, vSum / wSum));
      }
    }

    // Infinite Voronoi edge padding from pure interior foreground seed so zero backdrop white/black halo can ever bleed into UVs
    Asset3DEngine.dilateTextureEdges(texRgba, tSize, fgMask, N, removeBg && !hasTransparency);

    const maps = Asset3DEngine.synthesizePBRMaps(texRgba, tSize, delight, fgMask, N);

    // Tight foreground bounding box in [0..1] UV space (allows exporting centered real-aspect models)
    let minFgX = N - 1, maxFgX = 0, minFgY = N - 1, maxFgY = 0;
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        if (!fgMask[y * N + x]) continue;
        if (x < minFgX) minFgX = x;
        if (x > maxFgX) maxFgX = x;
        if (y < minFgY) minFgY = y;
        if (y > maxFgY) maxFgY = y;
      }
    }
    const aspectBox = maxFgX > minFgX && maxFgY > minFgY
      ? {
          minU: minFgX / (N - 1),
          maxU: maxFgX / (N - 1),
          minV: 1.0 - maxFgY / (N - 1),
          maxV: 1.0 - minFgY / (N - 1),
        }
      : { minU: 0, maxU: 1, minV: 0, maxV: 1 };

    return {
      resolution: N,
      fgMask,
      dist,
      maxDist,
      depth: denoisedDepth,
      sculptSignal,
      texSize: tSize,
      maps,
      aspectBox,
      boundsPx: { dx, dy, dw, dh },
    };
  }

  /**
   * Flood-fill outward from border in CIELAB so interior bright highlights never punch holes!
   */
  static extractBorderConnectedForeground(rgba, alpha, L, A, B, N, ctxInfo) {
    const total = N * N;
    const { dx, dy, dw, dh, removeBg, hasTransparency } = ctxInfo;
    const fg = new Uint8Array(total);
    const x0 = Math.max(0, Math.floor(dx));
    const y0 = Math.max(0, Math.floor(dy));
    const x1 = Math.min(N - 1, Math.ceil(dx + dw));
    const y1 = Math.min(N - 1, Math.ceil(dy + dh));

    if (hasTransparency) {
      for (let i = 0; i < total; i++) {
        if (alpha[i] >= 96) fg[i] = 1;
      }
      return Asset3DEngine.cleanSilhouetteMask(fg, N);
    }

    for (let y = y0; y <= y1; y++) {
      const row = y * N;
      for (let x = x0; x <= x1; x++) fg[row + x] = 1;
    }
    if (!removeBg) return fg;

    const samplesL = [];
    const samplesA = [];
    const samplesB = [];
    for (let x = x0; x <= x1; x += 2) {
      samplesL.push(L[y0 * N + x], L[y1 * N + x]);
      samplesA.push(A[y0 * N + x], A[y1 * N + x]);
      samplesB.push(B[y0 * N + x], B[y1 * N + x]);
    }
    for (let y = y0; y <= y1; y += 2) {
      samplesL.push(L[y * N + x0], L[y * N + x1]);
      samplesA.push(A[y * N + x0], A[y * N + x1]);
      samplesB.push(B[y * N + x0], B[y * N + x1]);
    }
    samplesL.sort((u, v) => u - v);
    samplesA.sort((u, v) => u - v);
    samplesB.sort((u, v) => u - v);
    const bgL = samplesL[Math.floor(samplesL.length * 0.5)] || 95;
    const bgA = samplesA[Math.floor(samplesA.length * 0.5)] || 0;
    const bgB = samplesB[Math.floor(samplesB.length * 0.5)] || 0;

    const visited = new Uint8Array(total);
    const queue = new Int32Array(total);
    let qH = 0;
    let qT = 0;

    const seedIfBg = (x, y) => {
      const idx = y * N + x;
      if (visited[idx]) return;
      const de = Math.hypot((L[idx] - bgL) * 0.85, A[idx] - bgA, B[idx] - bgB);
      if (de <= 22 || alpha[idx] < 64) {
        visited[idx] = 1;
        fg[idx] = 0;
        queue[qT++] = idx;
      }
    };

    for (let x = 0; x < N; x++) {
      seedIfBg(x, y0); seedIfBg(x, y1);
      seedIfBg(x, 0); seedIfBg(x, N - 1);
    }
    for (let y = 0; y < N; y++) {
      seedIfBg(x0, y); seedIfBg(x1, y);
      seedIfBg(0, y); seedIfBg(N - 1, y);
    }

    const dirs = [-1, 1, -N, N];
    while (qH < qT) {
      const u = queue[qH++];
      const ux = u % N;
      const uy = (u - ux) / N;
      for (let d = 0; d < 4; d++) {
        if (d === 0 && ux === 0) continue;
        if (d === 1 && ux === N - 1) continue;
        if (d === 2 && uy === 0) continue;
        if (d === 3 && uy === N - 1) continue;
        const v = u + dirs[d];
        if (visited[v]) continue;
        const deBg = Math.hypot((L[v] - bgL) * 0.85, A[v] - bgA, B[v] - bgB);
        const deStep = Math.hypot((L[v] - L[u]) * 0.85, A[v] - A[u], B[v] - B[u]);
        if (deBg <= 21 && deStep <= 14) {
          visited[v] = 1;
          fg[v] = 0;
          queue[qT++] = v;
        }
      }
    }

    let fgCount = 0;
    for (let i = 0; i < total; i++) fgCount += fg[i];
    if (fgCount < total * 0.04) {
      for (let y = y0; y <= y1; y++) {
        const row = y * N;
        for (let x = x0; x <= x1; x++) fg[row + x] = 1;
      }
    }
    return Asset3DEngine.cleanSilhouetteMask(fg, N);
  }

  static cleanSilhouetteMask(fg, N) {
    const closed = ImageVectorizer.morphClose(fg, N, N, 1);
    const total = N * N;
    const minCompPx = Math.max(18, Math.round(total * 0.002));
    const vis = new Uint8Array(total);
    const queue = new Int32Array(total);
    const dirs = [-1, 1, -N, N];

    for (let s = 0; s < total; s++) {
      if (!closed[s] || vis[s]) continue;
      let qH = 0, qT = 0;
      vis[s] = 1;
      queue[qT++] = s;
      while (qH < qT) {
        const u = queue[qH++];
        const ux = u % N;
        const uy = (u - ux) / N;
        for (let d = 0; d < 4; d++) {
          if (d === 0 && ux === 0) continue;
          if (d === 1 && ux === N - 1) continue;
          if (d === 2 && uy === 0) continue;
          if (d === 3 && uy === N - 1) continue;
          const v = u + dirs[d];
          if (closed[v] && !vis[v]) {
            vis[v] = 1;
            queue[qT++] = v;
          }
        }
      }
      if (qT < minCompPx) {
        for (let k = 0; k < qT; k++) closed[queue[k]] = 0;
      }
    }
    return closed;
  }

  /**
   * Industry-standard infinite Voronoi BFS padding from pure interior foreground texels across all cutout seams.
   * Eliminates 100% of white/black background halos on narrow tips, side bevels, and mipmap levels.
   */
  static dilateTextureEdges(texRgba, tSize, fgMask, fgRes, stripBorderBackdrop = false) {
    if (!fgMask || !fgRes) return;
    const fgN = fgRes * fgRes;
    const eroded1 = new Uint8Array(fgN);
    const eroded2 = new Uint8Array(fgN);
    let e1Count = 0;
    for (let y = 1; y < fgRes - 1; y++) {
      const row = y * fgRes;
      for (let x = 1; x < fgRes - 1; x++) {
        const i = row + x;
        if (fgMask[i] && fgMask[i - 1] && fgMask[i + 1] && fgMask[i - fgRes] && fgMask[i + fgRes]) {
          eroded1[i] = 1;
          e1Count++;
        }
      }
    }
    let e2Count = 0;
    for (let y = 1; y < fgRes - 1; y++) {
      const row = y * fgRes;
      for (let x = 1; x < fgRes - 1; x++) {
        const i = row + x;
        if (eroded1[i] && eroded1[i - 1] && eroded1[i + 1] && eroded1[i - fgRes] && eroded1[i + fgRes]) {
          eroded2[i] = 1;
          e2Count++;
        }
      }
    }
    const seedMask = e1Count > 16 ? eroded1 : fgMask;
    const safeInnerMask = e2Count > 16 ? eroded2 : null;

    // Sample corner backdrop RGB so residual high-res border anti-aliasing pixels don't seed Voronoi padding
    let bgR = 255, bgG = 255, bgB = 255;
    if (stripBorderBackdrop) {
      const cr = [0, (tSize - 1) * 4, (tSize * (tSize - 1)) * 4, (tSize * tSize - 1) * 4];
      bgR = Math.round((texRgba[cr[0]] + texRgba[cr[1]] + texRgba[cr[2]] + texRgba[cr[3]]) * 0.25);
      bgG = Math.round((texRgba[cr[0] + 1] + texRgba[cr[1] + 1] + texRgba[cr[2] + 1] + texRgba[cr[3] + 1]) * 0.25);
      bgB = Math.round((texRgba[cr[0] + 2] + texRgba[cr[1] + 2] + texRgba[cr[2] + 2] + texRgba[cr[3] + 2]) * 0.25);
    }

    const n = tSize * tSize;
    const filled = new Uint8Array(n);
    const queue = new Int32Array(n);
    let qH = 0;
    let qT = 0;
    const scale = (fgRes - 1) / Math.max(1, tSize - 1);

    for (let y = 0; y < tSize; y++) {
      const sy = Math.min(fgRes - 1, Math.round(y * scale));
      const fRow = sy * fgRes;
      const tRow = y * tSize;
      for (let x = 0; x < tSize; x++) {
        const sx = Math.min(fgRes - 1, Math.round(x * scale));
        const fi = fRow + sx;
        const idx = tRow + x;
        if (!seedMask[fi]) continue;
        const p = idx * 4;
        if (stripBorderBackdrop && safeInnerMask && !safeInnerMask[fi]) {
          const dBg = Math.hypot(texRgba[p] - bgR, texRgba[p + 1] - bgG, texRgba[p + 2] - bgB);
          if (dBg < 72 || texRgba[p + 3] < 220) continue;
        } else if (texRgba[p + 3] < 180) {
          continue;
        }
        filled[idx] = 1;
        queue[qT++] = idx;
        texRgba[p + 3] = 255;
      }
    }
    if (qT === 0) return;

    const dirs = [-1, 1, -tSize, tSize];
    while (qH < qT) {
      const u = queue[qH++];
      const ux = u % tSize;
      const uy = (u - ux) / tSize;
      const up = u * 4;
      for (let d = 0; d < 4; d++) {
        if (d === 0 && ux === 0) continue;
        if (d === 1 && ux === tSize - 1) continue;
        if (d === 2 && uy === 0) continue;
        if (d === 3 && uy === tSize - 1) continue;
        const v = u + dirs[d];
        if (filled[v]) continue;
        filled[v] = 1;
        const vp = v * 4;
        texRgba[vp] = texRgba[up];
        texRgba[vp + 1] = texRgba[up + 1];
        texRgba[vp + 2] = texRgba[up + 2];
        texRgba[vp + 3] = 255;
        queue[qT++] = v;
      }
    }
  }

  /**
   * Synthesize Base Color, Tangent Normal, Roughness, Metallic, AND GLTF-packed ORM maps.
   */
  static synthesizePBRMaps(rgba, size, delight = true, fgMask = null, fgRes = 0) {
    const n = size * size;
    const baseColor = new Uint8ClampedArray(rgba);
    const normal = new Uint8ClampedArray(n * 4);
    const roughness = new Uint8ClampedArray(n * 4);
    const metallic = new Uint8ClampedArray(n * 4);
    const metallicRoughness = new Uint8ClampedArray(n * 4);

    const lum = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const r = rgba[i * 4];
      const g = rgba[i * 4 + 1];
      const b = rgba[i * 4 + 2];
      lum[i] = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
    }

    const sat = ImageVectorizer.buildIntegralImage(lum, size, size);
    const rIllum = Math.max(12, Math.round(size * 0.09));
    const fgScale = fgMask && fgRes ? (fgRes - 1) / Math.max(1, size - 1) : 0;

    for (let y = 0; y < size; y++) {
      const sy = fgScale ? Math.min(fgRes - 1, Math.round(y * fgScale)) : 0;
      for (let x = 0; x < size; x++) {
        const i = y * size + x;
        const p = i * 4;
        const r = rgba[p];
        const g = rgba[p + 1];
        const b = rgba[p + 2];
        const a = rgba[p + 3];
        const inFg = !fgScale || fgMask[sy * fgRes + Math.min(fgRes - 1, Math.round(x * fgScale))] === 1;

        if (delight && a > 16) {
          const localMean = Math.max(0.18, ImageVectorizer.getLocalMean(sat, size, size, x, y, rIllum));
          const gain = Math.max(0.76, Math.min(1.38, (0.54 / localMean) * 0.42 + 0.58));
          baseColor[p] = Math.min(255, Math.round(r * gain));
          baseColor[p + 1] = Math.min(255, Math.round(g * gain));
          baseColor[p + 2] = Math.min(255, Math.round(b * gain));
          baseColor[p + 3] = 255;
        }

        const xl = Math.max(0, x - 1);
        const xr = Math.min(size - 1, x + 1);
        const yu = Math.max(0, y - 1);
        const yd = Math.min(size - 1, y + 1);
        const gradGain = inFg ? 3.1 : 0.4;
        const dx = (lum[y * size + xr] - lum[y * size + xl]) * gradGain;
        const dy = (lum[yd * size + x] - lum[yu * size + x]) * gradGain;
        let nx = -dx;
        let ny = dy;
        let nz = 1.0;
        const inv = 1.0 / Math.hypot(nx, ny, nz);
        nx *= inv; ny *= inv; nz *= inv;
        normal[p] = Math.round((nx * 0.5 + 0.5) * 255);
        normal[p + 1] = Math.round((ny * 0.5 + 0.5) * 255);
        normal[p + 2] = Math.round((nz * 0.5 + 0.5) * 255);
        normal[p + 3] = 255;

        const mx = Math.max(r, g, b);
        const mn = Math.min(r, g, b);
        const ch = (mx - mn) / 255.0;
        const gradMag = Math.min(1.0, Math.hypot(dx, dy) * 1.35);
        const roughVal = Math.max(0.24, Math.min(0.92, 0.72 - lum[i] * 0.24 + gradMag * 0.20));
        const rv = Math.round(roughVal * 255);
        roughness[p] = rv; roughness[p + 1] = rv; roughness[p + 2] = rv; roughness[p + 3] = 255;

        const isGoldOrSteel = lum[i] > 0.48 && (ch < 0.14 || (r > g && g > b && ch > 0.20));
        const mv = isGoldOrSteel ? Math.round((0.34 + lum[i] * 0.44) * 255) : 18;
        metallic[p] = mv; metallic[p + 1] = mv; metallic[p + 2] = mv; metallic[p + 3] = 255;

        // glTF 2.0 MetallicRoughness texture spec: G = Roughness, B = Metallic
        metallicRoughness[p] = 255;
        metallicRoughness[p + 1] = rv;
        metallicRoughness[p + 2] = mv;
        metallicRoughness[p + 3] = 255;
      }
    }

    return { size, baseColor, normal, roughness, metallic, metallicRoughness };
  }

  static buildAssetMeshData(analysis, options = {}) {
    const {
      assetMode = 'double-sided', // 'double-sided' | 'inflated' | 'lathe' | 'relief'
      widthMm = 100,
      heightMm = 100,
      thicknessMm = 22,
    } = options;

    if (assetMode === 'lathe') {
      return Asset3DEngine.buildLatheMeshData(analysis, widthMm, heightMm, thicknessMm);
    }
    return Asset3DEngine.buildVolumetricShellMeshData(analysis, assetMode, widthMm, heightMm, thicknessMm);
  }

  /**
   * Watertight volumetric shell with tangential boundary relaxation (zero staircasing!)
   * and split rim bevel vertices (crisp surface normals without side-wall darkening).
   */
  static buildVolumetricShellMeshData(analysis, assetMode, widthMm, heightMm, thicknessMm) {
    const N = analysis.resolution;
    const fg = analysis.fgMask;
    const depth = analysis.depth;
    const dist = analysis.dist;
    const maxDist = Math.max(1, analysis.maxDist);
    const C = N - 1;

    const activeCell = new Uint8Array(C * C);
    let cellCount = 0;
    for (let iy = 0; iy < C; iy++) {
      for (let ix = 0; ix < C; ix++) {
        const i00 = iy * N + ix;
        if (fg[i00] || fg[i00 + 1] || fg[i00 + N] || fg[i00 + N + 1]) {
          activeCell[iy * C + ix] = 1;
          cellCount++;
        }
      }
    }

    if (cellCount === 0) {
      for (let iy = 1; iy < C - 1; iy++) {
        for (let ix = 1; ix < C - 1; ix++) {
          activeCell[iy * C + ix] = 1;
          fg[iy * N + ix] = 1;
        }
      }
    }

    const hasCell = (ix, iy) => (ix >= 0 && iy >= 0 && ix < C && iy < C && activeCell[iy * C + ix] === 1);
    const activeVert = new Uint8Array(N * N);
    const isBoundaryVert = new Uint8Array(N * N);
    for (let iy = 0; iy < C; iy++) {
      for (let ix = 0; ix < C; ix++) {
        if (!hasCell(ix, iy)) continue;
        activeVert[iy * N + ix] = 1;
        activeVert[iy * N + ix + 1] = 1;
        activeVert[(iy + 1) * N + ix] = 1;
        activeVert[(iy + 1) * N + ix + 1] = 1;
      }
    }

    // Identify outer silhouette perimeter vertices & collect boundary chain neighbors for tangential fairing!
    const bndNbrs = new Map();
    const linkBoundary = (a, b) => {
      if (!bndNbrs.has(a)) bndNbrs.set(a, []);
      bndNbrs.get(a).push(b);
      isBoundaryVert[a] = 1;
    };

    const boundaryEdges = [];
    for (let iy = 0; iy < C; iy++) {
      for (let ix = 0; ix < C; ix++) {
        if (!hasCell(ix, iy)) continue;
        const i00 = iy * N + ix;
        const i10 = i00 + 1;
        const i01 = i00 + N;
        const i11 = i01 + 1;
        if (!hasCell(ix, iy - 1)) { boundaryEdges.push([i10, i00]); linkBoundary(i10, i00); linkBoundary(i00, i10); }
        if (!hasCell(ix, iy + 1)) { boundaryEdges.push([i01, i11]); linkBoundary(i01, i11); linkBoundary(i11, i01); }
        if (!hasCell(ix - 1, iy)) { boundaryEdges.push([i00, i01]); linkBoundary(i00, i01); linkBoundary(i01, i00); }
        if (!hasCell(ix + 1, iy)) { boundaryEdges.push([i11, i10]); linkBoundary(i11, i10); linkBoundary(i10, i11); }
      }
    }

    // Initialize continuous grid coordinates and apply 4-pass Taubin relaxation on outer perimeter ONLY!
    // Keeps front, back & side walls 100% stitched while eliminating pixel stairsteps along curves!
    const gridX = new Float32Array(N * N);
    const gridY = new Float32Array(N * N);
    for (let iy = 0; iy < N; iy++) {
      for (let ix = 0; ix < N; ix++) {
        const vi = iy * N + ix;
        gridX[vi] = ix;
        gridY[vi] = iy;
      }
    }
    const maxStepX = 0.42;
    const maxStepY = 0.42;
    for (let pass = 0; pass < 4; pass++) {
      const k = (pass & 1) ? -0.45 : 0.48;
      const nextX = new Float32Array(gridX);
      const nextY = new Float32Array(gridY);
      for (const [vi, nbrs] of bndNbrs.entries()) {
        if (nbrs.length < 2) continue;
        let sx = 0, sy = 0;
        for (const nb of nbrs) { sx += gridX[nb]; sy += gridY[nb]; }
        const avgX = sx / nbrs.length;
        const avgY = sy / nbrs.length;
        const ix0 = vi % N, iy0 = (vi - ix0) / N;
        nextX[vi] = Math.max(ix0 - maxStepX, Math.min(ix0 + maxStepX, gridX[vi] + k * (avgX - gridX[vi])));
        nextY[vi] = Math.max(iy0 - maxStepY, Math.min(iy0 + maxStepY, gridY[vi] + k * (avgY - gridY[vi])));
      }
      gridX.set(nextX);
      gridY.set(nextY);
    }

    const frontMap = new Int32Array(N * N).fill(-1);
    const backMap = new Int32Array(N * N).fill(-1);
    const sideFrontMap = new Int32Array(N * N).fill(-1);
    const sideBackMap = new Int32Array(N * N).fill(-1);
    const positions = [];
    const uvs = [];
    const frontIndices = [];
    const backIndices = [];
    const sideIndices = [];

    const isRelief = assetMode === 'relief';
    const halfT = thicknessMm * 0.5;
    const backScale = assetMode === 'inflated' ? 0.90 : (assetMode === 'double-sided' ? 0.58 : 0.0);
    const minRimZ = isRelief ? 0.0 : halfT * 0.14;

    const pushVert = (gx, gy, zVal) => {
      const u = Math.max(0, Math.min(1, gx / (N - 1)));
      const v = Math.max(0, Math.min(1, 1.0 - gy / (N - 1)));
      positions.push(u * widthMm, v * heightMm, zVal);
      uvs.push(u, v);
      return (positions.length / 3) - 1;
    };

    const frontZAt = new Float32Array(N * N);
    const backZAt = new Float32Array(N * N);
    for (let iy = 0; iy < N; iy++) {
      for (let ix = 0; ix < N; ix++) {
        const vi = iy * N + ix;
        if (!activeVert[vi]) continue;
        const gx = gridX[vi];
        const gy = gridY[vi];
        const dNorm = fg[vi] ? depth[vi] : 0.08;
        const t = Math.min(1.0, dist[vi] / Math.max(2.0, maxDist * 0.88));
        const macroSmooth = fg[vi] ? Math.sqrt(Math.max(0.0, 1.0 - (1.0 - t) * (1.0 - t))) : 0.0;
        const isRim = isBoundaryVert[vi] === 1;

        let zF;
        if (isRelief) {
          zF = isRim ? Math.max(0.3, dNorm * thicknessMm * 0.15) : dNorm * thicknessMm;
        } else if (assetMode === 'inflated') {
          zF = isRim ? minRimZ : Math.max(minRimZ, halfT * (macroSmooth * 0.56 + dNorm * 0.44));
        } else {
          zF = isRim ? minRimZ : Math.max(minRimZ, halfT * dNorm);
        }

        let zB;
        if (isRelief) {
          zB = -Math.max(1.8, thicknessMm * 0.22);
        } else if (assetMode === 'inflated') {
          zB = isRim ? -minRimZ : -Math.max(minRimZ, halfT * backScale * (macroSmooth * 0.72 + dNorm * 0.28));
        } else {
          zB = isRim ? -minRimZ : -Math.max(minRimZ, halfT * backScale * (macroSmooth * 0.65 + dNorm * 0.35));
        }

        frontZAt[vi] = zF;
        backZAt[vi] = zB;
        frontMap[vi] = pushVert(gx, gy, zF);
        backMap[vi] = pushVert(gx, gy, zB);
        if (isRim) {
          // Dedicated side-bevel vertices share exact XYZ position so rim stays 100% watertight
          // while preventing 90-degree side normals from darkening front/back surface shading!
          sideFrontMap[vi] = pushVert(gx, gy, zF);
          sideBackMap[vi] = pushVert(gx, gy, zB);
        }
      }
    }

    for (let iy = 0; iy < C; iy++) {
      for (let ix = 0; ix < C; ix++) {
        if (!hasCell(ix, iy)) continue;
        const i00 = iy * N + ix;
        const i10 = i00 + 1;
        const i01 = i00 + N;
        const i11 = i01 + 1;

        const f00 = frontMap[i00], f10 = frontMap[i10], f01 = frontMap[i01], f11 = frontMap[i11];
        frontIndices.push(f00, f01, f10, f10, f01, f11);

        const b00 = backMap[i00], b10 = backMap[i10], b01 = backMap[i01], b11 = backMap[i11];
        backIndices.push(b00, b10, b01, b10, b11, b01);
      }
    }

    for (const [va, vb] of boundaryEdges) {
      const fa = sideFrontMap[va], fb = sideFrontMap[vb];
      const ba = sideBackMap[va], bb = sideBackMap[vb];
      if (fa >= 0 && fb >= 0 && ba >= 0 && bb >= 0) {
        sideIndices.push(fa, fb, bb, fa, bb, ba);
      }
    }

    return Asset3DEngine.finalizeSplitNormalsMesh(
      positions,
      uvs,
      [frontIndices, backIndices, sideIndices],
      'Triangle'
    );
  }

  /**
   * Seam-free 360-degree revolved Lathe object with smoothed profile curve & bilinear photo relief.
   */
  static buildLatheMeshData(analysis, widthMm, heightMm, thicknessMm) {
    const N = analysis.resolution;
    const fg = analysis.fgMask;
    const depth = analysis.depth;

    let yMin = N - 1, yMax = 0;
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        if (fg[y * N + x]) {
          if (y < yMin) yMin = y;
          if (y > yMax) yMax = y;
        }
      }
    }
    if (yMax <= yMin + 2) {
      yMin = Math.floor(N * 0.1);
      yMax = Math.ceil(N * 0.9);
    }

    const rowData = [];
    for (let y = yMin; y <= yMax; y++) {
      let lx = N, rx = -1;
      for (let x = 0; x < N; x++) {
        if (fg[y * N + x]) {
          if (x < lx) lx = x;
          if (x > rx) rx = x;
        }
      }
      if (rx >= lx) {
        rowData.push({ y, cx: (lx + rx) * 0.5, rad: Math.max(1.5, (rx - lx) * 0.5) });
      } else if (rowData.length > 0) {
        rowData.push({ ...rowData[rowData.length - 1], y });
      }
    }

    // 5-tap Gaussian vertical fairing on silhouette profile so Lathe walls never step along pixels!
    const smoothRad = rowData.map((_, i) => {
      let s = 0, w = 0;
      for (let k = -2; k <= 2; k++) {
        const j = Math.max(0, Math.min(rowData.length - 1, i + k));
        const wt = k === 0 ? 4 : Math.abs(k) === 1 ? 2 : 1;
        s += rowData[j].rad * wt;
        w += wt;
      }
      return s / w;
    });
    const smoothCx = rowData.map((_, i) => {
      let s = 0, w = 0;
      for (let k = -2; k <= 2; k++) {
        const j = Math.max(0, Math.min(rowData.length - 1, i + k));
        const wt = k === 0 ? 4 : Math.abs(k) === 1 ? 2 : 1;
        s += rowData[j].cx * wt;
        w += wt;
      }
      return s / w;
    });

    const rows = rowData.length;
    const segTheta = 72;
    const cxWorld = widthMm * 0.5;
    const positions = [];
    const uvs = [];
    const bodyIndices = [];
    const capIndices = [];

    const sampleBilinearDepth = (xf, yf) => {
      const x0 = Math.max(0, Math.min(N - 2, Math.floor(xf)));
      const y0 = Math.max(0, Math.min(N - 2, Math.floor(yf)));
      const fx = xf - x0, fy = yf - y0;
      const d00 = depth[y0 * N + x0], d10 = depth[y0 * N + x0 + 1];
      const d01 = depth[(y0 + 1) * N + x0], d11 = depth[(y0 + 1) * N + x0 + 1];
      return (d00 * (1 - fx) + d10 * fx) * (1 - fy) + (d01 * (1 - fx) + d11 * fx) * fy;
    };

    for (let ri = 0; ri < rows; ri++) {
      const rInfo = rowData[ri];
      const worldY = (1.0 - rInfo.y / (N - 1)) * heightMm;
      const baseR = (smoothRad[ri] / (N - 1)) * widthMm;
      const vCoord = 1.0 - rInfo.y / (N - 1);

      for (let si = 0; si <= segTheta; si++) {
        const uFrac = si / segTheta;
        const theta = uFrac * Math.PI * 2 - Math.PI * 0.5;
        const dirX = Math.cos(theta);
        const dirZ = Math.sin(theta);
        const sampleXf = Math.max(0, Math.min(N - 1, smoothCx[ri] + dirX * smoothRad[ri]));
        const dMod = (sampleBilinearDepth(sampleXf, rInfo.y) - 0.5) * (thicknessMm * 0.06);
        const rFinal = Math.max(0.4, baseR + dMod);

        positions.push(cxWorld + dirX * rFinal, worldY, dirZ * rFinal);
        const texU = Math.max(0, Math.min(1, sampleXf / (N - 1)));
        uvs.push(texU, vCoord);
      }
    }

    const stride = segTheta + 1;
    // Outward CCW winding: as theta advances (-Z -> +X -> +Z -> -X) and ri goes top->bottom (-Y),
    // (b - a) x (c - a) points cleanly outward along (+cos(theta), 0, +sin(theta))!
    for (let ri = 0; ri < rows - 1; ri++) {
      for (let si = 0; si < segTheta; si++) {
        const a = ri * stride + si;
        const b = a + 1;
        const c = a + stride;
        const d = c + 1;
        bodyIndices.push(a, b, c, b, d, c);
      }
    }

    // Separate flat cap vertices so top/bottom end-caps have crisp outward (+Y / -Y) normals
    const topRingBase = positions.length / 3;
    const topY = (1.0 - rowData[0].y / (N - 1)) * heightMm;
    const topInnerRow = rowData[Math.min(rows - 1, 2)];
    const topCapV = 1.0 - topInnerRow.y / (N - 1);
    for (let si = 0; si <= segTheta; si++) {
      const orig = si * 3;
      positions.push(positions[orig], positions[orig + 1], positions[orig + 2]);
      uvs.push(uvs[si * 2], topCapV);
    }
    const topCenterIdx = positions.length / 3;
    positions.push(cxWorld, topY, 0);
    uvs.push(smoothCx[Math.min(rows - 1, 2)] / (N - 1), topCapV);
    for (let si = 0; si < segTheta; si++) {
      capIndices.push(topCenterIdx, topRingBase + si + 1, topRingBase + si);
    }

    const lastRowOrig = (rows - 1) * stride;
    const botRingBase = positions.length / 3;
    const botY = (1.0 - rowData[rows - 1].y / (N - 1)) * heightMm;
    const botInnerRow = rowData[Math.max(0, rows - 3)];
    const botCapV = 1.0 - botInnerRow.y / (N - 1);
    for (let si = 0; si <= segTheta; si++) {
      const src = (lastRowOrig + si) * 3;
      positions.push(positions[src], positions[src + 1], positions[src + 2]);
      uvs.push(uvs[(lastRowOrig + si) * 2], botCapV);
    }
    const botCenterIdx = positions.length / 3;
    positions.push(cxWorld, botY, 0);
    uvs.push(smoothCx[Math.max(0, rows - 3)] / (N - 1), botCapV);
    for (let si = 0; si < segTheta; si++) {
      capIndices.push(botCenterIdx, botRingBase + si, botRingBase + si + 1);
    }

    const result = Asset3DEngine.finalizeSplitNormalsMesh(positions, uvs, [bodyIndices, capIndices], 'Triangle');
    // Seam-weld azimuthal UV seam normals on Lathe cylinder so theta=0 and theta=2PI shade invisibly!
    for (let ri = 0; ri < rows; ri++) {
      const iStart = (ri * stride) * 3;
      const iEnd = (ri * stride + segTheta) * 3;
      const nx = result.normals[iStart] + result.normals[iEnd];
      const ny = result.normals[iStart + 1] + result.normals[iEnd + 1];
      const nz = result.normals[iStart + 2] + result.normals[iEnd + 2];
      const l = Math.hypot(nx, ny, nz) || 1;
      result.normals[iStart] = result.normals[iEnd] = nx / l;
      result.normals[iStart + 1] = result.normals[iEnd + 1] = ny / l;
      result.normals[iStart + 2] = result.normals[iEnd + 2] = nz / l;
    }
    return result;
  }

  /**
   * Compute clean normals separately per triangle group (prevents side-wall creases on Front/Back shells),
   * filter zero-area degenerate triangles, and return indexed BufferData.
   */
  static finalizeSplitNormalsMesh(positions, uvs, indexGroups, topology = 'Triangle') {
    const posArr = new Float32Array(positions);
    const uvArr = new Float32Array(uvs);
    const vCount = posArr.length / 3;
    const normals = new Float32Array(posArr.length);
    const validIndices = [];

    for (const grp of indexGroups) {
      for (let i = 0; i < grp.length; i += 3) {
        const a = grp[i], b = grp[i + 1], c = grp[i + 2];
        const i0 = a * 3, i1 = b * 3, i2 = c * 3;
        const ax = posArr[i1] - posArr[i0], ay = posArr[i1 + 1] - posArr[i0 + 1], az = posArr[i1 + 2] - posArr[i0 + 2];
        const bx = posArr[i2] - posArr[i0], by = posArr[i2 + 1] - posArr[i0 + 1], bz = posArr[i2 + 2] - posArr[i0 + 2];
        const nx = ay * bz - az * by;
        const ny = az * bx - ax * bz;
        const nz = ax * by - ay * bx;
        if (nx * nx + ny * ny + nz * nz < 1e-12) continue;
        validIndices.push(a, b, c);
        normals[i0] += nx; normals[i0 + 1] += ny; normals[i0 + 2] += nz;
        normals[i1] += nx; normals[i1 + 1] += ny; normals[i1 + 2] += nz;
        normals[i2] += nx; normals[i2 + 1] += ny; normals[i2 + 2] += nz;
      }
    }

    for (let i = 0; i < vCount; i++) {
      const k = i * 3;
      const l = Math.hypot(normals[k], normals[k + 1], normals[k + 2]) || 1;
      normals[k] /= l; normals[k + 1] /= l; normals[k + 2] /= l;
    }

    const idxArr = new Uint32Array(validIndices);
    return {
      positions: posArr,
      normals,
      uvs: uvArr,
      indices: idxArr,
      stats: {
        topology,
        faces: Math.floor(idxArr.length / 3),
        vertices: vCount,
      },
    };
  }
}
