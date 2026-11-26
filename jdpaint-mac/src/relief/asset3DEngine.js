import { ImageVectorizer } from '../cad/imageVectorizer.js';
import { VectorPuffEngine } from '../cad/vectorPuffEngine.js';

/**
 * Meshy-Style Image-to-3D Volumetric Asset Generator & PBR Map Synthesizer.
 * Converts any loaded image (or alpha silhouette) into a watertight 3D mesh
 * ('double-sided', 'inflated', 'lathe', or 'relief') along with Base Color,
 * Normal, Roughness, and Metallic maps ready for WebGL PBR preview & GLB/OBJ/STL export.
 */
export class Asset3DEngine {
  /**
   * Extract foreground silhouette, RGB albedo, de-lighted height field, and PBR channel maps from an image.
   */
  static analyzeImage(imgElement, options = {}) {
    const {
      resolution = 160,
      removeBg = true,
      delight = true,
      invert = false,
      edgeStrength = 0.35,
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

    // Also render high-res 512px texture buffer for crisp Base Color / PBR maps
    const texSize = 512;
    const texCanvas = document.createElement('canvas');
    texCanvas.width = texSize;
    texCanvas.height = texSize;
    const texCtx = texCanvas.getContext('2d', { willReadFrequently: true });
    texCtx.clearRect(0, 0, texSize, texSize);
    const texScale = texSize / N;
    texCtx.drawImage(imgElement, dx * texScale, dy * texScale, dw * texScale, dh * texScale);
    const texRgba = texCtx.getImageData(0, 0, texSize, texSize).data;

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

    // Compute smooth volumetric geodesic/Euclidean distance transform inside foreground
    const dist = VectorPuffEngine.computeInteriorDistance(fgMask, N);
    let maxDist = 1.0;
    for (let i = 0; i < total; i++) {
      if (dist[i] > maxDist) maxDist = dist[i];
    }

    // Light-normalized interior luminance so shadowed creases don't cave the whole back/side unfairly
    const lumaSmooth = ImageVectorizer.gaussianBlur3x3(luma, N, N);
    const sat = ImageVectorizer.buildIntegralImage(lumaSmooth, N, N);
    const illumR = Math.max(6, Math.round(N * 0.12));

    const detailHeight = new Float32Array(total);
    let dMin = 1.0;
    let dMax = 0.0;
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const i = y * N + x;
        if (!fgMask[i]) continue;
        const localMean = ImageVectorizer.getLocalMean(sat, N, N, x, y, illumR);
        let base = delight
          ? lumaSmooth[i] * 0.62 + (lumaSmooth[i] - localMean + 0.5) * 0.38
          : lumaSmooth[i];
        if (invert) base = 1.0 - base;
        base = Math.max(0.0, Math.min(1.0, base));
        detailHeight[i] = base;
        if (base < dMin) dMin = base;
        if (base > dMax) dMax = base;
      }
    }

    const dSpan = Math.max(0.08, dMax - dMin);
    const combinedDepth = new Float32Array(total);
    for (let y = 1; y < N - 1; y++) {
      for (let x = 1; x < N - 1; x++) {
        const i = y * N + x;
        if (!fgMask[i]) continue;
        const t = Math.min(1.0, dist[i] / Math.max(3.0, maxDist * 0.92));
        const macroDome = Math.sqrt(Math.max(0.0, 1.0 - (1.0 - t) * (1.0 - t)));
        const microRelief = (detailHeight[i] - dMin) / dSpan;
        const gx = lumaSmooth[i + 1] - lumaSmooth[i - 1];
        const gy = lumaSmooth[i + N] - lumaSmooth[i - N];
        const edgeAccent = Math.min(0.28, Math.hypot(gx, gy) * edgeStrength * 0.9);
        combinedDepth[i] = Math.max(
          0.03,
          Math.min(1.0, macroDome * 0.72 + microRelief * 0.28 + edgeAccent * 0.15)
        );
      }
    }

    const denoisedDepth = ImageVectorizer.guidedFilter(combinedDepth, N, N, 1, 0.015);
    for (let i = 0; i < total; i++) {
      if (!fgMask[i]) denoisedDepth[i] = 0;
      else denoisedDepth[i] = Math.max(0.04, Math.min(1.0, denoisedDepth[i]));
    }

    const maps = Asset3DEngine.synthesizePBRMaps(texRgba, texSize, delight);

    return {
      resolution: N,
      fgMask,
      dist,
      maxDist,
      depth: denoisedDepth,
      detailHeight,
      texSize,
      maps,
      boundsPx: { dx, dy, dw, dh },
    };
  }

  /**
   * Flood-fill outward from border in CIELAB so interior bright highlights stay solid and never punch holes!
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

    // Sample median border color inside image content box
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

    // Border-connected BFS background peel
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
      seedIfBg(x, y0);
      seedIfBg(x, y1);
      seedIfBg(x, 0);
      seedIfBg(x, N - 1);
    }
    for (let y = 0; y < N; y++) {
      seedIfBg(x0, y);
      seedIfBg(x1, y);
      seedIfBg(0, y);
      seedIfBg(N - 1, y);
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

    // Fallback: if border flood accidentally ate >92% (full-bleed photo), restore interior box
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
    // 1-pass majority close + prune tiny disconnected floater dots (< 0.4% of image) so GLB/STL stays clean
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
   * Synthesize Base Color, Normal, Roughness, and Metallic 512x512 Uint8ClampedArray maps from photo RGBA.
   */
  static synthesizePBRMaps(rgba, size, delight = true) {
    const n = size * size;
    const baseColor = new Uint8ClampedArray(rgba);
    const normal = new Uint8ClampedArray(n * 4);
    const roughness = new Uint8ClampedArray(n * 4);
    const metallic = new Uint8ClampedArray(n * 4);

    const lum = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const r = rgba[i * 4];
      const g = rgba[i * 4 + 1];
      const b = rgba[i * 4 + 2];
      lum[i] = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
    }

    const sat = ImageVectorizer.buildIntegralImage(lum, size, size);
    const rIllum = Math.max(12, Math.round(size * 0.09));

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const i = y * size + x;
        const p = i * 4;
        const r = rgba[p];
        const g = rgba[p + 1];
        const b = rgba[p + 2];
        const a = rgba[p + 3];

        // Gentle studio de-lighting so dark ambient shadows don't muddy lit 3D textures
        if (delight && a > 16) {
          const localMean = Math.max(0.18, ImageVectorizer.getLocalMean(sat, size, size, x, y, rIllum));
          const gain = Math.max(0.75, Math.min(1.42, 0.54 / localMean * 0.45 + 0.55));
          baseColor[p] = Math.min(255, Math.round(r * gain));
          baseColor[p + 1] = Math.min(255, Math.round(g * gain));
          baseColor[p + 2] = Math.min(255, Math.round(b * gain));
        }

        // Sobel tangent-space Normal map
        const xl = Math.max(0, x - 1);
        const xr = Math.min(size - 1, x + 1);
        const yu = Math.max(0, y - 1);
        const yd = Math.min(size - 1, y + 1);
        const dx = (lum[y * size + xr] - lum[y * size + xl]) * 2.8;
        const dy = (lum[yd * size + x] - lum[yu * size + x]) * 2.8;
        let nx = -dx;
        let ny = dy;
        let nz = 1.0;
        const inv = 1.0 / Math.hypot(nx, ny, nz);
        nx *= inv;
        ny *= inv;
        nz *= inv;
        normal[p] = Math.round((nx * 0.5 + 0.5) * 255);
        normal[p + 1] = Math.round((ny * 0.5 + 0.5) * 255);
        normal[p + 2] = Math.round((nz * 0.5 + 0.5) * 255);
        normal[p + 3] = 255;

        // Roughness & Metallic inference from local micro-variance + specular saturation
        const mx = Math.max(r, g, b);
        const mn = Math.min(r, g, b);
        const ch = (mx - mn) / 255.0;
        const gradMag = Math.min(1.0, Math.hypot(dx, dy) * 1.4);
        const roughVal = Math.max(0.22, Math.min(0.92, 0.72 - lum[i] * 0.25 + gradMag * 0.22));
        const rv = Math.round(roughVal * 255);
        roughness[p] = rv;
        roughness[p + 1] = rv;
        roughness[p + 2] = rv;
        roughness[p + 3] = 255;

        const isGoldOrSteel = (lum[i] > 0.48 && (ch < 0.14 || (r > g && g > b && ch > 0.20)));
        const mv = isGoldOrSteel ? Math.round((0.35 + lum[i] * 0.45) * 255) : 18;
        metallic[p] = mv;
        metallic[p + 1] = mv;
        metallic[p + 2] = mv;
        metallic[p + 3] = 255;
      }
    }

    return { size, baseColor, normal, roughness, metallic };
  }

  /**
   * Build indexed watertight 3D BufferData ({ positions, normals, uvs, indices, stats })
   * supporting 'double-sided', 'inflated', 'lathe', and 'relief' modes.
   */
  static buildAssetMeshData(analysis, options = {}) {
    const {
      assetMode = 'double-sided', // 'double-sided' | 'inflated' | 'lathe' | 'relief'
      widthMm = 100,
      heightMm = 100,
      thicknessMm = 18,
    } = options;

    if (assetMode === 'lathe') {
      return Asset3DEngine.buildLatheMeshData(analysis, widthMm, heightMm, thicknessMm);
    }
    return Asset3DEngine.buildVolumetricShellMeshData(analysis, assetMode, widthMm, heightMm, thicknessMm);
  }

  static buildVolumetricShellMeshData(analysis, assetMode, widthMm, heightMm, thicknessMm) {
    const N = analysis.resolution;
    const fg = analysis.fgMask;
    const depth = analysis.depth;
    const dist = analysis.dist;
    const maxDist = Math.max(1, analysis.maxDist);

    const activeCell = new Uint8Array((N - 1) * (N - 1));
    let cellCount = 0;
    for (let iy = 0; iy < N - 1; iy++) {
      for (let ix = 0; ix < N - 1; ix++) {
        const i00 = iy * N + ix;
        const on = fg[i00] || fg[i00 + 1] || fg[i00 + N] || fg[i00 + N + 1];
        if (on) {
          activeCell[iy * (N - 1) + ix] = 1;
          cellCount++;
        }
      }
    }

    // Fallback to center patch if mask is somehow empty
    if (cellCount === 0) {
      for (let iy = 1; iy < N - 2; iy++) {
        for (let ix = 1; ix < N - 2; ix++) {
          activeCell[iy * (N - 1) + ix] = 1;
          fg[iy * N + ix] = 1;
        }
      }
    }

    const activeVert = new Uint8Array(N * N);
    for (let iy = 0; iy < N - 1; iy++) {
      for (let ix = 0; ix < N - 1; ix++) {
        if (!activeCell[iy * (N - 1) + ix]) continue;
        activeVert[iy * N + ix] = 1;
        activeVert[iy * N + ix + 1] = 1;
        activeVert[(iy + 1) * N + ix] = 1;
        activeVert[(iy + 1) * N + ix + 1] = 1;
      }
    }

    const frontMap = new Int32Array(N * N).fill(-1);
    const backMap = new Int32Array(N * N).fill(-1);
    const positions = [];
    const uvs = [];
    const isRelief = assetMode === 'relief';
    const halfT = thicknessMm * 0.5;
    const backScale = assetMode === 'inflated' ? 0.92 : (assetMode === 'double-sided' ? 0.55 : 0.0);

    const addVertex = (ix, iy, zVal) => {
      const u = ix / (N - 1);
      const v = 1.0 - iy / (N - 1);
      const x = u * widthMm;
      const y = v * heightMm;
      positions.push(x, y, zVal);
      uvs.push(u, v);
      return (positions.length / 3) - 1;
    };

    for (let iy = 0; iy < N; iy++) {
      for (let ix = 0; ix < N; ix++) {
        const vi = iy * N + ix;
        if (!activeVert[vi]) continue;
        const dNorm = fg[vi] ? depth[vi] : 0.0;
        const t = Math.min(1.0, dist[vi] / Math.max(2.0, maxDist * 0.92));
        const macroSmooth = fg[vi] ? Math.sqrt(Math.max(0.0, 1.0 - (1.0 - t) * (1.0 - t))) : 0.0;
        const zFront = isRelief
          ? dNorm * thicknessMm
          : halfT * (assetMode === 'inflated' ? (macroSmooth * 0.72 + dNorm * 0.38) : (dNorm * 0.95));
        frontMap[vi] = addVertex(ix, iy, zFront);

        const zBack = isRelief
          ? -Math.max(1.5, thicknessMm * 0.2)
          : -halfT * backScale * (assetMode === 'inflated' ? macroSmooth : dNorm);
        backMap[vi] = addVertex(ix, iy, zBack);
      }
    }

    const indices = [];
    const C = N - 1;
    const hasCell = (ix, iy) => (ix >= 0 && iy >= 0 && ix < C && iy < C && activeCell[iy * C + ix] === 1);

    for (let iy = 0; iy < C; iy++) {
      for (let ix = 0; ix < C; ix++) {
        if (!hasCell(ix, iy)) continue;
        const i00 = iy * N + ix;
        const i10 = i00 + 1;
        const i01 = i00 + N;
        const i11 = i01 + 1;

        // Front faces (+Z outward, image Y top -> bottom so i01 is lower Y than i00)
        const f00 = frontMap[i00], f10 = frontMap[i10], f01 = frontMap[i01], f11 = frontMap[i11];
        indices.push(f00, f01, f10, f10, f01, f11);

        // Back faces (-Z outward)
        const b00 = backMap[i00], b10 = backMap[i10], b01 = backMap[i01], b11 = backMap[i11];
        indices.push(b00, b10, b01, b10, b11, b01);

        // Watertight perimeter side walls whenever neighbor cell is empty
        const stitchEdge = (va, vb) => {
          const fa = frontMap[va], fb = frontMap[vb];
          const ba = backMap[va], bb = backMap[vb];
          indices.push(fa, fb, bb, fa, bb, ba);
        };
        if (!hasCell(ix, iy - 1)) stitchEdge(i10, i00); // Top boundary (outward +Y)
        if (!hasCell(ix, iy + 1)) stitchEdge(i01, i11); // Bottom boundary (outward -Y)
        if (!hasCell(ix - 1, iy)) stitchEdge(i00, i01); // Left boundary (outward -X)
        if (!hasCell(ix + 1, iy)) stitchEdge(i11, i10); // Right boundary (outward +X)
      }
    }

    return Asset3DEngine.finalizeMeshStats(positions, uvs, indices, 'Triangle');
  }

  /**
   * Revolve the silhouette radial profile 360° around its vertical axis to build a full circular 3D object.
   */
  static buildLatheMeshData(analysis, widthMm, heightMm, thicknessMm) {
    const N = analysis.resolution;
    const fg = analysis.fgMask;
    const depth = analysis.depth;

    let yMin = N - 1;
    let yMax = 0;
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
      let lx = N;
      let rx = -1;
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

    const rows = rowData.length;
    const segTheta = 64;
    const cxWorld = widthMm * 0.5;
    const positions = [];
    const uvs = [];
    const indices = [];

    for (let ri = 0; ri < rows; ri++) {
      const rInfo = rowData[ri];
      const worldY = (1.0 - rInfo.y / (N - 1)) * heightMm;
      const baseR = (rInfo.rad / (N - 1)) * widthMm;
      const vCoord = 1.0 - rInfo.y / (N - 1);

      for (let si = 0; si <= segTheta; si++) {
        const uFrac = si / segTheta;
        const theta = uFrac * Math.PI * 2 - Math.PI * 0.5;
        // Sample photo relief detail subtly across both front/back azimuth
        const dirX = Math.cos(theta);
        const dirZ = Math.sin(theta);
        const sampleU = Math.max(0, Math.min(N - 1, Math.round(rInfo.cx + dirX * rInfo.rad)));
        const dMod = fg[rInfo.y * N + sampleU] ? (depth[rInfo.y * N + sampleU] - 0.5) * (thicknessMm * 0.14) : 0;
        const rFinal = Math.max(0.3, baseR + dMod);

        positions.push(cxWorld + dirX * rFinal, worldY, dirZ * rFinal);
        const texU = Math.max(0, Math.min(1, (rInfo.cx + dirX * rInfo.rad) / (N - 1)));
        uvs.push(texU, vCoord);
      }
    }

    const stride = segTheta + 1;
    for (let ri = 0; ri < rows - 1; ri++) {
      for (let si = 0; si < segTheta; si++) {
        const a = ri * stride + si;
        const b = a + 1;
        const c = a + stride;
        const d = c + 1;
        indices.push(a, c, b, b, c, d);
      }
    }

    // Close top and bottom caps cleanly for watertight STL/GLB export
    const topCenterIdx = positions.length / 3;
    positions.push(cxWorld, (1.0 - rowData[0].y / (N - 1)) * heightMm, 0);
    uvs.push(0.5, 1.0 - rowData[0].y / (N - 1));
    for (let si = 0; si < segTheta; si++) {
      indices.push(topCenterIdx, si, si + 1);
    }

    const botCenterIdx = positions.length / 3;
    const lastRowBase = (rows - 1) * stride;
    positions.push(cxWorld, (1.0 - rowData[rows - 1].y / (N - 1)) * heightMm, 0);
    uvs.push(0.5, 1.0 - rowData[rows - 1].y / (N - 1));
    for (let si = 0; si < segTheta; si++) {
      indices.push(botCenterIdx, lastRowBase + si + 1, lastRowBase + si);
    }

    return Asset3DEngine.finalizeMeshStats(positions, uvs, indices, 'Triangle');
  }

  static finalizeMeshStats(positions, uvs, indices, topology = 'Triangle') {
    const posArr = new Float32Array(positions);
    const uvArr = new Float32Array(uvs);
    const idxArr = new Uint32Array(indices);
    const vCount = posArr.length / 3;
    const normals = new Float32Array(posArr.length);

    for (let i = 0; i < idxArr.length; i += 3) {
      const i0 = idxArr[i] * 3;
      const i1 = idxArr[i + 1] * 3;
      const i2 = idxArr[i + 2] * 3;
      const ax = posArr[i1] - posArr[i0];
      const ay = posArr[i1 + 1] - posArr[i0 + 1];
      const az = posArr[i1 + 2] - posArr[i0 + 2];
      const bx = posArr[i2] - posArr[i0];
      const by = posArr[i2 + 1] - posArr[i0 + 1];
      const bz = posArr[i2 + 2] - posArr[i0 + 2];
      const nx = ay * bz - az * by;
      const ny = az * bx - ax * bz;
      const nz = ax * by - ay * bx;
      normals[i0] += nx; normals[i0 + 1] += ny; normals[i0 + 2] += nz;
      normals[i1] += nx; normals[i1 + 1] += ny; normals[i1 + 2] += nz;
      normals[i2] += nx; normals[i2 + 1] += ny; normals[i2 + 2] += nz;
    }

    for (let i = 0; i < vCount; i++) {
      const k = i * 3;
      const l = Math.hypot(normals[k], normals[k + 1], normals[k + 2]) || 1;
      normals[k] /= l;
      normals[k + 1] /= l;
      normals[k + 2] /= l;
    }

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
