import { ImageVectorizer } from '../cad/imageVectorizer.js';
import { VectorPuffEngine } from '../cad/vectorPuffEngine.js';

// 3D Relief Sculpting & High-Accuracy Heightmap Engine for JDPaint Mac
export class ReliefEngine {
  constructor(widthMm = 100, heightMm = 100, maxDepthMm = 8, resolution = 256) {
    this.width = widthMm;
    this.height = heightMm;
    this.maxDepth = maxDepthMm;
    this.baseHeight = 0;
    this.resolution = resolution;

    // Elevation data buffer normalized [0.0 to 1.0]
    this.data = new Float32Array(this.resolution * this.resolution);
    this.listeners = [];

    // Load initial classical floral relief
    this.loadPreset('rosette');
  }

  onChange(callback) {
    this.listeners.push(callback);
  }

  notify() {
    for (const cb of this.listeners) {
      cb(this);
    }
  }

  setResolution(res) {
    if (res === this.resolution) return;
    this.resolution = res;
    this.data = new Float32Array(res * res);
    this.loadPreset('rosette');
  }

  // Bilinear sampling to get continuous physical Z (mm) at any (x, y) in mm
  getZAt(xMm, yMm) {
    if (xMm < 0 || xMm > this.width || yMm < 0 || yMm > this.height) {
      return this.baseHeight;
    }

    const u = (xMm / this.width) * (this.resolution - 1);
    const v = (yMm / this.height) * (this.resolution - 1);

    const x0 = Math.floor(u);
    const y0 = Math.floor(v);
    const x1 = Math.min(this.resolution - 1, x0 + 1);
    const y1 = Math.min(this.resolution - 1, y0 + 1);

    const fx = u - x0;
    const fy = v - y0;

    const z00 = this.data[y0 * this.resolution + x0];
    const z10 = this.data[y0 * this.resolution + x1];
    const z01 = this.data[y1 * this.resolution + x0];
    const z11 = this.data[y1 * this.resolution + x1];

    const z0 = z00 * (1 - fx) + z10 * fx;
    const z1 = z01 * (1 - fx) + z11 * fx;
    const normZ = z0 * (1 - fy) + z1 * fy;

    return this.baseHeight + normZ * this.maxDepth;
  }

  // Surface Unit Normal Vector at (x, y) in mm for CAM ball-nose cutter compensation
  getNormalAt(xMm, yMm) {
    const delta = (this.width / this.resolution);
    const zC = this.getZAt(xMm, yMm);
    const zR = this.getZAt(xMm + delta, yMm);
    const zL = this.getZAt(xMm - delta, yMm);
    const zD = this.getZAt(xMm, yMm + delta);
    const zU = this.getZAt(xMm, yMm - delta);

    const gx = (zR - zL) / (2 * delta);
    const gy = (zD - zU) / (2 * delta);

    const len = Math.hypot(-gx, -gy, 1.0);
    return {
      x: -gx / len,
      y: -gy / len,
      z: 1.0 / len,
    };
  }

  // Interactive 3D Sculpting Brush (Raise, Lower/Carve, Smooth, Flatten)
  applySculptBrush(xMm, yMm, mode = 'raise', radiusMm = 10, strength = 0.3) {
    const N = this.resolution;
    const centerU = (xMm / this.width) * (N - 1);
    const centerV = (yMm / this.height) * (N - 1);
    const radU = (radiusMm / this.width) * (N - 1);

    const minX = Math.max(0, Math.floor(centerU - radU));
    const maxX = Math.min(N - 1, Math.ceil(centerU + radU));
    const minY = Math.max(0, Math.floor(centerV - radU));
    const maxY = Math.min(N - 1, Math.ceil(centerV + radU));

    const delta = strength * 0.05;

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const dist = Math.hypot(x - centerU, y - centerV);
        if (dist <= radU) {
          const factor = (1 + Math.cos((dist / radU) * Math.PI)) * 0.5;
          const idx = y * N + x;

          if (mode === 'raise') {
            this.data[idx] = Math.min(1.0, this.data[idx] + delta * factor);
          } else if (mode === 'lower') {
            this.data[idx] = Math.max(0.0, this.data[idx] - delta * factor);
          } else if (mode === 'flatten') {
            this.data[idx] = Math.max(0.0, this.data[idx] * (1.0 - factor * 0.5));
          } else if (mode === 'smooth') {
            let sum = 0, count = 0;
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                const ny = y + dy;
                const nx = x + dx;
                if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
                  sum += this.data[ny * N + nx];
                  count++;
                }
              }
            }
            const avg = sum / count;
            this.data[idx] = this.data[idx] * (1 - factor * 0.5) + avg * (factor * 0.5);
          }
        }
      }
    }

    this.notify();
  }

  // High-Accuracy Picture to 3D Relief Conversion Engine
  // Features: Auto-Background Polarity Detection + Volumetric Euclidean Distance Rounding + Edge Enhancement + Bilateral Denoise
  loadImageHighAccuracy(imgElement, options = {}) {
    const {
      style = 'sculpted',        // 'sculpted' (volumetric + edge enhanced), 'organic', 'linear'
      edgeStrength = 0.35,
      invert = false,
      removeWhiteBg = true,      // Auto-separates solid light/dark outer background onto Z=0 plane
      denoise = true,
      gamma = 0.85,
    } = options;

    const N = this.resolution;
    const canvas = document.createElement('canvas');
    canvas.width = N;
    canvas.height = N;
    const ctx = canvas.getContext('2d');

    // Fill white padding
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, N, N);

    // Preserve aspect ratio identically to ImageVectorizer.getImageFitBounds
    const aspect = (imgElement.naturalWidth || imgElement.width || 1) / (imgElement.naturalHeight || imgElement.height || 1);
    let dw = N, dh = N;
    let dx = 0, dy = 0;
    if (aspect > 1) {
      dh = N / aspect;
      dy = (N - dh) / 2;
    } else {
      dw = N * aspect;
      dx = (N - dw) / 2;
    }

    ctx.drawImage(imgElement, dx, dy, dw, dh);
    const pixels = ctx.getImageData(0, 0, N, N).data;

    const rawGray = new Float32Array(N * N);
    const rArr = new Float32Array(N * N);
    const gArr = new Float32Array(N * N);
    const bArr = new Float32Array(N * N);
    const histogram = new Uint32Array(256);

    // 1. High-Precision Rec. 709 Luminance + Histogram
    let globalSum = 0;
    for (let i = 0; i < N * N; i++) {
      rArr[i] = pixels[i * 4];
      gArr[i] = pixels[i * 4 + 1];
      bArr[i] = pixels[i * 4 + 2];
      const lByte = Math.round(0.2126 * rArr[i] + 0.7152 * gArr[i] + 0.0722 * bArr[i]);
      rawGray[i] = lByte / 255.0;
      globalSum += rawGray[i];
      histogram[lByte]++;
    }
    const globalMean = globalSum / (N * N);

    // 1b. Wood Plaque & Carving Photo Detection -> Multi-Scale Sculpted Relief (avoids inverting wood doors from white outer room walls!)
    const otsuStats = ImageVectorizer.computeOtsuStats(histogram, N * N);
    const isTexturedPhoto = otsuStats.separability < 0.85;
    const rawLuma = ImageVectorizer.gaussianBlur3x3(rawGray, N, N);

    if (style !== 'linear') {
      const luma255 = new Float32Array(N * N);
      for (let i = 0; i < N * N; i++) luma255[i] = rawGray[i] * 255.0;
      const roi = ImageVectorizer.detectFurnitureROI(rArr, gArr, bArr, luma255, N, N);
      if (roi.chromaMed >= 12) {
        const smL = ImageVectorizer.guidedFilter(luma255, N, N, 2, 130);
        const roiMask = new Uint8Array(N * N);
        const wmRaw = new Uint8Array(N * N);
        const wmTol = roi.chromaMed > 14 ? 14 : 20;

        for (let y = roi.y0; y <= roi.y1; y++) {
          const row = y * N;
          for (let x = roi.x0; x <= roi.x1; x++) {
            const i = row + x;
            const dc = roi.chromaDiff[i];
            const rgbSat = Math.max(rArr[i], gArr[i], bArr[i]) - Math.min(rArr[i], gArr[i], bArr[i]);
            const isNeutralOverlay = roi.chromaMed > 14 && roi.L[i] > 80 && rgbSat < 20;
            if (isNeutralOverlay || (dc > wmTol && (roi.L[i] > roi.lMed + 6 || dc > 21))) {
              wmRaw[i] = 1;
            } else {
              roiMask[i] = 1;
            }
          }
        }
        // Dilate watermark mask by 1px to suppress colored overlay AA edge halos completely
        const wmMask = new Uint8Array(wmRaw);
        for (let y = roi.y0 + 1; y < roi.y1; y++) {
          const row = y * N;
          for (let x = roi.x0 + 1; x < roi.x1; x++) {
            const i = row + x;
            if (wmRaw[i - 1] || wmRaw[i + 1] || wmRaw[i - N] || wmRaw[i + N]) {
              wmMask[i] = 1;
              roiMask[i] = 0;
            }
          }
        }

        // Check if engraved 2-tone plaque (e.g. laser/burnt wood sign) vs 3D bas-relief furniture door
        const rHist = new Uint32Array(256);
        let rCnt = 0;
        for (let i = 0; i < N * N; i++) if (roiMask[i]) { rHist[Math.max(0, Math.min(255, Math.round(smL[i])))]++; rCnt++; }
        const rOtsu = rCnt > 50 ? ImageVectorizer.computeOtsuStats(rHist, rCnt) : { separability: 0, threshold: 128, m1: 100, m2: 120 };

        if (rOtsu.separability >= 0.80 && Math.abs(rOtsu.m2 - rOtsu.m1) >= 65) {
          // Flat engraved wood plaque: cancel background bamboo planks completely & trace crisp relief
          const isDarkDesign = rOtsu.m2 > rOtsu.m1;
          const bgSeed = new Uint8Array(N * N);
          for (let i = 0; i < N * N; i++) {
            if (roi.chromaDiff[i] <= 24 && (isDarkDesign ? smL[i] >= rOtsu.threshold : smL[i] <= rOtsu.threshold)) bgSeed[i] = 1;
          }
          const planeR = Math.max(8, Math.round(Math.min(roi.w, roi.h) * 0.08));
          const bgPlane = ImageVectorizer.maskedBoxMean(smL, bgSeed, N, N, planeR);
          for (let i = 0; i < N * N; i++) {
            if (roi.chromaDiff[i] > 24) { this.data[i] = 0; continue; }
            const dVal = isDarkDesign ? Math.max(0, bgPlane[i] - smL[i]) : Math.max(0, smL[i] - bgPlane[i]);
            const cleanZ = Math.max(0, Math.min(1, (dVal - 26) / 72.0));
            this.data[i] = Math.pow(cleanZ, 0.85) * 0.72;
          }
          this.notify();
          return;
        }

        const minRoi = Math.min(roi.w, roi.h);
        const minR = Math.max(2, Math.min(5, Math.round(minRoi * 0.014)));
        const medR = Math.max(5, Math.min(14, Math.round(minRoi * 0.045)));
        const openedL = ImageVectorizer.separableExtrema(ImageVectorizer.separableExtrema(smL, N, N, minR, false), N, N, minR, true);
        const closedL = ImageVectorizer.separableExtrema(ImageVectorizer.separableExtrema(smL, N, N, minR, true), N, N, minR, false);
        const openedMed = ImageVectorizer.separableExtrema(ImageVectorizer.separableExtrema(smL, N, N, medR, false), N, N, medR, true);
        const closedMed = ImageVectorizer.separableExtrema(ImageVectorizer.separableExtrema(smL, N, N, medR, true), N, N, medR, false);
        const planeR = Math.max(10, Math.round(minRoi * 0.08));
        const plane = ImageVectorizer.maskedBoxMean(smL, roiMask, N, N, planeR);
        const baseLevel = invert ? 0.65 : 0.35;
        for (let i = 0; i < N * N; i++) {
          const inRoiBox = roiMask[i] || wmMask[i];
          if (!inRoiBox) {
            this.data[i] = 0;
            continue;
          }
          if (wmMask[i]) {
            // Suppress colored/white watermark bump smoothly onto flat local door plane
            this.data[i] = Math.pow(baseLevel, gamma);
            continue;
          }
          const wh = (Math.max(0, smL[i] - openedL[i]) * 0.45 + Math.max(0, smL[i] - openedMed[i]) * 0.55) / 72.0;
          const bh = (Math.max(0, closedL[i] - smL[i]) * 0.45 + Math.max(0, closedMed[i] - smL[i]) * 0.55) / 72.0;
          const slow = (smL[i] - plane[i]) / 100.0;
          const rawDelta = (wh * 1.15 - bh * 0.95 + slow * 0.42) * (invert ? -1 : 1);
          const zNorm = Math.max(0.05, Math.min(1.0, baseLevel + rawDelta));
          this.data[i] = Math.pow(zNorm, gamma);
        }
        this.notify();
        return;
      }
    }

    if (isTexturedPhoto) {
      const sat = ImageVectorizer.buildIntegralImage(rawLuma, N, N);
      const illumRadius = Math.max(12, Math.round(N * 0.14));
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const i = y * N + x;
          const localIllum = ImageVectorizer.getLocalMean(sat, N, N, x, y, illumRadius);
          // Gently cancel uneven directional shadows across wooden panels while preserving relief peaks
          rawLuma[i] = Math.max(0.0, Math.min(1.0, rawLuma[i] - (localIllum - globalMean) * 0.55));
        }
      }
    }

    // 2. Smart Background Polarity + Border RGB Chroma Detection
    const xStart = Math.max(0, Math.floor(dx) + 1);
    const xEnd = Math.min(N - 1, Math.ceil(dx + dw) - 2);
    const yStart = Math.max(0, Math.floor(dy) + 1);
    const yEnd = Math.min(N - 1, Math.ceil(dy + dh) - 2);
    let bSum = 0, bR = 0, bG = 0, bB = 0, bCount = 0;
    for (let x = xStart; x <= xEnd; x++) {
      const i1 = yStart * N + x, i2 = yEnd * N + x;
      bSum += rawLuma[i1] + rawLuma[i2];
      bR += (pixels[i1 * 4] + pixels[i2 * 4]) / 255;
      bG += (pixels[i1 * 4 + 1] + pixels[i2 * 4 + 1]) / 255;
      bB += (pixels[i1 * 4 + 2] + pixels[i2 * 4 + 2]) / 255;
      bCount += 2;
    }
    for (let y = yStart; y <= yEnd; y++) {
      const i1 = y * N + xStart, i2 = y * N + xEnd;
      bSum += rawLuma[i1] + rawLuma[i2];
      bR += (pixels[i1 * 4] + pixels[i2 * 4]) / 255;
      bG += (pixels[i1 * 4 + 1] + pixels[i2 * 4 + 1]) / 255;
      bB += (pixels[i1 * 4 + 2] + pixels[i2 * 4 + 2]) / 255;
      bCount += 2;
    }
    const borderMean = bCount > 0 ? bSum / bCount : 1.0;
    const meanBgR = bCount > 0 ? bR / bCount : 1.0;
    const meanBgG = bCount > 0 ? bG / bCount : 1.0;
    const meanBgB = bCount > 0 ? bB / bCount : 1.0;
    const isLightBg = borderMean > 0.58;

    // 3. Polarity-corrected elevation brightness + Chroma Contrast Boost (for colored artwork on similar-luma backgrounds)
    const heightLuma = new Float32Array(N * N);
    let minL = 1.0;
    let maxL = 0.0;
    for (let i = 0; i < N * N; i++) {
      const autoFlip = isLightBg;
      const shouldFlip = invert ? !autoFlip : autoFlip;
      const lBase = shouldFlip ? 1.0 - rawLuma[i] : rawLuma[i];
      const chromaDist = Math.hypot(
        pixels[i * 4] / 255 - meanBgR,
        pixels[i * 4 + 1] / 255 - meanBgG,
        pixels[i * 4 + 2] / 255 - meanBgB
      ) / 1.732;
      const l = Math.max(lBase, chromaDist * 0.9);
      heightLuma[i] = l;
      if (l < minL) minL = l;
      if (l > maxL) maxL = l;
    }

    // 4. Contrast Normalization + Foreground Mask via Otsu & Chroma
    const otsuNorm = otsuStats.threshold / 255.0;
    const fgThreshold = isLightBg ? 1.0 - otsuNorm : otsuNorm;
    const range = Math.max(0.05, maxL - minL);
    const normalized = new Float32Array(N * N);
    const fgMask = new Uint8Array(N * N);

    for (let i = 0; i < N * N; i++) {
      let val = Math.max(0.0, Math.min(1.0, (heightLuma[i] - minL) / range));
      const inBounds = (i % N) >= Math.floor(dx) && (i % N) <= Math.ceil(dx + dw) &&
                       Math.floor(i / N) >= Math.floor(dy) && Math.floor(i / N) <= Math.ceil(dy + dh);

      const chromaDist = Math.hypot(
        pixels[i * 4] / 255 - meanBgR,
        pixels[i * 4 + 1] / 255 - meanBgG,
        pixels[i * 4 + 2] / 255 - meanBgB
      ) / 1.732;

      if (!inBounds) {
        val = 0.0;
      } else if (removeWhiteBg && isLightBg && rawLuma[i] > 0.93 && chromaDist < 0.08) {
        val = 0.0;
      } else if (removeWhiteBg && !isLightBg && rawLuma[i] < 0.06 && chromaDist < 0.08) {
        val = 0.0;
      }

      normalized[i] = val;
      if (inBounds && val > Math.max(0.12, fgThreshold * 0.42)) {
        fgMask[i] = 1;
      }
    }

    // 5. Volumetric Distance-Field Dome inside Foreground Shapes (True 3D Carved Relief!)
    const distField = VectorPuffEngine.computeInteriorDistance(fgMask, N);
    let maxDist = 1.0;
    for (let i = 0; i < N * N; i++) {
      if (distField[i] > maxDist) maxDist = distField[i];
    }
    const normRadiusPx = Math.max(4.0, Math.min(N * 0.12, maxDist * 0.75));

    // 6. Sobel Edge Detection for Crisp Detail
    const edgeMap = new Float32Array(N * N);
    if (style === 'sculpted' || edgeStrength > 0) {
      for (let y = 1; y < N - 1; y++) {
        for (let x = 1; x < N - 1; x++) {
          const gx =
            -normalized[(y - 1) * N + (x - 1)] + normalized[(y - 1) * N + (x + 1)] +
            -2 * normalized[y * N + (x - 1)] + 2 * normalized[y * N + (x + 1)] +
            -normalized[(y + 1) * N + (x - 1)] + normalized[(y + 1) * N + (x + 1)];

          const gy =
            -normalized[(y - 1) * N + (x - 1)] - 2 * normalized[(y - 1) * N + x] - normalized[(y - 1) * N + (x + 1)] +
            normalized[(y + 1) * N + (x - 1)] + 2 * normalized[(y + 1) * N + x] + normalized[(y + 1) * N + (x + 1)];

          edgeMap[y * N + x] = Math.min(1.0, Math.hypot(gx, gy) * 1.25);
        }
      }
    }

    // 7. Combine Volumetric Dome + Surface Luminance Shading + Edge Crispness
    const blended = new Float32Array(N * N);
    for (let i = 0; i < N * N; i++) {
      const lumaVal = normalized[i];
      const edgeVal = edgeMap[i];
      const t = Math.min(1.0, distField[i] / normRadiusPx);
      const domeVal = fgMask[i] ? Math.sqrt(Math.max(0, 1.0 - Math.pow(1.0 - t, 2))) : 0.0;

      let finalZ = lumaVal;
      if (style === 'sculpted') {
        const baseVolume = domeVal * 0.55 + lumaVal * 0.45;
        finalZ = baseVolume * (1.0 - edgeStrength * 0.25) + edgeVal * edgeStrength * 0.35;
        finalZ = Math.pow(Math.max(0, finalZ), gamma);
      } else if (style === 'organic') {
        finalZ = Math.pow(domeVal * 0.65 + lumaVal * 0.35, 0.9);
      }

      blended[i] = Math.max(0.0, Math.min(1.0, finalZ));
    }

    // 8. Bilateral Edge-Preserving Denoising
    if (denoise) {
      for (let y = 1; y < N - 1; y++) {
        for (let x = 1; x < N - 1; x++) {
          const idx = y * N + x;
          const center = blended[idx];
          let weightSum = 0;
          let valSum = 0;

          for (let dy = -1; dy <= 1; dy++) {
            for (let dx2 = -1; dx2 <= 1; dx2++) {
              const nIdx = (y + dy) * N + (x + dx2);
              const neighbor = blended[nIdx];
              const diff = Math.abs(neighbor - center);
              const w = Math.exp(-(diff * diff) / 0.08);
              valSum += neighbor * w;
              weightSum += w;
            }
          }
          this.data[idx] = valSum / (weightSum || 1);
        }
      }
    } else {
      this.data.set(blended);
    }

    this.notify();
  }

  loadImage(imgElement) {
    this.loadImageHighAccuracy(imgElement, { style: 'sculpted', edgeStrength: 0.35, denoise: true });
  }

  // Preset Sculptures
  loadPreset(name) {
    const N = this.resolution;
    this.data.fill(0);

    if (name === 'rosette') {
      const cx = N / 2;
      const cy = N / 2;
      const maxR = N * 0.45;

      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const dx = x - cx;
          const dy = y - cy;
          const r = Math.sqrt(dx * dx + dy * dy);
          const theta = Math.atan2(dy, dx);

          if (r < maxR) {
            const dome = Math.cos((r / maxR) * (Math.PI / 2));
            const petals = Math.sin(theta * 8) * 0.25 + 0.25;
            const rings = Math.cos((r / maxR) * Math.PI * 6) * 0.2 + 0.2;
            const pearl = r < maxR * 0.2 ? Math.cos((r / (maxR * 0.2)) * (Math.PI / 2)) * 0.4 : 0;

            const val = dome * 0.4 + petals * (1 - r / maxR) * 0.4 + rings * 0.15 + pearl;
            this.data[y * N + x] = Math.max(0, Math.min(1, val));
          }
        }
      }
    } else if (name === 'dome') {
      const cx = N / 2;
      const cy = N / 2;
      const maxR = N * 0.42;

      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const r = Math.hypot(x - cx, y - cy);
          if (r < maxR) {
            this.data[y * N + x] = Math.cos((r / maxR) * (Math.PI / 2));
          }
        }
      }
    } else if (name === 'pyramid') {
      const cx = N / 2;
      const cy = N / 2;
      const maxR = N * 0.4;

      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const dx = Math.abs(x - cx);
          const dy = Math.abs(y - cy);
          const dist = Math.max(dx, dy);
          if (dist < maxR) {
            this.data[y * N + x] = 1.0 - dist / maxR;
          }
        }
      }
    }

    this.notify();
  }

  smooth(passes = 1) {
    const N = this.resolution;
    for (let p = 0; p < passes; p++) {
      const copy = new Float32Array(this.data);
      for (let y = 1; y < N - 1; y++) {
        for (let x = 1; x < N - 1; x++) {
          const idx = y * N + x;
          const avg =
            (copy[idx] * 4 +
              copy[idx - 1] * 2 +
              copy[idx + 1] * 2 +
              copy[idx - N] * 2 +
              copy[idx + N] * 2 +
              copy[idx - N - 1] +
              copy[idx - N + 1] +
              copy[idx + N - 1] +
              copy[idx + N + 1]) /
            16.0;
          this.data[idx] = avg;
        }
      }
    }
    this.notify();
  }

  invert() {
    for (let i = 0; i < this.data.length; i++) {
      if (this.data[i] > 0.001) {
        this.data[i] = 1.0 - this.data[i];
      }
    }
    this.notify();
  }

  clear() {
    this.data.fill(0);
    this.notify();
  }
}
