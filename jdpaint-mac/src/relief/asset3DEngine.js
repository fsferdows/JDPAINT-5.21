import { ImageVectorizer } from '../cad/imageVectorizer.js';
import { VectorPuffEngine } from '../cad/vectorPuffEngine.js';

/**
 * Meshy-Style Image-to-3D Volumetric Asset Generator & PBR Map Synthesizer.
 * Supports BOTH:
 * - Planar Architectural / Carved Wood Reliefs (panels, doors, plaques): aspect-fitted high-density grids,
 *   watermark suppression, grain smoothing, flat coplanar datum rails & flat print-ready backs.
 * - Organic 3D Cutout Assets (statues, coins, medallions, vases): smooth harmonic inflation (zero diagonal creases),
 *   Taubin perimeter fairing (zero staircasing), split bevel normals, seam-free Lathe 360, & Voronoi-padded 1024px PBR maps.
 */
export class Asset3DEngine {
  /**
   * Extract foreground/planar structure, adaptive aspect-matched depth field, and 1024px PBR maps.
   */
  static analyzeImage(imgElement, options = {}) {
    const {
      resolution = 160,
      texSize = 1024,
      removeBg = true,
      delight = true,
      invert = false,
      edgeStrength = 0.35,
    } = options;

    const baseRes = Math.max(48, Math.min(320, Math.round(resolution)));
    const iw = imgElement.naturalWidth || imgElement.width || 1;
    const ih = imgElement.naturalHeight || imgElement.height || 1;
    const imgAspect = iw / Math.max(1, ih);

    // Choose adaptive rectangular grid (resX x resY) matching source aspect & lifting High/Ultra detail resolution
    let resX = baseRes;
    let resY = baseRes;
    if (baseRes >= 128 || Math.abs(imgAspect - 1.0) >= 0.05) {
      const areaBudget = baseRes * baseRes * (baseRes >= 144 ? 1.65 : 1.45);
      resX = Math.max(84, Math.min(360, Math.round(Math.sqrt(areaBudget * imgAspect))));
      resY = Math.max(84, Math.min(360, Math.round(Math.sqrt(areaBudget / imgAspect))));
    }

    const drawToCanvas = (w, h, sx = 0, sy = 0, sw = iw, sh = ih) => {
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.clearRect(0, 0, w, h);
      if (ctx.imageSmoothingQuality !== undefined) ctx.imageSmoothingQuality = 'high';
      if (sx !== 0 || sy !== 0 || sw !== iw || sh !== ih) {
        ctx.drawImage(imgElement, sx, sy, sw, sh, 0, 0, w, h);
      } else {
        ctx.drawImage(imgElement, 0, 0, w, h);
      }
      return ctx.getImageData(0, 0, w, h).data;
    };

    let rgba = drawToCanvas(resX, resY);
    let activeAspect = imgAspect;
    let srcCropX = 0;
    let srcCropY = 0;
    let srcCropW = iw;
    let srcCropH = ih;

    // Detect if real browser image has wide uniform studio margins around a narrow subject and tighten crop for 2-3x detail
    const isRealDomImage = !!(imgElement && (imgElement.tagName === 'IMG' || imgElement.nodeName === 'IMG' || (typeof HTMLImageElement !== 'undefined' && imgElement instanceof HTMLImageElement)));
    let analysisData = Asset3DEngine._analyzeBuffer(rgba, resX, resY, { removeBg, delight, invert, edgeStrength });

    if (isRealDomImage && removeBg && analysisData.fgBox) {
      const { minX, maxX, minY, maxY } = analysisData.fgBox;
      const bw = maxX - minX + 1;
      const bh = maxY - minY + 1;
      const covX = bw / resX;
      const covY = bh / resY;
      if ((covX < 0.78 || covY < 0.78) && covX > 0.16 && covY > 0.16) {
        const padX = Math.max(2, Math.round(bw * 0.025));
        const padY = Math.max(2, Math.round(bh * 0.025));
        const cx0 = Math.max(0, minX - padX);
        const cy0 = Math.max(0, minY - padY);
        const cx1 = Math.min(resX - 1, maxX + padX);
        const cy1 = Math.min(resY - 1, maxY + padY);
        srcCropX = (cx0 / (resX - 1)) * iw;
        srcCropY = (cy0 / (resY - 1)) * ih;
        srcCropW = Math.max(8, ((cx1 - cx0) / (resX - 1)) * iw);
        srcCropH = Math.max(8, ((cy1 - cy0) / (resY - 1)) * ih);
        activeAspect = srcCropW / Math.max(1, srcCropH);
        const areaBudget = baseRes * baseRes * 1.5;
        resX = Math.max(96, Math.min(360, Math.round(Math.sqrt(areaBudget * activeAspect))));
        resY = Math.max(96, Math.min(360, Math.round(Math.sqrt(areaBudget / activeAspect))));
        rgba = drawToCanvas(resX, resY, srcCropX, srcCropY, srcCropW, srcCropH);
        analysisData = Asset3DEngine._analyzeBuffer(rgba, resX, resY, { removeBg, delight, invert, edgeStrength });
      }
    }

    const tSize = Math.max(256, Math.min(1024, Math.round(texSize)));
    const texRgba = drawToCanvas(tSize, tSize, srcCropX, srcCropY, srcCropW, srcCropH);

    // Infinite Voronoi edge padding from pure interior foreground seed so zero backdrop white/black halo bleeds into UVs
    Asset3DEngine.dilateTextureEdges(
      texRgba,
      tSize,
      analysisData.fgMask,
      resX,
      resY,
      removeBg && !analysisData.hasTransparency
    );

    const maps = Asset3DEngine.synthesizePBRMaps(texRgba, tSize, delight, analysisData.fgMask, resX, resY);

    return {
      resolution: Math.max(resX, resY),
      resX,
      resY,
      aspect: activeAspect,
      isPlanarPanel: analysisData.isPlanarPanel,
      fgMask: analysisData.fgMask,
      dist: analysisData.smoothDist,
      rawDist: analysisData.rawDist,
      maxDist: analysisData.maxDist,
      depth: analysisData.depth,
      sculptSignal: analysisData.sculptSignal,
      texSize: tSize,
      maps,
      aspectBox: analysisData.aspectBox,
      boundsPx: { dx: 0, dy: 0, dw: resX, dh: resY },
    };
  }

  /**
   * Analyze a rasterized (resX x resY) RGBA buffer into foreground mask, planar panel classification,
   * denoised multi-scale depth field, and smoothed distance envelope.
   */
  static _analyzeBuffer(rgba, resX, resY, opts) {
    const { removeBg, delight, invert, edgeStrength } = opts;
    const total = resX * resY;
    const rArr = new Float32Array(total);
    const gArr = new Float32Array(total);
    const bArr = new Float32Array(total);
    const luma = new Float32Array(total);
    const luma255 = new Float32Array(total);
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
      rArr[i] = r;
      gArr[i] = g;
      bArr[i] = b;
      alpha[i] = a;
      if (a < 230) hasTransparency = true;
      const lv = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
      luma[i] = lv;
      luma255[i] = lv * 255.0;
      const [cl, ca, cb] = ImageVectorizer.rgbToLab(r, g, b);
      chromaL[i] = cl;
      chromaA[i] = ca;
      chromaB[i] = cb;
    }

    const fgMask = Asset3DEngine.extractBorderConnectedForeground(
      alpha,
      chromaL,
      chromaA,
      chromaB,
      resX,
      resY,
      { removeBg, hasTransparency }
    );

    let minFgX = resX - 1, maxFgX = 0, minFgY = resY - 1, maxFgY = 0, fgCount = 0;
    for (let y = 0; y < resY; y++) {
      const row = y * resX;
      for (let x = 0; x < resX; x++) {
        if (!fgMask[row + x]) continue;
        fgCount++;
        if (x < minFgX) minFgX = x;
        if (x > maxFgX) maxFgX = x;
        if (y < minFgY) minFgY = y;
        if (y > maxFgY) maxFgY = y;
      }
    }
    if (fgCount === 0) {
      fgMask.fill(1);
      minFgX = 0; maxFgX = resX - 1; minFgY = 0; maxFgY = resY - 1;
      fgCount = total;
    }

    const fgBoxW = Math.max(1, maxFgX - minFgX + 1);
    const fgBoxH = Math.max(1, maxFgY - minFgY + 1);
    const fillRatio = fgCount / (fgBoxW * fgBoxH);

    // Check warm/furniture/architectural panel ROI and rectangularity -> prevents bulbous pillow distortion on carved doors & plaques!
    const roi = ImageVectorizer.detectFurnitureROI(rArr, gArr, bArr, luma255, resX, resY);
    const isWoodOrArchitectural = roi && roi.chromaMed >= 10 && fillRatio >= 0.78;
    const isSolidRectPanel = fillRatio >= 0.88 && fgBoxW > resX * 0.45 && fgBoxH > resY * 0.45;
    const isPlanarPanel = !!(isWoodOrArchitectural || isSolidRectPanel);

    // For carved architectural/wood panels framed near full bleed or tight ROI, align perimeter cleanly along rectangular edges
    if (isPlanarPanel && roi && roi.chromaMed >= 10 && roi.w > resX * 0.35 && roi.h > resY * 0.35 && removeBg && !hasTransparency) {
      for (let y = 0; y < resY; y++) {
        const row = y * resX;
        const inY = y >= roi.y0 && y <= roi.y1;
        for (let x = 0; x < resX; x++) {
          const inX = x >= roi.x0 && x <= roi.x1;
          fgMask[row + x] = (inX && inY) ? 1 : 0;
        }
      }
      minFgX = roi.x0; maxFgX = roi.x1; minFgY = roi.y0; maxFgY = roi.y1;
    }

    const rawDist = VectorPuffEngine.computeInteriorDistanceRect(fgMask, resX, resY);
    let maxDist = 1.0;
    for (let i = 0; i < total; i++) {
      if (rawDist[i] > maxDist) maxDist = rawDist[i];
    }

    // Smooth harmonic distance field eliminates diagonal "X" pyramid creases from rectangular Chamfer distance transforms
    const distSmoothR = Math.max(3, Math.round(Math.min(resX, resY) * 0.055));
    const smoothDistBox = ImageVectorizer.maskedBoxMean(rawDist, fgMask, resX, resY, distSmoothR);
    const smoothDist = new Float32Array(total);
    for (let i = 0; i < total; i++) {
      smoothDist[i] = fgMask[i] ? (smoothDistBox[i] * 0.65 + rawDist[i] * 0.35) : 0;
    }

    const minDim = Math.min(resX, resY);
    const depth = new Float32Array(total);
    const sculptSignal = new Float32Array(total);

    if (isPlanarPanel) {
      // Planar Wood / Architectural / 2-Tone Engraved Bas-Relief Pipeline
      const smL = ImageVectorizer.guidedFilter(luma255, resX, resY, 1, 55);
      const validMask = new Uint8Array(total);
      const wmRaw = new Uint8Array(total);
      const hasRoiChroma = roi && roi.chromaMed >= 10;
      const wmTol = hasRoiChroma && roi.chromaMed > 14 ? 12 : 18;

      for (let y = 0; y < resY; y++) {
        const row = y * resX;
        for (let x = 0; x < resX; x++) {
          const i = row + x;
          if (!fgMask[i]) continue;
          if (hasRoiChroma && roi.L[i] > roi.lMed + 5) {
            const dc = roi.chromaDiff[i];
            const rgbSat = Math.max(rArr[i], gArr[i], bArr[i]) - Math.min(rArr[i], gArr[i], bArr[i]);
            const isNeutralOverlay = roi.chromaMed > 14 && roi.L[i] > 76 && rgbSat < 32;
            const isFringeOverlay = roi.chromaMed > 14 && dc > 9 && rgbSat < 48;
            if (isNeutralOverlay || isFringeOverlay || dc > wmTol) {
              wmRaw[i] = 1;
              continue;
            }
          }
          validMask[i] = 1;
        }
      }

      // Dilate watermark mask by 2px so Lanczos text anti-aliasing fringes never spike
      let wmMask = new Uint8Array(wmRaw);
      for (let dPass = 0; dPass < 2; dPass++) {
        const nextWm = new Uint8Array(wmMask);
        for (let y = 1; y < resY - 1; y++) {
          const row = y * resX;
          for (let x = 1; x < resX - 1; x++) {
            const i = row + x;
            if (wmMask[i - 1] || wmMask[i + 1] || wmMask[i - resX] || wmMask[i + resX]) {
              nextWm[i] = 1;
              validMask[i] = 0;
            }
          }
        }
        wmMask = nextWm;
      }

      // Inpaint watermark regions BEFORE morphology/plane filtering so text bright pixels never leak outward
      const cleanL = new Float32Array(smL);
      let hasWm = false;
      for (let i = 0; i < total; i++) {
        if (wmMask[i]) { hasWm = true; break; }
      }
      if (hasWm) {
        const fillL = ImageVectorizer.maskedBoxMean(smL, validMask, resX, resY, Math.max(5, Math.round(minDim * 0.05)));
        for (let i = 0; i < total; i++) {
          if (wmMask[i]) cleanL[i] = fillL[i];
        }
      }

      // Measure robust tonal range [p10..p92] inside valid wood region so broad carvings stay solid all the way across!
      const vHist = new Uint32Array(256);
      let vCnt = 0;
      for (let i = 0; i < total; i++) {
        if (validMask[i]) {
          vHist[Math.max(0, Math.min(255, Math.round(cleanL[i])))]++;
          vCnt++;
        }
      }
      let p10 = 60, p92 = 200;
      if (vCnt > 50) {
        let acc = 0;
        for (let b = 0; b < 256; b++) {
          acc += vHist[b];
          if (acc >= vCnt * 0.10 && p10 === 60) p10 = b;
          if (acc >= vCnt * 0.92) { p92 = b; break; }
        }
      }
      const toneSpan = Math.max(36, p92 - p10);
      const rOtsu = vCnt > 50 ? ImageVectorizer.computeOtsuStats(vHist, vCnt) : { separability: 0, threshold: 128, m1: 100, m2: 120 };

      if (rOtsu.separability >= 0.72 && Math.abs(rOtsu.m2 - rOtsu.m1) >= 44) {
        // 2-Tone Laser/CNC Engraved Plaque or Solid Sign Logo (e.g. brand marks, typography, stamps):
        // Fit plane strictly across background pixels so board remains 100% flat without puffy letter halos!
        const bVals = [];
        for (let y = minFgY; y <= maxFgY; y++) {
          const nearY = (y - minFgY) < Math.max(3, Math.round(fgBoxH * 0.04)) || (maxFgY - y) < Math.max(3, Math.round(fgBoxH * 0.04));
          for (let x = minFgX; x <= maxFgX; x++) {
            const nearX = (x - minFgX) < Math.max(3, Math.round(fgBoxW * 0.04)) || (maxFgX - x) < Math.max(3, Math.round(fgBoxW * 0.04));
            if ((nearY || nearX) && validMask[y * resX + x]) bVals.push(cleanL[y * resX + x]);
          }
        }
        bVals.sort((u, v) => u - v);
        const borderMed = bVals.length > 0 ? bVals[Math.floor(bVals.length * 0.5)] : rOtsu.m2;
        const darkDesign = borderMed >= rOtsu.threshold;

        const bgSeed = new Uint8Array(total);
        let bgCnt = 0;
        for (let i = 0; i < total; i++) {
          if (validMask[i] && (darkDesign ? cleanL[i] >= rOtsu.threshold - 8 : cleanL[i] <= rOtsu.threshold + 8)) {
            bgSeed[i] = 1;
            bgCnt++;
          }
        }
        const planeR2 = Math.max(10, Math.round(minDim * 0.09));
        const bgPlane = bgCnt > 20 ? ImageVectorizer.maskedBoxMean(cleanL, bgSeed, resX, resY, planeR2) : new Float32Array(total).fill(borderMed);
        const cSpan = Math.max(38, Math.abs(rOtsu.m2 - rOtsu.m1) * 0.78);
        const embossRaised = invert || !darkDesign;

        const strokeMask = new Uint8Array(total);
        const bgMask = new Uint8Array(total);
        const tLinArr = new Float32Array(total);
        for (let i = 0; i < total; i++) {
          if (!fgMask[i]) continue;
          const dVal = darkDesign ? Math.max(0, bgPlane[i] - cleanL[i]) : Math.max(0, cleanL[i] - bgPlane[i]);
          const tLin = Math.max(0, Math.min(1, (dVal - cSpan * 0.14) / (cSpan * 0.72)));
          tLinArr[i] = tLin;
          if (tLin >= 0.48) strokeMask[i] = 1;
          else bgMask[i] = 1;
        }

        // Euclidean Signed Distance Field (SDF) beveling eliminates staircase jaggies on curves & diagonal typography!
        const distIn = VectorPuffEngine.computeInteriorDistanceRect(strokeMask, resX, resY);
        const distOut = VectorPuffEngine.computeInteriorDistanceRect(bgMask, resX, resY);
        const bevelPx = Math.max(1.8, Math.min(3.6, minDim * 0.015));

        for (let i = 0; i < total; i++) {
          if (!fgMask[i]) continue;
          const sdfGeom = strokeMask[i] ? Math.max(0, distIn[i] - 0.5) : -Math.max(0, distOut[i] - 0.5);
          const subpxOffset = tLinArr[i] - 0.5;
          const sdf = Math.abs(sdfGeom) <= 1.25 ? (sdfGeom * 0.55 + subpxOffset * 0.9) : sdfGeom;
          const uBev = Math.max(0, Math.min(1, 0.5 + sdf / bevelPx));
          const tBevel = uBev * uBev * (3.0 - 2.0 * uBev);
          const zVal = embossRaised
            ? (0.24 + tBevel * 0.72)
            : (0.74 - tBevel * 0.60);
          sculptSignal[i] = zVal;
          depth[i] = zVal;
        }
        const crispEdge = ImageVectorizer.guidedFilter(depth, resX, resY, 1, 0.004);
        for (let i = 0; i < total; i++) {
          depth[i] = fgMask[i] ? Math.max(0.10, Math.min(0.98, crispEdge[i])) : 0;
        }
      } else {
        const minR = Math.max(2, Math.min(5, Math.round(minDim * 0.015)));
        const medR = Math.max(5, Math.min(16, Math.round(minDim * 0.055)));
        const planeR = Math.max(18, Math.round(minDim * 0.24));

        const openedS = ImageVectorizer.separableExtrema(ImageVectorizer.separableExtrema(cleanL, resX, resY, minR, false), resX, resY, minR, true);
        const closedS = ImageVectorizer.separableExtrema(ImageVectorizer.separableExtrema(cleanL, resX, resY, minR, true), resX, resY, minR, false);
        const openedM = ImageVectorizer.separableExtrema(ImageVectorizer.separableExtrema(cleanL, resX, resY, medR, false), resX, resY, medR, true);
        const closedM = ImageVectorizer.separableExtrema(ImageVectorizer.separableExtrema(cleanL, resX, resY, medR, true), resX, resY, medR, false);
        const broadPlane = ImageVectorizer.maskedBoxMean(cleanL, fgMask, resX, resY, planeR);

        const pol = invert ? -1 : 1;
        for (let i = 0; i < total; i++) {
          if (!fgMask[i]) continue;
          // 56% illumination-balanced solid macro elevation + 44% crisp multi-scale detail
          const deLightedL = cleanL[i] - (broadPlane[i] - (p10 + toneSpan * 0.45)) * 0.32;
          let macroElev = (deLightedL - p10) / toneSpan;
          if (invert) macroElev = 1.0 - macroElev;
          macroElev = Math.max(0.05, Math.min(1.0, macroElev));

          const wh = (Math.max(0, cleanL[i] - openedS[i]) * 0.35 + Math.max(0, cleanL[i] - openedM[i]) * 0.65) / 72.0;
          const bh = (Math.max(0, closedS[i] - cleanL[i]) * 0.35 + Math.max(0, closedM[i] - cleanL[i]) * 0.65) / 72.0;
          const microDelta = (wh * 1.15 - bh * 0.90) * pol;
          const zVal = Math.max(0.10, Math.min(1.0, macroElev * 0.66 + 0.16 + microDelta * 0.38));
          sculptSignal[i] = zVal;
          depth[i] = zVal;
        }

        // Smooth fine wood grain ripples while keeping carved petals, fluted columns & recessed panel steps razor crisp
        const smoothPlanar = ImageVectorizer.guidedFilter(depth, resX, resY, 2, 0.012);
        for (let i = 0; i < total; i++) {
          depth[i] = fgMask[i] ? Math.max(0.10, Math.min(1.0, smoothPlanar[i])) : 0;
        }
      }
    } else {
      // Organic 3D Cutout Depth Extraction (Smooth Harmonic Volume + Denoised Multi-Scale Surface Detail)
      const lumaSmooth = ImageVectorizer.guidedFilter(luma, resX, resY, 2, 0.012);
      const rMed = Math.max(3, Math.min(9, Math.round(minDim * 0.032)));
      const openedMed = ImageVectorizer.separableExtrema(
        ImageVectorizer.separableExtrema(lumaSmooth, resX, resY, rMed, false),
        resX, resY, rMed, true
      );
      const closedMed = ImageVectorizer.separableExtrema(
        ImageVectorizer.separableExtrema(lumaSmooth, resX, resY, rMed, true),
        resX, resY, rMed, false
      );

      const sat = ImageVectorizer.buildIntegralImage(lumaSmooth, resX, resY);
      const illumR = Math.max(8, Math.round(minDim * 0.14));
      let sMin = 1e9;
      let sMax = -1e9;
      for (let y = 0; y < resY; y++) {
        for (let x = 0; x < resX; x++) {
          const i = y * resX + x;
          if (!fgMask[i]) continue;
          const localMean = ImageVectorizer.getLocalMean(sat, resX, resY, x, y, illumR);
          const wh = Math.max(0, lumaSmooth[i] - openedMed[i]);
          const bh = Math.max(0, closedMed[i] - lumaSmooth[i]);
          let tone = delight
            ? lumaSmooth[i] * 0.55 + (lumaSmooth[i] - localMean + 0.5) * 0.45
            : lumaSmooth[i];
          if (invert) tone = 1.0 - tone;
          const sig = tone * 0.54 + (wh * 1.25 - bh * 0.95) * (invert ? -1 : 1) * 0.46;
          sculptSignal[i] = sig;
          if (sig < sMin) sMin = sig;
          if (sig > sMax) sMax = sig;
        }
      }

      const sSpan = Math.max(0.08, sMax - sMin);
      const combinedDepth = new Float32Array(total);
      for (let y = 1; y < resY - 1; y++) {
        for (let x = 1; x < resX - 1; x++) {
          const i = y * resX + x;
          if (!fgMask[i]) continue;
          const t = Math.min(1.0, smoothDist[i] / Math.max(3.0, maxDist * 0.86));
          const macroDome = Math.sqrt(Math.max(0.0, 1.0 - (1.0 - t) * (1.0 - t)));
          const normSculpt = Math.max(0.0, Math.min(1.0, (sculptSignal[i] - sMin) / sSpan));
          const gx = lumaSmooth[i + 1] - lumaSmooth[i - 1];
          const gy = lumaSmooth[i + resX] - lumaSmooth[i - resX];
          const edgeCrest = Math.min(0.14, Math.hypot(gx, gy) * edgeStrength * 0.65);
          combinedDepth[i] = Math.max(
            0.08,
            Math.min(1.0, macroDome * 0.48 + normSculpt * 0.46 + edgeCrest * 0.14)
          );
        }
      }

      // Multi-pass Jacobi Poisson harmonic volume integration + edge-preserving surface fairing inside foreground
      const jacobiBuf = new Float32Array(combinedDepth);
      const jacobiNext = new Float32Array(combinedDepth);
      for (let jPass = 0; jPass < 10; jPass++) {
        for (let y = 1; y < resY - 1; y++) {
          const row = y * resX;
          for (let x = 1; x < resX - 1; x++) {
            const i = row + x;
            if (!fgMask[i]) continue;
            let s = 0, w = 0;
            const nbs = [i - 1, i + 1, i - resX, i + resX];
            for (const ni of nbs) {
              if (!fgMask[ni]) continue;
              const diff = Math.abs(jacobiBuf[ni] - jacobiBuf[i]);
              const wt = diff < 0.16 ? 1.0 : 0.28;
              s += jacobiBuf[ni] * wt;
              w += wt;
            }
            const nbrAvg = w > 0 ? s / w : combinedDepth[i];
            jacobiNext[i] = Math.max(0.08, Math.min(1.0, combinedDepth[i] * 0.34 + nbrAvg * 0.66));
          }
        }
        jacobiBuf.set(jacobiNext);
      }
      for (let i = 0; i < total; i++) {
        depth[i] = fgMask[i] ? Math.max(0.08, Math.min(1.0, jacobiBuf[i])) : 0;
      }
    }

    const aspectBox = maxFgX > minFgX && maxFgY > minFgY
      ? {
          minU: minFgX / Math.max(1, resX - 1),
          maxU: maxFgX / Math.max(1, resX - 1),
          minV: 1.0 - maxFgY / Math.max(1, resY - 1),
          maxV: 1.0 - minFgY / Math.max(1, resY - 1),
        }
      : { minU: 0, maxU: 1, minV: 0, maxV: 1 };

    return {
      fgMask,
      fgBox: { minX: minFgX, maxX: maxFgX, minY: minFgY, maxY: maxFgY },
      hasTransparency,
      isPlanarPanel,
      rawDist,
      smoothDist,
      maxDist,
      depth,
      sculptSignal,
      aspectBox,
    };
  }

  /**
   * Flood-fill outward from border in CIELAB so interior bright highlights never punch holes!
   */
  static extractBorderConnectedForeground(alpha, L, A, B, W, H, ctxInfo) {
    const total = W * H;
    const { removeBg, hasTransparency } = ctxInfo;
    const fg = new Uint8Array(total);

    if (hasTransparency) {
      for (let i = 0; i < total; i++) {
        if (alpha[i] >= 96) fg[i] = 1;
      }
      return Asset3DEngine.cleanSilhouetteMask(fg, W, H);
    }

    fg.fill(1);
    if (!removeBg) return fg;

    const samplesL = [];
    const samplesA = [];
    const samplesB = [];
    for (let x = 0; x < W; x += 2) {
      samplesL.push(L[x], L[(H - 1) * W + x]);
      samplesA.push(A[x], A[(H - 1) * W + x]);
      samplesB.push(B[x], B[(H - 1) * W + x]);
    }
    for (let y = 0; y < H; y += 2) {
      samplesL.push(L[y * W], L[y * W + W - 1]);
      samplesA.push(A[y * W], A[y * W + W - 1]);
      samplesB.push(B[y * W], B[y * W + W - 1]);
    }
    samplesL.sort((u, v) => u - v);
    samplesA.sort((u, v) => u - v);
    samplesB.sort((u, v) => u - v);
    const bgL = samplesL[Math.floor(samplesL.length * 0.5)] || 95;
    const bgA = samplesA[Math.floor(samplesA.length * 0.5)] || 0;
    const bgB = samplesB[Math.floor(samplesB.length * 0.5)] || 0;

    // Guard against flood-eating full-bleed wooden doors, stone slabs, or frames whose border IS the object itself!
    const borderChroma = Math.hypot(bgA, bgB);
    let uniformBorder = 0;
    for (let k = 0; k < samplesL.length; k++) {
      if (Math.hypot((samplesL[k] - bgL) * 0.85, samplesA[k] - bgA, samplesB[k] - bgB) <= 18) uniformBorder++;
    }
    const uniformRatio = uniformBorder / Math.max(1, samplesL.length);
    const inL = [], inA = [], inB = [];
    for (let ry = 0.30; ry <= 0.71; ry += 0.1) {
      const yy = Math.floor(H * ry);
      for (let rx = 0.30; rx <= 0.71; rx += 0.1) {
        const idx = yy * W + Math.floor(W * rx);
        inL.push(L[idx]); inA.push(A[idx]); inB.push(B[idx]);
      }
    }
    inL.sort((u, v) => u - v); inA.sort((u, v) => u - v); inB.sort((u, v) => u - v);
    const mid = Math.floor(inL.length * 0.5);
    const deInterior = Math.hypot((inL[mid] - bgL) * 0.85, inA[mid] - bgA, inB[mid] - bgB);
    const isNeutralBackdrop = borderChroma < 11 && uniformRatio >= 0.60 && (bgL > 72 || bgL < 22 || deInterior >= 12);
    const isUniformColorBackdrop = uniformRatio >= 0.85 && deInterior >= 22;
    if (!isNeutralBackdrop && !isUniformColorBackdrop) {
      return fg;
    }

    const visited = new Uint8Array(total);
    const queue = new Int32Array(total);
    let qH = 0;
    let qT = 0;

    const seedIfBg = (x, y) => {
      const idx = y * W + x;
      if (visited[idx]) return;
      const de = Math.hypot((L[idx] - bgL) * 0.85, A[idx] - bgA, B[idx] - bgB);
      if (de <= 22 || alpha[idx] < 64) {
        visited[idx] = 1;
        fg[idx] = 0;
        queue[qT++] = idx;
      }
    };

    for (let x = 0; x < W; x++) { seedIfBg(x, 0); seedIfBg(x, H - 1); }
    for (let y = 0; y < H; y++) { seedIfBg(0, y); seedIfBg(W - 1, y); }

    const dirs = [-1, 1, -W, W];
    while (qH < qT) {
      const u = queue[qH++];
      const ux = u % W;
      const uy = (u - ux) / W;
      for (let d = 0; d < 4; d++) {
        if (d === 0 && ux === 0) continue;
        if (d === 1 && ux === W - 1) continue;
        if (d === 2 && uy === 0) continue;
        if (d === 3 && uy === H - 1) continue;
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
    if (fgCount < total * 0.04) fg.fill(1);
    return Asset3DEngine.cleanSilhouetteMask(fg, W, H);
  }

  static cleanSilhouetteMask(fg, W, H) {
    const closed = ImageVectorizer.morphClose(fg, W, H, 1);
    const total = W * H;
    const minCompPx = Math.max(18, Math.round(total * 0.002));
    const vis = new Uint8Array(total);
    const queue = new Int32Array(total);
    const dirs = [-1, 1, -W, W];

    for (let s = 0; s < total; s++) {
      if (!closed[s] || vis[s]) continue;
      let qH = 0, qT = 0;
      vis[s] = 1;
      queue[qT++] = s;
      while (qH < qT) {
        const u = queue[qH++];
        const ux = u % W;
        const uy = (u - ux) / W;
        for (let d = 0; d < 4; d++) {
          if (d === 0 && ux === 0) continue;
          if (d === 1 && ux === W - 1) continue;
          if (d === 2 && uy === 0) continue;
          if (d === 3 && uy === H - 1) continue;
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
   * Infinite Voronoi BFS padding from pure interior foreground texels across cutout seams.
   */
  static dilateTextureEdges(texRgba, tSize, fgMask, fgW, fgH, stripBorderBackdrop = false) {
    if (!fgMask || !fgW || !fgH) return;
    const fgN = fgW * fgH;
    const eroded1 = new Uint8Array(fgN);
    const eroded2 = new Uint8Array(fgN);
    let e1Count = 0;
    for (let y = 1; y < fgH - 1; y++) {
      const row = y * fgW;
      for (let x = 1; x < fgW - 1; x++) {
        const i = row + x;
        if (fgMask[i] && fgMask[i - 1] && fgMask[i + 1] && fgMask[i - fgW] && fgMask[i + fgW]) {
          eroded1[i] = 1;
          e1Count++;
        }
      }
    }
    let e2Count = 0;
    for (let y = 1; y < fgH - 1; y++) {
      const row = y * fgW;
      for (let x = 1; x < fgW - 1; x++) {
        const i = row + x;
        if (eroded1[i] && eroded1[i - 1] && eroded1[i + 1] && eroded1[i - fgW] && eroded1[i + fgW]) {
          eroded2[i] = 1;
          e2Count++;
        }
      }
    }
    const seedMask = e1Count > 16 ? eroded1 : fgMask;
    const safeInnerMask = e2Count > 16 ? eroded2 : null;

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
    const scaleX = (fgW - 1) / Math.max(1, tSize - 1);
    const scaleY = (fgH - 1) / Math.max(1, tSize - 1);

    for (let y = 0; y < tSize; y++) {
      const sy = Math.min(fgH - 1, Math.round(y * scaleY));
      const fRow = sy * fgW;
      const tRow = y * tSize;
      for (let x = 0; x < tSize; x++) {
        const sx = Math.min(fgW - 1, Math.round(x * scaleX));
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
  static synthesizePBRMaps(rgba, size, delight = true, fgMask = null, fgW = 0, fgH = 0) {
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
    const fgScaleX = fgMask && fgW ? (fgW - 1) / Math.max(1, size - 1) : 0;
    const fgScaleY = fgMask && (fgH || fgW) ? ((fgH || fgW) - 1) / Math.max(1, size - 1) : 0;
    const activeFgW = fgW || 0;
    const activeFgH = fgH || fgW || 0;

    for (let y = 0; y < size; y++) {
      const sy = fgScaleY ? Math.min(activeFgH - 1, Math.round(y * fgScaleY)) : 0;
      for (let x = 0; x < size; x++) {
        const i = y * size + x;
        const p = i * 4;
        const r = rgba[p];
        const g = rgba[p + 1];
        const b = rgba[p + 2];
        const a = rgba[p + 3];
        const inFg = !fgMask || !activeFgW || fgMask[sy * activeFgW + Math.min(activeFgW - 1, Math.round(x * fgScaleX))] === 1;

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
        const gradGain = inFg ? 3.0 : 0.35;
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

    const aspect = analysis.aspect || 1.0;
    const fitW = aspect >= 1 ? widthMm : widthMm * aspect;
    const fitH = aspect >= 1 ? heightMm / aspect : heightMm;

    if (assetMode === 'lathe') {
      return Asset3DEngine.buildLatheMeshData(analysis, fitW, fitH, widthMm, heightMm, thicknessMm);
    }
    return Asset3DEngine.buildVolumetricShellMeshData(analysis, assetMode, fitW, fitH, widthMm, heightMm, thicknessMm);
  }

  /**
   * Watertight volumetric/planar shell centered inside (widthMm, heightMm) with exact source aspect (fitW x fitH).
   * Preserves flat coplanar architectural rails/stiles on planar panels, applies Taubin fairing on curved organic silhouettes,
   * and uses split rim bevel vertices so side walls never darken front/back normals.
   */
  static buildVolumetricShellMeshData(analysis, assetMode, fitW, fitH, stageW, stageH, thicknessMm) {
    const W = analysis.resX || analysis.resolution;
    const H = analysis.resY || analysis.resolution;
    const fg = analysis.fgMask;
    const depth = analysis.depth;
    const dist = analysis.dist;
    const maxDist = Math.max(1, analysis.maxDist);
    const isPlanar = !!analysis.isPlanarPanel;
    const CW = W - 1;
    const CH = H - 1;

    const activeCell = new Uint8Array(CW * CH);
    let cellCount = 0;
    for (let iy = 0; iy < CH; iy++) {
      for (let ix = 0; ix < CW; ix++) {
        const i00 = iy * W + ix;
        if (fg[i00] || fg[i00 + 1] || fg[i00 + W] || fg[i00 + W + 1]) {
          activeCell[iy * CW + ix] = 1;
          cellCount++;
        }
      }
    }

    if (cellCount === 0) {
      for (let iy = 1; iy < CH - 1; iy++) {
        for (let ix = 1; ix < CW - 1; ix++) {
          activeCell[iy * CW + ix] = 1;
          fg[iy * W + ix] = 1;
        }
      }
    }

    const hasCell = (ix, iy) => (ix >= 0 && iy >= 0 && ix < CW && iy < CH && activeCell[iy * CW + ix] === 1);
    const activeVert = new Uint8Array(W * H);
    const isBoundaryVert = new Uint8Array(W * H);
    for (let iy = 0; iy < CH; iy++) {
      for (let ix = 0; ix < CW; ix++) {
        if (!hasCell(ix, iy)) continue;
        activeVert[iy * W + ix] = 1;
        activeVert[iy * W + ix + 1] = 1;
        activeVert[(iy + 1) * W + ix] = 1;
        activeVert[(iy + 1) * W + ix + 1] = 1;
      }
    }

    const bndNbrs = new Map();
    const linkBoundary = (a, b) => {
      if (!bndNbrs.has(a)) bndNbrs.set(a, []);
      bndNbrs.get(a).push(b);
      isBoundaryVert[a] = 1;
    };

    const boundaryEdges = [];
    for (let iy = 0; iy < CH; iy++) {
      for (let ix = 0; ix < CW; ix++) {
        if (!hasCell(ix, iy)) continue;
        const i00 = iy * W + ix;
        const i10 = i00 + 1;
        const i01 = i00 + W;
        const i11 = i01 + 1;
        if (!hasCell(ix, iy - 1)) { boundaryEdges.push([i10, i00]); linkBoundary(i10, i00); linkBoundary(i00, i10); }
        if (!hasCell(ix, iy + 1)) { boundaryEdges.push([i01, i11]); linkBoundary(i01, i11); linkBoundary(i11, i01); }
        if (!hasCell(ix - 1, iy)) { boundaryEdges.push([i00, i01]); linkBoundary(i00, i01); linkBoundary(i01, i00); }
        if (!hasCell(ix + 1, iy)) { boundaryEdges.push([i11, i10]); linkBoundary(i11, i10); linkBoundary(i10, i11); }
      }
    }

    const gridX = new Float32Array(W * H);
    const gridY = new Float32Array(W * H);
    for (let iy = 0; iy < H; iy++) {
      const row = iy * W;
      for (let ix = 0; ix < W; ix++) {
        gridX[row + ix] = ix;
        gridY[row + ix] = iy;
      }
    }

    // Apply Taubin curve smoothing ONLY on organic silhouettes so rectangular architectural corners remain razor crisp!
    if (!isPlanar) {
      const maxStep = 0.42;
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
          const ix0 = vi % W;
          const iy0 = (vi - ix0) / W;
          nextX[vi] = Math.max(ix0 - maxStep, Math.min(ix0 + maxStep, gridX[vi] + k * (avgX - gridX[vi])));
          nextY[vi] = Math.max(iy0 - maxStep, Math.min(iy0 + maxStep, gridY[vi] + k * (avgY - gridY[vi])));
        }
        gridX.set(nextX);
        gridY.set(nextY);
      }
    }

    // Level-set isocontour vertex alignment: slides interior steep wall vertices along height contours so
    // diagonal letters, circles, and curved carvings keep clean diagonal edges without pixel sawtooths!
    const isoMaxShift = 0.36;
    for (let pass = 0; pass < 2; pass++) {
      const nextIsoX = new Float32Array(gridX);
      const nextIsoY = new Float32Array(gridY);
      for (let iy = 1; iy < H - 1; iy++) {
        const row = iy * W;
        for (let ix = 1; ix < W - 1; ix++) {
          const vi = row + ix;
          if (!activeVert[vi] || isBoundaryVert[vi]) continue;
          const zc = depth[vi];
          let minZ = zc, maxZ = zc;
          const nbs = [vi - 1, vi + 1, vi - W, vi + W, vi - W - 1, vi - W + 1, vi + W - 1, vi + W + 1];
          for (const ni of nbs) {
            if (!activeVert[ni]) continue;
            const zn = depth[ni];
            if (zn < minZ) minZ = zn;
            if (zn > maxZ) maxZ = zn;
          }
          if (maxZ - minZ < 0.12) continue;
          let sx = 0, sy = 0, sw = 0;
          for (const ni of nbs) {
            if (!activeVert[ni] || isBoundaryVert[ni]) continue;
            const dz = (depth[ni] - zc) * 9.0;
            const wt = Math.exp(-dz * dz);
            sx += gridX[ni] * wt;
            sy += gridY[ni] * wt;
            sw += wt;
          }
          if (sw > 0.25) {
            const tx = gridX[vi] + 0.55 * (sx / sw - gridX[vi]);
            const ty = gridY[vi] + 0.55 * (sy / sw - gridY[vi]);
            nextIsoX[vi] = Math.max(ix - isoMaxShift, Math.min(ix + isoMaxShift, tx));
            nextIsoY[vi] = Math.max(iy - isoMaxShift, Math.min(iy + isoMaxShift, ty));
          }
        }
      }
      gridX.set(nextIsoX);
      gridY.set(nextIsoY);
    }

    const frontMap = new Int32Array(W * H).fill(-1);
    const backMap = new Int32Array(W * H).fill(-1);
    const sideFrontMap = new Int32Array(W * H).fill(-1);
    const sideBackMap = new Int32Array(W * H).fill(-1);
    const positions = [];
    const uvs = [];
    const frontIndices = [];
    const backIndices = [];
    const sideIndices = [];

    const isRelief = assetMode === 'relief';
    const halfT = thicknessMm * 0.5;
    const backScale = assetMode === 'inflated' ? 0.90 : (assetMode === 'double-sided' ? (isPlanar ? 0.72 : 0.58) : 0.0);
    const minRimZ = isRelief ? 0.0 : (isPlanar ? Math.max(1.2, halfT * 0.34) : halfT * 0.14);
    const offX = (stageW - fitW) * 0.5;
    const offY = (stageH - fitH) * 0.5;

    const pushVert = (gx, gy, zVal) => {
      const u = Math.max(0, Math.min(1, gx / Math.max(1, W - 1)));
      const v = Math.max(0, Math.min(1, 1.0 - gy / Math.max(1, H - 1)));
      positions.push(offX + u * fitW, offY + v * fitH, zVal);
      uvs.push(u, v);
      return (positions.length / 3) - 1;
    };

    for (let iy = 0; iy < H; iy++) {
      for (let ix = 0; ix < W; ix++) {
        const vi = iy * W + ix;
        if (!activeVert[vi]) continue;
        const gx = gridX[vi];
        const gy = gridY[vi];
        const dNorm = fg[vi] ? depth[vi] : 0.10;
        const t = Math.min(1.0, dist[vi] / Math.max(2.0, maxDist * 0.86));
        const macroSmooth = fg[vi] ? Math.sqrt(Math.max(0.0, 1.0 - (1.0 - t) * (1.0 - t))) : 0.0;
        const isRim = isBoundaryVert[vi] === 1;

        let zF;
        let zB;
        if (isRelief) {
          // Flat-backed CNC / 3D print relief slab (dead-flat bottom plane at Z=0!)
          zF = isRim ? Math.max(0.6, thicknessMm * 0.18) : (thicknessMm * 0.18 + dNorm * thicknessMm * 0.82);
          zB = 0.0;
        } else if (isPlanar && assetMode !== 'inflated') {
          // Coplanar architectural/door relief on Front & Back (zero dome pillow warp, sturdy beveled rim)
          const rimChamfer = Math.min(dNorm * 0.88, 0.44);
          zF = isRim ? Math.max(minRimZ, halfT * rimChamfer) : Math.max(minRimZ, halfT * dNorm);
          zB = isRim ? -Math.max(minRimZ * 0.85, halfT * rimChamfer * backScale) : -Math.max(minRimZ * 0.85, halfT * backScale * dNorm);
        } else if (assetMode === 'inflated') {
          zF = isRim ? minRimZ : Math.max(minRimZ, halfT * (macroSmooth * 0.56 + dNorm * 0.44));
          zB = isRim ? -minRimZ : -Math.max(minRimZ, halfT * backScale * (macroSmooth * 0.72 + dNorm * 0.28));
        } else {
          zF = isRim ? minRimZ : Math.max(minRimZ, halfT * dNorm);
          zB = isRim ? -minRimZ : -Math.max(minRimZ, halfT * backScale * (macroSmooth * 0.65 + dNorm * 0.35));
        }

        frontMap[vi] = pushVert(gx, gy, zF);
        backMap[vi] = pushVert(gx, gy, zB);
        if (isRim) {
          sideFrontMap[vi] = pushVert(gx, gy, zF);
          sideBackMap[vi] = pushVert(gx, gy, zB);
        }
      }
    }

    for (let iy = 0; iy < CH; iy++) {
      for (let ix = 0; ix < CW; ix++) {
        if (!hasCell(ix, iy)) continue;
        const i00 = iy * W + ix;
        const i10 = i00 + 1;
        const i01 = i00 + W;
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
   * Seam-free 360-degree revolved Lathe object on (resX x resY) with outward normals & smoothed profile curve.
   */
  static buildLatheMeshData(analysis, fitW, fitH, stageW, stageH, thicknessMm) {
    const W = analysis.resX || analysis.resolution;
    const H = analysis.resY || analysis.resolution;
    const fg = analysis.fgMask;
    const depth = analysis.depth;

    let yMin = H - 1, yMax = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if (fg[y * W + x]) {
          if (y < yMin) yMin = y;
          if (y > yMax) yMax = y;
        }
      }
    }
    if (yMax <= yMin + 2) {
      yMin = Math.floor(H * 0.1);
      yMax = Math.ceil(H * 0.9);
    }

    const rowData = [];
    for (let y = yMin; y <= yMax; y++) {
      let lx = W, rx = -1;
      for (let x = 0; x < W; x++) {
        if (fg[y * W + x]) {
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
    const cxWorld = stageW * 0.5;
    const offY = (stageH - fitH) * 0.5;
    const positions = [];
    const uvs = [];
    const bodyIndices = [];
    const capIndices = [];

    const sampleBilinearDepth = (xf, yf) => {
      const x0 = Math.max(0, Math.min(W - 2, Math.floor(xf)));
      const y0 = Math.max(0, Math.min(H - 2, Math.floor(yf)));
      const fx = xf - x0, fy = yf - y0;
      const d00 = depth[y0 * W + x0], d10 = depth[y0 * W + x0 + 1];
      const d01 = depth[(y0 + 1) * W + x0], d11 = depth[(y0 + 1) * W + x0 + 1];
      return (d00 * (1 - fx) + d10 * fx) * (1 - fy) + (d01 * (1 - fx) + d11 * fx) * fy;
    };

    for (let ri = 0; ri < rows; ri++) {
      const rInfo = rowData[ri];
      const worldY = offY + (1.0 - rInfo.y / Math.max(1, H - 1)) * fitH;
      const baseR = (smoothRad[ri] / Math.max(1, W - 1)) * fitW;
      const vCoord = 1.0 - rInfo.y / Math.max(1, H - 1);

      for (let si = 0; si <= segTheta; si++) {
        const uFrac = si / segTheta;
        const theta = uFrac * Math.PI * 2 - Math.PI * 0.5;
        const dirX = Math.cos(theta);
        const dirZ = Math.sin(theta);
        const sampleXf = Math.max(0, Math.min(W - 1, smoothCx[ri] + dirX * smoothRad[ri]));
        const dMod = (sampleBilinearDepth(sampleXf, rInfo.y) - 0.5) * (thicknessMm * 0.06);
        const rFinal = Math.max(0.4, baseR + dMod);

        positions.push(cxWorld + dirX * rFinal, worldY, dirZ * rFinal);
        const texU = Math.max(0, Math.min(1, sampleXf / Math.max(1, W - 1)));
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
        bodyIndices.push(a, b, c, b, d, c);
      }
    }

    const topRingBase = positions.length / 3;
    const topY = offY + (1.0 - rowData[0].y / Math.max(1, H - 1)) * fitH;
    const topInnerRow = rowData[Math.min(rows - 1, 2)];
    const topCapV = 1.0 - topInnerRow.y / Math.max(1, H - 1);
    for (let si = 0; si <= segTheta; si++) {
      const orig = si * 3;
      positions.push(positions[orig], positions[orig + 1], positions[orig + 2]);
      uvs.push(uvs[si * 2], topCapV);
    }
    const topCenterIdx = positions.length / 3;
    positions.push(cxWorld, topY, 0);
    uvs.push(smoothCx[Math.min(rows - 1, 2)] / Math.max(1, W - 1), topCapV);
    for (let si = 0; si < segTheta; si++) {
      capIndices.push(topCenterIdx, topRingBase + si + 1, topRingBase + si);
    }

    const lastRowOrig = (rows - 1) * stride;
    const botRingBase = positions.length / 3;
    const botY = offY + (1.0 - rowData[rows - 1].y / Math.max(1, H - 1)) * fitH;
    const botInnerRow = rowData[Math.max(0, rows - 3)];
    const botCapV = 1.0 - botInnerRow.y / Math.max(1, H - 1);
    for (let si = 0; si <= segTheta; si++) {
      const src = (lastRowOrig + si) * 3;
      positions.push(positions[src], positions[src + 1], positions[src + 2]);
      uvs.push(uvs[(lastRowOrig + si) * 2], botCapV);
    }
    const botCenterIdx = positions.length / 3;
    positions.push(cxWorld, botY, 0);
    uvs.push(smoothCx[Math.max(0, rows - 3)] / Math.max(1, W - 1), botCapV);
    for (let si = 0; si < segTheta; si++) {
      capIndices.push(botCenterIdx, botRingBase + si, botRingBase + si + 1);
    }

    const result = Asset3DEngine.finalizeSplitNormalsMesh(positions, uvs, [bodyIndices, capIndices], 'Triangle');
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
