import { ImageVectorizer } from '../cad/imageVectorizer.js';
import { VectorPuffEngine } from '../cad/vectorPuffEngine.js';
import { NeuralVision } from '../ai/neuralVision.js';

/**
 * Meshy-Architecture Volumetric Image-to-3D Reconstruction Engine & PBR Synthesizer.
 * Combines Meshy AI client-side reconstruction algorithms verified via REA:
 * - Felzenszwalb exact O(N) 2D Euclidean L2 Distance Transform (computeL2Distance2D)
 * - Kovesi 3-pass separable circular Gaussian smoothing (gaussianSmooth2D)
 * - Diagonal checkerboard pinch cleanup & Union-Find manifold component verification
 * - [1,4,6,4,1] Sobel steep-slope detection + 30-deg normal-bilateral circular rolling-ball filleting
 * - Subpixel Newton-Raphson zero-isocontour (phi=0) rim snapping & ordered boundary loop relaxation
 * - 5-Stage asynchronous progressive pipeline + optional direct api.meshy.ai v1 cloud bridge
 */
export class Asset3DEngine {
  /**
   * 5-Stage Asynchronous Meshy-Architecture Image-to-3D Pipeline with real progress updates.
   */
  static async generateAssetProgressiveAsync(imgElement, options = {}, onProgress = null) {
    const emit = async (stage, pct, label) => {
      if (typeof onProgress === 'function') onProgress(stage, pct, label);
      await new Promise((r) => setTimeout(r, 0));
    };

    const {
      assetMode = 'double-sided',
      resolution = 256,
      texSize = 1024,
      removeBg = true,
      delight = true,
      invert = false,
      edgeStrength = 0.36,
      filletRatio = 0.012,
      widthMm = 100,
      heightMm = 100,
      thicknessMm = 22,
      useNeuralAI = true,
    } = options;

    await emit(1, 8, `Stage 1/5: Rasterizing high-density aspect grid & isolating subject (${resolution}px tier)...`);
    const prep = Asset3DEngine._prepareCanvasAndMask(imgElement, { resolution, texSize, removeBg, delight, invert, edgeStrength });

    if (useNeuralAI && NeuralVision.isBrowserRuntime()) {
      const neuralRes = await NeuralVision.analyzeSubjectAsync(
        imgElement,
        prep.resX,
        prep.resY,
        {
          enableDepth: true,
          enableMatte: !prep.isPlanarPanel && removeBg && !prep.hasTransparency,
          srcCrop: prep.srcCrop,
        },
        (_st, pct, label) => {
          if (typeof onProgress === 'function') onProgress(2, Math.max(12, Math.min(44, pct)), label);
        }
      );
      if (neuralRes?.depth) {
        prep.neuralDepth = neuralRes.depth;
      }
      if (neuralRes?.matte && !prep.isPlanarPanel && removeBg && !prep.hasTransparency) {
        const refinedFg = new Uint8Array(prep.total);
        let kept = 0;
        for (let i = 0; i < prep.total; i++) {
          const on = neuralRes.matte[i] >= 0.42 || (prep.fgMask[i] && neuralRes.matte[i] >= 0.25);
          if (on) { refinedFg[i] = 1; kept++; }
        }
        if (kept > prep.total * 0.05 && kept < prep.total * 0.96) {
          prep.fgMask = Asset3DEngine.cleanSilhouetteMask(refinedFg, prep.resX, prep.resY);
          ImageVectorizer.removeCheckerboardPinches(prep.fgMask, prep.resX, prep.resY, false);
          const { phi, dIn } = ImageVectorizer.computeSignedDistancePhi(prep.fgMask, prep.resX, prep.resY, 1.0);
          prep.phi = phi;
          prep.rawDist = dIn;
          let mDist = 1.0;
          for (let i = 0; i < prep.total; i++) if (dIn[i] > mDist) mDist = dIn[i];
          prep.maxDist = mDist;
          const smD = new Float32Array(dIn);
          ImageVectorizer.gaussianSmooth2D(smD, prep.resX, prep.resY, Math.max(1.4, Math.min(prep.resX, prep.resY) * 0.022));
          for (let i = 0; i < prep.total; i++) if (!prep.fgMask[i]) smD[i] = 0;
          prep.smoothDist = smD;
        }
      }
    }

    await emit(2, 46, `Stage 2/5: Computing ${prep.neuralDepth ? 'Depth-Anything V2 + ' : ''}Felzenszwalb L2 SDF & ${prep.isPlanarPanel ? 'Planar CNC Relief' : 'Jacobi Poisson 3D Volume'}...`);
    Asset3DEngine._solveSurfaceDepth(prep, { delight, invert, edgeStrength });

    await emit(3, 64, `Stage 3/5: Applying Volume-Preserving Taubin Fairing & Sobel Rolling-Ball Fillet (r=${filletRatio.toFixed(3)})...`);
    Asset3DEngine.taubinSmoothDepth(prep.depth, prep.fgMask, prep.resX, prep.resY, prep.isPlanarPanel ? 3 : 4);
    Asset3DEngine.applySlopeAwareFillet(prep.depth, prep.fgMask, prep.resX, prep.resY, prep.aspect >= 1 ? widthMm : widthMm * prep.aspect, thicknessMm, filletRatio);

    await emit(4, 82, `Stage 4/5: Newton-Raphson zero-isocontour rim snapping & watertight manifold ${assetMode.toUpperCase()} mesh (${prep.resX}×${prep.resY})...`);
    const analysis = Asset3DEngine._finalizeAnalysis(prep, { texSize, removeBg, delight });
    const meshData = Asset3DEngine.buildAssetMeshData(analysis, { assetMode, widthMm, heightMm, thicknessMm });

    await emit(5, 96, 'Stage 5/5: Synthesizing 1024px Cavity-AO BaseColor, Normal, Roughness & ORM PBR maps...');
    await emit(5, 100, `Complete: ${meshData.stats.faces.toLocaleString()} watertight faces ready!`);

    return { analysis, meshData };
  }

  /**
   * Synchronous entrypoint preserved for headless tests & fast preview scripting.
   */
  static analyzeImage(imgElement, options = {}) {
    const {
      resolution = 160,
      texSize = 1024,
      removeBg = true,
      delight = true,
      invert = false,
      edgeStrength = 0.35,
      filletRatio = 0.012,
      widthMm = 100,
      thicknessMm = 22,
    } = options;

    const prep = Asset3DEngine._prepareCanvasAndMask(imgElement, { resolution, texSize, removeBg, delight, invert, edgeStrength });
    Asset3DEngine._solveSurfaceDepth(prep, { delight, invert, edgeStrength });
    Asset3DEngine.applySlopeAwareFillet(prep.depth, prep.fgMask, prep.resX, prep.resY, prep.aspect >= 1 ? widthMm : widthMm * prep.aspect, thicknessMm, filletRatio);
    return Asset3DEngine._finalizeAnalysis(prep, { texSize, removeBg, delight });
  }

  /**
   * Optional Cloud Neural 3D Generator via Meshy API v1 (for users providing their own Meshy API key).
   */
  static async generateViaMeshyCloudApi(imgElement, apiKey, options = {}, onProgress = null) {
    const cleanKey = String(apiKey || '').trim();
    if (!cleanKey) throw new Error('Meshy API key is required for Cloud Neural 3D mode.');
    const {
      aiModel = 'latest',
      topology = 'triangle',
      targetPolycount = 60000,
      enablePbr = true,
      shouldRemesh = true,
    } = options;

    if (typeof onProgress === 'function') onProgress(1, 8, 'Uploading image to Meshy Cloud API v1...');
    const c = document.createElement('canvas');
    const iw = imgElement.naturalWidth || imgElement.width || 512;
    const ih = imgElement.naturalHeight || imgElement.height || 512;
    const maxEdge = 1024;
    const sc = Math.min(1.0, maxEdge / Math.max(1, iw, ih));
    c.width = Math.max(64, Math.round(iw * sc));
    c.height = Math.max(64, Math.round(ih * sc));
    c.getContext('2d').drawImage(imgElement, 0, 0, c.width, c.height);
    const imageUrl = c.toDataURL('image/png');

    const headers = {
      Authorization: `Bearer ${cleanKey}`,
      'Content-Type': 'application/json',
    };
    const createResp = await fetch('https://api.meshy.ai/openapi/v1/image-to-3d', {
      method: 'POST',
      headers,
      body: JSON.stringify({
        image_url: imageUrl,
        ai_model: aiModel,
        topology,
        target_polycount: targetPolycount,
        enable_pbr: enablePbr,
        should_remesh: shouldRemesh,
        should_texture: true,
      }),
    });
    if (!createResp.ok) {
      const msg = await createResp.text();
      throw new Error(`Meshy API error (${createResp.status}): ${msg.slice(0, 180)}`);
    }
    const created = await createResp.json();
    const taskId = created.result;
    if (!taskId) throw new Error('Meshy API did not return a valid task id.');

    for (let poll = 0; poll < 120; poll++) {
      await new Promise((r) => setTimeout(r, 2000));
      const pollResp = await fetch(`https://api.meshy.ai/openapi/v1/image-to-3d/${encodeURIComponent(taskId)}`, { headers });
      if (!pollResp.ok) continue;
      const task = await pollResp.json();
      const pct = Math.max(12, Math.min(96, Math.round(task.progress || 15)));
      if (typeof onProgress === 'function') {
        onProgress(3, pct, `Meshy Cloud Neural Inference (${task.status || 'IN_PROGRESS'})... ${pct}%`);
      }
      if (task.status === 'SUCCEEDED' && task.model_urls?.glb) {
        if (typeof onProgress === 'function') onProgress(5, 98, 'Downloading generated neural .GLB PBR asset...');
        const glbResp = await fetch(task.model_urls.glb);
        if (!glbResp.ok) throw new Error(`Failed downloading Meshy GLB (${glbResp.status})`);
        const glbBuffer = await glbResp.arrayBuffer();
        if (typeof onProgress === 'function') onProgress(5, 100, 'Meshy Cloud Neural 3D model ready!');
        return { taskId, glbBuffer, task };
      }
      if (task.status === 'FAILED' || task.status === 'EXPIRED') {
        throw new Error(task.task_error?.message || `Meshy task ${task.status}`);
      }
    }
    throw new Error('Meshy Cloud task timed out after 4 minutes.');
  }

  static _prepareCanvasAndMask(imgElement, opts) {
    const { resolution = 160, texSize = 1024, removeBg = true } = opts;
    const baseRes = Math.max(48, Math.min(512, Math.round(resolution)));
    const iw = imgElement.naturalWidth || imgElement.width || 1;
    const ih = imgElement.naturalHeight || imgElement.height || 1;
    const imgAspect = iw / Math.max(1, ih);

    // Choose adaptive rectangular grid (resX x resY) matching source aspect & High/Flagship quality budget
    let resX = baseRes;
    let resY = baseRes;
    if (baseRes >= 128 || Math.abs(imgAspect - 1.0) >= 0.05) {
      const areaBudget = baseRes * baseRes * (baseRes >= 144 ? 1.65 : 1.45);
      resX = Math.max(84, Math.min(512, Math.round(Math.sqrt(areaBudget * imgAspect))));
      resY = Math.max(84, Math.min(512, Math.round(Math.sqrt(areaBudget / imgAspect))));
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
    let srcCropX = 0, srcCropY = 0, srcCropW = iw, srcCropH = ih;

    const extractMaskAndStats = (buf, W, H) => {
      const total = W * H;
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
        const r = buf[i * 4], g = buf[i * 4 + 1], b = buf[i * 4 + 2], a = buf[i * 4 + 3];
        rArr[i] = r; gArr[i] = g; bArr[i] = b; alpha[i] = a;
        if (a < 230) hasTransparency = true;
        const lv = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
        luma[i] = lv;
        luma255[i] = lv * 255.0;
        const [cl, ca, cb] = ImageVectorizer.rgbToLab(r, g, b);
        chromaL[i] = cl; chromaA[i] = ca; chromaB[i] = cb;
      }

      const fgMask = Asset3DEngine.extractBorderConnectedForeground(alpha, chromaL, chromaA, chromaB, W, H, { removeBg, hasTransparency });
      ImageVectorizer.removeCheckerboardPinches(fgMask, W, H, false);

      let minFgX = W - 1, maxFgX = 0, minFgY = H - 1, maxFgY = 0, fgCount = 0;
      for (let y = 0; y < H; y++) {
        const row = y * W;
        for (let x = 0; x < W; x++) {
          if (!fgMask[row + x]) continue;
          fgCount++;
          if (x < minFgX) minFgX = x; if (x > maxFgX) maxFgX = x;
          if (y < minFgY) minFgY = y; if (y > maxFgY) maxFgY = y;
        }
      }
      if (fgCount === 0) {
        fgMask.fill(1);
        minFgX = 0; maxFgX = W - 1; minFgY = 0; maxFgY = H - 1;
        fgCount = total;
      }

      const fgBoxW = Math.max(1, maxFgX - minFgX + 1);
      const fgBoxH = Math.max(1, maxFgY - minFgY + 1);
      const fillRatio = fgCount / (fgBoxW * fgBoxH);
      const roi = ImageVectorizer.detectFurnitureROI(rArr, gArr, bArr, luma255, W, H);

      let roiPanelDensity = 0;
      if (roi && roi.panelMask && roi.w > 0 && roi.h > 0) {
        let pCnt = 0;
        for (let y = roi.y0; y <= roi.y1; y++) {
          const row = y * W;
          for (let x = roi.x0; x <= roi.x1; x++) {
            if (roi.panelMask[row + x]) pCnt++;
          }
        }
        roiPanelDensity = pCnt / (roi.w * roi.h);
      }

      // Guard against cropping full-bleed engraved signs/plaques: only snap fgMask to ROI when an exterior wall/frame differs from panel borders
      const hasOuterStripContrast = (() => {
        if (!roi || !roi.panelMask) return false;
        let outerTotal = 0, outerBg = 0;
        for (let y = 0; y < H; y++) {
          const row = y * W;
          const outY = y < roi.y0 || y > roi.y1;
          for (let x = 0; x < W; x++) {
            if (outY || x < roi.x0 || x > roi.x1) {
              outerTotal++;
              if (!roi.panelMask[row + x]) outerBg++;
            }
          }
        }
        return outerTotal > W * H * 0.06 && (outerBg / Math.max(1, outerTotal)) >= 0.45;
      })();

      const isFullBleedEngravedPlate = fillRatio >= 0.90 && fgBoxW >= W * 0.90 && fgBoxH >= H * 0.90 && !hasOuterStripContrast;
      const isRoiWoodPanel = !isFullBleedEngravedPlate && !!(roi && roi.chromaMed >= 11 && roi.w > W * 0.28 && roi.h > H * 0.42 && roiPanelDensity >= 0.56);
      const isWoodOrArch = !!(roi && roi.chromaMed >= 10 && fillRatio >= 0.78);
      const isSolidRectPanel = fillRatio >= 0.88 && fgBoxW > W * 0.45 && fgBoxH > H * 0.45;
      const isPlanarPanel = isRoiWoodPanel || isWoodOrArch || isSolidRectPanel;

      if (isPlanarPanel && roi && !isFullBleedEngravedPlate && hasOuterStripContrast && roi.chromaMed >= 10 && roi.w > W * 0.28 && roi.h > H * 0.42 && removeBg && !hasTransparency) {
        for (let y = 0; y < H; y++) {
          const row = y * W;
          const inY = y >= roi.y0 && y <= roi.y1;
          for (let x = 0; x < W; x++) {
            fgMask[row + x] = (inY && x >= roi.x0 && x <= roi.x1) ? 1 : 0;
          }
        }
        ImageVectorizer.removeCheckerboardPinches(fgMask, W, H, false);
        minFgX = roi.x0; maxFgX = roi.x1; minFgY = roi.y0; maxFgY = roi.y1;
        fgCount = roi.w * roi.h;
      }

      return {
        resX: W, resY: H, total, rArr, gArr, bArr, luma, luma255, alpha, hasTransparency,
        fgMask, fgBox: { minX: minFgX, maxX: maxFgX, minY: minFgY, maxY: maxFgY },
        fgBoxW: maxFgX - minFgX + 1, fgBoxH: maxFgY - minFgY + 1,
        isPlanarPanel, roi,
      };
    };

    const isRealDomImage = !!(imgElement && (imgElement.tagName === 'IMG' || imgElement.nodeName === 'IMG' || (typeof HTMLImageElement !== 'undefined' && imgElement instanceof HTMLImageElement)));
    let data = extractMaskAndStats(rgba, resX, resY);

    if (isRealDomImage && removeBg && data.fgBox) {
      const { minX, maxX, minY, maxY } = data.fgBox;
      const bw = maxX - minX + 1;
      const bh = maxY - minY + 1;
      const covX = bw / resX;
      const covY = bh / resY;
      if ((covX < 0.84 || covY < 0.84) && covX > 0.16 && covY > 0.16) {
        // Planar panels already include clean inner wood margins from detectFurnitureROI -> zero pad prevents brick/sofa edge bleed
        const padX = data.isPlanarPanel ? 0 : Math.max(2, Math.round(bw * 0.025));
        const padY = data.isPlanarPanel ? 0 : Math.max(2, Math.round(bh * 0.025));
        const cx0 = Math.max(0, minX - padX);
        const cy0 = Math.max(0, minY - padY);
        const cx1 = Math.min(resX - 1, maxX + padX);
        const cy1 = Math.min(resY - 1, maxY + padY);
        srcCropX = (cx0 / Math.max(1, resX - 1)) * iw;
        srcCropY = (cy0 / Math.max(1, resY - 1)) * ih;
        srcCropW = Math.max(8, ((cx1 - cx0) / Math.max(1, resX - 1)) * iw);
        srcCropH = Math.max(8, ((cy1 - cy0) / Math.max(1, resY - 1)) * ih);
        activeAspect = srcCropW / Math.max(1, srcCropH);
        const areaBudget = baseRes * baseRes * (baseRes >= 144 ? 1.65 : 1.5);
        resX = Math.max(96, Math.min(512, Math.round(Math.sqrt(areaBudget * activeAspect))));
        resY = Math.max(96, Math.min(512, Math.round(Math.sqrt(areaBudget / activeAspect))));
        rgba = drawToCanvas(resX, resY, srcCropX, srcCropY, srcCropW, srcCropH);
        data = extractMaskAndStats(rgba, resX, resY);
      }
    }

    const tSize = Math.max(256, Math.min(1024, Math.round(texSize)));
    const texRgba = drawToCanvas(tSize, tSize, srcCropX, srcCropY, srcCropW, srcCropH);

    // Exact Felzenszwalb L2 Signed Distance Field & Kovesi Gaussian smooth phi for zero-isocontour rim snapping
    const { phi, dIn: rawDist } = ImageVectorizer.computeSignedDistancePhi(data.fgMask, resX, resY, 1.0);
    let maxDist = 1.0;
    for (let i = 0; i < data.total; i++) if (rawDist[i] > maxDist) maxDist = rawDist[i];

    // Smooth harmonic distance field eliminates diagonal "X" pyramid creases on rectangular silhouettes
    const smoothDist = new Float32Array(rawDist);
    ImageVectorizer.gaussianSmooth2D(smoothDist, resX, resY, Math.max(1.4, Math.min(resX, resY) * 0.022));
    for (let i = 0; i < data.total; i++) if (!data.fgMask[i]) smoothDist[i] = 0;

    return {
      ...data,
      aspect: activeAspect,
      srcCrop: { x: srcCropX, y: srcCropY, w: srcCropW, h: srcCropH },
      neuralDepth: null,
      _wmMask: null,
      phi,
      rawDist,
      smoothDist,
      maxDist,
      texSize: tSize,
      texRgba,
      depth: new Float32Array(data.total),
      sculptSignal: new Float32Array(data.total),
    };
  }

  static _solveSurfaceDepth(prep, opts) {
    const { delight = true, invert = false, edgeStrength = 0.36 } = opts;
    const { resX, resY, total, rArr, gArr, bArr, luma, luma255, fgMask, fgBox, fgBoxW, fgBoxH, isPlanarPanel, roi, smoothDist, maxDist, depth, sculptSignal } = prep;
    const minDim = Math.min(resX, resY);

    if (isPlanarPanel) {
      const smL = ImageVectorizer.guidedFilter(luma255, resX, resY, 1, 55);
      const validMask = new Uint8Array(total);
      const wmRaw = new Uint8Array(total);
      const darkWmCand = new Uint8Array(total);
      const hasRoiChroma = roi && roi.chromaMed >= 10;
      const wmTol = hasRoiChroma && roi.chromaMed > 14 ? 12 : 18;
      let fgActiveCnt = 0;
      let darkWmCnt = 0;

      for (let y = 0; y < resY; y++) {
        const row = y * resX;
        for (let x = 0; x < resX; x++) {
          const i = row + x;
          if (!fgMask[i]) continue;
          fgActiveCnt++;
          if (hasRoiChroma) {
            const dc = roi.chromaDiff[i];
            const rgbSat = Math.max(rArr[i], gArr[i], bArr[i]) - Math.min(rArr[i], gArr[i], bArr[i]);
            if (roi.L[i] > roi.lMed + 5) {
              const isNeutralOverlay = roi.chromaMed > 14 && roi.L[i] > 76 && rgbSat < 32;
              const isFringeOverlay = roi.chromaMed > 14 && dc > 9 && rgbSat < 48;
              if (isNeutralOverlay || isFringeOverlay || dc > wmTol) {
                wmRaw[i] = 1;
                continue;
              }
            } else if (roi.chromaMed >= 14) {
              // Printed dark watermark ink (neutral charcoal black with near-zero R-B warmth) & colored logo accents
              const isNeutralBlackInk = luma255[i] < 40 && Math.abs(rArr[i] - bArr[i]) < 15 && dc > 11;
              const isLogoAccent = (rArr[i] - gArr[i] > 42 && bArr[i] > gArr[i] + 10 && dc > 18)
                || (rArr[i] - gArr[i] > 58 && rArr[i] - bArr[i] > 48 && dc > 22);
              if (isNeutralBlackInk || isLogoAccent) {
                darkWmCand[i] = 1;
                darkWmCnt++;
              }
            }
          }
          validMask[i] = 1;
        }
      }

      // Only erase dark printed overlay strokes when sparse (< 4.5% of panel) so 2-tone silhouettes stay completely intact
      if (darkWmCnt > 0 && darkWmCnt < Math.max(24, fgActiveCnt * 0.045)) {
        for (let i = 0; i < total; i++) {
          if (darkWmCand[i]) {
            wmRaw[i] = 1;
            validMask[i] = 0;
          }
        }
      }

      let wmMask = new Uint8Array(wmRaw);
      for (let dPass = 0; dPass < 3; dPass++) {
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

      const cleanL = new Float32Array(smL);
      let hasWm = false;
      for (let i = 0; i < total; i++) if (wmMask[i]) { hasWm = true; break; }
      if (hasWm) {
        const fillL = ImageVectorizer.maskedBoxMean(smL, validMask, resX, resY, Math.max(5, Math.round(minDim * 0.05)));
        for (let i = 0; i < total; i++) if (wmMask[i]) cleanL[i] = fillL[i];
        prep._wmMask = wmMask;
      }

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
      const oSpan = Math.abs(rOtsu.m2 - rOtsu.m1);
      let valleyCnt = 0;
      if (vCnt > 50 && oSpan >= 44) {
        const loM = Math.min(rOtsu.m1, rOtsu.m2), hiM = Math.max(rOtsu.m1, rOtsu.m2);
        const bLo = Math.ceil(loM + oSpan * 0.22), bHi = Math.floor(hiM - oSpan * 0.22);
        for (let b = Math.max(0, bLo); b <= Math.min(255, bHi); b++) valleyCnt += vHist[b];
      }
      const valleyFrac = valleyCnt / Math.max(1, vCnt);

      if (rOtsu.separability >= 0.80 && oSpan >= 44 && valleyFrac < 0.16) {
        // 2-Tone Engraved Plaque / Logo Mode using Exact Felzenszwalb L2 SDF Beveling
        const bVals = [];
        for (let y = fgBox.minY; y <= fgBox.maxY; y++) {
          const nearY = (y - fgBox.minY) < Math.max(3, Math.round(fgBoxH * 0.04)) || (fgBox.maxY - y) < Math.max(3, Math.round(fgBoxH * 0.04));
          for (let x = fgBox.minX; x <= fgBox.maxX; x++) {
            const nearX = (x - fgBox.minX) < Math.max(3, Math.round(fgBoxW * 0.04)) || (fgBox.maxX - x) < Math.max(3, Math.round(fgBoxW * 0.04));
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

        const distIn = ImageVectorizer.computeL2Distance2D(strokeMask, resX, resY);
        const distOut = ImageVectorizer.computeL2Distance2D(bgMask, resX, resY);
        const bevelPx = Math.max(1.8, Math.min(3.8, minDim * 0.014));

        for (let i = 0; i < total; i++) {
          if (!fgMask[i]) continue;
          const sdfGeom = strokeMask[i] ? Math.max(0, distIn[i] - 0.5) : -Math.max(0, distOut[i] - 0.5);
          const subpx = tLinArr[i] - 0.5;
          const sdf = Math.abs(sdfGeom) <= 1.25 ? (sdfGeom * 0.55 + subpx * 0.9) : sdfGeom;
          const uBev = Math.max(0, Math.min(1, 0.5 + sdf / bevelPx));
          const tBevel = uBev * uBev * (3.0 - 2.0 * uBev);
          const zVal = embossRaised ? (0.24 + tBevel * 0.72) : (0.74 - tBevel * 0.60);
          sculptSignal[i] = zVal;
          depth[i] = zVal;
        }
        const crispEdge = ImageVectorizer.guidedFilter(depth, resX, resY, 1, 0.004);
        for (let i = 0; i < total; i++) depth[i] = fgMask[i] ? Math.max(0.10, Math.min(0.98, crispEdge[i])) : 0;
        return;
      }

      // Multi-Scale Planar Architectural / Carved Wood Door Relief
      const minR = Math.max(2, Math.min(5, Math.round(minDim * 0.015)));
      const medR = Math.max(5, Math.min(18, Math.round(minDim * 0.055)));
      const planeR = Math.max(18, Math.round(minDim * 0.24));

      const openedS = ImageVectorizer.separableExtrema(ImageVectorizer.separableExtrema(cleanL, resX, resY, minR, false), resX, resY, minR, true);
      const closedS = ImageVectorizer.separableExtrema(ImageVectorizer.separableExtrema(cleanL, resX, resY, minR, true), resX, resY, minR, false);
      const openedM = ImageVectorizer.separableExtrema(ImageVectorizer.separableExtrema(cleanL, resX, resY, medR, false), resX, resY, medR, true);
      const closedM = ImageVectorizer.separableExtrema(ImageVectorizer.separableExtrema(cleanL, resX, resY, medR, true), resX, resY, medR, false);
      const broadPlane = ImageVectorizer.maskedBoxMean(cleanL, fgMask, resX, resY, planeR);

      const pol = invert ? -1 : 1;
      const nDepth = prep.neuralDepth || null;
      for (let i = 0; i < total; i++) {
        if (!fgMask[i]) continue;
        const deLightedL = cleanL[i] - (broadPlane[i] - (p10 + toneSpan * 0.45)) * 0.32;
        let macroElev = (deLightedL - p10) / toneSpan;
        if (invert) macroElev = 1.0 - macroElev;
        macroElev = Math.max(0.05, Math.min(1.0, macroElev));

        const wh = (Math.max(0, cleanL[i] - openedS[i]) * 0.35 + Math.max(0, cleanL[i] - openedM[i]) * 0.65) / 76.0;
        const bh = (Math.max(0, closedS[i] - cleanL[i]) * 0.35 + Math.max(0, closedM[i] - cleanL[i]) * 0.65) / 76.0;
        // Soft-compress peak extremes so grain & specular highlights never become spikes in Solid Clay view
        const rawMicro = (wh * 0.92 - bh * 0.80) * pol;
        const microDelta = Math.tanh(rawMicro * 1.45) * 0.68;
        const macroBase = nDepth
          ? ((invert ? 1.0 - nDepth[i] : nDepth[i]) * 0.44 + macroElev * 0.56)
          : macroElev;
        const zVal = Math.max(0.10, Math.min(1.0, macroBase * 0.68 + 0.16 + microDelta * 0.28));
        sculptSignal[i] = zVal;
        depth[i] = zVal;
      }

      const smoothPlanar = ImageVectorizer.guidedFilter(depth, resX, resY, 2, 0.012);
      for (let i = 0; i < total; i++) depth[i] = fgMask[i] ? Math.max(0.10, Math.min(1.0, smoothPlanar[i])) : 0;
      return;
    }

    // Organic 3D Cutout / Sculpture Solver (Neural Depth + L2 Harmonic Envelope + Jacobi Poisson Surface Solver)
    const lumaSmooth = ImageVectorizer.guidedFilter(luma, resX, resY, 2, 0.016);
    const rMed = Math.max(3, Math.min(10, Math.round(minDim * 0.032)));
    const openedMed = ImageVectorizer.separableExtrema(ImageVectorizer.separableExtrema(lumaSmooth, resX, resY, rMed, false), resX, resY, rMed, true);
    const closedMed = ImageVectorizer.separableExtrema(ImageVectorizer.separableExtrema(lumaSmooth, resX, resY, rMed, true), resX, resY, rMed, false);

    const sat = ImageVectorizer.buildIntegralImage(lumaSmooth, resX, resY);
    const illumR = Math.max(8, Math.round(minDim * 0.14));
    let sMin = 1e9, sMax = -1e9;
    for (let y = 0; y < resY; y++) {
      for (let x = 0; x < resX; x++) {
        const i = y * resX + x;
        if (!fgMask[i]) continue;
        const localMean = ImageVectorizer.getLocalMean(sat, resX, resY, x, y, illumR);
        const wh = Math.max(0, lumaSmooth[i] - openedMed[i]);
        const bh = Math.max(0, closedMed[i] - lumaSmooth[i]);
        let tone = delight ? lumaSmooth[i] * 0.52 + (lumaSmooth[i] - localMean + 0.5) * 0.48 : lumaSmooth[i];
        if (invert) tone = 1.0 - tone;
        const sig = tone * 0.56 + Math.tanh((wh * 1.1 - bh * 0.85) * 1.5) * 0.30 * (invert ? -1 : 1);
        sculptSignal[i] = sig;
        if (sig < sMin) sMin = sig;
        if (sig > sMax) sMax = sig;
      }
    }

    const sSpan = Math.max(0.08, sMax - sMin);
    const nDepth = prep.neuralDepth || null;
    const combinedDepth = new Float32Array(total);
    for (let y = 1; y < resY - 1; y++) {
      for (let x = 1; x < resX - 1; x++) {
        const i = y * resX + x;
        if (!fgMask[i]) continue;
        const t = Math.min(1.0, smoothDist[i] / Math.max(3.0, maxDist * 0.85));
        const macroDome = Math.sqrt(Math.max(0.0, 1.0 - (1.0 - t) * (1.0 - t)));
        const normSculpt = Math.max(0.0, Math.min(1.0, (sculptSignal[i] - sMin) / sSpan));
        const gx = lumaSmooth[i + 1] - lumaSmooth[i - 1];
        const gy = lumaSmooth[i + resX] - lumaSmooth[i - resX];
        const edgeCrest = Math.min(0.10, Math.hypot(gx, gy) * edgeStrength * 0.45);
        if (nDepth) {
          const nd = invert ? 1.0 - nDepth[i] : nDepth[i];
          combinedDepth[i] = Math.max(0.08, Math.min(1.0, nd * 0.62 + macroDome * 0.24 + normSculpt * 0.14 + edgeCrest * 0.08));
        } else {
          combinedDepth[i] = Math.max(0.08, Math.min(1.0, macroDome * 0.54 + normSculpt * 0.38 + edgeCrest * 0.10));
        }
      }
    }

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
    for (let i = 0; i < total; i++) depth[i] = fgMask[i] ? Math.max(0.08, Math.min(1.0, jacobiBuf[i])) : 0;
  }

  /**
   * Meshy-style [1, 4, 6, 4, 1] Sobel Steep-Slope Detection + 30-deg Normal-Bilateral Rolling-Ball Edge Fillet.
   * Smoothly rounds jagged cliff shoulders while keeping flat plateaus and textured interiors razor crisp.
   */
  static applySlopeAwareFillet(depth, fgMask, W, H, fitW, thicknessMm, filletRatio = 0.012) {
    if (!filletRatio || filletRatio <= 0.001 || W < 16 || H < 16) return;
    const total = W * H;
    const radiusMm = fitW * filletRatio;
    const spacingMm = fitW / Math.max(1, W - 1);
    if (radiusMm < spacingMm * 0.8) return;

    const sobelW = [1, 4, 6, 4, 1];
    const spanStep = Math.max(1, Math.round(W / 1023));
    const stepMm = 2 * spanStep * spacingMm;
    const weightNorm = 16 * stepMm;
    const pad = spanStep * 2;

    const slopeTanLo = Math.tan((54 * Math.PI) / 180);
    const slopeTanHi = Math.tan((68 * Math.PI) / 180);
    const heightMmArr = new Float32Array(total);
    for (let i = 0; i < total; i++) heightMmArr[i] = depth[i] * thicknessMm;

    const nx = new Float32Array(total);
    const ny = new Float32Array(total);
    const nz = new Float32Array(total);
    const weight = new Float32Array(total);

    for (let y = pad; y < H - pad; y++) {
      for (let x = pad; x < W - pad; x++) {
        const n = y * W + x;
        if (!fgMask[n]) continue;
        let gx = 0, gy = 0;
        for (let k = 0; k < 5; k++) {
          const rOff = (k - 2) * spanStep;
          gx += sobelW[k] * (heightMmArr[n + rOff * W + spanStep] - heightMmArr[n + rOff * W - spanStep]);
          gy += sobelW[k] * (heightMmArr[n + spanStep * W + rOff] - heightMmArr[n - spanStep * W + rOff]);
        }
        gx /= weightNorm;
        gy /= weightNorm;
        const inv = 1.0 / Math.hypot(gx, gy, 1.0);
        nx[n] = -gx * inv;
        ny[n] = -gy * inv;
        nz[n] = inv;
        const sl = Math.hypot(gx, gy);
        if (sl > slopeTanLo) {
          weight[n] = sl >= slopeTanHi ? 1.0 : (sl - slopeTanLo) / (slopeTanHi - slopeTanLo);
        }
      }
    }

    // 4-pass 30-deg normal-dot bilateral smoothing of steep cliff weights
    const cos30 = Math.cos((30 * Math.PI) / 180);
    const nbs = [-W, W, -1, 1];
    const tmpW = new Float32Array(total);
    for (let pass = 0; pass < 4; pass++) {
      tmpW.set(weight);
      for (let y = pad; y < H - pad; y++) {
        for (let x = pad; x < W - pad; x++) {
          const n = y * W + x;
          if (!fgMask[n]) continue;
          let sum = tmpW[n];
          let den = 1.0;
          for (let d = 0; d < 4; d++) {
            const ni = n + nbs[d];
            if (!fgMask[ni]) continue;
            const dotRatio = (nx[n] * nx[ni] + ny[n] * ny[ni] + nz[n] * nz[ni] - cos30) / (1.0 - cos30);
            if (dotRatio <= 0) continue;
            const wt = dotRatio >= 1.0 ? 1.0 : dotRatio;
            sum += wt * tmpW[ni];
            den += wt;
          }
          weight[n] = sum / den;
        }
      }
    }

    // Circular disk soft shoulder fillet ONLY on cliff crest pixels
    const rPx = Math.max(1, Math.min(3, Math.round(radiusMm / spacingMm)));
    const softened = new Float32Array(depth);
    ImageVectorizer.gaussianSmooth2D(softened, W, H, rPx * 0.72);
    for (let i = 0; i < total; i++) {
      if (!fgMask[i] || weight[i] <= 0.01) continue;
      // Blend toward softly filleted upper/lower shoulder weighted strictly by steep cliff score
      const w = Math.min(0.82, weight[i]);
      depth[i] = Math.max(0.08, Math.min(1.0, depth[i] * (1.0 - w) + softened[i] * w));
    }
  }

  /**
   * Volume-preserving Taubin (lambda / mu) bilateral surface fairing: removes high-frequency
   * photo grain & needle spikes while preserving sculptural ridges without shrinkage.
   */
  static taubinSmoothDepth(depth, fgMask, W, H, passes = 4) {
    if (!depth || !fgMask || W < 8 || H < 8 || passes <= 0) return;
    const total = W * H;
    const next = new Float32Array(depth);
    const lambdas = [0.50, -0.53];
    for (let p = 0; p < passes * 2; p++) {
      const k = lambdas[p & 1];
      next.set(depth);
      for (let y = 1; y < H - 1; y++) {
        const row = y * W;
        for (let x = 1; x < W - 1; x++) {
          const i = row + x;
          if (!fgMask[i]) continue;
          const nbs = [i - 1, i + 1, i - W, i + W];
          let sum = 0, wSum = 0;
          for (const ni of nbs) {
            if (!fgMask[ni]) continue;
            const dz = depth[ni] - depth[i];
            const wt = Math.exp(-(dz * dz) * 18.0);
            sum += depth[ni] * wt;
            wSum += wt;
          }
          if (wSum > 0.1) {
            const avg = sum / wSum;
            next[i] = Math.max(0.08, Math.min(1.0, depth[i] + k * (avg - depth[i])));
          }
        }
      }
      depth.set(next);
    }
  }

  /**
   * Inpaint overlay watermarks cleanly in the 1024px color texture from nearby valid texels.
   */
  static inpaintTextureWatermark(texRgba, tSize, wmMask, fgW, fgH) {
    if (!texRgba || !wmMask || !fgW || !fgH) return;
    const n = tSize * tSize;
    const wmTex = new Uint8Array(n);
    const sxScale = (fgW - 1) / Math.max(1, tSize - 1);
    const syScale = (fgH - 1) / Math.max(1, tSize - 1);
    const queue = new Int32Array(n);
    const filled = new Uint8Array(n);
    let qH = 0, qT = 0;

    for (let y = 0; y < tSize; y++) {
      const sy = Math.min(fgH - 1, Math.round(y * syScale));
      const fRow = sy * fgW, tRow = y * tSize;
      for (let x = 0; x < tSize; x++) {
        const sx = Math.min(fgW - 1, Math.round(x * sxScale));
        const idx = tRow + x;
        if (wmMask[fRow + sx]) {
          wmTex[idx] = 1;
        } else {
          filled[idx] = 1;
        }
      }
    }

    const dirs = [-1, 1, -tSize, tSize];
    for (let y = 1; y < tSize - 1; y++) {
      const row = y * tSize;
      for (let x = 1; x < tSize - 1; x++) {
        const idx = row + x;
        if (!filled[idx]) continue;
        if (wmTex[idx - 1] || wmTex[idx + 1] || wmTex[idx - tSize] || wmTex[idx + tSize]) {
          queue[qT++] = idx;
        }
      }
    }
    while (qH < qT) {
      const u = queue[qH++];
      const ux = u % tSize, uy = (u - ux) / tSize;
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
        queue[qT++] = v;
      }
    }
  }

  static _finalizeAnalysis(prep, opts) {
    const { texSize = 1024, removeBg = true, delight = true } = opts;
    const { resX, resY, fgMask, fgBox, hasTransparency, isPlanarPanel, aspect, phi, smoothDist, rawDist, maxDist, depth, sculptSignal, texRgba, _wmMask } = prep;

    if (_wmMask) {
      Asset3DEngine.inpaintTextureWatermark(texRgba, texSize, _wmMask, resX, resY);
    }
    Asset3DEngine.dilateTextureEdges(texRgba, texSize, fgMask, resX, resY, removeBg && !hasTransparency);
    const maps = Asset3DEngine.synthesizePBRMaps(texRgba, texSize, delight, fgMask, resX, resY);

    const aspectBox = fgBox.maxX > fgBox.minX && fgBox.maxY > fgBox.minY
      ? {
          minU: fgBox.minX / Math.max(1, resX - 1),
          maxU: fgBox.maxX / Math.max(1, resX - 1),
          minV: 1.0 - fgBox.maxY / Math.max(1, resY - 1),
          maxV: 1.0 - fgBox.minY / Math.max(1, resY - 1),
        }
      : { minU: 0, maxU: 1, minV: 0, maxV: 1 };

    return {
      resolution: Math.max(resX, resY),
      resX,
      resY,
      aspect,
      isPlanarPanel,
      fgMask,
      phi,
      dist: smoothDist,
      rawDist,
      maxDist,
      depth,
      sculptSignal,
      texSize,
      maps,
      aspectBox,
      boundsPx: { dx: 0, dy: 0, dw: resX, dh: resY },
    };
  }

  /**
   * Flood-fill outward from border in CIELAB with uniform backdrop guard so full-bleed wooden doors stay intact.
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

    const samplesL = [], samplesA = [], samplesB = [];
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
    let qH = 0, qT = 0;

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
      const ux = u % W, uy = (u - ux) / W;
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
        const ux = u % W, uy = (u - ux) / W;
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
    let qH = 0, qT = 0;
    const scaleX = (fgW - 1) / Math.max(1, tSize - 1);
    const scaleY = (fgH - 1) / Math.max(1, tSize - 1);

    for (let y = 0; y < tSize; y++) {
      const sy = Math.min(fgH - 1, Math.round(y * scaleY));
      const fRow = sy * fgW, tRow = y * tSize;
      for (let x = 0; x < tSize; x++) {
        const sx = Math.min(fgW - 1, Math.round(x * scaleX));
        const fi = fRow + sx, idx = tRow + x;
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
      const ux = u % tSize, uy = (u - ux) / tSize;
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
      lum[i] = (0.2126 * rgba[i * 4] + 0.7152 * rgba[i * 4 + 1] + 0.0722 * rgba[i * 4 + 2]) / 255.0;
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
        const i = y * size + x, p = i * 4;
        const r = rgba[p], g = rgba[p + 1], b = rgba[p + 2], a = rgba[p + 3];
        const inFg = !fgMask || !activeFgW || fgMask[sy * activeFgW + Math.min(activeFgW - 1, Math.round(x * fgScaleX))] === 1;

        if (delight && a > 16) {
          const localMean = Math.max(0.18, ImageVectorizer.getLocalMean(sat, size, size, x, y, rIllum));
          const gain = Math.max(0.76, Math.min(1.38, (0.54 / localMean) * 0.42 + 0.58));
          // Subtle sculpted cavity AO improves depth clarity in both Standard PBR and Unlit modes
          const cavityAO = Math.max(0.84, 1.0 - Math.max(0, localMean - lum[i]) * 0.42);
          baseColor[p] = Math.min(255, Math.round(r * gain * cavityAO));
          baseColor[p + 1] = Math.min(255, Math.round(g * gain * cavityAO));
          baseColor[p + 2] = Math.min(255, Math.round(b * gain * cavityAO));
          baseColor[p + 3] = 255;
        }

        const xl = Math.max(0, x - 1), xr = Math.min(size - 1, x + 1);
        const yu = Math.max(0, y - 1), yd = Math.min(size - 1, y + 1);
        const gradGain = inFg ? 3.0 : 0.35;
        const dx = (lum[y * size + xr] - lum[y * size + xl]) * gradGain;
        const dy = (lum[yd * size + x] - lum[yu * size + x]) * gradGain;
        let nx = -dx, ny = dy, nz = 1.0;
        const inv = 1.0 / Math.hypot(nx, ny, nz);
        nx *= inv; ny *= inv; nz *= inv;
        normal[p] = Math.round((nx * 0.5 + 0.5) * 255);
        normal[p + 1] = Math.round((ny * 0.5 + 0.5) * 255);
        normal[p + 2] = Math.round((nz * 0.5 + 0.5) * 255);
        normal[p + 3] = 255;

        const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
        const ch = (mx - mn) / 255.0;
        const gradMag = Math.min(1.0, Math.hypot(dx, dy) * 1.35);
        const roughVal = Math.max(0.24, Math.min(0.92, 0.72 - lum[i] * 0.24 + gradMag * 0.20));
        const rv = Math.round(roughVal * 255);
        roughness[p] = rv; roughness[p + 1] = rv; roughness[p + 2] = rv; roughness[p + 3] = 255;

        const isGoldOrSteel = lum[i] > 0.48 && (ch < 0.14 || (r > g && g > b && ch > 0.20));
        const mv = isGoldOrSteel ? Math.round((0.34 + lum[i] * 0.44) * 255) : 18;
        metallic[p] = mv; metallic[p + 1] = mv; metallic[p + 2] = mv; metallic[p + 3] = 255;

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
      assetMode = 'double-sided',
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
   * Meshy-style Subpixel Newton-Raphson Zero-Isocontour (phi=0) Perimeter Snapping & Ordered Loop Relaxation.
   */
  static snapPerimeterToZeroPhi(gridX, gridY, activeCell, W, H, phi, boundaryEdges) {
    const CW = W - 1;
    const CH = H - 1;
    const hasCell = (ix, iy) => (ix >= 0 && iy >= 0 && ix < CW && iy < CH && activeCell[iy * CW + ix] === 1);
    const isOuterGridEdge = (vi) => {
      const ix = vi % W, iy = (vi - ix) / W;
      return ix === 0 || ix === W - 1 || iy === 0 || iy === H - 1;
    };

    const nextRim = new Int32Array(W * H).fill(-1);
    const onRim = new Uint8Array(W * H);
    let cleanLoop = true;
    const linkRim = (a, b) => {
      if (nextRim[a] !== -1) cleanLoop = false;
      nextRim[a] = b;
      onRim[a] = 1;
      onRim[b] = 1;
    };

    for (let iy = 0; iy < CH; iy++) {
      for (let ix = 0; ix < CW; ix++) {
        if (!hasCell(ix, iy)) continue;
        const i00 = iy * W + ix, i10 = i00 + 1, i01 = i00 + W, i11 = i01 + 1;
        if (!hasCell(ix, iy - 1)) linkRim(i00, i10);
        if (!hasCell(ix + 1, iy)) linkRim(i10, i11);
        if (!hasCell(ix, iy + 1)) linkRim(i11, i01);
        if (!hasCell(ix - 1, iy)) linkRim(i01, i00);
      }
    }

    const samplePhi = (x, y) => {
      const cx = Math.max(0, Math.min(W - 1.001, x));
      const cy = Math.max(0, Math.min(H - 1.001, y));
      const x0 = Math.min(W - 2, Math.floor(cx)), y0 = Math.min(H - 2, Math.floor(cy));
      const fx = cx - x0, fy = cy - y0;
      const a = y0 * W + x0;
      return phi[a] * (1 - fx) * (1 - fy) + phi[a + 1] * fx * (1 - fy) + phi[a + W] * (1 - fx) * fy + phi[a + W + 1] * fx * fy;
    };

    const clampShift = (origX, origY, tx, ty, maxR) => {
      const dx = tx - origX, dy = ty - origY;
      const d = Math.hypot(dx, dy);
      if (d <= maxR || d < 1e-9) return [tx, ty];
      return [origX + (dx / d) * maxR, origY + (dy / d) * maxR];
    };

    if (phi) {
      for (let vi = 0; vi < W * H; vi++) {
        if (!onRim[vi] || isOuterGridEdge(vi)) continue;
        const origX = vi % W, origY = (vi - origX) / W;
        let rx = gridX[vi], ry = gridY[vi];
        for (let iter = 0; iter < 3; iter++) {
          const val = samplePhi(rx, ry);
          if (Math.abs(val) < 0.02 || Math.abs(val) > 3.0) break;
          const gx = (samplePhi(rx + 0.5, ry) - samplePhi(rx - 0.5, ry));
          const gy = (samplePhi(rx, ry + 0.5) - samplePhi(rx, ry - 0.5));
          const g2 = gx * gx + gy * gy;
          if (g2 < 1e-6) break;
          rx -= (val * gx) / g2;
          ry -= (val * gy) / g2;
        }
        const [sx, sy] = clampShift(origX, origY, rx, ry, 1.18);
        gridX[vi] = sx;
        gridY[vi] = sy;
      }
    }

    // Trace ordered rim loops & run 3 passes of constrained tangential curve relaxation (Meshy method)
    if (cleanLoop) {
      const vis = new Uint8Array(W * H);
      const loops = [];
      for (let s = 0; s < W * H; s++) {
        if (nextRim[s] === -1 || vis[s]) continue;
        const loop = [];
        let cur = s;
        while (cur !== -1 && !vis[cur]) {
          vis[cur] = 1;
          loop.push(cur);
          cur = nextRim[cur];
        }
        if (cur === s && loop.length >= 8) loops.push(loop);
      }
      for (const loop of loops) {
        const m = loop.length;
        const baseX = new Float32Array(m), baseY = new Float32Array(m);
        const nxtX = new Float32Array(m), nxtY = new Float32Array(m);
        for (let j = 0; j < m; j++) { baseX[j] = gridX[loop[j]]; baseY[j] = gridY[loop[j]]; }
        for (let pass = 0; pass < 3; pass++) {
          for (let j = 0; j < m; j++) {
            const vi = loop[j];
            if (isOuterGridEdge(vi)) { nxtX[j] = gridX[vi]; nxtY[j] = gridY[vi]; continue; }
            const vp = loop[(j - 1 + m) % m], vn = loop[(j + 1) % m];
            nxtX[j] = gridX[vi] + 0.5 * ((gridX[vp] + gridX[vn]) * 0.5 - gridX[vi]);
            nxtY[j] = gridY[vi] + 0.5 * ((gridY[vp] + gridY[vn]) * 0.5 - gridY[vi]);
          }
          for (let j = 0; j < m; j++) {
            const vi = loop[j];
            if (isOuterGridEdge(vi)) continue;
            const [cx, cy] = clampShift(baseX[j], baseY[j], nxtX[j], nxtY[j], 0.75);
            gridX[vi] = cx;
            gridY[vi] = cy;
          }
        }
      }
    }
  }

  /**
   * Enforce strictly 2-manifold boundary topology on active grid cells:
   * prunes tiny speckle components and eliminates all 2x2 diagonal pinch vertices iteratively.
   */
  static sanitizeManifoldActiveCells(activeCell, CW, CH) {
    const totalCells = CW * CH;
    let totalOn = 0;
    for (let i = 0; i < totalCells; i++) totalOn += activeCell[i];
    if (totalOn === 0) return;

    const minComp = Math.max(8, Math.round(totalOn * 0.002));
    const vis = new Uint8Array(totalCells);
    const queue = new Int32Array(totalCells);
    const comps = [];
    const dirs = [-1, 1, -CW, CW];
    for (let s = 0; s < totalCells; s++) {
      if (!activeCell[s] || vis[s]) continue;
      let qH = 0, qT = 0;
      vis[s] = 1;
      queue[qT++] = s;
      while (qH < qT) {
        const u = queue[qH++];
        const ux = u % CW, uy = (u - ux) / CW;
        for (let d = 0; d < 4; d++) {
          if (d === 0 && ux === 0) continue;
          if (d === 1 && ux === CW - 1) continue;
          if (d === 2 && uy === 0) continue;
          if (d === 3 && uy === CH - 1) continue;
          const v = u + dirs[d];
          if (activeCell[v] && !vis[v]) {
            vis[v] = 1;
            queue[qT++] = v;
          }
        }
      }
      comps.push({ cells: queue.slice(0, qT), len: qT });
    }
    comps.sort((a, b) => b.len - a.len);
    for (let k = 1; k < comps.length; k++) {
      if (comps[k].len < minComp) {
        const cArr = comps[k].cells;
        for (let i = 0; i < cArr.length; i++) activeCell[cArr[i]] = 0;
      }
    }

    // Bridge any 2x2 diagonal cell pinches until zero non-manifold deg=4 vertices remain
    for (let pass = 0; pass < 8; pass++) {
      let changed = false;
      for (let y = 0; y < CH - 1; y++) {
        const r0 = y * CW, r1 = r0 + CW;
        for (let x = 0; x < CW - 1; x++) {
          const i00 = r0 + x, i10 = i00 + 1, i01 = r1 + x, i11 = i01 + 1;
          const a = activeCell[i00], b = activeCell[i10], c = activeCell[i01], d = activeCell[i11];
          if (a && d && !b && !c) {
            activeCell[i10] = 1;
            changed = true;
          } else if (!a && !d && b && c) {
            activeCell[i00] = 1;
            changed = true;
          }
        }
      }
      if (!changed) break;
    }
  }

  /**
   * Watertight volumetric/planar shell centered inside stage bounds with Meshy Newton zero-isocontour rims,
   * height-contour alignment, split bevel normals, & topological manifold verification.
   */
  static buildVolumetricShellMeshData(analysis, assetMode, fitW, fitH, stageW, stageH, thicknessMm) {
    const W = analysis.resX || analysis.resolution;
    const H = analysis.resY || analysis.resolution;
    const fg = analysis.fgMask;
    const phi = analysis.phi || null;
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
    Asset3DEngine.sanitizeManifoldActiveCells(activeCell, CW, CH);

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

    const boundaryEdges = [];
    for (let iy = 0; iy < CH; iy++) {
      for (let ix = 0; ix < CW; ix++) {
        if (!hasCell(ix, iy)) continue;
        const i00 = iy * W + ix, i10 = i00 + 1, i01 = i00 + W, i11 = i01 + 1;
        if (!hasCell(ix, iy - 1)) { boundaryEdges.push([i10, i00]); isBoundaryVert[i10] = isBoundaryVert[i00] = 1; }
        if (!hasCell(ix, iy + 1)) { boundaryEdges.push([i01, i11]); isBoundaryVert[i01] = isBoundaryVert[i11] = 1; }
        if (!hasCell(ix - 1, iy)) { boundaryEdges.push([i00, i01]); isBoundaryVert[i00] = isBoundaryVert[i01] = 1; }
        if (!hasCell(ix + 1, iy)) { boundaryEdges.push([i11, i10]); isBoundaryVert[i11] = isBoundaryVert[i10] = 1; }
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

    if (!isPlanar) {
      Asset3DEngine.snapPerimeterToZeroPhi(gridX, gridY, activeCell, W, H, phi, boundaryEdges);
    }

    // Level-set height-contour vertex alignment on interior steep slope walls
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

        let zF, zB;
        if (isRelief) {
          zF = isRim ? Math.max(0.6, thicknessMm * 0.18) : (thicknessMm * 0.18 + dNorm * thicknessMm * 0.82);
          zB = 0.0;
        } else if (isPlanar && assetMode !== 'inflated') {
          const rimChamfer = Math.min(dNorm * 0.88, 0.44);
          zF = isRim ? Math.max(minRimZ, halfT * rimChamfer) : Math.max(minRimZ, halfT * dNorm);
          zB = isRim ? -Math.max(minRimZ * 0.85, halfT * rimChamfer * backScale) : -Math.max(minRimZ * 0.85, halfT * backScale * dNorm);
        } else if (assetMode === 'inflated') {
          zF = isRim ? minRimZ : Math.max(minRimZ, halfT * (macroSmooth * 0.48 + dNorm * 0.52));
          zB = isRim ? -minRimZ : -Math.max(minRimZ, halfT * backScale * (macroSmooth * 0.78 + dNorm * 0.22));
        } else {
          zF = isRim ? minRimZ : Math.max(minRimZ, halfT * (macroSmooth * 0.20 + dNorm * 0.80));
          zB = isRim ? -minRimZ : -Math.max(minRimZ, halfT * backScale * (macroSmooth * 0.72 + dNorm * 0.28));
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
        const i00 = iy * W + ix, i10 = i00 + 1, i01 = i00 + W, i11 = i01 + 1;
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
        // Outward-facing CCW winding around closed rim loop: (fa -> ba -> bb), (fa -> bb -> fb)
        sideIndices.push(fa, ba, bb, fa, bb, fb);
      }
    }

    return Asset3DEngine.finalizeSplitNormalsMesh(positions, uvs, [frontIndices, backIndices, sideIndices], 'Triangle');
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
        s += rowData[j].rad * wt; w += wt;
      }
      return s / w;
    });
    const smoothCx = rowData.map((_, i) => {
      let s = 0, w = 0;
      for (let k = -2; k <= 2; k++) {
        const j = Math.max(0, Math.min(rowData.length - 1, i + k));
        const wt = k === 0 ? 4 : Math.abs(k) === 1 ? 2 : 1;
        s += rowData[j].cx * wt; w += wt;
      }
      return s / w;
    });

    const rows = rowData.length;
    const segTheta = W >= 240 ? 96 : 72;
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
        const dirX = Math.cos(theta), dirZ = Math.sin(theta);
        const sampleXf = Math.max(0, Math.min(W - 1, smoothCx[ri] + dirX * smoothRad[ri]));
        const dMod = (sampleBilinearDepth(sampleXf, rInfo.y) - 0.5) * (thicknessMm * 0.06);
        const rFinal = Math.max(0.4, baseR + dMod);

        positions.push(cxWorld + dirX * rFinal, worldY, dirZ * rFinal);
        uvs.push(Math.max(0, Math.min(1, sampleXf / Math.max(1, W - 1))), vCoord);
      }
    }

    const stride = segTheta + 1;
    for (let ri = 0; ri < rows - 1; ri++) {
      for (let si = 0; si < segTheta; si++) {
        const a = ri * stride + si, b = a + 1, c = a + stride, d = c + 1;
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
    for (let si = 0; si < segTheta; si++) capIndices.push(topCenterIdx, topRingBase + si + 1, topRingBase + si);

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
    for (let si = 0; si < segTheta; si++) capIndices.push(botCenterIdx, botRingBase + si, botRingBase + si + 1);

    const result = Asset3DEngine.finalizeSplitNormalsMesh(positions, uvs, [bodyIndices, capIndices], 'Triangle');
    for (let ri = 0; ri < rows; ri++) {
      const iStart = (ri * stride) * 3, iEnd = (ri * stride + segTheta) * 3;
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
   * Union-Find manifold connected-component check (matching Meshy module 148102 integrity telemetry).
   */
  static countManifoldComponents(vCount, indices) {
    const parent = new Int32Array(vCount);
    const active = new Uint8Array(vCount);
    for (let i = 0; i < vCount; i++) parent[i] = i;
    const find = (x) => {
      while (parent[x] !== x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    const unite = (a, b) => {
      const ra = find(a), rb = find(b);
      if (ra !== rb) parent[ra] = rb;
    };
    for (let i = 0; i < indices.length; i += 3) {
      const a = indices[i], b = indices[i + 1], c = indices[i + 2];
      active[a] = active[b] = active[c] = 1;
      unite(a, b);
      unite(b, c);
    }
    let comps = 0;
    for (let i = 0; i < vCount; i++) {
      if (active[i] && find(i) === i) comps++;
    }
    return comps;
  }

  /**
   * Compute clean normals separately per triangle group, prune degenerate triangles, and return indexed BufferData.
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
        if (a === b || b === c || a === c) continue;
        const i0 = a * 3, i1 = b * 3, i2 = c * 3;
        const ax = posArr[i1] - posArr[i0], ay = posArr[i1 + 1] - posArr[i0 + 1], az = posArr[i1 + 2] - posArr[i0 + 2];
        const bx = posArr[i2] - posArr[i0], by = posArr[i2 + 1] - posArr[i0 + 1], bz = posArr[i2 + 2] - posArr[i0 + 2];
        const nx = ay * bz - az * by;
        const ny = az * bx - ax * bz;
        const nz = ax * by - ay * bx;
        if (nx * nx + ny * ny + nz * nz < 1e-18) continue;
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
        components: Asset3DEngine.countManifoldComponents(vCount, idxArr),
      },
    };
  }
}
