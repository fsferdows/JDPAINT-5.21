/**
 * Browser-Native Neural Vision Pipeline via @huggingface/transformers (ONNX WebGPU / WASM).
 * Runs offline-cached monocular depth estimation (Depth-Anything V2) and salient foreground cutout
 * completely client-side with zero backend servers required.
 */
export class NeuralVision {
  static _depthPipe = null;
  static _mattePipe = null;
  static _depthLoading = null;
  static _matteLoading = null;
  static _cache = new Map();

  static isBrowserRuntime() {
    if (typeof process !== 'undefined' && process.versions?.node) return false;
    return typeof window !== 'undefined'
      && typeof document !== 'undefined'
      && typeof navigator !== 'undefined'
      && typeof HTMLCanvasElement !== 'undefined';
  }

  static async _detectDevice() {
    if (!NeuralVision.isBrowserRuntime()) return 'wasm';
    try {
      if (navigator.gpu) {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) return 'webgpu';
      }
    } catch (_err) {
      // Fall back cleanly to WebAssembly
    }
    return 'wasm';
  }

  static _drawCroppedInput(imgElement, srcCrop, maxEdge = 672) {
    const iw = imgElement.naturalWidth || imgElement.width || 1;
    const ih = imgElement.naturalHeight || imgElement.height || 1;
    const sx = srcCrop?.w ? srcCrop.x : 0;
    const sy = srcCrop?.w ? srcCrop.y : 0;
    const sw = srcCrop?.w ? srcCrop.w : iw;
    const sh = srcCrop?.h ? srcCrop.h : ih;
    const sc = Math.min(1.0, maxEdge / Math.max(1, sw, sh));
    const dw = Math.max(64, Math.round(sw * sc));
    const dh = Math.max(64, Math.round(sh * sc));
    const canvas = document.createElement('canvas');
    canvas.width = dw;
    canvas.height = dh;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(imgElement, sx, sy, sw, sh, 0, 0, dw, dh);
    }
    return { canvas, dw, dh };
  }

  static _bilinearResample(src, sw, sh, dw, dh) {
    const out = new Float32Array(dw * dh);
    if (sw === dw && sh === dh) {
      out.set(src);
      return out;
    }
    const sxScale = (sw - 1) / Math.max(1, dw - 1);
    const syScale = (sh - 1) / Math.max(1, dh - 1);
    for (let y = 0; y < dh; y++) {
      const fy = y * syScale;
      const y0 = Math.floor(fy);
      const y1 = Math.min(sh - 1, y0 + 1);
      const wy = fy - y0;
      const r0 = y0 * sw, r1 = y1 * sw, outR = y * dw;
      for (let x = 0; x < dw; x++) {
        const fx = x * sxScale;
        const x0 = Math.floor(fx);
        const x1 = Math.min(sw - 1, x0 + 1);
        const wx = fx - x0;
        const top = src[r0 + x0] * (1 - wx) + src[r0 + x1] * wx;
        const bot = src[r1 + x0] * (1 - wx) + src[r1 + x1] * wx;
        out[outR + x] = top * (1 - wy) + bot * wy;
      }
    }
    return out;
  }

  static async _getDepthPipeline(onProgress = null) {
    if (NeuralVision._depthPipe) return NeuralVision._depthPipe;
    if (NeuralVision._depthLoading) return NeuralVision._depthLoading;

    NeuralVision._depthLoading = (async () => {
      const { pipeline, env } = await import('@huggingface/transformers');
      env.allowLocalModels = false;
      env.useBrowserCache = true;
      const device = await NeuralVision._detectDevice();
      const progressCb = (ev) => {
        if (!onProgress || !ev) return;
        if (ev.status === 'progress' && typeof ev.progress === 'number') {
          onProgress(1, 10 + Math.round(ev.progress * 0.25), `Loading Neural Depth AI (${device.toUpperCase()}): ${Math.round(ev.progress)}%`);
        }
      };
      try {
        NeuralVision._depthPipe = await pipeline('depth-estimation', 'onnx-community/depth-anything-v2-small', {
          device,
          dtype: 'q8',
          progress_callback: progressCb,
        });
      } catch (_primaryErr) {
        NeuralVision._depthPipe = await pipeline('depth-estimation', 'Xenova/depth-anything-small-hf', {
          device: 'wasm',
          progress_callback: progressCb,
        });
      }
      return NeuralVision._depthPipe;
    })();

    try {
      return await NeuralVision._depthLoading;
    } catch (err) {
      NeuralVision._depthLoading = null;
      throw err;
    }
  }

  static async _getMattePipeline(onProgress = null) {
    if (NeuralVision._mattePipe) return NeuralVision._mattePipe;
    if (NeuralVision._matteLoading) return NeuralVision._matteLoading;

    NeuralVision._matteLoading = (async () => {
      const { pipeline, env } = await import('@huggingface/transformers');
      env.allowLocalModels = false;
      env.useBrowserCache = true;
      const device = await NeuralVision._detectDevice();
      const progressCb = (ev) => {
        if (!onProgress || !ev) return;
        if (ev.status === 'progress' && typeof ev.progress === 'number') {
          onProgress(1, 5 + Math.round(ev.progress * 0.12), `Loading Salient Cutout AI (${device.toUpperCase()}): ${Math.round(ev.progress)}%`);
        }
      };
      NeuralVision._mattePipe = await pipeline('background-removal', 'Xenova/modnet', {
        device,
        dtype: 'q8',
        progress_callback: progressCb,
      });
      return NeuralVision._mattePipe;
    })();

    try {
      return await NeuralVision._matteLoading;
    } catch (err) {
      NeuralVision._matteLoading = null;
      throw err;
    }
  }

  /**
   * Run Neural Salient Cutout + Depth-Anything V2 Monocular Depth on an image/crop and resample to (targetW x targetH).
   * Returns `{ depth: Float32Array|null, matte: Float32Array|null }` or `{ depth: null, matte: null }` in non-browser/offline fallback.
   */
  static async analyzeSubjectAsync(imgElement, targetW, targetH, options = {}, onProgress = null) {
    if (!NeuralVision.isBrowserRuntime()) {
      return { depth: null, matte: null };
    }
    const {
      enableDepth = true,
      enableMatte = false,
      srcCrop = null,
    } = options;

    const iw = imgElement.naturalWidth || imgElement.width || 1;
    const ih = imgElement.naturalHeight || imgElement.height || 1;
    const cacheKey = `${imgElement.src || 'blob'}|${iw}x${ih}|${srcCrop ? `${Math.round(srcCrop.x)},${Math.round(srcCrop.y)},${Math.round(srcCrop.w)},${Math.round(srcCrop.h)}` : 'full'}|${enableDepth}|${enableMatte}`;
    const cached = NeuralVision._cache.get(cacheKey);
    if (cached && cached.targetW === targetW && cached.targetH === targetH) {
      return { depth: cached.depth, matte: cached.matte };
    }

    try {
      const { RawImage } = await import('@huggingface/transformers');
      const { canvas } = NeuralVision._drawCroppedInput(imgElement, srcCrop, 672);
      if (typeof HTMLCanvasElement === 'undefined' || !(canvas instanceof HTMLCanvasElement)) {
        return { depth: null, matte: null };
      }
      const rawImg = RawImage.fromCanvas(canvas);

    let matte = null;
    let depth = null;

    if (enableMatte) {
      try {
        if (onProgress) onProgress(1, 14, 'Running Neural Salient Subject Segmentation (MODNet)...');
        const mattePipe = await NeuralVision._getMattePipeline(onProgress);
        const matteImg = await mattePipe(rawImg);
        if (matteImg && matteImg.data && matteImg.width && matteImg.height) {
          const mw = matteImg.width, mh = matteImg.height, ch = matteImg.channels || 4;
          const mArr = new Float32Array(mw * mh);
          for (let i = 0; i < mw * mh; i++) {
            mArr[i] = matteImg.data[ch >= 4 ? i * ch + 3 : i * ch] / 255.0;
          }
          matte = NeuralVision._bilinearResample(mArr, mw, mh, targetW, targetH);
        }
      } catch (err) {
        console.warn('Neural matte fallback to geometric ROI:', err?.message || err);
      }
    }

    if (enableDepth) {
      try {
        if (onProgress) onProgress(2, 36, 'Running Neural Monocular Depth Estimation (Depth-Anything V2)...');
        const depthPipe = await NeuralVision._getDepthPipeline(onProgress);
        const out = await depthPipe(rawImg);
        const pred = out?.predicted_depth;
        if (pred && pred.data && pred.dims && pred.dims.length >= 2) {
          const dh = pred.dims[pred.dims.length - 2];
          const dw = pred.dims[pred.dims.length - 1];
          const rawData = pred.data;
          const len = dw * dh;
          const sorted = Float32Array.from(rawData.slice(0, len)).sort();
          const dLo = sorted[Math.floor(len * 0.02)] ?? 0;
          const dHi = sorted[Math.floor(len * 0.98)] ?? 1;
          const span = Math.max(1e-5, dHi - dLo);
          const normSrc = new Float32Array(len);
          for (let i = 0; i < len; i++) {
            normSrc[i] = Math.max(0, Math.min(1, (rawData[i] - dLo) / span));
          }
          depth = NeuralVision._bilinearResample(normSrc, dw, dh, targetW, targetH);
        } else if (out?.depth?.data) {
          const dw = out.depth.width, dh = out.depth.height;
          const normSrc = new Float32Array(dw * dh);
          for (let i = 0; i < dw * dh; i++) normSrc[i] = out.depth.data[i] / 255.0;
          depth = NeuralVision._bilinearResample(normSrc, dw, dh, targetW, targetH);
        }
      } catch (err) {
        console.warn('Neural depth fallback to Meshy geometric SDF:', err?.message || err);
      }
    }

    if (NeuralVision._cache.size >= 4) {
      const oldestKey = NeuralVision._cache.keys().next().value;
      NeuralVision._cache.delete(oldestKey);
    }
    NeuralVision._cache.set(cacheKey, { targetW, targetH, depth, matte });
    return { depth, matte };
    } catch (err) {
      console.warn('NeuralVision fallback:', err?.message || err);
      return { depth: null, matte: null };
    }
  }
}
