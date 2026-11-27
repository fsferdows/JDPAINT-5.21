/**
 * Browser client for the local Python OpenCascade (build123d + agentcad) B-Rep Kernel Bridge.
 * Gracefully degrades to offline Browser Mode on static hosting (e.g. Vercel) without errors.
 */
export class CadKernelBridge {
  static BASE_URL = 'http://127.0.0.1:5174';

  static _decodeBase64(b64) {
    if (!b64) return new Uint8Array(0);
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  static _isViteDevOrigin() {
    if (typeof window === 'undefined' || !window.location) return false;
    const host = window.location.hostname;
    return (host === 'localhost' || host === '127.0.0.1') && window.location.port === '5173';
  }

  static async checkHealth(timeoutMs = 1200, { passive = false } = {}) {
    if (typeof fetch !== 'function' || typeof AbortController === 'undefined') {
      return { ok: false };
    }
    const useProxy = CadKernelBridge._isViteDevOrigin();
    // On static hosts (e.g. Vercel), skip passive background probes so offline visits emit zero console errors.
    if (passive && !useProxy) {
      return { ok: false };
    }
    const url = useProxy ? '/__cad_health' : `${CadKernelBridge.BASE_URL}/health`;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const resp = await fetch(url, { signal: ctrl.signal });
      if (!resp.ok) return { ok: false };
      return await resp.json();
    } catch (_err) {
      return { ok: false };
    } finally {
      clearTimeout(timer);
    }
  }

  static async buildFromVectors(entities, options = {}) {
    const payload = {
      workpieceWidth: options.workpieceWidth ?? 100,
      workpieceHeight: options.workpieceHeight ?? 100,
      baseThicknessMm: options.baseThicknessMm ?? 6,
      extrudeHeightMm: options.extrudeHeightMm ?? 5,
      filletRadiusMm: options.filletRadiusMm ?? 0.4,
      entities: (entities || []).map((e) => ({
        type: e.type,
        x: e.x, y: e.y, w: e.w, h: e.h, cornerRadius: e.cornerRadius,
        cx: e.cx, cy: e.cy, r: e.r,
        points: e.points, holes: e.holes, isClosed: e.isClosed, visible: e.visible,
        relief: e.relief,
      })),
    };
    const resp = await fetch(`${CadKernelBridge.BASE_URL}/api/cad/from-vectors`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      throw new Error(data.error || `OpenCascade B-Rep build failed (${resp.status})`);
    }
    return {
      metrics: data.metrics,
      featuresBuilt: data.features_built || 0,
      stepBytes: CadKernelBridge._decodeBase64(data.step_b64),
      glbBuffer: CadKernelBridge._decodeBase64(data.glb_b64).buffer,
      stlBytes: CadKernelBridge._decodeBase64(data.stl_b64),
    };
  }

  static async buildParametricPreset(preset, params = {}) {
    const resp = await fetch(`${CadKernelBridge.BASE_URL}/api/cad/parametric-preset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preset, params }),
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      throw new Error(data.error || `Parametric B-Rep build failed (${resp.status})`);
    }
    return {
      preset,
      metrics: data.metrics,
      stepBytes: CadKernelBridge._decodeBase64(data.step_b64),
      glbBuffer: CadKernelBridge._decodeBase64(data.glb_b64).buffer,
      stlBytes: CadKernelBridge._decodeBase64(data.stl_b64),
    };
  }

  static _encodeImageB64(imgElement, maxEdge = 2048) {
    if (typeof document === 'undefined') return '';
    const iw = imgElement?.naturalWidth || imgElement?.width || 1;
    const ih = imgElement?.naturalHeight || imgElement?.height || 1;
    const sc = Math.min(1.0, maxEdge / Math.max(1, iw, ih));
    const c = document.createElement('canvas');
    c.width = Math.max(32, Math.round(iw * sc));
    c.height = Math.max(32, Math.round(ih * sc));
    const ctx = c.getContext('2d');
    if (ctx) {
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(imgElement, 0, 0, c.width, c.height);
    }
    return c.toDataURL('image/png');
  }

  static _decodeU16Buffer(b64) {
    const bytes = CadKernelBridge._decodeBase64(b64);
    const len = bytes.byteLength >> 1;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const out = new Float32Array(len);
    for (let i = 0; i < len; i++) out[i] = view.getUint16(i * 2, true) / 65535.0;
    return out;
  }

  static async vectorizeViaVision(imgElement, widthMm = 100, heightMm = 100, options = {}) {
    const health = await CadKernelBridge.checkHealth(2000, { passive: true });
    if (!health?.ok || !health?.vision_ai) return null;
    const imageB64 = CadKernelBridge._encodeImageB64(imgElement, 2400);
    if (!imageB64) return null;
    const resp = await fetch(`${CadKernelBridge.BASE_URL}/api/vision/vectorize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_b64: imageB64,
        trace_mode: options.traceMode || 'auto',
        detail: options.detail ?? 0.7,
        corners: options.corners ?? 0.65,
        color_count: options.colorCount ?? 6,
        ignore_bg: options.ignoreBg ?? true,
        width_mm: widthMm,
        height_mm: heightMm,
      }),
    });
    if (!resp.ok) return null;
    const data = await resp.json();
    if (data?.ok && Array.isArray(data.entities) && data.entities.length > 0) {
      if (data.convergence) {
        Object.defineProperty(data.entities, '_convergence', {
          value: data.convergence,
          enumerable: false,
          configurable: true,
        });
      }
      return data.entities;
    }
    return null;
  }

  static async reconstruct3DViaVision(imgElement, options = {}) {
    const health = await CadKernelBridge.checkHealth(2000, { passive: true });
    if (!health?.ok || !health?.vision_ai) return null;
    const imageB64 = CadKernelBridge._encodeImageB64(imgElement, 2400);
    if (!imageB64) return null;
    const resp = await fetch(`${CadKernelBridge.BASE_URL}/api/vision/reconstruct-3d`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_b64: imageB64,
        resolution: options.resolution || 384,
        remove_bg: options.removeBg ?? true,
        delight: options.delight ?? true,
        invert: options.invert ?? false,
        enhance_image: options.enhanceImage ?? true,
      }),
    });
    if (!resp.ok) return null;
    const data = await resp.json();
    if (!data?.ok) return null;
    return {
      width: data.width,
      height: data.height,
      aspect: data.aspect,
      isPlanarPanel: !!data.is_planar_panel,
      isGraphicLogo: !!data.is_graphic_logo,
      roiBox: data.roi_box || null,
      depth: CadKernelBridge._decodeU16Buffer(data.depth_u16_b64),
      fgMask: CadKernelBridge._decodeBase64(data.fg_u8_b64),
      cleanedDataUrl: `data:image/png;base64,${data.cleaned_png_b64}`,
    };
  }
}
