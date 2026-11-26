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
}
