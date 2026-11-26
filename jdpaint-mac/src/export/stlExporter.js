// Binary STL and 3D Mesh Exporter for JDPaint Mac
export class StlExporter {
  static exportReliefToBinarySTL(reliefEngine) {
    const res = reliefEngine.resolution;
    const w = reliefEngine.width;
    const h = reliefEngine.height;
    const dx = w / (res - 1);
    const dy = h / (res - 1);

    // Number of quad cells = (res - 1) * (res - 1)
    // 2 triangles per quad
    const numTriangles = (res - 1) * (res - 1) * 2;
    // Binary STL size = 80 header + 4 count + numTriangles * 50 bytes
    const totalBytes = 84 + numTriangles * 50;
    const buffer = new ArrayBuffer(totalBytes);
    const view = new DataView(buffer);

    // Write 80-byte header
    const headerStr = 'JDPaint 5.21 Mac Edition Binary STL 3D Relief Mesh';
    for (let i = 0; i < 80; i++) {
      view.setUint8(i, i < headerStr.length ? headerStr.charCodeAt(i) : 0);
    }

    // Write triangle count (Uint32 Little Endian)
    view.setUint32(80, numTriangles, true);

    let offset = 84;

    for (let iy = 0; iy < res - 1; iy++) {
      for (let ix = 0; ix < res - 1; ix++) {
        const x0 = ix * dx;
        const x1 = (ix + 1) * dx;
        const cadY0 = iy * dy;
        const cadY1 = (iy + 1) * dy;
        const yTop = h - cadY0; // Upright CNC Y+ axis!
        const yBot = h - cadY1;

        const z00 = reliefEngine.getZAt(x0, cadY0);
        const z10 = reliefEngine.getZAt(x1, cadY0);
        const z01 = reliefEngine.getZAt(x0, cadY1);
        const z11 = reliefEngine.getZAt(x1, cadY1);

        // Triangle 1 CCW (+Z outward normal): (x0, yTop), (x0, yBot), (x1, yTop)
        offset = StlExporter.writeTriangle(
          view,
          offset,
          x0, yTop, z00,
          x0, yBot, z01,
          x1, yTop, z10
        );

        // Triangle 2 CCW (+Z outward normal): (x1, yTop), (x0, yBot), (x1, yBot)
        offset = StlExporter.writeTriangle(
          view,
          offset,
          x1, yTop, z10,
          x0, yBot, z01,
          x1, yBot, z11
        );
      }
    }

    return new Blob([buffer], { type: 'application/octet-stream' });
  }

  static writeTriangle(view, offset, x1, y1, z1, x2, y2, z2, x3, y3, z3) {
    // Normal vector via cross product
    const ax = x2 - x1, ay = y2 - y1, az = z2 - z1;
    const bx = x3 - x1, by = y3 - y1, bz = z3 - z1;
    let nx = ay * bz - az * by;
    let ny = az * bx - ax * bz;
    let nz = ax * by - ay * bx;
    const len = Math.hypot(nx, ny, nz) || 1;
    nx /= len;
    ny /= len;
    nz /= len;

    // Normal
    view.setFloat32(offset, nx, true);
    view.setFloat32(offset + 4, ny, true);
    view.setFloat32(offset + 8, nz, true);

    // Vertex 1
    view.setFloat32(offset + 12, x1, true);
    view.setFloat32(offset + 16, y1, true);
    view.setFloat32(offset + 20, z1, true);

    // Vertex 2
    view.setFloat32(offset + 24, x2, true);
    view.setFloat32(offset + 28, y2, true);
    view.setFloat32(offset + 32, z2, true);

    // Vertex 3
    view.setFloat32(offset + 36, x3, true);
    view.setFloat32(offset + 40, y3, true);
    view.setFloat32(offset + 44, z3, true);

    // Attribute byte count
    view.setUint16(offset + 48, 0, true);

    return offset + 50;
  }

  static exportToOBJ(reliefEngine) {
    const res = reliefEngine.resolution;
    const w = reliefEngine.width;
    const h = reliefEngine.height;
    const dx = w / (res - 1);
    const dy = h / (res - 1);

    const lines = [];
    lines.push('# JDPaint 5.21 Mac Edition Wavefront OBJ Exporter');
    lines.push(`# Dimensions: ${w}x${h}x${reliefEngine.maxDepth}mm`);

    // Write vertices (Upright CNC Y+ axis)
    for (let iy = 0; iy < res; iy++) {
      for (let ix = 0; ix < res; ix++) {
        const x = (ix * dx).toFixed(3);
        const cadY = iy * dy;
        const y = (h - cadY).toFixed(3);
        const z = reliefEngine.getZAt(ix * dx, cadY).toFixed(3);
        lines.push(`v ${x} ${y} ${z}`);
      }
    }

    // Write faces (1-based indices, CCW outward +Z)
    for (let iy = 0; iy < res - 1; iy++) {
      for (let ix = 0; ix < res - 1; ix++) {
        const v00 = iy * res + ix + 1;
        const v10 = iy * res + ix + 2;
        const v01 = (iy + 1) * res + ix + 1;
        const v11 = (iy + 1) * res + ix + 2;

        lines.push(`f ${v00} ${v01} ${v10}`);
        lines.push(`f ${v10} ${v01} ${v11}`);
      }
    }

    return lines.join('\n');
  }

  /**
   * Export 2D CAD vector geometry to standard ASCII AutoCAD R12 DXF with per-layer separation (metric mm, upright CNC Y+ axis)
   */
  static exportToDXF(cadCanvas) {
    const h = cadCanvas.workpieceHeight || 100;
    const items = typeof cadCanvas.getPolylinesForCAMWithMeta === 'function'
      ? cadCanvas.getPolylinesForCAMWithMeta()
      : cadCanvas.getPolylinesForCAM().map((points) => ({ points, layerName: '0' }));

    const sanitizeLayer = (name) => String(name || '0').replace(/[^A-Za-z0-9_-]/g, '_').replace(/^_+|_+$/g, '') || '0';
    const layerNames = [];
    const seen = new Set();
    for (const item of items) {
      const ln = sanitizeLayer(item.layerName);
      if (!seen.has(ln)) { seen.add(ln); layerNames.push(ln); }
    }
    if (layerNames.length === 0) layerNames.push('0');

    const lines = [
      '0', 'SECTION',
      '2', 'HEADER',
      '9', '$INSUNITS', '70', '4', // 4 = Millimeters
      '0', 'ENDSEC',
      '0', 'SECTION',
      '2', 'TABLES',
      '0', 'TABLE',
      '2', 'LAYER',
      '70', String(layerNames.length),
    ];

    layerNames.forEach((ln, idx) => {
      const aciColor = String((idx % 7) + 1);
      lines.push('0', 'LAYER', '2', ln, '70', '0', '62', aciColor, '6', 'CONTINUOUS');
    });

    lines.push('0', 'ENDTAB', '0', 'ENDSEC', '0', 'SECTION', '2', 'ENTITIES');

    for (const item of items) {
      const poly = item.points;
      const layer = sanitizeLayer(item.layerName);
      if (!poly || poly.length < 2) continue;
      if (poly.length === 2) {
        lines.push(
          '0', 'LINE',
          '8', layer,
          '10', poly[0].x.toFixed(4), '20', (h - poly[0].y).toFixed(4), '30', '0.0',
          '11', poly[1].x.toFixed(4), '21', (h - poly[1].y).toFixed(4), '31', '0.0'
        );
      } else {
        const isClosed = Math.hypot(poly[0].x - poly[poly.length - 1].x, poly[0].y - poly[poly.length - 1].y) < 0.05;
        const pts = isClosed ? poly.slice(0, -1) : poly;
        lines.push('0', 'POLYLINE', '8', layer, '66', '1', '70', isClosed ? '1' : '0');
        for (const p of pts) {
          lines.push(
            '0', 'VERTEX', '8', layer,
            '10', p.x.toFixed(4),
            '20', (h - p.y).toFixed(4),
            '30', '0.0'
          );
        }
        lines.push('0', 'SEQEND');
      }
    }

    lines.push('0', 'ENDSEC', '0', 'EOF');
    return lines.join('\n');
  }

  /**
   * Export an arbitrary indexed 3D AssetBuffer ({ positions, indices }) directly to centered, upright Watertight Binary STL (mm).
   */
  static exportIndexedMeshToBinarySTL(meshData) {
    const pos = meshData.positions;
    const idx = meshData.indices;
    const vCount = pos.length / 3;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity;
    for (let i = 0; i < vCount; i++) {
      const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z;
    }
    const cx = isFinite(minX) ? (minX + maxX) * 0.5 : 0;
    const cy = isFinite(minY) ? (minY + maxY) * 0.5 : 0;
    const zShift = isFinite(minZ) ? -minZ : 0;

    const numTriangles = Math.floor(idx.length / 3);
    const buffer = new ArrayBuffer(84 + numTriangles * 50);
    const view = new DataView(buffer);
    const headerStr = 'JDPaint 3D Asset Studio Watertight Binary STL';
    for (let i = 0; i < 80; i++) view.setUint8(i, i < headerStr.length ? headerStr.charCodeAt(i) : 0);
    view.setUint32(80, numTriangles, true);

    let offset = 84;
    for (let t = 0; t < numTriangles; t++) {
      const i0 = idx[t * 3] * 3;
      const i1 = idx[t * 3 + 1] * 3;
      const i2 = idx[t * 3 + 2] * 3;
      offset = StlExporter.writeTriangle(
        view,
        offset,
        pos[i0] - cx, pos[i0 + 1] - cy, pos[i0 + 2] + zShift,
        pos[i1] - cx, pos[i1 + 1] - cy, pos[i1 + 2] + zShift,
        pos[i2] - cx, pos[i2 + 1] - cy, pos[i2 + 2] + zShift
      );
    }
    return new Blob([buffer], { type: 'application/octet-stream' });
  }

  /**
   * Export an indexed 3D AssetBuffer ({ positions, normals, uvs, indices }) to centered Wavefront OBJ (+Y up).
   */
  static exportIndexedMeshToOBJ(meshData) {
    const pos = meshData.positions;
    const norms = meshData.normals;
    const uvs = meshData.uvs;
    const idx = meshData.indices;
    const vCount = pos.length / 3;

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < vCount; i++) {
      const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    const cx = isFinite(minX) ? (minX + maxX) * 0.5 : 0;
    const cy = isFinite(minY) ? (minY + maxY) * 0.5 : 0;
    const cz = isFinite(minZ) ? (minZ + maxZ) * 0.5 : 0;

    const lines = ['# JDPaint 3D Asset Studio Wavefront OBJ Exporter (Centered)'];
    for (let i = 0; i < vCount; i++) {
      const k = i * 3;
      lines.push(`v ${(pos[k] - cx).toFixed(4)} ${(pos[k + 1] - cy).toFixed(4)} ${(pos[k + 2] - cz).toFixed(4)}`);
    }
    if (uvs) {
      for (let i = 0; i < vCount; i++) {
        const k = i * 2;
        lines.push(`vt ${uvs[k].toFixed(5)} ${uvs[k + 1].toFixed(5)}`);
      }
    }
    if (norms) {
      for (let i = 0; i < vCount; i++) {
        const k = i * 3;
        lines.push(`vn ${norms[k].toFixed(4)} ${norms[k + 1].toFixed(4)} ${norms[k + 2].toFixed(4)}`);
      }
    }
    const hasBoth = !!(uvs && norms);
    for (let t = 0; t < idx.length; t += 3) {
      const a = idx[t] + 1;
      const b = idx[t + 1] + 1;
      const c = idx[t + 2] + 1;
      lines.push(hasBoth ? `f ${a}/${a}/${a} ${b}/${b}/${b} ${c}/${c}/${c}` : `f ${a} ${b} ${c}`);
    }
    return lines.join('\n');
  }

  /**
   * Convert a regular ReliefEngine heightmap into indexed MeshData so Relief slabs also export to .GLB.
   */
  static reliefToMeshData(reliefEngine) {
    const res = reliefEngine.resolution;
    const w = reliefEngine.width;
    const h = reliefEngine.height;
    const dx = w / (res - 1);
    const dy = h / (res - 1);
    const vCount = res * res;
    const pos = new Float32Array(vCount * 3);
    const norms = new Float32Array(vCount * 3);
    const uvs = new Float32Array(vCount * 2);
    const indices = new Uint32Array((res - 1) * (res - 1) * 6);

    for (let iy = 0; iy < res; iy++) {
      for (let ix = 0; ix < res; ix++) {
        const vi = iy * res + ix;
        const x = ix * dx;
        const cadY = iy * dy;
        pos[vi * 3] = x;
        pos[vi * 3 + 1] = h - cadY;
        pos[vi * 3 + 2] = reliefEngine.getZAt(x, cadY);
        uvs[vi * 2] = ix / (res - 1);
        uvs[vi * 2 + 1] = 1.0 - iy / (res - 1);
      }
    }

    let p = 0;
    for (let iy = 0; iy < res - 1; iy++) {
      for (let ix = 0; ix < res - 1; ix++) {
        const v00 = iy * res + ix;
        const v10 = v00 + 1;
        const v01 = v00 + res;
        const v11 = v01 + 1;
        indices[p++] = v00; indices[p++] = v01; indices[p++] = v10;
        indices[p++] = v10; indices[p++] = v01; indices[p++] = v11;
      }
    }

    for (let i = 0; i < indices.length; i += 3) {
      const i0 = indices[i] * 3, i1 = indices[i + 1] * 3, i2 = indices[i + 2] * 3;
      const ax = pos[i1] - pos[i0], ay = pos[i1 + 1] - pos[i0 + 1], az = pos[i1 + 2] - pos[i0 + 2];
      const bx = pos[i2] - pos[i0], by = pos[i2 + 1] - pos[i0 + 1], bz = pos[i2 + 2] - pos[i0 + 2];
      const nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
      norms[i0] += nx; norms[i0 + 1] += ny; norms[i0 + 2] += nz;
      norms[i1] += nx; norms[i1 + 1] += ny; norms[i1 + 2] += nz;
      norms[i2] += nx; norms[i2 + 1] += ny; norms[i2 + 2] += nz;
    }
    for (let i = 0; i < vCount; i++) {
      const k = i * 3;
      const l = Math.hypot(norms[k], norms[k + 1], norms[k + 2]) || 1;
      norms[k] /= l; norms[k + 1] /= l; norms[k + 2] /= l;
    }
    return { positions: pos, normals: norms, uvs, indices };
  }

  /**
   * Export 3D mesh + Full PBR Textures (BaseColor, Normal, MetallicRoughness) into centered Binary glTF 2.0 (.glb).
   */
  static exportToGLB(meshData, options = {}) {
    const {
      baseColorPngBytes = null,
      normalPngBytes = null,
      metallicRoughnessPngBytes = null,
      roughness = 0.55,
      metalness = 0.08,
      normalScale = 0.85,
      centerAndNormalize = true,
      name = 'JDPaint3DAsset',
    } = options;

    const srcPos = meshData.positions;
    const srcNorms = meshData.normals;
    const vCount = srcPos.length / 3;
    const rawUvs = meshData.uvs;
    const uvs = new Float32Array(rawUvs.length);
    for (let i = 0; i < vCount; i++) {
      uvs[i * 2] = rawUvs[i * 2];
      uvs[i * 2 + 1] = 1.0 - rawUvs[i * 2 + 1]; // glTF 2.0 UV (0,0) is top-left of PNG
    }
    const idx = meshData.indices;

    let rawMinX = Infinity, rawMinY = Infinity, rawMinZ = Infinity;
    let rawMaxX = -Infinity, rawMaxY = -Infinity, rawMaxZ = -Infinity;
    for (let i = 0; i < vCount; i++) {
      const k = i * 3;
      const x = srcPos[k], y = srcPos[k + 1], z = srcPos[k + 2];
      if (x < rawMinX) rawMinX = x; if (x > rawMaxX) rawMaxX = x;
      if (y < rawMinY) rawMinY = y; if (y > rawMaxY) rawMaxY = y;
      if (z < rawMinZ) rawMinZ = z; if (z > rawMaxZ) rawMaxZ = z;
    }

    // Standard glTF 2.0 normalization: center at (0,0,0), fit max extent to 2.0 meters for Blender/QuickLook/Unity
    const cx = centerAndNormalize ? (rawMinX + rawMaxX) * 0.5 : 0;
    const cy = centerAndNormalize ? (rawMinY + rawMaxY) * 0.5 : 0;
    const cz = centerAndNormalize ? (rawMinZ + rawMaxZ) * 0.5 : 0;
    const maxExtent = Math.max(rawMaxX - rawMinX, rawMaxY - rawMinY, rawMaxZ - rawMinZ, 1e-6);
    const scale = centerAndNormalize ? (2.0 / maxExtent) : 1.0;

    const pos = new Float32Array(srcPos.length);
    const norms = new Float32Array(srcNorms.length);
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < vCount; i++) {
      const k = i * 3;
      const x = (srcPos[k] - cx) * scale;
      const y = (srcPos[k + 1] - cy) * scale;
      const z = (srcPos[k + 2] - cz) * scale;
      pos[k] = x; pos[k + 1] = y; pos[k + 2] = z;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;

      const nx = srcNorms[k], ny = srcNorms[k + 1], nz = srcNorms[k + 2];
      const nl = Math.hypot(nx, ny, nz) || 1;
      norms[k] = nx / nl; norms[k + 1] = ny / nl; norms[k + 2] = nz / nl;
    }

    const pad4 = (n) => (n + 3) & ~3;
    const chunks = [];
    let byteCursor = 0;
    const bufferViews = [];
    const appendView = (typedOrBytes, target) => {
      const u8 = typedOrBytes instanceof Uint8Array
        ? typedOrBytes
        : new Uint8Array(typedOrBytes.buffer, typedOrBytes.byteOffset, typedOrBytes.byteLength);
      const bv = { buffer: 0, byteOffset: byteCursor, byteLength: u8.byteLength };
      if (target) bv.target = target;
      const bvIdx = bufferViews.length;
      bufferViews.push(bv);
      chunks.push(u8);
      byteCursor += u8.byteLength;
      const rem = (4 - (byteCursor & 3)) & 3;
      if (rem > 0) {
        chunks.push(new Uint8Array(rem));
        byteCursor += rem;
      }
      return bvIdx;
    };

    const bvPos = appendView(pos, 34962);
    const bvNorm = appendView(norms, 34962);
    const bvUv = appendView(uvs, 34962);
    const bvIdx = appendView(idx, 34963);

    const hasOrmTex = !!(metallicRoughnessPngBytes && metallicRoughnessPngBytes.byteLength > 0);
    const materialDef = {
      name: `${name}_Mat`,
      pbrMetallicRoughness: {
        // When metallicRoughnessTexture (G=Roughness, B=Metallic) is supplied, it already encodes calibrated [0..1] channels!
        metallicFactor: hasOrmTex ? 1.0 : metalness,
        roughnessFactor: hasOrmTex ? 1.0 : roughness,
      },
      doubleSided: true,
    };

    const images = [];
    const textures = [];
    const samplers = [];
    const pushPngTexture = (pngBytes) => {
      if (!pngBytes || pngBytes.byteLength === 0) return -1;
      const bvTex = appendView(pngBytes, 0);
      const idxImg = images.length;
      if (samplers.length === 0) {
        samplers.push({ magFilter: 9729, minFilter: 9987, wrapS: 33071, wrapT: 33071 });
      }
      images.push({ bufferView: bvTex, mimeType: 'image/png' });
      textures.push({ sampler: 0, source: idxImg });
      return idxImg;
    };

    const baseTexIdx = pushPngTexture(baseColorPngBytes);
    if (baseTexIdx >= 0) {
      materialDef.pbrMetallicRoughness.baseColorTexture = { index: baseTexIdx };
    } else {
      materialDef.pbrMetallicRoughness.baseColorFactor = [0.85, 0.72, 0.56, 1.0];
    }

    const normTexIdx = pushPngTexture(normalPngBytes);
    if (normTexIdx >= 0) {
      materialDef.normalTexture = { index: normTexIdx, scale: Math.max(0.1, Math.min(2.0, normalScale)) };
    }

    const ormTexIdx = pushPngTexture(metallicRoughnessPngBytes);
    if (ormTexIdx >= 0) {
      materialDef.pbrMetallicRoughness.metallicRoughnessTexture = { index: ormTexIdx };
    }

    const gltf = {
      asset: { version: '2.0', generator: 'JDPaint 3D Asset Studio' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0, name }],
      meshes: [{
        name,
        primitives: [{
          attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 },
          indices: 3,
          material: 0,
        }],
      }],
      materials: [materialDef],
      accessors: [
        { bufferView: bvPos, componentType: 5126, count: vCount, type: 'VEC3', min: [minX, minY, minZ], max: [maxX, maxY, maxZ] },
        { bufferView: bvNorm, componentType: 5126, count: vCount, type: 'VEC3' },
        { bufferView: bvUv, componentType: 5126, count: vCount, type: 'VEC2' },
        { bufferView: bvIdx, componentType: 5125, count: idx.length, type: 'SCALAR' },
      ],
      bufferViews,
      buffers: [{ byteLength: byteCursor }],
    };
    if (images.length > 0) {
      gltf.samplers = samplers;
      gltf.images = images;
      gltf.textures = textures;
    }

    const jsonRaw = new TextEncoder().encode(JSON.stringify(gltf));
    const jsonPaddedLen = pad4(jsonRaw.byteLength);
    const totalLen = 12 + 8 + jsonPaddedLen + 8 + byteCursor;
    const out = new Uint8Array(totalLen);
    const dv = new DataView(out.buffer);

    // GLB Header: magic "glTF" (0x46546C67), version 2, total length
    dv.setUint32(0, 0x46546C67, true);
    dv.setUint32(4, 2, true);
    dv.setUint32(8, totalLen, true);

    // Chunk 0: JSON
    dv.setUint32(12, jsonPaddedLen, true);
    dv.setUint32(16, 0x4E4F534A, true);
    out.set(jsonRaw, 20);
    for (let p = 20 + jsonRaw.byteLength; p < 20 + jsonPaddedLen; p++) out[p] = 0x20;

    // Chunk 1: BIN
    const binHeaderPos = 20 + jsonPaddedLen;
    dv.setUint32(binHeaderPos, byteCursor, true);
    dv.setUint32(binHeaderPos + 4, 0x004E4942, true);
    let writePtr = binHeaderPos + 8;
    for (const ch of chunks) {
      out.set(ch, writePtr);
      writePtr += ch.byteLength;
    }

    return new Blob([out], { type: 'model/gltf-binary' });
  }

  static downloadBlob(filename, blob) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}
