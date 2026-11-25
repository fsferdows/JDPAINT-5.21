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
