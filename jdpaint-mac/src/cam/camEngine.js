// CNC CAM & Toolpath Calculation Engine for JDPaint Mac
// Engineered with True 3D Surface Normal Ball-Nose Cutter Compensation & G-Code Export

export class CamEngine {
  constructor() {
    this.tool = {
      type: 'ball', // 'ball', 'flat', 'vcarve'
      diameter: 3.175,
      stepover: 0.3,
      safeZ: 5.0,
      feedrate: 1500,
      plunge: 500,
      spindle: 18000,
    };

    this.stats = {
      points: 0,
      cutDistance: 0,
      rapidDistance: 0,
      estimatedTimeSec: 0,
    };
  }

  setToolConfig(config) {
    this.tool = { ...this.tool, ...config };
  }

  // 3D Relief Surface Finishing Toolpath (Raster Zigzag with Surface Normal Compensation)
  calculate3DReliefToolpath(reliefEngine) {
    const segments = [];
    const w = reliefEngine.width;
    const h = reliefEngine.height;
    const stepover = Math.max(0.05, this.tool.stepover);
    const safeZ = this.tool.safeZ;

    const numScanlines = Math.floor(h / stepover);
    const xStep = Math.max(0.15, stepover * 0.75); // Sample density along carving line

    let currentPos = { x: 0, y: 0, z: safeZ };
    let totalCutDist = 0;
    let totalRapidDist = 0;
    let pointCount = 0;

    // Ball-nose radius for normal compensation
    const R = this.tool.type === 'ball' ? this.tool.diameter / 2 : 0;

    for (let i = 0; i <= numScanlines; i++) {
      const cncY = Math.min(h, i * stepover);
      const cadY = h - cncY; // Bottom-left CNC origin (Y=0 bottom, Y=h top -> upright machining!)
      const isForward = i % 2 === 0;

      const startX = isForward ? 0 : w;
      const endX = isForward ? w : 0;
      const stepDirection = isForward ? xStep : -xStep;

      // Start of scanline with normal compensation
      const startNorm = reliefEngine.getNormalAt(startX, cadY);
      const startRawZ = reliefEngine.getZAt(startX, cadY);
      const startZ = R > 0 ? startRawZ + R * (1.0 - startNorm.z) : startRawZ;

      // Rapid to above start position
      if (i === 0) {
        segments.push({
          type: 'rapid',
          from: { ...currentPos },
          to: { x: startX, y: cncY, z: safeZ },
        });
        totalRapidDist += Math.hypot(startX - currentPos.x, cncY - currentPos.y);
      } else {
        // Step over along stock edge at cutting plane
        segments.push({
          type: 'cut',
          from: { ...currentPos },
          to: { x: startX, y: cncY, z: startZ },
        });
        totalCutDist += Math.hypot(startX - currentPos.x, cncY - currentPos.y, startZ - currentPos.z);
        currentPos = { x: startX, y: cncY, z: startZ };
        pointCount++;
      }

      // Plunge down on first scanline
      if (i === 0) {
        segments.push({
          type: 'plunge',
          from: { x: startX, y: cncY, z: safeZ },
          to: { x: startX, y: cncY, z: startZ },
        });
        totalRapidDist += Math.abs(safeZ - startZ);
        currentPos = { x: startX, y: cncY, z: startZ };
        pointCount++;
      }

      // Carve along the line
      let x = startX;
      while ((isForward && x <= endX) || (!isForward && x >= endX)) {
        const rawZ = reliefEngine.getZAt(x, cadY);
        let z = rawZ;

        // Apply true 3D normal contact compensation for ball-nose cutter
        if (R > 0) {
          const norm = reliefEngine.getNormalAt(x, cadY);
          z = rawZ + R * (1.0 - norm.z);
        }

        const dist = Math.hypot(x - currentPos.x, cncY - currentPos.y, z - currentPos.z);
        if (dist > 0.04) {
          segments.push({
            type: 'cut',
            from: { ...currentPos },
            to: { x, y: cncY, z },
          });
          totalCutDist += dist;
          currentPos = { x, y: cncY, z };
          pointCount++;
        }

        x += stepDirection;
      }

      // If at last line, retract to safeZ
      if (i === numScanlines) {
        segments.push({
          type: 'rapid',
          from: { ...currentPos },
          to: { x: currentPos.x, y: currentPos.y, z: safeZ },
        });
        totalRapidDist += Math.abs(safeZ - currentPos.z);
        currentPos.z = safeZ;
      }
    }

    // Time estimate: cut time + rapid time + plunge time
    const cutTimeMin = totalCutDist / this.tool.feedrate;
    const rapidTimeMin = totalRapidDist / 2500; // Assume 2500 mm/min rapid
    const totalTimeSec = (cutTimeMin + rapidTimeMin) * 60;

    this.stats = {
      points: pointCount,
      cutDistance: Math.round(totalCutDist),
      rapidDistance: Math.round(totalRapidDist),
      estimatedTimeSec: Math.round(totalTimeSec),
    };

    return segments;
  }

  // 2D Contour Profile Cutting (converted to upright CNC Y+ machine coordinates)
  calculate2DProfileToolpath(cadCanvas, targetDepthMm = 3.0, stepdownMm = 1.0) {
    const polylines = cadCanvas.getPolylinesForCAM();
    const h = cadCanvas.workpieceHeight || 100;
    const segments = [];
    const safeZ = this.tool.safeZ;

    let totalCutDist = 0;
    let totalRapidDist = 0;
    let pointCount = 0;

    const passes = Math.ceil(targetDepthMm / stepdownMm);
    let lastCncPos = { x: 0, y: 0, z: safeZ };

    for (const poly of polylines) {
      if (poly.length < 2) continue;

      const p0 = { x: poly[0].x, y: h - poly[0].y };

      // Rapid from previous position to start
      segments.push({
        type: 'rapid',
        from: { ...lastCncPos, z: safeZ },
        to: { x: p0.x, y: p0.y, z: safeZ },
      });
      totalRapidDist += Math.hypot(p0.x - lastCncPos.x, p0.y - lastCncPos.y);

      for (let p = 1; p <= passes; p++) {
        const currentZ = -Math.min(targetDepthMm, p * stepdownMm);

        // Plunge to pass depth
        segments.push({
          type: 'plunge',
          from: { x: p0.x, y: p0.y, z: p === 1 ? safeZ : currentZ + stepdownMm },
          to: { x: p0.x, y: p0.y, z: currentZ },
        });

        let lastPt = p0;
        for (let i = 1; i < poly.length; i++) {
          const pt = { x: poly[i].x, y: h - poly[i].y };
          segments.push({
            type: 'cut',
            from: { x: lastPt.x, y: lastPt.y, z: currentZ },
            to: { x: pt.x, y: pt.y, z: currentZ },
          });
          totalCutDist += Math.hypot(pt.x - lastPt.x, pt.y - lastPt.y);
          lastPt = pt;
          pointCount++;
        }
      }

      // Retract
      const lastRaw = poly[poly.length - 1];
      const lastPtCnc = { x: lastRaw.x, y: h - lastRaw.y };
      segments.push({
        type: 'rapid',
        from: { x: lastPtCnc.x, y: lastPtCnc.y, z: -targetDepthMm },
        to: { x: lastPtCnc.x, y: lastPtCnc.y, z: safeZ },
      });
      totalRapidDist += safeZ + targetDepthMm;
      lastCncPos = { x: lastPtCnc.x, y: lastPtCnc.y, z: safeZ };
    }

    const cutTimeMin = totalCutDist / this.tool.feedrate;
    const rapidTimeMin = totalRapidDist / 2500;
    const totalTimeSec = (cutTimeMin + rapidTimeMin) * 60;

    this.stats = {
      points: pointCount,
      cutDistance: Math.round(totalCutDist),
      rapidDistance: Math.round(totalRapidDist),
      estimatedTimeSec: Math.round(totalTimeSec),
    };

    return segments;
  }
}
