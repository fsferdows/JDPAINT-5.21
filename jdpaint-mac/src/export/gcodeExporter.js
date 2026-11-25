// CNC G-Code Post-Processor and Exporter for JDPaint Mac
export class GcodeExporter {
  static generate(segments, toolConfig, dialect = 'grbl') {
    if (!segments || segments.length === 0) {
      return '( No toolpaths calculated. Please calculate toolpaths first. )';
    }

    const lines = [];
    const dateStr = new Date().toISOString().replace('T', ' ').substring(0, 19);

    // Header
    lines.push(`( ================================================= )`);
    lines.push(`( JDPaint 5.21 Mac Edition - CNC Program )`);
    lines.push(`( Generated: ${dateStr} )`);
    lines.push(`( Dialect: ${dialect.toUpperCase()} )`);
    lines.push(`( Tool: ${toolConfig.type.toUpperCase()} D=${toolConfig.diameter}mm )`);
    lines.push(`( Stepover: ${toolConfig.stepover}mm | Safe Z: ${toolConfig.safeZ}mm )`);
    lines.push(`( Feedrate: ${toolConfig.feedrate} mm/min | Plunge: ${toolConfig.plunge} mm/min )`);
    lines.push(`( Spindle: ${toolConfig.spindle} RPM )`);
    lines.push(`( ================================================= )`);

    if (dialect === 'jingdiao') {
      // Beijing Jingdiao Engrave preamble
      lines.push('G90 G21 G17 G40 G49');
      lines.push(`M03 S${toolConfig.spindle}`);
      lines.push('G04 P2');
      lines.push(`G00 Z${toolConfig.safeZ.toFixed(3)}`);
    } else {
      // Standard ISO / GRBL / Mach3
      lines.push('G21 ; Metric millimeters');
      lines.push('G90 ; Absolute coordinates');
      lines.push('G17 ; XY Plane');
      lines.push('G54 ; Work Coordinate System');
      lines.push(`M03 S${toolConfig.spindle} ; Spindle ON Clockwise`);
      lines.push('G04 P1.5 ; Spindle dwell');
      lines.push(`G00 Z${toolConfig.safeZ.toFixed(3)} ; Move to safe Z`);
    }

    let lastF = 0;
    let lastZ = toolConfig.safeZ;

    for (const seg of segments) {
      const to = seg.to;
      const xStr = to.x.toFixed(3);
      const yStr = to.y.toFixed(3);
      const zStr = to.z.toFixed(3);

      if (seg.type === 'rapid') {
        lines.push(`G00 X${xStr} Y${yStr} Z${zStr}`);
      } else if (seg.type === 'plunge') {
        if (lastF !== toolConfig.plunge) {
          lines.push(`G01 Z${zStr} F${toolConfig.plunge}`);
          lastF = toolConfig.plunge;
        } else {
          lines.push(`G01 Z${zStr}`);
        }
      } else {
        // Cut
        if (lastF !== toolConfig.feedrate) {
          lines.push(`G01 X${xStr} Y${yStr} Z${zStr} F${toolConfig.feedrate}`);
          lastF = toolConfig.feedrate;
        } else {
          lines.push(`G01 X${xStr} Y${yStr} Z${zStr}`);
        }
      }
      lastZ = to.z;
    }

    // Footer
    lines.push(`( --- End of Toolpaths --- )`);
    lines.push(`G00 Z${(toolConfig.safeZ + 5).toFixed(3)} ; Retract to clearance`);
    lines.push(`M05 ; Spindle OFF`);
    lines.push(`G00 X0.000 Y0.000 ; Return home X0 Y0`);
    lines.push(`M30 ; Program End`);

    return lines.join('\n');
  }

  static download(filename, textContent) {
    const blob = new Blob([textContent], { type: 'text/plain;charset=utf-8' });
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
