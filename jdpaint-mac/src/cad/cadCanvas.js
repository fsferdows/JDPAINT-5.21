// 2D CAD Vector Drafting, Editing & Auto-Tracing Engine for JDPaint Mac
// Features: rAF-decoupled rendering, 8-handle Resize & Rotate, Freehand Brush, Pen Bezier,
// Node Edit, Knife/Scissors Split, Ellipse, Polygon/Star, 3-Point Arc, Polygon Offset, and ResizeObserver
import { ImageVectorizer } from './imageVectorizer.js';

export class CadCanvas {
  constructor(canvasElement, onCoordinateChange, callbacks = {}) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.onCoordinateChange = onCoordinateChange;
    this.onEntitiesChange = callbacks.onEntitiesChange || null;
    this.onSelectionChange = callbacks.onSelectionChange || null;
    this.onDraftChange = callbacks.onDraftChange || null;
    this._entitySeq = 1;

    // View transform
    this.scale = 2.5; // pixels per mm
    this.panX = 0;
    this.panY = 0;

    // Grid & Snapping (gridSnap defaults OFF so Pen/Freehand/Nodes never jump in coarse blocks!)
    this.gridVisible = true;
    this.showFills = true; // Subtle translucent / Illustrator fill on closed shapes
    this.showColorFills = true; // Vivid vector color fills when available
    this.gridSnap = false;
    this.objectSnap = true; // Magnetic OSNAP (Endpoint, Midpoint, Center, Close Path)
    this.snapGridSize = 1;  // 1 mm fine snap when enabled
    this.visualGridStep = 5; // 5 mm minor visual grid lines
    this.activeSnapTarget = null;

    // Background Reference Image for Tracing
    this.refImage = null;
    this.refImageOpacity = 0.5;
    this.showRefImage = true;
    this.refImageBounds = { x: 0, y: 0, w: 100, h: 100 };

    // State & Entities
    this.entities = [];
    this.selectedEntities = [];
    this.hoveredEntity = null;
    this.hoveredNodeIndex = -1;
    this.hoveredHoleIndex = -1;
    this.selectedNodeIndex = -1;
    this.selectedHoleIndex = -1;
    this.activeBezierHandle = null; // { holeIndex, nodeIndex, side: 'hIn'|'hOut' }
    // Tools: 'select', 'node', 'pen', 'freehand', 'polyline', 'line', 'rect', 'circle', 'ellipse', 'polygon', 'arc', 'knife', 'eraser', 'text'
    this.activeTool = 'select';

    // Polygon / Star parameters
    this.polygonSides = 6;
    this.polygonStarMode = false;
    this.reshapeBrushRadius = 6; // mm for Path Reshape Brush ('reshape')
    this.symmetryMode = 'none'; // 'none' | 'mirrorX' | 'mirrorY' | 'quad'

    // Interaction states
    this.isDragging = false;
    this.isPanning = false;
    this.isSpaceDown = false;
    this.isMouseDown = false;
    this.isBoxSelecting = false;
    this.activeTransformHandle = null; // 'nw','n','ne','e','se','s','sw','w','rot'
    this.transformOriginBounds = null;
    this.transformSnapshot = null;
    this.boxSelectStart = { x: 0, y: 0 };
    this.boxSelectEnd = { x: 0, y: 0 };

    this.dragStart = { x: 0, y: 0 };
    this.panStart = { x: 0, y: 0 };
    this.currentDraft = null;
    this.currentMousePos = { x: 0, y: 0, sx: 0, sy: 0 };

    // Undo / Redo History Stack
    this.undoStack = [];
    this.redoStack = [];

    // Render loop decoupling (rAF)
    this.renderRequested = false;

    // Workpiece bounds (mm)
    this.workpieceWidth = 100;
    this.workpieceHeight = 100;
    this._bboxCache = new WeakMap();

    this.initEvents();
    this.resize();
    this.centerView();
  }

  get selectedEntity() {
    return this.selectedEntities.length > 0 ? this.selectedEntities[0] : null;
  }

  set selectedEntity(ent) {
    if (ent) {
      this.selectedEntities = [ent];
    } else {
      this.selectedEntities = [];
    }
    this.notifySelectionChange();
  }

  notifyEntitiesChange() {
    this._bboxCache = new WeakMap();
    if (this.onEntitiesChange) this.onEntitiesChange(this.entities);
  }

  notifySelectionChange() {
    if (this.onSelectionChange) this.onSelectionChange(this.selectedEntities);
  }

  notifyDraftChange() {
    if (this.onDraftChange) this.onDraftChange(this.currentDraft);
  }

  ensureEntityMetadata(ent) {
    if (!ent) return ent;
    if (!ent.id) {
      ent.id = `layer_${Date.now().toString(36)}_${this._entitySeq++}`;
    }
    if (ent.visible === undefined) ent.visible = true;
    if (ent.locked === undefined) ent.locked = false;
    if (!ent.source) ent.source = 'user';
    if (!ent.name) {
      const labels = {
        curve: ent.shapeKind ? ent.shapeKind : 'Pen Curve',
        polyline: ent.shapeKind ? ent.shapeKind : 'Polyline',
        line: 'Line',
        rect: 'Rectangle',
        circle: 'Circle',
        text: `Text (${(ent.text || '').slice(0, 10)})`,
      };
      ent.name = `${labels[ent.type] || 'Vector'} #${this._entitySeq - 1}`;
    }
    if (!ent.relief) {
      const isOpen = (ent.type === 'line') || ((ent.type === 'curve' || ent.type === 'polyline') && !ent.isClosed);
      ent.relief = {
        enabled: true,
        shape: 'dome',
        heightMm: 4.0,
        radiusMm: isOpen ? 3.5 : 10.0,
        mode: 'max',
        openAsStroke: isOpen,
      };
    }
    return ent;
  }

  requestRender() {
    if (!this.renderRequested) {
      this.renderRequested = true;
      requestAnimationFrame(() => {
        this.render();
        this.renderRequested = false;
      });
    }
  }

  resize() {
    if (!this.canvas.parentElement) return;
    const rect = this.canvas.parentElement.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(rect.width * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height}px`;
    this.requestRender();
  }

  centerView() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.width / dpr;
    const h = this.canvas.height / dpr;
    if (w > 0 && h > 0) {
      const fitScale = Math.min((w * 0.82) / this.workpieceWidth, (h * 0.82) / this.workpieceHeight);
      if (fitScale > 0.4 && fitScale < 20) this.scale = fitScale;
    }
    this.panX = (w - this.workpieceWidth * this.scale) / 2;
    this.panY = (h - this.workpieceHeight * this.scale) / 2;
    this.requestRender();
  }

  // Undo / Redo History
  saveState() {
    const snapshot = JSON.stringify(this.entities);
    if (this.undoStack.length === 0 || this.undoStack[this.undoStack.length - 1] !== snapshot) {
      this.undoStack.push(snapshot);
      if (this.undoStack.length > 50) this.undoStack.shift();
      this.redoStack = [];
    }
  }

  undo() {
    if (this.undoStack.length > 0) {
      this.redoStack.push(JSON.stringify(this.entities));
      const state = this.undoStack.pop();
      this.entities = JSON.parse(state).map((e) => this.ensureEntityMetadata(e));
      this.selectedEntities = [];
      this.requestRender();
      this.notifyEntitiesChange();
      this.notifySelectionChange();
    }
  }

  redo() {
    if (this.redoStack.length > 0) {
      this.undoStack.push(JSON.stringify(this.entities));
      const state = this.redoStack.pop();
      this.entities = JSON.parse(state).map((e) => this.ensureEntityMetadata(e));
      this.selectedEntities = [];
      this.requestRender();
      this.notifyEntitiesChange();
      this.notifySelectionChange();
    }
  }

  // Reference Image
  setReferenceImage(img) {
    this.refImage = img;
    this.refImageBounds = ImageVectorizer.getImageFitBounds(img, this.workpieceWidth, this.workpieceHeight);
    this.showRefImage = true;
    this.requestRender();
  }

  clearReferenceImage() {
    this.refImage = null;
    this.requestRender();
  }

  // Auto-trace image into SEPARATE editable 2D CAD vector layers
  autoTraceReferenceImage(options = {}) {
    if (!this.refImage) return 0;
    const { autoEnable3D = true, ...traceOpts } = options;
    this.saveState();
    const vectors = ImageVectorizer.traceImageToVectors(
      this.refImage,
      this.workpieceWidth,
      this.workpieceHeight,
      traceOpts
    ).map((e) => {
      const ent = this.ensureEntityMetadata(e);
      if (!autoEnable3D) ent.relief.enabled = false;
      return ent;
    });

    if (vectors.length > 0) {
      this.entities.push(...vectors);
      // Do NOT lump all traced shapes into one multi-selection! Keep selection clear or single so user can immediately edit individual layers.
      this.selectedEntities = [];
      this.requestRender();
      this.notifyEntitiesChange();
      this.notifySelectionChange();
    }
    return vectors.length;
  }

  // Object Snapping (OSNAP: Close Path, Endpoints, Midpoints, Centers, AND On-Path)
  calculateSnap(rawX, rawY) {
    if (!this.objectSnap) return null;
    if (this.activeTool === 'freehand' || this.activeTool === 'eraser' || this.activeTool === 'reshape') return null;

    const snapRadius = 9 / this.scale;
    let bestTarget = null;
    let minDist = snapRadius;

    // Close Path snap for Pen / Polyline
    if (this.currentDraft && (this.currentDraft.type === 'curve' || this.currentDraft.type === 'polyline')) {
      if (this.currentDraft.points && this.currentDraft.points.length >= 2) {
        const p0 = this.currentDraft.points[0];
        if (Math.hypot(rawX - p0.x, rawY - p0.y) < snapRadius * 1.6) {
          return { x: p0.x, y: p0.y, type: 'Close Path' };
        }
      }
    }

    // Workpiece origin & center
    const origin = { x: 0, y: 0, type: 'Origin' };
    const center = { x: this.workpieceWidth / 2, y: this.workpieceHeight / 2, type: 'Center' };
    for (const p of [origin, center]) {
      const d = Math.hypot(rawX - p.x, rawY - p.y);
      if (d < minDist) {
        minDist = d;
        bestTarget = p;
      }
    }

    // Pass 1: Priority vertices (Endpoints, Midpoints, Centers) with Fast BBox rejection
    for (const ent of this.entities) {
      if (ent.visible === false) continue;
      const eb = this.getEntityBounds(ent);
      if (eb && (rawX < eb.minX - snapRadius || rawX > eb.maxX + snapRadius || rawY < eb.minY - snapRadius || rawY > eb.maxY + snapRadius)) {
        continue;
      }

      if (ent.type === 'line') {
        const p1 = { x: ent.x1, y: ent.y1, type: 'Endpoint', ent };
        const p2 = { x: ent.x2, y: ent.y2, type: 'Endpoint', ent };
        const mid = { x: (ent.x1 + ent.x2) / 2, y: (ent.y1 + ent.y2) / 2, type: 'Midpoint', ent };
        for (const p of [p1, p2, mid]) {
          const d = Math.hypot(rawX - p.x, rawY - p.y);
          if (d < minDist) { minDist = d; bestTarget = p; }
        }
      } else if (ent.type === 'circle') {
        const c = { x: ent.cx, y: ent.cy, type: 'Center', ent };
        const d = Math.hypot(rawX - c.x, rawY - c.y);
        if (d < minDist) { minDist = d; bestTarget = c; }
      } else if ((ent.type === 'polyline' || ent.type === 'curve') && ent.points) {
        for (let i = 0; i < ent.points.length; i++) {
          const pt = ent.points[i];
          const d = Math.hypot(rawX - pt.x, rawY - pt.y);
          if (d < minDist) { minDist = d; bestTarget = { x: pt.x, y: pt.y, type: 'Endpoint', ent }; }
        }
      }
    }

    if (bestTarget) return bestTarget;

    // Pass 2: Closest point ON any traced curve/line/circle edge with Fast BBox rejection
    let pathMin = snapRadius * 0.9;
    for (const ent of this.entities) {
      if (ent.visible === false) continue;
      const eb = this.getEntityBounds(ent);
      if (eb && (rawX < eb.minX - pathMin || rawX > eb.maxX + pathMin || rawY < eb.minY - pathMin || rawY > eb.maxY + pathMin)) {
        continue;
      }

      if (ent.type === 'line') {
        const proj = CadCanvas.closestPointOnSegment({ x: rawX, y: rawY }, { x: ent.x1, y: ent.y1 }, { x: ent.x2, y: ent.y2 });
        if (proj.dist < pathMin) { pathMin = proj.dist; bestTarget = { x: proj.x, y: proj.y, type: 'On Path', ent }; }
      } else if (ent.type === 'circle') {
        const d = Math.hypot(rawX - ent.cx, rawY - ent.cy);
        if (Math.abs(d - ent.r) < pathMin && d > 0.01) {
          pathMin = Math.abs(d - ent.r);
          bestTarget = {
            x: ent.cx + ((rawX - ent.cx) / d) * ent.r,
            y: ent.cy + ((rawY - ent.cy) / d) * ent.r,
            type: 'On Path',
            ent,
          };
        }
      } else if ((ent.type === 'polyline' || ent.type === 'curve') && ent.points?.length > 1) {
        const pts = ent.points;
        const segCount = ent.isClosed ? pts.length : pts.length - 1;
        for (let i = 0; i < segCount; i++) {
          const proj = CadCanvas.closestPointOnSegment({ x: rawX, y: rawY }, pts[i], pts[(i + 1) % pts.length]);
          if (proj.dist < pathMin) { pathMin = proj.dist; bestTarget = { x: proj.x, y: proj.y, type: 'On Path', ent }; }
        }
      }
    }

    return bestTarget;
  }

  static closestPointOnSegment(p, a, b) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const lenSq = dx * dx + dy * dy;
    if (lenSq === 0) return { x: a.x, y: a.y, dist: Math.hypot(p.x - a.x, p.y - a.y) };
    const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / lenSq));
    const x = a.x + t * dx;
    const y = a.y + t * dy;
    return { x, y, dist: Math.hypot(p.x - x, p.y - y) };
  }

  screenToWorld(sx, sy, bypassSnap = false) {
    const wx = (sx - this.panX) / this.scale;
    const wy = (sy - this.panY) / this.scale;

    if (!bypassSnap) {
      const snap = this.calculateSnap(wx, wy);
      if (snap) {
        this.activeSnapTarget = snap;
        return { x: snap.x, y: snap.y };
      }
    }
    this.activeSnapTarget = null;

    if (!bypassSnap && this.gridSnap && this.activeTool !== 'node' && this.activeTool !== 'eraser' && this.activeTool !== 'freehand' && this.activeTool !== 'knife') {
      return {
        x: Math.round(wx / this.snapGridSize) * this.snapGridSize,
        y: Math.round(wy / this.snapGridSize) * this.snapGridSize,
      };
    }
    return { x: wx, y: wy };
  }

  worldToScreen(wx, wy) {
    return {
      x: wx * this.scale + this.panX,
      y: wy * this.scale + this.panY,
    };
  }

  finishCurrentDraft(shouldClose = false) {
    if (!this.currentDraft) return;
    this.saveState();
    const draft = this.currentDraft;

    // Special conversion for Freehand sketch -> simplified + corner-preserving smoothed curve (with Auto-Splice check!)
    if (draft.type === 'freehand') {
      if (draft.points.length >= 2) {
        let simplified = ImageVectorizer.simplifyRDP(draft.points, 0.28);
        if (simplified.length >= 2) {
          const first = simplified[0];
          const last = simplified[simplified.length - 1];
          const autoClose = simplified.length >= 4 && Math.hypot(first.x - last.x, first.y - last.y) < 3.5;

          if (!autoClose && this.trySpliceIntoExistingCurve(simplified)) {
            this.currentDraft = null;
            this.requestRender();
            this.notifyEntitiesChange();
            this.notifyDraftChange();
            return;
          }

          if (autoClose) simplified[simplified.length - 1] = { x: first.x, y: first.y };
          if (simplified.length >= 5) {
            simplified = ImageVectorizer.cornerPreservingSmooth(simplified, autoClose);
            if (autoClose && simplified.length >= 3) simplified[simplified.length - 1] = { ...simplified[0] };
          }

          const curveEnt = this.ensureEntityMetadata({
            type: 'curve',
            shapeKind: 'Freehand Brush',
            points: simplified,
            isClosed: autoClose,
            color: '#38bdf8',
            width: 1.5,
          });
          this.entities.push(curveEnt);
          this.applySymmetryClones(curveEnt);
          this.selectedEntity = curveEnt;
        }
      }
      this.currentDraft = null;
      this.requestRender();
      this.notifyEntitiesChange();
      this.notifyDraftChange();
      return;
    }

    // Special conversion for 3-point Arc -> polyline/curve
    if (draft.type === 'arc_draft') {
      if (draft.pts && draft.pts.length === 3) {
        const curvePts = CadCanvas.sampleQuadraticArc(draft.pts[0], draft.pts[1], draft.pts[2], 24);
        const arcEnt = this.ensureEntityMetadata({
          type: 'curve',
          shapeKind: 'Arc',
          points: curvePts,
          isClosed: false,
          color: '#38bdf8',
          width: 1.5,
        });
        this.entities.push(arcEnt);
        this.applySymmetryClones(arcEnt);
        this.selectedEntity = arcEnt;
      }
      this.currentDraft = null;
      this.requestRender();
      this.notifyEntitiesChange();
      this.notifyDraftChange();
      return;
    }

    if (draft.type === 'curve' || draft.type === 'polyline') {
      while (draft.points.length > 2) {
        const a = draft.points[draft.points.length - 1];
        const b = draft.points[draft.points.length - 2];
        if (Math.hypot(a.x - b.x, a.y - b.y) < 0.25) draft.points.pop();
        else break;
      }
      if (draft.points.length >= 2) {
        // Check Smart Auto-Splice into existing design outline BEFORE saving as detached new layer!
        if (!shouldClose && this.trySpliceIntoExistingCurve(draft.points, draft._targetEnt)) {
          this.currentDraft = null;
          this.requestRender();
          this.notifyEntitiesChange();
          this.notifyDraftChange();
          return;
        }

        if (shouldClose && draft.points.length >= 3) {
          const first = draft.points[0];
          const last = draft.points[draft.points.length - 1];
          if (Math.hypot(first.x - last.x, first.y - last.y) > 0.05) {
            draft.points.push({ x: first.x, y: first.y });
          }
          draft.isClosed = true;
        }
        delete draft._targetEnt;
        this.ensureEntityMetadata(draft);
        this.entities.push(draft);
        this.applySymmetryClones(draft);
        this.selectedEntity = draft;
      }
    } else if (draft.type === 'ellipse_draft') {
      if (Math.abs(draft.rx) > 0.3 && Math.abs(draft.ry) > 0.3) {
        const pts = [];
        const segs = 36;
        for (let i = 0; i <= segs; i++) {
          const t = (i / segs) * Math.PI * 2;
          pts.push({
            x: Number((draft.cx + Math.cos(t) * Math.abs(draft.rx)).toFixed(3)),
            y: Number((draft.cy + Math.sin(t) * Math.abs(draft.ry)).toFixed(3)),
          });
        }
        const ent = this.ensureEntityMetadata({
          type: 'curve',
          shapeKind: 'Ellipse',
          points: pts,
          isClosed: true,
          color: '#38bdf8',
          width: 1.5,
        });
        this.entities.push(ent);
        this.applySymmetryClones(ent);
        this.selectedEntity = ent;
      }
    } else if (draft.type === 'polygon_draft') {
      if (draft.r > 0.5) {
        const pts = CadCanvas.buildPolygonPoints(draft.cx, draft.cy, draft.r, draft.sides, draft.isStar, draft.angle);
        const ent = this.ensureEntityMetadata({
          type: 'polyline',
          shapeKind: draft.isStar ? `${draft.sides}-Point Star` : `${draft.sides}-Gon`,
          points: pts,
          isClosed: true,
          color: '#38bdf8',
          width: 1.5,
        });
        this.entities.push(ent);
        this.applySymmetryClones(ent);
        this.selectedEntity = ent;
      }
    } else {
      let valid = true;
      if (draft.type === 'line' && Math.hypot(draft.x2 - draft.x1, draft.y2 - draft.y1) < 0.2) valid = false;
      if (draft.type === 'rect' && (Math.abs(draft.w) < 0.2 || Math.abs(draft.h) < 0.2)) valid = false;
      if (draft.type === 'circle' && draft.r < 0.2) valid = false;

      if (valid) {
        if (draft.type === 'rect') {
          if (draft.w < 0) { draft.x += draft.w; draft.w = -draft.w; }
          if (draft.h < 0) { draft.y += draft.h; draft.h = -draft.h; }
        }
        this.ensureEntityMetadata(draft);
        this.entities.push(draft);
        this.applySymmetryClones(draft);
        this.selectedEntity = draft;
      }
    }
    this.currentDraft = null;
    this.requestRender();
    this.notifyEntitiesChange();
    this.notifyDraftChange();
  }

  static sampleQuadraticArc(p0, p1, ctrl, steps = 24) {
    // Pass-through control adjustment: compute Bezier control Q so midpoint hits `ctrl`
    const qx = 2 * ctrl.x - 0.5 * (p0.x + p1.x);
    const qy = 2 * ctrl.y - 0.5 * (p0.y + p1.y);
    const pts = [];
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const mt = 1 - t;
      pts.push({
        x: Number((mt * mt * p0.x + 2 * mt * t * qx + t * t * p1.x).toFixed(3)),
        y: Number((mt * mt * p0.y + 2 * mt * t * qy + t * t * p1.y).toFixed(3)),
      });
    }
    return pts;
  }

  static buildPolygonPoints(cx, cy, r, sides = 6, isStar = false, startAngle = -Math.PI / 2) {
    const pts = [];
    const total = isStar ? sides * 2 : sides;
    for (let i = 0; i <= total; i++) {
      const a = startAngle + (i / total) * Math.PI * 2;
      const rad = isStar && i % 2 === 1 ? r * 0.45 : r;
      pts.push({
        x: Number((cx + Math.cos(a) * rad).toFixed(3)),
        y: Number((cy + Math.sin(a) * rad).toFixed(3)),
      });
    }
    return pts;
  }

  closeSelectedCurve() {
    if (!this.selectedEntity) return;
    const ent = this.selectedEntity;
    if (ent.type === 'curve' || ent.type === 'polyline') {
      if (ent.points && ent.points.length >= 3) {
        this.saveState();
        const first = ent.points[0];
        const last = ent.points[ent.points.length - 1];
        if (Math.hypot(first.x - last.x, first.y - last.y) > 0.05) {
          ent.points.push({ x: first.x, y: first.y });
        }
        ent.isClosed = true;
        if (ent.relief) ent.relief.openAsStroke = false;
        this.requestRender();
        this.notifyEntitiesChange();
      }
    }
  }

  convertSelectedToCurve() {
    if (this.selectedEntities.length === 0) return 0;
    this.saveState();
    let count = 0;
    const newSelection = [];

    for (let idx = 0; idx < this.entities.length; idx++) {
      const ent = this.entities[idx];
      if (!this.selectedEntities.includes(ent) || ent.locked) continue;

      let converted = null;
      if (ent.type === 'rect') {
        const pts = ent.cornerRadius > 0
          ? CadCanvas.buildRoundedRectPoints(ent.x, ent.y, ent.w, ent.h, ent.cornerRadius)
          : [
              { x: ent.x, y: ent.y },
              { x: ent.x + ent.w, y: ent.y },
              { x: ent.x + ent.w, y: ent.y + ent.h },
              { x: ent.x, y: ent.y + ent.h },
              { x: ent.x, y: ent.y },
            ];
        converted = { ...ent, type: ent.cornerRadius > 0 ? 'curve' : 'polyline', points: pts, isClosed: true };
        delete converted.cornerRadius;
      } else if (ent.type === 'circle') {
        const pts = [];
        const segs = 28;
        for (let i = 0; i <= segs; i++) {
          const a = (i / segs) * Math.PI * 2;
          pts.push({
            x: Number((ent.cx + Math.cos(a) * ent.r).toFixed(3)),
            y: Number((ent.cy + Math.sin(a) * ent.r).toFixed(3)),
          });
        }
        converted = { ...ent, type: 'curve', points: pts, isClosed: true };
      } else if (ent.type === 'line') {
        converted = {
          ...ent,
          type: 'polyline',
          points: [{ x: ent.x1, y: ent.y1 }, { x: ent.x2, y: ent.y2 }],
          isClosed: false,
        };
      } else if (ent.type === 'polyline') {
        converted = { ...ent, type: 'curve' };
      }

      if (converted) {
        delete converted.x; delete converted.y; delete converted.w; delete converted.h;
        delete converted.cx; delete converted.cy; delete converted.r;
        delete converted.x1; delete converted.y1; delete converted.x2; delete converted.y2;
        this.entities[idx] = this.ensureEntityMetadata(converted);
        newSelection.push(this.entities[idx]);
        count++;
      } else {
        newSelection.push(ent);
      }
    }

    if (count > 0) {
      this.selectedEntities = newSelection;
      this.requestRender();
      this.notifyEntitiesChange();
      this.notifySelectionChange();
    }
    return count;
  }

  smoothSelectedCurve() {
    if (!this.selectedEntity) return;
    const ent = this.selectedEntity;
    if ((ent.type === 'curve' || ent.type === 'polyline') && ent.points && ent.points.length > 3) {
      this.saveState();
      const pts = ent.points;
      const smoothed = [{ ...pts[0] }];
      for (let i = 1; i < pts.length - 1; i++) {
        smoothed.push({
          x: pts[i - 1].x * 0.25 + pts[i].x * 0.5 + pts[i + 1].x * 0.25,
          y: pts[i - 1].y * 0.25 + pts[i].y * 0.5 + pts[i + 1].y * 0.25,
        });
      }
      if (ent.isClosed) {
        smoothed.push({ ...smoothed[0] });
      } else {
        smoothed.push({ ...pts[pts.length - 1] });
      }
      ent.points = smoothed;
      this.requestRender();
      this.notifyEntitiesChange();
    }
  }

  // Reduce node density on dense traced curves so manual reshaping is easy
  simplifySelectedCurve(toleranceMm = 0.7) {
    if (this.selectedEntities.length === 0) return 0;
    this.saveState();
    let count = 0;
    for (const ent of this.selectedEntities) {
      if ((ent.type === 'curve' || ent.type === 'polyline') && ent.points && ent.points.length > 6) {
        const simplified = ImageVectorizer.simplifyRDP(ent.points, toleranceMm);
        if (simplified.length >= 3) {
          if (ent.isClosed) {
            const first = simplified[0];
            const last = simplified[simplified.length - 1];
            if (Math.hypot(first.x - last.x, first.y - last.y) > 0.05) {
              simplified.push({ x: first.x, y: first.y });
            }
          }
          ent.points = simplified;
          count++;
        }
      }
    }
    if (count > 0) {
      this.selectedNodeIndex = -1;
      this.requestRender();
      this.notifyEntitiesChange();
    }
    return count;
  }

  /**
   * Find closest visible unlocked curve/polyline entity within `maxDistMm`
   */
  findClosestCurveEntity(wx, wy, maxDistMm = 3.5) {
    let bestEnt = null;
    let bestD = maxDistMm;
    for (let i = this.entities.length - 1; i >= 0; i--) {
      const ent = this.entities[i];
      if (ent.visible === false || ent.locked) continue;
      if ((ent.type !== 'curve' && ent.type !== 'polyline') || !ent.points || ent.points.length < 2) continue;
      const eb = this.getEntityBounds(ent);
      if (eb && (wx < eb.minX - maxDistMm || wx > eb.maxX + maxDistMm || wy < eb.minY - maxDistMm || wy > eb.maxY + maxDistMm)) continue;
      const proj = CadCanvas.projectPointOntoCurve(ent, { x: wx, y: wy });
      if (proj && proj.dist < bestD) {
        bestD = proj.dist;
        bestEnt = ent;
      }
    }
    return bestEnt;
  }

  /**
   * Project point P onto a curve/polyline entity's segments -> { segIdx, t, u, point, dist, ring }
   */
  static projectPointOntoCurve(ent, pt) {
    const raw = ent.points;
    if (!raw || raw.length < 2) return null;
    let ring = raw;
    if (ent.isClosed && raw.length >= 3) {
      const f = raw[0], l = raw[raw.length - 1];
      if (Math.hypot(f.x - l.x, f.y - l.y) < 0.05) ring = raw.slice(0, -1);
    }
    const M = ring.length;
    if (M < 2) return null;

    const segCount = ent.isClosed ? M : M - 1;
    let best = null;
    for (let i = 0; i < segCount; i++) {
      const a = ring[i];
      const b = ring[(i + 1) % M];
      const dx = b.x - a.x, dy = b.y - a.y;
      const lenSq = dx * dx + dy * dy;
      const t = lenSq === 0 ? 0 : Math.max(0, Math.min(1, ((pt.x - a.x) * dx + (pt.y - a.y) * dy) / lenSq));
      const x = a.x + t * dx;
      const y = a.y + t * dy;
      const d = Math.hypot(pt.x - x, pt.y - y);
      if (!best || d < best.dist) {
        best = { segIdx: i, t, u: i + t, point: { x, y }, dist: d, ring };
      }
    }
    return best;
  }

  /**
   * Seamlessly splice a new Pen / Freehand replacement curve directly into an existing design outline!
   * Handles both replacing a local bumped span on a closed contour AND extending open curves.
   */
  trySpliceIntoExistingCurve(newPoints, preferredTarget = null) {
    if (!newPoints || newPoints.length < 2) return false;
    const pStart = newPoints[0];
    const pEnd = newPoints[newPoints.length - 1];
    const snapTol = Math.max(3.5, 12 / this.scale);

    // Candidate target curves
    const candidates = [];
    if (preferredTarget && (preferredTarget.type === 'curve' || preferredTarget.type === 'polyline') && !preferredTarget.locked && preferredTarget.visible !== false) {
      candidates.push(preferredTarget);
    }
    const startMatch = this.findClosestCurveEntity(pStart.x, pStart.y, snapTol);
    if (startMatch && !candidates.includes(startMatch)) candidates.push(startMatch);

    for (const target of candidates) {
      const projA = CadCanvas.projectPointOntoCurve(target, pStart);
      const projB = CadCanvas.projectPointOntoCurve(target, pEnd);
      if (!projA || projA.dist > snapTol) continue;

      // Case 1: Both ends attach onto the SAME existing design outline (separated by > 1.2mm) -> Replace local span!
      if (projB && projB.dist <= snapTol && Math.hypot(pEnd.x - pStart.x, pEnd.y - pStart.y) > 1.2) {
        const ring = projA.ring;
        const M = ring.length;
        const patch = newPoints.map((p) => ({ x: Number(p.x.toFixed(3)), y: Number(p.y.toFixed(3)) }));
        patch[0] = { x: Number(projA.point.x.toFixed(3)), y: Number(projA.point.y.toFixed(3)) };
        patch[patch.length - 1] = { x: Number(projB.point.x.toFixed(3)), y: Number(projB.point.y.toFixed(3)) };

        if (target.isClosed) {
          const extractArc = (uFrom, uTo, ptFrom, ptTo) => {
            const iFrom = Math.floor(uFrom) % M;
            const iTo = Math.floor(uTo) % M;
            const arc = [ptFrom];
            const steps = (iTo - iFrom + M) % M;
            if (steps === 0 && uFrom > uTo) {
              for (let k = 1; k <= M; k++) arc.push({ ...ring[(iFrom + k) % M] });
            } else {
              for (let k = 1; k <= steps; k++) arc.push({ ...ring[(iFrom + k) % M] });
            }
            arc.push(ptTo);
            let len = 0;
            for (let k = 0; k < arc.length - 1; k++) {
              len += Math.hypot(arc[k + 1].x - arc[k].x, arc[k + 1].y - arc[k].y);
            }
            return { arc, len };
          };

          const fwd = extractArc(projA.u, projB.u, patch[0], patch[patch.length - 1]);
          const bwd = extractArc(projB.u, projA.u, patch[patch.length - 1], patch[0]);

          let combined;
          if (fwd.len <= bwd.len) {
            combined = bwd.arc.concat(patch.slice(1));
          } else {
            combined = fwd.arc.concat([...patch].reverse().slice(1));
          }
          target.points = CadCanvas.dedupeClosePoints(combined, true);
          target.isClosed = true;
        } else {
          let aU = projA.u, bU = projB.u, orderedPatch = patch;
          if (aU > bU) {
            [aU, bU] = [bU, aU];
            orderedPatch = [...patch].reverse();
          }
          const head = ring.slice(0, Math.floor(aU) + 1);
          const tail = ring.slice(Math.floor(bU) + 1);
          target.points = CadCanvas.dedupeClosePoints([...head, ...orderedPatch, ...tail], false);
        }

        this.selectedEntity = target;
        return true;
      }

      // Case 2: Extending an OPEN existing curve from its start or end endpoint
      if (!target.isClosed) {
        const pts = target.points;
        const dFirst = Math.hypot(pStart.x - pts[0].x, pStart.y - pts[0].y);
        const dLast = Math.hypot(pStart.x - pts[pts.length - 1].x, pStart.y - pts[pts.length - 1].y);
        if (dLast <= snapTol && dLast <= dFirst) {
          target.points = CadCanvas.dedupeClosePoints([...pts, ...newPoints.slice(1)], false);
          this.selectedEntity = target;
          return true;
        } else if (dFirst <= snapTol) {
          target.points = CadCanvas.dedupeClosePoints([...[...newPoints].reverse().slice(0, -1), ...pts], false);
          this.selectedEntity = target;
          return true;
        }
      }
    }
    return false;
  }

  static dedupeClosePoints(pts, isClosed) {
    if (!pts || pts.length === 0) return [];
    const cleaned = [{ ...pts[0] }];
    for (let i = 1; i < pts.length; i++) {
      const prev = cleaned[cleaned.length - 1];
      if (Math.hypot(pts[i].x - prev.x, pts[i].y - prev.y) > 0.05) {
        cleaned.push({ ...pts[i] });
      }
    }
    if (isClosed && cleaned.length >= 3) {
      const f = cleaned[0], l = cleaned[cleaned.length - 1];
      if (Math.hypot(f.x - l.x, f.y - l.y) > 0.05) {
        cleaned.push({ x: f.x, y: f.y });
      }
    }
    return cleaned;
  }

  // Split curve/polyline/line at clicked point (Knife / Scissors tool)
  splitAt(wx, wy) {
    const tolerance = 8 / this.scale;
    const hit = this.findEntityAt(wx, wy, 8);
    if (!hit || hit.locked) return false;

    if (hit.type === 'line') {
      this.saveState();
      const idx = this.entities.indexOf(hit);
      const part1 = this.ensureEntityMetadata({
        ...JSON.parse(JSON.stringify(hit)),
        id: undefined,
        name: `${hit.name} (A)`,
        x2: wx,
        y2: wy,
      });
      const part2 = this.ensureEntityMetadata({
        ...JSON.parse(JSON.stringify(hit)),
        id: undefined,
        name: `${hit.name} (B)`,
        x1: wx,
        y1: wy,
      });
      this.entities.splice(idx, 1, part1, part2);
      this.selectedEntity = part2;
      this.requestRender();
      this.notifyEntitiesChange();
      return true;
    }

    if ((hit.type === 'curve' || hit.type === 'polyline') && hit.points && hit.points.length >= 2) {
      const pts = hit.points;
      let bestSeg = -1;
      let bestDist = tolerance;
      const segCount = hit.isClosed ? pts.length - 1 : pts.length - 1;
      for (let i = 0; i < segCount; i++) {
        const d = CadCanvas.distToSegment({ x: wx, y: wy }, pts[i], pts[i + 1]);
        if (d <= bestDist) {
          bestDist = d;
          bestSeg = i;
        }
      }
      if (bestSeg === -1) return false;

      this.saveState();
      const idx = this.entities.indexOf(hit);
      if (hit.isClosed) {
        // Open a closed contour cleanly starting & ending at cut point
        const openPts = [{ x: wx, y: wy }, ...pts.slice(bestSeg + 1, pts.length - 1), ...pts.slice(0, bestSeg + 1), { x: wx, y: wy }];
        // Re-order without duplicated closing point -> open curve
        openPts.pop();
        hit.points = openPts;
        hit.isClosed = false;
        if (hit.relief) hit.relief.openAsStroke = true;
        this.selectedEntity = hit;
      } else {
        const pts1 = [...pts.slice(0, bestSeg + 1), { x: wx, y: wy }];
        const pts2 = [{ x: wx, y: wy }, ...pts.slice(bestSeg + 1)];
        if (pts1.length >= 2 && pts2.length >= 2) {
          const cloneBase = JSON.parse(JSON.stringify(hit));
          const part1 = this.ensureEntityMetadata({ ...cloneBase, id: undefined, name: `${hit.name} (A)`, points: pts1, isClosed: false });
          const part2 = this.ensureEntityMetadata({ ...cloneBase, id: undefined, name: `${hit.name} (B)`, points: pts2, isClosed: false });
          this.entities.splice(idx, 1, part1, part2);
          this.selectedEntity = part2;
        }
      }
      this.requestRender();
      this.notifyEntitiesChange();
      return true;
    }
    return false;
  }

  duplicateSelected() {
    if (this.selectedEntities.length === 0) return 0;
    this.saveState();
    const clones = [];
    for (const ent of this.selectedEntities) {
      const copy = JSON.parse(JSON.stringify(ent));
      delete copy.id;
      copy.name = `${ent.name || 'Layer'} (Copy)`;
      this.moveEntity(copy, 4, 4);
      this.ensureEntityMetadata(copy);
      this.entities.push(copy);
      clones.push(copy);
    }
    this.selectedEntities = clones;
    this.requestRender();
    this.notifyEntitiesChange();
    this.notifySelectionChange();
    return clones.length;
  }

  static hasCubicHandles(points) {
    if (!points || points.length === 0) return false;
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      if (p.hIn || p.hOut || p.corner !== undefined) return true;
    }
    return false;
  }

  static mapAnchorPoint(p, fn) {
    const base = fn(p.x, p.y);
    const out = { x: Number(base.x.toFixed(3)), y: Number(base.y.toFixed(3)) };
    if (p.corner !== undefined) out.corner = p.corner;
    if (p.hIn) {
      const hi = fn(p.hIn.x, p.hIn.y);
      out.hIn = { x: Number(hi.x.toFixed(3)), y: Number(hi.y.toFixed(3)) };
    }
    if (p.hOut) {
      const ho = fn(p.hOut.x, p.hOut.y);
      out.hOut = { x: Number(ho.x.toFixed(3)), y: Number(ho.y.toFixed(3)) };
    }
    return out;
  }

  /**
   * Transform all coordinates on an entity (including Bezier handles & compound holes!).
   */
  static transformEntityPoints(ent, fn, scalarScale = 1) {
    if (ent.type === 'line') {
      const p1 = fn(ent.x1, ent.y1), p2 = fn(ent.x2, ent.y2);
      ent.x1 = p1.x; ent.y1 = p1.y; ent.x2 = p2.x; ent.y2 = p2.y;
    } else if (ent.type === 'rect') {
      const p1 = fn(ent.x, ent.y), p2 = fn(ent.x + ent.w, ent.y + ent.h);
      ent.x = Math.min(p1.x, p2.x);
      ent.y = Math.min(p1.y, p2.y);
      ent.w = Math.abs(p2.x - p1.x);
      ent.h = Math.abs(p2.y - p1.y);
      if (ent.cornerRadius && scalarScale !== 1) ent.cornerRadius = Math.max(0, ent.cornerRadius * scalarScale);
    } else if (ent.type === 'circle') {
      const c = fn(ent.cx, ent.cy);
      ent.cx = c.x; ent.cy = c.y;
      if (scalarScale !== 1) ent.r = Math.max(0.2, ent.r * scalarScale);
    } else if ((ent.type === 'curve' || ent.type === 'polyline') && ent.points) {
      ent.points = ent.points.map((p) => CadCanvas.mapAnchorPoint(p, fn));
      if (ent.holes && ent.holes.length > 0) {
        ent.holes = ent.holes.map((hole) => hole.map((p) => CadCanvas.mapAnchorPoint(p, fn)));
      }
    } else if (ent.type === 'text') {
      const p = fn(ent.x, ent.y);
      ent.x = p.x; ent.y = p.y;
      if (scalarScale !== 1) ent.size = Math.max(4, ent.size * scalarScale);
    }
  }

  /**
   * Mirror entity helper across (cx, cy) along axis 'x' | 'y' | 'xy'
   */
  static cloneMirrored(ent, cx, cy, flipX, flipY) {
    const copy = JSON.parse(JSON.stringify(ent));
    delete copy.id;
    copy.name = `${ent.name || 'Layer'} (Sym)`;
    CadCanvas.transformEntityPoints(copy, (px, py) => ({
      x: flipX ? 2 * cx - px : px,
      y: flipY ? 2 * cy - py : py,
    }));
    return copy;
  }

  applySymmetryClones(ent) {
    if (!this.symmetryMode || this.symmetryMode === 'none') return [];
    const cx = this.workpieceWidth / 2;
    const cy = this.workpieceHeight / 2;
    const clones = [];
    if (this.symmetryMode === 'mirrorX' || this.symmetryMode === 'quad') {
      clones.push(this.ensureEntityMetadata(CadCanvas.cloneMirrored(ent, cx, cy, true, false)));
    }
    if (this.symmetryMode === 'mirrorY' || this.symmetryMode === 'quad') {
      clones.push(this.ensureEntityMetadata(CadCanvas.cloneMirrored(ent, cx, cy, false, true)));
    }
    if (this.symmetryMode === 'quad') {
      clones.push(this.ensureEntityMetadata(CadCanvas.cloneMirrored(ent, cx, cy, true, true)));
    }
    this.entities.push(...clones);
    return clones;
  }

  mirrorCopySelected(axis = 'x') {
    const targets = this.getTransformTargets();
    if (targets.length === 0) return 0;
    this.saveState();
    const cx = this.workpieceWidth / 2;
    const cy = this.workpieceHeight / 2;
    const newClones = [];
    for (const ent of targets) {
      const copy = this.ensureEntityMetadata(CadCanvas.cloneMirrored(ent, cx, cy, axis === 'x', axis === 'y'));
      this.entities.push(copy);
      newClones.push(copy);
    }
    this.selectedEntities = [...targets, ...newClones];
    this.requestRender();
    this.notifyEntitiesChange();
    this.notifySelectionChange();
    return newClones.length;
  }

  /**
   * Precisely position & size selected entities in millimeters (for manual CNC numeric entry!)
   */
  setSelectionBoundsMm({ x, y, w, h }) {
    if (this.selectedEntities.length === 0) return false;
    const b = this.getSelectionBounds();
    if (!b) return false;
    this.saveState();
    const origW = Math.max(0.2, b.maxX - b.minX);
    const origH = Math.max(0.2, b.maxY - b.minY);
    const newX = isFinite(x) ? x : b.minX;
    const newY = isFinite(y) ? y : b.minY;
    const newW = isFinite(w) && w > 0.2 ? w : origW;
    const newH = isFinite(h) && h > 0.2 ? h : origH;
    const sx = newW / origW;
    const sy = newH / origH;

    for (const ent of this.selectedEntities) {
      if (ent.locked) continue;
      if (this._bboxCache) this._bboxCache.delete(ent);
      CadCanvas.transformEntityPoints(
        ent,
        (px, py) => ({ x: newX + (px - b.minX) * sx, y: newY + (py - b.minY) * sy }),
        (sx + sy) * 0.5
      );
    }
    this.requestRender();
    this.notifyEntitiesChange();
    this.notifySelectionChange();
    return true;
  }

  // Center selected (or all) vectors neatly onto workpiece center as a cohesive group
  centerOnWorkpiece() {
    const targets = this.getTransformTargets();
    if (targets.length === 0) return false;
    const b = this.getSelectionBounds(targets);
    if (!b) return false;
    this.saveState();
    const dx = (this.workpieceWidth / 2) - (b.minX + b.maxX) / 2;
    const dy = (this.workpieceHeight / 2) - (b.minY + b.maxY) / 2;
    for (const ent of targets) this.moveEntity(ent, dx, dy);
    this.requestRender();
    this.notifyEntitiesChange();
    this.notifySelectionChange();
    return true;
  }

  /**
   * Align selected entities:
   * - If 1 selected: aligns relative to Workpiece Stock borders/center!
   * - If 2+ selected: aligns each entity relative to selection bounding box!
   * mode: 'left' | 'centerH' | 'right' | 'top' | 'middleV' | 'bottom'
   */
  alignSelected(mode = 'centerH') {
    const targets = this.selectedEntities.filter((e) => !e.locked && e.visible !== false);
    if (targets.length === 0) return false;
    const ref = targets.length === 1
      ? { minX: 0, maxX: this.workpieceWidth, minY: 0, maxY: this.workpieceHeight }
      : this.getSelectionBounds(targets);
    if (!ref) return false;

    this.saveState();
    const refCx = (ref.minX + ref.maxX) / 2;
    const refCy = (ref.minY + ref.maxY) / 2;

    for (const ent of targets) {
      const eb = this.getEntityBounds(ent);
      if (!eb) continue;
      let dx = 0, dy = 0;
      if (mode === 'left') dx = ref.minX - eb.minX;
      else if (mode === 'centerH') dx = refCx - (eb.minX + eb.maxX) / 2;
      else if (mode === 'right') dx = ref.maxX - eb.maxX;
      else if (mode === 'top') dy = ref.minY - eb.minY;
      else if (mode === 'middleV') dy = refCy - (eb.minY + eb.maxY) / 2;
      else if (mode === 'bottom') dy = ref.maxY - eb.maxY;
      if (Math.abs(dx) > 1e-4 || Math.abs(dy) > 1e-4) {
        this.moveEntity(ent, dx, dy);
      }
    }
    this.requestRender();
    this.notifyEntitiesChange();
    this.notifySelectionChange();
    return true;
  }

  /**
   * Round / Fillet sharp corners on selected Rectangles or Polylines/Curves
   */
  filletSelected(radiusMm = 3.0) {
    const targets = this.getTransformTargets();
    if (targets.length === 0) return 0;
    this.saveState();
    let count = 0;
    const r = Math.max(0.2, radiusMm);

    for (const ent of targets) {
      if (this._bboxCache) this._bboxCache.delete(ent);
      if (ent.type === 'rect') {
        const maxR = Math.min(Math.abs(ent.w), Math.abs(ent.h)) / 2;
        ent.cornerRadius = Number(Math.min(maxR, r).toFixed(2));
        count++;
      } else if ((ent.type === 'polyline' || ent.type === 'curve') && ent.points && ent.points.length >= 3) {
        const raw = ent.points;
        const isClosed = !!ent.isClosed;
        let ring = raw;
        if (isClosed && raw.length >= 3 && Math.hypot(raw[0].x - raw[raw.length - 1].x, raw[0].y - raw[raw.length - 1].y) < 0.05) {
          ring = raw.slice(0, -1);
        }
        const M = ring.length;
        if (M < 3) continue;
        const out = [];
        for (let i = 0; i < M; i++) {
          if (!isClosed && (i === 0 || i === M - 1)) {
            out.push({ ...ring[i] });
            continue;
          }
          const prev = ring[(i - 1 + M) % M];
          const curr = ring[i];
          const next = ring[(i + 1) % M];
          const d1 = Math.hypot(curr.x - prev.x, curr.y - prev.y);
          const d2 = Math.hypot(next.x - curr.x, next.y - curr.y);
          const trim = Math.min(r, d1 * 0.4, d2 * 0.4);
          if (trim < 0.15) {
            out.push({ ...curr });
            continue;
          }
          const pIn = {
            x: Number((curr.x + ((prev.x - curr.x) / d1) * trim).toFixed(3)),
            y: Number((curr.y + ((prev.y - curr.y) / d1) * trim).toFixed(3)),
          };
          const pOut = {
            x: Number((curr.x + ((next.x - curr.x) / d2) * trim).toFixed(3)),
            y: Number((curr.y + ((next.y - curr.y) / d2) * trim).toFixed(3)),
          };
          const pMid = {
            x: Number((0.25 * pIn.x + 0.5 * curr.x + 0.25 * pOut.x).toFixed(3)),
            y: Number((0.25 * pIn.y + 0.5 * curr.y + 0.25 * pOut.y).toFixed(3)),
          };
          out.push(pIn, pMid, pOut);
        }
        if (isClosed && out.length > 0) out.push({ ...out[0] });
        ent.points = out;
        ent.type = 'curve';
        count++;
      }
    }

    if (count > 0) {
      this.requestRender();
      this.notifyEntitiesChange();
      this.notifySelectionChange();
    }
    return count;
  }

  // Scale & center selected (or all) vectors cleanly inside workpiece stock with margin
  fitToWorkpiece(marginMm = 5) {
    const targets = this.getTransformTargets();
    if (targets.length === 0) return false;
    const b = this.getSelectionBounds(targets);
    if (!b) return false;
    const bw = Math.max(0.5, b.maxX - b.minX);
    const bh = Math.max(0.5, b.maxY - b.minY);
    const targetW = Math.max(5, this.workpieceWidth - marginMm * 2);
    const targetH = Math.max(5, this.workpieceHeight - marginMm * 2);
    const s = Math.min(targetW / bw, targetH / bh);

    this.saveState();
    const cx = (b.minX + b.maxX) / 2;
    const cy = (b.minY + b.maxY) / 2;
    const wxCenter = this.workpieceWidth / 2;
    const wyCenter = this.workpieceHeight / 2;

    for (const ent of targets) {
      if (this._bboxCache) this._bboxCache.delete(ent);
      CadCanvas.transformEntityPoints(
        ent,
        (px, py) => ({ x: wxCenter + (px - cx) * s, y: wyCenter + (py - cy) * s }),
        s
      );
    }
    this.requestRender();
    this.notifyEntitiesChange();
    return true;
  }

  // Create Rectangular Grid Array of selected entities (classic JDPaint furniture/door pattern repeat)
  createGridArray(cols = 2, rows = 2, gapMm = 5) {
    if (this.selectedEntities.length === 0) return 0;
    const b = this.getSelectionBounds(this.selectedEntities);
    if (!b) return 0;
    const stepX = (b.maxX - b.minX) + gapMm;
    const stepY = (b.maxY - b.minY) + gapMm;
    this.saveState();
    let created = 0;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (r === 0 && c === 0) continue;
        for (const ent of this.selectedEntities) {
          const copy = JSON.parse(JSON.stringify(ent));
          delete copy.id;
          copy.name = `${ent.name || 'Layer'} [${r + 1},${c + 1}]`;
          this.moveEntity(copy, c * stepX, r * stepY);
          this.ensureEntityMetadata(copy);
          this.entities.push(copy);
          created++;
        }
      }
    }
    this.requestRender();
    this.notifyEntitiesChange();
    return created;
  }

  initEvents() {
    window.addEventListener('resize', () => this.resize());
    if (typeof ResizeObserver !== 'undefined' && this.canvas.parentElement) {
      this._ro = new ResizeObserver(() => this.resize());
      this._ro.observe(this.canvas.parentElement);
    }

    this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
    this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
    this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
    this.canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        e.preventDefault();
        const t = e.touches[0];
        this.onMouseDown({ clientX: t.clientX, clientY: t.clientY, button: 0, detail: 1, shiftKey: false, ctrlKey: false, metaKey: false });
      }
    }, { passive: false });
    this.canvas.addEventListener('touchmove', (e) => {
      if (e.touches.length === 1) {
        e.preventDefault();
        const t = e.touches[0];
        this.onMouseMove({ clientX: t.clientX, clientY: t.clientY, shiftKey: false });
      }
    }, { passive: false });
    this.canvas.addEventListener('touchend', () => this.onMouseUp(), { passive: true });
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    this.canvas.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });

    this.canvas.addEventListener('dblclick', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const world = this.screenToWorld(sx, sy);

      if (this.activeTool === 'node' && this.selectedEntity && !this.selectedEntity.locked) {
        const hitNode = this.findNodeTarget(this.selectedEntity, world.x, world.y);
        if (hitNode) {
          this.toggleSelectedNodeCorner(hitNode);
        } else {
          this.insertNodeOnEdge(this.selectedEntity, world.x, world.y);
        }
      } else if (this.activeTool === 'select') {
        const hit = this.findEntityAt(world.x, world.y);
        if (hit && hit.type === 'text' && !hit.locked) {
          const updated = prompt('Edit engraving text:', hit.text || '');
          if (updated !== null && updated.trim()) {
            this.saveState();
            hit.text = updated;
            hit.name = `Text (${updated.slice(0, 10)})`;
            if (this._bboxCache) this._bboxCache.delete(hit);
            this.requestRender();
            this.notifyEntitiesChange();
            this.notifySelectionChange();
          }
        }
      } else {
        this.finishCurrentDraft();
      }
    });

    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;

      if (e.code === 'Space') {
        e.preventDefault();
        if (!this.isSpaceDown) {
          this.isSpaceDown = true;
          this.canvas.style.cursor = 'grab';
        }
        return;
      }

      if (e.key === 'Enter') {
        this.finishCurrentDraft();
      } else if (e.key === 'Escape') {
        if (this.currentDraft) {
          this.currentDraft = null;
          this.requestRender();
          this.notifyDraftChange();
        } else {
          this.selectedEntities = [];
          this.requestRender();
          this.notifySelectionChange();
        }
      } else if (e.key === 'Backspace' && this.currentDraft && (this.currentDraft.type === 'curve' || this.currentDraft.type === 'polyline')) {
        // Backspace while drafting removes ONLY the last placed vertex instead of deleting entities!
        e.preventDefault();
        if (this.currentDraft.points.length > 1) {
          this.currentDraft.points.pop();
        } else {
          this.currentDraft = null;
          this.notifyDraftChange();
        }
        this.requestRender();
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        if (this.activeTool === 'node' && this.selectedEntity && this.selectedNodeIndex !== -1) {
          if (!this.selectedEntity.locked) this.deleteSelectedNode();
          return;
        }
        const deletable = this.selectedEntities.filter((sel) => !sel.locked);
        if (deletable.length > 0) {
          this.saveState();
          for (const sel of deletable) {
            const idx = this.entities.indexOf(sel);
            if (idx !== -1) this.entities.splice(idx, 1);
          }
          this.selectedEntities = this.selectedEntities.filter((sel) => sel.locked);
          this.requestRender();
          this.notifyEntitiesChange();
          this.notifySelectionChange();
        }
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) this.redo();
        else this.undo();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        this.redo();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'a') {
        e.preventDefault();
        this.selectedEntities = this.entities.filter((ent) => ent.visible !== false);
        this.requestRender();
        this.notifySelectionChange();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'd') {
        e.preventDefault();
        this.duplicateSelected();
      }
    });

    window.addEventListener('keyup', (e) => {
      if (e.code === 'Space') {
        this.isSpaceDown = false;
        this.isPanning = false;
      }
    });
  }

  onMouseDown(e) {
    const rect = this.canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    this.isMouseDown = true;

    // Right-Click during multi-point drafting (Pen / Polyline / Arc) finishes the path cleanly (AutoCAD/JDPaint standard)!
    if (e.button === 2 && this.currentDraft) {
      this.finishCurrentDraft(false);
      return;
    }

    // Pan with Spacebar+Drag, Middle Click, or Shift+Drag (when not selecting/shaping)
    if (this.isSpaceDown || e.button === 1 || (e.shiftKey && e.button === 0 && !['select', 'line', 'rect', 'ellipse', 'circle', 'polygon'].includes(this.activeTool))) {
      this.isPanning = true;
      this.panStart = { x: sx - this.panX, y: sy - this.panY };
      this.canvas.style.cursor = 'grabbing';
      return;
    }

    const world = this.screenToWorld(sx, sy);

    // Knife / Split Tool
    if (this.activeTool === 'knife') {
      this.splitAt(world.x, world.y);
      return;
    }

    // Path Reshape / Clay Push Brush (drag directly over any traced outline to bend it!)
    if (this.activeTool === 'reshape') {
      const raw = this.screenToWorld(sx, sy, true);
      this.saveState();
      this.isDragging = true;
      this.dragStart = { ...raw };
      return;
    }

    // Eraser Tool
    if (this.activeTool === 'eraser') {
      this.eraseAt(world.x, world.y);
      return;
    }

    // Node Edit Tool (Drag Bezier handle, drag anchor node, Alt+drag handle to break tangent, Right-Click node to delete, Double-Click node to toggle Corner/Smooth)
    if (this.activeTool === 'node') {
      if (this.selectedEntity && !this.selectedEntity.locked) {
        const bezHandle = this.findBezierHandleAt(this.selectedEntity, world.x, world.y);
        if (bezHandle) {
          this.saveState();
          this.activeBezierHandle = bezHandle;
          this.selectedNodeIndex = bezHandle.nodeIdx;
          this.selectedHoleIndex = bezHandle.holeIdx;
          this.isDragging = true;
          this.dragStart = { ...world };
          this.requestRender();
          return;
        }
        const nt = this.findNodeTarget(this.selectedEntity, world.x, world.y);
        if (nt) {
          if (e.button === 2) {
            this.selectedNodeIndex = nt.nodeIdx;
            this.selectedHoleIndex = nt.holeIdx;
            this.activeBezierHandle = null;
            this.deleteSelectedNode();
            return;
          }
          this.saveState();
          this.selectedNodeIndex = nt.nodeIdx;
          this.selectedHoleIndex = nt.holeIdx;
          this.activeBezierHandle = null;
          this.isDragging = true;
          this.dragStart = { ...world };
          this.requestRender();
          return;
        }
        // Check if clicking directly on an edge of selected curve (outer or hole) -> insert node & start dragging right away!
        const insertedIdx = this.insertNodeOnEdge(this.selectedEntity, world.x, world.y);
        if (insertedIdx !== -1) {
          this.activeBezierHandle = null;
          this.isDragging = true;
          this.dragStart = { ...world };
          return;
        }
      }
      const hit = this.findEntityAt(world.x, world.y);
      this.selectedEntity = hit;
      this.selectedNodeIndex = -1;
      this.selectedHoleIndex = -1;
      this.activeBezierHandle = null;
      this.requestRender();
      return;
    }

    // Select & Transform Tool (checks 8 resize handles + rotate handle BEFORE entity hit!)
    if (this.activeTool === 'select') {
      if (this.selectedEntities.length > 0) {
        const handle = this.findTransformHandleAt(sx, sy);
        if (handle) {
          this.saveState();
          this.activeTransformHandle = handle;
          this.transformOriginBounds = this.getSelectionBounds();
          this.transformSnapshot = JSON.parse(JSON.stringify(this.selectedEntities));
          this.dragStart = { ...world };
          this.requestRender();
          return;
        }
      }

      const hit = this.findEntityAt(world.x, world.y);
      if (hit) {
        this.saveState();
        if (e.metaKey || e.ctrlKey || e.shiftKey) {
          const idx = this.selectedEntities.indexOf(hit);
          if (idx === -1) this.selectedEntities.push(hit);
          else this.selectedEntities.splice(idx, 1);
          this.notifySelectionChange();
        } else if (!this.selectedEntities.includes(hit)) {
          this.selectedEntity = hit;
        }
        this.isDragging = true;
        this.dragStart = { ...world };
      } else {
        this.isBoxSelecting = true;
        this.boxSelectStart = { x: world.x, y: world.y };
        this.boxSelectEnd = { x: world.x, y: world.y };
        this.selectedEntities = [];
        this.notifySelectionChange();
      }
      this.requestRender();
      return;
    }

    // Freehand Sketch Brush
    if (this.activeTool === 'freehand') {
      const raw = this.screenToWorld(sx, sy, true);
      this.currentDraft = {
        type: 'freehand',
        points: [{ x: raw.x, y: raw.y }],
        color: '#38bdf8',
        width: 1.8,
      };
      this.requestRender();
      return;
    }

    // Pen Tool (Supports starting/ending ON an existing outline to seamlessly replace/add segments!)
    if (this.activeTool === 'pen') {
      if (e.detail >= 2) return; // Prevent duplicate trailing node on double-click finish
      if (this.currentDraft && this.currentDraft.points.length >= 3) {
        const p0 = this.currentDraft.points[0];
        if (Math.hypot(world.x - p0.x, world.y - p0.y) <= 12 / this.scale) {
          this.finishCurrentDraft(true);
          return;
        }
      }
      if (!this.currentDraft) {
        const startTarget = (this.activeSnapTarget && this.activeSnapTarget.ent)
          ? this.activeSnapTarget.ent
          : this.findClosestCurveEntity(world.x, world.y, 3.5);
        this.currentDraft = {
          type: 'curve',
          points: [{ x: world.x, y: world.y }],
          isClosed: false,
          color: '#38bdf8',
          width: 1.5,
          _targetEnt: startTarget || null,
        };
        this.notifyDraftChange();
      } else {
        const pts = this.currentDraft.points;
        const prev = pts[pts.length - 1];
        if (Math.hypot(world.x - prev.x, world.y - prev.y) > 0.25) {
          pts.push({ x: world.x, y: world.y });
          this.notifyDraftChange();
        }
      }
      this.requestRender();
      return;
    }

    // Polyline Tool
    if (this.activeTool === 'polyline') {
      if (e.detail >= 2) return;
      if (this.currentDraft && this.currentDraft.points.length >= 3) {
        const p0 = this.currentDraft.points[0];
        if (Math.hypot(world.x - p0.x, world.y - p0.y) <= 12 / this.scale) {
          this.finishCurrentDraft(true);
          return;
        }
      }
      if (!this.currentDraft) {
        this.currentDraft = {
          type: 'polyline',
          points: [{ x: world.x, y: world.y }],
          isClosed: false,
          color: '#38bdf8',
          width: 1.5,
        };
        this.notifyDraftChange();
      } else {
        this.currentDraft.points.push({ x: world.x, y: world.y });
        this.notifyDraftChange();
      }
      this.requestRender();
      return;
    }

    // 3-Point Arc Tool
    if (this.activeTool === 'arc') {
      if (!this.currentDraft) {
        this.currentDraft = {
          type: 'arc_draft',
          pts: [{ x: world.x, y: world.y }, { x: world.x, y: world.y }, { x: world.x, y: world.y }],
          stage: 1, // placing endpoint
          color: '#38bdf8',
          width: 1.5,
        };
      } else if (this.currentDraft.stage === 1) {
        this.currentDraft.pts[1] = { x: world.x, y: world.y };
        this.currentDraft.pts[2] = {
          x: (this.currentDraft.pts[0].x + world.x) / 2,
          y: (this.currentDraft.pts[0].y + world.y) / 2,
        };
        this.currentDraft.stage = 2; // placing bulge height
      } else if (this.currentDraft.stage === 2) {
        this.currentDraft.pts[2] = { x: world.x, y: world.y };
        this.finishCurrentDraft();
      }
      this.requestRender();
      return;
    }

    // Drag-to-Draw Geometric Shapes
    if (this.activeTool === 'line') {
      this.currentDraft = { type: 'line', x1: world.x, y1: world.y, x2: world.x, y2: world.y, color: '#38bdf8', width: 1.5 };
      this.requestRender();
    } else if (this.activeTool === 'rect') {
      this.currentDraft = { type: 'rect', x: world.x, y: world.y, w: 0, h: 0, color: '#38bdf8', width: 1.5 };
      this.requestRender();
    } else if (this.activeTool === 'circle') {
      this.currentDraft = { type: 'circle', cx: world.x, cy: world.y, r: 0, color: '#38bdf8', width: 1.5 };
      this.requestRender();
    } else if (this.activeTool === 'ellipse') {
      this.currentDraft = { type: 'ellipse_draft', cx: world.x, cy: world.y, rx: 0, ry: 0, color: '#38bdf8', width: 1.5 };
      this.requestRender();
    } else if (this.activeTool === 'polygon') {
      this.currentDraft = {
        type: 'polygon_draft',
        cx: world.x,
        cy: world.y,
        r: 0,
        sides: this.polygonSides,
        isStar: this.polygonStarMode,
        angle: -Math.PI / 2,
        color: '#38bdf8',
        width: 1.5,
      };
      this.requestRender();
    } else if (this.activeTool === 'text') {
      const text = prompt('Enter text for CNC engraving:', 'JDPaint');
      if (text) {
        this.saveState();
        const ent = this.ensureEntityMetadata({
          type: 'text',
          x: world.x,
          y: world.y,
          text,
          size: 18,
          fontFamily: 'serif',
          fontWeight: 'bold',
          color: '#38bdf8',
        });
        this.entities.push(ent);
        this.applySymmetryClones(ent);
        this.selectedEntity = ent;
        this.requestRender();
        this.notifyEntitiesChange();
      }
    }
  }

  onMouseMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;

    if (this.isPanning) {
      this.panX = sx - this.panStart.x;
      this.panY = sy - this.panStart.y;
      this.requestRender();
      return;
    }

    const bypassSnap = this.activeTool === 'freehand' || !!this.activeTransformHandle;
    const world = this.screenToWorld(sx, sy, bypassSnap);
    this.currentMousePos = { x: world.x, y: world.y, sx, sy };

    if (this.onCoordinateChange) {
      const snapNote = this.activeSnapTarget ? ` [${this.activeSnapTarget.type}]` : '';
      this.onCoordinateChange(world.x, world.y, snapNote);
    }

    this.updateCursor(world.x, world.y, sx, sy);

    // Path Reshape / Push Brush drag
    if (this.activeTool === 'reshape' && this.isMouseDown && this.isDragging) {
      const raw = this.screenToWorld(sx, sy, true);
      const dx = raw.x - this.dragStart.x;
      const dy = raw.y - this.dragStart.y;
      if (Math.hypot(dx, dy) > 0.05) {
        this.applyReshapeBrush(this.dragStart.x, this.dragStart.y, dx, dy);
        this.dragStart = { ...raw };
        this.requestRender();
      }
      return;
    }

    // Freehand recording (smooth minimal step 0.4mm)
    if (this.activeTool === 'freehand' && this.isMouseDown && this.currentDraft) {
      const pts = this.currentDraft.points;
      const prev = pts[pts.length - 1];
      if (Math.hypot(world.x - prev.x, world.y - prev.y) >= 0.4) {
        pts.push({ x: world.x, y: world.y });
        this.requestRender();
      }
      return;
    }

    if (this.activeTool === 'eraser' && this.isMouseDown) {
      this.eraseAt(world.x, world.y);
      this.requestRender();
      return;
    }

    if (this.isBoxSelecting) {
      this.boxSelectEnd = { x: world.x, y: world.y };
      this.requestRender();
      return;
    }

    // 8-Handle Scale / Rotate Transform Drag (supports Shift aspect lock!)
    if (this.activeTool === 'select' && this.activeTransformHandle && this.transformOriginBounds) {
      this.applyInteractiveTransform(world.x, world.y, e.shiftKey);
      this.requestRender();
      this.notifySelectionChange();
      return;
    }

    // Bezier Handle or Anchor Node Dragging
    if (this.activeTool === 'node' && this.isDragging && this.selectedEntity) {
      if (this.activeBezierHandle) {
        this.moveBezierHandle(this.selectedEntity, this.activeBezierHandle, world.x, world.y, e.altKey);
        this.requestRender();
        return;
      }
      if (this.selectedNodeIndex !== -1) {
        this.moveNode(this.selectedEntity, this.selectedNodeIndex, world.x, world.y, this.selectedHoleIndex);
        this.requestRender();
        return;
      }
    }

    // Entity Move Dragging
    if (this.activeTool === 'select' && this.isDragging && this.selectedEntities.length > 0) {
      const dx = world.x - this.dragStart.x;
      const dy = world.y - this.dragStart.y;
      for (const ent of this.selectedEntities) {
        if (!ent.locked) this.moveEntity(ent, dx, dy);
      }
      this.dragStart = { ...world };
      this.requestRender();
      this.notifySelectionChange();
      return;
    }

    // Active Drafting Live Preview (Shift locks Square, Circle, or 45-degree Line!)
    if (this.currentDraft) {
      if (this.currentDraft.type === 'line') {
        if (e.shiftKey) {
          const ang = Math.atan2(world.y - this.currentDraft.y1, world.x - this.currentDraft.x1);
          const snappedAng = Math.round(ang / (Math.PI / 4)) * (Math.PI / 4);
          const len = Math.hypot(world.x - this.currentDraft.x1, world.y - this.currentDraft.y1);
          this.currentDraft.x2 = this.currentDraft.x1 + Math.cos(snappedAng) * len;
          this.currentDraft.y2 = this.currentDraft.y1 + Math.sin(snappedAng) * len;
        } else {
          this.currentDraft.x2 = world.x;
          this.currentDraft.y2 = world.y;
        }
      } else if (this.currentDraft.type === 'rect') {
        let w = world.x - this.currentDraft.x;
        let h = world.y - this.currentDraft.y;
        if (e.shiftKey) {
          const s = Math.max(Math.abs(w), Math.abs(h));
          w = (w >= 0 ? 1 : -1) * s;
          h = (h >= 0 ? 1 : -1) * s;
        }
        this.currentDraft.w = w;
        this.currentDraft.h = h;
      } else if (this.currentDraft.type === 'circle') {
        this.currentDraft.r = Math.hypot(world.x - this.currentDraft.cx, world.y - this.currentDraft.cy);
      } else if (this.currentDraft.type === 'ellipse_draft') {
        let rx = Math.abs(world.x - this.currentDraft.cx);
        let ry = Math.abs(world.y - this.currentDraft.cy);
        if (e.shiftKey) { rx = ry = Math.max(rx, ry); }
        this.currentDraft.rx = rx;
        this.currentDraft.ry = ry;
      } else if (this.currentDraft.type === 'polygon_draft') {
        this.currentDraft.r = Math.hypot(world.x - this.currentDraft.cx, world.y - this.currentDraft.cy);
        this.currentDraft.angle = e.shiftKey ? -Math.PI / 2 : Math.atan2(world.y - this.currentDraft.cy, world.x - this.currentDraft.cx);
      } else if (this.currentDraft.type === 'arc_draft') {
        if (this.currentDraft.stage === 1) {
          this.currentDraft.pts[1] = { x: world.x, y: world.y };
          this.currentDraft.pts[2] = {
            x: (this.currentDraft.pts[0].x + world.x) / 2,
            y: (this.currentDraft.pts[0].y + world.y) / 2,
          };
        } else if (this.currentDraft.stage === 2) {
          this.currentDraft.pts[2] = { x: world.x, y: world.y };
        }
      }
      this.requestRender();
    } else if (this.activeTool === 'node' && this.selectedEntity) {
      const prevHover = this.hoveredNodeIndex;
      const prevHoleHover = this.hoveredHoleIndex;
      const nt = this.findNodeTarget(this.selectedEntity, world.x, world.y);
      this.hoveredNodeIndex = nt ? nt.nodeIdx : -1;
      this.hoveredHoleIndex = nt ? nt.holeIdx : -1;
      if (prevHover !== this.hoveredNodeIndex || prevHoleHover !== this.hoveredHoleIndex) this.requestRender();
    } else if (this.activeSnapTarget) {
      this.requestRender();
    }
  }

  onMouseUp() {
    const wasDragging = this.isDragging || !!this.activeTransformHandle || !!this.activeBezierHandle;
    this.isPanning = false;
    this.isDragging = false;
    this.isMouseDown = false;
    this.activeBezierHandle = null;
    this.activeTransformHandle = null;
    this.transformOriginBounds = null;
    this.transformSnapshot = null;

    if (this.isBoxSelecting) {
      this.isBoxSelecting = false;
      const minX = Math.min(this.boxSelectStart.x, this.boxSelectEnd.x);
      const maxX = Math.max(this.boxSelectStart.x, this.boxSelectEnd.x);
      const minY = Math.min(this.boxSelectStart.y, this.boxSelectEnd.y);
      const maxY = Math.max(this.boxSelectStart.y, this.boxSelectEnd.y);

      this.selectedEntities = this.entities.filter(
        (ent) => ent.visible !== false && !ent.locked && this.entityIntersectsBox(ent, minX, minY, maxX, maxY)
      );
      this.requestRender();
      this.notifySelectionChange();
      return;
    }

    if (wasDragging) {
      this.notifyEntitiesChange();
    }

    if (this.currentDraft && this.activeTool !== 'polyline' && this.activeTool !== 'pen' && this.activeTool !== 'arc') {
      this.finishCurrentDraft();
    }
  }

  onWheel(e) {
    e.preventDefault();
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
    const newScale = Math.max(0.3, Math.min(30, this.scale * zoomFactor));

    this.panX = mouseX - (mouseX - this.panX) * (newScale / this.scale);
    this.panY = mouseY - (mouseY - this.panY) * (newScale / this.scale);
    this.scale = newScale;

    this.requestRender();
  }

  updateCursor(wx, wy, sx, sy) {
    if (this.isSpaceDown || this.isPanning) {
      this.canvas.style.cursor = this.isPanning ? 'grabbing' : 'grab';
      return;
    }

    const prevHovered = this.hoveredEntity;
    let nextHovered = null;

    if (this.activeTool === 'eraser' || this.activeTool === 'knife' || this.activeTool === 'freehand') {
      this.canvas.style.cursor = 'crosshair';
      if (this.activeTool === 'knife' || this.activeTool === 'eraser') nextHovered = this.findEntityAt(wx, wy);
    } else if (this.activeTool === 'node') {
      if (this.selectedEntity && (this.findBezierHandleAt(this.selectedEntity, wx, wy) || this.findNodeTarget(this.selectedEntity, wx, wy))) {
        this.canvas.style.cursor = 'pointer';
        nextHovered = this.selectedEntity;
      } else {
        this.canvas.style.cursor = 'crosshair';
        nextHovered = this.findEntityAt(wx, wy);
      }
    } else if (this.activeTool === 'select') {
      const h = this.selectedEntities.length > 0 ? this.findTransformHandleAt(sx, sy) : null;
      if (h) {
        const map = { nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize', n: 'ns-resize', s: 'ns-resize', e: 'ew-resize', w: 'ew-resize', rot: 'grab' };
        this.canvas.style.cursor = map[h] || 'pointer';
      } else {
        nextHovered = this.findEntityAt(wx, wy);
        this.canvas.style.cursor = nextHovered ? 'move' : 'default';
      }
    } else {
      this.canvas.style.cursor = 'crosshair';
    }

    if (prevHovered !== nextHovered) {
      this.hoveredEntity = nextHovered;
      this.requestRender();
    }
  }

  // ============================================================
  // Interactive 8-Handle Bounding Box Scale & Rotate
  // ============================================================
  getEntityBounds(ent) {
    if (this._bboxCache && this._bboxCache.has(ent)) {
      return this._bboxCache.get(ent);
    }
    let bounds = null;
    if (ent.type === 'rect') {
      bounds = {
        minX: Math.min(ent.x, ent.x + ent.w),
        maxX: Math.max(ent.x, ent.x + ent.w),
        minY: Math.min(ent.y, ent.y + ent.h),
        maxY: Math.max(ent.y, ent.y + ent.h),
      };
    } else if (ent.type === 'circle') {
      bounds = { minX: ent.cx - ent.r, maxX: ent.cx + ent.r, minY: ent.cy - ent.r, maxY: ent.cy + ent.r };
    } else if (ent.type === 'line') {
      bounds = {
        minX: Math.min(ent.x1, ent.x2), maxX: Math.max(ent.x1, ent.x2),
        minY: Math.min(ent.y1, ent.y2), maxY: Math.max(ent.y1, ent.y2),
      };
    } else if ((ent.type === 'polyline' || ent.type === 'curve') && ent.points && ent.points.length > 0) {
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      const expandPt = (px, py) => {
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
      };
      const rings = ent.type === 'curve'
        ? [CadCanvas.sampleCurvePoints(ent.points, !!ent.isClosed, 6), ...(ent.holes || []).map((h) => CadCanvas.sampleCurvePoints(h, true, 6))]
        : [ent.points, ...(ent.holes || [])];
      for (const ring of rings) {
        for (const p of ring) expandPt(p.x, p.y);
      }
      if (isFinite(minX)) bounds = { minX, maxX, minY, maxY };
    } else if (ent.type === 'text') {
      const fontMm = (ent.size || 16) * 0.4;
      let estW = Math.max(4, (ent.text?.length || 1) * fontMm * 0.62);
      if (this.ctx && typeof this.ctx.measureText === 'function') {
        this.ctx.save();
        this.ctx.font = `${ent.fontWeight || 'bold'} 40px ${ent.fontFamily || 'sans-serif'}`;
        estW = Math.max(4, (this.ctx.measureText(ent.text || '').width / 40) * fontMm);
        this.ctx.restore();
      }
      const estH = Math.max(4, fontMm * 0.9);
      bounds = { minX: ent.x, maxX: ent.x + estW, minY: ent.y - estH, maxY: ent.y + fontMm * 0.15 };
    }
    if (bounds && this._bboxCache) this._bboxCache.set(ent, bounds);
    return bounds;
  }

  getSelectionBounds(list = this.selectedEntities) {
    if (!list || list.length === 0) return null;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const ent of list) {
      const b = this.getEntityBounds(ent);
      if (!b) continue;
      if (b.minX < minX) minX = b.minX;
      if (b.maxX > maxX) maxX = b.maxX;
      if (b.minY < minY) minY = b.minY;
      if (b.maxY > maxY) maxY = b.maxY;
    }
    return isFinite(minX) ? { minX, maxX, minY, maxY } : null;
  }

  getScreenTransformHandles(bounds) {
    if (!bounds) return [];
    const tl = this.worldToScreen(bounds.minX, bounds.minY);
    const br = this.worldToScreen(bounds.maxX, bounds.maxY);
    const x1 = tl.x - 4, y1 = tl.y - 4, x2 = br.x + 4, y2 = br.y + 4;
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    return [
      { id: 'nw', sx: x1, sy: y1 },
      { id: 'n', sx: mx, sy: y1 },
      { id: 'ne', sx: x2, sy: y1 },
      { id: 'w', sx: x1, sy: my },
      { id: 'e', sx: x2, sy: my },
      { id: 'sw', sx: x1, sy: y2 },
      { id: 's', sx: mx, sy: y2 },
      { id: 'se', sx: x2, sy: y2 },
      { id: 'rot', sx: mx, sy: y1 - 18 },
    ];
  }

  findTransformHandleAt(sx, sy) {
    const b = this.getSelectionBounds();
    if (!b) return null;
    const handles = this.getScreenTransformHandles(b);
    for (const h of handles) {
      if (Math.hypot(sx - h.sx, sy - h.sy) <= 8) return h.id;
    }
    return null;
  }

  applyInteractiveTransform(wx, wy, lockAspect = false) {
    const b = this.transformOriginBounds;
    const h = this.activeTransformHandle;
    const snaps = this.transformSnapshot;
    if (!b || !h || !snaps) return;

    const origW = Math.max(0.5, b.maxX - b.minX);
    const origH = Math.max(0.5, b.maxY - b.minY);
    const cx = (b.minX + b.maxX) / 2;
    const cy = (b.minY + b.maxY) / 2;

    if (h === 'rot') {
      const startAng = Math.atan2(this.dragStart.y - cy, this.dragStart.x - cx);
      const currAng = Math.atan2(wy - cy, wx - cx);
      const dTheta = currAng - startAng;
      const cos = Math.cos(dTheta), sin = Math.sin(dTheta);

      this.selectedEntities.forEach((ent, idx) => {
        if (ent.locked) return;
        if (this._bboxCache) this._bboxCache.delete(ent);
        const src = snaps[idx];
        const rotPt = (px, py) => ({
          x: cx + (px - cx) * cos - (py - cy) * sin,
          y: cy + (px - cx) * sin + (py - cy) * cos,
        });
        if (ent.type === 'line') {
          const p1 = rotPt(src.x1, src.y1), p2 = rotPt(src.x2, src.y2);
          ent.x1 = p1.x; ent.y1 = p1.y; ent.x2 = p2.x; ent.y2 = p2.y;
        } else if (ent.type === 'circle') {
          const c = rotPt(src.cx, src.cy);
          ent.cx = c.x; ent.cy = c.y;
        } else if ((ent.type === 'curve' || ent.type === 'polyline') && ent.points) {
          ent.points = src.points.map((p) => CadCanvas.mapAnchorPoint(p, rotPt));
          if (src.holes && src.holes.length > 0) {
            ent.holes = src.holes.map((hole) => hole.map((p) => CadCanvas.mapAnchorPoint(p, rotPt)));
          }
        } else if (ent.type === 'rect') {
          const c = rotPt(src.x + src.w / 2, src.y + src.h / 2);
          ent.x = c.x - src.w / 2; ent.y = c.y - src.h / 2;
        }
      });
      return;
    }

    // Scale around opposite anchor
    let anchorX = b.minX, anchorY = b.minY;
    let newW = origW, newH = origH;

    if (h.includes('w')) { anchorX = b.maxX; newW = Math.max(0.5, b.maxX - wx); }
    else if (h.includes('e')) { anchorX = b.minX; newW = Math.max(0.5, wx - b.minX); }

    if (h.includes('n')) { anchorY = b.maxY; newH = Math.max(0.5, b.maxY - wy); }
    else if (h.includes('s')) { anchorY = b.minY; newH = Math.max(0.5, wy - b.minY); }

    let sx = h.includes('w') || h.includes('e') ? newW / origW : 1;
    let sy = h.includes('n') || h.includes('s') ? newH / origH : 1;

    if (lockAspect && (h.includes('w') || h.includes('e')) && (h.includes('n') || h.includes('s'))) {
      const uniform = Math.max(sx, sy);
      sx = sy = uniform;
    }

    this.selectedEntities.forEach((ent, idx) => {
      if (ent.locked) return;
      if (this._bboxCache) this._bboxCache.delete(ent);
      const src = snaps[idx];
      const mapX = (px) => anchorX + (px - anchorX) * sx;
      const mapY = (py) => anchorY + (py - anchorY) * sy;

      if (ent.type === 'line') {
        ent.x1 = mapX(src.x1); ent.y1 = mapY(src.y1);
        ent.x2 = mapX(src.x2); ent.y2 = mapY(src.y2);
      } else if (ent.type === 'rect') {
        ent.x = mapX(src.x); ent.y = mapY(src.y);
        ent.w = src.w * sx; ent.h = src.h * sy;
      } else if (ent.type === 'circle') {
        ent.cx = mapX(src.cx); ent.cy = mapY(src.cy);
        ent.r = Math.max(0.3, src.r * ((sx + sy) / 2));
      } else if ((ent.type === 'curve' || ent.type === 'polyline') && ent.points) {
        const scalePt = (px, py) => ({ x: mapX(px), y: mapY(py) });
        ent.points = src.points.map((p) => CadCanvas.mapAnchorPoint(p, scalePt));
        if (src.holes && src.holes.length > 0) {
          ent.holes = src.holes.map((hole) => hole.map((p) => CadCanvas.mapAnchorPoint(p, scalePt)));
        }
      } else if (ent.type === 'text') {
        ent.x = mapX(src.x); ent.y = mapY(src.y);
        ent.size = Math.max(4, src.size * ((sx + sy) / 2));
      }
    });
  }

  eraseAt(wx, wy) {
    const hit = this.findEntityAt(wx, wy, 10);
    if (hit && !hit.locked) {
      this.saveState();
      const idx = this.entities.indexOf(hit);
      if (idx !== -1) {
        this.entities.splice(idx, 1);
        const selIdx = this.selectedEntities.indexOf(hit);
        if (selIdx !== -1) this.selectedEntities.splice(selIdx, 1);
        this.requestRender();
        this.notifyEntitiesChange();
        this.notifySelectionChange();
      }
    }
  }

  findNodeTarget(entity, x, y) {
    const tolerance = 9 / this.scale;
    if (entity.type === 'polyline' || entity.type === 'curve') {
      if (entity.points) {
        for (let i = 0; i < entity.points.length; i++) {
          const pt = entity.points[i];
          if (Math.hypot(x - pt.x, y - pt.y) <= tolerance) return { holeIdx: -1, nodeIdx: i };
        }
      }
      if (entity.holes && entity.holes.length > 0) {
        for (let hIdx = 0; hIdx < entity.holes.length; hIdx++) {
          const hole = entity.holes[hIdx];
          for (let i = 0; i < hole.length; i++) {
            const pt = hole[i];
            if (Math.hypot(x - pt.x, y - pt.y) <= tolerance) return { holeIdx: hIdx, nodeIdx: i };
          }
        }
      }
    } else if (entity.type === 'line') {
      if (Math.hypot(x - entity.x1, y - entity.y1) <= tolerance) return { holeIdx: -1, nodeIdx: 0 };
      if (Math.hypot(x - entity.x2, y - entity.y2) <= tolerance) return { holeIdx: -1, nodeIdx: 1 };
    }
    return null;
  }

  findNodeAt(entity, x, y) {
    const nt = this.findNodeTarget(entity, x, y);
    return nt && nt.holeIdx === -1 ? nt.nodeIdx : -1;
  }

  getVisibleBezierAnchors(entity) {
    if (!entity || (entity.type !== 'curve' && entity.type !== 'polyline')) return [];
    const out = [];
    const collectRing = (ring, holeIdx) => {
      if (!ring || ring.length === 0) return;
      const showAll = ring.length <= 24;
      for (let i = 0; i < ring.length; i++) {
        const isSel = holeIdx === this.selectedHoleIndex && (
          i === this.selectedNodeIndex ||
          i === (this.selectedNodeIndex + 1) % ring.length ||
          i === (this.selectedNodeIndex - 1 + ring.length) % ring.length
        );
        if (showAll || isSel) out.push({ ring, holeIdx, nodeIdx: i, pt: ring[i] });
      }
    };
    collectRing(entity.points, -1);
    if (entity.holes && entity.holes.length > 0) {
      entity.holes.forEach((h, hIdx) => collectRing(h, hIdx));
    }
    return out;
  }

  findBezierHandleAt(entity, x, y) {
    const tol = 8 / this.scale;
    for (const item of this.getVisibleBezierAnchors(entity)) {
      const { pt, holeIdx, nodeIdx } = item;
      if (pt.hOut && Math.hypot(x - pt.hOut.x, y - pt.hOut.y) <= tol) {
        return { holeIdx, nodeIdx, which: 'hOut' };
      }
      if (pt.hIn && Math.hypot(x - pt.hIn.x, y - pt.hIn.y) <= tol) {
        return { holeIdx, nodeIdx, which: 'hIn' };
      }
    }
    return null;
  }

  moveBezierHandle(entity, handle, newX, newY, breakSymmetry = false) {
    const ring = handle.holeIdx === -1 ? entity.points : entity.holes?.[handle.holeIdx];
    if (!ring || !ring[handle.nodeIdx]) return;
    if (this._bboxCache) this._bboxCache.delete(entity);
    const pt = ring[handle.nodeIdx];
    pt[handle.which] = { x: Number(newX.toFixed(3)), y: Number(newY.toFixed(3)) };

    if (!pt.corner && !breakSymmetry) {
      const oppKey = handle.which === 'hIn' ? 'hOut' : 'hIn';
      const vx = newX - pt.x, vy = newY - pt.y;
      const len = Math.hypot(vx, vy);
      if (len > 1e-4) {
        const oppLen = pt[oppKey] ? Math.hypot(pt[oppKey].x - pt.x, pt[oppKey].y - pt.y) : len;
        pt[oppKey] = {
          x: Number((pt.x - (vx / len) * oppLen).toFixed(3)),
          y: Number((pt.y - (vy / len) * oppLen).toFixed(3)),
        };
      }
    }
  }

  toggleSelectedNodeCorner(target = null) {
    const ent = this.selectedEntity;
    if (!ent || ent.locked || (ent.type !== 'curve' && ent.type !== 'polyline')) return false;
    const holeIdx = target ? target.holeIdx : this.selectedHoleIndex;
    const nodeIdx = target ? target.nodeIdx : this.selectedNodeIndex;
    const ring = holeIdx === -1 ? ent.points : ent.holes?.[holeIdx];
    if (!ring || !ring[nodeIdx]) return false;

    this.saveState();
    if (this._bboxCache) this._bboxCache.delete(ent);
    const pt = ring[nodeIdx];
    const M = ring.length;
    pt.corner = !pt.corner;
    if (pt.corner) {
      pt.hIn = null;
      pt.hOut = null;
    } else if (M >= 3) {
      const prev = ring[(nodeIdx - 1 + M) % M];
      const next = ring[(nodeIdx + 1) % M];
      const dx = next.x - prev.x, dy = next.y - prev.y;
      const d = Math.hypot(dx, dy) || 1;
      const dIn = Math.hypot(pt.x - prev.x, pt.y - prev.y) * 0.28;
      const dOut = Math.hypot(next.x - pt.x, next.y - pt.y) * 0.28;
      pt.hIn = { x: Number((pt.x - (dx / d) * dIn).toFixed(3)), y: Number((pt.y - (dy / d) * dIn).toFixed(3)) };
      pt.hOut = { x: Number((pt.x + (dx / d) * dOut).toFixed(3)), y: Number((pt.y + (dy / d) * dOut).toFixed(3)) };
    }
    this.selectedHoleIndex = holeIdx;
    this.selectedNodeIndex = nodeIdx;
    this.requestRender();
    this.notifyEntitiesChange();
    return true;
  }

  static distToSegment(p, a, b) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const lenSq = dx * dx + dy * dy;
    if (lenSq === 0) return Math.hypot(p.x - a.x, p.y - a.y);
    let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
  }

  insertNodeOnEdge(entity, x, y) {
    if (entity.locked || (entity.type !== 'polyline' && entity.type !== 'curve')) return -1;
    const tolerance = 7 / this.scale;
    const ringCandidates = [{ ring: entity.points, holeIdx: -1, closed: !!entity.isClosed }];
    if (entity.holes && entity.holes.length > 0) {
      entity.holes.forEach((h, idx) => ringCandidates.push({ ring: h, holeIdx: idx, closed: true }));
    }

    const lerp = (u, v, t) => ({ x: u.x + (v.x - u.x) * t, y: u.y + (v.y - u.y) * t });
    const roundPt = (p) => ({ x: Number(p.x.toFixed(3)), y: Number(p.y.toFixed(3)) });

    for (const cand of ringCandidates) {
      const pts = cand.ring;
      if (!pts || pts.length < 2) continue;
      const segCount = cand.closed ? pts.length : pts.length - 1;
      const isCubic = CadCanvas.hasCubicHandles(pts);

      for (let i = 0; i < segCount; i++) {
        const p1 = pts[i];
        const p2 = pts[(i + 1) % pts.length];
        if (isCubic && (p1.hOut || p2.hIn)) {
          const c1 = p1.hOut || p1;
          const c2 = p2.hIn || p2;
          let bestDist = Infinity, bestT = 0.5;
          for (let s = 1; s <= 12; s++) {
            const t = s / 12, mt = 1 - t;
            const bx = mt * mt * mt * p1.x + 3 * mt * mt * t * c1.x + 3 * mt * t * t * c2.x + t * t * t * p2.x;
            const by = mt * mt * mt * p1.y + 3 * mt * mt * t * c1.y + 3 * mt * t * t * c2.y + t * t * t * p2.y;
            const d = Math.hypot(x - bx, y - by);
            if (d < bestDist) { bestDist = d; bestT = t; }
          }
          if (bestDist <= tolerance) {
            this.saveState();
            const p01 = lerp(p1, c1, bestT);
            const p12 = lerp(c1, c2, bestT);
            const p23 = lerp(c2, p2, bestT);
            const p012 = lerp(p01, p12, bestT);
            const p123 = lerp(p12, p23, bestT);
            const mid = lerp(p012, p123, bestT);
            p1.hOut = roundPt(p01);
            p2.hIn = roundPt(p23);
            pts.splice(i + 1, 0, {
              ...roundPt(mid),
              corner: false,
              hIn: roundPt(p012),
              hOut: roundPt(p123),
            });
            this.selectedHoleIndex = cand.holeIdx;
            this.selectedNodeIndex = i + 1;
            if (this._bboxCache) this._bboxCache.delete(entity);
            this.requestRender();
            this.notifyEntitiesChange();
            return i + 1;
          }
        } else if (CadCanvas.distToSegment({ x, y }, p1, p2) <= tolerance) {
          this.saveState();
          pts.splice(i + 1, 0, { x: Number(x.toFixed(3)), y: Number(y.toFixed(3)), corner: true, hIn: null, hOut: null });
          this.selectedHoleIndex = cand.holeIdx;
          this.selectedNodeIndex = i + 1;
          if (this._bboxCache) this._bboxCache.delete(entity);
          this.requestRender();
          this.notifyEntitiesChange();
          return i + 1;
        }
      }
    }
    return -1;
  }

  /**
   * Smoothly bend/push nearby vertices along any traced outline (clay-like 2D curve sculpting)
   */
  applyReshapeBrush(cx, cy, dx, dy) {
    const rad = this.reshapeBrushRadius || 6;
    const targets = this.selectedEntities.length > 0
      ? this.selectedEntities.filter((e) => !e.locked && e.visible !== false)
      : this.entities.filter((e) => !e.locked && e.visible !== false);

    for (const ent of targets) {
      if (this._bboxCache) this._bboxCache.delete(ent);
      if ((ent.type === 'curve' || ent.type === 'polyline') && ent.points?.length >= 2) {
        const pts = ent.points;
        // Subdivide long segments passing through brush so curve bends cleanly even with sparse nodes
        for (let i = 0; i < pts.length - 1; i++) {
          const segLen = Math.hypot(pts[i + 1].x - pts[i].x, pts[i + 1].y - pts[i].y);
          if (segLen > rad * 0.65 && CadCanvas.distToSegment({ x: cx, y: cy }, pts[i], pts[i + 1]) <= rad) {
            pts.splice(i + 1, 0, {
              x: (pts[i].x + pts[i + 1].x) / 2,
              y: (pts[i].y + pts[i + 1].y) / 2,
            });
            i++;
          }
        }
        // Apply smooth cosine-falloff push
        for (let i = 0; i < pts.length; i++) {
          const d = Math.hypot(pts[i].x - cx, pts[i].y - cy);
          if (d < rad) {
            const w = 0.5 * (1 + Math.cos((d / rad) * Math.PI));
            pts[i].x += dx * w;
            pts[i].y += dy * w;
          }
        }
      } else if (ent.type === 'line') {
        const d1 = Math.hypot(ent.x1 - cx, ent.y1 - cy);
        const d2 = Math.hypot(ent.x2 - cx, ent.y2 - cy);
        if (d1 < rad) { const w = 0.5 * (1 + Math.cos((d1 / rad) * Math.PI)); ent.x1 += dx * w; ent.y1 += dy * w; }
        if (d2 < rad) { const w = 0.5 * (1 + Math.cos((d2 / rad) * Math.PI)); ent.x2 += dx * w; ent.y2 += dy * w; }
      }
    }
  }

  /**
   * Join / Weld selected open curves (e.g. a Knife-trimmed outline + new Pen replacement curve) into one path
   */
  joinSelectedCurves(toleranceMm = 4.0) {
    const curves = this.selectedEntities.filter(
      (e) => !e.locked && (e.type === 'curve' || e.type === 'polyline' || e.type === 'line')
    );
    if (curves.length < 2) return false;

    this.saveState();
    // Convert any line to point array
    const chains = curves.map((ent) => {
      if (ent.type === 'line') {
        return [{ x: ent.x1, y: ent.y1 }, { x: ent.x2, y: ent.y2 }];
      }
      return ent.points.map((p) => ({ ...p }));
    });

    // Greedily chain nearest endpoints
    let merged = chains.shift();
    while (chains.length > 0) {
      const mStart = merged[0];
      const mEnd = merged[merged.length - 1];
      let bestIdx = 0;
      let bestMode = 'end-start';
      let bestDist = Infinity;

      for (let i = 0; i < chains.length; i++) {
        const c = chains[i];
        const cStart = c[0], cEnd = c[c.length - 1];
        const candidates = [
          { mode: 'end-start', d: Math.hypot(mEnd.x - cStart.x, mEnd.y - cStart.y) },
          { mode: 'end-end', d: Math.hypot(mEnd.x - cEnd.x, mEnd.y - cEnd.y) },
          { mode: 'start-end', d: Math.hypot(mStart.x - cEnd.x, mStart.y - cEnd.y) },
          { mode: 'start-start', d: Math.hypot(mStart.x - cStart.x, mStart.y - cStart.y) },
        ];
        for (const cand of candidates) {
          if (cand.d < bestDist) {
            bestDist = cand.d; bestIdx = i; bestMode = cand.mode;
          }
        }
      }

      const nextChain = chains.splice(bestIdx, 1)[0];
      const skipFirst = bestDist <= toleranceMm;
      if (bestMode === 'end-start') {
        merged = merged.concat(skipFirst ? nextChain.slice(1) : nextChain);
      } else if (bestMode === 'end-end') {
        const rev = [...nextChain].reverse();
        merged = merged.concat(skipFirst ? rev.slice(1) : rev);
      } else if (bestMode === 'start-end') {
        merged = (skipFirst ? nextChain.slice(0, -1) : nextChain).concat(merged);
      } else if (bestMode === 'start-start') {
        const rev = [...nextChain].reverse();
        merged = (skipFirst ? rev.slice(0, -1) : rev).concat(merged);
      }
    }

    const closeGap = Math.hypot(merged[0].x - merged[merged.length - 1].x, merged[0].y - merged[merged.length - 1].y);
    const isClosed = closeGap <= toleranceMm;
    if (isClosed && closeGap > 0.05) {
      merged.push({ x: merged[0].x, y: merged[0].y });
    }

    const base = curves[0];
    const joined = this.ensureEntityMetadata({
      type: 'curve',
      points: merged,
      isClosed,
      color: base.color || '#38bdf8',
      width: 1.5,
      name: `${base.name || 'Joined Path'}`,
      relief: { ...(base.relief || {}), openAsStroke: !isClosed },
    });

    // Remove originals, insert joined curve
    this.entities = this.entities.filter((e) => !curves.includes(e));
    this.entities.push(joined);
    this.selectedEntity = joined;
    this.requestRender();
    this.notifyEntitiesChange();
    return true;
  }

  deleteSelectedNode() {
    if (!this.selectedEntity || this.selectedNodeIndex === -1) return;
    const ent = this.selectedEntity;
    if ((ent.type === 'curve' || ent.type === 'polyline')) {
      const isHole = this.selectedHoleIndex >= 0 && ent.holes?.[this.selectedHoleIndex];
      const ring = isHole ? ent.holes[this.selectedHoleIndex] : ent.points;
      if (!ring) return;
      if (isHole && ring.length <= 3) {
        this.saveState();
        ent.holes.splice(this.selectedHoleIndex, 1);
      } else if (ring.length > 2) {
        this.saveState();
        ring.splice(this.selectedNodeIndex, 1);
      } else {
        return;
      }
      if (this._bboxCache) this._bboxCache.delete(ent);
      this.selectedNodeIndex = -1;
      this.selectedHoleIndex = -1;
      this.activeBezierHandle = null;
      this.requestRender();
      this.notifyEntitiesChange();
    }
  }

  breakApartCompound() {
    const targets = this.getTransformTargets().filter((e) => e.holes && e.holes.length > 0);
    if (targets.length === 0) return 0;
    this.saveState();
    let count = 0;
    const newSel = [];
    for (const ent of targets) {
      if (this._bboxCache) this._bboxCache.delete(ent);
      const holes = ent.holes || [];
      holes.forEach((holePts, hIdx) => {
        const cutout = this.ensureEntityMetadata({
          type: ent.type || 'curve',
          points: holePts.map((p) => ({
            x: p.x,
            y: p.y,
            ...(p.corner !== undefined ? { corner: p.corner } : {}),
            ...(p.hIn ? { hIn: { ...p.hIn } } : {}),
            ...(p.hOut ? { hOut: { ...p.hOut } } : {}),
          })),
          holes: [],
          isClosed: true,
          color: ent.color || '#38bdf8',
          fillColor: ent.fillColor || ent.color || '#38bdf8',
          layerGroup: ent.layerGroup,
          width: ent.width || 1.5,
          name: `${ent.name || 'Shape'} Cutout #${hIdx + 1}`,
          relief: { ...(ent.relief || {}) },
        });
        this.entities.push(cutout);
        newSel.push(cutout);
        count++;
      });
      ent.holes = [];
      newSel.push(ent);
    }
    if (newSel.length > 0) this.selectedEntities = newSel;
    this.selectedHoleIndex = -1;
    this.selectedNodeIndex = -1;
    this.requestRender();
    this.notifyEntitiesChange();
    this.notifySelectionChange();
    return count;
  }

  moveNode(entity, nodeIndex, newX, newY, holeIndex = -1) {
    if (this._bboxCache) this._bboxCache.delete(entity);
    if (entity.type === 'polyline' || entity.type === 'curve') {
      const ring = holeIndex >= 0 ? entity.holes?.[holeIndex] : entity.points;
      const pt = ring?.[nodeIndex];
      if (pt) {
        const dx = newX - pt.x, dy = newY - pt.y;
        pt.x = newX; pt.y = newY;
        if (pt.hIn) { pt.hIn.x += dx; pt.hIn.y += dy; }
        if (pt.hOut) { pt.hOut.x += dx; pt.hOut.y += dy; }
      }
    } else if (entity.type === 'line') {
      if (nodeIndex === 0) { entity.x1 = newX; entity.y1 = newY; }
      else { entity.x2 = newX; entity.y2 = newY; }
    }
  }

  moveEntity(entity, dx, dy) {
    if (this._bboxCache) this._bboxCache.delete(entity);
    CadCanvas.transformEntityPoints(entity, (px, py) => ({ x: px + dx, y: py + dy }));
  }

  entityIntersectsBox(ent, minX, minY, maxX, maxY) {
    const b = this.getEntityBounds(ent);
    if (!b) return false;
    return b.minX <= maxX && b.maxX >= minX && b.minY <= maxY && b.maxY >= minY;
  }

  getTransformTargets() {
    if (this.selectedEntities.length > 0) {
      return this.selectedEntities.filter((e) => !e.locked);
    }
    return this.entities.filter((e) => e.visible !== false && !e.locked);
  }

  mirrorHorizontal() {
    const targets = this.getTransformTargets();
    if (targets.length === 0) return;
    this.saveState();
    const bounds = this.getSelectionBounds(targets);
    const cx = bounds ? (bounds.minX + bounds.maxX) / 2 : this.workpieceWidth / 2;
    for (const ent of targets) {
      if (this._bboxCache) this._bboxCache.delete(ent);
      CadCanvas.transformEntityPoints(ent, (px, py) => ({ x: 2 * cx - px, y: py }));
    }
    this.requestRender();
    this.notifyEntitiesChange();
  }

  mirrorVertical() {
    const targets = this.getTransformTargets();
    if (targets.length === 0) return;
    this.saveState();
    const bounds = this.getSelectionBounds(targets);
    const cy = bounds ? (bounds.minY + bounds.maxY) / 2 : this.workpieceHeight / 2;
    for (const ent of targets) {
      if (this._bboxCache) this._bboxCache.delete(ent);
      CadCanvas.transformEntityPoints(ent, (px, py) => ({ x: px, y: 2 * cy - py }));
    }
    this.requestRender();
    this.notifyEntitiesChange();
  }

  rotate90() {
    const targets = this.getTransformTargets();
    if (targets.length === 0) return;
    this.saveState();
    const bounds = this.getSelectionBounds(targets);
    const cx = bounds ? (bounds.minX + bounds.maxX) / 2 : this.workpieceWidth / 2;
    const cy = bounds ? (bounds.minY + bounds.maxY) / 2 : this.workpieceHeight / 2;
    for (const ent of targets) {
      if (this._bboxCache) this._bboxCache.delete(ent);
      if (ent.type === 'rect') {
        const rx = ent.x + ent.w / 2 - cx, ry = ent.y + ent.h / 2 - cy;
        const oldW = ent.w; ent.w = ent.h; ent.h = oldW;
        ent.x = cx - ry - ent.w / 2; ent.y = cy + rx - ent.h / 2;
      } else {
        CadCanvas.transformEntityPoints(ent, (px, py) => ({ x: cx - (py - cy), y: cy + (px - cx) }));
      }
    }
    this.requestRender();
    this.notifyEntitiesChange();
  }

  /**
   * True parallel edge-normal mitered polygon offset (for accurate CNC tool clearance & inlays!)
   * Supports positive (outward) and negative (inward) distances on convex AND concave shapes.
   */
  static computeMiterOffsetPoints(rawPts, isClosed, distMm) {
    if (!rawPts || rawPts.length < 2) return rawPts || [];
    let ring = rawPts;
    if (isClosed && rawPts.length >= 3 && Math.hypot(rawPts[0].x - rawPts[rawPts.length - 1].x, rawPts[0].y - rawPts[rawPts.length - 1].y) < 0.05) {
      ring = rawPts.slice(0, -1);
    }
    const M = ring.length;
    if (M < 2) return rawPts;

    // Determine polygon orientation via shoelace signed area so +distMm always goes outward!
    let area2 = 0;
    for (let i = 0; i < M; i++) {
      const a = ring[i], b = ring[(i + 1) % M];
      area2 += a.x * b.y - b.x * a.y;
    }
    // In screen/canvas Y-down coordinates: clockwise ring has area2 < 0 -> outward normal of edge (dx, dy) is (dy, -dx)/L
    // Wait: if (0,0)->(10,0)->(10,10)->(0,10), dx=10, dy=0 -> area2 = +200 -> inside is +Y, outward is -Y = (dy, -dx) = (0, -10)!
    // Check: clockwise on Y-down? (0,0)->(10,0)->(10,10)->(0,10) IS clockwise visually on Y-down canvas: x*y' = 10*10 - 0 + ... > 0 -> orientSign = area2 >= 0 ? 1 : -1;
    // Test outward for (0,0)->(10,0): right-hand perpendicular is (dy, -dx) = (0, -10) -> points UP (-Y, outward!).
    // If ring was opposite winding (area2 < 0), outward normal flips sign (-1).
    const orient = (!isClosed || area2 >= 0) ? 1 : -1;

    const edgeNormals = [];
    for (let i = 0; i < M; i++) {
      const a = ring[i], b = ring[(i + 1) % M];
      const dx = b.x - a.x, dy = b.y - a.y;
      const L = Math.hypot(dx, dy) || 1;
      edgeNormals.push({ nx: (dy / L) * orient, ny: (-dx / L) * orient });
    }

    const out = [];
    const miterLimit = 2.5;
    for (let i = 0; i < M; i++) {
      if (!isClosed && (i === 0 || i === M - 1)) {
        const n = i === 0 ? edgeNormals[0] : edgeNormals[M - 2];
        out.push({
          x: Number((ring[i].x + n.nx * distMm).toFixed(3)),
          y: Number((ring[i].y + n.ny * distMm).toFixed(3)),
        });
        continue;
      }
      const nPrev = edgeNormals[(i - 1 + M) % M];
      const nNext = edgeNormals[i];
      let bx = nPrev.nx + nNext.nx;
      let by = nPrev.ny + nNext.ny;
      const bLen = Math.hypot(bx, by);
      if (bLen < 1e-4) {
        out.push({
          x: Number((ring[i].x + nNext.nx * distMm).toFixed(3)),
          y: Number((ring[i].y + nNext.ny * distMm).toFixed(3)),
        });
        continue;
      }
      bx /= bLen;
      by /= bLen;
      const dot = Math.max(0.25, nPrev.nx * bx + nPrev.ny * by);
      const offsetScale = Math.min(miterLimit, 1.0 / dot) * distMm;
      out.push({
        x: Number((ring[i].x + bx * offsetScale).toFixed(3)),
        y: Number((ring[i].y + by * offsetScale).toFixed(3)),
      });
    }
    if (isClosed && out.length >= 2) out.push({ ...out[0] });
    return out;
  }

  // True normal parallel offset outward/inward for Rect, Circle, Curve, and Polyline!
  offsetSelected(distMm = 2.0, makeCopy = false) {
    const targets = this.getTransformTargets();
    if (targets.length === 0) return 0;
    this.saveState();
    let count = 0;
    const resultSelection = [];

    for (const srcEnt of targets) {
      const ent = makeCopy ? this.ensureEntityMetadata(JSON.parse(JSON.stringify(srcEnt))) : srcEnt;
      if (makeCopy) {
        delete ent.id;
        ent.name = `${srcEnt.name || 'Layer'} (Offset ${distMm > 0 ? '+' : ''}${distMm}mm)`;
        this.entities.push(ent);
      }
      if (this._bboxCache) this._bboxCache.delete(ent);

      if (ent.type === 'circle') {
        ent.r = Math.max(0.5, ent.r + distMm);
        count++;
      } else if (ent.type === 'rect') {
        const nextW = ent.w + distMm * 2;
        const nextH = ent.h + distMm * 2;
        if (nextW > 0.8 && nextH > 0.8) {
          ent.x -= distMm; ent.y -= distMm;
          ent.w = nextW; ent.h = nextH;
          if (ent.cornerRadius > 0) ent.cornerRadius = Math.max(0, ent.cornerRadius + distMm);
          count++;
        }
      } else if ((ent.type === 'curve' || ent.type === 'polyline') && ent.points && ent.points.length >= 2) {
        const baseRing = ent.type === 'curve' && CadCanvas.hasCubicHandles(ent.points)
          ? CadCanvas.sampleCurvePoints(ent.points, !!ent.isClosed, 8)
          : ent.points;
        ent.points = CadCanvas.computeMiterOffsetPoints(baseRing, !!ent.isClosed, distMm);
        if (ent.holes && ent.holes.length > 0) {
          ent.holes = ent.holes.map((hole) => {
            const sampledHole = ent.type === 'curve' && CadCanvas.hasCubicHandles(hole)
              ? CadCanvas.sampleCurvePoints(hole, true, 8)
              : hole;
            return CadCanvas.computeMiterOffsetPoints(sampledHole, true, -distMm);
          });
        }
        count++;
      }
      resultSelection.push(ent);
    }

    if (makeCopy && resultSelection.length > 0) this.selectedEntities = resultSelection;
    this.requestRender();
    this.notifyEntitiesChange();
    this.notifySelectionChange();
    return count;
  }

  findEntityAt(x, y, customTolerance = 6) {
    const tolerance = customTolerance / this.scale;
    const pt = { x, y };
    for (let i = this.entities.length - 1; i >= 0; i--) {
      const ent = this.entities[i];
      if (ent.visible === false || ent.locked) continue;

      const eb = this.getEntityBounds(ent);
      if (eb && (x < eb.minX - tolerance || x > eb.maxX + tolerance || y < eb.minY - tolerance || y > eb.maxY + tolerance)) {
        continue;
      }

      if (ent.type === 'circle') {
        const d = Math.hypot(x - ent.cx, y - ent.cy);
        if (Math.abs(d - ent.r) <= tolerance || d <= ent.r) return ent;
      } else if (ent.type === 'rect') {
        const minX = Math.min(ent.x, ent.x + ent.w);
        const maxX = Math.max(ent.x, ent.x + ent.w);
        const minY = Math.min(ent.y, ent.y + ent.h);
        const maxY = Math.max(ent.y, ent.y + ent.h);
        if (x >= minX - tolerance && x <= maxX + tolerance && y >= minY - tolerance && y <= maxY + tolerance) {
          return ent;
        }
      } else if (ent.type === 'polyline' || ent.type === 'curve') {
        const pts = ent.points;
        if (!pts || pts.length === 0) continue;
        const rings = [
          ent.type === 'curve' ? CadCanvas.sampleCurvePoints(pts, !!ent.isClosed, 6) : pts,
          ...(ent.holes || []).map((h) => (ent.type === 'curve' ? CadCanvas.sampleCurvePoints(h, true, 6) : h)),
        ];
        let edgeHit = false;
        for (let rIdx = 0; rIdx < rings.length && !edgeHit; rIdx++) {
          const ring = rings[rIdx];
          if (!ring || ring.length === 0) continue;
          const isRingClosed = rIdx > 0 || !!ent.isClosed;
          const segCount = isRingClosed ? ring.length : ring.length - 1;
          for (let k = 0; k < segCount; k++) {
            if (CadCanvas.distToSegment(pt, ring[k], ring[(k + 1) % ring.length]) <= tolerance) {
              edgeHit = true;
              break;
            }
          }
        }
        if (edgeHit) return ent;
        if (ent.isClosed && (this.showFills || this.showColorFills)) {
          let parity = 0;
          for (const ring of rings) {
            if (ring.length >= 3 && CadCanvas.pointInRing(x, y, ring)) parity ^= 1;
          }
          if (parity === 1) return ent;
        }
      } else if (ent.type === 'line') {
        if (CadCanvas.distToSegment(pt, { x: ent.x1, y: ent.y1 }, { x: ent.x2, y: ent.y2 }) <= tolerance) return ent;
      } else if (ent.type === 'text') {
        const tb = this.getEntityBounds(ent);
        if (tb && x >= tb.minX - tolerance && x <= tb.maxX + tolerance && y >= tb.minY - tolerance && y <= tb.maxY + tolerance) {
          return ent;
        }
      }
    }
    return null;
  }

  render() {
    const ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const w = this.canvas.width / dpr;
    const h = this.canvas.height / dpr;
    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = '#07090d';
    ctx.fillRect(0, 0, w, h);

    const origin = this.worldToScreen(0, 0);
    const corner = this.worldToScreen(this.workpieceWidth, this.workpieceHeight);

    // Subtle board elevation shadow & crisp CAD drafting surface
    ctx.save();
    ctx.shadowColor = 'rgba(0, 0, 0, 0.75)';
    ctx.shadowBlur = 18;
    ctx.fillStyle = '#0f131a';
    ctx.fillRect(origin.x, origin.y, corner.x - origin.x, corner.y - origin.y);
    ctx.restore();

    if (this.showRefImage && this.refImage) {
      ctx.save();
      ctx.globalAlpha = Math.max(0.05, Math.min(1.0, this.refImageOpacity));
      const imgPos = this.worldToScreen(this.refImageBounds.x, this.refImageBounds.y);
      const imgW = this.refImageBounds.w * this.scale;
      const imgH = this.refImageBounds.h * this.scale;
      ctx.drawImage(this.refImage, imgPos.x, imgPos.y, imgW, imgH);
      ctx.restore();
    }

    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 1.2;
    ctx.strokeRect(origin.x, origin.y, corner.x - origin.x, corner.y - origin.y);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px monospace';
    ctx.fillText(`${this.workpieceWidth}mm × ${this.workpieceHeight}mm`, origin.x + 6, origin.y + 14);

    if (this.gridVisible && this.scale > 0.9) {
      const minorStep = this.visualGridStep || 5;
      const majorStep = minorStep * 4; // 20mm major grid
      const startX = Math.floor(-this.panX / this.scale / minorStep) * minorStep;
      const endX = Math.ceil((w - this.panX) / this.scale / minorStep) * minorStep;
      for (let x = startX; x <= endX; x += minorStep) {
        const isMajor = Math.abs(Math.round(x / majorStep) * majorStep - x) < 1e-4;
        if (!isMajor && this.scale < 1.8) continue;
        ctx.strokeStyle = isMajor ? 'rgba(148, 163, 184, 0.16)' : 'rgba(148, 163, 184, 0.06)';
        ctx.lineWidth = isMajor ? 0.8 : 0.5;
        const s = this.worldToScreen(x, 0);
        ctx.beginPath(); ctx.moveTo(s.x, 0); ctx.lineTo(s.x, h); ctx.stroke();
      }

      const startY = Math.floor(-this.panY / this.scale / minorStep) * minorStep;
      const endY = Math.ceil((h - this.panY) / this.scale / minorStep) * minorStep;
      for (let y = startY; y <= endY; y += minorStep) {
        const isMajor = Math.abs(Math.round(y / majorStep) * majorStep - y) < 1e-4;
        if (!isMajor && this.scale < 1.8) continue;
        ctx.strokeStyle = isMajor ? 'rgba(148, 163, 184, 0.16)' : 'rgba(148, 163, 184, 0.06)';
        ctx.lineWidth = isMajor ? 0.8 : 0.5;
        const s = this.worldToScreen(0, y);
        ctx.beginPath(); ctx.moveTo(0, s.y); ctx.lineTo(w, s.y); ctx.stroke();
      }
    }

    // Coordinate Origin Crosshair + Edge Ruler Ticks
    ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(origin.x + 35, origin.y); ctx.stroke();
    ctx.strokeStyle = '#10b981';
    ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(origin.x, origin.y + 35); ctx.stroke();

    if (this.scale > 1.3) {
      ctx.save();
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.45)';
      ctx.fillStyle = 'rgba(148, 163, 184, 0.7)';
      ctx.font = '9px monospace';
      const tickStep = this.scale < 2.2 ? 20 : 10;
      for (let x = tickStep; x < this.workpieceWidth; x += tickStep) {
        const sp = this.worldToScreen(x, 0);
        ctx.beginPath(); ctx.moveTo(sp.x, origin.y); ctx.lineTo(sp.x, origin.y - 4); ctx.stroke();
        ctx.fillText(String(x), sp.x - 5, origin.y - 6);
      }
      for (let y = tickStep; y < this.workpieceHeight; y += tickStep) {
        const sp = this.worldToScreen(0, y);
        ctx.beginPath(); ctx.moveTo(origin.x, sp.y); ctx.lineTo(origin.x - 4, sp.y); ctx.stroke();
        ctx.fillText(String(y), origin.x - 18, sp.y + 3);
      }
      ctx.restore();
    }

    // Symmetry Guide Axes when Symmetry Drawing Mode is ON
    if (this.symmetryMode && this.symmetryMode !== 'none') {
      const cxS = this.worldToScreen(this.workpieceWidth / 2, 0);
      const cyS = this.worldToScreen(0, this.workpieceHeight / 2);
      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = 'rgba(232, 121, 249, 0.65)';
      ctx.lineWidth = 1.2;
      if (this.symmetryMode === 'mirrorX' || this.symmetryMode === 'quad') {
        ctx.beginPath(); ctx.moveTo(cxS.x, origin.y); ctx.lineTo(cxS.x, corner.y); ctx.stroke();
      }
      if (this.symmetryMode === 'mirrorY' || this.symmetryMode === 'quad') {
        ctx.beginPath(); ctx.moveTo(origin.x, cyS.y); ctx.lineTo(corner.x, cyS.y); ctx.stroke();
      }
      ctx.restore();
    }

    // Subtle Hover Glow under hovered entity
    if (this.hoveredEntity && this.hoveredEntity.visible !== false && !this.selectedEntities.includes(this.hoveredEntity)) {
      ctx.save();
      ctx.globalAlpha = 0.5;
      this.drawEntity(ctx, { ...this.hoveredEntity, color: '#38bdf8', width: (this.hoveredEntity.width || 1.5) + 2.5 }, false);
      ctx.restore();
    }

    // Render Entities
    for (const ent of this.entities) {
      if (ent.visible === false) continue;
      const isSelected = this.selectedEntities.includes(ent);
      this.drawEntity(ctx, ent, isSelected);
    }

    // Render Unified Selection Transform Box (8 Resize Handles + Rotate Handle) in Select mode
    if (this.activeTool === 'select' && this.selectedEntities.length > 0) {
      const selBounds = this.getSelectionBounds();
      if (selBounds) {
        this.drawSelectionTransformBox(ctx, selBounds);
        if (this.activeTransformHandle) {
          const bw = selBounds.maxX - selBounds.minX;
          const bh = selBounds.maxY - selBounds.minY;
          this.drawHudPill(ctx, this.currentMousePos.sx + 14, this.currentMousePos.sy - 14, `W: ${bw.toFixed(1)}  H: ${bh.toFixed(1)} mm`, '#fbbf24');
        }
      }
    }

    // Render Active Draft
    if (this.currentDraft) {
      this.drawDraftPreview(ctx, this.currentDraft);
    }

    // Box Selection Marquee
    if (this.isBoxSelecting) {
      const p1 = this.worldToScreen(this.boxSelectStart.x, this.boxSelectStart.y);
      const p2 = this.worldToScreen(this.boxSelectEnd.x, this.boxSelectEnd.y);
      const bx = Math.min(p1.x, p2.x), by = Math.min(p1.y, p2.y);
      const bw = Math.abs(p2.x - p1.x), bh = Math.abs(p2.y - p1.y);
      ctx.save();
      ctx.fillStyle = 'rgba(56, 189, 248, 0.15)';
      ctx.fillRect(bx, by, bw, bh);
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(bx, by, bw, bh);
      ctx.restore();
    }

    // OSNAP indicator
    if (this.activeSnapTarget) {
      const sp = this.worldToScreen(this.activeSnapTarget.x, this.activeSnapTarget.y);
      ctx.save();
      ctx.strokeStyle = this.activeSnapTarget.type === 'Close Path' ? '#10b981' : '#f59e0b';
      ctx.lineWidth = 2;
      ctx.strokeRect(sp.x - 6, sp.y - 6, 12, 12);
      ctx.fillStyle = this.activeSnapTarget.type === 'Close Path' ? '#34d399' : '#fbbf24';
      ctx.font = '11px sans-serif';
      ctx.fillText(this.activeSnapTarget.type, sp.x + 10, sp.y - 4);
      ctx.restore();
    }

    // Tool cursor ring for Eraser, Knife, & Path Reshape Brush
    if ((this.activeTool === 'eraser' || this.activeTool === 'knife' || this.activeTool === 'reshape') && this.currentMousePos.sx) {
      ctx.save();
      ctx.strokeStyle = this.activeTool === 'reshape' ? '#14b8a6' : this.activeTool === 'knife' ? '#f59e0b' : '#ef4444';
      ctx.lineWidth = 1.5;
      const rPx = this.activeTool === 'reshape' ? (this.reshapeBrushRadius * this.scale) : 10;
      ctx.beginPath();
      ctx.arc(this.currentMousePos.sx, this.currentMousePos.sy, Math.max(6, rPx), 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    ctx.restore();
  }

  drawHudPill(ctx, sx, sy, text, color = '#fde047') {
    if (!sx || !sy) return;
    ctx.save();
    ctx.font = 'bold 11px monospace';
    const w = ctx.measureText(text).width + 12;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.86)';
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.4)';
    ctx.lineWidth = 1;
    ctx.fillRect(sx, sy - 14, w, 18);
    ctx.strokeRect(sx, sy - 14, w, 18);
    ctx.fillStyle = color;
    ctx.fillText(text, sx + 6, sy - 1);
    ctx.restore();
  }

  drawDraftPreview(ctx, draft) {
    const mx = this.currentMousePos.sx || 0;
    const my = this.currentMousePos.sy || 0;

    if (draft.type === 'freehand') {
      this.drawEntity(ctx, { type: 'polyline', points: draft.points, color: '#38bdf8', width: 2 }, true);
      return;
    }
    if (draft.type === 'ellipse_draft') {
      const c = this.worldToScreen(draft.cx, draft.cy);
      ctx.save();
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.ellipse(c.x, c.y, Math.max(1, draft.rx * this.scale), Math.max(1, draft.ry * this.scale), 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
      this.drawHudPill(ctx, mx + 12, my - 12, `W: ${(draft.rx * 2).toFixed(1)}  H: ${(draft.ry * 2).toFixed(1)} mm`);
      return;
    }
    if (draft.type === 'polygon_draft') {
      const pts = CadCanvas.buildPolygonPoints(draft.cx, draft.cy, draft.r, draft.sides, draft.isStar, draft.angle);
      this.drawEntity(ctx, { type: 'polyline', points: pts, isClosed: true, color: '#fbbf24', width: 1.5 }, true);
      this.drawHudPill(ctx, mx + 12, my - 12, `R: ${draft.r.toFixed(1)} mm (${draft.sides} sides)`);
      return;
    }
    if (draft.type === 'arc_draft') {
      const pts = CadCanvas.sampleQuadraticArc(draft.pts[0], draft.pts[1], draft.pts[2], 24);
      this.drawEntity(ctx, { type: 'curve', points: pts, isClosed: false, color: '#fbbf24', width: 1.5 }, true);
      const hint = draft.stage === 1 ? 'Click Arc End Point' : 'Move & Click Arc Curve Bulge';
      this.drawHudPill(ctx, mx + 12, my - 12, hint);
      return;
    }

    // Highlight existing curve when Pen is anchored onto it for direct span reshaping!
    if (draft._targetEnt) {
      ctx.save();
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.7)';
      ctx.lineWidth = 3.5;
      this.drawEntity(ctx, draft._targetEnt, false);
      ctx.restore();
    }

    // Smooth Live Bezier Curve Preview through current cursor for Pen Tool!
    if (draft.type === 'curve' && draft.points.length >= 1) {
      const previewPts = [...draft.points, { x: this.currentMousePos.x, y: this.currentMousePos.y }];
      this.drawEntity(ctx, { ...draft, points: previewPts }, true);
      // Draw crisp placed vertices on draft
      for (let i = 0; i < draft.points.length; i++) {
        const sp = this.worldToScreen(draft.points[i].x, draft.points[i].y);
        const isStartClose = i === 0 && draft.points.length >= 3 && Math.hypot(this.currentMousePos.x - draft.points[0].x, this.currentMousePos.y - draft.points[0].y) <= 12 / this.scale;
        ctx.save();
        ctx.fillStyle = isStartClose ? '#10b981' : (i === 0 ? '#34d399' : '#ffffff');
        ctx.strokeStyle = '#0f172a';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(sp.x, sp.y, isStartClose ? 6.5 : 3.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      }
    } else {
      this.drawEntity(ctx, draft, true);
    }

    // Dimensional HUDs for Line, Rect, Circle, & Pen/Polyline
    if (draft.type === 'rect') {
      this.drawHudPill(ctx, mx + 12, my - 12, `W: ${Math.abs(draft.w).toFixed(1)}  H: ${Math.abs(draft.h).toFixed(1)} mm`);
    } else if (draft.type === 'circle') {
      this.drawHudPill(ctx, mx + 12, my - 12, `R: ${draft.r.toFixed(1)}  (Ø ${(draft.r * 2).toFixed(1)} mm)`);
    } else if (draft.type === 'line') {
      const len = Math.hypot(draft.x2 - draft.x1, draft.y2 - draft.y1);
      const ang = (Math.atan2(draft.y2 - draft.y1, draft.x2 - draft.x1) * 180 / Math.PI).toFixed(1);
      this.drawHudPill(ctx, mx + 12, my - 12, `L: ${len.toFixed(1)} mm  ∠ ${ang}°`);
    } else if (draft.points && draft.points.length > 0) {
      const lastPt = draft.points[draft.points.length - 1];
      const spLast = this.worldToScreen(lastPt.x, lastPt.y);
      const spMouse = this.worldToScreen(this.currentMousePos.x, this.currentMousePos.y);

      if (draft.type === 'polyline') {
        ctx.save();
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 1.3;
        ctx.beginPath(); ctx.moveTo(spLast.x, spLast.y); ctx.lineTo(spMouse.x, spMouse.y); ctx.stroke();
        ctx.restore();
      }

      const dist = Math.hypot(this.currentMousePos.x - lastPt.x, this.currentMousePos.y - lastPt.y);
      const label = draft._targetEnt
        ? `Reshape ${dist.toFixed(1)}mm — Click outline + R-Click/Enter`
        : `L: ${dist.toFixed(1)}mm (R-Click / Enter to finish)`;
      this.drawHudPill(ctx, spMouse.x + 12, spMouse.y - 12, label, draft._targetEnt ? '#34d399' : '#fde047');
    }
  }

  /**
   * Build smooth rounded-rectangle polygon points (for CAM, SVG export, and Puffing)
   */
  static buildRoundedRectPoints(x, y, w, h, radius = 0, segsPerCorner = 6) {
    const x0 = Math.min(x, x + w), y0 = Math.min(y, y + h);
    const rw = Math.abs(w), rh = Math.abs(h);
    const r = Math.max(0, Math.min(radius || 0, rw / 2, rh / 2));
    if (r < 0.05) {
      return [
        { x: x0, y: y0 },
        { x: x0 + rw, y: y0 },
        { x: x0 + rw, y: y0 + rh },
        { x: x0, y: y0 + rh },
        { x: x0, y: y0 },
      ];
    }
    const corners = [
      { cx: x0 + rw - r, cy: y0 + r, a0: -Math.PI / 2, a1: 0 },
      { cx: x0 + rw - r, cy: y0 + rh - r, a0: 0, a1: Math.PI / 2 },
      { cx: x0 + r, cy: y0 + rh - r, a0: Math.PI / 2, a1: Math.PI },
      { cx: x0 + r, cy: y0 + r, a0: Math.PI, a1: Math.PI * 1.5 },
    ];
    const pts = [];
    for (const c of corners) {
      for (let i = 0; i <= segsPerCorner; i++) {
        const a = c.a0 + ((c.a1 - c.a0) * i) / segsPerCorner;
        pts.push({
          x: Number((c.cx + Math.cos(a) * r).toFixed(3)),
          y: Number((c.cy + Math.sin(a) * r).toFixed(3)),
        });
      }
    }
    if (pts.length > 0) pts.push({ ...pts[0] });
    return pts;
  }

  static pointInRing(x, y, ring) {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const xi = ring[i].x, yi = ring[i].y;
      const xj = ring[j].x, yj = ring[j].y;
      if (((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / ((yj - yi) || 1e-12) + xi)) {
        inside = !inside;
      }
    }
    return inside;
  }

  static normalizeClosedRing(points, isClosed) {
    if (isClosed && points.length >= 3 && Math.hypot(points[0].x - points[points.length - 1].x, points[0].y - points[points.length - 1].y) < 0.05) {
      return points.slice(0, -1);
    }
    return points;
  }

  /**
   * Trace a smooth open OR closed Cubic Bezier / Quadratic B-spline ring onto any Canvas2D context.
   * Ensures ZERO cusp/kink at closed loop seam!
   */
  static traceSmoothCanvasPath(ctx, points, isClosed = false, mapX = (x) => x, mapY = (y) => y) {
    if (!points || points.length < 2) return;
    if (CadCanvas.hasCubicHandles(points)) {
      const ring = CadCanvas.normalizeClosedRing(points, isClosed);
      const M = ring.length;
      if (M < 2) return;
      ctx.moveTo(mapX(ring[0].x), mapY(ring[0].y));
      const segCount = isClosed ? M : M - 1;
      for (let i = 0; i < segCount; i++) {
        const a = ring[i];
        const b = ring[(i + 1) % M];
        if (!a.hOut && !b.hIn) {
          ctx.lineTo(mapX(b.x), mapY(b.y));
        } else {
          const c1 = a.hOut || a;
          const c2 = b.hIn || b;
          ctx.bezierCurveTo(mapX(c1.x), mapY(c1.y), mapX(c2.x), mapY(c2.y), mapX(b.x), mapY(b.y));
        }
      }
      if (isClosed) ctx.closePath();
      return;
    }

    const ring = CadCanvas.normalizeClosedRing(points, isClosed);
    const M = ring.length;
    if (M === 2 || !isClosed) {
      ctx.moveTo(mapX(points[0].x), mapY(points[0].y));
      if (points.length === 2) {
        ctx.lineTo(mapX(points[1].x), mapY(points[1].y));
      } else {
        for (let i = 1; i < points.length - 1; i++) {
          const pi = points[i], pnext = points[i + 1];
          ctx.quadraticCurveTo(mapX(pi.x), mapY(pi.y), mapX((pi.x + pnext.x) / 2), mapY((pi.y + pnext.y) / 2));
        }
        ctx.lineTo(mapX(points[points.length - 1].x), mapY(points[points.length - 1].y));
      }
      if (isClosed) ctx.closePath();
      return;
    }

    // Periodic closed B-spline: starts at mid(ring[M-1], ring[0]) and wraps seamlessly
    const startMidX = (ring[M - 1].x + ring[0].x) / 2;
    const startMidY = (ring[M - 1].y + ring[0].y) / 2;
    ctx.moveTo(mapX(startMidX), mapY(startMidY));
    for (let i = 0; i < M; i++) {
      const pi = ring[i];
      const pnext = ring[(i + 1) % M];
      ctx.quadraticCurveTo(mapX(pi.x), mapY(pi.y), mapX((pi.x + pnext.x) / 2), mapY((pi.y + pnext.y) / 2));
    }
    ctx.closePath();
  }

  /**
   * Sample smooth Cubic Bezier or Quadratic B-spline points in world mm (for CAM, DXF, and uniform exports)
   */
  static sampleCurvePoints(points, isClosed = false, segsPerSpan = 8) {
    if (!points || points.length < 2) return points ? [...points] : [];
    if (CadCanvas.hasCubicHandles(points)) {
      const ring = CadCanvas.normalizeClosedRing(points, isClosed);
      const M = ring.length;
      if (M < 2) return [...points];
      const out = [{ x: Number(ring[0].x.toFixed(3)), y: Number(ring[0].y.toFixed(3)) }];
      const segCount = isClosed ? M : M - 1;
      for (let i = 0; i < segCount; i++) {
        const a = ring[i];
        const b = ring[(i + 1) % M];
        if (!a.hOut && !b.hIn) {
          out.push({ x: Number(b.x.toFixed(3)), y: Number(b.y.toFixed(3)) });
        } else {
          const c1 = a.hOut || a;
          const c2 = b.hIn || b;
          for (let s = 1; s <= segsPerSpan; s++) {
            const t = s / segsPerSpan;
            const mt = 1 - t;
            const mt2 = mt * mt;
            const t2 = t * t;
            out.push({
              x: Number((mt2 * mt * a.x + 3 * mt2 * t * c1.x + 3 * mt * t2 * c2.x + t2 * t * b.x).toFixed(3)),
              y: Number((mt2 * mt * a.y + 3 * mt2 * t * c1.y + 3 * mt * t2 * c2.y + t2 * t * b.y).toFixed(3)),
            });
          }
        }
      }
      return out;
    }

    if (points.length === 2) return [{ ...points[0] }, { ...points[1] }];

    const out = [];
    const pushQuad = (ax, ay, cx, cy, bx, by) => {
      for (let s = 1; s <= segsPerSpan; s++) {
        const t = s / segsPerSpan;
        const mt = 1 - t;
        out.push({
          x: Number((mt * mt * ax + 2 * mt * t * cx + t * t * bx).toFixed(3)),
          y: Number((mt * mt * ay + 2 * mt * t * cy + t * t * by).toFixed(3)),
        });
      }
    };

    if (isClosed) {
      const ring = CadCanvas.normalizeClosedRing(points, true);
      const M = ring.length;
      if (M < 3) return [...points];
      let cx0 = (ring[M - 1].x + ring[0].x) / 2;
      let cy0 = (ring[M - 1].y + ring[0].y) / 2;
      out.push({ x: Number(cx0.toFixed(3)), y: Number(cy0.toFixed(3)) });
      for (let i = 0; i < M; i++) {
        const pi = ring[i];
        const pnext = ring[(i + 1) % M];
        const mx = (pi.x + pnext.x) / 2;
        const my = (pi.y + pnext.y) / 2;
        pushQuad(cx0, cy0, pi.x, pi.y, mx, my);
        cx0 = mx; cy0 = my;
      }
      return out;
    }

    let curX = points[0].x, curY = points[0].y;
    out.push({ x: curX, y: curY });
    for (let i = 1; i < points.length - 1; i++) {
      const pi = points[i], pnext = points[i + 1];
      const mx = (pi.x + pnext.x) / 2;
      const my = (pi.y + pnext.y) / 2;
      pushQuad(curX, curY, pi.x, pi.y, mx, my);
      curX = mx; curY = my;
    }
    const last = points[points.length - 1];
    out.push({ x: last.x, y: last.y });
    return out;
  }

  /**
   * Parse external SVG markup into normalized 2D CAD vector entities scaled cleanly into workpiece
   */
  static parseSVGToEntities(svgText, workW = 100, workH = 100) {
    if (!svgText || typeof svgText !== 'string') return [];
    const entities = [];
    const attr = (tag, name) => {
      const m = tag.match(new RegExp(`\\b${name}\\s*=\\s*["']([^"']*)["']`, 'i'));
      return m ? m[1] : null;
    };
    const numAttr = (tag, name, fallback = 0) => {
      const v = attr(tag, name);
      if (v === null) return fallback;
      const n = parseFloat(v);
      return isFinite(n) ? n : fallback;
    };

    // Parse <rect>, <circle>, <ellipse>, <line>, <polyline>, <polygon>, <path>
    const tagRegex = /<(rect|circle|ellipse|line|polyline|polygon|path)\b([^>]*?)\/?\s*>/gi;
    let m;
    while ((m = tagRegex.exec(svgText)) !== null) {
      const tag = m[1].toLowerCase();
      const attrs = m[2];
      const color = '#38bdf8';

      if (tag === 'rect') {
        const w = numAttr(attrs, 'width', 0);
        const h = numAttr(attrs, 'height', 0);
        const rx = numAttr(attrs, 'rx', numAttr(attrs, 'ry', 0));
        if (w > 0 && h > 0) {
          entities.push({ type: 'rect', x: numAttr(attrs, 'x', 0), y: numAttr(attrs, 'y', 0), w, h, cornerRadius: rx, isClosed: true, color, width: 1.5, source: 'svg-import' });
        }
      } else if (tag === 'circle') {
        const r = numAttr(attrs, 'r', 0);
        if (r > 0) entities.push({ type: 'circle', cx: numAttr(attrs, 'cx', 0), cy: numAttr(attrs, 'cy', 0), r, color, width: 1.5, source: 'svg-import' });
      } else if (tag === 'ellipse') {
        const cx = numAttr(attrs, 'cx', 0), cy = numAttr(attrs, 'cy', 0);
        const rx = numAttr(attrs, 'rx', 0), ry = numAttr(attrs, 'ry', 0);
        if (rx > 0 && ry > 0) {
          const pts = [];
          for (let i = 0; i <= 36; i++) {
            const a = (i / 36) * Math.PI * 2;
            pts.push({ x: cx + Math.cos(a) * rx, y: cy + Math.sin(a) * ry });
          }
          entities.push({ type: 'curve', shapeKind: 'Ellipse', points: pts, isClosed: true, color, width: 1.5, source: 'svg-import' });
        }
      } else if (tag === 'line') {
        entities.push({ type: 'line', x1: numAttr(attrs, 'x1', 0), y1: numAttr(attrs, 'y1', 0), x2: numAttr(attrs, 'x2', 0), y2: numAttr(attrs, 'y2', 0), color, width: 1.5, source: 'svg-import' });
      } else if (tag === 'polygon' || tag === 'polyline') {
        const rawPts = (attr(attrs, 'points') || '').trim().split(/[\s,]+/).map(Number).filter(isFinite);
        const pts = [];
        for (let i = 0; i + 1 < rawPts.length; i += 2) pts.push({ x: rawPts[i], y: rawPts[i + 1] });
        if (pts.length >= 2) {
          const isClosed = tag === 'polygon';
          if (isClosed) pts.push({ ...pts[0] });
          entities.push({ type: 'polyline', points: pts, isClosed, color, width: 1.5, source: 'svg-import' });
        }
      } else if (tag === 'path') {
        const d = attr(attrs, 'd');
        if (d) {
          const subPaths = CadCanvas.parseSvgPathD(d);
          for (const sp of subPaths) {
            if (sp.points.length >= 2) {
              entities.push({ type: 'curve', points: sp.points, isClosed: sp.isClosed, color, width: 1.5, source: 'svg-import' });
            }
          }
        }
      }
    }

    if (entities.length === 0) return [];

    // Compute raw bounds & normalize into workpiece center with 6mm margin
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    const expandPt = (px, py) => {
      if (px < minX) minX = px; if (px > maxX) maxX = px;
      if (py < minY) minY = py; if (py > maxY) maxY = py;
    };
    for (const e of entities) {
      if (e.type === 'rect') { expandPt(e.x, e.y); expandPt(e.x + e.w, e.y + e.h); }
      else if (e.type === 'circle') { expandPt(e.cx - e.r, e.cy - e.r); expandPt(e.cx + e.r, e.cy + e.r); }
      else if (e.type === 'line') { expandPt(e.x1, e.y1); expandPt(e.x2, e.y2); }
      else if (e.points) e.points.forEach((p) => expandPt(p.x, p.y));
    }
    const rawW = Math.max(1e-3, maxX - minX);
    const rawH = Math.max(1e-3, maxY - minY);
    const margin = Math.min(8, Math.min(workW, workH) * 0.08);
    const scale = Math.min((workW - margin * 2) / rawW, (workH - margin * 2) / rawH);
    const offX = (workW - rawW * scale) / 2 - minX * scale;
    const offY = (workH - rawH * scale) / 2 - minY * scale;
    const mx = (v) => Number((v * scale + offX).toFixed(3));
    const my = (v) => Number((v * scale + offY).toFixed(3));

    const palette = ['#38bdf8', '#34d399', '#fbbf24', '#f472b6', '#a78bfa', '#22d3ee'];
    entities.forEach((e, idx) => {
      e.name = `SVG Layer #${idx + 1}`;
      e.color = palette[idx % palette.length];
      if (e.type === 'rect') {
        e.x = mx(e.x); e.y = my(e.y);
        e.w = Number((e.w * scale).toFixed(3)); e.h = Number((e.h * scale).toFixed(3));
        if (e.cornerRadius) e.cornerRadius = Number((e.cornerRadius * scale).toFixed(3));
      } else if (e.type === 'circle') {
        e.cx = mx(e.cx); e.cy = my(e.cy); e.r = Number((e.r * scale).toFixed(3));
      } else if (e.type === 'line') {
        e.x1 = mx(e.x1); e.y1 = my(e.y1); e.x2 = mx(e.x2); e.y2 = my(e.y2);
      } else if (e.points) {
        e.points = e.points.map((p) => ({ x: mx(p.x), y: my(p.y) }));
      }
    });
    return entities;
  }

  static parseSvgPathD(d) {
    const tokens = d.match(/[a-zA-Z]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?/g);
    if (!tokens) return [];
    const subPaths = [];
    let pts = [];
    let cx = 0, cy = 0, sx = 0, sy = 0, lastCtrlX = 0, lastCtrlY = 0;
    let cmd = '';
    let i = 0;
    const nextNum = () => (i < tokens.length && !/^[a-zA-Z]$/.test(tokens[i])) ? parseFloat(tokens[i++]) : NaN;
    const pushPt = (x, y) => { cx = x; cy = y; pts.push({ x, y }); };

    while (i < tokens.length) {
      if (/^[a-zA-Z]$/.test(tokens[i])) {
        cmd = tokens[i++];
        if (cmd === 'Z' || cmd === 'z') {
          if (pts.length >= 2) {
            if (Math.hypot(cx - sx, cy - sy) > 0.01) pts.push({ x: sx, y: sy });
            subPaths.push({ points: pts, isClosed: true });
          }
          pts = []; cx = sx; cy = sy;
          continue;
        }
      }
      const isRel = cmd === cmd.toLowerCase();
      const C = cmd.toUpperCase();
      if (C === 'M') {
        const nx = nextNum(), ny = nextNum();
        if (!isFinite(nx) || !isFinite(ny)) break;
        if (pts.length >= 2) subPaths.push({ points: pts, isClosed: false });
        pts = [];
        const x = isRel ? cx + nx : nx, y = isRel ? cy + ny : ny;
        sx = x; sy = y; pushPt(x, y);
        cmd = isRel ? 'l' : 'L';
      } else if (C === 'L') {
        const nx = nextNum(), ny = nextNum();
        if (!isFinite(nx) || !isFinite(ny)) break;
        pushPt(isRel ? cx + nx : nx, isRel ? cy + ny : ny);
      } else if (C === 'H') {
        const nx = nextNum(); if (!isFinite(nx)) break;
        pushPt(isRel ? cx + nx : nx, cy);
      } else if (C === 'V') {
        const ny = nextNum(); if (!isFinite(ny)) break;
        pushPt(cx, isRel ? cy + ny : ny);
      } else if (C === 'C') {
        const x1 = nextNum(), y1 = nextNum(), x2 = nextNum(), y2 = nextNum(), x3 = nextNum(), y3 = nextNum();
        if (!isFinite(x3) || !isFinite(y3)) break;
        const ax = cx, ay = cy;
        const c1x = isRel ? ax + x1 : x1, c1y = isRel ? ay + y1 : y1;
        const c2x = isRel ? ax + x2 : x2, c2y = isRel ? ay + y2 : y2;
        const ex = isRel ? ax + x3 : x3, ey = isRel ? ay + y3 : y3;
        for (let s = 1; s <= 8; s++) {
          const t = s / 8, mt = 1 - t;
          pts.push({
            x: mt * mt * mt * ax + 3 * mt * mt * t * c1x + 3 * mt * t * t * c2x + t * t * t * ex,
            y: mt * mt * mt * ay + 3 * mt * mt * t * c1y + 3 * mt * t * t * c2y + t * t * t * ey,
          });
        }
        lastCtrlX = c2x; lastCtrlY = c2y; cx = ex; cy = ey;
      } else if (C === 'S' || C === 'Q') {
        if (C === 'Q') {
          const x1 = nextNum(), y1 = nextNum(), x2 = nextNum(), y2 = nextNum();
          if (!isFinite(x2) || !isFinite(y2)) break;
          const ax = cx, ay = cy;
          const qx = isRel ? ax + x1 : x1, qy = isRel ? ay + y1 : y1;
          const ex = isRel ? ax + x2 : x2, ey = isRel ? ay + y2 : y2;
          for (let s = 1; s <= 6; s++) {
            const t = s / 6, mt = 1 - t;
            pts.push({ x: mt * mt * ax + 2 * mt * t * qx + t * t * ex, y: mt * mt * ay + 2 * mt * t * qy + t * t * ey });
          }
          lastCtrlX = qx; lastCtrlY = qy; cx = ex; cy = ey;
        } else {
          const x2 = nextNum(), y2 = nextNum(), x3 = nextNum(), y3 = nextNum();
          if (!isFinite(x3) || !isFinite(y3)) break;
          const ax = cx, ay = cy;
          const c1x = 2 * ax - lastCtrlX, c1y = 2 * ay - lastCtrlY;
          const c2x = isRel ? ax + x2 : x2, c2y = isRel ? ay + y2 : y2;
          const ex = isRel ? ax + x3 : x3, ey = isRel ? ay + y3 : y3;
          for (let s = 1; s <= 8; s++) {
            const t = s / 8, mt = 1 - t;
            pts.push({
              x: mt * mt * mt * ax + 3 * mt * mt * t * c1x + 3 * mt * t * t * c2x + t * t * t * ex,
              y: mt * mt * mt * ay + 3 * mt * mt * t * c1y + 3 * mt * t * t * c2y + t * t * t * ey,
            });
          }
          lastCtrlX = c2x; lastCtrlY = c2y; cx = ex; cy = ey;
        }
      } else if (C === 'T') {
        const x2 = nextNum(), y2 = nextNum();
        if (!isFinite(x2) || !isFinite(y2)) break;
        const ax = cx, ay = cy;
        const qx = 2 * ax - lastCtrlX, qy = 2 * ay - lastCtrlY;
        const ex = isRel ? ax + x2 : x2, ey = isRel ? ay + y2 : y2;
        for (let s = 1; s <= 6; s++) {
          const t = s / 6, mt = 1 - t;
          pts.push({ x: mt * mt * ax + 2 * mt * t * qx + t * t * ex, y: mt * mt * ay + 2 * mt * t * qy + t * t * ey });
        }
        lastCtrlX = qx; lastCtrlY = qy; cx = ex; cy = ey;
      } else {
        i++;
      }
    }
    if (pts.length >= 2) subPaths.push({ points: pts, isClosed: false });
    return subPaths;
  }

  drawEntity(ctx, ent, isSelected) {
    const strokeCol = isSelected ? '#fbbf24' : ent.color || '#38bdf8';
    ctx.strokeStyle = strokeCol;
    ctx.lineWidth = isSelected ? (ent.width || 1.5) + 0.7 : (ent.width || 1.5);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const fillClosed = (targetEnt) => {
      const useColor = this.showColorFills && targetEnt.fillColor && targetEnt.isClosed !== false;
      if (!useColor && !this.showFills) return;
      ctx.save();
      ctx.fillStyle = useColor ? targetEnt.fillColor : strokeCol;
      ctx.globalAlpha = useColor ? (isSelected ? 0.50 : 0.34) : (isSelected ? 0.16 : 0.065);
      ctx.fill('evenodd');
      ctx.restore();
    };

    if (ent.type === 'line') {
      const p1 = this.worldToScreen(ent.x1, ent.y1);
      const p2 = this.worldToScreen(ent.x2, ent.y2);
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();

      if (this.activeTool === 'node' && isSelected) {
        this.drawNodeHandle(ctx, p1.x, p1.y, this.hoveredHoleIndex === -1 && 0 === this.hoveredNodeIndex, this.selectedHoleIndex === -1 && 0 === this.selectedNodeIndex, true);
        this.drawNodeHandle(ctx, p2.x, p2.y, this.hoveredHoleIndex === -1 && 1 === this.hoveredNodeIndex, this.selectedHoleIndex === -1 && 1 === this.selectedNodeIndex, true);
      }
    } else if (ent.type === 'rect') {
      const p = this.worldToScreen(ent.x, ent.y);
      const rw = ent.w * this.scale;
      const rh = ent.h * this.scale;
      const rr = (ent.cornerRadius || 0) * this.scale;
      ctx.beginPath();
      if (rr > 0.5 && typeof ctx.roundRect === 'function') {
        ctx.roundRect(p.x, p.y, rw, rh, Math.min(rr, Math.abs(rw) / 2, Math.abs(rh) / 2));
      } else {
        ctx.rect(p.x, p.y, rw, rh);
      }
      fillClosed(ent);
      ctx.stroke();
    } else if (ent.type === 'circle') {
      const c = this.worldToScreen(ent.cx, ent.cy);
      const r = ent.r * this.scale;
      ctx.beginPath(); ctx.arc(c.x, c.y, Math.max(0, r), 0, Math.PI * 2);
      fillClosed(ent);
      ctx.stroke();
    } else if (ent.type === 'polyline') {
      if (ent.points.length > 1) {
        ctx.beginPath();
        const tracePolyRing = (ring, closed) => {
          if (!ring || ring.length < 2) return;
          const first = this.worldToScreen(ring[0].x, ring[0].y);
          ctx.moveTo(first.x, first.y);
          for (let i = 1; i < ring.length; i++) {
            const pt = this.worldToScreen(ring[i].x, ring[i].y);
            ctx.lineTo(pt.x, pt.y);
          }
          if (closed) ctx.closePath();
        };
        tracePolyRing(ent.points, !!ent.isClosed);
        if (ent.holes && ent.holes.length > 0) {
          for (const hole of ent.holes) tracePolyRing(hole, true);
        }
        if (ent.isClosed) fillClosed(ent);
        ctx.stroke();

        if (this.activeTool === 'node' && isSelected) {
          this.drawCurveNodeHandles(ctx, ent);
        }
      } else if (ent.points.length === 1) {
        const sp = this.worldToScreen(ent.points[0].x, ent.points[0].y);
        this.drawNodeHandle(ctx, sp.x, sp.y, false, true, true);
      }
    } else if (ent.type === 'curve') {
      if (ent.points.length > 1) {
        const mapX = (x) => x * this.scale + this.panX;
        const mapY = (y) => y * this.scale + this.panY;
        ctx.beginPath();
        CadCanvas.traceSmoothCanvasPath(ctx, ent.points, !!ent.isClosed, mapX, mapY);
        if (ent.holes && ent.holes.length > 0) {
          for (const hole of ent.holes) {
            CadCanvas.traceSmoothCanvasPath(ctx, hole, true, mapX, mapY);
          }
        }
        if (ent.isClosed) fillClosed(ent);
        ctx.stroke();

        if (this.activeTool === 'node' && isSelected) {
          this.drawCurveNodeHandles(ctx, ent);
        }
      } else if (ent.points.length === 1) {
        const sp = this.worldToScreen(ent.points[0].x, ent.points[0].y);
        this.drawNodeHandle(ctx, sp.x, sp.y, false, true, true);
      }
    } else if (ent.type === 'text') {
      const p = this.worldToScreen(ent.x, ent.y);
      ctx.font = `${ent.fontWeight || 'bold'} ${(ent.size || 16) * this.scale * 0.4}px ${ent.fontFamily || 'sans-serif'}`;
      ctx.fillStyle = strokeCol;
      ctx.fillText(ent.text, p.x, p.y);
    }
  }

  drawCurveNodeHandles(ctx, ent) {
    // 1. Draw Bezier tangent stems & circular knobs for visible anchors
    const bezItems = this.getVisibleBezierAnchors(ent);
    if (bezItems.length > 0) {
      ctx.save();
      for (const { pt, holeIdx, nodeIdx } of bezItems) {
        const sa = this.worldToScreen(pt.x, pt.y);
        const drawKnob = (ctrl, which) => {
          if (!ctrl) return;
          const sc = this.worldToScreen(ctrl.x, ctrl.y);
          const isAct = this.activeBezierHandle && this.activeBezierHandle.holeIdx === holeIdx && this.activeBezierHandle.nodeIdx === nodeIdx && this.activeBezierHandle.which === which;
          ctx.strokeStyle = 'rgba(56, 189, 248, 0.72)';
          ctx.lineWidth = 1;
          ctx.setLineDash([2, 2]);
          ctx.beginPath(); ctx.moveTo(sa.x, sa.y); ctx.lineTo(sc.x, sc.y); ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = isAct ? '#fbbf24' : '#38bdf8';
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.2;
          ctx.beginPath(); ctx.arc(sc.x, sc.y, isAct ? 4.2 : 3.2, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        };
        drawKnob(pt.hIn, 'hIn');
        drawKnob(pt.hOut, 'hOut');
      }
      ctx.restore();
    }

    // 2. Draw outer & inner hole anchor nodes
    const drawRing = (ring, holeIdx) => {
      if (!ring) return;
      for (let i = 0; i < ring.length; i++) {
        const pt = ring[i];
        const sp = this.worldToScreen(pt.x, pt.y);
        const isHov = holeIdx === this.hoveredHoleIndex && i === this.hoveredNodeIndex;
        const isSel = holeIdx === this.selectedHoleIndex && i === this.selectedNodeIndex;
        this.drawNodeHandle(ctx, sp.x, sp.y, isHov, isSel, pt.corner !== false);
      }
    };
    drawRing(ent.points, -1);
    if (ent.holes && ent.holes.length > 0) {
      ent.holes.forEach((h, hIdx) => drawRing(h, hIdx));
    }
  }

  drawNodeHandle(ctx, sx, sy, isHovered, isSelected, isCorner = true) {
    ctx.save();
    if (isSelected) {
      ctx.fillStyle = '#ef4444'; ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2;
      if (isCorner) { ctx.fillRect(sx - 5, sy - 5, 10, 10); ctx.strokeRect(sx - 5, sy - 5, 10, 10); }
      else { ctx.beginPath(); ctx.arc(sx, sy, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); }
    } else if (isHovered) {
      ctx.fillStyle = '#f59e0b'; ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5;
      if (isCorner) { ctx.fillRect(sx - 4, sy - 4, 8, 8); ctx.strokeRect(sx - 4, sy - 4, 8, 8); }
      else { ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); }
    } else {
      ctx.fillStyle = '#ffffff'; ctx.strokeStyle = '#0054e3'; ctx.lineWidth = 1;
      if (isCorner) { ctx.fillRect(sx - 3, sy - 3, 6, 6); ctx.strokeRect(sx - 3, sy - 3, 6, 6); }
      else { ctx.beginPath(); ctx.arc(sx, sy, 3.2, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); }
    }
    ctx.restore();
  }

  drawSelectionTransformBox(ctx, bounds) {
    const tl = this.worldToScreen(bounds.minX, bounds.minY);
    const br = this.worldToScreen(bounds.maxX, bounds.maxY);
    const x = tl.x - 4, y = tl.y - 4;
    const w = (br.x - tl.x) + 8, h = (br.y - tl.y) + 8;

    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, w, h);

    // Stem to Rotate Handle
    const mx = x + w / 2;
    ctx.beginPath();
    ctx.moveTo(mx, y);
    ctx.lineTo(mx, y - 16);
    ctx.stroke();
    ctx.setLineDash([]);

    const handles = this.getScreenTransformHandles(bounds);
    for (const handle of handles) {
      if (handle.id === 'rot') {
        ctx.fillStyle = '#10b981';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(handle.sx, handle.sy, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      } else {
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#0054e3';
        ctx.lineWidth = 1.2;
        ctx.fillRect(handle.sx - 3.5, handle.sy - 3.5, 7, 7);
        ctx.strokeRect(handle.sx - 3.5, handle.sy - 3.5, 7, 7);
      }
    }
    ctx.restore();
  }

  static sampleEntityRings(ent, curveSegs = 8) {
    const closeRing = (pts) => {
      if (pts.length >= 3 && Math.hypot(pts[0].x - pts[pts.length - 1].x, pts[0].y - pts[pts.length - 1].y) > 0.05) {
        pts.push({ ...pts[0] });
      }
      return pts;
    };

    if (ent.type === 'rect') {
      return [CadCanvas.buildRoundedRectPoints(ent.x, ent.y, ent.w, ent.h, ent.cornerRadius || 0, 6)];
    }
    if (ent.type === 'circle') {
      const pts = [];
      const segments = 64;
      for (let i = 0; i <= segments; i++) {
        const theta = (i / segments) * Math.PI * 2;
        pts.push({ x: ent.cx + Math.cos(theta) * ent.r, y: ent.cy + Math.sin(theta) * ent.r });
      }
      return [pts];
    }
    if (ent.type === 'curve' && ent.points?.length > 1) {
      const outer = CadCanvas.sampleCurvePoints(ent.points, !!ent.isClosed, curveSegs);
      if (ent.isClosed) closeRing(outer);
      const rings = [outer];
      if (ent.holes && ent.holes.length > 0) {
        for (const hole of ent.holes) {
          if (hole.length > 1) rings.push(closeRing(CadCanvas.sampleCurvePoints(hole, true, curveSegs)));
        }
      }
      return rings;
    }
    if (ent.type === 'polyline' && ent.points?.length > 1) {
      const outer = ent.points.map((p) => ({ x: p.x, y: p.y }));
      if (ent.isClosed) closeRing(outer);
      const rings = [outer];
      if (ent.holes && ent.holes.length > 0) {
        for (const hole of ent.holes) {
          if (hole.length > 1) rings.push(closeRing(hole.map((p) => ({ x: p.x, y: p.y }))));
        }
      }
      return rings;
    }
    if (ent.type === 'line') {
      return [[{ x: ent.x1, y: ent.y1 }, { x: ent.x2, y: ent.y2 }]];
    }
    return [];
  }

  getPolylinesForCAMWithMeta() {
    const items = [];
    for (const ent of this.entities) {
      if (ent.visible === false) continue;
      const rings = CadCanvas.sampleEntityRings(ent, 8);
      const baseLayer = ent.layerGroup || ent.name || 'CAD_Layer';
      rings.forEach((pts, rIdx) => {
        if (pts && pts.length >= 2) {
          items.push({
            points: pts,
            isHole: rIdx > 0,
            layerName: rIdx > 0 ? `${baseLayer}_Holes` : baseLayer,
            entity: ent,
          });
        }
      });
    }
    return items;
  }

  getPolylinesForCAM() {
    return this.getPolylinesForCAMWithMeta().map((item) => item.points);
  }
}
