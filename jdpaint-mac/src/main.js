// Main Application Controller for JDPaint 5.21 (Mac Edition)
import { CadCanvas } from './cad/cadCanvas.js';
import { ImageVectorizer } from './cad/imageVectorizer.js';
import { VectorPuffEngine } from './cad/vectorPuffEngine.js';
import { ReliefEngine } from './relief/reliefEngine.js';
import { Asset3DEngine } from './relief/asset3DEngine.js';
import { Viewport3D } from './viewport/viewport3D.js';
import { CamEngine } from './cam/camEngine.js';
import { GcodeExporter } from './export/gcodeExporter.js';
import { StlExporter } from './export/stlExporter.js';

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const cadCanvasEl = document.getElementById('cad-canvas');
  const threeWrapper = document.getElementById('three-canvas-wrapper');
  const statusCursor = document.getElementById('status-cursor');
  const statusMode = document.getElementById('status-mode');
  const statusSnap = document.getElementById('status-snap-info');
  const refStatus = document.getElementById('ref-image-status');

  // Engines
  const reliefEngine = new ReliefEngine(100, 100, 8, 384);
  const viewport3D = new Viewport3D(threeWrapper, reliefEngine);
  const camEngine = new CamEngine();

  let activeToolpathSegments = [];
  let loadedImageElement = null;

  // Base Picture 3D Heightmap Layer state
  let basePhotoBuffer = null;
  let basePhotoEnabled = true;
  let basePhotoWeight = 1.0;
  let dirty2DFor3D = false;
  let autoSyncTimer = null;

  function scheduleAutoSync3D() {
    dirty2DFor3D = true;
    const chkAuto = document.getElementById('chk-auto-sync-3d');
    if (!chkAuto || !chkAuto.checked) return;
    if (cadCanvas.entities.length === 0 && !basePhotoBuffer) return;

    if (autoSyncTimer) clearTimeout(autoSyncTimer);
    autoSyncTimer = setTimeout(() => {
      rebuild3DFromLayers(true);
    }, 140);
  }

  // Real-time Coordinate Display + Layer Sync + Floating Quick-Edit Callbacks
  const cadCanvas = new CadCanvas(
    cadCanvasEl,
    (x, y, snapNote) => {
      const z = reliefEngine.getZAt(x, y);
      statusCursor.textContent = `X: ${x.toFixed(3)} mm  Y: ${y.toFixed(3)} mm  Z: ${z.toFixed(3)} mm${snapNote || ''}`;
    },
    {
      onEntitiesChange: () => {
        renderLayerList();
        syncSelectedLayerInspector();
        updateFloatingCadBar();
        scheduleAutoSync3D();
      },
      onSelectionChange: () => {
        renderLayerList();
        syncSelectedLayerInspector();
        updateFloatingCadBar();
      },
      onDraftChange: () => {
        updateFloatingCadBar();
      },
    }
  );

  // Floating Quick-Edit & Precision Dimension Pill Bar (renders over bottom of 2D canvas)
  const cadFloatingBar = document.getElementById('cad-floating-bar');
  function updateFloatingCadBar() {
    if (!cadFloatingBar) return;

    // Case 1: Active multi-point draft (Pen / Polyline / Arc)
    if (cadCanvas.currentDraft && (cadCanvas.currentDraft.type === 'curve' || cadCanvas.currentDraft.type === 'polyline' || cadCanvas.currentDraft.type === 'arc_draft')) {
      cadFloatingBar.style.display = 'flex';
      const ptsCount = cadCanvas.currentDraft.points ? cadCanvas.currentDraft.points.length : (cadCanvas.currentDraft.pts ? cadCanvas.currentDraft.stage : 1);
      cadFloatingBar.innerHTML = `
        <span style="color:#38bdf8;font-weight:bold;">Draft (${ptsCount} pts)</span>
        <button class="cad-float-btn primary" id="fl-finish">✓ Finish (R-Click/Enter)</button>
        ${cadCanvas.currentDraft.type !== 'arc_draft' ? '<button class="cad-float-btn emerald" id="fl-close">⤸ Close Shape</button>' : ''}
        ${cadCanvas.currentDraft.type !== 'arc_draft' ? '<button class="cad-float-btn" id="fl-bksp">↩ Undo Pt (Bksp)</button>' : ''}
        <button class="cad-float-btn" id="fl-cancel">✕ Esc</button>
      `;
      document.getElementById('fl-finish')?.addEventListener('click', () => cadCanvas.finishCurrentDraft(false));
      document.getElementById('fl-close')?.addEventListener('click', () => cadCanvas.finishCurrentDraft(true));
      document.getElementById('fl-bksp')?.addEventListener('click', () => {
        if (cadCanvas.currentDraft?.points?.length > 1) {
          cadCanvas.currentDraft.points.pop();
          cadCanvas.requestRender();
          updateFloatingCadBar();
        }
      });
      document.getElementById('fl-cancel')?.addEventListener('click', () => {
        cadCanvas.cancelCurrentDraft();
        updateFloatingCadBar();
      });
      return;
    }

    // Case 1.4: Click Pen Reshape Tool (Q) -> Click layer & click point-by-point to reshape curve!
    if (cadCanvas.activeTool === 'pen-edit' && !cadCanvas.currentDraft) {
      cadFloatingBar.style.display = 'flex';
      const pem = cadCanvas.penEditMode || 'span';
      cadFloatingBar.innerHTML = `
        <span style="color:#34d399;font-weight:bold;">Click Pen (Q):</span>
        <button class="cad-float-btn${pem === 'span' ? ' primary' : ''}" id="fl-pen-span" title="Click point 1 on layer outline -> click point 2..N -> click back on layer (or Enter/DblClick) to reshape curve!">🖊️ Click-by-Click Reshape</button>
        <button class="cad-float-btn${pem === 'point' ? ' primary' : ''}" id="fl-pen-point" title="Click directly on outline edge to insert & drag a smooth control point (Right-click deletes node)">✛ Click Point Insert</button>
        <span style="font-size:10px;color:#cbd5e1;">Drag handles/nodes directly anytime</span>
      `;
      document.getElementById('fl-pen-span')?.addEventListener('click', () => {
        cadCanvas.penEditMode = 'span';
        updateFloatingCadBar();
      });
      document.getElementById('fl-pen-point')?.addEventListener('click', () => {
        cadCanvas.penEditMode = 'point';
        updateFloatingCadBar();
      });
      return;
    }

    // Case 1.5: Smart Polyline Tool active -> Show Instant 1-by-1 Design Detection, Sensitivity & Tracing Sub-Modes!
    if (cadCanvas.activeTool === 'polyline' && !cadCanvas.currentDraft) {
      cadFloatingBar.style.display = 'flex';
      const sm = cadCanvas.polylineSmartMode || 'capture';
      const pScale = cadCanvas.polylinePartScale || 'small';
      const pSens = cadCanvas.smartPolySensitivity || 'balanced';
      const hasPic = !!cadCanvas._edgeGuide;
      cadFloatingBar.innerHTML = `
        <span style="color:#22d3ee;font-weight:bold;">Smart Poly${hasPic ? '' : ' (Load Picture)'}:</span>
        <button class="cad-float-btn${sm === 'capture' ? ' primary' : ''}" id="fl-poly-capture" title="Hover over any small petal/part & Left-Click to capture 1-by-1; Click shape/nodes to reshape">✨ Detect &amp; Grab</button>
        <button class="cad-float-btn${sm === 'magnetic' ? ' primary' : ''}" id="fl-poly-magnetic" title="Click points along outline — curve automatically hugs photo edges!">🧲 Edge Follower</button>
        <button class="cad-float-btn${sm === 'manual' ? ' primary' : ''}" id="fl-poly-manual" title="Classic straight point-to-point CAD polyline">📏 Straight Line</button>
        <button class="cad-float-btn${pScale === 'small' ? ' primary' : ''}" id="fl-poly-size" title="Toggle between Tight 1-by-1 Part Capture and Medium Panel Capture">${pScale === 'small' ? '🔍 1-by-1' : '⬛ Medium'}</button>
        <button class="cad-float-btn" id="fl-poly-sens" title="Cycle motif capture sensitivity: Tight / Balanced / Wide">🎯 ${pSens[0].toUpperCase() + pSens.slice(1)}</button>
      `;
      const setPolyMode = (mode) => {
        cadCanvas.polylineSmartMode = mode;
        cadCanvas.detectedShapePreview = null;
        cadCanvas.liveEdgePath = null;
        cadCanvas.requestRender();
        updateFloatingCadBar();
      };
      document.getElementById('fl-poly-capture')?.addEventListener('click', () => setPolyMode('capture'));
      document.getElementById('fl-poly-magnetic')?.addEventListener('click', () => setPolyMode('magnetic'));
      document.getElementById('fl-poly-manual')?.addEventListener('click', () => setPolyMode('manual'));
      document.getElementById('fl-poly-size')?.addEventListener('click', () => {
        cadCanvas.polylinePartScale = cadCanvas.polylinePartScale === 'small' ? 'medium' : 'small';
        cadCanvas.detectedShapePreview = null;
        cadCanvas.requestRender();
        updateFloatingCadBar();
      });
      document.getElementById('fl-poly-sens')?.addEventListener('click', () => {
        const order = ['tight', 'balanced', 'wide'];
        cadCanvas.smartPolySensitivity = order[(order.indexOf(pSens) + 1) % order.length];
        cadCanvas.detectedShapePreview = null;
        cadCanvas.requestRender();
        updateFloatingCadBar();
      });
      return;
    }

    // Case 2: One or more 2D entities selected -> Exact X, Y, W, H numeric inputs + Quick Actions!
    if (cadCanvas.selectedEntities.length > 0) {
      const b = cadCanvas.getSelectionBounds();
      if (!b) { cadFloatingBar.style.display = 'none'; return; }
      const bx = b.minX, by = b.minY, bw = b.maxX - b.minX, bh = b.maxY - b.minY;

      cadFloatingBar.style.display = 'flex';
      const existingX = document.getElementById('fl-dim-x');
      if (existingX) {
        // Update values in-place so Tab navigation & mouse drag stay 60fps without rebuilding DOM
        const pairs = [['fl-dim-x', bx], ['fl-dim-y', by], ['fl-dim-w', bw], ['fl-dim-h', bh]];
        for (const [id, val] of pairs) {
          const inp = document.getElementById(id);
          if (inp && document.activeElement !== inp) inp.value = val.toFixed(1);
        }
        return;
      }

      cadFloatingBar.innerHTML = `
        <label>X:<input type="number" id="fl-dim-x" step="0.5" value="${bx.toFixed(1)}"></label>
        <label>Y:<input type="number" id="fl-dim-y" step="0.5" value="${by.toFixed(1)}"></label>
        <label>W:<input type="number" id="fl-dim-w" step="0.5" min="0.5" value="${bw.toFixed(1)}"></label>
        <label>H:<input type="number" id="fl-dim-h" step="0.5" min="0.5" value="${bh.toFixed(1)}"></label>
        <button class="cad-float-btn" id="fl-nodes" title="Edit Nodes (N)">☩ Nodes</button>
        <button class="cad-float-btn" id="fl-pen-edit" title="Click Pen Reshape (Q)">🖊️ Click Pen</button>
        <button class="cad-float-btn" id="fl-reshape" title="Reshape Brush (H)">〰️ Reshape</button>
        <button class="cad-float-btn" id="fl-fillet" title="Round Corners (3mm Fillet)">◠ Fillet</button>
        <button class="cad-float-btn" id="fl-mir-copy" title="Mirror Duplicate Left ↔ Right">🪞 Mirror Copy</button>
        <button class="cad-float-btn" id="fl-dup" title="Duplicate (Ctrl+D)">⧉</button>
        <button class="cad-float-btn emerald" id="fl-to3d" title="Convert Selected to 3D">💨 To 3D</button>
      `;

      const applyDims = () => {
        cadCanvas.setSelectionBoundsMm({
          x: parseFloat(document.getElementById('fl-dim-x').value),
          y: parseFloat(document.getElementById('fl-dim-y').value),
          w: parseFloat(document.getElementById('fl-dim-w').value),
          h: parseFloat(document.getElementById('fl-dim-h').value),
        });
      };
      ['fl-dim-x', 'fl-dim-y', 'fl-dim-w', 'fl-dim-h'].forEach((id) => {
        document.getElementById(id)?.addEventListener('change', applyDims);
      });
      document.getElementById('fl-nodes')?.addEventListener('click', () => selectTool('node'));
      document.getElementById('fl-pen-edit')?.addEventListener('click', () => selectTool('pen-edit'));
      document.getElementById('fl-reshape')?.addEventListener('click', () => selectTool('reshape'));
      document.getElementById('fl-fillet')?.addEventListener('click', () => {
        const n = cadCanvas.filletSelected(3.0);
        if (n > 0) statusMode.textContent = `Applied 3.0mm rounded corner fillet to ${n} shape(s).`;
      });
      document.getElementById('fl-mir-copy')?.addEventListener('click', () => {
        const n = cadCanvas.mirrorCopySelected('x');
        statusMode.textContent = `Created symmetrical mirror copy across stock center (+${n} layer).`;
      });
      document.getElementById('fl-dup')?.addEventListener('click', () => triggerDuplicate());
      document.getElementById('fl-to3d')?.addEventListener('click', () => runVectorPuff());
      return;
    }

    cadFloatingBar.style.display = 'none';
  }

  // Symmetry Mode selector
  const selectSymmetry = document.getElementById('select-symmetry');
  if (selectSymmetry) {
    selectSymmetry.addEventListener('change', () => {
      cadCanvas.symmetryMode = selectSymmetry.value;
      cadCanvas.requestRender();
      statusMode.textContent = cadCanvas.symmetryMode === 'none'
        ? 'Symmetry drawing mode turned OFF.'
        : `Symmetry mode (${cadCanvas.symmetryMode}) ON — anything you draw manually automatically mirrors across stock center!`;
    });
  }

  // Synchronize Settings
  function syncSettings() {
    const w = parseFloat(document.getElementById('prop-width').value) || 100;
    const h = parseFloat(document.getElementById('prop-height').value) || 100;
    const maxZ = parseFloat(document.getElementById('prop-relief-depth').value) || 8;
    const baseZ = parseFloat(document.getElementById('prop-base-height').value) || 0;

    reliefEngine.width = w;
    reliefEngine.height = h;
    reliefEngine.maxDepth = maxZ;
    reliefEngine.baseHeight = baseZ;

    cadCanvas.workpieceWidth = w;
    cadCanvas.workpieceHeight = h;
    if (cadCanvas.refImage) {
      cadCanvas.refImageBounds = ImageVectorizer.getImageFitBounds(cadCanvas.refImage, w, h);
    }
    cadCanvas.render();

    viewport3D.buildMesh();

    const toolType = document.getElementById('tool-type').value;
    const diameter = parseFloat(document.getElementById('tool-diameter').value) || 3.175;
    const stepover = parseFloat(document.getElementById('tool-stepover').value) || 0.3;
    const safeZ = parseFloat(document.getElementById('cam-safe-z').value) || 5.0;
    const feed = parseFloat(document.getElementById('cam-feedrate').value) || 1500;
    const plunge = parseFloat(document.getElementById('cam-plunge').value) || 500;
    const spindle = parseInt(document.getElementById('cam-spindle').value, 10) || 18000;

    camEngine.setToolConfig({
      type: toolType,
      diameter,
      stepover,
      safeZ,
      feedrate: feed,
      plunge,
      spindle,
    });

    document.getElementById('stat-bounds').textContent = `${w} × ${h} × ${maxZ} mm`;
  }

  // Bind input listeners
  ['prop-width', 'prop-height', 'prop-relief-depth', 'prop-base-height',
   'tool-type', 'tool-diameter', 'tool-stepover', 'cam-safe-z',
   'cam-feedrate', 'cam-plunge', 'cam-spindle'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', syncSettings);
  });

  document.getElementById('prop-grid-res').addEventListener('change', (e) => {
    const res = parseInt(e.target.value, 10);
    reliefEngine.resolution = res;
    reliefEngine.data = new Float32Array(res * res);
    if (loadedImageElement && basePhotoEnabled) {
      const style = document.getElementById('relief-style')?.value || 'sculpted';
      const edgeStrength = parseFloat(document.getElementById('relief-edge-strength')?.value) || 0.35;
      const invert = document.getElementById('relief-invert-check')?.checked || false;
      const removeWhiteBg = document.getElementById('relief-remove-bg')?.checked ?? true;
      reliefEngine.loadImageHighAccuracy(loadedImageElement, { style, edgeStrength, invert, removeWhiteBg, denoise: true });
      basePhotoBuffer = new Float32Array(reliefEngine.data);
    }
    if (cadCanvas.entities.length > 0) {
      rebuild3DFromLayers(true);
    } else if (!loadedImageElement) {
      reliefEngine.loadPreset('rosette');
    }
    viewport3D.buildMesh();
  });

  // Right Inspector Tabs
  const inspTabs = document.querySelectorAll('.insp-tab');
  const inspContents = document.querySelectorAll('.insp-tab-content');
  inspTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      inspTabs.forEach((t) => t.classList.remove('active'));
      inspContents.forEach((c) => c.classList.remove('active'));
      tab.classList.add('active');
      const targetId = tab.dataset.tab;
      document.getElementById(targetId).classList.add('active');
    });
  });

  // Tool Switching (2D CAD & 3D Sculpt Brushes)
  function selectTool(toolName) {
    document.querySelectorAll('.tb-tool-btn, .act-tool').forEach((btn) => {
      btn.classList.remove('active');
      if (btn.dataset.tool === toolName) btn.classList.add('active');
    });

    if (toolName.startsWith('sculpt-')) {
      const mode = toolName.replace('sculpt-', '');
      viewport3D.activeSculptMode = mode;
      cadCanvas.activeTool = 'select';
      statusMode.textContent = `3D Sculpt Brush: ${mode.toUpperCase()}. Left-click and drag on the 3D relief surface to sculpt.`;
    } else {
      viewport3D.activeSculptMode = null;
      const prevTool = cadCanvas.activeTool;
      cadCanvas.activeTool = toolName;
      if (toolName !== 'polyline') {
        cadCanvas.detectedShapePreview = null;
        cadCanvas.liveEdgePath = null;
      }
      updateFloatingCadBar();

      // Re-clicking Polygon tool lets user set sides & Star mode quickly
      if (toolName === 'polygon' && prevTool === 'polygon') {
        const inp = prompt('Polygon sides (3-16) — add "s" for Star (e.g. "5s" or "6"):', `${cadCanvas.polygonSides}${cadCanvas.polygonStarMode ? 's' : ''}`);
        if (inp) {
          const isStar = inp.toLowerCase().includes('s');
          const sides = Math.max(3, Math.min(16, parseInt(inp, 10) || 6));
          cadCanvas.polygonSides = sides;
          cadCanvas.polygonStarMode = isStar;
        }
      }

      let hint = '';
      if (toolName === 'select') hint = 'Click shape/layer to select, drag to move, drag 8 handles to resize, green handle to rotate.';
      else if (toolName === 'pen') hint = 'Click ON an outline -> draw new shape -> click BACK on outline + Enter/DblClick to replace that section directly!';
      else if (toolName === 'pen-edit') hint = 'Click any traced layer edge -> click point 1..N to reshape curve -> click back on layer (or Enter/DblClick) to update!';
      else if (toolName === 'reshape') hint = 'Drag smoothly across ANY traced outline to bend & fix its shape like clay!';
      else if (toolName === 'freehand') hint = 'Hold & drag smoothly anywhere to sketch a freehand curve layer.';
      else if (toolName === 'node') hint = 'Click curve to reveal nodes, click edge to add+drag node immediately, right-click or Del to remove node.';
      else if (toolName === 'knife') hint = 'Click on any curve or line to split/cut it cleanly into separate layers.';
      else if (toolName === 'arc') hint = 'Click start point, click end point, then click bulge height.';
      else if (toolName === 'ellipse') hint = 'Click center and drag outward to draw Ellipse / Oval.';
      else if (toolName === 'polygon') hint = `Drag to draw ${cadCanvas.polygonSides}-${cadCanvas.polygonStarMode ? 'Point Star' : 'Gon'} (click Poly/Star again to change sides).`;
      else if (toolName === 'eraser') hint = 'Click or drag across vector layers to erase them.';
      else if (toolName === 'polyline') hint = 'Hover over any design shape on picture -> click to Capture Shape, or trace curves along photo edges automatically!';
      else hint = 'Click and drag in the 2D CAD plane to draft geometry.';
      statusMode.textContent = `2D Tool: ${toolName.toUpperCase()}. ${hint}`;
    }
  }

  document.querySelectorAll('.tb-tool-btn[data-tool], .act-tool[data-tool]').forEach((btn) => {
    btn.addEventListener('click', () => selectTool(btn.dataset.tool));
  });

  // Finish Path button
  const btnFinishCurve = document.getElementById('btn-finish-curve');
  if (btnFinishCurve) {
    btnFinishCurve.addEventListener('click', () => {
      cadCanvas.finishCurrentDraft();
      statusMode.textContent = 'Curve / Polyline path finished and added to canvas.';
    });
  }

  // Close Path button
  const btnClosePath = document.getElementById('btn-close-path');
  if (btnClosePath) {
    btnClosePath.addEventListener('click', () => {
      cadCanvas.closeSelectedCurve();
      statusMode.textContent = 'Selected open curve closed into a solid polygon.';
    });
  }

  // Smooth Curve button
  const btnSmoothCurve = document.getElementById('btn-smooth-curve');
  if (btnSmoothCurve) {
    btnSmoothCurve.addEventListener('click', () => {
      cadCanvas.smoothSelectedCurve();
      statusMode.textContent = 'Smoothed selected curve nodes.';
    });
  }

  // Sculpt Brush Sliders
  const brushRadSlider = document.getElementById('brush-radius');
  const brushStrSlider = document.getElementById('brush-strength');
  const brushRadVal = document.getElementById('brush-rad-val');
  const brushStrVal = document.getElementById('brush-str-val');

  brushRadSlider.addEventListener('input', (e) => {
    const r = parseFloat(e.target.value);
    brushRadVal.textContent = `${r}mm`;
    viewport3D.setBrushRadius(r);
  });
  brushStrSlider.addEventListener('input', (e) => {
    const s = parseInt(e.target.value, 10);
    brushStrVal.textContent = `${s}%`;
    viewport3D.brushStrength = s / 100;
  });

  // ============================================================
  // Background Reference Image for Tracing
  // ============================================================
  function handleReferenceImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        loadedImageElement = img;
        basePhotoBuffer = null;
        updateBasePhotoBox();
        cadCanvas.setReferenceImage(img);
        refStatus.textContent = `Picture: ${file.name} (Ready to trace / auto-vectorize)`;
        statusMode.textContent = `Reference picture '${file.name}' loaded! You can now use "⚡ Auto-Trace 2D", "✨ 1-Click (2D+3D)", or Pen/Freehand tools.`;
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  }

  const inputRefImage = document.getElementById('input-ref-image');
  const inputRefImageBar = document.getElementById('input-ref-image-bar');
  if (inputRefImage) inputRefImage.addEventListener('change', handleReferenceImageUpload);
  if (inputRefImageBar) inputRefImageBar.addEventListener('change', handleReferenceImageUpload);

  // Picture Opacity Slider
  const sliderPicOpacity = document.getElementById('slider-pic-opacity');
  const labelPicOpacity = document.getElementById('label-pic-opacity');
  sliderPicOpacity.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10);
    labelPicOpacity.textContent = `${val}%`;
    cadCanvas.refImageOpacity = val / 100;
    cadCanvas.render();
  });

  // Toggle Picture Visibility
  const btnTogglePic = document.getElementById('btn-toggle-pic');
  btnTogglePic.addEventListener('click', function () {
    this.classList.toggle('active');
    cadCanvas.showRefImage = this.classList.contains('active');
    cadCanvas.render();
  });

  // Remove Picture
  function removePic() {
    cadCanvas.clearReferenceImage();
    loadedImageElement = null;
    refStatus.textContent = 'No background image loaded';
    statusMode.textContent = 'Background reference picture removed.';
  }
  document.getElementById('btn-clear-pic').addEventListener('click', removePic);
  document.getElementById('menu-edit-clear-ref').addEventListener('click', removePic);

  // ============================================================
  // LAYER MANAGER & PER-LAYER 2D ↔ 3D EDITOR
  // ============================================================
  const layerListContainer = document.getElementById('layer-list-container');
  const layerCountBadge = document.getElementById('layer-count-badge');
  const basePhotoBox = document.getElementById('base-photo-layer-box');
  const chkBasePhotoEnabled = document.getElementById('chk-base-photo-enabled');
  const sliderBasePhotoWeight = document.getElementById('slider-base-photo-weight');
  const lblBasePhotoWeight = document.getElementById('lbl-base-photo-weight');
  const selectedLayerPropsBox = document.getElementById('selected-layer-props');

  const layerPropName = document.getElementById('layer-prop-name');
  const layerProp3dEnabled = document.getElementById('layer-prop-3d-enabled');
  const layerPropMode = document.getElementById('layer-prop-mode');
  const layerPropShape = document.getElementById('layer-prop-shape');
  const layerPropHeight = document.getElementById('layer-prop-height');
  const layerPropRadius = document.getElementById('layer-prop-radius');
  const layerPropOpenRow = document.getElementById('layer-prop-open-row');
  const layerPropOpenStyle = document.getElementById('layer-prop-open-style');
  const layerPropTextRow = document.getElementById('layer-prop-text-row');
  const layerPropText = document.getElementById('layer-prop-text');
  const layerPropFont = document.getElementById('layer-prop-font');

  function updateBasePhotoBox() {
    if (!basePhotoBox) return;
    basePhotoBox.style.display = basePhotoBuffer ? 'block' : 'none';
    if (chkBasePhotoEnabled) chkBasePhotoEnabled.checked = basePhotoEnabled;
    if (sliderBasePhotoWeight) sliderBasePhotoWeight.value = Math.round(basePhotoWeight * 100);
    if (lblBasePhotoWeight) lblBasePhotoWeight.textContent = `${Math.round(basePhotoWeight * 100)}%`;
  }

  function renderLayerList() {
    if (!layerListContainer) return;
    const entities = cadCanvas.entities;
    if (layerCountBadge) layerCountBadge.textContent = String(entities.length);

    if (entities.length === 0) {
      layerListContainer.innerHTML = '<div class="layer-empty-hint">Draw with Pen/Shapes or Auto-Trace a picture to edit layers here.</div>';
      return;
    }

    layerListContainer.innerHTML = '';
    // Render top-to-bottom (newest/topmost layer first, matching CAD/Photoshop convention)
    for (let i = entities.length - 1; i >= 0; i--) {
      const ent = cadCanvas.ensureEntityMetadata(entities[i]);
      const isSelected = cadCanvas.selectedEntities.includes(ent);
      const isVisible = ent.visible !== false;
      const isLocked = !!ent.locked;
      const is3D = !ent.relief || ent.relief.enabled !== false;

      const row = document.createElement('div');
      row.className = `layer-row${isSelected ? ' selected' : ''}${!isVisible ? ' hidden-layer' : ''}`;

      // Eye visibility button
      const eyeBtn = document.createElement('button');
      eyeBtn.className = 'layer-icon-btn';
      eyeBtn.textContent = isVisible ? '👁️' : '🚫';
      eyeBtn.title = isVisible ? 'Hide layer (2D & 3D)' : 'Show layer';
      eyeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        cadCanvas.saveState();
        ent.visible = !isVisible;
        cadCanvas.requestRender();
        renderLayerList();
        scheduleAutoSync3D();
      });

      // Lock toggle
      const lockBtn = document.createElement('button');
      lockBtn.className = 'layer-icon-btn';
      lockBtn.textContent = isLocked ? '🔒' : '🔓';
      lockBtn.title = isLocked ? 'Unlock layer for editing' : 'Lock layer';
      lockBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        ent.locked = !isLocked;
        renderLayerList();
      });

      // 2D / 3D mode badge (click to toggle 3D inclusion directly!)
      const badge = document.createElement('span');
      badge.className = `layer-badge ${is3D ? 'mode-3d' : 'mode-2d'}`;
      badge.textContent = is3D ? (ent.relief?.mode === 'sub' ? '3D-' : '3D+') : '2D';
      badge.title = 'Click to toggle 3D Relief ON/OFF for this layer';
      badge.addEventListener('click', (e) => {
        e.stopPropagation();
        cadCanvas.ensureEntityMetadata(ent);
        ent.relief.enabled = !is3D;
        renderLayerList();
        syncSelectedLayerInspector();
        scheduleAutoSync3D();
      });

      // Perceptual swatch chip
      const swatch = document.createElement('span');
      swatch.className = 'layer-swatch';
      swatch.style.backgroundColor = ent.fillColor || ent.color || '#38bdf8';
      swatch.title = `Layer Fill: ${ent.fillColor || ent.color || '#38bdf8'}`;

      // Layer name
      const nameSpan = document.createElement('span');
      nameSpan.className = 'layer-name';
      nameSpan.textContent = ent.name || `Layer #${i + 1}`;
      nameSpan.title = `${ent.name} (${ent.type})${ent.layerGroup ? ` — ${ent.layerGroup}` : ''}`;

      // Compound hole indicator badge
      let holeTag = null;
      if (ent.holes && ent.holes.length > 0) {
        holeTag = document.createElement('span');
        holeTag.className = 'layer-hole-tag';
        holeTag.textContent = `-${ent.holes.length}h`;
        holeTag.title = `${ent.holes.length} compound cutout hole(s) — click "Release Holes" to separate`;
      }

      // Move Up (higher draw/merge priority)
      const upBtn = document.createElement('button');
      upBtn.className = 'layer-icon-btn';
      upBtn.textContent = '▲';
      upBtn.title = 'Move Layer Up';
      upBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (i < entities.length - 1) {
          cadCanvas.saveState();
          [entities[i], entities[i + 1]] = [entities[i + 1], entities[i]];
          cadCanvas.requestRender();
          renderLayerList();
          scheduleAutoSync3D();
        }
      });

      // Move Down
      const downBtn = document.createElement('button');
      downBtn.className = 'layer-icon-btn';
      downBtn.textContent = '▼';
      downBtn.title = 'Move Layer Down';
      downBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (i > 0) {
          cadCanvas.saveState();
          [entities[i], entities[i - 1]] = [entities[i - 1], entities[i]];
          cadCanvas.requestRender();
          renderLayerList();
          scheduleAutoSync3D();
        }
      });

      // Delete layer
      const delBtn = document.createElement('button');
      delBtn.className = 'layer-icon-btn';
      delBtn.textContent = '🗑️';
      delBtn.title = 'Delete this layer';
      delBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        cadCanvas.saveState();
        entities.splice(i, 1);
        cadCanvas.selectedEntities = cadCanvas.selectedEntities.filter((s) => s !== ent);
        cadCanvas.requestRender();
        renderLayerList();
        syncSelectedLayerInspector();
        scheduleAutoSync3D();
      });

      row.addEventListener('click', (e) => {
        if (e.metaKey || e.ctrlKey) {
          const idx = cadCanvas.selectedEntities.indexOf(ent);
          if (idx === -1) cadCanvas.selectedEntities.push(ent);
          else cadCanvas.selectedEntities.splice(idx, 1);
        } else {
          cadCanvas.selectedEntities = [ent];
        }
        cadCanvas.requestRender();
        renderLayerList();
        syncSelectedLayerInspector();
      });

      if (holeTag) {
        row.append(eyeBtn, lockBtn, swatch, badge, nameSpan, holeTag, upBtn, downBtn, delBtn);
      } else {
        row.append(eyeBtn, lockBtn, swatch, badge, nameSpan, upBtn, downBtn, delBtn);
      }
      layerListContainer.appendChild(row);
    }
  }

  function syncSelectedLayerInspector() {
    if (!selectedLayerPropsBox) return;
    const ent = cadCanvas.selectedEntity;
    if (!ent) {
      selectedLayerPropsBox.style.display = 'none';
      return;
    }

    cadCanvas.ensureEntityMetadata(ent);
    selectedLayerPropsBox.style.display = 'block';

    layerPropName.value = ent.name || '';
    layerProp3dEnabled.checked = ent.relief.enabled !== false;
    layerPropMode.value = ent.relief.mode || 'max';
    layerPropShape.value = ent.relief.shape || 'dome';
    layerPropHeight.value = ent.relief.heightMm ?? 4.0;
    layerPropRadius.value = ent.relief.radiusMm ?? 10.0;

    const isOpenCapable = ent.type === 'line' || ent.type === 'curve' || ent.type === 'polyline';
    layerPropOpenRow.style.display = isOpenCapable ? 'flex' : 'none';
    if (isOpenCapable) {
      const isStroke = VectorPuffEngine.shouldStrokeEntity(ent, false);
      layerPropOpenStyle.value = isStroke ? 'stroke' : 'fill';
    }
    if (layerPropTextRow) {
      const isText = ent.type === 'text';
      layerPropTextRow.style.display = isText ? 'flex' : 'none';
      if (isText) {
        if (layerPropText) layerPropText.value = ent.text || '';
        if (layerPropFont) layerPropFont.value = ent.fontFamily || 'serif';
      }
    }
  }

  function applySelectedLayerPropChange(rebuild = true) {
    const targets = cadCanvas.selectedEntities;
    if (!targets || targets.length === 0) return;

    const primary = targets[0];
    cadCanvas.ensureEntityMetadata(primary);
    primary.name = layerPropName.value.trim() || primary.name;
    if (primary.type === 'text' && layerPropText && layerPropFont) {
      primary.text = layerPropText.value || primary.text;
      primary.fontFamily = layerPropFont.value || 'serif';
      if (cadCanvas._bboxCache) cadCanvas._bboxCache.delete(primary);
      cadCanvas.requestRender();
    }

    for (const ent of targets) {
      cadCanvas.ensureEntityMetadata(ent);
      ent.relief.enabled = layerProp3dEnabled.checked;
      ent.relief.mode = layerPropMode.value;
      ent.relief.shape = layerPropShape.value;
      ent.relief.heightMm = parseFloat(layerPropHeight.value) || 4.0;
      ent.relief.radiusMm = parseFloat(layerPropRadius.value) || 10.0;
      ent.relief.openAsStroke = layerPropOpenStyle.value === 'stroke';
      if (layerPropOpenStyle.value === 'fill' && (ent.type === 'curve' || ent.type === 'polyline') && ent.points.length >= 3) {
        ent.isClosed = true;
      }
    }

    renderLayerList();
    if (rebuild) scheduleAutoSync3D();
  }

  if (layerPropName) layerPropName.addEventListener('change', () => applySelectedLayerPropChange(false));
  [layerProp3dEnabled, layerPropMode, layerPropShape, layerPropHeight, layerPropRadius, layerPropOpenStyle, layerPropText, layerPropFont].forEach((el) => {
    if (el) el.addEventListener('change', () => applySelectedLayerPropChange(true));
  });

  if (chkBasePhotoEnabled) {
    chkBasePhotoEnabled.addEventListener('change', () => {
      basePhotoEnabled = chkBasePhotoEnabled.checked;
      rebuild3DFromLayers();
    });
  }
  if (sliderBasePhotoWeight) {
    sliderBasePhotoWeight.addEventListener('input', () => {
      basePhotoWeight = parseInt(sliderBasePhotoWeight.value, 10) / 100;
      if (lblBasePhotoWeight) lblBasePhotoWeight.textContent = `${Math.round(basePhotoWeight * 100)}%`;
    });
    sliderBasePhotoWeight.addEventListener('change', () => rebuild3DFromLayers());
  }
  const btnClearBasePhoto = document.getElementById('btn-clear-base-photo');
  if (btnClearBasePhoto) {
    btnClearBasePhoto.addEventListener('click', () => {
      basePhotoBuffer = null;
      // If no vector layer has 3D enabled yet, auto-enable visible layers so user sees vector 3D immediately
      const hasActive3dLayer = cadCanvas.entities.some((e) => e.visible !== false && (!e.relief || e.relief.enabled));
      if (!hasActive3dLayer) {
        cadCanvas.entities.forEach((e) => {
          cadCanvas.ensureEntityMetadata(e);
          if (e.visible !== false) e.relief.enabled = true;
        });
      }
      updateBasePhotoBox();
      renderLayerList();
      syncSelectedLayerInspector();
      rebuild3DFromLayers();
      statusMode.textContent = 'Base photo 3D cleared. 3D relief is now built purely from your editable 2D vector layers!';
    });
  }

  const chkClipBaseTo2d = document.getElementById('chk-clip-base-to-2d');
  if (chkClipBaseTo2d) {
    chkClipBaseTo2d.addEventListener('change', () => rebuild3DFromLayers(false));
  }

  function rebuild3DFromLayers(quiet = false) {
    // Finish active Pen/Polyline draft first if clicking Rebuild explicitly
    if (!quiet && cadCanvas.currentDraft) {
      cadCanvas.finishCurrentDraft();
    }
    if (viewport3D.assetMesh) {
      viewport3D.clearAssetMesh();
      const btnBackRel = document.getElementById('btn-back-relief');
      if (btnBackRel) btnBackRel.style.display = 'none';
    }
    const clipBase = chkClipBaseTo2d ? chkClipBaseTo2d.checked : true;
    const count = VectorPuffEngine.rebuildFromLayers(cadCanvas.entities, reliefEngine, {
      baseBuffer: basePhotoBuffer,
      baseEnabled: basePhotoEnabled,
      baseWeight: basePhotoWeight,
      clipBaseToEntities: clipBase,
      defaultPuff: {
        shape: document.getElementById('puff-shape')?.value || 'dome',
        heightMm: parseFloat(document.getElementById('puff-height')?.value) || 4.0,
        radiusMm: parseFloat(document.getElementById('puff-radius')?.value) || 12.0,
        mode: document.getElementById('puff-mode')?.value || 'max',
      },
    });
    dirty2DFor3D = false;
    if (!quiet) {
      if (panel3D.style.display === 'none') setViewMode('split');
      statusMode.textContent = `3D Relief updated from ${count} active 2D layers${basePhotoBuffer && basePhotoEnabled ? ' + Clipped Photo Base' : ''}.`;
    }
  }

  function extract2DFrom3D() {
    statusMode.textContent = 'Extracting editable 2D vector contour layers from 3D Relief heightmap...';
    setTimeout(() => {
      const vectors = ImageVectorizer.traceReliefToVectors(reliefEngine, {
        levels: [0.15, 0.45, 0.75],
        simplification: 0.25,
      }).map((e) => cadCanvas.ensureEntityMetadata(e));

      if (vectors.length === 0) {
        alert('3D Relief appears flat. Add a 3D heightmap or preset first before converting 3D → 2D.');
        return;
      }

      cadCanvas.saveState();
      cadCanvas.entities.push(...vectors);
      cadCanvas.selectedEntities = [...vectors];
      cadCanvas.requestRender();
      renderLayerList();
      syncSelectedLayerInspector();
      selectTool('select');
      statusMode.textContent = `Extracted ${vectors.length} editable 2D vector contour layers from 3D Relief!`;
    }, 20);
  }

  function convertSelectionToCurve() {
    const count = cadCanvas.convertSelectedToCurve();
    if (count > 0) {
      statusMode.textContent = `Converted ${count} shape(s) into editable 2D Node Curves. Use Node Edit (N) to reshape!`;
    } else {
      alert('Select a Rectangle, Circle, Line, or Polyline first to convert it to an editable Node Curve.');
    }
  }

  ['btn-rebuild-3d', 'btn-layer-rebuild-3d', 'layer-btn-apply-3d'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', () => rebuild3DFromLayers(false));
  });
  ['btn-extract-2d', 'btn-layer-extract-2d', 'tb-tool-to2d'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', extract2DFrom3D);
  });
  ['btn-to-curve', 'layer-btn-to-curve'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', convertSelectionToCurve);
  });

  function triggerJoinPaths() {
    if (cadCanvas.joinSelectedCurves(4.0)) {
      statusMode.textContent = 'Joined selected open curves cleanly into a unified outline!';
    } else {
      alert('Select 2 or more curves/lines (Ctrl/Shift-click or box select) to join/weld them together.');
    }
  }
  ['btn-join-paths', 'tb-tool-join'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', triggerJoinPaths);
  });

  function triggerSimplify() {
    const count = cadCanvas.simplifySelectedCurve(0.7);
    if (count > 0) statusMode.textContent = `Simplified nodes on ${count} selected layer(s) for easier reshaping.`;
    else alert('Select a dense traced curve first to simplify its nodes.');
  }
  const btnSimplifyCurve = document.getElementById('btn-simplify-curve');
  if (btnSimplifyCurve) btnSimplifyCurve.addEventListener('click', triggerSimplify);
  const menuTransSimplify = document.getElementById('menu-trans-simplify');
  if (menuTransSimplify) menuTransSimplify.addEventListener('click', triggerSimplify);

  function triggerDuplicate() {
    const count = cadCanvas.duplicateSelected();
    if (count > 0) statusMode.textContent = `Duplicated ${count} layer(s).`;
    else alert('Select a layer first to duplicate.');
  }
  const tbToolDuplicate = document.getElementById('tb-tool-duplicate');
  if (tbToolDuplicate) tbToolDuplicate.addEventListener('click', triggerDuplicate);
  const menuTransDuplicate = document.getElementById('menu-trans-duplicate');
  if (menuTransDuplicate) menuTransDuplicate.addEventListener('click', triggerDuplicate);

  const menuTransCenter = document.getElementById('menu-trans-center');
  if (menuTransCenter) {
    menuTransCenter.addEventListener('click', () => {
      if (cadCanvas.centerOnWorkpiece()) statusMode.textContent = 'Centered design onto workpiece stock.';
    });
  }
  const menuTransFit = document.getElementById('menu-trans-fit');
  if (menuTransFit) {
    menuTransFit.addEventListener('click', () => {
      if (cadCanvas.fitToWorkpiece(4)) statusMode.textContent = 'Scaled & fitted design cleanly onto workpiece stock.';
    });
  }
  const menuTransArray = document.getElementById('menu-trans-array');
  if (menuTransArray) {
    menuTransArray.addEventListener('click', () => {
      if (cadCanvas.selectedEntities.length === 0) {
        alert('Select a layer/motif first to repeat it as a grid array.');
        return;
      }
      const cols = Math.max(1, Math.min(8, parseInt(prompt('Columns (X repeat):', '2'), 10) || 2));
      const rows = Math.max(1, Math.min(8, parseInt(prompt('Rows (Y repeat):', '2'), 10) || 2));
      const gap = parseFloat(prompt('Spacing gap between motifs (mm):', '5')) || 5;
      const n = cadCanvas.createGridArray(cols, rows, gap);
      statusMode.textContent = `Created ${cols}×${rows} CNC Pattern Array (+${n} layers).`;
    });
  }

  const btnLayerNodeEdit = document.getElementById('layer-btn-node-edit');
  if (btnLayerNodeEdit) btnLayerNodeEdit.addEventListener('click', () => selectTool('node'));
  const btnLayerToggleCorner = document.getElementById('layer-btn-toggle-corner');
  if (btnLayerToggleCorner) {
    btnLayerToggleCorner.addEventListener('click', () => {
      if (!cadCanvas.toggleSelectedNodeCorner()) {
        statusMode.textContent = 'Select a node first in Node Edit (N) mode to toggle Corner/Smooth.';
      }
    });
  }
  const btnLayerClosePath = document.getElementById('layer-btn-close-path');
  if (btnLayerClosePath) btnLayerClosePath.addEventListener('click', () => cadCanvas.closeSelectedCurve());
  const btnLayerSmooth = document.getElementById('layer-btn-smooth');
  if (btnLayerSmooth) btnLayerSmooth.addEventListener('click', () => cadCanvas.smoothSelectedCurve());
  const btnLayerBreakHoles = document.getElementById('layer-btn-break-holes');
  if (btnLayerBreakHoles) {
    btnLayerBreakHoles.addEventListener('click', () => {
      const released = cadCanvas.breakApartCompound();
      if (released > 0) statusMode.textContent = `Released ${released} compound hole(s) into independent editable layers.`;
      else alert('Selected layer has no compound inner holes to release.');
    });
  }
  const btnLayerBreakNode = document.getElementById('layer-btn-break-node');
  if (btnLayerBreakNode) {
    btnLayerBreakNode.addEventListener('click', () => {
      if (cadCanvas.breakAtSelectedNode()) {
        renderLayerList();
        statusMode.textContent = 'Split path cleanly into editable Start (S) & End (E) points! Click ➕ Add End Point or drag with Pen (B).';
      } else {
        statusMode.textContent = 'Select a layer (or pick an inner node in Node Edit mode) to split into Start & End points.';
      }
    });
  }
  function triggerExtendEndpoint() {
    if (cadCanvas.startDrawingFromSelectedEndpoint('pen')) {
      selectTool('pen');
      statusMode.textContent = 'Continuing from Endpoint — Click anywhere to add End Points (Click Start point S to close, or Enter / R-Click to finish)!';
    } else {
      selectTool('pen');
      statusMode.textContent = 'Click any open Start (S) / End (E) point with Pen (B) to continue drawing from it!';
    }
  }
  const btnLayerExtendEnd = document.getElementById('layer-btn-extend-end');
  if (btnLayerExtendEnd) btnLayerExtendEnd.addEventListener('click', triggerExtendEndpoint);
  const btnExtendPathBar = document.getElementById('btn-extend-path');
  if (btnExtendPathBar) btnExtendPathBar.addEventListener('click', triggerExtendEndpoint);

  const btnLayerReversePath = document.getElementById('layer-btn-reverse-path');
  if (btnLayerReversePath) {
    btnLayerReversePath.addEventListener('click', () => {
      if (cadCanvas.reverseSelectedPath()) {
        statusMode.textContent = 'Flipped Start Point (S) ↔ End Point (E). Now you can extend from the opposite side!';
      }
    });
  }

  const btnShowAllLayers = document.getElementById('btn-layers-show-all');
  if (btnShowAllLayers) {
    btnShowAllLayers.addEventListener('click', () => {
      cadCanvas.saveState();
      cadCanvas.entities.forEach((e) => { e.visible = true; });
      cadCanvas.requestRender();
      renderLayerList();
      scheduleAutoSync3D();
    });
  }
  const btnDelSelLayers = document.getElementById('btn-layers-delete-sel');
  if (btnDelSelLayers) {
    btnDelSelLayers.addEventListener('click', () => triggerDelete());
  }

  // Trace Detail Sensitivity Slider readout
  const sliderTraceDetail = document.getElementById('slider-trace-detail');
  const lblTraceDetail = document.getElementById('lbl-trace-detail');
  if (sliderTraceDetail) {
    sliderTraceDetail.addEventListener('input', () => {
      if (lblTraceDetail) lblTraceDetail.textContent = `${sliderTraceDetail.value}%`;
    });
  }

  function getTraceOptions() {
    return {
      threshold: parseFloat(document.getElementById('trace-threshold-mode')?.value) || -1,
      invert: document.getElementById('relief-invert-check')?.checked || false,
      traceMode: document.getElementById('trace-mode-select')?.value || 'auto',
      detail: (parseInt(sliderTraceDetail?.value || '70', 10)) / 100,
      colorCount: Math.max(2, Math.min(16, parseInt(document.getElementById('trace-colors')?.value || '6', 10) || 6)),
      stackMethod: document.getElementById('trace-stack-mode')?.value || 'stacked',
      corners: (parseInt(document.getElementById('trace-corners')?.value || '65', 10)) / 100,
      noisePx: Math.max(0, parseInt(document.getElementById('trace-noise')?.value || '10', 10) || 0),
      ignoreBg: document.getElementById('trace-ignore-bg')?.checked ?? true,
      maxDepthMm: reliefEngine.maxDepth || 8.0,
    };
  }

  // ============================================================
  // AUTO-TRACE PICTURE TO 2D VECTORS (Step-by-Step Instant 2D Conversion!)
  // ============================================================
  async function runAutoTrace2D() {
    if (!loadedImageElement) {
      alert('Please load a picture first using "🖼️ Load Picture...".');
      return;
    }

    const opts = getTraceOptions();
    statusMode.textContent = `Step 1/5: Initializing (${opts.traceMode.toUpperCase()}, ${Math.round(opts.detail * 100)}% detail)...`;
    const count = await cadCanvas.autoTraceReferenceImageAsync({
      ...opts,
      smoothCurves: true,
      autoEnable3D: true,
    }, (prog) => {
      statusMode.textContent = prog.label;
    });
    selectTool('select');
    renderLayerList();
    syncSelectedLayerInspector();
    statusMode.textContent = `Auto-Trace complete! Generated ${count} accurate 2D layers (${opts.traceMode}). Use Click Pen (Q) or Node Edit (N) to customize any curve!`;
  }

  document.getElementById('btn-auto-trace-2d').addEventListener('click', runAutoTrace2D);
  document.getElementById('btn-side-trace-2d').addEventListener('click', runAutoTrace2D);
  document.getElementById('menu-file-trace-2d').addEventListener('click', runAutoTrace2D);

  // ============================================================
  // 2D VECTOR TO 3D RELIEF PUFFING (Pen / Curves / Shapes → 3D)
  // ============================================================
  function runVectorPuff() {
    // Automatically finish any in-progress Pen / Polyline draft so user doesn't get stuck!
    if (cadCanvas.currentDraft) {
      cadCanvas.finishCurrentDraft();
    }

    const shape = document.getElementById('puff-shape')?.value || 'dome';
    const heightMm = parseFloat(document.getElementById('puff-height')?.value) || 4.0;
    const radiusMm = parseFloat(document.getElementById('puff-radius')?.value) || 12.0;
    const mode = document.getElementById('puff-mode')?.value || 'max';

    const targetList = cadCanvas.selectedEntities.length > 0 ? cadCanvas.selectedEntities : cadCanvas.entities;

    if (!targetList || targetList.length === 0) {
      alert('No 2D vector entities found on canvas. Draw with Pen (B), Shapes, or Auto-Trace a picture first.');
      return;
    }

    // Store puff config onto each target layer so Layer Rebuild keeps it
    for (const ent of targetList) {
      cadCanvas.ensureEntityMetadata(ent);
      ent.relief.enabled = true;
      ent.relief.shape = shape;
      ent.relief.heightMm = heightMm;
      ent.relief.radiusMm = radiusMm;
      ent.relief.mode = mode;
    }

    statusMode.textContent = `Converting ${targetList.length} 2D vector(s) to 3D Relief (${shape.toUpperCase()})...`;
    setTimeout(() => {
      // Rebuild cleanly from layers so shrinking/reshaping outlines trims old unedited 3D peaks too!
      rebuild3DFromLayers(false);
      renderLayerList();
      syncSelectedLayerInspector();
      if (panel3D.style.display === 'none') setViewMode('split');
      statusMode.textContent = `Converted ${targetList.length} 2D layer(s) into 3D Relief (${shape.toUpperCase()} profile, ${heightMm}mm)!`;
    }, 20);
  }

  ['btn-puff-3d', 'btn-side-puff', 'btn-do-puff', 'tb-tool-puff'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', runVectorPuff);
  });

  // ============================================================
  // ONE-CLICK AUTO DUAL CONVERSION (100% Synced 2D Vector Layers + 3D Relief!)
  // ============================================================
  async function runOneClickAuto() {
    if (!loadedImageElement) {
      alert('Please load a picture first using "🖼️ Load Picture...".');
      return;
    }

    const opts = getTraceOptions();
    const style = document.getElementById('relief-style')?.value || 'sculpted';
    const edgeStrength = parseFloat(document.getElementById('relief-edge-strength')?.value) || 0.35;
    const removeWhiteBg = document.getElementById('relief-remove-bg')?.checked ?? true;

    statusMode.textContent = 'Step 1/5: Preparing relief depth field & image analysis...';
    await new Promise((r) => setTimeout(r, 0));

    // 1. Pre-compute high-accuracy relief heightmap buffer
    reliefEngine.loadImageHighAccuracy(loadedImageElement, {
      style,
      edgeStrength,
      invert: opts.invert,
      removeWhiteBg,
      denoise: true,
    });
    basePhotoBuffer = new Float32Array(reliefEngine.data);

    // 2. Step-by-step Illustrator-Grade 2D compound Bezier vector trace (with autoEnable3D: true!)
    const vectorCount = await cadCanvas.autoTraceReferenceImageAsync({
      ...opts,
      smoothCurves: true,
      autoEnable3D: true,
    }, (prog) => {
      statusMode.textContent = prog.label;
    });

    // 3. Drive 3D Relief cleanly from traced 2D layers (+ sculpted wood slab base on bas-relief doors)
    if (vectorCount > 0) {
      const tracedTail = cadCanvas.entities.slice(-vectorCount);
      const isWoodBasRelief = tracedTail.some((e) => e.layerGroup === 'Raised Petals & Motifs' || e.layerGroup === 'Carved Grooves & Panels');
      basePhotoEnabled = isWoodBasRelief;
      basePhotoWeight = isWoodBasRelief ? 0.65 : 0.25;
      if (chkClipBaseTo2d) chkClipBaseTo2d.checked = !isWoodBasRelief;
      updateBasePhotoBox();
      rebuild3DFromLayers(true);
    } else {
      basePhotoEnabled = true;
      basePhotoWeight = 1.0;
      updateBasePhotoBox();
    }

    selectTool('select');
    renderLayerList();
    syncSelectedLayerInspector();
    if (panel3D.style.display === 'none') setViewMode('split');
    statusMode.textContent = `1-Click Complete! ${vectorCount} compound 2D Layers ↔ 3D Relief synced 100% (${opts.traceMode}). Use Click Pen (Q) to reshape any layer!`;
  }

  const btnOneClick = document.getElementById('btn-one-click-auto');
  if (btnOneClick) btnOneClick.addEventListener('click', runOneClickAuto);

  // ============================================================
  // HIGH-ACCURACY PICTURE TO 3D RELIEF CONVERSION
  // ============================================================
  function convertPicTo3DRelief() {
    if (!loadedImageElement) {
      alert('Please load a background picture first using "🖼️ Load Picture...".');
      return;
    }

    const style = document.getElementById('relief-style')?.value || 'sculpted';
    const edgeStrength = parseFloat(document.getElementById('relief-edge-strength')?.value) || 0.35;
    const invert = document.getElementById('relief-invert-check')?.checked || false;
    const removeWhiteBg = document.getElementById('relief-remove-bg')?.checked ?? true;

    statusMode.textContent = 'Computing High-Accuracy 3D Relief (Bilateral Denoising & Edge Sharpening)...';
    setTimeout(() => {
      if (viewport3D.assetMesh) {
        viewport3D.clearAssetMesh();
        const btnBackRel = document.getElementById('btn-back-relief');
        if (btnBackRel) btnBackRel.style.display = 'none';
      }
      reliefEngine.loadImageHighAccuracy(loadedImageElement, {
        style,
        edgeStrength,
        invert,
        removeWhiteBg,
        denoise: true,
      });
      basePhotoBuffer = new Float32Array(reliefEngine.data);
      basePhotoEnabled = true;
      basePhotoWeight = 1.0;
      updateBasePhotoBox();
      if (panel3D.style.display === 'none') setViewMode('split');
      statusMode.textContent = 'High-Accuracy 3D Relief generated! Adjust Base Layer or add/edit 2D Vector Layers.';
    }, 30);
  }

  document.getElementById('btn-pic-to-relief').addEventListener('click', convertPicTo3DRelief);
  document.getElementById('btn-side-convert-pic').addEventListener('click', convertPicTo3DRelief);
  document.getElementById('menu-file-convert-3d').addEventListener('click', convertPicTo3DRelief);

  // ============================================================
  // VOLUMETRIC IMAGE-TO-3D ASSET STUDIO (Meshy Workflow + PBR)
  // ============================================================
  function readAssetMaterialAdjustment() {
    const r = (parseInt(document.getElementById('asset-roughness')?.value || '55', 10)) / 100;
    const m = (parseInt(document.getElementById('asset-metalness')?.value || '8', 10)) / 100;
    const c = (parseInt(document.getElementById('asset-contrast')?.value || '100', 10)) / 100;
    const i = (parseInt(document.getElementById('asset-intensity')?.value || '100', 10)) / 100;
    return { roughness: r, metalness: m, contrast: c, intensity: i };
  }

  async function generateVolumetric3DAsset() {
    if (!loadedImageElement) {
      alert('Please load a picture first using "🖼️ Load Picture...".');
      return;
    }

    const assetMode = document.getElementById('asset-mode-select')?.value || 'double-sided';
    const resolution = parseInt(document.getElementById('asset-res-select')?.value || '144', 10) || 144;
    const thicknessMm = Math.max(2, parseFloat(document.getElementById('asset-thickness')?.value || '22') || 22);
    const removeBg = document.getElementById('asset-remove-bg')?.checked ?? true;
    const delight = document.getElementById('asset-delight')?.checked ?? true;
    const invert = document.getElementById('relief-invert-check')?.checked || false;

    statusMode.textContent = `Step 1/3: Isolating subject silhouette & synthesizing PBR maps (${resolution}×${resolution})...`;
    await new Promise((r) => setTimeout(r, 0));

    const analysis = Asset3DEngine.analyzeImage(loadedImageElement, {
      resolution,
      removeBg,
      delight,
      invert,
      edgeStrength: parseFloat(document.getElementById('relief-edge-strength')?.value || '0.35') || 0.35,
    });

    statusMode.textContent = `Step 2/3: Reconstructing closed ${assetMode.toUpperCase()} volumetric 3D mesh...`;
    await new Promise((r) => setTimeout(r, 0));

    const meshData = Asset3DEngine.buildAssetMeshData(analysis, {
      assetMode,
      widthMm: reliefEngine.width || 100,
      heightMm: reliefEngine.height || 100,
      thicknessMm,
    });

    statusMode.textContent = 'Step 3/3: Uploading PBR Base Color, Normal, Roughness & Metallic maps to WebGL...';
    await new Promise((r) => setTimeout(r, 0));

    viewport3D.setMaterialAdjustment(readAssetMaterialAdjustment());
    viewport3D.setAssetMesh(meshData, analysis.maps);

    const btnBackRel = document.getElementById('btn-back-relief');
    if (btnBackRel) btnBackRel.style.display = 'inline-block';
    const assetTabBtn = document.getElementById('tab-btn-asset');
    if (assetTabBtn) assetTabBtn.click();
    if (panel3D.style.display === 'none') setViewMode('split');

    viewport3D.target.set((reliefEngine.width || 100) * 0.5, (reliefEngine.height || 100) * 0.5, 0);
    viewport3D.updateCameraPosition();

    const { faces, vertices } = meshData.stats;
    statusMode.textContent = `3D Asset ready (${assetMode}) — ${faces.toLocaleString()} triangles, ${vertices.toLocaleString()} vertices! Export .GLB / .STL / .OBJ anytime.`;
  }

  document.getElementById('btn-img-to-asset')?.addEventListener('click', generateVolumetric3DAsset);
  document.getElementById('btn-generate-asset')?.addEventListener('click', generateVolumetric3DAsset);

  // ============================================================
  // Transform Tools (Mirror, Rotate, Offset)
  // ============================================================
  document.getElementById('menu-trans-mirror-h').addEventListener('click', () => {
    cadCanvas.mirrorHorizontal();
    statusMode.textContent = 'Mirrored horizontally (Flip X)';
  });
  document.getElementById('menu-trans-mirror-v').addEventListener('click', () => {
    cadCanvas.mirrorVertical();
    statusMode.textContent = 'Mirrored vertically (Flip Y)';
  });
  document.getElementById('menu-trans-rot90').addEventListener('click', () => {
    cadCanvas.rotate90();
    statusMode.textContent = 'Rotated 90° clockwise';
  });

  function triggerOffset() {
    cadCanvas.offsetSelected(2.0);
    statusMode.textContent = 'Applied +2mm parallel normal contour offset to selected vector(s).';
  }
  function triggerOffsetInward() {
    cadCanvas.offsetSelected(-2.0);
    statusMode.textContent = 'Applied -2mm inward pocket contour offset to selected vector(s).';
  }
  function triggerFillet() {
    const count = cadCanvas.filletSelected(3.0);
    if (count > 0) statusMode.textContent = `Applied 3.0mm rounded corner fillet to ${count} shape(s).`;
    else alert('Select a Rectangle, Polyline, or Curve first to round its corners.');
  }
  document.getElementById('menu-trans-offset').addEventListener('click', triggerOffset);
  document.getElementById('menu-trans-offset-in')?.addEventListener('click', triggerOffsetInward);
  document.getElementById('menu-trans-fillet')?.addEventListener('click', triggerFillet);
  document.getElementById('btn-fillet-corners')?.addEventListener('click', triggerFillet);
  document.getElementById('tb-tool-offset').addEventListener('click', triggerOffset);

  // Precision Alignment Tools (Menus & Inspector Strip)
  const alignMappings = [
    ['btn-align-left', 'menu-align-left', 'left', 'Aligned Left edges'],
    ['btn-align-center-h', 'menu-align-center-h', 'centerH', 'Aligned Horizontal Centers'],
    ['btn-align-right', 'menu-align-right', 'right', 'Aligned Right edges'],
    ['btn-align-top', 'menu-align-top', 'top', 'Aligned Top edges'],
    ['btn-align-middle-v', 'menu-align-middle-v', 'middleV', 'Aligned Vertical Centers'],
    ['btn-align-bottom', 'menu-align-bottom', 'bottom', 'Aligned Bottom edges'],
  ];
  for (const [btnId, menuId, mode, msg] of alignMappings) {
    const handler = () => {
      if (cadCanvas.alignSelected(mode)) statusMode.textContent = msg;
      else statusMode.textContent = 'Select 1 layer (align to Stock) or multiple layers first.';
    };
    document.getElementById(btnId)?.addEventListener('click', handler);
    document.getElementById(menuId)?.addEventListener('click', handler);
  }
  document.getElementById('btn-align-center-stock')?.addEventListener('click', () => {
    if (cadCanvas.centerOnWorkpiece()) statusMode.textContent = 'Centered design onto stock workpiece center.';
  });

  // SVG Vector File Import into editable 2D CAD Layers
  function handleSvgImport(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      const svgText = String(evt.target?.result || '');
      const imported = CadCanvas.parseSVGToEntities(svgText, cadCanvas.workpieceWidth, cadCanvas.workpieceHeight)
        .map((ent) => cadCanvas.ensureEntityMetadata(ent));
      if (imported.length === 0) {
        alert('No supported vector shapes or <path> elements found in SVG file.');
        return;
      }
      cadCanvas.saveState();
      cadCanvas.entities.push(...imported);
      cadCanvas.selectedEntities = [...imported];
      cadCanvas.requestRender();
      cadCanvas.notifyEntitiesChange();
      cadCanvas.notifySelectionChange();
      selectTool('select');
      statusMode.textContent = `Imported ${imported.length} clean 2D vector layer(s) from "${file.name}"! Ready for 3D Relief or CNC Toolpaths.`;
    };
    reader.readAsText(file);
    e.target.value = '';
  }
  document.getElementById('input-import-svg')?.addEventListener('change', handleSvgImport);
  document.getElementById('input-import-svg-bar')?.addEventListener('change', handleSvgImport);

  // Stock Board Size Presets
  document.getElementById('prop-stock-preset')?.addEventListener('change', (e) => {
    if (!e.target.value) return;
    const [w, h, z] = e.target.value.split(',').map(Number);
    if (w && h && z) {
      document.getElementById('prop-width').value = w;
      document.getElementById('prop-height').value = h;
      document.getElementById('prop-relief-depth').value = z;
      syncSettings();
      cadCanvas.centerView();
      viewport3D.resetCamera();
      statusMode.textContent = `Stock size updated to ${w}×${h}×${z} mm.`;
    }
  });

  // ============================================================
  // Relief Operations
  // ============================================================
  document.getElementById('menu-relief-rosette').addEventListener('click', () => {
    basePhotoBuffer = null; updateBasePhotoBox();
    reliefEngine.loadPreset('rosette');
    statusMode.textContent = 'Loaded Classical Floral Medallion relief';
  });
  document.getElementById('menu-relief-dome').addEventListener('click', () => {
    basePhotoBuffer = null; updateBasePhotoBox();
    reliefEngine.loadPreset('dome');
    statusMode.textContent = 'Applied Radial Dome / Surface Puff';
  });
  document.getElementById('menu-relief-pyramid').addEventListener('click', () => {
    basePhotoBuffer = null; updateBasePhotoBox();
    reliefEngine.loadPreset('pyramid');
    statusMode.textContent = 'Applied Beveled Pyramid relief';
  });
  function smoothRelief() {
    reliefEngine.smooth(2);
    statusMode.textContent = 'Relief surface smoothed (Noise filter applied)';
  }
  function invertRelief() {
    reliefEngine.invert();
    statusMode.textContent = 'Relief inverted (Male/Female mold toggled)';
  }
  function clearRelief() {
    if (confirm('Clear 3D relief heights to flat plane?')) {
      reliefEngine.clear();
      statusMode.textContent = 'Relief heights cleared';
    }
  }

  document.getElementById('menu-relief-smooth').addEventListener('click', smoothRelief);
  document.getElementById('btn-side-smooth').addEventListener('click', smoothRelief);
  document.getElementById('menu-relief-invert').addEventListener('click', invertRelief);
  document.getElementById('btn-side-invert').addEventListener('click', invertRelief);
  document.getElementById('menu-relief-clear').addEventListener('click', clearRelief);

  // ============================================================
  // View Modes (Split, Top 2D, Iso 3D) & Resizable Splitter + Responsive Inspector Drawer
  // ============================================================
  const panelCad = document.getElementById('panel-cad');
  const panel3D = document.getElementById('panel-3d');
  const splitResizer = document.getElementById('split-resizer');
  const btnViewTop = document.getElementById('tb-view-top');
  const btnViewIso = document.getElementById('tb-view-iso');
  const btnViewSplit = document.getElementById('tb-view-split');

  function setViewMode(mode) {
    [btnViewTop, btnViewIso, btnViewSplit].forEach((b) => b && b.classList.remove('active'));
    panelCad.style.flex = '1 1 50%';
    panel3D.style.flex = '1 1 50%';

    if (mode === 'top') {
      if (btnViewTop) btnViewTop.classList.add('active');
      panelCad.style.display = 'flex';
      panel3D.style.display = 'none';
      if (splitResizer) splitResizer.style.display = 'none';
      statusMode.textContent = 'View: 2D CAD Vector Drafting (Top View)';
    } else if (mode === 'iso') {
      if (btnViewIso) btnViewIso.classList.add('active');
      panelCad.style.display = 'none';
      panel3D.style.display = 'flex';
      if (splitResizer) splitResizer.style.display = 'none';
      if (dirty2DFor3D && (cadCanvas.entities.length > 0 || basePhotoBuffer)) rebuild3DFromLayers(true);
      statusMode.textContent = 'View: 3D Relief & Toolpaths (Isometric View)';
    } else {
      if (btnViewSplit) btnViewSplit.classList.add('active');
      panelCad.style.display = 'flex';
      panel3D.style.display = 'flex';
      if (splitResizer) splitResizer.style.display = 'block';
      if (dirty2DFor3D && (cadCanvas.entities.length > 0 || basePhotoBuffer)) rebuild3DFromLayers(true);
      statusMode.textContent = 'View: Split Screen (2D CAD + 3D Relief)';
    }
    cadCanvas.resize();
    viewport3D.onResize();
  }

  // Draggable 2D <-> 3D Split Resizer (Desktop mouse & tablet touch)
  if (splitResizer) {
    let draggingSplit = false;
    const applySplitPos = (clientX, clientY) => {
      const container = document.getElementById('viewport-container');
      const rect = container.getBoundingClientRect();
      const isVerticalStack = window.innerWidth <= 1100;
      const pct = isVerticalStack
        ? Math.max(20, Math.min(80, ((clientY - rect.top) / Math.max(1, rect.height)) * 100))
        : Math.max(20, Math.min(80, ((clientX - rect.left) / Math.max(1, rect.width)) * 100));
      panelCad.style.flex = `0 0 ${pct}%`;
      panel3D.style.flex = `1 1 ${100 - pct}%`;
      cadCanvas.resize();
      viewport3D.onResize();
    };
    splitResizer.addEventListener('mousedown', (e) => { e.preventDefault(); draggingSplit = true; splitResizer.classList.add('dragging'); });
    splitResizer.addEventListener('touchstart', (e) => { if (e.touches.length === 1) { draggingSplit = true; splitResizer.classList.add('dragging'); } }, { passive: true });
    window.addEventListener('mousemove', (e) => { if (draggingSplit) applySplitPos(e.clientX, e.clientY); });
    window.addEventListener('touchmove', (e) => { if (draggingSplit && e.touches.length === 1) applySplitPos(e.touches[0].clientX, e.touches[0].clientY); }, { passive: true });
    const endSplit = () => { if (draggingSplit) { draggingSplit = false; splitResizer.classList.remove('dragging'); } };
    window.addEventListener('mouseup', endSplit);
    window.addEventListener('touchend', endSplit);
  }

  // Responsive Inspector Drawer Toggle & Window Maximize Control
  const inspectorPane = document.querySelector('.win-inspector-right');
  const btnToggleInspector = document.getElementById('tb-toggle-inspector');
  if (btnToggleInspector && inspectorPane) {
    if (window.innerWidth <= 960) {
      inspectorPane.classList.add('collapsed');
      btnToggleInspector.classList.remove('active');
    }
    btnToggleInspector.addEventListener('click', () => {
      const collapsed = inspectorPane.classList.toggle('collapsed');
      btnToggleInspector.classList.toggle('active', !collapsed);
      setTimeout(() => { cadCanvas.resize(); viewport3D.onResize(); }, 20);
    });
  }
  document.querySelector('.win-max')?.addEventListener('click', () => {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
    else document.exitFullscreen?.();
  });

  // Vector Interior Fill Tint & Illustrator Color Swatch Toggles
  const btnToggleFills = document.getElementById('btn-toggle-fills');
  if (btnToggleFills) {
    btnToggleFills.addEventListener('click', () => {
      btnToggleFills.classList.toggle('active');
      cadCanvas.showFills = btnToggleFills.classList.contains('active');
      cadCanvas.requestRender();
    });
  }
  const btnToggleColorFills = document.getElementById('btn-toggle-color-fills');
  if (btnToggleColorFills) {
    btnToggleColorFills.addEventListener('click', () => {
      btnToggleColorFills.classList.toggle('active');
      cadCanvas.showColorFills = btnToggleColorFills.classList.contains('active');
      cadCanvas.requestRender();
    });
  }

  if (btnViewTop) btnViewTop.addEventListener('click', () => setViewMode('top'));
  if (btnViewIso) btnViewIso.addEventListener('click', () => setViewMode('iso'));
  if (btnViewSplit) btnViewSplit.addEventListener('click', () => setViewMode('split'));

  document.getElementById('menu-view-cad').addEventListener('click', () => setViewMode('top'));
  document.getElementById('menu-view-3d').addEventListener('click', () => setViewMode('iso'));
  document.getElementById('menu-view-split').addEventListener('click', () => setViewMode('split'));

  // Fit & Zoom
  function fitView() {
    cadCanvas.centerView();
    viewport3D.resetCamera();
    statusMode.textContent = 'Workpiece centered in view';
  }
  document.getElementById('tb-fit').addEventListener('click', fitView);
  document.getElementById('menu-view-fit').addEventListener('click', fitView);

  document.getElementById('tb-zoomin').addEventListener('click', () => {
    cadCanvas.scale *= 1.25;
    cadCanvas.render();
  });
  document.getElementById('tb-zoomout').addEventListener('click', () => {
    cadCanvas.scale /= 1.25;
    cadCanvas.render();
  });
  document.getElementById('menu-view-zoomin').addEventListener('click', () => {
    cadCanvas.scale *= 1.25;
    cadCanvas.render();
  });
  document.getElementById('menu-view-zoomout').addEventListener('click', () => {
    cadCanvas.scale /= 1.25;
    cadCanvas.render();
  });

  // Snapping Toggles
  const btnSnapGrid = document.getElementById('snap-grid');
  const btnSnapEnd = document.getElementById('snap-end');
  const btnSnapOrtho = document.getElementById('snap-ortho');

  btnSnapGrid.addEventListener('click', function () {
    this.classList.toggle('active');
    cadCanvas.gridSnap = this.classList.contains('active');
    cadCanvas.render();
    updateSnapStatus();
  });

  btnSnapEnd.addEventListener('click', function () {
    this.classList.toggle('active');
    cadCanvas.objectSnap = this.classList.contains('active');
    updateSnapStatus();
  });

  btnSnapOrtho.addEventListener('click', function () {
    this.classList.toggle('active');
    updateSnapStatus();
  });

  function updateSnapStatus() {
    statusSnap.textContent = `Snap: ${cadCanvas.objectSnap ? 'ON' : 'OFF'} | Grid: ${cadCanvas.gridSnap ? 'ON' : 'OFF'} | Ortho: ${btnSnapOrtho.classList.contains('active') ? 'ON' : 'OFF'}`;
  }

  // 3D Material Shading Selector
  const selectMaterial = document.getElementById('select-material');
  const label3dMaterial = document.getElementById('vp-material-label');
  selectMaterial.addEventListener('change', (e) => {
    const mat = e.target.value;
    viewport3D.setMaterial(mat);
    label3dMaterial.textContent = `Material: ${e.target.options[e.target.selectedIndex].text}`;
  });

  // Toggle Wireframe & Toolpaths
  const btnWireframe = document.getElementById('tb-wireframe');
  btnWireframe.addEventListener('click', function () {
    this.classList.toggle('active');
    viewport3D.setWireframe(this.classList.contains('active'));
  });

  const btnShowPaths = document.getElementById('tb-show-paths');
  btnShowPaths.addEventListener('click', function () {
    this.classList.toggle('active');
    viewport3D.setToolpathVisibility(this.classList.contains('active'));
  });

  document.getElementById('btn-recenter-3d').addEventListener('click', () => {
    viewport3D.resetCamera();
  });

  // Meshy-style 3D Studio HUD Controls (Shading Mode, PBR Channels, Turntable, Stats, & Material Adjustment)
  const stat3dTopo = document.getElementById('stat-3d-topo');
  const stat3dFaces = document.getElementById('stat-3d-faces');
  const stat3dVerts = document.getElementById('stat-3d-verts');
  viewport3D.onStatsChange = (stats) => {
    if (stat3dTopo) stat3dTopo.textContent = stats.topology || 'Triangle';
    if (stat3dFaces) stat3dFaces.textContent = (stats.faces || 0).toLocaleString();
    if (stat3dVerts) stat3dVerts.textContent = (stats.vertices || 0).toLocaleString();
  };
  viewport3D._emitStats();

  document.querySelectorAll('#three-studio-hud [data-shade]').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#three-studio-hud [data-shade]').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      viewport3D.setShadingMode(btn.dataset.shade);
    });
  });

  document.querySelectorAll('#three-studio-hud [data-channel]').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#three-studio-hud [data-channel]').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      viewport3D.setPbrChannel(btn.dataset.channel);
    });
  });

  const btnTurntable3d = document.getElementById('btn-turntable-3d');
  if (btnTurntable3d) {
    btnTurntable3d.addEventListener('click', () => {
      btnTurntable3d.classList.toggle('active');
      viewport3D.setTurntable(btnTurntable3d.classList.contains('active'));
    });
  }

  const btnBackRelief = document.getElementById('btn-back-relief');
  if (btnBackRelief) {
    btnBackRelief.addEventListener('click', () => {
      viewport3D.clearAssetMesh();
      btnBackRelief.style.display = 'none';
      statusMode.textContent = 'Switched back to 2.5D CNC Relief Slab view.';
    });
  }

  const bindAssetSlider = (sliderId, lblId, key) => {
    const el = document.getElementById(sliderId);
    const lbl = document.getElementById(lblId);
    if (!el) return;
    el.addEventListener('input', () => {
      const val = parseInt(el.value, 10) / 100;
      if (lbl) lbl.textContent = val.toFixed(2);
      viewport3D.setMaterialAdjustment({ [key]: val });
    });
  };
  bindAssetSlider('asset-roughness', 'lbl-asset-rough', 'roughness');
  bindAssetSlider('asset-metalness', 'lbl-asset-metal', 'metalness');
  bindAssetSlider('asset-contrast', 'lbl-asset-contrast', 'contrast');
  bindAssetSlider('asset-intensity', 'lbl-asset-intensity', 'intensity');

  // ============================================================
  // CNC CAM Toolpath Calculations
  // ============================================================
  function updateCamStats() {
    const s = camEngine.stats;
    document.getElementById('stat-points').textContent = s.points.toLocaleString();
    document.getElementById('stat-distance').textContent = `${(s.cutDistance / 1000).toFixed(2)} m`;
    const min = Math.floor(s.estimatedTimeSec / 60);
    const sec = s.estimatedTimeSec % 60;
    document.getElementById('stat-time').textContent = `${min} min ${sec < 10 ? '0' : ''}${sec} s`;
  }

  function runSurfaceCAM() {
    statusMode.textContent = 'Calculating 3D Surface Relief Toolpath with Surface Normal Compensation (Ball-Nose)...';
    setTimeout(() => {
      syncSettings();
      activeToolpathSegments = camEngine.calculate3DReliefToolpath(reliefEngine);
      viewport3D.renderToolpaths(activeToolpathSegments);
      updateCamStats();
      btnShowPaths.classList.add('active');
      viewport3D.setToolpathVisibility(true);
      statusMode.textContent = `Toolpath calculated: ${camEngine.stats.points.toLocaleString()} points. Ready to export CNC G-Code.`;
    }, 20);
  }

  function runProfileCAM() {
    statusMode.textContent = 'Calculating 2D Contour Profile Cut...';
    setTimeout(() => {
      syncSettings();
      activeToolpathSegments = camEngine.calculate2DProfileToolpath(cadCanvas, 3.0, 1.0);
      viewport3D.renderToolpaths(activeToolpathSegments);
      updateCamStats();
      btnShowPaths.classList.add('active');
      viewport3D.setToolpathVisibility(true);
      statusMode.textContent = `2D Profile toolpath calculated: ${camEngine.stats.points.toLocaleString()} points.`;
    }, 20);
  }

  document.getElementById('tb-calc-cam').addEventListener('click', runSurfaceCAM);
  document.getElementById('btn-calc-surface-toolpath').addEventListener('click', runSurfaceCAM);
  document.getElementById('menu-cam-surface').addEventListener('click', runSurfaceCAM);
  document.getElementById('btn-calc-profile-toolpath').addEventListener('click', runProfileCAM);
  document.getElementById('menu-cam-profile').addEventListener('click', runProfileCAM);

  // Interactive 3D Spindle Toolpath Simulator
  const btnSimCam = document.getElementById('tb-sim-cam');
  let isSimulating = false;
  if (btnSimCam) {
    btnSimCam.addEventListener('click', () => {
      if (isSimulating) {
        viewport3D.stopSimulation();
        isSimulating = false;
        btnSimCam.textContent = '▶ Simulate';
        btnSimCam.classList.remove('active');
        statusMode.textContent = 'CNC Toolpath simulation stopped.';
        return;
      }
      if (!activeToolpathSegments || activeToolpathSegments.length === 0) {
        syncSettings();
        activeToolpathSegments = camEngine.calculate3DReliefToolpath(reliefEngine);
        viewport3D.renderToolpaths(activeToolpathSegments);
        updateCamStats();
      }
      if (panel3D.style.display === 'none') setViewMode('split');
      isSimulating = true;
      btnSimCam.textContent = '⏹ Stop Sim';
      btnSimCam.classList.add('active');
      viewport3D.startSimulation(
        (pct, pt) => {
          statusMode.textContent = `CNC Cutting Simulation: ${pct}% — Spindle (X:${pt.x.toFixed(1)} Y:${pt.y.toFixed(1)} Z:${pt.z.toFixed(2)} mm)`;
        },
        () => {
          isSimulating = false;
          btnSimCam.textContent = '▶ Simulate';
          btnSimCam.classList.remove('active');
          statusMode.textContent = 'CNC Cutting Simulation complete! G-Code is verified and ready to export.';
        }
      );
    });
  }

  ['btn-clear-toolpath', 'menu-cam-clear'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', () => {
      activeToolpathSegments = [];
      viewport3D.renderToolpaths([]);
      updateCamStats();
      statusMode.textContent = 'Toolpaths cleared';
    });
  });

  // ============================================================
  // G-Code Dialog & Exporters
  // ============================================================
  const dialogGcode = document.getElementById('dialog-gcode');
  const gcodeTextarea = document.getElementById('gcode-preview-text');
  const selectDialect = document.getElementById('select-dialect');

  function openGcodeModal() {
    if (!activeToolpathSegments || activeToolpathSegments.length === 0) {
      runSurfaceCAM();
    }
    const dialect = selectDialect.value;
    const gcode = GcodeExporter.generate(activeToolpathSegments, camEngine.tool, dialect);
    gcodeTextarea.value = gcode;
    dialogGcode.showModal();
  }

  selectDialect.addEventListener('change', () => {
    const dialect = selectDialect.value;
    gcodeTextarea.value = GcodeExporter.generate(activeToolpathSegments, camEngine.tool, dialect);
  });

  ['menu-file-export-gcode', 'menu-cam-export', 'tb-export-nc', 'btn-export-nc-tab'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', openGcodeModal);
  });

  document.getElementById('modal-close').addEventListener('click', () => dialogGcode.close());
  document.getElementById('btn-cancel-gcode').addEventListener('click', () => dialogGcode.close());

  document.getElementById('btn-copy-gcode').addEventListener('click', () => {
    navigator.clipboard.writeText(gcodeTextarea.value);
    alert('CNC G-Code copied to clipboard successfully!');
  });

  document.getElementById('btn-download-gcode').addEventListener('click', () => {
    const dialect = selectDialect.value;
    const ext = dialect === 'jingdiao' ? 'eng' : 'nc';
    GcodeExporter.download(`jdpaint-program.${ext}`, gcodeTextarea.value);
    dialogGcode.close();
  });

  // 3D Asset & Relief Exports (.GLB PBR, Binary .STL, Wavefront .OBJ)
  async function exportGLB() {
    statusMode.textContent = 'Exporting Binary glTF 2.0 (.glb) 3D Asset with PBR Textures...';
    const meshData = (viewport3D.assetMesh && viewport3D.assetData)
      ? viewport3D.assetData
      : StlExporter.reliefToMeshData(reliefEngine);

    let pngBytes = null;
    const baseCanvas = viewport3D.getBaseColorPngCanvas();
    if (baseCanvas && typeof baseCanvas.toBlob === 'function') {
      const pngBlob = await new Promise((resolve) => baseCanvas.toBlob(resolve, 'image/png'));
      if (pngBlob) pngBytes = new Uint8Array(await pngBlob.arrayBuffer());
    }

    const adj = readAssetMaterialAdjustment();
    const glbBlob = StlExporter.exportToGLB(meshData, {
      baseColorPngBytes: pngBytes,
      roughness: adj.roughness,
      metalness: adj.metalness,
      name: 'JDPaint3DAsset',
    });
    StlExporter.downloadBlob('jdpaint-3d-asset.glb', glbBlob);
    statusMode.textContent = `Exported Binary .GLB (${(glbBlob.size / 1024).toFixed(1)} KB, Blender/Unity/QuickLook ready).`;
  }

  function exportSTL() {
    statusMode.textContent = 'Exporting Watertight 3D Binary STL Mesh...';
    setTimeout(() => {
      const isAsset = !!(viewport3D.assetMesh && viewport3D.assetData);
      const blob = isAsset
        ? StlExporter.exportIndexedMeshToBinarySTL(viewport3D.assetData)
        : StlExporter.exportReliefToBinarySTL(reliefEngine);
      StlExporter.downloadBlob(isAsset ? 'jdpaint-3d-asset.stl' : 'jdpaint-relief.stl', blob);
      statusMode.textContent = `STL file exported (${isAsset ? 'Volumetric 3D Asset' : 'Relief Slab'}, ready for 3D printing/CAM).`;
    }, 20);
  }

  function exportOBJ() {
    statusMode.textContent = 'Exporting Wavefront OBJ model...';
    setTimeout(() => {
      const isAsset = !!(viewport3D.assetMesh && viewport3D.assetData);
      const text = isAsset
        ? StlExporter.exportIndexedMeshToOBJ(viewport3D.assetData)
        : StlExporter.exportToOBJ(reliefEngine);
      const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
      StlExporter.downloadBlob(isAsset ? 'jdpaint-3d-asset.obj' : 'jdpaint-relief.obj', blob);
      statusMode.textContent = `OBJ model exported (${isAsset ? 'with UVs & Normals' : 'Relief Surface'}).`;
    }, 20);
  }

  ['menu-file-export-glb', 'btn-export-glb-tab', 'btn-asset-export-glb'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', exportGLB);
  });
  ['menu-file-export-stl', 'btn-export-stl-tab', 'btn-asset-export-stl'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', exportSTL);
  });
  ['menu-file-export-obj', 'btn-export-obj-tab', 'btn-asset-export-obj'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', exportOBJ);
  });

  function exportDXF() {
    if (cadCanvas.entities.length === 0) {
      alert('No 2D vector entities to export. Draw or trace shapes first.');
      return;
    }
    const dxf = StlExporter.exportToDXF(cadCanvas);
    GcodeExporter.download('jdpaint-vectors.dxf', dxf);
    statusMode.textContent = `Exported ${cadCanvas.entities.length} 2D vector layers to AutoCAD DXF.`;
  }
  ['menu-file-export-dxf', 'btn-export-dxf-tab'].forEach((id) => {
    document.getElementById(id)?.addEventListener('click', exportDXF);
  });

  // Project Save / New
  function saveProject() {
    const project = {
      software: 'JDPaint 5.21 (Mac Edition)',
      version: '5.21.0',
      width: reliefEngine.width,
      height: reliefEngine.height,
      maxDepth: reliefEngine.maxDepth,
      resolution: reliefEngine.resolution,
      reliefData: Array.from(reliefEngine.data),
      cadEntities: cadCanvas.entities,
      tool: camEngine.tool,
    };
    GcodeExporter.download('jdpaint-project.jdp.json', JSON.stringify(project));
    statusMode.textContent = 'Project saved (.jdp.json)';
  }

  function newProject() {
    if (confirm('Start a new project? All unsaved work will be cleared.')) {
      cadCanvas.entities = [];
      cadCanvas.selectedEntities = [];
      cadCanvas.clearReferenceImage();
      cadCanvas.render();
      basePhotoBuffer = null;
      updateBasePhotoBox();
      reliefEngine.clear();
      activeToolpathSegments = [];
      viewport3D.renderToolpaths([]);
      renderLayerList();
      syncSelectedLayerInspector();
      statusMode.textContent = 'New project created.';
    }
  }

  ['menu-file-save', 'tb-save'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', saveProject);
  });
  ['menu-file-new', 'tb-new'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', newProject);
  });

  // ============================================================
  // UNDO / REDO (Ctrl+Z / Ctrl+Y / Ctrl+Shift+Z)
  // ============================================================
  function triggerUndo() {
    cadCanvas.undo();
    statusMode.textContent = `Undo (${cadCanvas.undoStack.length} states remaining)`;
  }
  function triggerRedo() {
    cadCanvas.redo();
    statusMode.textContent = `Redo (${cadCanvas.redoStack.length} states remaining)`;
  }
  function triggerDelete() {
    const deletable = cadCanvas.selectedEntities.filter((s) => !s.locked);
    if (deletable.length > 0) {
      cadCanvas.saveState();
      for (const sel of deletable) {
        const idx = cadCanvas.entities.indexOf(sel);
        if (idx !== -1) cadCanvas.entities.splice(idx, 1);
      }
      cadCanvas.selectedEntities = cadCanvas.selectedEntities.filter((s) => s.locked);
      cadCanvas.requestRender();
      cadCanvas.notifyEntitiesChange();
      cadCanvas.notifySelectionChange();
      statusMode.textContent = `Deleted ${deletable.length} layer(s).`;
    }
  }
  function triggerSelectAll() {
    cadCanvas.selectedEntities = cadCanvas.entities.filter((e) => e.visible !== false);
    cadCanvas.requestRender();
    cadCanvas.notifySelectionChange();
    statusMode.textContent = `Selected ${cadCanvas.selectedEntities.length} visible layers.`;
  }

  ['menu-edit-undo', 'tb-undo'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', triggerUndo);
  });
  ['menu-edit-redo', 'tb-redo'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', triggerRedo);
  });
  const elEditDelete = document.getElementById('menu-edit-delete');
  if (elEditDelete) elEditDelete.addEventListener('click', triggerDelete);
  const elEditSelectAll = document.getElementById('menu-edit-selectall');
  if (elEditSelectAll) elEditSelectAll.addEventListener('click', triggerSelectAll);

  // ============================================================
  // KEYBOARD SHORTCUTS (Single-Key Tool Switching + F3/F8/F9)
  // ============================================================
  window.addEventListener('keydown', (e) => {
    // Skip when typing in input fields
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
    // Ctrl/Cmd combos are handled in cadCanvas.initEvents for Z/Y/A
    if (e.ctrlKey || e.metaKey) return;

    switch (e.key) {
      case 'v': case 'V': selectTool('select'); break;
      case 'n': case 'N': selectTool('node'); break;
      case 'h': case 'H': selectTool('reshape'); break;
      case 'b': case 'B': selectTool('pen'); break;
      case 'q': case 'Q': selectTool('pen-edit'); break;
      case 'f': case 'F': selectTool('freehand'); break;
      case 'p': case 'P': selectTool('polyline'); break;
      case 'l': case 'L': selectTool('line'); break;
      case 'a': case 'A': selectTool('arc'); break;
      case 'r': case 'R': selectTool('rect'); break;
      case 'c': case 'C': selectTool('circle'); break;
      case 'o': case 'O': selectTool('ellipse'); break;
      case 'g': case 'G': selectTool('polygon'); break;
      case 'k': case 'K': selectTool('knife'); break;
      case 'j': case 'J': triggerJoinPaths(); break;
      case 'e': case 'E': selectTool('eraser'); break;
      case 't': case 'T': selectTool('text'); break;
      case 'F3':
        e.preventDefault();
        fitView();
        break;
      case 'F8':
        e.preventDefault();
        btnSnapOrtho.click(); // Toggle Ortho Lock
        break;
      case 'F9':
        e.preventDefault();
        runSurfaceCAM();
        break;
      default: break;
    }
  });

  // ============================================================
  // ORTHO LOCK CONSTRAINT (F8) — Constrain drawing to 90° axes
  // ============================================================
  const originalScreenToWorld = cadCanvas.screenToWorld.bind(cadCanvas);
  cadCanvas.screenToWorld = function(sx, sy, bypassSnap = false) {
    const result = originalScreenToWorld(sx, sy, bypassSnap);

    // Apply ortho constraint when active and drawing
    if (!bypassSnap && btnSnapOrtho.classList.contains('active') && cadCanvas.currentDraft) {
      const draft = cadCanvas.currentDraft;
      let refPt = null;

      if ((draft.type === 'curve' || draft.type === 'polyline') && draft.points && draft.points.length > 0) {
        refPt = draft.points[draft.points.length - 1];
      } else if (draft.type === 'line') {
        refPt = { x: draft.x1, y: draft.y1 };
      }

      if (refPt) {
        const dx = Math.abs(result.x - refPt.x);
        const dy = Math.abs(result.y - refPt.y);
        if (dx > dy) {
          result.y = refPt.y; // Lock horizontal
        } else {
          result.x = refPt.x; // Lock vertical
        }
      }
    }
    return result;
  };

  // ============================================================
  // GRID TOGGLE FROM MENU
  // ============================================================
  const menuViewGrid = document.getElementById('menu-view-grid');
  if (menuViewGrid) {
    menuViewGrid.addEventListener('click', () => {
      btnSnapGrid.click();
    });
  }

  // ============================================================
  // ZOOM PERCENTAGE DISPLAY (auto-update on render)
  // ============================================================
  const cadZoomLabel = document.getElementById('cad-zoom-label');
  const origRender = cadCanvas.render.bind(cadCanvas);
  cadCanvas.render = function() {
    origRender();
    // Update zoom label after each render
    if (cadZoomLabel) {
      const pct = Math.round(cadCanvas.scale * 100 / 2.5); // 2.5 is default = 100%
      cadZoomLabel.textContent = `Scale: ${pct}%`;
    }
  };

  function ringToSvgSubpath(points, isClosed) {
    if (!points || points.length < 2) return '';
    if (CadCanvas.hasCubicHandles(points)) {
      const ring = CadCanvas.normalizeClosedRing(points, isClosed);
      if (ring.length < 2) return '';
      const d = [`M ${ring[0].x.toFixed(3)} ${ring[0].y.toFixed(3)}`];
      const segCount = isClosed ? ring.length : ring.length - 1;
      for (let i = 0; i < segCount; i++) {
        const a = ring[i];
        const b = ring[(i + 1) % ring.length];
        if (!a.hOut && !b.hIn) {
          d.push(`L ${b.x.toFixed(3)} ${b.y.toFixed(3)}`);
        } else {
          const c1 = a.hOut || a;
          const c2 = b.hIn || b;
          d.push(`C ${c1.x.toFixed(3)} ${c1.y.toFixed(3)}, ${c2.x.toFixed(3)} ${c2.y.toFixed(3)}, ${b.x.toFixed(3)} ${b.y.toFixed(3)}`);
        }
      }
      if (isClosed) d.push('Z');
      return d.join(' ');
    }
    const sampled = CadCanvas.sampleCurvePoints(points, isClosed, 6);
    if (sampled.length < 2) return '';
    const d = [`M ${sampled[0].x.toFixed(3)} ${sampled[0].y.toFixed(3)}`];
    for (let i = 1; i < sampled.length; i++) d.push(`L ${sampled[i].x.toFixed(3)} ${sampled[i].y.toFixed(3)}`);
    if (isClosed) d.push('Z');
    return d.join(' ');
  }

  // ============================================================
  // SVG EXPORT (True Cubic Bezier + Compound Holes SVG Export)
  // ============================================================
  function exportSVG() {
    const w = cadCanvas.workpieceWidth;
    const h = cadCanvas.workpieceHeight;
    const entities = cadCanvas.entities;
    if (entities.length === 0) {
      alert('No 2D vector entities to export. Draw some shapes first.');
      return;
    }

    let svg = `<?xml version="1.0" encoding="UTF-8"?>\n`;
    svg += `<svg xmlns="http://www.w3.org/2000/svg" width="${w}mm" height="${h}mm" viewBox="0 0 ${w} ${h}">\n`;
    svg += `  <rect width="${w}" height="${h}" fill="none" stroke="#999" stroke-width="0.2"/>\n`;

    for (const ent of entities) {
      if (ent.visible === false) continue;
      const strokeCol = ent.color || '#38bdf8';
      const fillAttr = ent.isClosed && cadCanvas.showColorFills && ent.fillColor ? ent.fillColor : 'none';
      const sw = ((ent.width || 1.5) * 0.3).toFixed(3);
      const idAttr = ent.name ? ` id="${escapeXml(ent.name)}"` : '';

      if (ent.type === 'line') {
        svg += `  <line${idAttr} x1="${ent.x1.toFixed(3)}" y1="${ent.y1.toFixed(3)}" x2="${ent.x2.toFixed(3)}" y2="${ent.y2.toFixed(3)}" stroke="${strokeCol}" stroke-width="${sw}" fill="none"/>\n`;
      } else if (ent.type === 'rect') {
        const rxAttr = ent.cornerRadius > 0 ? ` rx="${ent.cornerRadius.toFixed(3)}" ry="${ent.cornerRadius.toFixed(3)}"` : '';
        svg += `  <rect${idAttr} x="${ent.x.toFixed(3)}" y="${ent.y.toFixed(3)}" width="${ent.w.toFixed(3)}" height="${ent.h.toFixed(3)}"${rxAttr} stroke="${strokeCol}" stroke-width="${sw}" fill="${fillAttr}"/>\n`;
      } else if (ent.type === 'circle') {
        svg += `  <circle${idAttr} cx="${ent.cx.toFixed(3)}" cy="${ent.cy.toFixed(3)}" r="${ent.r.toFixed(3)}" stroke="${strokeCol}" stroke-width="${sw}" fill="${fillAttr}"/>\n`;
      } else if ((ent.type === 'polyline' || ent.type === 'curve') && ent.points?.length >= 2) {
        const subpaths = [];
        if (ent.type === 'curve') {
          subpaths.push(ringToSvgSubpath(ent.points, !!ent.isClosed));
          if (ent.holes) {
            for (const hole of ent.holes) subpaths.push(ringToSvgSubpath(hole, true));
          }
        } else {
          const polySub = (pts, closed) => {
            if (!pts || pts.length < 2) return '';
            return `M ${pts.map((p) => `${p.x.toFixed(3)} ${p.y.toFixed(3)}`).join(' L ')}${closed ? ' Z' : ''}`;
          };
          subpaths.push(polySub(ent.points, !!ent.isClosed));
          if (ent.holes) {
            for (const hole of ent.holes) subpaths.push(polySub(hole, true));
          }
        }
        const dAttr = subpaths.filter(Boolean).join(' ');
        if (dAttr) {
          svg += `  <path${idAttr} d="${dAttr}" fill="${fillAttr}" fill-rule="evenodd" stroke="${strokeCol}" stroke-width="${sw}"/>\n`;
        }
      } else if (ent.type === 'text') {
        svg += `  <text${idAttr} x="${ent.x.toFixed(3)}" y="${ent.y.toFixed(3)}" font-size="${(ent.size || 16) * 0.4}" font-weight="${ent.fontWeight || 'bold'}" font-family="${escapeXml(ent.fontFamily || 'sans-serif')}" fill="${strokeCol}">${escapeXml(ent.text)}</text>\n`;
      }
    }

    svg += '</svg>\n';

    const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'jdpaint-vectors.svg';
    a.click();
    URL.revokeObjectURL(url);
    statusMode.textContent = `Exported ${entities.length} entities (Cubic Bezier SVG) to file.`;
  }

  function escapeXml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  const menuExportSvg = document.getElementById('menu-file-export-svg');
  if (menuExportSvg) menuExportSvg.addEventListener('click', exportSVG);
  const btnExportSvgTab = document.getElementById('btn-export-svg-tab');
  if (btnExportSvgTab) btnExportSvgTab.addEventListener('click', exportSVG);

  // ============================================================
  // PROJECT OPEN (Load .jdp.json file)
  // ============================================================
  function openProject() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json,.jdp.json';
    input.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (evt) => {
        try {
          const project = JSON.parse(evt.target.result);
          if (project.software && project.software.includes('JDPaint')) {
            // Restore settings
            if (project.width) document.getElementById('prop-width').value = project.width;
            if (project.height) document.getElementById('prop-height').value = project.height;
            if (project.maxDepth) document.getElementById('prop-relief-depth').value = project.maxDepth;

            // Restore relief heightmap
            if (project.reliefData && Array.isArray(project.reliefData)) {
              if (project.resolution) reliefEngine.setResolution(project.resolution);
              reliefEngine.width = project.width || 100;
              reliefEngine.height = project.height || 100;
              reliefEngine.maxDepth = project.maxDepth || 8;
              const len = Math.min(project.reliefData.length, reliefEngine.data.length);
              for (let i = 0; i < len; i++) {
                reliefEngine.data[i] = project.reliefData[i];
              }
            }

            // Restore CAD entities
            if (project.cadEntities && Array.isArray(project.cadEntities)) {
              cadCanvas.entities = project.cadEntities.map((ent) => cadCanvas.ensureEntityMetadata(ent));
              cadCanvas.selectedEntities = [];
            }

            // Restore tool config
            if (project.tool) {
              if (project.tool.type) document.getElementById('tool-type').value = project.tool.type;
              if (project.tool.diameter) document.getElementById('tool-diameter').value = project.tool.diameter;
              if (project.tool.stepover) document.getElementById('tool-stepover').value = project.tool.stepover;
            }

            basePhotoBuffer = null;
            updateBasePhotoBox();
            syncSettings();
            cadCanvas.render();
            renderLayerList();
            syncSelectedLayerInspector();
            statusMode.textContent = `Project "${file.name}" loaded successfully (${cadCanvas.entities.length} entities).`;

            // Update title bar
            document.querySelector('.win-title-text').textContent =
              `JDPaint 5.21 (Mac Edition) - CAD/CAM & 3D Relief Studio - [${file.name}]`;
            document.title = `Beijing Jingdiao CNC CAD/CAM System - [JDPaint 5.21 (Mac Edition) - ${file.name}]`;
          } else {
            alert('Invalid project file format. Expected JDPaint .jdp.json file.');
          }
        } catch (err) {
          alert(`Error loading project: ${err.message}`);
        }
      };
      reader.readAsText(file);
    });
    input.click();
  }

  ['menu-file-open', 'tb-open'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', openProject);
  });

  // ============================================================
  // FILE → EXIT
  // ============================================================
  const menuExit = document.getElementById('menu-file-exit');
  if (menuExit) {
    menuExit.addEventListener('click', () => {
      if (confirm('Exit JDPaint 5.21 (Mac Edition)? Unsaved changes will be lost.')) {
        window.close();
      }
    });
  }

  // About Dialog
  document.getElementById('menu-help-about').addEventListener('click', () => {
    alert(
      'JDPaint 5.21 (Mac Edition)\n\n' +
      'Version: 5.21.0 (Native Mac OS Engine)\n' +
      'Core Capabilities:\n' +
      '• 2D Vector CAD drafting with Pen / Bezier Curve, Polyline, Node Edit & Eraser\n' +
      '• Auto-Trace Picture into 2D CAD editable vector curves\n' +
      '• 2D Vector to 3D Relief Puffing (JDPaint Domain Forming) with Dome, Cone, Smooth S-Curve profiles\n' +
      '• High-Accuracy 3D Relief Heightmap generation with edge detection & auto-contrast\n' +
      '• Real-time manual 3D Sculpting Brushes (Raise, Carve, Smooth, Flatten)\n' +
      '• Realistic 3D PBR Shading (Jade, Walnut Wood, Brass, Granite)\n' +
      '• CNC CAM Toolpath calculation with surface normal ball-nose compensation & G-Code export\n\n' +
      'Keyboard Shortcuts:\n' +
      '  V = Select   N = Node Edit   B = Pen Curve   P = Polyline\n' +
      '  L = Line   R = Rectangle   C = Circle   E = Eraser   T = Text\n' +
      '  F3 = Fit View   F8 = Ortho Lock   F9 = Calculate Toolpaths\n' +
      '  Ctrl+Z = Undo   Ctrl+Y = Redo   Ctrl+A = Select All   Del = Delete'
    );
  });

  // ============================================================
  // INITIAL STATE
  // ============================================================
  syncSettings();
  updateSnapStatus(); // Show correct snap info on startup

  // Zoom label initial
  if (cadZoomLabel) {
    const pct = Math.round(cadCanvas.scale * 100 / 2.5);
    cadZoomLabel.textContent = `Scale: ${pct}%`;
  }
});
