// 3D WebGL Viewport using Three.js for JDPaint Mac
import * as THREE from 'three';

export class Viewport3D {
  constructor(containerElement, reliefEngine) {
    this.container = containerElement;
    this.relief = reliefEngine;

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.mesh = null;
    this.baseSlab = null;
    this.assetMesh = null;
    this.assetData = null;
    this.assetMaps = null;
    this._assetTextures = {};
    this.gridHelper = null;
    this.axesHelper = null;
    this.geometry = null;
    this.toolpathGroup = null;
    this.brushIndicator = null;
    this.cutterMesh = null;
    this.dirLight1 = null;
    this._textures = {};
    this._simTimer = null;
    this._simSegments = [];

    this.currentMaterialType = 'wood';
    this.shadingMode = 'standard'; // 'standard' | 'solid' | 'unlit' | 'lit'
    this.pbrChannel = 'pbr'; // 'pbr' | 'baseColor' | 'roughness' | 'metallic' | 'normal'
    this.materialAdjust = { roughness: 0.55, metalness: 0.08, contrast: 1.0, intensity: 1.0 };
    this.wireframe = false;
    this.showToolpaths = true;
    this.turntable = false;
    this.onStatsChange = null;

    // Sculpting brush state
    this.activeSculptMode = null; // 'raise', 'lower', 'smooth', 'flatten' or null
    this.brushRadius = 10;
    this.brushStrength = 0.3;
    this.isSculpting = false;
    this.raycaster = new THREE.Raycaster();
    this.mouseCoord = new THREE.Vector2();

    // Camera control state
    this.isRotating = false;
    this.isPanning = false;
    this.prevMouse = { x: 0, y: 0 };
    this.spherical = { radius: 175, theta: Math.PI / 5, phi: Math.PI / 4.0 };
    this.target = new THREE.Vector3(50, 50, 0);

    this.init();
    this.buildMesh();

    this.relief.onChange(() => {
      this.updateMeshHeights();
    });
  }

  init() {
    const width = this.container.clientWidth || 600;
    const height = this.container.clientHeight || 500;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0c1017);

    this.camera = new THREE.PerspectiveCamera(44, width / height, 1, 3000);
    this.updateCameraPosition();

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.12;
    this.container.appendChild(this.renderer.domElement);

    // 3-Point Studio Raking Lighting with 2048px soft shadows so every 3D petal/groove reads crisply!
    const hemiLight = new THREE.HemisphereLight(0xf8fafc, 0x3a2a1d, 0.64);
    this.scene.add(hemiLight);

    const dirLight1 = new THREE.DirectionalLight(0xfff7ea, 1.82);
    dirLight1.position.set(140, -90, 125);
    dirLight1.castShadow = true;
    dirLight1.shadow.mapSize.width = 2048;
    dirLight1.shadow.mapSize.height = 2048;
    dirLight1.shadow.bias = -0.00025;
    dirLight1.shadow.normalBias = 0.035;
    this.scene.add(dirLight1);
    this.dirLight1 = dirLight1;

    const dirLight2 = new THREE.DirectionalLight(0xbfdbfe, 0.66);
    dirLight2.position.set(-110, 150, 115);
    this.scene.add(dirLight2);

    const rimLight = new THREE.DirectionalLight(0xffffff, 0.35);
    rimLight.position.set(50, 220, 75);
    this.scene.add(rimLight);

    // Dynamic Grid & Coordinate Axis (scaled cleanly in updateStageGuides)
    this.updateStageGuides();

    // Group for toolpaths
    this.toolpathGroup = new THREE.Group();
    this.scene.add(this.toolpathGroup);

    // 3D CNC Cutter Head Mesh for Interactive Toolpath Simulation
    const cutterGroup = new THREE.Group();
    const tipGeom = new THREE.ConeGeometry(1.6, 4.5, 16);
    tipGeom.rotateX(-Math.PI / 2);
    tipGeom.translate(0, 0, 2.25);
    const tipMat = new THREE.MeshStandardMaterial({ color: 0xf59e0b, metalness: 0.85, roughness: 0.2 });
    const shaftGeom = new THREE.CylinderGeometry(1.6, 1.6, 14, 16);
    shaftGeom.rotateX(Math.PI / 2);
    shaftGeom.translate(0, 0, 11.5);
    const shaftMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, metalness: 0.9, roughness: 0.15 });
    const colletGeom = new THREE.CylinderGeometry(3.2, 2.6, 7, 16);
    colletGeom.rotateX(Math.PI / 2);
    colletGeom.translate(0, 0, 21.5);
    const colletMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.7, roughness: 0.35 });
    cutterGroup.add(new THREE.Mesh(tipGeom, tipMat));
    cutterGroup.add(new THREE.Mesh(shaftGeom, shaftMat));
    cutterGroup.add(new THREE.Mesh(colletGeom, colletMat));
    cutterGroup.visible = false;
    this.cutterMesh = cutterGroup;
    this.scene.add(this.cutterMesh);

    // Brush cursor indicator ring
    const ringGeom = new THREE.RingGeometry(this.brushRadius - 0.4, this.brushRadius, 32);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, side: THREE.DoubleSide, transparent: true, opacity: 0.85 });
    this.brushIndicator = new THREE.Mesh(ringGeom, ringMat);
    this.brushIndicator.visible = false;
    this.scene.add(this.brushIndicator);

    this.initControls();
    this.animate();

    window.addEventListener('resize', () => this.onResize());
    if (typeof ResizeObserver !== 'undefined') {
      this._ro = new ResizeObserver(() => this.onResize());
      this._ro.observe(this.container);
    }
  }

  initControls() {
    const dom = this.renderer.domElement;

    dom.addEventListener('mousedown', (e) => {
      if (this.activeSculptMode && e.button === 0 && !e.shiftKey) {
        this.isSculpting = true;
        this.performSculpt(e);
        return;
      }

      if (e.button === 0) {
        this.isRotating = true;
      } else if (e.button === 2 || e.button === 1 || e.shiftKey) {
        this.isPanning = true;
      }
      this.prevMouse = { x: e.clientX, y: e.clientY };
    });

    dom.addEventListener('contextmenu', (e) => e.preventDefault());

    dom.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        e.preventDefault();
        const t = e.touches[0];
        if (this.activeSculptMode) {
          this.isSculpting = true;
          this.performSculpt(t);
        } else {
          this.isRotating = true;
          this.prevMouse = { x: t.clientX, y: t.clientY };
        }
      }
    }, { passive: false });

    dom.addEventListener('touchmove', (e) => {
      if (e.touches.length === 1) {
        e.preventDefault();
        const t = e.touches[0];
        const dx = t.clientX - this.prevMouse.x;
        const dy = t.clientY - this.prevMouse.y;
        this.prevMouse = { x: t.clientX, y: t.clientY };
        if (this.isSculpting) this.performSculpt(t);
        else if (this.isRotating) {
          this.spherical.theta -= dx * 0.008;
          this.spherical.phi = Math.max(0.1, Math.min(Math.PI / 2 - 0.05, this.spherical.phi - dy * 0.008));
          this.updateCameraPosition();
        }
      }
    }, { passive: false });

    dom.addEventListener('touchend', () => {
      this.isRotating = false;
      this.isSculpting = false;
    });

    window.addEventListener('mousemove', (e) => {
      const rect = dom.getBoundingClientRect();
      this.mouseCoord.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouseCoord.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      // Update brush indicator
      if (this.activeSculptMode && this.mesh) {
        this.raycaster.setFromCamera(this.mouseCoord, this.camera);
        const intersects = this.raycaster.intersectObject(this.mesh);
        if (intersects.length > 0) {
          const pt = intersects[0].point;
          this.brushIndicator.position.set(pt.x, pt.y, pt.z + 0.2);
          this.brushIndicator.visible = true;

          if (this.isSculpting) {
            this.relief.applySculptBrush(pt.x, this.relief.height - pt.y, this.activeSculptMode, this.brushRadius, this.brushStrength);
          }
        } else {
          this.brushIndicator.visible = false;
        }
      } else {
        this.brushIndicator.visible = false;
      }

      const dx = e.clientX - this.prevMouse.x;
      const dy = e.clientY - this.prevMouse.y;
      this.prevMouse = { x: e.clientX, y: e.clientY };

      if (this.isRotating) {
        this.spherical.theta -= dx * 0.008;
        this.spherical.phi = Math.max(0.1, Math.min(Math.PI / 2 - 0.05, this.spherical.phi - dy * 0.008));
        this.updateCameraPosition();
      } else if (this.isPanning) {
        const factor = this.spherical.radius * 0.0012;
        const forward = new THREE.Vector3().subVectors(this.target, this.camera.position).normalize();
        const right = new THREE.Vector3().crossVectors(forward, this.camera.up).normalize();
        const up = new THREE.Vector3().crossVectors(right, forward).normalize();

        this.target.addScaledVector(right, -dx * factor);
        this.target.addScaledVector(up, dy * factor);
        this.updateCameraPosition();
      }
    });

    window.addEventListener('mouseup', () => {
      this.isRotating = false;
      this.isPanning = false;
      this.isSculpting = false;
    });

    dom.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY > 0 ? 1.1 : 0.9;
      this.spherical.radius = Math.max(20, Math.min(800, this.spherical.radius * zoomFactor));
      this.updateCameraPosition();
    }, { passive: false });
  }

  performSculpt(e) {
    if (!this.mesh) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouseCoord.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouseCoord.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.mouseCoord, this.camera);
    const intersects = this.raycaster.intersectObject(this.mesh);
    if (intersects.length > 0) {
      const pt = intersects[0].point;
      this.relief.applySculptBrush(pt.x, this.relief.height - pt.y, this.activeSculptMode, this.brushRadius, this.brushStrength);
    }
  }

  setBrushRadius(radiusMm) {
    this.brushRadius = radiusMm;
    if (this.brushIndicator) {
      this.brushIndicator.geometry.dispose();
      this.brushIndicator.geometry = new THREE.RingGeometry(radiusMm - 0.4, radiusMm, 32);
    }
  }

  updateCameraPosition() {
    const r = this.spherical.radius;
    const phi = this.spherical.phi;
    const theta = this.spherical.theta;

    this.camera.position.x = this.target.x + r * Math.sin(phi) * Math.sin(theta);
    this.camera.position.y = this.target.y - r * Math.sin(phi) * Math.cos(theta);
    this.camera.position.z = this.target.z + r * Math.cos(phi);
    this.camera.up.set(0, 0, 1);
    this.camera.lookAt(this.target);
  }

  updateStageGuides() {
    const w = this.relief.width || 100;
    const h = this.relief.height || 100;
    const size = Math.ceil(Math.max(w, h, 100) * 1.8 / 20) * 20;
    const divs = Math.max(10, Math.min(40, Math.round(size / 10)));

    if (this.gridHelper) {
      this.scene.remove(this.gridHelper);
      this.gridHelper.geometry?.dispose();
    }
    this.gridHelper = new THREE.GridHelper(size, divs, 0x334155, 0x1e293b);
    this.gridHelper.rotation.x = Math.PI / 2;
    this.gridHelper.position.set(w / 2, h / 2, -0.15);
    this.scene.add(this.gridHelper);

    if (this.axesHelper) {
      this.scene.remove(this.axesHelper);
      this.axesHelper.geometry?.dispose();
    }
    this.axesHelper = new THREE.AxesHelper(Math.max(25, Math.min(w, h) * 0.35));
    this.axesHelper.position.set(0, 0, 0.05);
    this.scene.add(this.axesHelper);

    if (this.dirLight1) {
      const half = size * 0.55;
      this.dirLight1.position.set(w / 2 + size * 0.65, h / 2 - size * 0.45, size * 0.6);
      this.dirLight1.target.position.set(w / 2, h / 2, 0);
      this.dirLight1.target.updateMatrixWorld();
      this.dirLight1.shadow.camera.left = -half;
      this.dirLight1.shadow.camera.right = half;
      this.dirLight1.shadow.camera.top = half;
      this.dirLight1.shadow.camera.bottom = -half;
      this.dirLight1.shadow.camera.far = size * 2.5;
      this.dirLight1.shadow.camera.updateProjectionMatrix();
    }
  }

  resetCamera() {
    const w = this.relief.width || 100;
    const h = this.relief.height || 100;
    this.target.set(w / 2, h / 2, (this.relief.maxDepth || 8) * 0.25);
    this.spherical = { radius: Math.max(120, Math.max(w, h) * 1.62), theta: Math.PI / 5, phi: Math.PI / 4.0 };
    this.updateCameraPosition();
  }

  buildMesh() {
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.geometry.dispose();
    }
    if (this.baseSlab) {
      this.scene.remove(this.baseSlab);
      this.baseSlab.geometry.dispose();
      this.baseSlab.material.dispose();
      this.baseSlab = null;
    }

    const res = this.relief.resolution;
    const w = this.relief.width;
    const h = this.relief.height;

    this.updateStageGuides();

    // Solid bevelled CNC stock backing board underneath Z=0 so relief sits on realistic timber/metal block
    const slabThickness = Math.max(4, Math.min(14, (this.relief.maxDepth || 8) * 0.85));
    const slabGeom = new THREE.BoxGeometry(w, h, slabThickness);
    slabGeom.translate(w / 2, h / 2, -slabThickness / 2 - 0.02);
    const slabMat = new THREE.MeshStandardMaterial({
      map: this.getProceduralTexture(this.currentMaterialType),
      color: this.getBaseTint(this.currentMaterialType),
      roughness: 0.65,
      metalness: this.currentMaterialType === 'brass' ? 0.7 : 0.06,
    });
    this.baseSlab = new THREE.Mesh(slabGeom, slabMat);
    this.baseSlab.receiveShadow = true;
    this.scene.add(this.baseSlab);

    this.geometry = new THREE.PlaneGeometry(w, h, res - 1, res - 1);
    this.geometry.translate(w / 2, h / 2, 0);

    const material = this.createMaterial(this.currentMaterialType);
    this.mesh = new THREE.Mesh(this.geometry, material);
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    this.scene.add(this.mesh);

    this.target.set(w / 2, h / 2, (this.relief.maxDepth || 8) * 0.25);
    this.updateMeshHeights();
  }

  updateMeshHeights() {
    if (!this.geometry) return;

    const pos = this.geometry.attributes.position;
    const res = this.relief.resolution;
    const data = this.relief.data;
    const maxZ = this.relief.maxDepth;
    const baseZ = this.relief.baseHeight;

    const colors = new Float32Array(res * res * 3);
    const tmpColor = new THREE.Color();
    const isHeatmap = this.currentMaterialType === 'heatmap';

    let pIdx = 0;
    for (let iy = 0; iy < res; iy++) {
      for (let ix = 0; ix < res; ix++) {
        const idx = iy * res + ix;
        const normZ = data[idx];
        const z = baseZ + normZ * maxZ;
        pos.setZ(pIdx, z);

        if (isHeatmap) {
          tmpColor.setHSL((1.0 - normZ) * 0.65, 0.9, 0.5);
          colors[pIdx * 3] = tmpColor.r;
          colors[pIdx * 3 + 1] = tmpColor.g;
          colors[pIdx * 3 + 2] = tmpColor.b;
        } else {
          // Soft cavity AO + gentle crest lift so sculpted 3D carvings stand out clearly without crushed blacks!
          let avgNbr = normZ;
          if (ix > 0 && ix < res - 1 && iy > 0 && iy < res - 1) {
            avgNbr = (data[idx - 1] + data[idx + 1] + data[idx - res] + data[idx + res]) * 0.25;
          }
          const cavity = Math.max(0, avgNbr - normZ);
          const shade = Math.max(0.84, Math.min(1.10, 0.94 + normZ * 0.14 - cavity * 0.85));
          colors[pIdx * 3] = shade;
          colors[pIdx * 3 + 1] = shade;
          colors[pIdx * 3 + 2] = shade;
        }

        pIdx++;
      }
    }

    pos.needsUpdate = true;
    this.geometry.computeVertexNormals();
    this.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  }

  getBaseTint(type) {
    if (type === 'wood') return 0xd6a472;
    if (type === 'oak') return 0xe8c496;
    if (type === 'jade') return 0x269b61;
    if (type === 'brass') return 0xdfb945;
    if (type === 'stone') return 0x64748b;
    return 0x64748b;
  }

  getProceduralTexture(type) {
    if (type === 'heatmap') return null;
    if (this._textures[type]) return this._textures[type];

    const size = 256;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(size, size);
    const d = imgData.data;

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const i = (y * size + x) * 4;
        const u = x / size;
        const v = y / size;
        let r = 210, g = 210, b = 210;

        if (type === 'wood' || type === 'oak') {
          // Subtle satin grain so real 3D carved geometry & shadows always read clearly without zebra striping!
          const grainFreq = type === 'wood' ? 12 : 10;
          const wave = Math.sin(v * Math.PI * 2 * grainFreq + Math.sin(u * Math.PI * 4) * 1.1) * 0.45;
          const fine = Math.sin(v * Math.PI * 24 + u * 5) * 0.08;
          const t = 0.5 + 0.25 * wave + fine;
          if (type === 'wood') {
            r = Math.round(176 + t * 34);
            g = Math.round(122 + t * 26);
            b = Math.round(78 + t * 18);
          } else {
            r = Math.round(202 + t * 28);
            g = Math.round(168 + t * 24);
            b = Math.round(122 + t * 18);
          }
        } else if (type === 'jade') {
          const swirl = Math.sin(u * 9 + Math.cos(v * 7) * 2) * Math.cos(v * 6 - u * 4);
          const t = 0.5 + 0.35 * swirl;
          r = Math.round(28 + t * 48);
          g = Math.round(122 + t * 72);
          b = Math.round(78 + t * 46);
        } else if (type === 'brass') {
          const brush = Math.sin(v * 180 + Math.sin(u * 6) * 0.5) * 0.5 + 0.5;
          r = Math.round(205 + brush * 42);
          g = Math.round(172 + brush * 36);
          b = Math.round(68 + brush * 22);
        } else if (type === 'stone') {
          const hash = Math.abs(Math.sin(x * 12.9898 + y * 78.233) * 43758.5453) % 1;
          const patch = Math.sin(u * 12) * Math.cos(v * 12) * 18;
          const base = 68 + hash * 44 + patch;
          r = g = b = Math.max(25, Math.min(180, Math.round(base)));
        }

        d[i] = r; d[i + 1] = g; d[i + 2] = b; d[i + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.colorSpace = THREE.SRGBColorSpace;
    this._textures[type] = tex;
    return tex;
  }

  setMaterial(type) {
    this.currentMaterialType = type;
    if (this.mesh) {
      this.mesh.material.dispose();
      this.mesh.material = this.createMaterial(type);
      this.mesh.material.wireframe = this.wireframe;
      if (this.baseSlab) {
        this.baseSlab.material.map = this.getProceduralTexture(type);
        this.baseSlab.material.color.setHex(this.getBaseTint(type));
        this.baseSlab.material.needsUpdate = true;
      }
      this.updateMeshHeights();
    }
  }

  createMaterial(type) {
    const tex = this.getProceduralTexture(type);
    if (type === 'jade') {
      return new THREE.MeshPhysicalMaterial({
        map: tex,
        vertexColors: true,
        color: 0xdfffe9,
        roughness: 0.22,
        metalness: 0.04,
        transmission: 0.12,
        reflectivity: 0.85,
        clearcoat: 0.65,
        clearcoatRoughness: 0.18,
      });
    } else if (type === 'wood' || type === 'oak') {
      return new THREE.MeshStandardMaterial({
        map: tex,
        vertexColors: true,
        color: 0xffffff,
        roughness: 0.48,
        metalness: 0.04,
      });
    } else if (type === 'brass') {
      return new THREE.MeshStandardMaterial({
        map: tex,
        vertexColors: true,
        color: 0xfff3c4,
        roughness: 0.26,
        metalness: 0.82,
      });
    } else if (type === 'stone') {
      return new THREE.MeshStandardMaterial({
        map: tex,
        vertexColors: true,
        color: 0xdbeafe,
        roughness: 0.78,
        metalness: 0.08,
      });
    } else if (type === 'heatmap') {
      return new THREE.MeshBasicMaterial({
        vertexColors: true,
      });
    }
    return new THREE.MeshStandardMaterial({ vertexColors: true, color: 0xe2e8f0 });
  }

  setWireframe(enabled) {
    this.wireframe = enabled;
    if (this.mesh && this.mesh.material) {
      this.mesh.material.wireframe = enabled;
    }
    if (this.assetMesh && this.assetMesh.material) {
      this.assetMesh.material.wireframe = enabled;
    }
  }

  _buildCanvasFromRgba(rgba, size) {
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(size, size);
    img.data.set(rgba);
    ctx.putImageData(img, 0, 0);
    return canvas;
  }

  _buildCanvasTextureFromRgba(rgba, size, isSRGB = false) {
    const canvas = this._buildCanvasFromRgba(rgba, size);
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = isSRGB ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.needsUpdate = true;
    tex._sourceCanvas = canvas;
    return tex;
  }

  _disposeAssetTextures() {
    for (const k of Object.keys(this._assetTextures)) {
      if (this._assetTextures[k] && typeof this._assetTextures[k].dispose === 'function') {
        this._assetTextures[k].dispose();
      }
    }
    this._assetTextures = {};
  }

  setAssetMesh(meshData, pbrMaps = null) {
    if (!meshData || !meshData.positions) return;
    this.assetData = meshData;
    this.assetMaps = pbrMaps;
    this._disposeAssetTextures();

    if (this.assetMesh) {
      this.scene.remove(this.assetMesh);
      this.assetMesh.geometry.dispose();
      this.assetMesh.material.dispose();
      this.assetMesh = null;
    }

    if (pbrMaps && pbrMaps.size) {
      this._assetTextures.baseColor = this._buildCanvasTextureFromRgba(pbrMaps.baseColor, pbrMaps.size, true);
      this._assetTextures.normal = this._buildCanvasTextureFromRgba(pbrMaps.normal, pbrMaps.size, false);
      this._assetTextures.roughness = this._buildCanvasTextureFromRgba(pbrMaps.roughness, pbrMaps.size, false);
      this._assetTextures.metallic = this._buildCanvasTextureFromRgba(pbrMaps.metallic, pbrMaps.size, false);
      if (pbrMaps.metallicRoughness) {
        this._assetTextures.metallicRoughnessCanvas = this._buildCanvasFromRgba(pbrMaps.metallicRoughness, pbrMaps.size);
      }
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(meshData.positions, 3));
    geom.setAttribute('normal', new THREE.BufferAttribute(meshData.normals, 3));
    geom.setAttribute('uv', new THREE.BufferAttribute(meshData.uvs, 2));
    geom.setIndex(new THREE.BufferAttribute(meshData.indices, 1));
    geom.computeBoundingBox();

    const mat = this._createAssetMaterial();
    this.assetMesh = new THREE.Mesh(geom, mat);
    this.assetMesh.castShadow = true;
    this.assetMesh.receiveShadow = true;

    // Lift asset cleanly above grid floor (Z >= 0) so both Front & Back rotate without slicing grid
    const minZ = geom.boundingBox ? geom.boundingBox.min.z : 0;
    if (minZ < 0) this.assetMesh.position.z = -minZ + 0.5;
    this.scene.add(this.assetMesh);

    if (this.mesh) this.mesh.visible = false;
    if (this.baseSlab) this.baseSlab.visible = false;
    this._emitStats();
  }

  clearAssetMesh() {
    if (this.assetMesh) {
      this.scene.remove(this.assetMesh);
      this.assetMesh.geometry.dispose();
      this.assetMesh.material.dispose();
      this.assetMesh = null;
    }
    this.assetData = null;
    this.assetMaps = null;
    this._disposeAssetTextures();
    if (this.mesh) this.mesh.visible = true;
    if (this.baseSlab) this.baseSlab.visible = true;
    this._emitStats();
  }

  _createAssetMaterial() {
    const mode = this.shadingMode || 'standard';
    const chan = this.pbrChannel || 'pbr';
    const adj = this.materialAdjust || { roughness: 0.55, metalness: 0.08, contrast: 1.0, intensity: 1.0 };

    if (chan !== 'pbr' && this._assetTextures[chan]) {
      return new THREE.MeshBasicMaterial({
        map: this._assetTextures[chan],
        wireframe: this.wireframe,
        side: THREE.DoubleSide,
      });
    }

    if (mode === 'solid') {
      return new THREE.MeshStandardMaterial({
        color: 0xd1d5db,
        roughness: Math.max(0.15, Math.min(0.95, adj.roughness)),
        metalness: 0.04,
        wireframe: this.wireframe,
        side: THREE.DoubleSide,
      });
    }

    if (mode === 'unlit') {
      return new THREE.MeshBasicMaterial({
        map: this._assetTextures.baseColor || null,
        color: this._assetTextures.baseColor ? 0xffffff : 0xd6a472,
        wireframe: this.wireframe,
        side: THREE.DoubleSide,
      });
    }

    const boost = Math.max(0.4, Math.min(1.6, adj.intensity || 1.0));
    const mat = new THREE.MeshStandardMaterial({
      map: this._assetTextures.baseColor || null,
      normalMap: this._assetTextures.normal || null,
      normalScale: new THREE.Vector2(0.85 * (adj.contrast || 1.0), 0.85 * (adj.contrast || 1.0)),
      roughnessMap: mode === 'lit' ? this._assetTextures.roughness || null : null,
      metalnessMap: mode === 'lit' ? this._assetTextures.metallic || null : null,
      color: new THREE.Color(boost, boost, boost),
      roughness: Math.max(0.08, Math.min(1.0, adj.roughness)),
      metalness: Math.max(0.0, Math.min(1.0, adj.metalness)),
      wireframe: this.wireframe,
      side: THREE.DoubleSide,
    });
    return mat;
  }

  _refreshAssetMaterial() {
    if (!this.assetMesh) return;
    this.assetMesh.material.dispose();
    this.assetMesh.material = this._createAssetMaterial();
  }

  setShadingMode(mode) {
    this.shadingMode = mode || 'standard';
    if (this.assetMesh) {
      this._refreshAssetMaterial();
    } else if (this.mesh) {
      if (mode === 'solid') {
        this.mesh.material.dispose();
        this.mesh.material = new THREE.MeshStandardMaterial({
          color: 0xd1d5db,
          roughness: this.materialAdjust.roughness,
          metalness: 0.04,
          wireframe: this.wireframe,
        });
      } else if (mode === 'unlit') {
        this.mesh.material.dispose();
        this.mesh.material = new THREE.MeshBasicMaterial({
          map: this.getProceduralTexture(this.currentMaterialType),
          vertexColors: true,
          wireframe: this.wireframe,
        });
      } else {
        this.setMaterial(this.currentMaterialType);
      }
    }
  }

  setPbrChannel(channel) {
    this.pbrChannel = channel || 'pbr';
    this._refreshAssetMaterial();
  }

  setMaterialAdjustment(partial) {
    this.materialAdjust = { ...this.materialAdjust, ...partial };
    this._refreshAssetMaterial();
  }

  setTurntable(enabled) {
    this.turntable = !!enabled;
  }

  getBaseColorPngCanvas() {
    return this._assetTextures.baseColor?._sourceCanvas || null;
  }

  /**
   * Prepare export-ready PNG canvases for .GLB export (applying user intensity, roughness & metalness adjustments).
   */
  getPbrExportCanvases(adj = null) {
    if (!this.assetMesh || !this.assetData) {
      const procTex = this.getProceduralTexture(this.currentMaterialType);
      return { baseColor: procTex?.image || null, normal: null, metallicRoughness: null };
    }

    const boost = Math.max(0.4, Math.min(1.6, adj?.intensity ?? this.materialAdjust?.intensity ?? 1.0));
    const targetRough = Math.max(0.08, Math.min(1.0, adj?.roughness ?? this.materialAdjust?.roughness ?? 0.55));
    const targetMetal = Math.max(0.0, Math.min(1.0, adj?.metalness ?? this.materialAdjust?.metalness ?? 0.08));

    let baseCanvas = this._assetTextures.baseColor?._sourceCanvas || null;
    if (baseCanvas && Math.abs(boost - 1.0) > 0.02) {
      const w = baseCanvas.width, h = baseCanvas.height;
      const out = document.createElement('canvas');
      out.width = w; out.height = h;
      const octx = out.getContext('2d');
      octx.drawImage(baseCanvas, 0, 0);
      const img = octx.getImageData(0, 0, w, h);
      const d = img.data;
      for (let i = 0; i < d.length; i += 4) {
        d[i] = Math.min(255, Math.round(d[i] * boost));
        d[i + 1] = Math.min(255, Math.round(d[i + 1] * boost));
        d[i + 2] = Math.min(255, Math.round(d[i + 2] * boost));
      }
      octx.putImageData(img, 0, 0);
      baseCanvas = out;
    }

    const normalCanvas = this._assetTextures.normal?._sourceCanvas || null;
    let ormCanvas = this._assetTextures.metallicRoughnessCanvas || null;
    if (ormCanvas && (Math.abs(targetRough - 0.55) > 0.02 || Math.abs(targetMetal - 0.08) > 0.02)) {
      const w = ormCanvas.width, h = ormCanvas.height;
      const out = document.createElement('canvas');
      out.width = w; out.height = h;
      const octx = out.getContext('2d');
      octx.drawImage(ormCanvas, 0, 0);
      const img = octx.getImageData(0, 0, w, h);
      const d = img.data;
      const rScale = targetRough / 0.55;
      for (let i = 0; i < d.length; i += 4) {
        const gRaw = d[i + 1] / 255.0;
        const bRaw = d[i + 2] / 255.0;
        d[i + 1] = Math.max(12, Math.min(255, Math.round(gRaw * rScale * 255)));
        const mBlended = Math.max(0.0, Math.min(1.0, targetMetal * 0.72 + bRaw * targetMetal * 0.45 + (targetMetal > 0.25 ? bRaw * 0.22 : 0)));
        d[i + 2] = Math.round(mBlended * 255);
      }
      octx.putImageData(img, 0, 0);
      ormCanvas = out;
    }

    return { baseColor: baseCanvas, normal: normalCanvas, metallicRoughness: ormCanvas };
  }

  _emitStats() {
    if (typeof this.onStatsChange !== 'function') return;
    if (this.assetMesh && this.assetData?.stats) {
      this.onStatsChange({ mode: 'asset', ...this.assetData.stats });
      return;
    }
    const res = this.relief.resolution || 256;
    const faces = (res - 1) * (res - 1) * 2;
    const vertices = res * res;
    this.onStatsChange({ mode: 'relief', topology: 'Triangle', faces, vertices });
  }

  renderToolpaths(toolpathSegments) {
    this.stopSimulation();
    this._simSegments = toolpathSegments || [];

    while (this.toolpathGroup.children.length > 0) {
      const obj = this.toolpathGroup.children[0];
      this.toolpathGroup.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    }

    if (!toolpathSegments || toolpathSegments.length === 0) return;

    const rapidPoints = [];
    const cutPoints = [];
    const plungePoints = [];

    // CamEngine coordinates already use upright CNC Y+ (0 bottom .. h top), matching 3D geometry 1:1!
    for (const seg of toolpathSegments) {
      if (seg.type === 'rapid') {
        rapidPoints.push(seg.from.x, seg.from.y, seg.from.z, seg.to.x, seg.to.y, seg.to.z);
      } else if (seg.type === 'plunge') {
        plungePoints.push(seg.from.x, seg.from.y, seg.from.z, seg.to.x, seg.to.y, seg.to.z);
      } else {
        cutPoints.push(seg.from.x, seg.from.y, seg.from.z, seg.to.x, seg.to.y, seg.to.z);
      }
    }

    if (rapidPoints.length > 0) {
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.Float32BufferAttribute(rapidPoints, 3));
      const mat = new THREE.LineBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.6 });
      this.toolpathGroup.add(new THREE.LineSegments(geom, mat));
    }

    if (plungePoints.length > 0) {
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.Float32BufferAttribute(plungePoints, 3));
      const mat = new THREE.LineBasicMaterial({ color: 0xfbbf24 });
      this.toolpathGroup.add(new THREE.LineSegments(geom, mat));
    }

    if (cutPoints.length > 0) {
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.Float32BufferAttribute(cutPoints, 3));
      const mat = new THREE.LineBasicMaterial({ color: 0x38bdf8 });
      this.toolpathGroup.add(new THREE.LineSegments(geom, mat));
    }
  }

  startSimulation(onProgress, onDone) {
    this.stopSimulation();
    if (!this._simSegments || this._simSegments.length === 0) return false;
    const total = this._simSegments.length;
    let idx = 0;
    const stepPerFrame = Math.max(1, Math.ceil(total / 360)); // ~6s smooth preview animation
    this.cutterMesh.visible = true;

    const tick = () => {
      if (!this.cutterMesh.visible) return;
      idx = Math.min(total - 1, idx + stepPerFrame);
      const seg = this._simSegments[idx];
      this.cutterMesh.position.set(seg.to.x, seg.to.y, seg.to.z);
      if (onProgress) onProgress(Math.round(((idx + 1) / total) * 100), seg.to);
      if (idx >= total - 1) {
        this._simTimer = null;
        if (onDone) onDone();
        return;
      }
      this._simTimer = requestAnimationFrame(tick);
    };
    this._simTimer = requestAnimationFrame(tick);
    return true;
  }

  stopSimulation() {
    if (this._simTimer) {
      cancelAnimationFrame(this._simTimer);
      this._simTimer = null;
    }
    if (this.cutterMesh) this.cutterMesh.visible = false;
  }

  setToolpathVisibility(visible) {
    this.showToolpaths = visible;
    this.toolpathGroup.visible = visible;
  }

  onResize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (width === 0 || height === 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    if (this.turntable && !this.isRotating) {
      this.spherical.theta += 0.008;
      this.updateCameraPosition();
    }
    this.renderer.render(this.scene, this.camera);
  }
}
