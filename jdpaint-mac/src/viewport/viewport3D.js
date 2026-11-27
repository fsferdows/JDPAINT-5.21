// 3D WebGL Viewport using Three.js for JDPaint Mac
import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

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
    this._externalGlbBytes = null;
    this._assetTextures = {};
    this.gridHelper = null;
    this.axesHelper = null;
    this.geometry = null;
    this.toolpathGroup = null;
    this.brushIndicator = null;
    this.cutterMesh = null;
    this.dirLight1 = null;
    this._textures = {};
    this._designPhotoTex = null;
    this._solidMatCapTex = null;
    this._simTimer = null;
    this._simSegments = [];

    this.currentMaterialType = 'wood';
    this.shadingMode = 'standard'; // 'standard' | 'solid' | 'unlit' | 'lit'
    this.pbrChannel = 'pbr'; // 'pbr' | 'baseColor' | 'roughness' | 'metallic' | 'normal'
    this.materialAdjust = { roughness: 0.55, metalness: 0.08, contrast: 1.0, intensity: 1.0 };
    this.wireframe = false;
    this.showToolpaths = true;
    this.turntable = false;
    this._dirty = true;
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

    // Meshy-grade Studio IBL reflections via PMREM RoomEnvironment so metallic & glossy PBR materials catch real highlights!
    try {
      const pmrem = new THREE.PMREMGenerator(this.renderer);
      this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
      this.scene.environmentIntensity = 0.85;
      pmrem.dispose();
    } catch (_err) {
      // Headless fallback safely leaves scene.environment unset
    }

    // 3-Point Studio Raking Lighting with 2048px soft shadows balanced for clay sculpting & PBR metals
    const hemiLight = new THREE.HemisphereLight(0xf8fafc, 0x3b2f26, 0.72);
    this.scene.add(hemiLight);

    const dirLight1 = new THREE.DirectionalLight(0xfff8ed, 1.48);
    dirLight1.position.set(140, -90, 125);
    dirLight1.castShadow = true;
    dirLight1.shadow.mapSize.width = 2048;
    dirLight1.shadow.mapSize.height = 2048;
    dirLight1.shadow.bias = -0.00025;
    dirLight1.shadow.normalBias = 0.035;
    this.scene.add(dirLight1);
    this.dirLight1 = dirLight1;

    const dirLight2 = new THREE.DirectionalLight(0xc7dcf7, 0.74);
    dirLight2.position.set(-110, 150, 115);
    this.scene.add(dirLight2);

    const rimLight = new THREE.DirectionalLight(0xffffff, 0.32);
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

  requestRender() {
    this._dirty = true;
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
    this._dirty = true;
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

  fitCameraToBox(box3, padding = 1.24) {
    if (!box3 || box3.isEmpty()) return;
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box3.getSize(size);
    box3.getCenter(center);
    this.target.copy(center);

    const vFov = ((this.camera?.fov || 44) * Math.PI) / 180;
    const aspect = Math.max(0.35, this.camera?.aspect || 1);
    const hFov = 2 * Math.atan(Math.tan(vFov * 0.5) * aspect);
    const radius = Math.max(12, 0.5 * Math.hypot(size.x, size.y, size.z));
    const minSin = Math.sin(Math.min(vFov, hFov) * 0.5);
    this.spherical.radius = Math.max(55, (radius / Math.max(0.14, minSin)) * padding);
    this.updateCameraPosition();
  }

  fitCameraToActive() {
    if (this.assetMesh) {
      const box = new THREE.Box3().setFromObject(this.assetMesh);
      this.fitCameraToBox(box, 1.26);
      return;
    }
    const w = this.relief.width || 100;
    const h = this.relief.height || 100;
    const d = this.relief.maxDepth || 8;
    const box = new THREE.Box3(new THREE.Vector3(0, 0, -4), new THREE.Vector3(w, h, d));
    this.fitCameraToBox(box, 1.18);
  }

  resetCamera() {
    this.spherical.theta = Math.PI / 5;
    this.spherical.phi = Math.PI / 4.0;
    this.fitCameraToActive();
  }

  buildMesh() {
    if (this.mesh) {
      this.scene.remove(this.mesh);
      this.mesh.material?.dispose();
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
    slabGeom.translate(w / 2, h / 2, -slabThickness / 2 - 0.18);
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
    const expectedVerts = this.relief.resolution * this.relief.resolution;
    if (this.geometry.attributes.position.count !== expectedVerts) {
      this.buildMesh();
      return;
    }

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
          const shade = Math.max(0.70, Math.min(1.14, 0.90 + normZ * 0.20 - cavity * 1.65));
          colors[pIdx * 3] = shade;
          colors[pIdx * 3 + 1] = shade;
          colors[pIdx * 3 + 2] = shade;
        }

        pIdx++;
      }
    }

    pos.needsUpdate = true;

    // Adaptive height-aligned quad diagonal split eliminates 45-degree sawtooth triangles in Solid zoom-in view
    const indexAttr = this.geometry.getIndex();
    if (indexAttr && res >= 4) {
      const arr = indexAttr.array;
      const quads = res - 1;
      let k = 0;
      for (let iy = 0; iy < quads; iy++) {
        const row0 = iy * res;
        const row1 = row0 + res;
        for (let ix = 0; ix < quads; ix++) {
          const a = row0 + ix;
          const b = a + 1;
          const c = row1 + ix;
          const d = c + 1;
          const za = data[a], zb = data[b], zc = data[c], zd = data[d];
          if (Math.abs(za - zd) <= Math.abs(zb - zc)) {
            arr[k++] = a; arr[k++] = c; arr[k++] = d;
            arr[k++] = a; arr[k++] = d; arr[k++] = b;
          } else {
            arr[k++] = a; arr[k++] = c; arr[k++] = b;
            arr[k++] = c; arr[k++] = d; arr[k++] = b;
          }
        }
      }
      indexAttr.needsUpdate = true;
    }

    this.geometry.computeVertexNormals();
    this.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    this._dirty = true;
    this._emitStats();
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

  /**
   * Align loaded reference picture onto the relief mesh UVs for true Standard photo-mapped PBR inspection.
   */
  setReferenceDesignTexture(imgElement, fitBounds = null) {
    if (this._designPhotoTex) {
      this._designPhotoTex.dispose();
      this._designPhotoTex = null;
    }
    if (imgElement && typeof document !== 'undefined') {
      const S = 1024;
      const canvas = document.createElement('canvas');
      canvas.width = S;
      canvas.height = S;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#8b6a4b';
      ctx.fillRect(0, 0, S, S);
      const ww = this.relief.width || 100;
      const wh = this.relief.height || 100;
      const b = fitBounds || { x: 0, y: 0, w: ww, h: wh };
      const dx = (b.x / ww) * S;
      const dy = (b.y / wh) * S;
      const dw = (b.w / ww) * S;
      const dh = (b.h / wh) * S;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(imgElement, dx, dy, dw, dh);
      const tex = new THREE.CanvasTexture(canvas);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.wrapS = THREE.ClampToEdgeWrapping;
      tex.wrapT = THREE.ClampToEdgeWrapping;
      tex.needsUpdate = true;
      this._designPhotoTex = tex;
    }
    if (this.mesh && !this.assetMesh) {
      this.setShadingMode(this.shadingMode);
    }
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
    const useDesignPhoto = type === 'wood' && this._designPhotoTex && (this.shadingMode === 'standard' || this.shadingMode === 'lit');
    const tex = useDesignPhoto ? this._designPhotoTex : this.getProceduralTexture(type);
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
        color: useDesignPhoto ? 0xfaf5f0 : 0xffffff,
        roughness: useDesignPhoto ? 0.52 : 0.48,
        metalness: 0.02,
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
    if (this.assetMesh) {
      this.assetMesh.traverse((child) => {
        if (child.isMesh && child.material) {
          const mats = Array.isArray(child.material) ? child.material : [child.material];
          mats.forEach((m) => { if (m) m.wireframe = enabled; });
        }
      });
    }
    this._dirty = true;
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
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = true;
    const maxAniso = this.renderer?.capabilities?.getMaxAnisotropy ? this.renderer.capabilities.getMaxAnisotropy() : 4;
    tex.anisotropy = Math.min(8, Math.max(1, maxAniso));
    tex.needsUpdate = true;
    tex._sourceCanvas = canvas;
    return tex;
  }

  /**
   * Procedural 512x512 Meshy-style Studio Clay MatCap (calibrated after Meshy Final Grey shader).
   * Reveals fine sculpting folds, cavity edges & crisp planar CAD faces cleanly under Solid Mode.
   */
  _getMeshyStudioMatCap() {
    if (this._solidMatCapTex) return this._solidMatCapTex;
    const S = 512;
    const canvas = document.createElement('canvas');
    canvas.width = S;
    canvas.height = S;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(S, S);
    const d = img.data;
    const kLen = Math.hypot(0.24, 0.68, 0.69);
    const lx = 0.24 / kLen, ly = 0.68 / kLen, lz = 0.69 / kLen;
    const hx = lx, hy = ly, hz = lz + 1.0;
    const hLen = Math.hypot(hx, hy, hz);
    const nhx = hx / hLen, nhy = hy / hLen, nhz = hz / hLen;
    const fLen = Math.hypot(-0.58, -0.34, 0.74);
    const fx = -0.58 / fLen, fy = -0.34 / fLen, fz = 0.74 / fLen;

    for (let py = 0; py < S; py++) {
      const ny = 1.0 - (2.0 * (py + 0.5)) / S;
      for (let px = 0; px < S; px++) {
        const nx = (2.0 * (px + 0.5)) / S - 1.0;
        const r2 = nx * nx + ny * ny;
        const nz = r2 < 1.0 ? Math.sqrt(1.0 - r2) : 0.0;
        let nnx = nx, nny = ny, nnz = nz;
        if (r2 >= 1.0) {
          const inv = 1.0 / Math.sqrt(r2);
          nnx *= inv; nny *= inv; nnz = 0.0;
        }
        const key = Math.max(0, nnx * lx + nny * ly + nnz * lz);
        const fill = Math.max(0, nnx * fx + nny * fy + nnz * fz);
        const hemi = 0.5 + 0.5 * nny;
        const fresnel = Math.pow(Math.max(0, 1.0 - nnz), 2.35);
        const spec = Math.pow(Math.max(0, nnx * nhx + nny * nhy + nnz * nhz), 28) * 0.16;
        const lum = Math.max(0.10, Math.min(0.92, 0.16 + hemi * 0.14 + key * 0.48 + fill * 0.10 + fresnel * 0.12 + spec));
        const rC = Math.round(Math.min(255, lum * 246 + fresnel * 8));
        const gC = Math.round(Math.min(255, lum * 248 + fresnel * 10));
        const bC = Math.round(Math.min(255, lum * 252 + fresnel * 16));
        const idx = (py * S + px) * 4;
        d[idx] = rC; d[idx + 1] = gC; d[idx + 2] = bC; d[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    this._solidMatCapTex = tex;
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

  _disposeAssetObject() {
    if (!this.assetMesh) return;
    this.scene.remove(this.assetMesh);
    this.assetMesh.traverse((child) => {
      if (child.isMesh) {
        child.geometry?.dispose();
        const mats = Array.isArray(child.material) ? child.material : [child.material];
        mats.forEach((m) => m?.dispose());
        if (child.userData?._origMat && child.userData._origMat !== child.material) {
          child.userData._origMat.dispose?.();
        }
        if (child.userData?._solidMat && child.userData._solidMat !== child.material) {
          child.userData._solidMat.dispose?.();
        }
      }
    });
    this.assetMesh = null;
    this._externalGlbBytes = null;
  }

  setAssetMesh(meshData, pbrMaps = null) {
    if (!meshData || !meshData.positions) return;
    this._disposeAssetObject();
    this.assetData = meshData;
    this.assetMaps = pbrMaps;
    this._disposeAssetTextures();

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

    // Lift asset cleanly above grid floor (Z >= 0) and auto-frame camera around asset center like Meshy studio
    const bb = geom.boundingBox;
    const minZ = bb ? bb.min.z : 0;
    if (minZ < 0) this.assetMesh.position.z = -minZ + 0.5;
    this.scene.add(this.assetMesh);

    if (bb) {
      const worldBox = bb.clone().translate(this.assetMesh.position);
      this.fitCameraToBox(worldBox, 1.26);
    }

    if (this.mesh) this.mesh.visible = false;
    if (this.baseSlab) this.baseSlab.visible = false;
    this._emitStats();
  }

  /**
   * Load an external Binary glTF 2.0 (.glb) buffer (e.g. from Meshy Cloud API v1) into the stage.
   */
  loadExternalGLBBuffer(arrayBuffer) {
    if (!arrayBuffer) return Promise.reject(new Error('Missing GLB buffer'));
    const loader = new GLTFLoader();
    return new Promise((resolve, reject) => {
      loader.parse(
        arrayBuffer,
        '',
        (gltf) => {
          this._disposeAssetObject();
          this._disposeAssetTextures();
          this._externalGlbBytes = new Uint8Array(arrayBuffer);
          this.assetMaps = null;

          const root = gltf.scene || new THREE.Group();
          root.rotation.x = Math.PI / 2; // Convert glTF Y-up convention to JDPaint Z-up stage
          root.updateMatrixWorld(true);

          const rawBox = new THREE.Box3().setFromObject(root);
          const rawSize = new THREE.Vector3();
          rawBox.getSize(rawSize);
          const maxSpan = Math.max(1e-4, rawSize.x, rawSize.y, rawSize.z);
          const normScale = 90 / maxSpan;
          root.scale.multiplyScalar(normScale);
          root.updateMatrixWorld(true);

          const scaledBox = new THREE.Box3().setFromObject(root);
          const scaledCenter = new THREE.Vector3();
          scaledBox.getCenter(scaledCenter);
          const stageW = this.relief.width || 100;
          const stageH = this.relief.height || 100;
          root.position.x += stageW * 0.5 - scaledCenter.x;
          root.position.y += stageH * 0.5 - scaledCenter.y;
          root.position.z += 0.5 - scaledBox.min.z;
          root.updateMatrixWorld(true);

          // Collect merged geometry buffers so .STL / .OBJ export also works on loaded neural GLBs
          const posChunks = [];
          const normChunks = [];
          const uvChunks = [];
          const idxChunks = [];
          let vOffset = 0;
          const normalMat = new THREE.Matrix3();
          const tmpPos = new THREE.Vector3();
          const tmpNorm = new THREE.Vector3();

          root.traverse((child) => {
            if (!child.isMesh || !child.geometry) return;
            child.castShadow = true;
            child.receiveShadow = true;
            child.userData._origMat = child.material;
            if (child.material && !Array.isArray(child.material)) {
              child.material.wireframe = this.wireframe;
            }
            const posAttr = child.geometry.getAttribute('position');
            if (!posAttr) return;
            if (!child.geometry.getAttribute('normal')) {
              child.geometry.computeVertexNormals();
            }
            const normAttr = child.geometry.getAttribute('normal');
            const uvAttr = child.geometry.getAttribute('uv');
            const idxAttr = child.geometry.getIndex();
            normalMat.getNormalMatrix(child.matrixWorld);

            const nVert = posAttr.count;
            const pArr = new Float32Array(nVert * 3);
            const nArr = new Float32Array(nVert * 3);
            const uArr = new Float32Array(nVert * 2);
            for (let i = 0; i < nVert; i++) {
              tmpPos.fromBufferAttribute(posAttr, i).applyMatrix4(child.matrixWorld);
              pArr[i * 3] = tmpPos.x;
              pArr[i * 3 + 1] = tmpPos.y;
              pArr[i * 3 + 2] = tmpPos.z;
              if (normAttr) {
                tmpNorm.fromBufferAttribute(normAttr, i).applyMatrix3(normalMat).normalize();
              } else {
                tmpNorm.set(0, 0, 1);
              }
              nArr[i * 3] = tmpNorm.x;
              nArr[i * 3 + 1] = tmpNorm.y;
              nArr[i * 3 + 2] = tmpNorm.z;
              if (uvAttr) {
                uArr[i * 2] = uvAttr.getX(i);
                uArr[i * 2 + 1] = uvAttr.getY(i);
              }
            }
            posChunks.push(pArr);
            normChunks.push(nArr);
            uvChunks.push(uArr);

            if (idxAttr) {
              const iArr = new Uint32Array(idxAttr.count);
              for (let i = 0; i < idxAttr.count; i++) iArr[i] = idxAttr.getX(i) + vOffset;
              idxChunks.push(iArr);
            } else {
              const iArr = new Uint32Array(nVert);
              for (let i = 0; i < nVert; i++) iArr[i] = i + vOffset;
              idxChunks.push(iArr);
            }
            vOffset += nVert;
          });

          const totalIdx = idxChunks.reduce((s, c) => s + c.length, 0);
          const positions = new Float32Array(vOffset * 3);
          const normals = new Float32Array(vOffset * 3);
          const uvs = new Float32Array(vOffset * 2);
          const indices = new Uint32Array(totalIdx);
          let pOff = 0, uOff = 0, iOff = 0;
          for (let c = 0; c < posChunks.length; c++) {
            positions.set(posChunks[c], pOff);
            normals.set(normChunks[c], pOff);
            pOff += posChunks[c].length;
            uvs.set(uvChunks[c], uOff);
            uOff += uvChunks[c].length;
            indices.set(idxChunks[c], iOff);
            iOff += idxChunks[c].length;
          }

          this.assetMesh = root;
          this.scene.add(root);
          const faces = Math.floor(indices.length / 3);
          this.assetData = {
            positions,
            normals,
            uvs,
            indices,
            stats: { topology: 'Triangle (Meshy Cloud)', faces, vertices: vOffset, components: 1 },
          };

          const finalBox = new THREE.Box3().setFromObject(root);
          this.fitCameraToBox(finalBox, 1.24);

          if (this.mesh) this.mesh.visible = false;
          if (this.baseSlab) this.baseSlab.visible = false;
          this._emitStats();
          resolve(this.assetData);
        },
        (err) => reject(err || new Error('Failed parsing GLB asset'))
      );
    });
  }

  clearAssetMesh() {
    this._disposeAssetObject();
    this.assetData = null;
    this.assetMaps = null;
    this._disposeAssetTextures();
    if (this.mesh) this.mesh.visible = true;
    if (this.baseSlab) this.baseSlab.visible = true;
    this.resetCamera();
    this._dirty = true;
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
      return new THREE.MeshMatcapMaterial({
        matcap: this._getMeshyStudioMatCap(),
        color: 0xeaf0f6,
        normalMap: this._assetTextures.normal || null,
        normalScale: new THREE.Vector2(0.88 * (adj.contrast || 1.0), 0.88 * (adj.contrast || 1.0)),
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
    if (this.assetMesh.isMesh) {
      this.assetMesh.material.dispose();
      this.assetMesh.material = this._createAssetMaterial();
      this._dirty = true;
      return;
    }
    const mode = this.shadingMode || 'standard';
    const chan = this.pbrChannel || 'pbr';
    const adj = this.materialAdjust || { roughness: 0.55, metalness: 0.08, contrast: 1.0, intensity: 1.0 };
    this.assetMesh.traverse((child) => {
      if (!child.isMesh) return;
      const orig = child.userData?._origMat;
      if (chan !== 'pbr' && orig) {
        const mapKey = chan === 'baseColor' ? 'map' : chan === 'normal' ? 'normalMap' : chan === 'roughness' ? 'roughnessMap' : chan === 'metallic' ? 'metalnessMap' : null;
        if (mapKey && orig[mapKey]) {
          if (child.material && child.material !== orig && child.material !== child.userData._solidMat) child.material.dispose?.();
          child.material = new THREE.MeshBasicMaterial({ map: orig[mapKey], wireframe: this.wireframe, side: THREE.DoubleSide });
          return;
        }
      }
      if (mode === 'solid') {
        if (!child.userData._solidMat) {
          child.userData._solidMat = new THREE.MeshMatcapMaterial({
            matcap: this._getMeshyStudioMatCap(),
            color: 0xf2f5f9,
            normalMap: orig?.normalMap || null,
            normalScale: new THREE.Vector2(0.55 * (adj.contrast || 1.0), 0.55 * (adj.contrast || 1.0)),
            side: THREE.DoubleSide,
          });
        }
        child.userData._solidMat.wireframe = this.wireframe;
        child.material = child.userData._solidMat;
      } else if (mode === 'unlit' && orig) {
        if (child.material && child.material !== orig && child.material !== child.userData._solidMat) child.material.dispose?.();
        child.material = new THREE.MeshBasicMaterial({
          map: orig.map || null,
          color: orig.color ? orig.color : new THREE.Color(0xdbe2ea),
          wireframe: this.wireframe,
          side: THREE.DoubleSide,
        });
      } else if (orig) {
        child.material = orig;
        const mats = Array.isArray(orig) ? orig : [orig];
        mats.forEach((m) => { if (m) m.wireframe = this.wireframe; });
      }
    });
    this._dirty = true;
  }

  setShadingMode(mode) {
    this.shadingMode = mode || 'standard';
    if (this.assetMesh) {
      this._refreshAssetMaterial();
    } else if (this.mesh) {
      if (mode === 'solid') {
        this.mesh.material.dispose();
        this.mesh.material = new THREE.MeshMatcapMaterial({
          matcap: this._getMeshyStudioMatCap(),
          vertexColors: true,
          color: 0xeaf0f6,
          wireframe: this.wireframe,
        });
      } else if (mode === 'unlit') {
        this.mesh.material.dispose();
        this.mesh.material = new THREE.MeshBasicMaterial({
          map: (this.currentMaterialType === 'wood' && this._designPhotoTex) ? this._designPhotoTex : this.getProceduralTexture(this.currentMaterialType),
          vertexColors: true,
          wireframe: this.wireframe,
        });
      } else {
        this.setMaterial(this.currentMaterialType);
      }
      this._dirty = true;
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
    if (this.assetMesh) {
      this.fitCameraToActive();
    }
    this._dirty = true;
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    if (this.turntable && !this.isRotating) {
      this.spherical.theta += 0.008;
      this.updateCameraPosition();
    }
    if (this._dirty || this.isRotating || this.isPanning || this._simTimer !== null) {
      this.renderer.render(this.scene, this.camera);
      this._dirty = false;
    }
  }
}
