# JDPaint 5.21 (Mac Edition)

A high-performance clean-room 2D CAD drafting, 3D relief sculpting, and CNC CAM toolpath calculation studio built natively for macOS and modern web browsers.

---

## Quick Launch on Mac

You can launch JDPaint in either of two ways:

### Option 1: Double-Click in Finder
Double-click the **`Start JDPaint.command`** file in this folder. It will start the local engine and automatically open JDPaint in your browser.

### Option 2: From Terminal
```bash
cd "/Users/user/Desktop/JDPaint 5.21/jdpaint-mac"
./launch.sh
```
Or:
```bash
cd "/Users/user/Desktop/JDPaint 5.21/jdpaint-mac"
npm install
npm run dev -- --open
```

---

## Key Features

### 1. 2D Vector CAD Drafting Engine
* **Drafting Tools**: Line, Polyline, Rectangle, Circle/Arc, Quadratic B-Spline Curve, Pen Tool, Freehand Pencil, and Engraving Text.
* **Node Editing & Reshape**: Direct vertex sub-selection, interactive handle manipulation, vertex addition, deletion, and closed contour welding.
* **Vector Transformation & Geometry**:
  * Precision Parallel Miter Offset (inward/outward contours).
  * Corner Filleting with radius specification.
  * 6-Way Precision CAD Alignment (Left, Center X, Right, Top, Center Y, Bottom).
  * Shape Boolean Operations: Union and Difference.
* **Multi-Layer CAD Architecture**: Group paths into distinct layers, control visibility, lock states, stroke colors, and layer ordering.
* **CAD Canvas Navigation**: Pan, smooth zoom, millimeter ruler guides, snap-to-grid, and touch/gesture support.

### 2. High-Precision Vector Auto-Tracing
* Direct raster-to-vector auto-tracing with multi-threshold Canny edge detection and Otsu binarization.
* Douglas-Peucker contour simplification with fine-tuned noise filtration for woodworking, furniture ornamentation, and CNC relief patterns.

### 3. 3D Relief Modeling & Sculpting Engine
* **Vector-to-3D Relief Puff Engine**:
  * Exact Felzenszwalb-Huttenlocher Euclidean Distance Transform (EDT) for instant, clean vector-to-relief conversion.
  * Multiple carving profiles: **Spherical Dome**, **Classical Ogee / S-Curve**, **V-Carve**, **Scoop / Concave Flute**, and **Flat Island / Pocket**.
  * Dynamic height, puff curvature angle, and sub-pixel anti-aliasing.
* **Bitmap Relief Heightmap Sculpting**:
  * Convert any grayscale or color graphic into a continuous 2.5D heightmap.
  * Sculpting filters: Gaussian smoothing (eliminates tool marks), inversion (convex to concave relief), base height offset.
  * Built-in relief generators: Classical Lotus Rosette Medallion, Radial Dome, and Geometric Pyramid.

### 4. Interactive 3D WebGL Viewport (Three.js)
* Real-time 3D orbital camera with OrbitControls (Rotate, Pan, Zoom).
* Realistic PBR materials:
  * **Hetian Jade**
  * **Walnut / Rosewood**
  * **Honey Oak**
  * **Polished Brass**
  * **Dark Slate / Granite**
  * **Height Heatmap**
* Multi-angle studio raking light with ACESFilmic tone mapping and drop shadows.
* Realistic backing stock slab visualization and interactive 3D Spindle Machining Simulation.

### 5. CNC CAM & Toolpath Generation
* **Cutter Library**: Ball-Nose, Flat Endmill, and V-Bit Engraver with customizable diameter, stepover, feedrate, plunge rate, and safe Z.
* **Machining Strategies**:
  * Full 3D surface relief zigzag raster finishing with cutter radius compensation.
  * 2D contour profile cutting (Inside, Outside, On-Path).
  * 2D pocket area clearing with multi-pass stepdown.
* **Toolpath Statistics**: Waypoint counts, total cutting distance (mm), and estimated machining time.

### 6. Post-Processors & Industry Export
* **CNC G-Code Export (`.nc`, `.gcode`, `.tap`, `.eng`)**: Compatible with standard ISO G-code, GRBL, Mach3, LinuxCNC, and Beijing Jingdiao CNC routers.
* **3D Binary STL & Wavefront OBJ Export**: Export clean CCW outward-normal watertight meshes for 3D printing and CAD software.
* **AutoCAD R12 DXF & SVG Export**: Vector interchange standard for laser cutters, vinyl cutters, and CNC machines.
* **Project Save/Load (`.jdp.json`)**: Preserves full vector layers, dimensions, and 3D relief parameters.

---

## Technical Stack
* **Frontend**: HTML5 Canvas, Vanilla ES2022 JavaScript, CSS3 Responsive Flex/Grid Layout
* **3D Engine**: Three.js (WebGL, PBR MeshStandardMaterial, Directional Raking Lights)
* **Build System**: Vite
* **Architecture**: Zero bloat, instant hot module reload, cross-platform macOS/Linux/Windows compatibility.

---

## License
MIT License.
