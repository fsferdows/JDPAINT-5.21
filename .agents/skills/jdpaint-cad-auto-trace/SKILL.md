---
name: jdpaint-cad-auto-trace
description: >-
  Manual-CAD Layer-by-Layer 2D Auto-Trace & 3D Bas-Relief architecture for
  JDPaint 5.21. Use whenever modifying, debugging, or triggering 2D image
  vectorization, CAD frame/curve extraction, layer hierarchy, wood/door/bed
  relief tracing, or Rust geometry fitting so output matches manual CAD drafting
  with zero cracked fragments, zero crooked lines, and zero background grain.
---

# JDPaint Manual-CAD Layer-by-Layer Auto-Trace

Build and maintain 2D CAD vector output that mirrors **human draftsman layer-by-layer tracing** directly over the input image instead of raw bitmap threshold confetti.

## Mandatory Semantic Layer Hierarchy

Every traced design (carved doors, bed headboards, architectural panels, logos, or arbitrary artwork) MUST separate cleanly into non-fighting layers:

1. **`Architectural Frame` (Layer 1 — Cyan `#22d3ee`)**
   - Extract continuous horizontal rails, vertical stiles, rectangular panel frames, fluted pilasters, and curved arch moldings.
   - Bridge collinear segments cleanly where edge support holds, and snap orthogonal rail/stile endpoints within corner tolerance so rectangular frames close at crisp 90-degree intersections (`type: "line"` or unbroken closed frame loops).
   - Never leave broken, jagged, double-walled bitmap outlines over straight structural moldings.

2. **`Carved Motifs & Petals` (Layer 2 — Green `#22c55e`)**
   - Trace raised sculptural elements (leaves, petals, scrolls, rosettes, figurative reliefs) as **complete, smooth, closed C1 Cubic Bézier shapes**.
   - **Never** erode or tear Layer 2 motif boundaries into half-broken fragments when building Layer 3 groove cutouts.
   - Use **straight-arm corner gating**: a curve point is pinned as a sharp corner ONLY when multi-scale turning angle >= `82°`, OR turning angle >= `55°` with both incoming/outgoing arms (`>= 5` contour steps) locally straight (`perp <= 0.85px`). Never collapse spans `< 8.5px` between non-architectural curve bends into straight lines (prevents square pixel-block artifacts inside circular flowers/rosettes).

3. **`Carved Grooves & Details` (Layer 3 — Blue `#38bdf8`)**
   - Suppress any groove pixel within a `<= 4px` corridor of Layer 1 straight architectural lines or Layer 2 raised motif outlines before vectorization.
   - Layer 3 carries strictly disjoint inner vein cuts, non-overlapping engraved recesses, and uncovered detail features — never duplicating or fighting Layer 1/Layer 2 boundaries.

## Clean ROI, Watermark & Grain Discipline

- Run `_inpaint_watermarks` cleanly across natural textured photos even when `detect_furniture_roi` selects full-image framing (`roi is None`), guarding dark charcoal plaques (`l_med > 82`) so dark engraved designs remain intact.
- Trim catalog footer banners (e.g., white/cream logo margins) and suppress disconnected margin text glyphs sitting strictly outside the main horizontal bottom/top door frame bounds.
- Reject 1D vertical wood-grain streaks along bare outer stiles (`x < 10%` or `x > 90%`) whenever they lack cross-rail intersections or multi-directional BlackHat carving energy.

## Visual & Automated Verification Gate

After modifying vectorization or Rust geometry code (`cad-kernel/jdpaint_rust_core/`, `cad-kernel/vision_worker.py`, `jdpaint-mac/src/cad/`):
1. Rebuild & run `cargo test` inside `cad-kernel/jdpaint_rust_core`.
2. Run `verify_illustrator_trace.mjs`, `rust_verify_report.py`, `test_convergence_lowlight.py`, and `qa_v6_full.py`.
3. **Visually inspect rendered screenshots** via `read_file` (`v6_doorsfair_vision_trace.png`, `v6_cropped_door_vision_trace.png`, `v6_dtech_vision_trace.png`, `v6_sagwan500_vision_trace.png`, `v6_eagle_vision_trace.png`, `v6_khoj24_vision_solid.png`):
   - Confirm zero wood-grain streaks on empty left/right door planks.
   - Confirm crisp closed straight rectangular corners and unbroken arches.
   - Confirm smooth closed curved petals with zero jagged square patches or cracked green/cyan tears.
