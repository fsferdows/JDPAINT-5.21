---
name: jdpaint-2d3d-self-healer
description: >-
  Autonomous 2D CAD vector and 3D relief/B-Rep self-diagnosis, root-cause localization,
  closed-loop repair, and multi-skill verification workflow for JDPaint. Use whenever
  input images produce inaccurate 2D outlines, missing/merged petals, jagged curves,
  wood-grain noise, watermark erasure inside artwork, collapsed/spiky 3D previews,
  disconnected 3D layers, or B-Rep/STEP failures.
---

# JDPaint Autonomous 2D+3D Self-Healing Developer System

Diagnose and repair any 2D vectorization, 3D relief/asset reconstruction, layered CNC sync, or B-Rep failure from arbitrary user input images using a deterministic **Reproduce -> Classify -> Ground -> Patch -> Verify -> Visually Inspect** loop across Python, Rust, and Three.js/JS.

## Mandatory Multi-Skill Orchestra

Compose all relevant skills together rather than working ad hoc (details in [references/skill-orchestration.md](references/skill-orchestration.md)):

1. **Engineering Discipline (`gravitas`, `gravitas-python`, `gravitas-typescript`, `gravitas-highstakes`):**
   - Ground before mutating: read the target function, callers, and tests first.
   - Never exact-retry a failed fix without changing diagnosis or strategy.
   - Require post-final-edit validator evidence before reporting `VERDICT: PASS`.
2. **2D CAD Layer Architecture (`jdpaint-cad-auto-trace`):**
   - Enforce disjoint 3-layer output: `Architectural Frame` (`#22d3ee`) -> `Raised Relief Motifs` (`#22c55e`) -> `Carved Grooves & Panels` (`#38bdf8`).
3. **3D Viewport & Mesh Health (`three-best-practices`):**
   - Verify `.dispose()` on rebuilt geometries/textures, live `_emitStats()`, curvature-preserving Taubin smoothing, and balanced PBR cavity lighting.
4. **Parametric B-Rep & Inspection (`agentcad`, `b123d-modeling`, `b123d-repair`, `cad-khana`, `use-rea`):**
   - Validate STEP solids via `BRepCheck` / `locate_gate_defects()` and inspect live browser runtime state when UI output diverges from backend JSON.

---

## Closed-Loop Workflow (Follow In Order)

```text
Progress Checklist:
- [ ] Step 1: Reproduce defects on the target image with scripts/diagnose_2d3d.py
- [ ] Step 2: Classify defect signatures via references/defect-taxonomy.md
- [ ] Step 3: Read exact failing source block & callers (zero assumed state)
- [ ] Step 4: Implement generalized algorithmic fix (Python / Rust / JS)
- [ ] Step 5: Run full post-edit verification gate via scripts/verify_2d3d_gates.py
- [ ] Step 6: Visually inspect diag_*.png and live Playwright Standard + Solid renders
```

### Step 1 — Run Pixel + Vector + 3D Reproduction

Execute the bundled analyzer (uses `/Users/user/Desktop/CNC/.venv/bin/python` where `cv2`, `numpy`, `skimage`, `torch` live):

```bash
"/Users/user/Desktop/CNC/.venv/bin/python" .agents/skills/jdpaint-2d3d-self-healer/scripts/diagnose_2d3d.py \
  --image "/path/to/input.jpg" \
  --out-dir "/tmp/jdpaint_diag"
```

Always inspect the generated 4-panel proof strip (`diag_<stem>.png`) via `read_file` before deciding which layer failed.

### Step 2 — Localize Root Cause & Critique Downstream Risk

Match `DEFECT_FLAGS` from Step 1 against [references/defect-taxonomy.md](references/defect-taxonomy.md). Before editing, verify downstream tests (`verify_illustrator_trace.mjs` Tests 1–11 and Rust `cargo test` 4/4) will not regress (for example, keeping `index.html` and `new ReliefEngine(100, 100, 8, 384)` initial defaults intact while upgrading dynamically at trace time).

### Step 3 — Implement Principled Fixes

- Keep fixes geometric and scale-invariant (Retinex reflectance normalization, local `White TopHat / env_wh` contrast adaptation, structural rail anchoring, multi-scale White TopHat + BlackHat Poisson depth, semantic layer relief modes) — never tune pixel coordinates for one file.
- Maintain **Zero-CJK** inside `jdpaint-mac/src/`.

### Step 4 — Deterministic Verification & Visual Proof Gate

Run the unified verification runner strictly **after** the final edit:

```bash
"/Users/user/Desktop/CNC/.venv/bin/python" .agents/skills/jdpaint-2d3d-self-healer/scripts/verify_2d3d_gates.py
```

And visually inspect both zoomed 2D vector splines and 3D `Standard` + `Solid` screenshots (`read_file`). Only emit `VERDICT: PASS` when every check exits `0` after the last file write.
