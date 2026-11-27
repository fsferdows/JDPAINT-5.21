# Multi-Skill Engineering Orchestration

How every skill installed in `.agents/skills/` and `~/.agents/skills/` participates in the self-healing development loop for JDPaint 5.21:

| Engineering Phase | Primary Skills Invoked | Concrete Responsibility |
| :--- | :--- | :--- |
| **1. Policy, Recon & Planning** | `gravitas`, `gravitas-python`, `gravitas-typescript`, `gravitas-highstakes` | Emit risk-calibrated Policy Record (`direct`/`compact`/`deep`), read target module + callers + tests before touching code, run 7-vector Adversarial Critique, require user confirmation before destructive resets. |
| **2. Visual & Runtime Inspection** | `jdpaint-2d3d-self-healer` (`scripts/diagnose_2d3d.py`), `use-rea` | Extract 4-panel visual mask breakdown (`Cubic Vectors`, `WhiteTopHat`, `Masks`, `Depth Heatmap`), inspect Chrome console/DOM/WebGL state if frontend renders diverge from backend JSON. |
| **3. 2D CAD & Layer Design** | `jdpaint-cad-auto-trace` | Enforce strict non-overlapping semantic hierarchy (`Architectural Frame`, `Raised Relief Motifs`, `Carved Grooves & Panels`), straight-arm corners, Retinex contrast balance, and grain rejection. |
| **4. 3D Mesh, Shaders & GPU Performance** | `three-best-practices`, `r3f-best-practices` | Guard WebGL `BufferGeometry`/`Texture` disposal on rebuild, preserve crisp directional cavity AO & balanced PBR roughness, update live triangle stats (`_emitStats()`). |
| **5. Parametric CAD & Solid Repair** | `agentcad`, `b123d-modeling`, `b123d-repair`, `b123d-drawing`, `b123d-edit`, `cad-khana` | Diagnose invalid B-Rep faces/open edges in `brep_builder.py` / `OpenCascade`, validate clearance & manifold `.STEP` export gates. |
| **6. Validation & Skill Evolution** | `gravitas`, `create-skill`, `create-rule` | Run full post-edit verification (`verify_2d3d_gates.py`, `cargo test`, `npm run build`, `verify_illustrator_trace.mjs`, `qa_v6_full.py`), inspect screenshots visually via `read_file`, and extend `defect-taxonomy.md` whenever a new failure class is discovered. |
