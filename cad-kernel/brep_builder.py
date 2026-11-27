"""Parametric B-Rep Solid Builder & ISO STEP Exporter powered by build123d (OpenCascade)."""

from __future__ import annotations

import base64
import math
import pathlib
import tempfile
from typing import Any

import build123d as b3d
import numpy as np
import trimesh
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
)
from OCP.BRep import BRep_Tool
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp_Face
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GeomAbs import GeomAbs_C2
from OCP.GeomAPI import GeomAPI_PointsToBSpline
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt, gp_Vec
from OCP.ShapeFix import ShapeFix_Face
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.TColgp import TColgp_Array1OfPnt

try:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.validation import make_valid as shapely_make_valid
except ImportError:
    ShapelyPolygon = None
    shapely_make_valid = None


def _clean_ring(raw_pts: list[tuple[float, float]], tol: float = 0.06) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for x, y in raw_pts:
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if not out or math.hypot(x - out[-1][0], y - out[-1][1]) >= tol:
            out.append((float(x), float(y)))
    if len(out) >= 2 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) < tol:
        out.pop()
    return out


def _sample_entity_ring(
    points: list[dict[str, Any]],
    work_h: float,
    samples_per_seg: int = 8,
) -> list[tuple[float, float]]:
    if not points or len(points) < 2:
        return []
    n = len(points)
    ring_pts = points
    if n >= 3 and math.hypot(points[0]["x"] - points[-1]["x"], points[0]["y"] - points[-1]["y"]) < 0.05:
        ring_pts = points[:-1]
    m = len(ring_pts)
    if m < 3:
        return []

    has_cubic = any("hOut" in p or "hIn" in p for p in ring_pts)
    eff_samples = 2 if m > 120 else (3 if m > 55 else samples_per_seg)
    sampled: list[tuple[float, float]] = []
    if has_cubic:
        for i in range(m):
            a = ring_pts[i]
            b = ring_pts[(i + 1) % m]
            if not a.get("hOut") and not b.get("hIn"):
                sampled.append((a["x"], work_h - a["y"]))
            else:
                p0 = (a["x"], work_h - a["y"])
                c1_raw = a.get("hOut") or a
                c2_raw = b.get("hIn") or b
                c1 = (c1_raw["x"], work_h - c1_raw["y"])
                c2 = (c2_raw["x"], work_h - c2_raw["y"])
                p3 = (b["x"], work_h - b["y"])
                for s in range(eff_samples):
                    t = s / eff_samples
                    mt = 1.0 - t
                    mt2, t2 = mt * mt, t * t
                    bx = mt2 * mt * p0[0] + 3 * mt2 * t * c1[0] + 3 * mt * t2 * c2[0] + t2 * t * p3[0]
                    by = mt2 * mt * p0[1] + 3 * mt2 * t * c1[1] + 3 * mt * t2 * c2[1] + t2 * t * p3[1]
                    sampled.append((bx, by))
    else:
        for p in ring_pts:
            sampled.append((p["x"], work_h - p["y"]))
    return _clean_ring(sampled, tol=0.09 if m > 120 else 0.05)


def _signed_area(ring: list[tuple[float, float]]) -> float:
    area = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


_XY_PLANE = gp_Pln(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0))


def _orient_ring(raw_pts: list[tuple[float, float]], ccw: bool, min_area: float = 0.06) -> list[tuple[float, float]]:
    pts = _clean_ring([(float(x), float(y)) for x, y in raw_pts], tol=0.04)
    area = _signed_area(pts)
    if len(pts) < 3 or abs(area) < min_area:
        return []
    return pts if (area > 0.0) == ccw else list(reversed(pts))


def _fit_bspline_span(pts: list[tuple[float, float]]) -> Any:
    """Fit an analytical C2 OpenCascade B-Spline edge across smooth sampled curve points."""
    n = len(pts)
    arr = TColgp_Array1OfPnt(1, n)
    for idx, (x, y) in enumerate(pts, start=1):
        arr.SetValue(idx, gp_Pnt(x, y, 0.0))
    curve = GeomAPI_PointsToBSpline(arr, 3, 6, GeomAbs_C2, 2e-3).Curve()
    return BRepBuilderAPI_MakeEdge(curve).Edge()


def _ring_to_ocp_wire(ring: list[tuple[float, float]]) -> Any:
    """Convert a closed 2D ring into an OpenCascade wire with analytical C2 B-Splines on curves & crisp lines on sharp corners."""
    n = len(ring)
    if n < 3:
        return None
    mw = BRepBuilderAPI_MakeWire()
    edges_added = 0

    # Detect sharp CAD corner indices (turn > ~34 deg) so rectangles/emblems stay sharp while ovals/petals become smooth C2 splines
    corners: list[int] = []
    cos_thresh = math.cos(math.radians(34.0))
    for i in range(n):
        ax, ay = ring[(i - 1) % n]
        bx, by = ring[i]
        cx, cy = ring[(i + 1) % n]
        u1x, u1y = bx - ax, by - ay
        u2x, u2y = cx - bx, cy - by
        l1, l2 = math.hypot(u1x, u1y), math.hypot(u2x, u2y)
        if l1 > 1e-4 and l2 > 1e-4:
            dot = (u1x * u2x + u1y * u2y) / (l1 * l2)
            if dot < cos_thresh:
                corners.append(i)

    def _add_polyline_span(span: list[tuple[float, float]]) -> None:
        nonlocal edges_added
        for k in range(len(span) - 1):
            x1, y1 = span[k]
            x2, y2 = span[k + 1]
            if math.hypot(x2 - x1, y2 - y1) > 1e-4:
                mw.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(x1, y1, 0.0), gp_Pnt(x2, y2, 0.0)).Edge())
                edges_added += 1

    if not corners and n >= 8:
        # Fully smooth closed loop -> split into 2 continuous C2 B-Spline halves to eliminate faceting cleanly
        half = n // 2
        for span in (ring[: half + 1], ring[half:] + [ring[0]]):
            try:
                mw.Add(_fit_bspline_span(span))
                edges_added += 1
            except Exception:
                _add_polyline_span(span)
    elif corners:
        m = len(corners)
        for ci in range(m):
            a_idx = corners[ci]
            b_idx = corners[(ci + 1) % m]
            span = [ring[a_idx]]
            cur = a_idx
            guard = 0
            while cur != b_idx and guard < n + 2:
                cur = (cur + 1) % n
                span.append(ring[cur])
                guard += 1
            chord = 0.0
            max_dev = 1.0
            if len(span) >= 5:
                # Check if span deviates noticeably from a straight line before fitting B-Spline
                p0, p1 = span[0], span[-1]
                chord = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                max_dev = 0.0
                if chord > 1e-4:
                    dx, dy = (p1[0] - p0[0]) / chord, (p1[1] - p0[1]) / chord
                    for px, py in span[1:-1]:
                        dev = abs(-(px - p0[0]) * dy + (py - p0[1]) * dx)
                        if dev > max_dev:
                            max_dev = dev
                if max_dev > 0.02:
                    try:
                        mw.Add(_fit_bspline_span(span))
                        edges_added += 1
                        continue
                    except Exception:
                        pass
            _add_polyline_span([span[0], span[-1]] if len(span) >= 5 and chord > 1e-4 and max_dev <= 0.015 else span)
    else:
        _add_polyline_span(ring + [ring[0]])

    if edges_added < 2 or not mw.IsDone():
        # Clean fallback straight-edge wire if spline chaining reported tolerance edge error
        mw_fb = BRepBuilderAPI_MakeWire()
        cnt = 0
        for i in range(n):
            x1, y1 = ring[i]
            x2, y2 = ring[(i + 1) % n]
            if math.hypot(x2 - x1, y2 - y1) > 1e-4:
                mw_fb.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(x1, y1, 0.0), gp_Pnt(x2, y2, 0.0)).Edge())
                cnt += 1
        return mw_fb.Wire() if cnt >= 3 and mw_fb.IsDone() else None
    return mw.Wire()


def _make_ocp_face(outer: list[tuple[float, float]], holes: list[list[tuple[float, float]]]) -> tuple[b3d.Face | None, int]:
    eo = _orient_ring(outer, ccw=True, min_area=0.08)
    if not eo:
        return None, 0
    ow = _ring_to_ocp_wire(eo)
    if ow is None:
        return None, 0
    mf = BRepBuilderAPI_MakeFace(_XY_PLANE, ow, True)
    for hr in holes:
        ei = _orient_ring(hr, ccw=False, min_area=0.05)
        if not ei:
            continue
        iw = _ring_to_ocp_wire(ei)
        if iw is not None:
            mf.Add(iw)
    if not mf.IsDone():
        return None, 0
    raw_face = mf.Face()
    try:
        if not BRepCheck_Analyzer(raw_face).IsValid():
            sf = ShapeFix_Face(raw_face)
            sf.Perform()
            raw_face = sf.Face()
            if not BRepCheck_Analyzer(raw_face).IsValid():
                return None, 0
    except Exception:
        return None, 0
    return b3d.Face(raw_face), len(eo)


def _polygon_to_faces_with_insets(
    outer: list[tuple[float, float]],
    holes: list[list[tuple[float, float]]],
    insets_mm: list[float] | None = None,
) -> dict[float, list[tuple[b3d.Face, int]]]:
    """Repair polygon topology via Shapely and generate optional inset tier faces for sculpted 3D crown bevels."""
    tiers: dict[float, list[tuple[b3d.Face, int]]] = {}
    if ShapelyPolygon is not None and shapely_make_valid is not None:
        try:
            repaired = shapely_make_valid(ShapelyPolygon(outer, holes))
            raw_polys = [
                g for g in ([repaired] if isinstance(repaired, ShapelyPolygon) else list(getattr(repaired, "geoms", [])))
                if isinstance(g, ShapelyPolygon) and g.area >= 0.08
            ]
            simp_tol = 0.09 if len(outer) > 120 else 0.02
            base_polys = [
                g.simplify(simp_tol, preserve_topology=True) if len(g.exterior.coords) > 90 else g
                for g in raw_polys
            ]
            for inset in (insets_mm or [0.0]):
                face_items: list[tuple[b3d.Face, int]] = []
                for bp in base_polys:
                    poly = bp if inset <= 0.01 else bp.buffer(-inset, join_style="round")
                    geoms = [poly] if isinstance(poly, ShapelyPolygon) else list(getattr(poly, "geoms", []))
                    for g in geoms:
                        if not isinstance(g, ShapelyPolygon) or g.area < 0.08:
                            continue
                        ext = list(g.exterior.coords)[:-1]
                        inners = [list(ring.coords)[:-1] for ring in g.interiors]
                        face, vc = _make_ocp_face(ext, inners)
                        if face is not None:
                            face_items.append((face, vc))
                if face_items:
                    tiers[inset] = face_items
            if 0.0 in tiers:
                return tiers
        except Exception:
            pass
    face, vc = _make_ocp_face(outer, holes)
    if face is None and holes:
        face, vc = _make_ocp_face(outer, [])
    if face is not None:
        tiers[0.0] = [(face, vc)]
    return tiers


def _build_entity_faces(ent: dict[str, Any], work_h: float, profile: str = "flat", h_mm: float = 4.0) -> list[tuple[b3d.Face, int, float, float]]:
    """Return list of (Face, vertex_count, z_offset_ratio, height_ratio) supporting curved B-Splines & sculpted crown tiers."""
    etype = ent.get("type")
    if etype == "circle":
        r = float(ent.get("r", 0))
        if r <= 0.15:
            return []
        cx = float(ent.get("cx", 0))
        cy = work_h - float(ent.get("cy", 0))
        with b3d.BuildSketch() as sk:
            with b3d.Locations((cx, cy)):
                b3d.Circle(r)
        return [(sk.sketch.faces()[0], 8, 0.0, 1.0)] if sk.sketch.faces() else []

    if etype == "rect":
        rw = abs(float(ent.get("w", 0)))
        rh = abs(float(ent.get("h", 0)))
        if rw <= 0.2 or rh <= 0.2:
            return []
        rx = float(ent.get("x", 0))
        ry = float(ent.get("y", 0))
        cr = max(0.0, min(float(ent.get("cornerRadius", 0) or 0), rw * 0.48, rh * 0.48))
        cx = rx + rw * 0.5
        cy = work_h - (ry + rh * 0.5)
        with b3d.BuildSketch() as sk:
            with b3d.Locations((cx, cy)):
                if cr > 0.15:
                    b3d.RectangleRounded(rw, rh, cr)
                else:
                    b3d.Rectangle(rw, rh)
        return [(sk.sketch.faces()[0], 8 if cr > 0.15 else 4, 0.0, 1.0)] if sk.sketch.faces() else []

    if etype in ("polyline", "curve") and ent.get("isClosed", True):
        outer = _sample_entity_ring(ent.get("points") or [], work_h, samples_per_seg=6)
        if len(outer) < 3 or abs(_signed_area(outer)) < 0.08:
            return []
        valid_holes: list[list[tuple[float, float]]] = []
        for hole_pts in ent.get("holes") or []:
            h_ring = _sample_entity_ring(hole_pts, work_h, samples_per_seg=6)
            if len(h_ring) >= 3 and abs(_signed_area(h_ring)) >= 0.05:
                valid_holes.append(h_ring)

        # Build smooth beveled/domed B-Rep relief tiers when shape is sculptural (skip slow multi-tier boolean fuse on ultra-dense contours)
        sculpted = profile in ("dome", "smooth", "cone", "ogee", "vcarve") and h_mm >= 0.9 and len(outer) <= 140
        insets = [0.0, 0.28, 0.65] if sculpted else [0.0]
        tier_map = _polygon_to_faces_with_insets(outer, valid_holes, insets)
        results: list[tuple[b3d.Face, int, float, float]] = []
        if sculpted and len(tier_map) >= 2:
            tier_specs = [(0.0, 0.0, 0.50), (0.28, 0.50, 0.30), (0.65, 0.80, 0.20)]
            for inset_key, z_frac, h_frac in tier_specs:
                for face, vc in tier_map.get(inset_key, []):
                    results.append((face, vc, z_frac, h_frac))
        else:
            for face, vc in tier_map.get(0.0, []):
                results.append((face, vc, 0.0, 1.0))
        return results

    return []


def export_solid_artifacts(shape: b3d.Shape, tol: float = 0.045) -> dict[str, Any]:
    """Validate OpenCascade B-Rep shape and encode STEP, Watertight GLB (Y-up), & Binary STL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        step_path = pathlib.Path(tmpdir) / "cad_solid.step"
        stl_path = pathlib.Path(tmpdir) / "cad_solid.stl"
        b3d.export_step(shape, str(step_path))
        b3d.export_stl(shape, str(stl_path), tolerance=tol, angular_tolerance=0.12)
        step_bytes = step_path.read_bytes()
        mesh_stl = trimesh.load_mesh(str(stl_path), process=True)
        mesh_stl.merge_vertices(digits_vertex=4)
        try:
            trimesh.repair.fill_holes(mesh_stl)
        except Exception:
            pass
        stl_bytes = mesh_stl.export(file_type="stl")

    # Extract analytical per-face B-Rep triangulations + NURBS surface normals directly from OpenCascade
    # so planar caps keep exact flat normals (zero diagonal corner shading) and C2 splines render smooth.
    BRepMesh_IncrementalMesh(shape.wrapped, tol, False, 0.12, True)
    verts_chunks: list[np.ndarray] = []
    faces_chunks: list[np.ndarray] = []
    norms_chunks: list[np.ndarray] = []
    v_offset = 0
    exp = TopExp_Explorer(shape.wrapped, TopAbs_FACE)
    p_tmp = gp_Pnt()
    n_tmp = gp_Vec()
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None and tri.NbNodes() > 0 and tri.NbTriangles() > 0:
            trsf = loc.Transformation()
            is_rev = face.Orientation() == TopAbs_REVERSED
            is_neg = bool(trsf.IsNegative())
            flip_wind = is_rev != is_neg
            n_nodes = tri.NbNodes()
            n_tris = tri.NbTriangles()
            f_verts = np.empty((n_nodes, 3), dtype=np.float64)
            f_norms = np.zeros((n_nodes, 3), dtype=np.float64)
            valid_n = np.zeros(n_nodes, dtype=bool)
            has_uv = tri.HasUVNodes()
            prop = BRepGProp_Face(face) if has_uv else None
            for idx_node in range(1, n_nodes + 1):
                pt = tri.Node(idx_node).Transformed(trsf)
                f_verts[idx_node - 1] = (pt.X(), pt.Z(), -pt.Y())
                if prop is not None:
                    uv = tri.UVNode(idx_node)
                    prop.Normal(uv.X(), uv.Y(), p_tmp, n_tmp)
                    mag = n_tmp.Magnitude()
                    if mag > 1e-12:
                        nt = gp_Vec(n_tmp.X(), n_tmp.Y(), n_tmp.Z()).Transformed(trsf)
                        if is_neg:
                            nt.Reverse()
                        f_norms[idx_node - 1] = (nt.X() / mag, nt.Z() / mag, -nt.Y() / mag)
                        valid_n[idx_node - 1] = True
            f_tris = np.empty((n_tris, 3), dtype=np.int32)
            for idx_tri in range(1, n_tris + 1):
                n1, n2, n3 = tri.Triangle(idx_tri).Get()
                if flip_wind:
                    f_tris[idx_tri - 1] = (n1 - 1, n3 - 1, n2 - 1)
                else:
                    f_tris[idx_tri - 1] = (n1 - 1, n2 - 1, n3 - 1)
            if not np.all(valid_n):
                tri_pts = f_verts[f_tris]
                geo_fn = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
                fallback = np.zeros_like(f_verts)
                np.add.at(fallback, f_tris.ravel(), np.repeat(geo_fn, 3, axis=0))
                fb_lens = np.linalg.norm(fallback, axis=1, keepdims=True)
                fallback = np.where(fb_lens > 1e-12, fallback / np.maximum(fb_lens, 1e-12), np.array([[0.0, 1.0, 0.0]]))
                f_norms[~valid_n] = fallback[~valid_n]
            verts_chunks.append(f_verts)
            faces_chunks.append(f_tris + v_offset)
            norms_chunks.append(f_norms)
            v_offset += n_nodes
        exp.Next()

    if verts_chunks:
        split_verts = np.vstack(verts_chunks)
        split_faces = np.vstack(faces_chunks)
        split_normals = np.vstack(norms_chunks)
    else:
        np_verts_zup = mesh_stl.vertices
        split_verts = np.column_stack([np_verts_zup[:, 0], np_verts_zup[:, 2], -np_verts_zup[:, 1]])
        split_faces = np.asarray(mesh_stl.faces, dtype=np.int32)
        fn_zup = np.asarray(mesh_stl.face_normals, dtype=np.float64)
        split_normals = np.column_stack([fn_zup[:, 0], fn_zup[:, 2], -fn_zup[:, 1]])
    mat = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[188, 202, 220, 255],
        roughnessFactor=0.38,
        metallicFactor=0.22,
    )
    mesh_glb = trimesh.Trimesh(
        vertices=split_verts,
        faces=split_faces,
        vertex_normals=split_normals,
        visual=trimesh.visual.TextureVisuals(material=mat),
        process=False,
    )
    glb_bytes = trimesh.Scene([mesh_glb]).export(file_type="glb", include_normals=True)

    is_valid = bool(BRepCheck_Analyzer(shape.wrapped).IsValid())
    bb = shape.bounding_box()
    solids_count = len(shape.solids()) if hasattr(shape, "solids") else 1
    faces_count = len(shape.faces()) if hasattr(shape, "faces") else 0

    return {
        "metrics": {
            "is_valid": is_valid,
            "watertight": bool(mesh_stl.is_watertight),
            "volume_mm3": round(float(getattr(shape, "volume", 0.0)), 3),
            "area_mm2": round(float(getattr(shape, "area", 0.0)), 3),
            "solids": solids_count,
            "brep_faces": faces_count,
            "mesh_triangles": int(len(mesh_stl.faces)),
            "bbox": {
                "min": [round(bb.min.X, 3), round(bb.min.Y, 3), round(bb.min.Z, 3)],
                "max": [round(bb.max.X, 3), round(bb.max.Y, 3), round(bb.max.Z, 3)],
            },
        },
        "step_b64": base64.b64encode(step_bytes).decode("ascii"),
        "glb_b64": base64.b64encode(glb_bytes).decode("ascii"),
        "stl_b64": base64.b64encode(stl_bytes).decode("ascii"),
    }


def build_solid_from_vectors(payload: dict[str, Any]) -> dict[str, Any]:
    """Extrude & boolean JDPaint 2D CAD vector entities into a validated OpenCascade B-Rep Solid."""
    work_w = max(5.0, float(payload.get("workpieceWidth", 100.0)))
    work_h = max(5.0, float(payload.get("workpieceHeight", 100.0)))
    base_mm = max(0.0, float(payload.get("baseThicknessMm", 6.0)))
    default_h = max(0.2, float(payload.get("extrudeHeightMm", 5.0)))
    fillet_mm = max(0.0, float(payload.get("filletRadiusMm", 0.0)))
    entities = payload.get("entities") or []

    # Batch entities by (is_cut, z_offset, extrude_amount) to avoid N slow 3D boolean fuse/cut passes
    groups: dict[tuple[bool, float, float], list[tuple[b3d.Face, int]]] = {}
    built_features = 0
    for ent in entities[:160]:
        if ent.get("visible", True) is False:
            continue
        relief = ent.get("relief") or {}
        if relief.get("enabled") is False:
            continue
        h_mm = round(max(0.25, float(relief.get("heightMm") or default_h)), 2)
        mode_str = str(relief.get("mode", "add")).lower()
        is_cut = base_mm > 0.2 and mode_str in ("subtract", "carve", "min", "cut")
        profile = str(relief.get("shape", "flat")).lower()
        try:
            entity_faces = _build_entity_faces(ent, work_h, profile="flat" if is_cut else profile, h_mm=h_mm)
        except Exception:
            entity_faces = []
        if not entity_faces:
            continue
        for face, vc, z_frac, h_frac in entity_faces:
            z_off = round(h_mm * z_frac, 2)
            ext_h = round(max(0.18, h_mm * h_frac), 2)
            groups.setdefault((is_cut, z_off, ext_h), []).append((face, vc))
        built_features += 1

    if base_mm <= 0.05 and built_features == 0:
        raise ValueError("No valid closed 2D CAD shapes with 3D height found to extrude.")

    with b3d.BuildPart() as bp:
        if base_mm > 0.05:
            with b3d.Locations((work_w * 0.5, work_h * 0.5, base_mm * 0.5)):
                b3d.Box(work_w, work_h, base_mm)

        for (is_cut, z_off, h_seg), face_items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            face_list = [f for f, _ in face_items]
            max_verts = max(vc for _, vc in face_items)
            try:
                if is_cut:
                    cut_depth = min(base_mm - 0.15, h_seg)
                    if cut_depth > 0.1:
                        solids = [b3d.extrude(f, amount=cut_depth).translate((0, 0, base_mm - cut_depth)) for f in face_list]
                        b3d.add(b3d.Compound(solids) if len(solids) > 1 else solids[0], mode=b3d.Mode.SUBTRACT)
                else:
                    z0 = (base_mm if base_mm > 0.05 else 0.0) + z_off
                    solids = [b3d.extrude(f, amount=h_seg).translate((0, 0, z0)) for f in face_list]
                    comp = b3d.Compound(solids) if len(solids) > 1 else solids[0]
                    if fillet_mm >= 0.1 and max_verts <= 36:
                        try:
                            top_edges = comp.edges().filter_by_position(b3d.Axis.Z, z0 + h_seg - 0.02, z0 + h_seg + 0.02)
                            if top_edges:
                                comp = comp.fillet(min(fillet_mm, h_seg * 0.40), edge_list=top_edges)
                        except Exception:
                            pass
                    b3d.add(comp, mode=b3d.Mode.ADD)
            except Exception:
                continue

        part = bp.part if hasattr(bp, "part") and bp.part is not None else b3d.Compound(bp.solids())

    result = export_solid_artifacts(part, tol=0.052)
    result["features_built"] = built_features
    return result


def build_parametric_preset(preset_id: str, raw_params: dict[str, Any]) -> dict[str, Any]:
    """Generate exact industrial B-Rep CAD parts via parametric build123d templates."""
    p = raw_params or {}
    pid = (preset_id or "bracket").strip().lower()

    with b3d.BuildPart() as bp:
        if pid == "bracket":
            w = max(20.0, min(300.0, float(p.get("width", 70.0))))
            d = max(20.0, min(300.0, float(p.get("depth", 55.0))))
            h = max(20.0, min(300.0, float(p.get("height", 60.0))))
            t = max(2.0, min(min(d, h) * 0.35, float(p.get("thickness", 6.0))))
            hole_d = max(2.0, min(min(w, d) * 0.4, float(p.get("holeDia", 6.5))))
            fillet_r = max(0.0, min(t * 0.45, float(p.get("fillet", 2.0))))

            with b3d.Locations((0, d * 0.5, t * 0.5)):
                b3d.Box(w, d, t)
            with b3d.Locations((0, t * 0.5, h * 0.5)):
                b3d.Box(w, t, h)
            # Side triangular strengthening gussets
            gusset_pts = [(t, t), (min(d - t * 0.8, h * 0.65), t), (t, min(h - t * 0.8, d * 0.65))]
            with b3d.BuildSketch(b3d.Plane.YZ):
                with b3d.BuildLine():
                    b3d.Polyline(*gusset_pts, close=True)
                b3d.make_face()
            b3d.extrude(amount=t * 0.5, both=True)
            # Mounting holes on base & vertical wall
            gx_off = w * 0.32
            with b3d.Locations((-gx_off, d * 0.65, t * 0.5), (gx_off, d * 0.65, t * 0.5)):
                b3d.Cylinder(hole_d * 0.5, t + 2.0, mode=b3d.Mode.SUBTRACT)
            with b3d.Locations((0, t * 0.5, h * 0.72)):
                b3d.Cylinder(hole_d * 0.5, t + 2.0, rotation=(90, 0, 0), mode=b3d.Mode.SUBTRACT)
            if fillet_r >= 0.2:
                try:
                    z_edges = bp.edges().filter_by(b3d.Axis.Z)
                    if z_edges:
                        b3d.fillet(z_edges, radius=min(fillet_r, 1.8))
                except Exception:
                    pass

        elif pid == "flange":
            od = max(24.0, min(300.0, float(p.get("outerDia", 86.0))))
            bore = max(4.0, min(od * 0.6, float(p.get("innerDia", 22.0))))
            hub_d = max(bore + 8.0, min(od * 0.8, float(p.get("hubDia", 42.0))))
            h_base = max(3.0, min(40.0, float(p.get("height", 8.0))))
            h_hub = max(2.0, min(60.0, float(p.get("hubHeight", 16.0))))
            bolts = max(3, min(12, int(p.get("boltCount", 6))))
            bolt_d = max(2.0, min((od - hub_d) * 0.38, float(p.get("boltDia", 6.5))))

            with b3d.Locations((0, 0, h_base * 0.5)):
                b3d.Cylinder(od * 0.5, h_base)
            with b3d.Locations((0, 0, h_base + h_hub * 0.5)):
                b3d.Cylinder(hub_d * 0.5, h_hub)
            with b3d.Locations((0, 0, (h_base + h_hub) * 0.5)):
                b3d.Cylinder(bore * 0.5, h_base + h_hub + 4.0, mode=b3d.Mode.SUBTRACT)
            pcd = (od * 0.5 + hub_d * 0.5) * 0.5
            with b3d.PolarLocations(pcd, bolts):
                with b3d.Locations((0, 0, h_base * 0.5)):
                    b3d.Cylinder(bolt_d * 0.5, h_base + 2.0, mode=b3d.Mode.SUBTRACT)
            try:
                b3d.chamfer(bp.edges().filter_by_position(b3d.Axis.Z, h_base + h_hub - 0.05, h_base + h_hub + 0.05), length=0.8)
            except Exception:
                pass

        elif pid == "enclosure":
            w = max(30.0, min(320.0, float(p.get("width", 96.0))))
            d = max(30.0, min(320.0, float(p.get("depth", 64.0))))
            h = max(12.0, min(180.0, float(p.get("height", 28.0))))
            wall = max(1.6, min(min(w, d, h) * 0.15, float(p.get("wall", 3.0))))
            cr = max(1.0, min(min(w, d) * 0.22, float(p.get("cornerRadius", 6.0))))

            with b3d.BuildSketch():
                b3d.RectangleRounded(w, d, cr)
            b3d.extrude(amount=h)
            with b3d.BuildSketch(b3d.Plane.XY.offset(wall)):
                b3d.RectangleRounded(w - wall * 2, d - wall * 2, max(0.6, cr - wall * 0.5))
            b3d.extrude(amount=h, mode=b3d.Mode.SUBTRACT)
            # Corner PCB mounting bosses
            bx = w * 0.5 - wall - 5.0
            by = d * 0.5 - wall - 5.0
            for sx in (-1, 1):
                for sy in (-1, 1):
                    with b3d.Locations((sx * bx, sy * by, wall + (h - wall) * 0.35)):
                        b3d.Cylinder(3.2, (h - wall) * 0.7)
                        b3d.Cylinder(1.4, (h - wall) * 0.85, mode=b3d.Mode.SUBTRACT)

        elif pid == "door_panel":
            w = max(40.0, min(1500.0, float(p.get("width", 90.0))))
            h = max(60.0, min(2600.0, float(p.get("height", 180.0))))
            t = max(8.0, min(60.0, float(p.get("thickness", 18.0))))
            stile = max(6.0, min(w * 0.28, float(p.get("stileWidth", 14.0))))
            pdepth = max(2.0, min(t * 0.65, float(p.get("panelDepth", 7.0))))

            with b3d.Locations((0, 0, t * 0.5)):
                b3d.Box(w, h, t)
            pw = w - stile * 2
            top_h = (h - stile * 3) * 0.56
            bot_h = (h - stile * 3) * 0.44
            cy_top = stile * 0.5 + top_h * 0.5
            cy_bot = -(stile * 0.5 + bot_h * 0.5)
            for cy, ph in ((cy_top, top_h), (cy_bot, bot_h)):
                with b3d.Locations((0, cy, t - pdepth * 0.5)):
                    b3d.Box(pw, ph, pdepth + 0.5, mode=b3d.Mode.SUBTRACT)
                panel_h = pdepth * 0.55
                with b3d.Locations((0, cy, (t - pdepth) + panel_h * 0.5 - 0.3)):
                    b3d.Box(max(4.0, pw - 5.0), max(4.0, ph - 5.0), panel_h + 0.6)
        else:
            raise ValueError(f"Unknown parametric preset: {preset_id}")

        part = bp.part

    return export_solid_artifacts(part, tol=0.04)
