#![forbid(unsafe_code)]

use std::env;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::process;
use std::thread;

#[derive(Clone, Copy, Debug)]
struct Pt2 {
    x: f32,
    y: f32,
}

impl Pt2 {
    #[inline]
    fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
    #[inline]
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y)
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y)
    }
    #[inline]
    fn mul(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s)
    }
    #[inline]
    fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y
    }
    #[inline]
    fn len_sq(self) -> f32 {
        self.dot(self)
    }
    #[inline]
    fn len(self) -> f32 {
        self.len_sq().sqrt()
    }
    #[inline]
    fn norm(self) -> Self {
        let l = self.len();
        if l > 1e-6 {
            self.mul(1.0 / l)
        } else {
            Self::new(0.0, 0.0)
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct CadNode {
    x: f32,
    y: f32,
    h_in_x: f32,
    h_in_y: f32,
    h_out_x: f32,
    h_out_y: f32,
    flags: u32, // bit 0 = corner, bit 1 = has_in, bit 2 = has_out
}

const FLAG_CORNER: u32 = 1;
const FLAG_HAS_IN: u32 = 2;
const FLAG_HAS_OUT: u32 = 4;

struct CadRing {
    is_hole: bool,
    parent_idx: i32,
    nodes: Vec<CadNode>,
}

#[derive(Clone, Copy)]
struct EdgeSeg {
    start_edge: u32,
    end_edge: u32,
    start_pt: Pt2,
}

fn read_u32_le<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_f32_le<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

#[inline]
fn lerp_edge(thresh: f32, va: f32, vb: f32) -> f32 {
    let denom = vb - va;
    if denom.abs() < 1e-5 {
        0.5
    } else {
        ((thresh - va) / denom).clamp(0.02, 0.98)
    }
}

#[inline]
fn cell_edge_key_and_pt(
    cx: usize,
    cy: usize,
    side: u8,
    pw: usize,
    thresh: f32,
    v0: f32,
    v1: f32,
    v2: f32,
    v3: f32,
) -> (u32, Pt2) {
    match side {
        0 => {
            // Top edge: (cx, cy) -> (cx+1, cy)
            let t = lerp_edge(thresh, v0, v1);
            let key = (((cy * pw + cx) << 1) | 0) as u32;
            (key, Pt2::new(cx as f32 + t - 1.0, cy as f32 - 1.0))
        }
        1 => {
            // Right edge: (cx+1, cy) -> (cx+1, cy+1)
            let t = lerp_edge(thresh, v1, v2);
            let key = (((cy * pw + (cx + 1)) << 1) | 1) as u32;
            (key, Pt2::new((cx + 1) as f32 - 1.0, cy as f32 + t - 1.0))
        }
        2 => {
            // Bottom edge: (cx+1, cy+1) -> (cx, cy+1), stored at left-to-right vertex (cx, cy+1)
            let t = lerp_edge(thresh, v3, v2);
            let key = ((((cy + 1) * pw + cx) << 1) | 0) as u32;
            (key, Pt2::new(cx as f32 + t - 1.0, (cy + 1) as f32 - 1.0))
        }
        _ => {
            // Left edge: (cx, cy+1) -> (cx, cy), stored at top-to-bottom vertex (cx, cy)
            let t = lerp_edge(thresh, v0, v3);
            let key = (((cy * pw + cx) << 1) | 1) as u32;
            (key, Pt2::new(cx as f32 - 1.0, cy as f32 + t - 1.0))
        }
    }
}

/// Extract closed subpixel Marching Squares contour loops with foreground on the right.
/// Outer contours wind negatively in screen Y-down coordinates; holes wind positively.
fn marching_squares_contours(field: &[u8], w: usize, h: usize, thresh: f32) -> Vec<Vec<Pt2>> {
    let pw = w + 2;
    let ph = h + 2;
    let mut padded = vec![0.0f32; pw * ph];
    for y in 0..h {
        let dst = (y + 1) * pw + 1;
        let src = y * w;
        for x in 0..w {
            padded[dst + x] = field[src + x] as f32;
        }
    }

    let edge_slots = pw * ph * 2;
    // Map start_edge_key -> segment index in `segs`; saddle edges share keys across 2 disjoint cells?
    // Wait: every edge between two corners is crossed at most once in one direction per cell,
    // EXCEPT wait: in a single edge shared by two cells, the directed edge leaves cell A on one side and enters cell B!
    // Each cell owns distinct `start_edge`? YES: walking with foreground on right determines a UNIQUE direction along every crossing edge!
    let mut edge_to_seg = vec![u32::MAX; edge_slots];
    let mut segs: Vec<EdgeSeg> = Vec::with_capacity((w + h) * 4);

    let mut push_pair = |cx: usize,
                         cy: usize,
                         s_side: u8,
                         e_side: u8,
                         v0: f32,
                         v1: f32,
                         v2: f32,
                         v3: f32| {
        let (sk, sp) = cell_edge_key_and_pt(cx, cy, s_side, pw, thresh, v0, v1, v2, v3);
        let (ek, _ep) = cell_edge_key_and_pt(cx, cy, e_side, pw, thresh, v0, v1, v2, v3);
        let idx = segs.len() as u32;
        segs.push(EdgeSeg {
            start_edge: sk,
            end_edge: ek,
            start_pt: sp,
        });
        edge_to_seg[sk as usize] = idx;
    };

    for cy in 0..(ph - 1) {
        let row0 = cy * pw;
        let row1 = (cy + 1) * pw;
        for cx in 0..(pw - 1) {
            let v0 = padded[row0 + cx];
            let v1 = padded[row0 + cx + 1];
            let v2 = padded[row1 + cx + 1];
            let v3 = padded[row1 + cx];
            let code = ((v0 >= thresh) as u8) << 3
                | ((v1 >= thresh) as u8) << 2
                | ((v2 >= thresh) as u8) << 1
                | ((v3 >= thresh) as u8);
            match code {
                0 | 15 => {}
                1 => push_pair(cx, cy, 3, 2, v0, v1, v2, v3),
                2 => push_pair(cx, cy, 2, 1, v0, v1, v2, v3),
                3 => push_pair(cx, cy, 3, 1, v0, v1, v2, v3),
                4 => push_pair(cx, cy, 1, 0, v0, v1, v2, v3),
                5 => {
                    let vc = 0.25 * (v0 + v1 + v2 + v3);
                    if vc >= thresh {
                        push_pair(cx, cy, 1, 2, v0, v1, v2, v3);
                        push_pair(cx, cy, 3, 0, v0, v1, v2, v3);
                    } else {
                        push_pair(cx, cy, 1, 0, v0, v1, v2, v3);
                        push_pair(cx, cy, 3, 2, v0, v1, v2, v3);
                    }
                }
                6 => push_pair(cx, cy, 2, 0, v0, v1, v2, v3),
                7 => push_pair(cx, cy, 3, 0, v0, v1, v2, v3),
                8 => push_pair(cx, cy, 0, 3, v0, v1, v2, v3),
                9 => push_pair(cx, cy, 0, 2, v0, v1, v2, v3),
                10 => {
                    let vc = 0.25 * (v0 + v1 + v2 + v3);
                    if vc >= thresh {
                        push_pair(cx, cy, 0, 1, v0, v1, v2, v3);
                        push_pair(cx, cy, 2, 3, v0, v1, v2, v3);
                    } else {
                        push_pair(cx, cy, 0, 3, v0, v1, v2, v3);
                        push_pair(cx, cy, 2, 1, v0, v1, v2, v3);
                    }
                }
                11 => push_pair(cx, cy, 0, 1, v0, v1, v2, v3),
                12 => push_pair(cx, cy, 1, 3, v0, v1, v2, v3),
                13 => push_pair(cx, cy, 1, 2, v0, v1, v2, v3),
                14 => push_pair(cx, cy, 2, 3, v0, v1, v2, v3),
                _ => {}
            }
        }
    }

    let mut visited = vec![false; segs.len()];
    let mut loops = Vec::new();
    for i in 0..segs.len() {
        if visited[i] {
            continue;
        }
        let mut pts = Vec::with_capacity(64);
        let start_k = segs[i].start_edge;
        let mut cur = i;
        for _ in 0..(segs.len() + 1) {
            if visited[cur] {
                break;
            }
            visited[cur] = true;
            let seg = segs[cur];
            // Deduplicate nearly identical vertices from clamped interpolation
            if pts
                .last()
                .map_or(true, |p: &Pt2| p.sub(seg.start_pt).len_sq() > 1e-4)
            {
                pts.push(seg.start_pt);
            }
            if seg.end_edge == start_k {
                break;
            }
            let nxt = edge_to_seg[seg.end_edge as usize];
            if nxt == u32::MAX {
                break;
            }
            cur = nxt as usize;
        }
        if pts.len() >= 3 && pts.first().unwrap().sub(*pts.last().unwrap()).len_sq() < 1e-4 {
            pts.pop();
        }
        if pts.len() >= 3 {
            loops.push(pts);
        }
    }
    loops
}

fn signed_area(pts: &[Pt2]) -> f32 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0f64;
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        s += (a.x as f64) * (b.y as f64) - (b.x as f64) * (a.y as f64);
    }
    (0.5 * s) as f32
}

fn point_in_poly(p: Pt2, poly: &[Pt2]) -> bool {
    let n = poly.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let pi = poly[i];
        let pj = poly[j];
        let cond = (pi.y > p.y) != (pj.y > p.y);
        if cond {
            let x_int = (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y + 1e-12) + pi.x;
            if p.x < x_int {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

fn arm_max_perp(raw: &[Pt2], start: usize, steps: usize, n: usize) -> f32 {
    if steps < 3 || n < 4 {
        return 0.0;
    }
    let a = raw[start % n];
    let b = raw[(start + steps) % n];
    let ab = b.sub(a);
    let len = ab.len();
    if len < 1.5 {
        return 0.0;
    }
    let ux = ab.x / len;
    let uy = ab.y / len;
    let mut max_p = 0.0f32;
    for s in 1..steps {
        let p = raw[(start + s) % n];
        let ap = p.sub(a);
        let perp = (ap.x * uy - ap.y * ux).abs();
        if perp > max_p {
            max_p = perp;
        }
    }
    max_p
}

/// Detect genuine structural straight-arm corners while keeping organic floral curves smooth C1 Beziers.
fn detect_corners_and_fair(raw: &[Pt2], corner_cos_thresh: f32) -> (Vec<Pt2>, Vec<bool>) {
    let n = raw.len();
    let k = (n / 8).clamp(4, 10);
    let mut pinned = vec![false; n];
    // Never pin staircase corners on small organic contours (< 44 vertices); keep them smooth C1 curves
    if n >= 44 {
        for i in 0..n {
            let p_prev = raw[(i + n - k) % n];
            let p_cur = raw[i];
            let p_next = raw[(i + k) % n];
            let d1 = p_cur.sub(p_prev).norm();
            let d2 = p_next.sub(p_cur).norm();
            let cs = d1.dot(d2);
            if cs < corner_cos_thresh && k >= 5 {
                let perp_in = arm_max_perp(raw, (i + n - k) % n, k, n);
                let perp_out = arm_max_perp(raw, i, k, n);
                if perp_in <= 0.85 && perp_out <= 0.85 {
                    let s1 = p_cur.sub(raw[(i + n - 2) % n]).norm();
                    let s2 = raw[(i + 2) % n].sub(p_cur).norm();
                    if s1.dot(s2) < 0.78 {
                        pinned[i] = true;
                    }
                }
            }
        }
        // Keep strongest corner inside +/-2 neighborhood
        let mut best_pinned = vec![false; n];
        for i in 0..n {
            if !pinned[i] {
                continue;
            }
            let sc_i = raw[i]
                .sub(raw[(i + n - k) % n])
                .norm()
                .dot(raw[(i + k) % n].sub(raw[i]).norm());
            let mut is_local_min_cos = true;
            for off in 1..=2 {
                let l = (i + n - off) % n;
                let r = (i + off) % n;
                if pinned[l] {
                    let sc_l = raw[l]
                        .sub(raw[(l + n - k) % n])
                        .norm()
                        .dot(raw[(l + k) % n].sub(raw[l]).norm());
                    if sc_l < sc_i || (sc_l == sc_i && l < i) {
                        is_local_min_cos = false;
                        break;
                    }
                }
                if pinned[r] {
                    let sc_r = raw[r]
                        .sub(raw[(r + n - k) % n])
                        .norm()
                        .dot(raw[(r + k) % n].sub(raw[r]).norm());
                    if sc_r < sc_i {
                        is_local_min_cos = false;
                        break;
                    }
                }
            }
            if is_local_min_cos {
                best_pinned[i] = true;
            }
        }
        pinned = best_pinned;
    }

    // Three-pass corner-pinned Taubin smoothing melts pixel jitter while keeping straight corners locked
    let mut pts = raw.to_vec();
    let mut tmp = pts.clone();
    for _pass in 0..3 {
        for (lambda, mu) in [(0.42f32, -0.44f32)] {
            for i in 0..n {
                if pinned[i] {
                    tmp[i] = pts[i];
                } else {
                    let avg = pts[(i + n - 1) % n].add(pts[(i + 1) % n]).mul(0.5);
                    tmp[i] = pts[i].add(avg.sub(pts[i]).mul(lambda));
                }
            }
            for i in 0..n {
                if pinned[i] {
                    pts[i] = tmp[i];
                } else {
                    let avg = tmp[(i + n - 1) % n].add(tmp[(i + 1) % n]).mul(0.5);
                    pts[i] = tmp[i].add(avg.sub(tmp[i]).mul(mu));
                }
            }
        }
    }
    (pts, pinned)
}

#[derive(Clone, Copy)]
struct CubicSeg {
    p0: Pt2,
    c1: Pt2,
    c2: Pt2,
    p3: Pt2,
    is_line: bool,
}

#[inline]
fn eval_cubic(s: &CubicSeg, t: f32) -> Pt2 {
    let mt = 1.0 - t;
    let b0 = mt * mt * mt;
    let b1 = 3.0 * mt * mt * t;
    let b2 = 3.0 * mt * t * t;
    let b3 = t * t * t;
    Pt2::new(
        b0 * s.p0.x + b1 * s.c1.x + b2 * s.c2.x + b3 * s.p3.x,
        b0 * s.p0.y + b1 * s.c1.y + b2 * s.c2.y + b3 * s.p3.y,
    )
}

#[inline]
fn eval_cubic_d1(s: &CubicSeg, t: f32) -> Pt2 {
    let mt = 1.0 - t;
    let d0 = s.c1.sub(s.p0).mul(3.0 * mt * mt);
    let d1 = s.c2.sub(s.c1).mul(6.0 * mt * t);
    let d2 = s.p3.sub(s.c2).mul(3.0 * t * t);
    d0.add(d1).add(d2)
}

#[inline]
fn eval_cubic_d2(s: &CubicSeg, t: f32) -> Pt2 {
    let a = s.c2.sub(s.c1.mul(2.0)).add(s.p0).mul(6.0 * (1.0 - t));
    let b = s.p3.sub(s.c2.mul(2.0)).add(s.c1).mul(6.0 * t);
    a.add(b)
}

fn generate_cubic(pts: &[Pt2], u: &[f32], t_hat1: Pt2, t_hat2: Pt2) -> CubicSeg {
    let n = pts.len();
    let p0 = pts[0];
    let p3 = pts[n - 1];
    let mut c00 = 0.0f64;
    let mut c01 = 0.0f64;
    let mut c11 = 0.0f64;
    let mut x0 = 0.0f64;
    let mut x1 = 0.0f64;

    for i in 0..n {
        let t = u[i];
        let mt = 1.0 - t;
        let b0 = mt * mt * mt;
        let b1 = 3.0 * mt * mt * t;
        let b2 = 3.0 * mt * t * t;
        let b3 = t * t * t;
        let a1 = t_hat1.mul(b1);
        let a2 = t_hat2.mul(b2);
        c00 += a1.dot(a1) as f64;
        c01 += a1.dot(a2) as f64;
        c11 += a2.dot(a2) as f64;
        let tmp = pts[i].sub(p0.mul(b0 + b1).add(p3.mul(b2 + b3)));
        x0 += tmp.dot(a1) as f64;
        x1 += tmp.dot(a2) as f64;
    }

    let det = c00 * c11 - c01 * c01;
    let mut alpha_l = 0.0f32;
    let mut alpha_r = 0.0f32;
    let chord = p3.sub(p0).len();
    if det.abs() > 1e-6 {
        alpha_l = ((c11 * x0 - c01 * x1) / det) as f32;
        alpha_r = ((c00 * x1 - c01 * x0) / det) as f32;
    }
    let eps = 1e-5 * chord.max(1e-3);
    if alpha_l < eps || alpha_r < eps || alpha_l > chord * 2.2 || alpha_r > chord * 2.2 {
        let dist = chord / 3.0;
        alpha_l = dist;
        alpha_r = dist;
    }
    CubicSeg {
        p0,
        c1: p0.add(t_hat1.mul(alpha_l)),
        c2: p3.add(t_hat2.mul(alpha_r)),
        p3,
        is_line: false,
    }
}

fn max_cubic_error(pts: &[Pt2], seg: &CubicSeg, u: &[f32]) -> (f32, usize) {
    let mut max_d_sq = 0.0f32;
    let mut split_idx = pts.len() / 2;
    for i in 1..(pts.len() - 1) {
        let d_sq = eval_cubic(seg, u[i]).sub(pts[i]).len_sq();
        if d_sq >= max_d_sq {
            max_d_sq = d_sq;
            split_idx = i;
        }
    }
    (max_d_sq, split_idx)
}

fn chord_len_params(pts: &[Pt2]) -> Vec<f32> {
    let mut u = vec![0.0f32; pts.len()];
    for i in 1..pts.len() {
        u[i] = u[i - 1] + pts[i].sub(pts[i - 1]).len();
    }
    let total = u[pts.len() - 1].max(1e-6);
    for v in &mut u {
        *v /= total;
    }
    u
}

fn check_straight_run(pts: &[Pt2], straight_tol: f32) -> bool {
    let n = pts.len();
    if n <= 2 {
        return true;
    }
    let a = pts[0];
    let b = pts[n - 1];
    let ab = b.sub(a);
    let len_sq = ab.len_sq();
    if len_sq < 1e-6 {
        return false;
    }
    let len = len_sq.sqrt();
    let ux = ab.x / len;
    let uy = ab.y / len;
    let mut prev_proj = -0.5f32;
    for p in &pts[1..(n - 1)] {
        let ap = p.sub(a);
        let perp = (ap.x * uy - ap.y * ux).abs();
        if perp > straight_tol {
            return false;
        }
        let proj = ap.x * ux + ap.y * uy;
        if proj < prev_proj - straight_tol || proj > len + straight_tol {
            return false;
        }
        prev_proj = prev_proj.max(proj);
    }
    true
}

/// Philip Schneider (1990) Recursive Least-Squares Cubic Bézier Curve Fitter with straight-run CAD snapping.
fn fit_span_schneider(
    pts: &[Pt2],
    t_hat1: Pt2,
    t_hat2: Pt2,
    err_sq: f32,
    straight_tol: f32,
    depth: u8,
    out: &mut Vec<CubicSeg>,
) {
    let n = pts.len();
    if n < 2 {
        return;
    }
    let chord_len = pts[n - 1].sub(pts[0]).len();
    // Collapse genuine straight CAD edges >= 9px & >= 6 samples so small organic curves always remain smooth C1 Beziers
    if n >= 6 && chord_len >= 9.0 && check_straight_run(pts, straight_tol) {
        let p0 = pts[0];
        let p3 = pts[n - 1];
        out.push(CubicSeg {
            p0,
            c1: p0,
            c2: p3,
            p3,
            is_line: true,
        });
        return;
    }

    let mut u = chord_len_params(pts);
    let mut seg = generate_cubic(pts, &u, t_hat1, t_hat2);
    let (mut max_d_sq, mut split_idx) = max_cubic_error(pts, &seg, &u);
    if max_d_sq <= err_sq {
        out.push(seg);
        return;
    }

    if max_d_sq <= err_sq * 4.5 && n <= 256 {
        for _ in 0..3 {
            for i in 1..(n - 1) {
                let q = eval_cubic(&seg, u[i]).sub(pts[i]);
                let q1 = eval_cubic_d1(&seg, u[i]);
                let q2 = eval_cubic_d2(&seg, u[i]);
                let denom = q1.dot(q1) + q.dot(q2);
                if denom.abs() > 1e-6 {
                    u[i] = (u[i] - q.dot(q1) / denom).clamp(0.0, 1.0);
                }
            }
            seg = generate_cubic(pts, &u, t_hat1, t_hat2);
            let (e2, sp2) = max_cubic_error(pts, &seg, &u);
            max_d_sq = e2;
            split_idx = sp2;
            if max_d_sq <= err_sq {
                out.push(seg);
                return;
            }
        }
    }

    if depth >= 14 || n <= 3 {
        out.push(seg);
        return;
    }

    split_idx = split_idx.clamp(1, n - 2);
    let t_center = pts[(split_idx + 1).min(n - 1)]
        .sub(pts[split_idx.saturating_sub(1)])
        .norm();
    let t_split_r = if t_center.len_sq() > 0.25 {
        t_center.mul(-1.0)
    } else {
        t_hat2
    };
    let t_split_l = if t_center.len_sq() > 0.25 {
        t_center
    } else {
        t_hat1
    };

    fit_span_schneider(&pts[..=split_idx], t_hat1, t_split_r, err_sq, straight_tol, depth + 1, out);
    fit_span_schneider(&pts[split_idx..], t_split_l, t_hat2, err_sq, straight_tol, depth + 1, out);
}

fn fit_loop_to_cad_nodes(
    pts: &[Pt2],
    pinned: &[bool],
    err_px: f32,
    straight_tol: f32,
    off_x: f32,
    off_y: f32,
    sx_mm: f32,
    sy_mm: f32,
) -> Vec<CadNode> {
    let n = pts.len();
    let mut corners: Vec<usize> = pinned
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| if p { Some(i) } else { None })
        .collect();

    // Split smooth cornerless loops cleanly at cardinal extrema (min_x, min_y, max_x, max_y)
    if corners.is_empty() && n >= 8 {
        let mut idx_min_x = 0usize;
        let mut idx_max_x = 0usize;
        let mut idx_min_y = 0usize;
        let mut idx_max_y = 0usize;
        for (i, &p) in pts.iter().enumerate() {
            if p.x < pts[idx_min_x].x { idx_min_x = i; }
            if p.x > pts[idx_max_x].x { idx_max_x = i; }
            if p.y < pts[idx_min_y].y { idx_min_y = i; }
            if p.y > pts[idx_max_y].y { idx_max_y = i; }
        }
        let mut cand = [idx_min_x, idx_max_y, idx_max_x, idx_min_y];
        cand.sort_unstable();
        let min_sep = (n / 12).max(2);
        for c in cand {
            if corners.last().map_or(true, |&prev| c.saturating_sub(prev) >= min_sep) {
                if corners.first().map_or(true, |&first| (n - c + first) >= min_sep) {
                    corners.push(c);
                }
            }
        }
    }
    if corners.is_empty() {
        let step = (n / 4).max(1);
        corners.push(0);
        if n >= 8 {
            corners.push(step);
            corners.push(step * 2);
            corners.push((step * 3).min(n - 1));
        } else if n >= 4 {
            corners.push(n / 2);
        }
    }

    let err_sq = err_px * err_px;
    let mut cubics: Vec<(CubicSeg, bool, bool)> = Vec::new();
    let m_c = corners.len();
    for ci in 0..m_c {
        let s_idx = corners[ci];
        let e_idx = corners[(ci + 1) % m_c];
        let mut span = Vec::new();
        let mut idx = s_idx;
        span.push(pts[idx]);
        loop {
            idx = (idx + 1) % n;
            span.push(pts[idx]);
            if idx == e_idx || span.len() > n + 1 {
                break;
            }
        }
        let s_pin = pinned[s_idx];
        let e_pin = pinned[e_idx];
        let k_tan = (span.len() / 4).clamp(1, 3);
        let t1 = if s_pin {
            span[1].sub(span[0]).norm()
        } else {
            let p_prev = pts[(s_idx + n - k_tan) % n];
            span[k_tan.min(span.len() - 1)].sub(p_prev).norm()
        };
        let t2 = if e_pin {
            span[span.len() - 2].sub(span[span.len() - 1]).norm()
        } else {
            let p_next = pts[(e_idx + k_tan) % n];
            span[span.len().saturating_sub(1 + k_tan)].sub(p_next).norm()
        };

        let mut span_cubics = Vec::new();
        fit_span_schneider(&span, t1, t2, err_sq, straight_tol, 0, &mut span_cubics);
        let sc_len = span_cubics.len();
        for (si, seg) in span_cubics.into_iter().enumerate() {
            cubics.push((seg, si == 0 && s_pin, si + 1 == sc_len && e_pin));
        }
    }

    let mut nodes = Vec::with_capacity(cubics.len());
    let num_c = cubics.len();
    if num_c == 0 {
        return nodes;
    }

    let to_mm = |p: Pt2| -> (f32, f32) { (off_x + p.x * sx_mm, off_y + p.y * sy_mm) };
    for i in 0..num_c {
        let prev = &cubics[(i + num_c - 1) % num_c].0;
        let (cur, is_start_pin, _) = &cubics[i];
        let (x, y) = to_mm(cur.p0);
        let has_in = !prev.is_line;
        let has_out = !cur.is_line;
        let is_corner = *is_start_pin || prev.is_line != cur.is_line || (prev.is_line && cur.is_line);
        let (hix, hiy) = if has_in { to_mm(prev.c2) } else { (x, y) };
        let (hox, hoy) = if has_out { to_mm(cur.c1) } else { (x, y) };
        let mut flags = 0u32;
        if is_corner {
            flags |= FLAG_CORNER;
        }
        if has_in {
            flags |= FLAG_HAS_IN;
        }
        if has_out {
            flags |= FLAG_HAS_OUT;
        }
        nodes.push(CadNode {
            x,
            y,
            h_in_x: hix,
            h_in_y: hiy,
            h_out_x: hox,
            h_out_y: hoy,
            flags,
        });
    }
    nodes
}

fn run_vectorize_pipeline(
    field: &[u8],
    w: usize,
    h: usize,
    thresh: f32,
    detail: f32,
    corners: f32,
    min_area_px: f32,
    straight_tol: f32,
    off_x: f32,
    off_y: f32,
    sx_mm: f32,
    sy_mm: f32,
) -> Vec<CadRing> {
    let raw_loops = marching_squares_contours(field, w, h, thresh);
    // Turning angle threshold for straight-arm corners (58 deg turn = 122 deg interior angle at corners=0.65)
    let turn_deg = 70.0 - corners.clamp(0.0, 1.0) * 18.0;
    let corner_cos = turn_deg.to_radians().cos();
    let err_px = (1.65 - detail.clamp(0.0, 1.0) * 1.25).max(0.32);

    struct PreLoop {
        abs_area: f32,
        bbox: [f32; 4],
        pts: Vec<Pt2>,
        pinned: Vec<bool>,
    }

    let mut outers: Vec<PreLoop> = Vec::new();
    let mut holes: Vec<PreLoop> = Vec::new();
    for lp in raw_loops {
        let sa = signed_area(&lp);
        let aa = sa.abs();
        // With foreground on the right in Y-down screen coords, outer loops have sa > 0 and holes sa < 0.
        let is_outer = sa > 0.0;
        if aa < if is_outer { min_area_px } else { (min_area_px * 0.55).max(4.0) } {
            continue;
        }
        let mut minx = f32::MAX;
        let mut miny = f32::MAX;
        let mut maxx = f32::MIN;
        let mut maxy = f32::MIN;
        for &p in &lp {
            minx = minx.min(p.x);
            miny = miny.min(p.y);
            maxx = maxx.max(p.x);
            maxy = maxy.max(p.y);
        }
        let (pts, pinned) = detect_corners_and_fair(&lp, corner_cos);
        let item = PreLoop {
            abs_area: aa,
            bbox: [minx, miny, maxx, maxy],
            pts,
            pinned,
        };
        if is_outer {
            outers.push(item);
        } else {
            holes.push(item);
        }
    }

    // Sort outer loops ascending by area so each hole binds to its tightest enclosing outer boundary
    outers.sort_by(|a, b| a.abs_area.partial_cmp(&b.abs_area).unwrap_or(std::cmp::Ordering::Equal));
    let mut hole_parents = vec![-1i32; holes.len()];
    for (hi, h_lp) in holes.iter().enumerate() {
        let probe = h_lp.pts[0];
        for (oi, o_lp) in outers.iter().enumerate() {
            if o_lp.abs_area <= h_lp.abs_area {
                continue;
            }
            if probe.x < o_lp.bbox[0]
                || probe.x > o_lp.bbox[2]
                || probe.y < o_lp.bbox[1]
                || probe.y > o_lp.bbox[3]
            {
                continue;
            }
            if point_in_poly(probe, &o_lp.pts) {
                hole_parents[hi] = oi as i32;
                break;
            }
        }
    }

    // Multi-core curve fitting across all rings
    let n_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8);

    enum RingSpec<'a> {
        Outer(&'a PreLoop),
        Hole(i32, &'a PreLoop),
    }

    let mut specs = Vec::with_capacity(outers.len() + holes.len());
    for o in &outers {
        specs.push(RingSpec::Outer(o));
    }
    for (hi, h_lp) in holes.iter().enumerate() {
        if hole_parents[hi] >= 0 {
            specs.push(RingSpec::Hole(hole_parents[hi], h_lp));
        }
    }

    let mut results: Vec<Option<CadRing>> = (0..specs.len()).map(|_| None).collect();
    if specs.len() < 16 || n_threads <= 1 {
        for (i, sp) in specs.iter().enumerate() {
            let (is_hole, parent_idx, lp) = match sp {
                RingSpec::Outer(o) => (false, -1, *o),
                RingSpec::Hole(pid, hl) => (true, *pid, *hl),
            };
            let nodes = fit_loop_to_cad_nodes(
                &lp.pts,
                &lp.pinned,
                err_px,
                straight_tol,
                off_x,
                off_y,
                sx_mm,
                sy_mm,
            );
            if nodes.len() >= 3 {
                results[i] = Some(CadRing {
                    is_hole,
                    parent_idx,
                    nodes,
                });
            }
        }
    } else {
        let chunk_sz = (specs.len() + n_threads - 1) / n_threads;
        thread::scope(|sc| {
            for (chunk_sp, chunk_res) in specs.chunks(chunk_sz).zip(results.chunks_mut(chunk_sz)) {
                sc.spawn(move || {
                    for (sp, dst) in chunk_sp.iter().zip(chunk_res.iter_mut()) {
                        let (is_hole, parent_idx, lp) = match sp {
                            RingSpec::Outer(o) => (false, -1, *o),
                            RingSpec::Hole(pid, hl) => (true, *pid, *hl),
                        };
                        let nodes = fit_loop_to_cad_nodes(
                            &lp.pts,
                            &lp.pinned,
                            err_px,
                            straight_tol,
                            off_x,
                            off_y,
                            sx_mm,
                            sy_mm,
                        );
                        if nodes.len() >= 3 {
                            *dst = Some(CadRing {
                                is_hole,
                                parent_idx,
                                nodes,
                            });
                        }
                    }
                });
            }
        });
    }

    // Re-index outer parent indices if any outer degenerated below 3 nodes (rare)
    let mut final_rings = Vec::with_capacity(results.len());
    let mut old_to_new = vec![-1i32; outers.len()];
    for i in 0..outers.len() {
        if let Some(r) = results[i].take() {
            old_to_new[i] = final_rings.len() as i32;
            final_rings.push(r);
        }
    }
    for i in outers.len()..results.len() {
        if let Some(mut r) = results[i].take() {
            let np = old_to_new[r.parent_idx as usize];
            if np >= 0 {
                r.parent_idx = np;
                final_rings.push(r);
            }
        }
    }
    final_rings
}

fn downsample_2x(src: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let nw = (w / 2).max(2);
    let nh = (h / 2).max(2);
    let mut dst = vec![0.0f32; nw * nh];
    for y in 0..nh {
        let sy = (y * 2).min(h - 1);
        let sy1 = (sy + 1).min(h - 1);
        for x in 0..nw {
            let sx = (x * 2).min(w - 1);
            let sx1 = (sx + 1).min(w - 1);
            dst[y * nw + x] = 0.25 * (src[sy * w + sx] + src[sy * w + sx1] + src[sy1 * w + sx] + src[sy1 * w + sx1]);
        }
    }
    (dst, nw, nh)
}

fn upsample_bilinear(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; dw * dh];
    let sx_scale = (sw as f32) / (dw as f32);
    let sy_scale = (sh as f32) / (dh as f32);
    for y in 0..dh {
        let fy = ((y as f32 + 0.5) * sy_scale - 0.5).clamp(0.0, (sh - 1) as f32);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let wy = fy - (y0 as f32);
        for x in 0..dw {
            let fx = ((x as f32 + 0.5) * sx_scale - 0.5).clamp(0.0, (sw - 1) as f32);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let wx = fx - (x0 as f32);
            let v00 = src[y0 * sw + x0];
            let v10 = src[y0 * sw + x1];
            let v01 = src[y1 * sw + x0];
            let v11 = src[y1 * sw + x1];
            dst[y * dw + x] = (v00 * (1.0 - wx) + v10 * wx) * (1.0 - wy) + (v01 * (1.0 - wx) + v11 * wx) * wy;
        }
    }
    dst
}

fn parallel_jacobi_relax(z: &mut Vec<f32>, rhs: &[f32], w: usize, h: usize, lambda: f32, iters: usize, omega: f32) {
    let inv_diag = omega / (4.0 + lambda);
    let one_m_w = 1.0 - omega;
    let mut z_next = z.clone();
    let n_threads = thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(4)
        .min(10);

    for _ in 0..iters {
        // Keep L2-resident grids on fast single-thread SIMD slices (avoids pthread respawn overhead per sweep)
        if w * h < 450_000 || n_threads <= 1 {
            for y in 0..h {
                let ym = y.saturating_sub(1);
                let yp = (y + 1).min(h - 1);
                let row = y * w;
                let row_m = ym * w;
                let row_p = yp * w;
                if w >= 2 {
                    let s0 = z[row] + z[row + 1] + z[row_m] + z[row_p];
                    z_next[row] = one_m_w * z[row] + (s0 + rhs[row]) * inv_diag;
                    for x in 1..(w - 1) {
                        let sum_n = z[row + x - 1] + z[row + x + 1] + z[row_m + x] + z[row_p + x];
                        z_next[row + x] = one_m_w * z[row + x] + (sum_n + rhs[row + x]) * inv_diag;
                    }
                    let xl = w - 1;
                    let sl = z[row + xl - 1] + z[row + xl] + z[row_m + xl] + z[row_p + xl];
                    z_next[row + xl] = one_m_w * z[row + xl] + (sl + rhs[row + xl]) * inv_diag;
                }
            }
        } else {
            let z_ref: &[f32] = z.as_slice();
            let rows_per = (h + n_threads - 1) / n_threads;
            let chunk_elems = rows_per * w;
            thread::scope(|sc| {
                for (band_idx, dst_band) in z_next.chunks_mut(chunk_elems).enumerate() {
                    let y_start = band_idx * rows_per;
                    let band_rows = dst_band.len() / w;
                    sc.spawn(move || {
                        for dy in 0..band_rows {
                            let y = y_start + dy;
                            let ym = y.saturating_sub(1);
                            let yp = (y + 1).min(h - 1);
                            let row = y * w;
                            let row_m = ym * w;
                            let row_p = yp * w;
                            let dst_row = &mut dst_band[dy * w..(dy + 1) * w];
                            // Interior SIMD-friendly slice path
                            if w >= 2 {
                                let sum_0 = z_ref[row] + z_ref[row + 1] + z_ref[row_m] + z_ref[row_p];
                                dst_row[0] = one_m_w * z_ref[row] + (sum_0 + rhs[row]) * inv_diag;
                                for x in 1..(w - 1) {
                                    let sum_n = z_ref[row + x - 1] + z_ref[row + x + 1] + z_ref[row_m + x] + z_ref[row_p + x];
                                    dst_row[x] = one_m_w * z_ref[row + x] + (sum_n + rhs[row + x]) * inv_diag;
                                }
                                let xl = w - 1;
                                let sum_l = z_ref[row + xl - 1] + z_ref[row + xl] + z_ref[row_m + xl] + z_ref[row_p + xl];
                                dst_row[xl] = one_m_w * z_ref[row + xl] + (sum_l + rhs[row + xl]) * inv_diag;
                            }
                        }
                    });
                }
            });
        }
        std::mem::swap(z, &mut z_next);
    }
}

/// 4-Level Coarse-to-Fine Multigrid Screened Poisson bas-relief solver with atan gradient compression.
fn solve_screened_poisson(
    p: &[f32],
    q: &[f32],
    z_macro: &[f32],
    grain_gate: &[f32],
    w: usize,
    h: usize,
    lambda: f32,
    iters: usize,
    grad_knee: f32,
    grad_scale: f32,
) -> Vec<f32> {
    let n = w * h;
    let mut pc = vec![0.0f32; n];
    let mut qc = vec![0.0f32; n];

    // Step 1: Compress extreme slopes + gate out flat background grain across CPU threads
    let n_threads = thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(4)
        .min(10);
    let chunk = (n + n_threads - 1) / n_threads;
    let knee = grad_knee.max(1e-4);
    thread::scope(|sc| {
        for ((p_ch, q_ch), ((gg_ch, pc_ch), qc_ch)) in p
            .chunks(chunk)
            .zip(q.chunks(chunk))
            .zip(grain_gate.chunks(chunk).zip(pc.chunks_mut(chunk)).zip(qc.chunks_mut(chunk)))
        {
            sc.spawn(move || {
                for i in 0..p_ch.len() {
                    let px = p_ch[i];
                    let qy = q_ch[i];
                    let mag = (px * px + qy * qy).sqrt();
                    if mag > 1e-6 {
                        let scale = gg_ch[i] * grad_scale * knee * (mag / knee).atan() / mag;
                        pc_ch[i] = px * scale;
                        qc_ch[i] = qy * scale;
                    }
                }
            });
        }
    });

    // Step 2: Zero-phase centered divergence: rhs = lambda * Z_macro - div(pc, qc)
    let pc_ref: &[f32] = &pc;
    let qc_ref: &[f32] = &qc;
    let mut rhs0 = vec![0.0f32; n];
    let rows_per = (h + n_threads - 1) / n_threads;
    thread::scope(|sc| {
        for (bi, rhs_band) in rhs0.chunks_mut(rows_per * w).enumerate() {
            let y_start = bi * rows_per;
            let b_rows = rhs_band.len() / w;
            sc.spawn(move || {
                for dy in 0..b_rows {
                    let y = y_start + dy;
                    let ym = y.saturating_sub(1);
                    let yp = (y + 1).min(h - 1);
                    let row = y * w;
                    let out_r = &mut rhs_band[dy * w..(dy + 1) * w];
                    for x in 0..w {
                        let xm = x.saturating_sub(1);
                        let xp = (x + 1).min(w - 1);
                        let dx_p = 0.5 * (pc_ref[row + xp] - pc_ref[row + xm]);
                        let dy_q = 0.5 * (qc_ref[yp * w + x] - qc_ref[ym * w + x]);
                        out_r[x] = lambda * z_macro[row + x] - (dx_p + dy_q);
                    }
                }
            });
        }
    });

    // Step 3: 4-level Coarse-to-Fine Multigrid warm-start + parallel Jacobi relaxation
    if w >= 128 && h >= 128 {
        let (rhs1, w1, h1) = downsample_2x(&rhs0, w, h);
        let rhs1_s: Vec<f32> = rhs1.iter().map(|&v| v * 4.0).collect();
        let (rhs2, w2, h2) = downsample_2x(&rhs1, w1, h1);
        let rhs2_s: Vec<f32> = rhs2.iter().map(|&v| v * 16.0).collect();
        let (rhs3, w3, h3) = downsample_2x(&rhs2, w2, h2);
        let rhs3_s: Vec<f32> = rhs3.iter().map(|&v| v * 64.0).collect();

        let (mac1, _, _) = downsample_2x(z_macro, w, h);
        let (mac2, _, _) = downsample_2x(&mac1, w1, h1);
        let (mut z3, _, _) = downsample_2x(&mac2, w2, h2);
        parallel_jacobi_relax(&mut z3, &rhs3_s, w3, h3, lambda * 64.0, iters.max(48), 0.90);

        let mut z2 = upsample_bilinear(&z3, w3, h3, w2, h2);
        parallel_jacobi_relax(&mut z2, &rhs2_s, w2, h2, lambda * 16.0, iters.max(32), 0.90);

        let mut z1 = upsample_bilinear(&z2, w2, h2, w1, h1);
        parallel_jacobi_relax(&mut z1, &rhs1_s, w1, h1, lambda * 4.0, iters.max(22), 0.90);

        let mut z0 = upsample_bilinear(&z1, w1, h1, w, h);
        parallel_jacobi_relax(&mut z0, &rhs0, w, h, lambda, iters.max(16), 0.90);
        z0
    } else if w >= 64 && h >= 64 {
        let (rhs1, w1, h1) = downsample_2x(&rhs0, w, h);
        let rhs1_s: Vec<f32> = rhs1.iter().map(|&v| v * 4.0).collect();
        let (mut z1, _, _) = downsample_2x(z_macro, w, h);
        parallel_jacobi_relax(&mut z1, &rhs1_s, w1, h1, lambda * 4.0, iters.max(32), 0.90);
        let mut z0 = upsample_bilinear(&z1, w1, h1, w, h);
        parallel_jacobi_relax(&mut z0, &rhs0, w, h, lambda, iters.max(18), 0.90);
        z0
    } else {
        let mut z0 = z_macro.to_vec();
        parallel_jacobi_relax(&mut z0, &rhs0, w, h, lambda, iters.max(40), 0.90);
        z0
    }
}

fn handle_vectorize_frame<R: Read, W: Write>(r: &mut R, out: &mut W) -> io::Result<()> {
    let w = read_u32_le(r)? as usize;
    let h = read_u32_le(r)? as usize;
    let thresh = read_f32_le(r)?;
    let detail = read_f32_le(r)?;
    let corners = read_f32_le(r)?;
    let min_area_px = read_f32_le(r)?;
    let straight_tol = read_f32_le(r)?;
    let off_x = read_f32_le(r)?;
    let off_y = read_f32_le(r)?;
    let sx_mm = read_f32_le(r)?;
    let sy_mm = read_f32_le(r)?;

    let mut field = vec![0u8; w * h];
    r.read_exact(&mut field)?;

    let rings = run_vectorize_pipeline(
        &field,
        w,
        h,
        thresh,
        detail,
        corners,
        min_area_px,
        straight_tol,
        off_x,
        off_y,
        sx_mm,
        sy_mm,
    );

    let mut body: Vec<u8> = Vec::with_capacity(4 + rings.len() * 256);
    body.extend_from_slice(&(rings.len() as u32).to_le_bytes());
    for ring in &rings {
        body.push(if ring.is_hole { 1u8 } else { 0u8 });
        body.extend_from_slice(&ring.parent_idx.to_le_bytes());
        body.extend_from_slice(&(ring.nodes.len() as u32).to_le_bytes());
        for nd in &ring.nodes {
            body.extend_from_slice(&nd.x.to_le_bytes());
            body.extend_from_slice(&nd.y.to_le_bytes());
            body.extend_from_slice(&nd.h_in_x.to_le_bytes());
            body.extend_from_slice(&nd.h_in_y.to_le_bytes());
            body.extend_from_slice(&nd.h_out_x.to_le_bytes());
            body.extend_from_slice(&nd.h_out_y.to_le_bytes());
            body.extend_from_slice(&nd.flags.to_le_bytes());
        }
    }
    out.write_all(b"JDR2")?;
    out.write_all(&(body.len() as u32).to_le_bytes())?;
    out.write_all(&body)?;
    out.flush()
}

fn handle_poisson_frame<R: Read, W: Write>(r: &mut R, out: &mut W) -> io::Result<()> {
    let w = read_u32_le(r)? as usize;
    let h = read_u32_le(r)? as usize;
    let lambda = read_f32_le(r)?;
    let iters = read_u32_le(r)? as usize;
    let grad_knee = read_f32_le(r)?;
    let grad_scale = read_f32_le(r)?;

    let n = w * h;
    let read_vec_f32 = |rdr: &mut R| -> io::Result<Vec<f32>> {
        let mut raw = vec![0u8; n * 4];
        rdr.read_exact(&mut raw)?;
        let mut out_v = Vec::with_capacity(n);
        for chunk in raw.chunks_exact(4) {
            out_v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out_v)
    };

    let p = read_vec_f32(r)?;
    let q = read_vec_f32(r)?;
    let z_macro = read_vec_f32(r)?;
    let grain_gate = read_vec_f32(r)?;

    let res = solve_screened_poisson(&p, &q, &z_macro, &grain_gate, w, h, lambda, iters, grad_knee, grad_scale);

    out.write_all(b"JDP2")?;
    for v in res {
        out.write_all(&v.to_le_bytes())?;
    }
    out.flush()
}

fn handle_vectorize() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut r = BufReader::new(stdin.lock());
    let mut out = BufWriter::new(stdout.lock());
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"JDV2" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic JDV2"));
    }
    handle_vectorize_frame(&mut r, &mut out)
}

fn handle_poisson() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut r = BufReader::new(stdin.lock());
    let mut out = BufWriter::new(stdout.lock());
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"JDP2" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic JDP2"));
    }
    handle_poisson_frame(&mut r, &mut out)
}

fn handle_serve() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut r = BufReader::new(stdin.lock());
    let mut out = BufWriter::new(stdout.lock());
    let mut magic = [0u8; 4];
    loop {
        match r.read_exact(&mut magic) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        }
        match &magic {
            b"JDV2" => handle_vectorize_frame(&mut r, &mut out)?,
            b"JDP2" => handle_poisson_frame(&mut r, &mut out)?,
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "unknown frame magic")),
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("");
    let res = match cmd {
        "vectorize" => handle_vectorize(),
        "poisson" => handle_poisson(),
        "serve" => handle_serve(),
        _ => {
            eprintln!("Usage: jdpaint_rust_core <vectorize|poisson|serve>");
            process::exit(2);
        }
    };
    if let Err(err) = res {
        eprintln!("jdpaint_rust_core error: {err}");
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circle_cubic_precision_subpixel() {
        let w = 160usize;
        let h = 160usize;
        let cx = 80.0f32;
        let cy = 80.0f32;
        let rad = 52.0f32;
        let mut field = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
                let cov = (rad + 0.5 - d).clamp(0.0, 1.0);
                field[y * w + x] = (cov * 255.0).round() as u8;
            }
        }
        let rings = run_vectorize_pipeline(&field, w, h, 127.5, 0.78, 0.65, 25.0, 0.85, 0.0, 0.0, 1.0, 1.0);
        assert_eq!(rings.len(), 1, "expected 1 outer circle ring");
        assert!(!rings[0].is_hole);
        for nd in &rings[0].nodes {
            let r_meas = ((nd.x - cx).powi(2) + (nd.y - cy).powi(2)).sqrt();
            assert!((r_meas - rad).abs() < 0.35, "circle radius error {} vs {}", r_meas, rad);
        }
    }

    #[test]
    fn test_square_with_hole_straight_line_collapse() {
        let w = 128usize;
        let h = 128usize;
        let mut field = vec![0u8; w * h];
        for y in 20..108 {
            for x in 20..108 {
                let in_hole = (44..84).contains(&x) && (44..84).contains(&y);
                field[y * w + x] = if in_hole { 0 } else { 255 };
            }
        }
        let rings = run_vectorize_pipeline(&field, w, h, 127.5, 0.75, 0.65, 20.0, 1.25, 0.0, 0.0, 1.0, 1.0);
        assert_eq!(rings.len(), 2, "expected 1 outer square + 1 inner hole");
        assert!(!rings[0].is_hole);
        assert!(rings[1].is_hole);
        assert_eq!(rings[1].parent_idx, 0);
        assert!(rings[0].nodes.len() <= 6, "straight square collapsed cleanly: got {} nodes", rings[0].nodes.len());
    }

    #[test]
    fn test_screened_poisson_smooth_dome_and_grain_suppression() {
        let w = 96usize;
        let h = 96usize;
        let mut p = vec![0.0f32; w * h];
        let mut q = vec![0.0f32; w * h];
        let mut z_macro = vec![0.2f32; w * h];
        let mut grain_gate = vec![0.0f32; w * h];
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let dx = x as f32 - 48.0;
                let dy = y as f32 - 48.0;
                let r = (dx * dx + dy * dy).sqrt();
                if r < 24.0 {
                    p[y * w + x] = -dx / 24.0;
                    q[y * w + x] = -dy / 24.0;
                    grain_gate[y * w + x] = 1.0;
                    z_macro[y * w + x] = 0.5;
                } else {
                    // Wild background noise outside dome that grain_gate MUST suppress cleanly
                    p[y * w + x] = if (x + y) % 2 == 0 { 8.0 } else { -8.0 };
                    q[y * w + x] = if (x * 3 + y) % 2 == 0 { 6.0 } else { -6.0 };
                }
            }
        }
        let out = solve_screened_poisson(&p, &q, &z_macro, &grain_gate, w, h, 0.025, 40, 2.5, 1.0);
        let center = out[48 * w + 48];
        let corner = out[4 * w + 4];
        assert!(center > corner + 0.35, "dome center {} must exceed flat bg {}", center, corner);
        assert!((corner - 0.2).abs() < 0.05, "flat grain must stay suppressed around z_macro=0.2, got {}", corner);
    }

    #[test]
    fn test_rosette_no_square_corners_keeps_cubic() {
        let w = 240usize;
        let h = 240usize;
        let mut field = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - 120.0;
                let dy = y as f32 - 120.0;
                let ang = dy.atan2(dx);
                let r = (dx * dx + dy * dy).sqrt();
                let rad_lim = 64.0 + 22.0 * (6.0 * ang).cos();
                if r <= rad_lim {
                    field[y * w + x] = 255;
                }
            }
        }
        let rings = run_vectorize_pipeline(&field, w, h, 127.5, 0.72, 0.65, 30.0, 1.05, 0.0, 0.0, 1.0, 1.0);
        assert_eq!(rings.len(), 1);
        let smooth_handles = rings[0]
            .nodes
            .iter()
            .filter(|nd| (nd.flags & FLAG_HAS_OUT) != 0 || (nd.flags & FLAG_HAS_IN) != 0)
            .count();
        assert!(
            smooth_handles > rings[0].nodes.len() / 2,
            "curved rosette must use smooth C1 Cubic Beziers rather than straight polygon teeth: {}/{}",
            smooth_handles,
            rings[0].nodes.len()
        );
    }
}
