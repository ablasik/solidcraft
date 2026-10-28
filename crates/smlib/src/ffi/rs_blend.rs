//! Fillet, chamfer and the selection plumbing on solidcore. Edges and faces are chosen by points as
//! the C API does: the nearest within the tolerance, by more than the body's tolerance nearer than
//! any other. Not ported ([`ERR_UNSUPPORTED`]): variable radii other than a straight law. Faces
//! moved or drafted, unequal chamfers and chord blends are in `rs_ops2`.
use solidcore::blend::fillet_executive::api::{fillet_edges, fillet_edges_per_edge, variable_radius_fillet};
use solidcore::blend::fillet_executive::session::FilletSession;
use solidcore::blend::fillet_solver::p1::{FilletSurfaceGenerator, FilletSurfaceGeneratorType, MakeSurfaceBlendSih};
use solidcore::geom::object::ObjectRef;
use solidcore::geom::tools::find_face;
use solidcore::kernel::Kernel;
use solidcore::math::{Point3, Vec3};
use solidcore::query::api_queries::{edge_closest_point, surface_closest_point};
use solidcore::topology::brep::p2::is_manifold_solid;
use solidcore::topology::brep::p5::{StitchAndOrientOptions, stitch_and_orient};
use solidcore::topology::brep_cache::{CacheObject, arena_identity};
use solidcore::topology::edgeuse::p1::face as edgeuse_face;
use solidcore::topology::face::p2::surface as face_surface;
use solidcore::topology::{Brep, BrepId, EdgeId, FaceId};
use solidcore::{EntityRef, SolidError};

use super::ops::{BLEND_CHAMFER, BLEND_CHORD, BLEND_CURVATURE, BLEND_ROUND, ERR_PICK_AMBIGUOUS, ERR_PICK_NONE, ERR_UNSUPPORTED, Pick, PickError};
use super::rs::{State, named_ref, state, status};
use super::rs_ops::copy_of;
use super::rs_ops2::finish;
use super::{ERR_ARG, Handle};

/// Largest radius, distance or thickness the C API accepts (mm).
const MAX_SIZE: f64 = 1e6;

fn plain(s: i32) -> PickError {
    (s, None)
}

fn point(p: &[f64; 3]) -> Point3 {
    Point3::new(p[0], p[1], p[2])
}

/// A tolerance, every point and a size the C API would accept.
pub(super) fn check_args(points: &[[f64; 3]], tolerance: f64, sizes: &[f64]) -> Result<(), PickError> {
    let ok = tolerance.is_finite()
        && tolerance > 0.0
        && points.iter().flatten().all(|v| v.is_finite())
        && sizes.iter().all(|s| s.is_finite() && s.abs() <= MAX_SIZE);
    if ok { Ok(()) } else { Err(plain(ERR_ARG)) }
}

/// The nearest of `found` (distance, item) if within `tolerance`, else the pick error for point `i`.
fn choose<T: Copy>(mut found: Vec<(f64, T)>, i: usize, tolerance: f64, body_tolerance: f64) -> Result<T, PickError> {
    found.sort_by(|a, b| a.0.total_cmp(&b.0));
    match found.as_slice() {
        [(d, _), ..] if *d > tolerance => Err((ERR_PICK_NONE, Some(i))),
        [(d0, _), (d1, _), ..] if d1 - d0 <= body_tolerance => Err((ERR_PICK_AMBIGUOUS, Some(i))),
        [(_, item), ..] => Ok(*item),
        [] => Err((ERR_PICK_NONE, Some(i))),
    }
}

/// The distance from `p` to every edge that has one.
fn edge_distances(k: &Kernel, b: &Brep, p: &[f64; 3]) -> Vec<(f64, EdgeId)> {
    b.edge_list().iter().filter_map(|&e| Some((edge_closest_point(k, b, e, point(p)).ok()?.closest.distance, e))).collect()
}

/// The edge nearest each point, once each.
pub(super) fn pick_edges(k: &Kernel, b: &Brep, points: &[[f64; 3]], tolerance: f64) -> Result<Vec<EdgeId>, PickError> {
    let mut picked = Vec::new();
    for (i, p) in points.iter().enumerate() {
        let e = choose(edge_distances(k, b, p), i, tolerance, b.tolerance())?;
        if !picked.contains(&e) {
            picked.push(e);
        }
    }
    Ok(picked)
}

/// The identities a cache keys `b`'s edge and face owners by, in the key domain of a copy.
fn cache_key(domain: u64) -> impl Fn(CacheObject) -> ObjectRef {
    move |object| match object {
        CacheObject::Edge(e) | CacheObject::Curve(e) => named_ref(EntityRef::Edge(e), domain),
        CacheObject::Face(f) | CacheObject::Surface(f) => named_ref(EntityRef::Face(f), domain),
        CacheObject::Vertex(v) => named_ref(EntityRef::Vertex(v), domain),
        other => arena_identity(other),
    }
}

/// The face nearest each point (`SmTools::FindFace`), once each, if its surface lies within
/// `tolerance`. A point within 100 body tolerances of an edge lies on two faces: ambiguous.
///
/// ponytail: the distance is to the face's whole surface, not its trimmed part, so a point beside
/// a face but on its surface's plane passes the tolerance test.
pub(super) fn pick_faces(k: &Kernel, b: &mut Brep, domain: u64, points: &[[f64; 3]], tolerance: f64) -> Result<Vec<FaceId>, PickError> {
    let keys = cache_key(domain);
    let mut picked = Vec::new();
    for (i, p) in points.iter().enumerate() {
        let on_edge = edge_distances(k, b, p).iter().any(|&(d, _)| d <= 100.0 * b.tolerance());
        let face = find_face(k, b, BrepId::from_index(0), point(p), &keys, &|_| None).map_err(|e| plain(status(&e)))?;
        let Some(f) = face else { return Err((ERR_PICK_NONE, Some(i))) };
        let surface = face_surface(b, f).map_err(|e| plain(status(&e)))?.ok_or(plain(ERR_ARG))?;
        let away = surface_closest_point(k, surface, point(p)).map_err(|e| plain(status(&e)))?.closest.distance;
        if away > tolerance {
            return Err((ERR_PICK_NONE, Some(i)));
        }
        if on_edge {
            return Err((ERR_PICK_AMBIGUOUS, Some(i)));
        }
        if !picked.contains(&f) {
            picked.push(f);
        }
    }
    Ok(picked)
}

/// A working copy of `h` for an operation that edits its input, with its key domain.
pub(super) fn working(s: &mut State, h: &Handle) -> Result<(Brep, u64), PickError> {
    copy_of(s, h).map_err(plain)
}

/// The chord between a chamfer's two contact lines for a setback of `distance` along each face of
/// `edge`: `2 d cos(phi / 2)`, `phi` the angle between the faces' outward normals at the edge's
/// midpoint (the interior angle is `pi - phi` on a convex edge and `pi + phi` on a concave one, and
/// `sin` of its half is `cos(phi / 2)` for both). A right angle gives `d sqrt 2`.
fn chamfer_chord(k: &Kernel, b: &Brep, edge: EdgeId, distance: f64) -> Result<f64, SolidError> {
    let e = b.edge(edge)?;
    let curve = e.curve.as_ref().ok_or(SolidError::sm_err("smlib chamfer: edge without curve"))?;
    let mid = curve.evaluate_point(k, 0.5 * (e.interval.min() + e.interval.max()))?;
    let mut faces: Vec<FaceId> = Vec::new();
    for &eu in &e.edgeuses {
        if let Some(f) = edgeuse_face(b, eu)? {
            if !faces.contains(&f) {
                faces.push(f);
            }
        }
    }
    let [fa, fb] = faces[..] else { return Ok(distance * std::f64::consts::SQRT_2) };
    let outward = |f: FaceId| -> Result<Vec3, SolidError> {
        let face = b.face(f)?;
        let surface = face.surface.as_ref().ok_or(SolidError::sm_err("smlib chamfer: face without surface"))?;
        let uv = *surface_closest_point(k, surface, mid)?.uvs.first().ok_or(SolidError::sm_err("smlib chamfer: no uv"))?;
        let d = surface.evaluate(k, uv, 1, 1, Default::default())?;
        let n = d[1][0].cross(d[0][1]).unitize()?;
        // The natural normal points out of the material when the face's own side faces a void.
        let own = b.faceuse(face.faceuse.ok_or(SolidError::sm_err("smlib chamfer: face without use"))?)?;
        let shell = own.shell.ok_or(SolidError::sm_err("smlib chamfer: face use without shell"))?;
        let region = b.shell(shell)?.region.ok_or(SolidError::sm_err("smlib chamfer: shell without region"))?;
        Ok(if b.region(region)?.is_void == 1 { n } else { -n })
    };
    let cos_phi = outward(fa)?.dot(outward(fb)?).clamp(-1.0, 1.0);
    // cos(phi / 2) = sqrt((1 + cos phi) / 2)
    Ok(2.0 * distance * (0.5 * (1.0 + cos_phi)).sqrt())
}

/// SMLib's chamfer: a constant distance solver per edge with a straight cross-section and the
/// corners bevelled, stitched and oriented after. The solver's distance is the chord between the
/// two contact lines: see [`chamfer_chord`] for the chord a setback of `distance` along each face makes.
fn chamfer(k: &mut Kernel, b: &mut Brep, edges: &[EdgeId], distances: &[f64]) -> Result<(), SolidError> {
    let tolerance = b.tolerance();
    let chords = edges.iter().zip(distances).map(|(&e, &d)| chamfer_chord(k, b, e, d)).collect::<Result<Vec<_>, _>>()?;
    {
        let mut session = FilletSession::new(k, b)?;
        let generator = session.add_surface_generator(FilletSurfaceGenerator::Linear)?;
        session.set_self_intersection_handler(Some(MakeSurfaceBlendSih::new(1.0)));
        session.set_do_global_merge(false);
        for (&edge, &chord) in edges.iter().zip(&chords) {
            let edgeuse = session.blend_edgeuse(k, edge)?.ok_or(SolidError::sm_err("smlib chamfer: edge has no blendable sector"))?;
            let (angle, tangent) = (30f64.to_radians(), 2f64.to_radians());
            let solver = session.new_constant_distance_solver(k, edgeuse, chord, tolerance, angle, tangent)?;
            session.set_solver_generator(solver, generator)?;
            session.load_solver(k, solver)?;
        }
        session.create_corners(k, None)?;
        for corner in session.fillet_corners()? {
            session.set_corner_bevel(corner, true)?;
        }
        session.do_filleting(k, None)?;
    }
    stitch_and_orient(k, b, &StitchAndOrientOptions::default())
}

pub fn pick_op(kind: Pick, h: &Handle, points: &[[f64; 3]], tolerance: f64, size: f64) -> Result<Handle, PickError> {
    check_args(points, tolerance, &[size])?;
    // A round or chamfer reaches `size` out from its edge; a wall has a side.
    if size == 0.0 || (size < 0.0 && matches!(kind, Pick::Fillet | Pick::Chamfer)) {
        return Err(plain(ERR_ARG));
    }
    let mut s = state();
    let (mut b, domain) = working(&mut s, h)?;
    let solid_in = is_manifold_solid(&b).map_err(|e| plain(status(&e)))?;
    match kind {
        Pick::Fillet | Pick::Chamfer => {
            let edges = pick_edges(&s.kernel, &b, points, tolerance)?;
            match kind {
                // Cross-section 1: circular.
                Pick::Fillet => fillet_edges(&mut s.kernel, &mut b, &edges, size, 1, 1, 1.0),
                _ => chamfer(&mut s.kernel, &mut b, &edges, &vec![size; edges.len()]),
            }
            .map_err(|e| plain(status(&e)))?;
        }
        Pick::Shell => return super::rs_shell::shell(&mut s, b, domain, points, tolerance, size),
        Pick::OffsetFaces => return super::rs_ops2::offset_faces(&mut s, b, domain, points, tolerance, size),
    }
    finish(&mut s, b, solid_in)
}

pub fn blend(kind: i32, h: &Handle, points: &[[f64; 3]], sizes: &[f64], tolerance: f64) -> Result<Handle, PickError> {
    check_args(points, tolerance, sizes)?;
    if sizes.len() != points.len() || sizes.iter().any(|&v| v <= 0.0) {
        return Err(plain(ERR_ARG));
    }
    let (generator, continuity) = match kind {
        BLEND_ROUND => (Some(FilletSurfaceGeneratorType::Circular), 1),
        BLEND_CHAMFER => (None, 1),
        BLEND_CURVATURE => (Some(FilletSurfaceGeneratorType::BlendCurve), 2),
        BLEND_CHORD => (Some(FilletSurfaceGeneratorType::Circular), 1),
        _ => return Err(plain(ERR_ARG)),
    };
    let mut s = state();
    let (mut b, _) = working(&mut s, h)?;
    let solid_in = is_manifold_solid(&b).map_err(|e| plain(status(&e)))?;
    // One size per edge: a point repeating an edge keeps the first size.
    let mut edges = Vec::new();
    let mut radii = Vec::new();
    for (i, p) in points.iter().enumerate() {
        let e = pick_edges(&s.kernel, &b, std::slice::from_ref(p), tolerance).map_err(|(c, _)| (c, Some(i)))?[0];
        if !edges.contains(&e) {
            edges.push(e);
            radii.push(sizes[i]);
        }
    }
    match generator {
        Some(_) if kind == BLEND_CHORD => super::rs_ops2::chord_edges(&mut s.kernel, &mut b, &edges, &radii),
        Some(g) => fillet_edges_per_edge(&mut s.kernel, &mut b, &edges, &radii, &vec![g; edges.len()], continuity, 1.0),
        None => chamfer(&mut s.kernel, &mut b, &edges, &radii),
    }
    .map_err(|e| plain(status(&e)))?;
    finish(&mut s, b, solid_in)
}

/// Only a straight law (degree 1, two radii) is ported; `start` says which end has the first radius.
pub fn fillet_variable(
    h: &Handle,
    at: &[f64; 3],
    tolerance: f64,
    start: &[f64; 3],
    degree: i32,
    knots: &[f64],
    radii: &[f64],
) -> Result<Handle, i32> {
    check_args(std::slice::from_ref(at), tolerance, radii).map_err(|e| e.0)?;
    if !start.iter().all(|v| v.is_finite()) || radii.iter().any(|&r| r < 0.0) || radii.iter().all(|&r| r == 0.0) {
        return Err(ERR_ARG);
    }
    if degree != 1 || knots != [0.0, 0.0, 1.0, 1.0] || radii.len() != 2 {
        return Err(ERR_UNSUPPORTED);
    }
    let mut s = state();
    let (mut b, _) = working(&mut s, h).map_err(|e| e.0)?;
    let solid_in = is_manifold_solid(&b).map_err(|e| status(&e))?;
    let edge = pick_edges(&s.kernel, &b, std::slice::from_ref(at), tolerance).map_err(|e| e.0)?[0];
    // The law runs from the edge's own start; the nearer end to `start` takes the first radius.
    let ends = {
        let e = b.edge(edge).map_err(|e| status(&e))?;
        let curve = e.curve.as_ref().ok_or(super::ERR_EXCEPTION)?;
        let end = |t| curve.evaluate_point(&s.kernel, t).map_err(|e| status(&e));
        [end(e.interval.min())?, end(e.interval.max())?]
    };
    let from = point(start);
    let forward = ends[0].distance(from) <= ends[1].distance(from);
    let (first, second) = if forward { (radii[0], radii[1]) } else { (radii[1], radii[0]) };
    variable_radius_fillet(&mut s.kernel, &mut b, &[edge], first, second, 1, 1).map_err(|e| status(&e))?;
    finish(&mut s, b, solid_in).map_err(|e| e.0)
}
