// Hardware ray-query variant of the two BVH traversals in path.wgsl (local Metal prototype for #3236).
// The hardware only enumerates candidate triangles along the ray; the per-triangle test, tie-breaking
// and hit rules are copied verbatim from the software functions, which this file's `trace_*_within` replace
// (mod.rs renames the software ones to `*_sw`). A software-accepted hit is confirmed so tmax shrinks.
@group(0) @binding(7) var acc: acceleration_structure;
fn trace_precise_within(o: vec3<f32>, o_low: vec3<f32>, d: vec3<f32>, any_hit: bool, excluded: u32, minimum: f32, inside: bool, limit: f32, blocker: f32) -> Hit {
    // Dispatch once before traversal. Ordinary rays keep the original lean BVH
    // loop, without precision-helper branches inside every candidate/node test.
    if all(o_low==vec3<f32>(0.0)) { return trace_ordered_within(o,d,any_hit,excluded,minimum,inside,limit,blocker); }
    var hit = Hit(limit, 0xffffffffu, vec3<f32>(0.0));
    // Project every candidate into the same ray-aligned 2D coordinate system.
    // Shared edges then use identical products with opposite signs, avoiding the
    // independently rounded barycentric comparisons of Moller-Trumbore.
    var kz = 0u;
    if abs(d.y)>abs(d.x) { kz=1u; }
    if abs(d.z)>abs(d[kz]) { kz=2u; }
    let kx=(kz+1u)%3u; let ky=(kx+1u)%3u;
    let shear=vec3<f32>(d[kx]/d[kz],d[ky]/d[kz],1.0/d[kz]);
    let order=select(-1.0,1.0,d[kz]>0.0);
    var best=FAR;
    // Only candidates near the origin need the integer-defined deltas. The
    // rounded origin is within a few ULPs of the true anchor plus offset, so
    // boxes grown by that bound reject conservatively: every point the true
    // ray reaches at depth t lies in the grown box along the rounded ray at
    // the same t. A triangle whose bounds are farther from the rounded origin
    // than the near radius uses ordinary arithmetic from it: that error cannot
    // reorder such a hit, but it can move the ray across an edge, so decisions
    // within the error bound of an edge or bound fall back to the exact path.
    let rounded=o+o_low;
    let scale=max(max(abs(rounded.x),max(abs(rounded.y),abs(rounded.z))),1.0);
    let origin_error=scale*(1.0/1048576.0);
    let pad=u.lighting.x+origin_error;
    var source: Triangle;
    if excluded!=0xffffffffu { source=triangle(excluded); }
    let near_radius=scale*(1.0/256.0);
    var rq: ray_query;
    rayQueryInitialize(&rq, acc, RayDesc(0u, 0xffu, 0.0, min(limit, FAR), rounded, d));
    loop {
        if !rayQueryProceed(&rq) { break; }
        let i = rayQueryGetCandidateIntersection(&rq).primitive_index;
        {
            if i == excluded { continue; }
            // A straight continuation cannot revisit its producing exact plane.
            // IDs join only connected, exactly coplanar uploaded f32 triangles
            // of one body; curved facets and other bodies remain candidates.
            if excluded!=0xffffffffu && geometry[geometry[3]+i]==geometry[geometry[3]+excluded] { continue; }
            let t = triangle(i);
            let lo=min(t.p0.xyz,min(t.p1.xyz,t.p2.xyz));
            let hi=max(t.p0.xyz,max(t.p1.xyz,t.p2.xyz));
            let gap=max(lo-rounded,rounded-hi);
            if any(gap>vec3<f32>(near_radius)) {
                // The rounded origin is within origin_error of the true one on each
                // axis. Bound and edge decisions that close to their threshold could
                // flip, while a nearby neighbour across the edge is tested exactly:
                // a crossing would fall between them. Such candidates take the exact
                // path below.
                if any((rounded<=lo-vec3<f32>(origin_error))&(d<vec3<f32>(0.0))) || any((rounded>=hi+vec3<f32>(origin_error))&(d>vec3<f32>(0.0))) { continue; }
                if any((d==vec3<f32>(0.0))&((rounded<lo-vec3<f32>(origin_error))|(rounded>hi+vec3<f32>(origin_error)))) { continue; }
                let qa=t.p0.xyz-rounded; let qb=t.p1.xyz-rounded; let qc=t.p2.xyz-rounded;
                let fa=vec2<f32>(qa[kx]-shear.x*qa[kz],qa[ky]-shear.y*qa[kz]);
                let fb=vec2<f32>(qb[kx]-shear.x*qb[kz],qb[ky]-shear.y*qb[kz]);
                let fc=vec2<f32>(qc[kx]-shear.x*qc[kz],qc[ky]-shear.y*qc[kz]);
                let far_weights=vec3<f32>(difference_products(fb.x,fc.y,fb.y,fc.x),difference_products(fc.x,fa.y,fc.y,fa.x),difference_products(fa.x,fb.y,fa.y,fb.x));
                let edge_error=origin_error*vec3<f32>(length(fb-fc),length(fc-fa),length(fa-fb));
                if all(abs(far_weights)>edge_error) {
                    if any(far_weights<vec3<f32>(0.0)) && any(far_weights>vec3<f32>(0.0)) { continue; }
                    let far_det=far_weights.x+far_weights.y+far_weights.z;
                    let far_bary=far_weights/far_det;
                    let far_depth=surface_ray_depth(t,far_bary,rounded,kz);
                    let far_distance=far_depth*shear.z;
                    let far_key=far_depth*order;
                    if far_distance > minimum && far_distance < limit && (far_key < best || (far_key == best && tie_preferred(t,d,inside))) {
                        hit = Hit(far_distance,i,far_bary); best=far_key; rayQueryConfirmIntersection(&rq);
                        if any_hit || (far_distance < blocker && (t.base.w<=0.0 || t.coat.w<=0.0)) { return hit; }
                    }
                    continue;
                }
            }
            if !triangle_ahead_precise(t,o,o_low,d) { continue; }
            let pa=precise_point_delta(t.p0.xyz,vec3<f32>(0.0),o,o_low);
            let pb=precise_point_delta(t.p1.xyz,vec3<f32>(0.0),o,o_low);
            let pc=precise_point_delta(t.p2.xyz,vec3<f32>(0.0),o,o_low);
            let a=vec2<f32>(pa[kx]-shear.x*pa[kz],pa[ky]-shear.y*pa[kz]);
            let b=vec2<f32>(pb[kx]-shear.x*pb[kz],pb[ky]-shear.y*pb[kz]);
            let c=vec2<f32>(pc[kx]-shear.x*pc[kz],pc[ky]-shear.y*pc[kz]);
            let weights=vec3<f32>(difference_products(b.x,c.y,b.y,c.x),difference_products(c.x,a.y,c.y,a.x),difference_products(a.x,b.y,a.y,b.x));
            if any(weights<vec3<f32>(0.0)) && any(weights>vec3<f32>(0.0)) { continue; }
            let det=weights.x+weights.y+weights.z;
            if det==0.0 { continue; }
            // Retain all three directly evaluated weights: 1-u-v can erase a
            // small corner clearance when the other two weights sum to one.
            let bary=weights/det;
            let depth=surface_ray_depth_precise(t,bary,o,o_low,kz);
            let distance=depth*shear.z;
            let key=depth*order;
            if excluded!=0xffffffffu && repeated_crossing(t,source,d,inside,distance,origin_error) { continue; }
            if distance > minimum && distance < limit && (key < best || (key == best && tie_preferred(t,d,inside))) {
                hit = Hit(distance,i,bary); best=key; rayQueryConfirmIntersection(&rq);
                if any_hit || (distance < blocker && (t.base.w<=0.0 || t.coat.w<=0.0)) { return hit; }
            }
        }
    }
    return hit;
}
fn trace_ordered_within(o: vec3<f32>, d: vec3<f32>, any_hit: bool, excluded: u32, minimum: f32, inside: bool, limit: f32, blocker: f32) -> Hit {
    var hit = Hit(limit, 0xffffffffu, vec3<f32>(0.0));
    let origin_error=max(max(abs(o.x),max(abs(o.y),abs(o.z))),1.0)*(1.0/1048576.0);
    var source: Triangle;
    if excluded!=0xffffffffu { source=triangle(excluded); }
    // Project every candidate into the same ray-aligned 2D coordinate system.
    // Shared edges then use identical products with opposite signs, avoiding the
    // independently rounded barycentric comparisons of Moller-Trumbore.
    var kz = 0u;
    if abs(d.y)>abs(d.x) { kz=1u; }
    if abs(d.z)>abs(d[kz]) { kz=2u; }
    let kx=(kz+1u)%3u; let ky=(kx+1u)%3u;
    let shear=vec3<f32>(d[kx]/d[kz],d[ky]/d[kz],1.0/d[kz]);
    let order=select(-1.0,1.0,d[kz]>0.0);
    var best=FAR;
    var rq: ray_query;
    rayQueryInitialize(&rq, acc, RayDesc(0u, 0xffu, 0.0, min(limit, FAR), o, d));
    loop {
        if !rayQueryProceed(&rq) { break; }
        let i = rayQueryGetCandidateIntersection(&rq).primitive_index;
        {
            if i == excluded { continue; }
            // A straight continuation cannot revisit its producing exact plane.
            // IDs join only connected, exactly coplanar uploaded f32 triangles
            // of one body; curved facets and other bodies remain candidates.
            if excluded!=0xffffffffu && geometry[geometry[3]+i]==geometry[geometry[3]+excluded] { continue; }
            let t = triangle(i);
            if !triangle_ahead(t,o,d) { continue; }
            let pa=t.p0.xyz-o; let pb=t.p1.xyz-o; let pc=t.p2.xyz-o;
            let a=vec2<f32>(pa[kx]-shear.x*pa[kz],pa[ky]-shear.y*pa[kz]);
            let b=vec2<f32>(pb[kx]-shear.x*pb[kz],pb[ky]-shear.y*pb[kz]);
            let c=vec2<f32>(pc[kx]-shear.x*pc[kz],pc[ky]-shear.y*pc[kz]);
            let weights=vec3<f32>(difference_products(b.x,c.y,b.y,c.x),difference_products(c.x,a.y,c.y,a.x),difference_products(a.x,b.y,a.y,b.x));
            if any(weights<vec3<f32>(0.0)) && any(weights>vec3<f32>(0.0)) { continue; }
            let det=weights.x+weights.y+weights.z;
            if det==0.0 { continue; }
            // Retain all three directly evaluated weights: 1-u-v can erase a
            // small corner clearance when the other two weights sum to one.
            let bary=weights/det;
            let depth=surface_ray_depth(t,bary,o,kz);
            let distance=depth*shear.z;
            let key=depth*order;
            if excluded!=0xffffffffu && repeated_crossing(t,source,d,inside,distance,origin_error) { continue; }
            if distance > minimum && distance < limit && (key < best || (key == best && tie_preferred(t,d,inside))) {
                hit = Hit(distance,i,bary); best=key; rayQueryConfirmIntersection(&rq);
                if any_hit || (distance < blocker && (t.base.w<=0.0 || t.coat.w<=0.0)) { return hit; }
            }
        }
    }
    return hit;
}
