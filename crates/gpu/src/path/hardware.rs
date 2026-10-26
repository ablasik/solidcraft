//! Hardware ray queries for the path tracer (local Metal/Vulkan prototype, #3236).
//!
//! The only `unsafe` in `solidcraft-gpu`: wgpu 30 refuses every `EXPERIMENTAL_*` feature without the token
//! `ExperimentalFeatures::enabled()`, whose contract is that the caller accepts possible UB in those APIs.
//! Opt-in only: SOLIDCRAFT_RAYQUERY=1 and an adapter that reports the feature; otherwise the software BVH runs.
#![allow(unsafe_code)]
use super::{Result, gpu};
use wgpu::util::DeviceExt;

pub(super) const FEATURE: wgpu::Features = wgpu::Features::EXPERIMENTAL_RAY_QUERY;

pub(super) fn requested(adapter: &wgpu::Adapter) -> bool {
    std::env::var_os("SOLIDCRAFT_RAYQUERY").is_some_and(|v| v == "1") && adapter.features().contains(FEATURE)
}

pub(super) fn token() -> wgpu::ExperimentalFeatures {
    // SAFETY: opt-in prototype; wgpu documents that experimental APIs may contain UB-containing bugs.
    unsafe { wgpu::ExperimentalFeatures::enabled() }
}

/// One BLAS over every uploaded triangle, in BVH leaf order, so `primitive_index` is the shader's triangle index.
pub(super) struct Scene {
    pub tlas: wgpu::Tlas,
    _blas: wgpu::Blas,
}

pub(super) fn build(device: &wgpu::Device, queue: &wgpu::Queue, geometry: &[u8]) -> Result<Scene> {
    let word = |i: usize| u32::from_le_bytes([geometry[4 * i], geometry[4 * i + 1], geometry[4 * i + 2], geometry[4 * i + 3]]);
    let triangles = word(0) as usize;
    let (vertices_at, materials_at) = (word(1) as usize, word(2) as usize);
    if triangles == 0 {
        return Err(gpu("hardware ray query needs at least one triangle"));
    }
    let mut indices = Vec::with_capacity(triangles * 12);
    for i in 0..triangles {
        for k in 0..3 {
            indices.extend_from_slice(&geometry[4 * (4 + 4 * i + k)..][..4]);
        }
    }
    // Vertices are 8 words (position, normal); the BLAS reads the position through a 32-byte stride.
    let vertex_bytes = &geometry[4 * vertices_at..4 * materials_at];
    let vertex_count = (vertex_bytes.len() / 32) as u32;
    let usage = wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE;
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("rq vertices"), contents: vertex_bytes, usage });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("rq indices"), contents: &indices, usage });
    let size = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(triangles as u32 * 3),
        flags: wgpu::AccelerationStructureGeometryFlags::empty(), // not OPAQUE: every triangle is a candidate for the shader
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("rq blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![size.clone()] },
    );
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("rq tlas"),
        max_instances: 1,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });
    let identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    *tlas.get_mut_single(0).ok_or_else(|| gpu("tlas slot"))? = Some(wgpu::TlasInstance::new(&blas, identity, 0, 0xff));
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rq build") });
    let entry = wgpu::BlasBuildEntry {
        blas: &blas,
        geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
            size: &size,
            vertex_buffer: &vertex_buffer,
            first_vertex: 0,
            vertex_stride: 32,
            index_buffer: Some(&index_buffer),
            first_index: Some(0),
            transform_buffer: None,
            transform_buffer_offset: None,
        }]),
    };
    encoder.build_acceleration_structures([&entry], [&tlas]);
    let submission = queue.submit([encoder.finish()]);
    device
        .poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: Some(std::time::Duration::from_secs(600)) })
        .map_err(gpu)?;
    Ok(Scene { tlas, _blas: blas })
}
