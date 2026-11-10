// SPDX-License-Identifier: MIT

//! Bind group layout: @group(0) has 3 bindings:
//!   @binding(0) params       storage,read       WorldgenParams + per-chunk header
//!   @binding(1) permutations storage,read       Perlin permutation tables
//!   @binding(2) out_bufs     storage,read_write 16 density channel output
//!
//! All fields are pure f32 / i32 at @align(4) — no f64, no padding mismatches.
//! Java/Rust/WGSL three-way assertion of every offset at startup (see
//! `constants.rs::_ASSERT_OFFSETS`).
//!
//! ## Phase 3 finalisation
//!
//! `create_layout` + `create_pipeline_layout` together produce the
//! `wgpu::PipelineLayout` consumed by `pipeline_cache::get_or_compile`
//! when it builds a `wgpu::ComputePipeline`. Both are derived from the
//! *static* bind group layout described in the BINDINGS_HEADER constant
//! in `codegen/wgsl_gen.rs` — they MUST stay in sync.
//!
//! Future per-world `wgpu::BindGroup` construction (binding concrete
//! buffers to these slots) is deferred to the scheduler phase — that's
//! when we know how many chunks are in the batch and can size `out_buf`
//! accordingly.

use crate::Result;

/// Bind group layout for `@group(0)` — 3 storage buffer bindings.
///
/// This must mirror the `BINDINGS_HEADER` const in
/// `codegen/wgsl_gen.rs`:
///   * binding 0: `var<storage, read> params: WorldgenParams`
///   * binding 1: `var<storage, read> permutations: array<i32>`
///   * binding 2: `var<storage, read_write> out_buf: array<f32>`
pub fn create_layout(device: &wgpu::Device) -> Result<wgpu::BindGroupLayout> {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("c2me_wgpu_bg_layout"),
        entries: &[
            // binding 0: params (read-only storage)
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 1: permutation tables (read-only storage)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 2: output density buffer (read-write storage)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 3: per-entry Perlin origin (x,y,z as f32), read-only
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 4: batch chunk positions, packed [cx,cz,...], read-only
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 5: SimplexNoiseSampler tables (EndIslands), read-only
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 6: external-input values for CPU-evaluated subtrees
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 7: block-state output (u32 per block), §3.5 block-output shard
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    Ok(layout)
}

/// Bind group layout for the standalone BlockOut shard (§3.5). Only the six
/// buffers that shard actually reads/writes: params(0), out_buf(2),
/// positions(4), block_buf(7), aq_p(8), aq_s(9). Keeping it separate from
/// the density layout keeps each group under `max_storage_buffers_per_shader_stage`.
pub fn create_block_out_layout(device: &wgpu::Device) -> Result<wgpu::BindGroupLayout> {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("c2me_wgpu_block_out_bg_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            // binding 10: static 24^3 Beardifier weight table (f32), §8.4.
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            // binding 11: per-chunk Beardifier pieces/junctions (u32), §8.4.
            wgpu::BindGroupLayoutEntry {
                binding: 11,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
        ],
    });
    Ok(layout)
}

/// Wrap the bind group layout into a pipeline layout (single bind group).
pub fn create_pipeline_layout(
    device: &wgpu::Device,
    bg_layout: &wgpu::BindGroupLayout,
) -> Result<wgpu::PipelineLayout> {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("c2me_wgpu_pipeline_layout"),
        bind_group_layouts: &[Some(bg_layout)],
        immediate_size: 0,
    });
    Ok(pipeline_layout)
}
