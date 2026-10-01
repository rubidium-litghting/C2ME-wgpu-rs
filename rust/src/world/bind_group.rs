/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2026 RubidiumLighting
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

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
