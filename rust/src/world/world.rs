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

//! Per-Minecraft-world context: AST + optimized WGSL + compiled pipeline
//! + the bind-group layout for params/permutations/output-buffers.

use crate::ast::AstNode;
use crate::ast::deserialize::{deserialize_full, ShapeConfig};
use crate::ast::opto;
use crate::ast::equality::hash_ast;
use crate::codegen::wgsl_gen::{self, Pass};
use crate::codegen::validate::validate_wgsl;
use crate::world::bind_group;
use crate::world::constants::PermutationTableSet;
use crate::world::pipeline_cache;
use crate::Result;
use wgpu::util::DeviceExt;

pub struct WorldContext {
    /// Parsed + optimized AST root (16-channel Root) — kept for fingerprinting
    /// and future shader hot-reload.
    #[allow(dead_code)]
    pub(crate) ast_root: AstNode,
    /// Parsed shape config (min_y / height / cell_*).
    pub(crate) shape: ShapeConfig,
    /// Biome tail from end of blob (currently `AstNode::BiomeTree { _placeholder }`
    /// with no further payload decoded). See plan.txt §4.4 for future protocol.
    #[allow(dead_code)]
    pub(crate) biome_source: AstNode,
    /// 32-bit fingerprint = hash(ast). Used for pipeline_cache lookups.
    pub(crate) fingerprint: u32,
    /// Emitted WGSL source for the single-pass `main_entry` shader.
    /// CacheInterpolated runs entirely within this single dispatch using
    /// `workgroupBarrier()` to synchronise corner computation and trilinear
    /// lerp. Validated via naga at `new` time.
    pub(crate) shader_src: String,
    /// Permutation tables for all `PerlinSampler` seeds discovered in the
    /// AST, packed into one flat `Vec<i32>` ready to upload to
    /// `@group(0) @binding(1)`.
    pub(crate) perm_table_set: PermutationTableSet,
    /// Compiled single-pass pipeline — `None` until `compile_pipeline` is
    /// called. Cached globally in `pipeline_cache::IN_MEMORY_CACHE` keyed
    /// by `(fingerprint, Pass::Single)`.
    pub(crate) pipeline: Option<wgpu::ComputePipeline>,
    /// GPU-side permutation buffer (binding 1). Created in
    /// `compile_pipeline`, kept alive for the lifetime of this
    /// `WorldContext` so the scheduler can bind it without re-uploading.
    pub(crate) perm_buffer: Option<wgpu::Buffer>,
}

impl WorldContext {
    /// Parse the Java-side blob, run AST opto passes, compute fingerprint,
    /// emit WGSL, and **smoke-test validate** the shader via naga (pure CPU).
    ///
    /// Fails fast on any emitter regression that produces invalid WGSL —
    /// long before a real `wgpu::Device::create_compute_pipeline` call
    /// would have caught it.
    ///
    /// Does **not** compile the GPU pipeline or upload any buffers. Call
    /// `compile_pipeline` afterwards with the singleton `WgpuInstance`'s
    /// device to produce a usable `wgpu::ComputePipeline` + perm buffer.
    /// This two-step split keeps `new` GPU-free so unit tests can run
    /// without a real adapter.
    pub fn new(config_blob: &[u8]) -> Result<Self> {
        // 1. Deserialize Java blob into AST + shape + biome tail
        let (ast_root, shape, biome_source) = deserialize_full(config_blob)?;

        // 2. Run 4-pass optimization pipeline (constant_fold → normalize → cse → dce)
        let ast_root = opto::run_pipeline(ast_root)?;

        // 3. Fingerprint = stable hash of optimized AST (Java ↔ Rust consistency)
        let fingerprint = hash_ast(&ast_root);

        // 4. Emit single-pass WGSL and smoke-test via naga.
        //    CacheInterpolated runs in single-pass mode using workgroupBarrier()
        //    to synchronise corner computation and trilinear lerp within one
        //    dispatch (var<workgroup> memory does NOT persist across dispatches).
        let shader_src = wgsl_gen::emit_shader(&ast_root)?;
        validate_wgsl(&shader_src).map_err(|e| {
            anyhow::anyhow!("emitted WGSL failed naga validation:\n{e}")
        })?;

        // 5. Scan AST for all PerlinSampler seeds and build the packed
        //    permutation-table buffer (CPU-side; upload happens in step 6).
        let seeds = collect_perlin_seeds(&ast_root);
        let perm_table_set = PermutationTableSet::from_seeds(seeds);

        Ok(Self {
            ast_root,
            shape,
            biome_source,
            fingerprint,
            shader_src,
            perm_table_set,
            pipeline: None,
            perm_buffer: None,
        })
    }

    /// Compile (or look up) the `wgpu::ComputePipeline` for this world's
    /// optimized AST + emit_shader output, and upload the permutation
    /// table buffer to the GPU.
    ///
    /// Idempotent: subsequent calls return the cached pipeline (cheap Arc
    /// clone) without re-uploading the permutation buffer.
    ///
    /// All ASTs — including those with `CacheInterpolated` — compile a
    /// single pipeline accessible via `pipeline()`. CacheInterpolated
    /// handles corner computation + `workgroupBarrier()` + trilinear lerp
    /// entirely within one dispatch, so no two-pass split is needed.
    pub fn compile_pipeline(&mut self, device: &wgpu::Device) -> Result<()> {
        // Fast path: already compiled.
        if self.pipeline.is_some() {
            return Ok(());
        }

        let bg_layout = bind_group::create_layout(device)?;
        let pipeline_layout = bind_group::create_pipeline_layout(device, &bg_layout)?;

        let pipeline = pipeline_cache::get_or_compile(
            self.fingerprint, Pass::Single, &self.shader_src, device, &pipeline_layout,
        )?;
        self.pipeline = Some(pipeline);

        // Upload the permutation table buffer (binding 1). Even if the
        // perm_table_set is empty (no PerlinSampler in the AST), we still
        // create a 1-i32 dummy buffer so the binding is valid — drivers
        // may reject empty storage bindings.
        if self.perm_buffer.is_none() {
            let buffer_data: Vec<i32> = if self.perm_table_set.buffer.is_empty() {
                vec![0]
            } else {
                self.perm_table_set.buffer.clone()
            };
            let perm_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("c2me_wgpu_permutations"),
                contents: bytemuck::cast_slice(&buffer_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            self.perm_buffer = Some(perm_buffer);
        }

        Ok(())
    }

    /// Borrow the compiled single-pass pipeline (None before
    /// `compile_pipeline` has been called).
    pub fn pipeline(&self) -> Option<&wgpu::ComputePipeline> {
        self.pipeline.as_ref()
    }

    /// Borrow the uploaded permutation buffer (binding 1), if
    /// `compile_pipeline` has been called.
    pub fn perm_buffer(&self) -> Option<&wgpu::Buffer> {
        self.perm_buffer.as_ref()
    }

    /// Borrow the CPU-side permutation table set (useful for unit tests
    /// and for the scheduler when it needs to cross-check offsets).
    pub fn perm_table_set(&self) -> &PermutationTableSet {
        &self.perm_table_set
    }

    /// Stable 32-bit fingerprint of the optimized AST. Same input NoiseRouter
    /// always produces the same fingerprint — used to key pipeline_cache.
    pub fn fingerprint(&self) -> u32 {
        self.fingerprint
    }

    /// Shape config (min_y / height / cell_width / cell_height) — needed by
    /// the scheduler to translate "MC cell index" ↔ "WGSL dispatch grid".
    pub fn shape(&self) -> &ShapeConfig {
        &self.shape
    }
}

// ============================================================================
// AST helpers
// ============================================================================

/// Recursively walk an AST and collect every `PerlinSampler.seed` value
/// found. Used by `WorldContext::new` to build the permutation table set.
///
/// Visits children in pre-order. Does **not** deduplicate — that's the
/// job of `PermutationTableSet::from_seeds`. The order is deterministic
/// (AST is a stable tree), so the resulting buffer layout is reproducible.
fn collect_perlin_seeds(root: &AstNode) -> Vec<i64> {
    let mut seeds = Vec::new();
    visit(root, &mut seeds);
    seeds
}

fn visit(node: &AstNode, out: &mut Vec<i64>) {
    match node {
        AstNode::PerlinSampler { seed, .. } => {
            out.push(*seed);
        }
        // Binary ops.
        AstNode::Add(a, b) | AstNode::Sub(a, b)
        | AstNode::Mul(a, b) | AstNode::Div(a, b)
        | AstNode::Min(a, b) | AstNode::Max(a, b)
        | AstNode::FloorDiv(a, b) => {
            visit(a, out);
            visit(b, out);
        }
        // Unary ops.
        AstNode::Abs(x) | AstNode::Square(x)
        | AstNode::Cube(x) | AstNode::Sqrt(x) => {
            visit(x, out);
        }
        // Noise samplers.
        AstNode::DoublePerlin { a, b } => {
            visit(a, out);
            visit(b, out);
        }
        AstNode::EndIslands(inner) => visit(inner, out),
        // Cache wrappers.
        AstNode::CacheFlat(x) | AstNode::Cache2D(x)
        | AstNode::CacheInterpolated(x) => {
            visit(x, out);
        }
        // Selection / surface nodes with child subtrees.
        AstNode::RangeChoice { input, min, max, in_bounds, out_of_bounds_lo, out_of_bounds_hi } => {
            visit(input, out);
            visit(min, out);
            visit(max, out);
            visit(in_bounds, out);
            visit(out_of_bounds_lo, out);
            visit(out_of_bounds_hi, out);
        }
        AstNode::FindTopSurface { density, initial, .. } => {
            visit(density, out);
            visit(initial, out);
        }
        // Root with channels.
        AstNode::Root(channels) => {
            for ch in channels {
                visit(ch, out);
            }
        }
        // Leaves — no children to recurse into.
        AstNode::Constant(_)
        | AstNode::InputX | AstNode::InputY | AstNode::InputZ
        | AstNode::SimplexNoise2D { .. } | AstNode::SimplexNoise3D { .. }
        | AstNode::Beardifier { .. }
        | AstNode::Spline { .. }
        | AstNode::IntervalSelect { .. }
        | AstNode::YClampedGradient { .. }
        | AstNode::Aquifer { .. }
        | AstNode::InterpolatedNoiseSampler { .. }
        | AstNode::BiomeTree { .. } => {}
    }
}

// ======================================================================
// Tests
// ======================================================================
#[cfg(test)]
mod tests{
    #[test]
    fn integration_full_nodes_pipeline() {
        use crate::ast::deserialize::deserialize_full;
        use crate::codegen::wgsl_gen::emit_shader;
        use crate::ast::opto::run_pipeline;

        // 构造包含多种节点的 blob（与之前相同）
        let mut blob = Vec::new();
        blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
        blob.push(0xFE);
        blob.extend_from_slice(&[0x06, 0x00]);
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.push(0x02);
        blob.extend_from_slice(&[0x80]);
        blob.extend_from_slice(&42i64.to_le_bytes());
        blob.extend_from_slice(&1i32.to_le_bytes());
        blob.extend_from_slice(&1i32.to_le_bytes());
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.extend_from_slice(&1i32.to_le_bytes());
        blob.extend_from_slice(&2.0f32.to_le_bytes());
        blob.extend_from_slice(&[0x82]);
        blob.extend_from_slice(&7i64.to_le_bytes());
        blob.extend_from_slice(&[0xB1]);
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&0.0f32.to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&2.0f32.to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&3.0f32.to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&(-1.0f32).to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.extend_from_slice(&[0xC0, 0x03]);
        blob.extend_from_slice(&(-64i32).to_le_bytes());
        blob.extend_from_slice(&384i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.extend_from_slice(&[0xF0]);
        blob.extend_from_slice(&123456789i64.to_le_bytes());

        // 1. 反序列化
        let (root, _, _) = deserialize_full(&blob).unwrap();

        // 2. 优化
        let optimized = run_pipeline(root).unwrap();

        // 3. 生成 WGSL（仅 CPU 端，不涉及 GPU）
        let wgsl = emit_shader(&optimized).unwrap();

        // 4. 验证生成的着色器包含关键函数调用
        assert!(wgsl.contains("range_"));
        assert!(wgsl.contains("range_"));
        assert!(wgsl.contains("out_buf"));
        println!("✅ 集成测试通过：反序列化、优化、WGSL生成成功！");
    }
        #[test]
    fn gpu_end_to_end_test() {
        use crate::instance::WgpuInstance;
        use crate::world::WorldContext;
        use crate::scheduler::batcher::{Batch, Priority};
        use crate::scheduler::executor::ExecutorThread;
        use std::sync::Arc;

        let mut blob = Vec::new();
        blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
        blob.push(0xFE);
        blob.extend_from_slice(&[0x06, 0x00]);
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.push(0x02);
        blob.extend_from_slice(&[0x80]);
        blob.extend_from_slice(&42i64.to_le_bytes());
        blob.extend_from_slice(&1i32.to_le_bytes());
        blob.extend_from_slice(&1i32.to_le_bytes());
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.extend_from_slice(&1i32.to_le_bytes());
        blob.extend_from_slice(&2.0f32.to_le_bytes());
        blob.extend_from_slice(&[0x82]);
        blob.extend_from_slice(&7i64.to_le_bytes());
        blob.extend_from_slice(&[0xB1]);
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&0.0f32.to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&2.0f32.to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&3.0f32.to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&(-1.0f32).to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.extend_from_slice(&[0xC0, 0x03]);
        blob.extend_from_slice(&(-64i32).to_le_bytes());
        blob.extend_from_slice(&384i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.extend_from_slice(&[0xF0]);
        blob.extend_from_slice(&123456789i64.to_le_bytes());

        let mut ctx = WorldContext::new(&blob).unwrap();
        let instance = WgpuInstance::instance().unwrap();
        ctx.compile_pipeline(&instance.device).unwrap();
        let world_arc = Arc::new(ctx);

        let executor = ExecutorThread::spawn(
            Arc::new(instance.device.clone()),
            Arc::new(instance.queue.clone()),
            Default::default(),
        ).unwrap();

        let batch = Batch {
            positions: vec![(0, 0)],
            max_priority: Priority::Normal,
        };

        let result = executor.request_chunks(world_arc, batch).unwrap();
        assert!(!result.results.is_empty());
        let blob_result = result.results.values().next().unwrap();
        assert!(blob_result.verify_crc());
        println!("✅ GPU 端到端测试通过！");
    }
    
    #[test]
    fn gpu_performance_benchmark() {
        use crate::instance::WgpuInstance;
        use crate::world::WorldContext;
        use crate::scheduler::batcher::{Batch, Priority};
        use crate::scheduler::executor::ExecutorThread;
        use std::sync::Arc;
        use std::time::Instant;

        // 构造一个包含多种节点的 blob（与之前相同）
        let mut blob = Vec::new();
        blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
        blob.push(0xFE);
        blob.extend_from_slice(&[0x06, 0x00]);
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.push(0x02);
        blob.extend_from_slice(&[0x80]);
        blob.extend_from_slice(&42i64.to_le_bytes());
        blob.extend_from_slice(&1i32.to_le_bytes());
        blob.extend_from_slice(&1i32.to_le_bytes());
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.extend_from_slice(&1i32.to_le_bytes());
        blob.extend_from_slice(&2.0f32.to_le_bytes());
        blob.extend_from_slice(&[0x82]);
        blob.extend_from_slice(&7i64.to_le_bytes());
        blob.extend_from_slice(&[0xB1]);
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&0.0f32.to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&2.0f32.to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&3.0f32.to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&(-1.0f32).to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.extend_from_slice(&[0xC0, 0x03]);
        blob.extend_from_slice(&(-64i32).to_le_bytes());
        blob.extend_from_slice(&384i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.extend_from_slice(&[0xF0]);
        blob.extend_from_slice(&123456789i64.to_le_bytes());

        // 创建 WorldContext 并编译 pipeline
        let mut ctx = WorldContext::new(&blob).unwrap();
        let instance = WgpuInstance::instance().unwrap();
        ctx.compile_pipeline(&instance.device).unwrap();
        let world_arc = Arc::new(ctx);

        // 创建执行器线程
        let executor = ExecutorThread::spawn(
            Arc::new(instance.device.clone()),
            Arc::new(instance.queue.clone()),
            Default::default(),
        ).unwrap();

        // 生成 64x64 = 4096 个区块坐标 (cx, cz) 从 (0,0) 到 (63,63)
        let mut all_positions = Vec::new();
        for cx in 0..64 {
            for cz in 0..64 {
                all_positions.push((cx, cz));
            }
        }

        // 分批处理，每批最多 64 个区块
        let batch_size = 64;
        let total_batches = (all_positions.len() + batch_size - 1) / batch_size;

        let start = Instant::now();
        let mut total_chunks = 0;

        for batch_idx in 0..total_batches {
            let start_pos = batch_idx * batch_size;
            let end_pos = (start_pos + batch_size).min(all_positions.len());
            let chunk_slice = &all_positions[start_pos..end_pos];

            let batch = Batch {
                positions: chunk_slice.to_vec(),
                max_priority: Priority::Normal,
            };

            let _result = executor.request_chunks(world_arc.clone(), batch).unwrap();
            total_chunks += chunk_slice.len();
        }

        let elapsed = start.elapsed();
        let total_seconds = elapsed.as_secs_f64();
        let chunks_per_second = total_chunks as f64 / total_seconds;

        println!(
            "✅ 性能测试完成：生成 {} 个区块，耗时 {:.3}s，吞吐量 {:.1} 区块/秒",
            total_chunks, total_seconds, chunks_per_second
        );

        // 如果时间太短，给出提示
        if total_seconds < 1.0 {
            println!("⚠️ 总耗时较短，建议增加区块数量以获得更稳定的平均数据。");
        }
    }
}