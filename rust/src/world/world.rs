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

use crate::Result;
use crate::ast::AstNode;
use crate::ast::deserialize::{PermEntry, ShapeConfig, deserialize_full_with_perm};
use crate::ast::equality::hash_ast;
use crate::ast::opto;
use crate::codegen::validate::validate_wgsl;
use crate::codegen::wgsl_gen::{self, Pass};
use crate::world::bind_group;
use crate::world::constants::PermutationTableSet;
use crate::world::pipeline_cache;
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
    /// Real Perlin permutation tables shipped by the Java side in the
    /// PERM_RESOURCE trailer. Stored for the upcoming noise emitters;
    /// not yet consumed by WGSL generation.
    #[allow(dead_code)]
    pub(crate) perm_entries: Vec<PermEntry>,
    /// Compiled single-pass pipeline — `None` until `compile_pipeline` is
    /// called. Cached globally in `pipeline_cache::IN_MEMORY_CACHE` keyed
    /// by `(fingerprint, Pass::Single)`.
    pub(crate) pipeline: Option<wgpu::ComputePipeline>,
    /// GPU-side permutation buffer (binding 1). Created in
    /// `compile_pipeline`, kept alive for the lifetime of this
    /// `WorldContext` so the scheduler can bind it without re-uploading.
    pub(crate) perm_buffer: Option<wgpu::Buffer>,
    /// GPU-side per-entry origin buffer (binding 3), f32 x3 per entry.
    pub(crate) origin_buffer: Option<wgpu::Buffer>,
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
        let (ast_root, shape, biome_source, perm_entries) =
            deserialize_full_with_perm(config_blob)?;

        // 2. Run 4-pass optimization pipeline (constant_fold → normalize → cse → dce)
        let ast_root = opto::run_pipeline(ast_root)?;

        // 3. Fingerprint = stable hash of optimized AST (Java ↔ Rust consistency)
        let fingerprint = hash_ast(&ast_root);

        // 4. Emit single-pass WGSL and smoke-test via naga.
        //    CacheInterpolated runs in single-pass mode using workgroupBarrier()
        //    to synchronise corner computation and trilinear lerp within one
        //    dispatch (var<workgroup> memory does NOT persist across dispatches).
        let shader_src = wgsl_gen::emit_shader(&ast_root)?;
        validate_wgsl(&shader_src)
            .map_err(|e| anyhow::anyhow!("emitted WGSL failed naga validation:\n{e}"))?;

        // 5. Build the packed permutation-table buffer from the REAL tables
        //    shipped in the Java PERM_RESOURCE trailer. `from_entries` returns
        //    an empty set when the trailer is absent (unit tests / old blobs);
        //    the buffer-upload path below already substitutes a 1-i32 dummy in
        //    that case so the binding stays valid.
        let perm_table_set = PermutationTableSet::from_entries(&perm_entries);

        Ok(Self {
            ast_root,
            shape,
            biome_source,
            fingerprint,
            shader_src,
            perm_table_set,
            perm_entries,
            pipeline: None,
            perm_buffer: None,
            origin_buffer: None,
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
            self.fingerprint,
            Pass::Single,
            &self.shader_src,
            device,
            &pipeline_layout,
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

        // Upload per-entry origin buffer (binding 3).
        if self.origin_buffer.is_none() {
            let origin_data: Vec<f32> = if self.perm_table_set.origins.is_empty() {
                vec![0.0f32, 0.0, 0.0]
            } else {
                self.perm_table_set.origins.clone()
            };
            let ob = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("c2me_wgpu_perm_origins"),
                contents: bytemuck::cast_slice(&origin_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            self.origin_buffer = Some(ob);
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

    /// Borrow the uploaded origin buffer (binding 3), if
    /// `compile_pipeline` has been called.
    pub fn origin_buffer(&self) -> Option<&wgpu::Buffer> {
        self.origin_buffer.as_ref()
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

// ======================================================================
// Tests
// ======================================================================
#[cfg(test)]
mod tests {
    #[test]
    fn integration_full_nodes_pipeline() {
        use crate::ast::deserialize::deserialize_full;
        use crate::ast::opto::run_pipeline;
        use crate::codegen::wgsl_gen::emit_shader;

        // 构造包含多种节点的 blob（与之前相同）
        let mut blob = Vec::new();
        blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
        blob.push(0xFE);
        blob.extend_from_slice(&[0x06, 0x00]);
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.push(0x02);
        // TAG_NOISE_SAMPLER(0x84)
        //   type=Perlin(0) flags=0 (single stack)
        //   amplitude=1.0 xz_scale=1.0 y_scale=1.0
        //   first stack: perm_offset=0 initial_freq=1.0 persistence=1.0
        //                octaves=1 amps=[1.0]
        blob.extend_from_slice(&[0x84, 0x00, 0x00]);
        blob.extend_from_slice(&1.0f32.to_le_bytes());   // amplitude
        blob.extend_from_slice(&1.0f32.to_le_bytes());   // xz_scale
        blob.extend_from_slice(&1.0f32.to_le_bytes());   // y_scale
        blob.extend_from_slice(&0i32.to_le_bytes());     // perm_offset
        blob.extend_from_slice(&1.0f32.to_le_bytes());   // initial_freq
        blob.extend_from_slice(&1.0f32.to_le_bytes());   // persistence
        blob.extend_from_slice(&1i32.to_le_bytes());     // octaves
        blob.extend_from_slice(&1.0f32.to_le_bytes());   // amp[0]
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
        use crate::scheduler::batcher::{Batch, Priority};
        use crate::scheduler::executor::ExecutorThread;
        use crate::world::WorldContext;
        use std::sync::Arc;

        let mut blob = Vec::new();
        blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
        blob.push(0xFE);
        blob.extend_from_slice(&[0x06, 0x00]);
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.push(0x02);
        // TAG_NOISE_SAMPLER(0x84)
        //   type=Perlin(0) flags=0 (single stack)
        //   amplitude=1.0 xz_scale=1.0 y_scale=1.0
        //   first stack: perm_offset=0 initial_freq=1.0 persistence=1.0
        //                octaves=1 amps=[1.0]
        blob.extend_from_slice(&[0x84, 0x00, 0x00]);
        blob.extend_from_slice(&1.0f32.to_le_bytes());   // amplitude
        blob.extend_from_slice(&1.0f32.to_le_bytes());   // xz_scale
        blob.extend_from_slice(&1.0f32.to_le_bytes());   // y_scale
        blob.extend_from_slice(&0i32.to_le_bytes());     // perm_offset
        blob.extend_from_slice(&1.0f32.to_le_bytes());   // initial_freq
        blob.extend_from_slice(&1.0f32.to_le_bytes());   // persistence
        blob.extend_from_slice(&1i32.to_le_bytes());     // octaves
        blob.extend_from_slice(&1.0f32.to_le_bytes());   // amp[0]
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
        )
        .unwrap();

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
        use crate::scheduler::batcher::{Batch, Priority};
        use crate::scheduler::executor::ExecutorThread;
        use crate::world::WorldContext;
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
        // TAG_NOISE_SAMPLER(0x84)
        //   type=Perlin(0) flags=0 (single stack)
        //   amplitude=1.0 xz_scale=1.0 y_scale=1.0
        //   first stack: perm_offset=0 initial_freq=1.0 persistence=1.0
        //                octaves=1 amps=[1.0]
        blob.extend_from_slice(&[0x84, 0x00, 0x00]);
        blob.extend_from_slice(&1.0f32.to_le_bytes());   // amplitude
        blob.extend_from_slice(&1.0f32.to_le_bytes());   // xz_scale
        blob.extend_from_slice(&1.0f32.to_le_bytes());   // y_scale
        blob.extend_from_slice(&0i32.to_le_bytes());     // perm_offset
        blob.extend_from_slice(&1.0f32.to_le_bytes());   // initial_freq
        blob.extend_from_slice(&1.0f32.to_le_bytes());   // persistence
        blob.extend_from_slice(&1i32.to_le_bytes());     // octaves
        blob.extend_from_slice(&1.0f32.to_le_bytes());   // amp[0]
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
        )
        .unwrap();

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

    // ── Numerical validation: GPU shader vs Rust CPU reference ──────────
    //
    // Uses a single-octave NoiseSampler with a known perm table (seed=42)
    // and no DOMAIN_SCALE, so the GPU result should exactly equal a
    // straightforward Rust transcription of MC's Perlin algorithm.

    fn cpu_maintain_precision(v: f32) -> f32 {
        let divisor = 33554432.0f32;
        v - (v / divisor + 0.5).floor() * divisor
    }

    fn cpu_fade(t: f32) -> f32 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    fn cpu_lerp(a: f32, b: f32, t: f32) -> f32 {
        t.mul_add(b - a, a)
    }

    fn cpu_trilinear(c000: f32, c100: f32, c010: f32, c110: f32,
                     c001: f32, c101: f32, c011: f32, c111: f32,
                     tx: f32, ty: f32, tz: f32) -> f32 {
        let x00 = cpu_lerp(c000, c100, tx);
        let x10 = cpu_lerp(c010, c110, tx);
        let x01 = cpu_lerp(c001, c101, tx);
        let x11 = cpu_lerp(c011, c111, tx);
        let y0 = cpu_lerp(x00, x10, ty);
        let y1 = cpu_lerp(x01, x11, ty);
        cpu_lerp(y0, y1, tz)
    }

    fn cpu_perlin3d_impl(perm: &[i32], p: i32, x: f32, y: f32, z: f32) -> f32 {
        let ix = x.floor() as i32;
        let iy = y.floor() as i32;
        let iz = z.floor() as i32;
        let fx = x - ix as f32;
        let fy = y - iy as f32;
        let fz = z - iz as f32;
        let xi = ix & 255;
        let yi = iy & 255;
        let zi = iz & 255;
        let idx = |i: i32| (p + i) as usize;
        let a  = perm[idx(xi)];
        let b  = perm[idx((xi + 1) & 255)];
        let aa = perm[idx((a + yi) & 255)];
        let ab = perm[idx((a + yi + 1) & 255)];
        let ba = perm[idx((b + yi) & 255)];
        let bb = perm[idx((b + yi + 1) & 255)];
        let aaa = perm[idx((aa + zi) & 255)];
        let aab = perm[idx((aa + zi + 1) & 255)];
        let aba = perm[idx((ab + zi) & 255)];
        let abb = perm[idx((ab + zi + 1) & 255)];
        let baa = perm[idx((ba + zi) & 255)];
        let bab = perm[idx((ba + zi + 1) & 255)];
        let bba = perm[idx((bb + zi) & 255)];
        let bbb = perm[idx((bb + zi + 1) & 255)];
        let g = |i: i32| -> [f32; 3] {
            match i & 15 {
                0 => [1.0, 1.0, 0.0],  1 => [-1.0, 1.0, 0.0],
                2 => [1.0, -1.0, 0.0], 3 => [-1.0, -1.0, 0.0],
                4 => [1.0, 0.0, 1.0],  5 => [-1.0, 0.0, 1.0],
                6 => [1.0, 0.0, -1.0], 7 => [-1.0, 0.0, -1.0],
                8 => [0.0, 1.0, 1.0],  9 => [0.0, -1.0, 1.0],
                10 => [0.0, 1.0, -1.0], 11 => [0.0, -1.0, -1.0],
                12 => [1.0, 1.0, 0.0], 13 => [0.0, -1.0, 1.0],
                14 => [-1.0, 1.0, 0.0], 15 => [0.0, -1.0, -1.0],
                _ => unreachable!(),
            }
        };
        let dot = |gr: [f32; 3], dx: f32, dy: f32, dz: f32| gr[0]*dx + gr[1]*dy + gr[2]*dz;
        let n000 = dot(g(aaa), fx, fy, fz);
        let n100 = dot(g(baa), fx - 1.0, fy, fz);
        let n010 = dot(g(aba), fx, fy - 1.0, fz);
        let n110 = dot(g(bba), fx - 1.0, fy - 1.0, fz);
        let n001 = dot(g(aab), fx, fy, fz - 1.0);
        let n101 = dot(g(bab), fx - 1.0, fy, fz - 1.0);
        let n011 = dot(g(abb), fx, fy - 1.0, fz - 1.0);
        let n111 = dot(g(bbb), fx - 1.0, fy - 1.0, fz - 1.0);
        let u = cpu_fade(fx);
        let v = cpu_fade(fy);
        let w = cpu_fade(fz);
        cpu_trilinear(n000, n100, n010, n110, n001, n101, n011, n111, u, v, w)
    }

    fn cpu_noise_perlin3d(perm: &[i32], p: i32, x: f32, y: f32, z: f32) -> f32 {
        let px = cpu_maintain_precision(x);
        let py = cpu_maintain_precision(y);
        let pz = cpu_maintain_precision(z);
        cpu_perlin3d_impl(perm, p, px, py, pz)
    }

    fn cpu_kahan_add(acc: &mut (f32, f32), v: f32) {
        let y = v - acc.1;
        let t = acc.0 + y;
        acc.1 = (t - acc.0) - y;
        acc.0 = t;
    }

    fn cpu_octave_stack(
        perm: &[i32], base: i32, initial_freq: f32, persistence: f32,
        amps: &[f32], x: f32, y: f32, z: f32,
    ) -> f32 {
        let mut freq = initial_freq;
        let mut amp = persistence;
        let mut acc = (0.0f32, 0.0f32);
        for i in 0..amps.len() {
            if amps[i] != 0.0 {
                let idx = base + (i as i32) * 256;
                let s = cpu_noise_perlin3d(perm, idx, x * freq, y * freq, z * freq);
                cpu_kahan_add(&mut acc, amps[i] * s * amp);
            }
            freq *= 2.0;
            amp *= 0.5;
        }
        acc.0
    }

    fn cpu_noise_sampler(
        perm: &[i32],
        first: (i32, f32, f32, &[f32]),
        second: Option<(i32, f32, f32, &[f32])>,
        amplitude: f32, xz: f32, ys: f32, x: f32, y: f32, z: f32,
    ) -> f32 {
        const D: f32 = 1.0181268882175227;
        let (fo, ff, fp, fa) = first;
        let vf = cpu_octave_stack(perm, fo, ff, fp, fa, x * xz, y * ys, z * xz);
        let total = if let Some((so, sf, sp, sa)) = second {
            let xzd = xz * D;
            let ysd = ys * D;
            vf + cpu_octave_stack(perm, so, sf, sp, sa, x * xzd, y * ysd, z * xzd)
        } else {
            vf
        };
        total * amplitude
    }

    fn build_run_blob(
        flags: u8, amplitude: f32, xz: f32, ys: f32,
        first: (i32, f32, f32, &[f32]),
        second: Option<(i32, f32, f32, &[f32])>,
        perms: &[[u32; 256]],
    ) -> Vec<f32> {
        use crate::instance::WgpuInstance;
        use crate::scheduler::batcher::{Batch, Priority};
        use crate::scheduler::executor::ExecutorThread;
        use crate::world::WorldContext;
        use std::sync::Arc;

        let mut blob = Vec::new();
        blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
        blob.push(0xFE);
        blob.extend_from_slice(&[0x01, 0x00]);
        blob.extend_from_slice(&[0x84, 0x00, flags]);
        blob.extend_from_slice(&amplitude.to_le_bytes());
        blob.extend_from_slice(&xz.to_le_bytes());
        blob.extend_from_slice(&ys.to_le_bytes());
        let (fo, ff, fp, fa) = first;
        blob.extend_from_slice(&fo.to_le_bytes());
        blob.extend_from_slice(&ff.to_le_bytes());
        blob.extend_from_slice(&fp.to_le_bytes());
        blob.extend_from_slice(&(fa.len() as i32).to_le_bytes());
        for &a in fa { blob.extend_from_slice(&a.to_le_bytes()); }
        if let Some((so, sf, sp, sa)) = second {
            blob.extend_from_slice(&so.to_le_bytes());
            blob.extend_from_slice(&sf.to_le_bytes());
            blob.extend_from_slice(&sp.to_le_bytes());
            blob.extend_from_slice(&(sa.len() as i32).to_le_bytes());
            for &a in sa { blob.extend_from_slice(&a.to_le_bytes()); }
        }
        blob.extend_from_slice(&(-64i32).to_le_bytes());
        blob.extend_from_slice(&384i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.push(0xF0);
        blob.extend_from_slice(&123456789i64.to_le_bytes());
        blob.extend_from_slice(&0x4D525750u32.to_le_bytes());
        blob.extend_from_slice(&(perms.len() as u32).to_le_bytes());
        for tb in perms {
            for &v in tb { blob.extend_from_slice(&v.to_le_bytes()); }
            for _ in 0..3 { blob.extend_from_slice(&0.0f64.to_le_bytes()); }
        }

        let mut ctx = WorldContext::new(&blob).unwrap();
        let instance = WgpuInstance::instance().unwrap();
        ctx.compile_pipeline(&instance.device).unwrap();
        let world_arc = Arc::new(ctx);
        let executor = ExecutorThread::spawn(
            Arc::new(instance.device.clone()),
            Arc::new(instance.queue.clone()),
            Default::default(),
        ).unwrap();
        let batch = Batch { positions: vec![(0, 0)], max_priority: Priority::Normal };
        let result = executor.request_chunks(world_arc, batch).unwrap();
        let blob_result = result.results.values().next().unwrap();
        let bytes = blob_result.as_bytes();
        let payload = &bytes[0x88..];
        (0..1536).map(|i| f32::from_le_bytes(payload[i*4..i*4+4].try_into().unwrap())).collect()
    }

    #[test]
    fn numerical_validation_offset_isolation() {
        use crate::world::constants::permutation_table_for_seed;

        let perms: Vec<[u32; 256]> = [11i64, 22, 33].iter()
            .map(|&sd| permutation_table_for_seed(sd)).collect();
        let perm_i32: Vec<i32> = perms.iter().flatten().map(|&v| v as i32).collect();

        // Case B: single octave at offset 0
        let amps1 = [1.0f32];
        let gpu_b = build_run_blob(0, 1.0, 0.3, 0.3, (0, 1.0, 1.0, &amps1), None, &perms);
        let mut max_b = 0.0f32;
        for cy in 0..96usize { for cz in 0..4usize { for cx in 0..4usize {
            let wx = (cx * 4) as f32;
            let wy = (-64 + cy as i32 * 4) as f32;
            let wz = (cz * 4) as f32;
            let cpu = cpu_octave_stack(&perm_i32, 0, 1.0, 1.0, &amps1, wx*0.3, wy*0.3, wz*0.3);
            let gpu = gpu_b[cx + cz*4 + cy*16];
            max_b = max_b.max((cpu - gpu).abs());
        }}}
        println!("Case B (1 octave, offset 0):    max_diff={:.6e}", max_b);

        // Case C: single octave at offset 256
        let gpu_c = build_run_blob(0, 1.0, 0.3, 0.3, (256, 1.0, 1.0, &amps1), None, &perms);
        let mut max_c = 0.0f32;
        for cy in 0..96usize { for cz in 0..4usize { for cx in 0..4usize {
            let wx = (cx * 4) as f32;
            let wy = (-64 + cy as i32 * 4) as f32;
            let wz = (cz * 4) as f32;
            let cpu = cpu_octave_stack(&perm_i32, 256, 1.0, 1.0, &amps1, wx*0.3, wy*0.3, wz*0.3);
            let gpu = gpu_c[cx + cz*4 + cy*16];
            max_c = max_c.max((cpu - gpu).abs());
        }}}
        println!("Case C (1 octave, offset 256):  max_diff={:.6e}", max_c);

        // Case D: two octaves
        let amps2 = [1.0f32, 0.5];
        let gpu_d = build_run_blob(0, 1.0, 0.3, 0.3, (0, 1.0, 1.0, &amps2), None, &perms);
        let mut max_d = 0.0f32;
        for cy in 0..96usize { for cz in 0..4usize { for cx in 0..4usize {
            let wx = (cx * 4) as f32;
            let wy = (-64 + cy as i32 * 4) as f32;
            let wz = (cz * 4) as f32;
            let cpu = cpu_octave_stack(&perm_i32, 0, 1.0, 1.0, &amps2, wx*0.3, wy*0.3, wz*0.3);
            let gpu = gpu_d[cx + cz*4 + cy*16];
            max_d = max_d.max((cpu - gpu).abs());
        }}}
        println!("Case D (2 octaves):             max_diff={:.6e}", max_d);

        // Debug: print cell (0,0,0) contributions
        let wx = 0.0f32; let wy = -64.0f32; let wz = 0.0f32;
        let perm_p = &perm_i32;
        let s0 = cpu_noise_perlin3d(perm_p, 0, wx*0.3, wy*0.3, wz*0.3);
        let s1 = cpu_noise_perlin3d(perm_p, 256, wx*0.3*2.0, wy*0.3*2.0, wz*0.3*2.0);
        let s2 = cpu_noise_perlin3d(perm_p, 512, wx*0.3*4.0, wy*0.3*4.0, wz*0.3*4.0);
        let cpu3 = s0 + 0.5*s1*0.5 + 0.25*s2*0.25;
        println!("  cell(0,0,0): oct0={:.6} oct1={:.6} oct2={:.6} -> cpu={:.6}", s0, s1, s2, cpu3);
        println!("  cell(0,0,0): gpu_A={:.6} gpu_D={:.6}", gpu_b[0], gpu_d[0]);

        assert!(max_b < 1e-4, "Case B diverged {:.6e}", max_b);
        assert!(max_c < 1e-4, "Case C diverged {:.6e}", max_c);
        assert!(max_d < 1e-4, "Case D diverged {:.6e}", max_d);
    }

    #[test]
    fn numerical_validation_stack_isolation() {
        use crate::world::constants::permutation_table_for_seed;

        let perms: Vec<[u32; 256]> = [11i64, 22, 33].iter()
            .map(|&sd| permutation_table_for_seed(sd)).collect();
        let perm_i32: Vec<i32> = perms.iter().flatten().map(|&v| v as i32).collect();
        let amps = [1.0f32, 0.5, 0.25];

        // Case A: single stack, 3 octaves, amplitude=1.0, no D.
        let gpu_a = build_run_blob(0, 1.0, 0.3, 0.3,
            (0, 1.0, 1.0, &amps), None, &perms);
        let mut max_a = 0.0f32;
        for cy in 0..96usize { for cz in 0..4usize { for cx in 0..4usize {
            let wx = (cx * 4) as f32;
            let wy = (-64 + cy as i32 * 4) as f32;
            let wz = (cz * 4) as f32;
            let cpu = cpu_octave_stack(&perm_i32, 0, 1.0, 1.0, &amps, wx*0.3, wy*0.3, wz*0.3);
            let gpu = gpu_a[cx + cz*4 + cy*16];
            max_a = max_a.max((cpu - gpu).abs());
        }}}
        println!("Case A (single stack, 3 octaves): max_diff={:.6e}", max_a);
        assert!(max_a < 1e-4, "Case A diverged {:.6e}", max_a);
    }

    #[test]
    fn numerical_validation_double_stack_multioctave() {
        use crate::instance::WgpuInstance;
        use crate::scheduler::batcher::{Batch, Priority};
        use crate::scheduler::executor::ExecutorThread;
        use crate::world::WorldContext;
        use crate::world::constants::permutation_table_for_seed;
        use std::sync::Arc;

        // 3 octaves in first stack (tables 0/256/512), 2 in second (768/1024).
        let seed_first = [11i64, 22, 33];
        let seed_second = [44i64, 55];
        let perms: Vec<[u32; 256]> = seed_first.iter().chain(seed_second.iter())
            .map(|&sd| permutation_table_for_seed(sd)).collect();
        let perm_i32: Vec<i32> = perms.iter().flatten().map(|&v| v as i32).collect();

        let amps_first = [1.0f32, 0.5, 0.25];
        let amps_second = [1.0f32, 0.5];

        let mut blob = Vec::new();
        blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
        blob.push(0xFE);
        blob.extend_from_slice(&[0x01, 0x00]);
        blob.extend_from_slice(&[0x84, 0x00, 0x01]);   // type=Perlin, flags=has_second
        blob.extend_from_slice(&0.8f32.to_le_bytes()); // amplitude
        blob.extend_from_slice(&0.3f32.to_le_bytes()); // xz_scale
        blob.extend_from_slice(&0.3f32.to_le_bytes()); // y_scale
        // first stack
        blob.extend_from_slice(&0i32.to_le_bytes());
        blob.extend_from_slice(&1.0f32.to_le_bytes());  // initial_freq
        blob.extend_from_slice(&1.0f32.to_le_bytes());  // persistence
        blob.extend_from_slice(&3i32.to_le_bytes());
        for a in amps_first { blob.extend_from_slice(&a.to_le_bytes()); }
        // second stack
        blob.extend_from_slice(&(3 * 256i32).to_le_bytes());
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.extend_from_slice(&2i32.to_le_bytes());
        for a in amps_second { blob.extend_from_slice(&a.to_le_bytes()); }
        // shape + biome
        blob.extend_from_slice(&(-64i32).to_le_bytes());
        blob.extend_from_slice(&384i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.push(0xF0);
        blob.extend_from_slice(&123456789i64.to_le_bytes());
        // PERM_RESOURCE
        blob.extend_from_slice(&0x4D525750u32.to_le_bytes());
        blob.extend_from_slice(&(perms.len() as u32).to_le_bytes());
        for tb in &perms {
            for &v in tb { blob.extend_from_slice(&v.to_le_bytes()); }
            for _ in 0..3 { blob.extend_from_slice(&0.0f64.to_le_bytes()); }
        }

        let mut ctx = WorldContext::new(&blob).unwrap();
        let instance = WgpuInstance::instance().unwrap();
        ctx.compile_pipeline(&instance.device).unwrap();
        let world_arc = Arc::new(ctx);

        let executor = ExecutorThread::spawn(
            Arc::new(instance.device.clone()),
            Arc::new(instance.queue.clone()),
            Default::default(),
        ).unwrap();
        let batch = Batch { positions: vec![(0, 0)], max_priority: Priority::Normal };
        let result = executor.request_chunks(world_arc, batch).unwrap();
        let blob_result = result.results.values().next().unwrap();
        assert!(blob_result.verify_crc());

        let bytes = blob_result.as_bytes();
        let payload = &bytes[0x88..];
        let ch0: Vec<f32> = (0..1536)
            .map(|i| f32::from_le_bytes(payload[i*4..i*4+4].try_into().unwrap()))
            .collect();

        let first = (0i32, 1.0f32, 1.0f32, &amps_first[..]);
        let second = Some((3 * 256i32, 1.0f32, 1.0f32, &amps_second[..]));
        let mut max_diff = 0.0f32;
        let mut sum_sq = 0.0f64;
        let mut n = 0usize;
        for cy in 0..96usize {
            for cz in 0..4usize {
                for cx in 0..4usize {
                    let wx = (cx * 4) as f32;
                    let wy = (-64 + cy as i32 * 4) as f32;
                    let wz = (cz * 4) as f32;
                    let cpu = cpu_noise_sampler(&perm_i32, first, second, 0.8, 0.3, 0.3, wx, wy, wz);
                    let gpu = ch0[cx + cz * 4 + cy * 16];
                    let d = (cpu - gpu).abs();
                    sum_sq += (d as f64) * (d as f64);
                    n += 1;
                    if d > max_diff { max_diff = d; }
                }
            }
        }
        let rms = (sum_sq / n as f64).sqrt();
        println!("double-stack/multioctave: max_diff={:.6e} rms={:.6e}", max_diff, rms);
        for &(cx, cy, cz) in &[(0usize,0usize,0usize),(1,10,2),(3,50,1),(2,95,3)] {
            let wx = (cx * 4) as f32;
            let wy = (-64 + cy as i32 * 4) as f32;
            let wz = (cz * 4) as f32;
            let cpu = cpu_noise_sampler(&perm_i32, first, second, 0.8, 0.3, 0.3, wx, wy, wz);
            let gpu = ch0[cx + cz * 4 + cy * 16];
            println!("  cell ({},{},{}) cpu={:.6} gpu={:.6}", cx, cy, cz, cpu, gpu);
        }
        assert!(max_diff < 1e-4, "double-stack divergence {:.6e}", max_diff);
    }

    // CPU reference WITH origin (MC order: maintainPrecision(coord) + origin).
    fn cpu_perlin3d_origin(perm: &[i32], p: i32, ox: f32, oy: f32, oz: f32,
                           x: f32, y: f32, z: f32) -> f32 {
        let px = cpu_maintain_precision(x) + ox;
        let py = cpu_maintain_precision(y) + oy;
        let pz = cpu_maintain_precision(z) + oz;
        cpu_perlin3d_impl(perm, p, px, py, pz)
    }

    fn build_run_blob_origin(
        flags: u8, amplitude: f32, xz: f32, ys: f32,
        first: (i32, f32, f32, &[f32]),
        second: Option<(i32, f32, f32, &[f32])>,
        perms: &[[u32; 256]],
        origins: &[[f64; 3]],
    ) -> Vec<f32> {
        use crate::instance::WgpuInstance;
        use crate::scheduler::batcher::{Batch, Priority};
        use crate::scheduler::executor::ExecutorThread;
        use crate::world::WorldContext;
        use std::sync::Arc;
        let mut blob = Vec::new();
        blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
        blob.push(0xFE);
        blob.extend_from_slice(&[0x01, 0x00]);
        blob.extend_from_slice(&[0x84, 0x00, flags]);
        blob.extend_from_slice(&amplitude.to_le_bytes());
        blob.extend_from_slice(&xz.to_le_bytes());
        blob.extend_from_slice(&ys.to_le_bytes());
        let (fo, ff, fp, fa) = first;
        blob.extend_from_slice(&fo.to_le_bytes());
        blob.extend_from_slice(&ff.to_le_bytes());
        blob.extend_from_slice(&fp.to_le_bytes());
        blob.extend_from_slice(&(fa.len() as i32).to_le_bytes());
        for &a in fa { blob.extend_from_slice(&a.to_le_bytes()); }
        if let Some((so, sf, sp, sa)) = second {
            blob.extend_from_slice(&so.to_le_bytes());
            blob.extend_from_slice(&sf.to_le_bytes());
            blob.extend_from_slice(&sp.to_le_bytes());
            blob.extend_from_slice(&(sa.len() as i32).to_le_bytes());
            for &a in sa { blob.extend_from_slice(&a.to_le_bytes()); }
        }
        blob.extend_from_slice(&(-64i32).to_le_bytes());
        blob.extend_from_slice(&384i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.push(0xF0);
        blob.extend_from_slice(&123456789i64.to_le_bytes());
        blob.extend_from_slice(&0x4D525750u32.to_le_bytes());
        blob.extend_from_slice(&(perms.len() as u32).to_le_bytes());
        for (i, tb) in perms.iter().enumerate() {
            for &v in tb { blob.extend_from_slice(&v.to_le_bytes()); }
            let o = origins.get(i).copied().unwrap_or([0.0, 0.0, 0.0]);
            for k in 0..3 { blob.extend_from_slice(&o[k].to_le_bytes()); }
        }
        let mut ctx = WorldContext::new(&blob).unwrap();
        let instance = WgpuInstance::instance().unwrap();
        ctx.compile_pipeline(&instance.device).unwrap();
        let world_arc = Arc::new(ctx);
        let executor = ExecutorThread::spawn(
            Arc::new(instance.device.clone()),
            Arc::new(instance.queue.clone()),
            Default::default(),
        ).unwrap();
        let batch = Batch { positions: vec![(0, 0)], max_priority: Priority::Normal };
        let result = executor.request_chunks(world_arc, batch).unwrap();
        let br = result.results.values().next().unwrap();
        let bytes = br.as_bytes();
        let payload = &bytes[0x88..];
        (0..1536).map(|i| f32::from_le_bytes(payload[i*4..i*4+4].try_into().unwrap())).collect()
    }

    #[test]
    fn cross_end_mc_reference() {
        use crate::instance::WgpuInstance;
        use crate::scheduler::batcher::{Batch, Priority};
        use crate::scheduler::executor::ExecutorThread;
        use crate::world::WorldContext;
        use std::sync::Arc;

        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../src/test/resources/mc_reference.txt");
        let txt = std::fs::read_to_string(path).expect("read mc_reference.txt");

        let mut perms: Vec<[u32; 256]> = Vec::new();
        let mut origins: Vec<[f64; 3]> = Vec::new();
        let mut amplitude = 1.0f32;
        let mut samples: Vec<(f32, f32, f32, f32)> = Vec::new();
        let mut cur_perm: Option<Vec<u32>> = None;
        let mut cur_perm_idx = 0usize;
        let mut in_samples = false;

        for line in txt.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            let mut it = line.split_whitespace();
            let tag = it.next().unwrap();
            match tag {
                "DOUBLE_AMPLITUDE" => amplitude = it.next().unwrap().parse().unwrap(),
                "SAMPLE_COUNT" => in_samples = true,
                "FIRST_PERM" | "SECOND_PERM" => {
                    let _i: usize = it.next().unwrap().parse().unwrap();
                    let vals: Vec<u32> = it.map(|v| v.parse().unwrap()).collect();
                    assert_eq!(vals.len(), 256, "perm len");
                    cur_perm = Some(vals);
                    cur_perm_idx = 0;
                }
                "FIRST_ORIGIN" | "SECOND_ORIGIN" => {
                    let _i: usize = it.next().unwrap().parse().unwrap();
                    let o: Vec<f64> = it.map(|v| v.parse().unwrap()).collect();
                    if let Some(pv) = cur_perm.take() {
                        let mut arr = [0u32; 256];
                        arr.copy_from_slice(&pv);
                        perms.push(arr);
                    }
                    origins.push([o[0], o[1], o[2]]);
                    let _ = cur_perm_idx;
                }
                _ => {
                    if in_samples {
                        // sample line: wx wy wz value
                        let wx: f32 = tag.parse().unwrap();
                        let wy: f32 = it.next().unwrap().parse().unwrap();
                        let wz: f32 = it.next().unwrap().parse().unwrap();
                        let v: f32 = it.next().unwrap().parse().unwrap();
                        samples.push((wx, wy, wz, v));
                    }
                }
            }
        }
        assert_eq!(perms.len(), 6, "expected 6 perm tables");
        assert_eq!(origins.len(), 6, "expected 6 origins");

        // Build blob: first stack (offset 0), second (offset 768).
        let amps = [1.0f32, 0.5, 0.25];
        let persistence = 4.0f32 / 7.0;   // 2^(n-1)/(2^n-1) for n=3
        let mut blob = Vec::new();
        blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
        blob.push(0xFE);
        blob.extend_from_slice(&[0x01, 0x00]);
        blob.extend_from_slice(&[0x84, 0x00, 0x01]);      // Perlin, has_second
        blob.extend_from_slice(&amplitude.to_le_bytes());
        blob.extend_from_slice(&0.3f32.to_le_bytes());    // xz_scale
        blob.extend_from_slice(&0.3f32.to_le_bytes());    // y_scale
        // first stack
        blob.extend_from_slice(&0i32.to_le_bytes());
        blob.extend_from_slice(&1.0f32.to_le_bytes());    // initial_freq = 2^0
        blob.extend_from_slice(&persistence.to_le_bytes());
        blob.extend_from_slice(&3i32.to_le_bytes());
        for a in amps { blob.extend_from_slice(&a.to_le_bytes()); }
        // second stack
        blob.extend_from_slice(&(3 * 256i32).to_le_bytes());
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.extend_from_slice(&persistence.to_le_bytes());
        blob.extend_from_slice(&3i32.to_le_bytes());
        for a in amps { blob.extend_from_slice(&a.to_le_bytes()); }
        blob.extend_from_slice(&(-64i32).to_le_bytes());
        blob.extend_from_slice(&384i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.push(0xF0);
        blob.extend_from_slice(&123456789i64.to_le_bytes());
        blob.extend_from_slice(&0x4D525750u32.to_le_bytes());
        blob.extend_from_slice(&(perms.len() as u32).to_le_bytes());
        for (i, tb) in perms.iter().enumerate() {
            for &v in tb { blob.extend_from_slice(&v.to_le_bytes()); }
            let o = origins[i];
            for k in 0..3 { blob.extend_from_slice(&o[k].to_le_bytes()); }
        }

        let mut ctx = WorldContext::new(&blob).unwrap();
        let instance = WgpuInstance::instance().unwrap();
        ctx.compile_pipeline(&instance.device).unwrap();
        let world_arc = Arc::new(ctx);
        let executor = ExecutorThread::spawn(
            Arc::new(instance.device.clone()),
            Arc::new(instance.queue.clone()),
            Default::default(),
        ).unwrap();
        let batch = Batch { positions: vec![(0, 0)], max_priority: Priority::Normal };
        let result = executor.request_chunks(world_arc, batch).unwrap();
        let br = result.results.values().next().unwrap();
        let bytes = br.as_bytes();
        let payload = &bytes[0x88..];
        let gpu: Vec<f32> = (0..1536)
            .map(|i| f32::from_le_bytes(payload[i*4..i*4+4].try_into().unwrap()))
            .collect();

        // Map MC world coord -> GPU cell index. cell_x = cx*4, cell_y = -64+cy*4, cell_z = cz*4.
        let mut max_diff = 0.0f32;
        let mut n = 0usize;
        for &(wx, wy, wz, mc) in &samples {
            let cx = (wx / 4.0) as usize;
            let cy = ((wy + 64.0) / 4.0) as usize;
            let cz = (wz / 4.0) as usize;
            if cx >= 4 || cz >= 4 || cy >= 96 { continue; }
            let idx = cx + cz * 4 + cy * 16;
            let d = (gpu[idx] - mc).abs();
            if d > max_diff {
                max_diff = d;
                println!("  worst: world ({},{},{}) mc={:.7} gpu={:.7}", wx, wy, wz, mc, gpu[idx]);
            }
            n += 1;
        }
        println!("cross-end MC vs GPU: n={} max_diff={:.6e}", n, max_diff);
        assert!(n > 0, "no samples mapped");
        assert!(max_diff < 1e-3, "cross-end divergence {:.6e}", max_diff);
    }

    #[test]
    fn numerical_validation_shifted_noise() {
        use crate::instance::WgpuInstance;
        use crate::scheduler::batcher::{Batch, Priority};
        use crate::scheduler::executor::ExecutorThread;
        use crate::world::WorldContext;
        use crate::world::constants::permutation_table_for_seed;
        use std::sync::Arc;

        let perms: Vec<[u32; 256]> = [11i64, 22, 33].iter()
            .map(|&sd| permutation_table_for_seed(sd)).collect();
        let perm_i32: Vec<i32> = perms.iter().flatten().map(|&v| v as i32).collect();

        // Build a blob: ROOT(1ch) = ShiftedNoise(xz=0.25, ys=0.0,
        //   shiftX = ShiftA(offset noise table0),
        //   shiftY = Constant(0.0),
        //   shiftZ = ShiftB(offset noise table1),
        //   noise  = Perlin table2)
        let mut blob = Vec::new();
        blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
        blob.push(0xFE);
        blob.extend_from_slice(&[0x01, 0x00]);
        blob.push(0xB5);
        blob.extend_from_slice(&0.25f32.to_le_bytes());
        blob.extend_from_slice(&0.0f32.to_le_bytes());
        // shiftX = Shift kind=1 (ShiftA)
        blob.push(0xB4); blob.push(1);
        push_simple_noise(&mut blob, 0);
        // shiftY = Constant(0.0)
        blob.push(0x01);
        blob.extend_from_slice(&0.0f32.to_le_bytes());
        // shiftZ = Shift kind=2 (ShiftB)
        blob.push(0xB4); blob.push(2);
        push_simple_noise(&mut blob, 256);
        // main noise
        push_simple_noise(&mut blob, 512);
        // shape
        blob.extend_from_slice(&(-64i32).to_le_bytes());
        blob.extend_from_slice(&384i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        // biome
        blob.push(0xF0);
        blob.extend_from_slice(&123456789i64.to_le_bytes());
        // PERM_RESOURCE
        blob.extend_from_slice(&0x4D525750u32.to_le_bytes());
        blob.extend_from_slice(&(perms.len() as u32).to_le_bytes());
        for tb in &perms {
            for &v in tb { blob.extend_from_slice(&v.to_le_bytes()); }
            for _ in 0..3 { blob.extend_from_slice(&0.0f64.to_le_bytes()); }
        }

        let mut ctx = WorldContext::new(&blob).unwrap();
        let instance = WgpuInstance::instance().unwrap();
        ctx.compile_pipeline(&instance.device).unwrap();
        let world_arc = Arc::new(ctx);
        let executor = ExecutorThread::spawn(
            Arc::new(instance.device.clone()),
            Arc::new(instance.queue.clone()),
            Default::default(),
        ).unwrap();
        // Use chunk coord 1 so world_x = 1 + cx*4 = 1,5,9... (non-grid)
        let batch = Batch { positions: vec![(1, 0)], max_priority: Priority::Normal };
        let result = executor.request_chunks(world_arc, batch).unwrap();
        let br = result.results.values().next().unwrap();
        let bytes = br.as_bytes();
        let payload = &bytes[0x88..];
        let gpu: Vec<f32> = (0..1536)
            .map(|i| f32::from_le_bytes(payload[i*4..i*4+4].try_into().unwrap()))
            .collect();

        // CPU reference (MC algorithm).
        let fx = (0i32, 1.0f32, 1.0f32, &[1.0f32][..]);
        let fz = (256i32, 1.0f32, 1.0f32, &[1.0f32][..]);
        let fnz = (512i32, 1.0f32, 1.0f32, &[1.0f32][..]);
        let shift_a = |x: f32, z: f32| -> f32 {
            cpu_noise_sampler(&perm_i32, fx, None, 1.0, 1.0, 1.0, x*0.25, 0.0, z*0.25) * 4.0
        };
        let shift_b = |x: f32, z: f32| -> f32 {
            cpu_noise_sampler(&perm_i32, fz, None, 1.0, 1.0, 1.0, z*0.25, x*0.25, 0.0) * 4.0
        };
        let cpu = |x: f32, y: f32, z: f32| -> f32 {
            let d = x*0.25 + shift_a(x, z);
            let e = y*0.0 + 0.0;
            let f = z*0.25 + shift_b(x, z);
            cpu_noise_sampler(&perm_i32, fnz, None, 1.0, 1.0, 1.0, d, e, f)
        };

        let mut max_d = 0.0f32;
        for cy in 0..96usize { for cz in 0..4usize { for cx in 0..4usize {
            let wx = (1 + cx as i32 * 4) as f32;
            let wy = (-64 + cy as i32 * 4) as f32;
            let wz = (cz * 4) as f32;
            let c = cpu(wx, wy, wz);
            let g = gpu[cx + cz*4 + cy*16];
            max_d = max_d.max((c - g).abs());
        }}}
        println!("shifted-noise: max_diff={:.6e}", max_d);
        assert!(max_d < 1e-4, "shifted-noise diverged {:.6e}", max_d);
    }

    #[test]
    fn numerical_validation_spline() {
        use crate::instance::WgpuInstance;
        use crate::scheduler::batcher::{Batch, Priority};
        use crate::scheduler::executor::ExecutorThread;
        use crate::world::WorldContext;
        use crate::world::constants::permutation_table_for_seed;
        use std::sync::Arc;

        let perms: Vec<[u32; 256]> = [7i64].iter()
            .map(|&sd| permutation_table_for_seed(sd)).collect();

        // Build: ROOT(1ch) = Spline over Y with 4 nodes:
        //   locations  = [-64, -20, 40, 320]
        //   derivatives= [ 0.05, 0.1, -0.02, 0.0]
        //   location   = InputY
        //   values     = [Constant(0.0), Constant(0.5),
        //                 PerlinNoise(table0), Constant(-0.3)]
        let mut blob = Vec::new();
        blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
        blob.push(0xFE);
        blob.extend_from_slice(&[0x01, 0x00]);

        blob.push(0xB0);           // Spline
        blob.extend_from_slice(&(-64.0f32).to_le_bytes()); // min
        blob.extend_from_slice(&(320.0f32).to_le_bytes()); // max
        blob.extend_from_slice(&4i32.to_le_bytes());       // count
        for l in [-64.0f32, -20.0, 40.0, 320.0] {
            blob.extend_from_slice(&l.to_le_bytes());
        }
        for d in [0.05f32, 0.1, -0.02, 0.0] {
            blob.extend_from_slice(&d.to_le_bytes());
        }
        // location = InputY
        blob.push(0x03);
        // values[0] = Constant(0.0)
        blob.push(0x01);
        blob.extend_from_slice(&0.0f32.to_le_bytes());
        // values[1] = Constant(0.5)
        blob.push(0x01);
        blob.extend_from_slice(&0.5f32.to_le_bytes());
        // values[2] = Perlin noise table0
        push_simple_noise(&mut blob, 0);
        // values[3] = Constant(-0.3)
        blob.push(0x01);
        blob.extend_from_slice(&(-0.3f32).to_le_bytes());

        // shape + biome
        blob.extend_from_slice(&(-64i32).to_le_bytes());
        blob.extend_from_slice(&384i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.push(0xF0);
        blob.extend_from_slice(&123456789i64.to_le_bytes());
        // PERM_RESOURCE
        blob.extend_from_slice(&0x4D525750u32.to_le_bytes());
        blob.extend_from_slice(&(perms.len() as u32).to_le_bytes());
        for tb in &perms {
            for &v in tb { blob.extend_from_slice(&v.to_le_bytes()); }
            for _ in 0..3 { blob.extend_from_slice(&0.0f64.to_le_bytes()); }
        }

        let mut ctx = WorldContext::new(&blob).unwrap();
        let instance = WgpuInstance::instance().unwrap();
        ctx.compile_pipeline(&instance.device).unwrap();
        let world_arc = Arc::new(ctx);
        let executor = ExecutorThread::spawn(
            Arc::new(instance.device.clone()),
            Arc::new(instance.queue.clone()),
            Default::default(),
        ).unwrap();
        let batch = Batch { positions: vec![(1, 0)], max_priority: Priority::Normal };
        let result = executor.request_chunks(world_arc, batch).unwrap();
        let br = result.results.values().next().unwrap();
        let bytes = br.as_bytes();
        let payload = &bytes[0x88..];
        let gpu: Vec<f32> = (0..1536)
            .map(|i| f32::from_le_bytes(payload[i*4..i*4+4].try_into().unwrap()))
            .collect();

        let perm_i32: Vec<i32> = perms.iter().flatten().map(|&v| v as i32).collect();
        let locs = [-64.0f32, -20.0, 40.0, 320.0];
        let ders = [0.05f32, 0.1, -0.02, 0.0];
        let cpu = |y: f32, wx: f32, wz: f32| -> f32 {
            let v: [f32; 4] = [
                0.0,
                0.5,
                cpu_noise_sampler(&perm_i32, (0, 1.0, 1.0, &[1.0f32][..]), None, 1.0, 1.0, 1.0, wx, y, wz),
                -0.3,
            ];
            let outside = |v: f32, d: f32, l: f32, t: f32| if d == 0.0 { v } else { v + d*(t-l) };
            if y < locs[0] { return outside(v[0], ders[0], locs[0], y); }
            if y >= locs[3] { return outside(v[3], ders[3], locs[3], y); }
            for k in 0..3 {
                if y < locs[k+1] {
                    let p = (y - locs[k]) / (locs[k+1] - locs[k]);
                    let a = ders[k]*(locs[k+1]-locs[k]) - (v[k+1]-v[k]);
                    let b = -ders[k+1]*(locs[k+1]-locs[k]) + (v[k+1]-v[k]);
                    let lerp = |t: f32, a: f32, b: f32| a + t*(b-a);
                    return lerp(p, v[k], v[k+1]) + p*(1.0-p)*lerp(p, a, b);
                }
            }
            v[3]
        };

        let mut max_d = 0.0f32;
        for cy in 0..96usize { for cz in 0..4usize { for cx in 0..4usize {
            let wx = (1 + cx as i32 * 4) as f32;
            let wy = (-64 + cy as i32 * 4) as f32;
            let wz = (cz * 4) as f32;
            let c = cpu(wy, wx, wz);
            let g = gpu[cx + cz*4 + cy*16];
            max_d = max_d.max((c - g).abs());
        }}}
        println!("spline: max_diff={:.6e}", max_d);
        assert!(max_d < 1e-4, "spline diverged {:.6e}", max_d);
    }

        fn push_simple_noise(blob: &mut Vec<u8>, perm_offset: i32) {
        blob.push(0x84);          // TAG_NOISE_SAMPLER
        blob.push(0);             // type = Perlin
        blob.push(0);             // flags = no second stack
        blob.extend_from_slice(&1.0f32.to_le_bytes()); // amplitude
        blob.extend_from_slice(&1.0f32.to_le_bytes()); // xz_scale
        blob.extend_from_slice(&1.0f32.to_le_bytes()); // y_scale
        blob.extend_from_slice(&perm_offset.to_le_bytes());
        blob.extend_from_slice(&1.0f32.to_le_bytes()); // initial_freq
        blob.extend_from_slice(&1.0f32.to_le_bytes()); // persistence
        blob.extend_from_slice(&1i32.to_le_bytes());   // octaves
        blob.extend_from_slice(&1.0f32.to_le_bytes()); // amp[0]
    }

    #[test]
    fn numerical_validation_nonzero_origin() {
        use crate::world::constants::permutation_table_for_seed;
        // 3 octaves, each with a distinct non-zero origin.
        let perms: Vec<[u32; 256]> = [11i64, 22, 33].iter()
            .map(|&sd| permutation_table_for_seed(sd)).collect();
        let perm_i32: Vec<i32> = perms.iter().flatten().map(|&v| v as i32).collect();
        let origins: Vec<[f64; 3]> = vec![
            [1.5, -2.25, 3.75],
            [-0.125, 7.5, 0.0625],
            [100.0, -50.5, 0.25],
        ];
        let amps = [1.0f32, 0.5, 0.25];
        let gpu = build_run_blob_origin(0, 1.0, 0.3, 0.3,
            (0, 1.0, 1.0, &amps), None, &perms, &origins);
        let mut max_d = 0.0f32;
        for cy in 0..96usize { for cz in 0..4usize { for cx in 0..4usize {
            let wx = (cx * 4) as f32;
            let wy = (-64 + cy as i32 * 4) as f32;
            let wz = (cz * 4) as f32;
            let mut acc = (0.0f32, 0.0f32);
            let mut freq = 1.0f32; let mut amp = 1.0f32;
            for i in 0..3usize {
                let p = (i as i32) * 256;
                let o = origins[i];
                let s = cpu_perlin3d_origin(&perm_i32, p, o[0] as f32, o[1] as f32, o[2] as f32,
                                            wx*0.3*freq, wy*0.3*freq, wz*0.3*freq);
                cpu_kahan_add(&mut acc, amps[i] * s * amp);
                freq *= 2.0; amp *= 0.5;
            }
            let g = gpu[cx + cz*4 + cy*16];
            max_d = max_d.max((acc.0 - g).abs());
        }}}
        println!("nonzero-origin (3 octaves): max_diff={:.6e}", max_d);
        assert!(max_d < 1e-4, "origin path diverged {:.6e}", max_d);
    }

    #[test]
    fn numerical_validation_noise_vs_gpu() {
        use crate::instance::WgpuInstance;
        use crate::scheduler::batcher::{Batch, Priority};
        use crate::scheduler::executor::ExecutorThread;
        use crate::world::WorldContext;
        use crate::world::constants::permutation_table_for_seed;
        use std::sync::Arc;

        let perm_u32 = permutation_table_for_seed(42);
        let perm_i32: Vec<i32> = perm_u32.iter().map(|&v| v as i32).collect();

        // Build minimal blob: ROOT(1 ch) = NoiseSampler{Perlin, single stack}.
        let mut blob = Vec::new();
        blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
        blob.push(0xFE);
        blob.extend_from_slice(&[0x01, 0x00]);
        blob.extend_from_slice(&[0x84, 0x00, 0x00]);
        blob.extend_from_slice(&1.0f32.to_le_bytes());  // amplitude
        blob.extend_from_slice(&0.3f32.to_le_bytes());  // xz_scale
        blob.extend_from_slice(&0.3f32.to_le_bytes());  // y_scale
        blob.extend_from_slice(&0i32.to_le_bytes());    // perm_offset
        blob.extend_from_slice(&1.0f32.to_le_bytes());  // initial_freq
        blob.extend_from_slice(&1.0f32.to_le_bytes());  // persistence
        blob.extend_from_slice(&1i32.to_le_bytes());    // octaves
        blob.extend_from_slice(&1.0f32.to_le_bytes());  // amp[0]
        blob.extend_from_slice(&(-64i32).to_le_bytes());
        blob.extend_from_slice(&384i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.push(0xF0);
        blob.extend_from_slice(&123456789i64.to_le_bytes());
        // PERM_RESOURCE
        blob.extend_from_slice(&0x4D525750u32.to_le_bytes());
        blob.extend_from_slice(&1u32.to_le_bytes());
        for &p in &perm_u32 { blob.extend_from_slice(&p.to_le_bytes()); }
        for _ in 0..3 { blob.extend_from_slice(&0.0f64.to_le_bytes()); }

        let mut ctx = WorldContext::new(&blob).unwrap();
        let instance = WgpuInstance::instance().unwrap();
        ctx.compile_pipeline(&instance.device).unwrap();
        let world_arc = Arc::new(ctx);

        let executor = ExecutorThread::spawn(
            Arc::new(instance.device.clone()),
            Arc::new(instance.queue.clone()),
            Default::default(),
        ).unwrap();

        let batch = Batch { positions: vec![(0, 0)], max_priority: Priority::Normal };
        let result = executor.request_chunks(world_arc, batch).unwrap();
        let blob_result = result.results.values().next().unwrap();
        assert!(blob_result.verify_crc());

        let bytes = blob_result.as_bytes();
        let payload = &bytes[0x88..];
        let ch0: Vec<f32> = (0..1536).map(|i| {
            f32::from_le_bytes(payload[i*4..i*4+4].try_into().unwrap())
        }).collect();

        let mut max_diff = 0.0f32;
        let mut max_cell = (0usize, 0usize, 0usize);
        let mut sum_sq = 0.0f64;
        let mut n = 0usize;
        for cy in 0..96usize {
            for cz in 0..4usize {
                for cx in 0..4usize {
                    let wx = (cx * 4) as f32;
                    let wy = (-64 + cy as i32 * 4) as f32;
                    let wz = (cz * 4) as f32;
                    let cpu = cpu_noise_perlin3d(&perm_i32, 0, wx * 0.3, wy * 0.3, wz * 0.3);
                    let gpu_idx = cx + cz * 4 + cy * 16;
                    let gpu = ch0[gpu_idx];
                    let d = (cpu - gpu).abs();
                    sum_sq += (d as f64) * (d as f64);
                    n += 1;
                    if d > max_diff { max_diff = d; max_cell = (cx, cy, cz); }
                }
            }
        }
        let rms = (sum_sq / n as f64).sqrt();
        println!("GPU vs CPU: max_diff={:.6e} at {:?}, rms={:.6e}", max_diff, max_cell, rms);
        for &(cx, cy, cz) in &[(0usize,0usize,0usize),(1,0,0),(2,0,0),(3,0,0),(1,30,1),(3,95,3)] {
            let wx = (cx * 4) as f32;
            let wy = (-64 + cy as i32 * 4) as f32;
            let wz = (cz * 4) as f32;
            let cpu = cpu_noise_perlin3d(&perm_i32, 0, wx * 0.3, wy * 0.3, wz * 0.3);
            let gpu_idx = cx + cz * 4 + cy * 16;
            println!("  cell ({},{},{}) world ({},{},{}) cpu={:.6} gpu={:.6}",
                     cx, cy, cz, wx, wy, wz, cpu, ch0[gpu_idx]);
        }
        assert!(max_diff < 1e-4, "GPU vs CPU divergence {:.6e}", max_diff);
    }

}
