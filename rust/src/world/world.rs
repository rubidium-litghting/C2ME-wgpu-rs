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
mod tests {
    use super::*;
    use crate::ast::AstNode;

    /// End-to-end phase 2 smoke test: take the minimal simple-root blob
    /// from the deserialize test suite, run WorldContext::new(), and assert
    /// the shape config is preserved and fingerprint is non-zero.
    #[test]
    fn world_context_happy_path_from_minimal_blob() {
        let blob: &[u8] = &[
            // HEADER
            0x4D,0x43,0x32,0x57,  0x01,0x00,  0x00,0x00,0x00,0x00,
            // ROOT(2 ch): Constant(3.14) + InputX
            0xFE, 0x02,0x00,
                0x01, 0xC3,0xF5,0x48,0x40,
                0x02,
            // SHAPE_CONFIG: min_y=-64, height=384, cw=4, ch=4
            0xC0,0xFF,0xFF,0xFF,  0x80,0x01,0x00,0x00,
            0x04,0x00,0x00,0x00,  0x04,0x00,0x00,0x00,
            // BIOME
            0xF0, 0x15, 0xCD, 0x5B, 0x07, 0x00, 0x00, 0x00, 0x00, 0x15, 0xCD, 0x5B, 0x07, 0x00, 0x00, 0x00, 0x00,
        ];

        let ctx = WorldContext::new(blob).expect("should build WorldContext");

        // Shape preserved
        assert_eq!(ctx.shape().min_y, -64);
        assert_eq!(ctx.shape().height, 384);
        assert_eq!(ctx.shape().cell_width, 4);
        assert_eq!(ctx.shape().cell_height, 4);

        // Fingerprint is non-default (not 0 — extremely rare if 0 anyway)
        assert_ne!(ctx.fingerprint(), 0,
            "fingerprint should be some non-zero value for a non-trivial AST");

        // AST root is a Root(2)
        assert!(matches!(&ctx.ast_root, AstNode::Root(v) if v.len() == 2),
            "expected Root with 2 children, got {:?}", &ctx.ast_root);
    }

    /// `WorldContext::new` does NOT compile the GPU pipeline — `pipeline`
    /// must start as `None` so unit tests can run without a GPU.
    /// `compile_pipeline` is the explicit GPU-touching step, deferred to
    /// when `WgpuInstance` is available.
    #[test]
    fn pipeline_is_none_until_compile_pipeline_called() {
        let blob: &[u8] = &[
            0x4D,0x43,0x32,0x57,  0x01,0x00,  0x00,0x00,0x00,0x00,
            0xFE, 0x01,0x00,
                0x01, 0x00,0x00,0x80,0x3F,  // Constant(1.0)
            0x00,0x00,0x00,0x00,  0x80,0x01,0x00,0x00,
            0x04,0x00,0x00,0x00,  0x04,0x00,0x00,0x00,
            0xF0, 0x15, 0xCD, 0x5B, 0x07, 0x00, 0x00, 0x00, 0x00, 0x15, 0xCD, 0x5B, 0x07, 0x00, 0x00, 0x00, 0x00,
        ];
        let ctx = WorldContext::new(blob).expect("should build WorldContext");
        assert!(ctx.pipeline().is_none(),
            "pipeline must be None before compile_pipeline is called");
    }

    /// Phase 4-3: a WorldContext whose AST contains `CacheInterpolated`
    /// runs in **single-pass** mode (corner computation + workgroupBarrier
    /// + trilinear lerp within one dispatch — workgroup memory does NOT
    /// persist across dispatches, so two-pass was incorrect).
    ///
    /// Verifies: `requires_two_pass() == false`, single-pass `shader_src`
    /// is populated with `main_entry`, and the pipeline starts as `None`.
    ///
    /// The inner subtree is `Add(InputX, InputY)` — a simple coordinate
    /// expression that the corner loop can safely evaluate per corner
    /// (no nested Cache2D/CacheFlat, which would break the loop model).
    #[test]
    fn cache_interpolated_world_context_single_pass() {
        let blob: &[u8] = &[
            // HEADER
            0x4D,0x43,0x32,0x57,  0x01,0x00,  0x00,0x00,0x00,0x00,
            // ROOT(1 ch): CacheInterpolated(Add(InputX, InputY))
            0xFE, 0x01,0x00,
                0xC2,            // CacheInterpolated
                    0x10,        // Add
                        0x02,     // InputX
                        0x03,     // InputY
            // SHAPE_CONFIG: min_y=-64, height=384, cw=4, ch=4
            0xC0,0xFF,0xFF,0xFF,  0x80,0x01,0x00,0x00,
            0x04,0x00,0x00,0x00,  0x04,0x00,0x00,0x00,
            // BIOME
            0xF0, 0x15, 0xCD, 0x5B, 0x07, 0x00, 0x00, 0x00, 0x00, 0x15, 0xCD, 0x5B, 0x07, 0x00, 0x00, 0x00, 0x00,
        ];

        let ctx = WorldContext::new(blob)
            .expect("should build single-pass WorldContext");

        // Single-pass mode — CacheInterpolated works with workgroupBarrier().
        // (requires_two_pass is always false as of phase 4-3.)
        assert!(!crate::codegen::wgsl_gen::requires_two_pass(&ctx.ast_root),
            "CacheInterpolated must NOT require two-pass (single-pass + barrier)");

        // Single-pass shader source is populated with main_entry + the
        // CacheInterpolated shared-memory + trilinear_lerp call.
        assert!(!ctx.shader_src.is_empty(),
            "single-pass shader_src must be emitted");
        assert!(ctx.shader_src.contains("fn main_entry("),
            "shader must define main_entry entry point");
        assert!(ctx.shader_src.contains("var<workgroup> ci_0: array<f32, 125>;"),
            "shader must declare 5×5×5 CacheInterpolated shared memory");
        assert!(ctx.shader_src.contains("trilinear_lerp("),
            "shader must call trilinear_lerp after workgroupBarrier");

        // Pipeline not compiled yet (no GPU in unit tests).
        assert!(ctx.pipeline().is_none(),
            "pipeline must be None before compile_pipeline");
    }

    /// `collect_perlin_seeds` walks an AST and picks up every
    /// `PerlinSampler.seed`, including ones nested inside Add/Cache/etc.
    #[test]
    fn collect_perlin_seeds_walks_ast() {
        use crate::ast::AstNode;
        let root = AstNode::Root(vec![
            AstNode::Constant(0.0),
            AstNode::Add(
                Box::new(AstNode::PerlinSampler {
                    seed: 100, octaves: 1,
                    amplitudes: vec![1.0], lacunaritys: vec![2.0],
                }),
                Box::new(AstNode::Mul(
                    Box::new(AstNode::PerlinSampler {
                        seed: 200, octaves: 2,
                        amplitudes: vec![1.0, 0.5], lacunaritys: vec![2.0, 2.0],
                    }),
                    Box::new(AstNode::InputX),
                )),
            ),
            AstNode::Cache2D(Box::new(AstNode::PerlinSampler {
                seed: 100, // duplicate of seed 100 above
                octaves: 1, amplitudes: vec![1.0], lacunaritys: vec![2.0],
            })),
        ]);

        let seeds = collect_perlin_seeds(&root);
        assert_eq!(seeds, vec![100, 200, 100],
            "should walk all PerlinSampler nodes in pre-order");
    }

    /// `WorldContext::new` builds a `perm_table_set` from the AST's seeds.
    /// The minimal blob has no PerlinSampler → empty perm_table_set.
    #[test]
    fn world_context_perm_table_set_empty_for_minimal_blob() {
        let blob: &[u8] = &[
            0x4D,0x43,0x32,0x57,  0x01,0x00,  0x00,0x00,0x00,0x00,
            0xFE, 0x01,0x00,
                0x01, 0x00,0x00,0x80,0x3F,
            0x00,0x00,0x00,0x00,  0x80,0x01,0x00,0x00,
            0x04,0x00,0x00,0x00,  0x04,0x00,0x00,0x00,
            0xF0, 0x15, 0xCD, 0x5B, 0x07, 0x00, 0x00, 0x00, 0x00, 0x15, 0xCD, 0x5B, 0x07, 0x00, 0x00, 0x00, 0x00,
        ];
        let ctx = WorldContext::new(blob).expect("should build WorldContext");
        assert_eq!(ctx.perm_table_set().len_i32(), 0,
            "minimal blob has no PerlinSampler, perm_table_set must be empty");
        assert!(ctx.perm_buffer().is_none(),
            "perm_buffer must be None before compile_pipeline is called");
    }

    /// Two semantically equivalent blobs (commutative + normalize) must
    /// produce the same fingerprint after the opto pipeline runs.
    ///
    /// This is the actual cache-validity contract of phase 3: changing Java
    /// constant argument order (which happens across MC patch builds or
    /// datapacks) must NOT trigger a pipeline re-compile.
    #[test]
    fn equivalent_blobs_share_fingerprint() {
        // Blob A: Root(1 ch) — Mul(InputX, Constant(2))
        let a = make_blob_1ch(|buf| {
            // Mul (0x12) — InputX then Constant(2.0)
            buf.extend_from_slice(&[0x12, 0x02, 0x01]);
            buf.extend_from_slice(&2.0f32.to_le_bytes());
        });
        // Blob B: Root(1 ch) — Mul(Constant(2), InputX)
        let b = make_blob_1ch(|buf| {
            // Mul — Constant(2.0) then InputX
            buf.extend_from_slice(&[0x12, 0x01]);
            buf.extend_from_slice(&2.0f32.to_le_bytes());
            buf.push(0x02);
        });

        let ctx_a = WorldContext::new(&a).unwrap();
        let ctx_b = WorldContext::new(&b).unwrap();

        // → opto pipeline (normalize) makes the two structurally equal,
        //   so their fingerprints are identical → pipeline cache hit!
        assert_eq!(ctx_a.fingerprint(), ctx_b.fingerprint(),
            "Mul(X,2) and Mul(2,X) must produce the same post-opto fingerprint");
    }

    // Helper: build a full blob with a ROOT(1 ch) whose body is written by the user closure.
    fn make_blob_1ch<F>(write_ch0: F) -> Vec<u8>
    where
        F: FnOnce(&mut Vec<u8>),
    {
        let mut v = Vec::new();
        v.extend_from_slice(&[0x4D,0x43,0x32,0x57, 0x01,0x00, 0x00,0x00,0x00,0x00]);
        v.extend_from_slice(&[0xFE, 0x01, 0x00]);
        write_ch0(&mut v);
        // SHAPE_CONFIG: 16 zeros
        v.extend_from_slice(&[0u8; 16]);
        // BIOME: 0xF0, 0x15, 0xCD, 0x5B, 0x07, 0x00, 0x00, 0x00, 0x00
        v.push(0xF0); v.push(0x15); v.push(0xCD); v.push(0x5B); v.push(0x07); v.push(0x00); v.push(0x00); v.push(0x00); v.push(0x00);
        v
    }
}

