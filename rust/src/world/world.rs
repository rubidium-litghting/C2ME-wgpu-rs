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

use crate::ast::deserialize::ShapeConfig;
use crate::Result;

pub struct WorldContext {
    /// Parsed + optimized AST root (16-channel Root) — kept for fingerprinting.
    #[allow(dead_code)]
    ast_root: (),
    /// Parsed shape config.
    #[allow(dead_code)]
    shape: ShapeConfig,
    // TODO real fields:
    //   pub pipeline: wgpu::ComputePipeline,
    //   pub bind_group_layout: wgpu::BindGroupLayout,
    //   pub params_bg: wgpu::BindGroup,         (constants + noise params)
    //   pub permutations_bg: wgpu::BindGroup,    (perlin permutation tables)
    //   pub fingerprint: u64,
}

impl WorldContext {
    /// Parse the Java-side blob, run AST opto passes, build shader,
    /// compile pipeline (try pipeline_cache first).
    pub fn new(_config_blob: &[u8]) -> Result<Self> {
        // TODO skeleton:
        //   1. deserialize_full(blob) -> (ast, shape)
        //   2. ast = opto::run_pipeline(ast)?
        //   3. fingerprint = fingerprint(ast, adapter, workarounds)
        //   4. try pipeline_cache.load(fingerprint).ok() -> Ok skip compile
        //   5. else emit_shader(&ast) -> create_shader_module -> create_compute_pipeline
        //   6. cache.save(&pipeline_data)
        //   7. build bind_group_layout + default bind groups
        Ok(Self {
            ast_root: (),
            shape: ShapeConfig { min_y: -64, height: 384, cell_width: 4, cell_height: 4 },
        })
    }
}
