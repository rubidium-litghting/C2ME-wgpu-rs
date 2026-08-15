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

//! Top-level WGSL code generator.
//!
//! Produces one shader module per WorldContext:
//!   math_lib.wgsl  +  per-channel let bindings  +  @compute main entry.
//!
//! The main entry dispatches workgroup (chunk_x, chunk_z, 1) over the
//! cell grid, then writes 16 f32 density channels per (cellX,cellY,cellZ)
//! into the output storage buffer in MC's native stride (see mc_stride.rs).

use crate::ast::AstNode;
use crate::Result;

/// Code generation context — accumulates WGSL source lines + var counter.
pub struct GenCtx {
    pub lines: Vec<String>,
    pub next_var: u32,
    // TODO: CSE let-binding dedup map: HashMap<hash_of_subtree, var_name>
}

impl GenCtx {
    pub fn new() -> Self {
        Self {
            lines: Vec::new(),
            next_var: 0,
        }
    }
    pub fn fresh_var(&mut self, prefix: &str) -> String {
        let n = self.next_var;
        self.next_var += 1;
        format!("{prefix}_{n}")
    }
    pub fn emit(&mut self, s: impl Into<String>) {
        self.lines.push(s.into());
    }
}

/// Emit a full shader module string for the given AST root.
pub fn emit_shader(root: &AstNode) -> Result<String> {
    let mut ctx = GenCtx::new();

    // TODO:
    //   1. Push MATH_LIB_WGSL verbatim
    //   2. Push the @group bindings header
    //   3. Iterate Root's 16 channels, emit each, store into output slot
    //   4. Push @compute main: for each cell in workgroup → write stride

    // Skeleton: just emit the library header + a no-op main so compilation works.
    ctx.emit(super::MATH_LIB_WGSL);
    ctx.emit(r#"@group(0) @binding(0) var<storage, read> params: array<u32>;
        @group(0) @binding(1) var<storage, read> permutations: array<i32>;
        @group(0) @binding(2) var<storage, read_write> out_buf: array<f32>;

        @compute @workgroup_size(8, 8, 1)
        fn main_entry(@builtin(global_invocation_id) gid: vec3<u32>) {
            // skeleton: do nothing
            _ = params[0];
        }"#);

    Ok(ctx.lines.join("\n"))
}
