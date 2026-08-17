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

//! Root emitter — binds all 16 density channels to the MC-native-stride
//! output buffer at their correct indices.
//!
//! This also declares the `cell_x / cell_y / cell_z` local-offset variables
//! that the coordinate emitter returns references to.

use crate::ast::AstNode;
use crate::codegen::emitters::emit_node;
use crate::codegen::wgsl_gen::GenCtx;
use crate::Result;
use anyhow::anyhow;

/// Emit each Root channel via `emit_node`, then write its value into
/// `out_buf[channel_idx * cells_per_chunk + cell_idx]` (channel-major
/// stride, matching `mc_stride.rs::cell_index`). Assumes `main_entry`
/// scope already declares `cell_idx` and `cells_per_chunk` (provided by
/// the prologue constants in `wgsl_gen.rs`).
pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    let AstNode::Root(channels) = node else {
        return Err(anyhow!("root emitter: expected AstNode::Root, got {node:?}"));
    };

    for (i, child) in channels.iter().enumerate() {
        let var = emit_node(child, ctx)?;
        let chan_u32 = i as u32;
        ctx.emit(format!(
            "  out_buf[{chan_u32}u * cells_per_chunk + cell_idx] = {var};"
        ));
    }

    // Root is an aggregate — no single value. The sentinel below should
    // never be referenced (Root only ever appears at the AST top).
    Ok("__root_aggregate".into())
}
