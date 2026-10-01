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

use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use crate::Result;
use anyhow::anyhow;

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    let AstNode::EndIslands { perm_index } = node else {
        return Err(anyhow!("end_islands emitter called on non-EndIslands node"));
    };
    let var = ctx.fresh_var("ei");
    // MC EndIslands.sample: (sample2d(sampler, blockX/8, blockZ/8) - 8) / 128
    // The WGSL helper end_islands(perm_index, x, z) returns the RAW sample2d
    // value; the -8 /128 transform is applied here.
    ctx.emit(format!(
        "let {var}: f32 = (end_islands({perm_index}, cell_x, cell_z) - 8.0) / 128.0;"
    ));
    Ok(var)
}
