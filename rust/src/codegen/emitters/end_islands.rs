// SPDX-License-Identifier: MIT

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
        "let {var}: f32 = (end_islands({perm_index}, i32(cell_x), i32(cell_z)) - 8.0) / 128.0;"
    ));
    Ok(var)
}
