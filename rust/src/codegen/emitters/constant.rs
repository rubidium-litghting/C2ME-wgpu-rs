// SPDX-License-Identifier: MIT

use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use crate::Result;

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    match node {
        AstNode::Constant(v) => {
            // Emit f32 literal; use precise hex representation via `{v:.9e}`
            // so bit-exact round-trip works for e.g. f32::MIN_POSITIVE.
            let var = ctx.fresh_var("c");
            ctx.emit(format!("let {var}: f32 = {v:.9e};"));
            Ok(var)
        }
        _ => unreachable!(),
    }
}
