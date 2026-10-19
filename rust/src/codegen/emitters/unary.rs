// SPDX-License-Identifier: MIT

use crate::ast::AstNode;
use crate::codegen::emitters::emit_node;
use crate::codegen::wgsl_gen::GenCtx;
use crate::Result;
use anyhow::anyhow;

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    match node {
        AstNode::Abs(a) => un(ctx, a, "abs"),
        AstNode::Square(a) => {
            let va = emit_node(a, ctx)?;
            let var = ctx.fresh_var("sq");
            ctx.emit(format!("let {var}: f32 = {va} * {va};"));
            Ok(var)
        }
        AstNode::Cube(a) => {
            let va = emit_node(a, ctx)?;
            let var = ctx.fresh_var("cu");
            ctx.emit(format!("let {var}: f32 = {va} * {va} * {va};"));
            Ok(var)
        }
        AstNode::Sqrt(a) => un(ctx, a, "sqrt"),
        AstNode::FloorDiv(a, b) => {
            let va = emit_node(a, ctx)?;
            let vb = emit_node(b, ctx)?;
            let var = ctx.fresh_var("fdiv");
            ctx.emit(format!("let {var}: f32 = floorDiv({va}, {vb});"));
            Ok(var)
        }
        _ => return Err(anyhow!("unary emitter got non-unary node: {node:?}")),
    }
}
fn un(ctx: &mut GenCtx, a: &AstNode, f: &str) -> Result<String> {
    let va = emit_node(a, ctx)?;
    let var = ctx.fresh_var("un");
    ctx.emit(format!("let {var}: f32 = {f}({va});"));
    Ok(var)
}
