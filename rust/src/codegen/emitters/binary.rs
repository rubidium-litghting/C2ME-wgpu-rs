// SPDX-License-Identifier: MIT

use crate::ast::AstNode;
use crate::codegen::emitters::emit_node;
use crate::codegen::wgsl_gen::GenCtx;
use crate::Result;
use anyhow::anyhow;

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    match node {
        AstNode::Add(a, b) => infix(ctx, a, b, "+"),
        AstNode::Sub(a, b) => infix(ctx, a, b, "-"),
        AstNode::Mul(a, b) => infix(ctx, a, b, "*"),
        AstNode::Div(a, b) => infix(ctx, a, b, "/"),
        AstNode::Min(a, b) => call(ctx, a, b, "min"),
        AstNode::Max(a, b) => call(ctx, a, b, "max"),
        _ => Err(anyhow!("binary emitter got non-binary node: {node:?}")),
    }
}

/// Emit `let v: f32 = <a> <op> <b>;` for WGSL's infix operators.
fn infix(ctx: &mut GenCtx, a: &AstNode, b: &AstNode, op: &str) -> Result<String> {
    let va = emit_node(a, ctx)?;
    let vb = emit_node(b, ctx)?;
    let var = ctx.fresh_var("bin");
    ctx.emit(format!("let {var}: f32 = {va} {op} {vb};"));
    Ok(var)
}

/// Emit `let v: f32 = <f>(<a>, <b>);` for WGSL builtin calls.
fn call(ctx: &mut GenCtx, a: &AstNode, b: &AstNode, f: &str) -> Result<String> {
    let va = emit_node(a, ctx)?;
    let vb = emit_node(b, ctx)?;
    let var = ctx.fresh_var("bin");
    ctx.emit(format!("let {var}: f32 = {f}({va}, {vb});"));
    Ok(var)
}
