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
use crate::codegen::emitters::emit_node;
use crate::codegen::wgsl_gen::GenCtx;
use crate::Result;

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    match node {
        AstNode::Add(a, b) => bin(ctx, a, b, "+"),
        AstNode::Sub(a, b) => bin(ctx, a, b, "-"),
        AstNode::Mul(a, b) => bin(ctx, a, b, "*"),
        AstNode::Div(a, b) => bin(ctx, a, b, "/"),
        AstNode::Min(a, b) => bin_fn(ctx, a, b, "min"),
        AstNode::Max(a, b) => bin_fn(ctx, a, b, "max"),
        _ => unreachable!("binary emitter got non-binary node"),
    }
}

fn bin(ctx: &mut GenCtx, a: &AstNode, b: &AstNode, op: &str) -> Result<String> {
    let va = emit_node(a, ctx)?;
    let vb = emit_node(b, ctx)?;
    let var = ctx.fresh_var("bin");
    ctx.emit(format!("let {var}: f32 = {va} {op} {vb};"));
    Ok(var)
}
fn bin_fn(ctx: &mut GenCtx, a: &AstNode, b: &AstNode, f: &str) -> Result<String> {
    let va = emit_node(a, ctx)?;
    let vb = emit_node(b, ctx)?;
    let var = ctx.fresh_var("bin");
    ctx.emit(format!("let {var}: f32 = {f}({va}, {vb});"));
    Ok(var)
}
