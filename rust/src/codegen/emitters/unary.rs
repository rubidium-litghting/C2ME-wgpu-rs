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
