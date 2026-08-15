/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2021-2026 RubidiumLighting
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

//! Coordinate inputs — this is where 铁律① local-offset is applied.
//!
//! Every reference to InputX / InputY / InputZ in the AST reads from
//! pre-declared `cell_x`/`cell_y`/`cell_z` f32 variables which are ALREADY
//! the origin-subtracted small magnitude values. The absolute world
//! coordinate is recovered in the individual noise emitter via
//! `noise_position_with_origin()`.

use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use crate::Result;

pub fn emit(node: &AstNode, _ctx: &mut GenCtx) -> Result<String> {
    // These variable names are declared once in main_entry() scope by root emitter.
    Ok(match node {
        AstNode::InputX => "cell_x".into(),
        AstNode::InputY => "cell_y".into(),
        AstNode::InputZ => "cell_z".into(),
        _ => unreachable!(),
    })
}
