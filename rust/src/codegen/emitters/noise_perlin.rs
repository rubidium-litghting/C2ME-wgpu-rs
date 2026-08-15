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

//! PerlinSampler / DoublePerlin emitter.
//!
//! 铁律② applies: when octaves ≥ 3, the generated code MUST use
//! `kahan_add()` loop; plain `accum += amp * sample;` is forbidden.

use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use crate::Result;

pub fn emit(node: &AstNode, _ctx: &mut GenCtx) -> Result<String> {
    match node {
        AstNode::PerlinSampler { octaves, .. } => {
            // TODO: declare amplitudes[]/lacunaritys[] via @group(0) @binding(1) permutation params
            // if *octaves >= 3 => generate kahan loop (铁律②)
            let _ = octaves;
            Ok("perlin_stub".into())
        }
        AstNode::DoublePerlin { .. } => Ok("double_perlin_stub".into()),
        _ => unreachable!(),
    }
}
