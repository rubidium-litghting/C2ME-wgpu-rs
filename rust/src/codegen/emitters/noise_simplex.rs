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

//! SimplexNoise2D / SimplexNoise3D emitter.
//!
//! MC's SimplexNoiseSampler is a *single-octave* noise:
//!   * no `amplitudes[]` / `lacunaritys[]`
//!   * no Kahan summation (铁律② does not apply)
//!   * direct call to `math_lib.wgsl::noise_simpleXd(seed, x, y[, z])`
//!
//! Same i64→i32 seed truncation caveat as Perlin (see `noise_perlin.rs`
//! module-level docstring): the real seed only matters at permutation-table
//! construction time.  This will be reworked when `noise_simplex2d/3d`
//! switches to a permutation-buffer-offset signature.

use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use crate::Result;
use anyhow::anyhow;

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    match node {
        AstNode::SimplexNoise2D { seed } => emit_simplex2d(ctx, *seed),
        AstNode::SimplexNoise3D { seed } => emit_simplex3d(ctx, *seed),
        _ => Err(anyhow!("noise_simplex emitter called on non-simplex node: {node:?}")),
    }
}

fn emit_simplex2d(ctx: &mut GenCtx, seed: i64) -> Result<String> {
    let seed_i32 = seed as i32;
    let result_name = ctx.fresh_var("sx2");
    ctx.emit(format!(
        "let {result_name}: f32 = noise_simplex2d({seed_i32}, cell_x, cell_y);"
    ));
    Ok(result_name)
}

fn emit_simplex3d(ctx: &mut GenCtx, seed: i64) -> Result<String> {
    let seed_i32 = seed as i32;
    let result_name = ctx.fresh_var("sx3");
    ctx.emit(format!(
        "let {result_name}: f32 = noise_simplex3d({seed_i32}, cell_x, cell_y, cell_z);"
    ));
    Ok(result_name)
}
