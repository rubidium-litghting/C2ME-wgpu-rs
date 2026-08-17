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
//!
//! Implementation note on the `seed: i64` AST field:
//!   WGSL does not have i64 integer type.  MC's PerlinNoiseSampler actually
//!   consumes the seed only once at *permutation-table construction time*
//!   (build_permutation_buffers writes the 256-entry table into
//!   @binding(1) `permutations` storage buffer) — the per-sample kernel
//!   never needs the original seed again.  For this phase-3 emitter we
//!   therefore **truncate `seed as i32`** for the `noise_perlin3d(seed+i, …)`
//!   call to match math_lib.wgsl's current placeholder signature.  A
//!   follow-up patch will change noise_perlin3d to take a permutation-table
//!   offset (not a seed) and remove this cast.

use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use crate::Result;
use anyhow::anyhow;

/// MC vanilla `DoublePerlinNoiseSampler` amplitude blending ratio.
///
/// Java source: `value = sampleA * 0.5 + sampleB * 0.5;` (aka the two
/// perlin samplers each contribute half amplitude).  Hard-coded to match
/// MC exactly — user-configurable ratios are rare.
const DOUBLE_PERLIN_RATIO: f32 = 0.5;

/// Hard upper bound on perlin octaves (conservative; MC vanilla ≤ 6).
/// The generated `for` loop uses this as compile-time upper bound and
/// emits a runtime `if (i32(i) >= octaves) { break; }` guard for short
/// samplers.  Keeping this small lets drivers unroll aggressively.
const MAX_OCTAVES: u32 = 8;

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    match node {
        AstNode::PerlinSampler { seed, octaves, amplitudes, lacunaritys } => {
            emit_perlin_sampler(ctx, *seed, *octaves, amplitudes, lacunaritys)
        }
        AstNode::DoublePerlin { a, b } => emit_double_perlin(ctx, a, b),
        _ => Err(anyhow!("noise_perlin emitter called on non-perlin node: {node:?}")),
    }
}

// =========================================================================
// PerlinSampler — inline the N-octave Kahan-summation loop per
// math_lib.wgsl reference template.
// =========================================================================

fn emit_perlin_sampler(
    ctx: &mut GenCtx,
    seed: i64,
    octaves: i32,
    amplitudes: &[f32],
    lacunaritys: &[f32],
) -> Result<String> {
    // ── sanity: contract from deserialize asserts amplitudes.len == lacunaritys.len == octaves
    if amplitudes.len() != octaves as usize || lacunaritys.len() != octaves as usize {
        return Err(anyhow!(
            "PerlinSampler contract violation: octaves={octaves}, \
             amplitudes.len()={}, lacunaritys.len()={}",
            amplitudes.len(), lacunaritys.len()
        ));
    }
    let n = amplitudes.len();
    let n_u32 = n as u32;

    // WGSL i64 workaround: truncate → i32 (see module-level docstring).
    let seed_i32 = seed as i32;

    // ── allocate 3 fresh var names
    let amps_name = ctx.fresh_var("amps");
    let lacs_name = ctx.fresh_var("lacs");
    let result_name = ctx.fresh_var("p");

    // ── emit amplitudes / lacunaritys compile-time arrays
    let amp_vals = amplitudes.iter().map(|v| format!("{v:.9e}")).collect::<Vec<_>>().join(", ");
    let lac_vals = lacunaritys.iter().map(|v| format!("{v:.9e}")).collect::<Vec<_>>().join(", ");
    ctx.emit(format!(
        "let {amps_name}: array<f32, {n_u32}> = array<f32, {n_u32}>({amp_vals});"
    ));
    ctx.emit(format!(
        "let {lacs_name}: array<f32, {n_u32}> = array<f32, {n_u32}>({lac_vals});"
    ));

    // ── Kahan accumulator + frequency locals
    let acc_name = ctx.fresh_var("kacc");
    let fx_name = ctx.fresh_var("fx");
    let fy_name = ctx.fresh_var("fy");
    let fz_name = ctx.fresh_var("fz");
    ctx.emit(format!("var {acc_name}: KahanSum = kahan_zero();"));
    ctx.emit(format!("var {fx_name}: f32 = cell_x;"));
    ctx.emit(format!("var {fy_name}: f32 = cell_y;"));
    ctx.emit(format!("var {fz_name}: f32 = cell_z;"));

    // ── for loop — bound by min(octaves, MAX_OCTAVES) as static upper bound
    let loop_bound = std::cmp::min(n_u32, MAX_OCTAVES);
    let octaves_u32 = octaves as u32;

    // Loop body variables (shared across iterations)
    let v_name = ctx.fresh_var("pv");

    let loop_head = format!(
        "for (var i: u32 = 0u; i < {loop_bound}u; i++) {{"
    );
    ctx.emit(loop_head);

    // Dynamic guard: for when static bound > actual octaves (only happens if
    // MAX_OCTAVES ceiling above is higher than the real n).
    if loop_bound > n_u32 || octaves_u32 != loop_bound {
        ctx.emit(format!(
            "  if (i >= {octaves_u32}u) {{ break; }}"
        ));
    }

    ctx.emit(format!(
        "  let amp = {amps_name}[i];"
    ));
    ctx.emit(format!(
        "  let lac = {lacs_name}[i];"
    ));
    // seed + i32(i) → per-octave seed offset (MC convention).
    ctx.emit(format!(
        "  let {v_name}: f32 = noise_perlin3d({seed_i32} + i32(i), \
         {fx_name}, {fy_name}, {fz_name}) * amp;"
    ));
    ctx.emit(format!(
        "  kahan_add(&{acc_name}, {v_name});"
    ));
    ctx.emit(format!("  {fx_name} *= lac;"));
    ctx.emit(format!("  {fy_name} *= lac;"));
    ctx.emit(format!("  {fz_name} *= lac;"));
    ctx.emit("}".to_string()); // close for

    // ── final result
    ctx.emit(format!(
        "let {result_name}: f32 = kahan_result({acc_name});"
    ));
    Ok(result_name)
}

// =========================================================================
// DoublePerlin — sample both children, blend with DOUBLE_PERLIN_RATIO.
// =========================================================================

fn emit_double_perlin(
    ctx: &mut GenCtx,
    a: &AstNode,
    b: &AstNode,
) -> Result<String> {
    use crate::codegen::emitters::emit_node;

    // NOTE: DoublePerlin's children are conceptually always PerlinSampler,
    // but the AST schema boxes them as generic AstNode so we just recurse
    // via `emit_node` — any child kind will produce an f32 expression.
    let va = emit_node(a, ctx)?;
    let vb = emit_node(b, ctx)?;

    let result_name = ctx.fresh_var("dp");
    ctx.emit(format!(
        "let {result_name}: f32 = {va} * {:.9e} + {vb} * {:.9e};",
        DOUBLE_PERLIN_RATIO, DOUBLE_PERLIN_RATIO
    ));
    Ok(result_name)
}
