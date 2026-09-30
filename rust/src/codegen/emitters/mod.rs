pub mod aquifer;
pub mod beardifier;
pub mod binary;
pub mod biome_tree;
pub mod cache_like;
pub mod constant;
pub mod coordinate;
pub mod end_islands;
pub mod find_top_surface;
pub mod interval_select;
pub mod noise_interpolated;
pub mod noise_simplex;
pub mod range_choice;

pub mod root;
pub mod shift;
pub mod spline;
pub mod unary;
pub mod y_clamped_gradient;

use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use anyhow::{Result, anyhow};

/// Emit a full MC `DoublePerlinNoiseSampler` (or single `OctavePerlinNoiseSampler`):
/// sample each octave stack with Kahan summation, sum the two stacks, scale by
/// `amplitude`. Coordinates are pre-scaled by `xz_scale` / `y_scale` (which Java
/// already multiplied by MC's DOMAIN_SCALE=1.0181268882175227).
fn emit_noise_sampler(
    ctx: &mut GenCtx,
    first: &crate::ast::OctaveSampler,
    second: Option<&crate::ast::OctaveSampler>,
    amplitude: f32,
    xz_scale: f32,
    y_scale: f32,
) -> Result<String> {
    // MC's DoublePerlinNoiseSampler applies DOMAIN_SCALE only to the SECOND
    // stack: first.sample(x,y,z) + second.sample(x*D, y*D, z*D).
    const DOMAIN_SCALE: f32 = 1.0181268882175227;
    let v_first = emit_octave_stack(ctx, first, "first", xz_scale, y_scale, 1.0);
    let total = if let Some(sec) = second {
        let v_second = emit_octave_stack(ctx, sec, "second", xz_scale, y_scale, DOMAIN_SCALE);
        let var = ctx.fresh_var("nsum");
        ctx.emit(format!("let {var}: f32 = {v_first} + {v_second};"));
        var
    } else {
        v_first
    };
    let var = ctx.fresh_var("ns");
    ctx.emit(format!("let {var}: f32 = {total} * {:.9e};", amplitude));
    Ok(var)
}

fn emit_octave_stack(
    ctx: &mut GenCtx,
    oct: &crate::ast::OctaveSampler,
    tag: &str,
    xz_scale: f32,
    y_scale: f32,
    domain_scale: f32,
) -> String {
    let n = oct.amplitudes.len();
    let n_u32 = n as u32;
    let amps_name = ctx.fresh_var(&format!("amps_{tag}"));
    let acc_name = ctx.fresh_var(&format!("kacc_{tag}"));
    let fx_name = ctx.fresh_var(&format!("fx_{tag}"));
    let fy_name = ctx.fresh_var(&format!("fy_{tag}"));
    let fz_name = ctx.fresh_var(&format!("fz_{tag}"));
    let freq_name = ctx.fresh_var(&format!("freq_{tag}"));
    let amp_name = ctx.fresh_var(&format!("ac_{tag}"));
    let v_name = ctx.fresh_var(&format!("pv_{tag}"));
    let out_name = ctx.fresh_var(&format!("os_{tag}"));

    let amp_vals = oct
        .amplitudes
        .iter()
        .map(|v| format!("{v:.9e}"))
        .collect::<Vec<_>>()
        .join(", ");
    ctx.emit(format!(
        "let {amps_name}: array<f32, {n_u32}> = array<f32, {n_u32}>({amp_vals});"
    ));
    let eff_xz = xz_scale * domain_scale;
    let eff_y = y_scale * domain_scale;
    ctx.emit(format!("var {fx_name}: f32 = cell_x * {eff_xz:.9e};"));
    ctx.emit(format!("var {fy_name}: f32 = cell_y * {eff_y:.9e};"));
    ctx.emit(format!("var {fz_name}: f32 = cell_z * {eff_xz:.9e};"));
    ctx.emit(format!("var {freq_name}: f32 = {:.9e};", oct.initial_freq));
    ctx.emit(format!("var {amp_name}: f32 = {:.9e};", oct.persistence));
    ctx.emit(format!("var {acc_name}: KahanSum = kahan_zero();"));
    ctx.emit(format!("for (var i: u32 = 0u; i < {n_u32}u; i++) {{"));
    ctx.emit(format!("  let a_i = {amps_name}[i];"));
    ctx.emit("  if (a_i != 0.0) {".to_string());
    ctx.emit(format!(
        "    let {v_name}: f32 = noise_perlin3d_origin({base}i + i32(i) * 256i, {fx_name} * {freq_name}, {fy_name} * {freq_name}, {fz_name} * {freq_name}) * a_i * {amp_name};",
        base = oct.perm_offset
    ));
    ctx.emit(format!("    kahan_add(&{acc_name}, {v_name});"));
    ctx.emit("  }".to_string());
    ctx.emit(format!("  {freq_name} *= 2.0;"));
    ctx.emit(format!("  {amp_name} *= 0.5;"));
    ctx.emit("}".to_string());
    ctx.emit(format!("let {out_name}: f32 = kahan_result({acc_name});"));
    out_name
}

/// MC `WeirdScaledSampler`: scale = mapper(input); result = scale *
/// abs(noise.sample(x/scale, y/scale, z/scale)).
fn emit_weird_scaled(
    ctx: &mut GenCtx,
    mapper: u8,
    input: &AstNode,
    noise: &AstNode,
) -> Result<String> {
    let input_var = emit_node(input, ctx)?;
    let scale_var = ctx.fresh_var("ws_scale");
    let fn_name = if mapper == 1 { "scale_tunnels" } else { "scale_caves" };
    ctx.emit(format!("let {scale_var}: f32 = {fn_name}({input_var});"));

    // Noise is sampled at (x/scale, y/scale, z/scale); shadow coords.
    let result = ctx.fresh_var("ws");
    ctx.emit(format!("var {result}: f32 = 0.0;"));
    let saved = ctx.in_shadow;
    ctx.in_shadow = true;
    ctx.emit("{".to_string());
    ctx.emit(format!("  let cell_x: f32 = cell_x / {scale_var};"));
    ctx.emit(format!("  let cell_y: f32 = cell_y / {scale_var};"));
    ctx.emit(format!("  let cell_z: f32 = cell_z / {scale_var};"));
    let n = emit_node(noise, ctx)?;
    ctx.emit(format!("  {result} = abs({n}) * {scale_var};"));
    ctx.emit("}".to_string());
    ctx.in_shadow = saved;
    Ok(result)
}

/// 统一节点分发函数。每个 emitter 模块都必须实现 `pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String>`。
pub fn emit_node(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    // ── Memoization ────────────────────────────────────────────────
    // MC's density tree is a DAG: the same subexpression (e.g. the
    // `depth` / `offset` references) appears dozens of times. Without
    // sharing, the emitter expands it every time -> 100k+ lines of WGSL.
    // We key by AST fingerprint; disabled while coordinates are shadowed
    // (Shift/ShiftedNoise/Cache) because then the same AST yields a
    // different value.
    // Always memoize. Cross-shadow reuse is prevented by the scope-stack
    // boundary flag (see GenCtx::memo_get), NOT by disabling memo here —
    // disabling it re-expands the whole subtree and blows WGSL up to
    // ~100k lines.
    let memo_key: Option<u64> = {
        use std::hash::{Hash, Hasher};
        let mut h = rustc_hash::FxHasher::default();
        node.hash(&mut h);
        Some(h.finish())
    };
    if let Some(k) = memo_key {
        if let Some(v) = ctx.memo_get(k) {
            return Ok(v.clone());
        }
    }

    let result = match node {
        AstNode::NoiseSampler { first, second, amplitude, xz_scale, y_scale, .. } => {
            emit_noise_sampler(ctx, first, second.as_ref(), *amplitude, *xz_scale, *y_scale)
        }
        AstNode::Constant(_) => constant::emit(node, ctx),
        AstNode::InputX | AstNode::InputY | AstNode::InputZ => coordinate::emit(node, ctx),
        AstNode::Add(_, _)
        | AstNode::Sub(_, _)
        | AstNode::Mul(_, _)
        | AstNode::Div(_, _)
        | AstNode::Min(_, _)
        | AstNode::Max(_, _)
        | AstNode::FloorDiv(_, _) => binary::emit(node, ctx),
        AstNode::Abs(_) | AstNode::Square(_) | AstNode::Cube(_) | AstNode::Sqrt(_) => {
            unary::emit(node, ctx)
        }
        AstNode::Squeeze(x) => {
            let v = emit_node(x, ctx)?;
            let var = ctx.fresh_var("sq");
            ctx.emit(format!("let {var}: f32 = squeeze({v});"));
            Ok(var)
        }
        AstNode::SimplexNoise2D { .. } | AstNode::SimplexNoise3D { .. } => {
            noise_simplex::emit(node, ctx)
        }
        AstNode::InterpolatedNoiseSampler { .. } => noise_interpolated::emit(node, ctx),
        AstNode::Shift { .. } | AstNode::ShiftedNoise { .. } => shift::emit(node, ctx),
        AstNode::WeirdScaledSampler { mapper, input, noise } => {
            emit_weird_scaled(ctx, *mapper, input, noise)
        }
        AstNode::Beardifier { .. } => beardifier::emit(node, ctx),
        AstNode::Spline { .. } => spline::emit(node, ctx),
        AstNode::RangeChoice { .. } => range_choice::emit(node, ctx),
        AstNode::IntervalSelect { .. } => interval_select::emit(node, ctx),
        AstNode::YClampedGradient { .. } => y_clamped_gradient::emit(node, ctx),
        AstNode::CacheFlat(_) | AstNode::Cache2D(_) | AstNode::CacheInterpolated(_) => {
            cache_like::emit(node, ctx)
        }
        AstNode::EndIslands(_) => end_islands::emit(node, ctx),
        AstNode::FindTopSurface { .. } => find_top_surface::emit(node, ctx),
        AstNode::Aquifer { .. } => aquifer::emit(node, ctx),
        AstNode::BiomeTree { .. } => biome_tree::emit(node, ctx),
        AstNode::Root(_) => unreachable!("Root should be handled by root::emit"),
    };

    if let (Some(k), Ok(v)) = (memo_key, &result) {
        ctx.memo_put(k, v.clone());
    }
    result
}
