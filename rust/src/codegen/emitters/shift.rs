// SPDX-License-Identifier: MIT

//! MC `Shift`/`ShiftA`/`ShiftB` (Offset) and `ShiftedNoise` emitters.
//!
//! Confirmed from bytecode:
//! ```text
//! Offset.sample(x,y,z) = offsetNoise.sample(x*.25, y*.25, z*.25) * 4.0
//! Shift.sample(pos)    = Offset.sample(blockX, blockY, blockZ)
//! ShiftA.sample(pos)   = Offset.sample(blockX, 0.0,    blockZ)
//! ShiftB.sample(pos)   = Offset.sample(blockZ, blockX, 0.0)
//!
//! ShiftedNoise.sample(pos):
//!   d = blockX*xzScale + shiftX.sample(pos)
//!   e = blockY*yScale  + shiftY.sample(pos)
//!   f = blockZ*xzScale + shiftZ.sample(pos)
//!   return noise.sample(d, e, f)
//! ```

use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use crate::Result;
use anyhow::anyhow;

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    match node {
        AstNode::Shift { kind, offset } => emit_shift(ctx, *kind, offset),
        AstNode::ShiftedNoise {
            xz_scale, y_scale, shift_x, shift_y, shift_z, noise,
        } => emit_shifted_noise(ctx, *xz_scale, *y_scale, shift_x, shift_y, shift_z, noise),
        _ => Err(anyhow!("shift emitter called on non-shift node: {node:?}")),
    }
}

fn emit_shift(ctx: &mut GenCtx, kind: u8, offset: &AstNode) -> Result<String> {
    // Sample the inner noise with kind-specific coordinate swizzle, all *0.25.
    let var = ctx.fresh_var("shift");
    // Declare the result OUTSIDE the block so it survives the shadow scope.
    ctx.emit(format!("var {var}: f32 = 0.0;"));
    // Coordinates are shadowed inside the block -> enter shadow scope
    // BEFORE emitting `{` so the frame is flagged as a memo boundary.
    let saved = ctx.in_shadow;
    ctx.in_shadow = true;
    ctx.emit("{".to_string());
    // Capture originals first so swizzles don't read already-shadowed names.
    ctx.emit("  let _ox: f32 = cell_x;".to_string());
    ctx.emit("  let _oy: f32 = cell_y;".to_string());
    ctx.emit("  let _oz: f32 = cell_z;".to_string());
    match kind {
        0 => {
            ctx.emit("  let cell_x: f32 = _ox * 0.25;".to_string());
            ctx.emit("  let cell_y: f32 = _oy * 0.25;".to_string());
            ctx.emit("  let cell_z: f32 = _oz * 0.25;".to_string());
        }
        1 => {
            ctx.emit("  let cell_x: f32 = _ox * 0.25;".to_string());
            ctx.emit("  let cell_y: f32 = 0.0;".to_string());
            ctx.emit("  let cell_z: f32 = _oz * 0.25;".to_string());
        }
        2 => {
            ctx.emit("  let cell_x: f32 = _oz * 0.25;".to_string());
            ctx.emit("  let cell_y: f32 = _ox * 0.25;".to_string());
            ctx.emit("  let cell_z: f32 = 0.0;".to_string());
        }
        _ => return Err(anyhow!("invalid Shift kind {kind}")),
    }
    let inner2 = crate::codegen::emitters::emit_node(offset, ctx)?;
    ctx.emit(format!("  {var} = {inner2} * 4.0;"));
    ctx.emit("}".to_string());
    ctx.in_shadow = saved;
    Ok(var)
}

fn emit_shifted_noise(
    ctx: &mut GenCtx,
    xz_scale: f32,
    y_scale: f32,
    shift_x: &AstNode,
    shift_y: &AstNode,
    shift_z: &AstNode,
    noise: &AstNode,
) -> Result<String> {
    let sx = crate::codegen::emitters::emit_node(shift_x, ctx)?;
    let sy = crate::codegen::emitters::emit_node(shift_y, ctx)?;
    let sz = crate::codegen::emitters::emit_node(shift_z, ctx)?;

    let dx = ctx.fresh_var("snx");
    let dy = ctx.fresh_var("sny");
    let dz = ctx.fresh_var("snz");
    ctx.emit(format!("let {dx}: f32 = cell_x * {xz_scale:.9e} + {sx};"));
    ctx.emit(format!("let {dy}: f32 = cell_y * {y_scale:.9e} + {sy};"));
    ctx.emit(format!("let {dz}: f32 = cell_z * {xz_scale:.9e} + {sz};"));

    // Sample the inner noise at the shifted coordinate. The inner emitter
    // reads cell_x/y/z, so shadow them inside a block.
    let var = ctx.fresh_var("sn");
    ctx.emit(format!("var {var}: f32 = 0.0;"));
    let saved = ctx.in_shadow;
    ctx.in_shadow = true;
    ctx.emit("{".to_string());
    ctx.emit(format!("  let cell_x: f32 = {dx};"));
    ctx.emit(format!("  let cell_y: f32 = {dy};"));
    ctx.emit(format!("  let cell_z: f32 = {dz};"));
    let nz = crate::codegen::emitters::emit_node(noise, ctx)?;
    ctx.emit(format!("  {var} = {nz};"));
    ctx.emit("}".to_string());
    ctx.in_shadow = saved;
    Ok(var)
}
