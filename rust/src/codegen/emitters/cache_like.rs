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

//! CacheFlat / Cache2D / CacheInterpolated — three addressing modes for MC's
//! ChunkNoiseSampler internal caches.
//!
//! ## Cache2D — column-wise cache (phase 3.5)
//!
//! MC semantics: all cells in the same (cell_x, cell_z) column share the
//! same cached value. We assume `inner` does not depend on `cell_y` (this
//! is MC vanilla's contract for `Cache2D`).
//!
//! Workgroup layout (4×4×4):
//!   * 4 invocations per column (lid.y ∈ {0,1,2,3})
//!   * 16 columns per workgroup (lid.x × lid.z)
//!
//! Implementation:
//!   ```wgsl
//!   var<workgroup> c2d_<slot>: array<f32, 16>;  // module scope
//!
//!   if (ly == 0u) {                              // first-in-column computes
//!     <emit inner subtree → result var>
//!     c2d_<slot>[lx * 4u + lz] = result;
//!   }
//!   workgroupBarrier();                          // sync
//!   let c2d_r_<n>: f32 = c2d_<slot>[lx * 4u + lz];
//!   ```
//!
//! ## CacheFlat — whole-workgroup cache (phase 3.5)
//!
//! MC semantics: the entire chunk (in our case, the entire workgroup)
//! shares a single cached value. `inner` must not depend on any of
//! `cell_x / cell_y / cell_z` for the cache to be correct (in practice
//! MC only wraps truly-Y-independent, X/Z-constant subtrees here).
//!
//! Implementation:
//!   ```wgsl
//!   var<workgroup> cf_<slot>: f32;               // module scope
//!
//!   if (lx == 0u && ly == 0u && lz == 0u) {      // first invocation computes
//!     <emit inner subtree → result var>
//!     cf_<slot> = result;
//!   }
//!   workgroupBarrier();
//!   let cf_r_<n>: f32 = cf_<slot>;
//!   ```
//!
//! ## CacheInterpolated — cell-corner trilinear lerp (phase 4-3)
//!
//! MC semantics: `inner` is sampled at cell-corner points (cx, cy, cz ∈
//! {0, 1, 2, 3, 4} for a 4×4×4 workgroup + 1-cell apron). Block-level
//! density is obtained by trilinear interpolation between the 8 corners
//! of the cell containing the block.
//!
//! Single-dispatch implementation (two phases within one `main_entry`):
//!   * **Phase 1** (corner computation): each invocation computes one or
//!     more corners of the 5×5×5 grid (125 corners, 64 invocations) and
//!     writes the result to `var<workgroup> ci_<slot>: array<f32, 5*5*5>`.
//!     Corner-assignment rule: corner (cx, cy, cz) is owned by invocation
//!     (min(cx,3), min(cy,3), min(cz,3)), so boundary invocations (lid==3)
//!     also compute the +1 apron corner on that axis — the (3,3,3)
//!     invocation computes all 8 corners of the (4,4,4) sub-cube.
//!   * **`workgroupBarrier()`**: guarantees all corners are written before
//!     any invocation reads them. This is the correct WGSL synchronisation
//!     primitive — `var<workgroup>` memory does NOT persist across separate
//!     dispatches, so the corner computation and lerp MUST run in the same
//!     dispatch.
//!   * **Phase 2** (trilinear lerp): each cell invocation reads its 8
//!     surrounding corners from shared memory and runs `trilinear_lerp`.
//!     Result is the cell's final value, returned to the root emitter for
//!     `out_buf` writeback.
//!
//! Slot layout: `array<f32, 5*5*5>` = 125 f32 = 500 bytes per slot.
//! Indexed by `cx * 25 + cy * 5 + cz` where each local coord is in
//! `[0, 5)` (cell coords + 1 apron).
//!
//! Why 5×5×5 and not 4×4×4? A 4×4×4 cell grid has 5×5×5 corner points
//! (vertices), not 4×4×4 cells. The +1 apron on each axis is needed so
//! that cells at lid==3 have a valid (lid+1) corner to interpolate
//! against.
//!
//! ## Nesting
//!
//! Each `Cache*` node allocates a fresh slot id via `GenCtx::alloc_cache*`,
//! so nested caches (e.g. `Cache2D(CacheFlat(Perlin))`, common in MC
//! vanilla `finalDensity`) get independent shared-memory regions.

use crate::ast::AstNode;
use crate::codegen::emitters::emit_node;
use crate::codegen::wgsl_gen::{GenCtx, Pass};
use crate::Result;
use anyhow::anyhow;

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    match node {
        AstNode::Cache2D(inner) => emit_cache2d(inner, ctx),
        AstNode::CacheFlat(inner) => emit_cache_flat(inner, ctx),
        AstNode::CacheInterpolated(inner) => emit_cache_interp(inner, ctx),
        _ => Err(anyhow!("cache_like emitter called on non-cache node: {node:?}")),
    }
}

/// Cache2D — column-wise shared memory cache.
///
/// Slot layout: `array<f32, 16>` indexed by `lx * 4 + lz` (4×4 XZ grid).
fn emit_cache2d(inner: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    let slot = ctx.alloc_cache2d_slot();
    let smem = format!("c2d_{slot}");

    // Module-scope declaration (WGSL forbids var<workgroup> in fn body).
    ctx.declare_shared(format!("var<workgroup> {smem}: array<f32, 16>;"));

    // Body: only lid.y == 0 invocations compute inner; all read after barrier.
    ctx.emit("if (ly == 0u) {");
    let inner_var = emit_node(inner, ctx)?;
    ctx.emit(format!("  {smem}[lx * 4u + lz] = {inner_var};"));
    ctx.emit("}");
    ctx.emit("workgroupBarrier();");

    let result = ctx.fresh_var("c2d_r");
    ctx.emit(format!("let {result}: f32 = {smem}[lx * 4u + lz];"));
    Ok(result)
}

/// CacheFlat — whole-workgroup shared memory cache.
///
/// Slot layout: single `f32`.
fn emit_cache_flat(inner: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    let slot = ctx.alloc_cacheflat_slot();
    let smem = format!("cf_{slot}");

    ctx.declare_shared(format!("var<workgroup> {smem}: f32;"));

    ctx.emit("if (lx == 0u && ly == 0u && lz == 0u) {");
    let inner_var = emit_node(inner, ctx)?;
    ctx.emit(format!("  {smem} = {inner_var};"));
    ctx.emit("}");
    ctx.emit("workgroupBarrier();");

    let result = ctx.fresh_var("cf_r");
    ctx.emit(format!("let {result}: f32 = {smem};"));
    Ok(result)
}

/// CacheInterpolated — cell-corner trilinear lerp within a single dispatch.
///
/// See module-level doc for the full algorithm description.
fn emit_cache_interp(inner: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    // Single-pass mode is the ONLY supported mode for CacheInterpolated:
    //   * Pass A (corner computation) + workgroupBarrier() + Pass B
    //     (trilinear lerp) must run in the SAME dispatch — `var<workgroup>`
    //     shared memory does NOT persist across separate dispatches, so a
    //     two-dispatch approach would leave pass B reading uninitialised
    //     memory. The `workgroupBarrier()` inside one dispatch is the
    //     correct WGSL synchronisation primitive for this pattern.
    //
    // The legacy Pass::A / Pass::B entry points are rejected here —
    // `requires_two_pass` now returns false for CacheInterpolated, so the
    // two-pipeline path in WorldContext is never taken.
    if ctx.pass != Pass::Single {
        return Err(anyhow!(
            "CacheInterpolated requires single-pass mode (workgroup memory \
             does not persist across dispatches); got pass {:?}. \
             Call emit_shader(root) instead of emit_shader_for_pass.",
            ctx.pass
        ));
    }

    let slot = ctx.alloc_cache_interp_slot();
    let smem = format!("ci_{slot}");

    // Shared memory: 5×5×5 = 125 f32 cell-corner values.
    ctx.declare_shared(format!("var<workgroup> {smem}: array<f32, 125>;"));

    // ── Phase 1: compute all assigned cell-corners into shared memory ──
    //
    // Corner-assignment rule: corner (cx, cy, cz) with cx, cy, cz ∈
    // {0, 1, 2, 3, 4} is computed by the invocation whose lid is
    //   (min(cx, 3), min(cy, 3), min(cz, 3)).
    // Interior invocations (lid < 3 on all axes) compute 1 corner;
    // boundary invocations (lid == 3) also compute the +1 apron corner
    // on that axis; the (3, 3, 3) invocation computes all 8 corners of
    // the (3..5, 3..5, 3..5) sub-cube.
    //
    // For invocation (lx, ly, lz), we loop dx ∈ [0, lx==3 ? 1 : 0],
    // dy ∈ [0, ly==3 ? 1 : 0], dz ∈ [0, lz==3 ? 1 : 0] and compute
    // corner (lx+dx, ly+dy, lz+dz).
    //
    // The inner subtree is emitted ONCE inside the loop body; each
    // iteration shadows `cell_x / cell_y / cell_z` with the corner's
    // world coords, so coordinate-dependent emitters (InputX/Y/Z,
    // PerlinSampler, …) sample at the corner instead of the cell center.
    // Inner MUST NOT itself contain a Cache2D / CacheFlat (those are
    // sibling cache layers in MC's NoiseRouter, never nested inside
    // CacheInterpolated) — if it did, their `lx/ly/lz` indexing would be
    // wrong inside this loop.
    ctx.emit("// CacheInterpolated phase 1: compute assigned cell-corners.");
    ctx.emit("let dx_max: u32 = select(0u, 1u, lx == 3u);");
    ctx.emit("let dy_max: u32 = select(0u, 1u, ly == 3u);");
    ctx.emit("let dz_max: u32 = select(0u, 1u, lz == 3u);");
    ctx.emit("for (var dx: u32 = 0u; dx <= dx_max; dx++) {");
    ctx.emit("  for (var dy: u32 = 0u; dy <= dy_max; dy++) {");
    ctx.emit("    for (var dz: u32 = 0u; dz <= dz_max; dz++) {");
    // Corner local coords — in [0, 4].
    ctx.emit("      let cx: u32 = lx + dx;");
    ctx.emit("      let cy: u32 = ly + dy;");
    ctx.emit("      let cz: u32 = lz + dz;");
    // Corner world coords. Corner 0 is one cell to the left of the
    // invocation's cell, hence the `- cell_width` offset.
    ctx.emit("      let corner_x_i32: i32 = world_x_i32 - i32(params.cell_width) + i32(cx) * i32(params.cell_width);");
    ctx.emit("      let corner_y_i32: i32 = world_y_i32 - i32(params.cell_height) + i32(cy) * i32(params.cell_height);");
    ctx.emit("      let corner_z_i32: i32 = world_z_i32 - i32(params.cell_width) + i32(cz) * i32(params.cell_width);");
    // Shadow cell_x/y/z in a nested block so inner reads corner coords.
    ctx.emit("      {");
    ctx.emit("        let cell_x: f32 = f32(corner_x_i32);");
    ctx.emit("        let cell_y: f32 = f32(corner_y_i32);");
    ctx.emit("        let cell_z: f32 = f32(corner_z_i32);");
    let inner_var = emit_node(inner, ctx)?;
    ctx.emit(format!("        {smem}[cx * 25u + cy * 5u + cz] = {inner_var};"));
    ctx.emit("      }");
    ctx.emit("    }");
    ctx.emit("  }");
    ctx.emit("}");

    // ── Phase 2: trilinear interpolation from shared memory ────────────
    //
    // workgroupBarrier() guarantees every invocation has written its
    // assigned corners before any invocation reads them. After the
    // barrier, each invocation reads its 8 surrounding corners and runs
    // trilinear_lerp to produce the cell's final density value.
    //
    // Corner index mapping (cell at lid (lx, ly, lz) interpolates between
    // corners (lx..lx+1, ly..ly+1, lz..lz+1)):
    //   corner (lx,   ly,   lz  ) → c000   corner (lx,   ly,   lz+1) → c001
    //   corner (lx+1, ly,   lz  ) → c100   corner (lx+1, ly,   lz+1) → c101
    //   corner (lx,   ly+1, lz  ) → c010   corner (lx,   ly+1, lz+1) → c011
    //   corner (lx+1, ly+1, lz  ) → c110   corner (lx+1, ly+1, lz+1) → c111
    //
    // Weights tx/ty/tz = 0.5 (cell centers for cell_width=4, the MC
    // default). For non-default cell sizes this would be
    // (cell_size-1)/cell_size, but MC vanilla always uses 4×4×4 cells.
    ctx.emit("workgroupBarrier();");
    ctx.emit("// CacheInterpolated phase 2: trilinear lerp from shared memory.");
    let r = ctx.fresh_var("ci_r");
    ctx.emit(format!("let {r}: f32 = trilinear_lerp("));
    ctx.emit(format!("  {smem}[(lx + 0u) * 25u + (ly + 0u) * 5u + (lz + 0u)],"));
    ctx.emit(format!("  {smem}[(lx + 1u) * 25u + (ly + 0u) * 5u + (lz + 0u)],"));
    ctx.emit(format!("  {smem}[(lx + 0u) * 25u + (ly + 1u) * 5u + (lz + 0u)],"));
    ctx.emit(format!("  {smem}[(lx + 1u) * 25u + (ly + 1u) * 5u + (lz + 0u)],"));
    ctx.emit(format!("  {smem}[(lx + 0u) * 25u + (ly + 0u) * 5u + (lz + 1u)],"));
    ctx.emit(format!("  {smem}[(lx + 1u) * 25u + (ly + 0u) * 5u + (lz + 1u)],"));
    ctx.emit(format!("  {smem}[(lx + 0u) * 25u + (ly + 1u) * 5u + (lz + 1u)],"));
    ctx.emit(format!("  {smem}[(lx + 1u) * 25u + (ly + 1u) * 5u + (lz + 1u)],"));
    ctx.emit("  0.5, 0.5, 0.5);");

    Ok(r)
}