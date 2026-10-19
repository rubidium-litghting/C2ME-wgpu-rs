// SPDX-License-Identifier: MIT

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
use crate::codegen::wgsl_gen::GenCtx;
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
    // Cache2D is pure memoisation (the wrapped value depends only on x,z).
    // A per-cell kernel evaluates exactly one cell per invocation, so we
    // simply evaluate the inner subtree in place. Shared memory is NOT
    // used: when Cache2D is nested inside CacheInterpolated's corner loop
    // its barrier would sit in a variable-trip-count loop (WGSL UB), which
    // silently biased the density in whole regions.
    emit_node(inner, ctx)
}

/// CacheFlat — per-column shared memory cache.
///
/// MC's `FlatCache` caches a value that does not depend on Y, but it
/// STILL varies per (x, z). So the slot is `array<f32, 16>` indexed by
/// `lx * 4 + lz` — NOT a single scalar. (Using a scalar silently drops
/// all x/z variation, which biases the density in whole regions.)
fn emit_cache_flat(inner: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    // FlatCache is pure memoisation (the wrapped value is Y-independent).
    // Same reasoning as Cache2D: evaluate in place, no shared memory and
    // no barrier (avoids the nested-in-loop UB).
    emit_node(inner, ctx)
}

/// CacheInterpolated — cell-corner trilinear lerp within a single dispatch.
///
/// See module-level doc for the full algorithm description.
pub(crate) fn emit_cache_interp(inner: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    // Pass-through. The dispatch now covers the corner grid
    // (cell_count + 1 per axis), so each invocation IS a distinct
    // corner sample. MC's ChunkNoiseSampler does the same: it samples
    // density at cell corners, then trilinearly interpolates per block.
    // We hand raw corner values to the Java write-back, which runs the
    // identical interpolation. No shared memory / barrier needed.
    emit_node(inner, ctx)
}
