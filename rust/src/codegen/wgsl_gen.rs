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

//! Top-level WGSL code generator.
//!
//! Produces one or two shader modules per `WorldContext`, depending on
//! whether the AST contains any `CacheInterpolated` node:
//!
//!   * **No CacheInterpolated** → single shader `main_entry` (phase 3.5
//!     behavior). One pipeline, one dispatch.
//!
//!   * **With CacheInterpolated** → two shaders `main_pass_a` and
//!     `main_pass_b` (phase 4-3). The scheduler dispatches A then B:
//!       - Pass A: each cell invocation computes the *inner* subtree at
//!         its 8 surrounding cell-corners and writes them to shared memory
//!         (`var<workgroup> ci_<slot>: array<f32, 5*5*5>`). `out_buf` is
//!         *not* written.
//!       - Pass B: each cell invocation reads the 8 surrounding corners
//!         from shared memory and runs trilinear interpolation to get
//!         the final cell value, then writes to `out_buf`.
//!
//! The two-pass split is needed because WGSL has no inter-dispatch
//! synchronization primitive — pass A's shared memory does not survive
//! into pass B. Pass B re-runs inner if it must (the corner data is
//! recomputed), but for typical NoiseRouter shapes inner is dominated
//! by noise samplers whose cost is amortised by Cache2D/CacheFlat layers
//! *inside* the CacheInterpolated wrapper.
//!
//! Pipeline cache key: `(fingerprint, pass_id)` — see `pipeline_cache.rs`.

use crate::ast::AstNode;
use crate::codegen::emitters;
use crate::Result;
use anyhow::anyhow;

/// Which pass to emit a shader for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pass {
    /// Single-pass mode — no CacheInterpolated in AST. Entry point
    /// `main_entry`. Use this when `requires_two_pass(root) == false`.
    Single,
    /// Pass A — compute cell-corners and write to shared memory only.
    /// Entry point `main_pass_a`. `out_buf` is not written.
    A,
    /// Pass B — trilinear-interpolate shared memory and write `out_buf`.
    /// Entry point `main_pass_b`.
    B,
}

impl Pass {
    pub fn entry_point(self) -> &'static str {
        match self {
            Pass::Single => "main_entry",
            Pass::A => "main_pass_a",
            Pass::B => "main_pass_b",
        }
    }
}

/// Code generation context — accumulates WGSL source lines + var counter.
///
/// Two output buckets:
///   * `lines`        — main_entry body lines (let bindings, if branches,
///                      barriers, out_buf writes, …) produced in call order.
///   * `shared_decls` — `var<workgroup>` declarations that MUST live at
///                      module scope (WGSL forbids `var<workgroup>` inside
///                      function bodies). `emit_shader_for_pass` splices
///                      these between `BINDINGS_HEADER` and the entry point.
pub struct GenCtx {
    pub lines: Vec<String>,
    pub shared_decls: Vec<String>,
    pub next_var: u32,
    /// Cache slot counters — each `Cache*` node gets a unique slot id so
    /// nested caches don't share storage.
    pub next_cache2d_slot: u32,
    pub next_cacheflat_slot: u32,
    pub next_cache_interp_slot: u32,
    /// Which pass we're emitting. Emitters consult this to decide
    /// whether to compute the inner subtree (A), do trilinear lerp (B),
    /// or just emit inner (Single).
    pub pass: Pass,
    /// Scope stack of memoized (fingerprint -> var name) bindings.
    /// Index 0 is the function top level; each `{` pushes, each `}` pops.
    /// A value memoized at scope k is visible from scope k and every
    /// nested scope, so reusing it is always WGSL-scope-safe.
    /// Each frame: (is_shadow_boundary, fingerprint -> var). A boundary
    /// frame means coordinates were shadowed when it was pushed, so
    /// memo_get must NOT search past it (outer values used different
    /// coordinates). Within the frame, memoization still works.
    pub memo_stack: Vec<(bool, rustc_hash::FxHashMap<u64, String>)>,
    /// Set while `cell_x/y/z` are shadowed (Shift/ShiftedNoise/Cache).
    /// Memoization is bypassed entirely then, because the same AST
    /// evaluates to a different value at shadowed coordinates.
    pub in_shadow: bool,
}

impl GenCtx {
    pub fn new(pass: Pass) -> Self {
        Self {
            lines: Vec::new(),
            shared_decls: Vec::new(),
            next_var: 0,
            next_cache2d_slot: 0,
            next_cacheflat_slot: 0,
            next_cache_interp_slot: 0,
            pass,
            memo_stack: vec![(false, rustc_hash::FxHashMap::default())],
            in_shadow: false,
        }
    }
    pub fn fresh_var(&mut self, prefix: &str) -> String {
        let n = self.next_var;
        self.next_var += 1;
        format!("{prefix}_{n}")
    }
    pub fn emit(&mut self, s: impl Into<String>) {
        let s = s.into();
        // Maintain the memo scope stack in lockstep with brace nesting.
        for ch in s.chars() {
            match ch {
                '{' => self.memo_stack.push((self.in_shadow, rustc_hash::FxHashMap::default())),
                '}' => {
                    if self.memo_stack.len() > 1 {
                        self.memo_stack.pop();
                    }
                }
                _ => {}
            }
        }
        self.lines.push(s);
    }

    /// Look up a memoized value in the current scope or enclosing ones,
    /// stopping at the first shadow boundary (values outside it were
    /// computed at different coordinates and must not be reused).
    pub fn memo_get(&self, key: u64) -> Option<&String> {
        for (is_boundary, m) in self.memo_stack.iter().rev() {
            if let Some(v) = m.get(&key) {
                return Some(v);
            }
            if *is_boundary {
                break;
            }
        }
        None
    }

    /// Record a memoized value in the innermost scope.
    pub fn memo_put(&mut self, key: u64, value: String) {
        if let Some((_, top)) = self.memo_stack.last_mut() {
            top.insert(key, value);
        }
    }
    /// Push a `var<workgroup>` declaration to module scope.
    /// Use this from emitters that need shared memory (Cache2D, CacheFlat,
    /// CacheInterpolated).
    pub fn declare_shared(&mut self, decl: impl Into<String>) {
        self.shared_decls.push(decl.into());
    }
    /// Allocate a fresh Cache2D slot id (each slot = `array<f32, 16>`,
    /// indexed by `lid.x * 4 + lid.z`).
    pub fn alloc_cache2d_slot(&mut self) -> u32 {
        let s = self.next_cache2d_slot;
        self.next_cache2d_slot += 1;
        s
    }
    /// Allocate a fresh CacheFlat slot id (each slot = single `f32`).
    pub fn alloc_cacheflat_slot(&mut self) -> u32 {
        let s = self.next_cacheflat_slot;
        self.next_cacheflat_slot += 1;
        s
    }
    /// Allocate a fresh CacheInterpolated slot id (each slot =
    /// `array<f32, 5*5*5>` for the 4×4×4 workgroup + 1-cell apron).
    pub fn alloc_cache_interp_slot(&mut self) -> u32 {
        let s = self.next_cache_interp_slot;
        self.next_cache_interp_slot += 1;
        s
    }
}

/// Check whether the AST requires two-pass dispatch.
///
/// **Always returns `false`** as of phase 4-3: `CacheInterpolated` now
/// runs in single-pass mode using `workgroupBarrier()` to synchronise
/// corner computation and trilinear lerp within one dispatch. This is
/// the correct WGSL pattern — `var<workgroup>` shared memory does NOT
/// persist across separate dispatches, so a two-dispatch approach would
/// leave pass B reading uninitialised memory.
///
/// The function is retained for API stability and as a hook in case a
/// future AST node genuinely needs multi-dispatch (e.g. a node that
/// requires a global memory fence between phases).
pub fn requires_two_pass(_root: &AstNode) -> bool {
    false
}

/// Emit a full shader module string for the given AST root and pass.
///
/// Most callers should use `emit_shader` (which auto-selects `Pass::Single`
/// when possible); use this directly only when explicitly compiling the
/// A/B pair for a `CacheInterpolated`-containing AST.
pub fn emit_shader_for_pass(root: &AstNode, pass: Pass) -> Result<String> {
    let AstNode::Root(_) = root else {
        return Err(anyhow!(
            "emit_shader_for_pass: expected AstNode::Root, got {root:?}"
        ));
    };

    let mut ctx = GenCtx::new(pass);

    // Phase A — emit entry-point body. Recursively walks the AST and
    // populates `ctx.lines` (body) + `ctx.shared_decls` (workgroup vars).
    emitters::root::emit(root, &mut ctx)?;

    // Phase B — assemble final shader source in the correct WGSL order.
    let mut out = String::with_capacity(8192);
    out.push_str(super::MATH_LIB_WGSL);
    out.push_str(BINDINGS_HEADER);
    for decl in &ctx.shared_decls {
        out.push_str(decl);
        out.push('\n');
    }
    // Pick the right prologue for this pass.
    let prologue = match pass {
        Pass::Single => MAIN_ENTRY_PROLOGUE,
        Pass::A => MAIN_PASS_A_PROLOGUE,
        Pass::B => MAIN_PASS_B_PROLOGUE,
    };
    out.push_str(prologue);
    for line in &ctx.lines {
        out.push_str(line);
        out.push('\n');
    }
    out.push('}');

    Ok(out)
}

/// Convenience: emit the single-pass `main_entry` shader for the given AST.
///
/// All ASTs — including those with `CacheInterpolated` — are emitted in
/// single-pass mode. CacheInterpolated handles corner computation +
/// `workgroupBarrier()` + trilinear lerp entirely within one dispatch
/// (workgroup memory does NOT persist across dispatches, so a two-pass
/// split would be incorrect).
pub fn emit_shader(root: &AstNode) -> Result<String> {
    emit_shader_for_pass(root, Pass::Single)
}

/// Storage bindings — one-to-one with `world/bind_group.rs::create_layout`
/// and `world/constants.rs::WorldgenParams` field order/offsets.
const BINDINGS_HEADER: &str = r#"
struct WorldgenParams {
  batch_origin_x: i32,
  batch_origin_z: i32,
  cell_width:     f32,
  cell_height:    f32,
  min_y_f:        f32,
  height_f:       f32,
  batch_count_x:  i32,
  batch_count_z:  i32,
  _pad:           array<u32, 8>,
};

@group(0) @binding(0) var<storage, read>        params:       WorldgenParams;
@group(0) @binding(1) var<storage, read>        perm: array<u32>;
@group(0) @binding(2) var<storage, read_write>  out_buf:      array<f32>;
@group(0) @binding(3) var<storage, read>        perm_origin:  array<f32>;
@group(0) @binding(4) var<storage, read>        positions:    array<i32>;
@group(0) @binding(5) var<storage, read>        simplex_perm: array<u32>;
"#;

/// `main_entry` opening (single-pass mode) — sets up `cell_x / cell_y /
/// cell_z` for the coordinate emitter and `cell_idx` / `cells_per_chunk`
/// for the root emitter's writeback. 铁律① applied here: world coords
/// computed in i32 first, then cast to f32 for noise input.
///
/// Workgroup layout (phase 3.5):
///   `@workgroup_size(4, 4, 4)` — one workgroup computes a 4×4×4 cell cube
///   = 64 invocations. Each invocation handles exactly one cell. The
///   `local_invocation_id` (`lid`) is exposed as `lx / ly / lz` for
///   Cache2D / CacheFlat emitters that need workgroup-local indexing.
const MAIN_ENTRY_PROLOGUE: &str = r#"
@compute @workgroup_size(4, 4, 4)
fn main_entry(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id)  lid: vec3<u32>,
) {
  // ── workgroup-local cell coordinates (u32, for shared-mem indexing) ─
  let lx: u32 = lid.x;
  let ly: u32 = lid.y;
  let lz: u32 = lid.z;

  // ── cell coordinates (i32, for linear indexing into out_buf) ────────
  let cx_i32: i32 = i32(gid.x);
  let cy_i32: i32 = i32(gid.y);
  let cz_i32: i32 = i32(gid.z);

  // ── chunk dimensions in cells (16×16 blocks per chunk) ──────────────
  let cell_count_x: i32 = 16 / i32(params.cell_width) + 1;
  let cell_count_z: i32 = 16 / i32(params.cell_width) + 1;
  let cell_count_y: i32 = i32(params.height_f) / i32(params.cell_height) + 1;
  let cells_per_chunk: u32 = u32(cell_count_x * cell_count_y * cell_count_z);

  // ── multi-chunk dispatch: gid.z spans cell_count_z × num_chunks ─────
  // Every chunk in the batch owns a contiguous band of `cell_count_z`
  // z-slices, so ONE dispatch generates all of them. The chunk's world
  // origin comes from the `positions` buffer (sorted [cx,cz,...]).
  let num_chunks: i32 = i32(arrayLength(&positions)) / 2;
  let chunk_index: i32 = cz_i32 / cell_count_z;
  let cz_local: i32 = cz_i32 % cell_count_z;
  if (cx_i32 >= cell_count_x || cy_i32 >= cell_count_y
      || cz_local >= cell_count_z || chunk_index >= num_chunks) {
    return;
  }

  let origin_x: i32 = positions[chunk_index * 2] * 16;
  let origin_z: i32 = positions[chunk_index * 2 + 1] * 16;

  // ── cell linear index (MC-native stride, see mc_stride.rs) ─────────
  // Local cell index within this chunk (MC-native stride).
  let cell_idx: u32 = u32(cx_i32
                         + cz_local * cell_count_x
                         + cy_i32 * cell_count_x * cell_count_z);
  // Chunk-major output base: 16 channels per chunk, so chunk i's channel c
  // lives at (i*16 + c) * cells_per_chunk. Matches Java's readback layout.
  let chunk_cell_base: u32 = u32(chunk_index) * 16u * cells_per_chunk;

  // ── 铁律①: i32 world coords via origin subtraction ──────────────────
  let world_x_i32: i32 = origin_x + cx_i32 * i32(params.cell_width);
  let world_z_i32: i32 = origin_z + cz_local * i32(params.cell_width);
  let world_y_i32: i32 = i32(params.min_y_f) + cy_i32 * i32(params.cell_height);

  // ── f32 noise inputs (the coordinate emitter returns these names) ───
  let cell_x: f32 = f32(world_x_i32);
  let cell_y: f32 = f32(world_y_i32);
  let cell_z: f32 = f32(world_z_i32);
"#;

/// `main_pass_a` opening — same prologue as `main_entry` but the entry
/// point name is `main_pass_a`. Pass A only writes to shared memory
/// (handled by the CacheInterpolated emitter); the root emitter's
/// `out_buf` writes are skipped at the GenCtx level when `pass == Pass::A`.
const MAIN_PASS_A_PROLOGUE: &str = r#"
@compute @workgroup_size(4, 4, 4)
fn main_pass_a(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id)  lid: vec3<u32>,
) {
  let lx: u32 = lid.x;
  let ly: u32 = lid.y;
  let lz: u32 = lid.z;
  let cx_i32: i32 = i32(gid.x);
  let cy_i32: i32 = i32(gid.y);
  let cz_i32: i32 = i32(gid.z);
  let cell_count_x: i32 = 16 / i32(params.cell_width) + 1;
  let cell_count_z: i32 = 16 / i32(params.cell_width) + 1;
  let cell_count_y: i32 = i32(params.height_f) / i32(params.cell_height) + 1;
  let cells_per_chunk: u32 = u32(cell_count_x * cell_count_y * cell_count_z);

  // ── multi-chunk dispatch: gid.z spans cell_count_z × num_chunks ─────
  // Every chunk in the batch owns a contiguous band of `cell_count_z`
  // z-slices, so ONE dispatch generates all of them. The chunk's world
  // origin comes from the `positions` buffer (sorted [cx,cz,...]).
  let num_chunks: i32 = i32(arrayLength(&positions)) / 2;
  let chunk_index: i32 = cz_i32 / cell_count_z;
  let cz_local: i32 = cz_i32 % cell_count_z;
  if (cx_i32 >= cell_count_x || cy_i32 >= cell_count_y
      || cz_local >= cell_count_z || chunk_index >= num_chunks) {
    return;
  }

  let origin_x: i32 = positions[chunk_index * 2] * 16;
  let origin_z: i32 = positions[chunk_index * 2 + 1] * 16;
  // Local cell index within this chunk (MC-native stride).
  let cell_idx: u32 = u32(cx_i32
                         + cz_local * cell_count_x
                         + cy_i32 * cell_count_x * cell_count_z);
  // Chunk-major output base: 16 channels per chunk, so chunk i's channel c
  // lives at (i*16 + c) * cells_per_chunk. Matches Java's readback layout.
  let chunk_cell_base: u32 = u32(chunk_index) * 16u * cells_per_chunk;
  let world_x_i32: i32 = origin_x + cx_i32 * i32(params.cell_width);
  let world_z_i32: i32 = origin_z + cz_local * i32(params.cell_width);
  let world_y_i32: i32 = i32(params.min_y_f) + cy_i32 * i32(params.cell_height);
  let cell_x: f32 = f32(world_x_i32);
  let cell_y: f32 = f32(world_y_i32);
  let cell_z: f32 = f32(world_z_i32);
"#;

/// `main_pass_b` opening — same as A but entry point is `main_pass_b`.
/// Pass B reads the shared memory populated by A and writes `out_buf`.
const MAIN_PASS_B_PROLOGUE: &str = r#"
@compute @workgroup_size(4, 4, 4)
fn main_pass_b(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id)  lid: vec3<u32>,
) {
  let lx: u32 = lid.x;
  let ly: u32 = lid.y;
  let lz: u32 = lid.z;
  let cx_i32: i32 = i32(gid.x);
  let cy_i32: i32 = i32(gid.y);
  let cz_i32: i32 = i32(gid.z);
  let cell_count_x: i32 = 16 / i32(params.cell_width) + 1;
  let cell_count_z: i32 = 16 / i32(params.cell_width) + 1;
  let cell_count_y: i32 = i32(params.height_f) / i32(params.cell_height) + 1;
  let cells_per_chunk: u32 = u32(cell_count_x * cell_count_y * cell_count_z);

  // ── multi-chunk dispatch: gid.z spans cell_count_z × num_chunks ─────
  // Every chunk in the batch owns a contiguous band of `cell_count_z`
  // z-slices, so ONE dispatch generates all of them. The chunk's world
  // origin comes from the `positions` buffer (sorted [cx,cz,...]).
  let num_chunks: i32 = i32(arrayLength(&positions)) / 2;
  let chunk_index: i32 = cz_i32 / cell_count_z;
  let cz_local: i32 = cz_i32 % cell_count_z;
  if (cx_i32 >= cell_count_x || cy_i32 >= cell_count_y
      || cz_local >= cell_count_z || chunk_index >= num_chunks) {
    return;
  }

  let origin_x: i32 = positions[chunk_index * 2] * 16;
  let origin_z: i32 = positions[chunk_index * 2 + 1] * 16;
  // Local cell index within this chunk (MC-native stride).
  let cell_idx: u32 = u32(cx_i32
                         + cz_local * cell_count_x
                         + cy_i32 * cell_count_x * cell_count_z);
  // Chunk-major output base: 16 channels per chunk, so chunk i's channel c
  // lives at (i*16 + c) * cells_per_chunk. Matches Java's readback layout.
  let chunk_cell_base: u32 = u32(chunk_index) * 16u * cells_per_chunk;
  let world_x_i32: i32 = origin_x + cx_i32 * i32(params.cell_width);
  let world_z_i32: i32 = origin_z + cz_local * i32(params.cell_width);
  let world_y_i32: i32 = i32(params.min_y_f) + cy_i32 * i32(params.cell_height);
  let cell_x: f32 = f32(world_x_i32);
  let cell_y: f32 = f32(world_y_i32);
  let cell_z: f32 = f32(world_z_i32);
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::AstNode;

    // ── helpers ─────────────────────────────────────────────────────────

    fn tiny_root(channels: Vec<AstNode>) -> AstNode {
        AstNode::Root(channels)
    }

    /// Returns (line index, line content) for the first line matching `pred`.
    fn find_line<'a, P: Fn(&str) -> bool>(
        src: &'a str, pred: P,
    ) -> Option<(usize, &'a str)> {
        src.lines().enumerate().find(|(_, l)| pred(l))
    }

    /// Extract the variable name from a `let <name>: f32 = ...;` line.
    ///
    /// Example: `let bin_5: f32 = sqrt(un_3);` → returns `"bin_5"`.
    fn var_name_of_let_binding(line: &str) -> String {
        let after_let = line.trim().strip_prefix("let ")
            .expect("caller must pass a `let ...` line");
        let colon = after_let.find(':')
            .expect("`let` line must have `:` type annotation");
        after_let[..colon].trim().to_string()
    }

    // ── happy path tests ────────────────────────────────────────────────

    /// Smallest-possible shader: single Constant channel.
    ///
    /// Assertions:
    ///   1. emit_shader returns Ok
    ///   2. output contains the bindings header (WorldgenParams struct)
    ///   3. output contains main_entry + cell_x/cell_y/cell_z declarations
    ///   4. output contains one `let c_0: f32 = <literal>;` (Constant emitter)
    ///   5. output contains `out_buf[0u * cells_per_chunk + cell_idx] = c_0;`
    #[test]
    fn emit_shader_single_constant_channel() {
        let root = tiny_root(vec![AstNode::Constant(3.14)]);
        let src = emit_shader(&root).expect("emit must succeed");

        // structural layout assertions
        assert!(src.contains("struct WorldgenParams"), "missing WorldgenParams struct");
        assert!(src.contains("fn main_entry(") && src.contains("@compute @workgroup_size(4, 4, 4)"),
                "missing main_entry fn with phase 3.5 workgroup_size");
        assert!(src.contains("let cell_x: f32"), "missing cell_x decl");
        assert!(src.contains("let cell_y: f32"), "missing cell_y decl");
        assert!(src.contains("let cell_z: f32"), "missing cell_z decl");
        assert!(src.contains("let lx: u32 = lid.x;"), "missing phase 3.5 lx/lid decl");

        // constant literal + out_buf write
        assert!(src.contains("let c_0: f32 ="), "missing constant let binding");
        assert!(src.contains("out_buf[chunk_cell_base + 0u * cells_per_chunk + cell_idx] = c_0;"),
                "missing ch0 out_buf write");
    }

    /// Binary op + coordinate inputs: Root([Constant, Add(InputX, InputY)]).
    ///
    /// Expected WGSL body (order of let bindings matches AST post-order):
    ///   let c_0: f32 = ...;         // ch0 constant
    ///   let bin_N: f32 = cell_x + cell_y;  // ch1 Add (N is fresh_var global counter)
    ///   out_buf[0u * ...] = c_0;
    ///   out_buf[1u * ...] = bin_N;
    ///
    /// NOTE: `GenCtx` uses one global counter for ALL prefixes (`c_`, `bin_`,
    /// `un_`, `sq_`…), so suffix numbers don't reset per-prefix.  Tests below
    /// use substring/prefix matching via `matches!` on lines rather than
    /// hard-coding exact numeric suffixes.
    #[test]
    fn emit_shader_arithmetic_add_xy() {
        let root = tiny_root(vec![
            AstNode::Constant(1.0),
            AstNode::Add(
                Box::new(AstNode::InputX),
                Box::new(AstNode::InputY),
            ),
        ]);
        let src = emit_shader(&root).expect("emit must succeed");

        // ch0 constant
        assert!(find_line(&src, |l| l.trim().starts_with("let c_0: f32 =")).is_some(),
                "missing ch0 constant let-binding");
        // ch1 Add: line starts with "let bin_" + ends with "cell_x + cell_y;"
        let add_line = find_line(&src, |l| {
            let t = l.trim();
            t.starts_with("let bin_")
                && t.contains(": f32 =")
                && t.ends_with("cell_x + cell_y;")
        });
        assert!(add_line.is_some(),
                "missing Add(InputX, InputY) let-binding; got:\n{src}");
        // Extract the Add-result var name for the out_buf write assertion.
        let add_var = var_name_of_let_binding(add_line.unwrap().1);
        // two out_buf writes at correct channel slots
        assert!(find_line(&src, |l| l.trim() == "out_buf[chunk_cell_base + 0u * cells_per_chunk + cell_idx] = c_0;").is_some(),
                "ch0 out_buf write absent");
        let ch1_write = format!("out_buf[chunk_cell_base + 1u * cells_per_chunk + cell_idx] = {add_var};");
        assert!(find_line(&src, |l| l.trim() == &ch1_write[..]).is_some(),
                "ch1 out_buf write absent, expected `{ch1_write}`");
    }

    /// Deep nesting: Root([Mul(Sqrt(Abs(Sub(InputZ, Constant(2.0)))), Constant(0.5))]).
    ///
    /// Validates that every arithmetic emitter (Sub → Abs → Sqrt → Mul × Constant)
    /// produces a let-binding, and that the final out_buf writes the Mul result.
    #[test]
    fn emit_shader_deep_nested_unary_and_binary() {
        let sub = AstNode::Sub(
            Box::new(AstNode::InputZ),
            Box::new(AstNode::Constant(2.0)),
        );
        let abs = AstNode::Abs(Box::new(sub));
        let sqrt = AstNode::Sqrt(Box::new(abs));
        let mul = AstNode::Mul(
            Box::new(sqrt),
            Box::new(AstNode::Constant(0.5)),
        );
        let root = tiny_root(vec![mul]);

        let src = emit_shader(&root).expect("emit must succeed");

        // No error sentinel lines leaked into the WGSL.
        assert!(!src.contains("binary emitter got"),
                "binary emitter error leaked into output:\n{src}");
        assert!(!src.contains("unary emitter got"),
                "unary emitter error leaked into output:\n{src}");

        // Emit ordering assertions (structural):
        //   - Sub binds to a bin_* var
        let sub_line = find_line(&src, |l| {
            l.trim().starts_with("let bin_")
                && l.trim().contains("cell_z - c_")
        }).expect("missing Sub(InputZ, Constant(2.0)) binding");
        let sub_var = var_name_of_let_binding(sub_line.1);
        //   - Abs binds to un_* and uses Sub var
        let abs_line = find_line(&src, |l| {
            let t = l.trim();
            t.starts_with("let un_")
                && t.contains("abs(")
                && t.contains(&format!("({sub_var})"))
        }).expect("missing Abs(Sub) binding");
        let abs_var = var_name_of_let_binding(abs_line.1);
        //   - Sqrt binds to un_* and uses Abs var
        let sqrt_line = find_line(&src, |l| {
            let t = l.trim();
            t.starts_with("let un_")
                && t.contains("sqrt(")
                && t.contains(&format!("({abs_var})"))
        }).expect("missing Sqrt(Abs) binding");
        let sqrt_var = var_name_of_let_binding(sqrt_line.1);
        //   - Mul binds to bin_* and uses Sqrt var + a c_* constant
        let mul_line = find_line(&src, |l| {
            let t = l.trim();
            t.starts_with("let bin_")
                && t.contains(&format!("{sqrt_var} * c_"))
        }).expect("missing Mul(Sqrt, Constant) binding");
        let mul_var = var_name_of_let_binding(mul_line.1);

        // Final out_buf write uses Mul var
        let expected_write = format!("out_buf[chunk_cell_base + 0u * cells_per_chunk + cell_idx] = {mul_var};");
        assert!(find_line(&src, |l| l.trim() == &expected_write[..]).is_some(),
                "ch0 out_buf write absent, expected `{expected_write}`");
    }

    // ── error path tests ────────────────────────────────────────────────

    /// Non-Root top-level node → Err.
    #[test]
    fn emit_shader_non_root_returns_err() {
        let err = emit_shader(&AstNode::Constant(0.0))
            .expect_err("non-Root top node must error");
        let msg = format!("{err:?}");
        assert!(msg.contains("expected AstNode::Root"),
                "error msg should mention Root, got: {msg}");
    }

    /// Empty Root(0 channels) → Ok but 0 writes (degenerate, no panic).
    #[test]
    fn emit_shader_empty_root_succeeds() {
        let src = emit_shader(&tiny_root(vec![]))
            .expect("empty root must not panic");
        assert!(src.contains("fn main_entry("),
                "at minimum must emit main_entry skeleton");
    }

}
