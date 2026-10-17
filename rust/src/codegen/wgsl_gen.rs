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
    /// Single-pass mode — the whole Root in one shader. Entry point
    /// `main_entry`. Used for tests and as the pre-split fallback.
    Single,
    /// Shard A — channels `[0, 11)` (barrier ... initialDensityWithoutJaggedness).
    /// Entry point `main_pass_a`.
    A,
    /// Shard B — channel 11 (`finalDensity`). Entry point `main_pass_b`.
    B,
    /// Shard C — channels `[12, N)` (vein x3). Entry point `main_pass_c`.
    C,
    /// §20 block-output shard: reads the finalDensity corner channel from
    /// `out_buf`, trilinearly interpolates per BLOCK, and writes one block
    /// state byte per block to `block_buf` (binding 7). Entry point
    /// `main_block`. Dispatched after the corner shards.
    BlockOut,
    /// §14 Pass E: inner shard that computes all `InnerRef` corners.
    Inner,
}

impl Pass {
    pub fn entry_point(self) -> &'static str {
        match self {
            Pass::Single => "main_entry",
            Pass::A => "main_pass_a",
            Pass::B => "main_pass_b",
            Pass::C => "main_pass_c",
            Pass::BlockOut => "main_block",
            Pass::Inner => "main_pass_inner",
        }
    }
}

/// One channel shard: a contiguous `[start, end)` range of Root channels
/// emitted into its own shader module + pipeline + dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shard {
    pub pass: Pass,
    pub start: usize,
    pub end: usize,
}

pub fn plan_shards(channel_count: usize) -> Vec<Shard> {
    // Only two channels are consumed by the Java write-back:
    //   11 = finalDensity  (solid/air decision + aquifer input)
    //   12 = veinToggle    (ore-vein coarse filter)
    // Channels 0..11 and 13.. are computed but never read back. Shard A
    // existed only to populate shared memory that CacheInterpolated no
    // longer uses (see cache_like.rs — it is now a pure pass-through),
    // so emitting channels 0..11 is pure waste. Dropping it cuts
    // Mali-G610 startup compile time (shard A ~= 1.15s) and GPU power,
    // with no change to the Java reader: out_buf slots remain global,
    // so ch11 / ch12 still land at their original offsets.
    const FD: usize = 11;  // finalDensity
    const VT: usize = 12;  // veinToggle
    // Shard C now emits all three ore-vein channels (12/13/14); BlockOut
    // interpolates their corners to decide vein placement.
    const VT_END: usize = 15;
    // Small diagnostic / unit-test blobs carry fewer channels than
    // finalDensity's index (11). They cannot use the sharded layout, so
    // fall back to a single all-channel shader — the emitters they
    // exercise stay reachable, and the numerical-validation tests keep
    // working unchanged. Real NoiseRouters always have >= 13 channels.
    if channel_count <= FD {
        if channel_count == 0 {
            return Vec::new();
        }
        return vec![Shard { pass: Pass::Single, start: 0, end: channel_count }];
    }
    let mut out = Vec::new();
    out.push(Shard { pass: Pass::B, start: FD, end: (FD + 1).min(channel_count) });
    if channel_count > VT {
        out.push(Shard { pass: Pass::C, start: VT, end: VT_END.min(channel_count) });
    }
    // §20 block-output shard: not bound to any channel (start==end); it reads
    // ch11 corners from out_buf and writes block-state bytes. Gated.
    if block_out_enabled() {
        out.push(Shard { pass: Pass::Inner, start: 0, end: 0 });
        out.push(Shard { pass: Pass::BlockOut, start: 0, end: 0 });
    }
    out
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
    pub inner_count: usize,
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
    /// Channel range `[start, end)` this shader is responsible for.
    /// `(0, usize::MAX)` means "all channels" (single-pass mode).
    pub channel_range: (usize, usize),
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
            inner_count: 0,
            memo_stack: vec![(false, rustc_hash::FxHashMap::default())],
            in_shadow: false,
            channel_range: (0, usize::MAX),
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
/// Prune `math_lib.wgsl` down to the top-level items (fn/struct/const)
/// that are reachable from `usage` (the concatenated shared-decls + prologue
/// + entry body).
///
/// The full lib is ~860 lines but a given NoiseRouter only references a
/// fraction of it. Mali-G610 aborts when asked to compile a ~5.4k-line
/// shader, so shaving unused helpers matters. Blocks are emitted in their
/// original order, so any declaration-order assumptions stay intact.
///
/// Safety: if any referenced name cannot be matched to a block, we fall back
/// to the full library rather than risk emitting an incomplete module.
fn emit_pruned_math_lib(usage: &str) -> String {
    use std::collections::HashSet;

    let lib = super::MATH_LIB_WGSL;

    struct Block {
        name: String,
        text: String,
    }

    // Split into blocks: a block starts at a top-level `fn`/`struct`/`const`
    // line (column 0) and runs until the next such line.
    let mut blocks: Vec<Block> = Vec::new();
    let mut cur = String::new();
    let mut cur_name: Option<String> = None;
    for line in lib.lines() {
        let is_start = line.starts_with("fn ")
            || line.starts_with("struct ")
            || line.starts_with("const ");
        if is_start {
            if let Some(n) = cur_name.take() {
                blocks.push(Block { name: n, text: std::mem::take(&mut cur) });
            } else {
                cur.clear(); // drop leading license/comment header
            }
            let rest = line.split_whitespace().nth(1).unwrap_or("");
            let name = rest
                .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .next()
                .unwrap_or("")
                .to_string();
            cur_name = Some(name);
        }
        cur.push_str(line);
        cur.push('\n');
    }
    if let Some(n) = cur_name.take() {
        blocks.push(Block { name: n, text: cur });
    }

    let name_set: HashSet<&str> = blocks.iter().map(|b| b.name.as_str()).collect();

    // Whole-word scan: collect defined names that appear in `text`.
    fn refs_in(text: &str, name_set: &HashSet<&str>) -> HashSet<String> {
        let mut out = HashSet::new();
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let c = bytes[i] as char;
            if c.is_ascii_alphabetic() || c == '_' {
                let start = i;
                while i < bytes.len() {
                    let d = bytes[i] as char;
                    if d.is_ascii_alphanumeric() || d == '_' { i += 1; } else { break; }
                }
                let word = &text[start..i];
                if name_set.contains(word) {
                    out.insert(word.to_string());
                }
            } else {
                i += 1;
            }
        }
        out
    }

    // Transitive closure from the usage roots.
    let mut needed: HashSet<String> = refs_in(usage, &name_set);
    let mut queue: Vec<String> = needed.iter().cloned().collect();
    while let Some(n) = queue.pop() {
        let Some(b) = blocks.iter().find(|b| b.name == n) else {
            // Unknown name referenced — bail out to the full library.
            return lib.to_string();
        };
        for r in refs_in(&b.text, &name_set) {
            if needed.insert(r.clone()) {
                queue.push(r);
            }
        }
    }

    let mut out = String::new();
    for b in &blocks {
        if needed.contains(&b.name) {
            out.push_str(&b.text);
        }
    }
    out
}

pub fn emit_shader_for_pass(root: &AstNode, pass: Pass) -> Result<String> {
    let ctx = GenCtx::new(pass);
    emit_shader_inner(root, ctx)
}

/// Emit the shader module for one channel `shard` (see [`Shard`]).
pub fn emit_shader_for_shard(
    root: &AstNode,
    shard: Shard,
    inners: &[AstNode],
    outer: Option<&AstNode>,
) -> Result<String> {
    if shard.pass == Pass::Inner {
        let mut ctx = GenCtx::new(Pass::Inner);
        let mut body = String::with_capacity(8192);
        for (i, inner) in inners.iter().enumerate() {
            let var = super::emitters::emit_node(inner, &mut ctx)?;
            body.push_str(&format!(
                "  out_buf[chunk_cell_base + {}u * cells_per_chunk + cell_idx] = {};\n",
                16 + i, var
            ));
        }
        
        let prologue = MAIN_PASS_C_PROLOGUE.replace("fn main_pass_c(", "fn main_pass_inner(");
        let mut usage = String::with_capacity(8192);
        usage.push_str(&prologue);
        usage.push_str(&body);
        for line in &ctx.lines { usage.push_str(line); }
        
        let math_lib = emit_pruned_math_lib(&usage);
        let mut out = String::with_capacity(math_lib.len() + usage.len() + 1024);
        out.push_str(&math_lib);
        out.push_str(BINDINGS_HEADER);
        for decl in &ctx.shared_decls {
            out.push_str(decl);
            out.push('\n');
        }
        out.push_str(&prologue);
        for line in &ctx.lines { out.push_str(line); out.push('\n'); }
        out.push_str(&body);
        out.push('}');
        return Ok(out);
    }

    if shard.pass == Pass::BlockOut {
        let mut ctx = GenCtx::new(Pass::BlockOut);
        
        let d_expr = if let Some(_out_tree) = outer {
            // DIAGNOSTIC 2: read the first inner directly to see if data is present.
            "tri_at(cbase + 16u * cpc, ix, iy, iz, ccx, s, tx, ty, tz)".to_string()
        } else {
            "d".to_string()
        };
        
        let mut injected = String::with_capacity(4096);
        // `emit_node` relies on these aliases for coordinate access.
        injected.push_str("  let cell_x: f32 = f32(wx);\n");
        injected.push_str("  let cell_y: f32 = f32(wy);\n");
        injected.push_str("  let cell_z: f32 = f32(wz);\n");
        for line in &ctx.lines {
            injected.push_str(line);
            injected.push('\n');
        }
        for decl in &ctx.shared_decls {
            // Not supported in BlockOut for now (no workgroup mem), but log if any.
            injected.push_str(decl);
            injected.push('\n');
        }
        injected.push_str(&format!("  let d_final: f32 = {};\n", d_expr));
        
        let mut usage = String::with_capacity(8192);
        usage.push_str(BLOCK_OUT_HEADER);
        usage.push_str(BLOCK_OUT_ENTRY);
        usage.push_str(&injected);
        
        let math_lib = emit_pruned_math_lib(&usage);
        let mut out = String::with_capacity(math_lib.len() + usage.len() + 1024);
        out.push_str(&math_lib);
        out.push_str(BLOCK_OUT_HEADER);
        
        // Replace the fixed density `d` with our per-block `d_final` and
        // inject the AST evaluation right after the block's world coordinates.
        let entry = BLOCK_OUT_ENTRY.replace(
            "  // §23: add the Beardifier contribution (per block, before the d>0 test).\n  var dd = d;",
            &format!("{}\n  // §23: add the Beardifier contribution.\n  var dd = d_final;", injected)
        );
        out.push_str(&entry);
        return Ok(out);
    }
    let mut ctx = GenCtx::new(shard.pass);
    ctx.channel_range = (shard.start, shard.end);
    emit_shader_inner(root, ctx)
}

fn emit_shader_inner(root: &AstNode, mut ctx: GenCtx) -> Result<String> {
    let AstNode::Root(_) = root else {
        return Err(anyhow!(
            "emit_shader_for_pass: expected AstNode::Root, got {root:?}"
        ));
    };

    // Bring-up: emit the tiny shader for *every* pass, so the diagnostic
    // keeps working now that WorldContext emits per-shard modules.
    if crate::minimal_shader() {
        crate::wgpu_debug!("[c2me-wgpu][codegen] MINIMAL shader requested");
        let mut out = String::with_capacity(2048);
        out.push_str(BINDINGS_HEADER);
        out.push_str(MINIMAL_ENTRY);
        return Ok(out);
    }

    // Phase A — emit entry-point body.
    emitters::root::emit(root, &mut ctx)?;

    // Phase B — assemble final shader source in the correct WGSL order.
    let prologue = match ctx.pass {
        Pass::Single => MAIN_ENTRY_PROLOGUE,
        Pass::A => MAIN_PASS_A_PROLOGUE,
        Pass::B => MAIN_PASS_B_PROLOGUE,
        Pass::C => MAIN_PASS_C_PROLOGUE,
        // BlockOut is emitted by emit_shader_for_shard's early-return path
        // and never reaches here.
        Pass::BlockOut | Pass::Inner => unreachable!("BlockOut/Inner handled separately"),
    };

    // Collect the code that actually *uses* the math library, so we can prune
    // unreferenced helpers out of it.
    let mut usage = String::new();
    for decl in &ctx.shared_decls {
        usage.push_str(decl);
        usage.push('\n');
    }
    usage.push_str(prologue);
    for line in &ctx.lines {
        usage.push_str(line);
        usage.push('\n');
    }

    let math_lib = emit_pruned_math_lib(&usage);

    let mut out = String::with_capacity(8192);
    out.push_str(&math_lib);
    out.push_str(BINDINGS_HEADER);
    for decl in &ctx.shared_decls {
        out.push_str(decl);
        out.push('\n');
    }
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

/// Minimal `main_entry` used by `C2ME_WGPU_MINIMAL_SHADER=1` (bring-up test).
///
/// Keeps the same bind group layout and entry signature; body just writes a
/// deterministic value into `out_buf`. If a device compiles+runs this but
/// aborts on the full shader, the problem is shader size, not wgpu integration.
/// §20 block-output entry point. Reads the finalDensity (ch11) corner values
/// that the corner shards already wrote into `out_buf`, trilinearly
/// interpolates one value per BLOCK, and writes a block-state byte:
///   0 = air (d <= 0), 1 = stone (d > 0).
/// This is the "GPU outputs block states" path; Java only writes the palette.
/// (Aquifer/vein/beard are NOT applied here yet — this is the minimal loop.)
const BLOCK_OUT_ENTRY: &str = r#"
// §23 Beardifier. Layout of beard_pj (per chunk, fixed stride 16 u32):
//   [0] piece_count, [1] junction_count,
//   [2..2+7*pc] pieces (7 i32: minX,minY,minZ,maxX,maxY,maxZ,adapt,delta),
//   junctions follow (3 i32 each).
const BEARD_STRIDE: u32 = 4096u;
const BEARD_W_OFF: u32 = 13824u; // 24*24*24, beard_w section length
const ADAPT_BURY: i32 = 1; const ADAPT_THIN: i32 = 2;
const ADAPT_BOX: i32 = 3; const ADAPT_ENCAP: i32 = 4;

fn beard_gsw(x: i32, y: i32, z: i32, yy: i32) -> f32 {
  let i = x + 12; let j = y + 12; let k = z + 12;
  if (i < 0 || i >= 24 || j < 0 || j >= 24 || k < 0 || k >= 24) { return 0.0; }
  let d = f32(yy) + 0.5;
  let e = f32(x * x) + d * d + f32(z * z);
  // MC uses a double fastInverseSqrt; f32 with exact inverseSqrt is within
  // ~1e-7 relative, negligible for structure shaping.
  let f = -d * inverseSqrt(e / 2.0) / 2.0;
  return f * bitcast<f32>(beard_buf[u32(k * 576 + i * 24 + j)]);
}
fn beard_mag(x: f32, y: f32, z: f32) -> f32 {
  let d = sqrt(x * x + y * y + z * z);
  return 1.0 - clamp(d / 6.0, 0.0, 1.0);
}
fn beard_sample(base: u32, bx: i32, by: i32, bz: i32) -> f32 {
  let pc = i32(beard_buf[BEARD_W_OFF + base]);
  let jc = i32(beard_buf[BEARD_W_OFF + base + 1u]);
  var d = 0.0;
  var o = base + 2u;
  for (var n: i32 = 0; n < pc; n = n + 1) {
    let minx = i32(beard_buf[BEARD_W_OFF + o]); let miny = i32(beard_buf[BEARD_W_OFF + o+1u]);
    let minz = i32(beard_buf[BEARD_W_OFF + o+2u]); let maxx = i32(beard_buf[BEARD_W_OFF + o+3u]);
    let maxy = i32(beard_buf[BEARD_W_OFF + o+4u]); let maxz = i32(beard_buf[BEARD_W_OFF + o+5u]);
    let ad = i32(beard_buf[BEARD_W_OFF + o+6u]); let gld = i32(beard_buf[BEARD_W_OFF + o+7u]);
    o = o + 8u;
    let m = max(0, max(minx - bx, bx - maxx));
    let nn = max(0, max(minz - bz, bz - maxz));
    let oo = miny + gld;
    let p = by - oo;
    var q = 0; if (ad == ADAPT_BURY || ad == ADAPT_THIN) { q = p; }
    else if (ad == ADAPT_BOX) { q = max(0, max(oo - by, by - maxy)); }
    else if (ad == ADAPT_ENCAP) { q = max(0, max(miny - by, by - maxy)); }
    if (ad == ADAPT_BURY) { d = d + beard_mag(f32(m), f32(q) * 0.5, f32(nn)); }
    else if (ad == ADAPT_THIN || ad == ADAPT_BOX) { d = d + beard_gsw(m, q, nn, p) * 0.8; }
    else if (ad == ADAPT_ENCAP) { d = d + beard_mag(f32(m) * 0.5, f32(q) * 0.5, f32(nn) * 0.5) * 0.8; }
  }
  for (var n: i32 = 0; n < jc; n = n + 1) {
    let sx = i32(beard_buf[BEARD_W_OFF + o]); let sy = i32(beard_buf[BEARD_W_OFF + o+1u]); let sz = i32(beard_buf[BEARD_W_OFF + o+2u]);
    o = o + 3u;
    let r = bx - sx; let l = by - sy; let mm = bz - sz;
    d = d + beard_gsw(r, l, mm, l) * 0.4;
  }
  return d;
}

fn aq_fdiv(a: i32, b: i32) -> i32 {
  let q = a / b; let r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) { return q - 1; }
  return q;
}

// ── Ore veins (§24) ─────────────────────────────────────────────────
// Mirrors net.minecraft.world.gen.OreVeinSampler. Runs on blocks the aquifer
// already classified as solid (density > 0), optionally replacing them with a
// vein ore / raw-ore / stone variant.

fn vein_clamped_map(v: f32, from_lo: f32, from_hi: f32, to_lo: f32, to_hi: f32) -> f32 {
  let t = clamp((v - from_lo) / (from_hi - from_lo), 0.0, 1.0);
  return t * (to_hi - to_lo) + to_lo;
}

// Trilinearly interpolate channel at `base` for the current block.
fn tri_at(base: u32, ix: i32, iy: i32, iz: i32, ccx: i32, s: i32,
          tx: f32, ty: f32, tz: f32) -> f32 {
  let c000 = out_buf[base + u32(ix + iz * ccx + iy * s)];
  let c100 = out_buf[base + u32(ix + 1 + iz * ccx + iy * s)];
  let c010 = out_buf[base + u32(ix + iz * ccx + (iy + 1) * s)];
  let c110 = out_buf[base + u32(ix + 1 + iz * ccx + (iy + 1) * s)];
  let c001 = out_buf[base + u32(ix + (iz + 1) * ccx + iy * s)];
  let c101 = out_buf[base + u32(ix + 1 + (iz + 1) * ccx + iy * s)];
  let c011 = out_buf[base + u32(ix + (iz + 1) * ccx + (iy + 1) * s)];
  let c111 = out_buf[base + u32(ix + 1 + (iz + 1) * ccx + (iy + 1) * s)];
  let x00 = c000 + tx * (c100 - c000);
  let x10 = c010 + tx * (c110 - c010);
  let x01 = c001 + tx * (c101 - c001);
  let x11 = c011 + tx * (c111 - c011);
  let y0 = x00 + ty * (x10 - x00);
  let y1 = x01 + ty * (x11 - x01);
  return y0 + tz * (y1 - y0);
}

// 0 = not a vein block (leave as stone); otherwise a block-state code 4..9.
fn vein_state(wx: i32, wy: i32, wz: i32, vt: f32, vr: f32, vg: f32) -> u32 {
  let is_copper = vt > 0.0;
  let min_y: i32 = select(-60, 0, is_copper);
  let max_y: i32 = select(-8, 50, is_copper);
  let e = abs(vt);
  let j = max_y - wy;
  let k = wy - min_y;
  if (k < 0 || j < 0) { return 0u; }
  let l = min(j, k);
  let f = vein_clamped_map(f32(l), 0.0, 20.0, -0.2, 0.0);
  if (e + f < 0.4) { return 0u; }
  // randomDeriver.split(wx, wy, wz): Xoroshiro(hashCode ^ seedLo, seedHi).
  let h = java_hash3(wx, wy, wz);
  let m = vec2<u32>(h.x ^ params._pad[5], h.y ^ params._pad[6]);
  var st: vec4<u32> = vec4<u32>(m.x, m.y, params._pad[7], params._pad[8]);
  if (xoro_next_float(&st) > 0.7) { return 0u; }
  if (vr >= 0.0) { return 0u; }
  let g = vein_clamped_map(e, 0.4, 0.6, 0.1, 0.3);
  if (xoro_next_float(&st) < g && vg > -0.3) {
    if (xoro_next_float(&st) < 0.02) {
      return select(9u, 7u, is_copper);   // raw iron : raw copper
    }
    return select(8u, 6u, is_copper);     // deepslate iron : copper ore
  }
  return select(5u, 4u, is_copper);       // tuff : granite
}

@compute @workgroup_size(4, 4, 4)
fn main_block(@builtin(global_invocation_id) gid: vec3<u32>) {
  let num_chunks: i32 = params.batch_count_x;
  let cx_i32: i32 = i32(gid.x);
  let cy_i32: i32 = i32(gid.y);
  let cz_i32: i32 = i32(gid.z);
  let chunk_index: i32 = cz_i32 / 16;
  let bz: i32 = cz_i32 % 16;
  let height_i: i32 = i32(params.height_f);
  if (cx_i32 >= 16 || cy_i32 >= height_i || bz >= 16 || chunk_index >= num_chunks) { return; }

  let cell_w: i32 = i32(params.cell_width);
  let cell_h: i32 = i32(params.cell_height);
  let ccx: i32 = 16 / cell_w + 1;
  let ccz: i32 = 16 / cell_w + 1;
  let ccy: i32 = height_i / cell_h + 1;
  let s: i32 = ccx * ccz;
  let cpc: u32 = u32(ccx * ccy * ccz);
  let cbase: u32 = u32(chunk_index) * params._pad[11] * cpc;
  let ch11_base: u32 = cbase + 11u * cpc;

  let ix: i32 = cx_i32 / cell_w;
  let iz: i32 = bz / cell_w;
  let iy: i32 = cy_i32 / cell_h;
  let tx: f32 = f32(cx_i32 % cell_w) / f32(cell_w);
  let tz: f32 = f32(bz % cell_w) / f32(cell_w);
  let ty: f32 = f32(cy_i32 % cell_h) / f32(cell_h);

  let c000 = out_buf[ch11_base + u32(ix + iz * ccx + iy * s)];
  let c100 = out_buf[ch11_base + u32(ix + 1 + iz * ccx + iy * s)];
  let c010 = out_buf[ch11_base + u32(ix + iz * ccx + (iy + 1) * s)];
  let c110 = out_buf[ch11_base + u32(ix + 1 + iz * ccx + (iy + 1) * s)];
  let c001 = out_buf[ch11_base + u32(ix + (iz + 1) * ccx + iy * s)];
  let c101 = out_buf[ch11_base + u32(ix + 1 + (iz + 1) * ccx + iy * s)];
  let c011 = out_buf[ch11_base + u32(ix + (iz + 1) * ccx + (iy + 1) * s)];
  let c111 = out_buf[ch11_base + u32(ix + 1 + (iz + 1) * ccx + (iy + 1) * s)];

  let x00 = c000 + tx * (c100 - c000);
  let x10 = c010 + tx * (c110 - c010);
  let x01 = c001 + tx * (c101 - c001);
  let x11 = c011 + tx * (c111 - c011);
  let y0 = x00 + ty * (x10 - x00);
  let y1 = x01 + ty * (x11 - x01);
  let d = y0 + tz * (y1 - y0);

  let per_chunk: u32 = u32(16 * height_i * 16);
  let bidx: u32 = u32(chunk_index) * per_chunk + u32(cx_i32 + bz * 16 + cy_i32 * 256);

  // §22 aquifer (12-cell scan + FluidLevel). Global block coords:
  let chunk_cx = positions[chunk_index * 2];
  let chunk_cz = positions[chunk_index * 2 + 1];
  let wx: i32 = chunk_cx * 16 + cx_i32;
  let wz: i32 = chunk_cz * 16 + bz;
  let wy: i32 = i32(params.min_y_f) + cy_i32;

  // §23: add the Beardifier contribution (per block, before the d>0 test).
  var dd = d;
  {
    let bbase = u32(chunk_index) * BEARD_STRIDE;
    let bcount = beard_buf[BEARD_W_OFF + bbase];
    if (bcount > 0u) { dd = dd + beard_sample(bbase, wx, wy, wz); }
  }

  // §22: per-dimension aquifer mode from params._pad[1..5].
  let has_aq = params._pad[1];
  if (has_aq == 0u) {
    // No aquifers (Nether/End): sea-level semantics, no 12-cell scan.
    if (dd > 0.0) { block_buf[bidx] = 1u; return; }
    let sea = i32(params._pad[2]);
    let dfl = params._pad[3];
    let lava_y = i32(params._pad[4]);
    if (wy < min(lava_y, sea)) { block_buf[bidx] = 3u; return; }
    if (wy < sea) { block_buf[bidx] = dfl; return; }
    block_buf[bidx] = 0u; return;
  }

  // d > 0 → solid. MC then lets OreVeinSampler replace solid blocks.
  if (dd > 0.0) {
    var st: u32 = 1u;
    if (params._pad[9] == 1u) {
      let vt = tri_at(cbase + 12u * cpc, ix, iy, iz, ccx, s, tx, ty, tz);
      let vr = tri_at(cbase + 13u * cpc, ix, iy, iz, ccx, s, tx, ty, tz);
      let vg = tri_at(cbase + 14u * cpc, ix, iy, iz, ccx, s, tx, ty, tz);
      let vs = vein_state(wx, wy, wz, vt, vr, vg);
      if (vs != 0u) { st = vs; }
    }
    block_buf[bidx] = st;
    return;
  }

  // FluidLevelSampler: y < min(-54, seaLevel) → LAVA.
  if (wy < -54) { block_buf[bidx] = 3u; return; }

  let min_y_i = i32(params.min_y_f);
  let start_y_i = aq_fdiv(min_y_i, 12) - 1;
  let aq_sy = aq_fdiv(min_y_i + height_i, 12) - aq_fdiv(min_y_i, 12) + 3;
  let aq_base = chunk_index * aq_sy * 9; // AQ_SX(3) * AQ_SY * AQ_SZ(3)
  // aquifer_buf holds aq_p in [0, half) and aq_s in [half, 2*half).
  let aq_half: u32 = params._pad[10];

  let l = aq_fdiv(wx - 5, 16);
  let m = aq_fdiv(wy + 1, 12);
  let n = aq_fdiv(wz - 5, 16);

  var o: i32 = 2147483647; var p: i32 = 2147483647; var ri: i32 = -1;
  for (var u: i32 = 0; u <= 1; u = u + 1) {
    for (var v: i32 = -1; v <= 1; v = v + 1) {
      for (var w: i32 = 0; w <= 1; w = w + 1) {
        let i = (l + u) - chunk_cx + 1;
        let j = (m + v) - start_y_i;
        let k = (n + w) - chunk_cz + 1;
        if (i < 0 || i >= 3 || k < 0 || k >= 3 || j < 0 || j >= aq_sy) { continue; }
        let idx = aq_base + (j * 3 + k) * 3 + i;
        let cp = aquifer_buf[u32(idx)];
        let ad = cp.x - wx; let ae = cp.y - wy; let af = cp.z - wz;
        let ag = ad * ad + ae * ae + af * af;
        if (o >= ag) { ri = idx; p = o; o = ag; }
        else if (p >= ag) { p = ag; }
      }
    }
  }
  if (ri < 0) { block_buf[bidx] = 1u; return; }
  let rcly = aquifer_buf[u32(ri)].w;
  block_buf[bidx] = select(0u, u32(aquifer_buf[aq_half + u32(ri)].x), wy < rcly);
}
"#;

/// `C2ME_WGPU_BLOCK_OUT=1` enables the block-output shard (§20 minimal loop).
/// BlockOut is the only GPU write-back path (the CPU density path was
/// removed as a measured net loss, §20). It is no longer switchable: a
/// session without it would write empty chunks.
pub fn block_out_enabled() -> bool {
    true
}

const MINIMAL_ENTRY: &str = r#"
@compute @workgroup_size(4, 4, 4)
fn main_entry(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id)  lid: vec3<u32>,
) {
  let idx: u32 = gid.x + gid.y * 4u + gid.z * 16u;
  if (idx < arrayLength(&out_buf)) {
    out_buf[idx] = f32(gid.x) + f32(gid.y) * 0.5 + f32(gid.z) * 0.25
                 + f32(lid.x) * 0.0;
  }
}
"#;

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
  _pad:           array<u32, 12>,
};

@group(0) @binding(0) var<storage, read>        params:       WorldgenParams;
@group(0) @binding(1) var<storage, read>        perm: array<u32>;
@group(0) @binding(2) var<storage, read_write>  out_buf:      array<f32>;
@group(0) @binding(3) var<storage, read>        perm_origin:  array<f32>;
@group(0) @binding(4) var<storage, read>        positions:    array<i32>;
@group(0) @binding(5) var<storage, read>        simplex_perm: array<u32>;
@group(0) @binding(6) var<storage, read>        external_inputs: array<f32>;
"#;

/// Bindings for the standalone BlockOut shard (§22). Deliberately separate
/// from `BINDINGS_HEADER`: this shard uses only params/out_buf/positions/
/// block_buf/aq_p/aq_s (6 storage buffers), so it gets its own bind group
/// layout that stays under `max_storage_buffers_per_shader_stage` (8).
const BLOCK_OUT_HEADER: &str = r#"
struct WorldgenParams {
  batch_origin_x: i32,
  batch_origin_z: i32,
  cell_width:     f32,
  cell_height:    f32,
  min_y_f:        f32,
  height_f:       f32,
  batch_count_x:  i32,
  batch_count_z:  i32,
  _pad:           array<u32, 12>,
};

@group(0) @binding(0) var<storage, read>        params:    WorldgenParams;
@group(0) @binding(2) var<storage, read_write>  out_buf:   array<f32>;
@group(0) @binding(4) var<storage, read>        positions: array<i32>;
@group(0) @binding(7) var<storage, read_write>  block_buf: array<u32>;
// Bindings are merged to stay within 8 storage buffers:
//   aquifer_buf = [ aq_p (N x vec4) | aq_s (N x vec4, only .x used) ]
//   beard_buf   = [ beard_w (13824 u32, bitcast f32) | beard_pj (stride 4096/chunk) ]
// perm / perm_origin are the same buffers the corner shards use.
@group(0) @binding(8) var<storage, read>        aquifer_buf: array<vec4<i32>>;
@group(0) @binding(9) var<storage, read>        beard_buf:   array<u32>;
@group(0) @binding(10) var<storage, read>       perm:        array<u32>;
@group(0) @binding(11) var<storage, read>       perm_origin: array<f32>;
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
  let chunk_cell_base: u32 = u32(chunk_index) * params._pad[11] * cells_per_chunk;

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
  let chunk_cell_base: u32 = u32(chunk_index) * params._pad[11] * cells_per_chunk;
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
  let chunk_cell_base: u32 = u32(chunk_index) * params._pad[11] * cells_per_chunk;
  let world_x_i32: i32 = origin_x + cx_i32 * i32(params.cell_width);
  let world_z_i32: i32 = origin_z + cz_local * i32(params.cell_width);
  let world_y_i32: i32 = i32(params.min_y_f) + cy_i32 * i32(params.cell_height);
  let cell_x: f32 = f32(world_x_i32);
  let cell_y: f32 = f32(world_y_i32);
  let cell_z: f32 = f32(world_z_i32);
"#;
/// `main_pass_c` opening — same prologue as B, entry point `main_pass_c`.
const MAIN_PASS_C_PROLOGUE: &str = r#"
@compute @workgroup_size(4, 4, 4)
fn main_pass_c(
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
  let chunk_cell_base: u32 = u32(chunk_index) * params._pad[11] * cells_per_chunk;
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

    #[test]
    fn plan_shards_only_emits_consumed_channels() {
        // Only ch11 (finalDensity) and ch12 (veinToggle) are read back
        // by Java; everything else is dropped to save shader compile time.
        // BlockOut is on by default, so it always trails the corner shards.
        let shards = plan_shards(16);
        let corner: Vec<_> = shards.iter()
            .filter(|s| s.pass != Pass::BlockOut && s.pass != Pass::Inner)
            .collect();
        assert_eq!(corner.len(), 2);
        assert_eq!((corner[0].pass, corner[0].start, corner[0].end), (Pass::B, 11, 12));
        assert_eq!((corner[1].pass, corner[1].start, corner[1].end), (Pass::C, 12, 15));
        assert_eq!(shards[shards.len() - 2].pass, Pass::Inner);
        assert_eq!(shards.last().unwrap().pass, Pass::BlockOut);
    }

    #[test]
    fn plan_shards_handles_small_roots() {
        assert!(plan_shards(0).is_empty());
        // 1..=11 channels: finalDensity absent -> single all-channel shader
        // so unit-test blobs can still exercise every emitter.
        for n in [1usize, 6, 11] {
            let shards = plan_shards(n);
            assert_eq!(shards.len(), 1, "n={n}");
            assert_eq!((shards[0].pass, shards[0].start, shards[0].end), (Pass::Single, 0, n), "n={n}");
        }
        // 12 channels: ch11 present, ch12 absent -> shard B only (+ Inner/BlockOut).
        let twelve: Vec<_> = plan_shards(12).into_iter()
            .filter(|s| s.pass != Pass::BlockOut && s.pass != Pass::Inner).collect();
        assert_eq!(twelve.len(), 1);
        assert_eq!((twelve[0].pass, twelve[0].start, twelve[0].end), (Pass::B, 11, 12));
        // 13+ channels: both ch11 and ch12 -> shards B and C (+ BlockOut).
        for n in [13usize, 14, 15, 16] {
            let shards: Vec<_> = plan_shards(n).into_iter()
                .filter(|s| s.pass != Pass::BlockOut && s.pass != Pass::Inner).collect();
            assert_eq!(shards.len(), 2, "n={n}");
            assert_eq!((shards[0].pass, shards[0].start, shards[0].end), (Pass::B, 11, 12), "n={n}");
            // Shard C covers all three vein channels, clipped to what exists.
            assert_eq!((shards[1].pass, shards[1].start, shards[1].end),
                       (Pass::C, 12, 15.min(n)), "n={n}");
        }
    }

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
