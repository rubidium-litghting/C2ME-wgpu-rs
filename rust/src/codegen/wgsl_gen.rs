// SPDX-License-Identifier: MIT

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
    /// §3.5 block-output shard: reads the finalDensity corner channel from
    /// `out_buf`, trilinearly interpolates per BLOCK, and writes one block
    /// state byte per block to `block_buf` (binding 7). Entry point
    /// `main_block`. Dispatched after the corner shards.
    BlockOut,
    /// Pass E (§15.2.2): inner shard that computes all `InnerRef` corners.
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
    // §3.5 block-output shard: not bound to any channel (start==end); it reads
    // ch11 corners from out_buf and writes block-state bytes. Gated.
    // BlockOut is the only GPU write-back path (the CPU density path was
    // removed as a measured net loss, §8.2), so it is unconditional.
    out.push(Shard { pass: Pass::Inner, start: 0, end: 0 });
    out.push(Shard { pass: Pass::BlockOut, start: 0, end: 0 });
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
/// Net change in `{`/`}` depth contributed by one WGSL line, skipping
/// braces inside `//` and `/* */` comments. WGSL block comments do not nest.
fn brace_delta(line: &str, in_block_comment: &mut bool) -> i32 {
    let b = line.as_bytes();
    let mut i = 0;
    let mut delta = 0i32;
    while i < b.len() {
        if *in_block_comment {
            if i + 1 < b.len() && b[i] == b'*' && b[i + 1] == b'/' {
                *in_block_comment = false;
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        if i + 1 < b.len() && b[i] == b'/' && b[i + 1] == b'/' {
            break;
        }
        if i + 1 < b.len() && b[i] == b'/' && b[i + 1] == b'*' {
            *in_block_comment = true;
            i += 2;
            continue;
        }
        match b[i] {
            b'{' => delta += 1,
            b'}' => delta -= 1,
            _ => {}
        }
        i += 1;
    }
    delta
}

/// Prune `math_lib.wgsl` down to the top-level items (fn/struct/const)
/// that are reachable from `usage` (the concatenated shared-decls + prologue
/// + entry body).
///
/// The full lib is ~1000 lines but a given NoiseRouter only references a
/// fraction of it. Mali-G610 aborts when asked to compile a ~5.4k-line
/// shader, so shaving unused helpers matters. Blocks are emitted in their
/// original order, so any declaration-order assumptions stay intact.
///
/// A block starts at a top-level (brace depth 0, not inside a comment)
/// `fn` / `struct` / `const` line. Tracking brace depth means a keyword
/// appearing inside a function body or a block comment can never be
/// mistaken for a declaration — the earlier "column-0 prefix" heuristic
/// could mis-split if the library ever grew such a line.
///
/// Safety: the blocks plus the dropped prefix are checked to reconstruct
/// the library exactly; on any mismatch (or if a referenced name cannot be
/// matched to a block) we fall back to the full library rather than risk
/// emitting an incomplete module.
fn emit_pruned_math_lib(usage: &str) -> String {
    use std::collections::HashSet;

    let lib = super::MATH_LIB_WGSL;

    struct Block {
        name: String,
        text: String,
    }

    let mut prefix = String::new();
    let mut blocks: Vec<Block> = Vec::new();
    let mut cur = String::new();
    let mut cur_name: Option<String> = None;
    let mut depth: i32 = 0;
    let mut in_block_comment = false;

    for line in lib.lines() {
        let starts_block = !in_block_comment
            && depth == 0
            && (line.starts_with("fn ")
                || line.starts_with("struct ")
                || line.starts_with("const "));
        if starts_block {
            let rest = line.split_whitespace().nth(1).unwrap_or("");
            let name = rest
                .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .next()
                .unwrap_or("")
                .to_string();
            if let Some(prev) = cur_name.take() {
                blocks.push(Block { name: prev, text: std::mem::take(&mut cur) });
            } else {
                prefix = std::mem::take(&mut cur);
            }
            cur_name = Some(name);
        }
        cur.push_str(line);
        cur.push('\n');
        depth += brace_delta(line, &mut in_block_comment);
    }
    if let Some(prev) = cur_name.take() {
        blocks.push(Block { name: prev, text: cur });
    } else {
        prefix.push_str(&cur);
    }

    // Lossless-reconstruction guard: header + blocks must equal the source.
    // Any dropped or duplicated line (e.g. a mis-split) trips this and we
    // ship the full library instead.
    let reconstructed: String = prefix
        .chars()
        .chain(blocks.iter().flat_map(|b| b.text.chars()))
        .collect();
    if reconstructed.trim_end() != lib.trim_end() {
        return lib.to_string();
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
    barrier_noise: Option<&AstNode>,
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
        
        let prologue = prologue_for(Pass::Inner);
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
        
        let d_expr = if let Some(out_tree) = outer {
            super::emitters::emit_node(out_tree, &mut ctx)?
        } else {
            "d".to_string()
        };
        
        let b_expr = if let Some(b_tree) = barrier_noise {
            super::emitters::emit_node(b_tree, &mut ctx)?
        } else {
            "0.0".to_string()
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
        injected.push_str(&format!("  let barrier_val: f32 = {};\n", b_expr));
        
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
        //
        // `str::replace` is silent when the needle is absent, which would
        // ship a shader that ignores the AST (DEVELOPMENT.md §3.5). Assert the
        // anchor exists, and assert the substitution actually happened.
        const BLOCK_OUT_INJECT_ANCHOR: &str =
            "  // \u{a7}8.4: add the Beardifier contribution (per block, before the d>0 test).\n  var dd = d;";
        assert!(
            BLOCK_OUT_ENTRY.contains(BLOCK_OUT_INJECT_ANCHOR),
            "block_out.wgsl lost the AST-injection anchor; update \
             BLOCK_OUT_INJECT_ANCHOR alongside the WGSL"
        );
        let entry = BLOCK_OUT_ENTRY.replace(
            BLOCK_OUT_INJECT_ANCHOR,
            &format!("{}\n  // \u{a7}8.4: add the Beardifier contribution.\n  var dd = d_final;", injected)
        );
        debug_assert!(entry.contains("var dd = d_final;"), "AST injection did not take effect");
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

    // Phase A — emit entry-point body.
    emitters::root::emit(root, &mut ctx)?;

    // Phase B — assemble final shader source in the correct WGSL order.
    // BlockOut/Inner are emitted by emit_shader_for_shard's early-return
    // paths and never reach here; prologue_for() still covers them.
    let prologue = prologue_for(ctx.pass);

    // Collect the code that actually *uses* the math library, so we can prune
    // unreferenced helpers out of it.
    let mut usage = String::new();
    for decl in &ctx.shared_decls {
        usage.push_str(decl);
        usage.push('\n');
    }
    usage.push_str(&prologue);
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
    out.push_str(&prologue);
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

/// Minimal `main_entry` (bring-up test, no longer wired to any config).
///
/// Keeps the same bind group layout and entry signature; body just writes a
/// deterministic value into `out_buf`. If a device compiles+runs this but
/// aborts on the full shader, the problem is shader size, not wgpu integration.
/// §3.5 block-output entry point. Reads the finalDensity (ch11) corner values
/// that the corner shards already wrote into `out_buf`, trilinearly
/// interpolates one value per BLOCK, and writes a block-state byte:
///   0 = air (d <= 0), 1 = stone (d > 0).
/// This is the "GPU outputs block states" path; Java only writes the palette.
/// (Aquifer/vein/beard are NOT applied here yet — this is the minimal loop.)
const BLOCK_OUT_ENTRY: &str = include_str!("block_out.wgsl");
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

/// Bindings for the standalone BlockOut shard (§3.5). Deliberately separate
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

/// `@compute` entry-point prologue shared by every pass.
///
/// All passes (`main_entry`, `main_pass_a/b/c`, `main_pass_inner`) run the
/// identical cell/chunk coordinate setup and differ only in the function
/// name, so the body lives here once and is instantiated per pass by
/// [`prologue_for`]. Keeping a single copy is deliberate: the stride /
/// origin arithmetic below is the contract with `mc_stride.rs` and the Java
/// readback layout, and a divergence between passes is a silent data bug.
const PROLOGUE_TEMPLATE: &str = r#"
@compute @workgroup_size(4, 4, 4)
fn __ENTRY__(
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
  let chunk_cell_base: u32 = u32(chunk_index) * 144u * cells_per_chunk;

  // ── 铁律①: i32 world coords via origin subtraction ──────────────────
  let world_x_i32: i32 = origin_x + cx_i32 * i32(params.cell_width);
  let world_z_i32: i32 = origin_z + cz_local * i32(params.cell_width);
  let world_y_i32: i32 = i32(params.min_y_f) + cy_i32 * i32(params.cell_height);

  // ── f32 noise inputs (the coordinate emitter returns these names) ───
  let cell_x: f32 = f32(world_x_i32);
  let cell_y: f32 = f32(world_y_i32);
  let cell_z: f32 = f32(world_z_i32);
"#;

/// Instantiate [`PROLOGUE_TEMPLATE`] for `pass`'s entry-point name.
fn prologue_for(pass: Pass) -> String {
    PROLOGUE_TEMPLATE.replace("__ENTRY__", pass.entry_point())
}

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
