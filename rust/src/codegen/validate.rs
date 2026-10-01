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

//! WGSL smoke-test validator — runs `naga` (pure CPU) parse + validation
//! on an emitted shader so we catch emitter bugs WITHOUT spinning up a
//! real wgpu::Device.
//!
//! Phase 3 use: called from `WorldContext::new` after `wgsl_gen::emit_shader`
//! to fail fast on any emitter regression that produces syntactically or
//! semantically invalid WGSL.
//!
//! Design notes:
//!   * naga is the same crate wgpu 30 internally uses for shader
//!     compilation, so a `validate_wgsl` pass here ≈ what the driver will
//!     see (modulo driver-specific capability limits).
//!   * We pass `Capabilities::all()` so naga accepts any extension the
//!     target GPU might support; the *real* device.compile_pipeline step
//!     is still the source of truth for capability enforcement.
//!   * Both `ParseError` and `WithSpan<ValidationError>` implement
//!     `Display` with line/col context, so we just pretty-print them.

use crate::Result;
use anyhow::anyhow;
use naga::front::wgsl::parse_str;
use naga::valid::{Capabilities, ValidationFlags, Validator};

/// Pure-CPU smoke test: parse + validate a WGSL shader string.
///
/// Returns `Ok(())` if the shader is accepted by naga, or `Err` with a
/// human-readable diagnostic (including line/col span) otherwise.
pub fn validate_wgsl(src: &str) -> Result<()> {
    // 1. Parse — catches lexing/syntax errors (missing `;`, unbalanced
    //    braces, unknown keywords, …).
    let module = parse_str(src).map_err(|e| {
        anyhow!("WGSL parse error:\n{e}\n--- shader source ---\n{src}")
    })?;

    // 2. Validate — catches type errors, unknown identifiers, missing
    //    entrypoints, layout violations, …
    let _info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map_err(|e| {
            anyhow!("WGSL validation error:\n{e}\n--- shader source ---\n{src}")
        })?;

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::AstNode;
    use crate::codegen::wgsl_gen::emit_shader;

    // ── happy paths: emit_shader output should always validate ──────────

    /// Minimal valid shader: single Constant channel.
    #[test]
    fn validate_accepts_minimal_emit() {
        let root = AstNode::Root(vec![AstNode::Constant(1.0)]);
        let src = emit_shader(&root).expect("emit");
        validate_wgsl(&src)
            .expect("minimal Constant emit must be valid WGSL");
    }

    /// Arithmetic + coordinate inputs.
    #[test]
    fn validate_accepts_arithmetic_emit() {
        let root = AstNode::Root(vec![
            AstNode::Constant(3.14),
            AstNode::Add(
                Box::new(AstNode::InputX),
                Box::new(AstNode::InputY),
            ),
            AstNode::Mul(
                Box::new(AstNode::Sub(
                    Box::new(AstNode::InputZ),
                    Box::new(AstNode::Constant(2.0)),
                )),
                Box::new(AstNode::Constant(0.5)),
            ),
        ]);
        let src = emit_shader(&root).expect("emit");
        validate_wgsl(&src)
            .expect("arithmetic emit must be valid WGSL");
    }

    /// SimplexNoise2D — direct call to math_lib.wgsl::noise_simplex2d.
    #[test]
    fn validate_accepts_simplex2d_emit() {
        let root = AstNode::Root(vec![AstNode::SimplexNoise2D { seed: 99 }]);
        let src = emit_shader(&root).expect("emit");
        validate_wgsl(&src).expect("Simplex2D emit must be valid WGSL");
    }

    /// SimplexNoise3D — direct call to math_lib.wgsl::noise_simplex3d.
    #[test]
    fn validate_accepts_simplex3d_emit() {
        let root = AstNode::Root(vec![AstNode::SimplexNoise3D { seed: -123 }]);
        let src = emit_shader(&root).expect("emit");
        validate_wgsl(&src).expect("Simplex3D emit must be valid WGSL");
    }

    /// Mixed Root with Perlin + Simplex + arithmetic — verifies the
    /// emitters compose correctly across multiple channels.
    #[test]
    fn validate_accepts_mixed_noise_root() {
        let root = AstNode::Root(vec![
            AstNode::Constant(0.5),
            AstNode::Add(
                Box::new(AstNode::SimplexNoise3D { seed: 1 }),
                Box::new(AstNode::SimplexNoise3D { seed: 2 }),
            ),
            AstNode::Mul(
                Box::new(AstNode::SimplexNoise2D { seed: 3 }),
                Box::new(AstNode::InputZ),
            ),
        ]);
        let src = emit_shader(&root).expect("emit");
        validate_wgsl(&src).expect("mixed-noise Root must be valid WGSL");
    }

    /// Cache wrappers — phase 4-3 real shared-memory implementation.
    /// Verifies that Cache2D + CacheFlat + CacheInterpolated all emit valid
    /// WGSL with proper `var<workgroup>` declarations, `workgroupBarrier()`,
    /// and shared-memory reads — all within a single-pass `main_entry`.
    #[test]
    /// Cache wrappers - all three degrade to pass-through memoisation.
    /// Cache2D / CacheFlat skip shared memory (avoids barrier-in-loop UB);
    /// CacheInterpolated likewise passes through because the dispatch grid
    /// now IS the corner grid (cell_count + 1 per axis).
    #[test]
    fn validate_accepts_cache_wrappers() {
        let root = AstNode::Root(vec![
            AstNode::CacheFlat(Box::new(AstNode::Constant(1.5))),
            AstNode::Cache2D(Box::new(AstNode::SimplexNoise2D { seed: 7 })),
            AstNode::CacheInterpolated(Box::new(AstNode::Add(
                Box::new(AstNode::InputX),
                Box::new(AstNode::InputY),
            ))),
            AstNode::Cache2D(Box::new(AstNode::CacheFlat(Box::new(
                AstNode::SimplexNoise3D { seed: 42 },
            )))),
        ]);
        assert!(!crate::codegen::wgsl_gen::requires_two_pass(&root),
            "caches must NOT require two-pass dispatch");

        let src = emit_shader(&root).expect("emit single-pass shader");
        validate_wgsl(&src).expect("cache-wrappers WGSL must be valid");

        assert!(src.contains("fn main_entry("),
            "must use single-pass entry point main_entry");
        assert!(!src.contains("workgroupBarrier();"),
            "cache wrappers must not emit barriers");
        assert!(!src.contains("var<workgroup>"),
            "cache wrappers must not declare shared memory");
        assert!(src.contains("+ 1;"),
            "must use corner grid (cell_count + 1)");
        assert!(src.contains("return;"),
            "must guard out-of-range invocations");
        assert!(src.contains("out_buf["),
            "single-pass mode must write out_buf");
    }

    // ── error paths: naga must reject obvious garbage ──────────────────

    /// Missing semicolon after `let` — must fail with parse error.
    #[test]
    fn validate_rejects_syntax_error() {
        let bad = "@compute @workgroup_size(1)
fn main() {
  let x: f32 = 1.0
}";
        let err = validate_wgsl(bad).expect_err("must reject missing semicolon");
        let msg = format!("{err}");
        assert!(msg.contains("parse"), "expected parse-phase error: {msg}");
    }

    /// Reference to undefined function — must fail validation.
    #[test]
    fn validate_rejects_undefined_function() {
        let bad = "@compute @workgroup_size(1)
fn main() {
  let x: f32 = undefined_func();
}";
        let err = validate_wgsl(bad).expect_err("must reject undefined func");
        let msg = format!("{err}");
        // naga reports unknown identifiers either at parse or validate phase
        assert!(msg.contains("undefined_func") || msg.contains("Unknown"),
                "expected identifier error mentioning undefined_func, got: {msg}");
    }

    /// Type mismatch: assigning vec3 to f32 scalar.
    #[test]
    fn validate_rejects_type_mismatch() {
        let bad = "@compute @workgroup_size(1)
fn main() {
  let x: f32 = vec3<f32>(1.0, 2.0, 3.0);
}";
        validate_wgsl(bad).expect_err("must reject type mismatch");
    }
}
