// SPDX-License-Identifier: MIT

//! fp32 WGSL code generation module.
//!
//! Pipeline per world:
//!   AstNode (optimized)  →  wgsl_gen.rs dispatches to emitters/*  →
//!   concat with math_lib.wgsl  →  ComputePipeline  →  cached to disk.

pub mod wgsl_gen;
pub mod fingerprint;
pub mod emitters;
pub mod validate;

/// The big math library — 1200+ lines, included verbatim into every shader
/// build via `include_str!`.
///
/// 铁律①②③ precision utilities, noise samplers, aquifer, biome access,
/// lerp/clamp/floorDiv/floorMod (semantics match Java exactly), etc.
pub const MATH_LIB_WGSL: &str = include_str!("math_lib.wgsl");
