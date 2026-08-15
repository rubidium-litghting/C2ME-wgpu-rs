/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2021-2026 RubidiumLighting
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

//! fp32 WGSL code generation module.
//!
//! Pipeline per world:
//!   AstNode (optimized)  →  wgsl_gen.rs dispatches to emitters/*  →
//!   concat with math_lib.wgsl  →  ComputePipeline  →  cached to disk.

pub mod wgsl_gen;
pub mod fingerprint;
pub mod emitters;

/// The big math library — 1200+ lines, included verbatim into every shader
/// build via `include_str!`.
///
/// 铁律①②③ precision utilities, noise samplers, aquifer, biome access,
/// lerp/clamp/floorDiv/floorMod (semantics match Java exactly), etc.
pub const MATH_LIB_WGSL: &str = include_str!("math_lib.wgsl");
