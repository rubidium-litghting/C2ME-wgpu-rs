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

//! AST module — intermediate representation of a Minecraft NoiseRouter.
//!
//! The AST is:
//!   1. Deserialized from Java-side NoiseRouterBlobifier binary protocol.
//!   2. Run through 4 optimization passes (constant_fold/cse/dce/normalize).
//!   3. Hashed for pipeline-cache fingerprint + equality.
//!   4. Emitted to fp32 WGSL by the codegen module.

pub mod deserialize;
pub mod equality;
pub mod opto;

/// All 30+ node variants. Recursive enum; children boxed.
///
/// All numeric leaves use `f32` (no f64 anywhere — 计划.txt D05).
#[derive(Debug, Clone, PartialEq)]
pub enum AstNode {
    // ── Scalars ─────────────────────────────────────────────────────────
    Constant(f32),

    // ── Inputs ──────────────────────────────────────────────────────────
    InputX,
    InputY,
    InputZ,

    // ── Binary ops ──────────────────────────────────────────────────────
    Add(Box<AstNode>, Box<AstNode>),
    Sub(Box<AstNode>, Box<AstNode>),
    Mul(Box<AstNode>, Box<AstNode>),
    Div(Box<AstNode>, Box<AstNode>),
    Min(Box<AstNode>, Box<AstNode>),
    Max(Box<AstNode>, Box<AstNode>),

    // ── Unary ops ───────────────────────────────────────────────────────
    Abs(Box<AstNode>),
    Square(Box<AstNode>),
    Cube(Box<AstNode>),
    Sqrt(Box<AstNode>),
    FloorDiv(Box<AstNode>, Box<AstNode>),

    // ── Noise samplers ──────────────────────────────────────────────────
    PerlinSampler {
        seed: i64,
        octaves: i32,
        amplitudes: Vec<f32>,
        lacunaritys: Vec<f32>,
    },
    DoublePerlin {
        a: Box<AstNode>, // PerlinSampler A
        b: Box<AstNode>, // PerlinSampler B
    },
    SimplexNoise2D { seed: i64 },
    SimplexNoise3D { seed: i64 },
    InterpolatedNoiseSampler {
        // TODO: full struct; see protocol appendix §4
        _placeholder: (),
    },

    // ── Structure terrain ───────────────────────────────────────────────
    EndIslands(Box<AstNode>),
    Beardifier {
        // TODO: full struct
        _placeholder: (),
    },

    // ── Shape / Selection ───────────────────────────────────────────────
    Spline {
        // TODO: spline control points + min/max
        _placeholder: (),
    },
    RangeChoice {
        input: Box<AstNode>,
        min: Box<AstNode>,
        max: Box<AstNode>,
        in_bounds: Box<AstNode>,
        out_of_bounds_lo: Box<AstNode>,
        out_of_bounds_hi: Box<AstNode>,
    },
    IntervalSelect {
        // 6 child nodes per protocol
        _placeholder: (),
    },
    YClampedGradient {
        // TODO: y_min, y_max offsets + slope
        _placeholder: (),
    },

    // ── Caches ──────────────────────────────────────────────────────────
    CacheFlat(Box<AstNode>),
    Cache2D(Box<AstNode>),
    CacheInterpolated(Box<AstNode>),

    // ── Surface / Aquifer ───────────────────────────────────────────────
    FindTopSurface {
        density: Box<AstNode>,
        initial: Box<AstNode>,
        min_y: i32,
        max_y: i32,
    },
    Aquifer {
        // TODO: full struct (noise_level + params)
        _placeholder: (),
    },

    // ── Biome ───────────────────────────────────────────────────────────
    BiomeTree {
        // MultiNoiseBiomeSource flattened decision tree
        _placeholder: (),
    },

    // ── Root ────────────────────────────────────────────────────────────
    /// Binds the 16 NoiseRouter density channels in the exact MC order:
    ///   0=barrier, 1=fluid_level_floodedness, ... 15=final_density
    /// (Exact list from NoiseRouter fields, double-checked against MC source.)
    Root(Vec<AstNode>),
}
