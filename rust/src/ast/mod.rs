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

use std::hash::{Hash, Hasher};

pub mod deserialize;
pub mod equality;
pub mod opto;

pub const TAG_CONSTANT: u8 = 0x01;
pub const TAG_INPUT_X: u8 = 0x02;
pub const TAG_INPUT_Y: u8 = 0x03;
pub const TAG_INPUT_Z: u8 = 0x04;
pub const TAG_ADD: u8 = 0x10;
pub const TAG_SUB: u8 = 0x11;
pub const TAG_MUL: u8 = 0x12;
pub const TAG_DIV: u8 = 0x13;
pub const TAG_MIN: u8 = 0x14;
pub const TAG_MAX: u8 = 0x15;
pub const TAG_ABS: u8 = 0x20;
pub const TAG_SQUARE: u8 = 0x21;
pub const TAG_CUBE: u8 = 0x22;
pub const TAG_SQRT: u8 = 0x23;
pub const TAG_FLOOR_DIV: u8 = 0x24;
pub const TAG_PERLIN: u8 = 0x80;
pub const TAG_DOUBLE_PERLIN: u8 = 0x81;
pub const TAG_SIMPLEX_2D: u8 = 0x82;
pub const TAG_SIMPLEX_3D: u8 = 0x83;
pub const TAG_NOISE_SAMPLER: u8 = 0x84;
pub const TAG_INTERP_NOISE: u8 = 0x90;
pub const TAG_END_ISLANDS: u8 = 0xA0;
pub const TAG_BEARDIFIER: u8 = 0xA1;
pub const TAG_SPLINE: u8 = 0xB0;
pub const TAG_RANGE_CHOICE: u8 = 0xB1;
pub const TAG_INTERVAL_SEL: u8 = 0xB2;
pub const TAG_Y_CLAMPED_GRAD: u8 = 0xB3;
pub const TAG_CACHE_FLAT: u8 = 0xC0;
pub const TAG_CACHE_2D: u8 = 0xC1;
pub const TAG_CACHE_INTERP: u8 = 0xC2;
pub const TAG_FIND_TOP_SURF: u8 = 0xD0;
pub const TAG_AQUIFER: u8 = 0xE0;
pub const TAG_BIOME_TREE: u8 = 0xF0;
pub const TAG_ROOT: u8 = 0xFE;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum NoiseType {
    Perlin = 0,
    Simplex = 1,
}

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
    NoiseSampler {
        noise_type: NoiseType,
        /// Index into the flat permutation buffer (binding 1); each entry is
        /// a 256-u32 table. Replaces `seed` because MC 1.21.1 discards the
        /// seed after table construction.
        perm_offset: i32,
        xz_scale: f32,
        y_scale: f32,
    },
    DoublePerlin {
        a: Box<AstNode>, // PerlinSampler A
        b: Box<AstNode>, // PerlinSampler B
    },
    SimplexNoise2D {
        seed: i64,
    },
    SimplexNoise3D {
        seed: i64,
    },
    InterpolatedNoiseSampler {
        child: Box<AstNode>,
    },

    // ── Structure terrain ───────────────────────────────────────────────
    EndIslands(Box<AstNode>),
    Beardifier {
        struct_index: i32,
    },

    // ── Shape / Selection ───────────────────────────────────────────────
    Spline {
        points: Vec<(f32, f32)>,
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
        input: Box<AstNode>,
        min: Box<AstNode>,
        max: Box<AstNode>,
        in_bounds: Box<AstNode>,
        out_of_bounds_lo: Box<AstNode>,
        out_of_bounds_hi: Box<AstNode>,
    },
    YClampedGradient {
        y_min: i32,
        y_max: i32,
        input: Box<AstNode>,
        gradient: Box<AstNode>,
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
        density: Box<AstNode>,
        fluid_level: Box<AstNode>,
    },

    // ── Biome ───────────────────────────────────────────────────────────
    BiomeTree {
        // MultiNoiseBiomeSource flattened decision tree
        seed: i64,
    },

    // ── Root ────────────────────────────────────────────────────────────
    /// Binds the 16 NoiseRouter density channels in the exact MC order:
    ///   0=barrier, 1=fluid_level_floodedness, ... 15=final_density
    /// (Exact list from NoiseRouter fields, double-checked against MC source.)
    Root(Vec<AstNode>),
}

/// 归一化 f32 为可哈希的 u32 bit pattern，保证 Hash 与 PartialEq 一致。
///
/// 关键：+0.0 和 -0.0 的 PartialEq 为 true，但 to_bits() 不同。必须统一。
#[inline]
fn f32_to_hash_bits(v: f32) -> u32 {
    if v == 0.0 {
        // +0.0 的 to_bits() 固定为 0x0000_0000
        0.0f32.to_bits()
    } else {
        v.to_bits()
    }
}

impl Hash for AstNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            // ── Scalars ─────────────────────────────────────────────────────
            AstNode::Constant(v) => {
                TAG_CONSTANT.hash(state);
                f32_to_hash_bits(*v).hash(state);
            }

            // ── Inputs ──────────────────────────────────────────────────────
            AstNode::InputX => TAG_INPUT_X.hash(state),
            AstNode::InputY => TAG_INPUT_Y.hash(state),
            AstNode::InputZ => TAG_INPUT_Z.hash(state),

            // ── Binary ops ──────────────────────────────────────────────────
            AstNode::Add(a, b) => {
                TAG_ADD.hash(state);
                a.hash(state);
                b.hash(state);
            }
            AstNode::Sub(a, b) => {
                TAG_SUB.hash(state);
                a.hash(state);
                b.hash(state);
            }
            AstNode::Mul(a, b) => {
                TAG_MUL.hash(state);
                a.hash(state);
                b.hash(state);
            }
            AstNode::Div(a, b) => {
                TAG_DIV.hash(state);
                a.hash(state);
                b.hash(state);
            }
            AstNode::Min(a, b) => {
                TAG_MIN.hash(state);
                a.hash(state);
                b.hash(state);
            }
            AstNode::Max(a, b) => {
                TAG_MAX.hash(state);
                a.hash(state);
                b.hash(state);
            }

            // ── Unary ops ───────────────────────────────────────────────────
            AstNode::Abs(a) => {
                TAG_ABS.hash(state);
                a.hash(state);
            }
            AstNode::Square(a) => {
                TAG_SQUARE.hash(state);
                a.hash(state);
            }
            AstNode::Cube(a) => {
                TAG_CUBE.hash(state);
                a.hash(state);
            }
            AstNode::Sqrt(a) => {
                TAG_SQRT.hash(state);
                a.hash(state);
            }
            AstNode::FloorDiv(a, b) => {
                TAG_FLOOR_DIV.hash(state);
                a.hash(state);
                b.hash(state);
            }

            // ── Noise samplers ──────────────────────────────────────────────
            AstNode::PerlinSampler {
                seed,
                octaves,
                amplitudes,
                lacunaritys,
            } => {
                TAG_PERLIN.hash(state);
                seed.hash(state);
                octaves.hash(state);
                amplitudes.len().hash(state);
                for &v in amplitudes {
                    f32_to_hash_bits(v).hash(state);
                }
                lacunaritys.len().hash(state);
                for &v in lacunaritys {
                    f32_to_hash_bits(v).hash(state);
                }
            }
            AstNode::NoiseSampler {
                noise_type,
                perm_offset,
                xz_scale,
                y_scale,
            } => {
                TAG_NOISE_SAMPLER.hash(state);
                0x84u8.hash(state);
                noise_type.hash(state);
                perm_offset.hash(state);
                xz_scale.to_bits().hash(state);
                y_scale.to_bits().hash(state);
            }
            AstNode::DoublePerlin { a, b } => {
                TAG_DOUBLE_PERLIN.hash(state);
                a.hash(state);
                b.hash(state);
            }
            AstNode::SimplexNoise2D { seed } => {
                TAG_SIMPLEX_2D.hash(state);
                seed.hash(state);
            }
            AstNode::SimplexNoise3D { seed } => {
                TAG_SIMPLEX_3D.hash(state);
                seed.hash(state);
            }
            AstNode::InterpolatedNoiseSampler { child } => {
                0x90u8.hash(state);
                child.hash(state);
            }

            // ── Structure terrain ───────────────────────────────────────────
            AstNode::EndIslands(a) => {
                TAG_END_ISLANDS.hash(state);
                a.hash(state);
            }
            AstNode::Beardifier { struct_index } => {
                0xA1u8.hash(state);
                struct_index.hash(state);
            }

            // ── Shape / Selection ───────────────────────────────────────────
            AstNode::Spline { points } => {
                0xB0u8.hash(state);
                points.len().hash(state);
                for (x, y) in points {
                    f32_to_hash_bits(*x).hash(state);
                    f32_to_hash_bits(*y).hash(state);
                }
            }
            AstNode::RangeChoice {
                input,
                min,
                max,
                in_bounds,
                out_of_bounds_lo,
                out_of_bounds_hi,
            } => {
                TAG_RANGE_CHOICE.hash(state);
                input.hash(state);
                min.hash(state);
                max.hash(state);
                in_bounds.hash(state);
                out_of_bounds_lo.hash(state);
                out_of_bounds_hi.hash(state);
            }
            AstNode::IntervalSelect {
                input,
                min,
                max,
                in_bounds,
                out_of_bounds_lo,
                out_of_bounds_hi,
            } => {
                0xB2u8.hash(state);
                input.hash(state);
                min.hash(state);
                max.hash(state);
                in_bounds.hash(state);
                out_of_bounds_lo.hash(state);
                out_of_bounds_hi.hash(state);
            }
            AstNode::YClampedGradient {
                y_min,
                y_max,
                input,
                gradient,
            } => {
                0xB3u8.hash(state);
                y_min.hash(state);
                y_max.hash(state);
                input.hash(state);
                gradient.hash(state);
            }

            // ── Caches ──────────────────────────────────────────────────────
            AstNode::CacheFlat(a) => {
                TAG_CACHE_FLAT.hash(state);
                a.hash(state);
            }
            AstNode::Cache2D(a) => {
                TAG_CACHE_2D.hash(state);
                a.hash(state);
            }
            AstNode::CacheInterpolated(a) => {
                TAG_CACHE_INTERP.hash(state);
                a.hash(state);
            }

            // ── Surface / Aquifer ───────────────────────────────────────────
            AstNode::FindTopSurface {
                density,
                initial,
                min_y,
                max_y,
            } => {
                TAG_FIND_TOP_SURF.hash(state);
                density.hash(state);
                initial.hash(state);
                min_y.hash(state);
                max_y.hash(state);
            }
            AstNode::Aquifer {
                density,
                fluid_level,
            } => {
                0xE0u8.hash(state);
                density.hash(state);
                fluid_level.hash(state);
            }

            // ── Biome ───────────────────────────────────────────────────────
            AstNode::BiomeTree { seed } => {
                0xF0u8.hash(state);
                seed.hash(state);
            }

            // ── Root ────────────────────────────────────────────────────────
            AstNode::Root(channels) => {
                TAG_ROOT.hash(state);
                channels.len().hash(state);
                for ch in channels {
                    ch.hash(state);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHasher;
    use std::hash::{Hash, Hasher};

    fn hash_one(n: &AstNode) -> u64 {
        let mut h = FxHasher::default();
        n.hash(&mut h);
        h.finish()
    }

    fn assert_hash_eq_if_partial_eq(a: &AstNode, b: &AstNode) {
        if a == b {
            assert_eq!(
                hash_one(a),
                hash_one(b),
                "Hash/PartialEq inconsistency: a==b but hash(a)!=hash(b)"
            );
        }
    }

    #[test]
    fn float_zero_plus_minus_hash_identical() {
        // +0.0 == -0.0（PartialEq 为 true），所以哈希必须相同
        let a = AstNode::Constant(0.0);
        let b = AstNode::Constant(-0.0);
        assert_eq!(a, b);
        assert_eq!(hash_one(&a), hash_one(&b));
    }

    #[test]
    fn structurally_different_have_different_hash() {
        let a = AstNode::Add(
            Box::new(AstNode::Constant(1.0)),
            Box::new(AstNode::Constant(2.0)),
        );
        let b = AstNode::Mul(
            Box::new(AstNode::Constant(1.0)),
            Box::new(AstNode::Constant(2.0)),
        );
        assert_ne!(a, b);
        // Add vs Mul：TAG 不同 → 哈希不同（不等价的情况我们不要求严格，但这里保证确实不同）
        assert_ne!(hash_one(&a), hash_one(&b));
    }

    #[test]
    fn perlin_vec_length_sensitive() {
        let p1 = AstNode::PerlinSampler {
            seed: 0,
            octaves: 1,
            amplitudes: vec![1.0],
            lacunaritys: vec![2.0],
        };
        let p2 = AstNode::PerlinSampler {
            seed: 0,
            octaves: 2,
            amplitudes: vec![1.0, 0.5],
            lacunaritys: vec![2.0, 2.0],
        };
        assert_ne!(hash_one(&p1), hash_one(&p2));
    }
}
