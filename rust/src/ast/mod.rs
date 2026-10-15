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

pub mod split;
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
pub const TAG_SQUEEZE: u8 = 0x25;
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

/// One MC `OctavePerlinNoiseSampler`: a stack of `octaves` Perlin samplers.
///
/// MC's sample loop (confirmed from bytecode):
/// ```text
/// freq = initial_freq        // = 2^firstOctave
/// amp  = persistence         // = 2^(n-1) / (2^n - 1)
/// acc  = 0
/// for i in 0..n:
///     if octaveSamplers[i] != null:          // <=> amplitudes[i] != 0.0
///         acc += amplitudes[i] * perlin(coord * freq) * amp
///     freq *= 2.0
///     amp  /= 2.0
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct OctaveSampler {
    /// Base i32 index into the flat perm buffer; octave `i` reads the
    /// 256-entry table starting at `perm_offset + i * 256`.
    pub perm_offset: i32,
    /// First octave's frequency multiplier (`2^firstOctave`).
    pub initial_freq: f32,
    /// First octave's amplitude multiplier (`2^(n-1)/(2^n-1)`).
    pub persistence: f32,
    /// Per-octave amplitude; a 0.0 entry means "no sampler at this octave".
    pub amplitudes: Vec<f32>,
}

impl Hash for OctaveSampler {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.perm_offset.hash(state);
        f32_to_hash_bits(self.initial_freq).hash(state);
        f32_to_hash_bits(self.persistence).hash(state);
        self.amplitudes.len().hash(state);
        for &v in &self.amplitudes {
            f32_to_hash_bits(v).hash(state);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AstNode {
    // ── Scalars ─────────────────────────────────────────────────────────
    Constant(f32),

    /// A density subtree the serializer could not translate (e.g. a
    /// modded node like Tectonic's `Invert`). Its values are evaluated on
    /// the CPU at the same corner grid and uploaded via the external-input
    /// buffer; the WGSL side just reads them. `index` selects the slot.
    ExternalInput {
        index: i32,
    },

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
    Squeeze(Box<AstNode>),

    // ── Noise samplers ──────────────────────────────────────────────────
    NoiseSampler {
        noise_type: NoiseType,
        /// First `OctavePerlinNoiseSampler` (always present).
        first: OctaveSampler,
        /// Second octave sampler — `Some` for MC's `DoublePerlinNoiseSampler`
        /// (the two are blended with a hard-coded 0.5 ratio); `None` for a
        /// plain single `OctavePerlinNoiseSampler`.
        second: Option<OctaveSampler>,
        /// Outer blend factor applied to `(first + second)`; MC uses 0.5 for
        /// `DoublePerlinNoiseSampler` and 1.0 for a plain octave stack.
        amplitude: f32,
        xz_scale: f32,
        y_scale: f32,
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

    // ── Coordinate shifts / shifted noise ───────────────────────────────
    /// MC `Shift`/`ShiftA`/`ShiftB` (Offset interface). `kind`:
    /// 0 = Shift (X,Y,Z), 1 = ShiftA (X,0,Z), 2 = ShiftB (Z,X,0).
    /// `offset` is a `NoiseSampler` child; sample = noise(coord*0.25) * 4.
    Shift {
        kind: u8,
        offset: Box<AstNode>,
    },
    /// MC `WeirdScaledSampler`: scale = mapper(input); result = scale *
    /// abs(noise.sample(x/scale, y/scale, z/scale)). `mapper`: 1 = TYPE1
    /// (scaleTunnels), 2 = TYPE2 (scaleCaves).
    WeirdScaledSampler {
        mapper: u8,
        input: Box<AstNode>,
        noise: Box<AstNode>,
    },
    /// MC standalone `InterpolatedNoiseSampler` (class_5822): 3 octave
    /// stacks (lower/upper/interp) blended by a clamped lerp.
    InterpNoiseSampler {
        xz_scale: f32, y_scale: f32, xz_factor: f32, y_factor: f32, smear: f32,
        lower_base: i32, lower_count: i32,
        upper_base: i32, upper_count: i32,
        interp_base: i32, interp_count: i32,
    },
    /// MC `ShiftedNoise`: shifts the sample coords by three 2D/3D noise
    /// fields, then samples `noise` at the displaced coordinate.
    ShiftedNoise {
        xz_scale: f32,
        y_scale: f32,
        shift_x: Box<AstNode>,
        shift_y: Box<AstNode>,
        shift_z: Box<AstNode>,
        noise: Box<AstNode>,
    },

    // ── Structure terrain ───────────────────────────────────────────────
    EndIslands {
        perm_index: i32,
    },
    Beardifier {
        struct_index: i32,
    },

    // ── Shape / Selection ───────────────────────────────────────────────
    Spline {
        /// Spline node locations (sorted ascending).
        locations: Vec<f32>,
        /// Per-node Hermite derivative.
        derivatives: Vec<f32>,
        /// The `ToFloatFunction` input (usually Y or a noise field).
        location: Box<AstNode>,
        /// Per-node values; each is a nested `Spline` or a `Constant` leaf.
        values: Vec<AstNode>,
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
        from_y: f32,
        to_y: f32,
        from_value: f32,
        to_value: f32,
    },

    // ── Caches ──────────────────────────────────────────────────────────
    CacheFlat(Box<AstNode>),
    Cache2D(Box<AstNode>),
    CacheInterpolated(Box<AstNode>),

    /// §14 fix: placeholder for the i-th extracted interpolated-boundary
    /// subtree of finalDensity. The corner pass computes the 8 extracted
    /// subtrees into out_buf slots 1..=8; the BlockOut pass trilinearly
    /// interpolates them to block resolution and substitutes the result
    /// here. Only appears inside the split outer tree; never evaluated by
    /// the corner pass.
    InnerRef(usize),

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

            AstNode::InnerRef(i) => { 0xEDu8.hash(state); i.hash(state); }

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
            AstNode::Squeeze(a) => {
                TAG_SQUEEZE.hash(state);
                a.hash(state);
            }

            // ── Noise samplers ──────────────────────────────────────────────
            AstNode::NoiseSampler {
                noise_type,
                first,
                second,
                amplitude,
                xz_scale,
                y_scale,
            } => {
                TAG_NOISE_SAMPLER.hash(state);
                0x84u8.hash(state);
                noise_type.hash(state);
                first.hash(state);
                second.hash(state);
                f32_to_hash_bits(*amplitude).hash(state);
                xz_scale.to_bits().hash(state);
                y_scale.to_bits().hash(state);
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
            AstNode::Shift { kind, offset } => {
                0xB4u8.hash(state);
                kind.hash(state);
                offset.hash(state);
            }
            AstNode::WeirdScaledSampler { mapper, input, noise } => {
                0x86u8.hash(state);
                mapper.hash(state);
                input.hash(state);
                noise.hash(state);
            }
            AstNode::InterpNoiseSampler {
                xz_scale, y_scale, xz_factor, y_factor, smear,
                lower_base, lower_count, upper_base, upper_count,
                interp_base, interp_count,
            } => {
                0x87u8.hash(state);
                for v in [xz_scale, y_scale, xz_factor, y_factor, smear] {
                    f32_to_hash_bits(*v).hash(state);
                }
                lower_base.hash(state); lower_count.hash(state);
                upper_base.hash(state); upper_count.hash(state);
                interp_base.hash(state); interp_count.hash(state);
            }
            AstNode::ShiftedNoise {
                xz_scale, y_scale, shift_x, shift_y, shift_z, noise,
            } => {
                0xB5u8.hash(state);
                f32_to_hash_bits(*xz_scale).hash(state);
                f32_to_hash_bits(*y_scale).hash(state);
                shift_x.hash(state);
                shift_y.hash(state);
                shift_z.hash(state);
                noise.hash(state);
            }

            // ── Structure terrain ───────────────────────────────────────────
            AstNode::EndIslands { perm_index } => {
                TAG_END_ISLANDS.hash(state);
                perm_index.hash(state);
            }
            AstNode::Beardifier { struct_index } => {
                0xA1u8.hash(state);
                struct_index.hash(state);
            }

            // ── Shape / Selection ───────────────────────────────────────────
            AstNode::Spline { locations, derivatives, location, values } => {
                0xB0u8.hash(state);
                locations.len().hash(state);
                for v in locations { f32_to_hash_bits(*v).hash(state); }
                for v in derivatives { f32_to_hash_bits(*v).hash(state); }
                location.hash(state);
                values.len().hash(state);
                for v in values { v.hash(state); }
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
            AstNode::YClampedGradient { from_y, to_y, from_value, to_value } => {
                0xB3u8.hash(state);
                f32_to_hash_bits(*from_y).hash(state);
                f32_to_hash_bits(*to_y).hash(state);
                f32_to_hash_bits(*from_value).hash(state);
                f32_to_hash_bits(*to_value).hash(state);
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

            AstNode::ExternalInput { index } => {
                0xFDu8.hash(state);
                index.hash(state);
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

}
