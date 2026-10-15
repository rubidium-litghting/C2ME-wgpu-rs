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

//! Pass 4: Normalize AST to canonical forms so semantically equivalent
//! expressions hit the same CSE entry and produce identical WGSL.
//!
//! Current rule set:
//!   1. Commutative binary ops (Add / Mul / Min / Max) reorder their two
//!      children by stable fingerprint. After this,
//!          Add(InputX, Constant(2)) == Add(Constant(2), InputX)
//!      structurally — a prerequisite for CSE to actually share them.
//!
//! n-ary associative flattening (e.g. Add(Add(a,b), c) → length-3 list) is
//! intentionally NOT implemented yet; the current AstNode enum only has
//! binary variants. We will revisit this when the WGSL emitter needs
//! KahanSum for long chains (see issue tracker).

use crate::ast::AstNode;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};

pub fn canonicalize(root: AstNode) -> AstNode {
    rec(root)
}

fn rec(node: AstNode) -> AstNode {
    use AstNode::*;
    match node {
        AstNode::NoiseSampler { .. } => node, // 或使用通配符
        // Leaves ----------------------------------------------------------
        Constant(_) | InputX | InputY | InputZ | ExternalInput { .. } | InnerRef(_) => node,

        // Commutative binaries: post-order, then hash-swap children -------
        Add(a, b) => order_commutative(*a, *b, Add, rec),
        Mul(a, b) => order_commutative(*a, *b, Mul, rec),
        Min(a, b) => order_commutative(*a, *b, Min, rec),
        Max(a, b) => order_commutative(*a, *b, Max, rec),

        // Non-commutative binaries: post-order only ----------------------
        Sub(a, b) => Sub(Box::new(rec(*a)), Box::new(rec(*b))),
        Div(a, b) => Div(Box::new(rec(*a)), Box::new(rec(*b))),
        FloorDiv(a, b) => FloorDiv(Box::new(rec(*a)), Box::new(rec(*b))),

        // Unaries: post-order only ---------------------------------------
        Abs(a) => Abs(Box::new(rec(*a))),
        Square(a) => Square(Box::new(rec(*a))),
        Cube(a) => Cube(Box::new(rec(*a))),
        Sqrt(a) => Sqrt(Box::new(rec(*a))),
        Squeeze(a) => Squeeze(Box::new(rec(*a))),

        // Noise / placeholders (no variable AstNode children) ------------
        | n @ SimplexNoise2D { .. }
        | n @ SimplexNoise3D { .. }
        | n @ InterpolatedNoiseSampler { .. }
        | n @ Beardifier { .. }
        | n @ IntervalSelect { .. }
                | n @ Aquifer { .. }
        | n @ BiomeTree { .. }
        | n @ InterpNoiseSampler { .. }
        | n @ YClampedGradient { .. } => n,

        Shift { kind, offset } => Shift { kind, offset: Box::new(rec(*offset)) },
        WeirdScaledSampler { mapper, input, noise } => WeirdScaledSampler {
            mapper,
            input: Box::new(rec(*input)),
            noise: Box::new(rec(*noise)),
        },
        Spline { locations, derivatives, location, values } => Spline {
            locations: locations.clone(), derivatives: derivatives.clone(),
            location: Box::new(rec(*location)),
            values: values.iter().map(|v| rec(v.clone())).collect(),
        },
        ShiftedNoise { xz_scale, y_scale, shift_x, shift_y, shift_z, noise } => ShiftedNoise {
            xz_scale, y_scale,
            shift_x: Box::new(rec(*shift_x)), shift_y: Box::new(rec(*shift_y)), shift_z: Box::new(rec(*shift_z)),
            noise: Box::new(rec(*noise)),
        },

        // Structured nodes with children: post-order only ----------------
        n @ EndIslands { .. } => n.clone(),
        RangeChoice {
            input,
            min,
            max,
            in_bounds,
            out_of_bounds_lo,
            out_of_bounds_hi,
        } => RangeChoice {
            input: Box::new(rec(*input)),
            min: Box::new(rec(*min)),
            max: Box::new(rec(*max)),
            in_bounds: Box::new(rec(*in_bounds)),
            out_of_bounds_lo: Box::new(rec(*out_of_bounds_lo)),
            out_of_bounds_hi: Box::new(rec(*out_of_bounds_hi)),
        },
        CacheFlat(a) => CacheFlat(Box::new(rec(*a))),
        Cache2D(a) => Cache2D(Box::new(rec(*a))),
        CacheInterpolated(a) => CacheInterpolated(Box::new(rec(*a))),
        FindTopSurface {
            density,
            initial,
            min_y,
            max_y,
        } => FindTopSurface {
            density: Box::new(rec(*density)),
            initial: Box::new(rec(*initial)),
            min_y,
            max_y,
        },

        Root(channels) => Root(channels.into_iter().map(rec).collect()),
    }
}

fn order_commutative<Ctor, RecFn>(a: AstNode, b: AstNode, ctor: Ctor, rec: RecFn) -> AstNode
where
    Ctor: FnOnce(Box<AstNode>, Box<AstNode>) -> AstNode,
    RecFn: Fn(AstNode) -> AstNode,
{
    let fa = rec(a);
    let fb = rec(b);
    if fingerprint(&fa) <= fingerprint(&fb) {
        ctor(Box::new(fa), Box::new(fb))
    } else {
        ctor(Box::new(fb), Box::new(fa))
    }
}

fn fingerprint(n: &AstNode) -> u64 {
    let mut h = FxHasher::default();
    n.hash(&mut h);
    h.finish()
}

// ======================================================================
// Tests
// ======================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use AstNode::*;

    fn boxed(n: AstNode) -> Box<AstNode> {
        Box::new(n)
    }

    /// Basic canonicalization: the two operand orders collapse into one.
    #[test]
    fn commutative_orders_collapse() {
        let a = Add(boxed(InputX), boxed(Constant(2.0)));
        let b = Add(boxed(Constant(2.0)), boxed(InputX));
        assert_ne!(a, b, "before canonicalize the two trees must differ");
        let ca = canonicalize(a);
        let cb = canonicalize(b);
        assert_eq!(
            ca, cb,
            "after canonicalize they must be structurally identical"
        );
    }

    /// Normalization is stable: applying canonicalize twice must give the
    /// same result as applying it once (idempotent).
    #[test]
    fn canonicalize_is_idempotent() {
        let orig = Mul(boxed(Constant(3.0)), boxed(InputZ));
        let once = canonicalize(orig.clone());
        let twice = canonicalize(once.clone());
        assert_eq!(once, twice, "canonicalize must be idempotent");
    }

    /// Nested commutativity: inner Add(Min(X,Y), Const) and Add(Const, Min(Y,X))
    /// must normalize to structurally equal trees after whole-tree post-order
    /// walk (inner Min first gets ordered, then outer Add sees same children).
    #[test]
    fn nested_commutative_orders_equalize() {
        // tree A: X+2 wrapped with Min then Add(Min, 1)
        let a = Add(
            boxed(Min(boxed(InputX), boxed(InputY))),
            boxed(Constant(1.0)),
        );
        // tree B: Y+X swapped Min then Add(1, Min) — structurally different
        let b = Add(
            boxed(Constant(1.0)),
            boxed(Min(boxed(InputY), boxed(InputX))),
        );
        assert_ne!(a, b);
        let ca = canonicalize(a);
        let cb = canonicalize(b);
        assert_eq!(ca, cb);
    }

    /// Non-commutative op Sub(a,b) and Sub(b,a) must NOT be equalized.
    #[test]
    fn non_commutative_preserved() {
        let a = Sub(boxed(InputX), boxed(Constant(2.0)));
        let b = Sub(boxed(Constant(2.0)), boxed(InputX));
        let ca = canonicalize(a);
        let cb = canonicalize(b);
        assert_ne!(ca, cb, "Sub must NOT be commutative under canonicalization");
    }
}
