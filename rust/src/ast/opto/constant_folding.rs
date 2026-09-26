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

//! Pass 1: constant folding. Any subtree whose **immediate children** are all
//! `Constant(f32)` is computed at optimization time and replaced with a single
//! `Constant`. Noise / coordinate / wrapper nodes are never folded.
//!
//! Edge case rules (must match MC runtime semantics):
//!   - Div / FloorDiv by zero → keep the node as-is (do NOT produce NaN/Inf).
//!     Vanilla DensityFunction.Div at runtime would trigger the same UB path
//!     when divisor == 0, so we preserve that behaviour.
//!   - Sqrt of negative → keep the node as-is (do NOT produce NaN silently).
//!   - `+0.0` vs `-0.0` are treated as numerically equal by f32 PartialEq,
//!     so `Constant(-0.0) + Constant(1.0)` correctly folds to `Constant(1.0)`.

use crate::Result;
use crate::ast::AstNode;

pub fn fold(root: AstNode) -> Result<AstNode> {
    Ok(rec(root))
}

fn rec(node: AstNode) -> AstNode {
    use AstNode::*;
    match node {
        // Levers / leaves that can't possibly fold ------------------------
        Constant(_) | InputX | InputY | InputZ => node,

        // Binary arithmetic ----------------------------------------------
        Add(a, b) => {
            let fa = rec(*a);
            let fb = rec(*b);
            if let (Constant(x), Constant(y)) = (&fa, &fb) {
                Constant(x + y)
            } else {
                Add(Box::new(fa), Box::new(fb))
            }
        }
        Sub(a, b) => {
            let fa = rec(*a);
            let fb = rec(*b);
            if let (Constant(x), Constant(y)) = (&fa, &fb) {
                Constant(x - y)
            } else {
                Sub(Box::new(fa), Box::new(fb))
            }
        }
        Mul(a, b) => {
            let fa = rec(*a);
            let fb = rec(*b);
            if let (Constant(x), Constant(y)) = (&fa, &fb) {
                Constant(x * y)
            } else {
                Mul(Box::new(fa), Box::new(fb))
            }
        }
        Div(a, b) => {
            let fa = rec(*a);
            let fb = rec(*b);
            if let (Constant(x), Constant(y)) = (&fa, &fb) {
                if *y == 0.0 {
                    return Div(Box::new(fa), Box::new(fb));
                }
                Constant(x / y)
            } else {
                Div(Box::new(fa), Box::new(fb))
            }
        }
        Min(a, b) => {
            let fa = rec(*a);
            let fb = rec(*b);
            if let (Constant(x), Constant(y)) = (&fa, &fb) {
                Constant(f32::min(*x, *y))
            } else {
                Min(Box::new(fa), Box::new(fb))
            }
        }
        Max(a, b) => {
            let fa = rec(*a);
            let fb = rec(*b);
            if let (Constant(x), Constant(y)) = (&fa, &fb) {
                Constant(f32::max(*x, *y))
            } else {
                Max(Box::new(fa), Box::new(fb))
            }
        }
        FloorDiv(a, b) => {
            let fa = rec(*a);
            let fb = rec(*b);
            if let (Constant(x), Constant(y)) = (&fa, &fb) {
                if *y == 0.0 {
                    return FloorDiv(Box::new(fa), Box::new(fb));
                }
                Constant(f32::floor(x / y))
            } else {
                FloorDiv(Box::new(fa), Box::new(fb))
            }
        }

        // Unary arithmetic -----------------------------------------------
        Abs(a) => {
            let fa = rec(*a);
            if let Constant(x) = fa {
                Constant(f32::abs(x))
            } else {
                Abs(Box::new(fa))
            }
        }
        Square(a) => {
            let fa = rec(*a);
            if let Constant(x) = fa {
                Constant(x * x)
            } else {
                Square(Box::new(fa))
            }
        }
        Cube(a) => {
            let fa = rec(*a);
            if let Constant(x) = fa {
                Constant(x * x * x)
            } else {
                Cube(Box::new(fa))
            }
        }
        Squeeze(a) => {
            let fa = rec(*a);
            Squeeze(Box::new(fa))
        }
        Sqrt(a) => {
            let fa = rec(*a);
            if let Constant(x) = fa {
                if x < 0.0 {
                    Sqrt(Box::new(Constant(x)))
                } else {
                    Constant(f32::sqrt(x))
                }
            } else {
                Sqrt(Box::new(fa))
            }
        }

        // Noise / Structure leaves ---------------------------------------
        // No variable AstNode children — return as-is.
        n @ SimplexNoise2D { .. }
        | n @ SimplexNoise3D { .. }
        | n @ InterpolatedNoiseSampler { .. }
        | n @ Beardifier { .. }
        | n @ IntervalSelect { .. }
                | n @ Aquifer { .. }
        | n @ BiomeTree { .. }
        | n @ YClampedGradient { .. } => n,

        // Nodes with variable AstNode children — recurse but never fold
        // the node itself (semantic meaning depends on coordinate input
        // or wrapper identity).
        AstNode::NoiseSampler { .. } => node, // 或使用通配符
        Shift { kind, offset } => Shift { kind, offset: Box::new(rec(*offset)) },
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
        EndIslands(a) => EndIslands(Box::new(rec(*a))),
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

    #[test]
    fn fold_scalar_constants() {
        // 1 + 2 = 3
        let n = Add(boxed(Constant(1.0)), boxed(Constant(2.0)));
        assert_eq!(fold(n).unwrap(), Constant(3.0));

        // 5 - 1 = 4
        let n = Sub(boxed(Constant(5.0)), boxed(Constant(1.0)));
        assert_eq!(fold(n).unwrap(), Constant(4.0));

        // 6 * 7 = 42
        let n = Mul(boxed(Constant(6.0)), boxed(Constant(7.0)));
        assert_eq!(fold(n).unwrap(), Constant(42.0));

        // 7 / 2 = 3.5
        let n = Div(boxed(Constant(7.0)), boxed(Constant(2.0)));
        assert_eq!(fold(n).unwrap(), Constant(3.5));

        // Min/Max
        let n = Min(boxed(Constant(3.0)), boxed(Constant(1.0)));
        assert_eq!(fold(n).unwrap(), Constant(1.0));
        let n = Max(boxed(Constant(3.0)), boxed(Constant(1.0)));
        assert_eq!(fold(n).unwrap(), Constant(3.0));

        // FloorDiv: 7 / 2 = 3.0 (floor(3.5)), -7 / 2 = -4.0 (floor(-3.5))
        let n = FloorDiv(boxed(Constant(7.0)), boxed(Constant(2.0)));
        assert_eq!(fold(n).unwrap(), Constant(3.0));
        let n = FloorDiv(boxed(Constant(-7.0)), boxed(Constant(2.0)));
        assert_eq!(fold(n).unwrap(), Constant(-4.0));
    }

    #[test]
    fn fold_unary_constants() {
        // Abs(-4.5) = 4.5
        let n = Abs(boxed(Constant(-4.5)));
        assert_eq!(fold(n).unwrap(), Constant(4.5));
        // Square(4) = 16
        let n = Square(boxed(Constant(4.0)));
        assert_eq!(fold(n).unwrap(), Constant(16.0));
        // Cube(3) = 27
        let n = Cube(boxed(Constant(3.0)));
        assert_eq!(fold(n).unwrap(), Constant(27.0));
        // Sqrt(4) = 2
        let n = Sqrt(boxed(Constant(4.0)));
        assert_eq!(fold(n).unwrap(), Constant(2.0));
    }

    #[test]
    fn guard_div_by_zero_and_negative_sqrt() {
        // Div / 0 → keep original node, don't produce Inf
        let n = Div(boxed(Constant(5.0)), boxed(Constant(0.0)));
        let out = fold(n.clone()).unwrap();
        // Must remain a Div node (not Constant), check variant discriminant
        assert!(
            matches!(out, Div(_, _)),
            "expected Div preserved, got {out:?}"
        );

        // FloorDiv / 0 → keep
        let n = FloorDiv(boxed(Constant(5.0)), boxed(Constant(0.0)));
        assert!(matches!(fold(n).unwrap(), FloorDiv(_, _)));

        // Sqrt(-1) → keep Sqrt node (don't silent-NaN fold)
        let n = Sqrt(boxed(Constant(-1.0)));
        assert!(matches!(fold(n).unwrap(), Sqrt(_)));
    }

    #[test]
    fn fold_nested_expressions() {
        // (1 + 2) * (3 - 1)  =  3 * 2  =  6
        let n = Mul(
            boxed(Add(boxed(Constant(1.0)), boxed(Constant(2.0)))),
            boxed(Sub(boxed(Constant(3.0)), boxed(Constant(1.0)))),
        );
        assert_eq!(fold(n).unwrap(), Constant(6.0));
    }

    #[test]
    fn fold_stops_at_non_constant() {
        // Add(Constant(3), InputX) → NOT foldable, result must stay Add
        let n = Add(boxed(Constant(3.0)), boxed(InputX));
        let out = fold(n).unwrap();
        // But the Constant(3) was already a leaf, so just assert structure
        assert!(matches!(out, Add(_, _)));

        // Square(Add(Const(2), InputY)) → outer Square can't fold (not Const child)
        // but inner Add also can't fold → structure preserved
        let n = Square(boxed(Add(boxed(Constant(2.0)), boxed(InputY))));
        let out = fold(n).unwrap();
        assert!(matches!(out, Square(_)));
    }

    #[test]
    fn fold_preserves_children_of_wrapper_nodes() {
        // CacheFlat(Add(Constant(1), Constant(2))) → inner folds to Constant(3)
        // but outer CacheFlat identity preserved.
        let n = CacheFlat(boxed(Add(boxed(Constant(1.0)), boxed(Constant(2.0)))));
        let out = fold(n).unwrap();
        match out {
            CacheFlat(inner) => assert_eq!(*inner, Constant(3.0)),
            other => panic!("expected CacheFlat, got {other:?}"),
        }
    }
}
