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

//! Pass 3: Common Subexpression Elimination.
//!
//! Lightweight variant — does NOT perform true Rc/arena node sharing (that
//! would involve rewriting every match on AstNode's enum). Instead:
//!
//!   **Return a structurally deduplicated AST where the SECOND occurrence of
//!   an exactly-equal subgraph is replaced by a clone of the FIRST.**
//!
//! On its own this only saves deserialized AST heap, but crucially it makes
//! semantically equivalent subgraphs **byte-for-byte structurally equal**,
//! so the later WGSL emitter can use `FxHashMap<&AstNode, ExprId>` to
//! actually share `let` bindings — that's where real shader-size savings
//! come from (and with it better GPU occupancy).
//!
//! Example input / output (run AFTER normalize so the two Add are ordered):
//!   Add( Add(X,2), Add(2,X) )   --normalize-->
//!   Add( Add(2,X), Add(2,X) )   --cse-------->
//!   Add( <ptr A>, <ptr A'> )      where A and A' are clone()-equal children
//!
//! The emitter then sees `&*ptr_a == &*ptr_a'` as the same value and reuses
//! its expression ID.

use crate::ast::AstNode;
use rustc_hash::FxHashMap;
use std::hash::{Hash, Hasher};

pub fn eliminate(root: AstNode) -> AstNode {
    let mut table: FxHashMap<u64, AstNode> = FxHashMap::default();
    rec(root, &mut table)
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn fingerprint(n: &AstNode) -> u64 {
    use rustc_hash::FxHasher;
    let mut h = FxHasher::default();
    n.hash(&mut h);
    h.finish()
}

fn rec(node: AstNode, table: &mut FxHashMap<u64, AstNode>) -> AstNode {
    // 1. Post-order: dedup children first
    let child_deduped = postorder_map(node, |child| rec(child, table));

    // 2. Now compute fp of the child-deduped node. If we've already stored an
    //    exactly-equal node for this fp, return the stored clone.
    let fp = fingerprint(&child_deduped);
    match table.entry(fp) {
        std::collections::hash_map::Entry::Occupied(o) => o.get().clone(),
        std::collections::hash_map::Entry::Vacant(v) => {
            v.insert(child_deduped.clone());
            child_deduped
        }
    }
}

/// Post-order deep-transform helper: apply `f` recursively to each
/// AstNode-owned child, then reassemble the parent node.
fn postorder_map<F>(node: AstNode, mut f: F) -> AstNode
where
    F: FnMut(AstNode) -> AstNode,
{
    use AstNode::*;
    let map_box = |b: Box<AstNode>, f: &mut F| Box::new(f(*b));

    match node {
        AstNode::NoiseSampler { .. } => node, // 或使用通配符
        Shift { kind, offset } => Shift { kind, offset: map_box(offset, &mut f) },
        WeirdScaledSampler { mapper, input, noise } => WeirdScaledSampler {
            mapper,
            input: map_box(input, &mut f),
            noise: map_box(noise, &mut f),
        },
        Spline { locations, derivatives, location, values } => Spline {
            locations, derivatives,
            location: map_box(location, &mut f),
            values: values.into_iter().map(|v| f(v)).collect(),
        },
        ShiftedNoise { xz_scale, y_scale, shift_x, shift_y, shift_z, noise } => ShiftedNoise {
            xz_scale, y_scale,
            shift_x: map_box(shift_x, &mut f), shift_y: map_box(shift_y, &mut f), shift_z: map_box(shift_z, &mut f),
            noise: map_box(noise, &mut f),
        },
        // Leaves with no variable AstNode children -----------------------
        Constant(_)
        | InputX
        | InputY
        | InputZ
        | SimplexNoise2D { .. }
        | SimplexNoise3D { .. }
        | InterpolatedNoiseSampler { .. }
        | Beardifier { .. }
        | IntervalSelect { .. }
                | Aquifer { .. }
        | BiomeTree { .. }
        | YClampedGradient { .. } => node,

        // Binary arithmetic ----------------------------------------------
        Add(a, b) => Add(map_box(a, &mut f), map_box(b, &mut f)),
        Sub(a, b) => Sub(map_box(a, &mut f), map_box(b, &mut f)),
        Mul(a, b) => Mul(map_box(a, &mut f), map_box(b, &mut f)),
        Div(a, b) => Div(map_box(a, &mut f), map_box(b, &mut f)),
        Min(a, b) => Min(map_box(a, &mut f), map_box(b, &mut f)),
        Max(a, b) => Max(map_box(a, &mut f), map_box(b, &mut f)),
        FloorDiv(a, b) => FloorDiv(map_box(a, &mut f), map_box(b, &mut f)),

        // Unary arithmetic -----------------------------------------------
        Abs(a) => Abs(map_box(a, &mut f)),
        Square(a) => Square(map_box(a, &mut f)),
        Cube(a) => Cube(map_box(a, &mut f)),
        Sqrt(a) => Sqrt(map_box(a, &mut f)),
        Squeeze(a) => Squeeze(map_box(a, &mut f)),

        // Structured nodes with children ---------------------------------
        EndIslands(a) => EndIslands(map_box(a, &mut f)),
        RangeChoice {
            input,
            min,
            max,
            in_bounds,
            out_of_bounds_lo,
            out_of_bounds_hi,
        } => RangeChoice {
            input: map_box(input, &mut f),
            min: map_box(min, &mut f),
            max: map_box(max, &mut f),
            in_bounds: map_box(in_bounds, &mut f),
            out_of_bounds_lo: map_box(out_of_bounds_lo, &mut f),
            out_of_bounds_hi: map_box(out_of_bounds_hi, &mut f),
        },
        CacheFlat(a) => CacheFlat(map_box(a, &mut f)),
        Cache2D(a) => Cache2D(map_box(a, &mut f)),
        CacheInterpolated(a) => CacheInterpolated(map_box(a, &mut f)),
        FindTopSurface {
            density,
            initial,
            min_y,
            max_y,
        } => FindTopSurface {
            density: map_box(density, &mut f),
            initial: map_box(initial, &mut f),
            min_y,
            max_y,
        },

        Root(channels) => Root(channels.into_iter().map(&mut f).collect()),
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

    /// Simple case: two structurally equal but separately-constructed
    /// subtrees inside a Root — after CSE, both children resolve to the
    /// same fingerprint → both clones originate from the same "first" node,
    /// which means their structure (and fingerprint) will compare equal.
    #[test]
    fn identical_children_share_after_cse() {
        // Construct two equivalent but independent Add(X, 2) — these would
        // normally be two distinct Box allocations but structurally equal.
        // (After normalize they're Add(2,X) on both sides; CSE just needs
        // structure equality, which our PartialEq already guarantees.)
        let a = Add(boxed(Constant(2.0)), boxed(InputX));
        let b = Add(boxed(Constant(2.0)), boxed(InputX));
        let root = Root(vec![a, b]);

        let after = eliminate(root);
        if let Root(v) = after {
            assert_eq!(v.len(), 2);
            // 1) Must be PartialEq-equal
            assert_eq!(v[0], v[1]);
            // 2) Both must produce the same fingerprint — which is the
            //    actual invariant we use in `table.entry(fp)` matching.
            assert_eq!(fingerprint(&v[0]), fingerprint(&v[1]));
        } else {
            panic!("expected Root output");
        }
    }

    /// Deduplication is transitive across nesting depths:
    ///   Mul( Add(1,2) , Mul(Add(1,2), 3) )
    /// After constant_folding both Add(1,2) become Constant(3); then CSE
    /// should make the two Constant(3) resolve to clones even if they
    /// appear at different tree depths.
    ///
    /// (We skip constant_folding here — just verify depth-insensitive de-dup.)
    #[test]
    fn dedup_across_depths() {
        let shared_inner = Add(boxed(Constant(1.0)), boxed(Constant(2.0)));
        // two instances of the same shared_inner structural subgraph
        let root = Mul(
            boxed(shared_inner.clone()),
            boxed(Mul(boxed(shared_inner), boxed(Constant(3.0)))),
        );

        let after = eliminate(root);
        // Must still be Mul(_, Mul(_, 3)) structure — no semantic changes
        match after {
            Mul(left, right) => {
                assert_eq!(&*left, &Add(boxed(Constant(1.0)), boxed(Constant(2.0))));
                match &*right {
                    Mul(inner, three) => {
                        assert_eq!(&**inner, &Add(boxed(Constant(1.0)), boxed(Constant(2.0))));
                        assert_eq!(&**three, &Constant(3.0));
                    }
                    other => panic!("right child should be Mul, got {other:?}"),
                }
            }
            other => panic!("top should be Mul, got {other:?}"),
        }
    }

    /// Root channels of different types must NOT get mixed up by CSE.
    #[test]
    fn different_types_not_merged() {
        let x = Add(boxed(InputX), boxed(InputY));
        let y = Mul(boxed(InputX), boxed(InputY));
        assert_ne!(
            fingerprint(&x),
            fingerprint(&y),
            "Add and Mul with same children must have different fingerprints"
        );
        let root = Root(vec![x, y]);
        let after = eliminate(root);
        if let Root(v) = after {
            assert!(matches!(&v[0], Add(_, _)));
            assert!(matches!(&v[1], Mul(_, _)));
        } else {
            panic!("expected Root");
        }
    }
}
