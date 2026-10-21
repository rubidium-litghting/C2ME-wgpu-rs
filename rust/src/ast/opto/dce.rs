// SPDX-License-Identifier: MIT

//! Pass 4: Dead-code elimination.
//!
//! **Currently a deliberate NO-OP.**
//!
//! # Why AstNode-as-Box-tree makes DCE "no-op by default"
//!
//! The AstNode enum owns every child with `Box<AstNode>`. Any subtree reachable
//! from `Root(channels)` is, by definition, **referenced by exactly one Box**
//! and **will be consumed by the WGSL emitter as part of its parent node**.
//! This means the classic DCE scenario ("unreferenced block of instructions")
//! simply cannot occur: if a node exists in memory, its parent points at it,
//! and if the parent is reachable from Root then the emitter will emit code
//! for it (because the parent uses it to compute its own value).
//!
//! # The cases DCE *could* help with (not yet implemented)
//!
//! DCE would make sense once one of the following is true:
//!
//! 1. **Semantic dead-branch elimination** — e.g. `RangeChoice` has six
//!    children, but if `max` provably ≤ `min` then `in_bounds` is NEVER
//!    reachable regardless of input. Removing it saves a few WGSL ops.
//!    This requires lightweight interval / value-range analysis over
//!    DensityFunctions, not just structural traversal.
//!
//! 2. **All-Constant channel pruning** — if all 16 Root channels are
//!    Constant(v) we could skip shader compilation entirely and just
//!    memcpy the density on the CPU. But that's handled upstream (Java
//!    throws `UnsupportedDFException` before we even deserialize).
//!
//! 3. **Emitter CSE-replaced clones** — after normalize + CSE, two semantically
//!    identical subtrees are two separate Box allocations with identical
//!    structure. The emitter deduplicates their WGSL `let` bindings via
//!    `FxHashMap<&AstNode, ExprId>`, so no extra WGSL code is emitted even
//!    though the Rust-side AST still has duplicate nodes. We could (costly)
//!    swap Box children for Rc to save Rust heap, but the benefit is O(MB) not
//!    O(100 MB) on NoiseRouter trees. For now we accept the duplication.
//!
//! We retain DCE as a pass slot in the 4-pipeline ordering as a hook point
//! for any of the above cases when we later decide the cost/benefit is worth
//! it. Callers can rely on strip_dead() always succeeding (no Result needed).

use crate::ast::AstNode;

pub fn strip_dead(root: AstNode) -> AstNode {
    // No-op — see module docs for rationale.
    root
}

// ======================================================================
// Tests
// ======================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use AstNode::*;
    use crate::ast::opto::test_support::boxed;

    /// DCE must be the identity function: the AST round-trips unchanged.
    #[test]
    fn dce_is_identity_for_now() {
        let tree = Root(vec![
            Add(boxed(Constant(1.0)), boxed(InputX)),
            Min(boxed(InputY), boxed(SimplexNoise3D { seed: 42 })),
        ]);
        let before = tree.clone();
        let after = strip_dead(tree);
        assert_eq!(before, after);
    }
}
