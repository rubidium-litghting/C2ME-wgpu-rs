// SPDX-License-Identifier: MIT

//! 4-pass optimization pipeline over AstNode.
//!
//! Order matters:
//!   1. constant_folding — evaluate pure-constant subtrees (e.g. 3+4 → 7)
//!   2. normalize        — canonical forms, e.g. commutative child ordering
//!   3. cse              — hash-cons structurally equal subtrees so the
//!                         emitter can reuse WGSL expressions
//!   4. dce              — drop nodes that Root channels never reference
//!
//! The normalize → cse dependency is strict: if CSE runs BEFORE normalization,
//! `Add(InputX, 2)` and `Add(2, InputX)` have two different fingerprints and
//! won't actually share anything.

pub mod constant_folding;
pub mod cse;
pub mod dce;
pub mod normalize;

use crate::ast::AstNode;
use crate::Result;

/// Run all 4 passes in order on a Root node, returning the optimized AST.
pub fn run_pipeline(root: AstNode) -> Result<AstNode> {
    let n = constant_folding::fold(root)?;
    let n = normalize::canonicalize(n);
    let n = cse::eliminate(n);
    let n = dce::strip_dead(n);
    Ok(n)
}

/// Helpers shared by the pass test modules.
#[cfg(test)]
pub(crate) mod test_support {
    use crate::ast::AstNode;

    /// Box a node, so test expressions read as `Add(boxed(InputX), boxed(..))`
    /// instead of `Box::new(...)` at every site.
    pub(crate) fn boxed(n: AstNode) -> Box<AstNode> {
        Box::new(n)
    }
}
