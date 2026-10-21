// SPDX-License-Identifier: MIT

//! Structural Hash + Eq for CSE (common subexpression elimination) and for
//! building the pipeline-cache fingerprint key.
//!
//! Uses fxhash (fast non-cryptographic) since AST hashes are never persisted
//! beyond the process lifetime or a single cache file that stores its own
//! stronger integrity check.

use std::hash::{Hash, Hasher};

use rustc_hash::FxHasher;

use crate::ast::AstNode;

/// Hash an AST subtree using FxHash32 (truncated).
///
/// Used for the pipeline-cache key, which is a `u32` fingerprint.
pub fn hash_ast(node: &AstNode) -> u32 {
    hash_ast_u64(node) as u32
}

/// Full-width FxHash64 of an AST subtree.
///
/// Used by the optimizer passes (CSE, normalize) where a 32-bit hash would
/// make accidental collisions between distinct subtrees likely enough to
/// matter. Both passes previously carried an identical private copy.
pub fn hash_ast_u64(node: &AstNode) -> u64 {
    let mut hasher = FxHasher::default();
    node.hash(&mut hasher);
    hasher.finish()
}

// AstNode already derives PartialEq for structural equality.
// We just provide a conveniece wrapper name.
pub type AstEq = AstNode;
