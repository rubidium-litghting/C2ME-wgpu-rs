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

/// Hash an AST subtree using FxHash32.
pub fn hash_ast(node: &AstNode) -> u32 {
    let mut hasher = FxHasher::default();
    node.hash(&mut hasher);
    hasher.finish() as u32
}

// AstNode already derives PartialEq for structural equality.
// We just provide a conveniece wrapper name.
pub type AstEq = AstNode;
