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
