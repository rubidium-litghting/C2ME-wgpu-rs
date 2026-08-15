/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2021-2026 RubidiumLighting
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

//! Pass 2: Common Subexpression Elimination.
//!
//! Builds a HashMap<(hash, AstNode), Rc<AstNode>> during post-order walk.
//! If two subtrees have equal hash AND structural equality, they share
//! the same Rc — the WGSL emitter then uses one `let` binding for both.
//!
//! This commonly triggers on the XYZ coordinate inputs plus the 8 copies
//! of `cache_like(erosion)` that the NoiseRouter wraps with different strides.

use crate::ast::AstNode;

pub fn eliminate(root: AstNode) -> AstNode {
    // TODO skeleton:
    //   use fxhash::FxHashMap;
    //   post-order walk, insert each node into map keyed by (hash_ast(&n), n.clone()),
    //   if present replace current pointer with the existing clone.
    //   (Since AstNode is Clone with Box children, real impl may need Rc<RefCell>
    //    or an arena to actually share nodes; spec uses Rc.)
    root
}
