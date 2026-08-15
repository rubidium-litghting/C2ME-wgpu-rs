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

//! Pass 4: Normalize AST to canonical forms so equivalent expressions hit
//! the same CSE entry and generate identical WGSL.
//!
//! Rules:
//!   • Commutative ops order children by stable hash:
//!         Mul(x, Add(a,b))  ↔  Mul(Add(a,b), x)  → order the same.
//!   • Flatten nested Add/Mul associative chains into n-ary (emitter then
//!     uses KahanSum reduction for length ≥ 3 — 铁律②).

use crate::ast::AstNode;

pub fn canonicalize(root: AstNode) -> AstNode {
    // TODO skeleton: post-order walk applying rewrite rules above.
    root
}
