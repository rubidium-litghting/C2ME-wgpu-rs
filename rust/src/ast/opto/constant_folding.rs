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

//! Pass 1: constant folding. Any subtree whose inputs are all Constant(f32)
//! is computed at optimization time and replaced with a single Constant.
//!
//! This drastically shrinks spline / range_choice trees that come from
//! hard-coded vanilla constants.

use crate::ast::AstNode;

pub fn fold(root: AstNode) -> AstNode {
    // TODO skeleton: post-order DFS, match on each (Const op Const) branch.
    //   e.g. Add(Constant(a), Constant(b)) => Constant(a + b)
    //   For noise nodes with all-Constant parameters, we don't fold (they
    //   need coordinate inputs). Only pure scalar arithmetic is folded.
    root
}
