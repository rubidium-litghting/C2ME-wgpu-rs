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

//! Pass 3: Dead Channel Elimination.
//!
//! NoiseRouter has 16 channels, but some modpacks / datapacks zero out some
//! of them. DCE walks from the Root's used channel indices, marks visited
//! subtrees, then drops any AstNode subtree that was never reached.
//!
//! In practice this prunes half the tree when JVM server-side biome blending
//! is off (BiomeTree unused).

use crate::ast::AstNode;

pub fn strip_dead(root: AstNode) -> AstNode {
    // TODO skeleton:
    //   match root {
    //       AstNode::Root(channels) => {
    //           // keep all 16 as-is for now; when Java side reports some channel
    //           // is Constant(0.0) or the same as another, swap to a cheap Constant stub
    //           AstNode::Root(channels)
    //       }
    //       other => other
    //   }
    root
}
