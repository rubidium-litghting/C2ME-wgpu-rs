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

//! 19+ node emitters, one per AstNode variant group.
//!
//! Each has a single function `pub fn emit(node: &AstNode, ctx: &mut GenCtx)
//! -> Result<String>` that returns the WGSL variable name (or expression
//! string) holding the value for this subtree.

pub mod binary;
pub mod unary;
pub mod constant;
pub mod coordinate;

pub mod noise_perlin;
pub mod noise_simplex;
pub mod noise_interpolated;

pub mod end_islands;
pub mod beardifier;
pub mod spline;
pub mod range_choice;
pub mod interval_select;
pub mod y_clamped_gradient;

pub mod cache_like;
pub mod find_top_surface;
pub mod aquifer;
pub mod biome_tree;

pub mod root;

use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use crate::Result;

/// Top-level dispatch — matches on AstNode variant, delegates to sub-emitter.
pub fn emit_node(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    use AstNode::*;
    match node {
        Constant(_) => constant::emit(node, ctx),
        InputX | InputY | InputZ => coordinate::emit(node, ctx),
        Add(_, _) | Sub(_, _) | Mul(_, _) | Div(_, _) | Min(_, _) | Max(_, _) => binary::emit(node, ctx),
        Abs(_) | Square(_) | Cube(_) | Sqrt(_) | FloorDiv(_, _) => unary::emit(node, ctx),

        PerlinSampler { .. } | DoublePerlin { .. } => noise_perlin::emit(node, ctx),
        SimplexNoise2D { .. } | SimplexNoise3D { .. } => noise_simplex::emit(node, ctx),
        InterpolatedNoiseSampler { .. } => noise_interpolated::emit(node, ctx),

        EndIslands(_) => end_islands::emit(node, ctx),
        Beardifier { .. } => beardifier::emit(node, ctx),
        Spline { .. } => spline::emit(node, ctx),
        RangeChoice { .. } => range_choice::emit(node, ctx),
        IntervalSelect { .. } => interval_select::emit(node, ctx),
        YClampedGradient { .. } => y_clamped_gradient::emit(node, ctx),

        CacheFlat(_) | Cache2D(_) | CacheInterpolated(_) => cache_like::emit(node, ctx),
        FindTopSurface { .. } => find_top_surface::emit(node, ctx),
        Aquifer { .. } => aquifer::emit(node, ctx),
        BiomeTree { .. } => biome_tree::emit(node, ctx),

        Root(_) => root::emit(node, ctx),
    }
}
