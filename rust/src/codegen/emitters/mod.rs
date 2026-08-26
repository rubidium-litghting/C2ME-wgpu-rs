pub mod aquifer;
pub mod beardifier;
pub mod binary;
pub mod biome_tree;
pub mod cache_like;
pub mod constant;
pub mod coordinate;
pub mod end_islands;
pub mod find_top_surface;
pub mod interval_select;
pub mod noise_interpolated;
pub mod noise_perlin;
pub mod noise_simplex;
pub mod range_choice;

pub mod root;
pub mod spline;
pub mod unary;
pub mod y_clamped_gradient;

use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use anyhow::{Result, anyhow};

/// 统一节点分发函数。每个 emitter 模块都必须实现 `pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String>`。
pub fn emit_node(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    match node {
        AstNode::NoiseSampler { .. } => {
            // TODO: implement noise sampler emitter
            Ok("0.0".to_string()) // 临时占位
        }
        AstNode::Constant(_) => constant::emit(node, ctx),
        AstNode::InputX | AstNode::InputY | AstNode::InputZ => coordinate::emit(node, ctx),
        AstNode::Add(_, _)
        | AstNode::Sub(_, _)
        | AstNode::Mul(_, _)
        | AstNode::Div(_, _)
        | AstNode::Min(_, _)
        | AstNode::Max(_, _)
        | AstNode::FloorDiv(_, _) => binary::emit(node, ctx),
        AstNode::Abs(_) | AstNode::Square(_) | AstNode::Cube(_) | AstNode::Sqrt(_) => {
            unary::emit(node, ctx)
        }
        AstNode::PerlinSampler { .. } => noise_perlin::emit(node, ctx),
        AstNode::DoublePerlin { .. } => noise_perlin::emit(node, ctx), // 复用 Perlin emitter
        AstNode::SimplexNoise2D { .. } | AstNode::SimplexNoise3D { .. } => {
            noise_simplex::emit(node, ctx)
        }
        AstNode::InterpolatedNoiseSampler { .. } => noise_interpolated::emit(node, ctx),
        AstNode::Beardifier { .. } => beardifier::emit(node, ctx),
        AstNode::Spline { .. } => spline::emit(node, ctx),
        AstNode::RangeChoice { .. } => range_choice::emit(node, ctx),
        AstNode::IntervalSelect { .. } => interval_select::emit(node, ctx),
        AstNode::YClampedGradient { .. } => y_clamped_gradient::emit(node, ctx),
        AstNode::CacheFlat(_) | AstNode::Cache2D(_) | AstNode::CacheInterpolated(_) => {
            cache_like::emit(node, ctx)
        }
        AstNode::EndIslands(_) => end_islands::emit(node, ctx),
        AstNode::FindTopSurface { .. } => find_top_surface::emit(node, ctx),
        AstNode::Aquifer { .. } => aquifer::emit(node, ctx),
        AstNode::BiomeTree { .. } => biome_tree::emit(node, ctx),
        AstNode::Root(_) => unreachable!("Root should be handled by root::emit"),
    }
}
