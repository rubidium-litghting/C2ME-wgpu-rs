//! InterpolatedNoiseSampler — MC wraps a density function and samples it at
//! cell corners, then trilinearly interpolates within the cell.
//!
//! This is semantically identical to CacheInterpolated (both describe MC's
//! cell-corner + trilinear-lerp behaviour), so we delegate to the
//! CacheInterpolated emitter rather than duplicating the WGSL.

use crate::ast::AstNode;
use crate::codegen::emitters::cache_like;
use crate::codegen::wgsl_gen::GenCtx;
use anyhow::{Result, anyhow};

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    if let AstNode::InterpolatedNoiseSampler { child } = node {
        cache_like::emit_cache_interp(child, ctx)
    } else {
        Err(anyhow!("expected InterpolatedNoiseSampler"))
    }
}
