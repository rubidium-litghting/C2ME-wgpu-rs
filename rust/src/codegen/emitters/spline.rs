use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use anyhow::{Result, anyhow};

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    if let AstNode::Spline { points: _ } = node {
        let var = ctx.fresh_var("spline");
        // For now, just call a placeholder function with a dummy value
        ctx.emit(format!("let {var}: f32 = 0.0; // Spline not fully implemented"));
        Ok(var)
    } else {
        Err(anyhow!("expected Spline"))
    }
}
