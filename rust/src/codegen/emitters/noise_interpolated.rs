use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use anyhow::{Result, anyhow};

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    if let AstNode::InterpolatedNoiseSampler { child } = node {
        let child_var = super::emit_node(child, ctx)?;
        let var = ctx.fresh_var("interp");
        ctx.emit(format!("let {var}: f32 = noise_interpolated_sampler(0, {child_var});"));
        Ok(var)
    } else {
        Err(anyhow!("expected InterpolatedNoiseSampler"))
    }
}
