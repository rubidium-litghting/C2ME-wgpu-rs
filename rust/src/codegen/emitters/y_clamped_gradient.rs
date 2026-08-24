use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use anyhow::{Result, anyhow};

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    if let AstNode::YClampedGradient { y_min, y_max, input, gradient } = node {
        let input_var = super::emit_node(input, ctx)?;
        let grad_var = super::emit_node(gradient, ctx)?;
        let var = ctx.fresh_var("yclamp");
        ctx.emit(format!(
            "let {var}: f32 = y_clamped_gradient({}i, {}i, {input_var}, {grad_var}, cell_y);",
            y_min, y_max
        ));
        Ok(var)
    } else {
        Err(anyhow!("expected YClampedGradient"))
    }
}
