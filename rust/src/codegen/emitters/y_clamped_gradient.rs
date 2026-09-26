use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use anyhow::{Result, anyhow};

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    let AstNode::YClampedGradient { from_y, to_y, from_value, to_value } = node else {
        return Err(anyhow!("expected YClampedGradient"));
    };
    let var = ctx.fresh_var("yclamp");
    ctx.emit(format!(
        "let {var}: f32 = clamped_map(cell_y, {from_y:.9e}, {to_y:.9e}, {from_value:.9e}, {to_value:.9e});"
    ));
    Ok(var)
}
