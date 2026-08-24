use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use anyhow::{Result, anyhow};

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    if let AstNode::Beardifier { struct_index } = node {
        let var = ctx.fresh_var("beard");
        ctx.emit(format!("let {var}: f32 = beardifier_adjust({}, cell_y, 0.0);", struct_index));
        Ok(var)
    } else {
        Err(anyhow!("expected Beardifier"))
    }
}
