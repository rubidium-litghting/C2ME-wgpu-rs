use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use anyhow::{Result, anyhow};

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    if let AstNode::Beardifier { struct_index } = node {
        let var = ctx.fresh_var("beard");
        // GPU path omits Beardifier; real beard is applied per-block on the Java side.
        let _ = struct_index;
        ctx.emit(format!("let {var}: f32 = 0.0;"));
        Ok(var)
    } else {
        Err(anyhow!("expected Beardifier"))
    }
}
