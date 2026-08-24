use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use anyhow::{Result, anyhow};

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    if let AstNode::BiomeTree { seed } = node {
        let var = ctx.fresh_var("biome");
        ctx.emit(format!(
            "let {var}: f32 = biome_tree({}, cell_x, cell_y, cell_z);",
            seed
        ));
        Ok(var)
    } else {
        Err(anyhow!("expected BiomeTree"))
    }
}
