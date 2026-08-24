use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use anyhow::{Result, anyhow};

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    if let AstNode::Aquifer { density, fluid_level } = node {
        let density_var = super::emit_node(density, ctx)?;
        let fluid_var = super::emit_node(fluid_level, ctx)?;
        let var = ctx.fresh_var("aquifer");
        ctx.emit(format!(
            "let {var}: f32 = aquifer_compute({density_var}, {fluid_var}, cell_x, cell_y, cell_z);"
        ));
        Ok(var)
    } else {
        Err(anyhow!("expected Aquifer"))
    }
}
