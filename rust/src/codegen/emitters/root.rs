use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use anyhow::{Result, anyhow};

pub fn emit(root: &AstNode, ctx: &mut GenCtx) -> Result<()> {
    match root {
        AstNode::Root(channels) => {
            for (i, ch) in channels.iter().enumerate() {
                let var = super::emit_node(ch, ctx)?;
                ctx.emit(format!("out_buf[chunk_cell_base + {}u * cells_per_chunk + cell_idx] = {};", i, var));
            }
            Ok(())
        }
        _ => Err(anyhow!("root::emit called with non-Root node")),
    }
}
