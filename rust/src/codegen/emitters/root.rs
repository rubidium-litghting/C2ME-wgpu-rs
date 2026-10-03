use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use anyhow::{Result, anyhow};

pub fn emit(root: &AstNode, ctx: &mut GenCtx) -> Result<()> {
    match root {
        AstNode::Root(channels) => {
            let limit = crate::max_channels();
            let take = if limit == 0 { channels.len() } else { limit.min(channels.len()) };
            for (i, ch) in channels.iter().take(take).enumerate() {
                let var = super::emit_node(ch, ctx)?;
                ctx.emit(format!("out_buf[chunk_cell_base + {}u * cells_per_chunk + cell_idx] = {};", i, var));
            }
            Ok(())
        }
        _ => Err(anyhow!("root::emit called with non-Root node")),
    }
}
