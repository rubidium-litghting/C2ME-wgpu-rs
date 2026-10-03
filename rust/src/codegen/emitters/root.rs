use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use anyhow::{Result, anyhow};

pub fn emit(root: &AstNode, ctx: &mut GenCtx) -> Result<()> {
    match root {
        AstNode::Root(channels) => {
            // Diagnostic cap: emit only the first N channels overall.
            let limit = crate::max_channels();
            let global_cap = if limit == 0 { channels.len() } else { limit.min(channels.len()) };
            // Shard window [start, end) from the GenCtx (single-pass => [0, MAX)).
            let (shard_start, shard_end) = ctx.channel_range;
            let start = shard_start.min(global_cap);
            let end = shard_end.min(global_cap);
            for i in start..end {
                let ch = &channels[i];
                let var = super::emit_node(ch, ctx)?;
                // Write at the channel's GLOBAL index so every shard lands
                // in the same out_buf slot the single-pass shader used.
                ctx.emit(format!("out_buf[chunk_cell_base + {}u * cells_per_chunk + cell_idx] = {};", i, var));
            }
            Ok(())
        }
        _ => Err(anyhow!("root::emit called with non-Root node")),
    }
}
