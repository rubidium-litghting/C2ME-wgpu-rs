use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use anyhow::{Result, anyhow};

/// Emit a read from the external-input buffer (binding 6).
///
/// Layout: [chunk0: ext0[G], ext1[G], ...][chunk1: ...]
/// where G = cells_per_chunk. `num_ext` lives in params._pad[0].
pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    let AstNode::ExternalInput { index } = node else {
        return Err(anyhow!("expected ExternalInput"));
    };
    let var = ctx.fresh_var("ext");
    // ext_idx = chunk_cell_base / 16 (chunk-local base) ... simpler: use
    // chunk_index and num_ext carried in params.
    ctx.emit(format!(
        "let {var}: f32 = external_inputs[chunk_index * (i32(params._pad[0]) * i32(cells_per_chunk)) + {i} * i32(cells_per_chunk) + i32(cell_idx)];",
        i = index,
    ));
    Ok(var)
}
