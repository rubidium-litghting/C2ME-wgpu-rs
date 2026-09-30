use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use anyhow::{Result, anyhow};

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    if let AstNode::RangeChoice { input, min, max, in_bounds, out_of_bounds_lo, out_of_bounds_hi } = node {
        let input_var = super::emit_node(input, ctx)?;
        let min_var = super::emit_node(min, ctx)?;
        let max_var = super::emit_node(max, ctx)?;
        let in_var = super::emit_node(in_bounds, ctx)?;
        let lo_var = super::emit_node(out_of_bounds_lo, ctx)?;
        let hi_var = super::emit_node(out_of_bounds_hi, ctx)?;
        let var = ctx.fresh_var("range");
        let out_var = ctx.fresh_var("range_oob");
        // MC semantics: min <= input < max ? in_bounds : (input < min ? lo : hi)
        // `select(a, b, cond)` = cond ? b : a.
        ctx.emit(format!(
            "let {out_var}: f32 = select({hi_var}, {lo_var}, {input_var} < {min_var});"
        ));
        ctx.emit(format!(
            "let {var}: f32 = select({out_var}, {in_var}, ({input_var} >= {min_var}) && ({input_var} < {max_var}));"
        ));
        Ok(var)
    } else {
        Err(anyhow!("expected RangeChoice"))
    }
}
