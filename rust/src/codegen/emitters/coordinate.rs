// SPDX-License-Identifier: MIT

//! Coordinate inputs — this is where 铁律① local-offset is applied.
//!
//! Every reference to InputX / InputY / InputZ in the AST reads from
//! pre-declared `cell_x`/`cell_y`/`cell_z` f32 variables which are ALREADY
//! the origin-subtracted small magnitude values. The absolute world
//! coordinate is recovered in the individual noise emitter via
//! `noise_position_with_origin()`.

use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use crate::Result;

pub fn emit(node: &AstNode, _ctx: &mut GenCtx) -> Result<String> {
    // These variable names are declared once in main_entry() scope by root emitter.
    Ok(match node {
        AstNode::InputX => "cell_x".into(),
        AstNode::InputY => "cell_y".into(),
        AstNode::InputZ => "cell_z".into(),
        _ => unreachable!(),
    })
}
