// SPDX-License-Identifier: MIT

//! FindTopSurface — 铁律③ i32-粗判 early-exit in upward search loop.

use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use crate::Result;

pub fn emit(_node: &AstNode, _ctx: &mut GenCtx) -> Result<String> {
    Ok("findtop_stub".into())
}
