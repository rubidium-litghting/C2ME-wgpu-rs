// SPDX-License-Identifier: MIT

//! `FindTopSurface` — present in the AST only for blob-format compatibility.
//!
//! The node does not exist in 1.21.1's `NoiseRouter` (it was removed before
//! 1.18), and the Java serialiser never emits `TAG_FIND_TOP_SURF`. The
//! deserialiser still accepts it so old blobs and the unit tests work, but
//! reaching code generation means something upstream is wrong.
//!
//! This used to return a placeholder string that was never defined in the
//! WGSL library, so hitting it produced a shader that failed to compile. An
//! explicit error is better: it propagates to the dimension-init catch and
//! falls that dimension back to CPU with a readable reason.

use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use crate::Result;
use anyhow::anyhow;

pub fn emit(node: &AstNode, _ctx: &mut GenCtx) -> Result<String> {
    Err(anyhow!(
        "FindTopSurface is not supported by the WGSL backend (node {node:?}); \
         it does not exist in 1.21.1 worldgen"
    ))
}