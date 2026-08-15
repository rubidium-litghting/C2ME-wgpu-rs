/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2021-2026 RubidiumLighting
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

//! Compute a stable pipeline-cache fingerprint key.
//!
//! Key = FxHash64(AST_opt_hash ++ adapter_info.vendor ++ adapter_info.device
//!                 ++ WGPU_VERSION_MICRO ++ workarounds_bitmask)
//!
//! Two worlds with the same NoiseRouter (e.g. same seed same generator) share
//! a single compiled ComputePipeline via this key — reducing startup cost
//! from ~200ms to <1ms on subsequent runs after the first cache hit.
//!
//! The cached blob is LZ4-compressed and stored on disk at
//! `$XDG_CACHE/c2me/wgpu_pipeline_<keyhex>.bin` (or equivalent Windows/macOS
//! cache directory).

use crate::ast::AstNode;
use crate::ast::equality::hash_ast;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};

/// Build the 64-bit cache fingerprint.
pub fn fingerprint(ast_root: &AstNode, adapter_vendor_id: u32, adapter_device_id: u32,
                   workaround_bits: u32, wgpu_version_micro: u32) -> u64 {
    let mut h = FxHasher::default();
    hash_ast(ast_root).hash(&mut h);
    adapter_vendor_id.hash(&mut h);
    adapter_device_id.hash(&mut h);
    workaround_bits.hash(&mut h);
    wgpu_version_micro.hash(&mut h);
    h.finish()
}
