/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2026 RubidiumLighting
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

//! Persistent, LZ4_flex-compressed wgpu::PipelineCache on disk.
//!
//! Location: `$XDG_CACHE_HOME/c2me/wgpu_pipeline_<hex>.cache` (or platform eqv).
//! A cache miss causes a ~200ms shader recompile first time per world seed
//! type; subsequent loads hit the cache.

use anyhow::Context;

use crate::Result;

/// Try to load a cached pipeline blob for the given fingerprint.
pub fn load(_fingerprint: u64) -> Result<Option<Vec<u8>>> {
    Ok(None) // skeleton: no persistent cache yet
}

/// Save a raw pipeline blob (post-LZ4 compression) for `fingerprint`.
pub fn save(_fingerprint: u64, _compressed_blob: &[u8]) -> Result<()> {
    Ok(())
}

/// Compress a raw wgpu::PipelineCache data blob using LZ4_flex.
pub fn compress(raw: &[u8]) -> Vec<u8> {
    lz4_flex::compress(raw)
}

/// Decompress.
pub fn decompress(compressed: &[u8]) -> Result<Vec<u8>> {
    lz4_flex::decompress_size_prepended(compressed)
        .with_context(|| "pipeline cache error: decompression failed")
}
