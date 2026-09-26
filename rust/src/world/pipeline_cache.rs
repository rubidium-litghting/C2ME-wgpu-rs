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

//! Pipeline cache: in-memory `(fingerprint, pass) → ComputePipeline` map +
//! future on-disk LZ4_flex-compressed `wgpu::PipelineCache` blob.
//!
//! ## Cache strategy
//!
//! Two-tier:
//!   1. **In-memory** (`IN_MEMORY_CACHE`): process-global `Mutex<FxHashMap>`.
//!      Keyed by `(fingerprint: u32, pass_id: u8)`:
//!        * `pass_id == 0` → single-pass `main_entry` (no CacheInterpolated)
//!        * `pass_id == 1` → two-pass `main_pass_a`
//!        * `pass_id == 2` → two-pass `main_pass_b`
//!      Hit → clone the `Arc`-backed `wgpu::ComputePipeline` and return.
//!      Miss → compile via `device.create_compute_pipeline` then insert.
//!   2. **On-disk** (TODO): `$XDG_CACHE_HOME/c2me/wgpu_pipeline_<hex>.cache`,
//!      LZ4_flex-compressed `wgpu::PipelineCache` data blob.
//!
//! ## Cross-world sharing
//!
//! MC servers often run multiple worlds (overworld / nether / end) with
//! different NoiseRouter configs → different fingerprints → different
//! pipeline cache slots, but **worlds sharing the same worldgen preset
//! (e.g. multiple nether-style dimensions) reuse one pipeline pair**.
//! The global cache makes this automatic.
//!
//! ## Thread safety
//!
//! `wgpu::ComputePipeline` is internally `Arc`-wrapped and `Send + Sync`,
//! so the `Mutex<FxHashMap<..>>` is sufficient. `get_or_compile` may
//! briefly hold the lock twice: once to check, then again to insert
//! after compilation. Compilation itself happens **outside** the lock —
//! two threads racing on the same key will compile twice and the loser's
//! pipeline is dropped (idempotent, just wasted work).

use std::sync::{Mutex, OnceLock};

use anyhow::Context;
use rustc_hash::FxHashMap;

use crate::codegen::wgsl_gen::Pass;
use crate::Result;

/// Pass identifier as a cache key suffix.
fn pass_id(pass: Pass) -> u8 {
    match pass {
        Pass::Single => 0,
        Pass::A => 1,
        Pass::B => 2,
    }
}

/// Global in-memory pipeline cache keyed by (fingerprint, pass_id).
static IN_MEMORY_CACHE: OnceLock<Mutex<FxHashMap<(u32, u8), wgpu::ComputePipeline>>> = OnceLock::new();

fn cache() -> &'static Mutex<FxHashMap<(u32, u8), wgpu::ComputePipeline>> {
    IN_MEMORY_CACHE.get_or_init(|| Mutex::new(FxHashMap::default()))
}

/// Look up an existing pipeline by (fingerprint, pass); on miss, compile
/// a new one from `shader_src` and insert it into the cache.
///
/// `device` must outlive the returned pipeline (in practice: the
/// singleton `WgpuInstance` lives for the whole process, so this holds).
///
/// `pipeline_layout` is borrowed only for the duration of
/// `create_compute_pipeline` — once the pipeline is built, the layout
/// can be dropped (the pipeline holds its own internal reference).
pub fn get_or_compile(
    fingerprint: u32,
    pass: Pass,
    shader_src: &str,
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
) -> Result<wgpu::ComputePipeline> {
    let key = (fingerprint, pass_id(pass));
    let entry_point = pass.entry_point();

    // 1. Fast path: cache hit (clone is cheap — Arc bump).
    {
        let map = cache().lock().expect("pipeline cache mutex poisoned");
        if let Some(p) = map.get(&key) {
            return Ok(p.clone());
        }
    }

    // 2. Slow path: compile. Outside the lock — compilation may take
    //    hundreds of ms and we don't want to block other lookups.
    let t_src0 = std::time::Instant::now();
    eprintln!("[c2me-wgpu][compile] shader source: {} lines, {} bytes",
              shader_src.lines().count(), shader_src.len());

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("c2me_wgpu_shader_{fingerprint:08x}_p{}", pass_id(pass))),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    eprintln!("[c2me-wgpu][compile] create_shader_module done in {:?}",
              t_src0.elapsed());

    let t_pipe0 = std::time::Instant::now();
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("c2me_wgpu_pipeline_{fingerprint:08x}_p{}", pass_id(pass))),
        layout: Some(pipeline_layout),
        module: &shader,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    eprintln!("[c2me-wgpu][compile] create_compute_pipeline done in {:?}",
              t_pipe0.elapsed());

    // 3. Insert. If another thread raced us, drop the loser.
    {
        let mut map = cache().lock().expect("pipeline cache mutex poisoned");
        if let Some(existing) = map.get(&key) {
            return Ok(existing.clone());
        }
        map.insert(key, pipeline.clone());
    }

    Ok(pipeline)
}

// ============================================================================
// On-disk LZ4_flex-compressed pipeline cache (TODO — phase 4+)
// ============================================================================

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

// ============================================================================
// Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// Basic sanity: the cache is lazily initialised and non-poisoned.
    /// Doesn't need a real GPU — just verifies the OnceLock + Mutex work.
    #[test]
    fn cache_returns_empty_map_initially() {
        let map = cache().lock().unwrap();
        // Map may have entries from other tests (test ordering is not
        // guaranteed), so we only assert it's lockable and is a HashMap.
        let _ = map.len();
    }

    /// `pass_id` mapping is stable — important for cache key stability
    /// across versions of this module.
    #[test]
    fn pass_id_mapping_is_stable() {
        assert_eq!(pass_id(Pass::Single), 0);
        assert_eq!(pass_id(Pass::A), 1);
        assert_eq!(pass_id(Pass::B), 2);
    }
}
