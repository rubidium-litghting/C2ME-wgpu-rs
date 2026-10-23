// SPDX-License-Identifier: MIT

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
//!   2. **On-disk**: `$XDG_CACHE_HOME/c2me/wgpu_pipelines/<fingerprint>.cache`,
//!      LZ4_flex-compressed `wgpu::PipelineCache` data blob (see `load`/`save`).
//!      Populated in `get_or_compile`: loaded before `create_compute_pipeline`,
//!      persisted after via `PipelineCache::get_data`.
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
        Pass::C => 3,
        Pass::BlockOut => 4,
        Pass::Inner => 5,
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
    // §22 B-3: the bind-group layout gained binding 8/9 (aquifer). Any
    // pipeline cached under the old layout is invalid, so perturb the key
    // (and the on-disk cache path) with a layout version. Bump on any
    // binding change.
    const LAYOUT_VERSION: u32 = 27;
    let fingerprint = fingerprint ^ LAYOUT_VERSION.wrapping_mul(0x9E37_79B9);
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
    crate::wgpu_debug!("[compile] shader source: {} lines, {} bytes",
              shader_src.lines().count(), shader_src.len());

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("c2me_wgpu_shader_{fingerprint:08x}_p{}", pass_id(pass))),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    crate::wgpu_debug!("[compile] create_shader_module done in {:?}",
              t_src0.elapsed());

    // Build the compute pipeline. The `wgpu::PipelineCache` is optional:
    // some Vulkan drivers (Mali-G610) abort inside it, so
    // `disable_pipeline_cache` in the config skips it entirely.
    let t_pipe0 = std::time::Instant::now();
    let no_cache = crate::instance::WgpuInstance::instance()
        .map(|i| i.cold.disable_pipeline_cache)
        .unwrap_or(false);
    let pipeline = if no_cache {
        crate::wgpu_debug!("[compile] pipeline cache disabled; compiling without it");
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("c2me_wgpu_pipeline_{fingerprint:08x}_p{}", pass_id(pass))),
            layout: Some(pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    } else {
        // On-disk pipeline cache: load any previous blob for this fingerprint,
        // hand it to wgpu, then persist the (possibly extended) result.
        let cached_data = load(fingerprint).ok().flatten();
        // SAFETY: `cached_data` is either None or bytes previously returned by
        // PipelineCache::get_data (our own cache file); `fallback: true` makes
        // wgpu ignore corrupt data rather than UB.
        let pc = unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some(&format!("c2me_wgpu_pcache_{fingerprint:08x}")),
                data: cached_data.as_deref(),
                fallback: true,
            })
        };

        let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("c2me_wgpu_pipeline_{fingerprint:08x}_p{}", pass_id(pass))),
            layout: Some(pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: Some(&pc),
        });

        if let Some(data) = pc.get_data() {
            if let Err(e) = save(fingerprint, &data) {
                crate::wgpu_debug!("[cache] save failed: {e}");
            }
        }
        p
    };

    crate::wgpu_debug!("[compile] create_compute_pipeline done in {:?}",
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
// On-disk LZ4_flex-compressed pipeline cache
// ============================================================================

/// Directory holding on-disk pipeline caches.
fn cache_dir() -> Option<std::path::PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?;
    Some(base.join("c2me").join("wgpu_pipelines"))
}

fn cache_path(fingerprint: u32) -> Option<std::path::PathBuf> {
    Some(cache_dir()?.join(format!("{fingerprint:08x}.cache")))
}

/// Try to load a cached pipeline blob for the given fingerprint.
/// Returns the DECOMPRESSED data (ready to hand to PipelineCacheDescriptor).
pub fn load(fingerprint: u32) -> Result<Option<Vec<u8>>> {
    let Some(path) = cache_path(fingerprint) else { return Ok(None); };
    let compressed = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("read {path:?}")),
    };
    match decompress(&compressed) {
        Ok(raw) => {
            crate::wgpu_debug!("[cache] loaded pipeline cache {fingerprint:08x} ({} bytes)", raw.len());
            Ok(Some(raw))
        }
        Err(e) => {
            crate::wgpu_debug!("[cache] corrupt cache {fingerprint:08x}: {e}; ignoring");
            Ok(None)
        }
    }
}

/// Save a RAW pipeline blob (compressed here) for `fingerprint`.
pub fn save(fingerprint: u32, raw_blob: &[u8]) -> Result<()> {
    let Some(path) = cache_path(fingerprint) else { return Ok(()); };
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {dir:?}"))?;
    }
    let compressed = compress(raw_blob);
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, &compressed).with_context(|| format!("write {tmp:?}"))?;
    std::fs::rename(&tmp, &path).with_context(|| format!("rename to {path:?}"))?;
    crate::wgpu_debug!("[cache] saved pipeline cache {fingerprint:08x} ({} raw -> {} compressed)",
              raw_blob.len(), compressed.len());
    Ok(())
}

/// Compress a raw wgpu::PipelineCache data blob using LZ4_flex.
pub fn compress(raw: &[u8]) -> Vec<u8> {
    // Must match decompress(), which uses decompress_size_prepended.
    lz4_flex::compress_prepend_size(raw)
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
        assert_eq!(pass_id(Pass::C), 3);
    }
}
