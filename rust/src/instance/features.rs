// SPDX-License-Identifier: MIT

//! Capability check — MUST PASS or we refuse to use the adapter.
//!
//! 4 mandatory checks:
//!   1. max_storage_buffer_binding_size ≥ 128 MiB
//!   2. max_compute_workgroups_per_dimension ≥ 65535
//!   3. max_buffer_address_space_bits ≥ 31  (i.e. ≥ 2 GiB addressable)
//!   4. Features::PIPELINE_CACHE is present
//!
//! Explicitly NOT required: SHADER_F64 — we are pure fp32.

use crate::Result;

/// Run all 4 mandatory capability checks against an adapter.
/// Returns Ok on pass, Err(MissingCapability(name)) on first failure.
pub fn check_required(adapter: &wgpu::Adapter, disable_pipeline_cache: bool) -> Result<()> {
    if adapter.limits().max_compute_workgroups_per_dimension < 65535 {
        return Err(anyhow::anyhow!("max_compute_workgroups_per_dimension < 65535"));
    }
    // 128 MiB is the WebGPU default minimum. We used to require 256 MiB to
    // accommodate worlds with hundreds of external nodes (e.g. Tectonic under
    // Sinytra Connector), but those are now capped at 64 per dimension, so
    // 128 MiB is plenty. Lowering this lets modern mobile GPUs (e.g. Adreno 750,
    // which reports exactly 128 MiB) use the GPU path instead of falling back.
    if adapter.limits().max_storage_buffer_binding_size < 128 * 1024 * 1024 {
        return Err(anyhow::anyhow!("max_storage_buffer_binding_size < 128 * 1024 * 1024"));
    }
    // PIPELINE_CACHE is only required when we actually use it. With
    // `disable_pipeline_cache` in the config we skip it, so adapters that lack the
    // feature can still run.
    if !disable_pipeline_cache && !adapter.features().contains(wgpu::Features::PIPELINE_CACHE) {
        return Err(anyhow::anyhow!("Features::PIPELINE_CACHE not present"));
    }
    Ok(())
}
