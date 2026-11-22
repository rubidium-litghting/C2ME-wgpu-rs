// SPDX-License-Identifier: MIT

//! Capability check — MUST PASS or we refuse to use the adapter.
//!
//! The gate is exactly "can this adapter create the device we request". Both
//! sides read [`required_limits`] / [`required_features`], so a number can no
//! longer be tightened in one place and left behind in the other.
//!
//! Checks:
//!   1. `max_storage_buffer_binding_size` ≥ the requested value
//!   2. `max_buffer_size` ≥ the requested value
//!   3. `max_compute_workgroups_per_dimension` ≥ the requested value
//!   4. `Features::PIPELINE_CACHE` is present (unless we disabled that cache)
//!
//! Explicitly NOT required: SHADER_F64 — we are pure fp32.
//!
//! # Why the floor is the WebGPU default, not a shape
//!
//! The device is process-global and created before any world shape is known,
//! so the request must cover the worst shape a session may later load. It
//! therefore asks for `wgpu::Limits::default()` and this gate accepts exactly
//! adapters that can satisfy it. How much of that budget a *particular* shape
//! uses is decided later, per dimension, by
//! [`BatchLimit::derive`](crate::scheduler::executor::BatchLimit::derive):
//! it clamps the batch size to whatever the requested limits and the shape
//! allow. That derivation can lower the batch, never raise the device floor —
//! which is why the floor stays at the spec default instead of being derived
//! from one dimension's grid.

use crate::Result;

/// The limits the device is created with.
///
/// Everything that filters adapters ([`check_required`]) and everything that
/// requests the device must go through this function. A literal in one and a
/// different literal in the other is how a gate starts accepting hardware the
/// subsequent `request_device` rejects.
pub fn required_limits() -> wgpu::Limits {
    wgpu::Limits::default()
}

/// Features the device is created with. `PIPELINE_CACHE` is optional so that
/// adapters without it can still run when the user disables the cache.
pub fn required_features(disable_pipeline_cache: bool) -> wgpu::Features {
    if disable_pipeline_cache {
        wgpu::Features::empty()
    } else {
        wgpu::Features::PIPELINE_CACHE
    }
}

/// Run every mandatory capability check against an adapter.
/// Returns Ok on pass, Err with the failing limit on the first mismatch.
pub fn check_required(adapter: &wgpu::Adapter, disable_pipeline_cache: bool) -> Result<()> {
    let required = required_limits();
    let limits = adapter.limits();

    // 128 MiB is the WebGPU default and the smallest binding we may need for a
    // full batch of a tall dimension. We used to require 256 MiB to
    // accommodate hundreds of external nodes (e.g. Tectonic under Sinytra
    // Connector), but those are deduplicated and capped, so 128 MiB is enough
    // and lets mobile GPUs that report exactly 128 MiB (e.g. Adreno 750) use
    // the GPU path instead of falling back.
    if limits.max_storage_buffer_binding_size < required.max_storage_buffer_binding_size {
        return Err(anyhow::anyhow!(
            "max_storage_buffer_binding_size {} < required {}",
            limits.max_storage_buffer_binding_size,
            required.max_storage_buffer_binding_size));
    }
    // The staging slot is a regular buffer holding one batch's density and
    // block outputs back to back, so it is bounded by max_buffer_size rather
    // than the binding limit. This was the one hard limit the gate never
    // checked; an adapter below the default would pass here and then fail in
    // request_device with a much less readable error.
    if limits.max_buffer_size < required.max_buffer_size {
        return Err(anyhow::anyhow!(
            "max_buffer_size {} < required {}",
            limits.max_buffer_size,
            required.max_buffer_size));
    }
    // The corner shard dispatches one workgroup per 4×4×4 cells and puts the
    // batched chunks on Z; at 1088 blocks with 64 chunks that is ~2000
    // workgroups, well inside the default 65535.
    if limits.max_compute_workgroups_per_dimension
        < required.max_compute_workgroups_per_dimension
    {
        return Err(anyhow::anyhow!(
            "max_compute_workgroups_per_dimension {} < required {}",
            limits.max_compute_workgroups_per_dimension,
            required.max_compute_workgroups_per_dimension));
    }
    let features = required_features(disable_pipeline_cache);
    if !adapter.features().contains(features) {
        return Err(anyhow::anyhow!("missing required features: {features:?}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gate and the device request must ask for the same limits; this
    /// pins the function to the WebGPU defaults so lowering one side without
    /// the other is a deliberate, visible edit.
    #[test]
    fn required_limits_are_the_webgpu_defaults() {
        let required = required_limits();
        assert_eq!(required.max_storage_buffer_binding_size, 128 << 20);
        assert_eq!(required.max_buffer_size, 256 << 20);
        assert_eq!(required.max_compute_workgroups_per_dimension, 65535);
    }

    #[test]
    fn pipeline_cache_is_only_required_when_enabled() {
        assert!(required_features(false).contains(wgpu::Features::PIPELINE_CACHE));
        assert_eq!(required_features(true), wgpu::Features::empty());
    }
}
