// SPDX-License-Identifier: MIT

//! Instance module: discover GPU adapters, pick best device, check features,
//! apply vendor-specific workarounds.

pub mod adapter_pick;
pub mod features;
pub mod workarounds;

use std::sync::Arc;
use parking_lot::RwLock;
use wgpu::wgt::DeviceDescriptor;
use crate::{ColdConfig, Result};
use workarounds::Workarounds;

/// The wgpu instance is not a process-wide immutable singleton any more: it is
/// replaceable, because the user can select a different GPU and that requires a
/// fresh adapter/device/queue triple. Reads are far more frequent than writes
/// (every batch touches the device), so an `RwLock` — not a `Mutex` — is the
/// right primitive; `parking_lot` keeps the read side cheap.
static INSTANCE: RwLock<Option<Arc<WgpuInstance>>> = RwLock::new(None);

/// Singleton wgpu instance + picked device + queue, plus the "cold" config
/// that produced them.
///
/// `WgpuInstance` owns `wgpu::Instance`, `wgpu::Adapter`, `wgpu::Device`,
/// `wgpu::Queue` and the [`ColdConfig`] used to create them. Rebuilding the
/// instance replaces all of these at once; callers that need the device should
/// therefore re-fetch [`instance()`](Self::instance) instead of caching it.
pub struct WgpuInstance {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pub workarounds: workarounds::Workarounds,
    /// The cold config this instance was built from. Upload path and
    /// pipeline-cache policy are read from here, so they always match the
    /// device they run on.
    pub cold: ColdConfig,
}

impl WgpuInstance {
    /// Get the current instance, initialising it on first use.
    ///
    /// On failure the previous state is left untouched and `Err` is returned;
    /// the next call retries. The write lock is only taken when there is no
    /// instance yet, so the steady state is a single read-lock acquire.
    pub fn instance() -> Result<Arc<Self>> {
        if let Some(inst) = INSTANCE.read().as_ref() {
            return Ok(Arc::clone(inst));
        }
        Self::instance_with(ColdConfig::default())
    }

    /// Like [`instance()`](Self::instance) but with explicit cold settings.
    ///
    /// If an instance already exists it is returned unchanged: cold settings
    /// only take effect through [`rebuild`](Self::rebuild), so a stale config
    /// cannot half-apply here.
    pub fn instance_with(cold: ColdConfig) -> Result<Arc<Self>> {
        if let Some(inst) = INSTANCE.read().as_ref() {
            return Ok(Arc::clone(inst));
        }
        // Slow path: create outside the write lock so concurrent readers are
        // not blocked by adapter enumeration. A racing thread may create a
        // second instance; the loser's `Arc` is simply dropped.
        let inst = Arc::new(Self::init(cold)?);
        let mut guard = INSTANCE.write();
        if let Some(existing) = guard.as_ref() {
            return Ok(Arc::clone(existing));
        }
        *guard = Some(Arc::clone(&inst));
        Ok(inst)
    }

    /// Create (or replace) the instance using `cold`.
    ///
    /// Called at startup and on save reload, when `gpu_index` /
    /// `disable_pipeline_cache` / `mapped_at_creation` may have changed. The
    /// caller is responsible for tearing down dependents first (executor,
    /// world handles); this function only swaps the GPU objects.
    pub fn rebuild(cold: ColdConfig) -> Result<Arc<Self>> {
        let inst = Arc::new(Self::init(cold)?);
        *INSTANCE.write() = Some(Arc::clone(&inst));
        Ok(inst)
    }

    /// Drop the current instance, if any, releasing the device and queue.
    ///
    /// Any `Arc<WgpuInstance>` handed out earlier keeps the old device alive
    /// until it is dropped; this only clears the slot so the next
    /// [`instance()`](Self::instance) builds a fresh one.
    pub fn shutdown() {
        *INSTANCE.write() = None;
    }

    fn init(cold: ColdConfig) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY.with_env(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = adapter_pick::pick_best(&instance, cold.gpu_index, cold.disable_pipeline_cache)?;

        features::check_required(&adapter, cold.disable_pipeline_cache)?;
        let adapter_info = adapter.get_info();
        let workarounds = Workarounds::detect(&adapter_info);
        if workarounds.any() {
            crate::wgpu_debug!("[c2me-wgpu-rs] enabled workarounds for {} ({}): {:?}", adapter_info.name, adapter_info.driver, workarounds);
        }

        let (device, queue) = pollster::block_on(
        adapter.request_device(&DeviceDescriptor{
            required_features: if cold.disable_pipeline_cache {
                wgpu::Features::empty()
            } else {
                wgpu::Features::PIPELINE_CACHE
            },
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            label: None,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))?;

        // Surface wgpu validation/OOM errors through our log instead of letting
        // the default handler panic (which, across the FFM boundary, aborts
        // the whole process with no message — exactly what we saw on Mali).
        device.on_uncaptured_error(std::sync::Arc::new(|err| {
            let msg = format!("[c2me-wgpu-rs][wgpu-error] {err}");
            eprintln!("{msg}");
            crate::debug_log_line(&msg);
        }));
        crate::wgpu_debug!("[c2me-wgpu-rs] device created, uncaptured-error handler installed");

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            adapter_info,
            workarounds,
            cold,
        })
    }
}
