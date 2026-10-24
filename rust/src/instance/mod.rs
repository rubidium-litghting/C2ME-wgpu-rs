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

/// Minimum `MemAvailable` (MiB) for keeping the old instance alive while a new
/// one is built.
///
/// Chosen conservatively: a 12 GB Android device was observed to hit lmkd with
/// both instances resident, while sitting around 3.2 GB available in normal
/// operation. 2 GiB keeps that device on the tight path and leaves desktops
/// (usually several GiB free) on the comfortable one.
const REBUILD_MEMORY_HEADROOM_MB: u64 = 2048;

/// `MemAvailable` from `/proc/meminfo`, in MiB.
///
/// `MemAvailable` rather than `MemFree`: the former accounts for reclaimable
/// page cache, which is what actually determines whether an allocation will
/// trigger the low-memory killer.
#[cfg(target_os = "linux")]
fn available_memory_mb() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb / 1024);
        }
    }
    None
}

/// Windows: `GlobalMemoryStatusEx`, whose `ullAvailPhys` is the closest
/// analogue of Linux's `MemAvailable` (free physical memory plus the standby
/// list, i.e. what can be handed out without paging).
#[cfg(target_os = "windows")]
fn available_memory_mb() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    // SAFETY: `status` is a fully initialised MEMORYSTATUSEX whose `dwLength`
    // the API checks to determine the caller's struct version.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return None;
    }
    Some(status.ullAvailPhys / (1024 * 1024))
}

/// macOS and the BSDs expose no file that reports this; the equivalent is the
/// Mach `host_statistics64` call (or parsing `vm_stat`), which is another API
/// surface entirely. Reporting "unknown" here means a rebuild on those
/// platforms always releases the old instance first — safe, but it gives up
/// the fallback. Worth implementing once macOS has been tested for real;
/// adding it blind would be unverifiable code.
#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn available_memory_mb() -> Option<u64> {
    None
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
    /// Read the current instance without creating one.
    ///
    /// Used where a caller only wants to annotate an existing result (e.g.
    /// marking which adapter is selected). Calling [`instance()`](Self::instance)
    /// there would silently build a device with default settings, which both
    /// hides the caller's own failure and leaves Rust holding a live device
    /// the Java side believes does not exist.
    pub fn peek() -> Option<Arc<Self>> {
        INSTANCE.read().as_ref().map(Arc::clone)
    }

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
    /// Replace the instance, choosing how to handle the overlap between old
    /// and new based on available memory.
    ///
    /// While the new device is being created the old one is still alive, so
    /// both sets of GPU resources exist at once. On desktop that is harmless;
    /// on a memory-tight Android device it can push the process over the
    /// limit, and on a Mali-G610 the failure was an OS-wide hang rather than a
    /// crash. `MemAvailable` therefore decides the order:
    ///
    /// * comfortable — build first, swap on success, keep the old device as a
    ///   fallback if the new one cannot be created;
    /// * tight — release the old device first so only one exists, and rebuild
    ///   with the default settings if the requested ones fail, because there
    ///   is nothing left to fall back to.
    ///
    /// If `MemAvailable` cannot be read the tight path is taken: an unreadable
    /// `/proc/meminfo` means an unusual environment, and the safe assumption
    /// there is "not much headroom".
    pub fn rebuild(cold: ColdConfig) -> Result<Arc<Self>> {
        let available = available_memory_mb();
        let tight = available.map_or(true, |mb| mb < REBUILD_MEMORY_HEADROOM_MB);

        // Logged unconditionally so the decision is observable on every
        // platform: the two paths differ in whether a failed rebuild can fall
        // back, and that is worth being able to confirm from a log.
        match available {
            Some(mb) => crate::debug_log_line(&format!(
                "[c2me-wgpu-rs] rebuild: {mb} MiB available (threshold {REBUILD_MEMORY_HEADROOM_MB}), {}",
                if tight { "releasing the old instance first" } else { "keeping the old instance for fallback" })),
            None => crate::debug_log_line(
                "[c2me-wgpu-rs] rebuild: available memory unknown, releasing the old instance first"),
        }

        if tight {
            Self::shutdown();
        }

        match Self::init(cold) {
            Ok(inst) => {
                let inst = Arc::new(inst);
                *INSTANCE.write() = Some(Arc::clone(&inst));
                Ok(inst)
            }
            Err(e) if tight => {
                // The old instance is already gone, so returning an error here
                // would leave the process without any GPU. Retry with the
                // automatic settings, which is the configuration most likely
                // to work on this machine.
                crate::debug_log_line(&format!(
                    "[c2me-wgpu-rs] rebuild failed ({e}); retrying with default settings"));
                let inst = Arc::new(Self::init(ColdConfig::default())?);
                *INSTANCE.write() = Some(Arc::clone(&inst));
                Ok(inst)
            }
            // Comfortable: the old instance is still installed and usable.
            Err(e) => Err(e),
        }
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
        // `WGPU_BACKEND` is deliberately not consulted: the config file is the
        // single source of truth, and it is the only one the GUI can write.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: cold.resolved_backends(),
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
