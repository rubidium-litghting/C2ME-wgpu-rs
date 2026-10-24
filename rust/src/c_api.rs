// SPDX-License-Identifier: MIT

//! 5 `pub extern "C"` entry points — the only surface area of this shared library.
//!
//! Contract:
//!   • Returns i64 handle → 0 on failure, error NUL-terminated string in err_buf
//!   • Returns i32 status → 0 PENDING / 1 DONE / 2 ERROR / -1 INVALID_HANDLE
//!   • All pointer pairs are (ptr, len). No bare un-sized pointers allowed.

use std::ffi::c_char;
use std::ptr;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::collections::HashMap;

use crate::error::write_err;
use crate::instance::WgpuInstance;
use crate::result::ResultBlob;
use crate::scheduler::batcher::{Batch, ChunkPos, Priority};
use crate::scheduler::executor::{ExecutorThread, ExecutorConfig};
use crate::scheduler::PendingQueue;
use crate::world::WorldContext;

/// Status constants (must match Java side WgpuThinBinding).
pub const STATUS_PENDING: i32 = 0;
pub const STATUS_DONE: i32 = 1;
pub const STATUS_ERROR: i32 = 2;
pub const STATUS_INVALID_HANDLE: i32 = -1;

/// Global pending queue — accumulates chunk requests before flushing.
static PENDING_QUEUE: OnceLock<Mutex<PendingQueue>> = OnceLock::new();

/// Global handle table: maps i64 handle → Arc<WorldContext>.
static HANDLE_TABLE: OnceLock<Mutex<HashMap<i64, Arc<WorldContext>>>> = OnceLock::new();

fn get_handle_table() -> &'static Mutex<HashMap<i64, Arc<WorldContext>>> {
    HANDLE_TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_pending_queue() -> &'static Mutex<PendingQueue> {
    PENDING_QUEUE.get_or_init(|| {
        // Size the queue to hold one full request. `request_chunks` pushes
        // ALL positions of a request before flushing, so the hard cap must be
        // >= the executor's max batch size, or a large batch overflows.
        // (Previously 16*2 = 32 < 64 => overflow once batches grew.)
        let cap = ExecutorConfig::default().max_batch_size;
        Mutex::new(PendingQueue::with_capacity(16, 50, cap))
    })
}

/// The executor owns a single worker thread and hands work to it over a
/// channel, so it needs no lock of its own: `submit_async` takes `&self` and
/// the channel serialises everything. The previous `Mutex<ExecutorThread>`
/// made every batch contend for a lock whose critical section was one send.
/// The executor owns the device/queue it was spawned with, so it must be
/// rebuilt whenever the wgpu instance is replaced. `RwLock<Option<..>>` rather
/// than `OnceLock` gives us that reset; the steady state is one read acquire.
static EXECUTOR: parking_lot::RwLock<Option<Arc<ExecutorThread>>> =
    parking_lot::RwLock::new(None);

fn get_or_create_executor() -> Result<Arc<ExecutorThread>, String> {
    if let Some(e) = EXECUTOR.read().as_ref() {
        return Ok(Arc::clone(e));
    }

    let instance = WgpuInstance::instance().map_err(|e| format!("WgpuInstance not initialised: {e}"))?;
    let device = Arc::new(instance.device.clone());
    let queue = Arc::new(instance.queue.clone());
    let config = ExecutorConfig::default();

    let executor = Arc::new(
        ExecutorThread::spawn(device, queue, config)
            .map_err(|e| format!("executor spawn failed: {e}"))?,
    );

    let mut guard = EXECUTOR.write();
    if let Some(existing) = guard.as_ref() {
        return Ok(Arc::clone(existing));
    }
    *guard = Some(Arc::clone(&executor));
    Ok(executor)
}

/// Stop the executor and wait for its thread to exit.
///
/// Must run before the device is replaced: the worker holds `Arc<Device>` /
/// `Arc<Queue>` clones, and its `shutdown()` drains in-flight batches so no
/// request handle is left dangling. Idempotent.
fn shutdown_executor() {
    let taken = EXECUTOR.write().take();
    if let Some(e) = taken {
        // Stop the worker unconditionally. Relying on `Drop` (via
        // `Arc::get_mut`) does not work here: a batch in flight holds its own
        // `Arc<ExecutorThread>`, so the last reference only drops when that
        // batch finishes — possibly long after the caller expects the device
        // to be released. The worker holds an `Arc<Device>`, so leaving it
        // running keeps the entire old device and its buffer pool alive.
        e.shutdown();
    }
}

// ---------------------------------------------------------------------------
// ① Create singleton WgpuInstance (adapter pick + feature check).
///
/// # Safety
/// `err_buf` must be null or point to a mutable buffer of `err_buf_len` bytes.
/// Resolve a world handle to its `Arc<WorldContext>`.
///
/// Every `world_h`-taking entry point needs this, and each one previously
/// inlined `lock().unwrap()` -- a poisoned mutex would panic across the FFI
/// boundary, which `DEVELOPMENT.md` #24 warns against. Returning `Err`
/// instead lets callers report a status code.
fn resolve_world(world_h: i64) -> Result<Arc<WorldContext>, &'static str> {
    if world_h == 0 {
        return Err("invalid world handle (null)");
    }
    let table = get_handle_table()
        .lock()
        .map_err(|_| "world handle table mutex poisoned")?;
    table
        .get(&world_h)
        .cloned()
        .ok_or("invalid world handle (not found in table)")
}

/// Borrow an FFI `(ptr, len)` byte buffer as a slice, mapping null/0 to empty.
///
/// # Safety
/// `ptr` must be valid for `len` reads when non-null.
unsafe fn bytes_from_ffi<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

/// Install the panic hook that flushes the log before aborting.
///
/// # Safety
/// No memory contract. May be called from any thread; installing the hook
/// twice is a harmless overwrite. Declared `unsafe` only for FFI symmetry.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn c2me_wgpu_setup_panic_hook() {
    crate::setup_panic_hook();
}

/// Push the hot-reloadable flags (debug / profiling) to Rust.
///
/// These are read on hot paths (every batch, every `wgpu_debug!`), so they are
/// plain atomics: callable at any time, no restart required. The cold settings
/// (`gpu_index`, `disable_pipeline_cache`, `mapped_at_creation`) travel through
/// [`c2me_wgpu_create_instance`] instead, because they only take effect when
/// the device is rebuilt.
#[unsafe(no_mangle)]
pub extern "C" fn c2me_wgpu_set_hot_flags(debug: u32, profile: u32) -> i32 {
    crate::set_hot_flags(debug != 0, profile != 0);
    0
}

/// Map the FFM `i32` encoding of `mapped_at_creation` to the Rust option:
/// -1 = platform default, 0 = force write_buffer, 1 = force mapped.
fn decode_mapped_at_creation(v: i32) -> Option<bool> {
    match v {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    }
}

/// Create the wgpu instance (adapter pick + feature check + device).
///
/// `gpu_index` is the adapter position from [`c2me_wgpu_list_gpus`], or -1 for
/// automatic selection. `disable_pipeline_cache` and `mapped_at_creation` are
/// the other cold settings: they belong to the device being created, so they
/// are passed here rather than through a separate config call.
///
/// Returns 1 on success, 0 on failure. Calling this when an instance already
/// exists is a no-op that returns 1: use [`c2me_wgpu_rebuild_instance`] to
/// apply new cold settings.
///
/// # Safety
/// `err_buf` must be valid for writes of `err_buf_len` bytes (or null with
/// `err_buf_len == 0`). It is only touched on the failure path.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn c2me_wgpu_create_instance(
    gpu_index: i32,
    backend: i32,
    disable_pipeline_cache: u32,
    mapped_at_creation: i32,
    err_buf: *mut c_char,
    err_buf_len: usize,
) -> i64 {
    let cold = crate::ColdConfig {
        gpu_index,
        disable_pipeline_cache: disable_pipeline_cache != 0,
        mapped_at_creation: decode_mapped_at_creation(mapped_at_creation),
        backend,
    };
    match std::panic::catch_unwind(|| match WgpuInstance::instance_with(cold) {
        Ok(_) => 1,
        Err(e) => {
            let msg = format!("{e}");
            unsafe { write_err(err_buf, err_buf_len, &msg) };
            0
        }
    }) {
        Ok(res) => res,
        Err(e) => {
            let msg = format!("Rust panicked during GPU instance creation (e.g. EGL BadAccess): {e:?}");
            crate::debug_log_line(&msg);
            unsafe { write_err(err_buf, err_buf_len, "Rust panicked during GPU instance creation") };
            0
        }
    }
}

/// Replace the wgpu instance using new cold settings.
///
/// The caller MUST have torn down dependents first: all world handles
/// destroyed and the executor stopped (`c2me_wgpu_shutdown_instance` does the
/// latter for you). This only swaps the GPU objects, so any world or pipeline
/// object still alive is bound to a dead device.
///
/// Returns 1 on success, 0 on failure. On failure the previous instance is
/// left in place, so the mod keeps working with the old GPU rather than dying.
///
/// # Safety
/// `err_buf` must be valid for writes of `err_buf_len` bytes (or null with
/// `err_buf_len == 0`).
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn c2me_wgpu_rebuild_instance(
    gpu_index: i32,
    backend: i32,
    disable_pipeline_cache: u32,
    mapped_at_creation: i32,
    err_buf: *mut c_char,
    err_buf_len: usize,
) -> i64 {
    let cold = crate::ColdConfig {
        gpu_index,
        disable_pipeline_cache: disable_pipeline_cache != 0,
        mapped_at_creation: decode_mapped_at_creation(mapped_at_creation),
        backend,
    };
    // The worker holds device/queue clones; stop it before the device dies.
    shutdown_executor();
    match std::panic::catch_unwind(|| match WgpuInstance::rebuild(cold) {
        Ok(_) => 1,
        Err(e) => {
            let msg = format!("{e}");
            unsafe { write_err(err_buf, err_buf_len, &msg) };
            0
        }
    }) {
        Ok(res) => res,
        Err(e) => {
            let msg = format!("Rust panicked during GPU instance rebuild: {e:?}");
            crate::debug_log_line(&msg);
            unsafe { write_err(err_buf, err_buf_len, "Rust panicked during GPU instance rebuild") };
            0
        }
    }
}

/// Tear down the executor and the wgpu instance.
///
/// Called on save switch / shutdown, before re-reading the config. World
/// handles must already be destroyed. Idempotent.
#[unsafe(no_mangle)]
pub extern "C" fn c2me_wgpu_shutdown_instance() -> i32 {
    shutdown_executor();
    WgpuInstance::shutdown();
    0
}

/// Write a JSON array describing every GPU the backend can see.
///
/// Shape (one object per adapter, order == `gpu_index`):
/// `[{"index":0,"name":"...","type":"IntegratedGpu","backend":"Vulkan"}, ...]`
///
/// When an instance already exists, its adapter is marked `"selected":true`,
/// so the Java side can show the effective choice even in automatic mode
/// without a second call.
///
/// Returns the number of bytes written (excluding NUL) on success, or a
/// negative value on error: -1 bad arguments, -2 buffer too small (the needed
/// size is then written to `out_len`). `buf` is always NUL-terminated when the
/// return is >= 0.
///
/// # Safety
/// `buf` must be valid for writes of `buf_len` bytes; `out_len` must be valid
/// for one write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn c2me_wgpu_list_gpus(
    backend: i32,
    buf: *mut c_char,
    buf_len: usize,
    out_len: *mut usize,
) -> i64 {
    if buf.is_null() || out_len.is_null() || buf_len == 0 {
        return -1;
    }
    // Enumerate exactly the backends the instance will be built with, not
    // `all()`. `gpu_index` is a position in this listing, and `pick_best`
    // indexes the adapters of the *instance*, which only exposes its enabled
    // backends. Enumerating more here than there would shift every index and
    // make the selection point at the wrong adapter. It also keeps duplicate
    // entries (the same GPU exposed through both Vulkan and GL) out of the
    // list unless the user explicitly asked for GL.
    // If a device is already live, answer from its cached listing and never
    // touch a driver. Enumerating adapters creates a real backend instance;
    // doing that while another device is live froze the renderer on Android
    // (the GLES path builds an EGL context and competes with the running one).
    // The listing is also the only meaningful one: a backend change requires a
    // restart, so previewing a different backend's adapters is no longer a
    // scenario the GUI has to support.
    if let Some(inst) = WgpuInstance::peek() {
        return unsafe { write_json(&inst.adapter_listing, buf, buf_len, out_len) };
    }

    let backends = if backend < 0 {
        wgpu::Backends::PRIMARY
    } else {
        wgpu::Backends::from_bits(backend as u32).unwrap_or(wgpu::Backends::PRIMARY)
    };
    // Creating an instance can panic inside wgpu-hal: the GL backend unwraps
    // a failed EGL context creation (egl.rs `make_current().unwrap()`). Letting
    // that unwind across the FFI boundary aborts the JVM, so the whole
    // enumeration runs inside catch_unwind. On failure we report an empty list
    // — the UI then shows only "automatic", which is the safe choice.
    let enumerated = std::panic::catch_unwind(|| {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        crate::instance::adapter_pick::enumerate(&instance)
    });
    let adapters = match enumerated {
        Ok(a) => a,
        Err(e) => {
            let msg = format!(
                "[c2me-wgpu-rs] enumerating adapters for backend bits {backends:?} panicked: {e:?}");
            crate::debug_log_line(&msg);
            Vec::new()
        }
    };
    // `peek`, not `instance`: the listing is informational and must never
    // create a device as a side effect. Doing so previously masked a failed
    // create_instance — Java reported "no GPU", Rust had one, and the two
    // disagreed for the rest of the session.
    let selected = WgpuInstance::peek()
        .map(|i| (i.adapter_info.name.clone(), i.adapter_info.backend));
    let mut json = String::from("[");
    for (idx, a) in adapters.iter().enumerate() {
        let info = a.get_info();
        if idx > 0 {
            json.push(',');
        }
        let is_selected = matches!(&selected, Some((n, b)) if *n == info.name && *b == info.backend);
        json.push_str(&format!(
            "{{\"index\":{idx},\"name\":{},\"type\":\"{:?}\",\"backend\":\"{:?}\"{}}}",
            json_escape(&info.name),
            info.device_type,
            info.backend,
            if is_selected { ",\"selected\":true" } else { "" },
        ));
    }
    json.push(']');
    let bytes = json.as_bytes();
    let need = bytes.len() + 1; // + NUL
    unsafe { *out_len = need };
    if need > buf_len {
        return -2;
    }
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buf, bytes.len());
        *buf.add(bytes.len()) = 0;
    }
    bytes.len() as i64
}

/// Copy a JSON string into the caller's buffer, reporting the required size.
///
/// Returns the byte count excluding the NUL on success, -2 when the buffer is
/// too small (the needed size is still written to `out_len`).
///
/// # Safety
/// `buf` must be valid for `buf_len` writes; `out_len` for one write.
unsafe fn write_json(
    json: &str,
    buf: *mut c_char,
    buf_len: usize,
    out_len: *mut usize,
) -> i64 {
    let bytes = json.as_bytes();
    let need = bytes.len() + 1;
    unsafe { *out_len = need };
    if need > buf_len {
        return -2;
    }
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buf, bytes.len());
        *buf.add(bytes.len()) = 0;
    }
    bytes.len() as i64
}

/// Minimal JSON string escaping for adapter names (backslash, quote, and
/// control characters). Adapter names are vendor strings; they can contain
/// quotes in theory, and emitting raw ones would produce invalid JSON.
pub(crate) fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ---------------------------------------------------------------------------
// ② Create a WorldContext from a serialized NoiseRouter blob.
///
/// # Safety
/// `config_blob_ptr` must be valid for `config_blob_len` contiguous bytes
/// (or both null/0 for empty blob in this skeleton).
/// `err_buf` must be null or valid mutable buffer of `err_buf_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn c2me_wgpu_world_destroy(world_h: i64) -> i32 {
    if world_h == 0 { return STATUS_INVALID_HANDLE; }
    let mut table = match get_handle_table().lock() {
        Ok(t) => t,
        Err(_) => return STATUS_ERROR,
    };
    let existed = table.remove(&world_h).is_some();
    // Release the table lock before dropping the WorldContext: its Drop may
    // free GPU resources and must not run while the global table is held.
    drop(table);
    if !existed {
        return STATUS_INVALID_HANDLE;
    }
    // `world_new` created the handle with `Arc::into_raw(world_arc.clone())`,
    // so the raw pointer owns one strong reference. The table entry held the
    // other; removing it dropped that one. Reclaim the raw-pointer reference
    // here, otherwise every destroyed world leaks its WorldContext.
    //
    // SAFETY: `world_h` came from `Arc::into_raw` in `world_new` and is only
    // converted back here, exactly once (a second call finds no table entry
    // and returns above without touching the pointer).
    unsafe { drop(Arc::from_raw(world_h as *const WorldContext)) };
    0
}

/// # Safety
/// `config_blob_ptr` must be valid for `config_blob_len` reads (or null with
/// len 0). `err_buf` must be valid for writes of `err_buf_len` bytes. The
/// returned handle owns an `Arc<WorldContext>` and must be released with
/// [`c2me_wgpu_world_destroy`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn c2me_wgpu_world_new(
    instance_h: i64,
    config_blob_ptr: *const u8,
    config_blob_len: usize,
    err_buf: *mut c_char,
    err_buf_len: usize,
) -> i64 {
    if instance_h == 0 {
        unsafe { write_err(err_buf, err_buf_len, "invalid instance handle (null)") };
        return 0;
    }
    let Ok(instance) = WgpuInstance::instance() else {
        unsafe { write_err(err_buf, err_buf_len, "instance not initialized") };
        return 0;
    };

    let blob_slice: &[u8] = if config_blob_ptr.is_null() {
        if config_blob_len != 0 {
            unsafe { write_err(err_buf, err_buf_len, "config_blob_ptr is null but len > 0") };
            return 0;
        }
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(config_blob_ptr, config_blob_len) }
    };

    let mut ctx = match WorldContext::new(blob_slice) {
        Ok(c) => c,
        Err(e) => {
            let msg = format!("{}", e);
            unsafe { write_err(err_buf, err_buf_len, &msg) };
            return 0;
        }
    };

    if let Err(e) = ctx.compile_pipeline(&instance.device) {
        let msg = format!("pipeline compile failed: {e}");
        unsafe { write_err(err_buf, err_buf_len, &msg) };
        return 0;
    }

    // Store in global handle table as Arc<WorldContext>.
    let world_arc = Arc::new(ctx);
    let handle = Arc::into_raw(world_arc.clone()) as i64;
    let mut table = match get_handle_table().lock() {
        Ok(t) => t,
        Err(_) => {
            // Handle was already leaked via into_raw; reclaim it so we do not
            // leak the WorldContext on a poisoned-table path.
            unsafe { drop(Arc::from_raw(handle as *const WorldContext)) };
            unsafe { write_err(err_buf, err_buf_len, "world handle table mutex poisoned") };
            return 0;
        }
    };
    table.insert(handle, world_arc);
    handle
}

// ---------------------------------------------------------------------------
// ③ Submit a batch of chunk (cx, cz) positions for GPU computation.
///
/// Real implementation: pushes each chunk into the global PendingQueue,
/// flushes the queue into a Batch, then dispatches the Batch through
/// the executor thread. Returns a BatchRequest handle that can be
/// polled for completion status.
///
/// # Safety
/// `positions_ptr` must point to `position_count * 2` packed i32 values:
/// [cx0, cz0, cx1, cz1, ...].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn c2me_wgpu_request_chunks(
    world_h: i64,
    positions_ptr: *const i32,
    position_count: i32,
    err_buf: *mut c_char,
    err_buf_len: usize,
) -> i64 {
    let world_arc = match resolve_world(world_h) {
        Ok(arc) => arc,
        Err(msg) => {
            unsafe { write_err(err_buf, err_buf_len, msg) };
            return 0;
        }
    };

    let positions: &[i32] = if positions_ptr.is_null() {
        if position_count != 0 {
            unsafe { write_err(err_buf, err_buf_len, "positions_ptr null but count>0") };
            return 0;
        }
        &[]
    } else {
        let n = (position_count as usize).checked_mul(2).unwrap_or(0);
        unsafe { std::slice::from_raw_parts(positions_ptr, n) }
    };

    // 1. Push each chunk into the pending queue.
    let queue = get_pending_queue();
    let batch = {
        let mut queue_guard = match queue.lock() {
            Ok(g) => g,
            Err(e) => {
                let msg = format!("pending queue lock failed: {e}");
                unsafe { write_err(err_buf, err_buf_len, &msg) };
                return 0;
            }
        };

        for i in 0..position_count as usize {
            let cx = positions[i * 2];
            let cz = positions[i * 2 + 1];
            let pos = ChunkPos { cx, cz, priority: Priority::Normal };
            if let Err(e) = queue_guard.push(pos) {
                let msg = format!("pending queue push failed: {e}");
                unsafe { write_err(err_buf, err_buf_len, &msg) };
                return 0;
            }
        }

        // 2. Flush the queue into a batch.
        queue_guard.flush().unwrap_or_else(|| Batch {
            positions: Vec::new(),
            max_priority: Priority::Background,
        })
    };

    if batch.positions.is_empty() {
        let req = Box::new(BatchRequest {
            status: AtomicI32::new(STATUS_DONE),
            result: Mutex::new(Some(ResultBlob::new_empty())),
            block: Mutex::new(Some(ResultBlob::new_empty())),
        });
        return Box::into_raw(req) as i64;
    }

    // 3. Dispatch on a BACKGROUND thread; return a STATUS_PENDING handle
    //    immediately so the coordinator can pipeline (submit the next batch
    //    without waiting for this one). The worker fills result/status; Java
    //    polls STATUS, then take_result_blob. See DEVELOPMENT §18.6.
    let req = Arc::new(BatchRequest {
        status: AtomicI32::new(STATUS_PENDING),
        result: Mutex::new(None),
        block: Mutex::new(None),
    });
    // Shape-dependent per-chunk cell count, computed before the move.
    let cells_per_channel_units: u64 = world_arc.cells_per_chunk() as u64;

    let executor = match get_or_create_executor() {
        Ok(e) => e,
        Err(_e) => {
            req.status.store(STATUS_ERROR, Ordering::Release);
            return Arc::into_raw(req) as i64;
        }
    };

    let req_bg = Arc::clone(&req);
    let submitted = executor.submit_async(world_arc, batch, move |executor_result| {
        let executed = match executor_result {
            Ok(e) => e,
            Err(_e) => {
                // Cannot use err_buf from here; signal via status.
                req_bg.status.store(STATUS_ERROR, Ordering::Release);
                return;
            }
        };

        // Merge all per-chunk results into one combined blob.
        // Each per-chunk blob now carries only the 2 consumed channels
        // (finalDensity, veinToggle) — see executor.rs OUT_CHANNELS.
        const OUT_CHANNELS: u64 = 2;
        let mut all_f32s: Vec<f32> = Vec::new();
        let mut combined_offsets = [0u64; 16];
        for ch in 0..OUT_CHANNELS as usize {
            combined_offsets[ch] = (ch as u64) * cells_per_channel_units * 4;
        }
        let mut chunk_idx = 0u16;

        // Emit chunks in the SAME order as batch.positions (sorted, deduped).
        // The previous FxHashMap iteration order was non-deterministic, so
        // multi-chunk blobs could not be mapped back to chunk coords.
        for &(cx, cz) in &executed.batch.positions {
            let key = ((cx as i64) << 32) | (cz as i64 & 0xFFFFFFFF);
            let blob = match executed.results.get(&key) {
                Some(b) => b,
                None => continue,
            };
            let chunk_f32s = decode_blob_f32s(blob);
            if chunk_f32s.is_empty() { continue; }
            all_f32s.extend_from_slice(&chunk_f32s);
            chunk_idx += 1;
        }

        let result_blob = if all_f32s.is_empty() {
            ResultBlob::new_empty()
        } else {
            ResultBlob::from_f32_data(&all_f32s, combined_offsets, chunk_idx)
        };
        match req_bg.result.lock() {
            Ok(mut slot) => *slot = Some(result_blob),
            Err(_) => {
                req_bg.status.store(STATUS_ERROR, Ordering::Release);
                return;
            }
        }
        // §20 block-state bytes as a separate blob (raw u32, no CRC needed
        // by Java's reader — reuse ResultBlob only as a byte container).
        if !executed.block_data.is_empty() {
            let bytes: Vec<u8> = executed.block_data.iter()
                .flat_map(|v| v.to_le_bytes()).collect();
            match req_bg.block.lock() {
                Ok(mut slot) => *slot = Some(ResultBlob::from_bytes(bytes)),
                Err(_) => {
                    req_bg.status.store(STATUS_ERROR, Ordering::Release);
                    return;
                }
            }
        }
        req_bg.status.store(STATUS_DONE, Ordering::Release);
    });

    if submitted.is_err() {
        req.status.store(STATUS_ERROR, Ordering::Release);
    }

    Arc::into_raw(req) as i64
}

// ---------------------------------------------------------------------------
// ④ Poll a BatchRequest for completion status.
#[unsafe(no_mangle)]
pub extern "C" fn c2me_wgpu_poll_status(request_h: i64) -> i32 {
    if request_h == 0 {
        return STATUS_INVALID_HANDLE;
    }
    // This is the heartbeat that drives wgpu's mapping callbacks. PollType::
    // Poll is explicitly non-blocking and does nothing on its own; a mapping
    // started by map_async only resolves when something polls the device.
    // Java calls this every 50us-2ms while a batch is in flight, which is a
    // convenient cadence — no timer of our own is needed.
    if let Ok(inst) = WgpuInstance::instance() {
        let _ = inst.device.poll(wgpu::PollType::Poll);
    }
    let req = unsafe { &*(request_h as *const BatchRequest) };
    req.status.load(Ordering::Acquire)
}

// ---------------------------------------------------------------------------
// ⑤ Consume a completed BatchRequest and write out the (ptr, len) of the
//   MC-native-stride density result blob.
///
/// After this returns successfully, the handle is freed and must not be reused.
/// Caller reads blob via MemorySegment.reinterpret(len); blob is valid until
/// the caller drops the segment.
/// # Safety
/// `request_h` must be a handle from [`c2me_wgpu_request_chunks`] that has
/// reached `STATUS_DONE` and has not been passed here before: this call does
/// `Arc::from_raw` and frees the request. `out_ptr`/`out_len` must be valid
/// for one write each. The returned blob must be released with
/// [`c2me_wgpu_free_result_blob`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn c2me_wgpu_take_result_blob(
    request_h: i64,
    out_ptr: *mut *const u8,
    out_len: *mut usize,
) -> i32 {
    if request_h == 0 {
        return STATUS_INVALID_HANDLE;
    }
    if out_ptr.is_null() || out_len.is_null() {
        return STATUS_ERROR;
    }
    let req: Arc<BatchRequest> =
        unsafe { Arc::from_raw(request_h as *const BatchRequest) };
    let status = req.status.load(Ordering::Acquire);
    if status != STATUS_DONE {
        return STATUS_ERROR;
    }
    let blob = match req.result.lock() {
        Ok(mut slot) => match slot.take() {
            Some(b) => b,
            None => return -1,
        },
        Err(_) => return STATUS_ERROR,
    };
    if !blob.verify_crc() {
        return 2;
    }
    let (ptr, len) = blob.into_raw_parts();
    unsafe {
        ptr::write(out_ptr, ptr as *const u8);
        ptr::write(out_len, len);
    }
    0
}

// ---------------------------------------------------------------------------
// ⑤b Consume the §20 per-block state blob (u32 per block). Same lifetime
//     rules as take_result_blob.
/// # Safety
/// `request_h` must be a live handle from [`c2me_wgpu_request_chunks`],
/// **not yet** consumed by [`c2me_wgpu_take_result_blob`] (this borrows the
/// request; that call frees it). `out_ptr`/`out_len` must be valid for one
/// write each. The returned blob must be released with
/// [`c2me_wgpu_free_result_blob`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn c2me_wgpu_take_block_blob(
    request_h: i64,
    out_ptr: *mut *const u8,
    out_len: *mut usize,
) -> i32 {
    if request_h == 0 { return STATUS_INVALID_HANDLE; }
    if out_ptr.is_null() || out_len.is_null() { return STATUS_ERROR; }
    // BORROW (do not Arc::from_raw): the handle is freed exactly once, by
    // take_result_blob. Java must call take_block_blob FIRST, then
    // take_result_blob. Two from_raw on one handle would free twice -> UAF.
    let req: &BatchRequest = unsafe { &*(request_h as *const BatchRequest) };
    if req.status.load(Ordering::Acquire) != STATUS_DONE { return STATUS_ERROR; }
    let blob = match req.block.lock() {
        Ok(mut slot) => match slot.take() {
            Some(b) => b,
            None => return -1,
        },
        Err(_) => return STATUS_ERROR,
    };
    let (ptr, len) = blob.into_raw_parts();
    unsafe {
        ptr::write(out_ptr, ptr as *const u8);
        ptr::write(out_len, len);
    }
    0
}

// ---------------------------------------------------------------------------
// ⑥ Free a result blob handed out by c2me_wgpu_take_result_blob.
//
/// # Safety
/// `ptr`/`len` must come from exactly one `take_result_blob` call and must not
/// have been freed already.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn c2me_wgpu_free_result_blob(ptr: *mut u8, len: usize) {
    unsafe { ResultBlob::free_raw_parts(ptr, len) };
}

// ---------------------------------------------------------------------------
// ⑦ Set CPU-evaluated external-input values for the given world.
//
/// # Safety
/// `data_ptr` must point to `data_len` f32 values (or be null with len 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn c2me_wgpu_set_external_inputs(
    world_h: i64,
    count: i32,
    data_ptr: *const f32,
    data_len: usize,
) -> i32 {
    let world_arc = match resolve_world(world_h) {
        Ok(arc) => arc,
        Err(_) => return STATUS_INVALID_HANDLE,
    };
    let data: Vec<f32> = if data_ptr.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(data_ptr, data_len) }.to_vec()
    };
    world_arc.set_external_inputs(count.max(0) as usize, data);
    0
}

// ⑧ Set a per-chunk aquifer table (called by Java before a request).
//
/// `table_ptr` points to `table_len` bytes of the Java-serialized compact
/// table (magic + n + seedLo/Hi + per-cell lvl_y/state). Copied into the
/// WorldContext's map, keyed by (cx, cz), consumed at the next dispatch.
//
/// # Safety
/// `table_ptr` must point to `table_len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn c2me_wgpu_set_aquifer_table(
    world_h: i64,
    cx: i32,
    cz: i32,
    table_ptr: *const u8,
    table_len: usize,
) -> i32 {
    let world_arc = match resolve_world(world_h) {
        Ok(arc) => arc,
        Err(_) => return STATUS_INVALID_HANDLE,
    };
    let bytes: Vec<u8> = unsafe { bytes_from_ffi(table_ptr, table_len) }.to_vec();
    if let Ok(mut m) = world_arc.aquifer_tables.lock() {
        m.insert((cx, cz), bytes);
    }
    0
}

// ⑨ Set a per-chunk Beardifier table (pieces/junctions), packed u32 LE.
//
/// # Safety
/// `ptr` must point to `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn c2me_wgpu_set_beard_table(
    world_h: i64,
    cx: i32,
    cz: i32,
    ptr: *const u8,
    len: usize,
) -> i32 {
    let world_arc = match resolve_world(world_h) {
        Ok(arc) => arc,
        Err(_) => return STATUS_INVALID_HANDLE,
    };
    let bytes: Vec<u8> = unsafe { bytes_from_ffi(ptr, len) }.to_vec();
    if let Ok(mut m) = world_arc.beard_tables.lock() { m.insert((cx, cz), bytes); }
    0
}

// ⑩ Set per-dimension aquifer mode (called by Java after world_new).
// ---------------------------------------------------------------------------
// ⑩c Set the ore-vein RNG seed (called once per world from Java).
//
/// # Safety
/// `world_h` must be a live handle from `c2me_wgpu_world_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn c2me_wgpu_set_ore_seed(
    world_h: i64,
    seed_lo: i64,
    seed_hi: i64,
) -> i32 {
    let world_arc = match resolve_world(world_h) {
        Ok(arc) => arc,
        Err(_) => return STATUS_INVALID_HANDLE,
    };
    world_arc.set_ore_seed(seed_lo, seed_hi);
    0
}

/// `has_aquifers`: 0/1. `default_fluid`: 0=air / 2=water / 3=lava.
/// # Safety
/// No memory contract: all arguments are by-value. An invalid `world_h`
/// returns [`STATUS_INVALID_HANDLE`] instead of dereferencing anything.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn c2me_wgpu_set_aquifer_mode(
    world_h: i64,
    has_aquifers: i32,
    sea_level: i32,
    default_fluid: i32,
    lava_level: i32,
) -> i32 {
    let world_arc = match resolve_world(world_h) {
        Ok(arc) => arc,
        Err(_) => return STATUS_INVALID_HANDLE,
    };
    world_arc.set_aquifer_mode(crate::world::world::AquiferMode {
        has_aquifers: has_aquifers != 0,
        sea_level,
        default_fluid: default_fluid.max(0) as u32,
        lava_level,
    });
    0
}

// ---------------------------------------------------------------------------
// ⑪ Set the Rust-side log file path (called by Java at startup).
//
/// # Safety
/// `ptr` must point to `len` UTF-8 bytes (or be null with len 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn c2me_wgpu_set_log_callback(cb: crate::LogCallback) {
    crate::set_log_callback(cb);
}

/// # Safety
/// `ptr` must be valid for `len` reads (or null, which is a no-op). The bytes
/// are interpreted as UTF-8; invalid UTF-8 is silently ignored.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn c2me_wgpu_set_log_path(ptr: *const u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    if let Ok(s) = std::str::from_utf8(bytes) {
        crate::set_log_path(s);
    }
}

// ---------------------------------------------------------------------------
// Internal helpers

/// Extract f32 data from a ResultBlob (skip the header + offsets).
fn decode_blob_f32s(blob: &ResultBlob) -> Vec<f32> {
    let bytes = blob.as_bytes();
    if bytes.len() < 136 { // 8 header + 128 offsets
        return Vec::new();
    }
    let payload_start = 8 + 16 * 8; // after header + offsets
    let payload_end = bytes.len().saturating_sub(4); // before CRC
    if payload_start >= payload_end {
        return Vec::new();
    }
    let payload = &bytes[payload_start..payload_end];
    payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// A BatchRequest tracks one GPU batch. `status`/`result` are shared between
/// the C caller and a background worker thread (which fills them), so they
/// live behind an `Arc` and the result behind a `Mutex`. The request is
/// returned to Java in STATUS_PENDING and completed asynchronously.
pub(crate) struct BatchRequest {
    pub status: AtomicI32,
    pub result: Mutex<Option<ResultBlob>>,
    /// §20 per-block state bytes (u32 per block, chunk-major). Empty unless
    /// the BlockOut shard is enabled. Separate from `result` so the density
    /// blob layout is untouched.
    pub block: Mutex<Option<ResultBlob>>,
}
