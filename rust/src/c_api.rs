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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, express OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

//! 5 `pub extern "C"` entry points — the only surface area of this shared library.
//!
//! Contract:
//!   • Returns i64 handle → 0 on failure, error NUL-terminated string in err_buf
//!   • Returns i32 status → 0 PENDING / 1 DONE / 2 ERROR / -1 INVALID_HANDLE
//!   • All pointer pairs are (ptr, len). No bare un-sized pointers allowed.

use std::ffi::{c_char, CString};
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
    PENDING_QUEUE.get_or_init(|| Mutex::new(PendingQueue::default()))
}

fn get_or_create_executor() -> Result<&'static Mutex<ExecutorThread>, String> {
    use std::sync::OnceLock;
    static EXECUTOR_LOCK: OnceLock<Mutex<ExecutorThread>> = OnceLock::new();

    // Check if already initialized
    if let Some(lock) = EXECUTOR_LOCK.get() {
        return Ok(lock);
    }

    // Try to initialize
    let instance = WgpuInstance::instance().map_err(|e| format!("WgpuInstance not initialised: {e}"))?;
    let device = Arc::new(instance.device.clone());
    let queue = Arc::new(instance.queue.clone());
    let config = ExecutorConfig::default();

    let executor = ExecutorThread::spawn(device, queue, config)
        .map_err(|e| format!("executor spawn failed: {e}"))?;

    match EXECUTOR_LOCK.set(Mutex::new(executor)) {
        Ok(()) => {}
        Err(_) => {
            // Race: another thread initialized it, just return the existing one
        }
    }

    Ok(EXECUTOR_LOCK.get().unwrap())
}

// ---------------------------------------------------------------------------
// ① Create singleton WgpuInstance (adapter pick + feature check).
///
/// # Safety
/// `err_buf` must be null or point to a mutable buffer of `err_buf_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn c2me_wgpu_create_instance(
    _features_mask: u64,
    err_buf: *mut c_char,
    err_buf_len: usize,
) -> i64 {
    match WgpuInstance::instance() {
        Ok(_) => 1,
        Err(e) => {
            let msg = format!("{}", e);
            unsafe { write_err(err_buf, err_buf_len, &msg) };
            0
        }
    }
}

// ---------------------------------------------------------------------------
// ② Create a WorldContext from a serialized NoiseRouter blob.
///
/// # Safety
/// `config_blob_ptr` must be valid for `config_blob_len` contiguous bytes
/// (or both null/0 for empty blob in this skeleton).
/// `err_buf` must be null or valid mutable buffer of `err_buf_len` bytes.
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
    let mut table = get_handle_table().lock().unwrap();
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
    if world_h == 0 {
        unsafe { write_err(err_buf, err_buf_len, "invalid world handle (null)") };
        return 0;
    }

    // Retrieve the Arc<WorldContext> from the handle table.
    let world_arc = {
        let table = get_handle_table().lock().unwrap();
        match table.get(&world_h) {
            Some(arc) => arc.clone(),
            None => {
                unsafe { write_err(err_buf, err_buf_len, "invalid world handle (not found in table)") };
                return 0;
            }
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
            result: Some(ResultBlob::new_empty()),
        });
        return Box::into_raw(req) as i64;
    }

    // 3. Dispatch the batch through the executor thread.
    let executor = match get_or_create_executor() {
        Ok(e) => e,
        Err(e) => {
            unsafe { write_err(err_buf, err_buf_len, &e) };
            return 0;
        }
    };

    let executor_result = executor.lock()
        .map_err(|e| {
            unsafe { write_err(err_buf, err_buf_len, &format!("executor lock: {e}")) };
            return 0;
        })
        .unwrap()
        .request_chunks(world_arc, batch);

    match executor_result {
        Ok(executed) => {
            // Merge all per-chunk results into one combined blob
            let mut all_f32s: Vec<f32> = Vec::new();
            const CELLS_PER_CHANNEL: u64 = 5 * 97 * 5;
            let mut combined_offsets = [0u64; 16];
            for (ch, off) in combined_offsets.iter_mut().enumerate() {
                *off = (ch as u64) * CELLS_PER_CHANNEL * 4;
            }
            let mut chunk_idx = 0u16;

            // Emit chunks in the SAME order as batch.positions (sorted,
            // deduped). The previous FxHashMap iteration order was
            // non-deterministic, so multi-chunk blobs could not be mapped
            // back to chunk coords by the caller. Batching depends on this.
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

            let req = Box::new(BatchRequest {
                status: AtomicI32::new(STATUS_DONE),
                result: Some(result_blob),
            });
            Box::into_raw(req) as i64
        }
        Err(e) => {
            let msg = format!("executor batch failed: {e}");
            unsafe { write_err(err_buf, err_buf_len, &msg) };
            let req = Box::new(BatchRequest {
                status: AtomicI32::new(STATUS_ERROR),
                result: Some(ResultBlob::new_empty()),
            });
            Box::into_raw(req) as i64
        }
    }
}

// ---------------------------------------------------------------------------
// ④ Poll a BatchRequest for completion status.
#[unsafe(no_mangle)]
pub extern "C" fn c2me_wgpu_poll_status(request_h: i64) -> i32 {
    if request_h == 0 {
        return STATUS_INVALID_HANDLE;
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
    let mut req: Box<BatchRequest> = unsafe { Box::from_raw(request_h as *mut BatchRequest) };
    let status = req.status.load(Ordering::Acquire);
    if status != STATUS_DONE {
        return STATUS_ERROR;
    }
    let Some(blob) = req.result.take() else {
        return -1;
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
    if world_h == 0 {
        return STATUS_INVALID_HANDLE;
    }
    let world_arc = {
        let table = get_handle_table().lock().unwrap();
        match table.get(&world_h) {
            Some(arc) => arc.clone(),
            None => return STATUS_INVALID_HANDLE,
        }
    };
    let data: Vec<f32> = if data_ptr.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(data_ptr, data_len) }.to_vec()
    };
    world_arc.set_external_inputs(count.max(0) as usize, data);
    0
}

// ---------------------------------------------------------------------------
// ⑧ Set the Rust-side log file path (called by Java at startup).
//
/// # Safety
/// `ptr` must point to `len` UTF-8 bytes (or be null with len 0).
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
// Extra helper: GPU name.
/// Returns a statically-allocated UTF-8 C string. Caller must NOT free.
#[unsafe(no_mangle)]
pub extern "C" fn c2me_wgpu_get_gpu_name() -> *const c_char {
    let name = match WgpuInstance::instance() {
        Ok(inst) => inst.adapter_info.name.clone(),
        Err(_) => "unknown".to_string(),
    };
    // 使用 CString 并泄漏，返回指向静态内存的指针
    let c_str = CString::new(name).unwrap_or_else(|_| CString::new("unknown").unwrap());
    let ptr = c_str.as_ptr();
    // 泄漏 CString 防止释放（整个程序生命周期内有效）
    std::mem::forget(c_str);
    ptr
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

/// A BatchRequest tracks one GPU batch.
pub(crate) struct BatchRequest {
    pub status: AtomicI32,
    pub result: Option<ResultBlob>,
}
