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

//! 5 `pub extern "C"` entry points — the only surface area of this shared library.
//!
//! Contract (see 计划.txt §3):
//!   • Returns i64 handle → 0 on failure, error NUL-terminated string in err_buf
//!   • Returns i32 status → 0 PENDING / 1 DONE / 2 ERROR / -1 INVALID_HANDLE
//!   • All pointer pairs are (ptr, len). No bare un-sized pointers allowed.

use std::ffi::{c_char, CString};
use std::ptr;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::OnceLock;

use crate::error::write_err;
use crate::instance::WgpuInstance;
use crate::result::ResultBlob;
use crate::world::WorldContext;

/// Status constants (must match Java side WgpuThinBinding).
pub const STATUS_PENDING: i32 = 0;
pub const STATUS_DONE: i32 = 1;
pub const STATUS_ERROR: i32 = 2;
pub const STATUS_INVALID_HANDLE: i32 = -1;

// ---------------------------------------------------------------------------
// Handle validity helpers: real impl stores handle in a slotmap / generational slab.
// Skeleton just uses raw Box::into_raw() pointers with null-checks.

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
    // instance_h 现在是哑参数，实际通过单例 API 取实例
    let Ok(_instance) = WgpuInstance::instance() else {
        unsafe { write_err(err_buf, err_buf_len, "instance not initialized") };
        return 0;
    };

    // Build a slice from the Java ByteBuffer MemorySegment.
    let blob_slice: &[u8] = if config_blob_ptr.is_null() {
        if config_blob_len != 0 {
            unsafe { write_err(err_buf, err_buf_len, "config_blob_ptr is null but len > 0") };
            return 0;
        }
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(config_blob_ptr, config_blob_len) }
    };

    match WorldContext::new(blob_slice) {
        Ok(ctx) => Box::into_raw(Box::new(ctx)) as i64,
        Err(e) => {
            let msg = format!("{}", e);
            unsafe { write_err(err_buf, err_buf_len, &msg) };
            0
        }
    }
}

// ---------------------------------------------------------------------------
// ③ Submit a batch of chunk (cx, cz) positions for GPU computation.
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
    let _world = unsafe { &*(world_h as *const WorldContext) };

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

    // Skeleton: immediately manufacture a fake "completed" BatchRequest with
    // an empty result blob. Real impl: push into scheduler batcher mpsc queue.
    let req = Box::new(BatchRequest {
        status: AtomicI32::new(STATUS_DONE),
        result: Some(ResultBlob::new_empty()),
    });
    let _ = positions; // silence unused in skeleton
    Box::into_raw(req) as i64
}

// ---------------------------------------------------------------------------
// ④ Poll a BatchRequest for completion status.
#[unsafe(no_mangle)]
pub extern "C" fn c2me_wgpu_poll_status(request_h: i64) -> i32 {
    if request_h == 0 {
        return STATUS_INVALID_HANDLE;
    }
    // Safety: caller guarantees handle came from request_chunks and not yet freed.
    let req = unsafe { &*(request_h as *const BatchRequest) };
    req.status.load(Ordering::Acquire)
}

// ---------------------------------------------------------------------------
// ⑤ Consume a completed BatchRequest and write out the (ptr, len) of the
//   MC-native-stride density result blob.
//
// After this returns successfully, the handle is freed and must not be reused.
// Caller reads blob via MemorySegment.reinterpret(len); blob is valid until
// the caller drops the segment (Rust keeps backing allocation behind the
// returned pointer via a static/leaked Arc; TODO real ownership model).
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
    // Take ownership of the Box.
    let mut req: Box<BatchRequest> = unsafe { Box::from_raw(request_h as *mut BatchRequest) };
    let status = req.status.load(Ordering::Acquire);
    if status != STATUS_DONE {
        // Leak nothing; re-box and drop. But since we just took ownership,
        // it's already consumed. Return ERROR status.
        return STATUS_ERROR;
    }
    let Some(blob) = req.result.take() else {
        return -1; // result already taken (double-take protection)
    };
    if !blob.verify_crc() {
        return 2; // blob corrupt
    }
    let (ptr, len) = blob.into_raw_parts();
    unsafe {
        ptr::write(out_ptr, ptr);
        ptr::write(out_len, len);
    }
    0
}

// ---------------------------------------------------------------------------
// Extra helper (not in §3 spec; used by Java ModuleEntryPoint to log GPU name).
/// Returns a statically-allocated UTF-8 C string. Caller must NOT free.
#[unsafe(no_mangle)]
pub extern "C" fn c2me_wgpu_get_gpu_name() -> *const c_char {
    static GPU_NAME: OnceLock<CString> = OnceLock::new();

    let Ok(inst) = WgpuInstance::instance() else {
        return b"unknown\0".as_ptr() as *const c_char;
    };
    let s = GPU_NAME.get_or_init(|| {
        CString::new(inst.adapter_info.name.as_str()).unwrap_or_default()
    });
    s.as_ptr()
}

// ---------------------------------------------------------------------------
// Internal: a BatchRequest tracks one GPU batch. Real impl lives in scheduler.

#[allow(dead_code)]
pub(crate) struct BatchRequest {
    pub status: AtomicI32,
    pub result: Option<ResultBlob>,
}
