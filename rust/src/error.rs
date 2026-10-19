// SPDX-License-Identifier: MIT

pub use anyhow::Error as C2meError;
pub use anyhow::Result;

/// Write a C2meError as a C string into the provided err_buf.
/// Always NUL-terminates (even on truncation). Returns bytes written
/// (excluding NUL), capped by err_buf_len.saturating_sub(1).
///
/// # Safety
/// err_buf must be a valid mutable pointer to a buffer of at least err_buf_len bytes,
/// or null (in which case nothing is written).
pub unsafe fn write_err(err_buf: *mut std::ffi::c_char, err_buf_len: usize, msg: &str) {
    if err_buf.is_null() || err_buf_len == 0 {
        return;
    }
    let bytes = msg.as_bytes();
    let copy_len = bytes.len().min(err_buf_len.saturating_sub(1));
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr() as *const std::ffi::c_char,
            err_buf,
            copy_len,
        );
        *err_buf.add(copy_len) = 0;
    }
}
