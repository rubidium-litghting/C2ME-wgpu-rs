/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2021-2026 RubidiumLighting
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
