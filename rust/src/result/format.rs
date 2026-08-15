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

//! Blob header + tail layout:
//!
//! Offset  Size  Field
//! 0x00    4     magic = 0xC2ME_WGPU0  (little-endian u32, constant below)
//! 0x04    2     version = 1
//! 0x06    2     chunk_count   (N chunks in this batch)
//! 0x08    8     offset[16]    per-channel byte offsets from start of payload
//! 0x88    ...   payload (MC-native-stride f32 arrays written by mc_stride.rs)
//! end-4   4     CRC32 (over entire blob except these 4 tail bytes)

use crate::Result;

/// Magic header: "WG00" in little-endian (b'W' b'G' b'0' b'0').
/// Matches Java-side HEADER sentinel value.
pub const MAGIC: u32 = u32::from_le_bytes(*b"WG00");
pub const VERSION: u16 = 1;

/// Opaque result blob. In real impl this holds a Vec<u8> + a Box<dyn Any>
/// to keep the wgpu staging buffer mapping alive until the blob is freed.
pub struct ResultBlob {
    data: Vec<u8>,
}

impl ResultBlob {
    /// Empty placeholder blob for skeleton C API.
    pub fn new_empty() -> Self {
        let mut v = Vec::<u8>::with_capacity(256);
        v.extend_from_slice(&MAGIC.to_le_bytes());
        v.extend_from_slice(&VERSION.to_le_bytes());
        v.extend_from_slice(&0u16.to_le_bytes()); // chunk_count
        v.resize(v.len() + 16 * 8, 0u8);           // offsets[16]
        let crc = crc32::crc32(&v);
        v.extend_from_slice(&crc.to_le_bytes());
        Self { data: v }
    }

    /// Returns (data_ptr, data_len). Leaks the backing Vec; caller must
    /// arrange to reclaim it later (TODO: real ownership via Atomic+Drop).
    pub fn into_raw_parts(self) -> (*const u8, usize) {
        let len = self.data.len();
        let ptr = self.data.as_ptr();
        std::mem::forget(self);
        (ptr, len)
    }

    /// Verify the 4-byte CRC32 trailer matches the rest of the blob.
    pub fn verify_crc(&self) -> bool {
        if self.data.len() < 4 { return false; }
        let (body, tail) = self.data.split_at(self.data.len() - 4);
        let expect = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]);
        crc32::crc32(body) == expect
    }
}

// Import from sibling module.
use super::crc32;
