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
//! 0x08    128   offset[16]    per-channel byte offsets from start of payload
//! 0x88    ...   payload (MC-native-stride f32 arrays written by mc_stride.rs)
//! end-4   4     CRC32 (over entire blob except these 4 tail bytes)

/// Magic header: "WG00" in little-endian (b'W' b'G' b'0' b'0').
/// Matches Java-side HEADER sentinel value.
pub const MAGIC: u32 = u32::from_le_bytes(*b"WG00");
pub const VERSION: u16 = 1;

/// Opaque result blob. Contains MC-native-stride f32 density data +
/// per-channel offsets + CRC32 trailer.
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
        let crc = crate::result::crc32::crc32(&v);
        v.extend_from_slice(&crc.to_le_bytes());
        Self { data: v }
    }

    /// Build a result blob from f32 density data + per-channel byte offsets.
    ///
    /// Layout (matching Java-side expectations):
    ///   0x00: magic (4B) = "WG00"
    ///   0x04: version (2B) = 1
    ///   0x06: chunk_count (2B) = N
    ///   0x08: offsets[16] (128B) — byte offset of each channel from payload start
    ///   0x88: payload — all f32 density data (MC-native stride)
    ///   end-4: CRC32 (4B) — over everything before this
    pub fn from_f32_data(f32_data: &[f32], channel_offsets: [u64; 16], chunk_count: u16) -> Self {
        let mut v = Vec::<u8>::new();

        // 1. Header: magic (4B) + version (2B) + chunk_count (2B)
        v.extend_from_slice(&MAGIC.to_le_bytes());
        v.extend_from_slice(&VERSION.to_le_bytes());
        v.extend_from_slice(&chunk_count.to_le_bytes());

        // 2. Offsets[16] (128B) — per-channel byte offsets from payload start (0x88).
        //    The offsets are relative to the payload start, not the blob start.
        //    Java side adds 0x88 to get absolute offset.
        for offset in &channel_offsets {
            v.extend_from_slice(offset.to_le_bytes().as_slice());
        }

        // 3. Payload — all f32 data, MC-native stride.
        v.extend_from_slice(bytemuck::cast_slice(f32_data));

        // 4. CRC32 trailer (over everything before this 4B).
        let crc = crate::result::crc32::crc32(&v);
        v.extend_from_slice(&crc.to_le_bytes());

        Self { data: v }
    }

    /// Returns (data_ptr, data_len). Leaks the backing Vec; caller must
    /// arrange to reclaim it later (TODO: real ownership via Arc+Drop).
    pub fn into_raw_parts(self) -> (*const u8, usize) {
        let len = self.data.len();
        let ptr = self.data.as_ptr();
        std::mem::forget(self.data);
        (ptr, len)
    }

    /// Verify the 4-byte CRC32 trailer matches the rest of the blob.
    pub fn verify_crc(&self) -> bool {
        if self.data.len() < 4 { return false; }
        let (body, tail) = self.data.split_at(self.data.len() - 4);
        let expect = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]);
        crate::result::crc32::crc32(body) == expect
    }

    /// Get the chunk_count from the header.
    pub fn chunk_count(&self) -> u16 {
        if self.data.len() < 8 { return 0; }
        u16::from_le_bytes([self.data[6], self.data[7]])
    }

    /// Get the byte offset for a given channel index (0..15).
    pub fn channel_offset(&self, channel: usize) -> u64 {
        if self.data.len() < 8 + channel * 8 + 8 { return 0; }
        let start = 8 + channel * 8;
        let bytes: [u8; 8] = self.data[start..start+8].try_into().unwrap_or([0u8; 8]);
        u64::from_le_bytes(bytes)
    }

    /// Get a reference to the raw data (for testing).
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_blob_has_valid_crc() {
        let blob = ResultBlob::new_empty();
        assert!(blob.verify_crc());
        assert_eq!(blob.chunk_count(), 0);
    }

    #[test]
    fn from_f32_data_has_valid_crc() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let offsets: [u64; 16] = [0u64; 16];
        let blob = ResultBlob::from_f32_data(&data, offsets, 1);
        assert!(blob.verify_crc());
        assert_eq!(blob.chunk_count(), 1);
    }

    #[test]
    fn channel_offsets_roundtrip() {
        let data = vec![0.5f32, 1.5, 2.5, 3.5];
        let mut offsets: [u64; 16] = [0u64; 16];
        offsets[0] = 0;
        offsets[1] = 16; // 4 f32s × 4 bytes
        offsets[2] = 32;
        offsets[3] = 48;

        let blob = ResultBlob::from_f32_data(&data, offsets, 1);
        assert!(blob.verify_crc());

        for ch in 0..16 {
            assert_eq!(blob.channel_offset(ch), offsets[ch]);
        }
    }

    #[test]
    fn blob_magic_and_version() {
        let blob = ResultBlob::new_empty();
        let bytes = blob.as_bytes();
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(magic, MAGIC);
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        assert_eq!(version, VERSION);
    }
}
