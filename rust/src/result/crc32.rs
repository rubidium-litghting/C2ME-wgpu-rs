// SPDX-License-Identifier: MIT

//! Tiny CRC32 wrapper — used at tail of every result blob so Java side can
//! detect GPU wild writes / truncated transfers before reading density data.

pub fn crc32(bytes: &[u8]) -> u32 {
    crc32fast::hash(bytes)
}
