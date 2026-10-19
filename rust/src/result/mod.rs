// SPDX-License-Identifier: MIT

//! Result format module — defines the blob layout returned from
//! `c2me_wgpu_take_result_blob`.
//!
//! Key design:
//!   • Layout matches MC cellX/cellY/cellZ stride exactly.
//!   • Java side does not have to reorder anything; 2 lines of code:
//!         buf.asFloatBuffer().get(rawF32);
//!         for (i..) dstDouble[i] = (double) rawF32[i];
//!   • CRC32 tail check guards against GPU memory corruption.

pub mod format;
pub mod mc_stride;
pub mod crc32;

pub use format::ResultBlob;
