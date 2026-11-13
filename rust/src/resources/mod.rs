// SPDX-License-Identifier: MIT

//! GPU buffer recycling.
//!
//! Staging buffers use a 3-phase fence chain so old data can't be read by new
//! dispatches (R07). Everything else is a fixed slot owned by the executor
//! (`InputBufferSet`), so a buffer is never handed from one purpose to another
//! just because the sizes happen to match.

pub mod staging_chain;
