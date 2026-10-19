// SPDX-License-Identifier: MIT

//! Memory pool module: recycle GPU buffers by size bucket instead of
//! reallocating per dispatch. Staging buffers use a 3-phase fence chain so
//! old data can't be read by new dispatches (R07).

pub mod buffer_pool;
pub mod staging_chain;
