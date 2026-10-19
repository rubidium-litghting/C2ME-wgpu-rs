// SPDX-License-Identifier: MIT

//! Scheduler module: batch many small chunk requests into a small number of
//! large GPU dispatches, improving SM utilization from ~5% to ~50%+ (A07).

pub mod batcher;
pub mod priority;
pub mod executor;

pub use batcher::{Batch, ChunkPos, PendingQueue, Priority};
pub use executor::{ExecutedBatch, ExecutorThread};
