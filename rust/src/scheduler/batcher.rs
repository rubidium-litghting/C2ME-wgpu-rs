// SPDX-License-Identifier: MIT

//! PendingQueue: accumulates chunk requests until (N ≥ THRESHOLD || elapsed ≥ X ms).
//!
//! Strategy (R06 risk mitigation):
//!   • `crossbeam::bounded(64)` channel on the submit side;
//!   • when full, return QueueFull error → Java fallbacks THIS BATCH only,
//!     no global blocking.
//!
//! The batcher groups incoming (cx, cz) chunk requests into batches
//! that are multiples of 4×4×4 workgroups (16×16×16 cells = one MC
//! chunk). A batch is always a 2D grid of chunks.

use crate::Result;
use anyhow::anyhow;
use std::time::Instant;

/// A single chunk position request with a priority level.
#[derive(Debug, Clone, Copy)]
pub struct ChunkPos {
    pub cx: i32,
    pub cz: i32,
    pub priority: Priority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Player-visible area (highest).
    Critical = 0,
    /// Short-term generation (within render distance).
    Normal = 1,
    /// Background generation (lowest).
    Background = 2,
}

impl Default for Priority {
    fn default() -> Self {
        // Only used by `Batch::default()`, i.e. test fixtures and the
        // empty-request path; the queue always overwrites it with the real
        // minimum priority of the pending positions.
        Priority::Background
    }
}

/// CPU-evaluated external-input values travelling with one batch.
///
/// The shader reads binding 6 as `[chunk][slot][cell]`, so the payload's
/// lifetime must match the batch that needs it. It used to live in a
/// process-global `WorldContext` field instead, which forced external worlds
/// to submit serially: `submit_async` returns as soon as the batch is queued,
/// but the executor only reads the global later, during prepare — a second
/// batch would overwrite data the first had not consumed yet. Owning the
/// payload here removes that constraint (DEVELOPMENT §9).
#[derive(Debug, Clone, Default)]
pub struct ExternalInputs {
    /// Slots per chunk — the index space the serializer assigned to
    /// CPU-evaluated subtrees (`NoiseRouterBlobifier.EXTERNAL_NODES` size).
    pub per_chunk: u32,
    /// `[chunk][slot][cell]`, `per_chunk * cells_per_chunk` values per chunk.
    pub data: Vec<f32>,
}

/// A batch of chunk requests, ready for GPU dispatch.
#[derive(Debug, Clone, Default)]
pub struct Batch {
    /// Sorted list of (cx, cz) positions in this batch.
    pub positions: Vec<(i32, i32)>,
    /// Priority of the highest-priority chunk in the batch (used for
    /// scheduler ordering).
    pub max_priority: Priority,
    /// Values for binding 6 for exactly these chunks, or `None` when the
    /// dimension has no CPU-evaluated subtrees. Owned by the batch so that
    /// several batches can be in flight or queued at once.
    pub externals: Option<ExternalInputs>,
}

/// Accumulates chunk requests until a batch is full or a timeout expires.
pub struct PendingQueue {
    /// Individual pending chunk positions.
    pending: Vec<ChunkPos>,
    /// When the first pending chunk was added (for timeout-based flush).
    first_added: Option<Instant>,
    /// Maximum number of chunks per batch before auto-flush.
    batch_threshold: usize,
    /// Maximum time (ms) a chunk can wait before being flushed with
    /// whatever is available.
    max_wait_ms: u64,
    /// Hard cap on buffered chunks. Must be >= the largest single request
    /// that can be pushed before a flush (i.e. ExecutorConfig.max_batch_size),
    /// or `request_chunks`'s push-all-then-flush will overflow. Derived from
    /// the executor config by the caller; the `new()` default keeps the old
    /// (small) bound for unit tests.
    max_pending: usize,
}

impl PendingQueue {
    /// Create a new queue with the given batching parameters.
    pub fn new(batch_threshold: usize, max_wait_ms: u64) -> Self {
        // Legacy default: capacity = 2x the batch threshold. Callers that
        // serve a specific executor size should use `with_capacity`.
        Self::with_capacity(batch_threshold, max_wait_ms, batch_threshold * 2)
    }

    /// Like `new`, but with an explicit hard cap on buffered chunks.
    pub fn with_capacity(batch_threshold: usize, max_wait_ms: u64, max_pending: usize) -> Self {
        Self {
            pending: Vec::with_capacity(batch_threshold),
            first_added: None,
            batch_threshold,
            max_wait_ms,
            max_pending: max_pending.max(1),
        }
    }

    /// Add a chunk request to the pending queue.
    /// Returns `Ok(())` on success, `Err` if the queue is full.
    pub fn push(&mut self, pos: ChunkPos) -> Result<()> {
        if self.pending.len() >= self.max_pending {
            return Err(anyhow!("PendingQueue overflow — too many pending chunks"));
        }
        if self.first_added.is_none() {
            self.first_added = Some(Instant::now());
        }
        self.pending.push(pos);
        Ok(())
    }

    /// Check if the queue should be flushed now (batch full or timeout).
    pub fn should_flush(&self) -> bool {
        if self.pending.is_empty() {
            return false;
        }
        if self.pending.len() >= self.batch_threshold {
            return true;
        }
        if let Some(first) = self.first_added {
            if first.elapsed().as_millis() >= self.max_wait_ms as u128 {
                return true;
            }
        }
        false
    }

    /// Flush the pending queue into a `Batch`. Returns None if the queue
    /// is empty.
    pub fn flush(&mut self) -> Option<Batch> {
        if self.pending.is_empty() {
            return None;
        }
        let mut positions: Vec<(i32, i32)> = self.pending
            .iter()
            .map(|p| (p.cx, p.cz))
            .collect();
        positions.sort_unstable();
        positions.dedup();

        let max_priority = self.pending
            .iter()
            .map(|p| p.priority)
            .min() // lower value = higher priority
            .unwrap_or(Priority::Background);

        self.pending.clear();
        self.first_added = None;

        Some(Batch {
            positions,
            max_priority,
            externals: None,
        })
    }

    /// Number of currently pending chunks.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

impl Default for PendingQueue {
    fn default() -> Self {
        Self::new(16, 50) // 16 chunks or 50ms timeout
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_increments_count() {
        let mut q = PendingQueue::new(16, 50);
        assert_eq!(q.pending_count(), 0);
        q.push(ChunkPos { cx: 0, cz: 0, priority: Priority::Normal }).unwrap();
        assert_eq!(q.pending_count(), 1);
    }

    #[test]
    fn flush_empty_returns_none() {
        let mut q = PendingQueue::new(16, 50);
        assert!(q.flush().is_none());
    }

    #[test]
    fn flush_returns_sorted_deduped_positions() {
        let mut q = PendingQueue::new(16, 50);
        q.push(ChunkPos { cx: 3, cz: 3, priority: Priority::Normal }).unwrap();
        q.push(ChunkPos { cx: 0, cz: 0, priority: Priority::Normal }).unwrap();
        q.push(ChunkPos { cx: 3, cz: 3, priority: Priority::Normal }).unwrap(); // duplicate
        q.push(ChunkPos { cx: 1, cz: 0, priority: Priority::Critical }).unwrap();

        let batch = q.flush().unwrap();
        assert_eq!(batch.positions, vec![(0, 0), (1, 0), (3, 3)]); // sorted + deduped
        assert_eq!(batch.max_priority, Priority::Critical);
        assert_eq!(q.pending_count(), 0);
    }

    #[test]
    fn should_flush_at_threshold() {
        let mut q = PendingQueue::new(4, 5000);
        for i in 0..4 {
            q.push(ChunkPos { cx: i, cz: i, priority: Priority::Normal }).unwrap();
        }
        assert!(q.should_flush());
    }

    #[test]
    fn should_not_flush_below_threshold() {
        let mut q = PendingQueue::new(16, 5000);
        q.push(ChunkPos { cx: 0, cz: 0, priority: Priority::Normal }).unwrap();
        assert!(!q.should_flush());
    }

    #[test]
    fn flush_clears_first_added() {
        let mut q = PendingQueue::new(16, 5000);
        q.push(ChunkPos { cx: 0, cz: 0, priority: Priority::Normal }).unwrap();
        let _ = q.flush();
        q.push(ChunkPos { cx: 1, cz: 1, priority: Priority::Normal }).unwrap();
        assert!(q.first_added.is_some());
    }
}
