// SPDX-License-Identifier: MIT

//! (BufferUsage, rounded_size) → LRU bucket of free buffers.
//!
//! Round-up rules (matching MC driver behaviour):
//!   • Default: round up to next multiple of 64 KB
//!   • radv_round_buf_256 workaround: round up to next multiple of 256 bytes
//!
//! The pool is **not** thread-safe — it's owned by the executor thread.
//! Multi-threaded access must go through the executor's single-owner
//! model (which serialises all GPU commands anyway).

use std::collections::VecDeque;

const DEFAULT_ALIGN: usize = 64 * 1024; // 64 KB
const RADV_ALIGN: usize = 256;

/// A key that uniquely identifies a buffer slot in the pool.
/// Buffers are keyed by (usage, rounded_size) so we never hand out a
/// buffer with incompatible usage flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PoolKey {
    usage: wgpu::BufferUsages,
    rounded_size: usize,
}

/// LRU bucket of free buffers for a given PoolKey.
#[derive(Debug, Default)]
struct Bucket {
    buffers: VecDeque<wgpu::Buffer>,
}

impl Bucket {
    fn pop(&mut self) -> Option<wgpu::Buffer> {
        self.buffers.pop_front()
    }
    fn push(&mut self, buf: wgpu::Buffer) {
        self.buffers.push_back(buf);
    }
}

/// Recycle GPU buffers by size bucket instead of reallocating per dispatch.
///
/// The pool is single-owner (executor thread). `acquire` returns a buffer
/// whose size is >= the requested size; `release` puts it back.
pub struct BufferPool {
    buckets: rustc_hash::FxHashMap<PoolKey, Bucket>,
    /// Total number of free buffers across all buckets (for stats).
    total_free: usize,
    /// Whether to use the radv 256-byte alignment workaround.
    radv_align_256: bool,
}

impl BufferPool {
    pub fn new(radv_align_256: bool) -> Self {
        Self {
            buckets: Default::default(),
            total_free: 0,
            radv_align_256,
        }
    }

    /// Round up `size` to the nearest alignment boundary.
    pub fn round_up(&self, size: usize) -> usize {
        let align = if self.radv_align_256 { RADV_ALIGN } else { DEFAULT_ALIGN };
        // Need at least 4 bytes (min storage buffer size on some drivers).
        let align = align.max(4);
        let r = (size + align - 1) & !(align - 1);
        r.max(align)
    }

    /// Try to acquire a buffer from the pool. Returns None if no suitable
    /// buffer is available — caller must create a new one.
    pub fn acquire(
        &mut self,
        usage: wgpu::BufferUsages,
        size: usize,
    ) -> Option<wgpu::Buffer> {
        let rounded = self.round_up(size);
        let key = PoolKey { usage, rounded_size: rounded };
        if let Some(bucket) = self.buckets.get_mut(&key) {
            if let Some(buf) = bucket.pop() {
                self.total_free -= 1;
                return Some(buf);
            }
        }
        None
    }

    /// Return a buffer to the pool for future reuse.
    pub fn release(&mut self, buf: wgpu::Buffer) {
        let key = PoolKey {
            usage: buf.usage(),
            rounded_size: buf.size() as usize,
        };
        self.buckets.entry(key).or_default().push(buf);
        self.total_free += 1;
    }

    /// Create a new buffer (when the pool can't satisfy the request).
    pub fn create_buffer(
        &self,
        device: &wgpu::Device,
        usage: wgpu::BufferUsages,
        size: usize,
        label: Option<&str>,
    ) -> wgpu::Buffer {
        let rounded = self.round_up(size);
        device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size: rounded as u64,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Number of free buffers currently in the pool.
    pub fn total_free(&self) -> usize {
        self.total_free
    }

    /// Number of distinct (usage, rounded_size) buckets.
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Evict all free buffers (useful for shutdown or memory pressure).
    pub fn clear(&mut self) {
        self.buckets.clear();
        self.total_free = 0;
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_up_64k_default() {
        let pool = BufferPool::new(false);
        assert_eq!(pool.round_up(0), 65536);
        assert_eq!(pool.round_up(1), 65536);
        assert_eq!(pool.round_up(65536), 65536);
        assert_eq!(pool.round_up(65537), 131072);
        assert_eq!(pool.round_up(100_000), 131072);
    }

    #[test]
    fn round_up_radv_256() {
        let pool = BufferPool::new(true);
        assert_eq!(pool.round_up(0), 256);
        assert_eq!(pool.round_up(1), 256);
        assert_eq!(pool.round_up(256), 256);
        assert_eq!(pool.round_up(257), 512);
        assert_eq!(pool.round_up(1000), 1024);
    }

    #[test]
    fn acquire_release_roundtrip() {
        let pool = BufferPool::new(false);
        // No GPU in unit tests — we can't create real buffers.
        // Just test the round_up logic and pool bookkeeping.
        assert_eq!(pool.total_free(), 0);
        assert_eq!(pool.bucket_count(), 0);
    }

    #[test]
    fn pool_key_equality() {
        let a = PoolKey { usage: wgpu::BufferUsages::STORAGE, rounded_size: 65536 };
        let b = PoolKey { usage: wgpu::BufferUsages::STORAGE, rounded_size: 65536 };
        let c = PoolKey { usage: wgpu::BufferUsages::COPY_DST, rounded_size: 65536 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
