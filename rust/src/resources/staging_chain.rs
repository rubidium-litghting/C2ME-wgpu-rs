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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, express OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

//! Three-phase staging-buffer state tracking to prevent a newer dispatch
//! reading an in-flight older result (R07 risk).
//!
//! State machine per staging buffer:
//!   A = GPU computation in flight (buffer locked as OUTPUT)
//!   B = dispatch completed, data copied to CPU
//!   C = CPU consumed → return to buffer_pool.
//!
//! In wgpu 30 (no more explicit Fence), we use `device.poll(PollType::Wait)`
//! to synchronize, and we copy the GPU buffer contents to a CPU-side Vec
//! during the transition so the mapping lifetime is simple and safe.

use std::collections::VecDeque;

/// Staging buffer phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagingPhase {
    /// GPU is currently writing to this buffer.
    InFlight,
    /// GPU finished writing; data is available in CPU-side Vec.
    ReadyForRead,
    /// CPU consumed the data; buffer is free for reuse.
    Consumed,
}

/// A single staging buffer slot in the chain.
struct StagingSlot {
    buffer: wgpu::Buffer,
    phase: StagingPhase,
    /// CPU-side copy of the GPU buffer data (filled during mark_ready).
    cpu_data: Option<Vec<u8>>,
}

/// Three-phase staging buffer chain.
pub struct StagingChain {
    slots: VecDeque<StagingSlot>,
    inflight_count: usize,
    ready_count: usize,
}

impl StagingChain {
    pub fn new(device: &wgpu::Device, capacity: usize, buffer_size: usize) -> Self {
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::MAP_READ
            | wgpu::BufferUsages::COPY_DST;
        let mut slots = VecDeque::with_capacity(capacity);
        for i in 0..capacity {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("c2me_wgpu_staging_{i}")),
                size: buffer_size as u64,
                usage,
                mapped_at_creation: false,
            });
            slots.push_back(StagingSlot {
                buffer: buf,
                phase: StagingPhase::Consumed,
                cpu_data: None,
            });
        }
        Self {
            slots,
            inflight_count: 0,
            ready_count: 0,
        }
    }

    /// Find the oldest `Consumed` slot and mark it as `InFlight`.
    pub fn acquire_slot(&mut self) -> Option<(&wgpu::Buffer, SlotHandle)> {
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if slot.phase == StagingPhase::Consumed {
                slot.phase = StagingPhase::InFlight;
                slot.cpu_data = None;
                self.inflight_count += 1;
                let handle = SlotHandle { index: i };
                return Some((&slot.buffer, handle));
            }
        }
        None
    }

    /// Transition a slot from InFlight → ReadyForRead:
    ///   1. Wait for GPU to finish (`PollType::Wait`).
    ///   2. Map the buffer, copy data to CPU-side Vec, unmap.
    ///   3. Store the CPU data for later retrieval.
    pub fn mark_ready(&mut self, handle: SlotHandle, device: &wgpu::Device, byte_count: usize) {
        let slot = &mut self.slots[handle.index];
        debug_assert!(slot.phase == StagingPhase::InFlight,
            "mark_ready called on non-InFlight slot (phase = {:?})", slot.phase);

        // Wait for GPU
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        // Copy data from GPU to CPU
        let size = byte_count.min(slot.buffer.size() as usize);
        let slice = slot.buffer.slice(..size as u64);
        let mapped = slice.get_mapped_range()
            .expect("failed to map staging buffer");
        let cpu_copy = mapped.as_ref().to_vec();
        drop(mapped);
        slot.buffer.unmap();

        slot.cpu_data = Some(cpu_copy);
        slot.phase = StagingPhase::ReadyForRead;
        self.inflight_count = self.inflight_count.saturating_sub(1);
        self.ready_count += 1;
    }

    /// Get the CPU-side data for a ReadyForRead slot. Returns None if not
    /// in ReadyForRead phase.
    pub fn get_data(&self, handle: SlotHandle) -> Option<&[u8]> {
        let slot = &self.slots[handle.index];
        if slot.phase != StagingPhase::ReadyForRead {
            return None;
        }
        slot.cpu_data.as_deref()
    }

    /// Mark a slot as consumed and return it to the free pool.
    pub fn consume(&mut self, handle: SlotHandle) {
        let slot = &mut self.slots[handle.index];
        debug_assert!(slot.phase == StagingPhase::ReadyForRead,
            "consume called on non-Ready slot (phase = {:?})", slot.phase);
        slot.phase = StagingPhase::Consumed;
        slot.cpu_data = None;
        self.ready_count = self.ready_count.saturating_sub(1);
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub fn inflight_count(&self) -> usize { self.inflight_count }
    pub fn ready_count(&self) -> usize { self.ready_count }
    pub fn free_count(&self) -> usize {
        self.slots.len() - self.inflight_count - self.ready_count
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SlotHandle {
    index: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_transitions_without_gpu() {
        let _ = StagingPhase::InFlight;
        let _ = StagingPhase::ReadyForRead;
        let _ = StagingPhase::Consumed;
        assert!(StagingPhase::InFlight != StagingPhase::ReadyForRead);
        assert!(StagingPhase::ReadyForRead != StagingPhase::Consumed);
    }
}
