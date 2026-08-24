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
use std::sync::mpsc;
use std::time::Duration;

pub struct StagingChain {
    slots: Vec<StagingSlot>,
    free_queue: VecDeque<usize>,
    inflight_count: usize,
    ready_count: usize,
}

struct StagingSlot {
    buffer: wgpu::Buffer,
    phase: StagingPhase,
    cpu_data: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StagingPhase {
    Free,
    InFlight,
    ReadyForRead,
    Consumed,
}

impl StagingChain {
    pub fn new(device: &wgpu::Device, capacity: usize, max_buffer_size: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        let mut free_queue = VecDeque::with_capacity(capacity);
        for i in 0..capacity {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("c2me_wgpu_staging_{}", i)),
                size: max_buffer_size as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            slots.push(StagingSlot {
                buffer,
                phase: StagingPhase::Free,
                cpu_data: None,
            });
            free_queue.push_back(i);
        }
        Self {
            slots,
            free_queue,
            inflight_count: 0,
            ready_count: 0,
        }
    }

    /// 返回 (staging_buffer, slot_handle) 供 executor 使用
    pub fn acquire_slot(&mut self) -> Option<(wgpu::Buffer, SlotHandle)> {
        let idx = self.free_queue.pop_front()?;
        let slot = &mut self.slots[idx];
        debug_assert!(slot.phase == StagingPhase::Free || slot.phase == StagingPhase::Consumed);
        slot.phase = StagingPhase::InFlight;
        self.inflight_count += 1;
        let handle = SlotHandle { index: idx };
        Some((slot.buffer.clone(), handle))
    }

    pub fn mark_ready(&mut self, handle: SlotHandle, device: &wgpu::Device, byte_count: usize) {
        let slot = &mut self.slots[handle.index];
        debug_assert!(slot.phase == StagingPhase::InFlight,
            "mark_ready called on non-InFlight slot (phase = {:?})", slot.phase);

        let size = byte_count.min(slot.buffer.size() as usize);
        let slice = slot.buffer.slice(..size as u64);

        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // 轮询直到映射完成，每次等待 50ms
        while let Ok(_) = rx.try_recv() {
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_millis(50)),
            });
        }
        let result = rx.recv().unwrap();
        result.expect("failed to map staging buffer");

        let mapped = slice.get_mapped_range().unwrap();
        let cpu_copy = mapped.as_ref().to_vec();
        drop(mapped);
        slot.buffer.unmap();

        slot.cpu_data = Some(cpu_copy);
        slot.phase = StagingPhase::ReadyForRead;
        self.inflight_count = self.inflight_count.saturating_sub(1);
        self.ready_count += 1;
    }

    pub fn get_data(&self, handle: SlotHandle) -> Option<&[u8]> {
        let slot = &self.slots[handle.index];
        if slot.phase != StagingPhase::ReadyForRead {
            return None;
        }
        slot.cpu_data.as_deref()
    }

    pub fn consume(&mut self, handle: SlotHandle) {
        let slot = &mut self.slots[handle.index];
        debug_assert!(slot.phase == StagingPhase::ReadyForRead,
            "consume called on non-Ready slot (phase = {:?})", slot.phase);
        slot.phase = StagingPhase::Consumed;
        slot.cpu_data = None;
        self.ready_count = self.ready_count.saturating_sub(1);
        self.free_queue.push_back(handle.index);
    }

    pub fn capacity(&self) -> usize { self.slots.len() }
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