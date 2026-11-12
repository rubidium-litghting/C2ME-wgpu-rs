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
    /// Set by `begin_map`, cleared by `try_finish_map`. Holding the receiver
    /// here is what lets the mapping be polled from an event loop instead of
    /// blocking until the GPU is done.
    pending_map: Option<PendingMap>,
}

struct PendingMap {
    rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    size: usize,
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
                pending_map: None,
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

    /// 返回 (staging_buffer, slot_handle)
    pub fn acquire_slot(&mut self) -> Option<(wgpu::Buffer, SlotHandle)> {
        let idx = self.free_queue.pop_front()?;
        let slot = &mut self.slots[idx];
        debug_assert!(slot.phase == StagingPhase::Free || slot.phase == StagingPhase::Consumed);
        slot.phase = StagingPhase::InFlight;
        self.inflight_count += 1;
        let handle = SlotHandle { index: idx };
        Some((slot.buffer.clone(), handle))
    }

    /// Start an asynchronous map of `byte_count` bytes. Returns immediately;
    /// the mapping is driven by `try_finish_map`, which the caller polls.
    ///
    /// This is the pipelining half of the readback: the GPU keeps computing
    /// the next batch while this one is being mapped, instead of the executor
    /// thread sitting in `poll(Wait)` for the whole round trip.
    pub fn begin_map(&mut self, handle: SlotHandle, device: &wgpu::Device, byte_count: usize) {
        let slot = &mut self.slots[handle.index];
        debug_assert!(slot.phase == StagingPhase::InFlight,
            "begin_map on non-InFlight slot (phase = {:?})", slot.phase);
        let size = byte_count.min(slot.buffer.size() as usize);
        let slice = slot.buffer.slice(..size as u64);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        slot.pending_map = Some(PendingMap { rx, size });
        // Drive one poll so the callback is queued.
        let _ = device.poll(wgpu::PollType::Poll);
    }

    /// Non-blocking check for a completed mapping. Returns true once the CPU
    /// copy is ready and the slot has moved to ReadyForRead.
    pub fn try_finish_map(&mut self, handle: SlotHandle, device: &wgpu::Device) -> bool {
        let idx = handle.index;
        if self.slots[idx].pending_map.is_none() {
            return false;
        }
        let _ = device.poll(wgpu::PollType::Poll);

        let size = {
            let pm = self.slots[idx].pending_map.as_ref().unwrap();
            match pm.rx.try_recv() {
                Ok(Ok(())) => pm.size,
                Ok(Err(e)) => panic!("map_async error: {:?}", e),
                Err(mpsc::TryRecvError::Empty) => return false,
                Err(mpsc::TryRecvError::Disconnected) => {
                    panic!("map_async channel disconnected");
                }
            }
        };

        let slot = &mut self.slots[idx];
        slot.pending_map = None;
        let slice = slot.buffer.slice(..size as u64);
        let mapped = slice.get_mapped_range().unwrap();
        let cpu_copy = mapped.as_ref().to_vec();
        drop(mapped);
        slot.buffer.unmap();
        slot.cpu_data = Some(cpu_copy);
        slot.phase = StagingPhase::ReadyForRead;
        self.inflight_count = self.inflight_count.saturating_sub(1);
        self.ready_count += 1;
        true
    }

    /// Drive the device once and finish any mapping that has completed.
    /// Returns the slots that became ready, so the caller can publish their
    /// results.
    ///
    /// The device poll happens here rather than on a timer because wgpu only
    /// resolves `map_async` callbacks when someone polls; Java's status
    /// polling is the heartbeat that makes this happen often enough.
    pub fn poll_pending(&mut self, device: &wgpu::Device) -> Vec<SlotHandle> {
        let _ = device.poll(wgpu::PollType::Poll);
        let mut done = Vec::new();
        for i in 0..self.slots.len() {
            if self.slots[i].pending_map.is_some() {
                let h = SlotHandle { index: i };
                if self.try_finish_map(h, device) {
                    done.push(h);
                }
            }
        }
        done
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

        loop {
            match rx.try_recv() {
                Ok(inner) => {
                    match inner {
                        Ok(()) => break,
                        Err(e) => panic!("map_async error: {:?}", e),
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    panic!("map_async channel disconnected");
                }
                _ => ()
            }

            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_millis(50)),
            });
        }

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
    pub fn slot_size(&self) -> usize { self.slots.first().map_or(0, |s| s.buffer.size() as usize) }
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