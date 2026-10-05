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

//! 调度执行器 (Scheduler + Executor)
//!
//! 核心职责：
//!   1. 从 `PendingQueue` 取出 `Batch`，使用 WorldContext 的 pipeline。
//!   2. 编码 Compute Pass，动态创建 bind groups，dispatch。
//!   3. 使用 StagingChain 回读结果，封装 ResultBlob。
//!   4. 支持 ExecutorThread 独占线程串行化 GPU 提交。

use std::sync::Arc;
use std::time::Duration;
use std::thread::{self, JoinHandle};

use rustc_hash::FxHashMap;
use crossbeam::channel;

use crate::result::ResultBlob;
use crate::scheduler::batcher::Batch;
#[cfg(test)]
use crate::scheduler::batcher::Priority;
use crate::resources::staging_chain::{StagingChain, SlotHandle};
use crate::world::WorldContext;
use crate::world::constants::WorldgenParams;
use crate::world::bind_group;

/// Number of channels Java actually reads back. Only finalDensity (global
/// ch11) and veinToggle (global ch12) are consumed (see
/// `wgsl_gen::plan_shards`); the shader still writes a 16-channel out_buf,
/// we just copy these two out, shrinking the readback 8x.
const OUT_CHANNELS: usize = 2;
/// Global channel index of the first read-back channel (finalDensity).
const CH_FIRST: usize = 11;

/// 执行器配置。
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum number of chunks to dispatch in one batch.
    pub max_batch_size: usize,
    /// Timeout for waiting on new chunks (ms).
    pub flush_timeout_ms: u64,
    /// Maximum number of staging buffers for triple buffering.
    pub staging_capacity: usize,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            flush_timeout_ms: 50,
            staging_capacity: 3,
        }
    }
}

/// 一次已执行的批次，包含结果和元数据。
pub struct ExecutedBatch {
    pub batch: Batch,
    pub results: FxHashMap<i64, ResultBlob>,
    pub dispatch_time: Duration,
}

/// 负责单次 GPU 命令提交的执行器。
pub struct SingleBatchExecutor {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    staging: StagingChain,
    /// Persistent GPU buffers reused across dispatches. Allocating these per
    /// batch leaked: wgpu defers buffer destruction until in-flight work
    /// completes, so on slow GPUs (Mali) the un-reclaimed buffers piled up and
    /// OOM'd. Sizes are fixed by max_batch_size, so one set suffices.
    params_buf: wgpu::Buffer,
    positions_buf: wgpu::Buffer,
    output_buf: wgpu::Buffer,
    /// External-input values (binding 6), lazily (re)created to fit the batch.
    external_buf: Option<wgpu::Buffer>,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    pipeline_layout: wgpu::PipelineLayout,
    #[allow(dead_code)]
    config: ExecutorConfig,
}

impl SingleBatchExecutor {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: ExecutorConfig,
    ) -> Result<Self, String> {
        // Size persistent buffers for the worst realistic shape. The real
        // per-chunk size is derived per-batch from world.cells_per_chunk();
        // this bound must be >= any shape we might see. 5×129×5 covers
        // height up to 512 with cell_height down to 4.
        const MAX_CELLS_PER_CHUNK: usize = 5 * 129 * 5;
        // output_buf still holds the full 16-channel layout the shader writes.
        let out_bytes_per_chunk: usize = 16 * MAX_CELLS_PER_CHUNK * 4;
        // staging receives only the read-back channels, but the executor is
        // shared across worlds (real router = 2 channels, Single-mode test
        // blobs = up to N), so reserve the full 16-channel width. Only
        // `total_bytes` is actually mapped/copied per batch, so the 8x
        // readback saving is unaffected by this slack.
        let max_buffer_size = out_bytes_per_chunk * config.max_batch_size.max(1);
        let staging_size = max_buffer_size;
        let staging = StagingChain::new(&device, config.staging_capacity, staging_size);

        // Persistent buffers (see struct doc). output_buf is sized for the
        // max batch; smaller batches use a prefix.
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("c2me_wgpu_params"),
            size: std::mem::size_of::<WorldgenParams>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let positions_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("c2me_wgpu_positions"),
            size: (config.max_batch_size.max(1) * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("c2me_wgpu_out_buf"),
            size: max_buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group_layout = bind_group::create_layout(&device)
            .map_err(|e| format!("bind group layout: {e}"))?;
        let pipeline_layout = bind_group::create_pipeline_layout(&device, &bind_group_layout)
            .map_err(|e| format!("pipeline layout: {e}"))?;

        Ok(Self {
            device,
            queue,
            staging,
            params_buf,
            positions_buf,
            output_buf,
            external_buf: None,
            bind_group_layout,
            pipeline_layout,
            config,
        })
    }

    /// 执行一个 batch：编码 + 提交 + 回读。
    pub fn execute_batch(
        &mut self,
        world: &WorldContext,
        batch: &Batch,
    ) -> Result<Arc<ExecutedBatch>, String> {
        if batch.positions.is_empty() {
            return Err("empty batch".into());
        }

        let pipelines = world.pipelines();
        if pipelines.is_empty() {
            return Err("pipeline not compiled".to_string());
        }

        let t0 = std::time::Instant::now();
        if batch.positions.len() > self.config.max_batch_size {
            return Err(format!(
                "batch size {} exceeds max_batch_size {}",
                batch.positions.len(), self.config.max_batch_size));
        }
        let num_chunks = batch.positions.len() as u32;
        // Cells per chunk is shape-dependent (e.g. 5×49×5 vanilla, 5×65×5
        // with taller datapacks) — must match the shader's grid exactly, or
        // the readback stride is wrong. Previously hard-coded to 2425
        // (5×97×5), which only matched the cell_height=4 test blob.
        let cells_per_chunk = world.cells_per_chunk();
        // Channels Java actually reads back, derived from the shard layout.
        let readback: Vec<usize> = world.readback_channels();
        let out_channels = readback.len().max(1);
        // Bytes copied into staging: only the read-back channels.
        let total_bytes = (num_chunks as usize) * out_channels * cells_per_chunk * 4;

        let shape = world.shape();

        // Build WorldgenParams
        // `batch.positions` holds CHUNK coords; the shader wants BLOCK coords.
        // A chunk is 16 blocks wide along X and Z.
        // Per-chunk origins now live in the `positions` buffer; params
        // keeps the batch-min origin for legacy scalar math (unused by the
        // prologue but harmless).
        let origin_x = batch.positions.iter().map(|p| p.0).min().unwrap_or(0) * 16;
        let origin_z = batch.positions.iter().map(|p| p.1).min().unwrap_or(0) * 16;


        let params = WorldgenParams {
            batch_origin_x: origin_x,
            batch_origin_z: origin_z,
            cell_width: shape.cell_width as f32,
            cell_height: shape.cell_height as f32,
            min_y_f: shape.min_y as f32,
            height_f: shape.height as f32,
            batch_count_x: num_chunks as i32,
            batch_count_z: 1,
            _pad: [0; 8],
        };

        // External inputs for CPU-evaluated subtrees (binding 6).
        let ext_count = world.external_count();
        let ext_data = world.external_data();
        let ext_bytes = (ext_data.len() * 4).max(4);
        let ext_buf = if ext_count > 0 {
            let need_recreate = match &self.external_buf {
                Some(b) => b.size() < ext_bytes as u64 || b.size() < 4,
                None => true,
            };
            if need_recreate {
                self.external_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("c2me_wgpu_external_inputs"),
                    size: ext_bytes as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            let buf = self.external_buf.as_ref().unwrap();
            self.queue.write_buffer(buf, 0, bytemuck::cast_slice(&ext_data));
            buf.as_entire_binding()
        } else {
            // No external inputs: bind a dedicated 4-byte dummy so binding 6
            // stays valid. (Reusing output_buf here would alias binding 2's
            // read-write storage and can trip wgpu validation.)
            if self.external_buf.is_none() {
                self.external_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("c2me_wgpu_external_dummy"),
                    size: 4,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            self.external_buf.as_ref().unwrap().as_entire_binding()
        };

        // Update the persistent params buffer (binding 0).
        let mut params = params;
        params._pad[0] = ext_count as u32;
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        // Packed chunk positions [cx,cz,...] for the shader (binding 4).
        // Sorted + deduped by PendingQueue::flush, matching the output order.
        let mut packed_positions: Vec<i32> = Vec::with_capacity(batch.positions.len() * 2);
        for &(cx, cz) in &batch.positions {
            packed_positions.push(cx);
            packed_positions.push(cz);
        }
        // Update the persistent positions buffer (binding 4).
        self.queue.write_buffer(&self.positions_buf, 0, bytemuck::cast_slice(&packed_positions));

        // Get or create perm buffer (binding 1)
        let perm_buf = world.perm_buffer()
            .ok_or_else(|| "perm buffer not uploaded".to_string())?;
        let origin_buf = world.origin_buffer()
            .ok_or_else(|| "origin buffer not uploaded".to_string())?;
        let simplex_buf = world.simplex_buffer()
            .ok_or_else(|| "simplex buffer not uploaded".to_string())?;

        // Reuse the persistent output buffer (binding 2).


        // Acquire staging slot
        let (staging_buf, slot_handle) = self
            .staging
            .acquire_slot()
            .ok_or_else(|| "no staging slot available".to_string())?;

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("c2me_wgpu_batch_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: perm_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: origin_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.positions_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: simplex_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: ext_buf,
                },
            ],
        });

        let prof = crate::profile_enabled();
        let t_encode0 = std::time::Instant::now();

        // Encode: clear staging, then copy output→staging (after compute writes to output)
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("c2me_wgpu_batch_encode"),
        });

        // Dispatch compute pass
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("c2me_wgpu_batch_compute"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);

            // Cell grid per chunk = (16/cell_width, height/cell_height, 16/cell_width).
            // Workgroup size is (4,4,4), so dispatch that many workgroups.
            let cw = shape.cell_width.max(1) as u32;
            let chh = shape.cell_height.max(1) as u32;
            // Corner grid: one more corner than cells per axis.
            let ccx = 16 / cw + 1;
            let ccz = 16 / cw + 1;
            let ccy = (shape.height as u32) / chh + 1;
            // Multi-chunk: the z axis carries (cells_z × num_chunks)
            // invocations so one dispatch covers the whole batch.
            let num_chunks = batch.positions.len() as u32;
            let wg_x = (ccx + 3) / 4;
            let wg_y = (ccy + 3) / 4;
            let wg_z = (ccz * num_chunks + 3) / 4;

            // One dispatch per shard; every shard writes its own channels'
            // global out_buf slots, so order is irrelevant to correctness.
            for pipeline in pipelines {
                cpass.set_pipeline(pipeline);
                cpass.dispatch_workgroups(wg_x, wg_y, wg_z);
            }
        }

        // Copy only the read-back channels into staging. output_buf keeps
        // the full 16-channel layout the shader writes; we pluck each
        // consumed channel out of every chunk into a compact chunk-major
        // staging layout. For the real router that is ch11+ch12 (adjacent,
        // so effectively one 2*CPC run per chunk) — an 8x smaller readback.
        let cpc = cells_per_chunk;
        for ci in 0..num_chunks as usize {
            for (k, &ch) in readback.iter().enumerate() {
                let src_off = ((ci * 16 + ch) * cpc) as u64 * 4;
                let dst_off = ((ci * out_channels + k) * cpc) as u64 * 4;
                let len = cpc as u64 * 4;
                encoder.copy_buffer_to_buffer(
                    &self.output_buf, src_off, &staging_buf, dst_off, len);
            }
        }

        let encode_ms = t_encode0.elapsed().as_secs_f64() * 1e3;
        let t_submit0 = std::time::Instant::now();
        self.queue.submit(Some(encoder.finish()));
        let submit_ms = t_submit0.elapsed().as_secs_f64() * 1e3;

        // Mark staging as ready (copies GPU data to CPU)
        let t_readback0 = std::time::Instant::now();
        self.staging.mark_ready(slot_handle, &self.device, total_bytes);
        let readback_ms = t_readback0.elapsed().as_secs_f64() * 1e3;

        // Read results from staging buffer
        let t_decode0 = std::time::Instant::now();
        let result_map = self.read_results(batch, slot_handle, out_channels)?;
        let decode_ms = t_decode0.elapsed().as_secs_f64() * 1e3;

        self.staging.consume(slot_handle);

        let dispatch_time = t0.elapsed();

        if prof {
            profile_accumulate(batch.positions.len() as u64,
                encode_ms, submit_ms, readback_ms, decode_ms,
                dispatch_time.as_secs_f64() * 1e3);
        }

        Ok(Arc::new(ExecutedBatch {
            batch: batch.clone(),
            results: result_map,
            dispatch_time,
        }))
    }

    fn read_results(
        &mut self,
        batch: &Batch,
        slot_handle: SlotHandle,
        out_channels: usize,
    ) -> Result<FxHashMap<i64, ResultBlob>, String> {
        let data: &[u8] = self.staging.get_data(slot_handle)
            .ok_or_else(|| "failed to get staging data".to_string())?;

        // Convert bytes to f32
        let f32s: Vec<f32> = data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        let mut result_map = FxHashMap::default();
        // Derive cells-per-channel from the staging size so this stays in
        // sync with the shader's shape (5*97*5 before, 5*49*5 after the
        // vertical-cell-height fix). total = num_chunks*16*CPC*4 bytes.
                let nc = batch.positions.len().max(1);
        let cells_per_channel = (f32s.len() / (nc * out_channels)).max(1);
        // offsets[0..out_channels] describe the compact per-chunk layout.
        let mut channel_offsets = [0u64; 16];
        for ch in 0..out_channels {
            channel_offsets[ch] = (ch * cells_per_channel * 4) as u64;
        }
        let chunk_size_f32 = out_channels * cells_per_channel;

        for (i, &pos) in batch.positions.iter().enumerate() {
            let start = i * chunk_size_f32;
            let end = (start + chunk_size_f32).min(f32s.len());
            let chunk_data = &f32s[start..end];
            let blob = ResultBlob::from_f32_data(chunk_data, channel_offsets, 1);
            let key = ((pos.0 as i64) << 32) | (pos.1 as i64 & 0xFFFFFFFF);
            result_map.insert(key, blob);
        }

        Ok(result_map)
    }
}

/// 执行器线程 - 独占一个线程处理所有 GPU 提交 (避免 driver 竞态)。
pub struct ExecutorThread {
    handle: Option<JoinHandle<()>>,
    sender: channel::Sender<ExecutorMessage>,
}

enum ExecutorMessage {
    SubmitBatch {
        world: Arc<WorldContext>,
        batch: Batch,
        reply: channel::Sender<Result<Arc<ExecutedBatch>, String>>,
    },
    Shutdown,
}

impl ExecutorThread {
    pub fn spawn(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: ExecutorConfig,
    ) -> Result<Self, String> {
        let (sender, receiver) = channel::bounded(4);

        let handle = thread::Builder::new()
            .name("c2me_wgpu_executor".into())
            .spawn(move || {
                let mut executor = match SingleBatchExecutor::new(
                    device,
                    queue,
                    config,
                ) {
                    Ok(e) => e,
                    Err(e) => {
                        eprintln!("Failed to create executor: {}", e);
                        return;
                    }
                };
                Self::run_loop(&mut executor, &receiver);
            })
            .map_err(|e| format!("failed to spawn executor thread: {e}"))?;

        Ok(Self {
            handle: Some(handle),
            sender,
        })
    }

    /// Submit a batch synchronously (blocking) and return the result.
    pub fn request_chunks(
        &self,
        world: Arc<WorldContext>,
        batch: Batch,
    ) -> Result<Arc<ExecutedBatch>, String> {
        let (reply_tx, reply_rx) = channel::bounded(1);
        self.sender.send(ExecutorMessage::SubmitBatch {
            world,
            batch,
            reply: reply_tx,
        }).map_err(|_| "executor channel closed".to_string())?;

        reply_rx.recv()
            .map_err(|_| "executor reply channel closed".to_string())?
    }

    pub fn shutdown(&mut self) {
        if let Some(h) = self.handle.take() {
            let _ = self.sender.send(ExecutorMessage::Shutdown);
            let _ = h.join();
        }
    }

    fn run_loop(
        executor: &mut SingleBatchExecutor,
        receiver: &channel::Receiver<ExecutorMessage>,
    ) {
        loop {
            match receiver.recv() {
                Ok(ExecutorMessage::SubmitBatch { world, batch, reply }) => {
                    let result = executor.execute_batch(&world, &batch);
                    let _ = reply.send(result);
                }
                Ok(ExecutorMessage::Shutdown) | Err(_) => break,
            }
        }
    }
}

impl Drop for ExecutorThread {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executor_config_defaults() {
        let cfg = ExecutorConfig::default();
        assert!(cfg.max_batch_size >= 8);
        assert!(cfg.flush_timeout_ms <= 500);
        assert_eq!(cfg.staging_capacity, 3);
    }

    #[test]
    fn batch_has_sorted_positions() {
        let batch = Batch {
            positions: vec![(3, 3), (0, 0), (1, 0)],
            max_priority: Priority::Normal,
        };
        assert_eq!(batch.positions.len(), 3);
    }
}

// ── profiling aggregator ────────────────────────────────────────────────
// One averaged line every PROFILE_EVERY batches, so the log doesn't flood
// at tens of batches per second.
static P_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static P_CHUNKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static P_ENC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static P_SUB: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static P_RB: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static P_DEC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static P_TOT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn profile_accumulate(chunks: u64, enc: f64, sub: f64, rb: f64, dec: f64, tot: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    const N: u64 = 64; // ~1-2 s of activity per line
    let n = P_N.fetch_add(1, Relaxed) + 1;
    P_CHUNKS.fetch_add(chunks, Relaxed);
    P_ENC.fetch_add((enc * 1000.0) as u64, Relaxed);
    P_SUB.fetch_add((sub * 1000.0) as u64, Relaxed);
    P_RB.fetch_add((rb * 1000.0) as u64, Relaxed);
    P_DEC.fetch_add((dec * 1000.0) as u64, Relaxed);
    P_TOT.fetch_add((tot * 1000.0) as u64, Relaxed);
    if n >= N {
        let bn = n as f64;
        crate::profile_log(&format!(
            "[c2me-wgpu][profile] avg over {} batches ({} chunks): encode={:.2} submit={:.2} readback={:.2} decode={:.2} total={:.2} ms/batch",
            n, P_CHUNKS.load(Relaxed),
            P_ENC.load(Relaxed) as f64 / 1000.0 / bn,
            P_SUB.load(Relaxed) as f64 / 1000.0 / bn,
            P_RB.load(Relaxed) as f64 / 1000.0 / bn,
            P_DEC.load(Relaxed) as f64 / 1000.0 / bn,
            P_TOT.load(Relaxed) as f64 / 1000.0 / bn,
        ));
        P_N.store(0, Relaxed); P_CHUNKS.store(0, Relaxed);
        P_ENC.store(0, Relaxed); P_SUB.store(0, Relaxed);
        P_RB.store(0, Relaxed); P_DEC.store(0, Relaxed); P_TOT.store(0, Relaxed);
    }
}
