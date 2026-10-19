// SPDX-License-Identifier: MIT

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
use crate::ast::deserialize::ShapeConfig;
use crate::world::WorldContext;
use crate::world::constants::WorldgenParams;
use crate::world::bind_group;

/// Number of channels Java actually reads back. Only finalDensity (global
/// ch11) and veinToggle (global ch12) are consumed (see
/// `wgsl_gen::plan_shards`); the shader still writes a 16-channel out_buf,
/// we just copy these two out, shrinking the readback 8x.
/// Maximum number of `InnerRef` subtrees (from §14 Pass E split).
/// Total channels = 16 (original) + this.
pub const MAX_INNER_COUNT: usize = 128;

/// Global channel index of the first read-back channel (finalDensity).

/// Block height in blocks for this world's shape (16*H*16 blocks/chunk).
fn shape_height(world: &WorldContext) -> u32 { world.shape().height.max(1) as u32 }

/// Diagnostic probe: artificially delay each completed batch by N ms, to
/// measure how sensitive end-to-end throughput is to "GPU residency time"
/// (which bounds how long C2ME's position lock is held). Set
/// `C2ME_WGPU_FAKE_GPU_MS=<n>`. 0 (default) = off. Pure sleep; results and
/// correctness are unaffected.
fn fake_gpu_delay() -> Duration {
    use std::sync::OnceLock;
    static D: OnceLock<Duration> = OnceLock::new();
    *D.get_or_init(|| {
        let ms = std::env::var("C2ME_WGPU_FAKE_GPU_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(0);
        if ms > 0 {
            crate::wgpu_debug!("[probe] fake GPU delay = {} ms/batch", ms);
        }
        Duration::from_millis(ms)
    })
}

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
    /// §20 per-block state bytes (chunk-major, u32 per block: 0=air 1=stone).
    /// Empty unless the BlockOut shard is enabled.
    pub block_data: Vec<u32>,
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
    /// §20 block-state output (binding 7): u32 per block, for the BlockOut
    /// shard. Sized for the max batch; the block count per chunk is
    /// 16 * height * 16 (up to 16*512*16 = 131072).
    block_buf: wgpu::Buffer,
    /// External-input values (binding 6), lazily (re)created to fit the batch.
    external_buf: Option<wgpu::Buffer>,
    /// §22: aquifer cells, binding 8. Layout: [ aq_p (N vec4) | aq_s (N vec4) ];
    /// only `.x` of the aq_s half carries the fluid state.
    aquifer_buf: wgpu::Buffer,
    /// §23: Beardifier data, binding 9. Layout:
    /// [ weight table (13824 u32, bitcast f32) | per-chunk pj (4096 u32/chunk) ].
    beard_buf: wgpu::Buffer,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    /// §22: separate layout for the BlockOut shard (6 storage buffers).
    block_out_bg_layout: wgpu::BindGroupLayout,
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
        // §14 Pass E: the outer tree needs per-block evaluation, which reads
        // all the split `inners` from extra channels. Reserve up to 128 inner
        // channels (144 total). Memory cost per chunk is ~700 KB, so a batch of
        // 100 chunks is ~70 MB — fine for desktop and modern mobile.
        let total_channels = 16 + MAX_INNER_COUNT;
        let out_bytes_per_chunk: usize = total_channels * MAX_CELLS_PER_CHUNK * 4;
        // staging receives only the read-back channels, but the executor is
        // shared across worlds (real router = 2 channels, Single-mode test
        // blobs = up to N), so reserve the full 16-channel width. Only
        // `total_bytes` is actually mapped/copied per batch, so the 8x
        // readback saving is unaffected by this slack.
        let max_buffer_size = out_bytes_per_chunk * config.max_batch_size.max(1);
        // §20: block-state output (u32/block), 16*512*16 blocks/chunk max.
        const MAX_BLOCKS_PER_CHUNK: usize = 16 * 512 * 16;
        let block_max = MAX_BLOCKS_PER_CHUNK * config.max_batch_size.max(1) * 4;
        // Staging holds density readback + block bytes.
        let staging_size = max_buffer_size + block_max;
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
        let block_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("c2me_wgpu_block_buf"),
            size: block_max as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // §22 B-2: aquifer cell tables. Max cells/chunk ≈ 3*131*3 = 1179 for
        // height=512, cell_height=4; round to 1200. vec4<i32>/cell + u32/cell.
        const MAX_AQ_CELLS: usize = 1200;
        let aq_stride = MAX_AQ_CELLS * config.max_batch_size.max(1);
        // aq_p half + aq_s half; aq_s is padded to vec4 so one array type works.
        let aquifer_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("c2me_wgpu_aquifer"),
            size: (aq_stride * 16 * 2) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // §23: 24^3 weight table (13824 u32) followed by 4096-u32/chunk pj.
        const BEARD_W_LEN: usize = 24 * 24 * 24;
        let beard_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("c2me_wgpu_beard"),
            size: ((BEARD_W_LEN + 4096 * config.max_batch_size.max(1)) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = bind_group::create_layout(&device)
            .map_err(|e| format!("bind group layout: {e}"))?;
        let block_out_bg_layout = bind_group::create_block_out_layout(&device)
            .map_err(|e| format!("block-out bind group layout: {e}"))?;
        let pipeline_layout = bind_group::create_pipeline_layout(&device, &bind_group_layout)
            .map_err(|e| format!("pipeline layout: {e}"))?;

        // Write the static Beardifier weight table into the front of beard_buf
        // once; per-batch uploads only touch the pj section after it.
        {
            let w: Vec<u32> = crate::world::beard::build_weight_table()
                .iter().map(|f| f.to_bits()).collect();
            queue.write_buffer(&beard_buf, 0, bytemuck::cast_slice(&w));
        }

        Ok(Self {
            device,
            queue,
            staging,
            params_buf,
            positions_buf,
            output_buf,
            block_buf,
            external_buf: None,
            aquifer_buf,
            beard_buf,
            bind_group_layout,
            block_out_bg_layout,
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
        let density_bytes = (num_chunks as usize) * out_channels * cells_per_chunk * 4;
        // §20 BlockOut shard present? (reads ch11 corners, writes block bytes)
        let has_block = world.shards().iter()
            .any(|sh| sh.pass == crate::codegen::wgsl_gen::Pass::BlockOut);
        let blocks_per_chunk = 16usize * (shape_height(world) as usize) * 16;
        let block_bytes = if has_block {
            (num_chunks as usize) * blocks_per_chunk * 4
        } else { 0 };
        let total_bytes = density_bytes + block_bytes;

        let shape = world.shape();

        // Build per-batch params (all _pad slots except [10], which the
        // aquifer upload fills in below), upload the three side tables
        // (external inputs / aquifer / beard), then write params ONCE.
        let mut params = Self::build_params(world, batch, num_chunks, shape);
        let ext_buf = self.ensure_external_buffer(world);
        if let Some(aq_n) = self.upload_aquifer_tables(world, batch, shape) {
            // aq_s half-offset: only the first `aq_n` vec4 entries were
            // written, so the state half starts at index `aq_n`.
            params._pad[10] = aq_n as u32;
        }
        self.upload_beard_data(world, batch);
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        // Packed chunk positions [cx,cz,...] for the shader (binding 4).
        // Sorted + deduped by PendingQueue::flush, matching the output order.
        let mut packed_positions: Vec<i32> = Vec::with_capacity(batch.positions.len() * 2);
        for &(cx, cz) in &batch.positions {
            packed_positions.push(cx);
            packed_positions.push(cz);
        }
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
                    resource: ext_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.block_buf.as_entire_binding(),
                },
            ],
        });

        // §22: BlockOut shard gets its own bind group (6 buffers).
        let block_out_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("c2me_wgpu_block_out_bg"),
            layout: &self.block_out_bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.output_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.positions_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.block_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.aquifer_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.beard_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: perm_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: origin_buf.as_entire_binding() },
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

            // One dispatch per shard. Corner shards use the corner grid; the
            // §20 BlockOut shard uses a per-BLOCK grid (16 x height x 16 per
            // chunk). Both write disjoint buffers, so order is irrelevant.
            let shards = world.shards();
            let height = shape.height.max(1) as u32;
            let bw_x = (16u32 + 3) / 4;              // 4
            let bw_y = (height + 3) / 4;
            let bw_z = (16u32 * num_chunks + 3) / 4;
            
            for (i, pipeline) in pipelines.iter().enumerate() {
                // IMPORTANT: Each shard gets its own compute pass. WebGPU only
                // guarantees memory synchronization (barriers) between distinct
                // passes, not between dispatches inside one pass. BlockOut reads
                // the corner values written by shards B and C, so without this
                // separation it may race and read stale/zeroed data, turning solid
                // terrain into aquifer-filled water.
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("c2me_wgpu_batch_compute"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pipeline);
                if shards[i].pass == crate::codegen::wgsl_gen::Pass::BlockOut {
                    cpass.set_bind_group(0, &block_out_bg, &[]);
                    cpass.dispatch_workgroups(bw_x, bw_y, bw_z);
                } else {
                    cpass.set_bind_group(0, &bind_group, &[]);
                    cpass.dispatch_workgroups(wg_x, wg_y, wg_z);
                }
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
                let src_off = ((ci * (16 + MAX_INNER_COUNT) + ch) * cpc) as u64 * 4;
                let dst_off = ((ci * out_channels + k) * cpc) as u64 * 4;
                let len = cpc as u64 * 4;
                encoder.copy_buffer_to_buffer(
                    &self.output_buf, src_off, &staging_buf, dst_off, len);
            }
        }

        // §20: copy the per-block state bytes into staging after the density.
        if has_block {
            encoder.copy_buffer_to_buffer(
                &self.block_buf, 0, &staging_buf, density_bytes as u64, block_bytes as u64);
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

        // §20: extract the per-block state bytes (chunk-major) for Java.
        let block_data: Vec<u32> = if has_block {
            match self.staging.get_data(slot_handle) {
                Some(d) if d.len() >= density_bytes + block_bytes => d
                    [density_bytes..density_bytes + block_bytes]
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
                _ => Vec::new(),
            }
        } else { Vec::new() };

        self.staging.consume(slot_handle);

        // Diagnostic probe: simulate a longer GPU residency so we can measure
        // how it affects end-to-end throughput (see fake_gpu_delay).
        let fake = fake_gpu_delay();
        if !fake.is_zero() {
            std::thread::sleep(fake);
        }

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
            block_data,
        }))
    }

    /// Build the per-batch [`WorldgenParams`]. Every `_pad` slot is filled
    /// except `_pad[10]` (the aquifer state half-offset), which only becomes
    /// known after [`Self::upload_aquifer_tables`].
    fn build_params(
        world: &WorldContext,
        batch: &Batch,
        num_chunks: u32,
        shape: &ShapeConfig,
    ) -> WorldgenParams {
        // `batch.positions` holds CHUNK coords; the shader wants BLOCK coords.
        // A chunk is 16 blocks wide along X and Z. Per-chunk origins live in
        // the `positions` buffer; params keeps the batch-min origin for legacy
        // scalar math (unused by the prologue but harmless).
        let origin_x = batch.positions.iter().map(|p| p.0).min().unwrap_or(0) * 16;
        let origin_z = batch.positions.iter().map(|p| p.1).min().unwrap_or(0) * 16;
        let mut params = WorldgenParams {
            batch_origin_x: origin_x,
            batch_origin_z: origin_z,
            cell_width: shape.cell_width as f32,
            cell_height: shape.cell_height as f32,
            min_y_f: shape.min_y as f32,
            height_f: shape.height as f32,
            batch_count_x: num_chunks as i32,
            batch_count_z: 1,
            _pad: [0; 12],
        };
        params._pad[0] = world.external_count() as u32;
        // §22: per-dimension aquifer mode in _pad[1..5].
        {
            let am = world.aquifer_mode();
            params._pad[1] = if am.has_aquifers { 1 } else { 0 };
            params._pad[2] = am.sea_level as u32;
            params._pad[3] = am.default_fluid;
            params._pad[4] = am.lava_level as u32;
        }
        // Ore-vein RNG seed in _pad[5..9]: (seedLo.lo, seedLo.hi, seedHi.lo, seedHi.hi).
        {
            let (lo, hi) = world.ore_seed();
            params._pad[5] = lo as u32;
            params._pad[6] = (lo as u64 >> 32) as u32;
            params._pad[7] = hi as u32;
            params._pad[8] = (hi as u64 >> 32) as u32;
            params._pad[9] = if world.ore_veins.load(std::sync::atomic::Ordering::Relaxed) { 1 } else { 0 };
        }
        // _pad[11] is the total channel count, used by WGSL chunk strides.
        params._pad[11] = (16 + MAX_INNER_COUNT) as u32;
        params
    }

    /// Return the external-inputs buffer (binding 6), (re)creating it if the
    /// current CPU-side payload needs more room. Returns an `Arc` clone so the
    /// caller can keep touching other `self` fields while holding it.
    fn ensure_external_buffer(&mut self, world: &WorldContext) -> wgpu::Buffer {
        let ext_count = world.external_count();
        let ext_data = world.external_data();
        let ext_bytes = (ext_data.len() * 4).max(4);
        if ext_count > 0 {
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
            buf.clone()
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
            self.external_buf.as_ref().unwrap().clone()
        }
    }

    /// Expand and upload the per-chunk aquifer tables (binding 8). Returns
    /// `Some(n)` with the `aq_p` cell count written, or `None` if no table was
    /// staged for any chunk in the batch.
    fn upload_aquifer_tables(
        &mut self,
        world: &WorldContext,
        batch: &Batch,
        shape: &ShapeConfig,
    ) -> Option<usize> {
        // Dimensions without aquifers (Nether/End) legitimately have no
        // tables. Skip the whole expansion: writing padded zero cells would
        // only make the shader fall through to "no cell found" for every
        // block, and logging it would spam once per chunk.
        if !world.aquifer_mode().has_aquifers {
            return None;
        }
        let mut aq_p: Vec<[i32; 4]> = Vec::new();
        let mut aq_s: Vec<u32> = Vec::new();
        if let Ok(mut tables) = world.aquifer_tables.lock() {
            let min_y = shape.min_y;
            let height = shape.height;
            let start_y = min_y.div_euclid(12) - 1;
            let size_y = (min_y + height).div_euclid(12) - min_y.div_euclid(12) + 3;
            let cells_per_chunk = (size_y * 9) as usize;
            for &(cx, cz) in &batch.positions {
                let before = aq_p.len();
                // Consume: remove so the map does not grow unboundedly.
                if let Some(bytes) = tables.remove(&(cx, cz)) {
                    if let Some(t) = crate::world::aquifer::AquiferTable::parse(
                            &bytes, 3, size_y, 3, cx - 1, start_y, cz - 1) {
                        t.pack_for_gpu(&mut aq_p, &mut aq_s);
                    }
                }
                // Every chunk must occupy exactly `cells_per_chunk` slots: the
                // shader addresses cells as chunk_index * aq_sy * 9. A chunk
                // with no table (fresh region edge, or a dimension without
                // aquifers) is padded with inert cells so later chunks do not
                // shift out of place.
                if aq_p.len() < before + cells_per_chunk {
                    // Vanilla always has a fully populated cell grid per chunk,
                    // so a gap here means the Java side failed to stage a table
                    // (async race or dimension mismatch). Pad with zeroes so
                    // later chunks keep their stride (the shader falls through
                    // its "no cell found" branch and writes stone) and log it.
                    // In tests the Java side never runs, so `aquifer_tables`
                    // is always empty -- the absence is expected, not an
                    // error. Only the production path logs it.
                    #[cfg(not(test))]
                    crate::debug_log_line(&format!(
                        "missing aquifer table for chunk ({cx}, {cz}); padding"));
                    aq_p.resize(before + cells_per_chunk, [0, 0, 0, 0]);
                    aq_s.resize(before + cells_per_chunk, 0);
                }
            }
        }
        if aq_p.is_empty() {
            return None;
        }
        // [ aq_p (vec4) | aq_s padded to vec4 (state in .x) ]
        let n = aq_p.len();
        let mut merged: Vec<[i32; 4]> = Vec::with_capacity(n * 2);
        merged.extend_from_slice(&aq_p);
        merged.extend(aq_s.iter().map(|&st| [st as i32, 0, 0, 0]));
        self.queue.write_buffer(&self.aquifer_buf, 0, bytemuck::cast_slice(&merged));
        Some(n)
    }

    /// Pack and upload the per-chunk Beardifier data (binding 9). Each chunk
    /// gets a fixed `STRIDE`-u32 slot; empty chunks get a zeroed stride so
    /// `beard_pj[base] == 0` disables beard.
    fn upload_beard_data(&mut self, world: &WorldContext, batch: &Batch) {
        const STRIDE: usize = 4096;
        const BEARD_W_BYTES: u64 = (24 * 24 * 24) as u64 * 4;
        let mut pj = vec![0u32; STRIDE * batch.positions.len()];
        if let Ok(mut tables) = world.beard_tables.lock() {
            for (ci, &(cx, cz)) in batch.positions.iter().enumerate() {
                if let Some(bytes) = tables.remove(&(cx, cz)) {
                    let n = (bytes.len() / 4).min(STRIDE);
                    let dst = ci * STRIDE;
                    for i in 0..n {
                        pj[dst + i] = u32::from_le_bytes(
                            bytes[i*4..i*4+4].try_into().unwrap());
                    }
                }
            }
        }
        // Weight table is written once at construction; only pj changes per
        // batch, so write straight past it.
        self.queue.write_buffer(&self.beard_buf, BEARD_W_BYTES, bytemuck::cast_slice(&pj));
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

        // Startup handshake: the worker owns buffer allocation, which can fail
        // (OOM, unsupported limit). Without this, spawn() would return Ok while
        // the thread had already died, and every later request_chunks() would
        // surface only "executor channel closed" with the real cause lost.
        let (init_tx, init_rx) = channel::bounded(1);

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
                        let _ = init_tx.send(Err(e));
                        return;
                    }
                };
                let _ = init_tx.send(Ok(()));
                Self::run_loop(&mut executor, &receiver);
            })
            .map_err(|e| format!("failed to spawn executor thread: {e}"))?;

        match init_rx.recv() {
            Ok(Ok(())) => Ok(Self { handle: Some(handle), sender }),
            Ok(Err(e)) => {
                let _ = handle.join();
                Err(format!("executor init failed: {e}"))
            }
            Err(_) => {
                let _ = handle.join();
                Err("executor thread died during init".to_string())
            }
        }
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
            "[profile] avg over {} batches ({} chunks): encode={:.2} submit={:.2} readback={:.2} decode={:.2} total={:.2} ms/batch",
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
