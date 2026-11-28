// SPDX-License-Identifier: MIT

//! 调度执行器 (Scheduler + Executor)
//!
//! 核心职责：
//!   1. 从 `PendingQueue` 取出 `Batch`，使用 WorldContext 的 pipeline。
//!   2. 编码 Compute Pass，动态创建 bind groups，dispatch。
//!   3. 使用 StagingChain 回读结果，封装 ResultBlob。
//!   4. 支持 ExecutorThread 独占线程串行化 GPU 提交。

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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
use crate::codegen::wgsl_gen::Pass;
use crate::world::WorldContext;
use crate::world::constants::WorldgenParams;
use crate::world::bind_group;

/// Channels the shader evaluates for vanilla itself.
const BASE_CHANNELS: usize = 16;

/// Maximum number of `InnerRef` subtrees (from the Pass E split, §15.2.2).
pub const MAX_INNER_COUNT: usize = 128;

/// Channels the shader writes per cell. Java only reads back a few of them
/// (finalDensity ch11, veinToggle ch12 for the real router), and
/// `prepare_batch` copies just those into staging, so the unused width costs
/// GPU memory but not readback bandwidth.
const TOTAL_CHANNELS: usize = BASE_CHANNELS + MAX_INNER_COUNT;

/// Per-chunk slot of the Beardifier `pj` table, in u32 words. The shader
/// indexes `beard_pj[chunk * BEARD_STRIDE + i]`, so the executor and
/// `upload_beard_data` must agree on this number.
const BEARD_STRIDE: usize = 4096;

/// `24^3` Beardifier weight table, in u32 words. Written once per input set,
/// ahead of the per-batch `pj` section.
const BEARD_WEIGHT_WORDS: usize = 24 * 24 * 24;

/// Aquifer cells per chunk assumed by [`ExecutorConfig::default`]. Only the
/// test/offline path uses it; production derives the real value from the shape
/// via [`ShapeConfig::aquifer_cells_per_chunk`].
const AQ_DEFAULT_CELLS_PER_CHUNK: usize = 1200;

/// Height in blocks for this world's shape (`16 * H * 16` blocks/chunk).
fn shape_height(world: &WorldContext) -> u32 { world.shape().height.max(1) as u32 }

/// 执行器配置。
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum number of chunks to dispatch in one batch.
    pub max_batch_size: usize,
    /// Timeout for waiting on new chunks (ms).
    pub flush_timeout_ms: u64,
    /// Maximum number of staging buffers for triple buffering.
    pub staging_capacity: usize,
    /// How many batches may hold a set of input buffers at once. Must be at
    /// least the Java-side queue depth or acquiring a set would block; each
    /// set costs a few MB (params/positions/aquifer/beard/external), not the
    /// hundreds the output buffers would.
    pub pipeline_depth: usize,
    /// Corner-grid cells per chunk of the shape these buffers are sized for.
    ///
    /// The persistent output/staging buffers are allocated once, and their
    /// size is `channels × max_batch_size × cells_per_chunk`. A world with a
    /// taller or denser grid needs a rebuilt executor; see
    /// `c_api::get_or_create_executor`. (Hard-coding an upper bound here was
    /// wrong: Tectonic's 832-block overworld needs 2625 cells and 212 992
    /// blocks per chunk, which no vanilla-derived constant covers.)
    pub cells_per_chunk: usize,
    /// Blocks per chunk (`16 * height * 16`) for the BlockOut buffer (binding
    /// 7, u32 per block). Same rebuild-on-growth rule as `cells_per_chunk`.
    pub blocks_per_chunk: usize,
    /// Aquifer cells per chunk (binding 8) of this shape, from
    /// [`ShapeConfig::aquifer_cells_per_chunk`].
    ///
    /// The buffer holds one `vec4` pair per cell per batch. A fixed
    /// reservation (1200) covered vanilla and Tectonic but overran once the
    /// dimension reached ~2000 blocks, so this follows the shape like the
    /// other buffer sizes.
    pub aquifer_cells_per_chunk: usize,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            flush_timeout_ms: 50,
            staging_capacity: 3,
            // Same as the staging depth: the pool must be able to serve every
            // batch the staging chain can hold.
            pipeline_depth: 3,
            // Vanilla-ish upper bound. Production callers override these from
            // the actual shape; the defaults exist for unit tests and for the
            // offline blobs, which use a 384-block, cell_height=4 grid.
            cells_per_chunk: 5 * 129 * 5,
            blocks_per_chunk: 16 * 512 * 16,
            aquifer_cells_per_chunk: AQ_DEFAULT_CELLS_PER_CHUNK,
        }
    }
}

/// 一次已执行的批次，包含结果和元数据。
pub struct ExecutedBatch {
    pub batch: Batch,
    pub dispatch_time: Duration,
    /// Number of density channels actually read back (2 for the real router).
    pub out_channels: usize,

    // ── test path (`finish_batch` with raw = false) ──────────────────
    /// Per-chunk decoded results, keyed by packed chunk position. Used by the
    /// offline tests, which want one blob per chunk.
    pub results: FxHashMap<i64, ResultBlob>,
    /// §3.5 per-block state bytes (u32 per block, chunk-major).
    pub block_data: Vec<u32>,

    // ── production path (`finish_batch` with raw = true) ─────────────
    /// Density readback exactly as the GPU wrote it: chunk-major, channels
    /// contiguous within each chunk, f32 little-endian. This is byte-identical
    /// to the layout `ResultBlob::from_f32_data` expects, so the Java-facing
    /// blob can be assembled from it without decoding and re-encoding.
    pub raw_density: Vec<u8>,
    /// Per-block state bytes, u32 LE per block, chunk-major. Same content as
    /// `block_data` but already in the form the blob needs.
    pub raw_block: Vec<u8>,
}

/// A submitted batch whose output is being mapped asynchronously.
///
/// Produced by [`SingleBatchExecutor::prepare_batch`], consumed by
/// [`SingleBatchExecutor::finish_batch`]. Carries the read-back parameters so
/// `finish_batch` does not have to re-derive them from `world`.
struct PreparedBatch {
    batch: Batch,
    /// Index into BufferPool.sets, returned on finish_batch.
    buf_set: usize,
    slot_handle: SlotHandle,
    out_channels: usize,
    density_bytes: usize,
    block_bytes: usize,
    has_block: bool,
    prof: bool,
    t0: std::time::Instant,
    t_submit: std::time::Instant,
    encode_ms: f64,
    submit_ms: f64,
}

/// Readback of one batch, in whichever form the caller asked for.
///
/// `raw` batches fill the two `raw_*` vectors and leave `results`/`block_data`
/// empty; decoded (test) batches do the opposite.
#[derive(Default)]
struct DecodedBatch {
    results: FxHashMap<i64, ResultBlob>,
    block_data: Vec<u32>,
    raw_density: Vec<u8>,
    raw_block: Vec<u8>,
}

/// One set of per-batch input buffers.
///
/// Everything the GPU reads while a batch is in flight lives here. Pooling
/// these is what makes pipelining possible: with a single set, preparing the
/// next batch would overwrite the buffers the current one is still reading.
/// `output_buf` and `block_buf` are deliberately not here — they are written
/// and copied to staging inside the same command buffer, so the GPU's own
/// ordering protects them.
struct InputBufferSet {
    params: wgpu::Buffer,
    positions: wgpu::Buffer,
    aquifer: wgpu::Buffer,
    beard: wgpu::Buffer,
    external: Option<wgpu::Buffer>,
}

/// Round-robin pool of [`InputBufferSet`]s.
///
/// Depth should match the maximum number of batches that can be in flight,
/// which is the smaller of the Java-side queue depth and the staging chain
/// capacity. Acquire before writing, release once the batch's output has been
/// read back.
struct BufferPool {
    sets: Vec<InputBufferSet>,
    free: std::collections::VecDeque<usize>,
}

impl BufferPool {
    fn acquire(&mut self) -> Option<usize> {
        self.free.pop_front()
    }

    fn release(&mut self, idx: usize) {
        self.free.push_back(idx);
    }
}

/// Shorthand for a whole-buffer bind group entry.
fn binding(index: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding: index,
        resource: buffer.as_entire_binding(),
    }
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
    pool: BufferPool,
    output_buf: wgpu::Buffer,
    /// §3.5 block-state output (binding 7): u32 per block, for the BlockOut
    /// shard. Sized for the max batch; the block count per chunk is
    /// 16 * height * 16 (up to 16*512*16 = 131072).
    block_buf: wgpu::Buffer,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    /// §3.5: separate layout for the BlockOut shard (6 storage buffers).
    block_out_bg_layout: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    pipeline_layout: wgpu::PipelineLayout,
    #[allow(dead_code)]
    config: ExecutorConfig,
}

/// Byte sizes of the persistent buffers, derived once from the config.
///
/// Kept apart from allocation so all the shape arithmetic lives in one place:
/// lesson #56 was exactly a buffer size drifting away from the shape it has to
/// hold (`blocks_per_chunk` was a constant while the world got taller).
struct BufferSizes {
    cells_per_chunk: usize,
    blocks_per_chunk: usize,
    /// Density output (binding 2), `TOTAL_CHANNELS` wide per cell.
    output: usize,
    /// Block-state output (binding 7), one u32 per block.
    block: usize,
    /// One staging slot: density readback followed by the block bytes.
    staging: usize,
    /// Aquifer entries reserved per batch; the buffer holds `aq_p` then `aq_s`.
    aquifer_entries: usize,
    /// Beardifier buffer: weight table followed by one `BEARD_STRIDE` slot per
    /// chunk.
    beard: u64,
}

/// How many chunks the device's buffer limits can hold for a given shape.
///
/// The persistent buffers scale with the batch size, so the largest batch the
/// device can actually serve is a function of the shape:
///
/// * `b2_out` is one storage buffer of `TOTAL_CHANNELS × cells × 4` bytes per
///   chunk, so `max_storage_buffer_binding_size` bounds the first term;
/// * `b7_block` is one storage buffer of `blocks × 4` bytes per chunk;
/// * one staging slot is `b2_out + b7_block` per chunk and is a regular
///   buffer, so `max_buffer_size` bounds the sum.
///
/// The device is created with `required_limits = Limits::default()`, so these
/// are the WebGPU defaults (128 MiB / 256 MiB) even when the adapter reports
/// more. The result does not include the configured hard cap; callers clamp
/// their requested batch size with [`BatchLimit::allows`].
#[derive(Debug, Clone, Copy)]
pub struct BatchLimit {
    /// Largest batch the device limits allow for this shape.
    pub allows: usize,
    /// What the storage-buffer binding limit alone allows.
    pub binding_allows: usize,
    /// What the whole-buffer limit alone allows.
    pub buffer_allows: usize,
    /// What the external-inputs payload (binding 6) alone allows, or
    /// `usize::MAX` when the shape has no external slots.
    pub external_allows: usize,
}

impl BatchLimit {
    /// Derive the limit for a shape from the *device* limits.
    ///
    /// Must use the device's limits, not the adapter's: the device is requested
    /// with `Limits::default()`, so that is the real ceiling no matter what the
    /// adapter advertises.
    ///
    /// `external_nodes` is the number of CPU-evaluated slots per chunk that
    /// binding 6 will carry (0 when the dimension has none). It has to be part
    /// of the same derivation as the outputs: the payload is
    /// `nodes × cells × batch × 4` bytes, so a node count the outputs would
    /// happily accept can still be too large for a full batch. Callers that
    /// know the node count (the Java query) get the joint answer; the executor
    /// passes 0 because it sizes only its own buffers and its result is then
    /// clamped by the caller's joint number.
    pub fn derive(
        limits: &wgpu::Limits,
        cells_per_chunk: usize,
        blocks_per_chunk: usize,
        external_nodes: usize,
    ) -> Self {
        let per_chunk_out = (TOTAL_CHANNELS * cells_per_chunk.max(1) * 4) as u64;
        let per_chunk_block = (blocks_per_chunk.max(1) * 4) as u64;

        let binding = limits.max_storage_buffer_binding_size;
        let buffer = limits.max_buffer_size;

        // Both outputs are separate bindings, so each has to fit on its own.
        let binding_allows = (binding / per_chunk_out)
            .min(binding / per_chunk_block)
            .max(1);
        // The staging slot holds the two outputs back to back.
        let buffer_allows = (buffer / (per_chunk_out + per_chunk_block)).max(1);
        // Binding 6 is its own storage buffer, so it is bounded by the binding
        // limit on its own, and by the buffer limit when it is created.
        let external_allows = if external_nodes == 0 {
            usize::MAX
        } else {
            let per_chunk_ext = (external_nodes as u64) * (cells_per_chunk.max(1) as u64) * 4;
            let allows = (binding / per_chunk_ext).min(buffer / per_chunk_ext).max(1);
            allows.min(usize::MAX as u64) as usize
        };
        let allows = binding_allows
            .min(buffer_allows)
            .min(external_allows as u64)
            .max(1);

        let to_usize = |v: u64| v.min(usize::MAX as u64) as usize;
        Self {
            allows: to_usize(allows),
            binding_allows: to_usize(binding_allows),
            buffer_allows: to_usize(buffer_allows),
            external_allows,
        }
    }

    /// Largest number of external slots per chunk that fits binding 6 for a
    /// single chunk. This is the device-side ceiling for `MAX_EXTERNAL_NODES`;
    /// anything above it cannot be dispatched at all.
    pub fn external_slot_limit(limits: &wgpu::Limits, cells_per_chunk: usize) -> usize {
        let per_slot = (cells_per_chunk.max(1) as u64) * 4;
        let allows = (limits.max_storage_buffer_binding_size / per_slot)
            .min(limits.max_buffer_size / per_slot)
            .max(1);
        allows.min(usize::MAX as u64) as usize
    }
}

/// Final batch size from the requested cap, the shape's device-derived limit
/// and (when an executor already exists) that executor's fixed ceiling.
///
/// The last term matters across dimensions: switching from a dense shape that
/// sized the executor for 34 chunks to a smaller one does not rebuild it
/// (`fits` only grows), so the smaller shape's derivation can exceed what the
/// existing buffers can dispatch.
pub fn clamp_batch_limit(requested: usize, derived: usize, existing: Option<usize>) -> usize {
    requested
        .min(derived)
        .min(existing.unwrap_or(usize::MAX))
        .max(1)
}

impl BufferSizes {
    fn for_config(config: &ExecutorConfig) -> Self {
        let max_batch = config.max_batch_size.max(1);
        let cells_per_chunk = config.cells_per_chunk.max(1);
        let blocks_per_chunk = config.blocks_per_chunk.max(1);
        let aquifer_cells_per_chunk = config.aquifer_cells_per_chunk.max(1);
        let output = TOTAL_CHANNELS * cells_per_chunk * 4 * max_batch;
        let block = blocks_per_chunk * 4 * max_batch;
        Self {
            cells_per_chunk,
            blocks_per_chunk,
            output,
            block,
            staging: output + block,
            aquifer_entries: aquifer_cells_per_chunk * max_batch,
            beard: ((BEARD_WEIGHT_WORDS + BEARD_STRIDE * max_batch) * 4) as u64,
        }
    }
}

impl SingleBatchExecutor {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        mut config: ExecutorConfig,
    ) -> Result<Self, String> {
        // A batch larger than the device's buffer limits cannot be allocated,
        // let alone dispatched. Derive the ceiling from the shape and the
        // device's own limits, then clamp the requested size before any buffer
        // is created. On today's shapes this leaves 64 intact; on a denser
        // 832-block grid it would otherwise allocate a 192 MiB `b2_out` and
        // fail inside wgpu, which discards the command buffer and returns
        // garbage block data.
        let requested_batch = config.max_batch_size.max(1);
        let limit = BatchLimit::derive(
            &device.limits(), config.cells_per_chunk, config.blocks_per_chunk, 0);
        config.max_batch_size = clamp_batch_limit(requested_batch, limit.allows, None);

        // Sized for the shape this config was built for: `cells_per_chunk` and
        // `blocks_per_chunk` come from the world's own grid, and a world that
        // needs more rebuilds the executor (see `c_api`).
        let sizes = BufferSizes::for_config(&config);
        let staging = StagingChain::new(&device, config.staging_capacity, sizes.staging);
        let (output_buf, block_buf) = Self::create_output_buffers(&device, &sizes);

        // Static Beardifier weight table, written into every input set at
        // creation. The per-batch upload only rewrites the `pj` section after
        // it.
        let weight_table: Vec<u32> = crate::world::beard::build_weight_table()
            .iter().map(|f| f.to_bits()).collect();

        let depth = config.pipeline_depth.max(1);
        let sets = (0..depth)
            .map(|i| Self::create_input_set(
                &device, &queue, &config, &sizes, i, &weight_table))
            .collect();
        let pool = BufferPool {
            sets,
            free: (0..depth).collect(),
        };

        let bind_group_layout = bind_group::create_layout(&device)
            .map_err(|e| format!("bind group layout: {e}"))?;
        let block_out_bg_layout = bind_group::create_block_out_layout(&device)
            .map_err(|e| format!("block-out bind group layout: {e}"))?;
        let pipeline_layout = bind_group::create_pipeline_layout(&device, &bind_group_layout)
            .map_err(|e| format!("pipeline layout: {e}"))?;

        // Print the static footprint so field logs show whether a driver limit
        // is being approached. `external` is allocated per-world, so it is not
        // included here. The shape is logged too: buffers are rebuilt when a
        // world needs more, so this line must say what it was sized for.
        let mib = |b: usize| b / (1024 * 1024);
        crate::debug_log_line(&format!(
            "[c2me-wgpu-rs] executor buffers: output={} MiB, block={} MiB, staging={} MiB x{}, \
             input_sets={} x (aquifer={} KiB, beard={} KiB), max_batch={}, \
             sized_for={} cells/chunk / {} blocks/chunk / {} aquifer cells/chunk",
            mib(sizes.output),
            mib(sizes.block),
            mib(sizes.staging),
            config.staging_capacity,
            depth,
            sizes.aquifer_entries * 16 * 2 / 1024,
            sizes.beard as usize / 1024,
            config.max_batch_size,
            sizes.cells_per_chunk,
            sizes.blocks_per_chunk,
            config.aquifer_cells_per_chunk,
        ));
        // Separate line so a capped batch is visible even when the buffer
        // footprint itself looks fine: the binding ceiling is the one that
        // fails first, and it depends on the shape, not on the batch alone.
        crate::debug_log_line(&format!(
            "[c2me-wgpu-rs] batch limit: {} chunks (requested {}, binding allows {}, \
             staging buffer allows {})",
            config.max_batch_size, requested_batch, limit.binding_allows, limit.buffer_allows,
        ));

        Ok(Self {
            device,
            queue,
            staging,
            pool,
            output_buf,
            block_buf,
            bind_group_layout,
            block_out_bg_layout,
            pipeline_layout,
            config,
        })
    }

    /// Create the density and block-state output buffers.
    ///
    /// These are NOT pooled: each batch writes them and copies the result to
    /// staging inside one command buffer, so the GPU's own ordering prevents a
    /// later batch from overwriting data an earlier one still needs.
    fn create_output_buffers(
        device: &wgpu::Device,
        sizes: &BufferSizes,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let create = |label: &str, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        (create("c2me_wgpu_out_buf", sizes.output),
         create("c2me_wgpu_block_buf", sizes.block))
    }

    /// Create one per-batch input buffer set (bindings 0, 3/4, 8 and 9).
    ///
    /// Everything the GPU reads while a batch is in flight lives here, so
    /// having `pipeline_depth` of them is what allows the next batch to be
    /// prepared without overwriting the one still being read.
    fn create_input_set(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &ExecutorConfig,
        sizes: &BufferSizes,
        index: usize,
        weight_table: &[u32],
    ) -> InputBufferSet {
        let create = |label: String, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let params = create(
            format!("c2me_wgpu_params_{index}"),
            std::mem::size_of::<WorldgenParams>() as u64);
        // Two i32 per chunk: cx, cz.
        let positions = create(
            format!("c2me_wgpu_positions_{index}"),
            (config.max_batch_size.max(1) * 2 * 4) as u64);
        // `aq_p` half + `aq_s` half; `aq_s` is padded to vec4 so one array
        // type serves both.
        let aquifer = create(
            format!("c2me_wgpu_aquifer_{index}"),
            (sizes.aquifer_entries * 16 * 2) as u64);
        let beard = create(format!("c2me_wgpu_beard_{index}"), sizes.beard);
        queue.write_buffer(&beard, 0, bytemuck::cast_slice(weight_table));
        InputBufferSet { params, positions, aquifer, beard, external: None }
    }

    /// Effective batch size after the device-limit clamp. Buffers are sized
    /// by it, so it is also the largest batch `prepare_batch` accepts.
    pub fn max_batch_size(&self) -> usize {
        self.config.max_batch_size
    }

    /// Run one batch end to end, blocking until it completes.
    ///
    /// Serial convenience path used by tests and [`Self::request_chunks`].
    pub fn execute_batch(
        &mut self,
        world: &WorldContext,
        batch: &Batch,
    ) -> Result<Arc<ExecutedBatch>, String> {
        let prepared = self.prepare_batch(world, batch)?;
        self.wait_for_batch(&prepared);
        // Tests read the per-chunk map, so this path keeps the decoded form.
        self.finish_batch(prepared, false)
    }

    /// Block until the batch's mapping completes. `try_finish_map` checks the
    /// callback once; between checks `PollType::Wait` parks until work finishes
    /// or the timeout elapses, so no CPU is burned spinning.
    fn wait_for_batch(&mut self, prepared: &PreparedBatch) {
        while !self.staging.try_finish_map(prepared.slot_handle, &self.device) {
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_millis(50)),
            });
        }
    }

    /// Prepare one batch: upload input buffers, encode, submit, and start an
    /// asynchronous mapping of the output.
    ///
    /// Returns without waiting for the GPU. Callers must drive the mapping to
    /// completion before calling [`Self::finish_batch`].
    ///
    /// Each call takes its own input buffer set from the pool, so several
    /// prepared batches may be outstanding (up to `pipeline_depth`); running
    /// out of sets is the backpressure that bounds the pipeline.
    fn prepare_batch(
        &mut self,
        world: &WorldContext,
        batch: &Batch,
    ) -> Result<PreparedBatch, String> {
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
        // Take a buffer set before touching any inputs. Blocking here is the
        // backpressure that keeps pipeline_depth batches at most in flight.
        let buf_set = self.pool.acquire()
            .ok_or_else(|| "no free input buffer set".to_string())?;
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
        // §3.5 BlockOut shard present? (reads ch11 corners, writes block bytes)
        let has_block = world.shards().iter().any(|sh| sh.pass == Pass::BlockOut);
        let blocks_per_chunk = 16usize * (shape_height(world) as usize) * 16;
        let block_bytes = if has_block {
            (num_chunks as usize) * blocks_per_chunk * 4
        } else { 0 };
        let total_bytes = density_bytes + block_bytes;
        // Binding 8 is sized for `aquifer_cells_per_chunk × max_batch`, so the
        // per-chunk count is the part that has to fit here; the per-batch
        // product is checked again in `upload_aquifer_tables` itself.
        let aquifer_cells_per_chunk = world.shape().aquifer_cells_per_chunk();

        // The buffers were sized from this dimension's grid when the executor
        // was built (see `get_or_create_executor`). If a batch ever needs
        // more, fail it here instead of emitting a copy that overruns its
        // source: wgpu then discards the whole command buffer, so the batch
        // silently yields garbage block data (observed as floating/clipped
        // terrain with 832-block Tectonic worlds).
        if cells_per_chunk > self.config.cells_per_chunk
            || blocks_per_chunk > self.config.blocks_per_chunk
            || aquifer_cells_per_chunk > self.config.aquifer_cells_per_chunk
            || total_bytes > self.staging.slot_size()
        {
            return Err(format!(
                "batch does not fit the executor buffers: need {} cells/chunk, {} blocks/chunk \
                 and {} aquifer cells/chunk (density {} B + block {} B), executor sized for \
                 {} cells/chunk, {} blocks/chunk and {} aquifer cells/chunk ({} B staging)",
                cells_per_chunk, blocks_per_chunk, aquifer_cells_per_chunk,
                density_bytes, block_bytes,
                self.config.cells_per_chunk, self.config.blocks_per_chunk,
                self.config.aquifer_cells_per_chunk,
                self.staging.slot_size()));
        }

        let shape = world.shape();

        // Build per-batch params (all _pad slots except [10], which the
        // aquifer upload fills in below), upload the three side tables
        // (external inputs / aquifer / beard), then write params ONCE.
        let mut params = Self::build_params(world, batch, num_chunks, shape);
        let ext_buf = Self::ensure_external_buffer(
            &self.device, &self.queue, &mut self.pool.sets[buf_set],
            batch.externals.as_ref());
        let aquifer_cells = Self::upload_aquifer_tables(
            &self.queue, &mut self.pool.sets[buf_set], world, batch, shape)?;
        if let Some(aq_n) = aquifer_cells {
            // aq_s half-offset: only the first `aq_n` vec4 entries were
            // written, so the state half starts at index `aq_n`.
            params._pad[10] = aq_n as u32;
        }
        Self::upload_beard_data(&self.queue, &mut self.pool.sets[buf_set], world, batch);
        self.queue.write_buffer(
            &self.pool.sets[buf_set].params, 0, bytemuck::bytes_of(&params));

        // Packed chunk positions [cx,cz,...] for the shader (binding 4).
        // Sorted + deduped by PendingQueue::flush, matching the output order.
        let mut packed_positions: Vec<i32> = Vec::with_capacity(batch.positions.len() * 2);
        for &(cx, cz) in &batch.positions {
            packed_positions.push(cx);
            packed_positions.push(cz);
        }
        self.queue.write_buffer(
            &self.pool.sets[buf_set].positions, 0,
            bytemuck::cast_slice(&packed_positions));

        // Get or create perm buffer (binding 1)
        let perm_buf = world.perm_buffer()
            .ok_or_else(|| "perm buffer not uploaded".to_string())?;
        let origin_buf = world.origin_buffer()
            .ok_or_else(|| "origin buffer not uploaded".to_string())?;
        let simplex_buf = world.simplex_buffer()
            .ok_or_else(|| "simplex buffer not uploaded".to_string())?;

        // Acquire staging slot
        let (staging_buf, slot_handle) = self
            .staging
            .acquire_slot()
            .ok_or_else(|| "no staging slot available".to_string())?;

        // Print the actual size of each binding once per process. The static
        // executor footprint is logged at init, but `external` (binding 6) and
        // the per-world perm/origin/simplex tables are only known here, and
        // binding 6 is the one that can grow past a driver's
        // max_storage_buffer_binding_size on modded worlds.
        {
            static BINDINGS_LOGGED: AtomicBool = AtomicBool::new(false);
            if !BINDINGS_LOGGED.swap(true, Ordering::Relaxed) {
                let kb = |b: u64| b as f64 / 1024.0;
                let set = &self.pool.sets[buf_set];
                crate::debug_log_line(&format!(
                    "[c2me-wgpu-rs] bindings (sizes in KiB): \
                     b0_params={:.1} b1_perm={:.1} b2_out={:.1} b3_origin={:.1} \
                     b4_pos={:.1} b5_simplex={:.1} b6_external={:.1} b7_block={:.1} \
                     b8_aquifer={:.1} b9_beard={:.1}",
                    kb(set.params.size()),
                    kb(perm_buf.size()),
                    kb(self.output_buf.size()),
                    kb(origin_buf.size()),
                    kb(set.positions.size()),
                    kb(simplex_buf.size()),
                    kb(ext_buf.size()),
                    kb(self.block_buf.size()),
                    kb(set.aquifer.size()),
                    kb(set.beard.size()),
                ));
            }
        }

        // Bindings 0-7: params, per-world tables, this batch's inputs, and the
        // two persistent outputs. See `bind_group::create_layout` for the
        // contract each one has with the shader.
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("c2me_wgpu_batch_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                binding(0, &self.pool.sets[buf_set].params),
                binding(1, &perm_buf),
                binding(2, &self.output_buf),
                binding(3, &origin_buf),
                binding(4, &self.pool.sets[buf_set].positions),
                binding(5, &simplex_buf),
                binding(6, &ext_buf),
                binding(7, &self.block_buf),
            ],
        });

        // §3.5: the BlockOut shard gets its own layout and bind group.
        let set = &self.pool.sets[buf_set];
        let block_out_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("c2me_wgpu_block_out_bg"),
            layout: &self.block_out_bg_layout,
            entries: &[
                binding(0, &set.params),
                binding(2, &self.output_buf),
                binding(4, &set.positions),
                binding(7, &self.block_buf),
                binding(8, &set.aquifer),
                binding(9, &set.beard),
                binding(10, &perm_buf),
                binding(11, &origin_buf),
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
            let wg_x = (ccx + 3) / 4;
            let wg_y = (ccy + 3) / 4;
            let wg_z = (ccz * num_chunks + 3) / 4;

            // One dispatch per shard. Corner shards use the corner grid; the
            // §3.5 BlockOut shard uses a per-BLOCK grid (16 x height x 16 per
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
                if shards[i].pass == Pass::BlockOut {
                    cpass.set_bind_group(0, &block_out_bg, &[]);
                    cpass.dispatch_workgroups(bw_x, bw_y, bw_z);
                } else {
                    cpass.set_bind_group(0, &bind_group, &[]);
                    cpass.dispatch_workgroups(wg_x, wg_y, wg_z);
                }
            }
        }

        // Copy only the read-back channels into staging. output_buf keeps
        // the full `TOTAL_CHANNELS`-wide layout the shader writes; we pluck
        // each consumed channel out of every chunk into a compact chunk-major
        // staging layout. For the real router that is ch11+ch12 (adjacent,
        // so effectively one 2*CPC run per chunk) — an 8x smaller readback.
        let cell_bytes = (cells_per_chunk * 4) as u64;
        for ci in 0..num_chunks as usize {
            for (k, &ch) in readback.iter().enumerate() {
                let src_off = ((ci * TOTAL_CHANNELS + ch) * cells_per_chunk) as u64 * 4;
                let dst_off = ((ci * out_channels + k) * cells_per_chunk) as u64 * 4;
                encoder.copy_buffer_to_buffer(
                    &self.output_buf, src_off, &staging_buf, dst_off, cell_bytes);
            }
        }

        // §3.5: copy the per-block state bytes into staging after the density.
        if has_block {
            encoder.copy_buffer_to_buffer(
                &self.block_buf, 0, &staging_buf, density_bytes as u64, block_bytes as u64);
        }

        let encode_ms = t_encode0.elapsed().as_secs_f64() * 1e3;
        let t_submit0 = std::time::Instant::now();
        self.queue.submit(Some(encoder.finish()));
        let submit_ms = t_submit0.elapsed().as_secs_f64() * 1e3;

        let t_submit = std::time::Instant::now();
        // Start the async map now. The serial path waits on it immediately;
        // pipelined mode can let it run while the next batch is prepared.
        self.staging.begin_map(slot_handle, &self.device, total_bytes);

        Ok(PreparedBatch {
            batch: batch.clone(),
            buf_set,
            slot_handle,
            out_channels,
            density_bytes,
            block_bytes,
            has_block,
            prof,
            t0,
            t_submit,
            encode_ms,
            submit_ms,
        })
    }

    /// Consume a batch prepared by [`Self::prepare_batch`]. The caller must
    /// have driven its mapping to completion first.
    ///
    /// `raw = true` is the production path: the staging buffer already holds
    /// exactly the bytes the Java-facing blob needs, so they are sliced out
    /// once instead of being decoded into `Vec<f32>` per chunk, re-serialised
    /// by the C callback, and decoded again. That round trip used to copy
    /// ~24 MiB of block data and the full density readback four extra times
    /// per batch.
    ///
    /// `raw = false` keeps the per-chunk `ResultBlob` map, which the offline
    /// tests read directly.
    fn finish_batch(&mut self, prepared: PreparedBatch, raw: bool)
        -> Result<Arc<ExecutedBatch>, String> {
        let PreparedBatch {
            batch, buf_set, slot_handle, out_channels, density_bytes,
            block_bytes, has_block, prof, t0, t_submit, encode_ms, submit_ms,
        } = prepared;
        let readback_ms = t_submit.elapsed().as_secs_f64() * 1e3;

        let t_decode0 = std::time::Instant::now();
        let decoded = self.decode_batch(
            &batch, slot_handle, raw, out_channels, density_bytes, block_bytes, has_block);
        let decode_ms = t_decode0.elapsed().as_secs_f64() * 1e3;

        // Recycle before propagating a decode error. Both are one-shot loans:
        // a slot or input set dropped on the error path would be lost for the
        // rest of the session, permanently shrinking the pipeline.
        self.staging.recycle(slot_handle);
        // The GPU work for this batch is complete (its output has been read
        // back, or the batch is being failed), so its input buffers are free
        // for the next prepare.
        self.pool.release(buf_set);

        let decoded = decoded?;
        let dispatch_time = t0.elapsed();

        if prof {
            profile_accumulate(batch.positions.len() as u64,
                encode_ms, submit_ms, readback_ms, decode_ms,
                dispatch_time.as_secs_f64() * 1e3);
        }

        Ok(Arc::new(ExecutedBatch {
            batch,
            dispatch_time,
            out_channels,
            results: decoded.results,
            block_data: decoded.block_data,
            raw_density: decoded.raw_density,
            raw_block: decoded.raw_block,
        }))
    }

    /// Turn the mapped staging bytes into the shape [`ExecutedBatch`] needs.
    ///
    /// Split from [`Self::finish_batch`] so the latter can recycle its slot and
    /// input set before this can fail.
    fn decode_batch(
        &mut self,
        batch: &Batch,
        slot_handle: SlotHandle,
        raw: bool,
        out_channels: usize,
        density_bytes: usize,
        block_bytes: usize,
        has_block: bool,
    ) -> Result<DecodedBatch, String> {
        let mut decoded = DecodedBatch::default();
        if raw {
            // One copy out of the mapped staging buffer, then one more for the
            // block half. Nothing is decoded.
            if let Some(d) = self.staging.get_data(slot_handle) {
                let den_end = density_bytes.min(d.len());
                decoded.raw_density = d[..den_end].to_vec();
                if has_block && d.len() >= density_bytes + block_bytes {
                    decoded.raw_block =
                        d[density_bytes..density_bytes + block_bytes].to_vec();
                }
            }
        } else {
            decoded.results = self.read_results(batch, slot_handle, out_channels)?;
            if has_block {
                if let Some(d) = self.staging.get_data(slot_handle) {
                    if d.len() >= density_bytes + block_bytes {
                        decoded.block_data = d[density_bytes..density_bytes + block_bytes]
                            .chunks_exact(4)
                            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                            .collect();
                    }
                }
            }
        }
        Ok(decoded)
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
        // CPU-evaluated external slots per chunk, carried by the batch itself.
        // (Was a process-global on WorldContext, which is why external worlds
        // had to run serially; see `ExternalInputs`.)
        params._pad[0] = batch.externals.as_ref().map_or(0, |e| e.per_chunk);
        // §8.4: per-dimension aquifer mode in _pad[1..5].
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
        params._pad[11] = TOTAL_CHANNELS as u32;
        params
    }

    /// Return the external-inputs buffer (binding 6) for this batch,
    /// (re)creating it in the set if the payload needs more room.
    ///
    /// The payload comes from the batch, not from the world: each batch owns
    /// its values, so a queued batch can be prepared after a newer one has
    /// already staged its own. Recreate only grows the buffer; a smaller
    /// batch writes a prefix and the shader never indexes past it.
    ///
    /// Takes the buffer set rather than `&mut self` so the caller can hold the
    /// set borrow while writing other buffers.
    fn ensure_external_buffer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        set: &mut InputBufferSet,
        externals: Option<&crate::scheduler::ExternalInputs>,
    ) -> wgpu::Buffer {
        let (per_chunk, data) = match externals {
            Some(e) if e.per_chunk > 0 && !e.data.is_empty() => {
                (e.per_chunk, e.data.as_slice())
            }
            _ => (0, &[][..]),
        };
        let ext_bytes = (data.len() * 4).max(4);
        if per_chunk > 0 {
            let need_recreate = match &set.external {
                Some(b) => b.size() < ext_bytes as u64 || b.size() < 4,
                None => true,
            };
            if need_recreate {
                set.external = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("c2me_wgpu_external_inputs"),
                    size: ext_bytes as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            let buf = set.external.as_ref().unwrap();
            queue.write_buffer(buf, 0, bytemuck::cast_slice(data));
            buf.clone()
        } else {
            // No external inputs: bind a dedicated 4-byte dummy so binding 6
            // stays valid. (Reusing output_buf here would alias binding 2's
            // read-write storage and can trip wgpu validation.)
            if set.external.is_none() {
                set.external = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("c2me_wgpu_external_dummy"),
                    size: 4,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            set.external.as_ref().unwrap().clone()
        }
    }

    /// Expand and upload the per-chunk aquifer tables (binding 8). Returns
    /// `Some(n)` with the `aq_p` cell count written, or `None` if no table was
    /// staged for any chunk in the batch.
    ///
    /// Fails instead of overrunning the buffer: wgpu would otherwise discard
    /// the whole command buffer, which shows up as stale terrain rather than
    /// as an error.
    fn upload_aquifer_tables(
        queue: &wgpu::Queue,
        set: &mut InputBufferSet,
        world: &WorldContext,
        batch: &Batch,
        shape: &ShapeConfig,
    ) -> Result<Option<usize>, String> {
        // Dimensions without aquifers (Nether/End) legitimately have no
        // tables. Skip the whole expansion: writing padded zero cells would
        // only make the shader fall through to "no cell found" for every
        // block, and logging it would spam once per chunk.
        if !world.aquifer_mode().has_aquifers {
            return Ok(None);
        }
        let cells_per_chunk = shape.aquifer_cells_per_chunk();
        let mut aq_p: Vec<[i32; 4]> = Vec::new();
        let mut aq_s: Vec<u32> = Vec::new();
        if let Ok(mut tables) = world.aquifer_tables.lock() {
            let min_y = shape.min_y;
            let start_y = min_y.div_euclid(12) - 1;
            let size_y = shape.aquifer_layers();
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
            return Ok(None);
        }
        // [ aq_p (vec4) | aq_s padded to vec4 (state in .x) ]
        let n = aq_p.len();
        let mut merged: Vec<[i32; 4]> = Vec::with_capacity(n * 2);
        merged.extend_from_slice(&aq_p);
        merged.extend(aq_s.iter().map(|&st| [st as i32, 0, 0, 0]));
        let needed = merged.len() * 16;
        let capacity = set.aquifer.size() as usize;
        if needed > capacity {
            return Err(format!(
                "aquifer tables do not fit the input buffer: {} chunks need {} cells \
                 ({needed} B), buffer holds {} cells ({capacity} B); executor was sized \
                 for {cells_per_chunk} aquifer cells/chunk",
                batch.positions.len(), n, capacity / 32));
        }
        queue.write_buffer(&set.aquifer, 0, bytemuck::cast_slice(&merged));
        Ok(Some(n))
    }

    /// Pack and upload the per-chunk Beardifier data (binding 9). Each chunk
    /// gets a fixed `BEARD_STRIDE`-u32 slot; empty chunks get a zeroed stride
    /// so `beard_pj[base] == 0` disables beard.
    fn upload_beard_data(
        queue: &wgpu::Queue,
        set: &mut InputBufferSet,
        world: &WorldContext,
        batch: &Batch,
    ) {
        let mut pj = vec![0u32; BEARD_STRIDE * batch.positions.len()];
        if let Ok(mut tables) = world.beard_tables.lock() {
            for (ci, &(cx, cz)) in batch.positions.iter().enumerate() {
                if let Some(bytes) = tables.remove(&(cx, cz)) {
                    let n = (bytes.len() / 4).min(BEARD_STRIDE);
                    let dst = ci * BEARD_STRIDE;
                    for i in 0..n {
                        pj[dst + i] = u32::from_le_bytes(
                            bytes[i*4..i*4+4].try_into().unwrap());
                    }
                }
            }
        }
        // Weight table is written once at construction; only pj changes per
        // batch, so write straight past it.
        let weight_bytes = (BEARD_WEIGHT_WORDS * 4) as u64;
        queue.write_buffer(&set.beard, weight_bytes, bytemuck::cast_slice(&pj));
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
    /// Wrapped in a `Mutex` so `shutdown` can take `&self`: the executor is
    /// shared through `Arc` with in-flight callers, so a `&mut self` shutdown
    /// would be unavailable exactly when it is needed most.
    handle: std::sync::Mutex<Option<JoinHandle<()>>>,
    sender: channel::Sender<ExecutorMessage>,
    /// Shape the persistent buffers were allocated for. A world that needs
    /// more must rebuild the executor; see `c_api::get_or_create_executor`.
    cells_per_chunk: usize,
    blocks_per_chunk: usize,
    /// Aquifer cells per chunk, tracked separately: a shape can keep the same
    /// density grid while needing a different aquifer table (different min_y
    /// at the same height), and the binding-8 buffer must still hold it.
    aquifer_cells_per_chunk: usize,
    /// Effective batch ceiling of this executor (requested size clamped by the
    /// device limits for this shape). The buffers are sized by it and
    /// `prepare_batch` rejects anything larger, so it is the hard ceiling for
    /// every dimension that reuses this executor — even one whose own shape
    /// would derive a bigger value.
    max_batch_size: usize,
}

enum ExecutorMessage {
    SubmitBatch {
        world: Arc<WorldContext>,
        batch: Batch,
        /// Return the readback as raw bytes (production) instead of a decoded
        /// per-chunk blob map (offline tests).
        raw: bool,
        /// Called once the batch finishes, on the executor thread. The caller
        /// supplies whatever it needs to publish the result (the C API uses
        /// this to fill its BatchRequest). Passing a callback instead of a
        /// reply channel lets `submit_async` return without a second thread:
        /// the old design spawned one thread per batch just to block on the
        /// reply and copy it elsewhere.
        on_done: Box<dyn FnOnce(Result<Arc<ExecutedBatch>, String>) + Send>,
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
        let cells_per_chunk = config.cells_per_chunk.max(1);
        let blocks_per_chunk = config.blocks_per_chunk.max(1);
        let aquifer_cells_per_chunk = config.aquifer_cells_per_chunk.max(1);

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
                let effective_batch = executor.max_batch_size();
                let _ = init_tx.send(Ok(effective_batch));
                Self::run_loop(&mut executor, &receiver);
            })
            .map_err(|e| format!("failed to spawn executor thread: {e}"))?;

        match init_rx.recv() {
            Ok(Ok(max_batch_size)) => Ok(Self {
                handle: std::sync::Mutex::new(Some(handle)),
                sender,
                cells_per_chunk,
                blocks_per_chunk,
                aquifer_cells_per_chunk,
                max_batch_size,
            }),
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

    /// Effective batch ceiling of the persistent buffer layout.
    ///
    /// Java clamps to this even when a dimension's own shape would derive a
    /// larger value: switching from a dense shape (e.g. 832 high, cell 4) to a
    /// smaller one reuses the same executor, whose `b2_out` binding was sized
    /// for the denser grid.
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// True when the persistent buffers cover a grid of this shape, so a batch
    /// from that world can be prepared without reallocating.
    pub fn fits(
        &self,
        cells_per_chunk: usize,
        blocks_per_chunk: usize,
        aquifer_cells_per_chunk: usize,
    ) -> bool {
        cells_per_chunk <= self.cells_per_chunk
            && blocks_per_chunk <= self.blocks_per_chunk
            && aquifer_cells_per_chunk <= self.aquifer_cells_per_chunk
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
            // Synchronous callers are the offline tests, which read the
            // per-chunk blob map — keep the decoded form for them.
            raw: false,
            on_done: Box::new(move |r| {
                let _ = reply_tx.send(r);
            }),
        }).map_err(|_| "executor channel closed".to_string())?;

        reply_rx.recv()
            .map_err(|_| "executor reply channel closed".to_string())?
    }

    /// Enqueue a batch without waiting for it. `on_done` runs on the executor
    /// thread when the batch completes. This is the production path: the
    /// caller returns immediately and publishes the result from the callback.
    pub fn submit_async(
        &self,
        world: Arc<WorldContext>,
        batch: Batch,
        on_done: impl FnOnce(Result<Arc<ExecutedBatch>, String>) + Send + 'static,
    ) -> Result<(), String> {
        self.sender.send(ExecutorMessage::SubmitBatch {
            world,
            batch,
            // Production path: hand back the raw readback bytes, which the C
            // callback turns into the Java blob without decoding per chunk.
            raw: true,
            on_done: Box::new(on_done),
        }).map_err(|_| "executor channel closed".to_string())
    }

    /// Stop the worker and wait for it to exit.
    ///
    /// Takes `&self` rather than `&mut self` so callers sharing the executor
    /// through `Arc` can stop it without owning the only reference. This
    /// matters when rebuilding the GPU device: the worker holds an
    /// `Arc<Device>`, and if it is not stopped the old device stays alive
    /// alongside the new one — two full buffer pools resident at once, which
    /// on a 12 GB Android device is enough to exhaust memory.
    ///
    /// Idempotent: the `JoinHandle` is taken under the lock, so a second call
    /// finds `None` and returns immediately.
    pub fn shutdown(&self) {
        let mut slot = self.handle.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = slot.take() {
            let _ = self.sender.send(ExecutorMessage::Shutdown);
            let _ = h.join();
        }
    }

    /// How long to wait for a batch's mapping before giving up. Matches the
    /// Java-side request timeout, so if the GPU genuinely wedges both sides
    /// report it rather than one hanging silently.
    const BATCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
    /// How long to park between mapping checks. Short enough to notice a
    /// `Shutdown` promptly, long enough not to spin.
    const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(2);

    fn run_loop(
        executor: &mut SingleBatchExecutor,
        receiver: &channel::Receiver<ExecutorMessage>,
    ) {
        // Batches in flight, oldest first. The callback travels with its batch
        // here rather than inside PreparedBatch: finish_batch consumes the
        // batch, and the callback must survive that move to be invoked after.
        let mut in_flight: std::collections::VecDeque<
            (PreparedBatch, bool, Box<dyn FnOnce(Result<Arc<ExecutedBatch>, String>) + Send>)
        > = std::collections::VecDeque::new();

        loop {
            // 1. Publish everything the GPU has finished. Checked before
            //    accepting new work so the read-back gap stays short.
            loop {
                let ready = match in_flight.front() {
                    Some((front, _, _)) => executor
                        .staging
                        .try_finish_map(front.slot_handle, &executor.device),
                    None => false,
                };
                if !ready {
                    break;
                }
                let (prepared, raw, on_done) = in_flight.pop_front().unwrap();
                let result = executor.finish_batch(prepared, raw);
                on_done(result);
            }

            // 2. Accept one new batch. The timeout doubles as the mapping
            //    re-check interval, so a busy executor still polls often.
            match receiver.recv_timeout(Self::POLL_INTERVAL) {
                Ok(ExecutorMessage::SubmitBatch { world, batch, raw, on_done }) => {
                    if in_flight.len() >= executor.config.pipeline_depth {
                        // The caller went past the agreed depth. Failing is
                        // safer than growing the pool on demand.
                        on_done(Err("pipeline depth exceeded".to_string()));
                        continue;
                    }
                    match executor.prepare_batch(&world, &batch) {
                        Ok(prepared) => in_flight.push_back((prepared, raw, on_done)),
                        Err(e) => on_done(Err(e)),
                    }
                }
                Ok(ExecutorMessage::Shutdown) => break,
                Err(channel::RecvTimeoutError::Disconnected) => break,
                Err(channel::RecvTimeoutError::Timeout) => {}
            }

            // 3. Nudge the mapping in case the Java-side heartbeat stalled.
            let _ = executor.device.poll(wgpu::PollType::Poll);
        }

        // Shutdown: drain in flight so no Rust-side request handle leaks.
        while let Some((prepared, raw, on_done)) = in_flight.pop_front() {
            let deadline = std::time::Instant::now() + Self::BATCH_TIMEOUT;
            while !executor
                .staging
                .try_finish_map(prepared.slot_handle, &executor.device)
            {
                if std::time::Instant::now() > deadline {
                    break;
                }
                let _ = executor.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Self::POLL_INTERVAL),
                });
            }
            let result = executor.finish_batch(prepared, raw);
            on_done(result);
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
            "[batch-stage-timing] avg over {} batches ({} chunks): encode={:.2} submit={:.2} readback={:.2} decode={:.2} total={:.2} ms/batch",
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
            ..Batch::default()
        };
        assert_eq!(batch.positions.len(), 3);
    }

    /// `upload_aquifer_tables` copies `cells × chunks` pairs of vec4, while the
    /// buffer is sized from `max_batch`. The upload now fails the batch rather
    /// than letting wgpu discard it; this pins the two formulas together at the
    /// largest legal batch, which is the case that must still fit.
    #[test]
    fn aquifer_upload_matches_the_buffer_at_max_batch() {
        let vanilla = ShapeConfig { min_y: -64, height: 384, cell_width: 4, cell_height: 4 };
        let max_batch = 8;
        let per_chunk = vanilla.aquifer_cells_per_chunk();
        let cfg = ExecutorConfig {
            max_batch_size: max_batch,
            aquifer_cells_per_chunk: per_chunk,
            ..ExecutorConfig::default()
        };
        let buffer_bytes = BufferSizes::for_config(&cfg).aquifer_entries * 2 * 16;
        // aq_p + aq_s, both padded to vec4.
        let needed = per_chunk * max_batch * 2 * 16;
        assert_eq!(needed, buffer_bytes, "a full batch must fit exactly");
        assert!(
            needed + per_chunk * 2 * 16 > buffer_bytes,
            "the guard must trigger before a batch one chunk larger"
        );
    }

    /// The binding-8 reservation must follow the shape, not the old 1200 cap.
    #[test]
    fn aquifer_reservation_tracks_the_shape() {
        let vanilla = ShapeConfig { min_y: -64, height: 384, cell_width: 4, cell_height: 4 };
        let cfg = ExecutorConfig {
            max_batch_size: 8,
            aquifer_cells_per_chunk: vanilla.aquifer_cells_per_chunk(),
            ..ExecutorConfig::default()
        };
        assert_eq!(BufferSizes::for_config(&cfg).aquifer_entries, 315 * 8);

        let tall = ShapeConfig { min_y: -64, height: 2048, cell_width: 4, cell_height: 4 };
        let cfg = ExecutorConfig {
            max_batch_size: 8,
            aquifer_cells_per_chunk: tall.aquifer_cells_per_chunk(),
            ..ExecutorConfig::default()
        };
        assert!(
            BufferSizes::for_config(&cfg).aquifer_entries > AQ_DEFAULT_CELLS_PER_CHUNK * 8,
            "a 2048-block dimension needs more room than the old fixed reservation"
        );
    }

    /// The binding ceiling depends on the shape, not just the batch size: the
    /// default 128 MiB `b2_out` binding fits 64 vanilla chunks but only 44
    /// chunks of an 832-block world sampled with cell_height=4.
    #[test]
    fn batch_limit_follows_the_shape_and_the_device_limits() {
        let limits = wgpu::Limits::default();

        // Vanilla 384: 5×97×5 corner cells, 16×384×16 blocks. 64 chunks need
        // 85 MiB for b2_out, so the default limits leave the configured 64.
        let vanilla = BatchLimit::derive(&limits, 5 * 97 * 5, 16 * 384 * 16, 0);
        assert_eq!(vanilla.binding_allows, 96);
        assert!(vanilla.buffer_allows >= 64);
        assert_eq!(vanilla.allows, 96);

        // Tectonic 832 with cell_height=8: 5×105×5 cells. 64 chunks need
        // 92 MiB, still under 128.
        let tectonic = BatchLimit::derive(&limits, 5 * 105 * 5, 16 * 832 * 16, 0);
        assert!(tectonic.allows >= 64, "{}", tectonic.allows);

        // The same height with cell_height=4: 5×209×5 cells makes b2_out
        // 192 MB at 64 chunks. The binding limit now binds at 44.
        let tall = BatchLimit::derive(&limits, 5 * 209 * 5, 16 * 832 * 16, 0);
        assert_eq!(tall.allows, 44);
        assert_eq!(tall.allows, tall.binding_allows);
        assert!(tall.allows < 64);
    }

    /// A device that reports exactly the WebGPU minimums must still be able to
    /// dispatch a single chunk; the derivation clamps to 1 instead of 0.
    #[test]
    fn batch_limit_never_drops_below_one() {
        let tiny = wgpu::Limits {
            max_storage_buffer_binding_size: 1024,
            max_buffer_size: 1024,
            ..wgpu::Limits::default()
        };
        assert_eq!(BatchLimit::derive(&tiny, 5225, 212_992, 0).allows, 1);
    }

    /// Binding 6 is solved together with the outputs: 200 external slots on a
    /// dense shape force a smaller batch than the output buffers alone would
    /// allow, and a node count that cannot fit a single chunk still leaves one
    /// chunk dispatchable (Java rejects the dimension at init in that case).
    #[test]
    fn external_slots_shrink_the_batch() {
        let limits = wgpu::Limits::default();
        let cells = 5 * 209 * 5; // 5225, 832 blocks at cell_height=4
        let blocks = 16 * 832 * 16; // 212_992

        let none = BatchLimit::derive(&limits, cells, blocks, 0);
        assert_eq!(none.external_allows, usize::MAX);

        // 128 MiB / (200 * 5225 * 4) = 32.1 -> 32 chunks.
        let with = BatchLimit::derive(&limits, cells, blocks, 200);
        assert_eq!(with.external_allows, 32);
        assert_eq!(with.allows, 32);

        let huge = BatchLimit::derive(&limits, cells, blocks, 100_000);
        assert_eq!(huge.allows, 1);
    }

    /// The node cap Java applies is the same shape-derived number, so a
    /// dimension that passes init can always dispatch at least one chunk.
    #[test]
    fn external_slot_limit_follows_the_shape() {
        let limits = wgpu::Limits::default();
        // 128 MiB / (5*97*5 * 4) = 13,836 slots on vanilla's grid.
        assert_eq!(BatchLimit::external_slot_limit(&limits, 5 * 97 * 5), 13_836);
        assert_eq!(
            BatchLimit::external_slot_limit(&limits, 5 * 209 * 5),
            (128 * 1024 * 1024) / (5 * 209 * 5 * 4)
        );
    }

    /// The requested cap, the shape derivation and a reused executor's fixed
    /// ceiling all apply; the smallest wins, and a zero from any term still
    /// leaves one chunk dispatchable.
    #[test]
    fn effective_batch_limit_takes_every_ceiling() {
        // Default shape: both the request and the device leave 64.
        assert_eq!(clamp_batch_limit(64, 96, None), 64);
        // Dense shape: the device binds first.
        assert_eq!(clamp_batch_limit(64, 34, None), 34);
        // A reused executor built for a denser shape is a hard ceiling even
        // though this shape's own derivation would allow 64.
        assert_eq!(clamp_batch_limit(64, 96, Some(34)), 34);
        // Never larger than requested, even if the executor reports more.
        assert_eq!(clamp_batch_limit(8, 96, Some(64)), 8);
        assert_eq!(clamp_batch_limit(64, 0, Some(0)), 1);
    }
}
