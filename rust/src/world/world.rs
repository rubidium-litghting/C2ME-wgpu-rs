// SPDX-License-Identifier: MIT

//! Per-Minecraft-world context: AST + optimized WGSL + compiled pipeline
//! + the bind-group layout for params/permutations/output-buffers.

use crate::Result;
use crate::ast::AstNode;
use crate::ast::deserialize::{PermEntry, ShapeConfig, deserialize_full_with_perm};
use crate::ast::equality::hash_ast;
use crate::ast::opto;
#[cfg(debug_assertions)]
use crate::codegen::validate::validate_wgsl;
use crate::codegen::wgsl_gen;
use crate::world::bind_group;
use crate::world::constants::PermutationTableSet;
use crate::world::pipeline_cache;
use wgpu::util::DeviceExt;

pub struct WorldContext {
    /// Parsed + optimized AST root (16-channel Root) — kept for fingerprinting
    /// and future shader hot-reload.
    #[allow(dead_code)]
    pub(crate) ast_root: AstNode,
    /// Parsed shape config (min_y / height / cell_*).
    pub(crate) shape: ShapeConfig,
    /// Biome tail from end of blob (currently `AstNode::BiomeTree { _placeholder }`
    /// with no further payload decoded). See plan.txt §4.4 for future protocol.
    #[allow(dead_code)]
    pub(crate) biome_source: AstNode,
    /// 32-bit fingerprint = hash(ast). Used for pipeline_cache lookups.
    pub(crate) fingerprint: u32,
    /// Channel shards (see `wgsl_gen::Shard`). Every shard is a separate
    /// WGSL module + pipeline + dispatch, but all write `out_buf` at the
    /// channel's global index, so the readback layout is unchanged.
    pub(crate) shards: Vec<wgsl_gen::Shard>,
    /// Emitted WGSL source per shard, parallel to `shards`.
    pub(crate) shader_srcs: Vec<String>,
    /// Permutation tables for all `PerlinSampler` seeds discovered in the
    /// AST, packed into one flat `Vec<i32>` ready to upload to
    /// `@group(0) @binding(1)`.
    pub(crate) perm_table_set: PermutationTableSet,
    /// Real Perlin permutation tables shipped by the Java side in the
    /// PERM_RESOURCE trailer. Stored for the upcoming noise emitters;
    /// not yet consumed by WGSL generation.
    #[allow(dead_code)]
    pub(crate) perm_entries: Vec<PermEntry>,
    pub(crate) simplex_entries: Vec<crate::ast::deserialize::SimplexEntry>,
    /// Compiled pipelines per shard, parallel to `shards`. Empty until
    /// `compile_pipeline` is called. Cached globally in
    /// `pipeline_cache::IN_MEMORY_CACHE` keyed by `(fingerprint, pass_id)`.
    pub(crate) pipelines: Vec<wgpu::ComputePipeline>,
    /// GPU-side permutation buffer (binding 1). Created in
    /// `compile_pipeline`, kept alive for the lifetime of this
    /// `WorldContext` so the scheduler can bind it without re-uploading.
    pub(crate) perm_buffer: Option<wgpu::Buffer>,
    /// SimplexNoiseSampler tables for EndIslands (binding 5).
    pub(crate) simplex_buffer: Option<wgpu::Buffer>,
    /// §23: static 24^3 Beardifier weight table (binding 10), uploaded once.
    pub(crate) beard_weight_buffer: Option<wgpu::Buffer>,
    /// GPU-side per-entry origin buffer (binding 3), f32 x3 per entry.
    pub(crate) origin_buffer: Option<wgpu::Buffer>,
    /// Number of CPU-evaluated (external) subtrees in this world's AST.
    pub(crate) external_count: std::sync::atomic::AtomicUsize,
    /// CPU-computed values for those subtrees, laid out
    /// [chunk][ext][cell]. Uploaded to binding 6 before each dispatch.
    pub(crate) external_data: std::sync::Mutex<Vec<f32>>,
    /// §22 Step B: per-chunk aquifer tables, keyed by (cx, cz). Each value is
    /// the compact Java-serialized table (magic + n + seedLo/Hi + per-cell
    /// lvl_y/state). Set via `c2me_wgpu_set_aquifer_table` before a request.
    pub(crate) aquifer_tables: std::sync::Mutex<std::collections::HashMap<(i32, i32), Vec<u8>>>,
    /// §23: per-chunk Beardifier pieces/junctions, same stage pattern as
    /// aquifer. Empty for structure-free chunks (beard contributes 0).
    pub(crate) beard_tables: std::sync::Mutex<std::collections::HashMap<(i32, i32), Vec<u8>>>,
    /// §22 fix: per-dimension aquifer mode. Dimensions without aquifers
    /// (Nether/End) skip the 12-cell scan and use sea-level semantics.
    pub(crate) aquifer_mode: std::sync::Mutex<AquiferMode>,
    /// §14: finalDensity (ch11) split into interpolated boundaries (`inners`)
    /// and the remaining per-block tree (`outer`). Empty when no split was
    /// needed (vanilla terrain has no non-linear wrapper outside an interp).
    pub(crate) density_inners: Vec<crate::ast::AstNode>,

    /// Ore-vein RNG seed (seedLo, seedHi) from MC's `oreRandomDeriver`.
    /// Set once per world via `c2me_wgpu_set_ore_seed`; (0,0) means unset.
    pub(crate) ore_seed: std::sync::Mutex<(i64, i64)>,
    /// True once the ore-vein seed has been staged (settings.oreVeins()).
    pub(crate) ore_veins: std::sync::atomic::AtomicBool,
}

/// Aquifer behaviour for one dimension.
#[derive(Clone, Copy)]
pub struct AquiferMode {
    pub has_aquifers: bool,
    pub sea_level: i32,
    /// Fluid state code above sea level: 2=water, 3=lava, 0=air.
    pub default_fluid: u32,
    /// y below this is always lava.
    pub lava_level: i32,
}
impl Default for AquiferMode {
    fn default() -> Self {
        AquiferMode { has_aquifers: true, sea_level: 63, default_fluid: 2, lava_level: -54 }
    }
}

/// Create a GPU buffer pre-filled with `contents`.
///
/// Default path is `create_buffer_init` (`mapped_at_creation`), which is the
/// fastest way to upload and is used on desktop. On **Android** we instead
/// default to `create_buffer` + `queue.write_buffer`: some Mali drivers
/// (observed on Mali-G610 / Dimensity 8100) reject the mapped-at-creation
/// path and abort the process, while Mali-G76 (Kirin 980) accepts it. The
/// write_buffer path costs one extra copy, but only once at init time.
///
/// Override with `debug_mapped_at_creation` in the config (true = mapped,
/// false = write_buffer) for debugging.
fn create_init_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
    use_mapped: bool,
) -> wgpu::Buffer {
    if use_mapped {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage,
        })
    } else {
        // create_buffer requires COPY_DST and a non-zero size.
        let size = contents.len().max(4) as u64;
        crate::wgpu_debug!("create_buffer({label}, {size} bytes)...");
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        crate::wgpu_debug!("create_buffer({label}) ok, write_buffer...");
        queue.write_buffer(&buf, 0, contents);
        crate::wgpu_debug!("write_buffer({label}) done");
        buf
    }
}

impl WorldContext {
    /// Parse the Java-side blob, run AST opto passes, compute fingerprint,
    /// emit WGSL, and **smoke-test validate** the shader via naga (pure CPU).
    ///
    /// Fails fast on any emitter regression that produces invalid WGSL —
    /// long before a real `wgpu::Device::create_compute_pipeline` call
    /// would have caught it.
    ///
    /// Does **not** compile the GPU pipeline or upload any buffers. Call
    /// `compile_pipeline` afterwards with the singleton `WgpuInstance`'s
    /// device to produce a usable `wgpu::ComputePipeline` + perm buffer.
    /// This two-step split keeps `new` GPU-free so unit tests can run
    /// without a real adapter.
    pub fn new(config_blob: &[u8]) -> Result<Self> {
        // 1. Deserialize Java blob into AST + shape + biome tail
        let (ast_root, shape, biome_source, perm_entries, simplex_entries) =
            deserialize_full_with_perm(config_blob)?;

        // 2. Run 4-pass optimization pipeline (constant_fold → normalize → cse → dce)
        let ast_root = opto::run_pipeline(ast_root)?;

        // §14 Pass E: pre-split finalDensity (ch11) into per-corner subtrees
        // (`inners`) and a per-block `outer` tree, so the GPU can evaluate
        // non-linear wrappers per block instead of per corner.
        let (density_inners, density_outer, barrier_noise) = match &ast_root {
            crate::ast::AstNode::Root(channels) if channels.len() > 11 => {
                let mut inners = Vec::new();
                let outer = crate::ast::split::split(&channels[11], &mut inners);
                let barrier = if !channels.is_empty() { Some(channels[0].clone()) } else { None };
                (inners, Some(outer), barrier)
            }
            _ => (Vec::new(), None, None),
        };

        // 3. Fingerprint = stable hash of optimized AST (Java ↔ Rust consistency)
        let fingerprint = hash_ast(&ast_root);

        // 4. Emit one WGSL module per channel shard and smoke-test via naga.
        //    Sharding keeps every module under the Mali-G610 driver's safe
        //    compile budget (DEVELOPMENT.md §5.15.5).
        let channel_count = match &ast_root {
            AstNode::Root(chs) => chs.len(),
            _ => 0,
        };
        let shards = wgsl_gen::plan_shards(channel_count);
        let mut shader_srcs: Vec<String> = Vec::with_capacity(shards.len());
        for shard in &shards {
            let src = wgsl_gen::emit_shader_for_shard(&ast_root, *shard, &density_inners, density_outer.as_ref(), barrier_noise.as_ref())?;
            crate::wgpu_debug!(
                "[codegen] shard {:?} [{}..{}) WGSL: {} lines, {} bytes",
                shard.pass, shard.start, shard.end,
                src.lines().count(), src.len()
            );
            // Debug-only smoke test: wgpu's create_shader_module re-runs naga
            // on this same source, so in release builds the extra parse+validate
            // is pure duplication.
            #[cfg(debug_assertions)]
            validate_wgsl(&src)
                .map_err(|e| anyhow::anyhow!("emitted WGSL failed naga validation:\n{e}"))?;
            shader_srcs.push(src);
        }

        // 5. Build the packed permutation-table buffer from the REAL tables
        //    shipped in the Java PERM_RESOURCE trailer. `from_entries` returns
        //    an empty set when the trailer is absent (unit tests / old blobs);
        //    the buffer-upload path below already substitutes a 1-i32 dummy in
        //    that case so the binding stays valid.
        let perm_table_set = PermutationTableSet::from_entries(&perm_entries);

        Ok(Self {
            ast_root,
            shape,
            biome_source,
            fingerprint,
            shards,
            shader_srcs,
            perm_table_set,
            perm_entries,
            simplex_entries,
            pipelines: Vec::new(),
            perm_buffer: None,
            simplex_buffer: None,
            beard_weight_buffer: None,
            origin_buffer: None,
            external_count: std::sync::atomic::AtomicUsize::new(0),
            external_data: std::sync::Mutex::new(Vec::new()),
            aquifer_tables: std::sync::Mutex::new(std::collections::HashMap::new()),
            beard_tables: std::sync::Mutex::new(std::collections::HashMap::new()),
            aquifer_mode: std::sync::Mutex::new(AquiferMode::default()),
            density_inners,
            ore_seed: std::sync::Mutex::new((0, 0)),
            ore_veins: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Compile (or look up) the `wgpu::ComputePipeline` for this world's
    /// optimized AST + emit_shader output, and upload the permutation
    /// table buffer to the GPU.
    ///
    /// Idempotent: subsequent calls return the cached pipeline (cheap Arc
    /// clone) without re-uploading the permutation buffer.
    ///
    /// All ASTs — including those with `CacheInterpolated` — compile a
    /// single pipeline accessible via `pipeline()`. CacheInterpolated
    /// handles corner computation + `workgroupBarrier()` + trilinear lerp
    /// entirely within one dispatch, so no two-pass split is needed.
    pub fn compile_pipeline(&mut self, device: &wgpu::Device) -> Result<()> {
        // Fast path: already compiled.
        if !self.pipelines.is_empty() {
            return Ok(());
        }
        // The queue and upload policy both come from the `WgpuInstance` that
        // owns `device`. Upload policy is cold config: it is fixed for the
        // lifetime of the device it was chosen for.
        let inst = crate::instance::WgpuInstance::instance()?;
        let queue = &inst.queue;
        let use_mapped = inst
            .cold
            .mapped_at_creation
            .unwrap_or(!cfg!(target_os = "android"));
        crate::wgpu_debug!("[compile] compile_pipeline start");

        crate::wgpu_debug!("[compile] creating bind group layout...");
        let bg_layout = bind_group::create_layout(device)?;
        crate::wgpu_debug!("[compile] bind group layout ok, creating pipeline layout...");
        let pipeline_layout = bind_group::create_pipeline_layout(device, &bg_layout)?;
        // §22: BlockOut uses its own (smaller) layout — 6 storage buffers,
        // under the 8-per-stage limit.
        let bo_bg_layout = bind_group::create_block_out_layout(device)?;
        let bo_pipeline_layout = bind_group::create_pipeline_layout(device, &bo_bg_layout)?;
        crate::wgpu_debug!("[compile] pipeline layout ok, compiling pipeline...");

        let mut pipelines: Vec<wgpu::ComputePipeline> =
            Vec::with_capacity(self.shards.len());
        for (i, shard) in self.shards.iter().enumerate() {
            crate::wgpu_debug!(
                "[compile] compiling shard {:?} [{}..{})...",
                shard.pass, shard.start, shard.end
            );
            let layout = if shard.pass == crate::codegen::wgsl_gen::Pass::BlockOut {
                &bo_pipeline_layout
            } else {
                &pipeline_layout
            };
            let pipeline = pipeline_cache::get_or_compile(
                self.fingerprint,
                shard.pass,
                &self.shader_srcs[i],
                device,
                layout,
            )?;
            pipelines.push(pipeline);
        }
        crate::wgpu_debug!("[compile] {} pipeline(s) compiled ok", pipelines.len());
        self.pipelines = pipelines;

        // Upload the permutation table buffer (binding 1). Even if the
        // perm_table_set is empty (no PerlinSampler in the AST), we still
        // create a 1-i32 dummy buffer so the binding is valid — drivers
        // may reject empty storage bindings.
        if self.perm_buffer.is_none() {
            let buffer_data: Vec<i32> = if self.perm_table_set.buffer.is_empty() {
                vec![0]
            } else {
                self.perm_table_set.buffer.clone()
            };
            crate::wgpu_debug!("[compile] uploading perm buffer ({} i32)...", buffer_data.len());
            let perm_buffer = create_init_buffer(
                device, queue, "c2me_wgpu_permutations",
                bytemuck::cast_slice(&buffer_data),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                use_mapped,
            );
            crate::wgpu_debug!("[compile] perm buffer ok");
            self.perm_buffer = Some(perm_buffer);
        }

        // Upload per-entry origin buffer (binding 3).
        if self.origin_buffer.is_none() {
            let origin_data: Vec<f32> = if self.perm_table_set.origins.is_empty() {
                vec![0.0f32, 0.0, 0.0]
            } else {
                self.perm_table_set.origins.clone()
            };
            crate::wgpu_debug!("[compile] uploading origin buffer...");
            let ob = create_init_buffer(
                device, queue, "c2me_wgpu_perm_origins",
                bytemuck::cast_slice(&origin_data),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                use_mapped,
            );
            crate::wgpu_debug!("[compile] origin buffer ok");
            self.origin_buffer = Some(ob);
        }

        // Upload simplex tables (binding 5). Each entry = 512 i32 (only the
        // first 256 meaningful, MC shuffles perm[0..256)). Empty -> dummy.
        if self.simplex_buffer.is_none() {
            let mut data: Vec<i32> = Vec::new();
            for e in &self.simplex_entries {
                data.extend_from_slice(&e.perm);
            }
            if data.is_empty() { data.push(0); }
            crate::wgpu_debug!("[compile] uploading simplex buffer ({} i32)...", data.len());
            let sb = create_init_buffer(
                device, queue, "c2me_wgpu_simplex",
                bytemuck::cast_slice(&data),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                use_mapped,
            );
            crate::wgpu_debug!("[compile] simplex buffer ok");
            self.simplex_buffer = Some(sb);
        }

        // §23: Beardifier static weight table (binding 10), once per world.
        if self.beard_weight_buffer.is_none() {
            let table = crate::world::beard::build_weight_table();
            let bb = create_init_buffer(
                device, queue, "c2me_wgpu_beard_weight",
                bytemuck::cast_slice(&table),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                use_mapped,
            );
            self.beard_weight_buffer = Some(bb);
        }

        crate::wgpu_debug!("[compile] compile_pipeline done");
        Ok(())
    }

    /// Borrow the compiled per-shard pipelines (empty before
    /// `compile_pipeline` has been called). Dispatched in order.
    pub fn pipelines(&self) -> &[wgpu::ComputePipeline] {
        &self.pipelines
    }

    /// Borrow the channel shards (parallel to `pipelines()`), so callers can
    /// tell a corner shard from the §20 BlockOut shard (different dispatch).
    pub fn shards(&self) -> &[wgsl_gen::Shard] {
        &self.shards
    }

    /// Borrow the uploaded permutation buffer (binding 1), if
    /// `compile_pipeline` has been called.
    /// Record CPU-computed external-input values (flat [chunk][ext][cell]).
    /// Called by Java right before a batch dispatch.
    pub fn set_external_inputs(&self, count: usize, data: Vec<f32>) {
        self.external_count
            .store(count, std::sync::atomic::Ordering::Relaxed);
        *self.external_data.lock().unwrap() = data;
    }

    /// Corner-grid cells per chunk for this world's shape:
    /// cell_count_x * cell_count_y * cell_count_z, where
    ///   cell_count = 16/cell_width + 1  (x, z)  and
    ///   cell_count_y = height/cell_height + 1.
    /// Must match the WGSL prologue's cells_per_chunk exactly.
    /// Global channel indices that the CPU actually reads back, derived
    /// from the shard layout. Shard B covers [start,end) = one channel
    /// (finalDensity); shard C covers the vein channels. In Single mode
    /// (small test blobs) every channel is read back. The executor uses
    /// this to size staging and to pluck the right out_buf slices.
    pub fn inner_count(&self) -> usize {
        self.density_inners.len()
    }

    pub fn readback_channels(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for sh in &self.shards {
            for ch in sh.start..sh.end {
                out.push(ch);
            }
        }
        if out.is_empty() {
            out.push(0);
        }
        out
    }

    pub fn cells_per_chunk(&self) -> usize {
        let cw = self.shape.cell_width.max(1);
        let ch = self.shape.cell_height.max(1);
        let gx = 16 / cw + 1;
        let gy = self.shape.height / ch + 1;
        let gz = 16 / cw + 1;
        (gx * gy * gz) as usize
    }

    pub fn external_count(&self) -> usize {
        self.external_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn external_data(&self) -> Vec<f32> {
        self.external_data.lock().unwrap().clone()
    }

    pub fn perm_buffer(&self) -> Option<&wgpu::Buffer> {
        self.perm_buffer.as_ref()
    }

    /// Borrow the uploaded origin buffer (binding 3), if
    /// `compile_pipeline` has been called.
    pub fn origin_buffer(&self) -> Option<&wgpu::Buffer> {
        self.origin_buffer.as_ref()
    }

    /// Borrow the simplex table buffer (binding 5), if uploaded.
    /// §22: set the per-dimension aquifer mode (called after world_new).
    pub fn set_aquifer_mode(&self, mode: AquiferMode) {
        if let Ok(mut m) = self.aquifer_mode.lock() { *m = mode; }
    }
    pub fn aquifer_mode(&self) -> AquiferMode {
        self.aquifer_mode.lock().map(|m| *m).unwrap_or_default()
    }

    /// Set the ore-vein RNG seed (from `NoiseConfig.getOreRandomDeriver()`).
    pub fn set_ore_seed(&self, lo: i64, hi: i64) {
        *self.ore_seed.lock().unwrap() = (lo, hi);
        self.ore_veins.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn ore_seed(&self) -> (i64, i64) {
        *self.ore_seed.lock().unwrap()
    }

    pub fn beard_weight_buffer(&self) -> Option<&wgpu::Buffer> {
        self.beard_weight_buffer.as_ref()
    }

    pub fn simplex_buffer(&self) -> Option<&wgpu::Buffer> {
        self.simplex_buffer.as_ref()
    }

    /// Borrow the CPU-side permutation table set (useful for unit tests
    /// and for the scheduler when it needs to cross-check offsets).
    pub fn perm_table_set(&self) -> &PermutationTableSet {
        &self.perm_table_set
    }

    /// Stable 32-bit fingerprint of the optimized AST. Same input NoiseRouter
    /// always produces the same fingerprint — used to key pipeline_cache.
    pub fn fingerprint(&self) -> u32 {
        self.fingerprint
    }

    /// Shape config (min_y / height / cell_width / cell_height) — needed by
    /// the scheduler to translate "MC cell index" ↔ "WGSL dispatch grid".
    pub fn shape(&self) -> &ShapeConfig {
        &self.shape
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
