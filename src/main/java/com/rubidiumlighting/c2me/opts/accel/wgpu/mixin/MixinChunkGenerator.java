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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

package com.rubidiumlighting.c2me.opts.accel.wgpu.mixin;

import com.rubidiumlighting.c2me.opts.accel.wgpu.ModuleEntryPoint;
import com.rubidiumlighting.c2me.opts.accel.wgpu.NoiseRouterBlobifier;
import com.rubidiumlighting.c2me.opts.accel.wgpu.WgpuBatchCoordinator;
import com.rubidiumlighting.c2me.opts.accel.wgpu.WgpuFallbackNotice;
import com.rubidiumlighting.c2me.opts.accel.wgpu.WgpuThinBinding;
import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.registry.entry.RegistryEntry;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.chunk.ChunkSection;
import net.minecraft.world.Heightmap;
import net.minecraft.world.gen.StructureAccessor;
import net.minecraft.world.gen.StructureWeightSampler;
import net.minecraft.world.gen.chunk.Blender;
import net.minecraft.world.gen.chunk.ChunkGeneratorSettings;
import net.minecraft.world.gen.chunk.NoiseChunkGenerator;
import net.minecraft.world.gen.noise.NoiseConfig;
import net.minecraft.world.gen.noise.NoiseRouter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import java.nio.ByteBuffer;
import java.util.concurrent.CompletableFuture;

@Mixin(NoiseChunkGenerator.class)
public abstract class MixinChunkGenerator {

    private static final Logger LOGGER = LoggerFactory.getLogger("c2me-wgpu");
    /** Per-dimension state, keyed by the dimension's noise-settings singleton.
     *  Each distinct settings (Overworld / LargeBiomes / Amplified / Nether /
     *  End / …) has its own NoiseRouter + shape → its own WorldContext. This
     *  also fixes a latent bug: previously the first-loaded dimension's context
     *  was reused for any later, differently-shaped dimension. */
    private static final java.util.concurrent.ConcurrentHashMap<ChunkGeneratorSettings, DimState>
            DIM_STATES = new java.util.concurrent.ConcurrentHashMap<>();
    /** Dimensions whose GPU init already failed. Once here we stay on the
     *  vanilla CPU path and never retry: re-running blobify + shader compile
     *  for every chunk would flood the log and waste CPU. */
    private static final java.util.Set<ChunkGeneratorSettings> FAILED_DIMS =
            java.util.concurrent.ConcurrentHashMap.newKeySet();

    /** Debug export switches, read once at class-init from the environment.
     *   - {@code C2ME_VANILLA_REF=1}  : skip GPU, let vanilla generate, then
     *     dump the resulting blocks (see the RETURN injection below) so GPU
     *     terrain can be compared against true vanilla terrain offline.
     *   - {@code C2ME_DUMP_DENSITY=1} : dump the GPU density values for the
     *     chunk at (0, 0).
     *   - {@code C2ME_DUMP_BLOCKS=1}  : dump the block states of the chunk at
     *     (0, 0) for offline comparison against vanilla.
     *   - {@code C2ME_DUMP_BLOB=<path>} : dump the serialized NoiseRouter blob. */
    private static final boolean VANILLA_REF =
            "1".equals(System.getenv("C2ME_VANILLA_REF"));
    private static final boolean DUMP_DENSITY =
            "1".equals(System.getenv("C2ME_DUMP_DENSITY"));
    private static final boolean DUMP_BLOCKS =
            "1".equals(System.getenv("C2ME_DUMP_BLOCKS"));
    /** Dump the aquifer per-cell table (waterLevels + blockPositions) for the
     *  chunk at C2ME_DUMP_CHUNK, for §20 GPU reference. C2ME_DUMP_AQUIFER=1. */
    private static final boolean DUMP_AQUIFER =
            "1".equals(System.getenv("C2ME_DUMP_AQUIFER"));
    /** Per-block aquifer reference: dump (x,y,z,d,beard,state) for the chunk
     *  at C2ME_DUMP_CHUNK, y in [-64,64]. C2ME_DUMP_AQ_BLOCKS=1. */
    private static final boolean DUMP_AQ_BLOCKS =
            "1".equals(System.getenv("C2ME_DUMP_AQ_BLOCKS"));
    /** Coarse per-stage timing of the CPU write-back path. Diagnostic only. */
    private static final boolean PROFILE =
            "1".equals(System.getenv("C2ME_WGPU_PROFILE"));
    /** Async scheduling (§18.6b): non-blocking GPU request + write-back on
     *  C2ME's worker pool. This is the only GPU path. Set C2ME_WGPU_ASYNC_SCHED=0
     *  to disable it entirely (GPU off for this run, plain C2ME CPU generation),
     *  useful as an A/B baseline. */

    /** §23: dump Beardifier pieces/junctions + per-block beard reference. */
    private static final boolean DUMP_BEARD =
            "1".equals(System.getenv("C2ME_DUMP_BEARD"));
    /** §23: dump only the FIRST structure-bearing chunk encountered. */
    private static final java.util.concurrent.atomic.AtomicBoolean BEARD_DUMPED =
            new java.util.concurrent.atomic.AtomicBoolean(false);

    private static final boolean ASYNC_SCHED =
            !"0".equals(System.getenv("C2ME_WGPU_ASYNC_SCHED"));
    @Inject(method = "populateNoise", at = @At("HEAD"), cancellable = true)
    private void onPopulateNoise(Blender blender, NoiseConfig noiseConfig,
                                 StructureAccessor structureAccessor, Chunk chunk,
                                 CallbackInfoReturnable<CompletableFuture<Chunk>> cir) {
        // 检测当前维度的世界种子是否发生变化（例如玩家切换了存档）。
        // 每个维度（通过 settings 区分）独立记录种子，多玩家在不同维度并发时不会互相干扰。
        long currentSeed = noiseConfig.getOreRandomDeriver().split(0L).nextLong();

        // GPU availability: use ModuleEntryPoint (which captures init failure
        // safely). WgpuThinBinding.isAvailable() would itself trigger the
        // class <clinit>; though now failure-safe, ModuleEntryPoint is the
        // single source of truth and also reports the reason to the player.
        if (!ModuleEntryPoint.isEnabled()) {
            return;
        }

        // Vanilla-reference mode: skip the GPU and let MC generate normally.
        // The RETURN injection below dumps the resulting blocks so we can
        // compare GPU terrain against true vanilla terrain offline.
        if (VANILLA_REF) {
            return;
        }

        try {
            // ── per-dimension state (context + coordinator + grid) ──
            NoiseChunkGenerator selfGen = (NoiseChunkGenerator) (Object) this;
            ChunkGeneratorSettings settings = selfGen.getSettings().value();
            DimState ds = DIM_STATES.get(settings);
            if (ds != null && ds.seed != currentSeed) {
                // 发现该维度的缓存属于旧世界，移除并延迟销毁它。
                DIM_STATES.remove(settings);
                long oldHandle = ds.worldHandle;
                if (oldHandle != 0L) {
                    java.util.concurrent.CompletableFuture.delayedExecutor(5, java.util.concurrent.TimeUnit.SECONDS)
                        .execute(() -> {
                            try { WgpuThinBinding.worldDestroy(oldHandle); } catch (Throwable ignored) {}
                        });
                }
                ds = null;
            }
            if (ds == null) {
                // This dimension already failed to initialise: stay on CPU.
                if (FAILED_DIMS.contains(settings)) {
                    return;
                }
                // Double-checked locking: world init is expensive (~7s shader
                // compile) and must run once per dimension. Without the lock,
                // concurrent first-chunks would each build a WorldContext, and
                // the loser's Rust handle + coordinator would leak forever.
                synchronized (DIM_STATES) {
                    ds = DIM_STATES.get(settings);
                    if (ds == null) {
                        if (FAILED_DIMS.contains(settings)) {
                            return;
                        }
                        ds = initDimState(settings, noiseConfig, chunk, currentSeed);
                        if (ds != null) {
                            DIM_STATES.put(settings, ds);
                        } else {
                            // Remember the failure so later chunks don't retry.
                            FAILED_DIMS.add(settings);
                        }
                    }
                }
                if (ds == null) {
                    return; // fall back to vanilla CPU path
                }
            }

            // ── GPU compute via the batch coordinator ────────────
            // Setup (synchronous): sampler + aquifer + ore-vein. Reuse MC's
            // own AquiferSampler; reuse the chunk's cached sampler (vanilla
            // buildSurface does the same — a fresh one per call rebuilds the
            // whole density-function graph).
            NoiseChunkGenerator gen = (NoiseChunkGenerator) (Object) this;
            final Blender fBlender = blender;
            final StructureAccessor fSa = structureAccessor;
            final NoiseConfig fNc = noiseConfig;
            net.minecraft.world.gen.chunk.ChunkNoiseSampler sampler =
                    chunk.getOrCreateChunkNoiseSampler(cs ->
                            ((NoiseChunkGeneratorInvoker) (Object) gen)
                                    .callCreateChunkNoiseSampler(cs, fSa, fBlender, fNc));
            final net.minecraft.world.gen.chunk.AquiferSampler aquifer = sampler.getAquiferSampler();
            // Aquifer table is mandatory: the BlockOut shader classifies
            // water/lava/stone from these cells. Without it every block below
            // sea level falls through to the "no cell found -> stone" branch.
            try {
                stageAquiferForGpu(ds, chunk, aquifer);
            } catch (Throwable t) {
                LOGGER.warn("[c2me-wgpu] aquifer stage failed", t);
            }
            if (DUMP_AQUIFER) {
                LOGGER.info("[c2me-wgpu] stageAquiferForGpu called for chunk: {}, DUMP_CHUNK: {},{}", chunk.getPos(), DUMP_CHUNK[0], DUMP_CHUNK[1]);
            }
            // §22 diagnostic: dump this chunk's aquifer table if requested.
            if (DUMP_AQUIFER
                    && chunk.getPos().x >= DUMP_CHUNK[0] && chunk.getPos().x <= DUMP_CHUNK[2]
                    && chunk.getPos().z >= DUMP_CHUNK[1] && chunk.getPos().z <= DUMP_CHUNK[3]) {
                try {
                    dumpAquifer(chunk, ds.worldHandle, aquifer, ds.rawRouter);
                } catch (Throwable t) {
                    LOGGER.warn("[c2me-wgpu] aquifer dump failed", t);
                }
            }
            try { maybeDumpChunkRef(chunk, ds, ds.rawRouter, noiseConfig); }
            catch (Throwable t) { LOGGER.warn("[c2me-wgpu] chunk-ref dump failed", t); }
            // Beardifier table is mandatory: structures shape the terrain
            // through it; without it buildings float / sink.
            {
                try {
                    stageBeardForGpu(ds, chunk, structureAccessor);
                } catch (Throwable t) {
                    LOGGER.warn("[c2me-wgpu] beard stage failed", t);
                }
            }
            if (DUMP_BEARD && BEARD_DUMPED.compareAndSet(false, true)) {
                try {
                    boolean hasStruct = !structureAccessor
                            .getStructureStarts(chunk.getPos(),
                                    s2 -> s2.getTerrainAdaptation()
                                            != net.minecraft.world.gen.StructureTerrainAdaptation.NONE)
                            .isEmpty();
                    if (hasStruct) {
                        dumpBeard(chunk, net.minecraft.world.gen.StructureWeightSampler
                                .createStructureWeightSampler(structureAccessor, chunk.getPos()));
                    } else {
                        BEARD_DUMPED.set(false); // keep looking
                    }
                } catch (Throwable t) {
                    LOGGER.warn("[c2me-wgpu] beard dump failed", t);
                    BEARD_DUMPED.set(false);
                }
            }
            // NOTE: the CPU ore-vein sampler used to be pulled out here for the
            // removed density write-back path. `sampler.getBlockStateSampler()`
            // re-runs the aquifer with raw density (a side effect) and is now
            // pure overhead, so it is gone. `extractOreVeinSampler` is kept for
            // the upcoming GPU vein work.

            // ── async scheduling path (§18.6b) ───────────────────
            // If enabled and we can reach C2ME's SchedulingManager, submit the
            // GPU request WITHOUT blocking, and run the write-back on C2ME's
            // worker pool (positionedExecutor — same pool, no new thread).
            // Returning a pending future frees the worker to submit more
            // chunks, so batches can grow.
            final long pos = chunk.getPos().toLong();
            final com.ishland.c2me.base.common.scheduler.SchedulingManager sched =
                    ASYNC_SCHED ? schedulingManagerOf(structureAccessor) : null;
            if (sched != null) {
                final Chunk fChunk = chunk;
                final DimState fDs = ds;
                final CompletableFuture<Chunk> result = new CompletableFuture<>();
                ds.coordinator.requestAsync(chunk.getPos().x, chunk.getPos().z)
                        .whenComplete((gpu, err) -> {
                            if (err != null) {
                                LOGGER.warn("[c2me-wgpu] async GPU request failed", err);
                                result.completeExceptionally(err);
                                return;
                            }
                            // Enqueue write-back onto C2ME's worker pool. The
                            // chunk's position lock is held by C2ME's
                            // ScheduledTask until `result` completes (§18.6b).
                            sched.positionedExecutor(pos).execute(() -> {
                                try {
                                    writeDensityToChunk(fChunk, gpu.blockSeg,
                                            gpu.batchIndex, fDs);
                                    if (DUMP_BLOCKS) {
                                        maybeDumpBlocks(fChunk, "gpu");
                                    }
                                    result.complete(fChunk);
                                } catch (Throwable t) {
                                    LOGGER.error("[c2me-wgpu] async write-back failed", t);
                                    result.completeExceptionally(t);
                                }
                            });
                        });
                cir.setReturnValue(result);
                return;
            }

            // ── no async scheduling available ────────────────────
            // Async scheduling is the ONLY GPU path. If C2ME's scheduler
            // cannot be reached (or is disabled), do NOT fall back to a
            // synchronous GPU request: §10.2 measured plain C2ME on the CPU
            // (desktop ~144 CPS) faster than our synchronous GPU path
            // (~80-110 CPS). Returning without cancelling lets vanilla/C2ME
            // CPU generation run.
            return;
        } catch (Throwable t) {
            LOGGER.error("[c2me-wgpu] GPU populateNoise failed, falling back to CPU", t);
        }
    }

    @Inject(method = "populateNoise", at = @At("RETURN"), cancellable = true)
    private void afterPopulateNoise(Blender blender, NoiseConfig noiseConfig,
                                    StructureAccessor structureAccessor, Chunk chunk,
                                    CallbackInfoReturnable<CompletableFuture<Chunk>> cir) {
        if (!VANILLA_REF) {
            return;
        }
        CompletableFuture<Chunk> f = cir.getReturnValue();
        if (f == null) {
            return;
        }
        cir.setReturnValue(f.thenApply(c -> {
            maybeDumpBlocks(c, "vanilla");
            return c;
        }));
    }

    /** Target chunk(s) for block dumping. Default (0,0). Override with
     *  C2ME_DUMP_CHUNK=x,z (or x1,z1,x2,z2 for a rectangle, inclusive). */
    private static final int[] DUMP_CHUNK = parseDumpChunk();

    /** True only when the user explicitly asked for a chunk dump. All the
     *  diagnostic writers below are gated on this (or on their own DUMP_*
     *  flag), so a normal session never touches the filesystem. */
    private static final boolean DUMP_ENABLED = parseDumpEnabled();

    private static boolean parseDumpEnabled() {
        String v = System.getenv("C2ME_DUMP_CHUNK");
        return v != null && !v.isBlank();
    }

    /** Directory for diagnostic dumps. `/tmp` does not exist on Android, so
     *  default to the platform temp dir; `C2ME_DUMP_DIR` overrides. */
    private static String dumpDir() {
        String d = System.getenv("C2ME_DUMP_DIR");
        if (d != null && !d.isBlank()) return d;
        String t = System.getProperty("java.io.tmpdir");
        if (t == null || t.isBlank()) t = ".";
        // Trailing separators would double up in the concatenations below.
        while (t.length() > 1 && (t.endsWith("/") || t.endsWith("\\"))) {
            t = t.substring(0, t.length() - 1);
        }
        return t;
    }

    private static int[] parseDumpChunk() {
        String v = System.getenv("C2ME_DUMP_CHUNK");
        if (v == null || v.isBlank()) return new int[]{98, -47, 98, -47}; // FORCED
        try {
            String[] p = v.split(",");
            if (p.length == 2) {
                int x = Integer.parseInt(p[0].trim()), z = Integer.parseInt(p[1].trim());
                return new int[]{x, z, x, z};
            }
            if (p.length == 4) {
                return new int[]{
                        Integer.parseInt(p[0].trim()), Integer.parseInt(p[1].trim()),
                        Integer.parseInt(p[2].trim()), Integer.parseInt(p[3].trim())};
            }
        } catch (Throwable ignored) {}
        return new int[]{0, 0, 0, 0};
    }

    // Aggregated CPU write-back timing: one averaged line every 128 chunks,
    // so the log isn't flooded (was ~1 line/chunk).
    private static final java.util.concurrent.atomic.AtomicLong CPUPROF_N =
            new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong CPUPROF_US =
            new java.util.concurrent.atomic.AtomicLong();

    private static void profileCpuAccumulate(long us) {
        // Synchronized: with two separate atomics, two threads could both
        // observe n>=128 and the second read cnt==0 after the first reset,
        // causing a divide-by-zero that aborted the whole write-back.
        synchronized (CPUPROF_N) {
            long n = CPUPROF_N.incrementAndGet();
            CPUPROF_US.addAndGet(us);
            if (n >= 128) {
                long cnt = CPUPROF_N.getAndSet(0);
                long total = CPUPROF_US.getAndSet(0);
                if (cnt > 0) {
                    LOGGER.info("[c2me-wgpu][profile-cpu] avg over {} chunks: write={}us/chunk",
                            cnt, total / cnt);
                }
            }
        }
    }

    /** Resolve C2ME's SchedulingManager from the worldgen StructureAccessor:
     *  world (ChunkRegion during worldgen) -> ServerWorld -> ServerChunkManager
     *  -> chunkLoadingManager (IVanillaChunkManager). Returns null on any
     *  failure, in which case the caller leaves generation to vanilla/C2ME on
     *  the CPU (no synchronous GPU fallback). */
    private static com.ishland.c2me.base.common.scheduler.SchedulingManager
            schedulingManagerOf(StructureAccessor sa) {
        try {
            Object w = sa.world;   // widened by accesswidener
            net.minecraft.server.world.ServerWorld sw;
            if (w instanceof net.minecraft.world.ChunkRegion cr) {
                sw = cr.toServerWorld();
            } else if (w instanceof net.minecraft.server.world.ServerWorld s) {
                sw = s;
            } else {
                return null;
            }
            net.minecraft.server.world.ServerChunkManager scm =
                    (net.minecraft.server.world.ServerChunkManager) sw.getChunkManager();
            net.minecraft.server.world.ServerChunkLoadingManager sclm = scm.chunkLoadingManager;
            if (sclm instanceof com.ishland.c2me.base.common.scheduler.IVanillaChunkManager vm) {
                return vm.c2me$getSchedulingManager();
            }
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] scheduling-manager lookup failed", t);
        }
        return null;
    }

    /** ChunkNoiseSampler.getBlockStateSampler() returns a ChainedBlockSource
     *  = [aquifer, oreVein]. Calling it re-runs the aquifer with the raw
     *  (non-interpolated) density, which can wrongly return air/water and
     *  clobber our solid result. Pull the ore-vein element out instead.
     *  Returns null when there is no separate vein sampler (oreVeins off). */
    private static net.minecraft.world.gen.chunk.ChunkNoiseSampler.BlockStateSampler
            extractOreVeinSampler(
                    net.minecraft.world.gen.chunk.ChunkNoiseSampler.BlockStateSampler full) {
        if (full == null) return null;
        try {
            for (java.lang.reflect.Field f : full.getClass().getDeclaredFields()) {
                if (java.util.List.class.isAssignableFrom(f.getType())) {
                    f.setAccessible(true);
                    java.util.List<?> list = (java.util.List<?>) f.get(full);
                    if (list == null) continue;
                    // Find the OreVein sampler by lambda enclosing class name;
                    // do NOT assume it is the last element (datapacks/mods can
                    // add more sources to the chain).
                    for (Object o : list) {
                        if (o != null && o.getClass().getName().contains("OreVein")) {
                            return (net.minecraft.world.gen.chunk.ChunkNoiseSampler.BlockStateSampler) o;
                        }
                    }
                    // Not found: no separate vein sampler in this chain.
                    return null;
                }
            }
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] extractOreVeinSampler failed", t);
        }
        return null;
    }

    /** Dump selected chunk block states for offline GPU-vs-vanilla comparison. */
    private static void maybeDumpBlocks(Chunk chunk, String mode) {
        int cx = chunk.getPos().x, cz = chunk.getPos().z;
        if (cx < DUMP_CHUNK[0] || cx > DUMP_CHUNK[2]
                || cz < DUMP_CHUNK[1] || cz > DUMP_CHUNK[3]) {
            return;
        }
        try {
            StringBuilder sb = new StringBuilder();
            int bottom = chunk.getBottomY();
            int top = chunk.getTopY();
            for (int y = bottom; y < top; y++) {
                int si = chunk.getSectionIndex(y);
                if (si < 0 || si >= chunk.getSectionArray().length) continue;
                ChunkSection sec = chunk.getSection(si);
                if (sec == null || sec.isEmpty()) continue;
                for (int z = 0; z < 16; z++) {
                    for (int x = 0; x < 16; x++) {
                        BlockState bs = sec.getBlockState(x, y & 15, z);
                        if (bs.isAir()) continue;
                        String id = net.minecraft.registry.Registries.BLOCK
                                .getId(bs.getBlock()).toString();
                        sb.append(x).append(' ').append(y).append(' ').append(z)
                          .append(' ').append(id).append('\n');
                    }
                }
            }
            String path = dumpDir() + "/chunk_blocks_" + mode
                    + "_" + cx + "_" + cz + ".txt";
            java.nio.file.Files.writeString(java.nio.file.Path.of(path), sb.toString());
            LOGGER.info("[c2me-wgpu] dumped {} block states ({} blocks) to {}",
                    mode, sb.length(), path);
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] block dump failed", t);
        }
    }

    /**
     * Write the GPU finalDensity channel (channel 11) to {@code chunk} as
     * STONE / AIR. Cell grid = (16/cellW) × (height/cellH) × (16/cellW)
     * = 4 × 96 × 4 = 1536 cells; each cell covers cellW×cellH×cellW blocks.
     */
    /** §20: write a chunk from GPU block-state bytes (u32/block: 0=air,
     *  1=stone). Chunk-major layout: [chunk][y][z][x] with x fastest,
     *  then z, then y (matches BLOCK_OUT_ENTRY). */
    private static void writeBlocksFromGpu(Chunk chunk, java.lang.foreign.MemorySegment blockSeg,
                                           int batchIndex, DimState ds) {
        final int BLOCKS_PER_CHUNK = 16 * ds.height * 16;
        final int minY = ds.minY;
        BlockState stone = ds.defaultBlock;
        BlockState air = Blocks.AIR.getDefaultState();
        // GPU state codes:
        //   0=air 1=stone 2=water 3=lava
        //   4=granite 5=tuff 6=copper_ore 7=raw_copper_block
        //   8=deepslate_iron_ore 9=raw_iron_block
        // 4..9 are ore-vein outputs (OreVeinSampler.VeinType).
        BlockState water = Blocks.WATER.getDefaultState();
        BlockState lava = Blocks.LAVA.getDefaultState();
        BlockState granite = Blocks.GRANITE.getDefaultState();
        BlockState tuff = Blocks.TUFF.getDefaultState();
        BlockState copperOre = Blocks.COPPER_ORE.getDefaultState();
        BlockState rawCopper = Blocks.RAW_COPPER_BLOCK.getDefaultState();
        BlockState ironOre = Blocks.DEEPSLATE_IRON_ORE.getDefaultState();
        BlockState rawIron = Blocks.RAW_IRON_BLOCK.getDefaultState();
        Heightmap oceanFloorWg = chunk.getHeightmap(Heightmap.Type.OCEAN_FLOOR_WG);
        Heightmap worldSurfaceWg = chunk.getHeightmap(Heightmap.Type.WORLD_SURFACE_WG);
        java.nio.ByteBuffer bb = blockSeg.asByteBuffer();
        bb.order(java.nio.ByteOrder.LITTLE_ENDIAN);
        long base = (long) batchIndex * BLOCKS_PER_CHUNK * 4;
        if (blockSeg.byteSize() < base + (long) BLOCKS_PER_CHUNK * 4) {
            LOGGER.warn("[c2me-wgpu] block blob too small for chunk {}", batchIndex);
            return;
        }
        for (int by = 0; by < ds.height; by++) {
            int globalY = minY + by;
            int secIdx = chunk.getSectionIndex(globalY);
            if (secIdx < 0 || secIdx >= chunk.getSectionArray().length) continue;
            ChunkSection sec = chunk.getSection(secIdx);
            int ly = globalY & 15;
            for (int bz = 0; bz < 16; bz++) {
                for (int bx = 0; bx < 16; bx++) {
                    int idx = bx + bz * 16 + by * 256;
                    int v = bb.getInt((int) (base + (long) idx * 4));
                    if (v == 0) continue;           // air
                    BlockState bs = switch (v) {
                        case 2 -> water;
                        case 3 -> lava;
                        case 4 -> granite;
                        case 5 -> tuff;
                        case 6 -> copperOre;
                        case 7 -> rawCopper;
                        case 8 -> ironOre;
                        case 9 -> rawIron;
                        default -> stone;
                    };
                    sec.setBlockState(bx, ly, bz, bs);
                    oceanFloorWg.trackUpdate(bx, globalY, bz, bs);
                    worldSurfaceWg.trackUpdate(bx, globalY, bz, bs);
                }
            }
        }
    }

    /** §20: the ONLY GPU write-back path. The GPU emits one block-state
     *  byte (0=air,1=stone,2=water,3=lava) per block; this copies them into
     *  the chunk palette and keeps the two WG heightmaps in sync.
     *
     *  The former "GPU density only, MC trilinear + aquifer + vein on the
     *  CPU" path was removed: §20 measured it as a net loss (CPU write-back
     *  pinned CPS at ~150 on desktop, ~6 on mobile). Ore veins are not yet
     *  emitted by the GPU — see the GPU-vein TODO.
     *
     *  `blockSeg` is null only if the BlockOut shard produced no blob (should
     *  not happen now that it is on by default); we log and leave the chunk
     *  empty rather than crashing the world. */
    private static void writeDensityToChunk(Chunk chunk,
                                            java.lang.foreign.MemorySegment blockSeg,
                                            int batchIndex, DimState ds) {
        if (blockSeg == null) {
            LOGGER.error("[c2me-wgpu] BlockOut produced no block blob for chunk {} {} "
                    + "— chunk left empty (block_out shard missing?)",
                    chunk.getPos().x, chunk.getPos().z);
            return;
        }
        writeBlocksFromGpu(chunk, blockSeg, batchIndex, ds);
    }

    /** §20 reference: dump the aquifer per-cell table so the GPU reimplementation
     *  of getWaterLevel can be checked against vanilla. Format:
     *    AQ startX startY startZ sizeX sizeZ
     *    CELL index blockX blockY blockZ y state
     *  blockPositions is a packed long = BlockPos.asLong(blockX,blockY,blockZ). */
    /**
     * §22 Step B: warm up the aquifer's lazy waterLevels table, then serialize
     * the compact table (seed + per-cell lvl_y/state) and stage it to Rust.
     * blockPositions are recomputed on the Rust side from the seed.
     */
    /**
     * §23: pack the Beardifier's pieces/junctions (u32 LE) and stage to Rust.
     * Layout: [0]=piece_count, [1]=junction_count, then per piece 8 i32
     * (minX,minY,minZ,maxX,maxY,maxZ,adaptation,groundLevelDelta), then per
     * junction 3 i32. Adaptation codes match beard.rs / WGSL.
     */
    private static void stageBeardForGpu(DimState ds, Chunk chunk,
            StructureAccessor sa) {
        boolean hasStruct = !sa.getStructureStarts(chunk.getPos(),
                s2 -> s2.getTerrainAdaptation()
                        != net.minecraft.world.gen.StructureTerrainAdaptation.NONE)
                .isEmpty();
        if (!hasStruct) return;
        var sw = net.minecraft.world.gen.StructureWeightSampler
                .createStructureWeightSampler(sa, chunk.getPos());
        var pit = sw.pieceIterator;
        java.util.List<net.minecraft.world.gen.StructureWeightSampler.Piece> pieces =
                new java.util.ArrayList<>();
        while (pit.hasNext()) pieces.add(pit.next());
        pit.back(Integer.MAX_VALUE);
        var jit = sw.junctionIterator;
        java.util.List<net.minecraft.structure.JigsawJunction> juncs =
                new java.util.ArrayList<>();
        while (jit.hasNext()) juncs.add(jit.next());
        jit.back(Integer.MAX_VALUE);
        if (pieces.isEmpty() && juncs.isEmpty()) return;

        int n = 2 + pieces.size() * 8 + juncs.size() * 3;
        java.nio.ByteBuffer tb = java.nio.ByteBuffer.allocate(n * 4)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        tb.putInt(pieces.size());
        tb.putInt(juncs.size());
        for (var pc : pieces) {
            var box = pc.box();
            tb.putInt(box.getMinX()); tb.putInt(box.getMinY()); tb.putInt(box.getMinZ());
            tb.putInt(box.getMaxX()); tb.putInt(box.getMaxY()); tb.putInt(box.getMaxZ());
            tb.putInt(switch (pc.terrainAdjustment()) {
                case NONE -> 0; case BURY -> 1; case BEARD_THIN -> 2;
                case BEARD_BOX -> 3; case ENCAPSULATE -> 4; });
            tb.putInt(pc.groundLevelDelta());
        }
        for (var jn : juncs) {
            tb.putInt(jn.getSourceX()); tb.putInt(jn.getSourceGroundY());
            tb.putInt(jn.getSourceZ());
        }
        WgpuThinBinding.setBeardTable(ds.worldHandle,
                chunk.getPos().x, chunk.getPos().z, tb.array());
    }

    /** §23: dump the Beardifier's per-chunk pieces/junctions and per-block
     *  beard values, for offline validation of the Rust/WGSL reproduction. */
    private static void dumpBeard(Chunk chunk,
            net.minecraft.world.gen.StructureWeightSampler sw) {
        try {
            StringBuilder sb = new StringBuilder();
            sb.append("CHUNK ").append(chunk.getPos().x).append(' ')
              .append(chunk.getPos().z).append('\n');
            // pieces
            var pit = sw.pieceIterator;
            int np = 0;
            StringBuilder pb = new StringBuilder();
            while (pit.hasNext()) {
                var pc = pit.next();
                var box = pc.box();
                String ad = pc.terrainAdjustment().name();
                pb.append(box.getMinX()).append(' ').append(box.getMinY()).append(' ')
                  .append(box.getMinZ()).append(' ').append(box.getMaxX()).append(' ')
                  .append(box.getMaxY()).append(' ').append(box.getMaxZ()).append(' ')
                  .append(ad).append(' ').append(pc.groundLevelDelta()).append('\n');
                np++;
            }
            pit.back(Integer.MAX_VALUE);
            sb.append("PIECE ").append(np).append('\n').append(pb);
            // junctions
            var jit = sw.junctionIterator;
            int nj = 0;
            StringBuilder jb = new StringBuilder();
            while (jit.hasNext()) {
                var jn = jit.next();
                jb.append(jn.getSourceX()).append(' ').append(jn.getSourceGroundY())
                  .append(' ').append(jn.getSourceZ()).append('\n');
                nj++;
            }
            jit.back(Integer.MAX_VALUE);
            sb.append("JUNC ").append(nj).append('\n').append(jb);
            // per-block beard (full chunk)
            int minY = -64, height = 384;
            var npos = new com.rubidiumlighting.c2me.opts.accel.wgpu.MutableNoisePos();
            sb.append("BLOCK ").append(16 * height * 16).append('\n');
            int sx = chunk.getPos().getStartX();
            int sz = chunk.getPos().getStartZ();
            for (int by = 0; by < height; by++) {
                int gy = minY + by;
                for (int bz = 0; bz < 16; bz++) {
                    for (int bx = 0; bx < 16; bx++) {
                        double bd = sw.sample(npos.set(sx + bx, gy, sz + bz));
                        sb.append(bx).append(' ').append(gy).append(' ').append(bz)
                          .append(' ').append(bd).append('\n');
                    }
                }
            }
            java.nio.file.Files.writeString(
                    java.nio.file.Path.of(dumpDir() + "/beard_" + chunk.getPos().x + "_"
                            + chunk.getPos().z + ".txt"), sb.toString());
            LOGGER.info("[c2me-wgpu] dumped beard ref ({} pieces, {} junc)", np, nj);
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] beard dump failed", t);
        }
    }

    private static void stageAquiferForGpu(DimState ds, Chunk chunk,
            net.minecraft.world.gen.chunk.AquiferSampler aquifer) {
        if (!(aquifer instanceof net.minecraft.world.gen.chunk.AquiferSampler.Impl impl)) {
            return;
        }
        AquiferImplAccessor a = (AquiferImplAccessor) (Object) impl;
        int startX = a.getStartX(), startY = a.getStartY(), startZ = a.getStartZ();
        int sizeX = a.getSizeX(), sizeZ = a.getSizeZ();
        net.minecraft.world.gen.chunk.AquiferSampler.FluidLevel[] wl = a.getWaterLevels();

        var dbg = new StringBuilder();
        a.getRandomDeriver().addDebugInfo(dbg);
        String dss = dbg.toString();
        long sLo = Long.parseLong(dss.split("seedLo: ")[1].split(",")[0].trim());
        long sHi = Long.parseLong(dss.split("seedHi: ")[1].trim());

        // Warm up every cell (MC fills waterLevels lazily on first apply).
        for (int idx = 0; idx < wl.length; idx++) {
            int ci = idx % sizeX;
            int ck = (idx / sizeX) % sizeZ;
            int cj = idx / (sizeX * sizeZ);
            int cx = startX + ci, cy = startY + cj, cz = startZ + ck;
            long h = net.minecraft.util.math.MathHelper.hashCode(cx, cy, cz);
            var rr = new net.minecraft.util.math.random.Xoroshiro128PlusPlusRandom(h ^ sLo, sHi);
            int bx = cx * 16 + rr.nextInt(10);
            int by = cy * 12 + rr.nextInt(9);
            int bz = cz * 16 + rr.nextInt(10);
            a.invokeGetWaterLevel(net.minecraft.util.math.BlockPos.asLong(bx, by, bz));
        }

        java.nio.ByteBuffer tb = java.nio.ByteBuffer.allocate(24 + wl.length * 8)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        tb.putInt(0x41515541); // "AQUA"
        tb.putInt(wl.length);
        tb.putLong(sLo); tb.putLong(sHi);
        for (int i = 0; i < wl.length; i++) {
            var lvl = wl[i];
            int ly = (lvl == null) ? Integer.MIN_VALUE : lvl.y;
            int sc = 0;
            if (lvl != null) {
                var blk = lvl.state.getBlock();
                if (blk == net.minecraft.block.Blocks.WATER) sc = 2;
                else if (blk == net.minecraft.block.Blocks.LAVA) sc = 3;
            }
            tb.putInt(ly); tb.putInt(sc);
        }
        WgpuThinBinding.setAquiferTable(ds.worldHandle,
                chunk.getPos().x, chunk.getPos().z, tb.array());
    }

    private static void dumpAquifer(Chunk chunk, long worldHandle,
            net.minecraft.world.gen.chunk.AquiferSampler aquifer,
            NoiseRouter router) {
        try {
                        if (!(aquifer instanceof net.minecraft.world.gen.chunk.AquiferSampler.Impl impl)) {
                LOGGER.warn("[c2me-wgpu] aquifer dump: not Impl, skipping");
                return;
            }
            AquiferImplAccessor a = (AquiferImplAccessor) (Object) impl;
            int startX = a.getStartX(), startY = a.getStartY(), startZ = a.getStartZ();
            int sizeX = a.getSizeX(), sizeZ = a.getSizeZ();
            net.minecraft.world.gen.chunk.AquiferSampler.FluidLevel[] wl = a.getWaterLevels();
            long[] bp = a.getBlockPositions();
            StringBuilder sb = new StringBuilder();
            sb.append("CHUNK ").append(chunk.getPos().x).append(' ')
              .append(chunk.getPos().z).append('\n');
            sb.append("AQ ").append(startX).append(' ').append(startY).append(' ')
              .append(startZ).append(' ').append(sizeX).append(' ').append(sizeZ)
              .append('\n');
            // §22: dump the aquifer RNG seed (seedLo/seedHi) so the Rust/WGSL
            // generator can reproduce blockPositions[] from scratch.
            {
                var dbg = new StringBuilder();
                a.getRandomDeriver().addDebugInfo(dbg);
                sb.append("SEED ").append(dbg).append('\n');
            }
            // §22: 预热所有 cell 的 waterLevels 表。MC 惰性填充（apply 访问到
            // 才填），GPU 需要完整表。用 seedLo/seedHi 算每 cell 的
            // blockPosition，再调 getWaterLevel 触发 getFluidLevel。
            {
                var dbg2 = new StringBuilder();
                a.getRandomDeriver().addDebugInfo(dbg2);
                String ds2 = dbg2.toString();
                long sLo = Long.parseLong(ds2.split("seedLo: ")[1].split(",")[0].trim());
                long sHi = Long.parseLong(ds2.split("seedHi: ")[1].trim());
                long t0 = System.nanoTime();
                int warmed = 0;
                for (int idx = 0; idx < wl.length; idx++) {
                    int ci = idx % sizeX;
                    int ck = (idx / sizeX) % sizeZ;
                    int cj = idx / (sizeX * sizeZ);
                    int cx = startX + ci, cy = startY + cj, cz = startZ + ck;
                    long h = net.minecraft.util.math.MathHelper.hashCode(cx, cy, cz);
                    var rr = new net.minecraft.util.math.random.Xoroshiro128PlusPlusRandom(
                            h ^ sLo, sHi);
                    int bx = cx * 16 + rr.nextInt(10);
                    int by = cy * 12 + rr.nextInt(9);
                    int bz = cz * 16 + rr.nextInt(10);
                    long pos = net.minecraft.util.math.BlockPos.asLong(bx, by, bz);
                    a.invokeGetWaterLevel(pos);
                    warmed++;
                }
                long dt = (System.nanoTime() - t0) / 1_000_000;
            }
            // §22 Step B: serialize the compact per-chunk aquifer table.
            // Layout: magic(4) n(4) seedLo(8) seedHi(8) then n * (lvl_y i32 +
            // state u32). blockPositions[] is NOT included — Rust recomputes
            // it from the seed (verified bit-exact in aquifer_positions.rs).
            {
                var dbg3 = new StringBuilder();
                a.getRandomDeriver().addDebugInfo(dbg3);
                String ds3 = dbg3.toString();
                long sLo3 = Long.parseLong(ds3.split("seedLo: ")[1].split(",")[0].trim());
                long sHi3 = Long.parseLong(ds3.split("seedHi: ")[1].trim());
                java.nio.ByteBuffer tb = java.nio.ByteBuffer.allocate(24 + wl.length * 8)
                        .order(java.nio.ByteOrder.LITTLE_ENDIAN);
                tb.putInt(0x41515541); // "AQUA"
                tb.putInt(wl.length);
                tb.putLong(sLo3); tb.putLong(sHi3);
                for (int i = 0; i < wl.length; i++) {
                    var lvl = wl[i];
                    int ly = (lvl == null) ? Integer.MIN_VALUE : lvl.y;
                    int sc = 0;
                    if (lvl != null) {
                        var blk = lvl.state.getBlock();
                        if (blk == net.minecraft.block.Blocks.WATER) sc = 2;
                        else if (blk == net.minecraft.block.Blocks.LAVA) sc = 3;
                    }
                    tb.putInt(ly); tb.putInt(sc);
                }
                byte[] arr = tb.array();
                java.nio.file.Files.write(java.nio.file.Path.of(dumpDir() + "/aquifer_table_"
                        + chunk.getPos().x + "_" + chunk.getPos().z + ".bin"), arr);
                if (worldHandle != 0L) {
                    WgpuThinBinding.setAquiferTable(worldHandle,
                            chunk.getPos().x, chunk.getPos().z, arr);
                }
            }
            for (int i = 0; i < wl.length; i++) {
                long packed = bp[i];
                boolean none = (packed == Long.MAX_VALUE);
                int bx = none ? 0 : net.minecraft.util.math.BlockPos.unpackLongX(packed);
                int by = none ? 0 : net.minecraft.util.math.BlockPos.unpackLongY(packed);
                int bz = none ? 0 : net.minecraft.util.math.BlockPos.unpackLongZ(packed);
                var lvl = wl[i];
                String state = (lvl == null) ? "null"
                        : net.minecraft.registry.Registries.BLOCK
                            .getId(lvl.state.getBlock()).toString();
                int y = (lvl == null) ? Integer.MIN_VALUE : lvl.y;
                sb.append(i).append(' ')
                  .append(none ? "none" : (bx + " " + by + " " + bz)).append(' ')
                  .append(y).append(' ').append(state).append('\n');
            }
            String path = dumpDir() + "/aquifer_" + chunk.getPos().x + "_" + chunk.getPos().z + ".txt";
            java.nio.file.Files.writeString(java.nio.file.Path.of(path), sb.toString());
            LOGGER.info("[c2me-wgpu] dumped aquifer table ({} cells) to {}", wl.length, path);

            // §20: dump, per cell, the 5 noise inputs at the exact points
            // getFluidLevel samples + the expected output. Rust/WGSL then only
            // needs to reproduce the control flow (no noise).
            var flooded = unwrapRouter(router.fluidLevelFloodednessNoise());
            var spread  = unwrapRouter(router.fluidLevelSpreadNoise());
            var lava    = unwrapRouter(router.lavaNoise());
            var erosion = unwrapRouter(router.erosion());
            var depth   = unwrapRouter(router.depth());
            StringBuilder nb = new StringBuilder();
            nb.append("CHUNK ").append(chunk.getPos().x).append(' ')
              .append(chunk.getPos().z).append('\n');
            var flSampler = a.getFluidLevelSampler();
            var cnsForSurf = a.getChunkNoiseSampler();
            for (int i = 0; i < wl.length; i++) {
                long packed = bp[i];
                if (packed == Long.MAX_VALUE) continue;
                int bx = net.minecraft.util.math.BlockPos.unpackLongX(packed);
                int by = net.minecraft.util.math.BlockPos.unpackLongY(packed);
                int bz = net.minecraft.util.math.BlockPos.unpackLongZ(packed);
                var p0 = new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(bx, by, bz);
                double f = flooded.sample(p0);
                double e = erosion.sample(p0);
                double d = depth.sample(p0);
                var ps = new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(
                        Math.floorDiv(bx,16), Math.floorDiv(by,40), Math.floorDiv(bz,16));
                double sp = spread.sample(ps);
                var pt = new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(
                        Math.floorDiv(bx,64), Math.floorDiv(by,40), Math.floorDiv(bz,64));
                double lv = lava.sample(pt);
                var lvl = wl[i];
                // initial FluidLevel from the sampler
                var fl0 = flSampler.getFluidLevel(bx, by, bz);
                nb.append(i).append(' ').append(bx).append(' ').append(by).append(' ')
                  .append(bz).append(' ')
                  .append((float) f).append(' ').append((float) sp).append(' ')
                  .append((float) lv).append(' ').append((float) e).append(' ')
                  .append((float) d).append(' ')
                  .append(fl0.y).append(' ')
                  .append(net.minecraft.registry.Registries.BLOCK.getId(fl0.state.getBlock()).toString())
                  .append(' ');
                // 13 neighbor surface heights
                for (int[] cs : new int[][]{
                        {0,0},{-2,-1},{-1,-1},{0,-1},{1,-1},{-3,0},{-2,0},{-1,0},{1,0},
                        {-2,1},{-1,1},{0,1},{1,1}}) {
                    int m = bx + cs[0] * 16;
                    int n = bz + cs[1] * 16;
                    nb.append(cnsForSurf.estimateSurfaceHeight(m, n)).append(' ');
                }
                nb.append(lvl == null ? Integer.MIN_VALUE : lvl.y).append(' ')
                  .append(lvl == null ? "null"
                        : net.minecraft.registry.Registries.BLOCK.getId(lvl.state.getBlock()).toString())
                  .append('\n');
            }
            String nbPath = dumpDir() + "/aquifer_noise_" + chunk.getPos().x + "_" + chunk.getPos().z + ".txt";
            java.nio.file.Files.writeString(java.nio.file.Path.of(nbPath), nb.toString());
            LOGGER.info("[c2me-wgpu] dumped aquifer noise inputs to {}", nbPath);

            // Also dump the per-column surface-height cache: getWaterLevel
            // needs estimateSurfaceHeight for 13 neighbour columns per cell,
            // and this is the CPU-provided input for the GPU reimplementation.
            net.minecraft.world.gen.chunk.ChunkNoiseSampler cns = a.getChunkNoiseSampler();
            var shCache = ((ChunkNoiseSamplerAccessor) (Object) cns).getSurfaceHeightEstimateCache();
            StringBuilder sh = new StringBuilder();
            sh.append("CHUNK ").append(chunk.getPos().x).append(' ')
              .append(chunk.getPos().z).append('\n');
            for (var e : shCache.long2IntEntrySet()) {
                long k = e.getLongKey();
                int cx = net.minecraft.util.math.ColumnPos.getX(k);
                int cz = net.minecraft.util.math.ColumnPos.getZ(k);
                sh.append(cx).append(' ').append(cz).append(' ')
                  .append(e.getIntValue()).append('\n');
            }
            String shPath = dumpDir() + "/aquifer_surfaces_" + chunk.getPos().x + "_" + chunk.getPos().z + ".txt";
            java.nio.file.Files.writeString(java.nio.file.Path.of(shPath), sh.toString());
            LOGGER.info("[c2me-wgpu] dumped surface-height cache ({} columns) to {}",
                    shCache.size(), shPath);
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] aquifer dump failed", t);
        }
    }

    /** Strip RegistryEntryHolder + Wrapper(cache) layers so a router
     *  node can be single-shot sampled for diagnostics. */
    /** Per-dimension GPU state: world handle, batch coordinator, grid dims. */
    private static final class DimState {
        final long seed;
        final long worldHandle;
        final WgpuBatchCoordinator coordinator;
        final int gridX, gridY, gridZ;   // corner counts per axis
        final int cellW, cellH;          // cell size in blocks
        final int minY, height;
        final NoiseRouter rawRouter;
        /** Dimension's solid block (Overworld stone / Nether netherrack /
         *  End end_stone). The aquifer returns null for "solid", and we must
         *  fill with THIS, not a hard-coded stone — otherwise the Nether got
         *  filled with stone. */
        final net.minecraft.block.BlockState defaultBlock;
        DimState(long seed, long worldHandle, WgpuBatchCoordinator coordinator,
                 int gridX, int gridY, int gridZ, int cellW, int cellH,
                 int minY, int height, NoiseRouter rawRouter,
                 net.minecraft.block.BlockState defaultBlock) {
            this.seed = seed;
            this.worldHandle = worldHandle;
            this.coordinator = coordinator;
            this.gridX = gridX; this.gridY = gridY; this.gridZ = gridZ;
            this.cellW = cellW; this.cellH = cellH;
            this.minY = minY; this.height = height;
            this.rawRouter = rawRouter;
            this.defaultBlock = defaultBlock;
        }
    }

    /** Build a WorldContext for one dimension's noise settings. Returns null
     *  on any failure so the caller can fall back to the vanilla CPU path. */
    private static DimState initDimState(ChunkGeneratorSettings settings,
                                         NoiseConfig noiseConfig, Chunk chunk, long seed) {
        try {
            NoiseRouter router = decompileRouter(noiseConfig.getNoiseRouter());
            if (router == null) {
                LOGGER.warn("[c2me-wgpu] NoiseRouter is null, falling back to CPU");
                return null;
            }
            net.minecraft.world.gen.chunk.GenerationShapeConfig gsc =
                    settings.generationShapeConfig();
            int hc = gsc.horizontalCellBlockCount();
            int vc = gsc.verticalCellBlockCount();
            int minY = gsc.minimumY();
            int height = gsc.height();
            int gx = 16 / hc + 1;
            int gz = 16 / hc + 1;
            int gy = height / vc + 1;
            LOGGER.info("[c2me-wgpu] shape: minY={} h={} cell={}x{} grid={}x{}x{}",
                    minY, height, hc, vc, gx, gy, gz);
            NoiseRouterBlobifier.ShapeConfig shape =
                    new NoiseRouterBlobifier.ShapeConfig(minY, height, hc, vc);
            ByteBuffer blob = NoiseRouterBlobifier.blobify(router, shape, noiseConfig);
            LOGGER.info("[c2me-wgpu] first chunk [{} {}], blob size={} bytes",
                    chunk.getPos().x, chunk.getPos().z, blob.remaining());

            maybeDumpBlobAndRef(router, noiseConfig, chunk, blob,
                    minY, height, hc, vc, gx, gy, gz);

            long inst = WgpuThinBinding.getInstanceHandle();
            long world = WgpuThinBinding.worldNew(inst, blob);
            if (world == 0L) {
                LOGGER.error("[c2me-wgpu] world_new returned 0; staying on CPU path");
                return null;
            }
            LOGGER.info("[c2me-wgpu] Rust WorldContext created, handle={}", world);
            // §22: per-dimension aquifer mode. Nether/End have no aquifers;
            // their default fluid is lava (Nether) / none (End), sea level 32/0.
            {
                int dfl = 0;
                var fb = settings.defaultFluid();
                if (fb.isOf(net.minecraft.block.Blocks.WATER)) dfl = 2;
                else if (fb.isOf(net.minecraft.block.Blocks.LAVA)) dfl = 3;
                WgpuThinBinding.setAquiferMode(world, settings.hasAquifers(),
                        settings.seaLevel(), dfl, -54);
            }
            // Ore-vein RNG seed: RandomSplitter exposes it through the public
            // addDebugInfo() hook (no reflection on obfuscated fields).
            if (settings.oreVeins()) {
                try {
                    StringBuilder sb = new StringBuilder();
                    noiseConfig.getOreRandomDeriver().addDebugInfo(sb);
                    java.util.regex.Matcher m = java.util.regex.Pattern
                            .compile("seedLo: (-?\\d+), seedHi: (-?\\d+)")
                            .matcher(sb);
                    if (m.find()) {
                        WgpuThinBinding.setOreSeed(world,
                                Long.parseLong(m.group(1)), Long.parseLong(m.group(2)));
                        LOGGER.info("[c2me-wgpu] ore-vein seed staged");
                    } else {
                        LOGGER.warn("[c2me-wgpu] could not parse ore seed from '{}'", sb);
                    }
                } catch (Throwable t) {
                    LOGGER.warn("[c2me-wgpu] ore seed staging failed", t);
                }
            }
            java.util.List<net.minecraft.world.gen.densityfunction.DensityFunction> extNodes =
                    new java.util.ArrayList<>(NoiseRouterBlobifier.EXTERNAL_NODES);
            if (!extNodes.isEmpty()) {
                LOGGER.info("[c2me-wgpu] {} external (CPU-evaluated) density subtree(s)",
                        extNodes.size());
            }
            WgpuBatchCoordinator coord = new WgpuBatchCoordinator(world, extNodes,
                    gx, gy, gz, hc, vc, minY);
            return new DimState(seed, world, coord, gx, gy, gz, hc, vc, minY, height, router,
                    settings.defaultBlock());
        } catch (Throwable t) {
            LOGGER.error("[c2me-wgpu] world init failed; staying on CPU path", t);
            String reason = (t.getMessage() != null)
                    ? t.getMessage()
                    : t.getClass().getSimpleName();
            WgpuFallbackNotice.report("dim-init",
                    "[c2me-wgpu] 该维度已回退 CPU：GPU 无法处理此地形（" + reason + "）");
            return null;
        }
    }

    /** Optional diagnostic: dump the router blob and a per-channel corner
     *  reference grid when a dump marker / system property is set. No-op
     *  otherwise (this is the source of src/test/resources/real_router.bin). */
    private static final java.util.concurrent.atomic.AtomicBoolean CHUNK_REF_DUMPED =
            new java.util.concurrent.atomic.AtomicBoolean(false);

    /** Dump the corner reference for the first chunk inside C2ME_DUMP_CHUNK,
     *  using the SAME decompiled tree as the blob (the `router` argument). */
    private static void maybeDumpChunkRef(Chunk chunk, DimState ds, NoiseRouter router,
                                          NoiseConfig noiseConfig) {
        if (!DUMP_ENABLED) return;
        int cx = chunk.getPos().x, cz = chunk.getPos().z;
        if (cx < DUMP_CHUNK[0] || cx > DUMP_CHUNK[2]
                || cz < DUMP_CHUNK[1] || cz > DUMP_CHUNK[3]) return;
        if (!CHUNK_REF_DUMPED.compareAndSet(false, true)) return;
        try {
            java.util.LinkedHashMap<String, net.minecraft.world.gen.densityfunction.DensityFunction> chans =
                    new java.util.LinkedHashMap<>();
            chans.put("barrierNoise", router.barrierNoise());
            chans.put("fluidLevelFloodednessNoise", router.fluidLevelFloodednessNoise());
            chans.put("fluidLevelSpreadNoise", router.fluidLevelSpreadNoise());
            chans.put("lavaNoise", router.lavaNoise());
            chans.put("temperature", router.temperature());
            chans.put("vegetation", router.vegetation());
            chans.put("continents", router.continents());
            chans.put("erosion", router.erosion());
            chans.put("depth", router.depth());
            chans.put("ridges", router.ridges());
            chans.put("initialDensityWithoutJaggedness", router.initialDensityWithoutJaggedness());
            chans.put("finalDensity", router.finalDensity());
            chans.put("veinToggle", router.veinToggle());
            chans.put("veinRidged", router.veinRidged());
            chans.put("veinGap", router.veinGap());
            StringBuilder sb = new StringBuilder();
            sb.append("# cx cz startX startZ\n");
            sb.append("CHUNK ").append(cx).append(' ').append(cz).append('\n');
            sb.append("DIMS ").append(ds.gridX).append(' ').append(ds.gridY).append(' ').append(ds.gridZ).append('\n');
            sb.append("CELL_W ").append(ds.cellW).append('\n');
            sb.append("MIN_Y ").append(ds.minY).append('\n');
            int sx = chunk.getPos().getStartX();
            int sz = chunk.getPos().getStartZ();
            for (var e : chans.entrySet()) {
                sb.append("CHANNEL ").append(e.getKey()).append('\n');
                net.minecraft.world.gen.densityfunction.DensityFunction fd = e.getValue();
                for (int guard = 0; guard < 32; guard++) {
                    if (fd instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.RegistryEntryHolder h) {
                        fd = h.function().value();
                    } else if (fd instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Wrapper w) {
                        fd = w.wrapped();
                    } else { break; }
                }
                for (int j = 0; j < ds.gridY; j++) {
                    int wy = ds.minY + j * ds.cellH;
                    for (int k = 0; k < ds.gridZ; k++) {
                        int wz = sz + k * ds.cellW;
                        for (int i = 0; i < ds.gridX; i++) {
                            int wx = sx + i * ds.cellW;
                            double v = fd.sample(new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(wx, wy, wz));
                            sb.append(i).append(' ').append(j).append(' ').append(k).append(' ').append((float) v).append('\n');
                        }
                    }
                }
            }
            String rp = dumpDir() + "/ref_chunk_" + cx + "_" + cz + ".txt";
            java.nio.file.Files.writeString(java.nio.file.Path.of(rp), sb.toString());
            LOGGER.info("[c2me-wgpu] dumped per-chunk ref (decompiled tree) to {}", rp);
            // §14 diag: also dump the COMPILED tree's depth, to compare against decompiled.
            try {
                NoiseRouter comp = noiseConfig.getNoiseRouter();
                StringBuilder cb = new StringBuilder();
                cb.append("CHANNEL depth\n");
                net.minecraft.world.gen.densityfunction.DensityFunction cfd = comp.depth();
                for (int guard=0; guard<32; guard++) {
                    if (cfd instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.RegistryEntryHolder h) cfd=h.function().value();
                    else if (cfd instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Wrapper w) cfd=w.wrapped();
                    else break;
                }
                for (int j=0;j<ds.gridY;j++){int wy=ds.minY+j*ds.cellH;
                  for (int k=0;k<ds.gridZ;k++){int wz=sz+k*ds.cellW;
                    for (int i=0;i<ds.gridX;i++){int wx=sx+i*ds.cellW;
                      double v=cfd.sample(new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(wx,wy,wz));
                      cb.append(i).append(' ').append(j).append(' ').append(k).append(' ').append((float)v).append('\n');}}}
                java.nio.file.Files.writeString(java.nio.file.Path.of(dumpDir() + "/ref_chunk_"+cx+"_"+cz+"_compiled.txt"), cb.toString());
                LOGGER.info("[c2me-wgpu] dumped COMPILED-tree depth ref to {}/ref_chunk_{}_{}_compiled.txt", dumpDir(), cx, cz);
            } catch (Throwable t2) { LOGGER.warn("[c2me-wgpu] compiled ref dump failed", t2); }
            // §14 diag: dump every Spline's locations/derivatives from MC's
            // decompiled depth tree, to diff against our Rust deserialization.
            try {
                final java.util.List<String> sp = new java.util.ArrayList<>();
                ds.rawRouter.depth().apply(new net.minecraft.world.gen.densityfunction.DensityFunction.DensityFunctionVisitor() {
                    @Override
                    public net.minecraft.world.gen.densityfunction.DensityFunction apply(
                            net.minecraft.world.gen.densityfunction.DensityFunction d) {
                        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Spline spl) {
                            try {
                                Object inner = spl.spline();
                                if (inner instanceof net.minecraft.util.math.Spline.Implementation<?,?> impl) {
                                    sp.add(java.util.Arrays.toString(impl.locations())
                                         + " | " + java.util.Arrays.toString(impl.derivatives()));
                                } else if (inner instanceof net.minecraft.util.math.Spline.FixedFloatFunction<?,?> f) {
                                    sp.add("FIXED " + f.value());
                                }
                            } catch (Throwable ignore) {}
                        }
                        return d;
                    }
                });
                java.nio.file.Files.writeString(java.nio.file.Path.of(dumpDir() + "/mc_depth_splines.txt"),
                        String.join("\n", sp));
                LOGGER.info("[c2me-wgpu] dumped {} MC splines to {}/mc_depth_splines.txt", sp.size(), dumpDir());
            } catch (Throwable t3) { LOGGER.warn("[c2me-wgpu] MC spline dump failed", t3); }
            // §14 diag: value trace of every depth node at the failing point.
            try {
                StringBuilder tb = new StringBuilder();
                var pos = new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(
                        sx + 0, ds.minY + 24 * ds.cellH, sz + 0); // cell(0,24,0) world(800,128,800)
                int[] counter = {0};
                valueTrace(ds.rawRouter.depth(), pos, tb, 0, counter);
                java.nio.file.Files.writeString(java.nio.file.Path.of(dumpDir() + "/mc_depth_values.txt"), tb.toString());
                LOGGER.info("[c2me-wgpu] dumped MC depth value trace ({} lines) to {}/mc_depth_values.txt", counter[0], dumpDir());
            } catch (Throwable t4) { LOGGER.warn("[c2me-wgpu] MC value trace failed", t4); }
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] per-chunk ref dump failed", t);
        }
    }

    private static String resolveType(net.minecraft.world.gen.densityfunction.DensityFunction d) {
        var T = net.minecraft.world.gen.densityfunction.DensityFunctionTypes.class;
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Constant) return "Const";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.RangeChoice) return "RangeChoice";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Spline) return "Spline";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.YClampedGradient) return "YClamp";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Noise) return "Noise";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.ShiftedNoise) return "ShiftedNoise";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.ShiftA) return "ShiftA";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.ShiftB) return "ShiftB";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Shift) return "Shift";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.UnaryOperation u) {
            return switch (u.type()) {
                case ABS -> "Abs"; case SQUARE -> "Square"; case CUBE -> "Cube";
                case SQUEEZE -> "Squeeze";
                case HALF_NEGATIVE -> "HalfNeg"; case QUARTER_NEGATIVE -> "QuarterNeg";
            };
        }
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Wrapper w) {
            return switch (w.type()) { case FLAT_CACHE -> "CacheFlat"; case CACHE2D -> "Cache2D";
                case CACHE_ONCE -> "CacheOnce"; case CACHE_ALL_IN_CELL -> "CacheCell"; case INTERPOLATED -> "Interp"; };
        }
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.RegistryEntryHolder h) {
            return "Holder(" + resolveType(h.function().value()) + ")";
        }
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.LinearOperation lo) {
            return lo.specificType() == net.minecraft.world.gen.densityfunction.DensityFunctionTypes.LinearOperation.SpecificType.ADD ? "Add" : "Mul";
        }
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.BinaryOperation bo) {
            return switch (bo.type()) { case ADD -> "Add"; case MUL -> "Mul"; case MIN -> "Min"; case MAX -> "Max"; };
        }
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Clamp) return "Clamp";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.WeirdScaledSampler) return "Weird";
        return d.getClass().getSimpleName();
    }

    private static void valueTrace(
            net.minecraft.world.gen.densityfunction.DensityFunction d,
            net.minecraft.world.gen.densityfunction.DensityFunction.NoisePos pos,
            StringBuilder sb, int ind, int[] counter) {
        // Transparent wrappers our AST does NOT represent (serialized as passthrough).
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.RegistryEntryHolder h) {
            valueTrace(h.function().value(), pos, sb, ind, counter); return;
        }
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Wrapper w
                && (w.type() == net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Wrapping.Type.CACHE_ONCE
                 || w.type() == net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Wrapping.Type.CACHE_ALL_IN_CELL)) {
            valueTrace(w.wrapped(), pos, sb, ind, counter); return;
        }
        String pad = "  ".repeat(Math.min(ind, 40));
        double v = d.sample(pos);
        String tag = resolveType(d);
        sb.append(pad).append(tag).append(" = ").append(String.format("%.6f", v)).append('\n');
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Spline sp) {
            try {
                Object inner = sp.spline();
                if (inner instanceof net.minecraft.util.math.Spline.Implementation<?,?> impl) {
                    float[] L = impl.locations();
                    if (L.length > 0 && Math.abs(L[0] - (-0.45f)) < 1e-4) {
                        sb.append(pad).append("  [SPLINE-STRUCT] ").append(impl.getDebugString()).append('\n');
                    }
                }
            } catch (Throwable ignore) {}
        }
        counter[0]++;
        if (ind > 300) return;
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.BinaryOperationLike b) {
            valueTrace(b.argument1(), pos, sb, ind+1, counter);
            valueTrace(b.argument2(), pos, sb, ind+1, counter);
        } else if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Unary u) {
            valueTrace(u.input(), pos, sb, ind+1, counter);
        } else if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Wrapper w) {
            valueTrace(w.wrapped(), pos, sb, ind+1, counter);
        } else if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.RangeChoice r) {
            valueTrace(r.input(), pos, sb, ind+1, counter);
        } else if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.ShiftedNoise sn) {
            valueTrace(sn.shiftX(), pos, sb, ind+1, counter);
            valueTrace(sn.shiftY(), pos, sb, ind+1, counter);
            valueTrace(sn.shiftZ(), pos, sb, ind+1, counter);
            // noise child (not a DensityFunction; our AST stores it as NoiseSampler)
            double dx = pos.blockX() * sn.xzScale() + sn.shiftX().sample(pos);
            double dy = pos.blockY() * sn.yScale() + sn.shiftY().sample(pos);
            double dz = pos.blockZ() * sn.xzScale() + sn.shiftZ().sample(pos);
            sb.append("  ".repeat(Math.min(ind+1, 40))).append("Noise = ")
              .append(String.format("%.6f", sn.noise().sample(dx, dy, dz))).append('\n');
            counter[0]++;
        }
    }

    private static void maybeDumpBlobAndRef(NoiseRouter router, NoiseConfig noiseConfig,
                                            Chunk chunk, ByteBuffer blob,
                                            int minY, int height, int cellW, int cellH,
                                            int gx, int gy, int gz) {
        String dumpPath = System.getProperty("c2me.dumpBlob");
        if (dumpPath == null || dumpPath.isEmpty()) {
            dumpPath = System.getenv("C2ME_DUMP_BLOB");
        }
        if (dumpPath == null || dumpPath.isEmpty()) {
            return;
        }
        // Per-dimension suffix so the three dimensions don't overwrite each
        // other's dumps, and we never clobber the committed Overworld fixture
        // src/test/resources/real_router.bin.
        dumpPath = dumpPath + "." + minY + "_" + height + "_" + cellW + "_" + cellH;
        try {
            java.nio.ByteBuffer dup = blob.duplicate();
            byte[] bytes = new byte[dup.remaining()];
            dup.get(bytes);
            java.nio.file.Files.write(java.nio.file.Path.of(dumpPath), bytes);
            LOGGER.info("[c2me-wgpu] dumped router blob ({} bytes) to {}", bytes.length, dumpPath);
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] blob dump failed", t);
        }
        try {
            StringBuilder sb = new StringBuilder();
            sb.append("# cx cz startX startZ\n");
            sb.append("CHUNK ").append(chunk.getPos().x).append(' ')
              .append(chunk.getPos().z).append('\n');
            sb.append("DIMS ").append(gx).append(" ").append(gy).append(" ").append(gz).append("\n");
            sb.append("CELL_W ").append(cellW).append("\nMIN_Y ").append(minY).append("\n");
            NoiseRouter CR = router;  // §5.8: must match the blob's tree (decompiled), not the compiled router
            java.util.LinkedHashMap<String, net.minecraft.world.gen.densityfunction.DensityFunction> chans =
                    new java.util.LinkedHashMap<>();
            chans.put("barrierNoise", CR.barrierNoise());
            chans.put("fluidLevelFloodednessNoise", CR.fluidLevelFloodednessNoise());
            chans.put("fluidLevelSpreadNoise", CR.fluidLevelSpreadNoise());
            chans.put("lavaNoise", CR.lavaNoise());
            chans.put("temperature", CR.temperature());
            chans.put("vegetation", CR.vegetation());
            chans.put("continents", CR.continents());
            chans.put("erosion", CR.erosion());
            chans.put("depth", CR.depth());
            chans.put("ridges", CR.ridges());
            chans.put("initialDensityWithoutJaggedness", CR.initialDensityWithoutJaggedness());
            chans.put("finalDensity", CR.finalDensity());
            chans.put("veinToggle", CR.veinToggle());
            chans.put("veinRidged", CR.veinRidged());
            chans.put("veinGap", CR.veinGap());
            int sx = chunk.getPos().getStartX();
            int sz = chunk.getPos().getStartZ();
            for (var e : chans.entrySet()) {
                sb.append("CHANNEL ").append(e.getKey()).append('\n');
                net.minecraft.world.gen.densityfunction.DensityFunction fd = e.getValue();
                for (int guard = 0; guard < 32; guard++) {
                    if (fd instanceof
                            net.minecraft.world.gen.densityfunction.DensityFunctionTypes.RegistryEntryHolder h) {
                        fd = h.function().value();
                    } else if (fd instanceof
                            net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Wrapper w) {
                        fd = w.wrapped();
                    } else {
                        break;
                    }
                }
                for (int j = 0; j < gy; j++) {
                    int wy = minY + j * cellH;
                    for (int k = 0; k < gz; k++) {
                        int wz = sz + k * cellW;
                        for (int i = 0; i < gx; i++) {
                            int wx = sx + i * cellW;
                            double v = fd.sample(
                                    new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(
                                            wx, wy, wz));
                            sb.append(i).append(' ').append(j).append(' ').append(k)
                              .append(' ').append((float) v).append('\n');
                        }
                    }
                }
            }
            String refPath = dumpPath + ".ref.txt";
            java.nio.file.Files.writeString(java.nio.file.Path.of(refPath), sb.toString());
            LOGGER.info("[c2me-wgpu] dumped corner reference to {}", refPath);
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] corner reference dump failed", t);
        }
    }

    /** Unwrap RegistryEntryHolder/Wrapper layers (same as the ref-dump). */
    private static net.minecraft.world.gen.densityfunction.DensityFunction unwrapRouter(
            net.minecraft.world.gen.densityfunction.DensityFunction fd) {
        for (int guard = 0; guard < 32; guard++) {
            if (fd instanceof
                    net.minecraft.world.gen.densityfunction.DensityFunctionTypes.RegistryEntryHolder h) {
                fd = h.function().value();
            } else if (fd instanceof
                    net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Wrapper w) {
                fd = w.wrapped();
            } else {
                break;
            }
        }
        return fd;
    }

    /** Get the raw (un-compiled) tree behind a c2me-opts-dfc compiled node.
     *  CompiledDensityFunction keeps the original DensityFunction in its
     *  blendingFallback; JIT evaluation is numerically identical to it. */
    private static net.minecraft.world.gen.densityfunction.DensityFunction decompileOne(
            net.minecraft.world.gen.densityfunction.DensityFunction df) {
        if (df == null) return null;
        net.minecraft.world.gen.densityfunction.DensityFunction cur = df;
        for (int guard = 0; guard < 8; guard++) {
            if (cur instanceof
                    net.minecraft.world.gen.densityfunction.DensityFunctionTypes.RegistryEntryHolder h) {
                cur = h.function().value();
                continue;
            }
            try {
                java.lang.reflect.Method m = null;
                for (Class<?> c = cur.getClass(); c != null && m == null; c = c.getSuperclass()) {
                    try { m = c.getDeclaredMethod("getFallback"); }
                    catch (NoSuchMethodException ignored) {}
                }
                if (m == null) break;
                m.setAccessible(true);
                Object o = m.invoke(cur);
                if (o instanceof net.minecraft.world.gen.densityfunction.DensityFunction d && d != cur) {
                    cur = d;
                    continue;
                }
                break;
            } catch (Throwable t) {
                break;
            }
        }
        return cur;
    }

    /** Rebuild a NoiseRouter whose channels are the raw trees behind the
     *  compiled router. Used as the blob source so GPU reproduces the SAME
     *  tree MC actually generates terrain with. */
    private static NoiseRouter decompileRouter(NoiseRouter c) {
        return new NoiseRouter(
                decompileOne(c.barrierNoise()),
                decompileOne(c.fluidLevelFloodednessNoise()),
                decompileOne(c.fluidLevelSpreadNoise()),
                decompileOne(c.lavaNoise()),
                decompileOne(c.temperature()),
                decompileOne(c.vegetation()),
                decompileOne(c.continents()),
                decompileOne(c.erosion()),
                decompileOne(c.depth()),
                decompileOne(c.ridges()),
                decompileOne(c.initialDensityWithoutJaggedness()),
                decompileOne(c.finalDensity()),
                decompileOne(c.veinToggle()),
                decompileOne(c.veinRidged()),
                decompileOne(c.veinGap()));
    }

}
