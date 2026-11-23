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

import com.rubidiumlighting.c2me.opts.accel.wgpu.ChunkDumper;
import com.rubidiumlighting.c2me.opts.accel.wgpu.DimState;
import com.rubidiumlighting.c2me.opts.accel.wgpu.ErrorTranslator;
import com.rubidiumlighting.c2me.opts.accel.wgpu.ModuleEntryPoint;
import com.rubidiumlighting.c2me.opts.accel.wgpu.NoiseRouterBlobifier;
import com.rubidiumlighting.c2me.opts.accel.wgpu.WgpuBatchCoordinator;
import com.rubidiumlighting.c2me.opts.accel.wgpu.WgpuDimensionRegistry;
import com.rubidiumlighting.c2me.opts.accel.wgpu.WgpuFallbackNotice;
import com.rubidiumlighting.c2me.opts.accel.wgpu.WgpuStatus;
import com.rubidiumlighting.c2me.opts.accel.wgpu.WgpuThinBinding;
import com.rubidiumlighting.c2me.opts.accel.wgpu.config.Config;

import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.server.world.ServerChunkLoadingManager;
import net.minecraft.server.world.ServerChunkManager;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.structure.JigsawJunction;
import net.minecraft.util.Formatting;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.random.RandomSplitter;
import net.minecraft.util.math.random.Xoroshiro128PlusPlusRandom;
import net.minecraft.world.ChunkRegion;
import net.minecraft.world.Heightmap;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.chunk.ChunkSection;
import net.minecraft.world.gen.StructureAccessor;
import net.minecraft.world.gen.StructureTerrainAdaptation;
import net.minecraft.world.gen.StructureWeightSampler;
import net.minecraft.world.gen.chunk.AquiferSampler;
import net.minecraft.world.gen.chunk.Blender;
import net.minecraft.world.gen.chunk.ChunkGeneratorSettings;
import net.minecraft.world.gen.chunk.ChunkNoiseSampler;
import net.minecraft.world.gen.chunk.ChunkNoiseSampler.BlockStateSampler;
import net.minecraft.world.gen.chunk.GenerationShapeConfig;
import net.minecraft.world.gen.chunk.NoiseChunkGenerator;
import net.minecraft.world.gen.densityfunction.DensityFunction;
import net.minecraft.world.gen.densityfunction.DensityFunctionTypes;
import net.minecraft.world.gen.noise.NoiseConfig;
import net.minecraft.world.gen.noise.NoiseRouter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;

@Mixin(NoiseChunkGenerator.class)
public abstract class MixinChunkGenerator {

    private static final Logger LOGGER = LoggerFactory.getLogger("c2me-wgpu");

    /**
     * Hard cap on distinct CPU-evaluated density subtrees per dimension.
     *
     * <p>Each slot costs a per-chunk CPU sampling pass and one entry in the
     * binding-6 storage buffer, so the cap bounds both time and memory. Above
     * it the dimension falls back to vanilla generation instead of generating
     * at a few chunks per second.
     */
    private static final int MAX_EXTERNAL_NODES = 64;
    // Per-dimension state now lives in WgpuDimensionRegistry: the save-switch
    // teardown has to iterate and destroy all of it from outside the mixin,
    // which is impossible for fields declared here.

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
        if (Config.get().debugVanillaRef) {
            return;
        }

        try {
            // ── per-dimension state (context + coordinator + grid) ──
            NoiseChunkGenerator selfGen = (NoiseChunkGenerator) (Object) this;
            final ChunkGeneratorSettings settings = selfGen.getSettings().value();
            final DimState ds = acquireDimState(settings, noiseConfig, chunk, currentSeed);
            if (ds == null) {
                return; // already failed, or init failed -> vanilla CPU path
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
            ChunkNoiseSampler sampler =
                    chunk.getOrCreateChunkNoiseSampler(cs ->
                            ((NoiseChunkGeneratorInvoker) (Object) gen)
                                    .callCreateChunkNoiseSampler(cs, fSa, fBlender, fNc));
            final AquiferSampler aquifer = sampler.getAquiferSampler();
            // Aquifer table is mandatory: the BlockOut shader classifies
            // water/lava/stone from these cells. Without it every block below
            // sea level falls through to the "no cell found -> stone" branch.
            try {
                stageAquiferForGpu(ds, chunk, aquifer);
            } catch (Throwable t) {
                LOGGER.warn("[c2me-wgpu] aquifer stage failed", t);
            }
            // §8.4 diagnostic: dump this chunk's aquifer table if requested.
            if (ChunkDumper.DUMP_AQUIFER
                    && chunk.getPos().x >= ChunkDumper.DUMP_CHUNK[0] && chunk.getPos().x <= ChunkDumper.DUMP_CHUNK[2]
                    && chunk.getPos().z >= ChunkDumper.DUMP_CHUNK[1] && chunk.getPos().z <= ChunkDumper.DUMP_CHUNK[3]) {
                try {
                    ChunkDumper.dumpAquifer(chunk, ds.worldHandle, aquifer, ds.rawRouter);
                } catch (Throwable t) {
                    LOGGER.warn("[c2me-wgpu] aquifer dump failed", t);
                }
            }
            try { ChunkDumper.maybeDumpChunkRef(chunk, ds, ds.rawRouter, noiseConfig); }
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
            if (ChunkDumper.DUMP_BEARD && ChunkDumper.markBeardDumped()) {
                try {
                    boolean hasStruct = !structureAccessor
                            .getStructureStarts(chunk.getPos(),
                                    s2 -> s2.getTerrainAdaptation()
                                            != StructureTerrainAdaptation.NONE)
                            .isEmpty();
                    if (hasStruct) {
                        ChunkDumper.dumpBeard(chunk, StructureWeightSampler
                                .createStructureWeightSampler(structureAccessor, chunk.getPos()));
                    } else {
                        ChunkDumper.clearBeardDumped(); // keep looking
                    }
                } catch (Throwable t) {
                    LOGGER.warn("[c2me-wgpu] beard dump failed", t);
                    ChunkDumper.clearBeardDumped();
                }
            }
            // NOTE: the CPU ore-vein sampler used to be pulled out here for the
            // removed density write-back path. `sampler.getBlockStateSampler()`
            // re-runs the aquifer with raw density (a side effect) and is now
            // pure overhead, so it is gone. `extractOreVeinSampler` is kept for
            // the upcoming GPU vein work.

            // ── async scheduling path (§3.5) ───────────────────
            // If enabled and we can reach C2ME's SchedulingManager, submit the
            // GPU request WITHOUT blocking, and run the write-back on C2ME's
            // worker pool (positionedExecutor — same pool, no new thread).
            // Returning a pending future frees the worker to submit more
            // chunks, so batches can grow.
            final long pos = chunk.getPos().toLong();
            final com.ishland.c2me.base.common.scheduler.SchedulingManager sched =
                    schedulingManagerOf(structureAccessor);
            if (sched != null) {
                final Chunk fChunk = chunk;
                final DimState fDs = ds;
                final CompletableFuture<Chunk> result = new CompletableFuture<>();
                ds.coordinator.requestAsync(chunk.getPos().x, chunk.getPos().z)
                        .whenComplete((gpu, err) -> {
                            if (err != null) {
                                // GPU failed. Completing `result` exceptionally
                                // is not enough: C2ME logs the failure and
                                // abandons the chunk, leaving it stuck in the
                                // "upgrade to noise" state forever — which
                                // deadlocks anything waiting on it.
                                //
                                // So regenerate this chunk with vanilla
                                // instead. Calling populateNoise again
                                // re-enters our HEAD injection, where
                                // acquireDimState sees the dimension marked
                                // failed and returns without cancelling, so
                                // vanilla runs normally.
                                LOGGER.error("[c2me-wgpu] GPU batch failed; regenerating chunk {} {} on the CPU",
                                        fChunk.getPos().x, fChunk.getPos().z, err);
                                // Mark before re-entering, so the nested
                                // onPopulateNoise takes the CPU path. Cleanup
                                // of the coordinator happens there too —
                                // deliberately not here, because this callback
                                // runs on the completion thread and
                                // shutdown() would join it.
                                WgpuDimensionRegistry.markFailed(settings);
                                CompletableFuture<Chunk> vanilla;
                                try {
                                    vanilla = gen.populateNoise(
                                            fBlender, fNc, fSa, fChunk);
                                } catch (Throwable t) {
                                    result.completeExceptionally(t);
                                    return;
                                }
                                if (vanilla == null) {
                                    result.completeExceptionally(
                                            new IllegalStateException("vanilla populateNoise returned null"));
                                    return;
                                }
                                vanilla.whenComplete((c, e2) -> {
                                    if (e2 != null) {
                                        result.completeExceptionally(e2);
                                    } else {
                                        result.complete(c);
                                    }
                                });
                                return;
                            }
                            // Enqueue write-back onto C2ME's worker pool. The
                            // chunk's position lock is held by C2ME's
                            // ScheduledTask until `result` completes (§3.5).
                            sched.positionedExecutor(pos).execute(() -> {
                                final boolean prof = Config.get().debugStageTiming;
                                final long tWrite = prof ? System.nanoTime() : 0L;
                                try {
                                    writeDensityToChunk(fChunk, gpu.blockSeg,
                                            gpu.batchIndex, fDs);
                                    if (prof) {
                                        profileCpuAccumulate((System.nanoTime() - tWrite) / 1000);
                                    }
                                    if (ChunkDumper.DUMP_BLOCKS) {
                                        ChunkDumper.maybeDumpBlocks(fChunk, "gpu");
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
            // synchronous GPU request: §5 measured plain C2ME on the CPU
            // (desktop ~144 CPS) faster than our synchronous GPU path
            // (~80-110 CPS). Returning without cancelling lets vanilla/C2ME
            // CPU generation run.
            return;
        } catch (Throwable t) {
            LOGGER.error("[c2me-wgpu] GPU populateNoise failed, falling back to CPU", t);
        }
    }

    /**
     * Return this dimension's {@link DimState}, creating it on first use.
     *
     * <p>Returns null when the dimension is already known to be unusable (init
     * failed once) or when this attempt's init fails — the caller then leaves
     * generation to vanilla/C2ME on the CPU.
     *
     * <p>World init is expensive (~7s shader compile) and must happen once per
     * dimension. Without the double-checked lock, concurrent first-chunks
     * would each build a WorldContext and the losers would leak their Rust
     * handle + coordinator.
     *
     * <p>A cached state whose seed no longer matches belongs to a previous
     * world (the player loaded a different save): it is dropped and its handle
     * destroyed off-thread after a grace period.
     */
    private static DimState acquireDimState(ChunkGeneratorSettings settings,
                                            NoiseConfig noiseConfig, Chunk chunk,
                                            long currentSeed) {
        // This must run BEFORE the cached-state lookup. The failure callback
        // re-enters this method to regenerate a chunk on the CPU; if the
        // cached (dead) DimState were returned first, the caller would keep
        // submitting to a coordinator that has already failed.
        //
        // No cleanup here: shutdown() joins the batch thread, and the failure
        // callback runs on the completion thread that would be joined. The
        // stale state is released on the next seed change instead.
        if (WgpuDimensionRegistry.hasFailed(settings)) {
            return null;
        }
        DimState ds = WgpuDimensionRegistry.get(settings);
        if (ds != null && ds.seed != currentSeed) {
            WgpuDimensionRegistry.remove(settings);
            // Stop the batch coordinator before the world handle goes away:
            // its threads still reference that handle, and any queued job must
            // have its future completed so C2ME can release the position lock.
            ds.coordinator.shutdown();
            // shutdown() blocks until both coordinator threads have exited
            // and every in-flight Rust request handle has been consumed, so
            // the world can be destroyed synchronously. The previous code
            // guessed "5 seconds should be enough", which was both needlessly
            // long on fast machines and not guaranteed on slow ones.
            //
            // Blocking here is fine: this runs on the save-switch path, where
            // a pause is expected. If the GPU is genuinely wedged the
            // completion thread's own 30s timeout halts the JVM first.
            long oldHandle = ds.worldHandle;
            if (oldHandle != 0L) {
                try {
                    WgpuThinBinding.worldDestroy(oldHandle);
                } catch (Throwable t) {
                    LOGGER.warn("[c2me-wgpu] world_destroy failed for old handle", t);
                }
            }
            ds = null;
        }
        if (ds != null) {
            return ds;
        }
        synchronized (WgpuDimensionRegistry.creationLock()) {
            ds = WgpuDimensionRegistry.get(settings);
            if (ds != null) {
                return ds;
            }
            if (WgpuDimensionRegistry.hasFailed(settings)) {
                return null;
            }
            ds = initDimState(settings, noiseConfig, chunk, currentSeed);
            if (ds != null) {
                WgpuDimensionRegistry.put(settings, ds);
            } else {
                // Remember the failure so later chunks don't retry.
                WgpuDimensionRegistry.markFailed(settings);
            }
            return ds;
        }
    }

    @Inject(method = "populateNoise", at = @At("RETURN"), cancellable = true)
    private void afterPopulateNoise(Blender blender, NoiseConfig noiseConfig,
                                    StructureAccessor structureAccessor, Chunk chunk,
                                    CallbackInfoReturnable<CompletableFuture<Chunk>> cir) {
        if (!Config.get().debugVanillaRef) {
            return;
        }
        CompletableFuture<Chunk> f = cir.getReturnValue();
        if (f == null) {
            return;
        }
        cir.setReturnValue(f.thenApply(c -> {
            ChunkDumper.maybeDumpBlocks(c, "vanilla");
            return c;
        }));
    }

    // Aggregated CPU write-back timing: one averaged line every 128 chunks,
    // so the log isn't flooded (was ~1 line/chunk).

    /** Resolve C2ME's SchedulingManager from the worldgen StructureAccessor:
     *  world (ChunkRegion during worldgen) -> ServerWorld -> ServerChunkManager
     *  -> chunkLoadingManager (IVanillaChunkManager). Returns null on any
     *  failure, in which case the caller leaves generation to vanilla/C2ME on
     *  the CPU (no synchronous GPU fallback). */
    private static com.ishland.c2me.base.common.scheduler.SchedulingManager
            schedulingManagerOf(StructureAccessor sa) {
        try {
            Object w = sa.world;   // widened by accesswidener
            ServerWorld sw;
            if (w instanceof ChunkRegion cr) {
                sw = cr.toServerWorld();
            } else if (w instanceof ServerWorld s) {
                sw = s;
            } else {
                return null;
            }
            ServerChunkManager scm =
                    (ServerChunkManager) sw.getChunkManager();
            ServerChunkLoadingManager sclm = scm.chunkLoadingManager;
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
    private static BlockStateSampler
            extractOreVeinSampler(
                    BlockStateSampler full) {
        if (full == null) return null;
        try {
            for (java.lang.reflect.Field f : full.getClass().getDeclaredFields()) {
                if (List.class.isAssignableFrom(f.getType())) {
                    f.setAccessible(true);
                    List<?> list = (List<?>) f.get(full);
                    if (list == null) continue;
                    // Find the OreVein sampler by lambda enclosing class name;
                    // do NOT assume it is the last element (datapacks/mods can
                    // add more sources to the chain).
                    for (Object o : list) {
                        if (o != null && o.getClass().getName().contains("OreVein")) {
                            return (BlockStateSampler) o;
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

    /**
     * Write the GPU finalDensity channel (channel 11) to {@code chunk} as
     * STONE / AIR. Cell grid = (16/cellW) × (height/cellH) × (16/cellW)
     * = 4 × 96 × 4 = 1536 cells; each cell covers cellW×cellH×cellW blocks.
     */
    /** §3.5: write a chunk from GPU block-state bytes (u32/block: 0=air,
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
        // State code -> BlockState. Index 0 is air (skipped); anything past
        // the table falls back to the dimension's solid block.
        final BlockState[] states = {
                Blocks.AIR.getDefaultState(),
                stone,
                Blocks.WATER.getDefaultState(),
                Blocks.LAVA.getDefaultState(),
                Blocks.GRANITE.getDefaultState(),
                Blocks.TUFF.getDefaultState(),
                Blocks.COPPER_ORE.getDefaultState(),
                Blocks.RAW_COPPER_BLOCK.getDefaultState(),
                Blocks.DEEPSLATE_IRON_ORE.getDefaultState(),
                Blocks.RAW_IRON_BLOCK.getDefaultState(),
        };
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
                    BlockState bs = v < states.length ? states[v] : stone;
                    sec.setBlockState(bx, ly, bz, bs);
                }
            }
        }
        // Rebuild both heightmaps in one pass instead of two trackUpdate calls
        // per non-air block (~67k calls for a typical chunk). The vanilla
        // helper walks the sections column-wise, which is far cheaper.
        Heightmap.populateHeightmaps(chunk,
                Set.of(Heightmap.Type.OCEAN_FLOOR_WG,
                                 Heightmap.Type.WORLD_SURFACE_WG));
    }

    /** §3.5: the ONLY GPU write-back path. The GPU emits one block-state
     *  byte (0=air,1=stone,2=water,3=lava) per block; this copies them into
     *  the chunk palette and keeps the two WG heightmaps in sync.
     *
     *  The former "GPU density only, MC trilinear + aquifer + vein on the
     *  CPU" path was removed: §8.2 measured it as a net loss (CPU write-back
     *  pinned CPS at ~150 on desktop, ~6 on mobile). Ore veins are not yet
     *  emitted by the GPU — see the GPU-vein TODO.
     *
     *  `blockSeg` is null only if the BlockOut shard produced no blob (should
     *  not happen now that it is on by default); we log and leave the chunk
     *  empty rather than crashing the world. */
    // Aggregated CPU write-back timing (the last stage: copying the GPU's
    // block bytes into the chunk). One averaged line every 128 chunks, so the
    // log is not flooded at ~100 chunks/s. Gated by debug_stage_timing.
    private static final AtomicLong CPUPROF_N =
            new AtomicLong();
    private static final AtomicLong CPUPROF_US =
            new AtomicLong();

    private static void profileCpuAccumulate(long us) {
        // Synchronized: with two independent atomics, two threads could both
        // observe n >= 128 and the second would read cnt == 0 after the first
        // reset, dividing by zero and aborting the write-back.
        synchronized (CPUPROF_N) {
            long n = CPUPROF_N.incrementAndGet();
            CPUPROF_US.addAndGet(us);
            if (n >= 128) {
                long cnt = CPUPROF_N.getAndSet(0);
                long total = CPUPROF_US.getAndSet(0);
                if (cnt > 0) {
                    LOGGER.info("[c2me-wgpu][writeback-timing] avg over {} chunks: write={}us/chunk",
                            cnt, total / cnt);
                }
            }
        }
    }

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

    /** §8.4 reference: dump the aquifer per-cell table so the GPU reimplementation
     *  of getWaterLevel can be checked against vanilla. Format:
     *    AQ startX startY startZ sizeX sizeZ
     *    CELL index blockX blockY blockZ y state
     *  blockPositions is a packed long = BlockPos.asLong(blockX,blockY,blockZ). */
    /**
     * §8.4 Step B: warm up the aquifer's lazy waterLevels table, then serialize
     * the compact table (seed + per-cell lvl_y/state) and stage it to Rust.
     * blockPositions are recomputed on the Rust side from the seed.
     */
    /**
     * §8.4: pack the Beardifier's pieces/junctions (u32 LE) and stage to Rust.
     * Layout: [0]=piece_count, [1]=junction_count, then per piece 8 i32
     * (minX,minY,minZ,maxX,maxY,maxZ,adaptation,groundLevelDelta), then per
     * junction 3 i32. Adaptation codes match beard.rs / WGSL.
     */
    private static void stageBeardForGpu(DimState ds, Chunk chunk,
            StructureAccessor sa) {
        boolean hasStruct = !sa.getStructureStarts(chunk.getPos(),
                s2 -> s2.getTerrainAdaptation()
                        != StructureTerrainAdaptation.NONE)
                .isEmpty();
        if (!hasStruct) return;
        var sw = StructureWeightSampler
                .createStructureWeightSampler(sa, chunk.getPos());
        var pit = sw.pieceIterator;
        List<StructureWeightSampler.Piece> pieces =
                new ArrayList<>();
        while (pit.hasNext()) pieces.add(pit.next());
        pit.back(Integer.MAX_VALUE);
        var jit = sw.junctionIterator;
        List<JigsawJunction> juncs =
                new ArrayList<>();
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

    private static void stageAquiferForGpu(DimState ds, Chunk chunk,
            AquiferSampler aquifer) {
        if (!(aquifer instanceof AquiferSampler.Impl impl)) {
            return;
        }
        AquiferImplAccessor a = (AquiferImplAccessor) (Object) impl;
        int startX = a.getStartX(), startY = a.getStartY(), startZ = a.getStartZ();
        int sizeX = a.getSizeX(), sizeZ = a.getSizeZ();
        AquiferSampler.FluidLevel[] wl = a.getWaterLevels();

        // Read the splitter's seed words directly instead of parsing
        // addDebugInfo()'s string: the formatter is a debug-command detail
        // and a Minecraft wording change would silently corrupt terrain.
        XoroshiroSplitterAccessor splitter =
                (XoroshiroSplitterAccessor) a.getRandomDeriver();
        long sLo = splitter.c2me$getSeedLo();
        long sHi = splitter.c2me$getSeedHi();

        // Warm up every cell (MC fills waterLevels lazily on first apply).
        for (int idx = 0; idx < wl.length; idx++) {
            int ci = idx % sizeX;
            int ck = (idx / sizeX) % sizeZ;
            int cj = idx / (sizeX * sizeZ);
            int cx = startX + ci, cy = startY + cj, cz = startZ + ck;
            long h = MathHelper.hashCode(cx, cy, cz);
            var rr = new Xoroshiro128PlusPlusRandom(h ^ sLo, sHi);
            int bx = cx * 16 + rr.nextInt(10);
            int by = cy * 12 + rr.nextInt(9);
            int bz = cz * 16 + rr.nextInt(10);
            a.invokeGetWaterLevel(BlockPos.asLong(bx, by, bz));
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
                if (blk == Blocks.WATER) sc = 2;
                else if (blk == Blocks.LAVA) sc = 3;
            }
            tb.putInt(ly); tb.putInt(sc);
        }
        WgpuThinBinding.setAquiferTable(ds.worldHandle,
                chunk.getPos().x, chunk.getPos().z, tb.array());
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
            GenerationShapeConfig gsc =
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

            ChunkDumper.maybeDumpBlobAndRef(router, noiseConfig, chunk, blob,
                    minY, height, hc, vc, gx, gy, gz);

            long inst = WgpuThinBinding.getInstanceHandle();
            long world = WgpuThinBinding.worldNew(inst, blob);
            if (world == 0L) {
                LOGGER.error("[c2me-wgpu] world_new returned 0; staying on CPU path");
                return null;
            }
            LOGGER.info("[c2me-wgpu] Rust WorldContext created, handle={}", world);
            // §8.4: per-dimension aquifer mode. Nether/End have no aquifers;
            // their default fluid is lava (Nether) / none (End), sea level 32/0.
            {
                int dfl = 0;
                var fb = settings.defaultFluid();
                if (fb.isOf(Blocks.WATER)) dfl = 2;
                else if (fb.isOf(Blocks.LAVA)) dfl = 3;
                WgpuThinBinding.setAquiferMode(world, settings.hasAquifers(),
                        settings.seaLevel(), dfl, -54);
            }
            // Ore-vein RNG seed: RandomSplitter exposes it through the public
            // addDebugInfo() hook (no reflection on obfuscated fields).
            if (settings.oreVeins()) {
                try {
                    // Read the splitter's seed words directly. The previous
                    // addDebugInfo() + regex route ties ore placement to the
                    // wording of a debug formatter that Minecraft does not
                    // guarantee -- the same hazard already fixed for the
                    // aquifer seed.
                    RandomSplitter deriver =
                            noiseConfig.getOreRandomDeriver();
                    if (deriver instanceof
                            Xoroshiro128PlusPlusRandom.Splitter sp) {
                        XoroshiroSplitterAccessor acc = (XoroshiroSplitterAccessor) (Object) sp;
                        WgpuThinBinding.setOreSeed(world, acc.c2me$getSeedLo(), acc.c2me$getSeedHi());
                        LOGGER.info("[c2me-wgpu] ore-vein seed staged");
                    } else {
                        LOGGER.warn("[c2me-wgpu] unexpected ore splitter type {}; "
                                + "ore veins stay on CPU", deriver.getClass().getName());
                    }
                } catch (Throwable t) {
                    LOGGER.warn("[c2me-wgpu] ore seed staging failed", t);
                }
            }
            List<DensityFunction> extNodes =
                    new ArrayList<>(NoiseRouterBlobifier.EXTERNAL_NODES);
            // Slots whose every reference sits under FLAT_CACHE / CACHE2D can
            // be sampled once per column and repeated across Y. Copy the flags
            // now, while the serializer's static state still describes this
            // dimension.
            boolean[] extYCacheable = new boolean[extNodes.size()];
            int yCached = 0;
            for (int i = 0; i < extYCacheable.length; i++) {
                extYCacheable[i] = i < NoiseRouterBlobifier.EXTERNAL_Y_CACHEABLE.size()
                        && Boolean.TRUE.equals(
                                NoiseRouterBlobifier.EXTERNAL_Y_CACHEABLE.get(i));
                if (extYCacheable[i]) yCached++;
            }
            if (!extNodes.isEmpty()) {
                // References vs distinct instances: the tree is a DAG and the
                // serializer inlines it, so the same node is written many
                // times. Only distinct instances cost a slot (see
                // NoiseRouterBlobifier.EXTERNAL_INDEX).
                LOGGER.info("[c2me-wgpu] {} external (CPU-evaluated) density subtree(s)"
                                + " from {} reference(s); {} sampled once per column"
                                + " (FLAT_CACHE/CACHE2D)",
                        extNodes.size(), NoiseRouterBlobifier.EXTERNAL_REFS, yCached);
            }
            // Every external subtree is sampled on the CPU over a corner
            // lattice, once per chunk per node, and the results are shipped to
            // the GPU as one storage buffer. Both cost and memory scale with
            // the number of *distinct* nodes, so the cap below counts slots.
            // Refuse up front instead of generating at 2 chunks/s.
            if (extNodes.size() > MAX_EXTERNAL_NODES) {
                throw new IllegalStateException(
                        "too many external density subtrees (" + extNodes.size()
                                + " distinct > " + MAX_EXTERNAL_NODES
                                + ") for GPU acceleration");
            }
            WgpuBatchCoordinator coord = new WgpuBatchCoordinator(world, extNodes,
                    extYCacheable, gx, gy, gz, hc, vc, minY);
            // Announce GPU acceleration here, not on world join: "the GPU is
            // active" is only true for a dimension that actually initialised.
            // Reporting it on join raced with the per-dimension fallback notice,
            // so a failed dimension showed both "fell back to CPU" and
            // "GPU acceleration enabled".
            WgpuFallbackNotice.report("dim-active",
                    Formatting.GREEN,
                    "c2me-opts-accel-wgpu.notify.enabled",
                    WgpuStatus.gpuName());
            return new DimState(seed, world, coord, gx, gy, gz, hc, vc, minY, height, router,
                    settings.defaultBlock());
        } catch (Throwable t) {
            LOGGER.error("[c2me-wgpu] world init failed; staying on CPU path", t);
            String reason = (t.getMessage() != null)
                    ? t.getMessage()
                    : t.getClass().getSimpleName();
            // Recognised low-level errors are replaced by a translated,
            // player-readable explanation; unrecognised ones are passed
            // through raw so no diagnostic detail is lost.
            WgpuFallbackNotice.report("dim-init",
                    "c2me-opts-accel-wgpu.notify.dim_fallback",
                    ErrorTranslator.localize(reason));
            return null;
        }
    }

    /** Get the raw (un-compiled) tree behind a c2me-opts-dfc compiled node.
     *  CompiledDensityFunction keeps the original DensityFunction in its
     *  blendingFallback; JIT evaluation is numerically identical to it. */
    private static DensityFunction decompileOne(
            DensityFunction df) {
        if (df == null) return null;
        DensityFunction cur = df;
        for (int guard = 0; guard < 8; guard++) {
            if (cur instanceof
                    DensityFunctionTypes.RegistryEntryHolder h) {
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
                if (o instanceof DensityFunction d && d != cur) {
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
