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
     *  The other {@code C2ME_DUMP_*} switches live in {@link ChunkDumper}.
     */
    private static final boolean VANILLA_REF =
            "1".equals(System.getenv("C2ME_VANILLA_REF"));
    /** Async scheduling (§18.6b): non-blocking GPU request + write-back on
     *  C2ME's worker pool. This is the only GPU path. Set C2ME_WGPU_ASYNC_SCHED=0
     *  to disable it entirely (GPU off for this run, plain C2ME CPU generation),
     *  useful as an A/B baseline. */


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
            if (ChunkDumper.DUMP_AQUIFER) {
                LOGGER.info("[c2me-wgpu] stageAquiferForGpu called for chunk: {}, DUMP_CHUNK: {},{}", chunk.getPos(), ChunkDumper.DUMP_CHUNK[0], ChunkDumper.DUMP_CHUNK[1]);
            }
            // §22 diagnostic: dump this chunk's aquifer table if requested.
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
                                            != net.minecraft.world.gen.StructureTerrainAdaptation.NONE)
                            .isEmpty();
                    if (hasStruct) {
                        ChunkDumper.dumpBeard(chunk, net.minecraft.world.gen.StructureWeightSampler
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
                                LOGGER.error("[c2me-wgpu] Async GPU request failed! This usually means the GPU crashed or the native library panicked.", err);
                                // First, try to complete the result exceptionally so C2ME's future chain unwinds.
                                result.completeExceptionally(err);
                                
                                // If C2ME swallows the exception and deadlocks, this will forcibly terminate the game.
                                // We use Error to bypass C2ME's normal Exception catching.
                                throw new Error("C2ME-wgpu GPU Crash: " + err.getMessage(), err);
                            }
                            // Enqueue write-back onto C2ME's worker pool. The
                            // chunk's position lock is held by C2ME's
                            // ScheduledTask until `result` completes (§18.6b).
                            sched.positionedExecutor(pos).execute(() -> {
                                try {
                                    writeDensityToChunk(fChunk, gpu.blockSeg,
                                            gpu.batchIndex, fDs);
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
            // synchronous GPU request: §10.2 measured plain C2ME on the CPU
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
        DimState ds = DIM_STATES.get(settings);
        if (ds != null && ds.seed != currentSeed) {
            DIM_STATES.remove(settings);
            long oldHandle = ds.worldHandle;
            if (oldHandle != 0L) {
                java.util.concurrent.CompletableFuture
                        .delayedExecutor(5, java.util.concurrent.TimeUnit.SECONDS)
                        .execute(() -> {
                            try { WgpuThinBinding.worldDestroy(oldHandle); }
                            catch (Throwable ignored) {}
                        });
            }
            ds = null;
        }
        if (ds != null) {
            return ds;
        }
        if (FAILED_DIMS.contains(settings)) {
            return null;
        }
        synchronized (DIM_STATES) {
            ds = DIM_STATES.get(settings);
            if (ds != null) {
                return ds;
            }
            if (FAILED_DIMS.contains(settings)) {
                return null;
            }
            ds = initDimState(settings, noiseConfig, chunk, currentSeed);
            if (ds != null) {
                DIM_STATES.put(settings, ds);
            } else {
                // Remember the failure so later chunks don't retry.
                FAILED_DIMS.add(settings);
            }
            return ds;
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


    private static void stageAquiferForGpu(DimState ds, Chunk chunk,
            net.minecraft.world.gen.chunk.AquiferSampler aquifer) {
        if (!(aquifer instanceof net.minecraft.world.gen.chunk.AquiferSampler.Impl impl)) {
            return;
        }
        AquiferImplAccessor a = (AquiferImplAccessor) (Object) impl;
        int startX = a.getStartX(), startY = a.getStartY(), startZ = a.getStartZ();
        int sizeX = a.getSizeX(), sizeZ = a.getSizeZ();
        net.minecraft.world.gen.chunk.AquiferSampler.FluidLevel[] wl = a.getWaterLevels();

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


    /** Strip RegistryEntryHolder + Wrapper(cache) layers so a router
     *  node can be single-shot sampled for diagnostics. */

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

            ChunkDumper.maybeDumpBlobAndRef(router, noiseConfig, chunk, blob,
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
