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

    /** §22 Step B: stage the per-chunk aquifer table for the GPU. */
    private static final boolean AQ_GPU =
            "1".equals(System.getenv("C2ME_WGPU_AQUIFER"));

    private static final boolean ASYNC_SCHED =
            !"0".equals(System.getenv("C2ME_WGPU_ASYNC_SCHED"));
    @Inject(method = "populateNoise", at = @At("HEAD"), cancellable = true)
    private void onPopulateNoise(Blender blender, NoiseConfig noiseConfig,
                                 StructureAccessor structureAccessor, Chunk chunk,
                                 CallbackInfoReturnable<CompletableFuture<Chunk>> cir) {
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
                        ds = initDimState(settings, noiseConfig, chunk);
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
            if (AQ_GPU) {
                try {
                    stageAquiferForGpu(ds, chunk, aquifer);
                } catch (Throwable t) {
                    LOGGER.warn("[c2me-wgpu] aquifer stage failed", t);
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
            net.minecraft.world.gen.chunk.ChunkNoiseSampler.BlockStateSampler fullBss =
                    ((ChunkNoiseSamplerAccessor) (Object) sampler).getBlockStateSampler();
            net.minecraft.world.gen.chunk.ChunkNoiseSampler.BlockStateSampler veinOnly =
                    extractOreVeinSampler(fullBss);
            final net.minecraft.world.gen.chunk.ChunkNoiseSampler.BlockStateSampler oreVein =
                    (veinOnly != null) ? veinOnly : fullBss;

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
                                    writeDensityToChunk(fChunk, gpu.seg, gpu.blockSeg,
                                            aquifer, oreVein, gpu.batchIndex, fSa, fDs);
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

    private static int[] parseDumpChunk() {
        String v = System.getenv("C2ME_DUMP_CHUNK");
        if (v == null || v.isBlank()) return new int[]{0, 0, 0, 0};
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
            String path = "/tmp/chunk_blocks_" + mode
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
        // §22 GPU state codes: 0=air, 1=stone, 2=water, 3=lava.
        BlockState water = Blocks.WATER.getDefaultState();
        BlockState lava = Blocks.LAVA.getDefaultState();
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
                    BlockState bs = (v == 2) ? water
                                  : (v == 3) ? lava
                                  : stone;
                    sec.setBlockState(bx, ly, bz, bs);
                    oceanFloorWg.trackUpdate(bx, globalY, bz, bs);
                    worldSurfaceWg.trackUpdate(bx, globalY, bz, bs);
                }
            }
        }
    }

    private static void writeDensityToChunk(Chunk chunk, java.lang.foreign.MemorySegment seg,
                                            java.lang.foreign.MemorySegment blockSeg,
                                            net.minecraft.world.gen.chunk.AquiferSampler aquifer,
                                            net.minecraft.world.gen.chunk.ChunkNoiseSampler.BlockStateSampler oreVein,
                                            int batchIndex,
                                            StructureAccessor structureAccessor,
                                            DimState ds) {
        // §20 minimal loop: if the GPU produced per-block state bytes, write
        // those directly (the GPU already did the d>0 decision). This skips
        // the CPU trilinear interp + aquifer entirely.
        if (blockSeg != null) {
            writeBlocksFromGpu(chunk, blockSeg, batchIndex, ds);
            return;
        }
        // Rust now copies out only the two consumed channels
        // (finalDensity -> slot 0, veinToggle -> slot 1); the other 14 are
        // never read back. See executor.rs OUT_CHANNELS.
        final int CHANNEL = 0;                  // finalDensity
        final int CELLS_X = ds.gridX;
        final int CELLS_Y = ds.gridY;
        final int CELLS_Z = ds.gridZ;
        final int CELLS_PER_CHANNEL = CELLS_X * CELLS_Y * CELLS_Z;
        final int PAYLOAD_BASE = 0x88;
        final int CELL_W = ds.cellW;
        final int CELL_H = ds.cellH;
        final int MIN_Y = ds.minY;
        final int OUT_CHANNELS = 2;

        // Blob layout (multi-chunk): payload = chunk0[2 channels][CPC],
        // chunk1[2 channels][CPC], ... The header channel_offsets only
        // describe chunk 0, so chunk i's channel c lives at
        //   PAYLOAD_BASE + i * 2 * CPC * 4 + c * CPC * 4
        long chunkStride = (long) OUT_CHANNELS * CELLS_PER_CHANNEL * 4;
        long need = PAYLOAD_BASE + (long) batchIndex * chunkStride
                + (long) (CHANNEL + 1) * CELLS_PER_CHANNEL * 4;
        if (seg.byteSize() < need) {
            return;
        }

        java.nio.ByteBuffer bb = seg.asByteBuffer();
        bb.order(java.nio.ByteOrder.LITTLE_ENDIAN);
        int base = (int) (PAYLOAD_BASE + (long) batchIndex * chunkStride
                + CHANNEL * CELLS_PER_CHANNEL * 4);

        final long tWriteStart = PROFILE ? System.nanoTime() : 0L;
        final int startX = chunk.getPos().getStartX();
        final int startZ = chunk.getPos().getStartZ();

        BlockState stone = ds.defaultBlock;   // dim-specific solid (stone/netherrack/end_stone)
        BlockState air = Blocks.AIR.getDefaultState();
        Heightmap oceanFloorWg = chunk.getHeightmap(Heightmap.Type.OCEAN_FLOOR_WG);
        Heightmap worldSurfaceWg = chunk.getHeightmap(Heightmap.Type.WORLD_SURFACE_WG);

        StringBuilder densityDump = null;
        StringBuilder aqBlocksDump = null;
        if (DUMP_AQ_BLOCKS
                && chunk.getPos().x >= DUMP_CHUNK[0] && chunk.getPos().x <= DUMP_CHUNK[2]
                && chunk.getPos().z >= DUMP_CHUNK[1] && chunk.getPos().z <= DUMP_CHUNK[3]) {
            aqBlocksDump = new StringBuilder();
        }
        if (DUMP_DENSITY
                && chunk.getPos().x >= DUMP_CHUNK[0] && chunk.getPos().x <= DUMP_CHUNK[2]
                && chunk.getPos().z >= DUMP_CHUNK[1] && chunk.getPos().z <= DUMP_CHUNK[3]) {
            densityDump = new StringBuilder();
        }

        // MC's real "Beardifier": shapes terrain around structures (flattens
        // / buries under villages so houses sit in the ground). The GPU
        // density lacks it, so structures floated/sank. Reuse MC's own
        // Only build the (expensive) structure-weight sampler when this chunk
        // actually contains a structure whose terrain adaptation is not NONE.
        // Vanilla's own predicate is `s.getTerrainAdaptation() != NONE` (see
        // StructureWeightSampler.method_42696); matching it exactly is what lets
        // us skip ~98k sample() calls on structure-free chunks. Using `s -> true`
        // here pulled in NONE-adaptation structures and effectively never skipped.
        boolean chunkHasStructure = !structureAccessor
                .getStructureStarts(chunk.getPos(),
                        s -> s.getTerrainAdaptation()
                                != net.minecraft.world.gen.StructureTerrainAdaptation.NONE)
                .isEmpty();
        StructureWeightSampler structureWeight = chunkHasStructure
                ? StructureWeightSampler.createStructureWeightSampler(
                        structureAccessor, chunk.getPos())
                : null;

        float dmin = Float.POSITIVE_INFINITY, dmax = Float.NEGATIVE_INFINITY;
        int stoneCells = 0, airCells = 0;
        int aqCalls = 0;
        // Sample every 16th aquifer call to estimate its cost without
        // wrecking JIT inlining of the hot block loop.
        int aqProbeCounter = 0;
        long aqProbeNanos = 0L;
        int aqProbeCalls = 0;

        // Precompute all cell values so we can trilinearly interpolate.
        float[] cells = new float[CELLS_PER_CHANNEL];
        for (int i = 0; i < CELLS_PER_CHANNEL; i++) {
            float d = bb.getFloat(base + i * 4);
            cells[i] = d;
            if (d < dmin) dmin = d;
            if (d > dmax) dmax = d;
            if (d > 0.0f) stoneCells++; else airCells++;
        }
        // veinToggle (ch12) corner cells — used only as a cheap coarse filter
        // for ore veins (the real vein logic is per-block + randomised).
        final int CHANNEL_VT = 1;
        int vtBase = (int) (PAYLOAD_BASE + (long) batchIndex * chunkStride
                + CHANNEL_VT * CELLS_PER_CHANNEL * 4);
        float[] vtCells = new float[CELLS_PER_CHANNEL];
        for (int i = 0; i < CELLS_PER_CHANNEL; i++) vtCells[i] = bb.getFloat(vtBase + i * 4);
        // Block-level trilinear interpolation.
        //
        // MC's ChunkNoiseSampler evaluates the density at every BLOCK via
        // trilinear interpolation of cell-corner samples. We only have one
        // value per cell (4x4x4 blocks), so we approximate: treat each
        // cell's value as its center sample and interpolate per block
        // between the two neighbouring cells in each axis. Without this,
        // an entire 4x4x4 cell flips between stone/air at once, which
        // fills the top few blocks of every ocean cell with stone.
        // Reusable noise position — one allocation per chunk instead of one
        // per block (~131k/chunk). See MutableNoisePos.
        final com.rubidiumlighting.c2me.opts.accel.wgpu.MutableNoisePos mutablePos =
                new com.rubidiumlighting.c2me.opts.accel.wgpu.MutableNoisePos();

        for (int by = 0; by < ds.height; by++) {
            int globalY = MIN_Y + by;
            int secIdx = chunk.getSectionIndex(globalY);
            if (secIdx < 0 || secIdx >= chunk.getSectionArray().length) continue;
            ChunkSection sec = chunk.getSection(secIdx);

            // Cell values are CENTER samples: cell (i,j,k) covers blocks
            // [4i, 4i+4) etc, centre at block 4i+2. A block bx in cell units
            // (centre at integer) sits at `bx/4 - 0.5`; floor gives the
            // lower cell, fraction the weight toward the upper one.
            // Cell values are CORNER samples: cell i covers blocks
            // [4i, 4i+4); its value is the density at block 4i. Interpolate
            // between corner[i] and corner[i+1] with weight (b%4)/4.
            int iy = by / CELL_H;
            int cy0 = iy, cy1 = iy + 1;
            float ty = (by % CELL_H) / (float) CELL_H;

            for (int bz = 0; bz < 16; bz++) {
                int iz = bz / CELL_W;
                int cz0 = iz, cz1 = iz + 1;
                float tz = (bz % CELL_W) / (float) CELL_W;

                for (int bx = 0; bx < 16; bx++) {
                    int ix = bx / CELL_W;
                    int cx0 = ix, cx1 = ix + 1;
                    float tx = (bx % CELL_W) / (float) CELL_W;

                    // cellAt() linearly extrapolates past the grid edge, so
                    // the last cell of a chunk is NOT flat — that flat strip
                    // was the visible chunk-boundary seam.
                    float c000 = cells[cx0 + cz0 * CELLS_X + cy0 * CELLS_X * CELLS_Z];
                    float c100 = cells[cx1 + cz0 * CELLS_X + cy0 * CELLS_X * CELLS_Z];
                    float c010 = cells[cx0 + cz0 * CELLS_X + cy1 * CELLS_X * CELLS_Z];
                    float c110 = cells[cx1 + cz0 * CELLS_X + cy1 * CELLS_X * CELLS_Z];
                    float c001 = cells[cx0 + cz1 * CELLS_X + cy0 * CELLS_X * CELLS_Z];
                    float c101 = cells[cx1 + cz1 * CELLS_X + cy0 * CELLS_X * CELLS_Z];
                    float c011 = cells[cx0 + cz1 * CELLS_X + cy1 * CELLS_X * CELLS_Z];
                    float c111 = cells[cx1 + cz1 * CELLS_X + cy1 * CELLS_X * CELLS_Z];

                    float x00 = c000 + tx * (c100 - c000);
                    float x10 = c010 + tx * (c110 - c010);
                    float x01 = c001 + tx * (c101 - c001);
                    float x11 = c011 + tx * (c111 - c011);
                    float y0 = x00 + ty * (x10 - x00);
                    float y1 = x01 + ty * (x11 - x01);
                    float d = y0 + tz * (y1 - y0);

                    // Let MC's aquifer decide: null = solid (stone),
                    // otherwise water / lava / air per local fluid level.
                    com.rubidiumlighting.c2me.opts.accel.wgpu.MutableNoisePos npos =
                            mutablePos.set(startX + bx, globalY, startZ + bz);
                    // Beard is applied PER BLOCK (not at cell corners):
                    // MC's mapAll swaps Beardifier for StructureWeightSampler
                    // as a direct density node, so it is NOT trilinearly
                    // interpolated. Doing so flattened the structure
                    // foundations and made buildings float again.
                    double beard = (structureWeight != null)
                            ? structureWeight.sample(npos) : 0.0;
                    BlockState bs;
                    if (PROFILE && (aqProbeCounter & 15) == 0) {
                        long tAq = System.nanoTime();
                        bs = aquifer.apply(npos, (double) d + beard);
                        aqProbeNanos += System.nanoTime() - tAq;
                        aqProbeCalls++;
                    } else {
                        bs = aquifer.apply(npos, (double) d + beard);
                    }
                    aqProbeCounter++;
                    if (PROFILE) aqCalls++;
                    if (bs == null) {
                        // Solid. Ore veins are per-block + randomised, so they
                        // cannot live on the corner grid. Only compute the
                        // veinToggle filter for solid blocks inside the vein Y
                        // band (copper 0..48, iron -60..-8 -> use -64..56);
                        // everything else skips the interpolation entirely.
                        BlockState vein = null;
                        if (oreVein != null && globalY >= -64 && globalY <= 56) {
                            float v000 = vtCells[cx0 + cz0 * CELLS_X + cy0 * CELLS_X * CELLS_Z];
                            float v100 = vtCells[cx1 + cz0 * CELLS_X + cy0 * CELLS_X * CELLS_Z];
                            float v010 = vtCells[cx0 + cz0 * CELLS_X + cy1 * CELLS_X * CELLS_Z];
                            float v110 = vtCells[cx1 + cz0 * CELLS_X + cy1 * CELLS_X * CELLS_Z];
                            float v001 = vtCells[cx0 + cz1 * CELLS_X + cy0 * CELLS_X * CELLS_Z];
                            float v101 = vtCells[cx1 + cz1 * CELLS_X + cy0 * CELLS_X * CELLS_Z];
                            float v011 = vtCells[cx0 + cz1 * CELLS_X + cy1 * CELLS_X * CELLS_Z];
                            float v111 = vtCells[cx1 + cz1 * CELLS_X + cy1 * CELLS_X * CELLS_Z];
                            float vx00 = v000 + tx * (v100 - v000);
                            float vx10 = v010 + tx * (v110 - v010);
                            float vx01 = v001 + tx * (v101 - v001);
                            float vx11 = v011 + tx * (v111 - v011);
                            float vy0 = vx00 + ty * (vx10 - vx00);
                            float vy1 = vx01 + ty * (vx11 - vx01);
                            float vt = vy0 + tz * (vy1 - vy0);
                            // MC's real gate is |veinToggle| + yClamp < 0.4
                            // -> skip. Interpolated vt is smoothed, so use a
                            // matching 0.4 threshold; this keeps oreCalls low.
                            if (Math.abs(vt) >= 0.4f) {
                                vein = oreVein.sample(npos);
                                // Guard: if extraction fell back to the
                                // chained source, its aquifer stage can return
                                // air/water (re-evaluated with raw density).
                                // OreVeinSampler never returns those.
                                if (vein != null && (vein.isAir()
                                        || vein.isOf(Blocks.WATER)
                                        || vein.isOf(Blocks.LAVA))) {
                                    vein = null;
                                }
                            }
                        }
                        bs = (vein != null) ? vein : stone;
                    }
                    if (densityDump != null) {
                        int lx0 = bx & 15, lz0 = bz & 15;
                        densityDump.append(lx0).append(' ').append(globalY).append(' ')
                          .append(lz0).append(' ').append((double) d + beard)
                          .append(' ')
                          .append(bs == air ? "air"
                              : net.minecraft.registry.Registries.BLOCK
                                  .getId(bs.getBlock()).toString())
                          .append('\n');
                    }
        if (aqBlocksDump != null) {
                        aqBlocksDump.append(startX + bx).append(' ')
                          .append(globalY).append(' ')
                          .append(startZ + bz).append(' ')
                          .append((double) d + beard).append(' ')
                          .append(bs == air ? "air"
                              : net.minecraft.registry.Registries.BLOCK
                                  .getId(bs.getBlock()).toString())
                          .append('\n');
                    }
                    if (bs == air) continue;
                    int lx = bx & 15;
                    int ly = globalY & 15;
                    int lz = bz & 15;
                    sec.setBlockState(lx, ly, lz, bs);
                    // Vanilla tracks both WG heightmaps right after writing.
                    // Skipping this left structure placement (villages!) with
                    // a stale surface height, so buildings floated.
                    oceanFloorWg.trackUpdate(lx, globalY, lz, bs);
                    worldSurfaceWg.trackUpdate(lx, globalY, lz, bs);
                }
            }
        }



        if (PROFILE) {
            long totalUs = (System.nanoTime() - tWriteStart) / 1_000L;
            profileCpuAccumulate(totalUs);
        }

        if (densityDump != null) {
            try {
                java.nio.file.Files.writeString(
                        java.nio.file.Path.of("/tmp/chunk_density_gpu_"
                                + chunk.getPos().x + "_" + chunk.getPos().z + ".txt"),
                        densityDump.toString());
                LOGGER.info("[c2me-wgpu] dumped GPU density ({} blocks)",
                        densityDump.length());
            } catch (Throwable t) {
                LOGGER.warn("[c2me-wgpu] density dump failed", t);
            }
        }

        if (DUMP_AQ_BLOCKS
                && chunk.getPos().x >= DUMP_CHUNK[0] && chunk.getPos().x <= DUMP_CHUNK[2]
                && chunk.getPos().z >= DUMP_CHUNK[1] && chunk.getPos().z <= DUMP_CHUNK[3]) {
            try {
                StringBuilder cb = new StringBuilder();
                cb.append("CHUNK ").append(chunk.getPos().x).append(' ')
                  .append(chunk.getPos().z).append('\n');
                cb.append("GRID ").append(CELLS_X).append(' ').append(CELLS_Y)
                  .append(' ').append(CELLS_Z).append(' ').append(CELL_W)
                  .append(' ').append(CELL_H).append(' ').append(MIN_Y).append('\n');
                for (int i = 0; i < CELLS_PER_CHANNEL; i++) {
                    cb.append(i).append(' ').append(cells[i]).append('\n');
                }
                java.nio.file.Files.writeString(
                        java.nio.file.Path.of("/tmp/aq_corners_"
                                + chunk.getPos().x + "_" + chunk.getPos().z + ".txt"),
                        cb.toString());
                LOGGER.info("[c2me-wgpu] dumped ch11 corners ({} cells)", CELLS_PER_CHANNEL);
            } catch (Throwable t) {
                LOGGER.warn("[c2me-wgpu] corner dump failed", t);
            }
        }

        if (aqBlocksDump != null) {
            try {
                java.nio.file.Files.writeString(
                        java.nio.file.Path.of("/tmp/aq_blocks_"
                                + chunk.getPos().x + "_" + chunk.getPos().z + ".txt"),
                        aqBlocksDump.toString());
                LOGGER.info("[c2me-wgpu] dumped per-block aquifer ref ({} blocks)",
                        aqBlocksDump.length());
            } catch (Throwable t) {
                LOGGER.warn("[c2me-wgpu] aq block dump failed", t);
            }
        }

        if (DUMP_AQUIFER
                && chunk.getPos().x >= DUMP_CHUNK[0] && chunk.getPos().x <= DUMP_CHUNK[2]
                && chunk.getPos().z >= DUMP_CHUNK[1] && chunk.getPos().z <= DUMP_CHUNK[3]) {
            dumpAquifer(chunk, ds.worldHandle, aquifer, ds.rawRouter);
        }

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
                    java.nio.file.Path.of("/tmp/beard_" + chunk.getPos().x + "_"
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
                java.nio.file.Files.write(java.nio.file.Path.of("/tmp/aquifer_table_"
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
            String path = "/tmp/aquifer_" + chunk.getPos().x + "_" + chunk.getPos().z + ".txt";
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
            String nbPath = "/tmp/aquifer_noise_" + chunk.getPos().x + "_" + chunk.getPos().z + ".txt";
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
            String shPath = "/tmp/aquifer_surfaces_" + chunk.getPos().x + "_" + chunk.getPos().z + ".txt";
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
        DimState(long worldHandle, WgpuBatchCoordinator coordinator,
                 int gridX, int gridY, int gridZ, int cellW, int cellH,
                 int minY, int height, NoiseRouter rawRouter,
                 net.minecraft.block.BlockState defaultBlock) {
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
                                         NoiseConfig noiseConfig, Chunk chunk) {
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
            java.util.List<net.minecraft.world.gen.densityfunction.DensityFunction> extNodes =
                    new java.util.ArrayList<>(NoiseRouterBlobifier.EXTERNAL_NODES);
            if (!extNodes.isEmpty()) {
                LOGGER.info("[c2me-wgpu] {} external (CPU-evaluated) density subtree(s)",
                        extNodes.size());
            }
            WgpuBatchCoordinator coord = new WgpuBatchCoordinator(world, extNodes,
                    gx, gy, gz, hc, vc, minY);
            return new DimState(world, coord, gx, gy, gz, hc, vc, minY, height, router,
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
            NoiseRouter CR = noiseConfig.getNoiseRouter();
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
