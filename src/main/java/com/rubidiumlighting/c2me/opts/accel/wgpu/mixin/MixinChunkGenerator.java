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

import com.rubidiumlighting.c2me.opts.accel.wgpu.NoiseRouterBlobifier;
import com.rubidiumlighting.c2me.opts.accel.wgpu.WgpuBatchCoordinator;
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
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

@Mixin(NoiseChunkGenerator.class)
public abstract class MixinChunkGenerator {

    private static final Logger LOGGER = LoggerFactory.getLogger("c2me-wgpu");
    private static final AtomicBoolean WORLD_INIT_ATTEMPTED = new AtomicBoolean(false);
    private static final AtomicLong WORLD_HANDLE = new AtomicLong(0L);
    private static volatile WgpuBatchCoordinator COORDINATOR = null;
    private static volatile NoiseRouter RAW_ROUTER = null;
    private static final AtomicBoolean TRUTH_DUMPED = new AtomicBoolean(false);

    // Cell grid dims (corner samples per axis). MC uses 5x97x5 corners;
    // we currently ship 4x96x4 + edge extrapolation (see cellAt).
    private static final int CELLS_X = 5;
    private static final int CELLS_Y = 97;
    private static final int CELLS_Z = 5;

    @Inject(method = "populateNoise", at = @At("HEAD"), cancellable = true)
    private void onPopulateNoise(Blender blender, NoiseConfig noiseConfig,
                                 StructureAccessor structureAccessor, Chunk chunk,
                                 CallbackInfoReturnable<CompletableFuture<Chunk>> cir) {
        if (!WgpuThinBinding.isAvailable()) {
            return;
        }

        try {
            // ── one-time world init ──────────────────────────────
            if (WORLD_INIT_ATTEMPTED.compareAndSet(false, true)) {
                // Use settings.noiseRouter(): the RAW, un-compiled node tree.
                // (noiseConfig.getNoiseRouter() returns c2me-opts-dfc's compiled
                // JVM bytecode, which cannot be serialized.)
                NoiseChunkGenerator self = (NoiseChunkGenerator) (Object) this;
                NoiseRouter router = getNoiseRouter(self);
                if (router == null) {
                    LOGGER.warn("[c2me-wgpu] NoiseRouter is null, falling back to CPU");
                    return;
                }
                NoiseRouterBlobifier.ShapeConfig shape = new NoiseRouterBlobifier.ShapeConfig(-64, 384, 4, 4);
                RAW_ROUTER = router;
                ByteBuffer blob = NoiseRouterBlobifier.blobify(router, shape, noiseConfig);
                LOGGER.info("[c2me-wgpu] first chunk [{} {}], blob size={} bytes",
                        chunk.getPos().x, chunk.getPos().z, blob.remaining());
                trySendToRust(blob);
            }

            long world = WORLD_HANDLE.get();
            if (world == 0L) {
                return;
            }
            WgpuBatchCoordinator coord = COORDINATOR;
            if (coord == null) {
                synchronized (MixinChunkGenerator.class) {
                    if (COORDINATOR == null) {
                        COORDINATOR = new WgpuBatchCoordinator(world);
                    }
                    coord = COORDINATOR;
                }
            }

            // ── GPU compute via the batch coordinator ────────────
            // Requests from all worker threads are coalesced into one
            // dispatch (see WgpuBatchCoordinator). The returned blob holds
            // the whole batch; batchIndex selects this chunk's slice.
            WgpuBatchCoordinator.GpuChunk gpu;
            try {
                gpu = coord.request(chunk.getPos().x, chunk.getPos().z);
            } catch (Throwable t) {
                LOGGER.warn("[c2me-wgpu] GPU batch request failed, falling back to CPU", t);
                return; // vanilla CPU path runs
            }
            java.lang.foreign.MemorySegment seg = gpu.seg;

            // Reuse MC's own AquiferSampler: it decides water/lava/air vs
            // solid from the density value we computed on the GPU. This
            // replaces the crude global sea-level fill.
            // Reuse the chunk's cached sampler (vanilla buildSurface does the
            // same). Creating a fresh ChunkNoiseSampler per call was a large
            // CPU cost (it rebuilds the whole density-function graph).
            NoiseChunkGenerator gen = (NoiseChunkGenerator) (Object) this;
            final Blender fBlender = blender;
            final StructureAccessor fSa = structureAccessor;
            final NoiseConfig fNc = noiseConfig;
            net.minecraft.world.gen.chunk.ChunkNoiseSampler sampler =
                    chunk.getOrCreateChunkNoiseSampler(cs ->
                            ((NoiseChunkGeneratorInvoker) (Object) gen)
                                    .callCreateChunkNoiseSampler(cs, fSa, fBlender, fNc));
            net.minecraft.world.gen.chunk.AquiferSampler aquifer = sampler.getAquiferSampler();

            writeDensityToChunk(chunk, seg, aquifer, gpu.batchIndex, structureAccessor);

            // Cancel vanilla generation and hand back the (now filled) chunk.
            cir.setReturnValue(CompletableFuture.completedFuture(chunk));
        } catch (Throwable t) {
            LOGGER.error("[c2me-wgpu] GPU populateNoise failed, falling back to CPU", t);
        }
    }

    private static void trySendToRust(ByteBuffer blob) {
        try {
            long inst = WgpuThinBinding.getInstanceHandle();
            long world = WgpuThinBinding.worldNew(inst, blob);
            WORLD_HANDLE.set(world);
            LOGGER.info("[c2me-wgpu] Rust WorldContext created, handle={}", world);
        } catch (Throwable t) {
            LOGGER.error("[c2me-wgpu] Rust world_new failed; staying on CPU path", t);
        }
    }

    /**
     * Write the GPU finalDensity channel (channel 11) to {@code chunk} as
     * STONE / AIR. Cell grid = (16/cellW) × (height/cellH) × (16/cellW)
     * = 4 × 96 × 4 = 1536 cells; each cell covers cellW×cellH×cellW blocks.
     */
    private static void writeDensityToChunk(Chunk chunk, java.lang.foreign.MemorySegment seg,
                                            net.minecraft.world.gen.chunk.AquiferSampler aquifer,
                                            int batchIndex,
                                            StructureAccessor structureAccessor) {
        final int CHANNEL = 11;                 // finalDensity
        final int CELLS_PER_CHANNEL = CELLS_X * CELLS_Y * CELLS_Z;
        final int PAYLOAD_BASE = 0x88;
        final int CELL_W = 4;
        final int CELL_H = 4;
        final int MIN_Y = -64;

        // Blob layout (multi-chunk): payload = chunk0[16 channels][CPC],
        // chunk1[16 channels][CPC], ... The header channel_offsets only
        // describe chunk 0, so chunk i's channel c lives at
        //   PAYLOAD_BASE + i * 16 * CPC * 4 + c * CPC * 4
        long chunkStride = 16L * CELLS_PER_CHANNEL * 4;
        long need = PAYLOAD_BASE + (long) batchIndex * chunkStride
                + (long) (CHANNEL + 1) * CELLS_PER_CHANNEL * 4;
        if (seg.byteSize() < need) {
            return;
        }

        java.nio.ByteBuffer bb = seg.asByteBuffer();
        bb.order(java.nio.ByteOrder.LITTLE_ENDIAN);
        int base = (int) (PAYLOAD_BASE + (long) batchIndex * chunkStride
                + CHANNEL * CELLS_PER_CHANNEL * 4);

        BlockState stone = Blocks.STONE.getDefaultState();
        BlockState air = Blocks.AIR.getDefaultState();
        Heightmap oceanFloorWg = chunk.getHeightmap(Heightmap.Type.OCEAN_FLOOR_WG);
        Heightmap worldSurfaceWg = chunk.getHeightmap(Heightmap.Type.WORLD_SURFACE_WG);

        // MC's real "Beardifier": shapes terrain around structures (flattens
        // / buries under villages so houses sit in the ground). The GPU
        // density lacks it, so structures floated/sank. Reuse MC's own
        // sampler — for chunks with no structure the piece list is empty and
        // sample() returns 0, so the cost is negligible.
        StructureWeightSampler structureWeight =
                StructureWeightSampler.createStructureWeightSampler(
                        structureAccessor, chunk.getPos());


        float dmin = Float.POSITIVE_INFINITY, dmax = Float.NEGATIVE_INFINITY;
        int stoneCells = 0, airCells = 0;

        // Precompute all cell values so we can trilinearly interpolate.
        float[] cells = new float[CELLS_PER_CHANNEL];
        for (int i = 0; i < CELLS_PER_CHANNEL; i++) {
            float d = bb.getFloat(base + i * 4);
            cells[i] = d;
            if (d < dmin) dmin = d;
            if (d > dmax) dmax = d;
            if (d > 0.0f) stoneCells++; else airCells++;
        }

        // Block-level trilinear interpolation.
        //
        // MC's ChunkNoiseSampler evaluates the density at every BLOCK via
        // trilinear interpolation of cell-corner samples. We only have one
        // value per cell (4x4x4 blocks), so we approximate: treat each
        // cell's value as its center sample and interpolate per block
        // between the two neighbouring cells in each axis. Without this,
        // an entire 4x4x4 cell flips between stone/air at once, which
        // fills the top few blocks of every ocean cell with stone.
        for (int by = 0; by < 384; by++) {
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
                    net.minecraft.world.gen.densityfunction.DensityFunction.NoisePos npos =
                            new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(
                                    chunk.getPos().getStartX() + bx, globalY,
                                    chunk.getPos().getStartZ() + bz);
                    // Beard is applied PER BLOCK (not at cell corners):
                    // MC's mapAll swaps Beardifier for StructureWeightSampler
                    // as a direct density node, so it is NOT trilinearly
                    // interpolated. Doing so flattened the structure
                    // foundations and made buildings float again.
                    double beard = structureWeight.sample(npos);
                    BlockState bs = aquifer.apply(npos, (double) d + beard);
                    if (bs == null) bs = stone;
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



        // One-shot: dump ch10's raw structure.
        if (TRUTH_DUMPED.compareAndSet(false, true) && RAW_ROUTER != null) {
            String ds = RAW_ROUTER.initialDensityWithoutJaggedness().toString();
            LOGGER.info("[c2me-wgpu][ch10] len={} head={}", ds.length(),
                    ds.substring(0, Math.min(2500, ds.length())));
        }
    }




    private NoiseRouter getNoiseRouter(NoiseChunkGenerator generator) {
        try {
            RegistryEntry<ChunkGeneratorSettings> entry = generator.getSettings();
            ChunkGeneratorSettings settings = entry.value();
            return settings.noiseRouter();
        } catch (Exception e) {
            LOGGER.error("Failed to get NoiseRouter", e);
            return null;
        }
    }
}
