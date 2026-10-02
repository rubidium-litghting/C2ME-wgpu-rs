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

    /** Marker file: when present, skip GPU entirely so MC generates vanilla
     *  blocks; the RETURN injection then dumps them for comparison. */
    private static final java.nio.file.Path VANILLA_REF_MARKER =
            java.nio.file.Path.of("/tmp/c2me_vanilla_ref");
    private static final java.nio.file.Path DENSITY_DUMP_MARKER =
            java.nio.file.Path.of("/tmp/c2me_dump_density");

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
        if (java.nio.file.Files.exists(VANILLA_REF_MARKER)) {
            return;
        }

        try {
            // ── per-dimension state (context + coordinator + grid) ──
            NoiseChunkGenerator selfGen = (NoiseChunkGenerator) (Object) this;
            ChunkGeneratorSettings settings = selfGen.getSettings().value();
            DimState ds = DIM_STATES.get(settings);
            if (ds == null) {
                // Double-checked locking: world init is expensive (~7s shader
                // compile) and must run once per dimension. Without the lock,
                // concurrent first-chunks would each build a WorldContext, and
                // the loser's Rust handle + coordinator would leak forever.
                synchronized (DIM_STATES) {
                    ds = DIM_STATES.get(settings);
                    if (ds == null) {
                        ds = initDimState(settings, noiseConfig, chunk);
                        if (ds != null) {
                            DIM_STATES.put(settings, ds);
                        }
                    }
                }
                if (ds == null) {
                    return; // fall back to vanilla CPU path
                }
            }

            // ── GPU compute via the batch coordinator ────────────
            // Requests from all worker threads are coalesced into one
            // dispatch (see WgpuBatchCoordinator). The returned blob holds
            // the whole batch; batchIndex selects this chunk's slice.
            WgpuBatchCoordinator.GpuChunk gpu;
            try {
                gpu = ds.coordinator.request(chunk.getPos().x, chunk.getPos().z);
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
            net.minecraft.world.gen.chunk.ChunkNoiseSampler.BlockStateSampler oreVein =
                    ((ChunkNoiseSamplerAccessor) (Object) sampler).getBlockStateSampler();

            writeDensityToChunk(chunk, seg, aquifer, oreVein, gpu.batchIndex, structureAccessor, ds);
            maybeDumpBlocks(chunk, "gpu");

            // Cancel vanilla generation and hand back the (now filled) chunk.
            cir.setReturnValue(CompletableFuture.completedFuture(chunk));
        } catch (Throwable t) {
            LOGGER.error("[c2me-wgpu] GPU populateNoise failed, falling back to CPU", t);
        }
    }

    @Inject(method = "populateNoise", at = @At("RETURN"), cancellable = true)
    private void afterPopulateNoise(Blender blender, NoiseConfig noiseConfig,
                                    StructureAccessor structureAccessor, Chunk chunk,
                                    CallbackInfoReturnable<CompletableFuture<Chunk>> cir) {
        if (!java.nio.file.Files.exists(VANILLA_REF_MARKER)) {
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

    /** Dump chunk (0,0) block states for offline GPU-vs-vanilla comparison. */
    private static void maybeDumpBlocks(Chunk chunk, String mode) {
        if (chunk.getPos().x != 0 || chunk.getPos().z != 0) {
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
            String path = "/tmp/chunk_blocks_" + mode + ".txt";
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
    private static void writeDensityToChunk(Chunk chunk, java.lang.foreign.MemorySegment seg,
                                            net.minecraft.world.gen.chunk.AquiferSampler aquifer,
                                            net.minecraft.world.gen.chunk.ChunkNoiseSampler.BlockStateSampler oreVein,
                                            int batchIndex,
                                            StructureAccessor structureAccessor,
                                            DimState ds) {
        final int CHANNEL = 11;                 // finalDensity
        final int CELLS_X = ds.gridX;
        final int CELLS_Y = ds.gridY;
        final int CELLS_Z = ds.gridZ;
        final int CELLS_PER_CHANNEL = CELLS_X * CELLS_Y * CELLS_Z;
        final int PAYLOAD_BASE = 0x88;
        final int CELL_W = ds.cellW;
        final int CELL_H = ds.cellH;
        final int MIN_Y = ds.minY;

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

        BlockState stone = ds.defaultBlock;   // dim-specific solid (stone/netherrack/end_stone)
        BlockState air = Blocks.AIR.getDefaultState();
        Heightmap oceanFloorWg = chunk.getHeightmap(Heightmap.Type.OCEAN_FLOOR_WG);
        Heightmap worldSurfaceWg = chunk.getHeightmap(Heightmap.Type.WORLD_SURFACE_WG);

        StringBuilder densityDump = null;
        if (java.nio.file.Files.exists(DENSITY_DUMP_MARKER)
                && chunk.getPos().x == 0 && chunk.getPos().z == 0) {
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
        final int CHANNEL_VT = 12;
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
                    net.minecraft.world.gen.densityfunction.DensityFunction.NoisePos npos =
                            new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(
                                    chunk.getPos().getStartX() + bx, globalY,
                                    chunk.getPos().getStartZ() + bz);
                    // Beard is applied PER BLOCK (not at cell corners):
                    // MC's mapAll swaps Beardifier for StructureWeightSampler
                    // as a direct density node, so it is NOT trilinearly
                    // interpolated. Doing so flattened the structure
                    // foundations and made buildings float again.
                    double beard = (structureWeight != null)
                            ? structureWeight.sample(npos) : 0.0;
                    BlockState bs = aquifer.apply(npos, (double) d + beard);
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
                            }
                        }
                        bs = (vein != null) ? vein : stone;
                    }
                    if (bs == air) continue;
                    if (densityDump != null) {
                        int gx = chunk.getPos().getStartX() + bx;
                        int gz = chunk.getPos().getStartZ() + bz;
                        densityDump.append(gx).append(' ').append(globalY).append(' ')
                          .append(gz).append(' ').append((double) d + beard)
                          .append(' ')
                          .append(net.minecraft.registry.Registries.BLOCK
                              .getId(bs.getBlock()).toString()).append('\n');
                    }
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



        if (densityDump != null) {
            try {
                java.nio.file.Files.writeString(
                        java.nio.file.Path.of("/tmp/chunk_density_gpu.txt"),
                        densityDump.toString());
                LOGGER.info("[c2me-wgpu] dumped GPU density ({} blocks)",
                        densityDump.length());
            } catch (Throwable t) {
                LOGGER.warn("[c2me-wgpu] density dump failed", t);
            }
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
            WgpuBatchCoordinator coord = new WgpuBatchCoordinator(world);
            return new DimState(world, coord, gx, gy, gz, hc, vc, minY, height, router,
                    settings.defaultBlock());
        } catch (Throwable t) {
            LOGGER.error("[c2me-wgpu] world init failed; staying on CPU path", t);
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
        if ((dumpPath == null || dumpPath.isEmpty())
                && java.nio.file.Files.exists(java.nio.file.Path.of("/tmp/c2me_dump_blob"))) {
            dumpPath = "/tmp/real_router.bin";
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
