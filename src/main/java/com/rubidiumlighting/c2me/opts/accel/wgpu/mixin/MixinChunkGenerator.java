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
import com.rubidiumlighting.c2me.opts.accel.wgpu.WgpuThinBinding;
import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.registry.entry.RegistryEntry;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.chunk.ChunkSection;
import net.minecraft.world.gen.StructureAccessor;
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
    private static volatile NoiseRouter REF_ROUTER = null;
    private static volatile NoiseRouter RAW_ROUTER = null;
    private static final AtomicBoolean TRUTH_DUMPED = new AtomicBoolean(false);

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
                REF_ROUTER = noiseConfig.getNoiseRouter();
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

            // ── per-chunk GPU compute + write-back ───────────────
            long req = WgpuThinBinding.requestChunks(world,
                    new int[]{chunk.getPos().x, chunk.getPos().z});
            if (req == 0L) {
                return;
            }
            int st = WgpuThinBinding.pollStatus(req);
            if (st != WgpuThinBinding.STATUS_DONE) {
                return;
            }
            java.lang.foreign.MemorySegment seg = WgpuThinBinding.takeResultBlob(req);
            writeDensityToChunk(chunk, seg);

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
    private static void writeDensityToChunk(Chunk chunk, java.lang.foreign.MemorySegment seg) {
        final int CHANNEL = 11;                 // finalDensity
        final int CELLS_PER_CHANNEL = 4 * 96 * 4;
        final int PAYLOAD_BASE = 0x88;
        final int CELL_W = 4;
        final int CELL_H = 4;
        final int CELLS_X = 4;
        final int CELLS_Y = 96;
        final int CELLS_Z = 4;
        final int MIN_Y = -64;

        long need = PAYLOAD_BASE + (long) (CHANNEL + 1) * CELLS_PER_CHANNEL * 4;
        if (seg.byteSize() < need) {
            return;
        }

        java.nio.ByteBuffer bb = seg.asByteBuffer();
        bb.order(java.nio.ByteOrder.LITTLE_ENDIAN);
        int base = PAYLOAD_BASE + CHANNEL * CELLS_PER_CHANNEL * 4;

        BlockState stone = Blocks.STONE.getDefaultState();
        BlockState air = Blocks.AIR.getDefaultState();


        float dmin = Float.POSITIVE_INFINITY, dmax = Float.NEGATIVE_INFINITY;
        int stoneCells = 0, airCells = 0;

        for (int cy = 0; cy < CELLS_Y; cy++) {
            for (int cz = 0; cz < CELLS_Z; cz++) {
                for (int cx = 0; cx < CELLS_X; cx++) {
                    int idx = cx + cz * CELLS_X + cy * CELLS_X * CELLS_Z;
                    float d = bb.getFloat(base + idx * 4);
                    if (d < dmin) dmin = d;
                    if (d > dmax) dmax = d;
                    if (d > 0.0f) stoneCells++; else airCells++;
                    BlockState bs = (d > 0.0f) ? stone : air;
                    for (int dy = 0; dy < CELL_H; dy++) {
                        int by = MIN_Y + cy * CELL_H + dy;
                        int secIdx = chunk.getSectionIndex(by);
                        if (secIdx < 0 || secIdx >= chunk.getSectionArray().length) continue;
                        ChunkSection sec = chunk.getSection(secIdx);
                        for (int dz = 0; dz < CELL_W; dz++) {
                            int bz = cz * CELL_W + dz;
                            for (int dx = 0; dx < CELL_W; dx++) {
                                int bx = cx * CELL_W + dx;
                                sec.setBlockState(bx & 15, by & 15, bz & 15, bs);
                            }
                        }
                    }
                }
            }
        }



        // Mismatch scan: compare ch6/8/10/11 MC vs GPU at y=8/-32.
        if (REF_ROUTER != null) {
            int cx0 = chunk.getPos().x * 16;
            int cz0 = chunk.getPos().z * 16;
            int[] wys = { -32, 8 };
            int[] chs = { 6, 8, 10, 11 };
            int[] bases = new int[4];
            for (int i = 0; i < 4; i++) bases[i] = PAYLOAD_BASE + chs[i] * CELLS_PER_CHANNEL * 4;
            StringBuilder sb = new StringBuilder();
            int any = 0;
            for (int wy : wys) {
                int cyCell = (wy - MIN_Y) / CELL_H;
                double[] mc = {
                    REF_ROUTER.continents().sample(new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(cx0, wy, cz0)),
                    REF_ROUTER.depth().sample(new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(cx0, wy, cz0)),
                    REF_ROUTER.initialDensityWithoutJaggedness().sample(new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(cx0, wy, cz0)),
                    REF_ROUTER.finalDensity().sample(new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(cx0, wy, cz0)),
                };
                int idx = cyCell * CELLS_X * CELLS_Z;
                for (int i = 0; i < 4; i++) {
                    float gpu = bb.getFloat(bases[i] + idx * 4);
                    double d = mc[i] - gpu;
                    if (Math.abs(d) > 0.02) {
                        any++;
                        sb.append(String.format(" y=%d ch%d mc=%.4f gpu=%.4f", wy, chs[i], mc[i], gpu));
                    }
                }
            }
            if (any > 0) {
                LOGGER.info("[c2me-wgpu][chdiv] ({},{}){}", chunk.getPos().x, chunk.getPos().z, sb);
            }
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
