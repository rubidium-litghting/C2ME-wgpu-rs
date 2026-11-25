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

package com.rubidiumlighting.c2me.opts.accel.wgpu;

import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.ChunkRegion;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.gen.StructureAccessor;
import net.minecraft.world.gen.chunk.Blender;
import net.minecraft.world.gen.chunk.NoiseChunkGenerator;
import net.minecraft.world.gen.noise.NoiseConfig;

import java.util.concurrent.CompletableFuture;

/**
 * The GPU work {@code NoiseChunkGenerator.populateNoise} does, reachable from
 * outside the mixin.
 *
 * <p>{@code MixinChunkGenerator} implements this, so the target generator gains
 * the method. The body is exactly what its {@code populateNoise} HEAD injection
 * runs: per-dimension state, sampler/aquifer/beard staging, then a
 * non-blocking submit to the batch coordinator. {@code onPopulateNoise} is now
 * only a call into it, which keeps the mixin thin and lets the {@code NOISE}
 * status start the same work earlier — that is Stage 3.
 *
 * <p>Nothing here is safe to call twice for the same chunk without checking the
 * stashed future first: it stages per-chunk GPU tables.
 */
public interface GpuNoiseSubmitter {

    /**
     * Sets this chunk's GPU work running.
     *
     * @param blender            the blender vanilla passes to {@code populateNoise}
     * @param noiseConfig        this dimension's noise config
     * @param structureAccessor  region-backed accessor for the beardifier
     * @param chunk              the chunk being generated
     * @return the future {@code populateNoise} should hand back, or
     *         {@code null} when this chunk must be left to vanilla/C2ME on the
     *         CPU (GPU disabled, dimension failed, or no async scheduler)
     */
    CompletableFuture<Chunk> c2me_wgpu$submitGpuNoise(Blender blender,
                                                      NoiseConfig noiseConfig,
                                                      StructureAccessor structureAccessor,
                                                      Chunk chunk);
}
