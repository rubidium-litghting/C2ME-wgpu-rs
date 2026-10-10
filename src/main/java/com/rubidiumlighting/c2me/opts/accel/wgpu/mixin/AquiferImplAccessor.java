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

import net.minecraft.world.gen.chunk.AquiferSampler;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/**
 * Accessors for {@link AquiferSampler.Impl}'s private per-cell caches, used
 * only by the aquifer reference dump (DEVELOPMENT §20). {@code waterLevels}
 * is the per-cell FluidLevel table; {@code blockPositions} packs each cell's
 * random sample point. startX/Y/Z + sizeX/Z define the cell grid.
 */
@Mixin(AquiferSampler.Impl.class)
public interface AquiferImplAccessor {
    @Accessor("waterLevels")
    AquiferSampler.FluidLevel[] getWaterLevels();

    @Accessor("blockPositions")
    long[] getBlockPositions();

    @Accessor("chunkNoiseSampler")
    net.minecraft.world.gen.chunk.ChunkNoiseSampler getChunkNoiseSampler();

    @Accessor("fluidLevelSampler")
    net.minecraft.world.gen.chunk.AquiferSampler.FluidLevelSampler getFluidLevelSampler();

    @Accessor("randomDeriver")
    net.minecraft.util.math.random.RandomSplitter getRandomDeriver();

    @org.spongepowered.asm.mixin.gen.Invoker("getWaterLevel")
    net.minecraft.world.gen.chunk.AquiferSampler.FluidLevel invokeGetWaterLevel(long pos);

    @Accessor("startX") int getStartX();
    @Accessor("startY") int getStartY();
    @Accessor("startZ") int getStartZ();
    @Accessor("sizeX") int getSizeX();
    @Accessor("sizeZ") int getSizeZ();
}
