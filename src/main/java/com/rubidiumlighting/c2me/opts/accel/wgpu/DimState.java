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

import net.minecraft.block.BlockState;
import net.minecraft.world.gen.noise.NoiseRouter;

/**
 * Per-dimension GPU state: world handle, batch coordinator, grid dimensions,
 * and the decompiled router the blob was built from.
 *
 * <p>This used to be a nested class inside {@code MixinChunkGenerator}. Mixin
 * renames nested classes of a {@code @Mixin} target, so any class outside the
 * mixin package that referenced it by its source name failed with
 * {@code NoSuchMethodError} at runtime. It is a plain data holder, not part of
 * the mixin, so it lives in its own file.
 */
public final class DimState {
    public final long seed;
    public final long worldHandle;
    public final WgpuBatchCoordinator coordinator;
    public final int gridX, gridY, gridZ;   // corner counts per axis
    public final int cellW, cellH;          // cell size in blocks
    public final int minY, height;
    public final NoiseRouter rawRouter;
    /** Dimension's solid block (Overworld stone / Nether netherrack /
     *  End end_stone). The aquifer returns null for "solid", and we must
     *  fill with THIS, not a hard-coded stone — otherwise the Nether got
     *  filled with stone. */
    public final net.minecraft.block.BlockState defaultBlock;
    public DimState(long seed, long worldHandle, WgpuBatchCoordinator coordinator,
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
