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

import net.minecraft.world.gen.densityfunction.DensityFunction;

/**
 * A mutable, reusable {@link DensityFunction.NoisePos}.
 *
 * <p>Our block-level write-back calls {@code aquifer.apply(pos, ...)} and
 * {@code oreVein.sample(pos)} once per block (~131k/чunk with tall shapes).
 * Allocating a fresh {@code UnblendedNoisePos} each time produced millions of
 * short-lived objects per second — a GC cost MC's own sampler avoids (it IS
 * its own NoisePos). Reusing one instance per thread removes that allocation.
 *
 * <p>Samples consume the position synchronously (they read blockX/Y/Z
 * immediately and never retain the object), so mutation between calls is safe.
 * Not thread-safe: use one instance per worker thread.
 */
public final class MutableNoisePos implements DensityFunction.NoisePos {
    public int x, y, z;

    public MutableNoisePos set(int x, int y, int z) {
        this.x = x; this.y = y; this.z = z;
        return this;
    }

    @Override public int blockX() { return x; }
    @Override public int blockY() { return y; }
    @Override public int blockZ() { return z; }
}
