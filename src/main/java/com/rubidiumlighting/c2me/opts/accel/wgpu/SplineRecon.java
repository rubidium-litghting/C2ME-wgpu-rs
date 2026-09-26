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

import net.minecraft.registry.entry.RegistryEntry;
import net.minecraft.util.math.Spline;
import net.minecraft.world.gen.densityfunction.DensityFunction;
import net.minecraft.world.gen.densityfunction.DensityFunctionTypes;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public final class SplineRecon {

    private static final AtomicInteger CALLS = new AtomicInteger();
    private static final int MAX_CALLS = 1;

    private static final Map<Object, Integer> LOC_FN_IDS = new IdentityHashMap<>();
    private static int nextLocFnId = 0;

    private static int implCount;

    private SplineRecon() {}

    public static void recon(DensityFunctionTypes.Spline sp) {
        if (CALLS.incrementAndGet() > MAX_CALLS) return;

        LOC_FN_IDS.clear();
        nextLocFnId = 0;
        implCount = 0;

        Spline<?, ?> root = sp.spline();
        System.err.println("[c2me-wgpu][spline-recon] ==== Spline #" + CALLS.get() + " ====");
        walk(root, 0);
        System.err.println("[c2me-wgpu][spline-recon] impls=" + implCount
                + " distinctLocFns=" + LOC_FN_IDS.size());
    }

    private static void walk(Spline<?, ?> node, int depth) {
        if (node == null) return;

        if (node instanceof Spline.Implementation<?, ?> impl) {
            implCount++;
            Object locFn = impl.locationFunction();
            int id = LOC_FN_IDS.computeIfAbsent(locFn, k -> nextLocFnId++);
            if (id < 3 || implCount <= 2) {
                System.err.println("[c2me-wgpu][spline-recon]   impl depth=" + depth
                        + " locFnId=" + id
                        + " locFnClass=" + (locFn == null ? "null" : locFn.getClass().getName()));
                describeLocFn(locFn);
            }
            List<? extends Spline<?, ?>> values = impl.values();
            if (values != null) {
                for (Spline<?, ?> v : values) {
                    walk(v, depth + 1);
                }
            }
        }
    }

    private static void describeLocFn(Object locFn) {
        if (!(locFn instanceof DensityFunctionTypes.Spline.DensityFunctionWrapper wrapper)) {
            System.err.println("[c2me-wgpu][spline-recon]     locFn not a DFWrapper");
            return;
        }
        RegistryEntry<DensityFunction> entry = wrapper.function();
        if (entry == null) {
            System.err.println("[c2me-wgpu][spline-recon]     wrapper.function()=null");
            return;
        }
        DensityFunction df = entry.value();
        System.err.println("[c2me-wgpu][spline-recon]     inner DF class="
                + (df == null ? "null" : df.getClass().getName()));
        System.err.println("[c2me-wgpu][spline-recon]     inner DF toString="
                + (df == null ? "null" : df.toString()));
    }
}
