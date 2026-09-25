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

import net.minecraft.util.math.noise.DoublePerlinNoiseSampler;
import net.minecraft.util.math.noise.OctavePerlinNoiseSampler;
import net.minecraft.util.math.noise.PerlinNoiseSampler;
import net.minecraft.world.gen.densityfunction.DensityFunction;
import net.minecraft.world.gen.densityfunction.DensityFunctionTypes;
import net.minecraft.world.gen.noise.NoiseConfig;
import net.minecraft.world.gen.noise.NoiseRouter;

import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;

public class NoiseRouterBlobifier {

    // ── Permutation resource collection ─────────────────────────────
    private static final int PERM_MAGIC = 0x4D525750; // "PWPM"
    private static final IdentityHashMap<Object, Void> SEEN_PERM = new IdentityHashMap<>();
    private static final List<Object> PERM_SAMPLERS = new ArrayList<>();
    private static Object lastDperlin = null;
    private static NoiseConfig CURRENT_NOISE_CONFIG = null;

    // ── Tags ────────────────────────────────────────────────────────
    private static final byte TAG_CONSTANT = 0x01;
    private static final byte TAG_INPUT_X = 0x02;
    private static final byte TAG_INPUT_Y = 0x03;
    private static final byte TAG_INPUT_Z = 0x04;
    private static final byte TAG_ADD = 0x10;
    private static final byte TAG_SUB = 0x11;
    private static final byte TAG_MUL = 0x12;
    private static final byte TAG_DIV = 0x13;
    private static final byte TAG_MIN = 0x14;
    private static final byte TAG_MAX = 0x15;
    private static final byte TAG_ABS = 0x20;
    private static final byte TAG_SQUARE = 0x21;
    private static final byte TAG_CUBE = 0x22;
    private static final byte TAG_SQRT = 0x23;
    private static final byte TAG_FLOOR_DIV = 0x24;
    private static final byte TAG_NOISE_SAMPLER = (byte) 0x84;
    private static final byte TAG_INTERP_NOISE = (byte) 0x90;
    private static final byte TAG_END_ISLANDS = (byte) 0xA0;
    private static final byte TAG_BEARDIFIER = (byte) 0xA1;
    private static final byte TAG_RANGE_CHOICE = (byte) 0xB1;
    private static final byte TAG_INTERVAL_SEL = (byte) 0xB2;
    private static final byte TAG_Y_CLAMPED_GRAD = (byte) 0xB3;
    private static final byte TAG_CACHE_FLAT = (byte) 0xC0;
    private static final byte TAG_CACHE_2D = (byte) 0xC1;
    private static final byte TAG_CACHE_INTERP = (byte) 0xC2;
    private static final byte TAG_FIND_TOP_SURF = (byte) 0xD0;
    private static final byte TAG_AQUIFER = (byte) 0xE0;
    private static final byte TAG_BIOME_TREE = (byte) 0xF0;
    private static final byte TAG_ROOT = (byte) 0xFE;

    public static ByteBuffer blobify(NoiseRouter router, ShapeConfig shape, NoiseConfig noiseConfig) {
        CURRENT_NOISE_CONFIG = noiseConfig;
        ByteBuffer buf = ByteBuffer.allocate(4 * 1024 * 1024).order(ByteOrder.LITTLE_ENDIAN);
        SEEN_PERM.clear();
        PERM_SAMPLERS.clear();
        // ── HEADER (10B) ─────────────────────────────────────────
        buf.put((byte) 0x4D);
        buf.put((byte) 0x43);
        buf.put((byte) 0x32);
        buf.put((byte) 0x57);
        buf.putShort((short) 1);
        buf.putInt(0);

        // ── ROOT + 15 channels ──────────────────────────────────
        buf.put(TAG_ROOT);
        buf.putShort((short) 15);

        writeDF(buf, router.barrierNoise());
        writeDF(buf, router.fluidLevelFloodednessNoise());
        writeDF(buf, router.fluidLevelSpreadNoise());
        writeDF(buf, router.lavaNoise());
        writeDF(buf, router.temperature());
        writeDF(buf, router.vegetation());
        writeDF(buf, router.continents());
        writeDF(buf, router.erosion());
        writeDF(buf, router.depth());
        writeDF(buf, router.ridges());
        writeDF(buf, router.initialDensityWithoutJaggedness());
        writeDF(buf, router.finalDensity());
        writeDF(buf, router.veinToggle());
        writeDF(buf, router.veinRidged());
        writeDF(buf, router.veinGap());

        // ── SHAPE_CONFIG (16B) ──────────────────────────────────
        buf.putInt(shape.minY);
        buf.putInt(shape.height);
        buf.putInt(shape.cellWidth);
        buf.putInt(shape.cellHeight);

        // ── BIOME_SOURCE placeholder ────────────────────────────
        buf.put(TAG_BIOME_TREE);
        buf.putLong(0);

        // ── PERM_RESOURCE (appended; Rust stores for later use) ─
        writePermResource(buf);

        buf.flip();
        return buf;
    }

    private static void writeConstant(ByteBuffer buf, float value) {
        buf.put(TAG_CONSTANT);
        buf.putFloat(value);
    }

    private static void writeDF(ByteBuffer buf, DensityFunction df) {
        if (df == null) {
            writeConstant(buf, 0.0f);
            return;
        }

        // ── Registry holder unwrap ──────────────────────────────
        if (df instanceof DensityFunctionTypes.RegistryEntryHolder h) {
            writeDF(buf, h.function().value());
            return;
        }

        // ── Blend pseudo-nodes ──────────────────────────────────
        if (df instanceof DensityFunctionTypes.BlendAlpha) {
            writeConstant(buf, 1.0f);
            return;
        }
        if (df instanceof DensityFunctionTypes.BlendOffset) {
            writeConstant(buf, 0.0f);
            return;
        }
        if (df instanceof DensityFunctionTypes.BlendDensity bd) {
            writeDF(buf, bd.input());
            return;
        }

        // ── Beardifier ──────────────────────────────────────────
        if (df instanceof DensityFunctionTypes.Beardifier) {
            buf.put(TAG_BEARDIFIER);
            buf.putInt(0);
            return;
        }

        // ── Constant ────────────────────────────────────────────
        if (df instanceof DensityFunctionTypes.Constant c) {
            writeConstant(buf, (float) c.value());
            return;
        }

        // ── BinaryOperationLike (BinaryOperation + LinearOperation) ──
        if (df instanceof DensityFunctionTypes.BinaryOperationLike b) {
            DensityFunction a1 = b.argument1();
            DensityFunction a2 = b.argument2();
            switch (b.type()) {
                case ADD -> buf.put(TAG_ADD);
                case MUL -> buf.put(TAG_MUL);
                case MIN -> buf.put(TAG_MIN);
                case MAX -> buf.put(TAG_MAX);
            }
            writeDF(buf, a1);
            writeDF(buf, a2);
            return;
        }

        // ── UnaryOperation ──────────────────────────────────────
        if (df instanceof DensityFunctionTypes.UnaryOperation u) {
            DensityFunction in = u.input();
            switch (u.type()) {
                case ABS -> { buf.put(TAG_ABS); writeDF(buf, in); }
                case SQUARE -> { buf.put(TAG_SQUARE); writeDF(buf, in); }
                case CUBE -> { buf.put(TAG_CUBE); writeDF(buf, in); }
                case HALF_NEGATIVE -> {
                    buf.put(TAG_MUL);
                    writeConstant(buf, -0.5f);
                    writeDF(buf, in);
                }
                case QUARTER_NEGATIVE -> {
                    buf.put(TAG_MUL);
                    writeConstant(buf, -0.25f);
                    writeDF(buf, in);
                }
                case SQUEEZE -> writeConstant(buf, 0.0f);
            }
            return;
        }

        // ── RangeChoice ─────────────────────────────────────────
        if (df instanceof DensityFunctionTypes.RangeChoice r) {
            buf.put(TAG_RANGE_CHOICE);
            writeDF(buf, r.input());
            writeConstant(buf, (float) r.minInclusive());
            writeConstant(buf, (float) r.maxExclusive());
            writeDF(buf, r.whenInRange());
            writeDF(buf, r.whenOutOfRange());
            writeDF(buf, r.whenOutOfRange());
            return;
        }

        // ── Clamp → min(max(x, min), max) ───────────────────────
        if (df instanceof DensityFunctionTypes.Clamp c) {
            buf.put(TAG_MIN);
            writeConstant(buf, (float) c.maxValue());
            buf.put(TAG_MAX);
            writeConstant(buf, (float) c.minValue());
            writeDF(buf, c.input());
            return;
        }

        // ── Wrapping (caches) ───────────────────────────────────
        if (df instanceof DensityFunctionTypes.Wrapping w) {
            switch (w.type()) {
                case FLAT_CACHE -> buf.put(TAG_CACHE_FLAT);
                case CACHE2D -> buf.put(TAG_CACHE_2D);
                case CACHE_ONCE, CACHE_ALL_IN_CELL -> buf.put(TAG_CACHE_INTERP);
                case INTERPOLATED -> buf.put(TAG_INTERP_NOISE);
            }
            writeDF(buf, w.wrapped());
            return;
        }

        // ── Collect noise samplers before the Phase-2 placeholder ──
        if (df instanceof DensityFunctionTypes.Noise tn) {
            registerNoiseSampler(tn.noise());
        }

        // ── Phase-2 placeholders (noise chain) ──────────────────
        if (df instanceof DensityFunctionTypes.YClampedGradient
                || df instanceof DensityFunctionTypes.Noise
                || df instanceof DensityFunctionTypes.ShiftedNoise
                || df instanceof DensityFunctionTypes.Shift
                || df instanceof DensityFunctionTypes.ShiftA
                || df instanceof DensityFunctionTypes.ShiftB
                || df instanceof DensityFunctionTypes.WeirdScaledSampler
                || df instanceof DensityFunctionTypes.Spline
                || df instanceof DensityFunctionTypes.EndIslands) {
            writeConstant(buf, 0.0f);
            return;
        }

        // ── Unknown node — diagnostic dump ──────────────────────
        System.err.println("[c2me-wgpu] UNSUPPORTED NODE: " + df.getClass().getName());
        for (Field f : df.getClass().getDeclaredFields()) {
            try {
                f.setAccessible(true);
                Object v = f.get(df);
                String repr = (v instanceof DensityFunction d)
                        ? ("DensityFunction(" + d.getClass().getSimpleName() + ")")
                        : String.valueOf(v);
                System.err.println("[c2me-wgpu]   field " + f.getName()
                        + " : " + f.getType().getSimpleName() + " = " + repr);
            } catch (Throwable ignored) {
            }
        }
        writeConstant(buf, 0.0f);
    }

    /**
     * Append the collected Perlin permutation tables.
     * Layout (little-endian):
     *   u32 magic = 0x4D525750 ("PWPM")
     *   u32 count
     *   for each entry:
     *     u32 perm[256]     (permutation bytes widened to u32)
     *     f64 origin_x, origin_y, origin_z
     */
    private static void writePermResource(ByteBuffer buf) {
        buf.putInt(PERM_MAGIC);
        buf.putInt(PERM_SAMPLERS.size());
        for (Object sampler : PERM_SAMPLERS) {
            try {
                byte[] perm = findPermutation(sampler);
                for (int i = 0; i < 256; i++) {
                    buf.putInt(perm[i] & 0xFF);
                }
                PerlinNoiseSampler p = (PerlinNoiseSampler) sampler;
                buf.putDouble(p.originX);
                buf.putDouble(p.originY);
                buf.putDouble(p.originZ);
            } catch (Throwable t) {
                System.err.println("[c2me-wgpu] perm resource: skip sampler: " + t);
                for (int i = 0; i < 256; i++) buf.putInt(0);
                buf.putDouble(0); buf.putDouble(0); buf.putDouble(0);
            }
        }
    }

    /** PerlinNoiseSampler has exactly one private byte[256] field (permutation). */
    private static byte[] findPermutation(Object sampler) throws Exception {
        for (Field f : sampler.getClass().getDeclaredFields()) {
            if (f.getType() == byte[].class) {
                f.setAccessible(true);
                return (byte[]) f.get(sampler);
            }
        }
        throw new NoSuchFieldException("permutation (byte[]) not found");
    }

    private static void registerOctaveSampler(Object octaveSampler) {
        if (octaveSampler == null) return;
        try {
            for (Field f : octaveSampler.getClass().getDeclaredFields()) {
                if (f.getType() == PerlinNoiseSampler[].class) {
                    f.setAccessible(true);
                    PerlinNoiseSampler[] arr = (PerlinNoiseSampler[]) f.get(octaveSampler);
                    for (PerlinNoiseSampler p : arr) {
                        if (p != null && SEEN_PERM.put(p, null) == null) {
                            PERM_SAMPLERS.add(p);
                        }
                    }
                    return;
                }
            }
        } catch (Throwable t) {
            System.err.println("[c2me-wgpu] registerOctaveSampler failed: " + t);
        }
    }

    private static void registerNoiseSampler(DensityFunction.Noise n) {
        // 1) Prefer the already-instantiated sampler (populated when the router
        //    comes from NoiseConfig; null for the raw settings template).
        Object dperlin = n.noise();
        // 2) Fall back to instantiating from NoiseConfig by registry key.
        if (!(dperlin instanceof DoublePerlinNoiseSampler) && CURRENT_NOISE_CONFIG != null) {
            try {
                var keyOpt = n.noiseData().getKey();
                if (keyOpt.isPresent()) {
                    dperlin = CURRENT_NOISE_CONFIG.getOrCreateSampler(keyOpt.get());
                }
            } catch (Throwable t) {
                System.err.println("[c2me-wgpu] getOrCreateSampler failed: " + t);
            }
        }
        if (!(dperlin instanceof DoublePerlinNoiseSampler)) {
            return;
        }
        if (dperlin == lastDperlin) return;
        lastDperlin = dperlin;
        try {
            for (Field f : dperlin.getClass().getDeclaredFields()) {
                if (OctavePerlinNoiseSampler.class.isAssignableFrom(f.getType())) {
                    f.setAccessible(true);
                    registerOctaveSampler(f.get(dperlin));
                }
            }
        } catch (Throwable t) {
            System.err.println("[c2me-wgpu] registerNoiseSampler failed: " + t);
        }
    }

    public static class ShapeConfig {
        public final int minY, height, cellWidth, cellHeight;
        public ShapeConfig(int minY, int height, int cellWidth, int cellHeight) {
            this.minY = minY; this.height = height; this.cellWidth = cellWidth; this.cellHeight = cellHeight;
        }
    }
}
