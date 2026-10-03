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
import net.minecraft.util.math.Spline;
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
    private static final int SIMPLEX_MAGIC = 0x4D525358; // "SXRM"
    private static final java.util.List<Object> SIMPLEX_SAMPLERS = new java.util.ArrayList<>();
    private static final IdentityHashMap<Object, Void> SEEN_PERM = new IdentityHashMap<>();
    private static final List<Object> PERM_SAMPLERS = new ArrayList<>();
    private static Object lastDperlin = null;
    /** IdentityHashMap: DoublePerlinNoiseSampler -> structured stack info. */
    private static final IdentityHashMap<Object, DpsInfo> DPS_INFO = new IdentityHashMap<>();

    private static final class StackInfo {
        final int permOffset;
        final float initialFreq;
        final float persistence;
        final float[] amplitudes;
        StackInfo(int permOffset, float initialFreq, float persistence, float[] amplitudes) {
            this.permOffset = permOffset;
            this.initialFreq = initialFreq;
            this.persistence = persistence;
            this.amplitudes = amplitudes;
        }
    }
    private static final class DpsInfo {
        final StackInfo first;
        final StackInfo second;
        final float amplitude;
        DpsInfo(StackInfo f, StackInfo sec, float amp) {
            first = f; second = sec; amplitude = amp;
        }
    }
    private static NoiseConfig CURRENT_NOISE_CONFIG = null;
    /** Every InterpolatedNoiseSampler encountered while serializing. */
    public static final java.util.List<Object> SEEN_INS = new java.util.ArrayList<>();
    private static final java.util.Map<String, Boolean> PLACEHOLDER_SEEN =
            new java.util.concurrent.ConcurrentHashMap<>();
    /** Dedup for UNSUPPORTED NODE field-dumps: one full dump per node class. */
    private static final java.util.Map<String, Boolean> UNSUPPORTED_SEEN =
            new java.util.concurrent.ConcurrentHashMap<>();

    /** Density subtrees we could not translate, evaluated on the CPU and fed
     *  to the GPU as external inputs (binding 6). Cleared per blobify(). */
    public static final java.util.List<DensityFunction> EXTERNAL_NODES =
            new java.util.ArrayList<>();

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
    private static final byte TAG_RANGE_CHOICE = (byte) 0xB1;
    private static final byte TAG_INTERVAL_SEL = (byte) 0xB2;
    private static final byte TAG_Y_CLAMPED_GRAD = (byte) 0xB3;
    private static final byte TAG_SPLINE = (byte) 0xB0;
    private static final byte TAG_SHIFT = (byte) 0xB4;
    private static final byte TAG_SHIFTED_NOISE = (byte) 0xB5;
    private static final byte TAG_WEIRD_SCALED = (byte) 0x86;
    private static final byte TAG_INTERP_NOISE_SAMPLER = (byte) 0x87;
    private static final byte TAG_CACHE_FLAT = (byte) 0xC0;
    private static final byte TAG_CACHE_2D = (byte) 0xC1;
    private static final byte TAG_CACHE_INTERP = (byte) 0xC2;
    private static final byte TAG_FIND_TOP_SURF = (byte) 0xD0;
    private static final byte TAG_AQUIFER = (byte) 0xE0;
    private static final byte TAG_BIOME_TREE = (byte) 0xF0;
    private static final byte TAG_EXTERNAL_INPUT = (byte) 0xFD;
    private static final byte TAG_ROOT = (byte) 0xFE;

    public static ByteBuffer blobify(NoiseRouter router, ShapeConfig shape, NoiseConfig noiseConfig) {
        CURRENT_NOISE_CONFIG = noiseConfig;
        // 32 MiB: modded datapacks (e.g. Tectonic) build far larger density
        // trees than vanilla. Overflow here previously surfaced as a
        // BufferOverflowException deep in writeStack and a CPU fallback.
        ByteBuffer buf = ByteBuffer.allocate(32 * 1024 * 1024).order(ByteOrder.LITTLE_ENDIAN);
        SEEN_PERM.clear();
        PERM_SAMPLERS.clear();
        DPS_INFO.clear();
        SEEN_INS.clear();
        SIMPLEX_SAMPLERS.clear();
        EXTERNAL_NODES.clear();
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
        writeSimplexResource(buf);

        buf.flip();
        return buf;
    }

    private static void writeConstant(ByteBuffer buf, float value) {
        buf.put(TAG_CONSTANT);
        buf.putFloat(value);
    }

    private static void writeDF(ByteBuffer buf, DensityFunction df) {
        // Unwrap MC's RegistryEntryHolder layer (e.g. class_6916$class_7051):
        // router channels come wrapped, the real tree is inside .function().
        if (df instanceof DensityFunctionTypes.RegistryEntryHolder h) {
            writeDF(buf, h.function().value());
            return;
        }

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
            // GPU density omits Beardifier; MC's real beard is applied
            // per-block in MixinChunkGenerator via StructureWeightSampler.
            // Emit constant 0 so the node contributes nothing on the GPU path.
            writeConstant(buf, 0.0f);
            return;
        }

        // ── InterpolatedNoiseSampler (standalone class) ─────────
        if (df instanceof net.minecraft.util.math.noise.InterpolatedNoiseSampler ins) {
            SEEN_INS.add(ins);
            writeInterpolatedNoiseSampler(buf, ins);
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
                    // MC: x > 0 ? x : x*0.5   ==   max(x, 0.5*x)
                    buf.put(TAG_MAX);
                    writeDF(buf, in);
                    buf.put(TAG_MUL);
                    writeConstant(buf, 0.5f);
                    writeDF(buf, in);
                }
                case QUARTER_NEGATIVE -> {
                    // MC: x > 0 ? x : x*0.25   ==   max(x, 0.25*x)
                    buf.put(TAG_MAX);
                    writeDF(buf, in);
                    buf.put(TAG_MUL);
                    writeConstant(buf, 0.25f);
                    writeDF(buf, in);
                }
                case SQUEEZE -> { buf.put((byte) 0x25); writeDF(buf, in); }
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
                // CACHE_ONCE / CACHE_ALL_IN_CELL are pure memoisation in MC:
                // they return the same value as their inner at the same
                // position. Treating them as INTERPOLATED (corner-blend)
                // silently changes the value -> wrong density. Emit as
                // passthrough (recurse into the wrapped node).
                case CACHE_ONCE, CACHE_ALL_IN_CELL -> {
                    writeDF(buf, w.wrapped());
                    return;
                }
                case INTERPOLATED -> buf.put(TAG_INTERP_NOISE);
            }
            writeDF(buf, w.wrapped());
            return;
        }

        // ── Shift / ShiftA / ShiftB (Offset interface) ──────────
        if (df instanceof DensityFunctionTypes.Offset off) {
            int kind = (df instanceof DensityFunctionTypes.ShiftA) ? 1
                     : (df instanceof DensityFunctionTypes.ShiftB) ? 2 : 0;
            buf.put(TAG_SHIFT);
            buf.put((byte) kind);
            buf.put(TAG_NOISE_SAMPLER);
            writeNoiseSamplerPayload(buf, off.offsetNoise(), 1.0f, 1.0f);
            return;
        }

        // ── ShiftedNoise ────────────────────────────────────────
        if (df instanceof DensityFunctionTypes.ShiftedNoise sn) {
            buf.put(TAG_SHIFTED_NOISE);
            buf.putFloat((float) sn.xzScale());
            buf.putFloat((float) sn.yScale());
            writeDF(buf, sn.shiftX());
            writeDF(buf, sn.shiftY());
            writeDF(buf, sn.shiftZ());
            buf.put(TAG_NOISE_SAMPLER);
            // sn.noise() is a raw DensityFunction.Noise: its sample(d,e,f) does
            // NOT re-apply xzScale (that already went into the coord shift).
            writeNoiseSamplerPayload(buf, sn.noise(), 1.0f, 1.0f);
            return;
        }

        // ── Noise: emit TAG_NOISE_SAMPLER with structured stacks ─
        if (df instanceof DensityFunctionTypes.Noise tn) {
            buf.put(TAG_NOISE_SAMPLER);
            writeNoiseSamplerPayload(buf, tn.noise(), (float) tn.xzScale(), (float) tn.yScale());
            return;
        }

        // ── Spline: recursive Hermite tree ──────────────────────
        if (df instanceof DensityFunctionTypes.Spline sp) {
            writeSpline(buf, sp.spline());
            return;
        }

        // ── YClampedGradient ────────────────────────────────────
        if (df instanceof DensityFunctionTypes.YClampedGradient g) {
            buf.put(TAG_Y_CLAMPED_GRAD);
            buf.putFloat((float) g.fromY());
            buf.putFloat((float) g.toY());
            buf.putFloat((float) g.fromValue());
            buf.putFloat((float) g.toValue());
            return;
        }

        // ── YClampedGradient ────────────────────────────────────
        if (df instanceof DensityFunctionTypes.YClampedGradient g) {
            buf.put(TAG_Y_CLAMPED_GRAD);
            buf.putFloat((float) g.fromY());
            buf.putFloat((float) g.toY());
            buf.putFloat((float) g.fromValue());
            buf.putFloat((float) g.toValue());
            return;
        }

        // ── WeirdScaledSampler ──────────────────────────────────
        if (df instanceof DensityFunctionTypes.WeirdScaledSampler ws) {
            buf.put(TAG_WEIRD_SCALED);
            buf.put(ws.rarityValueMapper() ==
                    DensityFunctionTypes.WeirdScaledSampler.RarityValueMapper.TYPE1
                    ? (byte) 1 : (byte) 2);
            writeDF(buf, ws.input());
            buf.put(TAG_NOISE_SAMPLER);
            writeNoiseSamplerPayload(buf, ws.noise(), 1.0f, 1.0f);
            return;
        }

        // ── EndIslands ──────────────────────────────────────────
        if (df instanceof DensityFunctionTypes.EndIslands) {
            try {
                java.lang.reflect.Field sf = null;
                for (java.lang.reflect.Field f : df.getClass().getDeclaredFields()) {
                    if (f.getType() == net.minecraft.util.math.noise.SimplexNoiseSampler.class) { sf = f; break; }
                }
                if (sf == null) throw new NoSuchFieldException("EndIslands.sampler");
                sf.setAccessible(true);
                int idx = SIMPLEX_SAMPLERS.size();
                SIMPLEX_SAMPLERS.add(sf.get(df));
                buf.put(TAG_END_ISLANDS);
                buf.putInt(idx);
            } catch (Throwable t) {
                System.err.println("[c2me-wgpu] EndIslands serialize failed: " + t);
                writeConstant(buf, 0.0f);
            }
            return;
        }

        // ── Phase-2 placeholders (noise chain) ──────────────────
        if (df instanceof DensityFunctionTypes.YClampedGradient
                || df instanceof DensityFunctionTypes.Noise
                || df instanceof DensityFunctionTypes.ShiftedNoise
                || df instanceof DensityFunctionTypes.Shift
                || df instanceof DensityFunctionTypes.ShiftA
                || df instanceof DensityFunctionTypes.ShiftB
                || df instanceof DensityFunctionTypes.WeirdScaledSampler
                || df instanceof DensityFunctionTypes.Spline) {
            PLACEHOLDER_SEEN.computeIfAbsent(df.getClass().getName(), k -> {
                System.err.println("[c2me-wgpu][placeholder] " + k);
                return Boolean.TRUE;
            });
            writeConstant(buf, 0.0f);
            return;
        }

        // ── Tectonic `Invert`: transform(d) = 1.0 / d ───────────
        // Tectonic's density functions are not a compile-time dependency, so
        // match by class name. Invert is NOT downgraded by mapAll(), so it
        // survives into the raw tree we serialize. min/max are pure range
        // metadata and do not affect the value.
        if (df.getClass().getName()
                .equals("dev.worldgen.tectonic.worldgen.densityfunction.Invert")) {
            DensityFunction in;
            try {
                Field inf = df.getClass().getDeclaredField("input");
                inf.setAccessible(true);
                in = (DensityFunction) inf.get(df);
            } catch (Throwable t) {
                throw new UnsupportedDensityFunctionException(df.getClass().getName());
            }
            if (in == null) {
                throw new UnsupportedDensityFunctionException(df.getClass().getName());
            }
            // 1.0 / input
            buf.put(TAG_DIV);
            buf.put(TAG_CONSTANT);
            buf.putFloat(1.0f);
            writeDF(buf, in);
            return;
        }

        // ── Unknown node — register as a CPU-evaluated external input ──
        // We can't translate this node, so we emit ExternalInput(index) and let
        // the CPU evaluate the whole subtree at the corner grid. The GPU reads
        // the values from binding 6. This keeps arbitrary modded density
        // functions working without per-node adapters.
        final String cls = df.getClass().getName();
        if (UNSUPPORTED_SEEN.putIfAbsent(cls, Boolean.TRUE) == null) {
            System.err.println("[c2me-wgpu] UNSUPPORTED NODE (externalized to CPU): " + cls);
        }
        int extIndex = EXTERNAL_NODES.size();
        EXTERNAL_NODES.add(df);
        buf.put(TAG_EXTERNAL_INPUT);
        buf.putInt(extIndex);
    }

    /** Thrown when the density tree contains a node this mod cannot serialize.
     *  Aborting is intentional: emitting a placeholder constant would silently
     *  generate wrong terrain. The caller falls back to vanilla CPU generation. */
    public static final class UnsupportedDensityFunctionException extends RuntimeException {
        public UnsupportedDensityFunctionException(String nodeClass) {
            super("unsupported density function node: " + nodeClass);
        }
    }

    /** Recursively serialize a MC Spline tree. FixedFloatFunction -> Constant. */
    @SuppressWarnings({"unchecked", "rawtypes"})
    private static void writeSpline(ByteBuffer buf, Object node) {
        if (node instanceof Spline.FixedFloatFunction<?, ?> f) {
            writeConstant(buf, f.value());
            return;
        }
        if (node instanceof Spline.Implementation<?, ?> impl) {
            buf.put(TAG_SPLINE);
            buf.putFloat(impl.min());
            buf.putFloat(impl.max());
            float[] locs = impl.locations();
            float[] ders = impl.derivatives();
            java.util.List<?> vals = impl.values();
            buf.putInt(locs.length);
            for (float l : locs) buf.putFloat(l);
            for (float d : ders) buf.putFloat(d);
            // location function: DensityFunctionWrapper -> inner DensityFunction
            Object locFn = impl.locationFunction();
            DensityFunction inner = unwrapLocFn(locFn);
            if (inner != null) {
                writeDF(buf, inner);
            } else {
                writeConstant(buf, 0.0f);
            }
            // values: recursive splines
            for (Object v : vals) {
                writeSpline(buf, v);
            }
            return;
        }
        // Unknown spline node -> constant 0
        System.err.println("[c2me-wgpu] unknown spline node: " + node.getClass().getName());
        writeConstant(buf, 0.0f);
    }

    /** Extract the wrapped DensityFunction from a Spline$DensityFunctionWrapper.
     *  Uses the record accessor (Loom remaps the method reference, unlike
     *  string-keyed reflection which fails on intermediary names). */
    private static DensityFunction unwrapLocFn(Object locFn) {
        if (locFn instanceof DensityFunctionTypes.Spline.DensityFunctionWrapper w) {
            return w.function().value();
        }
        System.err.println("[c2me-wgpu] unwrapLocFn: unexpected class "
                + locFn.getClass().getName());
        return null;
    }

    /** Write [type:u8][flags:u8][amplitude:f32][xz:f32][yz:f32][stacks] for a DensityFunction.Noise. */
    private static void writeNoiseSamplerPayload(ByteBuffer buf, DensityFunction.Noise n,
                                                 float xzScale, float yScale) {
        DpsInfo info = registerNoiseSampler(n);
        buf.put((byte) 0);                        // type: Perlin
        boolean hasSecond = (info != null && info.second != null);
        buf.put((byte) (hasSecond ? 1 : 0));      // flags
        buf.putFloat(info != null ? info.amplitude : 1.0f);
        buf.putFloat(xzScale);
        buf.putFloat(yScale);
        if (info != null && info.first != null) {
            writeStack(buf, info.first);
        } else {
            writeFallbackStack(buf);
        }
        if (hasSecond) {
            writeStack(buf, info.second);
        }
    }

    /** Serialize a standalone MC InterpolatedNoiseSampler (class_5822). */
    private static void writeInterpolatedNoiseSampler(
            ByteBuffer buf, net.minecraft.util.math.noise.InterpolatedNoiseSampler ins) {
        com.rubidiumlighting.c2me.opts.accel.wgpu.mixin.InterpolatedNoiseSamplerAccessor acc =
                (com.rubidiumlighting.c2me.opts.accel.wgpu.mixin.InterpolatedNoiseSamplerAccessor) ins;
        buf.put(TAG_INTERP_NOISE_SAMPLER);
        buf.putFloat((float) acc.getXzScale());
        buf.putFloat((float) acc.getYScale());
        buf.putFloat((float) acc.getXzFactor());
        buf.putFloat((float) acc.getYFactor());
        buf.putFloat((float) acc.getSmearScaleMultiplier());
        writeRawOctaveStack(buf, acc.getLowerInterpolatedNoise());
        writeRawOctaveStack(buf, acc.getUpperInterpolatedNoise());
        writeRawOctaveStack(buf, acc.getInterpolationNoise());
    }

    /** Register an OctavePerlinNoiseSampler's octaves and write [base:i32][count:i32]. */
    private static void writeRawOctaveStack(ByteBuffer buf, Object octave) {
        try {
            java.lang.reflect.Field af = null;
            for (java.lang.reflect.Field f : octave.getClass().getDeclaredFields()) {
                if (f.getType() == PerlinNoiseSampler[].class) { af = f; break; }
            }
            if (af == null) { buf.putInt(0); buf.putInt(0); return; }
            af.setAccessible(true);
            PerlinNoiseSampler[] arr = (PerlinNoiseSampler[]) af.get(octave);
            int baseIdx = PERM_SAMPLERS.size();
            // MC's OctavePerlinNoiseSampler.getOctave(i) returns
            // octaveSamplers[length-1-i] — REVERSED. InterpolatedNoiseSampler
            // (old_blended_noise) iterates i=0..n via getOctave, so we must
            // store the tables in reversed array order; the Rust/WGSL side
            // then iterates forward and reads the same perlin at the same
            // frequency (this is why regular noise, which uses the forward
            // sample() loop, was unaffected).
            for (int ai = arr.length - 1; ai >= 0; ai--) {
                PerlinNoiseSampler p = arr[ai];
                if (p == null) PERM_SAMPLERS.add(null);
                else { SEEN_PERM.put(p, null); PERM_SAMPLERS.add(p); }
            }
            buf.putInt(baseIdx * 256);
            buf.putInt(arr.length);
        } catch (Throwable t) {
            System.err.println("[c2me-wgpu] writeRawOctaveStack failed: " + t);
            buf.putInt(0); buf.putInt(0);
        }
    }


    private static void writeStack(ByteBuffer buf, StackInfo st) {
        buf.putInt(st.permOffset);
        buf.putFloat(st.initialFreq);
        buf.putFloat(st.persistence);
        buf.putInt(st.amplitudes.length);
        for (float a : st.amplitudes) buf.putFloat(a);
    }
    private static void writeFallbackStack(ByteBuffer buf) {
        buf.putInt(0);
        buf.putFloat(2.0f);
        buf.putInt(1);
        buf.putFloat(1.0f);
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
            if (sampler == null) {
                for (int i = 0; i < 256; i++) buf.putInt(0);
                buf.putDouble(0); buf.putDouble(0); buf.putDouble(0);
                continue;
            }
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

    /** Append captured SimplexNoiseSampler tables (int[512] + origin). */
    private static void writeSimplexResource(ByteBuffer buf) {
        buf.putInt(SIMPLEX_MAGIC);
        buf.putInt(SIMPLEX_SAMPLERS.size());
        for (Object sampler : SIMPLEX_SAMPLERS) {
            try {
                // Use the Loom-remapped @Accessor for the private
                // `permutation` table and direct field access for the public
                // originX/Y/Z. String reflection (getField("originX")) breaks
                // on intermediary names at runtime and used to silently write
                // 512 zeros, corrupting the EndIslands simplex table.
                net.minecraft.util.math.noise.SimplexNoiseSampler sns =
                        (net.minecraft.util.math.noise.SimplexNoiseSampler) sampler;
                int[] perm =
                        ((com.rubidiumlighting.c2me.opts.accel.wgpu.mixin.SimplexNoiseSamplerAccessor) sns)
                                .getPermutation();
                for (int i = 0; i < 512; i++) buf.putInt(perm[i]);
                buf.putDouble(sns.originX);
                buf.putDouble(sns.originY);
                buf.putDouble(sns.originZ);
            } catch (Throwable t) {
                System.err.println("[c2me-wgpu] simplex resource: skip: " + t);
                for (int i = 0; i < 512; i++) buf.putInt(0);
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

    private static float[] extractAmplitudes(Object octaveSampler, int expectedLen) {
        for (Field f : octaveSampler.getClass().getDeclaredFields()) {
            if (f.getType() == double[].class) {
                try {
                    f.setAccessible(true);
                    double[] arr = (double[]) f.get(octaveSampler);
                    float[] out = new float[arr.length];
                    for (int i = 0; i < arr.length; i++) out[i] = (float) arr[i];
                    return out;
                } catch (Throwable ignored) {}
            }
        }
        for (Field f : octaveSampler.getClass().getDeclaredFields()) {
            if (List.class.isAssignableFrom(f.getType())) {
                try {
                    f.setAccessible(true);
                    List<?> list = (List<?>) f.get(octaveSampler);
                    if (!list.isEmpty() && list.get(0) instanceof Double) {
                        float[] out = new float[list.size()];
                        for (int i = 0; i < list.size(); i++) {
                            out[i] = ((Double) list.get(i)).floatValue();
                        }
                        return out;
                    }
                } catch (Throwable ignored) {}
            }
        }
        float[] out = new float[Math.max(1, expectedLen)];
        java.util.Arrays.fill(out, 1.0f);
        return out;
    }

    private static int extractFirstOctave(Object octaveSampler) {
        try {
            Field f = octaveSampler.getClass().getDeclaredField("firstOctave");
            f.setAccessible(true);
            return f.getInt(octaveSampler);
        } catch (Throwable ignored) {}
        int best = 0; boolean found = false;
        for (Field f : octaveSampler.getClass().getDeclaredFields()) {
            if (f.getType() == int.class) {
                try {
                    f.setAccessible(true);
                    int v = f.getInt(octaveSampler);
                    if (v >= -20 && v <= 10 && (!found || v < best)) {
                        best = v; found = true;
                    }
                } catch (Throwable ignored) {}
            }
        }
        return best;
    }

    private static StackInfo registerOneStack(Object octaveSampler) {
        if (octaveSampler == null) return null;
        try {
            PerlinNoiseSampler[] arr = null;
            for (Field f : octaveSampler.getClass().getDeclaredFields()) {
                if (f.getType() == PerlinNoiseSampler[].class) {
                    f.setAccessible(true);
                    arr = (PerlinNoiseSampler[]) f.get(octaveSampler);
                    break;
                }
            }
            if (arr == null) {
                System.err.println("[c2me-wgpu] registerOneStack: no PerlinNoiseSampler[] field");
                return null;
            }
            int baseIdx = PERM_SAMPLERS.size();
            for (PerlinNoiseSampler p : arr) {
                if (p == null) {
                    PERM_SAMPLERS.add(null);
                } else {
                    SEEN_PERM.put(p, null);
                    PERM_SAMPLERS.add(p);
                }
            }
            float[] amps = extractAmplitudes(octaveSampler, arr.length);
            int n = amps.length;
            int firstOctave = extractFirstOctave(octaveSampler);
            float initialFreq = (float) Math.pow(2.0, firstOctave);
            float persistence = n <= 1
                    ? 1.0f
                    : (float) (Math.pow(2.0, n - 1) / (Math.pow(2.0, n) - 1.0));
            return new StackInfo(baseIdx * 256, initialFreq, persistence, amps);
        } catch (Throwable t) {
            System.err.println("[c2me-wgpu] registerOneStack failed: " + t);
            return null;
        }
    }

    private static float computeAmplitude(StackInfo st) {
        if (st == null || st.amplitudes.length == 0) return 1.0f;
        int minIdx = Integer.MAX_VALUE, maxIdx = Integer.MIN_VALUE;
        for (int i = 0; i < st.amplitudes.length; i++) {
            if (st.amplitudes[i] != 0.0f) {
                if (i < minIdx) minIdx = i;
                if (i > maxIdx) maxIdx = i;
            }
        }
        if (maxIdx < minIdx) return 1.0f;
        double n = maxIdx - minIdx;
        double createAmplitude = 0.1 * (1.0 + 1.0 / (n + 1.0));
        return (float) ((1.0 / 6.0) / createAmplitude);
    }

    private static DpsInfo registerNoiseSampler(DensityFunction.Noise n) {
        Object dperlin = n.noise();
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
            return null;
        }
        DpsInfo cached = DPS_INFO.get(dperlin);
        if (cached != null) return cached;
        try {
            Object first = null, second = null;
            for (Field f : dperlin.getClass().getDeclaredFields()) {
                if (OctavePerlinNoiseSampler.class.isAssignableFrom(f.getType())) {
                    f.setAccessible(true);
                    Object v = f.get(dperlin);
                    String nm = f.getName();
                    if (nm.contains("first")) first = v;
                    else if (nm.contains("second")) second = v;
                    else if (first == null) first = v;
                    else if (second == null) second = v;
                }
            }
            StackInfo fi = registerOneStack(first);
            StackInfo si = registerOneStack(second);
            float amp = computeAmplitude(fi);
            DpsInfo info = new DpsInfo(fi, si, amp);
            DPS_INFO.put(dperlin, info);
            return info;
        } catch (Throwable t) {
            System.err.println("[c2me-wgpu] registerNoiseSampler failed: " + t);
            return null;
        }
    }

    public static class ShapeConfig {
        public final int minY, height, cellWidth, cellHeight;
        public ShapeConfig(int minY, int height, int cellWidth, int cellHeight) {
            this.minY = minY; this.height = height; this.cellWidth = cellWidth; this.cellHeight = cellHeight;
        }
    }
}
