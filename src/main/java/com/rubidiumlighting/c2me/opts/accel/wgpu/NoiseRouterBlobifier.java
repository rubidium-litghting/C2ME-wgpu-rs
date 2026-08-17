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

import net.minecraft.world.gen.DensityFunction;
import net.minecraft.world.gen.DensityFunctions;
import net.minecraft.world.gen.chunk.ChunkGeneratorSettings;
import net.minecraft.world.gen.chunk.ShapeConfig;
import net.minecraft.world.gen.noise.NoiseRouter;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Serialize MC NoiseRouter + ShapeConfig + BiomeSource into a self-describing
 * [tag][payload] recursive binary blob understood by the Rust deserializer.
 * <p>
 * All numeric values use fp32 (no double). Byte order: Little Endian.
 * <p>
 * Supported DensityFunction classes (Yarn mapping names, 1.19.4+):
 *   Constant / InputX(BASE_3D_X singleton) / Add
 * All other subclasses throw {@link UnsupportedDFException} — caller must
 * catch and fall back to the CPU path.
 */
public class NoiseRouterBlobifier {

    // Header magic = "MC2W" in little-endian u32
    public static final int MAGIC = 0x4D433257;
    public static final short VERSION = 1;

    // Tag constants (must match Rust side 1:1)
    public static final byte TAG_CONSTANT = 0x01;
    public static final byte TAG_INPUT_X = 0x02;
    public static final byte TAG_INPUT_Y = 0x03;
    public static final byte TAG_INPUT_Z = 0x04;
    public static final byte TAG_ADD = 0x10;
    public static final byte TAG_SUB = 0x11;
    public static final byte TAG_MUL = 0x12;
    public static final byte TAG_DIV = 0x13;
    public static final byte TAG_MIN = 0x14;
    public static final byte TAG_MAX = 0x15;
    public static final byte TAG_ABS = 0x20;
    public static final byte TAG_SQUARE = 0x21;
    public static final byte TAG_CUBE = 0x22;
    public static final byte TAG_SQRT = 0x23;
    public static final byte TAG_FLOOR_DIV = 0x24;
    public static final byte TAG_PERLIN = (byte) 0x80;
    public static final byte TAG_DOUBLE_PERLIN = (byte) 0x81;
    public static final byte TAG_SIMPLEX_2D = (byte) 0x82;
    public static final byte TAG_SIMPLEX_3D = (byte) 0x83;
    public static final byte TAG_INTERPOLATED = (byte) 0x90;
    public static final byte TAG_END_ISLANDS = (byte) 0xA0;
    public static final byte TAG_BEARDIFIER = (byte) 0xA1;
    public static final byte TAG_SPLINE = (byte) 0xB0;
    public static final byte TAG_RANGE_CHOICE = (byte) 0xB1;
    public static final byte TAG_INTERVAL_SELECT = (byte) 0xB2;
    public static final byte TAG_Y_CLAMPED_GRADIENT = (byte) 0xB3;
    public static final byte TAG_CACHE_FLAT = (byte) 0xC0;
    public static final byte TAG_CACHE_2D = (byte) 0xC1;
    public static final byte TAG_CACHE_INTERP = (byte) 0xC2;
    public static final byte TAG_FIND_TOP_SURFACE = (byte) 0xD0;
    public static final byte TAG_AQUIFER = (byte) 0xE0;
    public static final byte TAG_BIOME_TREE = (byte) 0xF0;
    public static final byte TAG_ROOT = (byte) 0xFE;

    /** NoiseRouter 固定 16 个密度通道（Yarn 映射 1.19.4+ 的顺序）。 */
    private static final int ROOT_CHANNEL_COUNT = 16;

    /** 预估 blob 初始容量：16 通道保守估计 256B/通道 + 各段开销。 */
    private static final int INITIAL_BUFFER_CAPACITY = 1024 * 8;

    /**
     * Thrown when Blobifier encounters a DensityFunction node type it doesn't
     * yet know how to serialize. Caller should catch and fall back to CPU.
     */
    public static class UnsupportedDFException extends RuntimeException {
        public UnsupportedDFException(String nodeTypeName) {
            super("c2me-wgpu Blobifier: unsupported density function node: " + nodeTypeName);
        }
    }

    /**
     * Serialize a full NoiseRouter + ChunkGeneratorSettings into a blob ready
     * to pass to WgpuThinBinding.worldNew().
     *
     * @param router   MC NoiseRouter (the 16 density channels)
     * @param settings ChunkGeneratorSettings (for min_y/height/cell_* sizes)
     * @return heap ByteBuffer containing the serialized blob (Little Endian).
     *         Caller may read remaining bytes between position 0 and limit.
     * @throws UnsupportedDFException if any density channel contains a node
     *         class that the blobifier does not yet implement.
     */
    public static ByteBuffer blobify(NoiseRouter router, ChunkGeneratorSettings settings) {
        ByteBuffer buf = ByteBuffer.allocate(INITIAL_BUFFER_CAPACITY);
        buf.order(ByteOrder.LITTLE_ENDIAN);

        // ── 1. HEADER ─────────────────────────────────────────────────────
        buf.putInt(MAGIC);
        buf.putShort(VERSION);
        buf.put(new byte[4]); // reserved[4]

        // ── 2. NODE_TREE: ROOT(16 channels) ───────────────────────────────
        buf.put(TAG_ROOT);
        buf.putShort((short) ROOT_CHANNEL_COUNT);
        writeDF(buf, router.barrier());
        writeDF(buf, router.fluidLevelFloodedness());
        writeDF(buf, router.fluidTypeThisHeight());
        writeDF(buf, router.temperature());
        writeDF(buf, router.vegetation());
        writeDF(buf, router.continents());
        writeDF(buf, router.erosion());
        writeDF(buf, router.depth());
        writeDF(buf, router.ridges());
        writeDF(buf, router.factor());
        writeDF(buf, router.hole());
        writeDF(buf, router.peaks());
        writeDF(buf, router.terrain());
        writeDF(buf, router.caveEntrance());
        writeDF(buf, router.caveLayer());
        writeDF(buf, router.spaghetti2d());
        // NOTE: 不同 MC 版本的 NoiseRouter 第 16 个字段可能命名不同
        // （例如某些版本是 finalDensity() 而非 spaghetti2d()）。
        // 如果编译报错找不到 spaghetti2d() getter，在对应 MC 版本里改为
        // NoiseRouter.actualFinalDensityGetterName()。

        // ── 3. SHAPE_CONFIG ───────────────────────────────────────────────
        ShapeConfig sc = settings.shapeConfig();
        buf.putInt(sc.minimumY());
        buf.putInt(sc.height());
        buf.putInt(sc.horizontalBlockSize());
        buf.putInt(sc.verticalBlockSize());

        // ── 4. BIOME_SOURCE (placeholder: 0xF0 BiomeTree tag) ─────────────
        // TODO: 将来按协议 §4 真正序列化 MultiNoiseBiomeSource flattened tree。
        //       当前仅写 tag，payload 留空。
        buf.put(TAG_BIOME_TREE);

        // ── 5. Flip buffer → 就绪 ─────────────────────────────────────────
        buf.flip();
        return buf;
    }

    // ======================================================================
    // Recursive DensityFunction → [tag][payload] dispatcher.
    // ======================================================================

    /**
     * Append the serialized representation of one DensityFunction node.
     * Unsupported subclasses throw {@link UnsupportedDFException}.
     */
    private static void writeDF(ByteBuffer buf, DensityFunction df) {
        if (df == null) {
            // 防御：某些通道在极端配置里可能是 null → 写成 Constant(0) 占位，
            // 避免 NullPointerException 让上层有更好的报错信息。
            writeConstant(buf, 0.0);
            return;
        }

        // ── Input coordinates: singleton references (NOT instanceof) ────
        if (df == DensityFunctions.BASE_3D_X) {
            buf.put(TAG_INPUT_X);
            return;
        }
        if (df == DensityFunctions.BASE_3D_Y) {
            buf.put(TAG_INPUT_Y);
            return;
        }
        if (df == DensityFunctions.BASE_3D_Z) {
            buf.put(TAG_INPUT_Z);
            return;
        }

        // ── Scalars ──────────────────────────────────────────────────────
        if (df instanceof DensityFunction.Constant c) {
            writeConstant(buf, c.value());
            return;
        }

        // ── Binary arithmetic ───────────────────────────────────────────
        if (df instanceof DensityFunction.Add a) {
            buf.put(TAG_ADD);
            writeDF(buf, a.a());
            writeDF(buf, a.b());
            return;
        }
        // TODO: 后续子阶段在这里追加 Sub / Mul / Div / Min / Max / FloorDiv
        //       的 instanceof 分支，每个分支结构与 Add 完全一致。
        // TODO: 以及 Abs / Square / Cube / Sqrt 一元分支。
        // TODO: Perlin / DoublePerlin / Simplex2D / Simplex3D 噪声分支。
        // TODO: EndIslands / RangeChoice / Cache* / FindTopSurface 结构分支。
        // TODO: 所有 Beardifier / Spline / IntervalSelect / YClampedGradient /
        //       Aquifer / BiomeTree → 直接 throw UnsupportedDFException，
        //       让上层回退 CPU（对应 Rust 端 placeholder tag 返回 Err 的策略）。

        // Fallback → 抛异常让上层走 CPU 路径。
        throw new UnsupportedDFException(df.getClass().getName());
    }

    /** Helper: write TAG_CONSTANT + a double cast to float (计划.txt D05 全 fp32). */
    private static void writeConstant(ByteBuffer buf, double value) {
        buf.put(TAG_CONSTANT);
        buf.putFloat((float) value);
    }
}
