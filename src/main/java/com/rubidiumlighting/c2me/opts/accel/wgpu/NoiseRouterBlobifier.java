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

import net.minecraft.world.gen.chunk.ChunkGeneratorSettings;
import net.minecraft.world.gen.noise.NoiseRouter;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Serialize MC NoiseRouter + ShapeConfig + BiomeSource into a self-describing
 * [tag][payload] recursive binary blob understood by the Rust deserializer.
 * <p>
 * All numeric values use fp32 (no double). Byte order: Little Endian.
 * <p>
 * TODO: Implement all 30+ DensityFunction cases (refer to DFC McToAst.java instanceof chain).
 * Currently this is a skeleton returning an empty placeholder blob.
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
     * @return direct ByteBuffer containing the serialized blob (Little Endian)
     */
    public static ByteBuffer blobify(NoiseRouter router, ChunkGeneratorSettings settings) {
        // TODO: Implement recursive serializer with proper size estimation.
        // Skeleton: write header only. Real code must:
        //   1. Walk router.finalDensity / router.erosion / ... all 16 channels
        //   2. instanceof-match each DensityFunction type, emit [tag][payload]
        //   3. Append SHAPE_CONFIG = [min_y:i32, height:i32, cell_width:i32, cell_height:i32]
        //   4. Append BIOME_SOURCE = flattened MultiNoise decision tree

        ByteBuffer buf = ByteBuffer.allocate(1024);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        buf.putInt(MAGIC);
        buf.putShort(VERSION);
        buf.put(new byte[4]); // reserved

        // ---- PLACEHOLDER: emit TAG_ROOT with zero sub-nodes ----
        // Real implementation: 16 sub-nodes in NoiseRouter channel order.
        buf.put(TAG_ROOT);

        buf.flip();
        return buf;
    }

}
