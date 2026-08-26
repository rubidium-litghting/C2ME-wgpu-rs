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

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.IdentityHashMap;
import java.util.Map;

import net.minecraft.world.level.levelgen.DensityFunction;
import net.minecraft.world.level.levelgen.DensityFunctions;
import net.minecraft.world.level.levelgen.NoiseRouter;

/**
 * 将 26.2 的 NoiseRouter 序列化为二进制格式，使用反射访问私有字段。
 */
public class NoiseRouterBlobifier {

    // 标签常量
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
    private static final byte TAG_FLOOR_DIV = 0x24;
    private static final byte TAG_ABS = 0x20;
    private static final byte TAG_SQUARE = 0x21;
    private static final byte TAG_CUBE = 0x22;
    private static final byte TAG_SQRT = 0x23;
    private static final byte TAG_NOISE_SAMPLER = (byte) 0x84;
    private static final byte TAG_END_ISLANDS = (byte) 0xA0;
    private static final byte TAG_RANGE_CHOICE = (byte) 0xB1;
    private static final byte TAG_FIND_TOP_SURFACE = (byte) 0xD0;
    private static final byte TAG_ROOT = (byte) 0xFE;
    private static final byte TAG_SHIFTED_NOISE = 0x32;
    private static final byte TAG_CLAMP = 0x33;
    private static final byte TAG_MARKER = 0x34;

    private static final Map<DensityFunction, Void> visited = new IdentityHashMap<>();

    // 缓存类名以加速
    private static final String CONSTANT_CLASS = "net.minecraft.world.level.levelgen.DensityFunctions$Constant";
    private static final String MUL_OR_ADD_CLASS = "net.minecraft.world.level.levelgen.DensityFunctions$MulOrAdd";
    private static final String NOISE_CLASS = "net.minecraft.world.level.levelgen.DensityFunctions$Noise";
    private static final String RANGE_CHOICE_CLASS = "net.minecraft.world.level.levelgen.DensityFunctions$RangeChoice";
    private static final String FIND_TOP_SURFACE_CLASS = "net.minecraft.world.level.levelgen.DensityFunctions$FindTopSurface";
    private static final String END_ISLAND_CLASS = "net.minecraft.world.level.levelgen.DensityFunctions$EndIslandDensityFunction";

    public static ByteBuffer blobify(NoiseRouter router, ShapeConfig shape) {
        visited.clear();
        ByteBuffer buf = ByteBuffer.allocate(4096).order(ByteOrder.LITTLE_ENDIAN);
        buf.putInt(0x4D433257);
        buf.putShort((short) 1);
        buf.putInt(0);

        // 序列化 15 个密度函数
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
        writeDF(buf, router.preliminarySurfaceLevel());
        writeDF(buf, router.finalDensity());
        writeDF(buf, router.veinToggle());
        writeDF(buf, router.veinRidged());
        writeDF(buf, router.veinGap());

        buf.putInt(shape.minY);
        buf.putInt(shape.height);
        buf.putInt(shape.cellWidth);
        buf.putInt(shape.cellHeight);

        // BIOME_SOURCE: 占位 BiomeTree
        buf.put((byte) 0xF0);
        buf.putLong(0);

        buf.flip();
        return buf;
    }

    private static void writeDF(ByteBuffer buf, DensityFunction df) {
        System.out.println("[c2me-wgpu] writeDF: " + df.getClass().getName());
        if (df == null) {
            buf.put(TAG_CONSTANT);
            buf.putFloat(0.0f);
            return;
        }
        // 使用类名判断
        String className = df.getClass().getName();
        try {
            if (className.equals(CONSTANT_CLASS)) {
                double val = getDoubleField(df, "value");
                buf.put(TAG_CONSTANT);
                buf.putFloat((float) val);
            } else if (className.equals(MUL_OR_ADD_CLASS)) {
                // 读取 specificType, input, argument
                Object type = getField(df, "specificType");
                String typeName = type.toString(); // 可能是 "ADD" 或 "MUL"
                if ("ADD".equals(typeName)) {
                    buf.put(TAG_ADD);
                } else if ("MUL".equals(typeName)) {
                    buf.put(TAG_MUL);
                } else {
                    throw new UnsupportedOperationException("MulOrAdd type: " + typeName);
                }
                DensityFunction input = (DensityFunction) getField(df, "input");
                double argument = getDoubleField(df, "argument");
                writeDF(buf, input);
                buf.put(TAG_CONSTANT);
                buf.putFloat((float) argument);
            } else if (className.equals(NOISE_CLASS)) {
                Object noiseHolder = getField(df, "noise");
                double xzScale = getDoubleField(df, "xzScale");
                double yScale = getDoubleField(df, "yScale");
                byte noiseType = 0;
                long seed = 0;
                // 判断 noiseHolder 类型
                String holderClassName = noiseHolder.getClass().getName();
                if (holderClassName.contains("PerlinNoise")) {
                    noiseType = 0;
                    seed = getSeedFromNoise(noiseHolder);
                } else if (holderClassName.contains("SimplexNoise")) {
                    noiseType = 1;
                    seed = getSeedFromNoise(noiseHolder);
                } else {
                    noiseType = 0;
                    seed = 0;
                }
                buf.put(TAG_NOISE_SAMPLER);
                buf.put(noiseType);
                buf.putLong(seed);
                buf.putFloat((float) xzScale);
                buf.putFloat((float) yScale);
            } else if (className.equals(RANGE_CHOICE_CLASS)) {
                DensityFunction input = (DensityFunction) getField(df, "input");
                double minInclusive = getDoubleField(df, "minInclusive");
                double maxExclusive = getDoubleField(df, "maxExclusive");
                DensityFunction whenInRange = (DensityFunction) getField(df, "whenInRange");
                DensityFunction whenOutOfRange = (DensityFunction) getField(df, "whenOutOfRange");
                buf.put(TAG_RANGE_CHOICE);
                writeDF(buf, input);
                // 用常量包装 min 和 max
                buf.put(TAG_CONSTANT);
                buf.putFloat((float) minInclusive);
                buf.put(TAG_CONSTANT);
                buf.putFloat((float) maxExclusive);
                writeDF(buf, whenInRange);
                writeDF(buf, whenOutOfRange);
                // 多写一个 out_hi，用 whenOutOfRange 替代
                writeDF(buf, whenOutOfRange);
            } else if (className.equals(FIND_TOP_SURFACE_CLASS)) {
                DensityFunction density = (DensityFunction) getField(df, "density");
                DensityFunction upperBound = (DensityFunction) getField(df, "upperBound");
                int lowerBound = getIntField(df, "lowerBound");
                int cellHeight = getIntField(df, "cellHeight");
                buf.put(TAG_FIND_TOP_SURFACE);
                writeDF(buf, density);
                writeDF(buf, upperBound);
                buf.putInt(lowerBound);
                buf.putInt(cellHeight);
            } else if (className.equals(END_ISLAND_CLASS)) {
                buf.put(TAG_END_ISLANDS);
                buf.put(TAG_INPUT_X);
            } else if (className.equals("net.minecraft.world.level.levelgen.DensityFunctions$Clamp")) {
                // 字段: input, minValue, maxValue
                try {
                    DensityFunction input = (DensityFunction) getField(df, "input");
                    double minValue = getDoubleField(df, "minValue");
                    double maxValue = getDoubleField(df, "maxValue");
                    buf.put(TAG_CLAMP);
                    writeDF(buf, input);
                    // min 和 max 作为常量子节点
                    buf.put(TAG_CONSTANT);
                    buf.putFloat((float) minValue);
                    buf.put(TAG_CONSTANT);
                    buf.putFloat((float) maxValue);
                } catch (Exception e) {
                    System.err.println("[c2me-wgpu] ❌ Failed to serialize Clamp: " + e.getMessage());
                    buf.put(TAG_CONSTANT);
                    buf.putFloat(0.0f);
                }
            } else if (className.equals("net.minecraft.world.level.levelgen.DensityFunctions$Marker")) {
                // Marker 只是一个包装器，递归其 wrapped
                try {
                    DensityFunction wrapped = (DensityFunction) getField(df, "wrapped");
                    buf.put(TAG_MARKER);
                    writeDF(buf, wrapped);
                } catch (Exception e) {
                    System.err.println("[c2me-wgpu] ❌ Failed to serialize Marker: " + e.getMessage());
                    buf.put(TAG_CONSTANT);
                    buf.putFloat(0.0f);
                }
            } else if (className.equals("net.minecraft.world.level.levelgen.DensityFunctions$HolderHolder")) {
                try {
                    Object holder = getField(df, "holder");
                    java.lang.reflect.Method valueMethod = holder.getClass().getMethod("value");
                    DensityFunction unwrapped = (DensityFunction) valueMethod.invoke(holder);
                    writeDF(buf, unwrapped);
                } catch (Exception e) {
                    System.err.println("[c2me-wgpu] ❌ Failed to unwrap HolderHolder: " + e.getMessage());
                    buf.put(TAG_CONSTANT);
                    buf.putFloat(0.0f);
                }
            } else if (className.equals("net.minecraft.world.level.levelgen.DensityFunctions$Add")) {
                buf.put(TAG_ADD);
                writeDF(buf, (DensityFunction) getField(df, "argument1"));
                writeDF(buf, (DensityFunction) getField(df, "argument2"));
            } else if (className.equals("net.minecraft.world.level.levelgen.DensityFunctions$Mul")) {
                buf.put(TAG_MUL);
                writeDF(buf, (DensityFunction) getField(df, "argument1"));
                writeDF(buf, (DensityFunction) getField(df, "argument2"));
            } else if (className.equals("net.minecraft.world.level.levelgen.DensityFunctions$Min")) {
                buf.put(TAG_MIN);
                writeDF(buf, (DensityFunction) getField(df, "argument1"));
                writeDF(buf, (DensityFunction) getField(df, "argument2"));
            } else if (className.equals("net.minecraft.world.level.levelgen.DensityFunctions$Max")) {
                buf.put(TAG_MAX);
                writeDF(buf, (DensityFunction) getField(df, "argument1"));
                writeDF(buf, (DensityFunction) getField(df, "argument2"));
            } else if (className.contains("Ap2")) {
                try {
                    // 获取操作类型
                    Object type = null;
                    try {
                        // 尝试获取字段 "type"
                        Field typeField = df.getClass().getDeclaredField("type");
                        typeField.setAccessible(true);
                        type = typeField.get(df);
                    } catch (NoSuchFieldException e1) {
                        try {
                            // 尝试方法 "type()"
                            Method typeMethod = df.getClass().getMethod("type");
                            type = typeMethod.invoke(df);
                        } catch (NoSuchMethodException e2) {
                            // 尝试字段 "specificType"
                            Field specificTypeField = df.getClass().getDeclaredField("specificType");
                            specificTypeField.setAccessible(true);
                            type = specificTypeField.get(df);
                        }
                    }
                    if (type == null) {
                        // 回退到常量 0
                        buf.put(TAG_CONSTANT);
                        buf.putFloat(0.0f);
                        return;
                    }
                    String typeName = type.toString().toUpperCase();
                    byte tag;
                    switch (typeName) {
                        case "ADD": tag = TAG_ADD; break;
                        case "SUB": tag = TAG_SUB; break;
                        case "MUL": tag = TAG_MUL; break;
                        case "DIV": tag = TAG_DIV; break;
                        case "MIN": tag = TAG_MIN; break;
                        case "MAX": tag = TAG_MAX; break;
                        default: tag = TAG_ADD; break;
                    }
                    buf.put(tag);

                    // 获取左右子节点
                    DensityFunction left = null;
                    DensityFunction right = null;
                    try {
                        Field leftField = df.getClass().getDeclaredField("left");
                        leftField.setAccessible(true);
                        left = (DensityFunction) leftField.get(df);
                    } catch (NoSuchFieldException e) {
                        // 尝试 "input" 和 "argument"
                        Field inputField = df.getClass().getDeclaredField("input");
                        inputField.setAccessible(true);
                        left = (DensityFunction) inputField.get(df);
                        Field argField = df.getClass().getDeclaredField("argument");
                        argField.setAccessible(true);
                        // 如果 argument 是 double，则包装为常量
                        double arg = argField.getDouble(df);
                        right = DensityFunctions.constant(arg);
                    }
                    if (right == null) {
                        try {
                            Field rightField = df.getClass().getDeclaredField("right");
                            rightField.setAccessible(true);
                            right = (DensityFunction) rightField.get(df);
                        } catch (NoSuchFieldException e) {
                            // 尝试 "argument2" 等
                            Field arg2Field = df.getClass().getDeclaredField("argument2");
                            arg2Field.setAccessible(true);
                            double arg2 = arg2Field.getDouble(df);
                            right = DensityFunctions.constant(arg2);
                        }
                    }
                    if (left == null || right == null) {
                        buf.put(TAG_CONSTANT);
                        buf.putFloat(0.0f);
                        return;
                    }
                    writeDF(buf, left);
                    writeDF(buf, right);
                } catch (Exception e) {
                    // 如果失败，回退到常量 0
                    buf.put(TAG_CONSTANT);
                    buf.putFloat(0.0f);
                }
            } else if (className.contains("IntervalSelect")) {
                buf.put(TAG_CONSTANT);
                buf.putFloat(0.0f);
            // Unsupported: 
            } else if (className.contains("Spline")) {
                buf.put(TAG_CONSTANT);
                buf.putFloat(0.0f);
            // Unsupported: 
            } else if (className.contains("ShiftedNoise")) {
                try {
                    // 获取三个偏移函数和噪声对象
                    DensityFunction shiftX = (DensityFunction) getField(df, "shiftX");
                    DensityFunction shiftY = (DensityFunction) getField(df, "shiftY");
                    DensityFunction shiftZ = (DensityFunction) getField(df, "shiftZ");
                    Object noiseHolder = getField(df, "noise");
                    double xzScale = getDoubleField(df, "xzScale");
                    double yScale = getDoubleField(df, "yScale");
                    buf.put(TAG_SHIFTED_NOISE);
                    // 序列化三个偏移函数
                    writeDF(buf, shiftX);
                    writeDF(buf, shiftY);
                    writeDF(buf, shiftZ);
                    // 序列化噪声采样器（类似 Noise 节点）
                    byte noiseType = 0;
                    if (noiseHolder.getClass().getName().contains("Simplex")) noiseType = 1;
                    long seed = getSeedFromNoise(noiseHolder);
                    buf.put(noiseType);
                    buf.putLong(seed);
                    buf.putFloat((float) xzScale);
                    buf.putFloat((float) yScale);
                } catch (Exception e) {
                    System.err.println("[c2me-wgpu] ❌ Failed to serialize ShiftedNoise: " + e.getMessage());
                    buf.put(TAG_CONSTANT);
                    buf.putFloat(0.0f);
                }
            } else if (className.contains("BeardifierMarker")) {
                // 暂时用 Constant(0) 替代
                buf.put(TAG_CONSTANT);
                buf.putFloat(0.0f);
            }  
            else {
                // 【诊断模式】打印未识别节点的完整字段信息
                System.err.println("[c2me-wgpu] ⚠️ UNSUPPORTED NODE: " + className);
                System.err.println("[c2me-wgpu]   Available fields: ");
                for (java.lang.reflect.Field f : df.getClass().getDeclaredFields()) {
                    f.setAccessible(true);
                    Object val = null;
                    try { val = f.get(df); } catch (Exception ignored) {}
                    String valStr = (val instanceof DensityFunction)
                        ? "DensityFunction(" + val.getClass().getSimpleName() + ")"
                        : String.valueOf(val);
                    System.err.println("[c2me-wgpu]     - " + f.getName() + " : " + f.getType().getSimpleName() + " = " + valStr);
                }
                buf.put(TAG_CONSTANT);
                buf.putFloat(0.0f);
            }
        } catch (Exception e) {
            System.err.println("[c2me-wgpu] ❌ FAILED to serialize: " + className + " | " + e.getMessage());
            e.printStackTrace();
            buf.put(TAG_CONSTANT);
            buf.putFloat(0.0f);
        }
    }

    // 反射工具方法
    private static Object getField(Object obj, String fieldName) throws Exception {
        Field field = obj.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        return field.get(obj);
    }

    private static double getDoubleField(Object obj, String fieldName) throws Exception {
        return (double) getField(obj, fieldName);
    }

    private static int getIntField(Object obj, String fieldName) throws Exception {
        return (int) getField(obj, fieldName);
    }

    private static long getSeedFromNoise(Object noise) {
        try {
            // 尝试调用 seed() 方法（如果有）
            Method method = noise.getClass().getMethod("seed");
            return (long) method.invoke(noise);
        } catch (Exception e1) {
            try {
                // 尝试读取 seed 字段
                Field field = noise.getClass().getDeclaredField("seed");
                field.setAccessible(true);
                return field.getLong(noise);
            } catch (Exception e2) {
                return 0;
            }
        }
    }

    public static class ShapeConfig {
        public final int minY, height, cellWidth, cellHeight;
        public ShapeConfig(int minY, int height, int cellWidth, int cellHeight) {
            this.minY = minY; this.height = height; this.cellWidth = cellWidth; this.cellHeight = cellHeight;
        }
    }
}