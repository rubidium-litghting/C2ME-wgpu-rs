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

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * The external-input payload is the contract between Java and the WGSL corner
 * shard: {@code external_inputs[chunk * slots * cells + slot * cells + cell]}.
 * These tests pin that layout, the chunk stride, and both sampling paths
 * (single chunk serial, several chunks parallel) without loading Minecraft.
 */
class ExternalSamplerTest {

    /** Exact in f32 for the small coordinates these tests use, and it encodes
     *  the absolute world position, so a wrong origin or a wrong cell index
     *  both show up as a wrong value. */
    private static float expected(int x, int y, int z) {
        return (float) (x + z * 1000.0 + y * 1_000_000.0);
    }

    private static ExternalSampler.CornerSampler<Object> world() {
        return (scratch, x, y, z) -> expected(x, y, z);
    }

    private static ExternalSampler<Object> sampler(
            List<ExternalSampler.CornerSampler<Object>> nodes, boolean[] yCacheable,
            int gx, int gy, int gz) {
        // cellW = cellH = 1 and minY = 0: corner (i, j, k) sits at
        // (sx + i, j, sz + k), so expected() decodes the cell exactly.
        return new ExternalSampler<>(nodes, yCacheable,
                gx, gy, gz, 1, 1, 0, Object::new);
    }

    private static List<ExternalSampler.CornerSampler<Object>> nodes(int n) {
        List<ExternalSampler.CornerSampler<Object>> out = new ArrayList<>();
        for (int i = 0; i < n; i++) out.add(world());
        return out;
    }

    /** Cell index the shader uses: {@code ((j * gz) + k) * gx + i}. */
    private static int cell(int i, int j, int k, int gx, int gz) {
        return ((j * gz) + k) * gx + i;
    }

    private static void assertChunkSlice(float[] data, int chunkOffset, int cx, int cz,
                                         int gx, int gy, int gz, int slot, int slotCount,
                                         boolean column) {
        int cells = gx * gy * gz;
        int base = chunkOffset + slot * cells;
        int sx = cx * ExternalSampler.CHUNK_WIDTH;
        int sz = cz * ExternalSampler.CHUNK_WIDTH;
        for (int j = 0; j < gy; j++) {
            for (int k = 0; k < gz; k++) {
                for (int i = 0; i < gx; i++) {
                    int wy = column ? 0 : j;
                    float want = expected(sx + i, wy, sz + k);
                    assertEquals(want, data[base + cell(i, j, k, gx, gz)],
                            "slot " + slot + "/" + slotCount + " cell i=" + i
                                    + " j=" + j + " k=" + k + " of chunk " + cx + "," + cz);
                }
            }
        }
    }

    @Test
    void layoutIsSlotMajorThenYThenZThenX() {
        int gx = 3, gy = 2, gz = 4;
        // Slot 0 is Y-cacheable, slot 1 is not: one payload covers both.
        ExternalSampler<Object> sampler = sampler(nodes(2), new boolean[]{true, false},
                gx, gy, gz);

        float[] data = sampler.sampleBatch(new int[]{0, 0}, 1, null, 1);

        assertEquals(2 * gx * gy * gz, sampler.stride());
        assertEquals(gx * gy * gz, sampler.cellsPerChunk());
        assertEquals(2, sampler.nodeCount());
        assertEquals(sampler.stride(), data.length);
        assertChunkSlice(data, 0, 0, 0, gx, gy, gz, 0, 2, true);
        assertChunkSlice(data, 0, 0, 0, gx, gy, gz, 1, 2, false);
    }

    @Test
    void chunkOriginFollowsThePosition() {
        int gx = 2, gy = 2, gz = 2;
        ExternalSampler<Object> sampler = sampler(nodes(1), new boolean[]{false},
                gx, gy, gz);

        for (int[] pos : new int[][]{{0, 0}, {5, -3}, {-2, 7}}) {
            float[] data = sampler.sampleBatch(new int[]{pos[0], pos[1]}, 1, null, 1);
            assertChunkSlice(data, 0, pos[0], pos[1], gx, gy, gz, 0, 1, false);
        }
    }

    @Test
    void batchIsChunkMajorWithFixedStride() {
        int gx = 2, gy = 3, gz = 2;
        ExternalSampler<Object> sampler = sampler(nodes(2), new boolean[]{false, true},
                gx, gy, gz);
        int[][] positions = {{0, 0}, {3, 4}, {-1, 2}};

        float[] data = sampler.sampleBatch(
                new int[]{0, 0, 3, 4, -1, 2}, 3, null, 1);

        assertEquals(3 * sampler.stride(), data.length);
        for (int c = 0; c < positions.length; c++) {
            int offset = c * sampler.stride();
            assertChunkSlice(data, offset, positions[c][0], positions[c][1],
                    gx, gy, gz, 0, 2, false);
            assertChunkSlice(data, offset, positions[c][0], positions[c][1],
                    gx, gy, gz, 1, 2, true);
        }
    }

    @Test
    void parallelBatchMatchesSerial() {
        int gx = 3, gy = 2, gz = 3;
        ExternalSampler<Object> sampler = sampler(nodes(2), new boolean[]{true, false},
                gx, gy, gz);
        int chunks = 5;
        int[] positions = new int[chunks * 2];
        for (int c = 0; c < chunks; c++) {
            positions[c * 2] = c * 7 - 3;
            positions[c * 2 + 1] = 11 - c * 5;
        }

        float[] serial = sampler.sampleBatch(positions, chunks, null, 1);

        ExecutorService pool = Executors.newFixedThreadPool(3);
        try {
            // 5 chunks over 3 workers: partitions are [0,2) [2,4) [4,5).
            float[] parallel = sampler.sampleBatch(positions, chunks, pool, 3);
            assertArrayEquals(serial, parallel);
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    void parallelWorkersEachGetTheirOwnScratch() {
        int gx = 2, gy = 2, gz = 2;
        // Record which scratch instance each thread used. A shared scratch
        // would either appear under two threads or make the map inconsistent.
        Map<Thread, Object> scratchByThread = Collections.synchronizedMap(new IdentityHashMap<>());
        AtomicInteger conflicts = new AtomicInteger();
        List<ExternalSampler.CornerSampler<Object>> one = new ArrayList<>();
        one.add((scratch, x, y, z) -> {
            Object prev = scratchByThread.putIfAbsent(Thread.currentThread(), scratch);
            if (prev != null && prev != scratch) conflicts.incrementAndGet();
            return expected(x, y, z);
        });
        ExternalSampler<Object> sampler = sampler(one, new boolean[]{false}, gx, gy, gz);

        ExecutorService pool = Executors.newFixedThreadPool(2);
        try {
            sampler.sampleBatch(new int[]{0, 0, 1, 0, 2, 0, 3, 0}, 4, pool, 2);
        } finally {
            pool.shutdownNow();
        }

        assertEquals(0, conflicts.get(), "a scratch object leaked across workers");
        for (Map.Entry<Thread, Object> e : scratchByThread.entrySet()) {
            assertSame(e.getValue(), scratchByThread.get(e.getKey()));
        }
        // Each worker's scratch must be its own instance, not one shared object.
        Set<Object> distinct = Collections.newSetFromMap(new IdentityHashMap<>());
        distinct.addAll(scratchByThread.values());
        assertEquals(scratchByThread.size(), distinct.size());
    }

    @Test
    void columnSlotSamplesOncePerColumn() {
        int gx = 2, gy = 4, gz = 3;
        AtomicInteger columnCalls = new AtomicInteger();
        AtomicInteger fullCalls = new AtomicInteger();
        List<ExternalSampler.CornerSampler<Object>> one = new ArrayList<>();
        one.add((scratch, x, y, z) -> {
            columnCalls.incrementAndGet();
            return 1.0;
        });
        ExternalSampler<Object> sampler = sampler(one, new boolean[]{true}, gx, gy, gz);
        sampler.sampleBatch(new int[]{0, 0}, 1, null, 1);
        assertEquals(gx * gz, columnCalls.get());

        List<ExternalSampler.CornerSampler<Object>> two = new ArrayList<>();
        two.add((scratch, x, y, z) -> {
            fullCalls.incrementAndGet();
            return 1.0;
        });
        ExternalSampler<Object> perCorner = sampler(two, new boolean[]{false}, gx, gy, gz);
        perCorner.sampleBatch(new int[]{0, 0}, 1, null, 1);
        assertEquals(gx * gy * gz, fullCalls.get());
    }

    @Test
    void missingCacheableFlagMeansPerCorner() {
        int gx = 2, gy = 2, gz = 2;
        // yCacheable is shorter than the node list: slot 1 has no flag.
        ExternalSampler<Object> sampler = sampler(nodes(2), new boolean[]{true},
                gx, gy, gz);

        float[] data = sampler.sampleBatch(new int[]{0, 0}, 1, null, 1);

        assertChunkSlice(data, 0, 0, 0, gx, gy, gz, 0, 2, true);
        assertChunkSlice(data, 0, 0, 0, gx, gy, gz, 1, 2, false);
    }

    @Test
    void noNodesMeansNoPayload() {
        ExternalSampler<Object> sampler = sampler(nodes(0), new boolean[0], 2, 2, 2);
        assertEquals(0, sampler.stride());
        assertNull(sampler.sampleBatch(new int[]{0, 0, 1, 1}, 2, null, 1));
    }

    @Test
    void slotLimitTakesTheDeviceCeilingUnlessItIsUnknown() {
        // Device answer is the real limit; the policy cap only applies on top.
        assertEquals(32, ExternalSampler.slotLimit(32, 256));
        assertEquals(256, ExternalSampler.slotLimit(13_836, 256));
        // -1 (no device yet) and 0 fall back to the policy cap.
        assertEquals(256, ExternalSampler.slotLimit(-1, 256));
        assertEquals(256, ExternalSampler.slotLimit(0, 256));
    }

    @Test
    void samplerFailuresPropagate() {
        List<ExternalSampler.CornerSampler<Object>> boom = new ArrayList<>();
        boom.add((scratch, x, y, z) -> {
            throw new IllegalStateException("boom");
        });
        ExternalSampler<Object> sampler = sampler(boom, new boolean[]{false}, 2, 2, 2);

        assertThrows(IllegalStateException.class,
                () -> sampler.sampleBatch(new int[]{0, 0}, 1, null, 1));

        ExecutorService pool = Executors.newFixedThreadPool(2);
        try {
            assertThrows(IllegalStateException.class,
                    () -> sampler.sampleBatch(new int[]{0, 0, 1, 0}, 2, pool, 2));
        } finally {
            pool.shutdownNow();
        }
    }
}
