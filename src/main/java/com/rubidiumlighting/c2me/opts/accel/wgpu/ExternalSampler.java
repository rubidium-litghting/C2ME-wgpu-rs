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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.FutureTask;
import java.util.function.Supplier;

/**
 * Encodes CPU-evaluated external density subtrees into the payload binding 6
 * expects: {@code [chunk][slot][cell]}, all f32, chunks in the order of the
 * submitted positions.
 *
 * <p>The cell index inside one slot is {@code ((j * gz) + k) * gx + i}, the
 * same index the WGSL corner shard uses ({@code cell_idx}), so the shader can
 * read {@code external_inputs[chunk * slots * cells + slot * cells + cell]}.
 *
 * <p>Deliberately free of Minecraft types: the sampling callback receives
 * world coordinates and a per-worker scratch object, which is what makes the
 * layout testable without a game instance. The production adapter wraps a
 * {@link net.minecraft.world.gen.densityfunction.DensityFunction} and uses a
 * {@link MutableNoisePos} as scratch.
 *
 * <p>Chunk slices are independent, so sampling a batch can be split across
 * threads. Each worker gets its own scratch object, exactly like vanilla
 * samples the same density-function instances from several generation workers
 * at once.
 */
public final class ExternalSampler<P> {

    /** Horizontal chunk size in blocks; the corner grid starts at chunk×16. */
    public static final int CHUNK_WIDTH = 16;

    /** One external subtree, addressed by world coordinates. {@code scratch}
     *  is owned by the calling worker and may be mutated. */
    @FunctionalInterface
    public interface CornerSampler<P> {
        double sample(P scratch, int x, int y, int z);
    }

    private final List<CornerSampler<P>> nodes;
    /** Per slot: every reference was under FLAT_CACHE / CACHE2D, so MC's own
     *  cache semantics make the value Y-independent. One sample per column is
     *  then enough; {@link #fillChunk} repeats it across Y. False means the
     *  slot needs a sample at every corner. */
    private final boolean[] yCacheable;
    private final int gx, gy, gz;      // corner counts per axis
    private final int cellW, cellH;    // cell size in blocks
    private final int minY;
    private final Supplier<P> scratchFactory;

    public ExternalSampler(List<CornerSampler<P>> nodes, boolean[] yCacheable,
                           int gx, int gy, int gz, int cellW, int cellH, int minY,
                           Supplier<P> scratchFactory) {
        this.nodes = nodes;
        this.yCacheable = yCacheable;
        this.gx = gx; this.gy = gy; this.gz = gz;
        this.cellW = cellW; this.cellH = cellH;
        this.minY = minY;
        this.scratchFactory = scratchFactory;
    }

    public int nodeCount() {
        return nodes.size();
    }

    /**
     * Slot cap for a dimension: the device's shape-derived ceiling clamped by
     * the policy cost cap.
     *
     * <p>The device answer is the real limit ({@code slots × cells × 4} must
     * fit binding 6); the policy cap bounds CPU sampling cost. A non-positive
     * device answer means the query is unavailable (no device yet, or a native
     * library that predates it), in which case only the policy cap applies.
     */
    public static int slotLimit(int deviceSlots, int policyCap) {
        return deviceSlots > 0 ? Math.min(deviceSlots, policyCap) : policyCap;
    }

    /** Corner samples per chunk per slot. */
    public int cellsPerChunk() {
        return gx * gy * gz;
    }

    /** Floats per chunk slice: every slot's cells back to back. */
    public int stride() {
        return nodes.size() * cellsPerChunk();
    }

    /**
     * Sample {@code chunks} chunks into one payload.
     *
     * @param positions packed chunk coords {@code [cx0, cz0, cx1, cz1, ...]}
     * @param chunks    number of chunks (positions length is {@code chunks*2})
     * @param executor  where worker tasks run; only used when
     *                  {@code chunks > 1 && threads > 1}
     * @param threads   upper bound on workers
     * @return the payload, or {@code null} when there is nothing to sample
     */
    public float[] sampleBatch(int[] positions, int chunks,
                               Executor executor, int threads) {
        int stride = stride();
        if (stride == 0 || chunks <= 0) {
            return null;
        }
        float[] data = new float[chunks * stride];
        if (chunks > 1 && threads > 1 && executor != null) {
            int workers = Math.min(threads, chunks);
            int per = (chunks + workers - 1) / workers;
            List<FutureTask<Void>> tasks = new ArrayList<>(workers);
            for (int t = 0; t < workers; t++) {
                final int from = t * per;
                final int to = Math.min(chunks, from + per);
                if (from >= to) break;
                FutureTask<Void> task = new FutureTask<>(() -> {
                    P scratch = scratchFactory.get();
                    for (int c = from; c < to; c++) {
                        fillChunk(positions[c * 2], positions[c * 2 + 1],
                                data, c * stride, scratch);
                    }
                    return null;
                });
                tasks.add(task);
                executor.execute(task);
            }
            try {
                for (FutureTask<Void> f : tasks) f.get();
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("external sampling interrupted", ie);
            } catch (ExecutionException ee) {
                Throwable c = ee.getCause();
                if (c instanceof RuntimeException re) throw re;
                if (c instanceof Error e) throw e;
                throw new RuntimeException(c);
            }
        } else {
            P scratch = scratchFactory.get();
            for (int c = 0; c < chunks; c++) {
                fillChunk(positions[c * 2], positions[c * 2 + 1],
                        data, c * stride, scratch);
            }
        }
        return data;
    }

    /** Sample one chunk's slice: {@code [slot][j][k][i]} at
     *  {@code data[offset..]}. Safe to run concurrently for different chunks
     *  as long as each caller passes its own scratch. */
    public void fillChunk(int cx, int cz, float[] data, int offset, P scratch) {
        int sx = cx * CHUNK_WIDTH;
        int sz = cz * CHUNK_WIDTH;
        int plane = gx * gz;   // one Y slice per external node
        int idx = offset;
        for (int e = 0; e < nodes.size(); e++) {
            CornerSampler<P> ext = nodes.get(e);
            boolean column = e < yCacheable.length && yCacheable[e];
            if (column) {
                for (int k = 0; k < gz; k++) {
                    int wz = sz + k * cellW;
                    for (int i = 0; i < gx; i++) {
                        int wx = sx + i * cellW;
                        data[idx + k * gx + i] =
                                (float) ext.sample(scratch, wx, 0, wz);
                    }
                }
                for (int j = 1; j < gy; j++) {
                    System.arraycopy(data, idx, data, idx + j * plane, plane);
                }
                idx += gy * plane;
            } else {
                for (int j = 0; j < gy; j++) {
                    int wy = minY + j * cellH;
                    for (int k = 0; k < gz; k++) {
                        int wz = sz + k * cellW;
                        for (int i = 0; i < gx; i++) {
                            int wx = sx + i * cellW;
                            data[idx++] = (float) ext.sample(scratch, wx, wy, wz);
                        }
                    }
                }
            }
        }
    }
}
