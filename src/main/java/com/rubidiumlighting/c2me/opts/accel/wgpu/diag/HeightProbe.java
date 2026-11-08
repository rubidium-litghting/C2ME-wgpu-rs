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

package com.rubidiumlighting.c2me.opts.accel.wgpu.diag;

import com.rubidiumlighting.c2me.opts.accel.wgpu.WgpuBatchCoordinator;
import com.rubidiumlighting.c2me.opts.accel.wgpu.config.Config;

import net.minecraft.world.Heightmap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * Measures {@code ChunkGenerator.getHeight}: how often it is called, how long
 * it takes, how many of those columns are distinct, which threads call it, and
 * how much of it runs while the GPU pipeline is busy.
 *
 * <p>This exists to answer one question before any surface-height
 * optimisation is written (DEVELOPMENT.md §28.5): how much wall-clock time
 * could a cached heightmap actually save? The cost model says a single
 * {@code getHeight} is roughly 1/6 of a chunk's noise work, and that the
 * overworld pays one call per chunk because mineshafts use
 * {@code spacing=1}. Only a measurement can say whether those seconds are on
 * the critical path or hidden behind GPU readback.
 *
 * <p>Enabled by {@code debug_height_probe}. The flag is read once, at class
 * load: a mid-session change would split one run's counters across two modes.
 *
 * <p>Overhead is two {@code System.nanoTime()} calls plus a few counters per
 * query, far below the ~0.3–0.8 ms the query itself costs, so the numbers it
 * reports are not distorted by the measurement.
 */
public final class HeightProbe {

    private static final Logger LOGGER = LoggerFactory.getLogger("c2me-wgpu-height");

    private static final boolean ENABLED = Config.get().debugHeightProbe;

    /** Emit a line when either threshold is crossed, so short runs still get
     *  output and long runs do not spam. */
    private static final int REPORT_EVERY_CALLS = 256;
    private static final long REPORT_INTERVAL_NS = 2_000_000_000L;

    private static final int TYPES = Heightmap.Type.values().length;

    private static final ThreadLocal<long[]> START = ThreadLocal.withInitial(() -> new long[1]);

    private static final LongAdder CALLS = new LongAdder();
    private static final LongAdder TOTAL_NANOS = new LongAdder();
    private static final LongAdder GPU_BUSY_CALLS = new LongAdder();
    private static final LongAdder[] BY_TYPE = new LongAdder[TYPES];
    @SuppressWarnings("unchecked")
    private static final ConcurrentHashMap<Long, Boolean>[] COLUMNS =
            new ConcurrentHashMap[TYPES];
    private static final ConcurrentHashMap<String, LongAdder> BY_THREAD =
            new ConcurrentHashMap<>();
    private static final AtomicLong MAX_NANOS = new AtomicLong();

    private static final AtomicLong LAST_REPORT_CALLS = new AtomicLong();
    private static volatile long lastReportWall = System.nanoTime();

    static {
        for (int i = 0; i < TYPES; i++) {
            BY_TYPE[i] = new LongAdder();
            COLUMNS[i] = new ConcurrentHashMap<>();
        }
    }

    private HeightProbe() {
    }

    /** HEAD of {@code getHeight}. */
    public static void begin() {
        if (!ENABLED) {
            return;
        }
        START.get()[0] = System.nanoTime();
    }

    /** RETURN of {@code getHeight}. */
    public static void end(int x, int z, Heightmap.Type type) {
        if (!ENABLED) {
            return;
        }
        long[] slot = START.get();
        long start = slot[0];
        if (start == 0L) {
            // HEAD never ran (should not happen), or a previous call threw and
            // left a stale value that HEAD already overwrote. Nothing to add.
            return;
        }
        slot[0] = 0L;

        long now = System.nanoTime();
        long elapsed = now - start;
        CALLS.increment();
        TOTAL_NANOS.add(elapsed);
        MAX_NANOS.accumulateAndGet(elapsed, Math::max);
        if (WgpuBatchCoordinator.activeRequests() > 0) {
            GPU_BUSY_CALLS.increment();
        }

        int typeIndex = type.ordinal();
        BY_TYPE[typeIndex].increment();
        COLUMNS[typeIndex].putIfAbsent(((long) x << 32) | (z & 0xFFFFFFFFL), Boolean.TRUE);
        BY_THREAD.computeIfAbsent(Thread.currentThread().getName(), k -> new LongAdder())
                .increment();

        maybeReport(now);
    }

    private static void maybeReport(long now) {
        long calls = CALLS.sum();
        long last = LAST_REPORT_CALLS.get();
        boolean byCalls = calls - last >= REPORT_EVERY_CALLS;
        boolean byTime = now - lastReportWall >= REPORT_INTERVAL_NS;
        if (!byCalls && !byTime) {
            return;
        }
        if (!LAST_REPORT_CALLS.compareAndSet(last, calls)) {
            return;
        }
        lastReportWall = now;
        report(calls, calls - last);
    }

    private static void report(long calls, long windowCalls) {
        long unique = 0L;
        StringBuilder types = new StringBuilder();
        for (int i = 0; i < TYPES; i++) {
            long n = BY_TYPE[i].sum();
            if (n == 0L) {
                continue;
            }
            unique += COLUMNS[i].size();
            if (types.length() > 0) {
                types.append(',');
            }
            types.append(Heightmap.Type.values()[i].name()).append(':').append(n);
        }

        long totalNanos = TOTAL_NANOS.sum();
        double avgMs = calls > 0 ? totalNanos / 1_000_000.0 / calls : 0.0;
        double maxMs = MAX_NANOS.get() / 1_000_000.0;
        double gpuBusy = calls > 0 ? 100.0 * GPU_BUSY_CALLS.sum() / calls : 0.0;
        double dup = calls > 0 ? 100.0 * (calls - unique) / calls : 0.0;

        LOGGER.info("[c2me-wgpu-height] calls={} (+{}) unique={} dup={}% "
                        + "avg={}ms max={}ms total={}s gpuBusy={}% types=[{}] threads=[{}]",
                calls, windowCalls, unique, fmt(dup), fmt(avgMs), fmt(maxMs),
                fmt(totalNanos / 1_000_000_000.0), fmt(gpuBusy), types, topThreads());
    }

    /** Busiest four threads, so the line stays readable with a wide worker
     *  pool. */
    private static String topThreads() {
        List<Map.Entry<String, LongAdder>> entries = new ArrayList<>(BY_THREAD.entrySet());
        entries.sort(Comparator.comparingLong(e -> -e.getValue().sum()));
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < entries.size() && i < 4; i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append(entries.get(i).getKey()).append(':').append(entries.get(i).getValue().sum());
        }
        return sb.toString();
    }

    private static String fmt(double value) {
        return String.format(java.util.Locale.ROOT, "%.2f", value);
    }
}
