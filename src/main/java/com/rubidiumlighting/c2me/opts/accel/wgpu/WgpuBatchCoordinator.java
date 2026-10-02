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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Coalesces per-chunk GPU requests into a single dispatch.
 *
 * <p>C2ME calls {@code populateNoise} from many worker threads. If each
 * chunk issued its own {@code requestChunks()}, the GPU would run one tiny
 * 5x97x5 dispatch per chunk and the fixed per-dispatch cost (buffer alloc,
 * bind group, submit, readback) would dominate. Instead every request is
 * queued here; one coordinator thread drains the queue into a batch and
 * issues a single {@code requestChunks()} for all of them.
 *
 * <p>The Rust side sorts + dedups the positions it receives and emits the
 * result blob in that same order, so batch index i maps to the i-th sorted
 * position. We sort here identically before assigning indices.
 */
public final class WgpuBatchCoordinator {
    private static final Logger LOGGER = LoggerFactory.getLogger("c2me-wgpu-batch");

    /** Verbose per-window throughput logging, enabled with {@code C2ME_WGPU_DEBUG=1}. */
    private static final boolean DEBUG = "1".equals(System.getenv("C2ME_WGPU_DEBUG"));

    /** Max chunks coalesced into one GPU dispatch. */
    private static final int MAX_BATCH = 64;
    /** Max time (ms) to wait for more requests before flushing a short batch. */
    private static final long MAX_WAIT_MS = 5L;
    /** Abandon a GPU request after this long. */
    private static final long REQUEST_TIMEOUT_NS = 10_000_000_000L;

    /** One chunk's slice of a batch result. */
    public static final class GpuChunk {
        public final MemorySegment seg;
        public final int batchIndex;
        GpuChunk(MemorySegment seg, int batchIndex) {
            this.seg = seg;
            this.batchIndex = batchIndex;
        }
    }

    private static final class Job {
        final int cx, cz;
        final CompletableFuture<GpuChunk> future = new CompletableFuture<>();
        Job(int cx, int cz) { this.cx = cx; this.cz = cz; }
    }

    private final BlockingQueue<Job> queue = new LinkedBlockingQueue<>();
    private final long worldHandle;
    private volatile boolean running = true;

    // ── throughput stats (logged every STATS_INTERVAL_NS) ──────────
    private static final long STATS_INTERVAL_NS = 2_000_000_000L;
    private final AtomicLong statBatches = new AtomicLong();
    private final AtomicLong statChunks = new AtomicLong();
    private final AtomicLong statGpuNanos = new AtomicLong();
    private volatile long statsWindowStart = System.nanoTime();

    public WgpuBatchCoordinator(long worldHandle) {
        this.worldHandle = worldHandle;
        Thread t = new Thread(this::run, "c2me-wgpu-batch");
        t.setDaemon(true);
        t.start();
    }

    /** Submit one chunk and block until its density result is ready. */
    public GpuChunk request(int cx, int cz) throws Exception {
        Job job = new Job(cx, cz);
        queue.add(job);
        return job.future.get();
    }

    /** Stop the coordinator thread. */
    public void shutdown() {
        running = false;
    }

    private void run() {
        while (running) {
            try {
                Job first = queue.poll(50, TimeUnit.MILLISECONDS);
                if (first == null) continue;

                List<Job> batch = new ArrayList<>();
                batch.add(first);
                // Brief window for other workers to pile on. If the queue is
                // already non-empty, poll() returns immediately, so this only
                // actually waits when there is nothing more to take.
                long deadline = System.nanoTime() + MAX_WAIT_MS * 1_000_000L;
                while (batch.size() < MAX_BATCH) {
                    long remain = deadline - System.nanoTime();
                    if (remain <= 0) break;
                    Job j = queue.poll(remain, TimeUnit.NANOSECONDS);
                    if (j == null) break;
                    batch.add(j);
                }

                // Same ordering the Rust side applies (sort by cx, then cz).
                batch.sort((a, b) -> {
                    int c = Integer.compare(a.cx, b.cx);
                    return c != 0 ? c : Integer.compare(a.cz, b.cz);
                });

                int n = batch.size();
                int[] positions = new int[n * 2];
                for (int i = 0; i < n; i++) {
                    Job job = batch.get(i);
                    positions[i * 2] = job.cx;
                    positions[i * 2 + 1] = job.cz;
                }

                dispatch(batch, positions);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                return;
            } catch (Throwable t) {
                LOGGER.error("[c2me-wgpu] batch coordinator loop error", t);
            }
        }
    }

    private void dispatch(List<Job> batch, int[] positions) {
        long __t0 = System.nanoTime();
        long req;
        try {
            req = WgpuThinBinding.requestChunks(worldHandle, positions);
        } catch (Throwable t) {
            failAll(batch, t);
            return;
        }
        if (req == 0L) {
            failAll(batch, new RuntimeException("requestChunks returned 0"));
            return;
        }

        long deadline = System.nanoTime() + REQUEST_TIMEOUT_NS;
        int st;
        try {
            while ((st = WgpuThinBinding.pollStatus(req)) == WgpuThinBinding.STATUS_PENDING) {
                if (System.nanoTime() > deadline) {
                    failAll(batch, new RuntimeException("GPU request timed out"));
                    return;
                }
                java.util.concurrent.locks.LockSupport.parkNanos(100_000L);
            }
        } catch (Throwable t) {
            failAll(batch, t);
            return;
        }
        if (st != WgpuThinBinding.STATUS_DONE) {
            failAll(batch, new RuntimeException("GPU request status=" + st));
            return;
        }

        MemorySegment seg;
        try {
            seg = WgpuThinBinding.takeResultBlob(req);
        } catch (Throwable t) {
            failAll(batch, t);
            return;
        }

        // Hand the whole batch blob to each job; writeDensityToChunk slices
        // out the job's chunk via batchIndex.
        int n = batch.size();
        for (int i = 0; i < n; i++) {
            batch.get(i).future.complete(new GpuChunk(seg, i));
        }
        recordStats(n, System.nanoTime() - __t0);
    }

    private void recordStats(int chunks, long gpuNanos) {
        statBatches.incrementAndGet();
        statChunks.addAndGet(chunks);
        statGpuNanos.addAndGet(gpuNanos);
        if (!DEBUG) return;
        long now = System.nanoTime();
        long win = now - statsWindowStart;
        if (win < STATS_INTERVAL_NS) return;
        long b = statBatches.getAndSet(0);
        long c = statChunks.getAndSet(0);
        long g = statGpuNanos.getAndSet(0);
        statsWindowStart = now;
        if (b == 0) return;
        double sec = win / 1e9;
        double cps = c / sec;
        double avgBatchMs = (g / 1e6) / b;
        double avgChunkMs = (g / 1e6) / Math.max(1, c);
        LOGGER.info(String.format(
                "[c2me-wgpu] CPS=%.1f  batches=%d  chunks=%d  avgBatch=%.2fms  avgChunk=%.3fms  (window=%.2fs)",
                cps, b, c, avgBatchMs, avgChunkMs, sec));
    }

    private static void failAll(List<Job> batch, Throwable t) {
        for (Job job : batch) job.future.completeExceptionally(t);
    }
}
