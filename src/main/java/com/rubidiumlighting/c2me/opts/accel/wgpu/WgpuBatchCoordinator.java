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

import com.rubidiumlighting.c2me.opts.accel.wgpu.config.Config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.minecraft.world.gen.densityfunction.DensityFunction;

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


    /** Max chunks coalesced into one GPU dispatch. */
    private static final int MAX_BATCH = 64;
    /** Max time (ms) to wait for more requests before flushing a short batch. */
    private static final long MAX_WAIT_MS = 5L;
    /** Abandon a GPU request after this long. */
    private static final long REQUEST_TIMEOUT_NS = 30_000_000_000L;

    /** One chunk's slice of a batch result. */
    public static final class GpuChunk {
        public final MemorySegment seg;
        public final int batchIndex;
        /** §20 per-block state bytes (u32/block, chunk-major) for the whole
         *  batch. BlockOut is the only write-back path, so this is non-null
         *  whenever the request succeeded; the caller slices its chunk via
         *  batchIndex. */
        public final MemorySegment blockSeg;
        GpuChunk(MemorySegment seg, int batchIndex, MemorySegment blockSeg) {
            this.seg = seg;
            this.batchIndex = batchIndex;
            this.blockSeg = blockSeg;
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
    /** Set by {@link #run} when its loop exits. Lets {@link #completeLoop}
     *  distinguish "inflight is momentarily empty" from "no more batches can
     *  arrive" — otherwise it could exit in the window between the submit
     *  thread polling a batch and enqueuing it, stranding that batch. */
    private volatile boolean runDone = false;

    /** Number of batches allowed in flight (submitted to Rust, not yet
     *  completed). Matches the Rust staging chain depth so the GPU side can
     *  stay busy while Java assembles the next batch. */
    private static final int STAGING = 3;
    /** Submitted-but-not-completed batches. Bounded => backpressure (the
     *  submit loop blocks here instead of overflowing anything). */
    private final BlockingQueue<Inflight> inflight = new LinkedBlockingQueue<>(STAGING);
    /** Pipelining is safe only when there are no CPU-evaluated external
     *  inputs, because those share one Rust buffer (binding 6) that a later
     *  batch would overwrite. With externals we fall back to serial. */
    private final boolean pipeline;

    private static final class Inflight {
        final List<Job> batch;
        final long req;
        /** When this batch was submitted to Rust. Used to measure how long the
         *  batch occupied the pipeline; taking the timestamp at poll() time
         *  instead would only measure the drain, which is near zero once the
         *  GPU has finished. */
        final long submitT0;
        Inflight(List<Job> batch, long req, long submitT0) {
            this.batch = batch;
            this.req = req;
            this.submitT0 = submitT0;
        }
    }

    // ── external (CPU-evaluated) density subtrees ──────────────────
    /** Nodes the GPU serializer could not translate. Their values are
     *  computed here on the CPU at the corner grid and uploaded to binding 6. */
    private final List<DensityFunction> externalNodes;
    private final int gx, gy, gz;      // corner counts per axis
    private final int cellW, cellH;    // cell size in blocks
    private final int minY;

    // ── throughput stats (logged every STATS_INTERVAL_NS) ──────────
    private static final long STATS_INTERVAL_NS = 2_000_000_000L;
    private final AtomicLong statBatches = new AtomicLong();
    private final AtomicLong statChunks = new AtomicLong();
    private final AtomicLong statGpuNanos = new AtomicLong();
    private volatile long statsWindowStart = System.nanoTime();

    public WgpuBatchCoordinator(long worldHandle,
                                List<DensityFunction> externalNodes,
                                int gx, int gy, int gz, int cellW, int cellH, int minY) {
        this.worldHandle = worldHandle;
        this.externalNodes = externalNodes;
        this.gx = gx; this.gy = gy; this.gz = gz;
        this.cellW = cellW; this.cellH = cellH;
        this.minY = minY;
        this.pipeline = externalNodes.isEmpty();
        Thread t = new Thread(this::run, "c2me-wgpu-batch");
        t.setDaemon(true);
        t.start();
        if (pipeline) {
            Thread c = new Thread(this::completeLoop, "c2me-wgpu-complete");
            c.setDaemon(true);
            c.start();
        }
    }

    /** Submit one chunk and return a future completed when its result is
     *  ready. Does NOT block the caller — the C2ME-scheduling path (§18.6b)
     *  uses this so the worker is freed to submit more chunks and batches can
     *  grow. This is the only request entry point. */
    public java.util.concurrent.CompletableFuture<GpuChunk> requestAsync(int cx, int cz) {
        Job job = new Job(cx, cz);
        queue.add(job);
        return job.future;
    }

    /** Stop the coordinator and fail every request that has not been sent
     *  to Rust yet.
     *
     *  <p>Callers must not submit new chunks after this returns.
     *
     *  <p>Queued jobs hold no Rust-side handle, so their futures can be
     *  completed exceptionally right away — that is what lets C2ME release
     *  the chunk position locks (SchedulingManager releases locks from the
     *  future's whenComplete; a future that never completes holds the lock
     *  forever).
     *
     *  <p>In-flight batches are deliberately NOT failed here: each holds a
     *  Rust-side request handle that only {@code takeResultBlob} can consume.
     *  {@link #completeLoop} keeps draining until {@code inflight} is empty,
     *  so those handles are always released.
     */
    public void shutdown() {
        if (!running) {
            return;
        }
        running = false;
        List<Job> queued = new ArrayList<>();
        queue.drainTo(queued);
        if (!queued.isEmpty()) {
            failAll(queued, new RuntimeException("WgpuBatchCoordinator shut down"));
        }
    }

    private void run() {
        try {
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

                if (pipeline) {
                    submit(batch, positions);          // enqueue to inflight
                } else {
                    submitAndComplete(batch, positions); // serial (external inputs)
                }
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                return;
            } catch (Throwable t) {
                LOGGER.error("[c2me-wgpu] batch coordinator loop error", t);
            }
        }
        } finally {
            runDone = true;
        }
    }

    /** Evaluate every external subtree at each chunk's corner grid, in the
     *  exact [chunk][ext][cell] order the WGSL external_inputs[] index uses. */
    private void uploadExternalInputs(List<Job> batch) {
        int numExt = externalNodes.size();
        if (numExt == 0) {
            // No external subtrees: binding 6 is never read, skip the FFM call.
            return;
        }
        int cellsPerChunk = gx * gy * gz;
        float[] data = new float[batch.size() * numExt * cellsPerChunk];
        int idx = 0;
        for (Job job : batch) {
            int sx = job.cx * 16;
            int sz = job.cz * 16;
            for (DensityFunction ext : externalNodes) {
                for (int j = 0; j < gy; j++) {
                    int wy = minY + j * cellH;
                    for (int k = 0; k < gz; k++) {
                        int wz = sz + k * cellW;
                        for (int i = 0; i < gx; i++) {
                            int wx = sx + i * cellW;
                            double v = ext.sample(
                                    new DensityFunction.UnblendedNoisePos(wx, wy, wz));
                            data[idx++] = (float) v;
                        }
                    }
                }
            }
        }
        WgpuThinBinding.setExternalInputs(worldHandle, numExt, data);
    }

    /** Upload externals (no-op when none) and submit the batch to Rust.
     *  Returns the in-flight record, or null if the request could not be
     *  issued — in that case the batch's futures have already been failed. */
    private Inflight submitRequest(List<Job> batch, int[] positions) {
        uploadExternalInputs(batch);
        long req;
        try {
            req = WgpuThinBinding.requestChunks(worldHandle, positions);
        } catch (Throwable t) {
            failAll(batch, t);
            return null;
        }
        if (req == 0L) {
            failAll(batch, new RuntimeException("requestChunks returned 0"));
            return null;
        }
        return new Inflight(batch, req, System.nanoTime());
    }

    /** Pipeline mode: hand the submitted batch to the completion thread.
     *  Blocks on `inflight` when STAGING batches are already in flight
     *  (backpressure). */
    private void submit(List<Job> batch, int[] positions) throws InterruptedException {
        Inflight ib = submitRequest(batch, positions);
        if (ib != null) {
            inflight.put(ib);
        }
    }

    /** Serial mode (external inputs present): submit then wait inline. */
    private void submitAndComplete(List<Job> batch, int[] positions) {
        long t0 = System.nanoTime();
        Inflight ib = submitRequest(batch, positions);
        if (ib != null) {
            complete(ib, t0);
        }
    }

    private void completeLoop() {
        // Keep draining after `running` clears: each Inflight holds a Rust-side
        // request handle that only take_*_blob releases. Exiting early would
        // leak it.
        while (running || !runDone || !inflight.isEmpty()) {
            try {
                Inflight ib = inflight.poll(50, TimeUnit.MILLISECONDS);
                if (ib == null) continue;
                complete(ib, ib.submitT0);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                return;
            } catch (Throwable t) {
                LOGGER.error("[c2me-wgpu] FATAL: completion loop error. Main thread is likely deadlocked. Forcing JVM halt.", t);
                // Flush Log4j buffers before killing the process
                try {
                    org.apache.logging.log4j.LogManager.shutdown();
                } catch (Throwable ignored) {}
                
                // System.exit can deadlock if shutdown hooks wait on the deadlocked main thread.
                // halt(1) is the only safe way to kill the process.
                Runtime.getRuntime().halt(1); 
            }
        }
    }

    /** Wait for one submitted batch, take its blob, complete the jobs. */
    private void complete(Inflight ib, long t0) {
        long req = ib.req;
        long deadline = System.nanoTime() + REQUEST_TIMEOUT_NS;
        int st;
        try {
            while ((st = WgpuThinBinding.pollStatus(req)) == WgpuThinBinding.STATUS_PENDING) {
                if (System.nanoTime() > deadline) {
                    LOGGER.error("[c2me-wgpu] FATAL: GPU request timed out after 30s. The native worker is likely deadlocked. Terminating JVM.");
                    Runtime.getRuntime().halt(1);
                    return;
                }
                java.util.concurrent.locks.LockSupport.parkNanos(100_000L);
            }
        } catch (Throwable t) {
            LOGGER.error("[c2me-wgpu] FATAL: Exception while polling GPU status.", t);
            Runtime.getRuntime().halt(1);
            return;
        }
        if (st != WgpuThinBinding.STATUS_DONE) {
            LOGGER.error("[c2me-wgpu] FATAL: GPU request returned status=" + st + ". Terminating JVM to prevent deadlock.");
            try { org.apache.logging.log4j.LogManager.shutdown(); } catch (Throwable ignored) {}
            Runtime.getRuntime().halt(1);
            return;
        }

        // §20: take the BLOCK blob FIRST — it borrows the handle, and
        // takeResultBlob below frees it. (Reversed order = use-after-free.)
        MemorySegment blockSeg = null;
        try {
            blockSeg = WgpuThinBinding.takeBlockBlob(req);
            if (blockSeg != null && blockSeg.byteSize() == 0) blockSeg = null;
        } catch (Throwable t) {
            blockSeg = null;
        }

        MemorySegment seg;
        try {
            seg = WgpuThinBinding.takeResultBlob(req);
        } catch (Throwable t) {
            failAll(ib.batch, t);
            return;
        }

        int n = ib.batch.size();
        for (int i = 0; i < n; i++) {
            ib.batch.get(i).future.complete(new GpuChunk(seg, i, blockSeg));
        }
        recordStats(n, System.nanoTime() - t0);
    }

    private void recordStats(int chunks, long gpuNanos) {
        statBatches.incrementAndGet();
        statChunks.addAndGet(chunks);
        statGpuNanos.addAndGet(gpuNanos);
        if (!Config.get().debugVerboseLogging) return;
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
        // Two different questions, deliberately reported separately:
        //   batchMs — how long one batch took from submit to blob-taken. With
        //     pipelining this includes time spent queued behind other batches,
        //     so it is a LATENCY, not a compute cost.
        //   chunkMs — wall time per chunk, i.e. 1/cps scaled. This is the
        //     number that compares across versions; the old g/b/c form inflated
        //     it by roughly the pipeline depth.
        double avgBatchMs = (g / 1e6) / b;
        double avgChunkMs = c > 0 ? win / 1e6 / c : 0.0;
        // `g` sums each batch's submit-to-readback lifetime, so g/win is the
        // average number of batches occupying the pipeline at once. Compare it
        // to STAGING: near STAGING means the coordinator always had work
        // queued (GPU-bound); well below means it was starved waiting for
        // chunks (CPU/C2ME-bound). Note the cast — `g / win` on two longs is
        // integer division and truncates to 0, 1, 2, ...
        double inflight = (double) g / (win > 0 ? win : 1);
        // A batch counts as in flight from submit until its blob is taken, so
        // at most STAGING are queued plus the one completeLoop is draining.
        final int maxInflight = STAGING + 1;
        String line = String.format(
                "CPS=%.1f  batches=%d  chunks=%d  batchMs=%.2f  chunkMs=%.2f  "
                + "inflight=%.2f/%d  (window=%.2fs)",
                cps, b, c, avgBatchMs, avgChunkMs, inflight, maxInflight, sec);
        LOGGER.info(line);
        // Mirror to the chat HUD when the client can show it; the queue keeps
        // only the newest line, so headless/paused sessions do not build up.
        WgpuFallbackNotice.reportStats(line);
    }

    private static void failAll(List<Job> batch, Throwable t) {
        for (Job job : batch) job.future.completeExceptionally(t);
    }
}
