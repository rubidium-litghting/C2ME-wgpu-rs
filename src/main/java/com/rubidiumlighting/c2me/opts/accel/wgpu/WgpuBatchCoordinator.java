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
import java.util.concurrent.atomic.AtomicInteger;
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

    /** Batches submitted to the GPU and not yet read back, summed over every
     *  dimension. Only {@code HeightProbe} reads it: it is how the probe tells
     *  whether a CPU-side height query ran while the GPU pipeline was busy.
     *  Documentation: DEVELOPMENT.md §14.5. */
    private static final AtomicInteger ACTIVE_REQUESTS = new AtomicInteger();

    /** @see #ACTIVE_REQUESTS */
    public static int activeRequests() {
        return ACTIVE_REQUESTS.get();
    }
    /** Max time (ms) to wait for more requests before flushing a short batch.
     *
     *  Kept at 5ms after bracketing the value from both sides: 15ms produced
     *  no larger batches (C2ME, not the gather window, is the throughput
     *  limit) and only added latency; 1ms did keep the pipeline fuller
     *  (inflight 1.3→1.7, peak CPS 325→347) but shrank batches from 5.0 to
     *  3.4 chunks, so the per-batch overhead ate the gain — total time was a
     *  wash (1m04 → 1m05). 5ms is the sweet spot. See DEVELOPMENT.md §11. */
    private static final long MAX_WAIT_MS = 5L;
    /** Poll backoff. The first few polls are cheap (the batch may already be
     *  done), then the interval doubles up to POLL_MAX_NS. A 30ms batch needs
     *  roughly 25 wakeups instead of 300 at a fixed 100µs, and each wakeup is
     *  a cross-FFM call. */
    private static final long POLL_INITIAL_NS = 50_000L;   // 50µs
    private static final long POLL_MAX_NS = 2_000_000L;    // 2ms

    /** Abandon a GPU request after this long. */
    private static final long REQUEST_TIMEOUT_NS = 30_000_000_000L;

    /** One chunk's slice of a batch result. */
    public static final class GpuChunk {
        public final MemorySegment seg;
        public final int batchIndex;
        /** §3.5 per-block state bytes (u32/block, chunk-major) for the whole
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

    /** Hard cap on queued requests. This queue used to be unbounded, which
     *  turned into a leak on large pregeneration runs: C2ME submits chunks far
     *  faster than the GPU can retire them (an End pregen of radius 11820 is
     *  ~2.2M chunks), so hundreds of thousands of {@code Job} objects piled up,
     *  memory climbed, GC pressure starved the completion thread, and the GPU
     *  ended up idle waiting for work — throughput collapsed from ~500 CPS to
     *  single digits.
     *
     *  A bounded queue applies backpressure to the submitting C2ME worker
     *  thread instead, which is the correct place to stall: the GPU stays busy
     *  and memory stays bounded. Sized well above MAX_BATCH so batches still
     *  form instantly. */
    private static final int QUEUE_CAPACITY = 4096;
    private final BlockingQueue<Job> queue = new LinkedBlockingQueue<>(QUEUE_CAPACITY);
    private final long worldHandle;
    /** Kept so shutdown() can join them. Null in serial (non-pipeline) mode,
     *  where run() does the completion inline and no second thread exists. */
    private final Thread batchThread;
    private final Thread completeThread;
    private volatile boolean running = true;
    /** Set by {@link #run} when its loop exits. Lets {@link #completeLoop}
     *  distinguish "inflight is momentarily empty" from "no more batches can
     *  arrive" — otherwise it could exit in the window between the submit
     *  thread polling a batch and enqueuing it, stranding that batch. */
    private volatile boolean runDone = false;

    /** Set once a batch comes back STATUS_ERROR. From then on this dimension
     *  stops using the GPU: the batch coordinator cannot tell an execution
     *  failure (a poisoned mutex, an oversized batch, a Rust-side panic in one
     *  dispatch) from a device-level fault, and continuing to submit would
     *  just repeat the failure. The mixin checks this before submitting and
     *  falls back to vanilla CPU generation.
     *
     *  <p>Unlike a timeout — which means the native worker is unresponsive and
     *  halts the JVM — this is a recoverable condition: the failed futures are
     *  completed exceptionally, C2ME releases the chunk locks, and the chunks
     *  are regenerated on the CPU. */
    private volatile boolean gpuFailed = false;

    public boolean hasFailed() {
        return gpuFailed;
    }

    /** Number of batches allowed in flight (submitted to Rust, not yet
     *  completed). Matches the Rust staging chain depth so the GPU side can
     *  stay busy while Java assembles the next batch. */
    private static final int STAGING = 2;
    /** Submitted-but-not-completed batches. Bounded => backpressure (the
     *  submit loop blocks here instead of overflowing anything). */
    private final BlockingQueue<Inflight> inflight = new LinkedBlockingQueue<>(STAGING);

    /**
     * Acquired before handing a batch to Rust, released once its result has
     * been taken. The `inflight` queue alone is not enough: it is filled
     * *after* `requestChunks` has already sent the batch to the native side,
     * so with STAGING=1 a second batch could still reach Rust while the first
     * was in flight. The executor can only hold one prepared batch until input
     * buffers are pooled, and a second one there fails with
     * "another batch is already in flight".
     */
    private final java.util.concurrent.Semaphore submitPermits =
            new java.util.concurrent.Semaphore(STAGING);
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
    /** Same as statGpuNanos but each batch's contribution is clipped to the
     *  current window. `inflight` is "average batches occupying the pipeline",
     *  which must not count time a batch spent in a previous window. */
    private final AtomicLong statInflightNanos = new AtomicLong();
    private volatile long statsWindowStart = System.nanoTime();
    /** Size in bytes of the last external payload uploaded to binding 6. This
     *  is the only dynamically sized binding, and the one that grows with
     *  external-node count, so it is the one worth watching against
     *  max_storage_buffer_binding_size. Reported in the 2-second stats line. */
    private volatile long lastExternalBytes;

    public WgpuBatchCoordinator(long worldHandle,
                                List<DensityFunction> externalNodes,
                                int gx, int gy, int gz, int cellW, int cellH, int minY) {
        this.worldHandle = worldHandle;
        this.externalNodes = externalNodes;
        this.gx = gx; this.gy = gy; this.gz = gz;
        this.cellW = cellW; this.cellH = cellH;
        this.minY = minY;
        // External inputs force serial submission. This is a correctness
        // requirement, not just a memory one: the payload travels through the
        // process-global `world.external_data`, and `submit_async` returns as
        // soon as the batch is queued — the executor reads that global later,
        // during prepare. With several batches in flight the next batch's
        // upload would overwrite the data a queued batch has not consumed yet.
        // Serial mode blocks until each batch completes, so the global can only
        // be rewritten after its contents are safely on the GPU.
        // Fixing this properly means carrying the payload per batch instead of
        // through shared mutable state; until then, keep modded worlds serial.
        this.pipeline = externalNodes.isEmpty();
        this.batchThread = new Thread(this::run, "c2me-wgpu-batch");
        this.batchThread.setDaemon(true);
        this.batchThread.start();
        if (pipeline) {
            this.completeThread = new Thread(this::completeLoop, "c2me-wgpu-complete");
            this.completeThread.setDaemon(true);
            this.completeThread.start();
        } else {
            this.completeThread = null;
        }
    }

    /** Submit one chunk and return a future completed when its result is
     *  ready. Does NOT block the caller — the C2ME-scheduling path (§3.5)
     *  uses this so the worker is freed to submit more chunks and batches can
     *  grow. This is the only request entry point. */
    public java.util.concurrent.CompletableFuture<GpuChunk> requestAsync(int cx, int cz) {
        Job job = new Job(cx, cz);
        // Bounded queue: block the calling C2ME worker (never the main thread)
        // until there is room, so the GPU is never starved by a huge backlog.
        // Poll instead of put() so teardown can interrupt the wait: put() on a
        // full queue is not wakeable by `running = false`.
        while (running) {
            try {
                if (queue.offer(job, 50, TimeUnit.MILLISECONDS)) {
                    return job.future;
                }
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        // Shutting down (or interrupted): fail the request so C2ME regenerates
        // this chunk on the CPU rather than waiting forever.
        job.future.completeExceptionally(
                new IllegalStateException("c2me-wgpu coordinator is shutting down"));
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
        // Wait for both threads to actually exit before returning. Without
        // this the caller has no idea when the in-flight Rust handles have
        // been consumed, and has to guess with a delay before destroying the
        // world — a guess that is both too long on fast machines and possibly
        // too short on slow ones.
        //
        // The batch thread exits as soon as it notices `running == false`; the
        // completion thread then drains whatever is still in `inflight` and
        // exits once both the queue is empty and runDone is set.
        joinQuietly(batchThread);
        joinQuietly(completeThread);
    }

    /** Join {@code t}, ignoring interrupts and refusing to join ourselves
     *  (which would deadlock). */
    private static void joinQuietly(Thread t) {
        if (t == null || t == Thread.currentThread()) {
            return;
        }
        try {
            t.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
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
        lastExternalBytes = (long) data.length * 4L; // f32
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
        ACTIVE_REQUESTS.incrementAndGet();
        return new Inflight(batch, req, System.nanoTime());
    }

    /** Pipeline mode: hand the submitted batch to the completion thread.
     *  Blocks on `inflight` when STAGING batches are already in flight
     *  (backpressure). */
    private void submit(List<Job> batch, int[] positions) throws InterruptedException {
        // Wait for a free slot before submitting, not after: see submitPermits.
        submitPermits.acquire();
        Inflight ib;
        try {
            ib = submitRequest(batch, positions);
        } catch (Throwable t) {
            submitPermits.release();
            throw t;
        }
        if (ib == null) {
            submitPermits.release();
            return;
        }
        try {
            inflight.put(ib);
        } catch (InterruptedException ie) {
            // The batch is dropped, so no completion will ever run for it.
            ACTIVE_REQUESTS.decrementAndGet();
            throw ie;
        }
    }

    /** Serial mode (external inputs present): submit then wait inline. */
    private void submitAndComplete(List<Job> batch, int[] positions) {
        long t0 = System.nanoTime();
        Inflight ib = submitRequest(batch, positions);
        if (ib != null) {
            completeTracked(ib, t0);
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
                completeTracked(ib, ib.submitT0);
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

    /** {@link #complete} plus the probe bookkeeping for
     *  {@link #ACTIVE_REQUESTS}. Every {@link Inflight} passes through here
     *  exactly once, which is what keeps the counter honest. */
    private void completeTracked(Inflight ib, long t0) {
        try {
            complete(ib, t0);
        } finally {
            ACTIVE_REQUESTS.decrementAndGet();
        }
    }

    /** Wait for one submitted batch, take its blob, complete the jobs. */
    private void complete(Inflight ib, long t0) {
        long req = ib.req;
        long deadline = System.nanoTime() + REQUEST_TIMEOUT_NS;
        int st;
        try {
            long backoff = POLL_INITIAL_NS;
            while ((st = WgpuThinBinding.pollStatus(req)) == WgpuThinBinding.STATUS_PENDING) {
                if (System.nanoTime() > deadline) {
                    LOGGER.error("[c2me-wgpu] FATAL: GPU request timed out after 30s. The native worker is likely deadlocked. Terminating JVM.");
                    Runtime.getRuntime().halt(1);
                    return;
                }
                java.util.concurrent.locks.LockSupport.parkNanos(backoff);
                if (backoff < POLL_MAX_NS) {
                    backoff = Math.min(backoff * 2, POLL_MAX_NS);
                }
            }
        } catch (Throwable t) {
            LOGGER.error("[c2me-wgpu] FATAL: Exception while polling GPU status.", t);
            Runtime.getRuntime().halt(1);
            return;
        }
        if (st == WgpuThinBinding.STATUS_ERROR) {
            // Recoverable: mark this dimension as GPU-incapable and fail the
            // batch so C2ME releases the chunk locks. Later chunks go through
            // vanilla CPU generation.
            LOGGER.error("[c2me-wgpu] GPU batch failed (status={}); this dimension falls back to CPU", st);
            gpuFailed = true;
            // The Rust side consumed the handle regardless of status, so this
            // releases it. Its return value is meaningless here.
            try { WgpuThinBinding.takeResultBlob(req); } catch (Throwable ignored) {}
            failAll(ib.batch, new RuntimeException("GPU batch failed (STATUS_ERROR)"));
            submitPermits.release();
            return;
        }
        if (st != WgpuThinBinding.STATUS_DONE) {
            LOGGER.error("[c2me-wgpu] FATAL: GPU request returned unexpected status=" + st + ". Terminating JVM to prevent deadlock.");
            try { org.apache.logging.log4j.LogManager.shutdown(); } catch (Throwable ignored) {}
            Runtime.getRuntime().halt(1);
            return;
        }

        // §3.5: take the BLOCK blob FIRST — it borrows the handle, and
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
            submitPermits.release();
            return;
        }

        // Release as soon as the Rust-side handle is consumed: from here on
        // the permit no longer guards native resources, only the Java-side
        // fan-out below, so the submit chain can start the next batch while
        // this one is still distributing results.
        submitPermits.release();
        int n = ib.batch.size();
        for (int i = 0; i < n; i++) {
            ib.batch.get(i).future.complete(new GpuChunk(seg, i, blockSeg));
        }
        recordStats(n, t0, System.nanoTime());
    }

    /**
     * Runs on several C2ME worker threads at once. The window bookkeeping
     * (statsWindowStart plus the counters) is a multi-field update: without a
     * lock two threads can both decide the window has elapsed, and the second
     * one computes `now - statsWindowStart` against a value the first has
     * already moved, producing an inflight ratio above the real queue depth.
     * Called once per batch (~100ms apart), so contention is irrelevant.
     */
    private synchronized void recordStats(int chunks, long submitT0, long now) {
        long gpuNanos = now - submitT0;
        statBatches.incrementAndGet();
        statChunks.addAndGet(chunks);
        statGpuNanos.addAndGet(gpuNanos);
        // Clip to the current window. A batch submitted before this window
        // began still occupies the pipeline now, so only the part of its
        // lifetime inside the window counts — otherwise a batch straddling the
        // boundary is counted in full in both windows and inflight reads high.
        long clippedStart = Math.max(submitT0, statsWindowStart);
        statInflightNanos.addAndGet(Math.max(0, now - clippedStart));
        if (!Config.get().debugVerboseLogging) return;
        long win = now - statsWindowStart;
        if (win < STATS_INTERVAL_NS) return;
        long b = statBatches.getAndSet(0);
        long c = statChunks.getAndSet(0);
        long g = statGpuNanos.getAndSet(0);
        long gi = statInflightNanos.getAndSet(0);
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
        double inflight = (double) gi / (win > 0 ? win : 1);
        // In-flight spans three places: batches sitting in the queue, the one
        // completeLoop is draining, and the one submit() has handed to Rust but
        // not yet put() into the queue. Hence STAGING + 2, not + 1.
        final int maxInflight = STAGING + 2;
        // Binding 6 is the only one that grows with external-node count. The
        // rest (output/block/staging/input sets) are fixed at executor init and
        // logged once there. Showing it here makes a creeping external payload
        // visible before it hits max_storage_buffer_binding_size.
        double extMiB = lastExternalBytes / (1024.0 * 1024.0);
        String line = String.format(
                "CPS=%.1f  batches=%d  chunks=%d  batchMs=%.2f  chunkMs=%.2f  "
                + "inflight=%.2f/%d  ext=%.2fMiB  (window=%.2fs)",
                cps, b, c, avgBatchMs, avgChunkMs, inflight, maxInflight, extMiB, sec);
        LOGGER.info(line);
        // Mirror to the chat HUD when the client can show it; the queue keeps
        // only the newest line, so headless/paused sessions do not build up.
        WgpuFallbackNotice.reportStats(line);
    }

    private static void failAll(List<Job> batch, Throwable t) {
        for (Job job : batch) job.future.completeExceptionally(t);
    }
}
