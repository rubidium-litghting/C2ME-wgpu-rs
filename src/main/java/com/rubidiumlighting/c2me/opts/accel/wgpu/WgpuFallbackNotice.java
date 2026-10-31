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

import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Thread-safe bridge for reporting GPU → CPU fallbacks to the player.
 *
 * <p>World init runs on a C2ME worker thread, far from the client thread, so
 * it cannot touch the chat HUD directly. Instead it enqueues a message here
 * (de-duplicated by a stable key); the client mixin drains the queue on each
 * tick and prints whatever is pending.
 */
public final class WgpuFallbackNotice {

    private WgpuFallbackNotice() {}

    /** A pending notice as a translation key plus its arguments.
     *
     *  <p>Translation cannot happen here: notices are enqueued from C2ME
     *  worker threads, and resolving a key depends on the client's current
     *  language, which is a client-thread concern. The mixin builds the
     *  {@code Text} when it drains the queue. */
    public record Notice(String key, Object[] args, net.minecraft.util.Formatting format) {}

    private static final Queue<Notice> PENDING = new ConcurrentLinkedQueue<>();
    private static final Set<String> SEEN = ConcurrentHashMap.newKeySet();

    /** Enqueue a notice once per distinct {@code dedupKey}, in yellow. */
    public static void report(String dedupKey, String translationKey, Object... args) {
        report(dedupKey, net.minecraft.util.Formatting.YELLOW, translationKey, args);
    }

    /** Enqueue a notice once per distinct {@code dedupKey}, in {@code format}.
     *
     *  <p>Colour carries meaning here: yellow means "degraded / fell back",
     *  green means "working as intended". A success notice rendered in yellow
     *  reads like a warning, so callers reporting success pass
     *  {@link net.minecraft.util.Formatting#GREEN}. */
    public static void report(String dedupKey, net.minecraft.util.Formatting format,
                              String translationKey, Object... args) {
        if (SEEN.add(dedupKey)) {
            PENDING.add(new Notice(translationKey, args, format));
        }
    }

    /**
     * Forget a de-duplication key so its notice can be emitted again.
     *
     * <p>Needed for "GPU acceleration is active": that claim is only true for
     * the dimension currently being generated, so it must be re-announced
     * after a save switch rather than silently suppressed by the key left
     * over from the previous world.
     */
    public static void clearDedup(String dedupKey) {
        SEEN.remove(dedupKey);
    }

    /** Return the next pending notice, or {@code null} if none. */
    public static Notice poll() {
        return PENDING.poll();
    }

    /**
     * Latest periodic CPS line. Kept separate from {@link #PENDING} because
     * throughput stats repeat: de-duplicating them would swallow every sample
     * after the first. Only the newest one is retained, so a session with no
     * chat HUD (or a paused client) never accumulates a backlog.
     */
    private static final java.util.concurrent.atomic.AtomicReference<String> LATEST_STATS =
            new java.util.concurrent.atomic.AtomicReference<>();

    /** Publish the newest stats line for the client to print (if it can). */
    public static void reportStats(String message) {
        LATEST_STATS.set(message);
    }

    /** Return and clear the newest stats line, or {@code null} if none. */
    public static String pollStats() {
        return LATEST_STATS.getAndSet(null);
    }
}
