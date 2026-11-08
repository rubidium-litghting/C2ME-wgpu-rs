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

// Compile-time stub, NOT shipped in the jar (see build.gradle sourceSets).

package com.ishland.c2me.rewrites.chunksystem.common;

import com.ishland.c2me.base.common.scheduler.SchedulingManager;
import com.ishland.flowsched.scheduler.ItemHolder;
import com.ishland.flowsched.scheduler.KeyStatusPair;
import net.minecraft.server.world.ServerChunkLoadingManager;
import net.minecraft.util.math.ChunkPos;

public final class ChunkLoadingContext {
    /** Mirrors the record's canonical constructor so a batch anchor can build
     *  a context for a member chunk. The descriptor is erased, so this links
     *  to the real constructor at runtime. */
    public ChunkLoadingContext(
            ItemHolder<ChunkPos, ChunkState, ChunkLoadingContext, ?> holder,
            ServerChunkLoadingManager tacs,
            SchedulingManager schedulingManager,
            TheChunkSystem theChunkSystem,
            KeyStatusPair<ChunkPos, ChunkState, ChunkLoadingContext>[] dependencies) {
        throw new UnsupportedOperationException("stub");
    }
    public ItemHolder<ChunkPos, ChunkState, ChunkLoadingContext, ?> holder() {
        throw new UnsupportedOperationException("stub");
    }
    public ServerChunkLoadingManager tacs() {
        throw new UnsupportedOperationException("stub");
    }
    public SchedulingManager schedulingManager() {
        throw new UnsupportedOperationException("stub");
    }
    public TheChunkSystem theChunkSystem() {
        throw new UnsupportedOperationException("stub");
    }
    public KeyStatusPair<ChunkPos, ChunkState, ChunkLoadingContext>[] dependencies() {
        throw new UnsupportedOperationException("stub");
    }
}
