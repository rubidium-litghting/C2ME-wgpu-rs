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

// Compile-time stub for C2ME's chunk-system NewChunkStatus. NOT shipped in
// the jar (compileOnly source set); at runtime C2ME's real class is used.
// Declares only the members this mod calls, so it does not pin a C2ME version.
package com.ishland.c2me.rewrites.chunksystem.common;

import com.ishland.flowsched.scheduler.ItemStatus;
import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.chunk.ChunkStatus;

public abstract class NewChunkStatus
        implements ItemStatus<ChunkPos, ChunkState, ChunkLoadingContext> {

    /** Populated by C2ME's <clinit> after every status (vanilla + hooks) has
     *  been registered. Our own subclass reads it from {@code getAllStatuses}
     *  once the scheduler starts, so it is always populated by then. */
    public static NewChunkStatus[] ALL_STATUSES;

    protected NewChunkStatus(int ordinal, ChunkStatus effectiveVanillaStatus) {
        throw new UnsupportedOperationException("stub");
    }

    public static NewChunkStatus fromVanillaStatus(ChunkStatus status) {
        throw new UnsupportedOperationException("stub");
    }

    public ChunkStatus getEffectiveVanillaStatus() {
        throw new UnsupportedOperationException("stub");
    }

    // Implemented by C2ME's real class (it stores the ordinal passed to the
    // constructor and returns the process-wide ALL_STATUSES). Declared here
    // with concrete (throwing) bodies so subclass stubs do not need to
    // override them just to satisfy the interface.
    @Override
    public int ordinal() {
        throw new UnsupportedOperationException("stub");
    }

    @Override
    @SuppressWarnings("unchecked")
    public ItemStatus<ChunkPos, ChunkState, ChunkLoadingContext>[] getAllStatuses() {
        throw new UnsupportedOperationException("stub");
    }
}