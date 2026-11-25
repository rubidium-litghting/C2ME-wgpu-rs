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

// Compile-time stub for FlowSched's ItemHolder. NOT shipped in the jar
// (compileOnly source set); at runtime C2ME's bundled FlowSched class is used.
// Declares only the members this mod calls.
package com.ishland.flowsched.scheduler;

import java.util.concurrent.atomic.AtomicReference;

public class ItemHolder<K, V, Ctx, Ext> {
    public K getKey() {
        throw new UnsupportedOperationException("stub");
    }

    public AtomicReference<V> getItem() {
        throw new UnsupportedOperationException("stub");
    }

    /** C2ME stores the vanilla-side {@code AbstractChunkHolder} here; it is
     *  how the scheduler exposes a chunk to {@code BoundedRegionArray}. */
    public AtomicReference<Ext> getUserData() {
        throw new UnsupportedOperationException("stub");
    }

    public ItemStatus<K, V, Ctx> getStatus() {
        throw new UnsupportedOperationException("stub");
    }
}