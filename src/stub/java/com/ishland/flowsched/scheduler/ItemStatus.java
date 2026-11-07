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

// Compile-time stub for FlowSched's ItemStatus. NOT shipped in the jar
// (compileOnly source set); at runtime C2ME's bundled FlowSched class is used.
// Declares only the members this mod calls, so it does not pin a version.
package com.ishland.flowsched.scheduler;

import io.reactivex.rxjava3.core.Completable;

public interface ItemStatus<K, V, Ctx> {
    KeyStatusPair[] EMPTY_DEPENDENCIES = new KeyStatusPair[0];

    ItemStatus<K, V, Ctx>[] getAllStatuses();

    int ordinal();

    Completable upgradeToThis(Ctx ctx, Cancellable cancellable);

    Completable postUpgradeToThis(Ctx ctx);

    Completable preDowngradeFromThis(Ctx ctx, Cancellable cancellable);

    Completable downgradeFromThis(Ctx ctx, Cancellable cancellable);

    KeyStatusPair<K, V, Ctx>[] getDependencies(ItemHolder<K, V, Ctx, ?> holder);

    @SuppressWarnings("unchecked")
    default KeyStatusPair<K, V, Ctx>[] getDependenciesToRemove(ItemHolder<K, V, Ctx, ?> holder) {
        return EMPTY_DEPENDENCIES;
    }

    @SuppressWarnings("unchecked")
    default KeyStatusPair<K, V, Ctx>[] getDependenciesToAdd(ItemHolder<K, V, Ctx, ?> holder) {
        return EMPTY_DEPENDENCIES;
    }
}