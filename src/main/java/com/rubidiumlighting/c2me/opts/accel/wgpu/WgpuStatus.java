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

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * Read-only view of what the GPU backend is currently doing: which adapter was
 * selected and how fast it is generating.
 *
 * <p>The adapter list comes from {@link WgpuThinBinding#listGpus()}, which
 * marks the effective adapter. Cached because it changes only when the
 * instance is rebuilt.
 */
public final class WgpuStatus {

    private WgpuStatus() {}

    private static volatile String cachedName;
    private static volatile boolean nameResolved;

    /**
     * Name of the adapter in use, or a placeholder when it cannot be
     * determined (native unavailable, or no adapter selected yet).
     */
    public static String gpuName() {
        if (!nameResolved) {
            synchronized (WgpuStatus.class) {
                if (!nameResolved) {
                    cachedName = resolveName();
                    nameResolved = true;
                }
            }
        }
        return cachedName;
    }

    /** Drop the cached name, so the next query reflects a rebuilt instance. */
    public static void invalidate() {
        synchronized (WgpuStatus.class) {
            nameResolved = false;
            cachedName = null;
        }
    }

    private static String resolveName() {
        String json = WgpuThinBinding.listGpus();
        if (json == null) {
            return "unknown";
        }
        try {
            JsonArray arr = JsonParser.parseString(json).getAsJsonArray();
            // Prefer the adapter Rust marked as selected; fall back to the
            // first entry, which is what automatic selection picks.
            for (JsonElement el : arr) {
                JsonObject o = el.getAsJsonObject();
                if (o.has("selected") && o.get("selected").getAsBoolean()) {
                    return o.get("name").getAsString();
                }
            }
            if (!arr.isEmpty()) {
                return arr.get(0).getAsJsonObject().get("name").getAsString();
            }
        } catch (Throwable ignored) {
            // Malformed JSON here would only cost us the name in the overlay.
        }
        return "unknown";
    }
}
