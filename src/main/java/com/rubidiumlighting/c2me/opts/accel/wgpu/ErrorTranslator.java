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

import net.minecraft.text.Text;

/**
 * Translates low-level Rust/wgpu error strings into user-facing translation
 * keys.
 *
 * <p>The native side reports conditions in engineering terms
 * ({@code Features::PIPELINE_CACHE not present}), which are actionable for a
 * developer but meaningless to a player. This maps the known ones onto
 * translation keys; anything unrecognised is passed through unchanged so no
 * diagnostic information is lost.
 */
public final class ErrorTranslator {

    private ErrorTranslator() {}

    /**
     * @param detail the raw error string from the native layer, may be null
     * @return a translation key, or null when the error is not recognised
     */
    public static String getErrorKey(String detail) {
        if (detail == null) return null;
        if (detail.contains("PIPELINE_CACHE")) {
            return "c2me-opts-accel-wgpu.error.pipeline_cache";
        }
        if (detail.contains("max_storage_buffer_binding_size")) {
            return "c2me-opts-accel-wgpu.error.gpu_too_weak";
        }
        // Rust reports both shader-module and compute-pipeline validation
        // failures as "… for pass X failed validation: <driver text>". They
        // are the same class of problem for the player: this GPU/driver
        // rejected the generated shader.
        if (detail.contains("failed validation")) {
            return "c2me-opts-accel-wgpu.error.shader_validation";
        }
        if (detail.contains("panicked")) {
            return "c2me-opts-accel-wgpu.error.driver_panic";
        }
        if (detail.contains("no suitable wgpu adapter") || detail.contains("No suitable adapter")) {
            return "c2me-opts-accel-wgpu.error.no_adapter";
        }
        if (detail.contains("native library symbol mismatch")) {
            return "c2me-opts-accel-wgpu.error.symbol_mismatch";
        }
        return null;
    }

    /**
     * Wrap {@code detail} in a translatable component when it is a recognised
     * error, or return the raw string otherwise. Suitable for passing as an
     * argument to {@code Text.translatable(...)}.
     */
    public static Object localize(String detail) {
        String key = getErrorKey(detail);
        return key != null ? Text.translatable(key) : detail;
    }
}