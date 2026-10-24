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

package com.rubidiumlighting.c2me.opts.accel.wgpu.config;

import java.util.ArrayList;
import java.util.List;

/**
 * The graphics backends the user may pick from.
 *
 * <p>Deliberately static rather than queried from wgpu. Listing backends used
 * to be a C API call, which had to create a temporary
 * {@code wgpu::Instance} over {@code Backends::all()}. For GL that is not a
 * read-only operation: {@code wgpu-hal}'s {@code enumerate_adapters} calls
 * {@code eglMakeCurrent} and then invokes GL entry points, so it builds a real
 * GL context inside the game process. When that fails, wgpu-hal panics, and on
 * a Mali-G610 the driver hang took down the whole OS; on Mesa it left the
 * driver in a state that wedged Minecraft's own render thread.
 *
 * <p>The set of backends a platform can offer is a compile-time property of
 * wgpu, not something that needs probing, so it is hardcoded here. Nothing in
 * this class touches a driver.
 */
public final class Backends {

    private Backends() {}

    /** One selectable backend. {@code bits} is the {@code wgpu::Backends} value. */
    public record Entry(int bits, String label) {}

    /**
     * Backends offered on this platform, excluding "automatic".
     *
     * <p>Follows wgpu's own platform support table rather than what has been
     * tested here:
     * <ul>
     *   <li>Vulkan: Windows, Linux and Android. macOS/iOS would need
     *       MoltenVK, which the default build does not enable.</li>
     *   <li>Metal: macOS/iOS.</li>
     *   <li>DX12: Windows.</li>
     *   <li>OpenGL: best effort everywhere; macOS needs ANGLE, which the
     *       default build does not enable.</li>
     * </ul>
     *
     * <p>GL is listed on Linux and Android even though it has been seen
     * panicking inside EGL context creation on this project's own hardware.
     * That is a driver-level failure: refusing the option outright would hide
     * a working configuration on other drivers, and the rebuild path already
     * handles failure by keeping the previous device and telling the user.
     */
    public static List<Entry> available() {
        List<Entry> out = new ArrayList<>();
        String os = System.getProperty("os.name", "").toLowerCase();

        if (os.contains("win")) {
            out.add(new Entry(8, "DX12"));
            out.add(new Entry(2, "Vulkan"));
            out.add(new Entry(16, "GL"));
        } else if (os.contains("mac") || os.contains("darwin")) {
            // No Vulkan (needs MoltenVK) and no GL (needs ANGLE): neither is
            // built by default, so listing them would only fail on rebuild.
            out.add(new Entry(4, "Metal"));
        } else {
            // Linux, and Android which reports os.name as "Linux".
            out.add(new Entry(2, "Vulkan"));
            out.add(new Entry(16, "GL"));
        }

        return out;
    }
}
