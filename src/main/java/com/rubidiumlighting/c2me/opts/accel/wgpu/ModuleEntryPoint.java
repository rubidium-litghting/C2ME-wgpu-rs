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

//import com.ishland.c2me.base.common.config.ConfigSystem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ModuleEntryPoint {

    private static final Logger LOGGER = LoggerFactory.getLogger("c2me-wgpu");

    private static final boolean enabled;

    static {
        /*boolean configured = new ConfigSystem.ConfigAccessor()
                .key("worldGenOptimizations.wgpuAcceleration.enabled")
                .comment("""
                        Enable GPU-accelerated world generation using wgpu.
                        If the GPU is unavailable or initialization fails, this will automatically
                        fall back to the vanilla CPU path with a warning log.
                        """)
                .getBoolean(true, false);
        */
        boolean actuallyEnabled = false;
            try {
                actuallyEnabled = NativeLoader.isAvailable();
                if (actuallyEnabled) {
                    String gpuName = WgpuThinBinding.getGpuName();
                    LOGGER.info("[c2me-wgpu] wgpu acceleration enabled, GPU: {}", gpuName);
                } else {
                    LOGGER.warn("[c2me-wgpu] Native library not available for this platform, falling back to CPU");
                }
            } catch (IllegalCallerException e) {
                LOGGER.warn("""
                                
                                [c2me-wgpu] Panama FFM native access is not enabled.
                                Please add the following JVM argument:
                                  --enable-native-access=com.rubidiumlighting.c2me.opts.accel.wgpu
                                Falling back to CPU world generation.
                                """);
                actuallyEnabled = false;
            } catch (Throwable t) {
                LOGGER.warn("[c2me-wgpu] Failed to initialize wgpu acceleration, falling back to CPU", t);
                actuallyEnabled = false;
            }
        enabled = actuallyEnabled;
    }

    public static boolean isEnabled() {
        return enabled;
    }

}
