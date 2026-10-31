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

    /** GPU acceleration status, consumed by the client notifier. */
    public enum Status {
        OK,                 // GPU acceleration active
        NATIVE_UNAVAILABLE, // native library missing for this platform
        FFM_DISABLED,       // --enable-native-access not set
        INIT_FAILED         // wgpu instance / adapter creation failed
    }

    private static final Status status;
    private static final String detail;

    static {
        Status st;
        String dt;
        try {
            if (!NativeLoader.isAvailable()) {
                st = Status.NATIVE_UNAVAILABLE;
                dt = "native library not available for this platform";
                LOGGER.warn("[c2me-wgpu] {}, falling back to CPU", dt);
            } else if (!WgpuThinBinding.isAvailable()) {
                st = Status.INIT_FAILED;
                String linkErr = WgpuThinBinding.getLinkError();
                if (linkErr != null) {
                    // The native library loaded but a downcall symbol is
                    // missing — that is a packaging/version mismatch, NOT a
                    // GPU problem. Say so, otherwise this looks like a
                    // driver issue and sends debugging in the wrong direction.
                    dt = "native library symbol mismatch (" + linkErr + ")";
                } else {
                    String instErr = WgpuThinBinding.getLastInstanceError();
                    dt = (instErr != null) ? instErr : "no suitable wgpu adapter (driver missing or unsupported GPU)";
                }
                LOGGER.warn("[c2me-wgpu] {}, falling back to CPU", dt);
            } else {
                st = Status.OK;
                dt = "wgpu acceleration enabled";
                LOGGER.info("[c2me-wgpu] wgpu acceleration enabled");
            }
        } catch (IllegalCallerException e) {
            st = Status.FFM_DISABLED;
            dt = "--enable-native-access=com.rubidiumlighting.c2me.opts.accel.wgpu";
            LOGGER.warn("""

                    [c2me-wgpu] Panama FFM native access is not enabled.
                    Please add the following JVM argument:
                      {}
                    Falling back to CPU world generation.
                    """, dt);
        } catch (Throwable t) {
            st = Status.INIT_FAILED;
            dt = String.valueOf(t.getMessage());
            LOGGER.warn("[c2me-wgpu] Failed to initialise wgpu acceleration, falling back to CPU", t);
        }
        status = st;
        detail = dt;
    }

    /**
     * Whether GPU generation can be used <em>right now</em>.
     *
     * <p>The {@link #status} captured at class-load time is the *start-up*
     * result and never changes. That is fine for reporting why the mod started
     * on the CPU path, but wrong as a gate: the instance can be rebuilt later
     * (a save switch after the user picks a different backend, for example),
     * and a start-up failure would otherwise disable GPU generation for the
     * rest of the session even though the device is now healthy.
     *
     * <p>So the live availability of the native binding decides. The start-up
     * status still short-circuits the two cases a rebuild cannot fix: no
     * native library for this platform, and FFM being disabled.
     */
    public static boolean isEnabled() {
        if (status == Status.NATIVE_UNAVAILABLE || status == Status.FFM_DISABLED) {
            return false;
        }
        return WgpuThinBinding.isAvailable();
    }

    /**
     * Current status, recomputed from live availability.
     *
     * <p>Differs from the start-up snapshot only when a rebuild changed the
     * outcome, which is exactly the case the UI needs to reflect.
     */
    public static Status getStatus() {
        if (status == Status.NATIVE_UNAVAILABLE || status == Status.FFM_DISABLED) {
            return status;
        }
        return WgpuThinBinding.isAvailable() ? Status.OK : status;
    }

    public static String getDetail() {
        return detail;
    }

    /**
     * Returns a translation key for the failure detail, if it matches a known
     * hardware/driver condition. Returns null if the error is unknown, in
     * which case the caller should display the raw detail.
     */
    public static String getErrorKey() {
        if (status != Status.INIT_FAILED) return null;
        return ErrorTranslator.getErrorKey(detail);
    }
}
