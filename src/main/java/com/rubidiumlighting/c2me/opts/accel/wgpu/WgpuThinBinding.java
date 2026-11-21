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

import com.rubidiumlighting.c2me.opts.accel.wgpu.config.Config;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.lang.invoke.MethodType;
import java.nio.ByteBuffer;
//import java.lang.foreign.Arena;

/**
 * Thin Panama FFM binding to the Rust wgpu shared library.
 * <p>
 * Downcall handles over a handle + poll model, plus one upcall for Rust log
 * forwarding. The exact set of downcalls is an internal detail and changes
 * as features land.
 * All failures throw RuntimeException (fallback catches at Mixin level).
 */
public class WgpuThinBinding {
    private static final Logger LOGGER = LoggerFactory.getLogger("c2me-wgpu");

    // Status codes returned by poll_status / take_result_blob
    public static final int STATUS_PENDING = 0;
    public static final int STATUS_DONE = 1;
    public static final int STATUS_ERROR = 2;
    public static final int STATUS_INVALID_HANDLE = -1;

    private static MethodHandle MH_set_hot_flags;
    private static MethodHandle MH_create_instance;
    private static MethodHandle MH_rebuild_instance;
    private static MethodHandle MH_shutdown_instance;
    private static MethodHandle MH_list_gpus;
    private static MethodHandle MH_world_new;
    private static MethodHandle MH_world_destroy;
    private static MethodHandle MH_request_chunks;
    private static MethodHandle MH_poll_status;
    private static MethodHandle MH_take_result_blob;
    private static MethodHandle MH_take_block_blob;
    private static MethodHandle MH_free_result_blob;
    private static MethodHandle MH_set_aquifer_table;
    private static MethodHandle MH_set_beard_table;
    private static MethodHandle MH_set_aquifer_mode;
    private static MethodHandle MH_set_ore_seed;
    private static MethodHandle MH_set_log_path;
    private static MethodHandle MH_setup_panic_hook;
    private static MethodHandle MH_set_log_callback;
    /**
     * Optional: reports shader-compile progress to the overlay. Resolved
     * separately from the main link, so a native library from before this
     * feature (e.g. a stale dev build) degrades to "no overlay" instead of
     * disabling GPU acceleration entirely.
     */
    private static MethodHandle MH_compile_progress;
    /**
     * Optional: largest batch the live device can dispatch for a shape.
     * Resolved separately for the same reason as {@link #MH_compile_progress}:
     * a native library from before this query should degrade to the caller's
     * own cap, not disable GPU acceleration.
     */
    private static MethodHandle MH_batch_limit;

    /** Non-null when the FFM downcall linking failed (e.g. a symbol is
     *  missing from the native library). Lets callers report the real cause
     *  instead of blaming the GPU adapter. */
    private static String linkError = null;

    /** The last error reported by create_instance, for user-facing notices. */
    private static volatile String lastInstanceError = null;

    // Declared before the static initialiser: Java runs static field
    // initialisers in textual order, and the block below calls
    // invalidateListings(), which would otherwise NPE on a null field.
    /**
     * GPU listings keyed by backend selection. The listing depends on which
     * backends are enumerated, so one cache entry would serve the wrong list
     * after the user changes the backend.
     */
    private static final java.util.Map<Integer, String> cachedGpus =
            new java.util.concurrent.ConcurrentHashMap<>();

    private static long instanceHandle;
    /** True only if the native lib linked AND a wgpu instance was created. */
    private static boolean available;

    /**
     * The cold settings the live device was actually built with.
     *
     * <p>Change detection compares against this rather than against
     * {@code Config.get()}. The GUI edits that singleton in place, so by the
     * time a save is reloaded the in-memory value already equals the file and
     * a "current vs on-disk" comparison would always report "unchanged" —
     * silently skipping the rebuild the user just asked for. The snapshot
     * records what the GPU is really running with, which is the only
     * meaningful baseline.
     */
    private record AppliedCold(int gpuIndex, int backend, boolean disablePipelineCache, Boolean mappedAtCreation) {}

    private static volatile AppliedCold appliedCold;

    private static AppliedCold coldOf(Config c) {
        return new AppliedCold(c.gpuIndex, c.backend, c.disablePipelineCache, c.mappedAtCreation);
    }

    /**
     * Render a backend bitmask as names, for log and chat text.
     *
     * <p>The bits mirror {@code wgpu::Backends}: each is {@code 1 << n} where
     * n is the {@code wgpu::Backend} discriminant (Vulkan=1, Metal=2, Dx12=3,
     * Gl=4).
     */
    private static String describeBackend(int bits) {
        if (bits < 0) {
            return "Automatic";
        }
        StringBuilder sb = new StringBuilder();
        // bit = 1 << wgpu::Backend discriminant:
        // Vulkan=1 -> 2, Metal=2 -> 4, Dx12=3 -> 8, Gl=4 -> 16.
        int[][] table = {{2, 0}, {4, 1}, {8, 2}, {16, 3}};
        String[] names = {"Vulkan", "Metal", "DX12", "GL"};
        for (int[] row : table) {
            if ((bits & row[0]) != 0) {
                if (sb.length() > 0) sb.append('+');
                sb.append(names[row[1]]);
            }
        }
        return sb.length() == 0 ? "Automatic" : sb.toString();
    }

    /**
     * @return true when the config's cold settings differ from those the live
     *         device was built with, i.e. a rebuild is required for them to
     *         take effect. False when no device has been created yet.
     */
    /**
     * Whether the graphics backend differs from the one the live device was
     * built with.
     *
     * <p>Only meaningful on Android, where a GLES device does not release its
     * native resources on drop (measured 15 MiB freed out of ~4 GB RSS,
     * against 645 MiB for a Vulkan device), so tearing one down while building
     * another exhausts memory. There a backend change requires a restart.
     *
     * <p>On desktop the same measurement shows no such retention, and the
     * hot-swap path is useful, so this always returns {@code false} there. The
     * platform check lives here rather than at the call site so the two
     * behaviours cannot drift apart.
     */
    public static boolean backendChanged(Config cfg) {
        if (!NativeLoader.IS_ANDROID) {
            return false;
        }
        AppliedCold applied = appliedCold;
        if (applied == null) {
            return false;
        }
        return applied.backend() != cfg.backend;
    }

    public static boolean coldConfigChanged(Config cfg) {
        AppliedCold applied = appliedCold;
        if (applied == null) {
            // No device was ever built from this config, which means the last
            // attempt failed (a rejected backend, a driver refusing the
            // device, ...). Reporting "unchanged" here would pin the session
            // to the CPU path forever: the user fixes the setting, and nothing
            // rebuilds because there is no baseline to compare against.
            // Treating it as changed makes the next save load retry.
            return true;
        }
        return !applied.equals(coldOf(cfg));
    }

    /** True once every downcall symbol resolved. Set in the static block. */
    private static boolean linked = true;

    static {
        try {
            MH_set_hot_flags = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_set_hot_flags").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,        // -> i32 (0 = ok)
                            ValueLayout.JAVA_INT,        // debug: u32
                            ValueLayout.JAVA_INT         // profile: u32
                    )
            );

            // create / rebuild share the same C signature: the three cold
            // settings (gpu_index, disable_pipeline_cache, mapped_at_creation)
            // travel with the call, because they belong to the device being
            // built and only take effect by building a new one.
            MH_create_instance = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_create_instance").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_LONG,       // -> 1 ok / 0 error
                            ValueLayout.JAVA_INT,        // gpu_index: i32
                            ValueLayout.JAVA_INT,        // backend: i32
                            ValueLayout.JAVA_INT,        // disable_pipeline_cache: u32
                            ValueLayout.JAVA_INT,        // mapped_at_creation: i32
                            ValueLayout.ADDRESS,         // err_buf: *mut c_char
                            ValueLayout.JAVA_LONG        // err_buf_len: usize
                    )
            );

            MH_rebuild_instance = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_rebuild_instance").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_LONG,       // -> 1 ok / 0 error
                            ValueLayout.JAVA_INT,        // gpu_index: i32
                            ValueLayout.JAVA_INT,        // backend: i32
                            ValueLayout.JAVA_INT,        // disable_pipeline_cache: u32
                            ValueLayout.JAVA_INT,        // mapped_at_creation: i32
                            ValueLayout.ADDRESS,         // err_buf: *mut c_char
                            ValueLayout.JAVA_LONG        // err_buf_len: usize
                    )
            );

            MH_shutdown_instance = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_shutdown_instance").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT)
            );

            MH_list_gpus = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_list_gpus").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_LONG,       // -> bytes written, or < 0
                            ValueLayout.JAVA_INT,        // backend: i32
                            ValueLayout.ADDRESS,         // buf: *mut c_char
                            ValueLayout.JAVA_LONG,       // buf_len: usize
                            ValueLayout.ADDRESS          // out_len: *mut usize
                    )
            );

            MH_world_destroy = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_world_destroy").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_LONG
                    )
            );
            MH_world_new = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_world_new").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_LONG,       // -> i64 world_handle (0 on error)
                            ValueLayout.JAVA_LONG,       // instance_h: i64
                            ValueLayout.ADDRESS,         // config_blob_ptr: *const u8
                            ValueLayout.JAVA_LONG,       // config_blob_len: usize
                            ValueLayout.ADDRESS,         // err_buf: *mut c_char
                            ValueLayout.JAVA_LONG        // err_buf_len: usize
                    )
            );

            MH_request_chunks = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_request_chunks").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_LONG,       // -> i64 request_handle (0 on error)
                            ValueLayout.JAVA_LONG,       // world_h: i64
                            ValueLayout.ADDRESS,         // positions_ptr: *const i32 (packed cx,cz)
                            ValueLayout.JAVA_INT,        // position_count: i32
                            ValueLayout.JAVA_INT,        // external_count: i32 (slots per chunk)
                            ValueLayout.ADDRESS,         // external_ptr: *const f32
                            ValueLayout.JAVA_LONG,       // external_len: usize
                            ValueLayout.ADDRESS,         // err_buf: *mut c_char
                            ValueLayout.JAVA_LONG        // err_buf_len: usize
                    )
            );

            MH_poll_status = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_poll_status").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,        // -> i32 status
                            ValueLayout.JAVA_LONG        // request_h: i64
                    )
            );

            MH_take_result_blob = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_take_result_blob").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,        // -> i32 status
                            ValueLayout.JAVA_LONG,       // request_h: i64
                            ValueLayout.ADDRESS,         // out_ptr: *mut *const u8
                            ValueLayout.ADDRESS          // out_len: *mut usize
                    )
            );

            MH_take_block_blob = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_take_block_blob").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS
                    )
            );

            MH_free_result_blob = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_free_result_blob").orElseThrow(),
                    FunctionDescriptor.ofVoid(
                            ValueLayout.ADDRESS,         // ptr: *mut u8
                            ValueLayout.JAVA_LONG        // len: usize
                    )
            );

            MH_set_aquifer_mode = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_set_aquifer_mode").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT));
            MH_set_ore_seed = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_set_ore_seed").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_LONG));
            MH_set_beard_table = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_set_beard_table").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
            MH_set_aquifer_table = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_set_aquifer_table").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,        // -> i32 status
                            ValueLayout.JAVA_LONG,       // world_h: i64
                            ValueLayout.JAVA_INT,        // cx: i32
                            ValueLayout.JAVA_INT,        // cz: i32
                            ValueLayout.ADDRESS,         // table_ptr: *const u8
                            ValueLayout.JAVA_LONG        // table_len: usize
                    )
            );

            MH_setup_panic_hook = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_setup_panic_hook").orElseThrow(),
                    FunctionDescriptor.ofVoid()
            );
            MH_set_log_path = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_set_log_path").orElseThrow(),
                    FunctionDescriptor.ofVoid(
                            ValueLayout.ADDRESS,         // ptr: *const u8
                            ValueLayout.JAVA_LONG        // len: usize
                    )
            );
            try {
                MH_compile_progress = NativeLoader.linker.downcallHandle(
                        NativeLoader.lookup.find("c2me_wgpu_compile_progress").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.JAVA_LONG)
                );
            } catch (Throwable t) {
                MH_compile_progress = null;
            }
            try {
                MH_batch_limit = NativeLoader.linker.downcallHandle(
                        NativeLoader.lookup.find("c2me_wgpu_batch_limit").orElseThrow(),
                        FunctionDescriptor.of(
                                ValueLayout.JAVA_INT,        // -> i32 chunks, -1 unknown
                                ValueLayout.JAVA_INT,        // cells_per_chunk
                                ValueLayout.JAVA_INT         // blocks_per_chunk
                        )
                );
            } catch (Throwable t) {
                MH_batch_limit = null;
            }
        } catch (Throwable t) {
            System.err.println("[c2me-wgpu] FFM link failed: " + t);
            linkError = String.valueOf(t);
            linked = false;
        }

        // Auto-create the singleton WgpuInstance on class load. IMPORTANT: this
        // block must NEVER throw — a class whose <clinit> throws is permanently
        // poisoned (NoClassDefFoundError afterwards), so the fallback guard in
        // callers would itself explode. On any failure we just leave
        // instanceHandle=0 / available=false.
        if (linked) {

        }

        if (linked) {
            // Registration of the panic hook and the log callback happens
            // exactly once per JVM, even though the instance itself can be
            // rebuilt later: both are process-level hooks on the Rust side.
            try {
                MH_setup_panic_hook.invokeExact();
            } catch (Throwable t) {
                System.err.println("[c2me-wgpu] setup_panic_hook failed: " + t);
            }
            try {
                MethodHandle target = MethodHandles.lookup().findStatic(
                        WgpuThinBinding.class, "onRustLog",
                        MethodType.methodType(void.class, MemorySegment.class, long.class));
                MemorySegment upcall = NativeLoader.linker.upcallStub(
                        target, FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG),
                        Arena.global());
                MH_set_log_callback = NativeLoader.linker.downcallHandle(
                        NativeLoader.lookup.find("c2me_wgpu_set_log_callback").orElseThrow(),
                        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
                MH_set_log_callback.invokeExact(upcall);
            } catch (Throwable t) {
                System.err.println("[c2me-wgpu] Failed to register Rust log callback: " + t);
            }

            // Build the first instance from the config on disk.
            try {
                pushHotFlags();
                ensureInstance();
            } catch (Throwable t) {
                System.err.println("[c2me-wgpu] create_instance failed: " + t);
            }
        }
    }

    /** Push the hot-reloadable flags (debug / profiling). Safe at any time. */
    public static void pushHotFlags() throws Throwable {
        if (!linked) return;
        var cfg = Config.get();
        int st = (int) MH_set_hot_flags.invokeExact(
                cfg.debugVerboseLogging ? 1 : 0,
                cfg.debugStageTiming ? 1 : 0);
        if (st != 0) {
            LOGGER.warn("[c2me-wgpu] set_hot_flags returned {}", st);
        }
    }

    /**
     * Create the wgpu instance if it does not exist yet, using the cold
     * settings currently on disk. A no-op when an instance is already live.
     *
     * @return true if an instance is available afterwards.
     */
    public static synchronized boolean ensureInstance() throws Throwable {
        if (!linked) return false;
        if (available) return true;
        var cfg = Config.get();
        try (Arena errArena = Arena.ofConfined()) {
            MemorySegment errBuf = errArena.allocate(1024);
            long r = (long) MH_create_instance.invokeExact(
                    cfg.gpuIndex,
                    cfg.backend,
                    cfg.disablePipelineCache ? 1 : 0,
                    cfg.mappedAtCreation == null ? -1 : (cfg.mappedAtCreation ? 1 : 0),
                    errBuf,
                    errBuf.byteSize());
            if (r == 0) {
                String err = errBuf.getString(0);
                lastInstanceError = err;
                LOGGER.warn("[c2me-wgpu] Failed to create wgpu instance: {}", err);
                return false;
            }
        }
        lastInstanceError = null;
        available = true;
        instanceHandle = 1L;
        appliedCold = coldOf(cfg);
        invalidateListings();
        return true;
    }

    /**
     * Tear down the instance and rebuild it from the current config.
     *
     * MUST only be called when no world handle is alive and no batch is in
     * flight: the Rust side stops the executor, but worlds hold pipelines
     * bound to the old device. The save-switch path guarantees this.
     *
     * @return true if a usable instance exists afterwards.
     */
    public static synchronized boolean rebuildFromConfig() throws Throwable {
        if (!linked) return false;
        var cfg = Config.get();
        try (Arena errArena = Arena.ofConfined()) {
            MemorySegment errBuf = errArena.allocate(1024);
            long r = (long) MH_rebuild_instance.invokeExact(
                    cfg.gpuIndex,
                    cfg.backend,
                    cfg.disablePipelineCache ? 1 : 0,
                    cfg.mappedAtCreation == null ? -1 : (cfg.mappedAtCreation ? 1 : 0),
                    errBuf,
                    errBuf.byteSize());
            if (r == 0) {
                // Rust keeps the previous instance when the new device cannot
                // be built, so GPU generation still works — but with settings
                // that differ from the config. Silently carrying on is the
                // worst outcome: the user asked for one backend and got
                // another with no indication. Report it explicitly.
                //
                // `appliedCold` deliberately keeps describing the live device,
                // so `coldConfigChanged()` stays true and the next save load
                // retries the switch.
                String reason = errBuf.getString(0);
                LOGGER.warn("[c2me-wgpu] Failed to rebuild wgpu instance: {}", reason);

                // Capture what was requested *before* touching the config:
                // `cfg` is the Config singleton, so writing the effective
                // settings back below would otherwise rewrite the requested
                // value too and the message would claim a switch from X to X.
                int requestedBackend = cfg.backend;
                AppliedCold live = appliedCold;

                // The file is the source of truth for "what the user wants";
                // after a failed switch that would be a lie, and the next
                // launch would retry the rejected setting and drop to the CPU
                // path. Reset the rejected field rather than copying the live
                // device's backend: copying would pin the session to whichever
                // backend happened to be in use, which is wrong on hardware
                // where that backend is not the best available (an HD 4000 has
                // DX12 but no Vulkan).
                //
                // `backend` goes back to automatic; the automatic policy picks
                // a working backend for this machine. `gpu_index` is kept only
                // if it still points at something.
                Config c = Config.get();
                c.backend = -1;
                if (live != null) {
                    c.disablePipelineCache = live.disablePipelineCache();
                    c.mappedAtCreation = live.mappedAtCreation();
                }
                c.save();

                // The message names only what the user asked for and what
                // happens next. Naming the "current" backend was misleading:
                // it described the configured value, which after the reset is
                // simply "automatic", not the device actually in use.
                WgpuFallbackNotice.report("rebuild-failed-" + reason,
                        "c2me-opts-accel-wgpu.notify.rebuild_failed",
                        describeBackend(requestedBackend),
                        ErrorTranslator.localize(reason));
                return available;
            }
        }
        available = true;
        instanceHandle = 1L;
        appliedCold = coldOf(cfg);
        invalidateListings();
        pushHotFlags();
        // The adapter may have changed, so the cached name is stale.
        WgpuStatus.invalidate();
        return true;
    }

    /** Drop the instance and its executor. Idempotent. */
    public static synchronized void shutdownInstance() {
        if (!linked || !available) return;
        try {
            int st = (int) MH_shutdown_instance.invokeExact();
            if (st != 0) {
                LOGGER.warn("[c2me-wgpu] shutdown_instance returned {}", st);
            }
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] shutdown_instance failed", t);
        } finally {
            available = false;
            instanceHandle = 0L;
            // No device is live, so there is nothing to compare against; the
            // next ensureInstance() records a fresh snapshot.
            appliedCold = null;
            invalidateListings();
        }
    }

    /**
     * Enumerate the GPUs visible to the backend as a JSON array.
     *
     * @return the JSON text, or {@code null} if the query failed.
     */
    public static String listGpus(int backend) {
        if (!linked) return null;
        try (Arena arena = Arena.ofConfined()) {
            // First call with a small buffer tells us the required size via
            // out_len; then retry with exactly that. Two calls avoid guessing
            // a fixed cap that a machine with many adapters could exceed.
            MemorySegment outLen = arena.allocate(ValueLayout.JAVA_LONG);
            MemorySegment small = arena.allocate(64);
            long r = (long) MH_list_gpus.invokeExact(backend, small, small.byteSize(), outLen);
            if (r >= 0) {
                return small.getString(0);
            }
            long need = outLen.get(ValueLayout.JAVA_LONG, 0);
            if (r != -2 || need <= 0 || need > 1 << 20) {
                return null;
            }
            MemorySegment buf = arena.allocate(need);
            r = (long) MH_list_gpus.invokeExact(backend, buf, buf.byteSize(), outLen);
            return r >= 0 ? buf.getString(0) : null;
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] list_gpus failed", t);
            return null;
        }
    }

    public static long getInstanceHandle() {
        return instanceHandle;
    }

    public static boolean isAvailable() {
        return available;
    }

    /**
     * Packed shader-compile progress: {@code active << 32 | done << 16 | total}.
     *
     * <p>Returns {@code -1} when the native library predates the symbol or the
     * call fails, which the overlay treats as "nothing compiling". Never
     * throws: this is polled from the render thread every frame.
     */
    public static long compileProgress() {
        MethodHandle mh = MH_compile_progress;
        if (mh == null) return -1L;
        try {
            return (long) mh.invokeExact();
        } catch (Throwable t) {
            return -1L;
        }
    }

    /**
     * Largest batch the live device can dispatch for this shape, in chunks.
     *
     * <p>Read-only: the Rust side derives it from the device's own buffer
     * limits (128 MiB storage binding / 256 MiB buffer by default) and the
     * per-chunk cell/block counts, without creating or resizing anything.
     * The executor clamps its buffers with the same derivation, so a batch at
     * or below this number is guaranteed to fit.
     *
     * @return the limit, or {@code -1} when no device exists or the native
     *         library predates the query (callers should use a conservative
     *         fallback)
     */
    public static int batchLimit(int cellsPerChunk, int blocksPerChunk) {
        MethodHandle mh = MH_batch_limit;
        if (mh == null) return -1;
        try {
            return (int) mh.invokeExact(cellsPerChunk, blocksPerChunk);
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] batch_limit query failed", t);
            return -1;
        }
    }

    public static String getLastInstanceError() {
        return lastInstanceError;
    }

    /** @return the FFM linking failure detail, or {@code null} if linking was fine. */
    public static String getLinkError() {
        return linkError;
    }

    /**
     * Create a new WorldContext from a serialized NoiseRouter blob.
     *
     * @param instanceH instance handle from getInstanceHandle()
     * @param blob      serialized NoiseRouter blob (from NoiseRouterBlobifier)
     * @return world handle (non-zero)
     */
    public static void worldDestroy(long worldH) {
        if (worldH == 0L) return;
        try {
            int st = (int) MH_world_destroy.invokeExact(worldH);
            if (st != 0) throw new RuntimeException("world_destroy status=" + st);
        } catch (Throwable e) {
            throw new RuntimeException("[c2me-wgpu] world_destroy failed", e);
        }
    }

    public static long worldNew(long instanceH, ByteBuffer blob) {
        if (!blob.isDirect()) {
            ByteBuffer direct = ByteBuffer.allocateDirect(blob.remaining());
            direct.put(blob);
            direct.flip();
            blob = direct;
        }
        MemorySegment blobSeg = MemorySegment.ofBuffer(blob);
        try (Arena errArena = Arena.ofConfined()) {
            MemorySegment errBuf = errArena.allocate(1024);
            long h = (long) MH_world_new.invokeExact(
                    instanceH,
                    blobSeg,
                    (long) blob.remaining(),
                    errBuf,
                    errBuf.byteSize()
            );
            if (h == 0) {
                String msg = errBuf.getString(0);
                throw new RuntimeException("[c2me-wgpu] world_new failed: " + msg);
            }
            return h;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException(t);
        }
    }

    /**
     * Submit a batch of chunk positions for GPU generation.
     *
     * @param worldH     world handle
     * @param positions  packed [cx0,cz0, cx1,cz1, ...]
     * @return request handle
     */
    /**
     * Upload CPU-evaluated external-input values for the given world.
     * The data is copied into native memory by Rust, so the caller's array
     * can be GC'd afterwards; nothing is retained across calls.
     *
     * @param worldH world handle
     * @param count  number of external subtrees
     * @param data   flat [chunk][ext][cell] f32 values
     */
    /**
     * §8.4 Step B: stage a per-chunk aquifer table for the next request.
     * `table` is the compact Java-serialized layout (magic + n + seed + cells).
     */
    /** §8.4: set per-dimension aquifer mode (hasAquifers/seaLevel/fluid/lavaY). */
    public static void setAquiferMode(long worldH, boolean hasAquifers,
                                      int seaLevel, int defaultFluid, int lavaLevel) {
        try {
            int st = (int) MH_set_aquifer_mode.invokeExact(worldH,
                    hasAquifers ? 1 : 0, seaLevel, defaultFluid, lavaLevel);
            if (st != 0) throw new RuntimeException("set_aquifer_mode status=" + st);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("[c2me-wgpu] set_aquifer_mode failed", t);
        }
    }

    /** Set the ore-vein RNG seed (seedLo, seedHi) from MC's oreRandomDeriver. */
    public static void setOreSeed(long worldH, long seedLo, long seedHi) {
        try {
            int st = (int) MH_set_ore_seed.invokeExact(worldH, seedLo, seedHi);
            if (st != 0) throw new RuntimeException("set_ore_seed status=" + st);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("[c2me-wgpu] set_ore_seed failed", t);
        }
    }

    /** §8.4: stage a per-chunk Beardifier table (packed u32 LE). */
    public static void setBeardTable(long worldH, int cx, int cz, byte[] table) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocateFrom(ValueLayout.JAVA_BYTE, table);
            int st = (int) MH_set_beard_table.invokeExact(worldH, cx, cz, seg, (long) table.length);
            if (st != 0) throw new RuntimeException("set_beard_table status=" + st);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("[c2me-wgpu] set_beard_table failed", t);
        }
    }

    public static void setAquiferTable(long worldH, int cx, int cz, byte[] table) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocateFrom(ValueLayout.JAVA_BYTE, table);
            int st = (int) MH_set_aquifer_table.invokeExact(worldH, cx, cz, seg, (long) table.length);
            if (st != 0) throw new RuntimeException("set_aquifer_table status=" + st);
        } catch (Throwable t) {
            throw new RuntimeException("[c2me-wgpu] set_aquifer_table failed", t);
        }
    }

    /**
     * Submit a batch of chunks plus the CPU-evaluated external-input values
     * that belong to exactly this batch.
     *
     * <p>The payload travels with the request instead of through a
     * process-global setter, so several batches can be in flight: the native
     * side copies it into the batch and the executor only reads that copy.
     *
     * @param positions  packed chunk coords `[cx0, cz0, cx1, cz1, ...]`,
     *                   already sorted the way the native queue sorts them
     * @param externalCount slots per chunk (0 when the dimension has no
     *                      CPU-evaluated subtrees)
     * @param externals  `[chunk][slot][cell]` values matching {@code positions},
     *                   or {@code null} when {@code externalCount == 0}
     */
    public static long requestChunks(long worldH, int[] positions,
                                     int externalCount, float[] externals) {
        try (Arena arena = Arena.ofConfined(); Arena errArena = Arena.ofConfined()) {
            MemorySegment posSeg = arena.allocateFrom(ValueLayout.JAVA_INT, positions);
            boolean hasExt = externalCount > 0 && externals != null && externals.length > 0;
            MemorySegment extSeg = hasExt
                    ? arena.allocateFrom(ValueLayout.JAVA_FLOAT, externals)
                    : MemorySegment.NULL;
            long extLen = hasExt ? externals.length : 0L;
            MemorySegment errBuf = errArena.allocate(1024);
            long h = (long) MH_request_chunks.invokeExact(
                    worldH,
                    posSeg,
                    positions.length / 2,
                    hasExt ? externalCount : 0,
                    extSeg,
                    extLen,
                    errBuf,
                    errBuf.byteSize()
            );
            if (h == 0) {
                String msg = errBuf.getString(0);
                throw new RuntimeException("[c2me-wgpu] request_chunks failed: " + msg);
            }
            return h;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException(t);
        }
    }

    /**
     * Poll request status.
     *
     * @return 0=PENDING 1=DONE 2=ERROR -1=INVALID_HANDLE
     */
    public static int pollStatus(long requestH) {
        try {
            return (int) MH_poll_status.invokeExact(requestH);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException(t);
        }
    }

    /**
     * Take the result blob out. Must be called only once after pollStatus == DONE.
     *
     * @return MemorySegment wrapping the Rust-owned blob (valid until take).
     * Use .asByteBuffer() then .asFloatBuffer() to read density channels.
     */
    /**
     * §3.5: take the per-block state blob (raw u32 per block, chunk-major).
     * Same lifetime/copy semantics as {@link #takeResultBlob}.
     */
    public static MemorySegment takeBlockBlob(long requestH) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outPtr = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment outLen = arena.allocate(ValueLayout.JAVA_LONG);
            int status = (int) MH_take_block_blob.invokeExact(requestH, outPtr, outLen);
            if (status != 0) {
                throw new RuntimeException("[c2me-wgpu] take_block_blob failed, status=" + status);
            }
            MemorySegment dataPtr = outPtr.get(ValueLayout.ADDRESS, 0);
            long len = outLen.get(ValueLayout.JAVA_LONG, 0);
            if (dataPtr.address() == 0 || len <= 0) {
                if (MH_free_result_blob != null) MH_free_result_blob.invokeExact(dataPtr, len);
                return MemorySegment.ofArray(new byte[0]);
            }
            byte[] copy = new byte[(int) len];
            MemorySegment.copy(dataPtr.reinterpret(len), 0, MemorySegment.ofArray(copy), 0, len);
            MH_free_result_blob.invokeExact(dataPtr, len);
            // Skip the 8-byte ResultBlob header (magic+version+chunk_count).
            return MemorySegment.ofArray(copy).asSlice(8);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException(t);
        }
    }

    public static MemorySegment takeResultBlob(long requestH) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outPtr = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment outLen = arena.allocate(ValueLayout.JAVA_LONG);
            int status = (int) MH_take_result_blob.invokeExact(requestH, outPtr, outLen);
            if (status != 0) {
                throw new RuntimeException("[c2me-wgpu] take_result_blob failed, status=" + status);
            }
            MemorySegment dataPtr = outPtr.get(ValueLayout.ADDRESS, 0);
            long len = outLen.get(ValueLayout.JAVA_LONG, 0);
            if (dataPtr.address() == 0 || len <= 0) {
                if (MH_free_result_blob != null) {
                    MH_free_result_blob.invokeExact(dataPtr, len);
                }
                return MemorySegment.ofArray(new byte[0]);
            }
            // Copy the Rust-owned blob into a Java heap array, then reclaim the
            // native buffer immediately. Without this, each batch's result Vec
            // leaked (into_raw_parts forgot it, nothing ever freed) -> ~10 GB
            // after ~65k chunks.
            byte[] copy = new byte[(int) len];
            MemorySegment.copy(dataPtr.reinterpret(len), 0,
                    MemorySegment.ofArray(copy), 0, len);
            MH_free_result_blob.invokeExact(dataPtr, len);
            return MemorySegment.ofArray(copy);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException(t);
        }
    }

    /**
     * Cached results of the JSON listing queries.
     *
     * <p>Both queries create a throwaway `wgpu::Instance` and enumerate
     * adapters, which on some Vulkan drivers blocks for hundreds of
     * milliseconds. They were previously called from the config screen's
     * {@code init()}, i.e. on the render thread, and the resulting stall looked
     * like the game had frozen. The listings only change when the device is
     * rebuilt, so they are cached and invalidated there.
     */
    /** Drop the cached listings; called whenever the instance is replaced. */
    private static void invalidateListings() {
        cachedGpus.clear();
    }

    /** @return the GPU listing for {@code backend}, enumerating once per selection. */
    public static String listGpusCached(int backend) {
        return cachedGpus.computeIfAbsent(backend, WgpuThinBinding::listGpus);
    }

    /**
     * Two-call helper shared by the JSON listing queries: the first call with
     * a tiny buffer reports the required size via {@code out_len}, the second
     * fills a buffer of exactly that size.
     */
    private static String queryJsonArray(String what, MethodHandle handle) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment outLen = arena.allocate(ValueLayout.JAVA_LONG);
            MemorySegment small = arena.allocate(64);
            long r = (long) handle.invokeExact(small, small.byteSize(), outLen);
            if (r >= 0) {
                return small.getString(0);
            }
            long need = outLen.get(ValueLayout.JAVA_LONG, 0);
            if (r != -2 || need <= 0 || need > 1 << 20) {
                return null;
            }
            MemorySegment buf = arena.allocate(need);
            r = (long) handle.invokeExact(buf, buf.byteSize(), outLen);
            return r >= 0 ? buf.getString(0) : null;
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] {} failed", what, t);
            return null;
        }
    }

    /** Upcall receiver from Rust. */
    private static void onRustLog(MemorySegment msgPtr, long msgLen) {
        if (msgPtr.address() == 0L || msgLen <= 0 || msgLen > 1024 * 1024) return;
        try {
            byte[] bytes = msgPtr.reinterpret(msgLen).toArray(ValueLayout.JAVA_BYTE);
            String msg = new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
            LOGGER.info("[c2me-wgpu] " + msg);
        } catch (Throwable t) {
            LOGGER.error("[c2me-wgpu] onRustLog callback failed", t);
        }
    }
}
