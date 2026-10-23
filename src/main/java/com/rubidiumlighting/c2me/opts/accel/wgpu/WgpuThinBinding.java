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
    private static MethodHandle MH_set_external_inputs;
    private static MethodHandle MH_set_aquifer_table;
    private static MethodHandle MH_set_beard_table;
    private static MethodHandle MH_set_aquifer_mode;
    private static MethodHandle MH_set_ore_seed;
    private static MethodHandle MH_set_log_path;
    private static MethodHandle MH_setup_panic_hook;
    private static MethodHandle MH_set_log_callback;

    /** Non-null when the FFM downcall linking failed (e.g. a symbol is
     *  missing from the native library). Lets callers report the real cause
     *  instead of blaming the GPU adapter. */
    private static String linkError = null;

    private static long instanceHandle;
    /** True only if the native lib linked AND a wgpu instance was created. */
    private static boolean available;

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

            MH_set_external_inputs = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_set_external_inputs").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,        // -> i32 status
                            ValueLayout.JAVA_LONG,       // world_h: i64
                            ValueLayout.JAVA_INT,        // count: i32
                            ValueLayout.ADDRESS,         // data_ptr: *const f32
                            ValueLayout.JAVA_LONG        // data_len: usize
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
                    cfg.disablePipelineCache ? 1 : 0,
                    cfg.mappedAtCreation == null ? -1 : (cfg.mappedAtCreation ? 1 : 0),
                    errBuf,
                    errBuf.byteSize());
            if (r == 0) {
                LOGGER.warn("[c2me-wgpu] Failed to create wgpu instance: {}", errBuf.getString(0));
                return false;
            }
        }
        available = true;
        instanceHandle = 1L;
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
                    cfg.disablePipelineCache ? 1 : 0,
                    cfg.mappedAtCreation == null ? -1 : (cfg.mappedAtCreation ? 1 : 0),
                    errBuf,
                    errBuf.byteSize());
            if (r == 0) {
                // The previous instance is left in place by Rust, so we keep
                // reporting whatever availability we had rather than dropping
                // GPU acceleration for the rest of the session.
                LOGGER.warn("[c2me-wgpu] Failed to rebuild wgpu instance: {}", errBuf.getString(0));
                return available;
            }
        }
        available = true;
        instanceHandle = 1L;
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
        }
    }

    /**
     * Enumerate the GPUs visible to the backend as a JSON array.
     *
     * @return the JSON text, or {@code null} if the query failed.
     */
    public static String listGpus() {
        if (!linked) return null;
        try (Arena arena = Arena.ofConfined()) {
            // First call with a small buffer tells us the required size via
            // out_len; then retry with exactly that. Two calls avoid guessing
            // a fixed cap that a machine with many adapters could exceed.
            MemorySegment outLen = arena.allocate(ValueLayout.JAVA_LONG);
            MemorySegment small = arena.allocate(64);
            long r = (long) MH_list_gpus.invokeExact(small, small.byteSize(), outLen);
            if (r >= 0) {
                return small.getString(0);
            }
            long need = outLen.get(ValueLayout.JAVA_LONG, 0);
            if (r != -2 || need <= 0 || need > 1 << 20) {
                return null;
            }
            MemorySegment buf = arena.allocate(need);
            r = (long) MH_list_gpus.invokeExact(buf, buf.byteSize(), outLen);
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
     * §22 Step B: stage a per-chunk aquifer table for the next request.
     * `table` is the compact Java-serialized layout (magic + n + seed + cells).
     */
    /** §22: set per-dimension aquifer mode (hasAquifers/seaLevel/fluid/lavaY). */
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

    /** §23: stage a per-chunk Beardifier table (packed u32 LE). */
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

    public static void setExternalInputs(long worldH, int count, float[] data) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment dataSeg = (data != null && data.length > 0)
                    ? arena.allocateFrom(ValueLayout.JAVA_FLOAT, data)
                    : MemorySegment.NULL;
            long len = (data != null) ? data.length : 0;
            int st = (int) MH_set_external_inputs.invokeExact(
                    worldH,
                    count,
                    dataSeg,
                    len
            );
            if (st != 0) {
                throw new RuntimeException("[c2me-wgpu] set_external_inputs status=" + st);
            }
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException(t);
        }
    }

    public static long requestChunks(long worldH, int[] positions) {
        try (Arena arena = Arena.ofConfined(); Arena errArena = Arena.ofConfined()) {
            MemorySegment posSeg = arena.allocateFrom(ValueLayout.JAVA_INT, positions);
            MemorySegment errBuf = errArena.allocate(1024);
            long h = (long) MH_request_chunks.invokeExact(
                    worldH,
                    posSeg,
                    positions.length / 2,
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
     * §20: take the per-block state blob (raw u32 per block, chunk-major).
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
