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

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.ByteBuffer;
//import java.lang.foreign.Arena;

/**
 * Thin Panama FFM binding to the Rust wgpu shared library.
 * <p>
 * 5 downcall handles, handle + poll model, no upcall callbacks.
 * All failures throw RuntimeException (fallback catches at Mixin level).
 */
public class WgpuThinBinding {

    // Status codes returned by poll_status / take_result_blob
    public static final int STATUS_PENDING = 0;
    public static final int STATUS_DONE = 1;
    public static final int STATUS_ERROR = 2;
    public static final int STATUS_INVALID_HANDLE = -1;

    private static final MethodHandle MH_create_instance;
    private static final MethodHandle MH_world_new;
    private static final MethodHandle MH_request_chunks;
    private static final MethodHandle MH_poll_status;
    private static final MethodHandle MH_take_result_blob;
    private static final MethodHandle MH_get_gpu_name;

    private static final long instanceHandle;

    static {
        try {
            MH_create_instance = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_create_instance").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_LONG,       // -> i64 instance_handle (0 on error)
                            ValueLayout.JAVA_LONG,       // features_mask: u64
                            ValueLayout.ADDRESS,         // err_buf: *mut c_char
                            ValueLayout.JAVA_LONG        // err_buf_len: usize
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

            MH_get_gpu_name = NativeLoader.linker.downcallHandle(
                    NativeLoader.lookup.find("c2me_wgpu_get_gpu_name").orElseThrow(),
                    FunctionDescriptor.of(
                            ValueLayout.ADDRESS          // -> *const c_char (static UTF-8)
                    )
            );
        } catch (Throwable t) {
            throw new RuntimeException("[c2me-wgpu] Failed to link FFM downcall handles", t);
        }

        // Auto-create the singleton WgpuInstance on class load
        try (Arena errArena = Arena.ofConfined()) {
            MemorySegment errBuf = errArena.allocate(1024);
            long h = (long) MH_create_instance.invokeExact(
                    0L,                    // features_mask: 0 = default
                    errBuf,
                    errBuf.byteSize()
            );
            if (h == 0) {
                String msg = errBuf.getString(0);
                throw new RuntimeException("[c2me-wgpu] Failed to create wgpu instance: " + msg);
            }
            instanceHandle = h;
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException("[c2me-wgpu] create_instance failed", t);
        }
    }

    public static String getGpuName() {
        try {
            var result = MH_get_gpu_name.invoke();
            if (result instanceof MemorySegment) {
                var segment = (MemorySegment) result;
                if (segment.address() == 0) {
                    return "unknown (null address)";
                }
                // 如果 byteSize 为 0，重新创建一个段，假设长度 256
                if (segment.byteSize() == 0) {
                    segment = MemorySegment.ofAddress(segment.address())
                            .reinterpret(256, Arena.global(), null);
                }
                try {
                    return segment.getString(0);
                } catch (IndexOutOfBoundsException | IllegalArgumentException e) {
                    return "unknown (invalid string)";
                }
            } else {
                return "unknown (not a MemorySegment)";
            }
        } catch (Throwable t) {
            t.printStackTrace();
            return "unknown (exception)";
        }
    }

    public static long getInstanceHandle() {
        return instanceHandle;
    }

    public static boolean isAvailable() {
        return NativeLoader.isAvailable();
    }

    /**
     * Create a new WorldContext from a serialized NoiseRouter blob.
     *
     * @param instanceH instance handle from getInstanceHandle()
     * @param blob      serialized NoiseRouter blob (from NoiseRouterBlobifier)
     * @return world handle (non-zero)
     */
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
            return dataPtr.reinterpret(len);
        } catch (Throwable t) {
            if (t instanceof RuntimeException re) throw re;
            throw new RuntimeException(t);
        }
    }

}
