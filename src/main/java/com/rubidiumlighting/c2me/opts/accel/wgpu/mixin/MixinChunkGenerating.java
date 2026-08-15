/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2021-2026 RubidiumLighting
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

package com.rubidiumlighting.c2me.opts.accel.wgpu.mixin;

import com.rubidiumlighting.c2me.opts.accel.wgpu.ModuleEntryPoint;
import com.rubidiumlighting.c2me.opts.accel.wgpu.NoiseRouterBlobifier;
import com.rubidiumlighting.c2me.opts.accel.wgpu.WgpuThinBinding;
import net.minecraft.world.gen.chunk.ChunkNoiseSampler;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

/**
 * Intercepts the NOISE generation phase entry of the chunk generator.
 * <p>
 * Flow (skeleton):
 *   1. If ModuleEntryPoint disabled -> return, run vanilla CPU path.
 *   2. Collect batch of pending ChunkPos.
 *   3. Blobify NoiseRouter + ShapeCfg -> submit to wgpu via requestChunks.
 *   4. poll loop with Thread.onSpinWait() until DONE.
 *   5. take_result_blob -> FloatBuffer -> System.arraycopy f32->double[].
 *   6. NaN/clamp defensive checks; on any exception, fall back to CPU path
 *      (re-invoke original noise sampler for those chunks).
 * <p>
 * TODO: Locate the actual target method (ChunkNoiseSampler / NoiseChunk
 * generation entry) once real implementation begins. Skeleton just prints
 * a log line on injection hit.
 */
@Mixin(ChunkNoiseSampler.class)
public abstract class MixinChunkGenerating {

    @Inject(method = "<init>", at = @At("TAIL"))
    private void c2me_wgpu$onChunkNoiseInit(CallbackInfo ci) {
        if (!ModuleEntryPoint.isEnabled()) {
            return; // wgpu disabled or unavailable -> CPU path, nothing to do
        }
        // Skeleton: this is where we'd hook NOISE phase batching.
        // For now, just touch the classes so the JVM loads them (probes links work).
        try {
            Class.forName(NoiseRouterBlobifier.class.getName());
            Class.forName(WgpuThinBinding.class.getName());
        } catch (ClassNotFoundException ignored) {
        }
    }

    // --- Helpers planned for real impl ---

    /**
     * Copy f32 density array from Rust blob into MC's double[] destination,
     * with NaN-stripping and clamp defense.
     */
    private static void copyF32ToDouble(MemorySegment blob, double[] dst) {
        FloatBuffer fbuf = blob.asByteBuffer().asFloatBuffer();
        float[] rawF32 = new float[fbuf.remaining()];
        fbuf.get(rawF32);
        for (int i = 0; i < rawF32.length && i < dst.length; i++) {
            float v = rawF32[i];
            if (Float.isNaN(v) || Float.isInfinite(v)) {
                throw new IllegalStateException("[c2me-wgpu] Bad density value at index " + i + ": " + v);
            }
            dst[i] = (double) v;
        }
    }

}
