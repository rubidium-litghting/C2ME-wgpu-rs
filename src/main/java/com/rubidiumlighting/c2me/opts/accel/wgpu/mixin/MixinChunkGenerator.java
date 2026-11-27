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


package com.rubidiumlighting.c2me.opts.accel.wgpu.mixin;

import com.rubidiumlighting.c2me.opts.accel.wgpu.ChunkDumper;
import com.rubidiumlighting.c2me.opts.accel.wgpu.GpuNoiseSubmitter;
import com.rubidiumlighting.c2me.opts.accel.wgpu.GpuNoiseSubmission;
import com.rubidiumlighting.c2me.opts.accel.wgpu.chunkstatus.StatusNoiseContext;
import com.rubidiumlighting.c2me.opts.accel.wgpu.config.Config;

import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.gen.StructureAccessor;
import net.minecraft.world.gen.chunk.Blender;
import net.minecraft.world.gen.chunk.NoiseChunkGenerator;
import net.minecraft.world.gen.noise.NoiseConfig;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import java.util.concurrent.CompletableFuture;

@Mixin(NoiseChunkGenerator.class)
public abstract class MixinChunkGenerator implements GpuNoiseSubmitter {

    @Inject(method = "populateNoise", at = @At("HEAD"))
    private void c2me_wgpu$observeStatusSampler(Blender blender, NoiseConfig noiseConfig,
                                                StructureAccessor structureAccessor, Chunk chunk,
                                                CallbackInfoReturnable<CompletableFuture<Chunk>> cir) {
        // Debug probe for the status-side sampling context: compare the sampler
        // this call will use with the one the NOISE status observed. No-op
        // unless debug_status_sampler is on, and never touches generation.
        StatusNoiseContext.notePopulateNoise(chunk);
    }

    @Inject(method = "populateNoise", at = @At("HEAD"), cancellable = true)
    private void onPopulateNoise(Blender blender, NoiseConfig noiseConfig,
                                 StructureAccessor structureAccessor, Chunk chunk,
                                 CallbackInfoReturnable<CompletableFuture<Chunk>> cir) {
        CompletableFuture<Chunk> submitted = c2me_wgpu$submitGpuNoise(
                blender, noiseConfig, structureAccessor, chunk);
        if (submitted != null) {
            cir.setReturnValue(submitted);
        }
    }

    @Inject(method = "populateNoise", at = @At("RETURN"), cancellable = true)
    private void afterPopulateNoise(Blender blender, NoiseConfig noiseConfig,
                                    StructureAccessor structureAccessor, Chunk chunk,
                                    CallbackInfoReturnable<CompletableFuture<Chunk>> cir) {
        if (!Config.get().debugVanillaRef) {
            return;
        }
        CompletableFuture<Chunk> f = cir.getReturnValue();
        if (f == null) {
            return;
        }
        cir.setReturnValue(f.thenApply(c -> {
            ChunkDumper.maybeDumpBlocks(c, "vanilla");
            return c;
        }));
    }

    /** Delegates to {@link GpuNoiseSubmission}. The mixin itself keeps only the
     *  injections and this entry point, so the submit path has a single home
     *  that is not a mixin class. */
    @Override
    public CompletableFuture<Chunk> c2me_wgpu$submitGpuNoise(
            Blender blender, NoiseConfig noiseConfig,
            StructureAccessor structureAccessor, Chunk chunk) {
        return GpuNoiseSubmission.submit((NoiseChunkGenerator) (Object) this,
                blender, noiseConfig, structureAccessor, chunk);
    }
}
