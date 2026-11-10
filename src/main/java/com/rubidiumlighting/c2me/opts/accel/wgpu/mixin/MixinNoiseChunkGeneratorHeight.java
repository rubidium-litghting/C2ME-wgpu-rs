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

import com.rubidiumlighting.c2me.opts.accel.wgpu.diag.HeightProbe;

import net.minecraft.world.HeightLimitView;
import net.minecraft.world.Heightmap;
import net.minecraft.world.gen.chunk.NoiseChunkGenerator;
import net.minecraft.world.gen.noise.NoiseConfig;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Times every vanilla height query. See {@link HeightProbe} and
 * DEVELOPMENT.md §14.5.
 *
 * <p>Targets {@code NoiseChunkGenerator} rather than the concrete
 * {@code getHeightOnGround}/{@code getHeightInGround} wrappers: callers reach
 * {@code getHeight} both through those wrappers (structure corner heights) and
 * directly ({@code MineshaftStructure}), and both land here.
 */
@Mixin(NoiseChunkGenerator.class)
public abstract class MixinNoiseChunkGeneratorHeight {

    @Inject(method = "getHeight", at = @At("HEAD"))
    private void c2meWgpu$heightProbeBegin(int x, int z, Heightmap.Type heightmap,
                                           HeightLimitView world, NoiseConfig noiseConfig,
                                           CallbackInfoReturnable<Integer> cir) {
        HeightProbe.begin();
    }

    @Inject(method = "getHeight", at = @At("RETURN"))
    private void c2meWgpu$heightProbeEnd(int x, int z, Heightmap.Type heightmap,
                                         HeightLimitView world, NoiseConfig noiseConfig,
                                         CallbackInfoReturnable<Integer> cir) {
        HeightProbe.end(x, z, heightmap);
    }
}
