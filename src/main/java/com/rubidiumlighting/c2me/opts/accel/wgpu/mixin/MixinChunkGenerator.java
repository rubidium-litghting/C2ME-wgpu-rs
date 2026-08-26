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

import com.rubidiumlighting.c2me.opts.accel.wgpu.NoiseRouterBlobifier;
import com.rubidiumlighting.c2me.opts.accel.wgpu.WgpuThinBinding;
import net.minecraft.core.Holder;
import net.minecraft.server.level.WorldGenRegion;
import net.minecraft.world.level.StructureManager;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.levelgen.NoiseBasedChunkGenerator;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.NoiseRouter;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.blending.Blender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.util.concurrent.CompletableFuture;

@Mixin(NoiseBasedChunkGenerator.class)
public abstract class MixinChunkGenerator {

    private static final Logger LOGGER = LoggerFactory.getLogger("c2me-wgpu");

    @Inject(method = "fillFromNoise", at = @At("HEAD"), cancellable = true)
    private void onFillFromNoise(Blender blender, RandomState randomState,
                                 StructureManager structureManager, ChunkAccess chunk,
                                 CallbackInfoReturnable<CompletableFuture<ChunkAccess>> cir) {
        if (!WgpuThinBinding.isAvailable()) {
            return;
        }

        try {
            NoiseBasedChunkGenerator self = (NoiseBasedChunkGenerator) (Object) this;

            NoiseRouter router = getNoiseRouter(self);
            if (router == null) {
                LOGGER.warn("[c2me-wgpu] NoiseRouter is null, falling back to CPU");
                return;
            }

            LOGGER.info("[c2me-wgpu] 🔥 onFillFromNoise intercepted! chunk={}", chunk.getPos());

            NoiseRouterBlobifier.ShapeConfig shape = new NoiseRouterBlobifier.ShapeConfig(-64, 384, 4, 4);
            ByteBuffer blob = NoiseRouterBlobifier.blobify(router, shape);
            LOGGER.info("[c2me-wgpu] ✅ blobify completed, blob size={} bytes", blob.remaining());

            // TODO: 调用 WgpuThinBinding 执行 GPU compute 并将结果写回 chunk
            // 目前仅验证序列化链路，不 cancel 原版生成以避免坏档
            // cir.cancel(); // ← 当 GPU 写回逻辑完成后取消注释

        } catch (Exception e) {
            LOGGER.error("[c2me-wgpu] ❌ Failed to use GPU acceleration, falling back to CPU", e);
        }
    }

    private NoiseRouter getNoiseRouter(NoiseBasedChunkGenerator generator) {
        try {
            Field settingsField = generator.getClass().getDeclaredField("settings");
            settingsField.setAccessible(true);
            Holder<NoiseGeneratorSettings> holder = (Holder<NoiseGeneratorSettings>) settingsField.get(generator);
            NoiseGeneratorSettings settings = holder.value();
            Field noiseRouterField = settings.getClass().getDeclaredField("noiseRouter");
            noiseRouterField.setAccessible(true);
            return (NoiseRouter) noiseRouterField.get(settings);
        } catch (Exception e) {
            LOGGER.error("Failed to get NoiseRouter", e);
            return null;
        }
    }
}
