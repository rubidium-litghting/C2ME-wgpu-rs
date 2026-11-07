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

package com.rubidiumlighting.c2me.opts.accel.wgpu.chunkstatus;

import com.ishland.c2me.rewrites.chunksystem.common.ChunkLoadingContext;
import com.ishland.c2me.rewrites.chunksystem.common.ChunkState;
import com.ishland.c2me.rewrites.chunksystem.common.NewChunkStatus;
import com.ishland.flowsched.scheduler.Cancellable;
import com.ishland.flowsched.scheduler.ItemHolder;
import com.ishland.flowsched.scheduler.KeyStatusPair;
import io.reactivex.rxjava3.core.Completable;
import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.chunk.ChunkStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Minimal C2ME chunk-system integration, stage 1: prove that a custom
 * {@link NewChunkStatus} can be registered through the official
 * {@code NewStatusHook} extension point and take part in the 1.21.1 chunk
 * pipeline.
 *
 * <p>This status is intentionally <strong>transparent</strong>. It sits
 * between vanilla {@code BIOMES} and vanilla {@code NOISE} and delegates the
 * actual upgrade to C2ME's own {@code VanillaWorldGenerationDelegate} for
 * {@code NOISE}. That delegate eventually calls
 * {@code NoiseChunkGenerator.populateNoise}, which is already intercepted by
 * {@code MixinChunkGenerator} and served by the existing
 * {@link com.rubidiumlighting.c2me.opts.accel.wgpu.WgpuBatchCoordinator}.
 *
 * <p>Why not submit from this status directly? Aquifer and beardifier staging
 * need the vanilla generation step's {@code Blender}, {@code StructureAccessor}
 * and the chunk's already-built {@code ChunkNoiseSampler}. Reproducing that
 * outside the vanilla step is exactly the kind of premature rewrite this
 * validation round is meant to avoid. Delegating keeps the GPU path byte-for-
 * byte identical, so a failure here can only come from the hook/ordering, not
 * from a second generation implementation. A later round (4x4 batching) will
 * take ownership of the async request once the hook is proven.
 *
 * <p>Dependencies are delegated too: the scheduler must hold the same
 * neighbour chunks at {@code STRUCTURE_STARTS} that vanilla {@code NOISE}
 * requires, otherwise the delegated step would run too early.
 */
public final class WgpuNoiseStatus extends NewChunkStatus {
    private static final Logger LOGGER = LoggerFactory.getLogger("c2me-wgpu-status");
    private static final AtomicBoolean ANNOUNCED = new AtomicBoolean();

    /**
     * Resolved lazily on the first scheduler call. It cannot be resolved in the
     * constructor: the constructor runs from {@code NewChunkStatus.<clinit>}
     * while the hook is about to append the vanilla NOISE delegate, so
     * {@code VANILLA_WORLDGEN_PIPELINE[NOISE]} is still null at that moment.
     */
    private static volatile NewChunkStatus vanillaNoise;

    public WgpuNoiseStatus(int ordinal) {
        super(ordinal, ChunkStatus.NOISE);
    }

    private static NewChunkStatus vanillaNoise() {
        NewChunkStatus s = vanillaNoise;
        if (s == null) {
            s = NewChunkStatus.fromVanillaStatus(ChunkStatus.NOISE);
            vanillaNoise = s;
        }
        return s;
    }

    @Override
    public Completable upgradeToThis(ChunkLoadingContext ctx, Cancellable cancellable) {
        NewChunkStatus vanilla = vanillaNoise();
        if (vanilla == null || vanilla == this) {
            // Should be impossible after NewChunkStatus.<clinit>; completing
            // without doing work lets the scheduler advance to the real
            // vanilla NOISE delegate instead of deadlocking.
            LOGGER.warn("[c2me-wgpu-status] vanilla NOISE delegate unavailable");
            return Completable.complete();
        }
        if (ANNOUNCED.compareAndSet(false, true)) {
            LOGGER.info("[c2me-wgpu-status] active: ordinal={}, delegating vanilla NOISE "
                    + "(GPU work stays in the existing populateNoise path)", ordinal());
        }
        return vanilla.upgradeToThis(ctx, cancellable);
    }

    @Override
    public KeyStatusPair<ChunkPos, ChunkState, ChunkLoadingContext>[] getDependencies(
            ItemHolder<ChunkPos, ChunkState, ChunkLoadingContext, ?> holder) {
        NewChunkStatus vanilla = vanillaNoise();
        if (vanilla == null || vanilla == this) {
            return EMPTY_DEPENDENCIES;
        }
        return vanilla.getDependencies(holder);
    }

    @Override
    public Completable postUpgradeToThis(ChunkLoadingContext ctx) {
        return Completable.complete();
    }

    @Override
    public Completable preDowngradeFromThis(ChunkLoadingContext ctx, Cancellable cancellable) {
        return Completable.complete();
    }

    @Override
    public Completable downgradeFromThis(ChunkLoadingContext ctx, Cancellable cancellable) {
        return Completable.complete();
    }

    @Override
    public String toString() {
        return "c2me-wgpu:wgpu_noise";
    }
}
