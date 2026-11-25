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

import com.ishland.c2me.base.mixin.access.IThreadedAnvilChunkStorage;
import com.ishland.c2me.rewrites.chunksystem.common.ChunkLoadingContext;
import com.ishland.c2me.rewrites.chunksystem.common.ChunkState;
import com.ishland.flowsched.scheduler.ItemHolder;
import com.rubidiumlighting.c2me.opts.accel.wgpu.config.Config;
import com.rubidiumlighting.c2me.opts.accel.wgpu.mixin.ChunkAccessor;
import com.rubidiumlighting.c2me.opts.accel.wgpu.mixin.NoiseChunkGeneratorInvoker;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.crash.CrashException;
import net.minecraft.util.collection.BoundedRegionArray;
import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.chunk.AbstractChunkHolder;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.chunk.ChunkGenerationContext;
import net.minecraft.world.chunk.ChunkGenerationStep;
import net.minecraft.world.chunk.ChunkGenerationSteps;
import net.minecraft.world.ChunkRegion;
import net.minecraft.world.chunk.ChunkStatus;
import net.minecraft.world.gen.StructureAccessor;
import net.minecraft.world.gen.chunk.Blender;
import net.minecraft.world.gen.chunk.ChunkGenerator;
import net.minecraft.world.gen.chunk.ChunkNoiseSampler;
import net.minecraft.world.gen.chunk.NoiseChunkGenerator;
import net.minecraft.world.gen.noise.NoiseConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The sampling context vanilla builds for the {@code NOISE} step, rebuilt from
 * a {@link ChunkLoadingContext}.
 *
 * <p>Vanilla assembles four things inside {@code ChunkGenerating.populateNoise}
 * from {@code ChunkGenerationContext} plus the generation step:
 *
 * <ol>
 *   <li>{@code BoundedRegionArray} over the step's dependency radius, filled
 *       from the chunk loading manager ({@code ChunkLoader.create});</li>
 *   <li>a {@link ChunkRegion} around the chunk;</li>
 *   <li>{@code Blender.getBlender(region)};</li>
 *   <li>{@code world.getStructureAccessor().forRegion(region)}.</li>
 * </ol>
 *
 * <p>This class performs the same four steps from the scheduler side, using
 * {@code theChunkSystem().getHolder(..).getUserData().get()} for the region
 * array — the same accessor C2ME's own vanilla delegate uses to expose
 * holders. The generator, world and {@link NoiseConfig} come from
 * {@code IThreadedAnvilChunkStorage.getGenerationContext()}, which is what
 * C2ME hands to every vanilla generation step.
 *
 * <p>Nothing is sampled here. The point is to prove that the status can hold
 * the context <em>before</em> {@code populateNoise} runs, which is the
 * prerequisite for evaluating one batch against one shared context.
 *
 * <p>Only the {@code debug_status_sampler} probe calls this today. It is off
 * by default and every failure degrades to the vanilla path.
 */
public final class StatusNoiseContext {

    private static final Logger LOGGER = LoggerFactory.getLogger("c2me-wgpu-status");

    private static final AtomicBoolean WARNED = new AtomicBoolean();

    /** Samplers the status observed (built, or already cached on the chunk).
     *  Weak keys: the chunk owns the sampler, so entries vanish with it. */
    private static final Map<ChunkNoiseSampler, Boolean> OBSERVED =
            Collections.synchronizedMap(new WeakHashMap<>());

    private static final AtomicLong CONTEXTS = new AtomicLong();
    private static final AtomicLong SAMPLERS_BUILT = new AtomicLong();
    private static final AtomicLong POPULATE_CALLS = new AtomicLong();
    private static final AtomicLong POPULATE_MATCHED = new AtomicLong();

    private final ChunkRegion region;
    private final Blender blender;
    private final NoiseConfig noiseConfig;
    private final StructureAccessor structureAccessor;
    private final ChunkGenerator generator;
    private final int radius;
    /** Region slots the scheduler had no holder for (vanilla's outer ring). */
    private final int missing;

    private StatusNoiseContext(ChunkRegion region, Blender blender,
                               NoiseConfig noiseConfig,
                               StructureAccessor structureAccessor,
                               ChunkGenerator generator, int radius, int missing) {
        this.region = region;
        this.blender = blender;
        this.noiseConfig = noiseConfig;
        this.structureAccessor = structureAccessor;
        this.generator = generator;
        this.radius = radius;
        this.missing = missing;
    }

    /**
     * Build the context for {@code chunk} without sampling it.
     *
     * @return the context, or {@code null} when the dimension does not use a
     *         {@link NoiseChunkGenerator} (nothing to pre-build).
     */
    public static StatusNoiseContext create(ChunkLoadingContext ctx, Chunk chunk) {
        ChunkGenerationContext generation =
                ((IThreadedAnvilChunkStorage) ctx.tacs()).getGenerationContext();
        ChunkGenerator generator = generation.generator();
        if (!(generator instanceof NoiseChunkGenerator)) {
            return null;
        }
        ServerWorld world = generation.world();
        ChunkPos pos = chunk.getPos();

        // Same step and radius ChunkLoader.create uses for NOISE: the
        // accumulated dependency level of EMPTY. For NOISE that is 9, one ring
        // wider than the direct dependency radius (8) our status declares.
        ChunkGenerationStep step = ChunkGenerationSteps.GENERATION.get(ChunkStatus.NOISE);
        int radius = step.getAdditionalLevel(ChunkStatus.EMPTY);
        int[] missing = new int[1];
        BoundedRegionArray<AbstractChunkHolder> chunks = BoundedRegionArray.create(
                pos.x, pos.z, radius, (x, z) -> {
                    ItemHolder<ChunkPos, ChunkState, ChunkLoadingContext, ?> holder =
                            ctx.theChunkSystem().getHolder(new ChunkPos(x, z));
                    Object userData = holder == null ? null : holder.getUserData().get();
                    if (!(userData instanceof AbstractChunkHolder abstractHolder)) {
                        // Vanilla's ChunkLoader fills this array through
                        // ChunkLoadingManager#acquire, which yields null for a
                        // position that is not loaded. The outer ring is only
                        // needed to size the array: noise sampling touches the
                        // centre chunk, the beardifier radius (8) and the biome
                        // radius (1), all of which the status does declare.
                        missing[0]++;
                        return null;
                    }
                    return abstractHolder;
                });

        ChunkRegion region = new ChunkRegion(world, chunks, step, chunk);
        return new StatusNoiseContext(
                region,
                Blender.getBlender(region),
                world.getChunkManager().getNoiseConfig(),
                world.getStructureAccessor().forRegion(region),
                generator,
                radius,
                missing[0]);
    }

    /**
     * Debug probe: build the context for the chunk the status is about to
     * advance and record the sampler it will hand to {@code populateNoise}.
     * No-op unless {@code debug_status_sampler} is on.
     */
    public static void observe(ChunkLoadingContext ctx) {
        if (!Config.get().debugStatusSampler) {
            return;
        }
        try {
            ChunkState state = ctx.holder().getItem().get();
            if (state == null) {
                return;
            }
            Chunk chunk = state.chunk();
            if (chunk == null) {
                return;
            }
            StatusNoiseContext context = create(ctx, chunk);
            if (context == null) {
                return;
            }
            ChunkNoiseSampler sampler = context.observeSampler(chunk);

            long n = CONTEXTS.incrementAndGet();
            if (n <= 4 || n % 512 == 0) {
                LOGGER.info("[c2me-wgpu-status] context at {}: radius={}, missing={}, "
                                + "needsBlending={}, sampler={}, aquifer={}, contexts={}",
                        chunk.getPos(), context.radius, context.missing,
                        // 2 = BiomeCoords.toChunk(BLENDING_BIOME_DISTANCE_THRESHOLD + 3),
                        // the check Blender itself performs.
                        context.region.needsBlending(chunk.getPos(), 2),
                        sampler == null ? "absent" : "present",
                        sampler != null && sampler.getAquiferSampler() != null,
                        n);
            }
        } catch (Throwable t) {
            if (WARNED.compareAndSet(false, true)) {
                // ChunkRegion#getChunk throws a CrashException carrying the
                // request details (chunk, requested status, distance). Log
                // them: they say exactly which neighbour the status-side
                // context needed but the scheduler had not advanced yet.
                String details = "";
                if (t instanceof CrashException ce) {
                    details = "\n" + ce.getReport().getMessage()
                            + "\n" + ce.getReport().getCauseAsString();
                }
                LOGGER.warn("[c2me-wgpu-status] status sampling context unavailable; "
                        + "vanilla builds it in populateNoise as usual" + details, t);
            }
        }
    }

    /** Return the chunk's cached sampler, creating it with this context's
     *  inputs if the chunk has none yet (in practice BIOMES already made one). */
    private ChunkNoiseSampler observeSampler(Chunk chunk) {
        ChunkNoiseSampler existing = ((ChunkAccessor) (Object) chunk).getChunkNoiseSampler();
        if (existing == null) {
            existing = chunk.getOrCreateChunkNoiseSampler(cs ->
                    ((NoiseChunkGeneratorInvoker) (Object) generator)
                            .callCreateChunkNoiseSampler(cs, structureAccessor, blender, noiseConfig));
            SAMPLERS_BUILT.incrementAndGet();
        }
        if (existing != null) {
            OBSERVED.put(existing, Boolean.TRUE);
        }
        return existing;
    }

    /**
     * Debug probe: called from {@code populateNoise}. Reports whether the
     * sampler that call is about to use is the one the status observed, i.e.
     * whether the status-side context is the same state carrier vanilla ends
     * up with. Aggregated; a few lines per run, not one per chunk.
     */
    public static void notePopulateNoise(Chunk chunk) {
        if (!Config.get().debugStatusSampler || chunk == null) {
            return;
        }
        ChunkNoiseSampler sampler = ((ChunkAccessor) (Object) chunk).getChunkNoiseSampler();
        boolean matched = sampler != null && OBSERVED.containsKey(sampler);
        long total = POPULATE_CALLS.incrementAndGet();
        if (matched) {
            POPULATE_MATCHED.incrementAndGet();
        }
        if (total <= 4 || total % 512 == 0) {
            LOGGER.info("[c2me-wgpu-status] populateNoise at {}: sampler {}, "
                            + "matched {}/{} (built by status: {})",
                    chunk.getPos(), matched ? "is the status one" : "differs",
                    POPULATE_MATCHED.get(), total, SAMPLERS_BUILT.get());
        }
    }
}
