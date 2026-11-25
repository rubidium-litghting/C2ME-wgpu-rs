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
import io.reactivex.rxjava3.core.Completable;
import net.minecraft.server.world.ServerChunkLoadingManager;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.collection.BoundedRegionArray;
import net.minecraft.util.crash.CrashException;
import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.ChunkRegion;
import net.minecraft.world.chunk.AbstractChunkHolder;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.chunk.ChunkGenerationContext;
import net.minecraft.world.chunk.ChunkGenerationStep;
import net.minecraft.world.chunk.ChunkGenerationSteps;
import net.minecraft.world.chunk.ChunkStatus;
import net.minecraft.world.gen.StructureAccessor;
import net.minecraft.world.gen.StructureTerrainAdaptation;
import net.minecraft.world.gen.chunk.Blender;
import net.minecraft.world.gen.chunk.ChunkGenerator;
import net.minecraft.world.gen.chunk.ChunkNoiseSampler;
import net.minecraft.world.gen.chunk.NoiseChunkGenerator;
import net.minecraft.world.gen.noise.NoiseConfig;
import net.minecraft.world.gen.structure.Structure;
import it.unimi.dsi.fastutil.longs.LongIterator;
import it.unimi.dsi.fastutil.longs.LongSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.concurrent.ConcurrentHashMap;
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
 * <p>{@link #build} performs the same four steps from the scheduler side, then
 * asks the chunk for its {@link ChunkNoiseSampler} (creating it only when
 * {@code BIOMES} left none). The generator, world and {@link NoiseConfig} come
 * from {@code IThreadedAnvilChunkStorage.getGenerationContext()}, which is what
 * C2ME hands to every vanilla generation step; the region array is filled from
 * {@code theChunkSystem().getHolder(..).getUserData().get()}, the same accessor
 * C2ME's own vanilla delegate uses.
 *
 * <h2>Why the build checks readiness instead of taking locks</h2>
 *
 * <p>In practice this status almost never creates the sampler: vanilla's
 * {@code BIOMES} step calls {@code NoiseChunkGenerator.populateBiomes}, which
 * does {@code chunk.getOrCreateChunkNoiseSampler(...)} with the biome-level
 * region, so by {@code NOISE} the chunk already carries one (measured in game:
 * 67 created out of 5632 contexts). {@code Chunk.getOrCreateChunkNoiseSampler}
 * therefore returns the existing instance here and the context adopts it — the
 * region, sampler-creator and accessor are only built because the batch submit
 * will read through them, and because the sampler really is created here when
 * it is absent.
 *
 * <p>Creating the sampler walks the region: {@code StructureWeightSampler}
 * reads the chunk's structure references through {@code ChunkRegion.getChunk},
 * which throws a crash report when a neighbour is not <em>actually</em> at the
 * status the step allows. The scheduler guarantees that every dependency holds
 * a ticket at its status, and the centre chunk itself is past {@code BIOMES},
 * so the read normally succeeds; it failed once in ~7000 observations.
 *
 * <p>Pinning the whole 19×19 region with {@code runTaskWithLockRadius} removes
 * that race but is unusable: an all-or-nothing lock set of 361 positions
 * overlaps every neighbour's own generation task, which serialises worldgen.
 * Measured in game, that halved chunk throughput (CPS median 83 → 36), so this
 * class does the opposite: it takes <em>no</em> locks and instead verifies the
 * positions the build will actually read before it builds anything.
 *
 * <p>Those positions are exactly the ones vanilla reads from the region, and
 * they are few: the chunk itself ({@code StructureWeightSampler} asks for its
 * structure references), each chunk its structure references point at, and — on
 * an old-chunk border — the blending neighbourhood. Checking the whole 19×19
 * instead skipped roughly a quarter of all builds, because a holder whose
 * chunk sits at a later status routinely answers {@code null} from
 * {@code getUncheckedOrNull} for an earlier one: that is why the check asks the
 * same question {@code ChunkRegion.getChunk} asks, and why
 * {@code getActualStatus} is useless here (it reports {@code biomes} for a
 * position that still cannot serve {@code biomes}).
 *
 * <p>A position that is not ready is logged and the build is skipped, leaving
 * the chunk to vanilla, so the crash cannot happen: the failure mode becomes a
 * skipped optimisation with a diagnostic line naming the position, the status
 * it needs and the status it has.
 *
 * <h2>Lifecycle</h2>
 *
 * <p>One context per chunk, cached in {@link Cache} for as long as the chunk is
 * inside this status. Re-entry into {@code upgradeToThis} therefore reuses the
 * finished build instead of rebuilding the region, and eviction happens on
 * {@code postUpgradeToThis} / {@code downgradeFromThis}. A failed or skipped
 * build is not cached, so a later attempt can try again, and generation always
 * continues: the vanilla delegate creates the sampler itself.
 *
 * <p>Only the {@code debug_status_sampler} probe drives this today.
 */
public final class StatusNoiseContext {

    private static final Logger LOGGER = LoggerFactory.getLogger("c2me-wgpu-status");

    /**
     * The step whose context {@code populateNoise} runs with. Everything below
     * (required statuses, region radius) is derived from it rather than from
     * constants, so a vanilla or modded step change cannot silently skew the
     * reconstruction.
     */
    private static final ChunkGenerationStep STEP =
            ChunkGenerationSteps.GENERATION.get(ChunkStatus.NOISE);

    /**
     * Radius of the region vanilla builds for {@code NOISE}:
     * {@code ChunkLoader.create} uses {@code step.getAdditionalLevel(EMPTY)},
     * which is 9 — one ring wider than the direct dependency radius of 8,
     * because BIOMES occupies levels 0 and 1 of the accumulated set.
     *
     * <p>The extra ring is only ever null-filled: {@code ChunkRegion.getChunk}
     * refuses every request at a distance of {@code directDependencies().size()}
     * or more, so those slots are never dereferenced.
     */
    private static final int REGION_RADIUS = STEP.getAdditionalLevel(ChunkStatus.EMPTY);

    /** Status a position at each Chebyshev distance must actually hold before
     *  {@code ChunkRegion.getChunk} will serve it. Level 0 is the chunk itself,
     *  so the array always has at least one entry. */
    private static final List<ChunkStatus> REQUIRED_AT_DISTANCE =
            STEP.directDependencies().getDependencies();

    /**
     * How far the blending path reaches. {@code Blender.getBlender} scans
     * {@code BLENDING_CHUNK_DISTANCE_THRESHOLD} (= 2) chunks around the centre
     * and then reads each of those chunks' eight neighbours through
     * {@code getAdjacentChunksWithNoise}, so one more ring covers every read.
     */
    private static final int BLENDING_READ_RADIUS = 3;

    /** Samplers this class has seen. Weak keys: the chunk owns the sampler, so
     *  entries disappear with it. Only used by the identity check below. */
    private static final Map<ChunkNoiseSampler, Boolean> OBSERVED =
            Collections.synchronizedMap(new WeakHashMap<>());

    private static final AtomicLong CONTEXTS = new AtomicLong();
    private static final AtomicLong SAMPLERS_CREATED = new AtomicLong();
    private static final AtomicLong FAILURES = new AtomicLong();
    private static final AtomicLong NOT_READY = new AtomicLong();
    private static final AtomicLong POPULATE_CALLS = new AtomicLong();
    private static final AtomicLong POPULATE_MATCHED = new AtomicLong();
    private static final AtomicBoolean CACHE_CAP_LOGGED = new AtomicBoolean();

    private final ChunkRegion region;
    private final ChunkNoiseSampler sampler;
    private final int missing;
    private final boolean createdSampler;

    private StatusNoiseContext(ChunkRegion region, ChunkNoiseSampler sampler,
                               int missing, boolean createdSampler) {
        this.region = region;
        this.sampler = sampler;
        this.missing = missing;
        this.createdSampler = createdSampler;
    }

    /**
     * Build the context for {@code chunk}.
     *
     * @return the context, or {@code null} when nothing should be built: the
     *         dimension does not use a {@link NoiseChunkGenerator}, or a
     *         neighbour is not at the status the step requires yet
     */
    private static StatusNoiseContext build(ChunkLoadingContext ctx, Chunk chunk) {
        ChunkGenerationContext generation =
                ((IThreadedAnvilChunkStorage) ctx.tacs()).getGenerationContext();
        ChunkGenerator generator = generation.generator();
        if (!(generator instanceof NoiseChunkGenerator)) {
            return null;
        }
        ServerWorld world = generation.world();
        ChunkPos pos = chunk.getPos();

        int[] missing = new int[1];
        BoundedRegionArray<AbstractChunkHolder> chunks = BoundedRegionArray.create(
                pos.x, pos.z, REGION_RADIUS, (x, z) -> {
                    ItemHolder<ChunkPos, ChunkState, ChunkLoadingContext, ?> holder =
                            ctx.theChunkSystem().getHolder(new ChunkPos(x, z));
                    Object userData = holder == null ? null : holder.getUserData().get();
                    if (!(userData instanceof AbstractChunkHolder abstractHolder)) {
                        // Vanilla's ChunkLoader fills this array through
                        // ChunkLoadingManager#acquire, which yields null for a
                        // position that is not loaded. Only the outer ring can
                        // be absent here: ChunkRegion#getChunk refuses any
                        // request at distance >= 9 anyway, so these nulls are
                        // never dereferenced.
                        missing[0]++;
                        return null;
                    }
                    return abstractHolder;
                });

        ChunkRegion region = new ChunkRegion(world, chunks, STEP, chunk);
        String unready = firstUnready(pos, chunk, region, chunks);
        if (unready != null) {
            noteNotReady(pos, unready);
            return null;
        }

        Blender blender;
        try {
            blender = Blender.getBlender(region);
        } catch (ClassCastException e) {
            // C2ME's Blender mixin replaces the needsBlending call with
            // `((ProtoChunkExtension) region.getChunk(pos)).getNeedBlending()`,
            // so it only accepts its own proto chunks. Reached ~3 times in
            // 17000 builds, always for a batch member whose chunk is still a
            // WorldChunk; by the time vanilla's populateNoise runs it is no
            // longer one, so this is simply "too early" rather than a problem
            // with the chunk. Our own needsBlending check above is not
            // redirected, which is why it can pass and this still throws.
            noteNotReady(pos, "C2ME Blender cannot handle this chunk yet ("
                    + e.getMessage() + ")");
            return null;
        }
        NoiseConfig noiseConfig = world.getChunkManager().getNoiseConfig();
        StructureAccessor structureAccessor = world.getStructureAccessor().forRegion(region);

        ChunkNoiseSampler before = ((ChunkAccessor) (Object) chunk).getChunkNoiseSampler();
        ChunkNoiseSampler sampler = chunk.getOrCreateChunkNoiseSampler(cs ->
                ((NoiseChunkGeneratorInvoker) (Object) generator)
                        .callCreateChunkNoiseSampler(cs, structureAccessor, blender, noiseConfig));
        boolean created = before == null && sampler != null;
        if (sampler != null) {
            OBSERVED.put(sampler, Boolean.TRUE);
        }
        if (created) {
            SAMPLERS_CREATED.incrementAndGet();
        }
        StatusNoiseContext context =
                new StatusNoiseContext(region, sampler, missing[0], created);
        context.logBuilt();
        return context;
    }

    /**
     * Checks the positions creating the sampler will read through
     * {@code ChunkRegion.getChunk}: the chunk itself, the chunks its structure
     * references point at, and the blending neighbourhood when the chunk sits
     * on an old-chunk border.
     *
     * @return {@code null} when all of them can serve the read, otherwise a
     *         one-line description of the first position that cannot
     */
    private static String firstUnready(ChunkPos center, Chunk chunk,
                                       ChunkRegion region,
                                       BoundedRegionArray<AbstractChunkHolder> chunks) {
        String problem = unreadyAt(center, center, chunks);
        if (problem != null) {
            return problem;
        }
        // StructureWeightSampler ignores structures without terrain adaptation,
        // so checking those references would skip work for nothing.
        for (Map.Entry<Structure, LongSet> entry : chunk.getStructureReferences().entrySet()) {
            if (entry.getKey().getTerrainAdaptation() == StructureTerrainAdaptation.NONE) {
                continue;
            }
            LongIterator it = entry.getValue().iterator();
            while (it.hasNext()) {
                problem = unreadyAt(center, new ChunkPos(it.nextLong()), chunks);
                if (problem != null) {
                    return problem;
                }
            }
        }
        // Same radius Blender.getBlender uses for its own decision.
        boolean blending;
        try {
            blending = region.needsBlending(center, 2);
        } catch (ClassCastException e) {
            // C2ME redirects ChunkRegion#needsBlending to
            // `((ProtoChunkExtension) region.getChunk(pos)).getNeedBlending()`,
            // and that cast fails when the chunk is already a WorldChunk
            // instead of one of C2ME's proto chunks. Observed for a handful of
            // batch members (~3 in 17000 builds), never for an anchor. Without
            // the answer we cannot bound what Blender will read, so leave the
            // chunk to vanilla rather than guess.
            return "blending state unavailable (" + e.getClass().getSimpleName()
                    + "; C2ME expects its own proto chunk there)";
        }
        if (blending) {
            for (int dx = -BLENDING_READ_RADIUS; dx <= BLENDING_READ_RADIUS; dx++) {
                for (int dz = -BLENDING_READ_RADIUS; dz <= BLENDING_READ_RADIUS; dz++) {
                    problem = unreadyAt(center,
                            new ChunkPos(center.x + dx, center.z + dz), chunks);
                    if (problem != null) {
                        return problem;
                    }
                }
            }
        }
        return null;
    }

    /** Asks the exact question {@code ChunkRegion.getChunk} will ask. */
    private static String unreadyAt(ChunkPos center, ChunkPos target,
                                    BoundedRegionArray<AbstractChunkHolder> chunks) {
        int distance = center.getChebyshevDistance(target.x, target.z);
        if (distance >= REQUIRED_AT_DISTANCE.size()) {
            // ChunkRegion#getChunk turns this into a crash report, so vanilla
            // never asks; a reference this far out is still worth reporting.
            return target + " (distance " + distance + ") is outside the step's dependency radius";
        }
        ChunkStatus required = REQUIRED_AT_DISTANCE.get(distance);
        AbstractChunkHolder holder = chunks.get(target.x, target.z);
        if (holder == null) {
            return target + " (distance " + distance + ", needs " + required.getId()
                    + ") has no holder";
        }
        if (holder.getUncheckedOrNull(required) == null) {
            ChunkStatus actual = holder.getActualStatus();
            return target + " (distance " + distance + ") needs " + required.getId()
                    + " but cannot serve it (status " + (actual == null ? "absent" : actual.getId()) + ")";
        }
        return null;
    }

    private void logBuilt() {
        long n = CONTEXTS.incrementAndGet();
        if (n > 4 && n % 512 != 0) {
            return;
        }
        LOGGER.info("[c2me-wgpu-status] context at {}: radius={}, missing={}, "
                        + "needsBlending={}, sampler={}, aquifer={}, "
                        + "contexts={}, created={}, notReady={}",
                region.getCenterPos(), REGION_RADIUS, missing,
                // 2 = BiomeCoords.toChunk(BLENDING_BIOME_DISTANCE_THRESHOLD + 3),
                // the check Blender itself performs.
                region.needsBlending(region.getCenterPos(), 2),
                sampler == null ? "absent" : "present",
                sampler != null && sampler.getAquiferSampler() != null,
                n, SAMPLERS_CREATED.get(), NOT_READY.get());
    }

    private static void noteNotReady(ChunkPos pos, String detail) {
        long n = NOT_READY.incrementAndGet();
        // First few attempts, then a sparse sample: enough to see which causes
        // are common without flooding the log when a whole area is behind.
        if (n <= 8 || n % 64 == 0) {
            LOGGER.info("[c2me-wgpu-status] context at {} skipped, region not ready: {} "
                    + "({} total; vanilla builds the sampler in populateNoise)", pos, detail, n);
        }
    }

    private static void noteFailure(Throwable t) {
        long n = FAILURES.incrementAndGet();
        if (n > 3 && n % 256 != 0) {
            return;
        }
        String details = "";
        if (t instanceof CrashException ce) {
            // Should be unreachable now that readiness is checked first; if it
            // ever comes back the report names what was missing.
            details = "\n" + ce.getReport().getMessage()
                    + "\n" + ce.getReport().getCauseAsString();
        }
        LOGGER.warn("[c2me-wgpu-status] status-side sampling context failed "
                + "({} total); vanilla builds it in populateNoise{}", n, details, t);
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
        long matchedTotal = matched
                ? POPULATE_MATCHED.incrementAndGet() : POPULATE_MATCHED.get();
        if (total <= 4 || total % 512 == 0) {
            LOGGER.info("[c2me-wgpu-status] populateNoise at {}: sampler {}, "
                            + "matched {}/{} (created by status: {}, regions not ready: {})",
                    chunk.getPos(), matched ? "is the status one" : "differs",
                    matchedTotal, total, SAMPLERS_CREATED.get(), NOT_READY.get());
        }
    }

    /**
     * Status-side context cache, split per world.
     *
     * <p>{@link WgpuNoiseStatus} is created once while C2ME builds its global
     * status table, so a single instance serves every dimension: the cache has
     * to separate them itself. Buckets are keyed by the world's
     * {@code ServerChunkLoadingManager}, which is identity-stable for a world's
     * whole lifetime, and held weakly so an unloaded world takes its contexts
     * with it.
     *
     * <p>Within a bucket, a context is built once per chunk: the scheduler can
     * enter {@code upgradeToThis} several times for the same chunk, and every
     * entry after the first reuses the finished build instead of rebuilding the
     * region. Entries are evicted when the chunk leaves the status
     * ({@code postUpgradeToThis} / {@code downgradeFromThis}).
     */
    public static final class Cache {
        private static final int MAX_ENTRIES_PER_WORLD = 8192;

        private final Map<ServerChunkLoadingManager, World> worlds =
                Collections.synchronizedMap(new WeakHashMap<>());

        /**
         * Builds the context for {@code ctx} if it is not built already. Runs
         * on the calling scheduler thread, which is the same thread the vanilla
         * delegate would build it on.
         *
         * @return a {@link Completable} that is already complete: nothing here
         *         is asynchronous today, but keeping the shape lets batch
         *         submission wait on a context later without changing callers
         */
        public Completable prepare(ChunkLoadingContext ctx) {
            if (!Config.get().debugStatusSampler || !C2meCompat.verifySamplingContext()) {
                return Completable.complete();
            }
            ServerChunkLoadingManager world = ctx.tacs();
            World bucket;
            synchronized (worlds) {
                bucket = worlds.computeIfAbsent(world, ignored -> new World());
            }
            try {
                bucket.prepare(ctx);
            } catch (Throwable t) {
                // Generation must never stall on the probe; vanilla builds the
                // sampler itself right after this status.
                noteFailure(t);
            }
            return Completable.complete();
        }

        /** @return whether {@code pos} currently has a built context in
         *          {@code world}; the probe uses this to describe a batch */
        public boolean isReady(ServerChunkLoadingManager world, ChunkPos pos) {
            synchronized (worlds) {
                World bucket = worlds.get(world);
                return bucket != null && bucket.entries.containsKey(pos);
            }
        }

        /** Drop the cached context for {@code pos} in {@code world}. */
        public void evict(ServerChunkLoadingManager world, ChunkPos pos) {
            synchronized (worlds) {
                World bucket = worlds.get(world);
                if (bucket != null) {
                    bucket.entries.remove(pos);
                }
            }
        }

        /** Contexts of one world. */
        private static final class World {
            private final ConcurrentHashMap<ChunkPos, StatusNoiseContext> entries =
                    new ConcurrentHashMap<>();
            /**
             * Positions whose build is running on some thread right now.
             * Creating the sampler is not reentrant (it mutates the chunk), and
             * a batch anchor and the member's own status could otherwise race
             * for the same chunk. A second caller simply skips this round: the
             * finished context is there for the next one.
             */
            private final Set<ChunkPos> building = ConcurrentHashMap.newKeySet();

            private void prepare(ChunkLoadingContext ctx) {
                ChunkPos pos = ctx.holder().getKey();
                if (entries.containsKey(pos) || !building.add(pos)) {
                    return;
                }
                try {
                    prepare0(ctx, pos);
                } finally {
                    building.remove(pos);
                }
            }

            private void prepare0(ChunkLoadingContext ctx, ChunkPos pos) {
                ChunkState state = ctx.holder().getItem().get();
                Chunk chunk = state == null ? null : state.chunk();
                if (chunk == null) {
                    return;
                }
                StatusNoiseContext context = build(ctx, chunk);
                if (context == null) {
                    // Not ready (or not a noise dimension): do not cache, so a
                    // later attempt can succeed once the region catches up.
                    return;
                }
                if (entries.size() >= MAX_ENTRIES_PER_WORLD) {
                    // Safety valve, not the normal bound: contexts are evicted
                    // when the chunk leaves the status. Loud once, so a leak is
                    // visible instead of silently growing.
                    if (CACHE_CAP_LOGGED.compareAndSet(false, true)) {
                        LOGGER.warn("[c2me-wgpu-status] status context cache reached "
                                + "{} entries; new contexts are not kept", MAX_ENTRIES_PER_WORLD);
                    }
                    return;
                }
                entries.putIfAbsent(pos, context);
            }
        }
    }
}
