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
import net.minecraft.world.gen.chunk.ChunkGeneratorSettings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Owns every per-dimension {@link DimState} and the save-switch teardown.
 *
 * <p>This lives outside the mixin package on purpose: methods declared in a
 * {@code @Mixin} class are merged into the target class and cannot be called
 * by their source name from elsewhere (see the note on {@code DimState}). The
 * mixin calls into this class, not the other way round.
 *
 * <p>Teardown order matters and is enforced here: stop each coordinator (which
 * joins its threads and drains in-flight Rust requests), then destroy the
 * world handle, and only then allow the wgpu instance to be rebuilt. A world
 * handle holds pipelines bound to a specific device, so destroying it after
 * the device is gone would be a use-after-free on the GPU side.
 */
public final class WgpuDimensionRegistry {

    private static final Logger LOGGER = LoggerFactory.getLogger("c2me-wgpu");

    /** Per-dimension state, keyed by the dimension's noise-settings singleton.
     *  Each distinct settings (Overworld / LargeBiomes / Amplified / Nether /
     *  End / …) has its own NoiseRouter + shape, hence its own WorldContext. */
    private static final Map<ChunkGeneratorSettings, DimState> DIM_STATES =
            new ConcurrentHashMap<>();

    /** Dimensions whose GPU init already failed. Once here we stay on the
     *  vanilla CPU path and never retry: re-running blobify + shader compile
     *  for every chunk would flood the log and waste CPU. */
    private static final Set<ChunkGeneratorSettings> FAILED_DIMS =
            ConcurrentHashMap.newKeySet();

    private WgpuDimensionRegistry() {}

    public static DimState get(ChunkGeneratorSettings settings) {
        return DIM_STATES.get(settings);
    }

    public static void put(ChunkGeneratorSettings settings, DimState state) {
        DIM_STATES.put(settings, state);
    }

    public static void remove(ChunkGeneratorSettings settings) {
        DIM_STATES.remove(settings);
    }

    /** Serialises creation so two worker threads cannot both build the same
     *  dimension's world handle (which would leak one of them). */
    public static Object creationLock() {
        return DIM_STATES;
    }

    public static boolean hasFailed(ChunkGeneratorSettings settings) {
        return FAILED_DIMS.contains(settings);
    }

    public static void markFailed(ChunkGeneratorSettings settings) {
        FAILED_DIMS.add(settings);
    }

    /**
     * Destroy every dimension's GPU state. Blocking: returns once all
     * coordinators have stopped and all world handles are released.
     *
     * <p>Called on save switch before the instance is rebuilt. Must not run on
     * a coordinator's own completion thread (that thread is joined here); the
     * save-switch path runs on the client thread, which satisfies this.
     */
    public static synchronized void destroyAll() {
        if (DIM_STATES.isEmpty() && FAILED_DIMS.isEmpty()) {
            return;
        }
        for (Map.Entry<ChunkGeneratorSettings, DimState> e : DIM_STATES.entrySet()) {
            DimState ds = e.getValue();
            if (ds == null) continue;
            try {
                ds.coordinator.shutdown();
            } catch (Throwable t) {
                LOGGER.warn("[c2me-wgpu] coordinator shutdown failed on save switch", t);
            }
            if (ds.worldHandle != 0L) {
                try {
                    WgpuThinBinding.worldDestroy(ds.worldHandle);
                } catch (Throwable t) {
                    LOGGER.warn("[c2me-wgpu] world_destroy failed on save switch", t);
                }
            }
        }
        DIM_STATES.clear();
        // A GPU failure may have been transient (e.g. the device that is being
        // replaced); retry on the new instance rather than staying on the CPU
        // path for the rest of the session.
        FAILED_DIMS.clear();
    }

    /**
     * Save-switch entry point: re-read the config and, if the cold settings
     * changed, tear everything down and rebuild the wgpu instance.
     *
     * <p>When the cold settings are unchanged this only refreshes the hot
     * flags, which costs nothing and lets {@code debug_*} toggles apply without
     * rebuilding the device.
     *
     * @return true if a usable GPU instance exists afterwards.
     */
    public static synchronized boolean onSaveSwitch() {
        // The file is the single source of truth for "what the user wants",
        // and `appliedCold` for "what the device is running". Comparing those
        // two avoids the cached Config singleton entirely: the GUI edits that
        // object in place, so using it as either side of the comparison would
        // report "unchanged" no matter what the user picked.
        Config cfg = Config.reload();
        boolean coldChanged = WgpuThinBinding.coldConfigChanged(cfg);

        if (!coldChanged) {
            try {
                WgpuThinBinding.pushHotFlags();
            } catch (Throwable t) {
                LOGGER.warn("[c2me-wgpu] pushing hot flags failed", t);
            }
            return WgpuThinBinding.isAvailable();
        }

        LOGGER.info("[c2me-wgpu] GPU config changed; rebuilding instance "
                + "(gpu_index={}, backend={}, disable_pipeline_cache={}, mapped_at_creation={})",
                cfg.gpuIndex, cfg.backend, cfg.disablePipelineCache, cfg.mappedAtCreation);
        destroyAll();
        try {
            boolean ok = WgpuThinBinding.rebuildFromConfig();
            if (!ok) {
                WgpuFallbackNotice.report("rebuild-failed",
                        "c2me-opts-accel-wgpu.notify.init_failed",
                        "config changed but GPU rebuild failed; see log");
            }
            return ok;
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] instance rebuild failed", t);
            return false;
        }
    }
}
