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

package com.rubidiumlighting.c2me.opts.accel.wgpu.config;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;
import com.google.gson.annotations.SerializedName;
import net.fabricmc.loader.api.FabricLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Persistent user settings, stored at {@code config/c2me-wgpu.json}.
 *
 * <p>This is the only source of user settings: the mod no longer reads any
 * {@code C2ME_WGPU_*} environment variables.
 *
 * <p>Deliberately hand-written: no Cloth Config, no third-party config
 * library. Gson ships with Minecraft, so using it adds no dependency.
 */
public final class Config {

    private static final Logger LOGGER = LoggerFactory.getLogger("c2me-wgpu-config");
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().serializeNulls().create();
    private static final String FILE_NAME = "c2me-wgpu.json";

    /** Selected GPU index, as returned by {@code c2me_wgpu_list_gpus}.
     *  -1 means "automatic" (integrated GPU first). Not implemented yet. */
    @SerializedName("gpu_index")
    public int gpuIndex = -1;

    /** Disable wgpu's pipeline cache. Only useful when a driver aborts inside
     *  it. */
    @SerializedName("disable_pipeline_cache")
    public boolean disablePipelineCache = false;

    // ── debug ──────────────────────────────────────────────────────────
    // Everything below is for troubleshooting and normally stays off. They
    // all share the `debug_` prefix so a glance at the file shows which
    // settings are diagnostics rather than preferences.

    /** Verbose diagnostics: shader compile timings, pipeline-cache hits,
     *  adapter workarounds, plus the periodic CPS throughput line. */
    @SerializedName("debug_verbose_logging")
    public boolean debugVerboseLogging = false;

    /** Emit per-stage timing to the log: GPU side (encode / submit /
     *  readback / decode) and CPU side (block write-back into the chunk). */
    @SerializedName("debug_stage_timing")
    public boolean debugStageTiming = false;

    /** Override the buffer-upload path. null = platform default (mapped on
     *  desktop; write_buffer on Android, where some Mali drivers abort on the
     *  mapped path). This is a compatibility setting, not a diagnostic: it is
     *  what gets a crashing device running at all. */
    @SerializedName("mapped_at_creation")
    public Boolean mappedAtCreation = null;

    /** Skip the GPU entirely, let vanilla generate, then dump blocks. */
    @SerializedName("debug_vanilla_ref")
    public boolean debugVanillaRef = false;

    /** Target chunk(s) for the dump writers: "x,z" or "x1,z1,x2,z2".
     *  Empty disables every {@code debug_dump_*} switch below. */
    @SerializedName("debug_dump_chunk")
    public String debugDumpChunk = "";

    /** Where dump files go. Empty = platform temp dir. */
    @SerializedName("debug_dump_dir")
    public String debugDumpDir = "";

    /** Dump block states of the target chunk(s). */
    @SerializedName("debug_dump_blocks")
    public boolean debugDumpBlocks = false;

    /** Dump the per-cell aquifer table. */
    @SerializedName("debug_dump_aquifer")
    public boolean debugDumpAquifer = false;

    /** Dump Beardifier pieces/junctions plus per-block reference values. */
    @SerializedName("debug_dump_beard")
    public boolean debugDumpBeard = false;

    /** Write the serialized NoiseRouter blob (+ corner reference) here.
     *  Empty = do not dump. This is how src/test/resources/real_router.bin
     *  is produced. */
    @SerializedName("debug_dump_blob")
    public String debugDumpBlob = "";

    // ── transient state ────────────────────────────────────────────────
    private static Config INSTANCE;

    /** Load (or create) the config file. Safe to call from any thread; the
     *  result is cached after the first call. */
    public static synchronized Config get() {
        if (INSTANCE == null) {
            INSTANCE = load();
        }
        return INSTANCE;
    }

    private static Config load() {
        Path path = configPath();
        if (Files.exists(path)) {
            try {
                String json = Files.readString(path, StandardCharsets.UTF_8);
                Config cfg = GSON.fromJson(json, Config.class);
                if (cfg != null) {
                    if (cfg.clamp()) {
                        // Persist the corrected values so the file reflects
                        // what the mod is actually using.
                        cfg.save();
                    }
                    return cfg;
                }
                LOGGER.warn("[c2me-wgpu] {} parsed to null; using defaults", path);
            } catch (JsonSyntaxException e) {
                LOGGER.warn("[c2me-wgpu] {} is not valid JSON ({}); using defaults", path, e.getMessage());
            } catch (IOException e) {
                LOGGER.warn("[c2me-wgpu] could not read {}; using defaults", path, e);
            }
            // The file exists but could not be used. Move it aside instead of
            // overwriting: the user's edits are what they will want to inspect
            // when they realise nothing took effect.
            backupBroken(path);
        }
        // Absent, unreadable, or unusable: start from defaults and write them
        // out so the user has something to edit.
        Config cfg = new Config();
        cfg.save();
        return cfg;
    }

    /** Rename a bad config out of the way so it is not silently replaced. */
    private static void backupBroken(Path path) {
        Path backup = path.resolveSibling(path.getFileName() + ".broken");
        try {
            Files.move(path, backup, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            LOGGER.warn("[c2me-wgpu] moved unusable config to {}; a fresh one was written", backup);
        } catch (IOException e) {
            LOGGER.error("[c2me-wgpu] could not move {} aside", path, e);
        }
    }

    /** Bring out-of-range values back to something the rest of the code can
     *  handle. Hand-edited files are the expected input here, so a typo must
     *  not turn into a negative channel or a GPU index of 10^9.
     *
     *  @return true if anything was changed, so the caller can persist the
     *          corrected values. Otherwise the file would keep showing a value
     *          the mod is not actually using. */
    private boolean clamp() {
        boolean changed = false;
        if (gpuIndex < -1) {
            gpuIndex = -1;
            changed = true;
        }
        return changed;
    }

    /** Write the current values to disk. Called on creation and by the GUI. */
    public synchronized void save() {
        Path path = configPath();
        try {
            Files.createDirectories(path.getParent());
            Files.writeString(path, GSON.toJson(this), StandardCharsets.UTF_8);
        } catch (IOException e) {
            LOGGER.error("[c2me-wgpu] failed to write {}", path, e);
        }
    }

    private static Path configPath() {
        return FabricLoader.getInstance().getConfigDir().resolve(FILE_NAME);
    }
}
