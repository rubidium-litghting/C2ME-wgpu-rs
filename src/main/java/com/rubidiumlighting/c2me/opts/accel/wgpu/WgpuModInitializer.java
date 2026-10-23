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
import net.fabricmc.api.ModInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Creates the config file at startup.
 *
 * <p>{@link Config#get()} is lazy: without this, the file would only appear
 * the first time a world loads. Reading it early also surfaces a malformed
 * file in the log before the player is in-game, instead of the first time
 * terrain generation touches it.
 */
public class WgpuModInitializer implements ModInitializer {

    private static final Logger LOGGER = LoggerFactory.getLogger("c2me-wgpu");

    @Override
    public void onInitialize() {
        Config.get();
        LOGGER.info("[c2me-wgpu] config loaded");

        // Write the GPU list into the config file as a comment. Done here (not
        // when the instance is created) so the comment is present even for
        // players who never load a world, and so the file on disk explains its
        // own `gpu_index` field. `listGpus` only enumerates adapters via a
        // throwaway instance; it does not create a device, so this does not
        // start any GPU work early.
        try {
            Config.updateGpuListing(WgpuThinBinding.listGpus());
        } catch (Throwable t) {
            // Purely cosmetic: the mod works without the comment.
            LOGGER.debug("[c2me-wgpu] could not write GPU listing comment", t);
        }
    }
}
