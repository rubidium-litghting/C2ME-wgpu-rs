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

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Correction rules for {@code external_threads}. These run without a Fabric
 * loader because {@link Config#clamp()} does not touch the filesystem; only
 * reading or writing the file does.
 */
class ConfigTest {

    @BeforeEach
    void drainPendingCorrections() {
        while (Config.pollCorrection() != null) {
            // Nothing should carry over between tests.
        }
    }

    @Test
    void negativeExternalThreadsBecomesAutomaticWithOneWarning() {
        Config cfg = new Config();
        cfg.externalThreads = -3;

        assertTrue(cfg.clamp(), "a negative value must be reported as changed");
        assertEquals(0, cfg.externalThreads);

        Config.Correction correction = Config.pollCorrection();
        assertNotNull(correction, "the correction must be surfaced to the player");
        assertEquals("c2me-opts-accel-wgpu.notify.external_threads_negative", correction.key());
        assertEquals(-3, correction.args()[0]);
        assertNull(Config.pollCorrection(), "exactly one notice per correction");
    }

    @Test
    void externalThreadsAboveTheCapIsClampedSilently() {
        Config cfg = new Config();
        cfg.externalThreads = Config.MAX_EXTERNAL_THREADS + 1;

        assertTrue(cfg.clamp());
        assertEquals(Config.MAX_EXTERNAL_THREADS, cfg.externalThreads);
        assertNull(Config.pollCorrection(),
                "only negative values are worth a chat warning");
    }

    @Test
    void validExternalThreadsIsLeftAlone() {
        Config cfg = new Config();
        // Guard against a single-core test machine, where even 2 exceeds the
        // machine-derived ceiling.
        int valid = Math.min(2, Config.MAX_EXTERNAL_THREADS);
        cfg.externalThreads = valid;

        assertFalse(cfg.clamp());
        assertEquals(valid, cfg.externalThreads);
        assertNull(Config.pollCorrection());
    }

    @Test
    void automaticIsLeftAlone() {
        Config cfg = new Config();
        cfg.externalThreads = 0;

        assertFalse(cfg.clamp());
        assertEquals(0, cfg.externalThreads);
        assertNull(Config.pollCorrection());
    }
}
