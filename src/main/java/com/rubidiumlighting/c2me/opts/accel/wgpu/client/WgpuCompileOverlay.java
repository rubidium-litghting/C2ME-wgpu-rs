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

package com.rubidiumlighting.c2me.opts.accel.wgpu.client;

import com.rubidiumlighting.c2me.opts.accel.wgpu.WgpuThinBinding;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.text.Text;
import net.minecraft.util.Util;
import net.minecraft.util.math.MathHelper;

/**
 * Full-screen compile-progress overlay.
 *
 * <p>The first chunk of a dimension compiles its shaders on a C2ME worker
 * thread; on a slow device that can take seconds, during which the client
 * looks frozen. Chat messages cannot help here: the terrain-download screen
 * covers the HUD. So the overlay is drawn from {@code GameRenderer.render}
 * TAIL, i.e. <em>after</em> the world, HUD and any open screen, but still
 * inside the frame's framebuffer.
 *
 * <p>Vanilla draws its profiler pie in exactly this position with a fresh
 * {@link DrawContext} ({@code MinecraftClient.render}), which is why this is
 * safe without touching the render pipeline state.
 *
 * <p>Polling, not callbacks: see {@code rust/src/world/progress.rs} for why
 * the Rust side exposes a pull-only snapshot. One FFM call per frame while a
 * compile is active is negligible; when nothing compiles the first call
 * returns immediately and no draw context is built.
 */
public final class WgpuCompileOverlay {

    private WgpuCompileOverlay() {}

    private static boolean active;
    private static long startedAtMs;
    private static int done;
    private static int total;

    /** Set once if the overlay ever throws, so a broken frame cannot repeat. */
    private static boolean broken;

    /**
     * Poll the native snapshot and report whether this frame should draw.
     *
     * <p>Called from the render thread. Never throws: an unavailable native
     * library or a bad handle just means "nothing to show".
     */
    public static boolean shouldDraw() {
        if (broken) return false;
        long packed = WgpuThinBinding.compileProgress();
        if (packed < 0) {
            active = false;
            return false;
        }
        boolean nowActive = ((packed >>> 32) & 1L) != 0;
        if (!nowActive) {
            active = false;
            return false;
        }
        if (!active) {
            // First frame of this compile: start the elapsed-time clock.
            active = true;
            startedAtMs = Util.getMeasuringTimeMs();
        }
        done = (int) ((packed >>> 16) & 0xFFFFL);
        total = (int) (packed & 0xFFFFL);
        return true;
    }

    /**
     * Draw the progress panel.
     *
     * <p>Placed near the top edge: that area is empty both on the title screen
     * and in-game, and unlike the centre it does not collide with the
     * terrain-download screen's "Loading terrain..." text or the hotbar.
     */
    public static void draw(MinecraftClient mc, DrawContext ctx) {
        try {
            int sw = mc.getWindow().getScaledWidth();
            int sh = mc.getWindow().getScaledHeight();

            long elapsedSec = Math.max(0L, (Util.getMeasuringTimeMs() - startedAtMs) / 1000L);
            int shownDone = total > 0 ? MathHelper.clamp(done, 0, total) : 0;
            Text label = Text.translatable("c2me-opts-accel-wgpu.overlay.compiling",
                    shownDone, total, elapsedSec);

            int textW = mc.textRenderer.getWidth(label);
            int barW = Math.max(180, textW + 24);
            int x = (sw - barW) / 2;
            int y = Math.max(8, sh / 12);

            // Panel: dark, mostly opaque, so the text stays readable over both
            // the dirt loading background and bright in-game scenes.
            ctx.fill(x - 2, y - 3, x + barW + 2, y + 30, 0xC8101010);
            ctx.fill(x - 2, y - 3, x + barW + 2, y - 2, 0xFF3A3A3A);

            ctx.drawCenteredTextWithShadow(mc.textRenderer, label,
                    x + barW / 2, y + 3, 0xFFFFFFFF);

            int trackY = y + 21;
            int trackX0 = x + 6;
            int trackX1 = x + barW - 6;
            ctx.fill(trackX0, trackY, trackX1, trackY + 5, 0xFF303030);
            if (total > 0 && shownDone > 0) {
                int fillX = trackX0 + (trackX1 - trackX0) * shownDone / total;
                ctx.fill(trackX0, trackY, fillX, trackY + 5, 0xFF66BB6A);
            }
        } catch (Throwable t) {
            broken = true;
            org.slf4j.LoggerFactory.getLogger("c2me-wgpu")
                    .warn("[c2me-wgpu] compile overlay disabled after a render error", t);
        }
    }
}
