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

package com.rubidiumlighting.c2me.opts.accel.wgpu.mixin.client;

import com.rubidiumlighting.c2me.opts.accel.wgpu.client.WgpuCompileOverlay;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.render.GameRenderer;
import net.minecraft.client.render.RenderTickCounter;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Draws the shader-compile overlay above everything else in the frame.
 *
 * <p>{@code GameRenderer.render} is the last thing the client renders before
 * the framebuffer is blitted to the window, and it already includes the HUD
 * and the current {@code Screen}. Injecting at TAIL therefore puts the
 * panel above the terrain-download screen as well as above the chat HUD —
 * which is the case that matters, because the first compile happens while
 * that screen is up.
 */
@Mixin(GameRenderer.class)
public abstract class MixinGameRendererOverlay {

    @Inject(method = "render", at = @At("TAIL"))
    private void c2me$drawCompileOverlay(RenderTickCounter tickCounter, boolean tick,
                                        CallbackInfo ci) {
        MinecraftClient mc = MinecraftClient.getInstance();
        if (mc == null || mc.getWindow() == null) return;
        if (!WgpuCompileOverlay.shouldDraw()) return;
        // Same pattern vanilla uses for the profiler pie: a throwaway context
        // drawn on top of the frame that was just produced.
        DrawContext ctx = new DrawContext(mc, mc.getBufferBuilders().getEntityVertexConsumers());
        WgpuCompileOverlay.draw(mc, ctx);
        ctx.draw();
    }
}
