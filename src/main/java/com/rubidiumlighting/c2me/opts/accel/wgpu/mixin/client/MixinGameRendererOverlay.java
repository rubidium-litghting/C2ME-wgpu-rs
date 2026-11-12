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
 * <p>{@code GameRenderer.render} already contains the world, the HUD and the
 * current {@code Screen}, so drawing at the end of it puts the panel above
 * the terrain-download screen — which is the case that matters, because the
 * first compile happens while that screen is up.
 *
 * <p><b>Not TAIL.</b> The method ends with
 * {@code DrawContext.draw(); modelViewStack.popMatrix(); applyModelViewMatrix();}
 * — i.e. by the time TAIL runs, the GUI matrix (the {@code translate(0,0,
 * -11000)} vanilla pushes for 2D drawing) has already been popped while the
 * orthographic projection with near=1000 is still bound. Quads at {@code z=0}
 * then sit behind the near plane and are clipped: the overlay draws nothing
 * and raises no error. Injecting on the single {@code DrawContext.draw()}
 * call keeps the GUI matrix live, and vanilla's own {@code draw()} flushes
 * the vertices we queued into the shared immediate buffer.
 */
@Mixin(GameRenderer.class)
public abstract class MixinGameRendererOverlay {

    @Inject(method = "render",
            at = @At(value = "INVOKE",
                     target = "Lnet/minecraft/client/gui/DrawContext;draw()V"))
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
