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

import com.rubidiumlighting.c2me.opts.accel.wgpu.ModuleEntryPoint;
import com.rubidiumlighting.c2me.opts.accel.wgpu.NativeLoader;
import com.rubidiumlighting.c2me.opts.accel.wgpu.WgpuFallbackNotice;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.hud.ChatHud;
import net.minecraft.text.Text;
import net.minecraft.util.Formatting;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Notifies the player (chat HUD) when GPU acceleration can't run, or when the
 * JVM isn't using a low-pause GC. Silent on a healthy setup.
 */
@Mixin(MinecraftClient.class)
public class MixinMinecraftClientNotify {

    private static boolean c2me$gcWarned = false;

    /** Drain GPU→CPU fallback notices queued by worker threads. */
    @Inject(method = "tick", at = @At("HEAD"))
    private void c2me$drainFallbackNotices(CallbackInfo ci) {
        MinecraftClient mc = (MinecraftClient) (Object) this;
        if (mc.inGameHud == null) return;
        ChatHud chat = mc.inGameHud.getChatHud();
        if (chat == null) return;
        String m;
        while ((m = WgpuFallbackNotice.poll()) != null) {
            chat.addMessage(Text.literal(m).formatted(Formatting.YELLOW));
        }
    }

    @Inject(method = "joinWorld", at = @At("RETURN"))
    private void c2me$notifyOnJoin(CallbackInfo ci) {
        MinecraftClient mc = (MinecraftClient) (Object) this;
        if (mc.inGameHud == null) return;
        ChatHud chat = mc.inGameHud.getChatHud();
        if (chat == null) return;

        // 1. GPU status
        ModuleEntryPoint.Status st = ModuleEntryPoint.getStatus();
        Text msg = switch (st) {
            case FFM_DISABLED -> Text.literal("[c2me-wgpu] ")
                    .append(Text.literal("Panama FFM 未启用，GPU 加速不可用。").formatted(Formatting.YELLOW))
                    .append(Text.literal("\n  JVM 参数需加：--enable-native-access=com.rubidiumlighting.c2me.opts.accel.wgpu").formatted(Formatting.GRAY));
            case NATIVE_UNAVAILABLE -> Text.literal("[c2me-wgpu] ")
                    .append(Text.literal("此平台没有可用的原生库，GPU 加速不可用。").formatted(Formatting.YELLOW));
            case INIT_FAILED -> Text.literal("[c2me-wgpu] ")
                    .append(Text.literal("wgpu 初始化失败，已回退 CPU：").formatted(Formatting.YELLOW))
                    .append(Text.literal(String.valueOf(ModuleEntryPoint.getDetail())).formatted(Formatting.GRAY));
            default -> null;
        };
        if (msg != null) chat.addMessage(msg);

        // 2. GC hint (once per session). NOT on Android: the custom Android JVM
        // does not support ZGC (enabling it crashes the whole app on launch),
        // so the desktop-oriented advice would be actively harmful there.
        if (!NativeLoader.IS_ANDROID && !c2me$gcWarned && !c2me$isZgc()) {
            c2me$gcWarned = true;
            chat.addMessage(Text.literal("[c2me-wgpu] ")
                    .append(Text.literal("检测到未使用 ZGC，区块生成可能因 GC 停顿而变慢。").formatted(Formatting.YELLOW))
                    .append(Text.literal("\n  建议加入 JVM 参数：-XX:+UseZGC").formatted(Formatting.GRAY)));
        }
    }

    private static boolean c2me$isZgc() {
        try {
            return java.lang.management.ManagementFactory
                    .getGarbageCollectorMXBeans().stream()
                    .map(java.lang.management.GarbageCollectorMXBean::getName)
                    .anyMatch(n -> n.contains("ZGC"));
        } catch (Throwable t) {
            return true; // don't nag if we can't tell
        }
    }
}
