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
        WgpuFallbackNotice.Notice n;
        while ((n = WgpuFallbackNotice.poll()) != null) {
            chat.addMessage(Text.translatable(n.key(), n.args()).formatted(Formatting.YELLOW));
        }

        // Throughput stats. These are only produced when debug_verbose_logging
        // is on, so the "spam" is confined to debug sessions; no throttle here.
        String m;
        while ((m = WgpuFallbackNotice.pollStats()) != null) {
            chat.addMessage(Text.literal(m).formatted(Formatting.GRAY));
        }
    }

    @Inject(method = "joinWorld", at = @At("HEAD"))
    private void c2me$onSaveSwitch(net.minecraft.client.world.ClientWorld world,
                                   net.minecraft.client.gui.screen.DownloadingTerrainScreen.WorldEntryReason reason,
                                   CallbackInfo ci) {
        // Fires before the new world's chunks start generating. That is the
        // one moment where no GPU batch can be in flight, so the instance may
        // be rebuilt safely.
        //
        // This also covers the very first join: a world that was loaded before
        // (previous session) may have left a config on disk that the JVM read
        // at startup, but a player who edited the file while the title screen
        // was up expects it to apply here. `onSaveSwitch` is cheap when the
        // cold settings are unchanged (it only re-reads the file and refreshes
        // the hot flags), so running it unconditionally is simpler and more
        // predictable than special-casing the first join.
        try {
            com.rubidiumlighting.c2me.opts.accel.wgpu.WgpuDimensionRegistry.onSaveSwitch();
        } catch (Throwable t) {
            org.slf4j.LoggerFactory.getLogger("c2me-wgpu")
                    .warn("[c2me-wgpu] save-switch reload failed", t);
        }
    }

    @Inject(method = "joinWorld", at = @At("RETURN"))
    private void c2me$notifyOnJoin(net.minecraft.client.world.ClientWorld world,
                                   net.minecraft.client.gui.screen.DownloadingTerrainScreen.WorldEntryReason reason,
                                   CallbackInfo ci) {
        MinecraftClient mc = (MinecraftClient) (Object) this;
        if (mc.inGameHud == null) return;
        ChatHud chat = mc.inGameHud.getChatHud();
        if (chat == null) return;

        // 1. GPU status
        ModuleEntryPoint.Status st = ModuleEntryPoint.getStatus();
        Text msg = switch (st) {
            case FFM_DISABLED -> prefix()
                    .append(Text.translatable("c2me-opts-accel-wgpu.notify.ffm_disabled").formatted(Formatting.YELLOW))
                    .append(Text.translatable("c2me-opts-accel-wgpu.notify.ffm_disabled.hint").formatted(Formatting.GRAY));
            case NATIVE_UNAVAILABLE -> prefix()
                    .append(Text.translatable("c2me-opts-accel-wgpu.notify.native_unavailable").formatted(Formatting.YELLOW));
            case INIT_FAILED -> prefix()
                    .append(Text.translatable("c2me-opts-accel-wgpu.notify.init_failed",
                            String.valueOf(ModuleEntryPoint.getDetail())).formatted(Formatting.YELLOW));
            case OK -> prefix()
                    .append(Text.translatable("c2me-opts-accel-wgpu.notify.enabled",
                            com.rubidiumlighting.c2me.opts.accel.wgpu.WgpuStatus.gpuName())
                            .formatted(Formatting.GREEN));
            default -> null;
        };
        if (msg != null) chat.addMessage(msg);

        // 2. GC hint (once per session). NOT on Android: the custom Android JVM
        // does not support ZGC (enabling it crashes the whole app on launch),
        // so the desktop-oriented advice would be actively harmful there.
        if (!NativeLoader.IS_ANDROID && !c2me$gcWarned && !c2me$isZgc()) {
            c2me$gcWarned = true;
            chat.addMessage(prefix()
                    .append(Text.translatable("c2me-opts-accel-wgpu.notify.gc_hint").formatted(Formatting.YELLOW))
                    .append(Text.translatable("c2me-opts-accel-wgpu.notify.gc_hint.hint").formatted(Formatting.GRAY)));
        }
    }

    /** The "[c2me-wgpu] " tag. Not translated: it identifies the mod, and
     *  players grep logs/chat for it regardless of their language. */
    private static net.minecraft.text.MutableText prefix() {
        return Text.literal("[c2me-wgpu] ");
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
