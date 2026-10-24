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

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import com.rubidiumlighting.c2me.opts.accel.wgpu.WgpuThinBinding;
import com.rubidiumlighting.c2me.opts.accel.wgpu.config.Config;
import com.rubidiumlighting.c2me.opts.accel.wgpu.config.ConfigOption;
import com.terraformersmc.modmenu.api.ConfigScreenFactory;
import com.terraformersmc.modmenu.api.ModMenuApi;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.gui.screen.Screen;
import net.minecraft.client.gui.widget.ButtonWidget;
import net.minecraft.client.gui.widget.ClickableWidget;
import net.minecraft.client.gui.widget.CyclingButtonWidget;
import net.minecraft.client.gui.widget.TextFieldWidget;
import net.minecraft.text.Text;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * ModMenu config screen, generated from the {@link ConfigOption} annotations on
 * {@link Config}.
 *
 * <p>ModMenu is a soft dependency: Fabric only instantiates this class when
 * the mod is present, and the JSON file stays fully usable without it.
 *
 * <p>There is deliberately no per-setting code here. The screen walks
 * {@code Config}'s fields once, sorts them by {@link ConfigOption#order()}, and
 * builds the widget named by {@link ConfigOption#kind()}. Adding a setting
 * therefore means adding one annotated field — the GUI cannot drift out of
 * sync with the file.
 *
 * <p>Built from vanilla widgets; no Cloth Config or other GUI library.
 */
public class WgpuModMenuApi implements ModMenuApi {

    @Override
    public ConfigScreenFactory<?> getModConfigScreenFactory() {
        return WgpuConfigScreen::new;
    }

    /**
     * Shared plumbing for the main and debug screens: field discovery,
     * widget construction and value writing.
     *
     * <p>Both screens differ only in which fields they show and their footer
     * buttons, so the per-kind widget code lives here once. The layout is a
     * simple two-column form — label on the left, control on the right — since
     * the field count is small enough that a scrolling layout engine would be
     * more machinery than the problem deserves.
     */
    private abstract static class WgpuOptionScreen extends Screen {

        protected final Screen parent;
        protected final Config cfg;

        protected static final int ROW_H = 20;
        protected static final int ROW_GAP = 6;
        protected static final int TOP = 36;
        protected static final int WIDTH = 220;
        /** Gutter between the label column and the control column. */
        protected static final int GAP = 8;

        protected WgpuOptionScreen(Text title, Screen parent) {
            super(title);
            this.parent = parent;
            // Re-read the file every time the screen opens. The config is
            // cached at startup, so a file edited while the game is running
            // would otherwise show stale values here — confusing in itself,
            // and it would also make the "did the device settings change"
            // comparison use the wrong baseline.
            Config.reload();
            this.cfg = Config.get();
        }

        /** @return true for fields that belong on this screen. */
        protected abstract boolean accepts(ConfigOption opt, String serializedName);

        /** Called after widgets are laid out, to add footer buttons. */
        protected abstract void addFooter(int centerX, int width, int footerY);

        protected int contentTop() {
            return TOP;
        }

        @Override
        protected void init() {
            int cx = this.width / 2;
            int left = cx - WIDTH / 2;
            int y = contentTop();

            for (Field f : sortedFields()) {
                ConfigOption opt = f.getAnnotation(ConfigOption.class);
                String key = serializedName(f);
                if (!accepts(opt, key)) {
                    continue;
                }
                ClickableWidget widget = buildWidget(f, opt, left, y);
                if (widget != null) {
                    this.addDrawableChild(widget);
                    y += ROW_H + ROW_GAP;
                }
            }

            addFooter(cx, WIDTH, this.height - 28);
        }

        /** Annotated fields, ordered by {@link ConfigOption#order()}. */
        private static List<Field> sortedFields() {
            List<Field> fields = new ArrayList<>();
            for (Field f : Config.class.getFields()) {
                if (f.isAnnotationPresent(ConfigOption.class)) {
                    fields.add(f);
                }
            }
            fields.sort(Comparator.comparingInt(f -> f.getAnnotation(ConfigOption.class).order()));
            return fields;
        }

        /** The JSON key, which is also what decides the screen for debug options. */
        private static String serializedName(Field f) {
            var sn = f.getAnnotation(com.google.gson.annotations.SerializedName.class);
            return sn != null ? sn.value() : f.getName();
        }

        /**
         * One row: a left-aligned label plus the control for this field's kind.
         *
         * @return the control, or {@code null} if the field's type does not
         *         match its declared kind (a programming error; skipping the
         *         row beats taking the screen down).
         */
        private ClickableWidget buildWidget(Field f, ConfigOption opt, int left, int y) {
            int labelW = 100;
            int ctrlX = left + labelW + GAP;
            int ctrlW = WIDTH - labelW - GAP;
            try {
                return switch (opt.kind()) {
                    // `omitKeyText` drops the widget's own "<key>: " prefix.
                    // The label is drawn separately to the left, so letting the
                    // button compose its own key would render a stray ": ".
                    case BOOL -> CyclingButtonWidget.onOffBuilder(f.getBoolean(cfg))
                            .omitKeyText()
                            .build(ctrlX, y, ctrlW, ROW_H, Text.empty(),
                                    (btn, v) -> setField(f, v));
                    case TRI_STATE -> {
                        Boolean cur = (Boolean) f.get(cfg);
                        yield CyclingButtonWidget.<Integer>builder(v -> Text.translatable(
                                        switch (v) {
                                            case 1 -> "c2me-opts-accel-wgpu.config.tristate.on";
                                            case 0 -> "c2me-opts-accel-wgpu.config.tristate.off";
                                            default -> "c2me-opts-accel-wgpu.config.tristate.auto";
                                        }))
                                .values(List.of(-1, 1, 0))
                                .initially(cur == null ? -1 : (cur ? 1 : 0))
                                .omitKeyText()
                                .build(ctrlX, y, ctrlW, ROW_H, Text.empty(),
                                        (btn, v) -> setField(f, v == -1 ? null : (v == 1)));
                    }
                    case GPU_INDEX -> buildGpuWidget(f, ctrlX, y, ctrlW);
                    case BACKEND -> buildBackendWidget(f, ctrlX, y, ctrlW);
                    case STRING -> new TextFieldWidget(this.textRenderer, ctrlX, y, ctrlW, ROW_H,
                            Text.empty()) {{
                        setMaxLength(512);
                        setText((String) getFieldQuietly(f));
                        setChangedListener(v -> setField(f, v));
                    }};
                };
            } catch (ReflectiveOperationException | ClassCastException e) {
                org.slf4j.LoggerFactory.getLogger("c2me-wgpu")
                        .warn("[c2me-wgpu] cannot build widget for {}", f.getName(), e);
                return null;
            }
        }

        private ClickableWidget buildGpuWidget(Field f, int x, int y, int w)
                throws ReflectiveOperationException {
            List<Integer> indices = new ArrayList<>();
            List<String> names = new ArrayList<>();
            indices.add(-1);
            names.add(Text.translatable("c2me-opts-accel-wgpu.config.gpu.auto").getString());
            String json = WgpuThinBinding.listGpusCached(cfg.backend);
            if (json != null) {
                try {
                    JsonArray arr = JsonParser.parseString(json).getAsJsonArray();
                    for (JsonElement el : arr) {
                        var o = el.getAsJsonObject();
                        int idx = o.get("index").getAsInt();
                        indices.add(idx);
                        names.add(idx + ": " + o.get("name").getAsString());
                    }
                } catch (Throwable ignored) {
                    // Unreadable list: the "automatic" entry alone still works.
                }
            }
            int cur = f.getInt(cfg);
            int pos = indices.indexOf(cur);
            return CyclingButtonWidget.<Integer>builder(
                            v -> Text.literal(names.get(Math.max(0, indices.indexOf(v)))))
                    .values(indices)
                    .initially(pos >= 0 ? cur : -1)
                    .omitKeyText()
                    .build(x, y, w, ROW_H, Text.empty(),
                            (btn, v) -> setField(f, v));
        }

        /**
         * Backend picker: automatic plus every backend that actually has an
         * adapter. Listing only usable ones mirrors the GPU list — offering a
         * backend the machine cannot run would just fail on rebuild.
         */
        private ClickableWidget buildBackendWidget(Field f, int x, int y, int w)
                throws ReflectiveOperationException {
            List<Integer> values = new ArrayList<>();
            List<String> names = new ArrayList<>();
            values.add(-1);
            names.add(Text.translatable("c2me-opts-accel-wgpu.config.backend.auto").getString());
            // Static per-platform list: querying wgpu would have to build a
            // real GL context inside the game process, which can wedge the
            // driver. See Backends for the details.
            for (var e : com.rubidiumlighting.c2me.opts.accel.wgpu.config.Backends.available()) {
                values.add(e.bits());
                names.add(e.label());
            }
            int cur = f.getInt(cfg);
            int pos = values.indexOf(cur);
            return CyclingButtonWidget.<Integer>builder(
                            v -> Text.literal(names.get(Math.max(0, values.indexOf(v)))))
                    .values(values)
                    .initially(pos >= 0 ? cur : -1)
                    .omitKeyText()
                    .build(x, y, w, ROW_H, Text.empty(),
                            (btn, v) -> setField(f, v));
        }

        private Object getFieldQuietly(Field f) {
            try {
                return f.get(cfg);
            } catch (ReflectiveOperationException e) {
                return "";
            }
        }

        protected void setField(Field f, Object value) {
            try {
                f.set(cfg, value);
            } catch (ReflectiveOperationException e) {
                org.slf4j.LoggerFactory.getLogger("c2me-wgpu")
                        .warn("[c2me-wgpu] could not set {}", f.getName(), e);
            }
        }

        /** Persist and apply. Shared by both screens' "done"/"back" buttons. */
        protected void saveAndApply() {
            cfg.save();
            // Debug flags take effect at once; anything that belongs to the
            // device (GPU choice, pipeline cache, upload path) needs the
            // instance rebuilt, which happens on the next save load.
            try {
                WgpuThinBinding.pushHotFlags();
            } catch (Throwable ignored) {
                // The file was written; a failed push only delays the effect.
            }
        }

        @Override
        public void render(DrawContext ctx, int mouseX, int mouseY, float delta) {
            super.render(ctx, mouseX, mouseY, delta);
            ctx.drawCenteredTextWithShadow(this.textRenderer, this.title,
                    this.width / 2, 16, 0xFFFFFF);

            // Labels are drawn rather than turned into widgets: a vanilla
            // label widget would add a hitbox the player can click for no
            // effect. Positions mirror `buildWidget` exactly.
            int cx = this.width / 2;
            int left = cx - WIDTH / 2;
            int y = contentTop();
            for (Field f : sortedFields()) {
                ConfigOption opt = f.getAnnotation(ConfigOption.class);
                if (!accepts(opt, serializedName(f))) {
                    continue;
                }
                ctx.drawTextWithShadow(this.textRenderer, Text.translatable(opt.label()),
                        left, y + 6, 0xA0A0A0);
                y += ROW_H + ROW_GAP;
            }
        }
    }

    /** The screen ModMenu opens: everything that is not a {@code debug_} field. */
    private static class WgpuConfigScreen extends WgpuOptionScreen {

        WgpuConfigScreen(Screen parent) {
            super(Text.translatable("c2me-opts-accel-wgpu.config.title"), parent);
        }

        @Override
        protected boolean accepts(ConfigOption opt, String serializedName) {
            return !serializedName.startsWith("debug_");
        }

        @Override
        protected void addFooter(int cx, int width, int footerY) {
            // Half-width buttons: "debug options" on the left, "done" on the
            // right, so the primary action stays where vanilla puts it.
            int half = (width - GAP) / 2;
            int left = cx - width / 2;
            this.addDrawableChild(ButtonWidget.builder(
                            Text.translatable("c2me-opts-accel-wgpu.config.debug_button"),
                            b -> {
                                saveAndApply();
                                if (this.client != null) {
                                    this.client.setScreen(new WgpuDebugScreen(this));
                                }
                            })
                    .dimensions(left, footerY, half, ROW_H)
                    .build());
            this.addDrawableChild(ButtonWidget.builder(
                            Text.translatable("gui.done"), b -> this.close())
                    .dimensions(left + half + GAP, footerY, half, ROW_H)
                    .build());
        }

        @Override
        public void close() {
            saveAndApply();
            if (this.client != null) {
                this.client.setScreen(parent);
            }
        }

        @Override
        public void render(DrawContext ctx, int mouseX, int mouseY, float delta) {
            super.render(ctx, mouseX, mouseY, delta);
            // Settings are split by when they apply, and the difference is not
            // guessable from the controls: debug flags take effect now, while
            // device settings need the instance rebuilt, which happens when a
            // save is reloaded. Saying so here is cheaper than a bug report
            // asking why the GPU choice did nothing.
            ctx.drawCenteredTextWithShadow(this.textRenderer,
                    Text.translatable("c2me-opts-accel-wgpu.config.hint"),
                    this.width / 2, this.height - 44, 0x808080);
        }
    }

    /** Diagnostics: every {@code debug_} field, including the dump writers. */
    private static class WgpuDebugScreen extends WgpuOptionScreen {

        WgpuDebugScreen(Screen parent) {
            super(Text.translatable("c2me-opts-accel-wgpu.config.debug_title"), parent);
        }

        @Override
        protected boolean accepts(ConfigOption opt, String serializedName) {
            return serializedName.startsWith("debug_");
        }

        @Override
        protected int contentTop() {
            // Extra room for the explanatory line under the title.
            return TOP + 8;
        }

        @Override
        protected void addFooter(int cx, int width, int footerY) {
            this.addDrawableChild(ButtonWidget.builder(
                            Text.translatable("gui.back"), b -> this.close())
                    .dimensions(cx - width / 2, footerY, width, ROW_H)
                    .build());
        }

        @Override
        public void close() {
            // Saving here too: the player may edit a dump path and go straight
            // back without touching "done" on the other screen.
            saveAndApply();
            if (this.client != null) {
                this.client.setScreen(parent);
            }
        }

        @Override
        public void render(DrawContext ctx, int mouseX, int mouseY, float delta) {
            super.render(ctx, mouseX, mouseY, delta);
            ctx.drawCenteredTextWithShadow(this.textRenderer,
                    Text.translatable("c2me-opts-accel-wgpu.config.debug_hint"),
                    this.width / 2, TOP - 6, 0x808080);
        }
    }
}
