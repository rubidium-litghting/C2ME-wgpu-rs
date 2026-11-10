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

package com.rubidiumlighting.c2me.opts.accel.wgpu.chunkstatus;

import net.minecraft.world.chunk.ChunkStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Array;
import java.lang.reflect.Method;
import java.lang.reflect.RecordComponent;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Shape check for the C2ME chunk-system API used by {@link WgpuNoiseStatus}.
 *
 * <p>The mod compiles against stubs, so a newer C2ME jar is linked only at
 * runtime. If C2ME renames a record component or a method, the first scheduler
 * call would fail with {@code NoSuchMethodError} in the middle of world
 * generation. This class detects that up front and lets the mixin skip
 * registration, keeping the pre-0.1.7 {@code populateNoise} GPU path working.
 *
 * <p>Everything C2ME-side is reached through {@link Class#forName(String)}:
 * a direct reference in this class would itself throw
 * {@code NoClassDefFoundError} before the check could report anything.
 * Vanilla {@link ChunkStatus} is referenced directly because Loom remaps it.
 */
public final class C2meCompat {
    private static final Logger LOGGER = LoggerFactory.getLogger("c2me-wgpu-status");

    private static final String CONTEXT =
            "com.ishland.c2me.rewrites.chunksystem.common.ChunkLoadingContext";
    private static final String THE_CHUNK_SYSTEM =
            "com.ishland.c2me.rewrites.chunksystem.common.TheChunkSystem";
    private static final String NEW_CHUNK_STATUS =
            "com.ishland.c2me.rewrites.chunksystem.common.NewChunkStatus";
    private static final String CHUNK_STATE =
            "com.ishland.c2me.rewrites.chunksystem.common.ChunkState";
    private static final String ITEM_HOLDER =
            "com.ishland.flowsched.scheduler.ItemHolder";
    private static final String KEY_STATUS_PAIR =
            "com.ishland.flowsched.scheduler.KeyStatusPair";

    /**
     * Record components that {@link WgpuNoiseStatus} reads by name. The order
     * matters because the canonical constructor is called with exactly this
     * argument order.
     */
    private static final String[] CONTEXT_COMPONENTS = {
            "holder", "tacs", "schedulingManager", "theChunkSystem", "dependencies"
    };

    private static Boolean compatible;

    private C2meCompat() {
    }

    /**
     * Verifies the API once per JVM.
     *
     * @return {@code true} when {@link WgpuNoiseStatus} can be registered
     */
    public static synchronized boolean verify() {
        if (compatible == null) {
            String mismatch = mismatch();
            compatible = mismatch == null;
            if (compatible) {
                LOGGER.info("[c2me-wgpu-status] C2ME chunk-system API shape verified");
            } else {
                LOGGER.error("[c2me-wgpu-status] incompatible C2ME chunk-system API; "
                        + "the aligned NOISE status is disabled and world generation "
                        + "continues on the regular populateNoise GPU path. "
                        + "Reason: {}. Update c2me-opts-accel-wgpu after checking "
                        + "the API shape (see DEVELOPMENT.md §13.6).", mismatch);
            }
        }
        return compatible;
    }

    /** @return {@code null} when every checked signature matches */
    private static String mismatch() {
        try {
            Class<?> context = load(CONTEXT);
            Class<?> chunkSystem = load(THE_CHUNK_SYSTEM);
            Class<?> newChunkStatus = load(NEW_CHUNK_STATUS);
            Class<?> chunkState = load(CHUNK_STATE);
            Class<?> itemHolder = load(ITEM_HOLDER);
            Class<?> keyStatusPair = load(KEY_STATUS_PAIR);

            if (!context.isRecord()) {
                return CONTEXT + " is not a record";
            }
            RecordComponent[] components = context.getRecordComponents();
            if (components.length != CONTEXT_COMPONENTS.length) {
                return "ChunkLoadingContext has " + components.length
                        + " components: " + names(components);
            }
            for (int i = 0; i < CONTEXT_COMPONENTS.length; i++) {
                if (!components[i].getName().equals(CONTEXT_COMPONENTS[i])) {
                    return "ChunkLoadingContext component " + i + " is '"
                            + components[i].getName() + "', expected '"
                            + CONTEXT_COMPONENTS[i] + "'";
                }
            }
            if (!components[0].getType().equals(itemHolder)) {
                return "ChunkLoadingContext.holder is "
                        + components[0].getType().getName();
            }
            if (!components[3].getType().equals(chunkSystem)) {
                return "ChunkLoadingContext.theChunkSystem is "
                        + components[3].getType().getName();
            }
            Class<?> dependencyArray = Array.newInstance(keyStatusPair, 0).getClass();
            if (!components[4].getType().equals(dependencyArray)) {
                return "ChunkLoadingContext.dependencies is "
                        + components[4].getType().getName();
            }
            // The record, its components and isRecord() together guarantee the
            // accessors; only the canonical constructor still needs a lookup.
            Class<?>[] constructorTypes = Arrays.stream(components)
                    .map(RecordComponent::getType)
                    .toArray(Class<?>[]::new);
            context.getDeclaredConstructor(constructorTypes);

            Method getHolder = chunkSystem.getMethod("getHolder", Object.class);
            if (!getHolder.getReturnType().equals(itemHolder)) {
                return "TheChunkSystem.getHolder returns "
                        + getHolder.getReturnType().getName();
            }
            Method getItem = itemHolder.getMethod("getItem");
            if (!getItem.getReturnType().equals(AtomicReference.class)) {
                return "ItemHolder.getItem returns "
                        + getItem.getReturnType().getName();
            }
            itemHolder.getMethod("getKey");
            Method reachedStatus = chunkState.getMethod("reachedStatus");
            if (!reachedStatus.getReturnType().equals(ChunkStatus.class)) {
                return "ChunkState.reachedStatus returns "
                        + reachedStatus.getReturnType().getName();
            }

            newChunkStatus.getDeclaredConstructor(int.class, ChunkStatus.class);
            newChunkStatus.getDeclaredMethod(
                    "relativeToAbsoluteDependencies", itemHolder, dependencyArray);
            newChunkStatus.getMethod("fromVanillaStatus", ChunkStatus.class);
            return null;
        } catch (ClassNotFoundException | NoSuchMethodException e) {
            return e.toString();
        } catch (LinkageError e) {
            Throwable cause = e.getCause();
            return cause == null ? e.toString() : e + " caused by " + cause;
        }
    }

    private static String names(RecordComponent[] components) {
        StringBuilder sb = new StringBuilder();
        for (RecordComponent component : components) {
            if (sb.length() > 0) {
                sb.append(", ");
            }
            sb.append(component.getName());
        }
        return sb.toString();
    }

    /**
     * Loads without initializing. This runs from {@code NewChunkStatus.<clinit>}
     * (the hook fires while C2ME is building its status table), so running
     * another C2ME or FlowSched static initializer here could observe a
     * half-built {@code ALL_STATUSES} and deadlock or crash.
     */
    private static Class<?> load(String name) throws ClassNotFoundException {
        return Class.forName(name, false, C2meCompat.class.getClassLoader());
    }
}
