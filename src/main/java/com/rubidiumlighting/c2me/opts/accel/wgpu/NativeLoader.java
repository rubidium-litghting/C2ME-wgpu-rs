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

import io.netty.util.internal.SystemPropertyUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.Linker;
import java.lang.foreign.SymbolLookup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;

public class NativeLoader {

    private static final Logger LOGGER = LoggerFactory.getLogger(NativeLoader.class);

    /** Android's custom JVM reports os.name as "Linux"; detect it explicitly so
     *  we look under natives/android-<arch>/ instead of linux-<arch>/. The two
     *  are NOT ABI-compatible (Android = bionic, desktop Linux = glibc). */
    public static final boolean IS_ANDROID = detectAndroid();
    public static final boolean IS_OHOS = detectOhos();

    public static final String NORMALIZED_ARCH = normalizeArch(SystemPropertyUtil.get("os.arch", ""));
    public static final String NORMALIZED_OS = IS_ANDROID ? "android" : (IS_OHOS ? "ohos" : normalizeOs(SystemPropertyUtil.get("os.name", "")));

    private static final Arena arena = Arena.ofAuto();
    public static final SymbolLookup lookup;
    public static final Linker linker = Linker.nativeLinker();

    static {
        // Truncate the log file as the very first action.
        try {
            java.nio.file.Path logPath = net.fabricmc.loader.api.FabricLoader
                    .getInstance().getGameDir()
                    .resolve("logs").resolve("c2me-wgpu.log");
            java.nio.file.Files.createDirectories(logPath.getParent());
            try (java.io.FileOutputStream fos = new java.io.FileOutputStream(logPath.toFile(), false)) {
                // Truncates to 0 bytes.
            }
        } catch (Throwable t) {
            System.err.println("[c2me-wgpu] Failed to truncate log file: " + t);
        }

        String libName = String.format("natives/%s-%s/%s",
                NORMALIZED_OS, NORMALIZED_ARCH,
                System.mapLibraryName("c2me_wgpu"));
        lookup = load0(libName);
    }

    public static boolean isAvailable() {
        return lookup != null;
    }

    public static String getAvailabilityString() {
        if (lookup != null) {
            return "Available";
        } else {
            return String.format("Unavailable (no native library for %s-%s)", NORMALIZED_OS, NORMALIZED_ARCH);
        }
    }

    private static boolean detectAndroid() {
        try {
            if (System.getenv("ANDROID_ROOT") != null || System.getenv("ANDROID_DATA") != null) {
                return true;
            }
        } catch (Throwable ignored) {}
        try {
            return Files.exists(Path.of("/system/build.prop"));
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static boolean detectOhos() {
        try {
            // Allow explicit override (e.g., running AMCL on HarmonyOS where os.version might be ambiguous)
            String force = SystemPropertyUtil.get("c2me.native.platform", "");
            if (force.contains("ohos")) {
                return true;
            }
            // Some OHOS environments report "OpenHarmony" or "HarmonyOS" in os.version
            String osVersion = SystemPropertyUtil.get("os.version", "").toLowerCase();
            if (osVersion.contains("harmony") || osVersion.contains("ohos")) {
                return true;
            }
        } catch (Throwable ignored) {}
        return false;
    }

    private static SymbolLookup load0(String libName) {
        final String fileName = libName.substring(libName.lastIndexOf('/') + 1);
        java.util.List<Path> candidates = new java.util.ArrayList<>();

        // 1. Build candidate paths in order of preference
        if (Boolean.getBoolean("c2me.wgpu.preserveNative")) {
            // Debug mode: drop it directly in the working directory (.minecraft/versions/xxx)
            candidates.add(Path.of(".", fileName));
        } else {
            // Default: try gameDir cache first
            try {
                Path gameDir = net.fabricmc.loader.api.FabricLoader.getInstance().getGameDir();
                Path nativeCacheDir = gameDir.resolve("c2me-wgpu-natives");
                candidates.add(nativeCacheDir.resolve(fileName));
            } catch (Throwable e) {
                LOGGER.warn("[c2me-wgpu] Failed to resolve game dir cache path", e);
            }

            // Fallback: system temp dir
            try {
                candidates.add(Files.createTempFile(null, fileName));
            } catch (Throwable t) {
                LOGGER.warn("[c2me-wgpu] Failed to create temp file for native extraction", t);
            }
        }

        if (candidates.isEmpty()) {
            LOGGER.error("[c2me-wgpu] No valid extraction paths available for native library");
            return null;
        }

        // 2. Iterate through candidates: extract -> load
        for (Path targetPath : candidates) {
            boolean isTmp = targetPath.toString().startsWith(System.getProperty("java.io.tmpdir"));
            try {
                if (targetPath.getParent() != null) {
                    Files.createDirectories(targetPath.getParent());
                }
                
                extractResource(libName, targetPath);
                
                if (isTmp && !Boolean.getBoolean("c2me.wgpu.preserveNative")) {
                    targetPath.toFile().deleteOnExit();
                }
                
                // Try to load. If this throws (e.g., noexec mount), we fall through to the next candidate.
                SymbolLookup lookup = SymbolLookup.libraryLookup(targetPath, arena);
                LOGGER.info("[c2me-wgpu] Successfully loaded native library from {}", targetPath);
                return lookup;
                
            } catch (Throwable e) {
                LOGGER.warn("[c2me-wgpu] Failed to extract/load native library from " + targetPath + ", trying next fallback...", e);
                // Clean up the failed file so the next attempt starts fresh
                try { Files.deleteIfExists(targetPath); } catch (Throwable ignored) {}
            }
        }

        LOGGER.error("[c2me-wgpu] All native library extraction and loading attempts failed.");
        return null;
    }

    private static void extractResource(String libName, Path target) throws Exception {
        try (final InputStream in = NativeLoader.class.getClassLoader().getResourceAsStream(libName)) {
            if (in == null) {
                throw new IllegalStateException("Cannot find native library " + libName);
            }
            Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
            
            // On POSIX systems (Linux, Android, OHOS), dlopen usually only requires
            // read access. However, some strict SELinux policies or directories
            // mounted with `noexec` might reject loading a library without the
            // executable bit set. We explicitly set it to be safe, but ignore
            // any failures (e.g. on Windows or read-only filesystems).
            if (!NORMALIZED_OS.equals("windows")) {
                boolean success = target.toFile().setExecutable(true);
                if (!success) {
                    LOGGER.debug("[c2me-wgpu] Could not set executable permission on {}", target);
                }
            }
        }
    }

    private static String normalize(String value) {
        return value.toLowerCase(Locale.US).replaceAll("[^a-z0-9]+", "");
    }

    private static String normalizeArch(String value) {
        value = normalize(value);
        if (value.matches("^(x8664|amd64|ia32e|em64t|x64)$")) {
            return "x86_64";
        }
        if (value.matches("^(x8632|x86|i[3-6]86|ia32|x32)$")) {
            return "x86_32";
        }
        // "arm64" is what some Apple JDKs report on Apple Silicon; aarch64 is
        // the canonical directory name. Bare "arm" is deliberately NOT mapped:
        // that means 32-bit ARMv7, and "unknown" produces a clearer "no native
        // library" message than a mis-mapped dlopen failure would.
        if ("aarch64".equals(value) || "arm64".equals(value)) {
            return "aarch64";
        }
        if (value.matches("^(armv7l|armv7|armeabi|armeabiv7a)$")) {
            return "armv7";
        }
        return "unknown";
    }

    private static String normalizeOs(String value) {
        value = normalize(value);
        if (value.startsWith("linux")) {
            return "linux";
        }
        if (value.startsWith("macosx") || value.startsWith("osx") || value.startsWith("darwin")) {
            return "macos";
        }
        if (value.startsWith("windows")) {
            return "windows";
        }
        return "unknown";
    }

}
