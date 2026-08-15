/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2021-2026 RubidiumLighting
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

    public static final String NORMALIZED_ARCH = normalizeArch(SystemPropertyUtil.get("os.arch", ""));
    public static final String NORMALIZED_OS = normalizeOs(SystemPropertyUtil.get("os.name", ""));

    private static final Arena arena = Arena.ofAuto();
    public static final SymbolLookup lookup;
    public static final Linker linker = Linker.nativeLinker();

    static {
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

    private static SymbolLookup load0(String libName) {
        try (final InputStream in = NativeLoader.class.getClassLoader().getResourceAsStream(libName)) {
            if (in == null) {
                LOGGER.warn("Cannot find native library {}, wgpu acceleration unavailable for this platform", libName);
                return null;
            }
            final Path tempFile;
            if (Boolean.getBoolean("c2me.wgpu.preserveNative")) {
                tempFile = Path.of(".", libName.substring(libName.lastIndexOf('/') + 1));
            } else {
                tempFile = Files.createTempFile(null, libName.substring(libName.lastIndexOf('/') + 1));
                tempFile.toFile().deleteOnExit();
            }
            Files.copy(in, tempFile, StandardCopyOption.REPLACE_EXISTING);
            return SymbolLookup.libraryLookup(tempFile, arena);
        } catch (Throwable e) {
            LOGGER.warn("Failed to load wgpu native library", e);
            return null;
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
        if ("aarch64".equals(value)) {
            return "aarch64";
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
