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

import net.minecraft.util.math.noise.DoublePerlinNoiseSampler;
import net.minecraft.util.math.noise.OctavePerlinNoiseSampler;
import net.minecraft.util.math.noise.PerlinNoiseSampler;
import net.minecraft.util.math.random.Random;
import org.junit.jupiter.api.Test;

import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.util.List;

public class McReferenceExportTest {

    @Test
    void exportReference() throws Exception {
        long seed = 12345L;
        int firstOctave = 0;
        double[] amps = {1.0, 0.5, 0.25};

        Random r = Random.create(seed);
        DoublePerlinNoiseSampler sampler =
                DoublePerlinNoiseSampler.create(r, firstOctave, amps);

        PrintWriter w = new PrintWriter("src/test/resources/mc_reference.txt");
        w.println("# MC DoublePerlinNoiseSampler reference");
        w.println("SEED " + seed);
        w.println("FIRST_OCTAVE " + firstOctave);
        w.println("AMP_LEN " + amps.length);
        for (double a : amps) w.println("AMP " + a);

        Object first = getField(sampler, "firstSampler");
        Object second = getField(sampler, "secondSampler");
        dumpStack(w, "FIRST", first);
        dumpStack(w, "SECOND", second);

        double amplitude = (double) getField(sampler, "amplitude");
        w.println("DOUBLE_AMPLITUDE " + amplitude);

        // Sample points: world coords (cx*4, -64+cy*4, cz*4), scale 0.3
        double xzScale = 0.3, yScale = 0.3;
        int count = 0;
        StringBuilder samples = new StringBuilder();
        for (int cy = 0; cy < 8; cy++) {
            for (int cz = 0; cz < 2; cz++) {
                for (int cx = 0; cx < 2; cx++) {
                    double wx = cx * 4;
                    double wy = -64 + cy * 4;
                    double wz = cz * 4;
                    double v = sampler.sample(wx * xzScale, wy * yScale, wz * xzScale);
                    samples.append(wx).append(' ').append(wy).append(' ')
                           .append(wz).append(' ').append(v).append('\n');
                    count++;
                }
            }
        }
        w.println("SAMPLE_COUNT " + count);
        w.print(samples);
        w.close();
        System.out.println("EXPORTED " + count + " samples to src/test/resources/mc_reference.txt");
    }

    private static void dumpStack(PrintWriter w, String tag, Object octave) throws Exception {
        Object arrObj = getField(octave, "octaveSamplers");
        PerlinNoiseSampler[] arr = (PerlinNoiseSampler[]) arrObj;
        int firstOctave = (int) getField(octave, "firstOctave");
        w.println(tag + "_FIRST_OCTAVE " + firstOctave);
        w.println(tag + "_OCTAVE_COUNT " + arr.length);
        for (int i = 0; i < arr.length; i++) {
            PerlinNoiseSampler p = arr[i];
            if (p == null) {
                w.println(tag + "_NULL " + i);
                continue;
            }
            byte[] perm = (byte[]) getField(p, "permutation");
            StringBuilder sb = new StringBuilder(tag).append("_PERM ").append(i).append(' ');
            for (int k = 0; k < 256; k++) {
                sb.append(perm[k] & 0xFF).append(k == 255 ? "" : " ");
            }
            w.println(sb);
            double ox = (double) getField(p, "originX");
            double oy = (double) getField(p, "originY");
            double oz = (double) getField(p, "originZ");
            w.println(tag + "_ORIGIN " + i + " " + ox + " " + oy + " " + oz);
        }
        // amplitudes as DoubleList
        Object ampsObj = getField(octave, "amplitudes");
        List<?> list = (List<?>) ampsObj;
        StringBuilder ab = new StringBuilder(tag).append("_AMPS");
        for (Object o : list) ab.append(' ').append(o);
        w.println(ab);
    }

    private static Object getField(Object o, String name) throws Exception {
        Class<?> c = o.getClass();
        while (c != null) {
            try {
                Field f = c.getDeclaredField(name);
                f.setAccessible(true);
                return f.get(o);
            } catch (NoSuchFieldException e) {
                c = c.getSuperclass();
            }
        }
        throw new NoSuchFieldException(name);
    }
}
