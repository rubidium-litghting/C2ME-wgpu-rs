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

import com.rubidiumlighting.c2me.opts.accel.wgpu.config.Config;

import com.rubidiumlighting.c2me.opts.accel.wgpu.mixin.AquiferImplAccessor;
import com.rubidiumlighting.c2me.opts.accel.wgpu.mixin.ChunkNoiseSamplerAccessor;
import com.rubidiumlighting.c2me.opts.accel.wgpu.mixin.XoroshiroSplitterAccessor;
import net.minecraft.block.BlockState;
import net.minecraft.world.chunk.Chunk;

import java.nio.ByteBuffer;
import net.minecraft.world.chunk.ChunkSection;
import net.minecraft.world.gen.StructureWeightSampler;
import net.minecraft.world.gen.noise.NoiseConfig;
import net.minecraft.world.gen.noise.NoiseRouter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Offline diagnostic dumps, moved out of {@link MixinChunkGenerator}.
 *
 * <p>Everything here is gated on the {@code debug_dump_*} config fields and
 * writes files for GPU-vs-vanilla comparison. None of it runs in a normal session,
 * so it lives in its own class to keep the mixin readable.
 *
 * <p>These writers are the source of the fixtures committed under
 * {@code src/test/resources/}: {@code maybeDumpBlobAndRef} produces
 * {@code real_router.bin} + {@code real_router.ref.txt}.
 */
public final class ChunkDumper {

    private static final Logger LOGGER = LoggerFactory.getLogger("c2me-wgpu");

    /** Target chunk(s) for block dumping, parsed once from config.
     *  Defaults to {0,0,0,0} when debug_dump_chunk is empty. */
    public static final int[] DUMP_CHUNK = parseDumpChunk();

    /** True only when the user explicitly asked for a chunk dump. All the
     *  diagnostic writers below are gated on this (or on their own config
     *  switch), so a normal session never touches the filesystem. */
    public static final boolean DUMP_ENABLED = !Config.get().debugDumpChunk.isBlank();

    /** Dump the block states of the target chunk(s) (CPU or GPU result). */
    public static final boolean DUMP_BLOCKS = Config.get().debugDumpBlocks;
    /** Dump the per-cell aquifer table for the target chunk(s). */
    public static final boolean DUMP_AQUIFER = Config.get().debugDumpAquifer;
    /** §23: dump Beardifier pieces/junctions + per-block beard reference. */
    public static final boolean DUMP_BEARD = Config.get().debugDumpBeard;
    /** §23: dump only the FIRST structure-bearing chunk encountered. */
    private static final java.util.concurrent.atomic.AtomicBoolean BEARD_DUMPED =
            new java.util.concurrent.atomic.AtomicBoolean(false);
    /** One-shot guard for the per-chunk corner reference dump. */
    private static final java.util.concurrent.atomic.AtomicBoolean CHUNK_REF_DUMPED =
            new java.util.concurrent.atomic.AtomicBoolean(false);

    /** Directory for diagnostic dumps. `/tmp` does not exist on Android, so
     *  default to the platform temp dir; the config field overrides. */
    static String dumpDir() {
        String d = Config.get().debugDumpDir;
        String dir;
        if (d != null && !d.isBlank()) {
            dir = d;
        } else {
            String t = System.getProperty("java.io.tmpdir");
            if (t == null || t.isBlank()) t = ".";
            // Trailing separators would double up in the concatenations below.
            while (t.length() > 1 && (t.endsWith("/") || t.endsWith("\\"))) {
                t = t.substring(0, t.length() - 1);
            }
            dir = t;
        }
        // Files.writeString/write do not create parent directories: a
        // user-configured path that does not exist yet fails with
        // NoSuchFileException. Create it here, once, for every writer.
        try {
            java.nio.file.Files.createDirectories(java.nio.file.Path.of(dir));
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] could not create dump dir {}", dir, t);
        }
        return dir;
    }

    static int[] parseDumpChunk() {
        String v = Config.get().debugDumpChunk;
        if (v == null || v.isBlank()) return new int[]{0, 0, 0, 0};
        try {
            String[] p = v.split(",");
            if (p.length == 2) {
                int x = Integer.parseInt(p[0].trim()), z = Integer.parseInt(p[1].trim());
                return new int[]{x, z, x, z};
            }
            if (p.length == 4) {
                return new int[]{
                        Integer.parseInt(p[0].trim()), Integer.parseInt(p[1].trim()),
                        Integer.parseInt(p[2].trim()), Integer.parseInt(p[3].trim())};
            }
        } catch (Throwable ignored) {}
        return new int[]{0, 0, 0, 0};
    }
    /** Dump selected chunk block states for offline GPU-vs-vanilla comparison. */
    public static void maybeDumpBlocks(Chunk chunk, String mode) {
        int cx = chunk.getPos().x, cz = chunk.getPos().z;
        if (cx < DUMP_CHUNK[0] || cx > DUMP_CHUNK[2]
                || cz < DUMP_CHUNK[1] || cz > DUMP_CHUNK[3]) {
            return;
        }
        try {
            StringBuilder sb = new StringBuilder();
            int bottom = chunk.getBottomY();
            int top = chunk.getTopY();
            for (int y = bottom; y < top; y++) {
                int si = chunk.getSectionIndex(y);
                if (si < 0 || si >= chunk.getSectionArray().length) continue;
                ChunkSection sec = chunk.getSection(si);
                if (sec == null || sec.isEmpty()) continue;
                for (int z = 0; z < 16; z++) {
                    for (int x = 0; x < 16; x++) {
                        BlockState bs = sec.getBlockState(x, y & 15, z);
                        if (bs.isAir()) continue;
                        String id = net.minecraft.registry.Registries.BLOCK
                                .getId(bs.getBlock()).toString();
                        sb.append(x).append(' ').append(y).append(' ').append(z)
                          .append(' ').append(id).append('\n');
                    }
                }
            }
            String path = dumpDir() + "/chunk_blocks_" + mode
                    + "_" + cx + "_" + cz + ".txt";
            java.nio.file.Files.writeString(java.nio.file.Path.of(path), sb.toString());
            LOGGER.info("[c2me-wgpu] dumped {} block states ({} blocks) to {}",
                    mode, sb.length(), path);
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] block dump failed", t);
        }
    }
    /** §23: dump the Beardifier's per-chunk pieces/junctions and per-block
     *  beard values, for offline validation of the Rust/WGSL reproduction. */
    public static void dumpBeard(Chunk chunk,
            net.minecraft.world.gen.StructureWeightSampler sw) {
        try {
            StringBuilder sb = new StringBuilder();
            sb.append("CHUNK ").append(chunk.getPos().x).append(' ')
              .append(chunk.getPos().z).append('\n');
            // pieces
            var pit = sw.pieceIterator;
            int np = 0;
            StringBuilder pb = new StringBuilder();
            while (pit.hasNext()) {
                var pc = pit.next();
                var box = pc.box();
                String ad = pc.terrainAdjustment().name();
                pb.append(box.getMinX()).append(' ').append(box.getMinY()).append(' ')
                  .append(box.getMinZ()).append(' ').append(box.getMaxX()).append(' ')
                  .append(box.getMaxY()).append(' ').append(box.getMaxZ()).append(' ')
                  .append(ad).append(' ').append(pc.groundLevelDelta()).append('\n');
                np++;
            }
            pit.back(Integer.MAX_VALUE);
            sb.append("PIECE ").append(np).append('\n').append(pb);
            // junctions
            var jit = sw.junctionIterator;
            int nj = 0;
            StringBuilder jb = new StringBuilder();
            while (jit.hasNext()) {
                var jn = jit.next();
                jb.append(jn.getSourceX()).append(' ').append(jn.getSourceGroundY())
                  .append(' ').append(jn.getSourceZ()).append('\n');
                nj++;
            }
            jit.back(Integer.MAX_VALUE);
            sb.append("JUNC ").append(nj).append('\n').append(jb);
            // per-block beard (full chunk)
            int minY = -64, height = 384;
            var npos = new com.rubidiumlighting.c2me.opts.accel.wgpu.MutableNoisePos();
            sb.append("BLOCK ").append(16 * height * 16).append('\n');
            int sx = chunk.getPos().getStartX();
            int sz = chunk.getPos().getStartZ();
            for (int by = 0; by < height; by++) {
                int gy = minY + by;
                for (int bz = 0; bz < 16; bz++) {
                    for (int bx = 0; bx < 16; bx++) {
                        double bd = sw.sample(npos.set(sx + bx, gy, sz + bz));
                        sb.append(bx).append(' ').append(gy).append(' ').append(bz)
                          .append(' ').append(bd).append('\n');
                    }
                }
            }
            java.nio.file.Files.writeString(
                    java.nio.file.Path.of(dumpDir() + "/beard_" + chunk.getPos().x + "_"
                            + chunk.getPos().z + ".txt"), sb.toString());
            LOGGER.info("[c2me-wgpu] dumped beard ref ({} pieces, {} junc)", np, nj);
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] beard dump failed", t);
        }
    }
    public static void dumpAquifer(Chunk chunk, long worldHandle,
            net.minecraft.world.gen.chunk.AquiferSampler aquifer,
            NoiseRouter router) {
        try {
                        if (!(aquifer instanceof net.minecraft.world.gen.chunk.AquiferSampler.Impl impl)) {
                LOGGER.warn("[c2me-wgpu] aquifer dump: not Impl, skipping");
                return;
            }
            AquiferImplAccessor a = (AquiferImplAccessor) (Object) impl;
            int startX = a.getStartX(), startY = a.getStartY(), startZ = a.getStartZ();
            int sizeX = a.getSizeX(), sizeZ = a.getSizeZ();
            net.minecraft.world.gen.chunk.AquiferSampler.FluidLevel[] wl = a.getWaterLevels();
            long[] bp = a.getBlockPositions();
            StringBuilder sb = new StringBuilder();
            sb.append("CHUNK ").append(chunk.getPos().x).append(' ')
              .append(chunk.getPos().z).append('\n');
            sb.append("AQ ").append(startX).append(' ').append(startY).append(' ')
              .append(startZ).append(' ').append(sizeX).append(' ').append(sizeZ)
              .append('\n');
            // §22: dump the aquifer RNG seed (seedLo/seedHi) so the Rust/WGSL
            // generator can reproduce blockPositions[] from scratch.
            {
                var dbg = new StringBuilder();
                a.getRandomDeriver().addDebugInfo(dbg);
                sb.append("SEED ").append(dbg).append('\n');
            }
            // §22: 预热所有 cell 的 waterLevels 表。MC 惰性填充（apply 访问到
            // 才填），GPU 需要完整表。用 seedLo/seedHi 算每 cell 的
            // blockPosition，再调 getWaterLevel 触发 getFluidLevel。
            {
                long[] seed = seedWords(a);
                long sLo = seed[0], sHi = seed[1];
                for (int idx = 0; idx < wl.length; idx++) {
                    int ci = idx % sizeX;
                    int ck = (idx / sizeX) % sizeZ;
                    int cj = idx / (sizeX * sizeZ);
                    int cx = startX + ci, cy = startY + cj, cz = startZ + ck;
                    long h = net.minecraft.util.math.MathHelper.hashCode(cx, cy, cz);
                    var rr = new net.minecraft.util.math.random.Xoroshiro128PlusPlusRandom(
                            h ^ sLo, sHi);
                    int bx = cx * 16 + rr.nextInt(10);
                    int by = cy * 12 + rr.nextInt(9);
                    int bz = cz * 16 + rr.nextInt(10);
                    long pos = net.minecraft.util.math.BlockPos.asLong(bx, by, bz);
                    a.invokeGetWaterLevel(pos);
                }
            }
            // §22 Step B: serialize the compact per-chunk aquifer table.
            // Layout: magic(4) n(4) seedLo(8) seedHi(8) then n * (lvl_y i32 +
            // state u32). blockPositions[] is NOT included — Rust recomputes
            // it from the seed (verified bit-exact in aquifer_positions.rs).
            {
                long[] seed = seedWords(a);
                long sLo3 = seed[0], sHi3 = seed[1];
                java.nio.ByteBuffer tb = java.nio.ByteBuffer.allocate(24 + wl.length * 8)
                        .order(java.nio.ByteOrder.LITTLE_ENDIAN);
                tb.putInt(0x41515541); // "AQUA"
                tb.putInt(wl.length);
                tb.putLong(sLo3); tb.putLong(sHi3);
                for (int i = 0; i < wl.length; i++) {
                    var lvl = wl[i];
                    int ly = (lvl == null) ? Integer.MIN_VALUE : lvl.y;
                    int sc = 0;
                    if (lvl != null) {
                        var blk = lvl.state.getBlock();
                        if (blk == net.minecraft.block.Blocks.WATER) sc = 2;
                        else if (blk == net.minecraft.block.Blocks.LAVA) sc = 3;
                    }
                    tb.putInt(ly); tb.putInt(sc);
                }
                byte[] arr = tb.array();
                java.nio.file.Files.write(java.nio.file.Path.of(dumpDir() + "/aquifer_table_"
                        + chunk.getPos().x + "_" + chunk.getPos().z + ".bin"), arr);
                if (worldHandle != 0L) {
                    WgpuThinBinding.setAquiferTable(worldHandle,
                            chunk.getPos().x, chunk.getPos().z, arr);
                }
            }
            for (int i = 0; i < wl.length; i++) {
                long packed = bp[i];
                boolean none = (packed == Long.MAX_VALUE);
                int bx = none ? 0 : net.minecraft.util.math.BlockPos.unpackLongX(packed);
                int by = none ? 0 : net.minecraft.util.math.BlockPos.unpackLongY(packed);
                int bz = none ? 0 : net.minecraft.util.math.BlockPos.unpackLongZ(packed);
                var lvl = wl[i];
                String state = (lvl == null) ? "null"
                        : net.minecraft.registry.Registries.BLOCK
                            .getId(lvl.state.getBlock()).toString();
                int y = (lvl == null) ? Integer.MIN_VALUE : lvl.y;
                sb.append(i).append(' ')
                  .append(none ? "none" : (bx + " " + by + " " + bz)).append(' ')
                  .append(y).append(' ').append(state).append('\n');
            }
            String path = dumpDir() + "/aquifer_" + chunk.getPos().x + "_" + chunk.getPos().z + ".txt";
            java.nio.file.Files.writeString(java.nio.file.Path.of(path), sb.toString());
            LOGGER.info("[c2me-wgpu] dumped aquifer table ({} cells) to {}", wl.length, path);

            // §20: dump, per cell, the 5 noise inputs at the exact points
            // getFluidLevel samples + the expected output. Rust/WGSL then only
            // needs to reproduce the control flow (no noise).
            var flooded = unwrapRouter(router.fluidLevelFloodednessNoise());
            var spread  = unwrapRouter(router.fluidLevelSpreadNoise());
            var lava    = unwrapRouter(router.lavaNoise());
            var erosion = unwrapRouter(router.erosion());
            var depth   = unwrapRouter(router.depth());
            StringBuilder nb = new StringBuilder();
            nb.append("CHUNK ").append(chunk.getPos().x).append(' ')
              .append(chunk.getPos().z).append('\n');
            var flSampler = a.getFluidLevelSampler();
            var cnsForSurf = a.getChunkNoiseSampler();
            for (int i = 0; i < wl.length; i++) {
                long packed = bp[i];
                if (packed == Long.MAX_VALUE) continue;
                int bx = net.minecraft.util.math.BlockPos.unpackLongX(packed);
                int by = net.minecraft.util.math.BlockPos.unpackLongY(packed);
                int bz = net.minecraft.util.math.BlockPos.unpackLongZ(packed);
                var p0 = new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(bx, by, bz);
                double f = flooded.sample(p0);
                double e = erosion.sample(p0);
                double d = depth.sample(p0);
                var ps = new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(
                        Math.floorDiv(bx,16), Math.floorDiv(by,40), Math.floorDiv(bz,16));
                double sp = spread.sample(ps);
                var pt = new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(
                        Math.floorDiv(bx,64), Math.floorDiv(by,40), Math.floorDiv(bz,64));
                double lv = lava.sample(pt);
                var lvl = wl[i];
                // initial FluidLevel from the sampler
                var fl0 = flSampler.getFluidLevel(bx, by, bz);
                nb.append(i).append(' ').append(bx).append(' ').append(by).append(' ')
                  .append(bz).append(' ')
                  .append((float) f).append(' ').append((float) sp).append(' ')
                  .append((float) lv).append(' ').append((float) e).append(' ')
                  .append((float) d).append(' ')
                  .append(fl0.y).append(' ')
                  .append(net.minecraft.registry.Registries.BLOCK.getId(fl0.state.getBlock()).toString())
                  .append(' ');
                // 13 neighbor surface heights
                for (int[] cs : new int[][]{
                        {0,0},{-2,-1},{-1,-1},{0,-1},{1,-1},{-3,0},{-2,0},{-1,0},{1,0},
                        {-2,1},{-1,1},{0,1},{1,1}}) {
                    int m = bx + cs[0] * 16;
                    int n = bz + cs[1] * 16;
                    nb.append(cnsForSurf.estimateSurfaceHeight(m, n)).append(' ');
                }
                nb.append(lvl == null ? Integer.MIN_VALUE : lvl.y).append(' ')
                  .append(lvl == null ? "null"
                        : net.minecraft.registry.Registries.BLOCK.getId(lvl.state.getBlock()).toString())
                  .append('\n');
            }
            String nbPath = dumpDir() + "/aquifer_noise_" + chunk.getPos().x + "_" + chunk.getPos().z + ".txt";
            java.nio.file.Files.writeString(java.nio.file.Path.of(nbPath), nb.toString());
            LOGGER.info("[c2me-wgpu] dumped aquifer noise inputs to {}", nbPath);

            // Also dump the per-column surface-height cache: getWaterLevel
            // needs estimateSurfaceHeight for 13 neighbour columns per cell,
            // and this is the CPU-provided input for the GPU reimplementation.
            net.minecraft.world.gen.chunk.ChunkNoiseSampler cns = a.getChunkNoiseSampler();
            var shCache = ((ChunkNoiseSamplerAccessor) (Object) cns).getSurfaceHeightEstimateCache();
            StringBuilder sh = new StringBuilder();
            sh.append("CHUNK ").append(chunk.getPos().x).append(' ')
              .append(chunk.getPos().z).append('\n');
            for (var e : shCache.long2IntEntrySet()) {
                long k = e.getLongKey();
                int cx = net.minecraft.util.math.ColumnPos.getX(k);
                int cz = net.minecraft.util.math.ColumnPos.getZ(k);
                sh.append(cx).append(' ').append(cz).append(' ')
                  .append(e.getIntValue()).append('\n');
            }
            String shPath = dumpDir() + "/aquifer_surfaces_" + chunk.getPos().x + "_" + chunk.getPos().z + ".txt";
            java.nio.file.Files.writeString(java.nio.file.Path.of(shPath), sh.toString());
            LOGGER.info("[c2me-wgpu] dumped surface-height cache ({} columns) to {}",
                    shCache.size(), shPath);
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] aquifer dump failed", t);
        }
    }
    /** Dump the corner reference for the first chunk inside C2ME_DUMP_CHUNK,
     *  using the SAME decompiled tree as the blob (the `router` argument). */
    public static void maybeDumpChunkRef(Chunk chunk, DimState ds, NoiseRouter router,
                                          NoiseConfig noiseConfig) {
        if (!DUMP_ENABLED) return;
        int cx = chunk.getPos().x, cz = chunk.getPos().z;
        if (cx < DUMP_CHUNK[0] || cx > DUMP_CHUNK[2]
                || cz < DUMP_CHUNK[1] || cz > DUMP_CHUNK[3]) return;
        if (!CHUNK_REF_DUMPED.compareAndSet(false, true)) return;
        try {
            java.util.LinkedHashMap<String, net.minecraft.world.gen.densityfunction.DensityFunction> chans =
                    new java.util.LinkedHashMap<>();
            chans.put("barrierNoise", router.barrierNoise());
            chans.put("fluidLevelFloodednessNoise", router.fluidLevelFloodednessNoise());
            chans.put("fluidLevelSpreadNoise", router.fluidLevelSpreadNoise());
            chans.put("lavaNoise", router.lavaNoise());
            chans.put("temperature", router.temperature());
            chans.put("vegetation", router.vegetation());
            chans.put("continents", router.continents());
            chans.put("erosion", router.erosion());
            chans.put("depth", router.depth());
            chans.put("ridges", router.ridges());
            chans.put("initialDensityWithoutJaggedness", router.initialDensityWithoutJaggedness());
            chans.put("finalDensity", router.finalDensity());
            chans.put("veinToggle", router.veinToggle());
            chans.put("veinRidged", router.veinRidged());
            chans.put("veinGap", router.veinGap());
            StringBuilder sb = new StringBuilder();
            sb.append("# cx cz startX startZ\n");
            sb.append("CHUNK ").append(cx).append(' ').append(cz).append('\n');
            sb.append("DIMS ").append(ds.gridX).append(' ').append(ds.gridY).append(' ').append(ds.gridZ).append('\n');
            sb.append("CELL_W ").append(ds.cellW).append('\n');
            sb.append("MIN_Y ").append(ds.minY).append('\n');
            int sx = chunk.getPos().getStartX();
            int sz = chunk.getPos().getStartZ();
            for (var e : chans.entrySet()) {
                sb.append("CHANNEL ").append(e.getKey()).append('\n');
                net.minecraft.world.gen.densityfunction.DensityFunction fd = e.getValue();
                for (int guard = 0; guard < 32; guard++) {
                    if (fd instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.RegistryEntryHolder h) {
                        fd = h.function().value();
                    } else if (fd instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Wrapper w) {
                        fd = w.wrapped();
                    } else { break; }
                }
                for (int j = 0; j < ds.gridY; j++) {
                    int wy = ds.minY + j * ds.cellH;
                    for (int k = 0; k < ds.gridZ; k++) {
                        int wz = sz + k * ds.cellW;
                        for (int i = 0; i < ds.gridX; i++) {
                            int wx = sx + i * ds.cellW;
                            double v = fd.sample(new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(wx, wy, wz));
                            sb.append(i).append(' ').append(j).append(' ').append(k).append(' ').append((float) v).append('\n');
                        }
                    }
                }
            }
            String rp = dumpDir() + "/ref_chunk_" + cx + "_" + cz + ".txt";
            java.nio.file.Files.writeString(java.nio.file.Path.of(rp), sb.toString());
            LOGGER.info("[c2me-wgpu] dumped per-chunk ref (decompiled tree) to {}", rp);
            // §14 diag: also dump the COMPILED tree's depth, to compare against decompiled.
            try {
                NoiseRouter comp = noiseConfig.getNoiseRouter();
                StringBuilder cb = new StringBuilder();
                cb.append("CHANNEL depth\n");
                net.minecraft.world.gen.densityfunction.DensityFunction cfd = comp.depth();
                for (int guard=0; guard<32; guard++) {
                    if (cfd instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.RegistryEntryHolder h) cfd=h.function().value();
                    else if (cfd instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Wrapper w) cfd=w.wrapped();
                    else break;
                }
                for (int j=0;j<ds.gridY;j++){int wy=ds.minY+j*ds.cellH;
                  for (int k=0;k<ds.gridZ;k++){int wz=sz+k*ds.cellW;
                    for (int i=0;i<ds.gridX;i++){int wx=sx+i*ds.cellW;
                      double v=cfd.sample(new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(wx,wy,wz));
                      cb.append(i).append(' ').append(j).append(' ').append(k).append(' ').append((float)v).append('\n');}}}
                java.nio.file.Files.writeString(java.nio.file.Path.of(dumpDir() + "/ref_chunk_"+cx+"_"+cz+"_compiled.txt"), cb.toString());
                LOGGER.info("[c2me-wgpu] dumped COMPILED-tree depth ref to {}/ref_chunk_{}_{}_compiled.txt", dumpDir(), cx, cz);
            } catch (Throwable t2) { LOGGER.warn("[c2me-wgpu] compiled ref dump failed", t2); }
            // §14 diag: dump every Spline's locations/derivatives from MC's
            // decompiled depth tree, to diff against our Rust deserialization.
            try {
                final java.util.List<String> sp = new java.util.ArrayList<>();
                ds.rawRouter.depth().apply(new net.minecraft.world.gen.densityfunction.DensityFunction.DensityFunctionVisitor() {
                    @Override
                    public net.minecraft.world.gen.densityfunction.DensityFunction apply(
                            net.minecraft.world.gen.densityfunction.DensityFunction d) {
                        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Spline spl) {
                            try {
                                Object inner = spl.spline();
                                if (inner instanceof net.minecraft.util.math.Spline.Implementation<?,?> impl) {
                                    sp.add(java.util.Arrays.toString(impl.locations())
                                         + " | " + java.util.Arrays.toString(impl.derivatives()));
                                } else if (inner instanceof net.minecraft.util.math.Spline.FixedFloatFunction<?,?> f) {
                                    sp.add("FIXED " + f.value());
                                }
                            } catch (Throwable ignore) {}
                        }
                        return d;
                    }
                });
                java.nio.file.Files.writeString(java.nio.file.Path.of(dumpDir() + "/mc_depth_splines.txt"),
                        String.join("\n", sp));
                LOGGER.info("[c2me-wgpu] dumped {} MC splines to {}/mc_depth_splines.txt", sp.size(), dumpDir());
            } catch (Throwable t3) { LOGGER.warn("[c2me-wgpu] MC spline dump failed", t3); }
            // §14 diag: value trace of every depth node at the failing point.
            try {
                StringBuilder tb = new StringBuilder();
                var pos = new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(
                        sx + 0, ds.minY + 24 * ds.cellH, sz + 0); // cell(0,24,0) world(800,128,800)
                int[] counter = {0};
                valueTrace(ds.rawRouter.depth(), pos, tb, 0, counter);
                java.nio.file.Files.writeString(java.nio.file.Path.of(dumpDir() + "/mc_depth_values.txt"), tb.toString());
                LOGGER.info("[c2me-wgpu] dumped MC depth value trace ({} lines) to {}/mc_depth_values.txt", counter[0], dumpDir());
            } catch (Throwable t4) { LOGGER.warn("[c2me-wgpu] MC value trace failed", t4); }
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] per-chunk ref dump failed", t);
        }
    }
    /** The aquifer splitter's two seed words, read from the fields rather
     *  than parsed out of {@code addDebugInfo()}'s text (see
     *  {@link XoroshiroSplitterAccessor}). */
    private static long[] seedWords(AquiferImplAccessor a) {
        XoroshiroSplitterAccessor sp =
                (XoroshiroSplitterAccessor) a.getRandomDeriver();
        return new long[]{sp.c2me$getSeedLo(), sp.c2me$getSeedHi()};
    }

    static String resolveType(net.minecraft.world.gen.densityfunction.DensityFunction d) {
        var T = net.minecraft.world.gen.densityfunction.DensityFunctionTypes.class;
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Constant) return "Const";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.RangeChoice) return "RangeChoice";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Spline) return "Spline";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.YClampedGradient) return "YClamp";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Noise) return "Noise";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.ShiftedNoise) return "ShiftedNoise";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.ShiftA) return "ShiftA";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.ShiftB) return "ShiftB";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Shift) return "Shift";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.UnaryOperation u) {
            return switch (u.type()) {
                case ABS -> "Abs"; case SQUARE -> "Square"; case CUBE -> "Cube";
                case SQUEEZE -> "Squeeze";
                case HALF_NEGATIVE -> "HalfNeg"; case QUARTER_NEGATIVE -> "QuarterNeg";
            };
        }
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Wrapper w) {
            return switch (w.type()) { case FLAT_CACHE -> "CacheFlat"; case CACHE2D -> "Cache2D";
                case CACHE_ONCE -> "CacheOnce"; case CACHE_ALL_IN_CELL -> "CacheCell"; case INTERPOLATED -> "Interp"; };
        }
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.RegistryEntryHolder h) {
            return "Holder(" + resolveType(h.function().value()) + ")";
        }
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.LinearOperation lo) {
            return lo.specificType() == net.minecraft.world.gen.densityfunction.DensityFunctionTypes.LinearOperation.SpecificType.ADD ? "Add" : "Mul";
        }
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.BinaryOperation bo) {
            return switch (bo.type()) { case ADD -> "Add"; case MUL -> "Mul"; case MIN -> "Min"; case MAX -> "Max"; };
        }
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Clamp) return "Clamp";
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.WeirdScaledSampler) return "Weird";
        return d.getClass().getSimpleName();
    }
    static void valueTrace(
            net.minecraft.world.gen.densityfunction.DensityFunction d,
            net.minecraft.world.gen.densityfunction.DensityFunction.NoisePos pos,
            StringBuilder sb, int ind, int[] counter) {
        // Transparent wrappers our AST does NOT represent (serialized as passthrough).
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.RegistryEntryHolder h) {
            valueTrace(h.function().value(), pos, sb, ind, counter); return;
        }
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Wrapper w
                && (w.type() == net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Wrapping.Type.CACHE_ONCE
                 || w.type() == net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Wrapping.Type.CACHE_ALL_IN_CELL)) {
            valueTrace(w.wrapped(), pos, sb, ind, counter); return;
        }
        String pad = "  ".repeat(Math.min(ind, 40));
        double v = d.sample(pos);
        String tag = resolveType(d);
        sb.append(pad).append(tag).append(" = ").append(String.format("%.6f", v)).append('\n');
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Spline sp) {
            try {
                Object inner = sp.spline();
                if (inner instanceof net.minecraft.util.math.Spline.Implementation<?,?> impl) {
                    float[] L = impl.locations();
                    if (L.length > 0 && Math.abs(L[0] - (-0.45f)) < 1e-4) {
                        sb.append(pad).append("  [SPLINE-STRUCT] ").append(impl.getDebugString()).append('\n');
                    }
                }
            } catch (Throwable ignore) {}
        }
        counter[0]++;
        if (ind > 300) return;
        if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.BinaryOperationLike b) {
            valueTrace(b.argument1(), pos, sb, ind+1, counter);
            valueTrace(b.argument2(), pos, sb, ind+1, counter);
        } else if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Unary u) {
            valueTrace(u.input(), pos, sb, ind+1, counter);
        } else if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Wrapper w) {
            valueTrace(w.wrapped(), pos, sb, ind+1, counter);
        } else if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.RangeChoice r) {
            valueTrace(r.input(), pos, sb, ind+1, counter);
        } else if (d instanceof net.minecraft.world.gen.densityfunction.DensityFunctionTypes.ShiftedNoise sn) {
            valueTrace(sn.shiftX(), pos, sb, ind+1, counter);
            valueTrace(sn.shiftY(), pos, sb, ind+1, counter);
            valueTrace(sn.shiftZ(), pos, sb, ind+1, counter);
            // noise child (not a DensityFunction; our AST stores it as NoiseSampler)
            double dx = pos.blockX() * sn.xzScale() + sn.shiftX().sample(pos);
            double dy = pos.blockY() * sn.yScale() + sn.shiftY().sample(pos);
            double dz = pos.blockZ() * sn.xzScale() + sn.shiftZ().sample(pos);
            sb.append("  ".repeat(Math.min(ind+1, 40))).append("Noise = ")
              .append(String.format("%.6f", sn.noise().sample(dx, dy, dz))).append('\n');
            counter[0]++;
        }
    }
    public static void maybeDumpBlobAndRef(NoiseRouter router, NoiseConfig noiseConfig,
                                            Chunk chunk, ByteBuffer blob,
                                            int minY, int height, int cellW, int cellH,
                                            int gx, int gy, int gz) {
        String dumpPath = Config.get().debugDumpBlob;
        if (dumpPath.isBlank()) {
            return;
        }
        // Per-dimension suffix so the three dimensions don't overwrite each
        // other's dumps, and we never clobber the committed Overworld fixture
        // src/test/resources/real_router.bin.
        dumpPath = dumpPath + "." + minY + "_" + height + "_" + cellW + "_" + cellH;
        try {
            java.nio.ByteBuffer dup = blob.duplicate();
            byte[] bytes = new byte[dup.remaining()];
            dup.get(bytes);
            java.nio.file.Files.write(java.nio.file.Path.of(dumpPath), bytes);
            LOGGER.info("[c2me-wgpu] dumped router blob ({} bytes) to {}", bytes.length, dumpPath);
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] blob dump failed", t);
        }
        try {
            StringBuilder sb = new StringBuilder();
            sb.append("# cx cz startX startZ\n");
            sb.append("CHUNK ").append(chunk.getPos().x).append(' ')
              .append(chunk.getPos().z).append('\n');
            sb.append("DIMS ").append(gx).append(" ").append(gy).append(" ").append(gz).append("\n");
            sb.append("CELL_W ").append(cellW).append("\nMIN_Y ").append(minY).append("\n");
            NoiseRouter CR = router;  // §5.8: must match the blob's tree (decompiled), not the compiled router
            java.util.LinkedHashMap<String, net.minecraft.world.gen.densityfunction.DensityFunction> chans =
                    new java.util.LinkedHashMap<>();
            chans.put("barrierNoise", CR.barrierNoise());
            chans.put("fluidLevelFloodednessNoise", CR.fluidLevelFloodednessNoise());
            chans.put("fluidLevelSpreadNoise", CR.fluidLevelSpreadNoise());
            chans.put("lavaNoise", CR.lavaNoise());
            chans.put("temperature", CR.temperature());
            chans.put("vegetation", CR.vegetation());
            chans.put("continents", CR.continents());
            chans.put("erosion", CR.erosion());
            chans.put("depth", CR.depth());
            chans.put("ridges", CR.ridges());
            chans.put("initialDensityWithoutJaggedness", CR.initialDensityWithoutJaggedness());
            chans.put("finalDensity", CR.finalDensity());
            chans.put("veinToggle", CR.veinToggle());
            chans.put("veinRidged", CR.veinRidged());
            chans.put("veinGap", CR.veinGap());
            int sx = chunk.getPos().getStartX();
            int sz = chunk.getPos().getStartZ();
            for (var e : chans.entrySet()) {
                sb.append("CHANNEL ").append(e.getKey()).append('\n');
                net.minecraft.world.gen.densityfunction.DensityFunction fd = e.getValue();
                for (int guard = 0; guard < 32; guard++) {
                    if (fd instanceof
                            net.minecraft.world.gen.densityfunction.DensityFunctionTypes.RegistryEntryHolder h) {
                        fd = h.function().value();
                    } else if (fd instanceof
                            net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Wrapper w) {
                        fd = w.wrapped();
                    } else {
                        break;
                    }
                }
                for (int j = 0; j < gy; j++) {
                    int wy = minY + j * cellH;
                    for (int k = 0; k < gz; k++) {
                        int wz = sz + k * cellW;
                        for (int i = 0; i < gx; i++) {
                            int wx = sx + i * cellW;
                            double v = fd.sample(
                                    new net.minecraft.world.gen.densityfunction.DensityFunction.UnblendedNoisePos(
                                            wx, wy, wz));
                            sb.append(i).append(' ').append(j).append(' ').append(k)
                              .append(' ').append((float) v).append('\n');
                        }
                    }
                }
            }
            String refPath = dumpPath + ".ref.txt";
            java.nio.file.Files.writeString(java.nio.file.Path.of(refPath), sb.toString());
            LOGGER.info("[c2me-wgpu] dumped corner reference to {}", refPath);
        } catch (Throwable t) {
            LOGGER.warn("[c2me-wgpu] corner reference dump failed", t);
        }
    }
    /** Unwrap RegistryEntryHolder/Wrapper layers (same as the ref-dump). */
    static net.minecraft.world.gen.densityfunction.DensityFunction unwrapRouter(
            net.minecraft.world.gen.densityfunction.DensityFunction fd) {
        for (int guard = 0; guard < 32; guard++) {
            if (fd instanceof
                    net.minecraft.world.gen.densityfunction.DensityFunctionTypes.RegistryEntryHolder h) {
                fd = h.function().value();
            } else if (fd instanceof
                    net.minecraft.world.gen.densityfunction.DensityFunctionTypes.Wrapper w) {
                fd = w.wrapped();
            } else {
                break;
            }
        }
        return fd;
    }

    /** Exposed for {@link MixinChunkGenerator}: one-shot guard so only the
     *  first structure-bearing chunk gets its beard dumped. */
    public static boolean markBeardDumped() {
        return BEARD_DUMPED.compareAndSet(false, true);
    }

    public static void clearBeardDumped() {
        BEARD_DUMPED.set(false);
    }

}
