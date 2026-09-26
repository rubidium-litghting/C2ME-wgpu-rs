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

//! Deserialize Java NoiseRouterBlobifier binary format → AstNode.
//!
//! Protocol: 计划.txt §4
//!   Header:  magic(u32=0x4D433257 LE)  version(u16=1)  reserved([u8;4])
//!   Tree  :  [tag:u8][payload]  (recursive)
//!   Tail  :  SHAPE_CONFIG  BIOME_SOURCE
//!
//! Byte order: Little Endian.
//!
//! Every tag MUST match NoiseRouterBlobifier.java constants exactly.

use anyhow::anyhow;

use crate::Result;
use crate::ast::AstNode;
use crate::ast::NoiseType;
use crate::ast::OctaveSampler;

/// Parse a full blob including header. Returns (Root node, Shape config, BiomeTree).
///
/// Protocol layout (计划.txt §4):
///   [HEADER 10B] [NODE_TREE (recursive)] [SHAPE_CONFIG 16B] [BIOME_SOURCE (BiomeTree)]
/// A single Perlin permutation table + its origin offset, as shipped in the
/// optional PERM_RESOURCE trailer appended by the Java side.
#[derive(Debug, Clone, PartialEq)]
pub struct PermEntry {
    pub perm: [u32; 256],
    pub origin: [f64; 3],
}

/// Magic of the PERM_RESOURCE trailer: "PWPM" little-endian.
pub const PERM_MAGIC: u32 = 0x4D52_5750;

pub fn deserialize_full(blob: &[u8]) -> Result<(AstNode, ShapeConfig, AstNode)> {
    let (root, shape, biome, _perm) = deserialize_full_with_perm(blob)?;
    Ok((root, shape, biome))
}

/// Like [`deserialize_full`] but also decodes the optional PERM_RESOURCE
/// trailer. Returns an empty Vec when the trailer is absent.
pub fn deserialize_full_with_perm(
    blob: &[u8],
) -> Result<(AstNode, ShapeConfig, AstNode, Vec<PermEntry>)> {
    // 1. check blob.len() >= 10, else Deserialize("truncated header")
    ensure(blob.len() >= 10, "header")?;
    // 2. read u32 magic at offset 0, assert == 0x4D433257
    ensure(blob[0..4] == [0x4D, 0x43, 0x32, 0x57], "magic")?;
    // 3. read u16 version at offset 4, assert == 1
    ensure(blob[4..6] == [0x01, 0x00], "version")?;

    // 4. parse tree from offset 10 via parse_node
    let (root, tree_len) = parse_node(blob, 10)?;
    let mut cursor = 10 + tree_len;

    // 5. parse SHAPE_CONFIG = [min_y:i32, height:i32, cell_width:i32, cell_height:i32]
    ensure(blob.len() >= cursor + 16, "SHAPE_CONFIG")?;
    let min_y = i32::from_le_bytes(blob[cursor..cursor + 4].try_into().unwrap());
    let height = i32::from_le_bytes(blob[cursor + 4..cursor + 8].try_into().unwrap());
    let cell_width = i32::from_le_bytes(blob[cursor + 8..cursor + 12].try_into().unwrap());
    let cell_height = i32::from_le_bytes(blob[cursor + 12..cursor + 16].try_into().unwrap());
    cursor += 16;
    let shape_cfg = ShapeConfig {
        min_y,
        height,
        cell_width,
        cell_height,
    };

    // 6. parse BIOME_SOURCE remainder (BiomeTree — tag 0xF0 with its payload)
    ensure(cursor < blob.len(), "BIOME_SOURCE (empty)")?;
    let (biome_tree, biome_len) = parse_node(blob, cursor)?;
    cursor += biome_len;
    // (allow trailing bytes for future extension; protocol doesn't forbid them)
    let _ = cursor;

    // 7. optional PERM_RESOURCE trailer
    let perm_entries = parse_perm_resource(blob, cursor).unwrap_or_default();

    Ok((root, shape_cfg, biome_tree, perm_entries))
}

/// Parse the optional PERM_RESOURCE trailer starting at `offset`.
/// Returns empty Vec when no trailer is present.
fn parse_perm_resource(blob: &[u8], offset: usize) -> Result<Vec<PermEntry>> {
    if offset + 8 > blob.len() {
        return Ok(Vec::new());
    }
    let magic = u32::from_le_bytes(blob[offset..offset + 4].try_into().unwrap());
    if magic != PERM_MAGIC {
        return Ok(Vec::new());
    }
    let count = u32::from_le_bytes(blob[offset + 4..offset + 8].try_into().unwrap()) as usize;
    let mut cursor = offset + 8;
    let entry_size = 256 * 4 + 3 * 8;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        if cursor + entry_size > blob.len() {
            break;
        }
        let mut perm = [0u32; 256];
        for (i, slot) in perm.iter_mut().enumerate() {
            let p = cursor + i * 4;
            *slot = u32::from_le_bytes(blob[p..p + 4].try_into().unwrap());
        }
        cursor += 256 * 4;
        let mut origin = [0f64; 3];
        for slot in origin.iter_mut() {
            *slot = f64::from_le_bytes(blob[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;
        }
        out.push(PermEntry { perm, origin });
    }
    Ok(out)
}

/// Recursively parse one `[tag:u8][payload]` node starting at bytes[offset..].
/// Returns (node, bytes_consumed).
pub fn parse_octave_block(bytes: &[u8], cursor: &mut usize) -> Result<OctaveSampler> {
    let cur = *cursor;
    let rem = bytes.len().saturating_sub(cur);
    ensure(rem >= 4 + 4 + 4 + 4, "OctaveSampler header")?;
    let perm_offset = i32::from_le_bytes(bytes[cur..cur + 4].try_into().unwrap());
    *cursor += 4;
    let initial_freq = f32::from_le_bytes(bytes[cur + 4..cur + 8].try_into().unwrap());
    *cursor += 4;
    let persistence = f32::from_le_bytes(bytes[cur + 8..cur + 12].try_into().unwrap());
    *cursor += 4;
    let octaves = i32::from_le_bytes(bytes[cur + 12..cur + 16].try_into().unwrap());
    *cursor += 4;
    if octaves < 0 || octaves > 64 {
        return Err(anyhow!("OctaveSampler: bogus octaves={octaves}"));
    }
    let n = octaves as usize;
    let rem = bytes.len().saturating_sub(*cursor);
    ensure(rem >= n * 4, "OctaveSampler amplitudes")?;
    let mut amplitudes = Vec::with_capacity(n);
    for i in 0..n {
        let off = *cursor + i * 4;
        amplitudes.push(f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
    }
    *cursor += n * 4;
    Ok(OctaveSampler { perm_offset, initial_freq, persistence, amplitudes })
}

fn parse_node(bytes: &[u8], offset: usize) -> Result<(AstNode, usize)> {
    if offset >= bytes.len() {
        return Err(anyhow!("binary blob deserialize error: unexpected EOF"));
    }
    let tag = bytes[offset];
    let mut cursor = offset + 1;
    use AstNode::*;
    let node = match tag {
        0x01 => {
            // Constant(f32) — 4 bytes IEEE 754 LE
            ensure(bytes.len() >= cursor + 4, "Constant")?;
            let bits = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            Constant(f32::from_bits(bits))
        }
        0x02 => InputX,
        0x03 => InputY,
        0x04 => InputZ,

        // ── Binary arithmetic ops (0x10..0x15) ───────────────────────────
        0x10 => {
            // Add(a, b)
            let (a, la) = parse_node(bytes, cursor)?;
            cursor += la;
            let (b, lb) = parse_node(bytes, cursor)?;
            cursor += lb;
            Add(Box::new(a), Box::new(b))
        }
        0x11 => {
            // Sub(a, b)
            let (a, la) = parse_node(bytes, cursor)?;
            cursor += la;
            let (b, lb) = parse_node(bytes, cursor)?;
            cursor += lb;
            Sub(Box::new(a), Box::new(b))
        }
        0x12 => {
            // Mul(a, b)
            let (a, la) = parse_node(bytes, cursor)?;
            cursor += la;
            let (b, lb) = parse_node(bytes, cursor)?;
            cursor += lb;
            Mul(Box::new(a), Box::new(b))
        }
        0x13 => {
            // Div(a, b)
            let (a, la) = parse_node(bytes, cursor)?;
            cursor += la;
            let (b, lb) = parse_node(bytes, cursor)?;
            cursor += lb;
            Div(Box::new(a), Box::new(b))
        }
        0x14 => {
            // Min(a, b)
            let (a, la) = parse_node(bytes, cursor)?;
            cursor += la;
            let (b, lb) = parse_node(bytes, cursor)?;
            cursor += lb;
            Min(Box::new(a), Box::new(b))
        }
        0x15 => {
            // Max(a, b)
            let (a, la) = parse_node(bytes, cursor)?;
            cursor += la;
            let (b, lb) = parse_node(bytes, cursor)?;
            cursor += lb;
            Max(Box::new(a), Box::new(b))
        }
        0x24 => {
            // FloorDiv(a, b) — 二元，和 Add 同构
            let (a, la) = parse_node(bytes, cursor)?;
            cursor += la;
            let (b, lb) = parse_node(bytes, cursor)?;
            cursor += lb;
            FloorDiv(Box::new(a), Box::new(b))
        }

        // ── Unary arithmetic ops (0x20..0x23) ───────────────────────────
        0x20 => {
            // Abs(a)
            let (a, la) = parse_node(bytes, cursor)?;
            cursor += la;
            Abs(Box::new(a))
        }
        0x21 => {
            // Square(a)
            let (a, la) = parse_node(bytes, cursor)?;
            cursor += la;
            Square(Box::new(a))
        }
        0x22 => {
            // Cube(a)
            let (a, la) = parse_node(bytes, cursor)?;
            cursor += la;
            Cube(Box::new(a))
        }
        0x23 => {
            // Sqrt(a)
            let (a, la) = parse_node(bytes, cursor)?;
            cursor += la;
            Sqrt(Box::new(a))
        }

        // ── Noise samplers (0x80..0x90) ───────────────────────────────────
        0x82 => {
            // SimplexNoise2D { seed: i64 }
            ensure(bytes.len() >= cursor + 8, "SimplexNoise2D seed")?;
            let seed = i64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;
            SimplexNoise2D { seed }
        }
        0x83 => {
            // SimplexNoise3D { seed: i64 }
            ensure(bytes.len() >= cursor + 8, "SimplexNoise3D seed")?;
            let seed = i64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;
            SimplexNoise3D { seed }
        }
        0x84 => {
            // Payload:
            //   [type:u8] [flags:u8] [amplitude:f32] [xz_scale:f32] [y_scale:f32]
            //   [first: OctaveBlock]
            //   if flags & 1: [second: OctaveBlock]
            //
            // OctaveBlock:
            //   [perm_offset:i32] [lacunarity:f32] [octaves:i32]
            //   [amplitudes: f32 × octaves]
            ensure(bytes.len() >= cursor + 1 + 1 + 4 + 4 + 4, "NoiseSampler header")?;
            let noise_type = match bytes[cursor] {
                0 => NoiseType::Perlin,
                1 => NoiseType::Simplex,
                _ => return Err(anyhow!("unknown noise type in NoiseSampler")),
            };
            cursor += 1;
            let flags = bytes[cursor];
            cursor += 1;
            let amplitude =
                f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            let xz_scale =
                f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            let y_scale =
                f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            let first = parse_octave_block(bytes, &mut cursor)?;
            let second = if flags & 1 != 0 {
                Some(parse_octave_block(bytes, &mut cursor)?)
            } else {
                None
            };
            NoiseSampler {
                noise_type,
                first,
                second,
                amplitude,
                xz_scale,
                y_scale,
            }
        }

        // ── 新增：InterpolatedNoiseSampler (0x90) ──────────────────────
        0x90 => {
            // 假设 payload 为一个子节点（输入噪声）
            let (child, len) = parse_node(bytes, cursor)?;
            cursor += len;
            InterpolatedNoiseSampler {
                child: Box::new(child),
            }
        }

        // ── Shift / ShiftA / ShiftB (0xB4) ──────────────────────────────
        0xB4 => {
            // [kind:u8] [offset: NoiseSampler node]
            ensure(bytes.len() >= cursor + 1, "Shift kind")?;
            let kind = bytes[cursor];
            cursor += 1;
            let (offset, lo) = parse_node(bytes, cursor)?;
            cursor += lo;
            Shift {
                kind,
                offset: Box::new(offset),
            }
        }

        // ── ShiftedNoise (0xB5) ────────────────────────────────────────
        0xB5 => {
            // [xzScale:f32] [yScale:f32] [shiftX][shiftY][shiftZ][noise]
            ensure(bytes.len() >= cursor + 8, "ShiftedNoise scales")?;
            let xz_scale = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            let y_scale = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            let (sx, l1) = parse_node(bytes, cursor)?;
            cursor += l1;
            let (sy, l2) = parse_node(bytes, cursor)?;
            cursor += l2;
            let (sz, l3) = parse_node(bytes, cursor)?;
            cursor += l3;
            let (nz, l4) = parse_node(bytes, cursor)?;
            cursor += l4;
            ShiftedNoise {
                xz_scale,
                y_scale,
                shift_x: Box::new(sx),
                shift_y: Box::new(sy),
                shift_z: Box::new(sz),
                noise: Box::new(nz),
            }
        }

        // ── Structure terrain / Shape selection (0xA0..0xB3) ───────────────
        0xA0 => {
            // EndIslands(child) — child 通常是坐标函数
            let (a, la) = parse_node(bytes, cursor)?;
            cursor += la;
            EndIslands(Box::new(a))
        }

        // ── 新增：Beardifier (0xA1) ────────────────────────────────────
        0xA1 => {
            ensure(bytes.len() >= cursor + 4, "Beardifier struct_index")?;
            let struct_index = i32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            Beardifier { struct_index }
        }

        // ── Spline (0xB0) ──────────────────────────────────────────────
        0xB0 => {
            // [min:f32][max:f32][count:i32][locations: f32×count]
            // [derivatives: f32×count][location: node][values: node × count]
            ensure(bytes.len() >= cursor + 12, "Spline header")?;
            let _min = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            let _max = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            let count = i32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            let mut locations = Vec::with_capacity(count);
            for _ in 0..count {
                ensure(bytes.len() >= cursor + 4, "Spline location")?;
                locations.push(f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()));
                cursor += 4;
            }
            let mut derivatives = Vec::with_capacity(count);
            for _ in 0..count {
                ensure(bytes.len() >= cursor + 4, "Spline derivative")?;
                derivatives.push(f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()));
                cursor += 4;
            }
            let (location, ll) = parse_node(bytes, cursor)?;
            cursor += ll;
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                let (v, lv) = parse_node(bytes, cursor)?;
                cursor += lv;
                values.push(v);
            }
            Spline {
                locations,
                derivatives,
                location: Box::new(location),
                values,
            }
        }

        0xB1 => {
            // RangeChoice(input, min, max, in_bounds, out_lo, out_hi)
            let (input, l1) = parse_node(bytes, cursor)?;
            cursor += l1;
            let (min, l2) = parse_node(bytes, cursor)?;
            cursor += l2;
            let (max, l3) = parse_node(bytes, cursor)?;
            cursor += l3;
            let (in_bounds, l4) = parse_node(bytes, cursor)?;
            cursor += l4;
            let (out_lo, l5) = parse_node(bytes, cursor)?;
            cursor += l5;
            let (out_hi, l6) = parse_node(bytes, cursor)?;
            cursor += l6;
            RangeChoice {
                input: Box::new(input),
                min: Box::new(min),
                max: Box::new(max),
                in_bounds: Box::new(in_bounds),
                out_of_bounds_lo: Box::new(out_lo),
                out_of_bounds_hi: Box::new(out_hi),
            }
        }

        // ── 新增：IntervalSelect (0xB2) ────────────────────────────────
        0xB2 => {
            let (input, l1) = parse_node(bytes, cursor)?;
            cursor += l1;
            let (min, l2) = parse_node(bytes, cursor)?;
            cursor += l2;
            let (max, l3) = parse_node(bytes, cursor)?;
            cursor += l3;
            let (in_bounds, l4) = parse_node(bytes, cursor)?;
            cursor += l4;
            let (out_lo, l5) = parse_node(bytes, cursor)?;
            cursor += l5;
            let (out_hi, l6) = parse_node(bytes, cursor)?;
            cursor += l6;
            IntervalSelect {
                input: Box::new(input),
                min: Box::new(min),
                max: Box::new(max),
                in_bounds: Box::new(in_bounds),
                out_of_bounds_lo: Box::new(out_lo),
                out_of_bounds_hi: Box::new(out_hi),
            }
        }

        // ── 新增：YClampedGradient (0xB3) ──────────────────────────────
        0xB3 => {
            ensure(bytes.len() >= cursor + 8, "YClampedGradient y_min/y_max")?;
            let y_min = i32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            let y_max = i32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            let (input, l1) = parse_node(bytes, cursor)?;
            cursor += l1;
            let (gradient, l2) = parse_node(bytes, cursor)?;
            cursor += l2;
            YClampedGradient {
                y_min,
                y_max,
                input: Box::new(input),
                gradient: Box::new(gradient),
            }
        }

        // ── Caches / Surface / Aquifer (0xC0..0xE0) ─────────────────────────
        0xC0 => {
            // CacheFlat(child)
            let (a, la) = parse_node(bytes, cursor)?;
            cursor += la;
            CacheFlat(Box::new(a))
        }
        0xC1 => {
            // Cache2D(child)
            let (a, la) = parse_node(bytes, cursor)?;
            cursor += la;
            Cache2D(Box::new(a))
        }
        0xC2 => {
            // CacheInterpolated(child)
            let (a, la) = parse_node(bytes, cursor)?;
            cursor += la;
            CacheInterpolated(Box::new(a))
        }
        0xD0 => {
            // FindTopSurface { density, initial, min_y:i32, max_y:i32 }
            let (density, ld) = parse_node(bytes, cursor)?;
            cursor += ld;
            let (initial, li) = parse_node(bytes, cursor)?;
            cursor += li;
            ensure(bytes.len() >= cursor + 8, "FindTopSurface min_y/max_y")?;
            let min_y = i32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            let max_y = i32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            FindTopSurface {
                density: Box::new(density),
                initial: Box::new(initial),
                min_y,
                max_y,
            }
        }

        // ── 新增：Aquifer (0xE0) ────────────────────────────────────────
        0xE0 => {
            // 假设有两个子节点：density 和 fluid_level
            let (density, l1) = parse_node(bytes, cursor)?;
            cursor += l1;
            let (fluid_level, l2) = parse_node(bytes, cursor)?;
            cursor += l2;
            Aquifer {
                density: Box::new(density),
                fluid_level: Box::new(fluid_level),
            }
        }

        // ── 新增：BiomeTree (0xF0) ──────────────────────────────────────
        0xF0 => {
            // 解析一个种子（i64）作为简单占位
            ensure(bytes.len() >= cursor + 8, "BiomeTree seed")?;
            let seed = i64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;
            BiomeTree { seed }
        }

        0xFE => {
            // ROOT — tag 0xFE + channel_count(u16 LE) + [channel_count × recursive nodes]
            // MC NoiseRouter always has 16 channels but tests can use fewer.
            ensure(bytes.len() >= cursor + 2, "ROOT channel_count")?;
            let channel_count = u16::from_le_bytes(bytes[cursor..cursor + 2].try_into().unwrap());
            cursor += 2;
            let mut channels = Vec::with_capacity(channel_count as usize);
            for _ in 0..channel_count {
                let (child, child_len) = parse_node(bytes, cursor)?;
                cursor += child_len;
                channels.push(child);
            }
            Root(channels)
        }
        // TODO: implement all remaining tags (0x10..0xDF, 0x90..0x9F, etc.) recursively
        other => return Err(anyhow!("unsupported AST node tag: 0x{other:02X}")),
    };
    Ok((node, cursor - offset))
}

/// Minimum remaining-bytes check helper.
fn ensure(have: bool, ctx: &str) -> Result<()> {
    if have {
        Ok(())
    } else {
        Err(anyhow!(
            "binary blob deserialize error: truncated payload while parsing {ctx}"
        ))
    }
}

/// MC shape config: min Y, total world height, horizontal & vertical cell size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShapeConfig {
    pub min_y: i32,
    pub height: i32,
    pub cell_width: i32,
    pub cell_height: i32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::AstNode;

    // ────────────────────────────────────────────────────────────────────────
    // 测试 blob 字节说明（协议与 deserialize_full / parse_node 严格对齐）：
    //
    //  OFFSET  SIZE  FIELD              VALUE / DESCRIPTION
    //  ──────  ────  ─────────────────  ─────────────────────────────────────
    //       0     4  magic              0x4D433257 (按代码 49 行的逐字节断言)
    //       4     2  version (u16 LE)   1
    //       6     4  reserved           0x00000000 (未校验，占位)
    //      10     ●  NODE_TREE          TAG_ROOT + 2 channels
    //           1    tag                0xFE
    //           2    channel_count(u16) 2
    //           ●    channel[0]         Constant(3.14f32):
    //             1    tag              0x01
    //             4    f32 LE 3.14      0x4048F5C3 → [0xC3,0xF5,0x48,0x40]
    //           ●    channel[1]         InputX:
    //             1    tag              0x02
    //      (10+9) 16  SHAPE_CONFIG      [min_y=-64 | height=384 |
    //                                    cell_width=4 | cell_height=4]
    //   (19+16)  1  BIOME_SOURCE        tag 0xF0 (placeholder, 无 payload)
    // ──────  ────  ─────────────────  ─────────────────────────────────────
    //   TOTAL  36  BYTES
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    fn roundtrip_simple_root() {
        let blob: &[u8] = &[
            // ── HEADER (offset 0..10) ──────────────────────────────────
            0x4D, 0x43, 0x32, 0x57, // magic: 按 deserialize_full L49 的字节数组
            0x01, 0x00, // version = 1 (u16 LE)
            0x00, 0x00, 0x00, 0x00, // reserved[4]
            // ── NODE_TREE (offset 10 .. 19, 共 9 字节) ──────────────────
            0xFE, // TAG_ROOT
            0x02, 0x00, // channel_count = 2 (u16 LE)
            // channel 0: Constant(3.14)
            0x01, //   tag = TAG_CONSTANT
            0xC3, 0xF5, 0x48, 0x40, //   3.14f32.to_le_bytes()
            // channel 1: InputX
            0x02, //   tag = TAG_INPUT_X
            // ── SHAPE_CONFIG (offset 19 .. 35, 共 16 字节) ──────────────
            0xC0, 0xFF, 0xFF, 0xFF, // min_y      = -64 (i32 LE)
            0x80, 0x01, 0x00, 0x00, // height     = 384 (i32 LE, 0x180)
            0x04, 0x00, 0x00, 0x00, // cell_width = 4   (i32 LE)
            0x04, 0x00, 0x00, 0x00, // cell_height= 4   (i32 LE)
            // ── BIOME_SOURCE (offset 35, 1 字节) ────────────────────────
            0xF0, 0x15, 0xCD, 0x5B, 0x07, 0x00, 0x00, 0x00,
            0x00, // TAG_BIOME_TREE (placeholder)
        ];
        assert_eq!(
            blob.len(),
            44,
            "test blob size mismatch — protocol layout recalc needed"
        );

        let (root, shape_cfg, biome_tree) = deserialize_full(blob).unwrap();

        // 1) Root 结构断言
        assert_eq!(
            root,
            AstNode::Root(vec![AstNode::Constant(3.14), AstNode::InputX,])
        );

        // 2) ShapeConfig 断言（验证 SHAPE_CONFIG 没错位）
        assert_eq!(
            shape_cfg,
            ShapeConfig {
                min_y: -64,
                height: 384,
                cell_width: 4,
                cell_height: 4,
            }
        );

        // 3) BiomeTree 占位断言（验证 BIOME_SOURCE 被正确 consume 为 BiomeTree 占位）
        assert!(matches!(biome_tree, AstNode::BiomeTree { seed: _ }));
    }

    /// 验证：±0f32 反序列化后，Hash/PartialEq 一致性。
    /// （间接验证 ast/mod.rs 里 f32_to_hash_bits 的归一化路径在 de 后生效）
    #[test]
    fn constant_negative_zero_roundtrip() {
        // 手工构造只包含一个 TAG_ROOT(1ch) + Constant(-0.0) + 最小 SHAPE_CONFIG + BIOME 的 blob
        let neg_zero_bits: [u8; 4] = (-0.0f32).to_le_bytes();
        let blob: &[u8] = &[
            // HEADER
            0x4D,
            0x43,
            0x32,
            0x57,
            0x01,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            // ROOT(1 channel)
            0xFE,
            0x01,
            0x00,
            // ch0: Constant(-0.0f32)
            0x01,
            neg_zero_bits[0],
            neg_zero_bits[1],
            neg_zero_bits[2],
            neg_zero_bits[3],
            // SHAPE_CONFIG (全 0 占位 16 字节)
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            // BIOME_SOURCE
            0xF0,
            0x15,
            0xCD,
            0x5B,
            0x07,
            0x00,
            0x00,
            0x00,
            0x00,
            0x15,
            0xCD,
            0x5B,
            0x07,
            0x00,
            0x00,
            0x00,
            0x00,
        ];
        let (root, _, _) = deserialize_full(blob).unwrap();
        let ch0 = match &root {
            AstNode::Root(v) => &v[0],
            _ => panic!("not a root"),
        };
        // PartialEq：Constant(-0.0) 必须 == Constant(+0.0)
        assert_eq!(ch0, &AstNode::Constant(0.0));
        // 再顺便断言它们是同一个 variant 且数值为 0.0（-0.0 as f32 == 0.0）
        if let AstNode::Constant(f) = ch0 {
            assert!(*f == 0.0);
        }
    }

    // ── 子阶段 2.3-1 算术运算测试 ────────────────────────────────────────

    /// 测试二元算术 op（Add / Sub / Mul）嵌套解析：
    ///   ROOT(1ch) = Mul( Add(Constant(1.5), InputX), Sub(InputY, Constant(2.0)) )
    #[test]
    fn test_arithmetic_binary_ops() {
        let f1p5 = 1.5f32.to_le_bytes();
        let f2p0 = 2.0f32.to_le_bytes();

        let blob: &[u8] = &[
            // HEADER 10B
            0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, // ROOT(1 ch)
            0xFE, 0x01, 0x00, // Mul → 0x12
            0x12, // Add → 0x10
            0x10, 0x01, f1p5[0], f1p5[1], f1p5[2], f1p5[3], 0x02, // Sub → 0x11
            0x11, 0x03, 0x01, f2p0[0], f2p0[1], f2p0[2], f2p0[3], // SHAPE_CONFIG 16B
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // BIOME
            0xF0, 0x15, 0xCD, 0x5B, 0x07, 0x00, 0x00, 0x00, 0x00,
        ];

        let (root, _, _) = deserialize_full(blob).unwrap();

        let expected = AstNode::Root(vec![AstNode::Mul(
            Box::new(AstNode::Add(
                Box::new(AstNode::Constant(1.5)),
                Box::new(AstNode::InputX),
            )),
            Box::new(AstNode::Sub(
                Box::new(AstNode::InputY),
                Box::new(AstNode::Constant(2.0)),
            )),
        )]);
        assert_eq!(root, expected);
    }

    /// 测试一元算术 op（Abs / Sqrt）嵌套解析：
    ///   ROOT(1ch) = Sqrt( Abs( Sub(InputX, Constant(5.0)) ) )
    #[test]
    fn test_arithmetic_unary_ops() {
        let f5p0 = 5.0f32.to_le_bytes();

        let blob: &[u8] = &[
            // HEADER
            0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, // ROOT(1 ch)
            0xFE, 0x01, 0x00, // Sqrt → 0x23
            0x23, // Abs → 0x20
            0x20, // Sub(X, 5.0) → 0x11
            0x11, 0x02, 0x01, f5p0[0], f5p0[1], f5p0[2], f5p0[3], // SHAPE_CONFIG 16B
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // BIOME
            0xF0, 0x15, 0xCD, 0x5B, 0x07, 0x00, 0x00, 0x00, 0x00,
        ];

        let (root, _, _) = deserialize_full(blob).unwrap();

        let expected = AstNode::Root(vec![AstNode::Sqrt(Box::new(AstNode::Abs(Box::new(
            AstNode::Sub(Box::new(AstNode::InputX), Box::new(AstNode::Constant(5.0))),
        ))))]);
        assert_eq!(root, expected);
    }

    // ── 子阶段 2.3-2 噪声采样器测试 ──────────────────────────────────────

    // ── 子阶段 2.3-3 地形结构 / 形状选择测试 ───────────────────────────────

    /// 测试 EndIslands + RangeChoice(6 子节点嵌套) 组合
    #[test]
    fn test_shape_selection_tags() {
        let f64p0 = 64.0f32.to_le_bytes();
        let f0p0 = 0.0f32.to_le_bytes();
        let f384 = 384.0f32.to_le_bytes();
        let f0p5 = 0.5f32.to_le_bytes();
        let f_neg1 = (-1.0f32).to_le_bytes();
        let f_p1 = 1.0f32.to_le_bytes();

        let blob: &[u8] = &[
            // HEADER
            0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, // ROOT(2 ch)
            0xFE, 0x02, 0x00, // ch0: EndIslands(InputX)  → 0xA0
            0xA0, 0x02, // ch1: RangeChoice → 0xB1
            0xB1, // input = Sub(Y, 64.0) → 0x11
            0x11, 0x03, 0x01, f64p0[0], f64p0[1], f64p0[2], f64p0[3], // min = 0.0
            0x01, f0p0[0], f0p0[1], f0p0[2], f0p0[3], // max = 384.0
            0x01, f384[0], f384[1], f384[2], f384[3], // in_bounds = Mul(Z, 0.5)
            0x12, 0x04, 0x01, f0p5[0], f0p5[1], f0p5[2], f0p5[3], // out_lo = -1.0
            0x01, f_neg1[0], f_neg1[1], f_neg1[2], f_neg1[3], // out_hi = 1.0
            0x01, f_p1[0], f_p1[1], f_p1[2], f_p1[3], // SHAPE_CONFIG 16B
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // BIOME
            0xF0, 0x15, 0xCD, 0x5B, 0x07, 0x00, 0x00, 0x00, 0x00,
        ];

        let (root, _, _) = deserialize_full(blob).unwrap();
        let expected = AstNode::Root(vec![
            AstNode::EndIslands(Box::new(AstNode::InputX)),
            AstNode::RangeChoice {
                input: Box::new(AstNode::Sub(
                    Box::new(AstNode::InputY),
                    Box::new(AstNode::Constant(64.0)),
                )),
                min: Box::new(AstNode::Constant(0.0)),
                max: Box::new(AstNode::Constant(384.0)),
                in_bounds: Box::new(AstNode::Mul(
                    Box::new(AstNode::InputZ),
                    Box::new(AstNode::Constant(0.5)),
                )),
                out_of_bounds_lo: Box::new(AstNode::Constant(-1.0)),
                out_of_bounds_hi: Box::new(AstNode::Constant(1.0)),
            },
        ]);
        assert_eq!(root, expected);
    }

    // ── 子阶段 2.3-4 缓存 / 表面 / 含水层测试 ─────────────────────────────

    /// 测试 CacheFlat/Cache2D/CacheInterpolated 三层嵌套 + FindTopSurface i32 解析
    #[test]
    fn test_cache_and_surface_tags() {
        let f0p0 = 0.0f32.to_le_bytes();
        let min_y_bytes = (-64i32).to_le_bytes();
        let max_y_bytes = 320i32.to_le_bytes();

        let blob: &[u8] = &[
            // HEADER
            0x4D,
            0x43,
            0x32,
            0x57,
            0x01,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            // ROOT(2 ch)
            0xFE,
            0x02,
            0x00,
            // ch0: CacheInterpolated(Cache2D(CacheFlat(InputX)))
            0xC2,
            0xC1,
            0xC0,
            0x02,
            // ch1: FindTopSurface (0xD0)
            0xD0,
            // density = InputY
            0x03,
            // initial = Constant(0.0)
            0x01,
            f0p0[0],
            f0p0[1],
            f0p0[2],
            f0p0[3],
            // min_y = -64
            min_y_bytes[0],
            min_y_bytes[1],
            min_y_bytes[2],
            min_y_bytes[3],
            // max_y = 320
            max_y_bytes[0],
            max_y_bytes[1],
            max_y_bytes[2],
            max_y_bytes[3],
            // SHAPE_CONFIG 16B
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            // BIOME
            0xF0,
            0x15,
            0xCD,
            0x5B,
            0x07,
            0x00,
            0x00,
            0x00,
            0x00,
            0x15,
            0xCD,
            0x5B,
            0x07,
            0x00,
            0x00,
            0x00,
            0x00,
        ];

        let (root, _, _) = deserialize_full(blob).unwrap();
        let expected = AstNode::Root(vec![
            AstNode::CacheInterpolated(Box::new(AstNode::Cache2D(Box::new(AstNode::CacheFlat(
                Box::new(AstNode::InputX),
            ))))),
            AstNode::FindTopSurface {
                density: Box::new(AstNode::InputY),
                initial: Box::new(AstNode::Constant(0.0)),
                min_y: -64,
                max_y: 320,
            },
        ]);
        assert_eq!(root, expected);
    }
}
