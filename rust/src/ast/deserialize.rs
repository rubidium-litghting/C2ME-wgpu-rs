// SPDX-License-Identifier: MIT

//! Deserialize Java NoiseRouterBlobifier binary format → AstNode.
//!
//! Protocol:
//!   Header:  magic(u32=0x4D433257 LE)  version(u16=1)  reserved([u8;4])
//!   Tree  :  [tag:u8][payload]  (recursive)
//!   Tail  :  SHAPE_CONFIG  BIOME_SOURCE
//!
//! Byte order: Little Endian.
//!
//! Every tag MUST match NoiseRouterBlobifier.java constants exactly.

use anyhow::anyhow;

use crate::Result;
use crate::ast::{
    AstNode, NoiseType, OctaveSampler,
    TAG_ABS, TAG_ADD, TAG_AQUIFER, TAG_BEARDIFIER, TAG_BIOME_TREE,
    TAG_CACHE_2D, TAG_CACHE_FLAT, TAG_CACHE_INTERP, TAG_CONSTANT,
    TAG_CUBE, TAG_DIV, TAG_END_ISLANDS, TAG_EXTERNAL_INPUT,
    TAG_FIND_TOP_SURF, TAG_FLOOR_DIV, TAG_INPUT_X, TAG_INPUT_Y,
    TAG_INPUT_Z, TAG_INTERP_NOISE, TAG_INTERP_NOISE_SAMPLER,
    TAG_INTERVAL_SEL, TAG_MAX, TAG_MIN, TAG_MUL, TAG_NOISE_SAMPLER,
    TAG_RANGE_CHOICE, TAG_ROOT, TAG_SHIFT, TAG_SHIFTED_NOISE,
    TAG_SIMPLEX_2D, TAG_SIMPLEX_3D, TAG_SPLINE, TAG_SQRT, TAG_SQUARE,
    TAG_SQUEEZE, TAG_SUB, TAG_WEIRD_SCALED, TAG_Y_CLAMPED_GRAD,
};

/// A single Perlin permutation table + its origin offset, as shipped in the
/// optional PERM_RESOURCE trailer appended by the Java side.
#[derive(Debug, Clone, PartialEq)]
pub struct PermEntry {
    pub perm: [u32; 256],
    pub origin: [f64; 3],
}

/// Magic of the PERM_RESOURCE trailer: "PWPM" little-endian.
pub const PERM_MAGIC: u32 = 0x4D52_5750;

/// A captured SimplexNoiseSampler table (EndIslands): int[512] permutation
/// plus its origin offset, shipped in the optional SIMPLEX_RESOURCE trailer.
#[derive(Debug, Clone, PartialEq)]
pub struct SimplexEntry {
    pub perm: [i32; 512],
    pub origin: [f64; 3],
}

/// Magic of the SIMPLEX_RESOURCE trailer: "SXRM" little-endian.
pub const SIMPLEX_MAGIC: u32 = 0x4D52_5358;

/// Bounds-checked little-endian cursor over a blob.
///
/// Every read in this module goes through this type, so a truncated or
/// malformed blob yields an `Err` naming the exact field that ran out of
/// bytes instead of a `unwrap()` panic or a silent misparse. Adding a new
/// node no longer requires hand-computing offsets or writing guard checks.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize, ctx: &str) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(anyhow!(
                "binary blob deserialize error: truncated payload while parsing {ctx}"
            ));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u8(&mut self, ctx: &str) -> Result<u8> {
        Ok(self.take(1, ctx)?[0])
    }
    fn u16(&mut self, ctx: &str) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2, ctx)?.try_into().unwrap()))
    }
    fn u32(&mut self, ctx: &str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4, ctx)?.try_into().unwrap()))
    }
    fn i32(&mut self, ctx: &str) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4, ctx)?.try_into().unwrap()))
    }
    fn i64(&mut self, ctx: &str) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8, ctx)?.try_into().unwrap()))
    }
    fn f32(&mut self, ctx: &str) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4, ctx)?.try_into().unwrap()))
    }
    fn f64(&mut self, ctx: &str) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8, ctx)?.try_into().unwrap()))
    }
}

pub fn deserialize_full(blob: &[u8]) -> Result<(AstNode, ShapeConfig, AstNode)> {
    let (root, shape, biome, _perm, _simplex) = deserialize_full_with_perm(blob)?;
    Ok((root, shape, biome))
}

pub fn deserialize_full_with_perm(
    blob: &[u8],
) -> Result<(AstNode, ShapeConfig, AstNode, Vec<PermEntry>, Vec<SimplexEntry>)> {
    let mut r = Reader::new(blob);

    // Header: magic(u32) version(u16) reserved([u8;4]).
    if r.take(4, "header")? != [0x4D, 0x43, 0x32, 0x57] {
        return Err(anyhow!("binary blob deserialize error: bad magic"));
    }
    let version = r.u16("header version")?;
    if version != 1 {
        return Err(anyhow!(
            "binary blob deserialize error: unsupported version {version}"
        ));
    }
    r.take(4, "header reserved")?;

    // NODE_TREE (Root) — SHAPE_CONFIG — BIOME_SOURCE.
    let root = parse_node(&mut r)?;
    let shape_cfg = ShapeConfig {
        min_y: r.i32("SHAPE_CONFIG min_y")?,
        height: r.i32("SHAPE_CONFIG height")?,
        cell_width: r.i32("SHAPE_CONFIG cell_width")?,
        cell_height: r.i32("SHAPE_CONFIG cell_height")?,
    };
    let biome_tree = parse_node(&mut r)?;

    // Optional trailers (per-table, then simplex table). Trailing bytes past
    // them are allowed for future extension.
    let perm_entries = parse_perm_resource(&mut r).unwrap_or_default();
    let simplex_entries = parse_simplex_resource(&mut r).unwrap_or_default();

    Ok((root, shape_cfg, biome_tree, perm_entries, simplex_entries))
}

/// Parse the optional PERM_RESOURCE trailer. Returns an empty Vec (and leaves
/// the cursor untouched) when the trailer is absent.
fn parse_perm_resource(r: &mut Reader<'_>) -> Result<Vec<PermEntry>> {
    let saved = r.pos;
    if r.remaining() < 8 {
        return Ok(Vec::new());
    }
    if r.u32("PERM_RESOURCE magic")? != PERM_MAGIC {
        r.pos = saved;
        return Ok(Vec::new());
    }
    let count = r.u32("PERM_RESOURCE count")? as usize;
    let entry_size = 256 * 4 + 3 * 8;
    let mut out = Vec::with_capacity(count.min(r.remaining() / entry_size));
    for _ in 0..count {
        if r.remaining() < entry_size {
            break; // tolerate a truncated trailer
        }
        let mut perm = [0u32; 256];
        for slot in perm.iter_mut() {
            *slot = r.u32("PERM_RESOURCE perm")?;
        }
        let mut origin = [0f64; 3];
        for slot in origin.iter_mut() {
            *slot = r.f64("PERM_RESOURCE origin")?;
        }
        out.push(PermEntry { perm, origin });
    }
    Ok(out)
}

/// Parse the optional SIMPLEX_RESOURCE trailer.
/// Each entry is int[512] (2048 B) + 3×f64 origin (24 B) = 2072 B.
fn parse_simplex_resource(r: &mut Reader<'_>) -> Result<Vec<SimplexEntry>> {
    let saved = r.pos;
    if r.remaining() < 8 {
        return Ok(Vec::new());
    }
    if r.u32("SIMPLEX_RESOURCE magic")? != SIMPLEX_MAGIC {
        r.pos = saved;
        return Ok(Vec::new());
    }
    let count = r.u32("SIMPLEX_RESOURCE count")? as usize;
    let entry_size = 512 * 4 + 3 * 8;
    let mut out = Vec::with_capacity(count.min(r.remaining() / entry_size));
    for _ in 0..count {
        if r.remaining() < entry_size {
            break;
        }
        let mut perm = [0i32; 512];
        for slot in perm.iter_mut() {
            *slot = r.i32("SIMPLEX_RESOURCE perm")?;
        }
        let mut origin = [0f64; 3];
        for slot in origin.iter_mut() {
            *slot = r.f64("SIMPLEX_RESOURCE origin")?;
        }
        out.push(SimplexEntry { perm, origin });
    }
    Ok(out)
}

/// Parse one `OctaveBlock` payload:
///   [perm_offset:i32] [initial_freq:f32] [persistence:f32]
///   [octaves:i32] [amplitudes: f32 × octaves]
fn parse_octave_block(r: &mut Reader<'_>) -> Result<OctaveSampler> {
    let perm_offset = r.i32("OctaveSampler perm_offset")?;
    let initial_freq = r.f32("OctaveSampler initial_freq")?;
    let persistence = r.f32("OctaveSampler persistence")?;
    let octaves = r.i32("OctaveSampler octaves")?;
    if !(0..=64).contains(&octaves) {
        return Err(anyhow!("OctaveSampler: bogus octaves={octaves}"));
    }
    let n = octaves as usize;
    let amp_bytes = r.take(n * 4, "OctaveSampler amplitudes")?;
    let mut amplitudes = Vec::with_capacity(n);
    for chunk in amp_bytes.chunks_exact(4) {
        amplitudes.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(OctaveSampler { perm_offset, initial_freq, persistence, amplitudes })
}

// ── Child-parsing helpers ───────────────────────────────────────────────
// Most nodes are "N recursively parsed children, optionally with scalar
// payloads interleaved". These helpers collapse the repeated
// `parse_node(...)?; cursor += len;` boilerplate.

fn sub(r: &mut Reader<'_>) -> Result<Box<AstNode>> {
    Ok(Box::new(parse_node(r)?))
}

fn binary(r: &mut Reader<'_>) -> Result<(Box<AstNode>, Box<AstNode>)> {
    Ok((sub(r)?, sub(r)?))
}

fn sext(r: &mut Reader<'_>) -> Result<[Box<AstNode>; 6]> {
    Ok([sub(r)?, sub(r)?, sub(r)?, sub(r)?, sub(r)?, sub(r)?])
}

/// Recursively parse one `[tag:u8][payload]` node.
///
/// Tags are matched against the `TAG_*` constants in [`crate::ast`], which are
/// the single source of truth shared with `NoiseRouterBlobifier.java` and the
/// `Hash` impl. Adding a node means adding its constant there and one arm here.
fn parse_node(r: &mut Reader<'_>) -> Result<AstNode> {
    use AstNode::*;
    let tag = r.u8("node tag")?;
    Ok(match tag {
        TAG_CONSTANT => Constant(f32::from_bits(r.u32("Constant")?)),
        TAG_INPUT_X => InputX,
        TAG_INPUT_Y => InputY,
        TAG_INPUT_Z => InputZ,

        // ── Binary ops ──────────────────────────────────────────────────
        TAG_ADD => { let (a, b) = binary(r)?; Add(a, b) }
        TAG_SUB => { let (a, b) = binary(r)?; Sub(a, b) }
        TAG_MUL => { let (a, b) = binary(r)?; Mul(a, b) }
        TAG_DIV => { let (a, b) = binary(r)?; Div(a, b) }
        TAG_MIN => { let (a, b) = binary(r)?; Min(a, b) }
        TAG_MAX => { let (a, b) = binary(r)?; Max(a, b) }
        TAG_FLOOR_DIV => { let (a, b) = binary(r)?; FloorDiv(a, b) }

        // ── Unary ops ───────────────────────────────────────────────────
        TAG_ABS => Abs(sub(r)?),
        TAG_SQUARE => Square(sub(r)?),
        TAG_CUBE => Cube(sub(r)?),
        TAG_SQRT => Sqrt(sub(r)?),
        TAG_SQUEEZE => Squeeze(sub(r)?),

        // ── Noise samplers ──────────────────────────────────────────────
        TAG_SIMPLEX_2D => SimplexNoise2D { seed: r.i64("SimplexNoise2D seed")? },
        TAG_SIMPLEX_3D => SimplexNoise3D { seed: r.i64("SimplexNoise3D seed")? },
        TAG_NOISE_SAMPLER => {
            let noise_type = match r.u8("NoiseSampler type")? {
                0 => NoiseType::Perlin,
                1 => NoiseType::Simplex,
                other => return Err(anyhow!("unknown noise type in NoiseSampler: {other}")),
            };
            let flags = r.u8("NoiseSampler flags")?;
            let amplitude = r.f32("NoiseSampler amplitude")?;
            let xz_scale = r.f32("NoiseSampler xz_scale")?;
            let y_scale = r.f32("NoiseSampler y_scale")?;
            let first = parse_octave_block(r)?;
            let second = if flags & 1 != 0 {
                Some(parse_octave_block(r)?)
            } else {
                None
            };
            NoiseSampler { noise_type, first, second, amplitude, xz_scale, y_scale }
        }
        TAG_INTERP_NOISE => InterpolatedNoiseSampler { child: sub(r)? },
        TAG_SHIFT => {
            let kind = r.u8("Shift kind")?;
            Shift { kind, offset: sub(r)? }
        }
        TAG_SHIFTED_NOISE => {
            let xz_scale = r.f32("ShiftedNoise xz_scale")?;
            let y_scale = r.f32("ShiftedNoise y_scale")?;
            let shift_x = sub(r)?;
            let shift_y = sub(r)?;
            let shift_z = sub(r)?;
            let noise = sub(r)?;
            ShiftedNoise { xz_scale, y_scale, shift_x, shift_y, shift_z, noise }
        }
        TAG_WEIRD_SCALED => {
            let mapper = r.u8("WeirdScaled mapper")?;
            let input = sub(r)?;
            let noise = sub(r)?;
            WeirdScaledSampler { mapper, input, noise }
        }
        TAG_INTERP_NOISE_SAMPLER => InterpNoiseSampler {
            xz_scale: r.f32("InterpNoiseSampler xz_scale")?,
            y_scale: r.f32("InterpNoiseSampler y_scale")?,
            xz_factor: r.f32("InterpNoiseSampler xz_factor")?,
            y_factor: r.f32("InterpNoiseSampler y_factor")?,
            smear: r.f32("InterpNoiseSampler smear")?,
            lower_base: r.i32("InterpNoiseSampler lower_base")?,
            lower_count: r.i32("InterpNoiseSampler lower_count")?,
            upper_base: r.i32("InterpNoiseSampler upper_base")?,
            upper_count: r.i32("InterpNoiseSampler upper_count")?,
            interp_base: r.i32("InterpNoiseSampler interp_base")?,
            interp_count: r.i32("InterpNoiseSampler interp_count")?,
        },

        // ── Structure terrain ───────────────────────────────────────────
        TAG_END_ISLANDS => EndIslands { perm_index: r.i32("EndIslands perm_index")? },
        TAG_BEARDIFIER => Beardifier { struct_index: r.i32("Beardifier struct_index")? },

        // ── Shape / selection ───────────────────────────────────────────
        TAG_SPLINE => {
            // min/max are recomputable from `locations` and are dropped.
            let _min = r.f32("Spline min")?;
            let _max = r.f32("Spline max")?;
            let count = r.i32("Spline count")? as usize;
            let mut locations = Vec::with_capacity(count);
            for _ in 0..count { locations.push(r.f32("Spline location")?); }
            let mut derivatives = Vec::with_capacity(count);
            for _ in 0..count { derivatives.push(r.f32("Spline derivative")?); }
            let location = sub(r)?;
            let mut values = Vec::with_capacity(count);
            for _ in 0..count { values.push(parse_node(r)?); }
            Spline { locations, derivatives, location, values }
        }
        TAG_RANGE_CHOICE => {
            let [input, min, max, in_bounds, out_lo, out_hi] = sext(r)?;
            RangeChoice {
                input, min, max, in_bounds,
                out_of_bounds_lo: out_lo,
                out_of_bounds_hi: out_hi,
            }
        }
        TAG_INTERVAL_SEL => {
            let [input, min, max, in_bounds, out_lo, out_hi] = sext(r)?;
            IntervalSelect {
                input, min, max, in_bounds,
                out_of_bounds_lo: out_lo,
                out_of_bounds_hi: out_hi,
            }
        }
        TAG_Y_CLAMPED_GRAD => YClampedGradient {
            from_y: r.f32("YClampedGradient from_y")?,
            to_y: r.f32("YClampedGradient to_y")?,
            from_value: r.f32("YClampedGradient from_value")?,
            to_value: r.f32("YClampedGradient to_value")?,
        },

        // ── Caches / surface / aquifer ──────────────────────────────────
        TAG_CACHE_FLAT => CacheFlat(sub(r)?),
        TAG_CACHE_2D => Cache2D(sub(r)?),
        TAG_CACHE_INTERP => CacheInterpolated(sub(r)?),
        TAG_FIND_TOP_SURF => FindTopSurface {
            density: sub(r)?,
            initial: sub(r)?,
            min_y: r.i32("FindTopSurface min_y")?,
            max_y: r.i32("FindTopSurface max_y")?,
        },
        TAG_AQUIFER => Aquifer { density: sub(r)?, fluid_level: sub(r)? },

        // ── Biome / external / root ─────────────────────────────────────
        TAG_BIOME_TREE => BiomeTree { seed: r.i64("BiomeTree seed")? },
        TAG_EXTERNAL_INPUT => ExternalInput { index: r.i32("ExternalInput index")? },
        TAG_ROOT => {
            let channel_count = r.u16("ROOT channel_count")?;
            let mut channels = Vec::with_capacity(channel_count as usize);
            for _ in 0..channel_count { channels.push(parse_node(r)?); }
            Root(channels)
        }
        // All tags emitted by NoiseRouterBlobifier are handled above.
        // Unknown tags are a hard error (the Java side logs UNSUPPORTED NODE
        // and falls back to a constant 0, so they should not reach here).
        other => return Err(anyhow!("unsupported AST node tag: 0x{other:02X}")),
    })
}

/// MC shape config: min Y, total world height, horizontal & vertical cell size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShapeConfig {
    pub min_y: i32,
    pub height: i32,
    pub cell_width: i32,
    pub cell_height: i32,
}

impl ShapeConfig {
    /// Vertical aquifer layers needed to cover the world at vanilla's 12-block
    /// vertical spacing, plus the three layers it reserves around the range.
    ///
    /// This is the `size_y` of the staged aquifer table (see
    /// `AquiferTable::parse`), not the density grid's `height / cell_height`.
    pub fn aquifer_layers(&self) -> i32 {
        let top = self.min_y.saturating_add(self.height);
        top.div_euclid(12) - self.min_y.div_euclid(12) + 3
    }

    /// Aquifer table cells per chunk: a 3×3 grid in X/Z (48 blocks at vanilla
    /// 16-block columns) times [`Self::aquifer_layers`].
    ///
    /// Must track the shape rather than a fixed 1200 reservation: the table is
    /// uploaded into a buffer sized for `cells × max_batch`, and a dimension
    /// taller than ~2000 blocks has more cells than that.
    pub fn aquifer_cells_per_chunk(&self) -> usize {
        (self.aquifer_layers().max(1) as usize) * 9
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::AstNode;

    /// Test-only structural encoder for the NoiseRouterBlobifier wire format.
    ///
    /// Tests describe the tree shape (`Node::add(Node::input_x(), ...)`) and
    /// the envelope is emitted here once, instead of hand-assembling byte
    /// arrays whose offsets drift with the protocol. `roundtrip_simple_root`
    /// deliberately keeps its raw byte array: it is the independent wire
    /// canary that catches a builder/parser that drifted together.
    mod blob {
        use crate::ast::{
            TAG_ABS, TAG_ADD, TAG_BIOME_TREE, TAG_CACHE_2D, TAG_CACHE_FLAT,
            TAG_CACHE_INTERP, TAG_CONSTANT, TAG_END_ISLANDS, TAG_FIND_TOP_SURF,
            TAG_INPUT_X, TAG_INPUT_Y, TAG_INPUT_Z, TAG_MUL, TAG_RANGE_CHOICE,
            TAG_ROOT, TAG_SQRT, TAG_SUB,
        };

        /// A node's serialized bytes: `[tag:u8][payload]`.
        pub struct Node(Vec<u8>);

        impl Node {
            fn tagged(tag: u8, payload: impl AsRef<[u8]>) -> Self {
                let mut v = vec![tag];
                v.extend_from_slice(payload.as_ref());
                Node(v)
            }
            fn unary(tag: u8, child: Node) -> Self {
                let mut v = vec![tag];
                v.extend(child.0);
                Node(v)
            }
            fn binary(tag: u8, a: Node, b: Node) -> Self {
                let mut v = vec![tag];
                v.extend(a.0);
                v.extend(b.0);
                Node(v)
            }

            pub fn constant(v: f32) -> Self { Self::tagged(TAG_CONSTANT, v.to_le_bytes()) }
            pub fn input_x() -> Self { Node(vec![TAG_INPUT_X]) }
            pub fn input_y() -> Self { Node(vec![TAG_INPUT_Y]) }
            pub fn input_z() -> Self { Node(vec![TAG_INPUT_Z]) }

            pub fn add(a: Node, b: Node) -> Self { Self::binary(TAG_ADD, a, b) }
            pub fn sub(a: Node, b: Node) -> Self { Self::binary(TAG_SUB, a, b) }
            pub fn mul(a: Node, b: Node) -> Self { Self::binary(TAG_MUL, a, b) }

            pub fn abs(a: Node) -> Self { Self::unary(TAG_ABS, a) }
            pub fn sqrt(a: Node) -> Self { Self::unary(TAG_SQRT, a) }

            pub fn end_islands(perm_index: i32) -> Self {
                Self::tagged(TAG_END_ISLANDS, perm_index.to_le_bytes())
            }
            pub fn cache_flat(c: Node) -> Self { Self::unary(TAG_CACHE_FLAT, c) }
            pub fn cache_2d(c: Node) -> Self { Self::unary(TAG_CACHE_2D, c) }
            pub fn cache_interp(c: Node) -> Self { Self::unary(TAG_CACHE_INTERP, c) }

            pub fn range_choice(
                input: Node, min: Node, max: Node,
                in_bounds: Node, out_lo: Node, out_hi: Node,
            ) -> Self {
                let mut v = vec![TAG_RANGE_CHOICE];
                for n in [input, min, max, in_bounds, out_lo, out_hi] { v.extend(n.0); }
                Node(v)
            }
            pub fn find_top_surface(density: Node, initial: Node, min_y: i32, max_y: i32) -> Self {
                let mut v = vec![TAG_FIND_TOP_SURF];
                v.extend(density.0);
                v.extend(initial.0);
                v.extend(min_y.to_le_bytes());
                v.extend(max_y.to_le_bytes());
                Node(v)
            }
            pub fn root(channels: Vec<Node>) -> Self {
                let mut v = vec![TAG_ROOT];
                v.extend((channels.len() as u16).to_le_bytes());
                for c in channels { v.extend(c.0); }
                Node(v)
            }
            pub fn biome_tree(seed: i64) -> Self {
                Self::tagged(TAG_BIOME_TREE, seed.to_le_bytes())
            }
        }

        /// Envelope assembler: header + root tree + SHAPE_CONFIG + BIOME_SOURCE.
        pub struct BlobBuilder {
            root: Node,
            shape: [i32; 4],
            biome: Node,
        }

        pub fn blob(root: Node) -> BlobBuilder {
            BlobBuilder {
                root,
                shape: [-64, 384, 4, 4],
                biome: Node::biome_tree(0),
            }
        }

        impl BlobBuilder {
            pub fn shape(mut self, min_y: i32, height: i32, cell_width: i32, cell_height: i32) -> Self {
                self.shape = [min_y, height, cell_width, cell_height];
                self
            }
            pub fn biome(mut self, seed: i64) -> Self {
                self.biome = Node::biome_tree(seed);
                self
            }
            pub fn build(self) -> Vec<u8> {
                let mut v = vec![0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0, 0, 0, 0];
                v.extend(self.root.0);
                for x in self.shape { v.extend(x.to_le_bytes()); }
                v.extend(self.biome.0);
                v
            }
        }
    }


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

    /// 二元算术 op 嵌套解析：
    ///   ROOT(1ch) = Mul( Add(Constant(1.5), InputX), Sub(InputY, Constant(2.0)) )
    #[test]
    fn test_arithmetic_binary_ops() {
        use blob::Node;
        let root = Node::root(vec![Node::mul(
            Node::add(Node::constant(1.5), Node::input_x()),
            Node::sub(Node::input_y(), Node::constant(2.0)),
        )]);
        let (parsed, _, _) = deserialize_full(&blob::blob(root).build()).unwrap();

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
        assert_eq!(parsed, expected);
    }

    /// 一元算术 op 嵌套解析：
    ///   ROOT(1ch) = Sqrt( Abs( Sub(InputX, Constant(5.0)) ) )
    #[test]
    fn test_arithmetic_unary_ops() {
        use blob::Node;
        let root = Node::root(vec![Node::sqrt(Node::abs(Node::sub(
            Node::input_x(),
            Node::constant(5.0),
        )))]);
        let (parsed, _, _) = deserialize_full(&blob::blob(root).build()).unwrap();

        let expected = AstNode::Root(vec![AstNode::Sqrt(Box::new(AstNode::Abs(Box::new(
            AstNode::Sub(Box::new(AstNode::InputX), Box::new(AstNode::Constant(5.0))),
        ))))]);
        assert_eq!(parsed, expected);
    }

    // ── 子阶段 2.3-3 地形结构 / 形状选择测试 ───────────────────────────────

    /// EndIslands + RangeChoice(6 子节点嵌套) 组合
    #[test]
    fn test_shape_selection_tags() {
        use blob::Node;
        let root = Node::root(vec![
            Node::end_islands(5),
            Node::range_choice(
                Node::sub(Node::input_y(), Node::constant(64.0)),
                Node::constant(0.0),
                Node::constant(384.0),
                Node::mul(Node::input_z(), Node::constant(0.5)),
                Node::constant(-1.0),
                Node::constant(1.0),
            ),
        ]);
        let (parsed, _, _) = deserialize_full(&blob::blob(root).build()).unwrap();

        let expected = AstNode::Root(vec![
            AstNode::EndIslands { perm_index: 5 },
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
        assert_eq!(parsed, expected);
    }

    // ── 子阶段 2.3-4 缓存 / 表面 / 含水层测试 ─────────────────────────────

    /// CacheFlat/Cache2D/CacheInterpolated 三层嵌套 + FindTopSurface i32 解析
    #[test]
    fn test_cache_and_surface_tags() {
        use blob::Node;
        let root = Node::root(vec![
            Node::cache_interp(Node::cache_2d(Node::cache_flat(Node::input_x()))),
            Node::find_top_surface(
                Node::input_y(),
                Node::constant(0.0),
                -64,
                320,
            ),
        ]);
        let (parsed, _, _) = deserialize_full(&blob::blob(root).build()).unwrap();

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
        assert_eq!(parsed, expected);
    }

    // ── Envelope canary ─────────────────────────────────────────────────
    //
    // Unlike the structural tests above, this one hand-writes every byte.
    // It is the independent check that `mod blob`'s envelope (header,
    // SHAPE_CONFIG, BIOME_SOURCE ordering) matches the protocol as actually
    // written by NoiseRouterBlobifier.java — if the builder and parser
    // drift together, this test is what still fails.

    #[test]
    fn envelope_wire_canary() {
        use blob::Node;
        // Same tree as roundtrip_simple_root, built by the builder.
        let built = blob::blob(Node::root(vec![
            Node::constant(3.14),
            Node::input_x(),
        ]))
        .shape(-64, 384, 4, 4)
        .biome(0x0000_0007_5BCD_1500)
        .build();

        // Independent, hand-written reference for that same tree.
        let manual: &[u8] = &[
            0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0xFE, 0x02, 0x00,
            0x01, 0xC3, 0xF5, 0x48, 0x40,
            0x02,
            0xC0, 0xFF, 0xFF, 0xFF,
            0x80, 0x01, 0x00, 0x00,
            0x04, 0x00, 0x00, 0x00,
            0x04, 0x00, 0x00, 0x00,
            0xF0, 0x00, 0x15, 0xCD, 0x5B, 0x07, 0x00, 0x00, 0x00,
        ];
        assert_eq!(built, manual, "mod blob no longer matches the wire format");
    }

    /// Aquifer reservations the executor used to hardcode as 1200.
    #[test]
    fn aquifer_cells_follow_the_shape() {
        let vanilla = ShapeConfig { min_y: -64, height: 384, cell_width: 4, cell_height: 4 };
        // (320 / 12) - (-64 / 12) + 3 = 26 + 6 + 3 = 35 layers, 3×3 per layer.
        assert_eq!(vanilla.aquifer_layers(), 35);
        assert_eq!(vanilla.aquifer_cells_per_chunk(), 315);

        let tectonic = ShapeConfig { min_y: -64, height: 832, cell_width: 4, cell_height: 8 };
        // 768 / 12 = 64, minus floor(-64/12) = -6, plus 3 => 73.
        assert_eq!(tectonic.aquifer_cells_per_chunk(), 657);

        // Tall enough that the old fixed 1200 would have overrun the buffer.
        let tall = ShapeConfig { min_y: -64, height: 2048, cell_width: 4, cell_height: 4 };
        assert!(tall.aquifer_cells_per_chunk() > 1200);
    }
}
