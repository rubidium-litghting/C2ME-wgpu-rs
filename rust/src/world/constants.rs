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

//! WorldgenParams memory layout (header of @binding(0) buffer).
//!
//! — ALL values f32 or i32, @align(4). No f64 anywhere (D05).
//! — Must match `bytemuck::Pod` so we can cast raw bytes safely.
//! — The `batch_origin_*` fields enable 铁律① local-offset coordinates.

use bytemuck::{Pod, Zeroable};

/// Parameters header that goes at byte offset 0 of @group(0) @binding(0).
/// `#[repr(C)]` order preserved exactly across Java ↔ Rust ↔ WGSL.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WorldgenParams {
    /// Batch world-coordinate origin X (i32). Iron rule ①: subtract this
    /// from per-cell i32 BEFORE converting to f32, to keep magnitudes small.
    pub batch_origin_x: i32,
    pub batch_origin_z: i32,

    /// Cell size in world units (MC vanilla = 4.0 for both horizontal).
    pub cell_width: f32,
    pub cell_height: f32,

    /// World height range as f32 for clamp / gradient math.
    pub min_y_f: f32,
    pub height_f: f32,

    /// Number of chunks in this dispatch batch (width along x/z).
    pub batch_count_x: i32,
    pub batch_count_z: i32,

    /// Reserved padding to 64-byte multiple for alignment paranoia.
    pub _pad: [u32; 8],
}

// Three-way offset assertion (called once at startup from Instance creation).
// Panics on mismatch so we never send a misaligned header to the GPU.
#[allow(dead_code)]
const _ASSERT_OFFSETS: () = {
    use std::mem::offset_of;
    assert!(offset_of!(WorldgenParams, batch_origin_x) == 0);
    assert!(offset_of!(WorldgenParams, batch_origin_z) == 4);
    assert!(offset_of!(WorldgenParams, cell_width) == 8);
    assert!(offset_of!(WorldgenParams, cell_height) == 12);
    assert!(std::mem::size_of::<WorldgenParams>() % 64 == 0);
};
