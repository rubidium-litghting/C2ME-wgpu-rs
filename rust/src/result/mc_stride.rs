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

//! Write back density results in the EXACT same cell-stride order that
//! Minecraft Java expects. If this stride is off by one f32, terrain will
//! look "tiled" — every 4 or 8 blocks repeating the pattern.
//!
//! MC Vanilla ChunkNoiseSampler cell order for 16 density channels:
//!   For each chunk:
//!     For each channel in NoiseRouter order (barrier…final_density):
//!       linear index k = 0..cells_total:
//!           cellX = k % cellCountX
//!           tmp   = k / cellCountX
//!           cellZ = tmp % cellCountZ
//!           cellY = tmp / cellCountZ
//!           write buf[offset + k] = density_value_f32

use crate::ast::deserialize::ShapeConfig;

/// Compute the MC-native linear index for a (cx, cy, cz) cell position.
#[inline]
pub fn cell_index(cfg: &ShapeConfig, cell_count_x: usize, cell_count_z: usize,
                  cx: usize, cy: usize, cz: usize) -> usize {
    let cells_y = (cfg.height / cfg.cell_height) as usize;
    let _ = cells_y; // silence until used
    cx + cz * cell_count_x + cy * cell_count_x * cell_count_z
}
