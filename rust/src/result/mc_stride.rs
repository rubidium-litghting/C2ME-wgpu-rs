// SPDX-License-Identifier: MIT

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
