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

//! WorldgenParams memory layout (header of @group(0) @binding(0)) and
//! Perlin permutation table builder (binding 1).
//!
//! — ALL values f32 or i32, @align(4). No f64 anywhere (D05).
//! — Must match `bytemuck::Pod` so we can cast raw bytes safely.
//! — The `batch_origin_*` fields enable 铁律① local-offset coordinates.

use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;

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

// ============================================================================
// Perlin permutation tables (binding 1)
// ============================================================================

/// Number of i32 entries per Perlin permutation table. MC vanilla = 256.
pub const PERM_TABLE_SIZE: usize = 256;

/// Derive a 256-entry permutation table from a 64-bit seed.
///
/// **Phase 4-2 placeholder**: uses a simple LCG (Linear Congruential
/// Generator) + Fisher-Yates shuffle. This is NOT the algorithm MC vanilla
/// uses (which is based on `WorldgenRandom` / `Xoroshiro128PlusPlusRng`
/// with SHA-derived entropy). When the Java-side `NoiseRouterBlobifier`
/// handshake is finalised, this function will be replaced with the
/// exact MC algorithm to ensure Java ↔ GPU numerical equivalence.
///
/// The placeholder is deterministic, uniform, and bijective over [0, 256)
/// for a given seed — sufficient for testing the permutation-buffer upload
/// pipeline and shader indexing logic.
///
/// Returns: `[u32; 256]` where each entry is in `[0, 256)`.
pub fn permutation_table_for_seed(seed: i64) -> [u32; PERM_TABLE_SIZE] {
    // 1. Initialise identity permutation.
    let mut table = [0u32; PERM_TABLE_SIZE];
    for (i, slot) in table.iter_mut().enumerate() {
        *slot = i as u32;
    }

    // 2. Seed a 64-bit LCG (Numerical Recipes constants).
    let mut state: u64 = (seed as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let next = |s: &mut u64| -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s
    };

    // 3. Fisher-Yates shuffle from the end.
    for i in (1..PERM_TABLE_SIZE).rev() {
        let j = (next(&mut state) as usize) % (i + 1);
        table.swap(i, j);
    }

    table
}

/// A collection of permutation tables for all `PerlinSampler` seeds
/// discovered in a `WorldContext`'s AST, packed into a single buffer.
///
/// Layout in the final `Vec<i32>`:
///   ```text
///   [seed_0_table_0..256] [seed_1_table_0..256] ...
///   ```
/// Each `PerlinSampler` looks up its offset via `offset_for(seed)`.
#[derive(Debug, Clone, Default)]
pub struct PermutationTableSet {
    /// Flat packed buffer: concatenation of all permutation tables.
    /// This is what gets uploaded to `@group(0) @binding(1)`.
    pub buffer: Vec<i32>,
    /// `seed → byte offset / 4` (= i32 index) within `buffer`.
    /// Used by `wgsl_gen` to emit `perm_offset_i32` in place of `seed_i32`.
    pub offsets: FxHashMap<i64, i32>,
}

impl PermutationTableSet {
    /// Build a `PermutationTableSet` from an arbitrary collection of seeds.
    ///
    /// Deduplicates: the same seed appearing twice in `seeds` only
    /// contributes one table to the buffer.
    ///
    /// Order: insertion order into the deduplicated `Vec<i64>` determines
    /// offset assignment — first occurrence gets offset 0, second gets 256, etc.
    pub fn from_seeds(seeds: impl IntoIterator<Item = i64>) -> Self {
        let mut set = Self::default();
        for seed in seeds {
            if set.offsets.contains_key(&seed) {
                continue;
            }
            let table = permutation_table_for_seed(seed);
            let offset = set.buffer.len() as i32;
            set.buffer.extend(table.iter().map(|&x| x as i32));
            set.offsets.insert(seed, offset);
        }
        set
    }

    /// Build from REAL permutation tables shipped by the Java side
    /// (PERM_RESOURCE trailer). Entry `i` occupies buffer indices
    /// `[i*256, (i+1)*256)`; `perm_offset` in `NoiseSampler` == `i * 256`.
    pub fn from_entries(entries: &[crate::ast::deserialize::PermEntry]) -> Self {
        let mut set = Self::default();
        for (i, e) in entries.iter().enumerate() {
            let offset = (i * 256) as i32;
            set.buffer.extend(e.perm.iter().map(|&x| x as i32));
            set.offsets.insert(i as i64, offset);
        }
        set
    }

    /// Look up the i32 index (NOT byte offset) where a seed's permutation
    /// table starts in `buffer`. Returns `None` if the seed wasn't in the
    /// set used to build this `PermutationTableSet`.
    pub fn offset_for(&self, seed: i64) -> Option<i32> {
        self.offsets.get(&seed).copied()
    }

    /// Total number of i32 entries in the packed buffer.
    pub fn len_i32(&self) -> usize {
        self.buffer.len()
    }

    /// Byte size of the packed buffer — useful for `wgpu::BufferDescriptor`.
    pub fn len_bytes(&self) -> usize {
        self.buffer.len() * std::mem::size_of::<i32>()
    }
}

// ============================================================================
// Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// Permutation table for a fixed seed is deterministic — same input
    /// always produces the same output.
    #[test]
    fn permutation_table_is_deterministic() {
        let a = permutation_table_for_seed(42);
        let b = permutation_table_for_seed(42);
        assert_eq!(a, b, "same seed must produce identical permutation table");
    }

    /// Different seeds must produce different tables (probabilistic but
    /// astronomically unlikely to collide for 256-element shuffles).
    #[test]
    fn permutation_table_differs_across_seeds() {
        let a = permutation_table_for_seed(1);
        let b = permutation_table_for_seed(2);
        assert_ne!(a, b, "different seeds must produce different tables");
    }

    /// Every permutation table is a bijection over [0, 256) — every value
    /// appears exactly once.
    #[test]
    fn permutation_table_is_bijection() {
        for seed in [0i64, 1, -1, 42, 1_000_000, i64::MAX, i64::MIN] {
            let table = permutation_table_for_seed(seed);
            let mut seen = [false; PERM_TABLE_SIZE];
            for &v in &table {
                let v = v as usize;
                assert!(v < PERM_TABLE_SIZE, "value {v} out of range for seed {seed}");
                assert!(!seen[v], "value {v} appears twice in table for seed {seed}");
                seen[v] = true;
            }
            assert!(seen.iter().all(|&x| x),
                "table for seed {seed} is not a complete permutation");
        }
    }

    /// `PermutationTableSet` deduplicates seeds.
    #[test]
    fn permutation_table_set_dedups_seeds() {
        let set = PermutationTableSet::from_seeds([1, 2, 1, 3, 2]);
        assert_eq!(set.offsets.len(), 3, "expected 3 unique seeds, got {}",
            set.offsets.len());
        assert_eq!(set.buffer.len(), 3 * PERM_TABLE_SIZE);
        assert_eq!(set.len_i32(), 3 * PERM_TABLE_SIZE);
        assert_eq!(set.len_bytes(), 3 * PERM_TABLE_SIZE * 4);
    }

    /// Offset lookup returns the correct i32 index for each seed.
    #[test]
    fn permutation_table_set_offsets_are_correct() {
        let set = PermutationTableSet::from_seeds([10, 20, 30]);
        assert_eq!(set.offset_for(10), Some(0));
        assert_eq!(set.offset_for(20), Some(PERM_TABLE_SIZE as i32));
        assert_eq!(set.offset_for(30), Some(2 * PERM_TABLE_SIZE as i32));
        assert_eq!(set.offset_for(999), None);
    }

    /// Buffer contents at offset N match `permutation_table_for_seed` for
    /// the Nth inserted seed.
    #[test]
    fn permutation_table_set_buffer_matches_per_seed_tables() {
        let seeds = [100i64, 200, 300];
        let set = PermutationTableSet::from_seeds(seeds.iter().copied());
        for &seed in &seeds {
            let offset = set.offset_for(seed).expect("seed missing from set") as usize;
            let expected = permutation_table_for_seed(seed);
            let actual: &[i32] = &set.buffer[offset..offset + PERM_TABLE_SIZE];
            for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
                assert_eq!(e as i32, a,
                    "perm mismatch at seed={seed} idx={i}: expected {e}, got {a}");
            }
        }
    }

    /// Empty seed set → empty buffer.
    #[test]
    fn permutation_table_set_empty() {
        let set = PermutationTableSet::from_seeds(std::iter::empty::<i64>());
        assert!(set.buffer.is_empty());
        assert!(set.offsets.is_empty());
        assert_eq!(set.len_i32(), 0);
        assert_eq!(set.len_bytes(), 0);
    }

    /// Negative seeds are valid and produce distinct tables from their
    /// positive counterparts.
    #[test]
    fn permutation_table_handles_negative_seeds() {
        let pos = permutation_table_for_seed(42);
        let neg = permutation_table_for_seed(-42);
        assert_ne!(pos, neg, "negative seed must produce different table");

        let set = PermutationTableSet::from_seeds([42, -42]);
        assert_eq!(set.offsets.len(), 2);
        assert_eq!(set.buffer.len(), 2 * PERM_TABLE_SIZE);
    }
}
