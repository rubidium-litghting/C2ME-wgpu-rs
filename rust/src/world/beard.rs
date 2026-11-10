//! §8.4: StructureWeightSampler ("Beardifier") reproduction.
//! Static 24^3 weight table + per-chunk pieces/junctions. Per-block, applied
//! to the density before aquifer classification so structures sit on terrain.

pub const ADAPT_NONE: u8 = 0;
pub const ADAPT_BURY: u8 = 1;
pub const ADAPT_BEARD_THIN: u8 = 2;
pub const ADAPT_BEARD_BOX: u8 = 3;
pub const ADAPT_ENCAPSULATE: u8 = 4;

#[derive(Clone, Copy)]
pub struct Piece {
    pub min_x: i32, pub min_y: i32, pub min_z: i32,
    pub max_x: i32, pub max_y: i32, pub max_z: i32,
    pub adaptation: u8,
    pub ground_level_delta: i32,
}

#[derive(Clone, Copy)]
pub struct Junction {
    pub source_x: i32, pub source_ground_y: i32, pub source_z: i32,
}

/// Quake-style fast inverse sqrt (MC's `MathHelper.fastInverseSqrt`, double).
fn fast_inverse_sqrt(x: f64) -> f64 {
    let half = 0.5 * x;
    let mut l = x.to_bits();
    l = 6910469410427058090u64.wrapping_sub(l >> 1);
    let x = f64::from_bits(l);
    x * (1.5 - half * x * x)
}

fn structure_weight(x: i32, y: f64, z: i32) -> f64 {
    let d = (x as f64) * (x as f64) + y * y + (z as f64) * (z as f64);
    (-d / 16.0).exp()
}
fn calculate_structure_weight(x: i32, y: i32, z: i32) -> f64 {
    structure_weight(x, y as f64 + 0.5, z)
}

/// 24^3 table: `[i*576 + j*24 + k] = calc(j-12, k-12, i-12)`.
pub fn build_weight_table() -> Vec<f32> {
    let mut t = vec![0f32; 13824];
    for i in 0..24i32 { for j in 0..24i32 { for k in 0..24i32 {
        t[(i * 576 + j * 24 + k) as usize] =
            calculate_structure_weight(j - 12, k - 12, i - 12) as f32;
    }}}
    t
}

fn index_in_bounds(i: i32) -> bool { (0..24).contains(&i) }

fn get_structure_weight(table: &[f32], x: i32, y: i32, z: i32, yy: i32) -> f64 {
    let i = x + 12; let j = y + 12; let k = z + 12;
    if index_in_bounds(i) && index_in_bounds(j) && index_in_bounds(k) {
        let d = yy as f64 + 0.5;
        let e = (x * x) as f64 + d * d + (z * z) as f64;
        let f = -d * fast_inverse_sqrt(e / 2.0) / 2.0;
        f * table[(k * 576 + i * 24 + j) as usize] as f64
    } else { 0.0 }
}

fn get_magnitude_weight(x: f64, y: f64, z: f64) -> f64 {
    let d = (x * x + y * y + z * z).sqrt();
    1.0 - (d / 6.0).clamp(0.0, 1.0)
}

pub fn sample(table: &[f32], pieces: &[Piece], junctions: &[Junction],
              bx: i32, by: i32, bz: i32) -> f64 {
    let mut d = 0.0f64;
    for pc in pieces {
        let m = 0.max((pc.min_x - bx).max(bx - pc.max_x));
        let n = 0.max((pc.min_z - bz).max(bz - pc.max_z));
        let o = pc.min_y + pc.ground_level_delta;
        let p = by - o;
        let q: i32 = match pc.adaptation {
            ADAPT_NONE => 0,
            ADAPT_BURY | ADAPT_BEARD_THIN => p,
            ADAPT_BEARD_BOX => 0.max((o - by).max(by - pc.max_y)),
            ADAPT_ENCAPSULATE => 0.max((pc.min_y - by).max(by - pc.max_y)),
            _ => 0,
        };
        d += match pc.adaptation {
            ADAPT_NONE => 0.0,
            ADAPT_BURY => get_magnitude_weight(m as f64, q as f64 / 2.0, n as f64),
            ADAPT_BEARD_THIN | ADAPT_BEARD_BOX =>
                get_structure_weight(table, m, q, n, p) * 0.8,
            ADAPT_ENCAPSULATE =>
                get_magnitude_weight(m as f64 / 2.0, q as f64 / 2.0, n as f64 / 2.0) * 0.8,
            _ => 0.0,
        };
    }
    for jn in junctions {
        let r = bx - jn.source_x;
        let l = by - jn.source_ground_y;
        let mm = bz - jn.source_z;
        d += get_structure_weight(table, r, l, mm, l) * 0.4;
    }
    d
}
