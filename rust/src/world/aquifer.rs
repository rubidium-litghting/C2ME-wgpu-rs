//! §22 Step B: per-chunk aquifer table — parse Java's compact layout and
//! regenerate blockPositions[] from the dumped RNG seed. Verified bit-exact
//! in `tests/aquifer_positions.rs` (315/315).

type P = (u32, u32);

fn add(a: P, b: P) -> P { let l = a.0 as u64 + b.0 as u64; (l as u32, a.1.wrapping_add(b.1).wrapping_add((l >> 32) as u32)) }
fn xor(a: P, b: P) -> P { (a.0 ^ b.0, a.1 ^ b.1) }
fn shl(a: P, s: u32) -> P { if s == 0 { return a; } if s >= 64 { return (0,0); } if s >= 32 { (0, a.0 << (s-32)) } else { (a.0 << s, (a.1 << s) | (a.0 >> (32-s))) } }
fn shr(a: P, s: u32) -> P { if s == 0 { return a; } if s >= 64 { return (0,0); } if s >= 32 { (a.1 >> (s-32), 0) } else { ((a.0 >> s) | (a.1 << (32-s)), a.1 >> s) } }
fn rotl(a: P, s: u32) -> P { let s = s % 64; if s == 0 { return a; } let (l, r) = (shl(a, s), shr(a, 64-s)); (l.0 | r.0, l.1 | r.1) }
fn mul32(a: u32, b: u32) -> P {
    let (al, ah) = (a & 0xFFFF, a >> 16); let (bl, bh) = (b & 0xFFFF, b >> 16);
    let (ll, lh, hl, hh) = (al*bl, al*bh, ah*bl, ah*bh);
    let cross = (ll >> 16) + (lh & 0xFFFF) + (hl & 0xFFFF);
    ((ll & 0xFFFF) | (cross << 16), hh + (lh >> 16) + (hl >> 16) + (cross >> 16))
}
fn mul(a: P, b: P) -> P { let p00 = mul32(a.0, b.0); let p01 = mul32(a.0, b.1); let p10 = mul32(a.1, b.0); add(p00, (0, add(p01, p10).0)) }
fn sgn(v: i64) -> P { ((v as u64) as u32, (v as u64 >> 32) as u32) }
fn hash3(x: i32, y: i32, z: i32) -> P {
    let a = sgn((x.wrapping_mul(3129871)) as i64);
    let bz = sgn((z as i64).wrapping_mul(116129781));
    let l = xor(xor(a, bz), sgn(y as i64));
    let l2 = add(mul(mul(l, l), sgn(42317861)), mul(l, sgn(11)));
    let r = shr(l2, 16);
    if l2.1 & 0x8000_0000 != 0 { (r.0, r.1 | 0xFFFF_0000) } else { r }
}
struct X { lo: P, hi: P }
impl X {
    fn next(&mut self) -> P { let (s0, s1) = (self.lo, self.hi); let res = add(rotl(add(s0, s1), 17), s0);
        let s1n = xor(s1, s0); self.lo = xor(xor(rotl(s0, 49), s1n), shl(s1n, 21)); self.hi = rotl(s1n, 28); res }
    fn next_int(&mut self, bound: i32) -> i32 {
        let b = bound as u64; let mut r = self.next().0 as u64; let mut m = r.wrapping_mul(b); let mut mlo = m & 0xFFFF_FFFF;
        if mlo < b { let thr = ((-bound) as u32 % bound as u32) as u64; while mlo < thr { r = self.next().0 as u64; m = r.wrapping_mul(b); mlo = m & 0xFFFF_FFFF; } }
        (m >> 32) as i32
    }
}

/// Parsed per-chunk aquifer table. `n` cells, laid out cell-major with
/// strides (sizeX, sizeY, sizeZ) — the same ordering MC uses internally.
pub struct AquiferTable {
    pub n: usize,
    pub size_x: i32,
    pub size_y: i32,
    pub size_z: i32,
    pub start_x: i32,
    pub start_y: i32,
    pub start_z: i32,
    pub seed_lo: i64,
    pub seed_hi: i64,
    pub lvl_y: Vec<i32>,
    pub lvl_state: Vec<u32>,
}

impl AquiferTable {
    /// Parse the compact Java layout: magic "AUQA" (LE bytes 41 55 51 41),
    /// n:i32, seedLo:i64, seedHi:i64, then n * (lvl_y:i32, state:u32).
    /// `size_x/y/z` and `start_*` must be supplied by the caller (they are
    /// shape + chunk dependent, not in the serialized header).
    pub fn parse(bytes: &[u8], size_x: i32, size_y: i32, size_z: i32,
                 start_x: i32, start_y: i32, start_z: i32) -> Option<Self> {
        if bytes.len() < 24 || &bytes[0..4] != b"AUQA" { return None; }
        let n = i32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
        let seed_lo = i64::from_le_bytes(bytes[8..16].try_into().ok()?);
        let seed_hi = i64::from_le_bytes(bytes[16..24].try_into().ok()?);
        if bytes.len() < 24 + n * 8 { return None; }
        let mut lvl_y = Vec::with_capacity(n);
        let mut lvl_state = Vec::with_capacity(n);
        for i in 0..n {
            let o = 24 + i * 8;
            lvl_y.push(i32::from_le_bytes(bytes[o..o+4].try_into().ok()?));
            lvl_state.push(u32::from_le_bytes(bytes[o+4..o+8].try_into().ok()?));
        }
        Some(AquiferTable { n, size_x, size_y, size_z,
            start_x, start_y, start_z, seed_lo, seed_hi, lvl_y, lvl_state })
    }

    /// Pack into the GPU layout: `cell_p` = per cell [px, py, pz, lvl_y],
    /// `cell_s` = per cell [state]. Appends to the caller's vectors so
    /// multiple chunks can share one buffer.
    pub fn pack_for_gpu(&self, out_p: &mut Vec<[i32; 4]>, out_s: &mut Vec<u32>) {
        for i in 0..self.n {
            let ci = (i as i32) % self.size_x;
            let ck = (i as i32 / self.size_x) % self.size_z;
            let cj = i as i32 / (self.size_x * self.size_z);
            let cx = self.start_x + ci;
            let cy = self.start_y + cj;
            let cz = self.start_z + ck;
            let h = hash3(cx, cy, cz);
            let h64 = (((h.1 as u64) << 32) | h.0 as u64) as i64;
            let m = h64 ^ self.seed_lo;
            let mut r = X { lo: (m as u32, (m as u64 >> 32) as u32),
                            hi: (self.seed_hi as u32, (self.seed_hi as u64 >> 32) as u32) };
            let px = cx * 16 + r.next_int(10);
            let py = cy * 12 + r.next_int(9);
            let pz = cz * 16 + r.next_int(10);
            out_p.push([px, py, pz, self.lvl_y[i]]);
            out_s.push(self.lvl_state[i]);
        }
    }
}
