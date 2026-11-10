//! §8.4: 用 dump 的 aquifer 种子 (seedLo/seedHi) 复现 blockPositions[]。
//! 对照 /tmp/aquifer_0_0.txt 里每 cell 的 (bx,by,bz)。
//! 逻辑：cell(x,y,z) -> hash3 -> split(hash^seedLo, seedHi) ->
//!       nextInt(10/9/10) -> (x*16+dx, y*12+dy, z*16+dz)。

type P = (u32, u32);

fn add(a: P, b: P) -> P { let l = a.0 as u64 + b.0 as u64; (l as u32, a.1.wrapping_add(b.1).wrapping_add((l >> 32) as u32)) }
fn xor(a: P, b: P) -> P { (a.0 ^ b.0, a.1 ^ b.1) }
fn shl(a: P, s: u32) -> P { if s == 0 { return a; } if s >= 64 { return (0,0); } if s >= 32 { (0, a.0 << (s-32)) } else { (a.0 << s, (a.1 << s) | (a.0 >> (32-s))) } }
fn shr(a: P, s: u32) -> P { if s == 0 { return a; } if s >= 64 { return (0,0); } if s >= 32 { (a.1 >> (s-32), 0) } else { ((a.0 >> s) | (a.1 << (32-s)), a.1 >> s) } }
fn rotl(a: P, s: u32) -> P { let s = s % 64; if s == 0 { return a; } let (l, r) = (shl(a, s), shr(a, 64-s)); (l.0 | r.0, l.1 | r.1) }
fn mul32(a: u32, b: u32) -> P {
    let (al, ah) = (a & 0xFFFF, a >> 16);
    let (bl, bh) = (b & 0xFFFF, b >> 16);
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
    fn next(&mut self) -> P {
        let (s0, s1) = (self.lo, self.hi);
        let res = add(rotl(add(s0, s1), 17), s0);
        let s1n = xor(s1, s0);
        self.lo = xor(xor(rotl(s0, 49), s1n), shl(s1n, 21));
        self.hi = rotl(s1n, 28);
        res
    }
    fn next_i32(&mut self) -> u32 { self.next().0 }
    fn next_int(&mut self, bound: i32) -> i32 {
        let b = bound as u64;
        let mut r = self.next_i32() as u64;
        let mut m = r.wrapping_mul(b);
        let mut mlo = m & 0xFFFF_FFFF;
        if mlo < b {
            let thr = ((-bound) as u32 % bound as u32) as u64;
            while mlo < thr { r = self.next_i32() as u64; m = r.wrapping_mul(b); mlo = m & 0xFFFF_FFFF; }
        }
        (m >> 32) as i32
    }
}

#[test]
fn aquifer_block_positions_reproduce() {
    let p = "/tmp/aquifer_0_0.txt";
    if !std::path::Path::new(p).exists() { eprintln!("skip"); return; }
    let txt = std::fs::read_to_string(p).unwrap();
    let mut seed_lo = 0i64; let mut seed_hi = 0i64;
    let (start_x, start_y, start_z, size_x, size_z) = (-1i32, -7i32, -1i32, 3i32, 3i32);
    let mut total = 0usize; let mut ok = 0usize;
    let mut bad_examples: Vec<String> = Vec::new();
    for line in txt.lines() {
        if let Some(rest) = line.strip_prefix("SEED seedLo: ") {
            let parts: Vec<&str> = rest.split(", seedHi: ").collect();
            seed_lo = parts[0].parse().unwrap();
            seed_hi = parts[1].parse().unwrap();
        } else if !line.starts_with("CHUNK") && !line.starts_with("AQ") && !line.is_empty() {
            let t: Vec<&str> = line.split_whitespace().collect();
            if t.len() < 4 { continue; }
            let idx: usize = t[0].parse().unwrap();
            let want_bx: i32 = t[1].parse().unwrap();
            let want_by: i32 = t[2].parse().unwrap();
            let want_bz: i32 = t[3].parse().unwrap();
            // 从 index 反推 cell (x,y,z): index = (j*sizeZ + k)*sizeX + i
            let i = (idx as i32) % size_x;
            let k = ((idx as i32) / size_x) % size_z;
            let j = (idx as i32) / (size_x * size_z);
            let x = i + start_x; let y = j + start_y; let z = k + start_z;
            // 生成
            let h = hash3(x, y, z);
            let h64 = (((h.1 as u64) << 32) | h.0 as u64) as i64;
            let m = h64 ^ seed_lo;
            let mut r = X { lo: ((m as u64) as u32, (m as u64 >> 32) as u32),
                            hi: ((seed_hi as u64) as u32, (seed_hi as u64 >> 32) as u32) };
            let dx = r.next_int(10); let dy = r.next_int(9); let dz = r.next_int(10);
            let got_bx = x * 16 + dx; let got_by = y * 12 + dy; let got_bz = z * 16 + dz;
            total += 1;
            if got_bx == want_bx && got_by == want_by && got_bz == want_bz { ok += 1; }
            else if bad_examples.len() < 5 {
                bad_examples.push(format!("idx={idx} cell=({x},{y},{z}) want=({want_bx},{want_by},{want_bz}) got=({got_bx},{got_by},{got_bz})"));
            }
        }
    }
    println!("aquifer positions: {ok}/{total} match (seedLo={seed_lo} seedHi={seed_hi})");
    for e in &bad_examples { println!("  BAD {e}"); }
    assert!(total > 0, "no cells parsed");
    assert!(ok == total, "{}/{} mismatch", total - ok, total);
}
