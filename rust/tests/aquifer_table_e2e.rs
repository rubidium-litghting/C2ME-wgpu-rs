//! §8.4 Step B 端到端：读序列化的 aquifer 表（seed + per-cell lvl_y/state），
//! Rust 现算 blockPositions，复现 per-block 方块状态。对照 /tmp/aq_blocks_0_0.txt。
//! 这验证"Java 只传 compact 表 + Rust 现算"的最终架构。

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
fn i32le(b: &[u8], o: usize) -> i32 { i32::from_le_bytes([b[o], b[o+1], b[o+2], b[o+3]]) }

#[test]
fn aquifer_table_e2e() {
    let tp = "/tmp/aquifer_table_0_0.bin";
    let bp = "/tmp/aq_blocks_0_0.txt";
    if !std::path::Path::new(tp).exists() || !std::path::Path::new(bp).exists() { eprintln!("skip"); return; }
    let t = std::fs::read(tp).unwrap();
    assert_eq!(&t[0..4], b"AUQA");
    let n = i32le(&t, 4) as usize;
    let s_lo = i64::from_le_bytes(t[8..16].try_into().unwrap());
    let s_hi = i64::from_le_bytes(t[16..24].try_into().unwrap());
    // per-cell: lvl_y(i32) + state(u32)
    let (start_x, start_y, start_z, size_x, size_z) = (-1i32, -7i32, -1i32, 3i32, 3i32);
    let mut cpx = vec![0i32; n]; let mut cpy = vec![0i32; n]; let mut cpz = vec![0i32; n];
    let mut cly = vec![0i32; n]; let mut cls = vec![0u32; n];
    for i in 0..n {
        let o = 24 + i * 8;
        cly[i] = i32le(&t, o); cls[i] = i32le(&t, o+4) as u32;
        // 现算 blockPositions
        let ci = (i as i32) % size_x; let ck = (i as i32 / size_x) % size_z; let cj = i as i32 / (size_x * size_z);
        let cx = start_x + ci; let cy = start_y + cj; let cz = start_z + ck;
        let h = hash3(cx, cy, cz);
        let h64 = (((h.1 as u64) << 32) | h.0 as u64) as i64;
        let m = h64 ^ s_lo;
        let mut r = X { lo: (m as u32, (m as u64 >> 32) as u32), hi: (s_hi as u32, (s_hi as u64 >> 32) as u32) };
        cpx[i] = cx * 16 + r.next_int(10); cpy[i] = cy * 12 + r.next_int(9); cpz[i] = cz * 16 + r.next_int(10);
    }
    let norm = |s: &str| -> u32 { match s.rsplit(':').next().unwrap_or(s) {
        "air" => 0, "stone" => 1, "water" => 2, "lava" => 3, _ => 0 } };
    let mut total = 0usize; let mut ok = 0usize;
    for line in std::fs::read_to_string(bp).unwrap().lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 5 { continue; }
        let bx: i32 = f[0].parse().unwrap(); let by: i32 = f[1].parse().unwrap();
        let bz: i32 = f[2].parse().unwrap(); let d: f64 = f[3].parse().unwrap();
        let want = norm(f[4]);
        let got = if d > 0.0 { 1u32 } else if by < -54 { 3u32 } else {
            let l = (bx - 5).div_euclid(16); let m = (by + 1).div_euclid(12); let nn = (bz - 5).div_euclid(16);
            let (mut o, mut p) = (i32::MAX, i32::MAX); let mut ri: i32 = -1;
            for u in 0..=1 { for v in -1..=1 { for w in 0..=1 {
                let x = l+u; let y = m+v; let z = nn+w;
                let ii = x - start_x; let kk = z - start_z; let jj = y - start_y;
                if ii < 0 || jj < 0 || kk < 0 || ii >= size_x || kk >= size_z { continue; }
                let idx = (jj * size_z + kk) * size_x + ii;
                if idx as usize >= n { continue; }
                let ad = cpx[idx as usize]-bx; let ae = cpy[idx as usize]-by; let af = cpz[idx as usize]-bz;
                let ag = ad*ad+ae*ae+af*af;
                if o >= ag { ri = idx; p = o; o = ag; } else if p >= ag { p = ag; }
            }}}
            if ri < 0 { 1 } else if by < cly[ri as usize] { cls[ri as usize] } else { 0 }
        };
        total += 1; if got == want { ok += 1; }
    }
    let rate = ok as f64 / total as f64;
    println!("aquifer table e2e: {ok}/{total} = {:.3}% (seedLo={s_lo})", rate*100.0);
    assert!(rate > 0.995, "{:.3}%", rate*100.0);
}
