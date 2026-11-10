//! §8.4: reproduce the Beardifier's per-block `sample` from dumped pieces.
//! Reads /tmp/beard_X_Z.txt (PIECE/JUNC/BLOCK sections) and compares the
//! Rust `beard::sample` against MC's per-block reference.

use c2me_wgpu::world::beard::{self, Piece, Junction, ADAPT_NONE, ADAPT_BURY,
    ADAPT_BEARD_THIN, ADAPT_BEARD_BOX, ADAPT_ENCAPSULATE};

fn adapt(s: &str) -> u8 {
    match s { "NONE" => ADAPT_NONE, "BURY" => ADAPT_BURY,
        "BEARD_THIN" => ADAPT_BEARD_THIN, "BEARD_BOX" => ADAPT_BEARD_BOX,
        "ENCAPSULATE" => ADAPT_ENCAPSULATE, _ => ADAPT_NONE }
}

#[test]
fn beard_repro() {
    let path = std::env::var("BEARD_DUMP")
        .unwrap_or_else(|_| "/tmp/beard_15_4.txt".into());
    if !std::path::Path::new(&path).exists() { eprintln!("skip: {path}"); return; }
    let txt = std::fs::read_to_string(&path).unwrap();
    let mut lines = txt.lines();
    let mut pieces = Vec::new();
    let mut junctions = Vec::new();
    let mut blocks: Vec<(i32, i32, i32, f64)> = Vec::new();
    while let Some(l) = lines.next() {
        if let Some(rest) = l.strip_prefix("PIECE ") {
            let n: usize = rest.trim().parse().unwrap();
            for _ in 0..n {
                let t: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
                pieces.push(Piece {
                    min_x: t[0].parse().unwrap(), min_y: t[1].parse().unwrap(),
                    min_z: t[2].parse().unwrap(), max_x: t[3].parse().unwrap(),
                    max_y: t[4].parse().unwrap(), max_z: t[5].parse().unwrap(),
                    adaptation: adapt(t[6]), ground_level_delta: t[7].parse().unwrap(),
                });
            }
        } else if let Some(rest) = l.strip_prefix("JUNC ") {
            let n: usize = rest.trim().parse().unwrap();
            for _ in 0..n {
                let t: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
                junctions.push(Junction {
                    source_x: t[0].parse().unwrap(),
                    source_ground_y: t[1].parse().unwrap(),
                    source_z: t[2].parse().unwrap(),
                });
            }
        } else if let Some(rest) = l.strip_prefix("BLOCK ") {
            let n: usize = rest.trim().parse().unwrap();
            blocks.reserve(n);
            for _ in 0..n {
                let t: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
                blocks.push((t[0].parse().unwrap(), t[1].parse().unwrap(),
                    t[2].parse().unwrap(), t[3].parse().unwrap()));
            }
        }
    }
    let table = beard::build_weight_table();
    let mut bad = 0usize; let mut maxd = 0f64;
    for &(bx, gy, bz, want) in &blocks {
        let got = beard::sample(&table, &pieces, &junctions, bx, gy, bz);
        let dd = (got - want).abs();
        if dd > maxd { maxd = dd; }
        if dd > 1e-6 { bad += 1; }
    }
    println!("beard repro: {} pieces, {} junc, {} blocks, bad={} maxd={:.3e}",
        pieces.len(), junctions.len(), blocks.len(), bad, maxd);
    assert_eq!(bad, 0, "{bad}/{} mismatch, maxd={maxd:e}", blocks.len());
}
