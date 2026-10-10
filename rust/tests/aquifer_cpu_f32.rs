//! §20 Step B 预研：CPU-f32 复现 MC `AquiferSampler.Impl.apply` 的**控制流**。
//! 读 /tmp/aquifer_noise_*.txt（per-cell 表：blockPositions + 最终 waterLevel）
//! 与 /tmp/aq_blocks_*.txt（per-block 参考：x y z d state）。
//! 只复现 12-cell 扫描 + 查表 + 分类；barrier 噪声（calculateDensity）先跳过。
//! 目标：验证控制流框架可复现（接受边界误差）。

use std::collections::HashMap;

const START_X: i32 = -1; const START_Y: i32 = -7; const START_Z: i32 = -1;
const SIZE_X: i32 = 3;  const SIZE_Z: i32 = 3;

#[derive(Clone)]
struct Cell {
    bx: i32, by: i32, bz: i32,
    lvl_y: i32, lvl_state: String,
}

fn index(x: i32, y: i32, z: i32) -> usize {
    let i = x - START_X; let j = y - START_Y; let k = z - START_Z;
    ((j * SIZE_Z + k) * SIZE_X + i) as usize
}

fn max_distance(a: i32, b: i32) -> f64 { 1.0 - (b - a).abs() as f64 / 25.0 }

/// 复现 apply 的分类（跳过 calculateDensity → 只到 d<=0 分支）。
fn classify(cells: &[Cell], block_x: i32, block_y: i32, block_z: i32,
            density: f64) -> String {
    if density > 0.0 { return "stone".into(); }
    // 入口 fluidLevelSampler：overworld sea=63, -54<63 → y<-54 lava else water
    let fl_lava = block_y < -54;
    if fl_lava { return "lava".into(); }

    let l = (block_x - 5).div_euclid(16);
    let m = (block_y + 1).div_euclid(12);
    let n = (block_z - 5).div_euclid(16);
    let mut o = i32::MAX; let mut p = i32::MAX;
    let mut r_cell: Option<usize> = None; let mut s_cell: Option<usize> = None;
    let mut r_d = 0i32; let mut s_d = 0i32;

    for u in 0..=1 { for v in -1..=1 { for w in 0..=1 {
        let x = l + u; let y = m + v; let z = n + w;
        let i = x - START_X; let k = z - START_Z; let j = y - START_Y;
        if i < 0 || j < 0 || k < 0 || i >= SIZE_X || k >= SIZE_Z { continue; }
        let idx = index(x, y, z);
        if idx >= cells.len() { continue; }
        let c = &cells[idx];
        let ad = c.bx - block_x; let ae = c.by - block_y; let af = c.bz - block_z;
        let ag = ad * ad + ae * ae + af * af;
        if o >= ag { s_cell = r_cell; s_d = r_d; r_cell = Some(idx); r_d = ag;
                     p = o; o = ag; }
        else if p >= ag { s_cell = r_cell; s_d = r_d; r_cell = Some(idx); r_d = ag;
                          p = ag; }
    }}}

    let rc = match r_cell { Some(i) => &cells[i], None => return "stone".into() };
    let d = max_distance(o, p);
    // FluidLevel.getBlockState(y): y < lvl.y -> state, else AIR
    let air = "minecraft:air".to_string();
    let bs: &String = if block_y < rc.lvl_y { &rc.lvl_state } else { &air };
    if d <= 0.0 { return bs.clone(); }
    // 跳过 calculateDensity 分支：保守返回该 fluid level 状态
    let _ = (s_cell, s_d);
    bs.clone()
}

#[test]
fn aquifer_cpu_f32_reproduces_reference() {
    let np = "/tmp/aquifer_noise_0_0.txt";
    let bp = "/tmp/aq_blocks_0_0.txt";
    if !std::path::Path::new(np).exists() || !std::path::Path::new(bp).exists() {
        eprintln!("skipping: dumps absent"); return;
    }
    // 解析 cell 表
    let mut cells: Vec<Option<Cell>> = vec![None; 512];
    for line in std::fs::read_to_string(np).unwrap().lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() < 20 || t[0] == "CHUNK" { continue; }
        let idx: usize = t[0].parse().unwrap();
        let bx: i32 = t[1].parse().unwrap(); let by: i32 = t[2].parse().unwrap();
        let bz: i32 = t[3].parse().unwrap();
        let lvl_y: i32 = t[t.len()-2].parse().unwrap();
        let lvl_state = t[t.len()-1].to_string();
        if idx < cells.len() {
            cells[idx] = Some(Cell { bx, by, bz, lvl_y, lvl_state });
        }
    }
    // 解析 per-block 参考
    let mut total = 0usize; let mut match_ = 0usize;
    let mut mism: Vec<String> = Vec::new();
    for line in std::fs::read_to_string(bp).unwrap().lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() != 5 { continue; }
        let x: i32 = t[0].parse().unwrap(); let y: i32 = t[1].parse().unwrap();
        let z: i32 = t[2].parse().unwrap(); let d: f64 = t[3].parse().unwrap();
        let want = t[4].to_string();
        let flat: Vec<Cell> = cells.iter().map(|c| c.clone().unwrap_or(Cell{bx:0,by:0,bz:0,lvl_y:i32::MIN,lvl_state:"null".into()})).collect();
        let got = classify(&flat, x, y, z, d);
        // 归一化：minecraft:water / water / air 等
        let norm = |s: &str| s.rsplit(':').next().unwrap_or(s).to_string();
        total += 1;
        if norm(&got) == norm(&want) { match_ += 1; }
        else if mism.len() < 20 { mism.push(format!("({x},{y},{z}) d={d:.4} want={want} got={got}")); }
    }
    let rate = match_ as f64 / total as f64;
    println!("aquifer cpu-f32: {match_}/{total} = {:.1}% match", rate * 100.0);
    for m in &mism { println!("  MISMATCH {m}"); }
    assert!(rate > 0.995, "匹配率过低: {:.1}% (barrier 分支未实现属预期)", rate * 100.0);
}
