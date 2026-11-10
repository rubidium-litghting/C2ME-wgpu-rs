// §8.4 Beardifier. Layout of beard_pj (per chunk, fixed stride 16 u32):
//   [0] piece_count, [1] junction_count,
//   [2..2+7*pc] pieces (7 i32: minX,minY,minZ,maxX,maxY,maxZ,adapt,delta),
//   junctions follow (3 i32 each).
const BEARD_STRIDE: u32 = 4096u;
const BEARD_W_OFF: u32 = 13824u; // 24*24*24, beard_w section length
const ADAPT_BURY: i32 = 1; const ADAPT_THIN: i32 = 2;
const ADAPT_BOX: i32 = 3; const ADAPT_ENCAP: i32 = 4;

fn beard_gsw(x: i32, y: i32, z: i32, yy: i32) -> f32 {
  let i = x + 12; let j = y + 12; let k = z + 12;
  if (i < 0 || i >= 24 || j < 0 || j >= 24 || k < 0 || k >= 24) { return 0.0; }
  let d = f32(yy) + 0.5;
  let e = f32(x * x) + d * d + f32(z * z);
  // MC uses a double fastInverseSqrt; f32 with exact inverseSqrt is within
  // ~1e-7 relative, negligible for structure shaping.
  let f = -d * inverseSqrt(e / 2.0) / 2.0;
  return f * bitcast<f32>(beard_buf[u32(k * 576 + i * 24 + j)]);
}
fn beard_mag(x: f32, y: f32, z: f32) -> f32 {
  let d = sqrt(x * x + y * y + z * z);
  return 1.0 - clamp(d / 6.0, 0.0, 1.0);
}
fn beard_sample(base: u32, bx: i32, by: i32, bz: i32) -> f32 {
  let pc = i32(beard_buf[BEARD_W_OFF + base]);
  let jc = i32(beard_buf[BEARD_W_OFF + base + 1u]);
  var d = 0.0;
  var o = base + 2u;
  for (var n: i32 = 0; n < pc; n = n + 1) {
    let minx = i32(beard_buf[BEARD_W_OFF + o]); let miny = i32(beard_buf[BEARD_W_OFF + o+1u]);
    let minz = i32(beard_buf[BEARD_W_OFF + o+2u]); let maxx = i32(beard_buf[BEARD_W_OFF + o+3u]);
    let maxy = i32(beard_buf[BEARD_W_OFF + o+4u]); let maxz = i32(beard_buf[BEARD_W_OFF + o+5u]);
    let ad = i32(beard_buf[BEARD_W_OFF + o+6u]); let gld = i32(beard_buf[BEARD_W_OFF + o+7u]);
    o = o + 8u;
    let m = max(0, max(minx - bx, bx - maxx));
    let nn = max(0, max(minz - bz, bz - maxz));
    let oo = miny + gld;
    let p = by - oo;
    var q = 0; if (ad == ADAPT_BURY || ad == ADAPT_THIN) { q = p; }
    else if (ad == ADAPT_BOX) { q = max(0, max(oo - by, by - maxy)); }
    else if (ad == ADAPT_ENCAP) { q = max(0, max(miny - by, by - maxy)); }
    if (ad == ADAPT_BURY) { d = d + beard_mag(f32(m), f32(q) * 0.5, f32(nn)); }
    else if (ad == ADAPT_THIN || ad == ADAPT_BOX) { d = d + beard_gsw(m, q, nn, p) * 0.8; }
    else if (ad == ADAPT_ENCAP) { d = d + beard_mag(f32(m) * 0.5, f32(q) * 0.5, f32(nn) * 0.5) * 0.8; }
  }
  for (var n: i32 = 0; n < jc; n = n + 1) {
    let sx = i32(beard_buf[BEARD_W_OFF + o]); let sy = i32(beard_buf[BEARD_W_OFF + o+1u]); let sz = i32(beard_buf[BEARD_W_OFF + o+2u]);
    o = o + 3u;
    let r = bx - sx; let l = by - sy; let mm = bz - sz;
    d = d + beard_gsw(r, l, mm, l) * 0.4;
  }
  return d;
}

fn aq_fdiv(a: i32, b: i32) -> i32 {
  let q = a / b; let r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) { return q - 1; }
  return q;
}

fn max_dist(i: i32, a: i32) -> f32 {
  let d = 25.0;
  return 1.0 - abs(f32(a - i)) / 25.0;
}

fn calc_density(
    wy: i32,
    lvl_y1: i32, st1: u32,
    lvl_y2: i32, st2: u32,
    barrier_val: f32,
) -> f32 {
    // MC FluidLevel.getBlockState(y): blocks above the fluid level are AIR.
    let actual_st1 = select(0u, st1, wy < lvl_y1);
    let actual_st2 = select(0u, st2, wy < lvl_y2);
    let is_lava1 = actual_st1 == 3u; let is_water1 = actual_st1 == 2u;
    let is_lava2 = actual_st2 == 3u; let is_water2 = actual_st2 == 2u;
    if (is_lava1 && is_water2) || (is_water1 && is_lava2) { return 2.0; }
    let j = abs(lvl_y1 - lvl_y2);
    if (j == 0) { return 0.0; }
    let d = 0.5 * f32(lvl_y1 + lvl_y2);
    let e = f32(wy) + 0.5 - d;
    let f = f32(j) / 2.0;
    let o = f - abs(e);
    var q: f32 = 0.0;
    if (e > 0.0) {
        let p = o;
        if (p > 0.0) { q = p / 1.5; } else { q = p / 2.5; }
    } else {
        let p = 3.0 + o;
        if (p > 0.0) { q = p / 3.0; } else { q = p / 10.0; }
    }
    var r = 0.0;
    if (q >= -2.0 && q <= 2.0) {
        r = barrier_val;
    }
    return 2.0 * (r + q);
}

// ── Ore veins (§3.5) ─────────────────────────────────────────────────
// Mirrors net.minecraft.world.gen.OreVeinSampler. Runs on blocks the aquifer
// already classified as solid (density > 0), optionally replacing them with a
// vein ore / raw-ore / stone variant.

fn vein_clamped_map(v: f32, from_lo: f32, from_hi: f32, to_lo: f32, to_hi: f32) -> f32 {
  let t = clamp((v - from_lo) / (from_hi - from_lo), 0.0, 1.0);
  return t * (to_hi - to_lo) + to_lo;
}

// Trilinearly interpolate channel at `base` for the current block.
fn tri_at(base: u32, ix: i32, iy: i32, iz: i32, ccx: i32, s: i32,
          tx: f32, ty: f32, tz: f32) -> f32 {
  let c000 = out_buf[base + u32(ix + iz * ccx + iy * s)];
  let c100 = out_buf[base + u32(ix + 1 + iz * ccx + iy * s)];
  let c010 = out_buf[base + u32(ix + iz * ccx + (iy + 1) * s)];
  let c110 = out_buf[base + u32(ix + 1 + iz * ccx + (iy + 1) * s)];
  let c001 = out_buf[base + u32(ix + (iz + 1) * ccx + iy * s)];
  let c101 = out_buf[base + u32(ix + 1 + (iz + 1) * ccx + iy * s)];
  let c011 = out_buf[base + u32(ix + (iz + 1) * ccx + (iy + 1) * s)];
  let c111 = out_buf[base + u32(ix + 1 + (iz + 1) * ccx + (iy + 1) * s)];
  let x00 = c000 + tx * (c100 - c000);
  let x10 = c010 + tx * (c110 - c010);
  let x01 = c001 + tx * (c101 - c001);
  let x11 = c011 + tx * (c111 - c011);
  let y0 = x00 + ty * (x10 - x00);
  let y1 = x01 + ty * (x11 - x01);
  return y0 + tz * (y1 - y0);
}

// 0 = not a vein block (leave as stone); otherwise a block-state code 4..9.
fn vein_state(wx: i32, wy: i32, wz: i32, vt: f32, vr: f32, vg: f32) -> u32 {
  let is_copper = vt > 0.0;
  let min_y: i32 = select(-60, 0, is_copper);
  let max_y: i32 = select(-8, 50, is_copper);
  let e = abs(vt);
  let j = max_y - wy;
  let k = wy - min_y;
  if (k < 0 || j < 0) { return 0u; }
  let l = min(j, k);
  let f = vein_clamped_map(f32(l), 0.0, 20.0, -0.2, 0.0);
  if (e + f < 0.4) { return 0u; }
  // randomDeriver.split(wx, wy, wz): Xoroshiro(hashCode ^ seedLo, seedHi).
  let h = java_hash3(wx, wy, wz);
  let m = vec2<u32>(h.x ^ params._pad[5], h.y ^ params._pad[6]);
  var st: vec4<u32> = vec4<u32>(m.x, m.y, params._pad[7], params._pad[8]);
  if (xoro_next_float(&st) > 0.7) { return 0u; }
  if (vr >= 0.0) { return 0u; }
  let g = vein_clamped_map(e, 0.4, 0.6, 0.1, 0.3);
  if (xoro_next_float(&st) < g && vg > -0.3) {
    if (xoro_next_float(&st) < 0.02) {
      return select(9u, 7u, is_copper);   // raw iron : raw copper
    }
    return select(8u, 6u, is_copper);     // deepslate iron : copper ore
  }
  return select(5u, 4u, is_copper);       // tuff : granite
}

@compute @workgroup_size(4, 4, 4)
fn main_block(@builtin(global_invocation_id) gid: vec3<u32>) {
  let num_chunks: i32 = params.batch_count_x;
  let cx_i32: i32 = i32(gid.x);
  let cy_i32: i32 = i32(gid.y);
  let cz_i32: i32 = i32(gid.z);
  let chunk_index: i32 = cz_i32 / 16;
  let bz: i32 = cz_i32 % 16;
  let height_i: i32 = i32(params.height_f);
  if (cx_i32 >= 16 || cy_i32 >= height_i || bz >= 16 || chunk_index >= num_chunks) { return; }

  let cell_w: i32 = i32(params.cell_width);
  let cell_h: i32 = i32(params.cell_height);
  let ccx: i32 = 16 / cell_w + 1;
  let ccz: i32 = 16 / cell_w + 1;
  let ccy: i32 = height_i / cell_h + 1;
  let s: i32 = ccx * ccz;
  let cpc: u32 = u32(ccx * ccy * ccz);
  let cbase: u32 = u32(chunk_index) * 144u * cpc;
  let ch11_base: u32 = cbase + 11u * cpc;

  let ix: i32 = cx_i32 / cell_w;
  let iz: i32 = bz / cell_w;
  let iy: i32 = cy_i32 / cell_h;
  let tx: f32 = f32(cx_i32 % cell_w) / f32(cell_w);
  let tz: f32 = f32(bz % cell_w) / f32(cell_w);
  let ty: f32 = f32(cy_i32 % cell_h) / f32(cell_h);

  let c000 = out_buf[ch11_base + u32(ix + iz * ccx + iy * s)];
  let c100 = out_buf[ch11_base + u32(ix + 1 + iz * ccx + iy * s)];
  let c010 = out_buf[ch11_base + u32(ix + iz * ccx + (iy + 1) * s)];
  let c110 = out_buf[ch11_base + u32(ix + 1 + iz * ccx + (iy + 1) * s)];
  let c001 = out_buf[ch11_base + u32(ix + (iz + 1) * ccx + iy * s)];
  let c101 = out_buf[ch11_base + u32(ix + 1 + (iz + 1) * ccx + iy * s)];
  let c011 = out_buf[ch11_base + u32(ix + (iz + 1) * ccx + (iy + 1) * s)];
  let c111 = out_buf[ch11_base + u32(ix + 1 + (iz + 1) * ccx + (iy + 1) * s)];

  let x00 = c000 + tx * (c100 - c000);
  let x10 = c010 + tx * (c110 - c010);
  let x01 = c001 + tx * (c101 - c001);
  let x11 = c011 + tx * (c111 - c011);
  let y0 = x00 + ty * (x10 - x00);
  let y1 = x01 + ty * (x11 - x01);
  let d = y0 + tz * (y1 - y0);

  let per_chunk: u32 = u32(16 * height_i * 16);
  let bidx: u32 = u32(chunk_index) * per_chunk + u32(cx_i32 + bz * 16 + cy_i32 * 256);

  // §8.4 aquifer (12-cell scan + FluidLevel). Global block coords:
  let chunk_cx = positions[chunk_index * 2];
  let chunk_cz = positions[chunk_index * 2 + 1];
  let wx: i32 = chunk_cx * 16 + cx_i32;
  let wz: i32 = chunk_cz * 16 + bz;
  let wy: i32 = i32(params.min_y_f) + cy_i32;

  // §8.4: add the Beardifier contribution (per block, before the d>0 test).
  var dd = d;
  {
    let bbase = u32(chunk_index) * BEARD_STRIDE;
    let bcount = beard_buf[BEARD_W_OFF + bbase];
    if (bcount > 0u) { dd = dd + beard_sample(bbase, wx, wy, wz); }
  }

  // §8.4: per-dimension aquifer mode from params._pad[1..5].
  let has_aq = params._pad[1];
  if (has_aq == 0u) {
    // No aquifers (Nether/End): sea-level semantics, no 12-cell scan.
    if (dd > 0.0) { block_buf[bidx] = 1u; return; }
    let sea = i32(params._pad[2]);
    let dfl = params._pad[3];
    let lava_y = i32(params._pad[4]);
    if (wy < min(lava_y, sea)) { block_buf[bidx] = 3u; return; }
    if (wy < sea) { block_buf[bidx] = dfl; return; }
    block_buf[bidx] = 0u; return;
  }

  // d > 0 → solid. MC then lets OreVeinSampler replace solid blocks.
  if (dd > 0.0) {
    var st: u32 = 1u;
    if (params._pad[9] == 1u) {
      let vt = tri_at(cbase + 12u * cpc, ix, iy, iz, ccx, s, tx, ty, tz);
      let vr = tri_at(cbase + 13u * cpc, ix, iy, iz, ccx, s, tx, ty, tz);
      let vg = tri_at(cbase + 14u * cpc, ix, iy, iz, ccx, s, tx, ty, tz);
      let vs = vein_state(wx, wy, wz, vt, vr, vg);
      if (vs != 0u) { st = vs; }
    }
    block_buf[bidx] = st;
    return;
  }

  // FluidLevelSampler: global lava level check.
  let lava_level = i32(params._pad[4]);
  if (wy < lava_level) { block_buf[bidx] = 3u; return; }

  let min_y_i = i32(params.min_y_f);
  let start_y_i = aq_fdiv(min_y_i, 12) - 1;
  let aq_sy = aq_fdiv(min_y_i + height_i, 12) - aq_fdiv(min_y_i, 12) + 3;
  let aq_base = chunk_index * aq_sy * 9; // AQ_SX(3) * AQ_SY * AQ_SZ(3)
  // aquifer_buf holds aq_p in [0, half) and aq_s in [half, 2*half).
  let aq_half: u32 = params._pad[10];

  let l = aq_fdiv(wx - 5, 16);
  let m = aq_fdiv(wy + 1, 12);
  let n = aq_fdiv(wz - 5, 16);

  var o: i32 = 2147483647; var p: i32 = 2147483647; var q: i32 = 2147483647;
  var ri: i32 = -1; var si: i32 = -1; var ti: i32 = -1;
  for (var u: i32 = 0; u <= 1; u = u + 1) {
    for (var v: i32 = -1; v <= 1; v = v + 1) {
      for (var w: i32 = 0; w <= 1; w = w + 1) {
        let i = (l + u) - chunk_cx + 1;
        let j = (m + v) - start_y_i;
        let k = (n + w) - chunk_cz + 1;
        if (i < 0 || i >= 3 || k < 0 || k >= 3 || j < 0 || j >= aq_sy) { continue; }
        let idx = aq_base + (j * 3 + k) * 3 + i;
        let cp = aquifer_buf[u32(idx)];
        let ad = cp.x - wx; let ae = cp.y - wy; let af = cp.z - wz;
        let ag = ad * ad + ae * ae + af * af;
        if (o >= ag) {
          q = p; p = o; o = ag;
          ti = si; si = ri; ri = idx;
        } else if (p >= ag) {
          q = p; p = ag;
          ti = si; si = idx;
        } else if (q >= ag) {
          q = ag; ti = idx;
        }
      }
    }
  }
  if (ri < 0) { block_buf[bidx] = 1u; return; }
  
  let r_lvl = aquifer_buf[u32(ri)].w;
  let r_st = u32(aquifer_buf[aq_half + u32(ri)].x);
  let r_actual_st = select(0u, r_st, wy < r_lvl);
  
  let dist_op = max_dist(o, p);
  if (dist_op <= 0.0) { block_buf[bidx] = r_actual_st; return; }
  
  if (r_actual_st == 2u && wy - 1 < lava_level) {
      block_buf[bidx] = r_actual_st; return;
  }
  
  var s_lvl: i32 = r_lvl;
  var s_st: u32 = r_st;
  if (si >= 0) {
    s_lvl = aquifer_buf[u32(si)].w;
    s_st = u32(aquifer_buf[aq_half + u32(si)].x);
  }
  var t_lvl: i32 = r_lvl;
  var t_st: u32 = r_st;
  if (ti >= 0) {
    t_lvl = aquifer_buf[u32(ti)].w;
    t_st = u32(aquifer_buf[aq_half + u32(ti)].x);
  }
  
  let e_val = dist_op * calc_density(wy, r_lvl, r_st, s_lvl, s_st, barrier_val);
  if (dd + e_val > 0.0) { block_buf[bidx] = 1u; return; }
  
  let dist_oq = max_dist(o, q);
  if (dist_oq > 0.0) {
      let g_val = dist_op * dist_oq * calc_density(wy, r_lvl, r_st, t_lvl, t_st, barrier_val);
      if (dd + g_val > 0.0) { block_buf[bidx] = 1u; return; }
  }
  
  let dist_pq = max_dist(p, q);
  if (dist_pq > 0.0) {
      let h_val = dist_op * dist_pq * calc_density(wy, s_lvl, s_st, t_lvl, t_st, barrier_val);
      if (dd + h_val > 0.0) { block_buf[bidx] = 1u; return; }
  }
  
  block_buf[bidx] = r_actual_st;
}
