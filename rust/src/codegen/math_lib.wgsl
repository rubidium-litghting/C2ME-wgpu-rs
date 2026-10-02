// ============================================================================
//  C2ME wgpu math_lib.wgsl  —  fp32 precision
//  License: MIT
//  铁律 (from 计划.txt §5):
//    ① 局部 offset 坐标系 — noise_position_with_origin()
//    ② Kahan 累加器      — struct KahanSum + kahan_add(), for len ≥ 3
//    ③ 整数边界          — i32 粗判 + f32 细插 for aquifer/find_top/y_clamped
// ============================================================================

// ----------------------------------------------------------------------------
// § Basic primitives (aligned to Java semantics exactly)
// ----------------------------------------------------------------------------

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return fma(t, b - a, a);
}

fn clamp01(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

// Trilinear interpolation across the 8 corners of a unit cube.
fn trilinear_lerp(
    c000: f32, c100: f32, c010: f32, c110: f32,
    c001: f32, c101: f32, c011: f32, c111: f32,
    tx: f32, ty: f32, tz: f32,
) -> f32 {
    let x00 = lerp(c000, c100, tx);
    let x10 = lerp(c010, c110, tx);
    let x01 = lerp(c001, c101, tx);
    let x11 = lerp(c011, c111, tx);
    let y0 = lerp(x00, x10, ty);
    let y1 = lerp(x01, x11, ty);
    return lerp(y0, y1, tz);
}

fn floorDiv(x: f32, y: f32) -> f32 {
    return floor(x / y);
}
fn floorMod(x: f32, y: f32) -> f32 {
    let q = floorDiv(x, y);
    return x - q * y;
}

// ----------------------------------------------------------------------------
// 铁律②: KahanSum
// ----------------------------------------------------------------------------

struct KahanSum {
    sum: f32,
    corr: f32,
};

fn kahan_zero() -> KahanSum {
    return KahanSum(0.0f, 0.0f);
}

fn kahan_add(s: ptr<function, KahanSum>, v: f32) {
    var t: KahanSum = *s;
    let y = v - t.corr;
    let nsum = t.sum + y;
    t.corr = (nsum - t.sum) - y;
    t.sum = nsum;
    *s = t;
}

fn kahan_result(s: KahanSum) -> f32 { return s.sum; }

struct DD { hi: f32, lo: f32 }

// ----------------------------------------------------------------------------
// 铁律①: Local-offset coordinate helper.
// ----------------------------------------------------------------------------

fn noise_position_with_origin(local: f32, origin_times_freq: f32, local_times_freq: f32) -> f32 {
    return origin_times_freq + local_times_freq;
}

// ----------------------------------------------------------------------------
// § Perlin noise — 全局绑定 & 梯度表
// ----------------------------------------------------------------------------

// 全局置换表（单表，用于简单场景）

// 多表支持（用于精确复现 OctavePerlin）

const GRAD3: array<vec3<f32>, 16> = array<vec3<f32>, 16>(
    vec3( 1.0,  1.0,  0.0), vec3(-1.0,  1.0,  0.0),
    vec3( 1.0, -1.0,  0.0), vec3(-1.0, -1.0,  0.0),
    vec3( 1.0,  0.0,  1.0), vec3(-1.0,  0.0,  1.0),
    vec3( 1.0,  0.0, -1.0), vec3(-1.0,  0.0, -1.0),
    vec3( 0.0,  1.0,  1.0), vec3( 0.0, -1.0,  1.0),
    vec3( 0.0,  1.0, -1.0), vec3( 0.0, -1.0, -1.0),
    vec3( 1.0,  1.0,  0.0), vec3( 0.0, -1.0,  1.0),
    vec3(-1.0,  1.0,  0.0), vec3( 0.0, -1.0, -1.0)
);

fn scale_tunnels(x: f32) -> f32 {
    if (x < -0.5) { return 0.75; }
    if (x < 0.0)  { return 1.0; }
    if (x < 0.5)  { return 1.5; }
    return 2.0;
}

fn scale_caves(x: f32) -> f32 {
    if (x < -0.75) { return 0.5; }
    if (x < -0.5)  { return 0.75; }
    if (x < 0.5)   { return 1.0; }
    if (x < 0.75)  { return 2.0; }
    return 3.0;
}

fn perlin_ysmear(perm_offset: i32, x: f32, y: f32, z: f32, yfreq: f32, ysmear: f32) -> f32 {
    let e = perm_offset / 256;
    let px = x + perm_origin[e * 3];
    let py = y + perm_origin[e * 3 + 1];
    let pz = z + perm_origin[e * 3 + 2];
    let ix = i32(floor(px));
    let iy = i32(floor(py));
    let iz = i32(floor(pz));
    let fx = px - f32(ix);
    let fy = py - f32(iy);
    let fz = pz - f32(iz);
    var o: f32 = 0.0;
    if (yfreq != 0.0) {
        let m = select(fy, ysmear, (ysmear >= 0.0) && (ysmear < fy));
        o = floor(m / yfreq + 1.0e-7) * yfreq;
    }
    let dy = fy - o;
    let p = perm_offset;
    let xi = ix & 255;
    let yi = iy & 255;
    let zi = iz & 255;
    let a  = i32(perm[p + xi]);
    let b  = i32(perm[p + (((xi + 1)) & 255)]);
    let aa = i32(perm[p + (((a + yi)) & 255)]);
    let ab = i32(perm[p + (((a + yi + 1)) & 255)]);
    let ba = i32(perm[p + (((b + yi)) & 255)]);
    let bb = i32(perm[p + (((b + yi + 1)) & 255)]);
    let aaa = i32(perm[p + (((aa + zi)) & 255)]);
    let aab = i32(perm[p + (((aa + zi + 1)) & 255)]);
    let aba = i32(perm[p + (((ab + zi)) & 255)]);
    let abb = i32(perm[p + (((ab + zi + 1)) & 255)]);
    let baa = i32(perm[p + (((ba + zi)) & 255)]);
    let bab = i32(perm[p + (((ba + zi + 1)) & 255)]);
    let bba = i32(perm[p + (((bb + zi)) & 255)]);
    let bbb = i32(perm[p + (((bb + zi + 1)) & 255)]);
    let g000 = GRAD3[aaa & 15];
    let g100 = GRAD3[baa & 15];
    let g010 = GRAD3[aba & 15];
    let g110 = GRAD3[bba & 15];
    let g001 = GRAD3[aab & 15];
    let g101 = GRAD3[bab & 15];
    let g011 = GRAD3[abb & 15];
    let g111 = GRAD3[bbb & 15];
    let n000 = dot(g000, vec3(fx,       dy,       fz));
    let n100 = dot(g100, vec3(fx - 1.0, dy,       fz));
    let n010 = dot(g010, vec3(fx,       dy - 1.0, fz));
    let n110 = dot(g110, vec3(fx - 1.0, dy - 1.0, fz));
    let n001 = dot(g001, vec3(fx,       dy,       fz - 1.0));
    let n101 = dot(g101, vec3(fx - 1.0, dy,       fz - 1.0));
    let n011 = dot(g011, vec3(fx,       dy - 1.0, fz - 1.0));
    let n111 = dot(g111, vec3(fx - 1.0, dy - 1.0, fz - 1.0));
    let u = fade(fx);
    let v = fade(fy);
    let w = fade(fz);
    return trilinear_lerp(n000, n100, n010, n110, n001, n101, n011, n111, u, v, w);
}

fn interp_noise_sampler(
    lb: i32, lc: i32, ub: i32, uc: i32, ib: i32, ic: i32,
    bx: f32, by: f32, bz: f32,
    xz_scale: f32, y_scale: f32, xz_factor: f32, y_factor: f32, smear: f32
) -> f32 {
    let sxz = 684.412 * xz_scale;
    let sy = 684.412 * y_scale;
    let d = bx * sxz;
    let e = by * sy;
    let f = bz * sxz;
    let g = d / xz_factor;
    let h = e / y_factor;
    let l = f / xz_factor;
    let mm = sy * smear / y_factor;
    let smear_y_raw = sy * smear;

    var q = 0.0;
    var r = 1.0;
    for (var i = 0; i < ic; i++) {
        q += perlin_ysmear(ib + i * 256,
                           maintain_precision(g * r), maintain_precision(h * r),
                           maintain_precision(l * r), mm * r, h * r) / r;
        r *= 0.5;
    }
    let interp = (q / 10.0 + 1.0) / 2.0;
    let noLower = interp >= 1.0;
    let noUpper = interp <= 0.0;

    var lo = 0.0;
    var up = 0.0;
    r = 1.0;
    // lower/upper use the RAW (un-divided) coords and a different
    // arg4/arg5 scaling than the interp stack (per MC bytecode).
    for (var i = 0; i < lc; i++) {
        if (!noLower) {
            lo += perlin_ysmear(lb + i * 256,
                                maintain_precision(d * r), maintain_precision(e * r),
                                maintain_precision(f * r), smear_y_raw * r, e * r) / r;
        }
        if (!noUpper) {
            up += perlin_ysmear(ub + i * 256,
                                maintain_precision(d * r), maintain_precision(e * r),
                                maintain_precision(f * r), smear_y_raw * r, e * r) / r;
        }
        r *= 0.5;
    }
    let t = clamp(interp, 0.0, 1.0);
    return (lo / 512.0 + t * (up / 512.0 - lo / 512.0)) / 128.0;
}

fn squeeze(x: f32) -> f32 {
    // MC: x = clamp(x, -1, 1); return x/2 - x^3/24
    let c = clamp(x, -1.0, 1.0);
    return c * 0.5 - c * c * c / 24.0;
}

fn clamped_map(v: f32, from_min: f32, from_max: f32, to_min: f32, to_max: f32) -> f32 {
    let t = clamp((v - from_min) / (from_max - from_min), 0.0, 1.0);
    return to_min + t * (to_max - to_min);
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn maintain_precision(value: f32) -> f32 {
    let divisor = 33554432.0; // 2^25
    return value - floor(value / divisor + 0.5) * divisor;
}

// ---------- 核心 Perlin (使用指定置换表) ----------
fn perlin3d_with_perm(perm_offset: i32, x: f32, y: f32, z: f32) -> f32 {
    let ix = i32(floor(x));
    let iy = i32(floor(y));
    let iz = i32(floor(z));
    let fx = x - f32(ix);
    let fy = y - f32(iy);
    let fz = z - f32(iz);

    let xi = ix & 255;
    let yi = iy & 255;
    let zi = iz & 255;
    // Read this sampler's 256-entry table starting at perm_offset.
    let p = perm_offset;
    let a  = i32(perm[p + xi]);
    let b  = i32(perm[p + (((xi + 1)) & 255)]);
    let aa = i32(perm[p + (((a + yi)) & 255)]);
    let ab = i32(perm[p + (((a + yi + 1)) & 255)]);
    let ba = i32(perm[p + (((b + yi)) & 255)]);
    let bb = i32(perm[p + (((b + yi + 1)) & 255)]);
    let aaa = i32(perm[p + (((aa + zi)) & 255)]);
    let aab = i32(perm[p + (((aa + zi + 1)) & 255)]);
    let aba = i32(perm[p + (((ab + zi)) & 255)]);
    let abb = i32(perm[p + (((ab + zi + 1)) & 255)]);
    let baa = i32(perm[p + (((ba + zi)) & 255)]);
    let bab = i32(perm[p + (((ba + zi + 1)) & 255)]);
    let bba = i32(perm[p + (((bb + zi)) & 255)]);
    let bbb = i32(perm[p + (((bb + zi + 1)) & 255)]);

    let g000 = GRAD3[aaa & 15];
    let g100 = GRAD3[baa & 15];
    let g010 = GRAD3[aba & 15];
    let g110 = GRAD3[bba & 15];
    let g001 = GRAD3[aab & 15];
    let g101 = GRAD3[bab & 15];
    let g011 = GRAD3[abb & 15];
    let g111 = GRAD3[bbb & 15];

    let n000 = dot(g000, vec3(fx,      fy,      fz));
    let n100 = dot(g100, vec3(fx - 1.0, fy,      fz));
    let n010 = dot(g010, vec3(fx,      fy - 1.0, fz));
    let n110 = dot(g110, vec3(fx - 1.0, fy - 1.0, fz));
    let n001 = dot(g001, vec3(fx,      fy,      fz - 1.0));
    let n101 = dot(g101, vec3(fx - 1.0, fy,      fz - 1.0));
    let n011 = dot(g011, vec3(fx,      fy - 1.0, fz - 1.0));
    let n111 = dot(g111, vec3(fx - 1.0, fy - 1.0, fz - 1.0));

    let u = fade(fx);
    let v = fade(fy);
    let w = fade(fz);
    return trilinear_lerp(
        n000, n100, n010, n110,
        n001, n101, n011, n111,
        u, v, w
    );
}

// 带 origin 偏移的版本
fn perlin3d_with_perm_and_origin(perm_offset: i32, origin: vec3<f32>, x: f32, y: f32, z: f32) -> f32 {
    let px = maintain_precision(x + origin.x);
    let py = maintain_precision(y + origin.y);
    let pz = maintain_precision(z + origin.z);
    return perlin3d_with_perm(perm_offset, px, py, pz);
}

// ---------- 公开 API (使用全局 perm) ----------
fn noise_perlin3d(perm_offset: i32, x: f32, y: f32, z: f32) -> f32 {
    let px = maintain_precision(x);
    let py = maintain_precision(y);
    let pz = maintain_precision(z);
    return perlin3d_with_perm(perm_offset, px, py, pz);
}

// MC-exact: maintainPrecision(coord*freq) THEN + origin, per octave.
// `perm_offset` selects the table; its entry index (= perm_offset/256)
// selects the (x,y,z) origin triple in binding 3.
fn noise_perlin3d_origin(perm_offset: i32, x: f32, y: f32, z: f32) -> f32 {
    let e = perm_offset / 256;
    let px = maintain_precision(x)   + perm_origin[e * 3];
    let py = maintain_precision(y)   + perm_origin[e * 3 + 1];
    let pz = maintain_precision(z)   + perm_origin[e * 3 + 2];
    return perlin3d_with_perm(perm_offset, px, py, pz);
}

fn noise_perlin2d(perm_offset: i32, x: f32, y: f32) -> f32 {
    return noise_perlin3d(perm_offset, x, y, 0.0);
}

// ----------------------------------------------------------------------------
// § Simplex noise (标准实现，非原版)
// ----------------------------------------------------------------------------

fn hash3_perm(a: i32, b: i32, c: i32) -> u32 {
    let idx = i32(perm[(i32(perm[(i32(perm[(a & 255)]) + b) & 255]) + c) & 255]);
    return u32(idx & 15);
}

fn simplex3d(x: f32, y: f32, z: f32) -> f32 {
    let F3 = 1.0 / 3.0;
    let G3 = 1.0 / 6.0;
    let s = (x + y + z) * F3;
    let i = floor(x + s);
    let j = floor(y + s);
    let k = floor(z + s);
    let t = (i + j + k) * G3;
    let X0 = i - t;
    let Y0 = j - t;
    let Z0 = k - t;
    let x0 = x - X0;
    let y0 = y - Y0;
    let z0 = z - Z0;

    var i1: f32 = 0.0; var j1: f32 = 0.0; var k1: f32 = 0.0;
    var i2: f32 = 0.0; var j2: f32 = 0.0; var k2: f32 = 0.0;
    if (x0 >= y0) {
        if (y0 >= z0) {
            i1 = 1.0; j1 = 0.0; k1 = 0.0;
            i2 = 1.0; j2 = 1.0; k2 = 0.0;
        } else if (x0 >= z0) {
            i1 = 1.0; j1 = 0.0; k1 = 0.0;
            i2 = 1.0; j2 = 0.0; k2 = 1.0;
        } else {
            i1 = 0.0; j1 = 0.0; k1 = 1.0;
            i2 = 1.0; j2 = 0.0; k2 = 1.0;
        }
    } else {
        if (y0 < z0) {
            i1 = 0.0; j1 = 0.0; k1 = 1.0;
            i2 = 0.0; j2 = 1.0; k2 = 1.0;
        } else if (x0 < z0) {
            i1 = 0.0; j1 = 1.0; k1 = 0.0;
            i2 = 0.0; j2 = 1.0; k2 = 1.0;
        } else {
            i1 = 0.0; j1 = 1.0; k1 = 0.0;
            i2 = 1.0; j2 = 1.0; k2 = 0.0;
        }
    }

    let x1 = x0 - i1 + G3;
    let y1 = y0 - j1 + G3;
    let z1 = z0 - k1 + G3;
    let x2 = x0 - i2 + 2.0*G3;
    let y2 = y0 - j2 + 2.0*G3;
    let z2 = z0 - k2 + 2.0*G3;
    let x3 = x0 - 1.0 + 3.0*G3;
    let y3 = y0 - 1.0 + 3.0*G3;
    let z3 = z0 - 1.0 + 3.0*G3;

    let xi = i32(floor(i)) & 255;
    let yi = i32(floor(j)) & 255;
    let zi = i32(floor(k)) & 255;

    let n0 = dot(GRAD3[hash3_perm(xi, yi, zi)], vec3(x0, y0, z0));
    let n1 = dot(GRAD3[hash3_perm(xi + i32(i1), yi + i32(j1), zi + i32(k1))], vec3(x1, y1, z1));
    let n2 = dot(GRAD3[hash3_perm(xi + i32(i2), yi + i32(j2), zi + i32(k2))], vec3(x2, y2, z2));
    let n3 = dot(GRAD3[hash3_perm(xi + 1, yi + 1, zi + 1)], vec3(x3, y3, z3));

    let r0 = (x0*x0 + y0*y0 + z0*z0);
    let r1 = (x1*x1 + y1*y1 + z1*z1);
    let r2 = (x2*x2 + y2*y2 + z2*z2);
    let r3 = (x3*x3 + y3*y3 + z3*z3);
    var val = 0.0;
    if (r0 < 1.0) { val += (1.0 - r0)*(1.0 - r0)*(1.0 - r0) * n0; }
    if (r1 < 1.0) { val += (1.0 - r1)*(1.0 - r1)*(1.0 - r1) * n1; }
    if (r2 < 1.0) { val += (1.0 - r2)*(1.0 - r2)*(1.0 - r2) * n2; }
    if (r3 < 1.0) { val += (1.0 - r3)*(1.0 - r3)*(1.0 - r3) * n3; }

    return val * 16.0;
}

fn noise_simplex3d(seed: i32, x: f32, y: f32, z: f32) -> f32 {
    let px = maintain_precision(x);
    let py = maintain_precision(y);
    let pz = maintain_precision(z);
    return simplex3d(px, py, pz);
}

fn noise_simplex2d(seed: i32, x: f32, y: f32) -> f32 {
    return noise_simplex3d(seed, x, y, 0.0);
}

// ----------------------------------------------------------------------------
// § FBM (Fractional Brownian Motion) using global perm
// ----------------------------------------------------------------------------

fn fbm_perlin3d(
    seed: i32,
    x: f32,
    y: f32,
    z: f32,
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    initial_amplitude: f32,
    initial_frequency: f32
) -> f32 {
    var acc: KahanSum = kahan_zero();
    var amp: f32 = initial_amplitude;
    var freq: f32 = initial_frequency;
    var i: u32 = 0u;
    loop {
        if (i >= octaves) { break; }
        let noise_val = noise_perlin3d(seed + i32(i), x * freq, y * freq, z * freq);
        kahan_add(&acc, amp * noise_val);
        amp *= gain;
        freq *= lacunarity;
        i += 1u;
    }
    return kahan_result(acc);
}

fn fbm_perlin2d(
    seed: i32,
    x: f32,
    y: f32,
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    initial_amplitude: f32,
    initial_frequency: f32
) -> f32 {
    return fbm_perlin3d(seed, x, y, 0.0, octaves, lacunarity, gain, initial_amplitude, initial_frequency);
}

// 归一化版本
fn fbm_perlin3d_normalized(
    seed: i32,
    x: f32,
    y: f32,
    z: f32,
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    initial_amplitude: f32,
    initial_frequency: f32
) -> f32 {
    var acc: KahanSum = kahan_zero();
    var total_amp: f32 = 0.0;
    var amp: f32 = initial_amplitude;
    var freq: f32 = initial_frequency;
    var i: u32 = 0u;
    loop {
        if (i >= octaves) { break; }
        let noise_val = noise_perlin3d(seed + i32(i), x * freq, y * freq, z * freq);
        kahan_add(&acc, amp * noise_val);
        total_amp += amp;
        amp *= gain;
        freq *= lacunarity;
        i += 1u;
    }
    if (total_amp > 0.0) {
        return kahan_result(acc) / total_amp;
    } else {
        return 0.0;
    }
}

// ----------------------------------------------------------------------------
// § 精确复现原版 OctavePerlinNoiseSampler (使用独立表)
// ----------------------------------------------------------------------------


fn sample_octave_perlin(x: f32, y: f32, z: f32, octaves: u32, lacunarity: f32, persistence: f32) -> f32 { return 0.0; }

// ----------------------------------------------------------------------------
// § InterpolatedNoiseSampler (直接调用 perlin2d 或 simplex2d)
// ----------------------------------------------------------------------------

fn interpolated_noise_sampler_perlin(
    seed: i32,
    x: f32, y: f32, z: f32
) -> f32 {
    let ix = floor(x);
    let iz = floor(z);
    let fx = x - ix;
    let fz = z - iz;
    let u = fade(fx);
    let w = fade(fz);

    let n00 = noise_perlin2d(seed, ix, iz);
    let n10 = noise_perlin2d(seed, ix + 1.0, iz);
    let n01 = noise_perlin2d(seed, ix, iz + 1.0);
    let n11 = noise_perlin2d(seed, ix + 1.0, iz + 1.0);

    var acc: KahanSum = kahan_zero();
    let w00 = (1.0 - u) * (1.0 - w); kahan_add(&acc, n00 * w00);
    let w10 = (u)       * (1.0 - w); kahan_add(&acc, n10 * w10);
    let w01 = (1.0 - u) * (w);       kahan_add(&acc, n01 * w01);
    let w11 = (u)       * (w);       kahan_add(&acc, n11 * w11);
    return kahan_result(acc);
}

fn interpolated_noise_sampler_simplex(
    seed: i32,
    x: f32, y: f32, z: f32
) -> f32 {
    let ix = floor(x);
    let iz = floor(z);
    let fx = x - ix;
    let fz = z - iz;
    let u = fade(fx);
    let w = fade(fz);

    let n00 = noise_simplex2d(seed, ix, iz);
    let n10 = noise_simplex2d(seed, ix + 1.0, iz);
    let n01 = noise_simplex2d(seed, ix, iz + 1.0);
    let n11 = noise_simplex2d(seed, ix + 1.0, iz + 1.0);

    var acc: KahanSum = kahan_zero();
    let w00 = (1.0 - u) * (1.0 - w); kahan_add(&acc, n00 * w00);
    let w10 = (u)       * (1.0 - w); kahan_add(&acc, n10 * w10);
    let w01 = (1.0 - u) * (w);       kahan_add(&acc, n01 * w01);
    let w11 = (u)       * (w);       kahan_add(&acc, n11 * w11);
    return kahan_result(acc);
}

// ---------- 核心批处理 Perlin (精确匹配原版私有 sample) ----------
// 参数：
//   perm_table  : 置换表
//   ix,iy,iz    : 整数晶格坐标（已经取整）
//   local_x, local_y, local_z : 各方向的分数偏移（未经过 fade）
//   fade_local_x : 用于 y 方向 fade 的原始 y 分数（即 h，不是 local_y）
// 返回噪声值
fn perlin3d_with_perm_batched(
    perm_offset: i32,
    ix: i32, iy: i32, iz: i32,
    local_x: f32, local_y: f32, local_z: f32,
    fade_local_x: f32
) -> f32 {
    // 1. 计算晶格索引（包装）
    let xi = ix & 255;
    let yi = iy & 255;
    let zi = iz & 255;
    let p = perm_offset;

    let a  = i32(perm[p + xi]);
    let b  = i32(perm[p + (((xi + 1)) & 255)]);
    let aa = i32(perm[p + (((a + yi)) & 255)]);
    let ab = i32(perm[p + (((a + yi + 1)) & 255)]);
    let ba = i32(perm[p + (((b + yi)) & 255)]);
    let bb = i32(perm[p + (((b + yi + 1)) & 255)]);
    let aaa = i32(perm[p + (((aa + zi)) & 255)]);
    let aab = i32(perm[p + (((aa + zi + 1)) & 255)]);
    let aba = i32(perm[p + (((ab + zi)) & 255)]);
    let abb = i32(perm[p + (((ab + zi + 1)) & 255)]);
    let baa = i32(perm[p + (((ba + zi)) & 255)]);
    let bab = i32(perm[p + (((ba + zi + 1)) & 255)]);
    let bba = i32(perm[p + (((bb + zi)) & 255)]);
    let bbb = i32(perm[p + (((bb + zi + 1)) & 255)]);

    // 2. 梯度向量
    let g000 = GRAD3[aaa & 15];
    let g100 = GRAD3[baa & 15];
    let g010 = GRAD3[aba & 15];
    let g110 = GRAD3[bba & 15];
    let g001 = GRAD3[aab & 15];
    let g101 = GRAD3[bab & 15];
    let g011 = GRAD3[abb & 15];
    let g111 = GRAD3[bbb & 15];

    // 3. 点积（使用 local_x/y/z 及 -1 偏移）
    let n000 = dot(g000, vec3(local_x,      local_y,      local_z));
    let n100 = dot(g100, vec3(local_x - 1.0, local_y,      local_z));
    let n010 = dot(g010, vec3(local_x,      local_y - 1.0, local_z));
    let n110 = dot(g110, vec3(local_x - 1.0, local_y - 1.0, local_z));
    let n001 = dot(g001, vec3(local_x,      local_y,      local_z - 1.0));
    let n101 = dot(g101, vec3(local_x - 1.0, local_y,      local_z - 1.0));
    let n011 = dot(g011, vec3(local_x,      local_y - 1.0, local_z - 1.0));
    let n111 = dot(g111, vec3(local_x - 1.0, local_y - 1.0, local_z - 1.0));

    // 4. 计算 fade 因子
    //    x 方向使用 local_x 的 fade
    //    y 方向使用 fade_local_x（即原始 h）的 fade
    //    z 方向使用 local_z 的 fade
    let u = fade(local_x);          // x 方向
    let v = fade(fade_local_x);     // y 方向 (注意这里！)
    let w = fade(local_z);          // z 方向

    // 5. 三线性插值
    return trilinear_lerp(
        n000, n100, n010, n110,
        n001, n101, n011, n111,
        u, v, w
    );
}

// ---------- 带 yScale / yMax 的公开采样 ----------
// 完全匹配原版 PerlinNoiseSampler.sample(x,y,z,yScale,yMax)
fn perlin3d_with_y_scale(
    perm_offset: i32,
    origin: vec3<f32>,
    x: f32, y: f32, z: f32,
    y_scale: f32,
    y_max: f32
) -> f32 {
    // 应用 origin 偏移
    let d = x + origin.x;
    let e = y + origin.y;
    let f = z + origin.z;

    // 取整得到晶格坐标和分数
    let i = floor(d);
    let j = floor(e);
    let k = floor(f);
    let g = d - i;  // localX
    let h = e - j;  // localY (原始 y 分数)
    let l = f - k;  // localZ

    // 计算分层偏移 o (针对 y 方向)
    var o: f32 = 0.0;
    if (y_scale != 0.0) {
        let m = select(h, y_max, y_max >= 0.0 && y_max < h);
        o = floor(m / y_scale + 1.0e-7f) * y_scale;
    }

    // 调用批处理核心
    return perlin3d_with_perm_batched(
        perm_offset,
        i32(i), i32(j), i32(k),
        g,           // local_x
        h - o,       // local_y (减去分层偏移)
        l,           // local_z
        h            // fade_local_x = 原始 y 分数
    );
}

// ----------------------------------------------------------------------------
// 铁律③: Integer boundary helpers
// ----------------------------------------------------------------------------

fn aquifer_below_fluid(
    fluid_level_i32: i32, pos_y_i32: i32,
    fluid_level_frac: f32, pos_y_frac: f32,
) -> bool {
    let dy_i32 = fluid_level_i32 - pos_y_i32;
    if (dy_i32 > 0) { return true; }
    if (dy_i32 < 0) { return false; }
    return fluid_level_frac >= pos_y_frac;
}

fn find_top_should_stop(search_y_i32: i32, target_top_i32: i32) -> bool {
    let d = search_y_i32 - target_top_i32;
    if (d > 0)  { return true; }
    if (d < 0)  { return false; }
    return true;
}

fn y_clamped_gradient(
    y_val_i32: i32, y_min_i32: i32, y_max_i32: i32,
    y_val_frac: f32, slope: f32,
) -> f32 {
    if (y_val_i32 <= y_min_i32) { return 0.0; }
    if (y_val_i32 >= y_max_i32) { return 1.0; }
    let t = (f32(y_val_i32 - y_min_i32) + y_val_frac) * slope;
    return clamp01(t);
}

// ----------------------------------------------------------------------------
// § Biome 4-corner interpolation
// ----------------------------------------------------------------------------

fn biome_access_4corner(
    c00: f32, c10: f32, c01: f32, c11: f32,
    tx: f32, tz: f32,
) -> f32 {
    var acc: KahanSum = kahan_zero();
    let w00 = (1.0 - tx) * (1.0 - tz); kahan_add(&acc, c00 * w00);
    let w10 = (tx)       * (1.0 - tz); kahan_add(&acc, c10 * w10);
    let w01 = (1.0 - tx) * (tz);       kahan_add(&acc, c01 * w01);
    let w11 = (tx)       * (tz);       kahan_add(&acc, c11 * w11);
    return kahan_result(acc);
}

// 完整含水层计算（模拟 MC 1.18 的 Aquifer 逻辑）
fn aquifer_compute(
    noise_level: f32,
    pos_xi: i32, pos_yi: i32, pos_zi: i32,
    pos_xf: f32, pos_yf: f32, pos_zf: f32,
) -> f32 {
    // 基础含水层判定：根据噪声值决定是否含水
    // 这里使用简化的阈值，实际 MC 会结合流体水位和地形
    let threshold = 0.2;
    let water_level = 63; // 海平面高度（Y=63）
    
    // 如果噪声低于阈值且方块在海平面以下，返回 1.0（表示水）
    if (noise_level < threshold && pos_yi < water_level) {
        return 1.0;
    } else {
        return 0.0;
    }
}

// 末地岛地形生成（基于坐标的简单公式）
fn end_islands_calc(_cx: i32, _cz: i32, _in: f32) -> f32 {
    // 使用坐标计算一个简单的岛形轮廓
    let cx_f = f32(_cx);
    let cz_f = f32(_cz);
    // 中心岛半径约为 100 个区块
    let radius = 100.0;
    let dist = sqrt(cx_f*cx_f + cz_f*cz_f);
    // 岛中心高度最高，向外递减
    let height = max(0.0, 1.0 - dist / radius);
    return height * 64.0; // 最大高度 64 格
}

// ----------------------------------------------------------------------------
// § 统一入口（示例）
// ----------------------------------------------------------------------------

fn sample_world_noise(
    seed: i32,
    block_x: i32, block_y: i32, block_z: i32,
    noise_type: u32
) -> f32 {
    let fx = f32(block_x);
    let fy = f32(block_y);
    let fz = f32(block_z);
    switch noise_type {
        case 0u: { return noise_perlin3d(seed, fx, fy, fz); }
        case 1u: { return noise_simplex3d(seed, fx, fy, fz); }
        case 2u: {
            return fbm_perlin3d(seed, fx, fy, fz, 8u, 2.0, 0.5, 1.0, 1.0);
        }
        case 3u: {
            return sample_octave_perlin(fx, fy, fz, 8u, 2.0, 0.5);
        }
        default: { return 0.0; }
    }
}

// ============================================================================
// § EndIslands (End dimension) — MC SimplexNoiseSampler 2D port
// ============================================================================
// simplex_perm (binding 5) holds, per entry, 512 ints; entry i starts at
// i*512. Only the first 256 are non-zero (MC shuffles perm[0..256)).
// map(a) = perm[a & 255]; grad uses GRADIENTS[h%12] dot (x, y, 0).

const SX_SKEW_2D:   f32 = 0.3660254037844386;  // 0.5*(sqrt(3)-1)
const SX_UNSKEW_2D: f32 = 0.21132486540518713; // (3-sqrt(3))/6

fn sx_map(perm_base: i32, a: i32) -> i32 {
    return i32(simplex_perm[perm_base + (a & 255)]);
}

fn sx_grad(h: i32, x: f32, y: f32) -> f32 {
    let t = 0.5 - x * x - y * y;
    if (t < 0.0) { return 0.0; }
    let t4 = t * t * t * t;
    var dp: f32 = 0.0;
    // GRADIENTS[h] dot (x, y, 0) for h in 0..11
    if (h == 0) { dp = x + y; }
    else if (h == 1) { dp = -x + y; }
    else if (h == 2) { dp = x - y; }
    else if (h == 3) { dp = -x - y; }
    else if (h == 4 || h == 6) { dp = x; }
    else if (h == 5 || h == 7) { dp = -x; }
    else if (h == 8 || h == 10) { dp = y; }
    else { dp = -y; }
    return t4 * dp;
}

fn simplex2d(perm_base: i32, x: f32, y: f32) -> f32 {
    let s = (x + y) * SX_SKEW_2D;
    let i = floor(x + s);
    let j = floor(y + s);
    let t = (i + j) * SX_UNSKEW_2D;
    let x0 = x - (i - t);
    let y0 = y - (j - t);
    var i1: f32 = 0.0;
    var j1: f32 = 0.0;
    if (x0 > y0) { i1 = 1.0; j1 = 0.0; } else { i1 = 0.0; j1 = 1.0; }
    let x1 = x0 - i1 + SX_UNSKEW_2D;
    let y1 = y0 - j1 + SX_UNSKEW_2D;
    let x2 = x0 - 1.0 + 2.0 * SX_UNSKEW_2D;
    let y2 = y0 - 1.0 + 2.0 * SX_UNSKEW_2D;
    let ii = i32(i) & 255;
    let jj = i32(j) & 255;
    let gi0 = sx_map(perm_base, ii + sx_map(perm_base, jj)) % 12;
    let gi1 = sx_map(perm_base, ii + i32(i1) + sx_map(perm_base, jj + i32(j1))) % 12;
    let gi2 = sx_map(perm_base, ii + 1 + sx_map(perm_base, jj + 1)) % 12;
    return 70.0 * (sx_grad(gi0, x0, y0) + sx_grad(gi1, x1, y1) + sx_grad(gi2, x2, y2));
}

// MC DensityFunctionTypes$EndIslands.sample: blockX/8, blockZ/8 then the
// static 25x25 island scan. Returns the raw sample() value; the caller
// applies (v - 8) / 128.
fn end_islands(perm_index: i32, block_x: i32, block_z: i32) -> f32 {
    let base = perm_index * 512;
    let x = block_x / 8;
    let z = block_z / 8;
    let x2 = x / 2;
    let z2 = z / 2;
    let xr = x % 2;
    let zr = z % 2;
    var d = clamp(100.0 - sqrt(f32(x * x + z * z)) * 8.0, -100.0, 80.0);
    for (var dx: i32 = -12; dx <= 12; dx = dx + 1) {
        for (var dz: i32 = -12; dz <= 12; dz = dz + 1) {
            let lx = x2 + dx;
            let lz = z2 + dz;
            if (lx * lx + lz * lz <= 4096) { continue; }
            if (simplex2d(base, f32(lx), f32(lz)) < -0.9) {
                let f = f32(abs(lx)) * 3439.0 + f32(abs(lz)) * 147.0;
                let ff = (f % 13.0) + 9.0;
                let gx = f32(xr - 2 * dx);
                let gz = f32(zr - 2 * dz);
                let g = clamp(100.0 - sqrt(gx * gx + gz * gz) * ff, -100.0, 80.0);
                d = max(d, g);
            }
        }
    }
    return d;
}


