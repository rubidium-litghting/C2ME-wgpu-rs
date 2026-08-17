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
// Corners are indexed:
//   c000 c100 c010 c110 c001 c101 c011 c111
// where bit 0 = +x, bit 1 = +y, bit 2 = +z.
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

// WARNING: WGSL / GLSL floor is "towards -inf"; Rust / C is "towards zero".
// Java Math.floorDiv and Math.floorMod match WGSL floor — but only for the
// divisor positive case. These wrappers reproduce the exact Java edge case:
//   floorDiv(-1,  4) = -1 in Java (ours)
//   floorDiv(-1, -4) =  0 in Java (handled below, noise never uses negative denom)
fn floorDiv(x: f32, y: f32) -> f32 {
    // TODO: sign-aware correct floorDiv matching StrictMath.floorDiv
    return floor(x / y);
}
fn floorMod(x: f32, y: f32) -> f32 {
    let q = floorDiv(x, y);
    return x - q * y;
}

// ----------------------------------------------------------------------------
// 铁律②: KahanSum — mandatory for 3+ octave Perlin, Spline control-point
//                    weighted sums, Biome 4/8 corner trilinear weights
// ----------------------------------------------------------------------------

struct KahanSum {
    sum: f32,
    corr: f32,
};

fn kahan_zero() -> KahanSum {
    return KahanSum(0.0f, 0.0f);
}

fn kahan_add(s: ptr<function, KahanSum>, v: f32) {
    // WGSL: function-ptr member access only works for reads. To mutate,
    // we must load → modify → store.
    var t: KahanSum = *s;
    let y = v - t.corr;
    let nsum = t.sum + y;
    t.corr = (nsum - t.sum) - y;
    t.sum = nsum;
    *s = t;
}

fn kahan_result(s: KahanSum) -> f32 { return s.sum; }

// Optional extended precision pair — used only for very-long-distance noise
// where offset + f32 still drifts beyond 1e-4. Disabled below x < 100_000 to
// save registers.
struct DD { hi: f32, lo: f32 }

// ----------------------------------------------------------------------------
// 铁律①: Local-offset coordinate helper.
//
//   world_x_i32  = origin_x_i32 + cell_x   (true 32-bit integer math)
//   lx = f32(cell_x)  ←  small, relative to batch origin
//   absolute = origin_as_f32 * freq + lx * freq   ←  the actual sampling coord
//
// The key point: we NEVER do (big_world_coord * freq) in f32 directly —
// we subtract the i32 origin BEFORE f32 cast.
// ----------------------------------------------------------------------------

fn noise_position_with_origin(local: f32, origin_times_freq: f32, local_times_freq: f32) -> f32 {
    // origin_times_freq has the large-magnitude component; local_times_freq is tiny.
    // Adding in this order keeps precision (Kahan-style split sum optional).
    return origin_times_freq + local_times_freq;
}

// ----------------------------------------------------------------------------
// § 2D / 3D Perlin + Simplex + DoublePerlin noise
// ----------------------------------------------------------------------------

// TODO — ~1200 lines total, each validated vs Java FastNoise reference impl
//        with per-function wgpu unit test. For skeleton we just declare the
//        function names so Rust-side codegen can call them.

fn noise_perlin2d(seed: i32, x: f32, y: f32) -> f32 {
    // TODO: reference perlin with fade=6t^5-15t^4+10t^3, grad=[-1,1] lattice
    return 0.0;
}

fn noise_perlin3d(seed: i32, x: f32, y: f32, z: f32) -> f32 {
    return 0.0;
}

fn noise_simplex2d(seed: i32, x: f32, y: f32) -> f32 {
    return 0.0;
}
fn noise_simplex3d(seed: i32, x: f32, y: f32, z: f32) -> f32 {
    return 0.0;
}

// N-octave Perlin with Kahan summation (铁律②).
//
// NOTE on the WGSL address-space model: WGSL function-scope pointers can only
// point to a single element (no runtime-array base-pointer math à la C). The
// octaves/amplitudes/lacunaritys are known at shader-emit time from the
// PerlinSampler AST node, so the Rust `noise_perlin.rs` emitter INLINES this
// loop into the caller scope — reading amplitudes/lacunaritys from a
// compile-time-sized `let amps: array<f32, N> = (...);` binding and indexing
// with `amps[i]`.  That skips the need for a runtime array-pointer signature
// here entirely.  Reference (manual-inlined) form:
//
//   var acc: KahanSum = kahan_zero();
//   var fx = x; var fy = y; var fz = z;
//   for (var i: u32 = 0u; i < 8u; i++) {  // octaves hard-coded, break when done
//       if (i32(i) >= octaves) { break; }
//       let amp = amplitudes[i];
//       let lac = lacunaritys[i];
//       let v = noise_perlin3d(seed + i32(i), fx, fy, fz) * amp;
//       kahan_add(&acc, v);
//       fx *= lac; fy *= lac; fz *= lac;
//   }
//   return kahan_result(acc);

// InterpolatedNoiseSampler — MC's weird cache-interpolated 2D-over-3D noise.
fn noise_interpolated_sampler(_seed: i32, _x: f32, _y: f32, _z: f32) -> f32 { return 0.0; }

// ----------------------------------------------------------------------------
// 铁律③: Integer boundary helpers — i32 coarse branch, f32 fine interpolation
// ----------------------------------------------------------------------------

// Returns true when a point is below or exactly on an aquifer fluid level.
// When exactly on the integer boundary (dy_i32 == 0) we use f32 fractional
// comparison to avoid 1-block water/stone flip-flops at chunk borders.
fn aquifer_below_fluid(
    fluid_level_i32: i32, pos_y_i32: i32,
    fluid_level_frac: f32, pos_y_frac: f32,
) -> bool {
    let dy_i32 = fluid_level_i32 - pos_y_i32;
    if (dy_i32 > 0) { return true; }
    if (dy_i32 < 0) { return false; }
    // exact integer boundary — compare f32 fractions
    return fluid_level_frac >= pos_y_frac;
}

// Same pattern for FindTopSurface upward search early-exit.
fn find_top_should_stop(search_y_i32: i32, target_top_i32: i32) -> bool {
    let d = search_y_i32 - target_top_i32;
    if (d > 0)  { return true; }   // passed the target
    if (d < 0)  { return false; }  // still below
    return true;                   // exactly at — stop and f32 refine
}

// YClampedGradient: hard clip at y_min / y_max with i32 branch first.
fn y_clamped_gradient(
    y_val_i32: i32, y_min_i32: i32, y_max_i32: i32,
    y_val_frac: f32, slope: f32,
) -> f32 {
    if (y_val_i32 <= y_min_i32) { return 0.0; }
    if (y_val_i32 >= y_max_i32) { return 1.0; }
    // in-range: lerp along the slope using fractional y
    let t = (f32(y_val_i32 - y_min_i32) + y_val_frac) * slope;
    return clamp01(t);
}

// ----------------------------------------------------------------------------
// § Biome 4-corner + trilinear access
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

// ----------------------------------------------------------------------------
// § Aquifer full search — placeholder
// ----------------------------------------------------------------------------
fn aquifer_compute(
    noise_level: f32,
    pos_xi: i32, pos_yi: i32, pos_zi: i32,
    pos_xf: f32, pos_yf: f32, pos_zf: f32,
) -> f32 {
    // TODO: 铁律③ i32 boundary check at each fluid height
    return noise_level;
}

// ----------------------------------------------------------------------------
// § End islands, beardifier, spline — placeholder names
// ----------------------------------------------------------------------------
fn end_islands_calc(_cx: i32, _cz: i32, _in: f32) -> f32 { return 0.0; }
fn beardifier_adjust(_struct_idx: i32, _y: i32, _in: f32) -> f32 { return 0.0; }
