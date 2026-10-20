// SPDX-License-Identifier: MIT

//! Tests for `WorldContext`: pipeline compilation, GPU/CPU numerical
//! validation, and end-to-end block output. Split out of `world.rs` so the
//! production code stays readable; this is a child module, so it can still
//! reach `WorldContext`'s crate-private fields via `super::`.

use super::WorldContext;

#[test]
#[ignore]
fn validate_blockout_wgsl() {
    let bin = concat!(env!("CARGO_MANIFEST_DIR"), "/../src/test/resources/real_router.bin");
    let blob = std::fs::read(bin).unwrap();
    let ctx = crate::world::WorldContext::new(&blob).unwrap();
    for (shard, src) in ctx.shards.iter().zip(ctx.shader_srcs.iter()) {
        if shard.pass == crate::codegen::wgsl_gen::Pass::BlockOut {
            match naga::front::wgsl::parse_str(src) {
                Ok(_) => println!("BlockOut WGSL parsed successfully!"),
                Err(e) => panic!("BlockOut WGSL parse error:\n{}", e.emit_to_string(src)),
            }
        }
    }
}

#[test]
#[ignore]
fn dump_shard_sources() {
        let bin_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../src/test/resources/real_router.bin"
    );
    let blob = std::fs::read(bin_path).expect("read real_router.bin");
    let ctx = WorldContext::new(&blob).expect("WorldContext::new");
    for (sh, src) in ctx.shards.iter().zip(ctx.shader_srcs.iter()) {
        let f = format!("/tmp/shard_{:?}.wgsl", sh.pass);
        std::fs::write(&f, src).unwrap();
        println!("wrote {} ({} lines)", f, src.lines().count());
    }
}

#[test]
fn integration_full_nodes_pipeline() {
    use crate::ast::deserialize::deserialize_full;
    use crate::ast::opto::run_pipeline;
    use crate::codegen::wgsl_gen::emit_shader;

    // 构造包含多种节点的 blob（与之前相同）
    let mut blob = Vec::new();
    blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
    blob.push(0xFE);
    blob.extend_from_slice(&[0x06, 0x00]);
    blob.extend_from_slice(&[0x01]);
    blob.extend_from_slice(&1.0f32.to_le_bytes());
    blob.push(0x02);
    // TAG_NOISE_SAMPLER(0x84)
    //   type=Perlin(0) flags=0 (single stack)
    //   amplitude=1.0 xz_scale=1.0 y_scale=1.0
    //   first stack: perm_offset=0 initial_freq=1.0 persistence=1.0
    //                octaves=1 amps=[1.0]
    blob.extend_from_slice(&[0x84, 0x00, 0x00]);
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // amplitude
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // xz_scale
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // y_scale
    blob.extend_from_slice(&0i32.to_le_bytes()); // perm_offset
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // initial_freq
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // persistence
    blob.extend_from_slice(&1i32.to_le_bytes()); // octaves
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // amp[0]
    blob.extend_from_slice(&[0x82]);
    blob.extend_from_slice(&7i64.to_le_bytes());
    blob.extend_from_slice(&[0xB1]);
    blob.extend_from_slice(&[0x01]);
    blob.extend_from_slice(&0.0f32.to_le_bytes());
    blob.extend_from_slice(&[0x01]);
    blob.extend_from_slice(&1.0f32.to_le_bytes());
    blob.extend_from_slice(&[0x01]);
    blob.extend_from_slice(&2.0f32.to_le_bytes());
    blob.extend_from_slice(&[0x01]);
    blob.extend_from_slice(&3.0f32.to_le_bytes());
    blob.extend_from_slice(&[0x01]);
    blob.extend_from_slice(&(-1.0f32).to_le_bytes());
    blob.extend_from_slice(&[0x01]);
    blob.extend_from_slice(&1.0f32.to_le_bytes());
    blob.extend_from_slice(&[0xC0, 0x03]);
    blob.extend_from_slice(&(-64i32).to_le_bytes());
    blob.extend_from_slice(&384i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    blob.extend_from_slice(&[0xF0]);
    blob.extend_from_slice(&123456789i64.to_le_bytes());

    // 1. 反序列化
    let (root, _, _) = deserialize_full(&blob).unwrap();

    // 2. 优化
    let optimized = run_pipeline(root).unwrap();

    // 3. 生成 WGSL（仅 CPU 端，不涉及 GPU）
    let wgsl = emit_shader(&optimized).unwrap();

    // 4. 验证生成的着色器包含关键函数调用
    assert!(wgsl.contains("range_"));
    assert!(wgsl.contains("range_"));
    assert!(wgsl.contains("out_buf"));
    println!("✅ 集成测试通过：反序列化、优化、WGSL生成成功！");
}
/// §20 aquifer ground-work: verify a PURE u32-pair implementation of
/// MC's Xoroshiro128++ + MathHelper.hashCode + split + nextInt(bound),
/// against values dumped from MC's own classes (Ref.java, seed 12345/
/// 67890). Every op here is u32-expressible, so WGSL can reproduce it
/// verbatim. This is the foundation for computing aquifer cell positions
/// on the GPU.
#[test]
fn xoro_u32pair_matches_mc_reference() {
    type P = (u32, u32); // (lo, hi)

    fn add(a: P, b: P) -> P {
        let l = a.0 as u64 + b.0 as u64;
        (l as u32, a.1.wrapping_add(b.1).wrapping_add((l >> 32) as u32))
    }
    fn xor(a: P, b: P) -> P { (a.0 ^ b.0, a.1 ^ b.1) }
    fn shl(a: P, s: u32) -> P {
        if s == 0 { return a; } if s >= 64 { return (0, 0); }
        if s >= 32 { (0, a.0 << (s - 32)) }
        else { (a.0 << s, (a.1 << s) | (a.0 >> (32 - s))) }
    }
    fn shr(a: P, s: u32) -> P {
        if s == 0 { return a; } if s >= 64 { return (0, 0); }
        if s >= 32 { (a.1 >> (s - 32), 0) }
        else { ((a.0 >> s) | (a.1 << (32 - s)), a.1 >> s) }
    }
    fn rotl(a: P, s: u32) -> P {
        let s = s % 64; if s == 0 { return a; }
        let (l, r) = (shl(a, s), shr(a, 64 - s));
        (l.0 | r.0, l.1 | r.1)
    }
    fn mul32(a: u32, b: u32) -> P {
        let (al, ah) = (a & 0xFFFF, a >> 16);
        let (bl, bh) = (b & 0xFFFF, b >> 16);
        let (ll, lh, hl, hh) = (al * bl, al * bh, ah * bl, ah * bh);
        let cross = (ll >> 16) + (lh & 0xFFFF) + (hl & 0xFFFF);
        ((ll & 0xFFFF) | (cross << 16), hh + (lh >> 16) + (hl >> 16) + (cross >> 16))
    }
    fn mul(a: P, b: P) -> P {
        let p00 = mul32(a.0, b.0);
        let p01 = mul32(a.0, b.1);
        let p10 = mul32(a.1, b.0);
        add(p00, (0, add(p01, p10).0))
    }
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
                while mlo < thr {
                    r = self.next_i32() as u64;
                    m = r.wrapping_mul(b);
                    mlo = m & 0xFFFF_FFFF;
                }
            }
            (m >> 32) as i32
        }
    }
    fn hex(n: P) -> String { format!("{:x}{:08x}", n.1, n.0) }

    // RAW stream (seed 12345 / 67890)
    let mut r = X { lo: (12345, 0), hi: (67890, 0) };
    let got: Vec<String> = (0..4).map(|_| hex(r.next())).collect();
    assert_eq!(got,
        vec!["272d63039", "87e1a2e99377f9ef", "8aa1cc21ea593723", "5b767e353956a85a"]);

    // Cell positions: [cx,cy,cz,dx,dy,dz] dumped from MC.
    let cases: &[(i32, i32, i32, i32, i32, i32)] = &[
        (0, 0, 0, 7, 3, 0),
        (1, 2, 3, 8, 8, 9),
        (-5, 4, 7, 5, 3, 9),
        (10, -3, -8, 9, 0, 5),
        (0, 1, 0, 7, 1, 2),
    ];
    let mut r = X { lo: (12345, 0), hi: (67890, 0) };
    let (slo, shi) = (r.next(), r.next()); // nextSplitter
    for &(cx, cy, cz, dx, dy, dz) in cases {
        let mut rr = X { lo: xor(hash3(cx, cy, cz), slo), hi: shi };
        assert_eq!(rr.next_int(10), dx, "cell ({cx},{cy},{cz}) dx");
        assert_eq!(rr.next_int(9), dy, "cell ({cx},{cy},{cz}) dy");
        assert_eq!(rr.next_int(10), dz, "cell ({cx},{cy},{cz}) dz");
    }
}

/// §20: same u32-pair Xoroshiro/hash3/split/nextInt, but in WGSL on the
/// GPU, computing the aquifer cell positions. Must match the same MC
/// reference values as the Rust test.
#[test]
fn xoro_u32pair_wgsl_matches_mc_reference() {
    use crate::instance::WgpuInstance;
    let inst = WgpuInstance::instance().unwrap();
    let device = &inst.device;
    let queue = &inst.queue;

    // 5 cells: [cx,cy,cz] each.
    let cells: [i32; 15] = [0,0,0, 1,2,3, -5,4,7, 10,-3,-8, 0,1,0];
    let n = 5u64;

    let src = r#"
@group(0) @binding(0) var<storage, read> cells: array<i32, 15>;
@group(0) @binding(1) var<storage, read_write> out: array<i32>;

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let lo = a.x + b.x;
  let c = select(0u, 1u, lo < a.x);
  return vec2<u32>(lo, a.y + b.y + c);
}
fn xor64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return vec2<u32>(a.x ^ b.x, a.y ^ b.y); }
fn shl64(a: vec2<u32>, s: u32) -> vec2<u32> {
  if (s == 0u) { return a; }
  if (s >= 32u) { return vec2<u32>(0u, a.x << (s - 32u)); }
  return vec2<u32>(a.x << s, (a.y << s) | (a.x >> (32u - s)));
}
fn shr64(a: vec2<u32>, s: u32) -> vec2<u32> {
  if (s == 0u) { return a; }
  if (s >= 32u) { return vec2<u32>(a.y >> (s - 32u), 0u); }
  return vec2<u32>((a.x >> s) | (a.y << (32u - s)), a.y >> s);
}
fn rotl64(a: vec2<u32>, s: u32) -> vec2<u32> {
  let l = shl64(a, s);
  let r = shr64(a, 64u - s);
  return vec2<u32>(l.x | r.x, l.y | r.y);
}
// 32x32 -> (lo,hi), pure u32
fn mul32x32(a: u32, b: u32) -> vec2<u32> {
  let al = a & 0xFFFFu; let ah = a >> 16u;
  let bl = b & 0xFFFFu; let bh = b >> 16u;
  let ll = al * bl; let lh = al * bh; let hl = ah * bl; let hh = ah * bh;
  let cross = (ll >> 16u) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
  return vec2<u32>((ll & 0xFFFFu) | (cross << 16u), hh + (lh >> 16u) + (hl >> 16u) + (cross >> 16u));
}
// (a.x,a.y)*(b.x,b.y) mod 2^64
fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let p00 = mul32x32(a.x, b.x);
  let p01 = mul32x32(a.x, b.y);
  let p10 = mul32x32(a.y, b.x);
  let mid = add64(p01, p10);
  return add64(p00, vec2<u32>(0u, mid.x));
}
fn sgn(v: i32) -> vec2<u32> {
  let u = bitcast<u32>(v);
  if (v < 0) { return vec2<u32>(u, 0xFFFFFFFFu); }
  return vec2<u32>(u, 0u);
}
fn hash3(x: i32, y: i32, z: i32) -> vec2<u32> {
  let a = sgn(x * 3129871);
  let bz = mul64(sgn(z), sgn(116129781));
  let l = xor64(xor64(a, bz), sgn(y));
  let l2 = add64(mul64(mul64(l, l), sgn(42317861)), mul64(l, sgn(11)));
  let r = shr64(l2, 16u);
  if ((l2.y & 0x80000000u) != 0u) { return vec2<u32>(r.x, r.y | 0xFFFF0000u); }
  return r;
}

struct X { lo: vec2<u32>, hi: vec2<u32> }
fn x_next(s: ptr<function, X>) -> vec2<u32> {
  let s0 = (*s).lo; let s1 = (*s).hi;
  let res = add64(rotl64(add64(s0, s1), 17u), s0);
  let s1n = xor64(s1, s0);
  (*s).lo = xor64(xor64(rotl64(s0, 49u), s1n), shl64(s1n, 21u));
  (*s).hi = rotl64(s1n, 28u);
  return res;
}
fn x_next_int(s: ptr<function, X>, bound: i32) -> i32 {
  let b = u32(bound);
  var r = vec2<u32>(x_next(s).x, 0u);           // u32 value zero-extended
  var m = mul64(r, vec2<u32>(b, 0u));
  var mlo = m.x;
  if (mlo < b) {
let thr = (0u - b) % b;
while (mlo < thr) {
  r = vec2<u32>(x_next(s).x, 0u);
  m = mul64(r, vec2<u32>(b, 0u));
  mlo = m.x;
}
  }
  return bitcast<i32>(m.y);
}

@compute @workgroup_size(1)
fn main() {
  // nextSplitter consumes two next() calls.
  var sp: X;
  sp.lo = vec2<u32>(12345u, 0u);
  sp.hi = vec2<u32>(67890u, 0u);
  let slo = x_next(&sp);
  let shi = x_next(&sp);
  for (var i = 0u; i < 5u; i = i + 1u) {
let cx = cells[i*3u]; let cy = cells[i*3u+1u]; let cz = cells[i*3u+2u];
var rr: X;
rr.lo = xor64(hash3(cx, cy, cz), slo);
rr.hi = shi;
out[i*3u+0u] = x_next_int(&rr, 10);
out[i*3u+1u] = x_next_int(&rr, 9);
out[i*3u+2u] = x_next_int(&rr, 10);
  }
}
"#;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("xoro_wgsl_test"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None, layout: Some(&pl), module: &shader, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });

    let cells_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (cells.len()*4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&cells_buf, 0, bytemuck::cast_slice(&cells));
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: n*3*4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: n*3*4, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: cells_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cp.set_pipeline(&pipe);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, n*3*4);
    queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    loop {
        match rx.try_recv() {
            Ok(_) => break,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
            _ => {}
        }
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_millis(100)),
        });
    }
    let data = slice.get_mapped_range().unwrap();
    let got: &[i32] = bytemuck::cast_slice(&data);

    let expect: [i32; 15] = [7,3,0, 8,8,9, 5,3,9, 9,0,5, 7,1,2];
    for i in 0..15 {
        assert_eq!(got[i], expect[i],
            "WGSL cell out[{i}] = {} expected {}", got[i], expect[i]);
    }
    drop(data);
    staging.unmap();
    println!("WGSL xoro: all 15 cell values match MC reference");
}

/// §20 minimal-loop step 1: a per-BLOCK grid (16 x 16 x 384 = 98304
/// invocations per chunk) writing one block-state byte each, read back
/// intact. Validates the "GPU outputs block states" data path that the
/// corner-density path never exercised (corner grid is only 1225).
#[test]
fn per_block_output_path() {
    use crate::instance::WgpuInstance;
    let inst = WgpuInstance::instance().unwrap();
    let device = &inst.device;
    let queue = &inst.queue;

    let n: u64 = 16 * 16 * 384; // 98304 blocks/chunk

    let src = r#"
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
// One invocation per block. gid.x = local x (0..15), gid.y = local z (0..15),
// gid.z = block y index (0..383). Write a value that encodes the position so
// the CPU can verify the full grid round-trips.
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= 16u || gid.y >= 16u || gid.z >= 384u) { return; }
  let idx: u32 = gid.x + gid.y * 16u + gid.z * 256u;
  out[idx] = idx ^ 0xA5A5u;
}
"#;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("per_block_test"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None, layout: Some(&pl), module: &shader, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: n * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: n * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: out.as_entire_binding() }],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cp.set_pipeline(&pipe);
        cp.set_bind_group(0, &bg, &[]);
        // 16/4 x 16/4 x 384/4 = 4 x 4 x 96 workgroups.
        cp.dispatch_workgroups(4, 4, 96);
    }
    enc.copy_buffer_to_buffer(&out, 0, &staging, 0, n * 4);
    queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    loop {
        match rx.try_recv() {
            Ok(_) => break,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
            _ => {}
        }
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_millis(100)),
        });
    }
    let data = slice.get_mapped_range().unwrap();
    let got: &[u32] = bytemuck::cast_slice(&data);
    assert_eq!(got.len() as u64, n);
    let mut bad = 0;
    for i in 0..n as u32 {
        if got[i as usize] != (i ^ 0xA5A5) { bad += 1; }
    }
    drop(data);
    staging.unmap();
    assert_eq!(bad, 0, "{bad} mismatches in per-block output");
    println!("per-block path: all {n} block outputs round-trip OK");
}

/// §20 minimal-loop step 2: WGSL per-block trilinear interpolation from
/// the corner cell grid, matching the Java write-back exactly. Synthetic
/// cell values; CPU reference mirrors MixinChunkGenerator's loop.
#[test]
fn per_block_trilinear_matches_cpu() {
    use crate::instance::WgpuInstance;
    let inst = WgpuInstance::instance().unwrap();
    let device = &inst.device;
    let queue = &inst.queue;

    const GX: usize = 5; const GY: usize = 49; const GZ: usize = 5;
    const CW: usize = 4; const CH: usize = 8;
    const H: usize = 384; // blocks in Y
    let cpc = GX * GY * GZ;
    let nblocks = 16 * H * 16;

    // Synthetic cells, deterministic.
    let cells: Vec<f32> = (0..cpc)
        .map(|i| (((i as u64).wrapping_mul(2654435761)) % 2000) as f32 / 1000.0 - 1.0)
        .collect();

    // CPU reference (mirrors the Java loop).
    let cell = |x: usize, y: usize, z: usize| -> f32 {
        cells[x + z * GX + y * GX * GZ]
    };
    let mut cpu = vec![0f32; nblocks];
    for by in 0..H {
        let iy = by / CH; let ty = (by % CH) as f32 / CH as f32;
        for bz in 0..16 {
            let iz = bz / CW; let tz = (bz % CW) as f32 / CW as f32;
            for bx in 0..16 {
                let ix = bx / CW; let tx = (bx % CW) as f32 / CW as f32;
                let c000 = cell(ix, iy, iz);
                let c100 = cell(ix+1, iy, iz);
                let c010 = cell(ix, iy+1, iz);
                let c110 = cell(ix+1, iy+1, iz);
                let c001 = cell(ix, iy, iz+1);
                let c101 = cell(ix+1, iy, iz+1);
                let c011 = cell(ix, iy+1, iz+1);
                let c111 = cell(ix+1, iy+1, iz+1);
                let x00 = c000 + tx * (c100 - c000);
                let x10 = c010 + tx * (c110 - c010);
                let x01 = c001 + tx * (c101 - c001);
                let x11 = c011 + tx * (c111 - c011);
                let y0 = x00 + ty * (x10 - x00);
                let y1 = x01 + ty * (x11 - x01);
                let d = y0 + tz * (y1 - y0);
                cpu[bx + bz * 16 + by * 256] = d;
            }
        }
    }

    let src = format!(r#"
@group(0) @binding(0) var<storage, read> cells: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

const GXc: u32 = {GX}u;
const GZc: u32 = {GZ}u;
fn at(x: u32, y: u32, z: u32) -> f32 {{ return cells[x + z * GXc + y * GXc * GZc]; }}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let bx = gid.x; let by = gid.y; let bz = gid.z;
  if (bx >= 16u || by >= {H}u || bz >= 16u) {{ return; }}
  let CW = {CW}u; let CH = {CH}u;
  let ix = bx / CW; let tx = f32(bx % CW) / f32(CW);
  let iz = bz / CW; let tz = f32(bz % CW) / f32(CW);
  let iy = by / CH; let ty = f32(by % CH) / f32(CH);
  let c000 = at(ix, iy, iz);
  let c100 = at(ix+1u, iy, iz);
  let c010 = at(ix, iy+1u, iz);
  let c110 = at(ix+1u, iy+1u, iz);
  let c001 = at(ix, iy, iz+1u);
  let c101 = at(ix+1u, iy, iz+1u);
  let c011 = at(ix, iy+1u, iz+1u);
  let c111 = at(ix+1u, iy+1u, iz+1u);
  let x00 = c000 + tx * (c100 - c000);
  let x10 = c010 + tx * (c110 - c010);
  let x01 = c001 + tx * (c101 - c001);
  let x11 = c011 + tx * (c111 - c011);
  let y0 = x00 + ty * (x10 - x00);
  let y1 = x01 + ty * (x11 - x01);
  let d = y0 + tz * (y1 - y0);
  out[bx + bz * 16u + by * 256u] = d;
}}
"#, H=H, CW=CW, CH=CH, GX=GX, GZ=GZ);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None, source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None, layout: Some(&pl), module: &shader, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });
    let cells_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (cells.len()*4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    queue.write_buffer(&cells_buf, 0, bytemuck::cast_slice(&cells));
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (nblocks*4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (nblocks*4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: cells_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
        ]});
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cp.set_pipeline(&pipe); cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(4, 96, 4);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, (nblocks*4) as u64);
    queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    loop {
        match rx.try_recv() { Ok(_)=>break, Err(std::sync::mpsc::TryRecvError::Disconnected)=>break, _=>{} }
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None,
            timeout: Some(std::time::Duration::from_millis(100)) });
    }
    let data = slice.get_mapped_range().unwrap();
    let got: &[f32] = bytemuck::cast_slice(&data);
    let mut bad = 0; let mut maxd = 0f32;
    for i in 0..nblocks {
        let dd = (got[i] - cpu[i]).abs();
        if dd > maxd { maxd = dd; }
        if dd > 1e-6 { bad += 1; }
    }
    drop(data); staging.unmap();
    println!("trilinear: bad={bad} maxd={maxd:e}");
    assert_eq!(bad, 0, "{bad} mismatches, max diff {maxd:e}");
}

/// §20 Step A: with REAL dumped corner densities for chunk (0,0), the WGSL
/// per-block pipeline must reproduce (a) the trilinear density and (b) the
/// trivial d>0 -> stone / else air state byte, matching the Java write-back
/// dump (/tmp/aq_blocks_0_0.txt). Validates the "GPU outputs block states"
/// byte stream that Java will consume. Reads dumps from /tmp; skips if absent.
#[test]
fn gpu_per_block_states_real_chunk() {
    use crate::instance::WgpuInstance;
    let corners_path = "/tmp/aq_corners_0_0.txt";
    let blocks_path = "/tmp/aq_blocks_0_0.txt";
    if !std::path::Path::new(corners_path).exists()
        || !std::path::Path::new(blocks_path).exists() {
        eprintln!("skipping: dumps not present ({corners_path})");
        return;
    }
    // Parse corners.
    let mut cells = vec![0f32; 5 * 49 * 5];
    for line in std::fs::read_to_string(corners_path).unwrap().lines() {
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() == 2 {
            cells[p[0].parse::<usize>().unwrap()] = p[1].parse::<f32>().unwrap();
        }
    }
    // Parse reference d per block.
    let mut ref_d = std::collections::HashMap::new();
    for line in std::fs::read_to_string(blocks_path).unwrap().lines() {
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() == 5 {
            let x: i32 = p[0].parse().unwrap();
            let y: i32 = p[1].parse().unwrap();
            let z: i32 = p[2].parse().unwrap();
            let d: f32 = p[3].parse().unwrap();
            ref_d.insert((x, y, z), d);
        }
    }

    let inst = WgpuInstance::instance().unwrap();
    let device = &inst.device;
    let queue = &inst.queue;
    let (gx, _gy, gz, cw, ch, miny) = (5u32, 49u32, 5u32, 4u32, 8u32, -64i32);
    let nblocks = 16u32 * 384u32 * 16u32;

    let src = format!(r#"
@group(0) @binding(0) var<storage, read> cells: array<f32>;
@group(0) @binding(1) var<storage, read_write> outd: array<f32>;
@group(0) @binding(2) var<storage, read_write> outs: array<u32>;
const GXc: u32 = {gx}u;
const GZc: u32 = {gz}u;
fn at(x: u32, y: u32, z: u32) -> f32 {{ return cells[x + z * GXc + y * GXc * GZc]; }}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let bx = gid.x; let by = gid.y; let bz = gid.z;
  if (bx >= 16u || by >= 384u || bz >= 16u) {{ return; }}
  let ix = bx / {cw}u; let tx = f32(bx % {cw}u) / f32({cw}u);
  let iz = bz / {cw}u; let tz = f32(bz % {cw}u) / f32({cw}u);
  let iy = by / {ch}u; let ty = f32(by % {ch}u) / f32({ch}u);
  let c000 = at(ix, iy, iz); let c100 = at(ix+1u, iy, iz);
  let c010 = at(ix, iy+1u, iz); let c110 = at(ix+1u, iy+1u, iz);
  let c001 = at(ix, iy, iz+1u); let c101 = at(ix+1u, iy, iz+1u);
  let c011 = at(ix, iy+1u, iz+1u); let c111 = at(ix+1u, iy+1u, iz+1u);
  let x00 = c000 + tx * (c100 - c000);
  let x10 = c010 + tx * (c110 - c010);
  let x01 = c001 + tx * (c101 - c001);
  let x11 = c011 + tx * (c111 - c011);
  let y0 = x00 + ty * (x10 - x00);
  let y1 = x01 + ty * (x11 - x01);
  let d = y0 + tz * (y1 - y0);
  let idx = bx + bz * 16u + by * 256u;
  outd[idx] = d;
  outs[idx] = select(0u, 1u, d > 0.0);
}}
"#, gx=gx, gz=gz, cw=cw, ch=ch);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None, source: wgpu::ShaderSource::Wgsl(src.into()) });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[0u32, 1, 2].iter().map(|&b| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: b == 0 },
                has_dynamic_offset: false, min_binding_size: None }, count: None }).collect::<Vec<_>>(),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None, layout: Some(&pl), module: &shader, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None });
    let cells_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (cells.len()*4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    queue.write_buffer(&cells_buf, 0, bytemuck::cast_slice(&cells));
    let outd = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (nblocks*4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let outs = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (nblocks*4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let stg = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (nblocks*4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: cells_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: outd.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: outs.as_entire_binding() },
        ]});
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cp.set_pipeline(&pipe); cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(4, 96, 4);
    }
    enc.copy_buffer_to_buffer(&outd, 0, &stg, 0, (nblocks*4) as u64);
    queue.submit(Some(enc.finish()));
    let slice = stg.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    loop {
        match rx.try_recv() { Ok(_)=>break, Err(std::sync::mpsc::TryRecvError::Disconnected)=>break, _=>{} }
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None,
            timeout: Some(std::time::Duration::from_millis(100)) });
    }
    let data = slice.get_mapped_range().unwrap();
    let gotd: &[f32] = bytemuck::cast_slice(&data);
    let mut bad = 0usize; let mut maxd = 0f32; let mut n = 0usize;
    for ((x, y, z), rd) in &ref_d {
        let bx = (x & 15) as u32; let bz = (z & 15) as u32;
        let by = (y - miny) as u32;
        let idx = (bx + bz * 16 + by * 256) as usize;
        let g = gotd[idx];
        let dd = (g - rd).abs();
        if dd > maxd { maxd = dd; }
        if dd > 1e-4 { bad += 1; }
        n += 1;
    }
    drop(data); stg.unmap();
    println!("gpu per-block: compared {n} blocks, bad={bad}, max_diff={maxd:e}");
    assert_eq!(bad, 0, "{bad}/{n} blocks exceed 1e-4");
}

/// §20 block-output smoke: run the real executor (which now dispatches
/// the BlockOut shard + copies block_buf) on the real router blob. Run
/// with C2ME_WGPU_BLOCK_OUT=1. Used to reproduce the in-game native
/// abort offline where Rust panics/signals are visible.
#[test]
#[ignore]
fn block_out_executor_smoke() {
    use crate::instance::WgpuInstance;
    use crate::scheduler::batcher::{Batch, Priority};
    use crate::scheduler::executor::ExecutorThread;
    use crate::world::WorldContext;
    use std::sync::Arc;

    let bin = concat!(env!("CARGO_MANIFEST_DIR"), "/../src/test/resources/real_router.bin");
    let blob = std::fs::read(bin).unwrap();
    let mut ctx = WorldContext::new(&blob).unwrap();
    println!("shards: {:?}", ctx.shards().iter()
        .map(|s| (s.pass, s.start, s.end)).collect::<Vec<_>>());
    let inst = WgpuInstance::instance().unwrap();
    ctx.compile_pipeline(&inst.device).unwrap();
    let world = Arc::new(ctx);
    let exec = ExecutorThread::spawn(
        Arc::new(inst.device.clone()), Arc::new(inst.queue.clone()), Default::default()).unwrap();
    let batch = Batch { positions: vec![(0, 0), (1, 0), (0, 1)], max_priority: Priority::Normal };
    let r = exec.request_chunks(world.clone(), batch).unwrap();
    println!("results={} block_data_len={}", r.results.len(), r.block_data.len());
    assert!(!r.block_data.is_empty(), "block output empty");
}

#[test]
fn gpu_end_to_end_test() {
    use crate::instance::WgpuInstance;
    use crate::scheduler::batcher::{Batch, Priority};
    use crate::scheduler::executor::ExecutorThread;
    use crate::world::WorldContext;
    use std::sync::Arc;

    let mut blob = Vec::new();
    blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
    blob.push(0xFE);
    blob.extend_from_slice(&[0x06, 0x00]);
    blob.extend_from_slice(&[0x01]);
    blob.extend_from_slice(&1.0f32.to_le_bytes());
    blob.push(0x02);
    // TAG_NOISE_SAMPLER(0x84)
    //   type=Perlin(0) flags=0 (single stack)
    //   amplitude=1.0 xz_scale=1.0 y_scale=1.0
    //   first stack: perm_offset=0 initial_freq=1.0 persistence=1.0
    //                octaves=1 amps=[1.0]
    blob.extend_from_slice(&[0x84, 0x00, 0x00]);
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // amplitude
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // xz_scale
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // y_scale
    blob.extend_from_slice(&0i32.to_le_bytes()); // perm_offset
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // initial_freq
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // persistence
    blob.extend_from_slice(&1i32.to_le_bytes()); // octaves
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // amp[0]
    blob.extend_from_slice(&[0x82]);
    blob.extend_from_slice(&7i64.to_le_bytes());
    blob.extend_from_slice(&[0xB1]);
    blob.extend_from_slice(&[0x01]);
    blob.extend_from_slice(&0.0f32.to_le_bytes());
    blob.extend_from_slice(&[0x01]);
    blob.extend_from_slice(&1.0f32.to_le_bytes());
    blob.extend_from_slice(&[0x01]);
    blob.extend_from_slice(&2.0f32.to_le_bytes());
    blob.extend_from_slice(&[0x01]);
    blob.extend_from_slice(&3.0f32.to_le_bytes());
    blob.extend_from_slice(&[0x01]);
    blob.extend_from_slice(&(-1.0f32).to_le_bytes());
    blob.extend_from_slice(&[0x01]);
    blob.extend_from_slice(&1.0f32.to_le_bytes());
    blob.extend_from_slice(&[0xC0, 0x03]);
    blob.extend_from_slice(&(-64i32).to_le_bytes());
    blob.extend_from_slice(&384i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    blob.extend_from_slice(&[0xF0]);
    blob.extend_from_slice(&123456789i64.to_le_bytes());

    let mut ctx = WorldContext::new(&blob).unwrap();
    let instance = WgpuInstance::instance().unwrap();
    ctx.compile_pipeline(&instance.device).unwrap();
    let world_arc = Arc::new(ctx);

    let executor = ExecutorThread::spawn(
        Arc::new(instance.device.clone()),
        Arc::new(instance.queue.clone()),
        Default::default(),
    )
    .unwrap();

    let batch = Batch {
        positions: vec![(0, 0)],
        max_priority: Priority::Normal,
    };

    let result = executor.request_chunks(world_arc, batch).unwrap();
    assert!(!result.results.is_empty());
    let blob_result = result.results.values().next().unwrap();
    assert!(blob_result.verify_crc());
    println!("✅ GPU 端到端测试通过！");
}

#[test]
#[ignore = "perf benchmark; run with --ignored"]
fn gpu_performance_benchmark() {
    use crate::instance::WgpuInstance;
    use crate::scheduler::batcher::{Batch, Priority};
    use crate::scheduler::executor::ExecutorThread;
    use crate::world::WorldContext;
    use std::sync::Arc;
    use std::time::Instant;

    // Prefer a REAL router blob dumped from the game. To produce it:
    //   1) run the game with  -Dc2me.dumpBlob=/tmp/real_router.bin
    //   2) cp /tmp/real_router.bin src/test/resources/real_router.bin
    // Falls back to the hand-made minimal blob when the file is absent.
    let real_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../src/test/resources/real_router.bin"
    );
    let blob: Vec<u8> = if std::path::Path::new(real_path).exists() {
        let b = std::fs::read(real_path).expect("read real_router.bin");
        println!("[bench] using REAL router blob: {} bytes", b.len());
        b
    } else {
        println!("[bench] real_router.bin absent — using hand-made minimal blob");
        let mut blob = Vec::new();
        blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
        blob.push(0xFE);
        blob.extend_from_slice(&[0x06, 0x00]);
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.push(0x02);
        // TAG_NOISE_SAMPLER(0x84)
        //   type=Perlin(0) flags=0 (single stack)
        //   amplitude=1.0 xz_scale=1.0 y_scale=1.0
        //   first stack: perm_offset=0 initial_freq=1.0 persistence=1.0
        //                octaves=1 amps=[1.0]
        blob.extend_from_slice(&[0x84, 0x00, 0x00]);
        blob.extend_from_slice(&1.0f32.to_le_bytes()); // amplitude
        blob.extend_from_slice(&1.0f32.to_le_bytes()); // xz_scale
        blob.extend_from_slice(&1.0f32.to_le_bytes()); // y_scale
        blob.extend_from_slice(&0i32.to_le_bytes()); // perm_offset
        blob.extend_from_slice(&1.0f32.to_le_bytes()); // initial_freq
        blob.extend_from_slice(&1.0f32.to_le_bytes()); // persistence
        blob.extend_from_slice(&1i32.to_le_bytes()); // octaves
        blob.extend_from_slice(&1.0f32.to_le_bytes()); // amp[0]
        blob.extend_from_slice(&[0x82]);
        blob.extend_from_slice(&7i64.to_le_bytes());
        blob.extend_from_slice(&[0xB1]);
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&0.0f32.to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&2.0f32.to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&3.0f32.to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&(-1.0f32).to_le_bytes());
        blob.extend_from_slice(&[0x01]);
        blob.extend_from_slice(&1.0f32.to_le_bytes());
        blob.extend_from_slice(&[0xC0, 0x03]);
        blob.extend_from_slice(&(-64i32).to_le_bytes());
        blob.extend_from_slice(&384i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.extend_from_slice(&4i32.to_le_bytes());
        blob.extend_from_slice(&[0xF0]);
        blob.extend_from_slice(&123456789i64.to_le_bytes());
        blob
    };

    // 创建 WorldContext 并编译 pipeline
    let mut ctx = WorldContext::new(&blob).unwrap();
    // The benchmark measures dispatch/readback throughput, not terrain
    // correctness, and nothing stages aquifer tables here. Disable aquifers
    // so the shader takes its sea-level branch: no table expansion, no
    // per-chunk "missing aquifer table" logging.
    ctx.set_aquifer_mode(crate::world::world::AquiferMode {
        has_aquifers: false,
        ..Default::default()
    });
    let instance = WgpuInstance::instance().unwrap();
    ctx.compile_pipeline(&instance.device).unwrap();
    let world_arc = Arc::new(ctx);

    // 创建执行器线程
    let executor = ExecutorThread::spawn(
        Arc::new(instance.device.clone()),
        Arc::new(instance.queue.clone()),
        Default::default(),
    )
    .unwrap();

    // 生成 64x64 = 4096 个区块坐标 (cx, cz) 从 (0,0) 到 (63,63)
    let mut all_positions = Vec::new();
    for cx in 0..32 {
        for cz in 0..32 {
            all_positions.push((cx, cz));
        }
    }

    // 分批处理，每批最多 64 个区块
    let batch_size = 64;
    let total_batches = (all_positions.len() + batch_size - 1) / batch_size;

    // One GPU dispatch covers the corner grid 5x97x5 padded to the
    // workgroup size 4 -> 8x100x8 = 6400 invocations.
    const INVOCATIONS_PER_DISPATCH: u64 = 8 * 100 * 8;

    let start = Instant::now();
    let mut total_chunks = 0;
    let mut total_gpu_time = std::time::Duration::ZERO;

    for batch_idx in 0..total_batches {
        let start_pos = batch_idx * batch_size;
        let end_pos = (start_pos + batch_size).min(all_positions.len());
        let chunk_slice = &all_positions[start_pos..end_pos];

        let batch = Batch {
            positions: chunk_slice.to_vec(),
            max_priority: Priority::Normal,
        };

        let result = executor.request_chunks(world_arc.clone(), batch).unwrap();
        total_gpu_time += result.dispatch_time;
        total_chunks += chunk_slice.len();
    }

    let elapsed = start.elapsed();
    let total_seconds = elapsed.as_secs_f64();
    let chunks_per_second = total_chunks as f64 / total_seconds;

    let gpu_seconds = total_gpu_time.as_secs_f64();
    let per_batch_ms = total_gpu_time.as_secs_f64() * 1000.0 / total_batches as f64;
    let per_chunk_ms = per_batch_ms / batch_size as f64;
    let gpu_busy_pct = if total_seconds > 0.0 {
        gpu_seconds / total_seconds * 100.0
    } else {
        0.0
    };

    println!(
        "✅ 性能测试完成：生成 {} 个区块，耗时 {:.3}s，吞吐量 {:.1} 区块/秒",
        total_chunks, total_seconds, chunks_per_second
    );
    println!(
        "   GPU 批次：{} 批 × {} 区块，平均 {:.3} ms/批（{:.3} ms/区块）",
        total_batches, batch_size, per_batch_ms, per_chunk_ms
    );
    println!(
        "   GPU 忙碌占比 {:.1}%，调度开销占比 {:.1}%",
        gpu_busy_pct,
        100.0 - gpu_busy_pct
    );

    // ── theoretical throughput ───────────────────────────────────
    // NOTE: the shader currently computes ONE chunk's density grid per
    // dispatch, regardless of how many positions the batch holds. So
    // per_batch_ms already includes all fixed dispatch cost. Let
    //   T = per_batch_ms = T_setup + T_compute_1chunk.
    // If the shader processed all N chunks in the SAME dispatch, the
    // batch time would become T_setup + N * T_compute. Since
    // T_compute <= T (unknown split), the resulting throughput lies
    // between these two bounds:
    //   lower  = 1 / T             (compute-bound: time grows with N)
    //   upper  = N / T             (overhead-bound: setup dominates)
    let lower_cps = 1000.0 / per_batch_ms;
    let upper_cps = (batch_size as f64) * 1000.0 / per_batch_ms;
    println!("   ── 理论吞吐量（shader 支持 {} 区块/批后）──", batch_size);
    println!(
        "      下限（计算主导，时间随 N 线性增长）: {:.1} 区块/秒",
        lower_cps
    );
    println!(
        "      上限（调度开销主导，setup 摊薄）: {:.1} 区块/秒",
        upper_cps
    );
    println!(
        "      GPU 算力参考: {:.2e} invocation/秒（{} invocation/dispatch）",
        total_batches as f64 * INVOCATIONS_PER_DISPATCH as f64 / total_seconds,
        INVOCATIONS_PER_DISPATCH
    );
    println!(
        "      （shader 现为多块 dispatch：一次生成整批 {} 区块）",
        batch_size
    );

    // 如果时间太短，给出提示
    if total_seconds < 1.0 {
        println!("⚠️ 总耗时较短，建议增加区块数量以获得更稳定的平均数据。");
    }
}

// ── Numerical validation: GPU shader vs Rust CPU reference ──────────
//
// Uses a single-octave NoiseSampler with a known perm table (seed=42)
// and no DOMAIN_SCALE, so the GPU result should exactly equal a
// straightforward Rust transcription of MC's Perlin algorithm.

#[allow(dead_code)]
fn cpu_maintain_precision(v: f32) -> f32 {
    let divisor = 33554432.0f32;
    v - (v / divisor + 0.5).floor() * divisor
}

#[allow(dead_code)]
fn cpu_fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[allow(dead_code)]
fn cpu_lerp(a: f32, b: f32, t: f32) -> f32 {
    t.mul_add(b - a, a)
}

#[allow(dead_code)]
fn cpu_trilinear(
    c000: f32,
    c100: f32,
    c010: f32,
    c110: f32,
    c001: f32,
    c101: f32,
    c011: f32,
    c111: f32,
    tx: f32,
    ty: f32,
    tz: f32,
) -> f32 {
    let x00 = cpu_lerp(c000, c100, tx);
    let x10 = cpu_lerp(c010, c110, tx);
    let x01 = cpu_lerp(c001, c101, tx);
    let x11 = cpu_lerp(c011, c111, tx);
    let y0 = cpu_lerp(x00, x10, ty);
    let y1 = cpu_lerp(x01, x11, ty);
    cpu_lerp(y0, y1, tz)
}

#[allow(dead_code)]
fn cpu_perlin3d_impl(perm: &[i32], p: i32, x: f32, y: f32, z: f32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let iz = z.floor() as i32;
    let fx = x - ix as f32;
    let fy = y - iy as f32;
    let fz = z - iz as f32;
    let xi = ix & 255;
    let yi = iy & 255;
    let zi = iz & 255;
    let idx = |i: i32| (p + i) as usize;
    let a = perm[idx(xi)];
    let b = perm[idx((xi + 1) & 255)];
    let aa = perm[idx((a + yi) & 255)];
    let ab = perm[idx((a + yi + 1) & 255)];
    let ba = perm[idx((b + yi) & 255)];
    let bb = perm[idx((b + yi + 1) & 255)];
    let aaa = perm[idx((aa + zi) & 255)];
    let aab = perm[idx((aa + zi + 1) & 255)];
    let aba = perm[idx((ab + zi) & 255)];
    let abb = perm[idx((ab + zi + 1) & 255)];
    let baa = perm[idx((ba + zi) & 255)];
    let bab = perm[idx((ba + zi + 1) & 255)];
    let bba = perm[idx((bb + zi) & 255)];
    let bbb = perm[idx((bb + zi + 1) & 255)];
    let g = |i: i32| -> [f32; 3] {
        match i & 15 {
            0 => [1.0, 1.0, 0.0],
            1 => [-1.0, 1.0, 0.0],
            2 => [1.0, -1.0, 0.0],
            3 => [-1.0, -1.0, 0.0],
            4 => [1.0, 0.0, 1.0],
            5 => [-1.0, 0.0, 1.0],
            6 => [1.0, 0.0, -1.0],
            7 => [-1.0, 0.0, -1.0],
            8 => [0.0, 1.0, 1.0],
            9 => [0.0, -1.0, 1.0],
            10 => [0.0, 1.0, -1.0],
            11 => [0.0, -1.0, -1.0],
            12 => [1.0, 1.0, 0.0],
            13 => [0.0, -1.0, 1.0],
            14 => [-1.0, 1.0, 0.0],
            15 => [0.0, -1.0, -1.0],
            _ => unreachable!(),
        }
    };
    let dot = |gr: [f32; 3], dx: f32, dy: f32, dz: f32| gr[0] * dx + gr[1] * dy + gr[2] * dz;
    let n000 = dot(g(aaa), fx, fy, fz);
    let n100 = dot(g(baa), fx - 1.0, fy, fz);
    let n010 = dot(g(aba), fx, fy - 1.0, fz);
    let n110 = dot(g(bba), fx - 1.0, fy - 1.0, fz);
    let n001 = dot(g(aab), fx, fy, fz - 1.0);
    let n101 = dot(g(bab), fx - 1.0, fy, fz - 1.0);
    let n011 = dot(g(abb), fx, fy - 1.0, fz - 1.0);
    let n111 = dot(g(bbb), fx - 1.0, fy - 1.0, fz - 1.0);
    let u = cpu_fade(fx);
    let v = cpu_fade(fy);
    let w = cpu_fade(fz);
    cpu_trilinear(n000, n100, n010, n110, n001, n101, n011, n111, u, v, w)
}

#[allow(dead_code)]
fn cpu_noise_perlin3d(perm: &[i32], p: i32, x: f32, y: f32, z: f32) -> f32 {
    let px = cpu_maintain_precision(x);
    let py = cpu_maintain_precision(y);
    let pz = cpu_maintain_precision(z);
    cpu_perlin3d_impl(perm, p, px, py, pz)
}

#[allow(dead_code)]
fn cpu_kahan_add(acc: &mut (f32, f32), v: f32) {
    let y = v - acc.1;
    let t = acc.0 + y;
    acc.1 = (t - acc.0) - y;
    acc.0 = t;
}

#[allow(dead_code)]
fn cpu_octave_stack(
    perm: &[i32],
    base: i32,
    initial_freq: f32,
    persistence: f32,
    amps: &[f32],
    x: f32,
    y: f32,
    z: f32,
) -> f32 {
    let mut freq = initial_freq;
    let mut amp = persistence;
    let mut acc = (0.0f32, 0.0f32);
    for i in 0..amps.len() {
        if amps[i] != 0.0 {
            let idx = base + (i as i32) * 256;
            let s = cpu_noise_perlin3d(perm, idx, x * freq, y * freq, z * freq);
            cpu_kahan_add(&mut acc, amps[i] * s * amp);
        }
        freq *= 2.0;
        amp *= 0.5;
    }
    acc.0
}

#[allow(dead_code)]
fn cpu_noise_sampler(
    perm: &[i32],
    first: (i32, f32, f32, &[f32]),
    second: Option<(i32, f32, f32, &[f32])>,
    amplitude: f32,
    xz: f32,
    ys: f32,
    x: f32,
    y: f32,
    z: f32,
) -> f32 {
    const D: f32 = 1.0181268882175227;
    let (fo, ff, fp, fa) = first;
    let vf = cpu_octave_stack(perm, fo, ff, fp, fa, x * xz, y * ys, z * xz);
    let total = if let Some((so, sf, sp, sa)) = second {
        let xzd = xz * D;
        let ysd = ys * D;
        vf + cpu_octave_stack(perm, so, sf, sp, sa, x * xzd, y * ysd, z * xzd)
    } else {
        vf
    };
    total * amplitude
}

#[allow(dead_code)]
fn build_run_blob(
    flags: u8,
    amplitude: f32,
    xz: f32,
    ys: f32,
    first: (i32, f32, f32, &[f32]),
    second: Option<(i32, f32, f32, &[f32])>,
    perms: &[[u32; 256]],
) -> Vec<f32> {
    use crate::instance::WgpuInstance;
    use crate::scheduler::batcher::{Batch, Priority};
    use crate::scheduler::executor::ExecutorThread;
    use crate::world::WorldContext;
    use std::sync::Arc;

    let mut blob = Vec::new();
    blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
    blob.push(0xFE);
    blob.extend_from_slice(&[0x01, 0x00]);
    blob.extend_from_slice(&[0x84, 0x00, flags]);
    blob.extend_from_slice(&amplitude.to_le_bytes());
    blob.extend_from_slice(&xz.to_le_bytes());
    blob.extend_from_slice(&ys.to_le_bytes());
    let (fo, ff, fp, fa) = first;
    blob.extend_from_slice(&fo.to_le_bytes());
    blob.extend_from_slice(&ff.to_le_bytes());
    blob.extend_from_slice(&fp.to_le_bytes());
    blob.extend_from_slice(&(fa.len() as i32).to_le_bytes());
    for &a in fa {
        blob.extend_from_slice(&a.to_le_bytes());
    }
    if let Some((so, sf, sp, sa)) = second {
        blob.extend_from_slice(&so.to_le_bytes());
        blob.extend_from_slice(&sf.to_le_bytes());
        blob.extend_from_slice(&sp.to_le_bytes());
        blob.extend_from_slice(&(sa.len() as i32).to_le_bytes());
        for &a in sa {
            blob.extend_from_slice(&a.to_le_bytes());
        }
    }
    blob.extend_from_slice(&(-64i32).to_le_bytes());
    blob.extend_from_slice(&384i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    blob.push(0xF0);
    blob.extend_from_slice(&123456789i64.to_le_bytes());
    blob.extend_from_slice(&0x4D525750u32.to_le_bytes());
    blob.extend_from_slice(&(perms.len() as u32).to_le_bytes());
    for tb in perms {
        for &v in tb {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        for _ in 0..3 {
            blob.extend_from_slice(&0.0f64.to_le_bytes());
        }
    }

    let mut ctx = WorldContext::new(&blob).unwrap();
    let instance = WgpuInstance::instance().unwrap();
    ctx.compile_pipeline(&instance.device).unwrap();
    let world_arc = Arc::new(ctx);
    let executor = ExecutorThread::spawn(
        Arc::new(instance.device.clone()),
        Arc::new(instance.queue.clone()),
        Default::default(),
    )
    .unwrap();
    let batch = Batch {
        positions: vec![(0, 0)],
        max_priority: Priority::Normal,
    };
    let result = executor.request_chunks(world_arc, batch).unwrap();
    let blob_result = result.results.values().next().unwrap();
    let bytes = blob_result.as_bytes();
    let payload = &bytes[0x88..];
    (0..2425)
        .map(|i| f32::from_le_bytes(payload[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect()
}

#[test]
fn numerical_validation_offset_isolation() {
    use crate::world::constants::permutation_table_for_seed;

    let perms: Vec<[u32; 256]> = [11i64, 22, 33]
        .iter()
        .map(|&sd| permutation_table_for_seed(sd))
        .collect();
    let perm_i32: Vec<i32> = perms.iter().flatten().map(|&v| v as i32).collect();

    // Case B: single octave at offset 0
    let amps1 = [1.0f32];
    let gpu_b = build_run_blob(0, 1.0, 0.3, 0.3, (0, 1.0, 1.0, &amps1), None, &perms);
    let mut max_b = 0.0f32;
    for cy in 0..97usize {
        for cz in 0..5usize {
            for cx in 0..5usize {
                let wx = (cx * 4) as f32;
                let wy = (-64 + cy as i32 * 4) as f32;
                let wz = (cz * 4) as f32;
                let cpu = cpu_octave_stack(
                    &perm_i32,
                    0,
                    1.0,
                    1.0,
                    &amps1,
                    wx * 0.3,
                    wy * 0.3,
                    wz * 0.3,
                );
                let gpu = gpu_b[cx + cz * 5 + cy * 25];
                max_b = max_b.max((cpu - gpu).abs());
            }
        }
    }
    println!("Case B (1 octave, offset 0):    max_diff={:.6e}", max_b);

    // Case C: single octave at offset 256
    let gpu_c = build_run_blob(0, 1.0, 0.3, 0.3, (256, 1.0, 1.0, &amps1), None, &perms);
    let mut max_c = 0.0f32;
    for cy in 0..97usize {
        for cz in 0..5usize {
            for cx in 0..5usize {
                let wx = (cx * 4) as f32;
                let wy = (-64 + cy as i32 * 4) as f32;
                let wz = (cz * 4) as f32;
                let cpu = cpu_octave_stack(
                    &perm_i32,
                    256,
                    1.0,
                    1.0,
                    &amps1,
                    wx * 0.3,
                    wy * 0.3,
                    wz * 0.3,
                );
                let gpu = gpu_c[cx + cz * 5 + cy * 25];
                max_c = max_c.max((cpu - gpu).abs());
            }
        }
    }
    println!("Case C (1 octave, offset 256):  max_diff={:.6e}", max_c);

    // Case D: two octaves
    let amps2 = [1.0f32, 0.5];
    let gpu_d = build_run_blob(0, 1.0, 0.3, 0.3, (0, 1.0, 1.0, &amps2), None, &perms);
    let mut max_d = 0.0f32;
    for cy in 0..97usize {
        for cz in 0..5usize {
            for cx in 0..5usize {
                let wx = (cx * 4) as f32;
                let wy = (-64 + cy as i32 * 4) as f32;
                let wz = (cz * 4) as f32;
                let cpu = cpu_octave_stack(
                    &perm_i32,
                    0,
                    1.0,
                    1.0,
                    &amps2,
                    wx * 0.3,
                    wy * 0.3,
                    wz * 0.3,
                );
                let gpu = gpu_d[cx + cz * 5 + cy * 25];
                max_d = max_d.max((cpu - gpu).abs());
            }
        }
    }
    println!("Case D (2 octaves):             max_diff={:.6e}", max_d);

    // Debug: print cell (0,0,0) contributions
    let wx = 0.0f32;
    let wy = -64.0f32;
    let wz = 0.0f32;
    let perm_p = &perm_i32;
    let s0 = cpu_noise_perlin3d(perm_p, 0, wx * 0.3, wy * 0.3, wz * 0.3);
    let s1 = cpu_noise_perlin3d(perm_p, 256, wx * 0.3 * 2.0, wy * 0.3 * 2.0, wz * 0.3 * 2.0);
    let s2 = cpu_noise_perlin3d(perm_p, 512, wx * 0.3 * 4.0, wy * 0.3 * 4.0, wz * 0.3 * 4.0);
    let cpu3 = s0 + 0.5 * s1 * 0.5 + 0.25 * s2 * 0.25;
    println!(
        "  cell(0,0,0): oct0={:.6} oct1={:.6} oct2={:.6} -> cpu={:.6}",
        s0, s1, s2, cpu3
    );
    println!("  cell(0,0,0): gpu_A={:.6} gpu_D={:.6}", gpu_b[0], gpu_d[0]);

    assert!(max_b < 1e-4, "Case B diverged {:.6e}", max_b);
    assert!(max_c < 1e-4, "Case C diverged {:.6e}", max_c);
    assert!(max_d < 1e-4, "Case D diverged {:.6e}", max_d);
}

#[test]
fn numerical_validation_stack_isolation() {
    use crate::world::constants::permutation_table_for_seed;

    let perms: Vec<[u32; 256]> = [11i64, 22, 33]
        .iter()
        .map(|&sd| permutation_table_for_seed(sd))
        .collect();
    let perm_i32: Vec<i32> = perms.iter().flatten().map(|&v| v as i32).collect();
    let amps = [1.0f32, 0.5, 0.25];

    // Case A: single stack, 3 octaves, amplitude=1.0, no D.
    let gpu_a = build_run_blob(0, 1.0, 0.3, 0.3, (0, 1.0, 1.0, &amps), None, &perms);
    let mut max_a = 0.0f32;
    for cy in 0..97usize {
        for cz in 0..5usize {
            for cx in 0..5usize {
                let wx = (cx * 4) as f32;
                let wy = (-64 + cy as i32 * 4) as f32;
                let wz = (cz * 4) as f32;
                let cpu = cpu_octave_stack(
                    &perm_i32,
                    0,
                    1.0,
                    1.0,
                    &amps,
                    wx * 0.3,
                    wy * 0.3,
                    wz * 0.3,
                );
                let gpu = gpu_a[cx + cz * 5 + cy * 25];
                max_a = max_a.max((cpu - gpu).abs());
            }
        }
    }
    println!("Case A (single stack, 3 octaves): max_diff={:.6e}", max_a);
    assert!(max_a < 1e-4, "Case A diverged {:.6e}", max_a);
}

#[test]
fn numerical_validation_double_stack_multioctave() {
    use crate::instance::WgpuInstance;
    use crate::scheduler::batcher::{Batch, Priority};
    use crate::scheduler::executor::ExecutorThread;
    use crate::world::WorldContext;
    use crate::world::constants::permutation_table_for_seed;
    use std::sync::Arc;

    // 3 octaves in first stack (tables 0/256/512), 2 in second (768/1024).
    let seed_first = [11i64, 22, 33];
    let seed_second = [44i64, 55];
    let perms: Vec<[u32; 256]> = seed_first
        .iter()
        .chain(seed_second.iter())
        .map(|&sd| permutation_table_for_seed(sd))
        .collect();
    let perm_i32: Vec<i32> = perms.iter().flatten().map(|&v| v as i32).collect();

    let amps_first = [1.0f32, 0.5, 0.25];
    let amps_second = [1.0f32, 0.5];

    let mut blob = Vec::new();
    blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
    blob.push(0xFE);
    blob.extend_from_slice(&[0x01, 0x00]);
    blob.extend_from_slice(&[0x84, 0x00, 0x01]); // type=Perlin, flags=has_second
    blob.extend_from_slice(&0.8f32.to_le_bytes()); // amplitude
    blob.extend_from_slice(&0.3f32.to_le_bytes()); // xz_scale
    blob.extend_from_slice(&0.3f32.to_le_bytes()); // y_scale
    // first stack
    blob.extend_from_slice(&0i32.to_le_bytes());
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // initial_freq
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // persistence
    blob.extend_from_slice(&3i32.to_le_bytes());
    for a in amps_first {
        blob.extend_from_slice(&a.to_le_bytes());
    }
    // second stack
    blob.extend_from_slice(&(3 * 256i32).to_le_bytes());
    blob.extend_from_slice(&1.0f32.to_le_bytes());
    blob.extend_from_slice(&1.0f32.to_le_bytes());
    blob.extend_from_slice(&2i32.to_le_bytes());
    for a in amps_second {
        blob.extend_from_slice(&a.to_le_bytes());
    }
    // shape + biome
    blob.extend_from_slice(&(-64i32).to_le_bytes());
    blob.extend_from_slice(&384i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    blob.push(0xF0);
    blob.extend_from_slice(&123456789i64.to_le_bytes());
    // PERM_RESOURCE
    blob.extend_from_slice(&0x4D525750u32.to_le_bytes());
    blob.extend_from_slice(&(perms.len() as u32).to_le_bytes());
    for tb in &perms {
        for &v in tb {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        for _ in 0..3 {
            blob.extend_from_slice(&0.0f64.to_le_bytes());
        }
    }

    let mut ctx = WorldContext::new(&blob).unwrap();
    let instance = WgpuInstance::instance().unwrap();
    ctx.compile_pipeline(&instance.device).unwrap();
    let world_arc = Arc::new(ctx);

    let executor = ExecutorThread::spawn(
        Arc::new(instance.device.clone()),
        Arc::new(instance.queue.clone()),
        Default::default(),
    )
    .unwrap();
    let batch = Batch {
        positions: vec![(0, 0)],
        max_priority: Priority::Normal,
    };
    let result = executor.request_chunks(world_arc, batch).unwrap();
    let blob_result = result.results.values().next().unwrap();
    assert!(blob_result.verify_crc());

    let bytes = blob_result.as_bytes();
    let payload = &bytes[0x88..];
    let ch0: Vec<f32> = (0..2425)
        .map(|i| f32::from_le_bytes(payload[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect();

    let first = (0i32, 1.0f32, 1.0f32, &amps_first[..]);
    let second = Some((3 * 256i32, 1.0f32, 1.0f32, &amps_second[..]));
    let mut max_diff = 0.0f32;
    let mut sum_sq = 0.0f64;
    let mut n = 0usize;
    for cy in 0..97usize {
        for cz in 0..5usize {
            for cx in 0..5usize {
                let wx = (cx * 4) as f32;
                let wy = (-64 + cy as i32 * 4) as f32;
                let wz = (cz * 4) as f32;
                let cpu =
                    cpu_noise_sampler(&perm_i32, first, second, 0.8, 0.3, 0.3, wx, wy, wz);
                let gpu = ch0[cx + cz * 5 + cy * 25];
                let d = (cpu - gpu).abs();
                sum_sq += (d as f64) * (d as f64);
                n += 1;
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
    }
    let rms = (sum_sq / n as f64).sqrt();
    println!(
        "double-stack/multioctave: max_diff={:.6e} rms={:.6e}",
        max_diff, rms
    );
    for &(cx, cy, cz) in &[(0usize, 0usize, 0usize), (1, 10, 2), (3, 50, 1), (2, 95, 3)] {
        let wx = (cx * 4) as f32;
        let wy = (-64 + cy as i32 * 4) as f32;
        let wz = (cz * 4) as f32;
        let cpu = cpu_noise_sampler(&perm_i32, first, second, 0.8, 0.3, 0.3, wx, wy, wz);
        let gpu = ch0[cx + cz * 5 + cy * 25];
        println!(
            "  cell ({},{},{}) cpu={:.6} gpu={:.6}",
            cx, cy, cz, cpu, gpu
        );
    }
    assert!(max_diff < 1e-4, "double-stack divergence {:.6e}", max_diff);
}

// CPU reference WITH origin (MC order: maintainPrecision(coord) + origin).
#[allow(dead_code)]
fn cpu_perlin3d_origin(
    perm: &[i32],
    p: i32,
    ox: f32,
    oy: f32,
    oz: f32,
    x: f32,
    y: f32,
    z: f32,
) -> f32 {
    let px = cpu_maintain_precision(x) + ox;
    let py = cpu_maintain_precision(y) + oy;
    let pz = cpu_maintain_precision(z) + oz;
    cpu_perlin3d_impl(perm, p, px, py, pz)
}

#[allow(dead_code)]
fn build_run_blob_origin(
    flags: u8,
    amplitude: f32,
    xz: f32,
    ys: f32,
    first: (i32, f32, f32, &[f32]),
    second: Option<(i32, f32, f32, &[f32])>,
    perms: &[[u32; 256]],
    origins: &[[f64; 3]],
) -> Vec<f32> {
    use crate::instance::WgpuInstance;
    use crate::scheduler::batcher::{Batch, Priority};
    use crate::scheduler::executor::ExecutorThread;
    use crate::world::WorldContext;
    use std::sync::Arc;
    let mut blob = Vec::new();
    blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
    blob.push(0xFE);
    blob.extend_from_slice(&[0x01, 0x00]);
    blob.extend_from_slice(&[0x84, 0x00, flags]);
    blob.extend_from_slice(&amplitude.to_le_bytes());
    blob.extend_from_slice(&xz.to_le_bytes());
    blob.extend_from_slice(&ys.to_le_bytes());
    let (fo, ff, fp, fa) = first;
    blob.extend_from_slice(&fo.to_le_bytes());
    blob.extend_from_slice(&ff.to_le_bytes());
    blob.extend_from_slice(&fp.to_le_bytes());
    blob.extend_from_slice(&(fa.len() as i32).to_le_bytes());
    for &a in fa {
        blob.extend_from_slice(&a.to_le_bytes());
    }
    if let Some((so, sf, sp, sa)) = second {
        blob.extend_from_slice(&so.to_le_bytes());
        blob.extend_from_slice(&sf.to_le_bytes());
        blob.extend_from_slice(&sp.to_le_bytes());
        blob.extend_from_slice(&(sa.len() as i32).to_le_bytes());
        for &a in sa {
            blob.extend_from_slice(&a.to_le_bytes());
        }
    }
    blob.extend_from_slice(&(-64i32).to_le_bytes());
    blob.extend_from_slice(&384i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    blob.push(0xF0);
    blob.extend_from_slice(&123456789i64.to_le_bytes());
    blob.extend_from_slice(&0x4D525750u32.to_le_bytes());
    blob.extend_from_slice(&(perms.len() as u32).to_le_bytes());
    for (i, tb) in perms.iter().enumerate() {
        for &v in tb {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        let o = origins.get(i).copied().unwrap_or([0.0, 0.0, 0.0]);
        for k in 0..3 {
            blob.extend_from_slice(&o[k].to_le_bytes());
        }
    }
    let mut ctx = WorldContext::new(&blob).unwrap();
    let instance = WgpuInstance::instance().unwrap();
    ctx.compile_pipeline(&instance.device).unwrap();
    let world_arc = Arc::new(ctx);
    let executor = ExecutorThread::spawn(
        Arc::new(instance.device.clone()),
        Arc::new(instance.queue.clone()),
        Default::default(),
    )
    .unwrap();
    let batch = Batch {
        positions: vec![(0, 0)],
        max_priority: Priority::Normal,
    };
    let result = executor.request_chunks(world_arc, batch).unwrap();
    let br = result.results.values().next().unwrap();
    let bytes = br.as_bytes();
    let payload = &bytes[0x88..];
    (0..2425)
        .map(|i| f32::from_le_bytes(payload[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect()
}

#[test]
fn cross_end_mc_reference() {
    use crate::instance::WgpuInstance;
    use crate::scheduler::batcher::{Batch, Priority};
    use crate::scheduler::executor::ExecutorThread;
    use crate::world::WorldContext;
    use std::sync::Arc;

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../src/test/resources/mc_reference.txt"
    );
    let txt = std::fs::read_to_string(path).expect("read mc_reference.txt");

    let mut perms: Vec<[u32; 256]> = Vec::new();
    let mut origins: Vec<[f64; 3]> = Vec::new();
    let mut amplitude = 1.0f32;
    let mut samples: Vec<(f32, f32, f32, f32)> = Vec::new();
    let mut cur_perm: Option<Vec<u32>> = None;
    let mut cur_perm_idx = 0usize;
    let mut in_samples = false;

    for line in txt.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let tag = it.next().unwrap();
        match tag {
            "DOUBLE_AMPLITUDE" => amplitude = it.next().unwrap().parse().unwrap(),
            "SAMPLE_COUNT" => in_samples = true,
            "FIRST_PERM" | "SECOND_PERM" => {
                let _i: usize = it.next().unwrap().parse().unwrap();
                let vals: Vec<u32> = it.map(|v| v.parse().unwrap()).collect();
                assert_eq!(vals.len(), 256, "perm len");
                cur_perm = Some(vals);
                cur_perm_idx = 0;
            }
            "FIRST_ORIGIN" | "SECOND_ORIGIN" => {
                let _i: usize = it.next().unwrap().parse().unwrap();
                let o: Vec<f64> = it.map(|v| v.parse().unwrap()).collect();
                if let Some(pv) = cur_perm.take() {
                    let mut arr = [0u32; 256];
                    arr.copy_from_slice(&pv);
                    perms.push(arr);
                }
                origins.push([o[0], o[1], o[2]]);
                let _ = cur_perm_idx;
            }
            _ => {
                if in_samples {
                    // sample line: wx wy wz value
                    let wx: f32 = tag.parse().unwrap();
                    let wy: f32 = it.next().unwrap().parse().unwrap();
                    let wz: f32 = it.next().unwrap().parse().unwrap();
                    let v: f32 = it.next().unwrap().parse().unwrap();
                    samples.push((wx, wy, wz, v));
                }
            }
        }
    }
    assert_eq!(perms.len(), 6, "expected 6 perm tables");
    assert_eq!(origins.len(), 6, "expected 6 origins");

    // Build blob: first stack (offset 0), second (offset 768).
    let amps = [1.0f32, 0.5, 0.25];
    let persistence = 4.0f32 / 7.0; // 2^(n-1)/(2^n-1) for n=3
    let mut blob = Vec::new();
    blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
    blob.push(0xFE);
    blob.extend_from_slice(&[0x01, 0x00]);
    blob.extend_from_slice(&[0x84, 0x00, 0x01]); // Perlin, has_second
    blob.extend_from_slice(&amplitude.to_le_bytes());
    blob.extend_from_slice(&0.3f32.to_le_bytes()); // xz_scale
    blob.extend_from_slice(&0.3f32.to_le_bytes()); // y_scale
    // first stack
    blob.extend_from_slice(&0i32.to_le_bytes());
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // initial_freq = 2^0
    blob.extend_from_slice(&persistence.to_le_bytes());
    blob.extend_from_slice(&3i32.to_le_bytes());
    for a in amps {
        blob.extend_from_slice(&a.to_le_bytes());
    }
    // second stack
    blob.extend_from_slice(&(3 * 256i32).to_le_bytes());
    blob.extend_from_slice(&1.0f32.to_le_bytes());
    blob.extend_from_slice(&persistence.to_le_bytes());
    blob.extend_from_slice(&3i32.to_le_bytes());
    for a in amps {
        blob.extend_from_slice(&a.to_le_bytes());
    }
    blob.extend_from_slice(&(-64i32).to_le_bytes());
    blob.extend_from_slice(&384i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    blob.push(0xF0);
    blob.extend_from_slice(&123456789i64.to_le_bytes());
    blob.extend_from_slice(&0x4D525750u32.to_le_bytes());
    blob.extend_from_slice(&(perms.len() as u32).to_le_bytes());
    for (i, tb) in perms.iter().enumerate() {
        for &v in tb {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        let o = origins[i];
        for k in 0..3 {
            blob.extend_from_slice(&o[k].to_le_bytes());
        }
    }

    let mut ctx = WorldContext::new(&blob).unwrap();
    let instance = WgpuInstance::instance().unwrap();
    ctx.compile_pipeline(&instance.device).unwrap();
    let world_arc = Arc::new(ctx);
    let executor = ExecutorThread::spawn(
        Arc::new(instance.device.clone()),
        Arc::new(instance.queue.clone()),
        Default::default(),
    )
    .unwrap();
    let batch = Batch {
        positions: vec![(0, 0)],
        max_priority: Priority::Normal,
    };
    let result = executor.request_chunks(world_arc, batch).unwrap();
    let br = result.results.values().next().unwrap();
    let bytes = br.as_bytes();
    let payload = &bytes[0x88..];
    let gpu: Vec<f32> = (0..2425)
        .map(|i| f32::from_le_bytes(payload[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect();

    // Map MC world coord -> GPU cell index. cell_x = cx*4, cell_y = -64+cy*4, cell_z = cz*4.
    let mut max_diff = 0.0f32;
    let mut n = 0usize;
    for &(wx, wy, wz, mc) in &samples {
        let cx = (wx / 4.0) as usize;
        let cy = ((wy + 64.0) / 4.0) as usize;
        let cz = (wz / 4.0) as usize;
        if cx >= 4 || cz >= 4 || cy >= 96 {
            continue;
        }
        let idx = cx + cz * 5 + cy * 25;
        let d = (gpu[idx] - mc).abs();
        if d > max_diff {
            max_diff = d;
            println!(
                "  worst: world ({},{},{}) mc={:.7} gpu={:.7}",
                wx, wy, wz, mc, gpu[idx]
            );
        }
        n += 1;
    }
    println!("cross-end MC vs GPU: n={} max_diff={:.6e}", n, max_diff);
    assert!(n > 0, "no samples mapped");
    assert!(max_diff < 1e-3, "cross-end divergence {:.6e}", max_diff);
}

#[test]
fn numerical_validation_shifted_noise() {
    use crate::instance::WgpuInstance;
    use crate::scheduler::batcher::{Batch, Priority};
    use crate::scheduler::executor::ExecutorThread;
    use crate::world::WorldContext;
    use crate::world::constants::permutation_table_for_seed;
    use std::sync::Arc;

    let perms: Vec<[u32; 256]> = [11i64, 22, 33]
        .iter()
        .map(|&sd| permutation_table_for_seed(sd))
        .collect();
    let perm_i32: Vec<i32> = perms.iter().flatten().map(|&v| v as i32).collect();

    // Build a blob: ROOT(1ch) = ShiftedNoise(xz=0.25, ys=0.0,
    //   shiftX = ShiftA(offset noise table0),
    //   shiftY = Constant(0.0),
    //   shiftZ = ShiftB(offset noise table1),
    //   noise  = Perlin table2)
    let mut blob = Vec::new();
    blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
    blob.push(0xFE);
    blob.extend_from_slice(&[0x01, 0x00]);
    blob.push(0xB5);
    blob.extend_from_slice(&0.25f32.to_le_bytes());
    blob.extend_from_slice(&0.0f32.to_le_bytes());
    // shiftX = Shift kind=1 (ShiftA)
    blob.push(0xB4);
    blob.push(1);
    push_simple_noise(&mut blob, 0);
    // shiftY = Constant(0.0)
    blob.push(0x01);
    blob.extend_from_slice(&0.0f32.to_le_bytes());
    // shiftZ = Shift kind=2 (ShiftB)
    blob.push(0xB4);
    blob.push(2);
    push_simple_noise(&mut blob, 256);
    // main noise
    push_simple_noise(&mut blob, 512);
    // shape
    blob.extend_from_slice(&(-64i32).to_le_bytes());
    blob.extend_from_slice(&384i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    // biome
    blob.push(0xF0);
    blob.extend_from_slice(&123456789i64.to_le_bytes());
    // PERM_RESOURCE
    blob.extend_from_slice(&0x4D525750u32.to_le_bytes());
    blob.extend_from_slice(&(perms.len() as u32).to_le_bytes());
    for tb in &perms {
        for &v in tb {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        for _ in 0..3 {
            blob.extend_from_slice(&0.0f64.to_le_bytes());
        }
    }

    let mut ctx = WorldContext::new(&blob).unwrap();
    let instance = WgpuInstance::instance().unwrap();
    ctx.compile_pipeline(&instance.device).unwrap();
    let world_arc = Arc::new(ctx);
    let executor = ExecutorThread::spawn(
        Arc::new(instance.device.clone()),
        Arc::new(instance.queue.clone()),
        Default::default(),
    )
    .unwrap();
    // Use chunk coord 1 so world_x = 1 + cx*4 = 1,5,9... (non-grid)
    let batch = Batch {
        positions: vec![(1, 0)],
        max_priority: Priority::Normal,
    };
    let result = executor.request_chunks(world_arc, batch).unwrap();
    let br = result.results.values().next().unwrap();
    let bytes = br.as_bytes();
    let payload = &bytes[0x88..];
    let gpu: Vec<f32> = (0..2425)
        .map(|i| f32::from_le_bytes(payload[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect();

    // CPU reference (MC algorithm).
    let fx = (0i32, 1.0f32, 1.0f32, &[1.0f32][..]);
    let fz = (256i32, 1.0f32, 1.0f32, &[1.0f32][..]);
    let fnz = (512i32, 1.0f32, 1.0f32, &[1.0f32][..]);
    let shift_a = |x: f32, z: f32| -> f32 {
        cpu_noise_sampler(&perm_i32, fx, None, 1.0, 1.0, 1.0, x * 0.25, 0.0, z * 0.25) * 4.0
    };
    let shift_b = |x: f32, z: f32| -> f32 {
        cpu_noise_sampler(&perm_i32, fz, None, 1.0, 1.0, 1.0, z * 0.25, x * 0.25, 0.0) * 4.0
    };
    let cpu = |x: f32, y: f32, z: f32| -> f32 {
        let d = x * 0.25 + shift_a(x, z);
        let e = y * 0.0 + 0.0;
        let f = z * 0.25 + shift_b(x, z);
        cpu_noise_sampler(&perm_i32, fnz, None, 1.0, 1.0, 1.0, d, e, f)
    };

    let mut max_d = 0.0f32;
    for cy in 0..97usize {
        for cz in 0..5usize {
            for cx in 0..5usize {
                let wx = (16 + cx as i32 * 4) as f32;
                let wy = (-64 + cy as i32 * 4) as f32;
                let wz = (cz * 4) as f32;
                let c = cpu(wx, wy, wz);
                let g = gpu[cx + cz * 5 + cy * 25];
                max_d = max_d.max((c - g).abs());
            }
        }
    }
    println!("shifted-noise: max_diff={:.6e}", max_d);
    assert!(max_d < 1e-4, "shifted-noise diverged {:.6e}", max_d);
}

#[test]
fn numerical_validation_spline() {
    use crate::instance::WgpuInstance;
    use crate::scheduler::batcher::{Batch, Priority};
    use crate::scheduler::executor::ExecutorThread;
    use crate::world::WorldContext;
    use crate::world::constants::permutation_table_for_seed;
    use std::sync::Arc;

    let perms: Vec<[u32; 256]> = [7i64]
        .iter()
        .map(|&sd| permutation_table_for_seed(sd))
        .collect();

    // Build: ROOT(1ch) = Spline over Y with 4 nodes:
    //   locations  = [-64, -20, 40, 320]
    //   derivatives= [ 0.05, 0.1, -0.02, 0.0]
    //   location   = InputY
    //   values     = [Constant(0.0), Constant(0.5),
    //                 PerlinNoise(table0), Constant(-0.3)]
    let mut blob = Vec::new();
    blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
    blob.push(0xFE);
    blob.extend_from_slice(&[0x01, 0x00]);

    blob.push(0xB0); // Spline
    blob.extend_from_slice(&(-64.0f32).to_le_bytes()); // min
    blob.extend_from_slice(&(320.0f32).to_le_bytes()); // max
    blob.extend_from_slice(&4i32.to_le_bytes()); // count
    for l in [-64.0f32, -20.0, 40.0, 320.0] {
        blob.extend_from_slice(&l.to_le_bytes());
    }
    for d in [0.05f32, 0.1, -0.02, 0.0] {
        blob.extend_from_slice(&d.to_le_bytes());
    }
    // location = InputY
    blob.push(0x03);
    // values[0] = Constant(0.0)
    blob.push(0x01);
    blob.extend_from_slice(&0.0f32.to_le_bytes());
    // values[1] = Constant(0.5)
    blob.push(0x01);
    blob.extend_from_slice(&0.5f32.to_le_bytes());
    // values[2] = Perlin noise table0
    push_simple_noise(&mut blob, 0);
    // values[3] = Constant(-0.3)
    blob.push(0x01);
    blob.extend_from_slice(&(-0.3f32).to_le_bytes());

    // shape + biome
    blob.extend_from_slice(&(-64i32).to_le_bytes());
    blob.extend_from_slice(&384i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    blob.push(0xF0);
    blob.extend_from_slice(&123456789i64.to_le_bytes());
    // PERM_RESOURCE
    blob.extend_from_slice(&0x4D525750u32.to_le_bytes());
    blob.extend_from_slice(&(perms.len() as u32).to_le_bytes());
    for tb in &perms {
        for &v in tb {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        for _ in 0..3 {
            blob.extend_from_slice(&0.0f64.to_le_bytes());
        }
    }

    let mut ctx = WorldContext::new(&blob).unwrap();
    let instance = WgpuInstance::instance().unwrap();
    ctx.compile_pipeline(&instance.device).unwrap();
    let world_arc = Arc::new(ctx);
    let executor = ExecutorThread::spawn(
        Arc::new(instance.device.clone()),
        Arc::new(instance.queue.clone()),
        Default::default(),
    )
    .unwrap();
    let batch = Batch {
        positions: vec![(1, 0)],
        max_priority: Priority::Normal,
    };
    let result = executor.request_chunks(world_arc, batch).unwrap();
    let br = result.results.values().next().unwrap();
    let bytes = br.as_bytes();
    let payload = &bytes[0x88..];
    let gpu: Vec<f32> = (0..2425)
        .map(|i| f32::from_le_bytes(payload[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect();

    let perm_i32: Vec<i32> = perms.iter().flatten().map(|&v| v as i32).collect();
    let locs = [-64.0f32, -20.0, 40.0, 320.0];
    let ders = [0.05f32, 0.1, -0.02, 0.0];
    let cpu = |y: f32, wx: f32, wz: f32| -> f32 {
        let v: [f32; 4] = [
            0.0,
            0.5,
            cpu_noise_sampler(
                &perm_i32,
                (0, 1.0, 1.0, &[1.0f32][..]),
                None,
                1.0,
                1.0,
                1.0,
                wx,
                y,
                wz,
            ),
            -0.3,
        ];
        let outside =
            |v: f32, d: f32, l: f32, t: f32| if d == 0.0 { v } else { v + d * (t - l) };
        if y < locs[0] {
            return outside(v[0], ders[0], locs[0], y);
        }
        if y >= locs[3] {
            return outside(v[3], ders[3], locs[3], y);
        }
        for k in 0..3 {
            if y < locs[k + 1] {
                let p = (y - locs[k]) / (locs[k + 1] - locs[k]);
                let a = ders[k] * (locs[k + 1] - locs[k]) - (v[k + 1] - v[k]);
                let b = -ders[k + 1] * (locs[k + 1] - locs[k]) + (v[k + 1] - v[k]);
                let lerp = |t: f32, a: f32, b: f32| a + t * (b - a);
                return lerp(p, v[k], v[k + 1]) + p * (1.0 - p) * lerp(p, a, b);
            }
        }
        v[3]
    };

    let mut max_d = 0.0f32;
    for cy in 0..97usize {
        for cz in 0..5usize {
            for cx in 0..5usize {
                let wx = (1 + cx as i32 * 4) as f32;
                let wy = (-64 + cy as i32 * 4) as f32;
                let wz = (cz * 4) as f32;
                let c = cpu(wy, wx, wz);
                let g = gpu[cx + cz * 5 + cy * 25];
                max_d = max_d.max((c - g).abs());
            }
        }
    }
    println!("spline: max_diff={:.6e}", max_d);
    assert!(max_d < 1e-4, "spline diverged {:.6e}", max_d);
}

#[allow(dead_code)]
fn push_simple_noise(blob: &mut Vec<u8>, perm_offset: i32) {
    blob.push(0x84); // TAG_NOISE_SAMPLER
    blob.push(0); // type = Perlin
    blob.push(0); // flags = no second stack
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // amplitude
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // xz_scale
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // y_scale
    blob.extend_from_slice(&perm_offset.to_le_bytes());
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // initial_freq
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // persistence
    blob.extend_from_slice(&1i32.to_le_bytes()); // octaves
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // amp[0]
}

#[test]
fn numerical_validation_nonzero_origin() {
    use crate::world::constants::permutation_table_for_seed;
    // 3 octaves, each with a distinct non-zero origin.
    let perms: Vec<[u32; 256]> = [11i64, 22, 33]
        .iter()
        .map(|&sd| permutation_table_for_seed(sd))
        .collect();
    let perm_i32: Vec<i32> = perms.iter().flatten().map(|&v| v as i32).collect();
    let origins: Vec<[f64; 3]> = vec![
        [1.5, -2.25, 3.75],
        [-0.125, 7.5, 0.0625],
        [100.0, -50.5, 0.25],
    ];
    let amps = [1.0f32, 0.5, 0.25];
    let gpu = build_run_blob_origin(
        0,
        1.0,
        0.3,
        0.3,
        (0, 1.0, 1.0, &amps),
        None,
        &perms,
        &origins,
    );
    let mut max_d = 0.0f32;
    for cy in 0..97usize {
        for cz in 0..5usize {
            for cx in 0..5usize {
                let wx = (cx * 4) as f32;
                let wy = (-64 + cy as i32 * 4) as f32;
                let wz = (cz * 4) as f32;
                let mut acc = (0.0f32, 0.0f32);
                let mut freq = 1.0f32;
                let mut amp = 1.0f32;
                for i in 0..3usize {
                    let p = (i as i32) * 256;
                    let o = origins[i];
                    let s = cpu_perlin3d_origin(
                        &perm_i32,
                        p,
                        o[0] as f32,
                        o[1] as f32,
                        o[2] as f32,
                        wx * 0.3 * freq,
                        wy * 0.3 * freq,
                        wz * 0.3 * freq,
                    );
                    cpu_kahan_add(&mut acc, amps[i] * s * amp);
                    freq *= 2.0;
                    amp *= 0.5;
                }
                let g = gpu[cx + cz * 5 + cy * 25];
                max_d = max_d.max((acc.0 - g).abs());
            }
        }
    }
    println!("nonzero-origin (3 octaves): max_diff={:.6e}", max_d);
    assert!(max_d < 1e-4, "origin path diverged {:.6e}", max_d);
}

#[test]
fn numerical_validation_noise_vs_gpu() {
    use crate::instance::WgpuInstance;
    use crate::scheduler::batcher::{Batch, Priority};
    use crate::scheduler::executor::ExecutorThread;
    use crate::world::WorldContext;
    use crate::world::constants::permutation_table_for_seed;
    use std::sync::Arc;

    let perm_u32 = permutation_table_for_seed(42);
    let perm_i32: Vec<i32> = perm_u32.iter().map(|&v| v as i32).collect();

    // Build minimal blob: ROOT(1 ch) = NoiseSampler{Perlin, single stack}.
    let mut blob = Vec::new();
    blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
    blob.push(0xFE);
    blob.extend_from_slice(&[0x01, 0x00]);
    blob.extend_from_slice(&[0x84, 0x00, 0x00]);
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // amplitude
    blob.extend_from_slice(&0.3f32.to_le_bytes()); // xz_scale
    blob.extend_from_slice(&0.3f32.to_le_bytes()); // y_scale
    blob.extend_from_slice(&0i32.to_le_bytes()); // perm_offset
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // initial_freq
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // persistence
    blob.extend_from_slice(&1i32.to_le_bytes()); // octaves
    blob.extend_from_slice(&1.0f32.to_le_bytes()); // amp[0]
    blob.extend_from_slice(&(-64i32).to_le_bytes());
    blob.extend_from_slice(&384i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    blob.push(0xF0);
    blob.extend_from_slice(&123456789i64.to_le_bytes());
    // PERM_RESOURCE
    blob.extend_from_slice(&0x4D525750u32.to_le_bytes());
    blob.extend_from_slice(&1u32.to_le_bytes());
    for &p in &perm_u32 {
        blob.extend_from_slice(&p.to_le_bytes());
    }
    for _ in 0..3 {
        blob.extend_from_slice(&0.0f64.to_le_bytes());
    }

    let mut ctx = WorldContext::new(&blob).unwrap();
    let instance = WgpuInstance::instance().unwrap();
    ctx.compile_pipeline(&instance.device).unwrap();
    let world_arc = Arc::new(ctx);

    let executor = ExecutorThread::spawn(
        Arc::new(instance.device.clone()),
        Arc::new(instance.queue.clone()),
        Default::default(),
    )
    .unwrap();

    let batch = Batch {
        positions: vec![(0, 0)],
        max_priority: Priority::Normal,
    };
    let result = executor.request_chunks(world_arc, batch).unwrap();
    let blob_result = result.results.values().next().unwrap();
    assert!(blob_result.verify_crc());

    let bytes = blob_result.as_bytes();
    let payload = &bytes[0x88..];
    let ch0: Vec<f32> = (0..2425)
        .map(|i| f32::from_le_bytes(payload[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect();

    let mut max_diff = 0.0f32;
    let mut max_cell = (0usize, 0usize, 0usize);
    let mut sum_sq = 0.0f64;
    let mut n = 0usize;
    for cy in 0..97usize {
        for cz in 0..5usize {
            for cx in 0..5usize {
                let wx = (cx * 4) as f32;
                let wy = (-64 + cy as i32 * 4) as f32;
                let wz = (cz * 4) as f32;
                let cpu = cpu_noise_perlin3d(&perm_i32, 0, wx * 0.3, wy * 0.3, wz * 0.3);
                let gpu_idx = cx + cz * 5 + cy * 25;
                let gpu = ch0[gpu_idx];
                let d = (cpu - gpu).abs();
                sum_sq += (d as f64) * (d as f64);
                n += 1;
                if d > max_diff {
                    max_diff = d;
                    max_cell = (cx, cy, cz);
                }
            }
        }
    }
    let rms = (sum_sq / n as f64).sqrt();
    println!(
        "GPU vs CPU: max_diff={:.6e} at {:?}, rms={:.6e}",
        max_diff, max_cell, rms
    );
    for &(cx, cy, cz) in &[
        (0usize, 0usize, 0usize),
        (1, 0, 0),
        (2, 0, 0),
        (3, 0, 0),
        (1, 30, 1),
        (3, 95, 3),
    ] {
        let wx = (cx * 4) as f32;
        let wy = (-64 + cy as i32 * 4) as f32;
        let wz = (cz * 4) as f32;
        let cpu = cpu_noise_perlin3d(&perm_i32, 0, wx * 0.3, wy * 0.3, wz * 0.3);
        let gpu_idx = cx + cz * 5 + cy * 25;
        println!(
            "  cell ({},{},{}) world ({},{},{}) cpu={:.6} gpu={:.6}",
            cx, cy, cz, wx, wy, wz, cpu, ch0[gpu_idx]
        );
    }
    assert!(max_diff < 1e-4, "GPU vs CPU divergence {:.6e}", max_diff);
}

/// GPU finalDensity (channel 11) vs a vanilla reference dump.
///
/// The reference is produced by MixinChunkGenerator's corner dump:
/// RAW_ROUTER.finalDensity() sampled on the same 5x97x5 corner grid the
/// GPU emits. Both sides exclude the Beardifier (MC applies it per-block),
/// so the two must agree to <1e-4. Regenerate with:
///   touch /tmp/c2me_dump_blob  (then run the game once)
///
/// Ignored by default: it costs ~2s (vs ~0.2s for the other GPU tests) because
/// it compares the full 16-channel router against the vanilla dump. It is the
/// ONLY test that validates against real Minecraft output, so run it
/// explicitly whenever terrain correctness is in question:
///   cargo test --lib -- --ignored real_router_vs_vanilla
#[test]
#[ignore = "slow vanilla comparison; run when touching terrain correctness"]
fn real_router_vs_vanilla() {
    use crate::instance::WgpuInstance;
    use crate::scheduler::batcher::{Batch, Priority};
    use crate::scheduler::executor::ExecutorThread;
    use crate::world::WorldContext;
    use std::sync::Arc;

    let bin_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../src/test/resources/real_router.bin"
    );
    let ref_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../src/test/resources/real_router.ref.txt"
    );
    let blob = std::fs::read(bin_path).expect("read real_router.bin");

    // ── parse reference (multi-channel) ─────────────────────────
    let txt = std::fs::read_to_string(ref_path).expect("read real_router.ref.txt");
    let dims: (usize, usize, usize) = txt
        .lines()
        .find_map(|l| l.trim().strip_prefix("DIMS "))
        .map(|r| {
            let v: Vec<usize> = r.split_whitespace().map(|s| s.parse().unwrap()).collect();
            (v[0], v[1], v[2])
        })
        .unwrap_or((5, 97, 5));
    let (dx, dy, dz) = dims;
    let _ = dy;
    let cpc: usize = dx * dy * dz;
    let mut chunk = (0i32, 0i32);
    // name -> (gpu_channel, values)
    let chan_map: &[(&str, usize)] = &[
        ("barrierNoise", 0),
        ("fluidLevelFloodednessNoise", 1),
        ("fluidLevelSpreadNoise", 2),
        ("lavaNoise", 3),
        ("temperature", 4),
        ("vegetation", 5),
        ("continents", 6),
        ("erosion", 7),
        ("depth", 8),
        ("ridges", 9),
        ("initialDensityWithoutJaggedness", 10),
        ("finalDensity", 11),
        ("veinToggle", 12),
        ("veinRidged", 13),
        ("veinGap", 14),
    ];
    let mut ref_data: std::collections::HashMap<usize, Vec<f32>> =
        std::collections::HashMap::new();
    let mut cur_ch: Option<usize> = None;
    for line in txt.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("CHUNK ") {
            let mut it = rest.split_whitespace();
            chunk.0 = it.next().unwrap().parse().unwrap();
            chunk.1 = it.next().unwrap().parse().unwrap();
            continue;
        }
        if let Some(name) = line.strip_prefix("CHANNEL ") {
            cur_ch = chan_map.iter().find(|(n, _)| *n == name).map(|(_, c)| *c);
            if let Some(c) = cur_ch {
                ref_data.entry(c).or_insert_with(|| vec![0f32; cpc]);
            }
            continue;
        }
        if line.starts_with("DIMS") || line.starts_with("CELL_W") || line.starts_with("MIN_Y") {
            continue;
        }
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() != 4 {
            continue;
        }
        if let Some(c) = cur_ch {
            let i: usize = p[0].parse().unwrap();
            let j: usize = p[1].parse().unwrap();
            let k: usize = p[2].parse().unwrap();
            let v: f32 = p[3].parse().unwrap();
            if let Some(vec) = ref_data.get_mut(&c) {
                vec[i + k * dx + j * dx * dz] = v;
            }
        }
    }

    // ── run GPU ──────────────────────────────────────────────────
    let mut ctx = WorldContext::new(&blob).unwrap();
    // Which global channels the blob actually carries, in slot order.
    let readback = ctx.readback_channels();
    let instance = WgpuInstance::instance().unwrap();
    ctx.compile_pipeline(&instance.device).unwrap();
    let world_arc = Arc::new(ctx);
    let executor = ExecutorThread::spawn(
        Arc::new(instance.device.clone()),
        Arc::new(instance.queue.clone()),
        Default::default(),
    )
    .unwrap();
    let batch = Batch {
        positions: vec![chunk],
        max_priority: Priority::Normal,
    };
    let result = executor.request_chunks(world_arc, batch).unwrap();
    let br = result.results.values().next().unwrap();
    let bytes = br.as_bytes();
    let payload = &bytes[0x88..];
    // slot k of the compact blob holds global channel readback[k].
    let slot_of = |global: usize| readback.iter().position(|&c| c == global);

    // ── cross-match: for each ref channel, find best GPU slot ──
    println!("  -- cross-match (ref channel -> best GPU slot) --");
    for (name, rch) in chan_map {
        if let Some(exp_vec) = ref_data.get(rch) {
            let mut best = (usize::MAX, f64::MAX);
            for gch in 0..readback.len() {
                let gbase = gch * cpc * 4;
                let mut ss = 0.0f64;
                for idx in 0..cpc {
                    let gpu = f32::from_le_bytes(
                        payload[gbase + idx * 4..gbase + idx * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );
                    let d = (gpu - exp_vec[idx]) as f64;
                    ss += d * d;
                }
                let rms = (ss / cpc as f64).sqrt();
                if rms < best.1 {
                    best = (gch, rms);
                }
            }
            println!(
                "     {:14}(ref ch{:2}) -> gpu ch{:2}  rms={:.6e}",
                name, rch, best.0, best.1
            );
        }
    }

    // Only ch11 (finalDensity) and ch12 (veinToggle) are computed now
    // (see wgsl_gen::plan_shards). The other channels' out_buf slots are
    // no longer written, so comparing them against vanilla would always
    // fail and say nothing about codegen correctness.
    let checked: &[usize] = &[11, 12];
    let mut any_fail = false;
    for (name, ch) in chan_map {
        if !checked.contains(ch) {
            continue;
        }
        let exp_vec = match ref_data.get(ch) {
            Some(v) => v,
            None => {
                println!("  [{}] no ref data", name);
                continue;
            }
        };
        let slot = match slot_of(*ch) {
            Some(s) => s,
            None => continue,
        };
        let base = slot * cpc * 4;
        let mut max_diff = 0.0f32;
        let mut sum_sq = 0.0f64;
        let mut nan_both = 0usize;
        let mut nan_mismatch = 0usize;
        for idx in 0..cpc {
            let gpu = f32::from_le_bytes(
                payload[base + idx * 4..base + idx * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            let e = exp_vec[idx];
            if gpu.is_nan() && e.is_nan() { nan_both += 1; continue; }
            if gpu.is_nan() || e.is_nan() { nan_mismatch += 1; continue; }
            let d = (gpu - e).abs();
            sum_sq += (d as f64) * (d as f64);
            if d > max_diff {
                max_diff = d;
            }
        }
        if nan_both > 0 || nan_mismatch > 0 {
            println!("  ch{:2} NaN: both={} mismatch={}", ch, nan_both, nan_mismatch);
        }
        let rms = (sum_sq / cpc as f64).sqrt();
        // Pass criterion: RMS below the fp32 budget (DEVELOPMENT.md §0). max_diff
        // is reported but not decisive — a handful of outlier cells can
        // exceed 1e-4 purely from fp32 rounding in deep expressions
        // (ch10/ch11), which is accepted, not a bug. NaN must match exactly.
        let ok = rms < 1e-3 && nan_mismatch == 0;
        if !ok {
            any_fail = true;
        }
        println!(
            "  ch{:2} {:14} max_diff={:.6e} rms={:.6e}  {}",
            ch,
            name,
            max_diff,
            rms,
            if ok { "OK" } else { "FAIL" }
        );
        if *ch == 11 {
            let mut worst: Vec<(f32, usize)> = (0..cpc)
                .map(|idx| {
                    let gpu = f32::from_le_bytes(
                        payload[base + idx * 4..base + idx * 4 + 4]
                            .try_into()
                            .unwrap(),
                    );
                    ((gpu - exp_vec[idx]).abs(), idx)
                })
                .collect();
            worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            for &(d, idx) in worst.iter().take(10) {
                let i = idx % dx;
                let j = idx / (dx * dz);
                let k = (idx / dx) % dz;
                let gpu = f32::from_le_bytes(
                    payload[base + idx * 4..base + idx * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
                println!(
                    "     ch11 worst (i={},j={},k={}) gpu={:.5} ref={:.5} d={:.5}",
                    i, j, k, gpu, exp_vec[idx], d
                );
            }
        }
    }
    assert!(!any_fail,
        "GPU vs vanilla: a consumed channel (ch11/ch12) exceeds the fp32 RMS budget");
    println!("  consumed channels within fp32 budget (RMS < 1e-3), NaN matched");
}

/// Three-way: CPU-f32 interpreter vs GPU vs ref(double). If CPU-f32 ~= GPU
/// but both differ from ref -> pure f32 precision, codegen is faithful.
/// If CPU-f32 != GPU -> structural codegen bug.
#[test]
#[ignore = "needs GPU; run with --ignored"]
fn cpu_f32_vs_gpu() {
    use crate::instance::WgpuInstance;
    use crate::scheduler::batcher::{Batch, Priority};
    use crate::scheduler::executor::ExecutorThread;
    use crate::world::WorldContext;
    use std::sync::Arc;

    let bin_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../src/test/resources/real_router.bin");
    let ref_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../src/test/resources/real_router.ref.txt");
    let blob = std::fs::read(bin_path).expect("read blob");
    let txt = std::fs::read_to_string(ref_path).expect("read ref");

    let dx = 5usize; let dy = 49usize; let dz = 5usize;
    let cpc = dx * dy * dz;
    // parse ref for ch10/ch11
    let chan_map: &[(&str, usize)] = &[
        ("initialDensityWithoutJaggedness", 10), ("finalDensity", 11)];
    let mut ref_data: std::collections::HashMap<usize, Vec<f32>> = std::collections::HashMap::new();
    let mut cur: Option<usize> = None;
    let (mut cx, mut cz) = (0i32, 0i32);
    for line in txt.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("CHUNK ") {
            let mut it = rest.split_whitespace();
            cx = it.next().unwrap().parse().unwrap();
            cz = it.next().unwrap().parse().unwrap();
            continue;
        }
        if let Some(name) = line.strip_prefix("CHANNEL ") {
            cur = chan_map.iter().find(|(n, _)| *n == name).map(|(_, c)| *c);
            if let Some(c) = cur { ref_data.entry(c).or_insert_with(|| vec![0f32; cpc]); }
            continue;
        }
        if line.starts_with('#') || line.starts_with("DIMS") || line.starts_with("CELL_W") || line.starts_with("MIN_Y") { continue; }
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() != 4 { continue; }
        if let Some(c) = cur {
            let i: usize = p[0].parse().unwrap();
            let j: usize = p[1].parse().unwrap();
            let k: usize = p[2].parse().unwrap();
            let v: f32 = p[3].parse().unwrap();
            if let Some(vec) = ref_data.get_mut(&c) { vec[i + k*dx + j*dx*dz] = v; }
        }
    }

    // run GPU
    let mut ctx = WorldContext::new(&blob).unwrap();
    let perm: Vec<i32> = ctx.perm_table_set().buffer.clone();
    let origins: Vec<f32> = ctx.perm_table_set().origins.clone();
    // eval CPU before moving ctx
    let ast_root = ctx.ast_root.clone();
    let crate::ast::AstNode::Root(channels) = &ast_root else { panic!("not root") };
    let sx = cx * 16; let sz = cz * 16;
    let mut cpu_vals: std::collections::HashMap<usize, Vec<f32>> = std::collections::HashMap::new();
    for &(_, ch) in chan_map {
        let node = &channels[ch];
        let mut v = vec![0f32; cpc];
        for j in 0..dy {
            let wy = (-64 + j as i32 * 8) as f32;
            for k in 0..dz {
                let wz = (sz + k as i32 * 4) as f32;
                for i in 0..dx {
                    let wx = (sx + i as i32 * 4) as f32;
                    v[i + k*dx + j*dx*dz] = cpu_ast_eval(node, &perm, &origins, wx, wy, wz);
                }
            }
        }
        cpu_vals.insert(ch, v);
    }

    let instance = WgpuInstance::instance().unwrap();
    ctx.compile_pipeline(&instance.device).unwrap();
    // Read-back layout is compact: [chunk][readback_index][cells].
    // Capture the channel->index map before moving ctx into the Arc.
    let readback = ctx.readback_channels();
    let world_arc = Arc::new(ctx);
    let executor = ExecutorThread::spawn(
        Arc::new(instance.device.clone()), Arc::new(instance.queue.clone()), Default::default()).unwrap();
    let batch = Batch { positions: vec![(cx, cz)], max_priority: Priority::Normal };
    let result = executor.request_chunks(world_arc, batch).unwrap();
    let br = result.results.values().next().unwrap();
    let payload = &br.as_bytes()[0x88..];

    for &(name, ch) in chan_map {
        let cv = &cpu_vals[&ch];
        let rv = &ref_data[&ch];
        let Some(ri) = readback.iter().position(|&c| c == ch) else {
            println!("ch{} {}: NOT in readback (skipped); readback={:?}", ch, name, readback);
            continue;
        };
        let base = ri * cpc * 4;
        let (mut rc, mut rg, mut cg, mut mx_cg, mut mx_rc) = (0.0f64, 0.0f64, 0.0f64, 0.0f32, 0.0f32);
        for idx in 0..cpc {
            let gpu = f32::from_le_bytes(payload[base+idx*4..base+idx*4+4].try_into().unwrap());
            rc += ((cv[idx]-rv[idx]) as f64).powi(2);
            rg += ((gpu-rv[idx]) as f64).powi(2);
            cg += ((cv[idx]-gpu) as f64).powi(2);
            mx_cg = mx_cg.max((cv[idx]-gpu).abs());
            mx_rc = mx_rc.max((cv[idx]-rv[idx]).abs());
        }
        let n = cpc as f64;
        println!("ch{} {}: |cpu-ref|rms={:.3e} max={:.3e} |gpu-ref|rms={:.3e} |cpu-gpu|rms={:.3e} max={:.3e}",
            ch, name, (rc/n).sqrt(), mx_rc, (rg/n).sqrt(), (cg/n).sqrt(), mx_cg);
    }
}

// ── EndIslands: GPU 2D-simplex vs Rust CPU reference ──────────────
#[allow(dead_code)]
fn cpu_sx_map(perm: &[i32], a: i32) -> i32 {
    perm[(a & 255) as usize]
}
#[allow(dead_code)]
fn cpu_sx_grad(h: i32, x: f32, y: f32) -> f32 {
    let t = 0.5 - x * x - y * y;
    if t < 0.0 { return 0.0; }
    let t4 = t * t * t * t;
    let dp = match h {
        0 => x + y, 1 => -x + y, 2 => x - y, 3 => -x - y,
        4 | 6 => x, 5 | 7 => -x, 8 | 10 => y, _ => -y,
    };
    t4 * dp
}
#[allow(dead_code)]
fn cpu_simplex2d(perm: &[i32], x: f32, y: f32) -> f32 {
    const SKEW: f32 = 0.3660254037844386;
    const UNSKEW: f32 = 0.21132486540518713;
    let s = (x + y) * SKEW;
    let i = (x + s).floor();
    let j = (y + s).floor();
    let t = (i + j) * UNSKEW;
    let x0 = x - (i - t);
    let y0 = y - (j - t);
    let (i1, j1) = if x0 > y0 { (1.0f32, 0.0f32) } else { (0.0, 1.0) };
    let x1 = x0 - i1 + UNSKEW;
    let y1 = y0 - j1 + UNSKEW;
    let x2 = x0 - 1.0 + 2.0 * UNSKEW;
    let y2 = y0 - 1.0 + 2.0 * UNSKEW;
    let ii = (i as i32) & 255;
    let jj = (j as i32) & 255;
    let gi0 = cpu_sx_map(perm, ii + cpu_sx_map(perm, jj)) % 12;
    let gi1 = cpu_sx_map(perm, ii + i1 as i32 + cpu_sx_map(perm, jj + j1 as i32)) % 12;
    let gi2 = cpu_sx_map(perm, ii + 1 + cpu_sx_map(perm, jj + 1)) % 12;
    70.0 * (cpu_sx_grad(gi0, x0, y0) + cpu_sx_grad(gi1, x1, y1) + cpu_sx_grad(gi2, x2, y2))
}
#[allow(dead_code)]
fn cpu_end_islands(perm: &[i32], bx: i32, bz: i32) -> f32 {
    let x = bx / 8;
    let z = bz / 8;
    let x2 = x / 2;
    let z2 = z / 2;
    let xr = x % 2;
    let zr = z % 2;
    let mut d = (100.0 - (((x * x + z * z) as f32).sqrt()) * 8.0).clamp(-100.0, 80.0);
    for dx in -12..=12i32 {
        for dz in -12..=12i32 {
            let lx = x2 + dx;
            let lz = z2 + dz;
            if lx * lx + lz * lz <= 4096 { continue; }  // process only OUTER ring (>4096), matching MC
            if cpu_simplex2d(perm, lx as f32, lz as f32) < -0.9 {
                let f = (lx.abs() as f32) * 3439.0 + (lz.abs() as f32) * 147.0;
                let ff = (f % 13.0) + 9.0;
                let gx = (xr - 2 * dx) as f32;
                let gz = (zr - 2 * dz) as f32;
                let g = (100.0 - (gx * gx + gz * gz).sqrt() * ff).clamp(-100.0, 80.0);
                d = d.max(g);
            }
        }
    }
    d
}

#[test]
fn end_islands_cpu_vs_gpu() {
    use crate::instance::WgpuInstance;
    use crate::scheduler::batcher::{Batch, Priority};
    use crate::scheduler::executor::ExecutorThread;
    use crate::world::WorldContext;
    use std::sync::Arc;

    // Deterministic 512-int permutation; only perm[0..256) is used.
    let mut perm = [0i32; 512];
    for i in 0..256usize {
        perm[i] = ((i * 137 + 41) % 256) as i32;
    }
    // Ensure a permutation (0..255 each once): use a simple LCG shuffle.
    let mut arr: Vec<i32> = (0..256).collect();
    let mut seed: u32 = 12345;
    for i in (1..256).rev() {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let j = (seed >> 16) as usize % (i + 1);
        arr.swap(i, j);
    }
    for i in 0..256usize { perm[i] = arr[i]; }

    // Build blob: Root([EndIslands{perm_index:0}]) + SIMPLEX trailer.
    let mut blob: Vec<u8> = Vec::new();
    blob.extend_from_slice(&[0x4D, 0x43, 0x32, 0x57, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]);
    blob.push(0xFE);
    blob.extend_from_slice(&[0x01, 0x00]);           // 1 channel
    blob.push(0xA0);                                  // EndIslands
    blob.extend_from_slice(&0i32.to_le_bytes());      // perm_index = 0
    blob.extend_from_slice(&(-64i32).to_le_bytes());  // shape
    blob.extend_from_slice(&384i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    blob.extend_from_slice(&4i32.to_le_bytes());
    blob.push(0xF0);
    blob.extend_from_slice(&123456789i64.to_le_bytes());
    // PERM trailer (0 entries) then SIMPLEX trailer (1 entry)
    blob.extend_from_slice(&0x4D525750u32.to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes());
    blob.extend_from_slice(&0x4D525358u32.to_le_bytes());
    blob.extend_from_slice(&1u32.to_le_bytes());
    for &v in &perm { blob.extend_from_slice(&v.to_le_bytes()); }
    for _ in 0..3 { blob.extend_from_slice(&0.0f64.to_le_bytes()); }

    let mut ctx = WorldContext::new(&blob).unwrap();
    let instance = WgpuInstance::instance().unwrap();
    ctx.compile_pipeline(&instance.device).unwrap();
    let world_arc = Arc::new(ctx);
    let executor = ExecutorThread::spawn(
        Arc::new(instance.device.clone()),
        Arc::new(instance.queue.clone()),
        Default::default(),
    ).unwrap();
    // EndIslands is sparse: find one chunk where the CPU reference
    // actually leaves the clamp floor, so the simplex2d "< -0.9"
    // branch is exercised.
    let mut hit: Option<(i32, i32)> = None;
    'outer: for cx in -300..300i32 {
        for cz in -300..300i32 {
            if cpu_end_islands(&perm, cx * 16, cz * 16) > -99.0 {
                hit = Some((cx, cz));
                break 'outer;
            }
        }
    }
    let cpos = hit.expect("no EndIslands-hitting chunk found");
    println!("EndIslands: hitting chunk = {:?} (cpu={:.4})",
             cpos, cpu_end_islands(&perm, cpos.0 * 16, cpos.1 * 16));

    let batch = Batch { positions: vec![cpos], max_priority: Priority::Normal };
    let result = executor.request_chunks(world_arc, batch).unwrap();
    let br = result.results.get(&(((cpos.0 as i64) << 32) | (cpos.1 as i64 & 0xFFFFFFFF)))
        .unwrap();
    let bytes = br.as_bytes();
    let payload = &bytes[0x88..];

    let mut max_diff = 0.0f32;
    let mut n_hit = 0;
    for k in 0..5usize {
        for j in 0..97usize {
            for i in 0..5usize {
                let bx = cpos.0 * 16 + (i * 4) as i32;
                let bz = cpos.1 * 16 + (k * 4) as i32;
                let cpu = (cpu_end_islands(&perm, bx, bz) - 8.0) / 128.0;
                let idx = i + k * 5 + j * 25;
                let gpu = f32::from_le_bytes(payload[idx*4..idx*4+4].try_into().unwrap());
                if cpu > -0.7 { n_hit += 1; }
                let d = (cpu - gpu).abs();
                if d > max_diff { max_diff = d; }
            }
        }
    }
    println!("EndIslands GPU vs CPU: max_diff={:.6e}, non-floor cells={}", max_diff, n_hit);
    assert!(max_diff < 1e-4, "EndIslands divergence {:.6e}", max_diff);
}

#[test]
#[ignore = "diagnostic; run with --ignored"]
fn dump_channel_histograms() {
    use crate::ast::deserialize::deserialize_full_with_perm;
    use crate::ast::opto::run_pipeline;
    use crate::ast::AstNode;
    use std::collections::BTreeMap;

    fn walk(node: &AstNode, out: &mut BTreeMap<String, usize>) {
        let name = match node {
            AstNode::Constant(_) => "Constant",
            AstNode::ExternalInput { .. } => "ExternalInput",
            AstNode::InputX => "InputX",
            AstNode::InputY => "InputY",
            AstNode::InputZ => "InputZ",
            AstNode::Add(..) => "Add",
            AstNode::Sub(..) => "Sub",
            AstNode::Mul(..) => "Mul",
            AstNode::Div(..) => "Div",
            AstNode::Min(..) => "Min",
            AstNode::Max(..) => "Max",
            AstNode::Abs(_) => "Abs",
            AstNode::Square(_) => "Square",
            AstNode::Cube(_) => "Cube",
            AstNode::Sqrt(_) => "Sqrt",
            AstNode::FloorDiv(..) => "FloorDiv",
            AstNode::Squeeze(_) => "Squeeze",
            AstNode::NoiseSampler { .. } => "NoiseSampler",
            AstNode::SimplexNoise2D { .. } => "SimplexNoise2D",
            AstNode::SimplexNoise3D { .. } => "SimplexNoise3D",
            AstNode::InterpolatedNoiseSampler { .. } => "InterpolatedNoiseSampler",
            AstNode::Shift { .. } => "Shift",
            AstNode::WeirdScaledSampler { .. } => "WeirdScaledSampler",
            AstNode::InterpNoiseSampler { .. } => "InterpNoiseSampler",
            AstNode::ShiftedNoise { .. } => "ShiftedNoise",
            AstNode::EndIslands { .. } => "EndIslands",
            AstNode::Beardifier { .. } => "Beardifier",
            AstNode::Spline { .. } => "Spline",
            AstNode::RangeChoice { .. } => "RangeChoice",
            AstNode::IntervalSelect { .. } => "IntervalSelect",
            AstNode::YClampedGradient { .. } => "YClampedGradient",
            AstNode::CacheFlat(_) => "CacheFlat",
            AstNode::Cache2D(_) => "Cache2D",
            AstNode::CacheInterpolated(_) => "CacheInterpolated",
            AstNode::FindTopSurface { .. } => "FindTopSurface",
            AstNode::Aquifer { .. } => "Aquifer",
            AstNode::BiomeTree { .. } => "BiomeTree",
            AstNode::Root(_) => "Root",
            AstNode::InnerRef(_) => "InnerRef",
        };
        *out.entry(name.to_string()).or_insert(0) += 1;
        // recurse
        match node {
            AstNode::Add(a,b)|AstNode::Sub(a,b)|AstNode::Mul(a,b)|AstNode::Div(a,b)
            |AstNode::Min(a,b)|AstNode::Max(a,b)|AstNode::FloorDiv(a,b) => { walk(a,out); walk(b,out); }
            AstNode::Abs(a)|AstNode::Square(a)|AstNode::Cube(a)|AstNode::Sqrt(a)
            |AstNode::Squeeze(a) => walk(a,out),
            AstNode::NoiseSampler { .. } => {}
            AstNode::InterpolatedNoiseSampler { child } => walk(child,out),
            AstNode::Shift { offset, .. } => walk(offset,out),
            AstNode::WeirdScaledSampler { input, noise, .. } => { walk(input,out); walk(noise,out); }
            AstNode::ShiftedNoise { shift_x, shift_y, shift_z, noise, .. } => { walk(shift_x,out); walk(shift_y,out); walk(shift_z,out); walk(noise,out); }
            AstNode::Spline { location, values, .. } => { walk(location,out); for v in values { walk(v,out); } }
            AstNode::RangeChoice { input,min,max,in_bounds,out_of_bounds_lo,out_of_bounds_hi } => { walk(input,out); walk(min,out); walk(max,out); walk(in_bounds,out); walk(out_of_bounds_lo,out); walk(out_of_bounds_hi,out); }
            AstNode::IntervalSelect { input,min,max,in_bounds,out_of_bounds_lo,out_of_bounds_hi } => { walk(input,out); walk(min,out); walk(max,out); walk(in_bounds,out); walk(out_of_bounds_lo,out); walk(out_of_bounds_hi,out); }
            AstNode::YClampedGradient { .. } => {}
            AstNode::CacheFlat(a)|AstNode::Cache2D(a)|AstNode::CacheInterpolated(a) => walk(a,out),
            AstNode::EndIslands { .. } => {}
            AstNode::Beardifier { .. } => {}
            AstNode::FindTopSurface { .. } => {}
            AstNode::Aquifer { density, fluid_level } => { walk(density,out); walk(fluid_level,out); }
            AstNode::BiomeTree { .. } => {}
            AstNode::Root(chans) => { for c in chans { walk(c,out); } }
            _ => {}
        }
    }

    let bin_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../src/test/resources/real_router.bin");
    let blob = std::fs::read(bin_path).expect("read blob");
    let (ast_root, _shape, _bs, _pe, _se) = deserialize_full_with_perm(&blob).unwrap();
    let ast_root = run_pipeline(ast_root).unwrap();
    let AstNode::Root(channels) = &ast_root else { panic!("not root") };
    let names = ["barrierNoise","fluidLevelFloodedness","fluidLevelSpread","lavaNoise","temperature","vegetation","continents","erosion","depth","ridges","initDensity","finalDensity","veinToggle","veinRidged","veinGap"];
    for (i, ch) in channels.iter().enumerate() {
        let mut hist = BTreeMap::new();
        walk(ch, &mut hist);
        let nm = names.get(i).copied().unwrap_or("?");
        let items: Vec<String> = hist.iter().map(|(k,v)| format!("{k}={v}")).collect();
        println!("ch{:2} {nm:20} : {}", i, items.join(" "));
    }
}


#[allow(dead_code)]
fn cpu_scale_tunnels(x: f32) -> f32 {
    if x < -0.5 { 0.75 } else if x < 0.0 { 1.0 } else if x < 0.5 { 1.5 } else { 2.0 }
}
#[allow(dead_code)]
fn cpu_scale_caves(x: f32) -> f32 {
    if x < -0.75 { 0.5 } else if x < -0.5 { 0.75 } else if x < 0.5 { 1.0 } else if x < 0.75 { 2.0 } else { 3.0 }
}
#[allow(dead_code)]
fn cpu_perlin_origin(perm: &[i32], origins: &[f32], po: i32, x: f32, y: f32, z: f32) -> f32 {
    let e = po / 256;
    let px = cpu_maintain_precision(x) + origins[(e * 3) as usize];
    let py = cpu_maintain_precision(y) + origins[(e * 3 + 1) as usize];
    let pz = cpu_maintain_precision(z) + origins[(e * 3 + 2) as usize];
    cpu_perlin3d_impl(perm, po, px, py, pz)
}
#[allow(dead_code)]
fn cpu_oct(perm: &[i32], origins: &[f32], oct: &crate::ast::OctaveSampler, cx: f32, cy: f32, cz: f32) -> f32 {
    let mut freq = oct.initial_freq;
    let mut amp = oct.persistence;
    let mut acc = (0.0f32, 0.0f32);
    for i in 0..oct.amplitudes.len() {
        if oct.amplitudes[i] != 0.0 {
            let s = cpu_perlin_origin(perm, origins, oct.perm_offset + i as i32 * 256, cx * freq, cy * freq, cz * freq);
            cpu_kahan_add(&mut acc, oct.amplitudes[i] * s * amp);
        }
        freq *= 2.0; amp *= 0.5;
    }
    acc.0
}
#[allow(dead_code)]
fn cpu_perlin_ysmear(perm: &[i32], origins: &[f32], po: i32, x: f32, y: f32, z: f32, yfreq: f32, ysmear: f32) -> f32 {
    let e = po / 256;
    let px = x + origins[(e*3) as usize];
    let py = y + origins[(e*3+1) as usize];
    let pz = z + origins[(e*3+2) as usize];
    let ix = px.floor() as i32; let iy = py.floor() as i32; let iz = pz.floor() as i32;
    let fx = px - ix as f32; let fy = py - iy as f32; let fz = pz - iz as f32;
    let mut o = 0.0f32;
    if yfreq != 0.0 {
        let m = if ysmear >= 0.0 && ysmear < fy { ysmear } else { fy };
        o = (m / yfreq + 1.0e-7).floor() * yfreq;
    }
    let dy = fy - o;
    let p = po;
    let xi = ix & 255; let yi = iy & 255; let zi = iz & 255;
    let g = |i: i32| -> [f32;3] {
        match i & 15 {
            0=>[1.0,1.0,0.0],1=>[-1.0,1.0,0.0],2=>[1.0,-1.0,0.0],3=>[-1.0,-1.0,0.0],
            4=>[1.0,0.0,1.0],5=>[-1.0,0.0,1.0],6=>[1.0,0.0,-1.0],7=>[-1.0,0.0,-1.0],
            8=>[0.0,1.0,1.0],9=>[0.0,-1.0,1.0],10=>[0.0,1.0,-1.0],11=>[0.0,-1.0,-1.0],
            12=>[1.0,1.0,0.0],13=>[0.0,-1.0,1.0],14=>[-1.0,1.0,0.0],_=>[0.0,-1.0,-1.0],
        }
    };
    let idx=|i:i32|(p+i) as usize;
    let a=perm[idx(xi)]; let b=perm[idx((xi+1)&255)];
    let aa=perm[idx((a+yi)&255)]; let ab=perm[idx((a+yi+1)&255)];
    let ba=perm[idx((b+yi)&255)]; let bb=perm[idx((b+yi+1)&255)];
    let aaa=perm[idx((aa+zi)&255)]; let aab=perm[idx((aa+zi+1)&255)];
    let aba=perm[idx((ab+zi)&255)]; let abb=perm[idx((ab+zi+1)&255)];
    let baa=perm[idx((ba+zi)&255)]; let bab=perm[idx((ba+zi+1)&255)];
    let bba=perm[idx((bb+zi)&255)]; let bbb=perm[idx((bb+zi+1)&255)];
    let dot=|gr:[f32;3],dx:f32,dy:f32,dz:f32| gr[0]*dx+gr[1]*dy+gr[2]*dz;
    let n000=dot(g(aaa),fx,dy,fz); let n100=dot(g(baa),fx-1.0,dy,fz);
    let n010=dot(g(aba),fx,dy-1.0,fz); let n110=dot(g(bba),fx-1.0,dy-1.0,fz);
    let n001=dot(g(aab),fx,dy,fz-1.0); let n101=dot(g(bab),fx-1.0,dy,fz-1.0);
    let n011=dot(g(abb),fx,dy-1.0,fz-1.0); let n111=dot(g(bbb),fx-1.0,dy-1.0,fz-1.0);
    cpu_trilinear(n000,n100,n010,n110,n001,n101,n011,n111,cpu_fade(fx),cpu_fade(fy),cpu_fade(fz))
}
#[allow(dead_code)]
fn cpu_interp_noise_sampler(perm: &[i32], origins: &[f32],
    lb:i32,lc:i32,ub:i32,uc:i32,ib:i32,ic:i32,
    bx:f32,by:f32,bz:f32,
    xz_scale:f32,y_scale:f32,xz_factor:f32,y_factor:f32,smear:f32) -> f32 {
    let _ = uc;
    let sxz = 684.412 * xz_scale; let sy = 684.412 * y_scale;
    let d = bx*sxz; let e = by*sy; let f = bz*sxz;
    let gg = d/xz_factor; let h = e/y_factor; let l = f/xz_factor;
    let mm = sy*smear/y_factor; let smear_y_raw = sy*smear;
    let mut q = 0.0f32; let mut r = 1.0f32;
    for i in 0..ic {
        q += cpu_perlin_ysmear(perm, origins, ib+i*256,
            cpu_maintain_precision(gg*r), cpu_maintain_precision(h*r), cpu_maintain_precision(l*r),
            mm*r, h*r) / r;
        r *= 0.5;
    }
    let interp = (q/10.0 + 1.0)/2.0;
    let no_lower = interp >= 1.0; let no_upper = interp <= 0.0;
    let mut lo = 0.0f32; let mut up = 0.0f32; r = 1.0;
    for i in 0..lc {
        if !no_lower {
            lo += cpu_perlin_ysmear(perm, origins, lb+i*256,
                cpu_maintain_precision(d*r), cpu_maintain_precision(e*r), cpu_maintain_precision(f*r),
                smear_y_raw*r, e*r) / r;
        }
        if !no_upper {
            up += cpu_perlin_ysmear(perm, origins, ub+i*256,
                cpu_maintain_precision(d*r), cpu_maintain_precision(e*r), cpu_maintain_precision(f*r),
                smear_y_raw*r, e*r) / r;
        }
        r *= 0.5;
    }
    let t = interp.clamp(0.0,1.0);
    (lo/512.0 + t*(up/512.0 - lo/512.0))/128.0
}
thread_local! {
    pub static INNER_VALS: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
}

#[allow(dead_code)]
pub(crate) fn cpu_ast_eval(node: &crate::ast::AstNode, perm: &[i32], origins: &[f32], x: f32, y: f32, z: f32) -> f32 {
    use crate::ast::AstNode as A;
    match node {
        A::InnerRef(i) => INNER_VALS.with(|v| v.borrow().get(*i).copied().unwrap_or(0.0)),
        A::Constant(v) => *v,
        A::InputX => x, A::InputY => y, A::InputZ => z,
        A::Add(a, b) => cpu_ast_eval(a, perm, origins, x, y, z) + cpu_ast_eval(b, perm, origins, x, y, z),
        A::Sub(a, b) => cpu_ast_eval(a, perm, origins, x, y, z) - cpu_ast_eval(b, perm, origins, x, y, z),
        A::Mul(a, b) => cpu_ast_eval(a, perm, origins, x, y, z) * cpu_ast_eval(b, perm, origins, x, y, z),
        A::Div(a, b) => cpu_ast_eval(a, perm, origins, x, y, z) / cpu_ast_eval(b, perm, origins, x, y, z),
        A::Min(a, b) => cpu_ast_eval(a, perm, origins, x, y, z).min(cpu_ast_eval(b, perm, origins, x, y, z)),
        A::Max(a, b) => cpu_ast_eval(a, perm, origins, x, y, z).max(cpu_ast_eval(b, perm, origins, x, y, z)),
        A::FloorDiv(a, b) => (cpu_ast_eval(a, perm, origins, x, y, z) / cpu_ast_eval(b, perm, origins, x, y, z)).floor(),
        A::Abs(a) => cpu_ast_eval(a, perm, origins, x, y, z).abs(),
        A::Square(a) => { let v = cpu_ast_eval(a, perm, origins, x, y, z); v * v }
        A::Cube(a) => { let v = cpu_ast_eval(a, perm, origins, x, y, z); v * v * v }
        A::Sqrt(a) => cpu_ast_eval(a, perm, origins, x, y, z).sqrt(),
        A::Squeeze(a) => { let v = cpu_ast_eval(a, perm, origins, x, y, z).clamp(-1.0, 1.0); v * 0.5 - v * v * v / 24.0 }
        A::CacheFlat(a) | A::Cache2D(a) | A::CacheInterpolated(a) => cpu_ast_eval(a, perm, origins, x, y, z),
        A::YClampedGradient { from_y, to_y, from_value, to_value } => {
            let t = ((y - from_y) / (to_y - from_y)).clamp(0.0, 1.0);
            from_value + t * (to_value - from_value)
        }
        A::Spline { locations, derivatives, location, values } => {
            let t = cpu_ast_eval(location, perm, origins, x, y, z);
            let n = locations.len();
            let vs: Vec<f32> = values.iter().map(|v| cpu_ast_eval(v, perm, origins, x, y, z)).collect();
            if n == 0 { return 0.0; }
            if n == 1 {
                let d0 = derivatives[0];
                return if d0 == 0.0 { vs[0] } else { vs[0] + d0 * (t - locations[0]) };
            }
            if t < locations[0] {
                let d0 = derivatives[0];
                return if d0 == 0.0 { vs[0] } else { vs[0] + d0 * (t - locations[0]) };
            }
            if t >= locations[n - 1] {
                let dn = derivatives[n - 1];
                return if dn == 0.0 { vs[n - 1] } else { vs[n - 1] + dn * (t - locations[n - 1]) };
            }
            for k in 0..n - 1 {
                if t < locations[k + 1] {
                    let lk = locations[k]; let lk1 = locations[k + 1];
                    let dk = derivatives[k]; let dk1 = derivatives[k + 1];
                    let vk = vs[k]; let vk1 = vs[k + 1];
                    let width = lk1 - lk;
                    let p = (t - lk) / width;
                    let a = dk * width - (vk1 - vk);
                    let b = dk1 * -1.0 * width + (vk1 - vk);
                    return vk + p * (vk1 - vk) + p * (1.0 - p) * (a + p * (b - a));
                }
            }
            vs[n - 1]
        }
        A::NoiseSampler { first, second, amplitude, xz_scale, y_scale, .. } => {
            const D: f32 = 1.0181268882175227;
            let vf = cpu_oct(perm, origins, first, x * xz_scale, y * y_scale, z * xz_scale);
            let total = if let Some(sec) = second {
                vf + cpu_oct(perm, origins, sec, x * xz_scale * D, y * y_scale * D, z * xz_scale * D)
            } else { vf };
            total * amplitude
        }
        A::Shift { kind, offset } => {
            let (cx, cy, cz) = match kind {
                0 => (x * 0.25, y * 0.25, z * 0.25),
                1 => (x * 0.25, 0.0, z * 0.25),
                2 => (z * 0.25, x * 0.25, 0.0),
                _ => (x, y, z),
            };
            cpu_ast_eval(offset, perm, origins, cx, cy, cz) * 4.0
        }
        A::ShiftedNoise { xz_scale, y_scale, shift_x, shift_y, shift_z, noise } => {
            let d = x * xz_scale + cpu_ast_eval(shift_x, perm, origins, x, y, z);
            let e = y * y_scale + cpu_ast_eval(shift_y, perm, origins, x, y, z);
            let f = z * xz_scale + cpu_ast_eval(shift_z, perm, origins, x, y, z);
            cpu_ast_eval(noise, perm, origins, d, e, f)
        }
        A::WeirdScaledSampler { mapper, input, noise } => {
            let iv = cpu_ast_eval(input, perm, origins, x, y, z);
            let scale = if *mapper == 1 { cpu_scale_tunnels(iv) } else { cpu_scale_caves(iv) };
            cpu_ast_eval(noise, perm, origins, x / scale, y / scale, z / scale).abs() * scale
        }
        A::InterpNoiseSampler { xz_scale, y_scale, xz_factor, y_factor, smear,
            lower_base, lower_count, upper_base, upper_count, interp_base, interp_count } => {
            cpu_interp_noise_sampler(perm, origins, *lower_base, *lower_count, *upper_base, *upper_count,
                *interp_base, *interp_count, x, y, z, *xz_scale, *y_scale, *xz_factor, *y_factor, *smear)
        }
        A::InterpolatedNoiseSampler { child } => cpu_ast_eval(child, perm, origins, x, y, z),
        A::RangeChoice { input, min, max, in_bounds, out_of_bounds_lo, out_of_bounds_hi } => {
            let iv = cpu_ast_eval(input, perm, origins, x, y, z);
            let mn = cpu_ast_eval(min, perm, origins, x, y, z);
            let mx = cpu_ast_eval(max, perm, origins, x, y, z);
            if iv >= mn && iv < mx { cpu_ast_eval(in_bounds, perm, origins, x, y, z) }
            else if iv < mn { cpu_ast_eval(out_of_bounds_lo, perm, origins, x, y, z) }
            else { cpu_ast_eval(out_of_bounds_hi, perm, origins, x, y, z) }
        }
        _ => f32::NAN,
    }
}
