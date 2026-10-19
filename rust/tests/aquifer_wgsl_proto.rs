//! §22 Step B 原型：aquifer 12-cell 扫描的 WGSL 实现（跳过 barrier 分支）。
//! 表从 /tmp/aquifer_0_0.txt 读，per-block 密度+参考从 /tmp/aq_blocks_0_0.txt。
//! 目标：验证控制流在 GPU 上可行，复现 CPU-f32 的 99.95%。

fn state_code(s: &str) -> u32 {
    let n = s.rsplit(':').next().unwrap_or(s);
    match n { "air" => 0, "stone" => 1, "water" => 2, "lava" => 3, _ => 0 }
}

#[test]
fn aquifer_wgsl_proto() {
    use c2me_wgpu::instance::WgpuInstance;
    let ap = "/tmp/aquifer_0_0.txt";
    let bp = "/tmp/aq_blocks_0_0.txt";
    if !std::path::Path::new(ap).exists() || !std::path::Path::new(bp).exists() {
        eprintln!("skip: dumps absent"); return;
    }
    // 解析 cell 表（315 项）：pos + lvl_y + state
    let mut cpx = vec![0i32; 315]; let mut cpy = vec![0i32; 315];
    let mut cpz = vec![0i32; 315]; let mut cly = vec![0i32; 315];
    let mut cls = vec![0u32; 315];
    for line in std::fs::read_to_string(ap).unwrap().lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() < 6 || t[0] == "CHUNK" || t[0] == "AQ" || t[0] == "SEED" { continue; }
        let idx: usize = t[0].parse().unwrap();
        if idx >= 315 { continue; }
        cpx[idx] = t[1].parse().unwrap(); cpy[idx] = t[2].parse().unwrap();
        cpz[idx] = t[3].parse().unwrap();
        cly[idx] = t[t.len()-2].parse().unwrap();
        cls[idx] = state_code(t[t.len()-1]);
    }
    // 解析 per-block 参考
    let n = 16u32 * 16u32 * 384u32;
    let mut din = vec![0f32; n as usize];
    let mut want = vec![0u32; n as usize];
    for line in std::fs::read_to_string(bp).unwrap().lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() != 5 { continue; }
        let x: i32 = t[0].parse().unwrap(); let y: i32 = t[1].parse().unwrap();
        let z: i32 = t[2].parse().unwrap(); let d: f32 = t[3].parse().unwrap();
        let idx = (x + z * 16 + (y + 64) * 256) as usize;
        din[idx] = d; want[idx] = state_code(&t[4]);
    }

    let inst = WgpuInstance::instance().unwrap();
    let device = &inst.device; let queue = &inst.queue;

    let src = r#"
@group(0) @binding(0) var<storage, read> cell_p: array<vec4<i32>>;
@group(0) @binding(1) var<storage, read> cell_s: array<u32>;
@group(0) @binding(2) var<storage, read> din: array<f32>;
@group(0) @binding(3) var<storage, read_write> sout: array<u32>;
const SX: i32 = 3; const SY: i32 = 35; const SZ: i32 = 3;
const SX0: i32 = -1; const SY0: i32 = -7; const SZ0: i32 = -1;
fn fdiv(a: i32, b: i32) -> i32 {
  let q = a / b; let r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) { return q - 1; }
  return q;
}
fn cidx(x: i32, y: i32, z: i32) -> i32 {
  let i = x - SX0; let j = y - SY0; let k = z - SZ0;
  if (i < 0 || j < 0 || k < 0 || i >= SX || k >= SZ || j >= SY) { return -1; }
  return (j * SZ + k) * SX + i;
}
@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let bx = i32(gid.x); let bz = i32(gid.y); let byi = i32(gid.z);
  if (bx >= 16 || bz >= 16 || byi >= 384) { return; }
  let by = byi - 64;
  let idx = bx + bz * 16 + byi * 256;
  let density = din[idx];
  if (density > 0.0) { sout[idx] = 1u; return; }
  if (by < -54) { sout[idx] = 3u; return; }
  let l = fdiv(bx - 5, 16); let m = fdiv(by + 1, 12); let n = fdiv(bz - 5, 16);
  var o: i32 = 2147483647; var p: i32 = 2147483647; var q: i32 = 2147483647;
  var ri: i32 = -1;
  for (var u: i32 = 0; u <= 1; u = u + 1) {
    for (var v: i32 = -1; v <= 1; v = v + 1) {
      for (var w: i32 = 0; w <= 1; w = w + 1) {
        let ci = cidx(l + u, m + v, n + w);
        if (ci < 0) { continue; }
        let cp = cell_p[ci];
        let ad = cp.x - bx; let ae = cp.y - by; let af = cp.z - bz;
        let ag = ad * ad + ae * ae + af * af;
        if (o >= ag) { ri = ci; q = p; p = o; o = ag; }
        else if (p >= ag) { q = p; p = ag; }
      }
    }
  }
  if (ri < 0) { sout[idx] = 1u; return; }
  let rcly = cell_p[ri].w;
  let dd = 1.0 - f32(abs(p - o)) / 25.0;
  var bs: u32 = select(0u, cell_s[ri], by < rcly);
  bs = bs + u32(dd * 0.0);
  sout[idx] = bs;
}
"#;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None, source: wgpu::ShaderSource::Wgsl(src.into()) });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[0u32, 1, 2, 3].iter().map(|&b| wgpu::BindGroupLayoutEntry {
            binding: b, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: b != 3 },
                has_dynamic_offset: false, min_binding_size: None }, count: None }).collect::<Vec<_>>(),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None, layout: Some(&pl), module: &shader, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None });
    // 打包 cell_p: vec4(px,py,pz,lvl_y)
    let mut cpv: Vec<[i32; 4]> = Vec::with_capacity(315);
    for i in 0..315 { cpv.push([cpx[i], cpy[i], cpz[i], cly[i]]); }
    let mk = |bytes: u64, ro: bool| device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
            | if ro { wgpu::BufferUsages::empty() } else { wgpu::BufferUsages::COPY_SRC },
        mapped_at_creation: false });
    let b0 = mk(315 * 16, true); let b1 = mk(315 * 4, true);
    let b2 = mk(n as u64 * 4, true); let b3 = mk(n as u64 * 4, false);
    queue.write_buffer(&b0, 0, bytemuck::cast_slice(&cpv));
    queue.write_buffer(&b1, 0, bytemuck::cast_slice(&cls));
    queue.write_buffer(&b2, 0, bytemuck::cast_slice(&din));
    let stg = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: n as u64 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: b0.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b1.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: b2.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: b3.as_entire_binding() },
        ]});
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    { let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
      cp.set_pipeline(&pipe); cp.set_bind_group(0, &bg, &[]);
      cp.dispatch_workgroups(2, 2, 96); }
    enc.copy_buffer_to_buffer(&b3, 0, &stg, 0, n as u64 * 4);
    queue.submit(Some(enc.finish()));
    let slice = stg.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    loop {
        match rx.try_recv() { Ok(_) => break, Err(std::sync::mpsc::TryRecvError::Disconnected) => break, _ => {} }
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None,
            timeout: Some(std::time::Duration::from_millis(100)) });
    }
    let data = slice.get_mapped_range().unwrap();
    let got: &[u32] = bytemuck::cast_slice(&data);
    let mut ok = 0usize; let mut bad = 0usize;
    for i in 0..n as usize {
        if got[i] == want[i] { ok += 1; } else { bad += 1; }
    }
    drop(data); stg.unmap();
    let rate = ok as f64 / n as f64;
    println!("aquifer WGSL proto: {ok}/{n} = {:.3}% match (bad={bad})", rate * 100.0);
    assert!(rate > 0.995, "匹配率过低: {:.3}%", rate * 100.0);
}
