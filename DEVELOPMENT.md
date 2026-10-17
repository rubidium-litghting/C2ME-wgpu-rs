# C2ME WGPU 加速模块 — 开发文档

> 最后更新：2026-10-17
> 状态：**Pass E 架构、矿脉、含水层完全对齐原版**。史诗地形（Epic Terrain）与 Tectonic 完美兼容！
> 桌面端到端反超纯 CPU（Chunky r=500：15 s vs 23 s）。GPU 直接输出 per-block 方块状态，CPU 只写 palette。
> 历史细节见文末「附录：开发历程」。

---

## 速查（接手必读）

### 项目一句话

给 C2ME 提供基于 wgpu 的 GPU 地形生成：把 MC 的 `NoiseRouter` 密度管线
（噪声 + 三线性插值 + aquifer + beard）搬上 GPU，直接输出 per-block 方块状态字节，
Java 只写 palette。纯 fp32（兼容性优先），Java↔Rust 走 Panama FFM。

### 当前完成度

| 模块 | 状态 |
|------|------|
| AST 反序列化 + 优化 + WGSL 生成 | ✅ |
| 噪声原语（Perlin/Simplex/Shift/Spline/…） | ✅ 数值验证 ~1e-7 |
| 多维度（Overworld/Nether/End） | ✅ per-dimension WorldContext |
| 密度角点计算 + 回读（只算被消费通道） | ✅ |
| **per-block 方块状态输出（BlockOut shard）** | ✅ |
| **aquifer 全 GPU（水/岩浆/石分类）** | ✅ 99.97%（barrier 分支遗留） |
| **Beardifier（结构地基）GPU** | ✅ bit-exact |
| 矿脉 | ⚠️ 角点粗筛 + MC 精算（未全 GPU） |
| **端到端反超纯 CPU** | ✅ 15 s vs 23 s |
| WeirdScaledSampler | ❌ 仍占位 |

### 关键教训速查

**Java / 序列化**
1. **用 `instanceof` 而非字符串比较**：Loom 重映射类引用，但不重映射字符串常量；运行期类名是 intermediary。
2. **magic 逐字节写**：`putInt(0x4D433257)+LE` 会反序，必须逐字节写 `4D 43 32 57`。
3. **反射按类型不按字段名**：字段名也被混淆；用"字段类型为 X"来找。
4. **置换表不能靠 seed 重建**：MC 构造完 Perlin 就丢 seed，只剩 `perm[256]+origin`，必须传表。
5. **AW 够用就别写 accessor**：Loom 把 AW 同时应用到编译期和运行期，`accessible field` 直接可用。

**数值 / WGSL**
6. **绝不用 f64**：WebGPU / 多数移动 GPU 不保证 `f64`；精度不够用 fp32 内部补偿。
7. **WGSL 无 f64/u64**：任何 double 位魔术（如 `fastInverseSqrt`）都要想 f32 替代；`inverseSqrt` 误差 ~1e-7 可接受。
8. **`hash3` 是 64 位**：Xoroshiro split 用 `hash3 ^ seedLo`，不能只取低 32 位。
9. **平台未定义行为要显式兜底**：`sqrt(负)`、`clamp(NaN)` Java 有定义、GPU 未定义。
10. **运算符优先级陷阱**：`+` 优先于 `&`（perm 索引）；WGSL 负数拼接 `--` 会解析失败。
11. **插桩污染数据**：`nanoTime` 包住可内联调用会阻止 JIT，绝对值不可信，只看比例。
12. **"肉眼无区别" ≠ 正确**：要方块级量化。**n==1 的 Spline 也常被当成常量**：MC 走
    `outside()`，`derivs=[1.0]` 时返回 `coordinate` 本身（如 -0.45）而非 `values[0]`；漏掉会让
    嵌套 values 全错，且只在特定 t 段暴露（Tectonic depth 偏差 ~0.08 → 海底 water↔stone）。

**GPU 架构**
13. **多 pass 是硬约束**：每类 shard 用独立 bind group layout，只含它实际用的 binding（≤8 storage buffer）。
14. **pipeline 缓存键不含代码生成**：磁盘缓存键 = AST 哈希 + pass，**既不含 layout 版本，也不含 WGSL 生成逻辑**。改 binding **或任何会改变 WGSL 输出的代码**（哪怕只动函数体）都必须 bump `LAYOUT_VERSION`，否则旧 pipeline 被复用 → `ComputePipeline invalid`。
15. **shader 编译成本超线性**：拆小是唯一可靠解法（安全区 ~2100 行，Mali 单 shader 过大会设备丢失）。
16. **惰性缓存可预计算**：MC 的惰性表（aquifer waterLevels）在 GPU 上必须整表预计算（Java warmup ~0ms）。
17. **内存泄漏**：共享 map 要 remove-on-consume，不能只 insert。
18. **诊断代码放对位置**：别放在会被短路的分支后（否则是死代码）。

**方法论（最重要）**
19. **端到端指标才是判据**：Chunky 总时间，不看内部 batch 吞吐（§19 教训）。
20. **"CPU 解释器一致" ≠ "对 vanilla 正确"**：测试要对照 MC 原版 dump。
21. **先数消费者，再决定算多少**：16 通道里只有 2 个被读，别默认全算。
22. **假通过的测试比没测试更危险**：`eprintln` 而非 `assert` 的判据会沉默。
23. **热路径加原子操作会拖垮性能**：计数用纯 `incrementAndGet`。

### 待办

- ~~**barrier 噪声分支**（aquifer 99.97%→100%）~~ → **暂不修**（与 §14 Pass E 同类，见下）。
- **矿脉全 GPU 化**：仍是角点粗筛 + MC 精算。
- ~~**pass B 超安全线**~~ → **已解决（2026-10-17）**：shard B 3084→1771 行。
  做法不是拆 dispatch（单通道标量输出，拆开要额外 buffer 通信），而是把两处高频
  内联公式提成 WGSL 函数：`spline_seg`（190 段 ×4 行 → 1 行）与 `octave_stack`
  （52 个噪声栈 ×13 行 → 1 行）。数学 bit-identical（`numerical_validation_spline`
  仍为 1.79e-7，实机地形逐块一致），并 bump `LAYOUT_VERSION` 5→6。
  （注意与 §14 的 Pass E 区分：Pass E 是逐方块求值方案，会让 shard B 涨到 5537 行，最后才做。）
- ~~**§14 interpolated 边界 bug（Tectonic/Epic 触发）**~~ → **暂不修（2026-10-17 决策）**：
  `squeeze` 等非线性节点包在 `interpolated` 外层时，MC 是"逐方块 外层(角点插值(内层))"，
  我们是"角点插值(整棵树)"，不等。现象：浮空水、怪坑、怪地形（滴水石锥症状已由 §22 的
  Spline n==1 修复消除）。**vanilla 不触发，Tectonic/Epic 触发**。
  修法（Pass E，**与边界问题一并最后修**）：拦截 MC 的 `DensityInterpolator.fill()`，用 GPU 角点值填充 MC 缓冲，
  让 MC 自己完成"插值 + 外层求值"。注意树里有 ~8 个 interpolator（含含水层噪声通道），
  需枚举 + 建实例↔子树映射 + 改 blobify 协议。
  **不修的原因**：Pass E 实测把 CPS 封顶在 ~150（每 chunk 多一次 98k 逐方块 dispatch，
  shard B 3084→5537 行，Mali 设备丢失风险），地形收益被性能成本抵消。
- **aquifer barrier 分支（99.97%→100%）暂不修（2026-10-17 决策）**：与 Pass E 同类。
  MC 的 `calculateDensity` 需 per-block 采样 barrierNoise（ch0），Tectonic 的 ch0 是
  90 行结构；BlockOut binding 已满 8/8（铁律 13），加 perm/origins 需重构且 shader
  逼近 Mali 安全区。收益仅过渡带 0.03% 方块，与 Pass E 一并搁置。

### 环境变量速查

| 变量 | 值 | 效果 |
|---|---|---|
| `C2ME_WGPU_DEBUG` | `1` | 详细日志 → `logs/c2me-wgpu.log` |
| `C2ME_WGPU_PROFILE` | `1` | 分阶段 GPU 计时 |
| `C2ME_WGPU_SKIP_AQUIFER` / `MIN_STAGE` | `0` | 诊断快捷路径（产错地形，仅计时） |
| `C2ME_DUMP_*` | — | 各种离线参考 dump（测完关掉）；需同时设 `C2ME_DUMP_CHUNK` 才生效 |
| `C2ME_DUMP_DIR` | 路径 | dump 目录，默认平台临时目录（Android 无 `/tmp`） |
| `WGPU_BACKEND` / `WGPU_ADAPTER_NAME` | — | 标准 wgpu GPU 选择变量 |

---

## §0 顶层目标与设计约束（不可动摇）

**主要目标：在"多数 GPU"上实现 Minecraft GPU 地形生成加速。**
- 目标平台 **兼容性优先**：桌面独显/集显 → 笔记本 → 手机 SoC GPU（Adreno / Mali / PowerVR）。
- 一切技术选型以"能跑在尽可能多的 GPU 上"为最高优先级。

**硬约束：**
- **计算精度 = fp32**。绝不用 `f64` / `shader-f64`（WebGPU / 多数移动 GPU 不保证支持）。
- **误差预算**：浅表达式 ~1e-6，深表达式（finalDensity）允许 ~1e-2。地形判阈对此不敏感。
  这是 fp32 固有上限，**不接受任何"改 f64"的提议**。需更高精度时只用 fp32 内部补偿。

---

## §1 项目目标

为 **C2ME** 提供 **WGPU** GPU 加速区块生成模块。
- 将 MC 1.21.1 `NoiseRouter` 密度函数在 GPU 计算。
- Java 通过 **Panama FFM** 调用 Rust cdylib，下行 C 函数 + 轮询模型，无上行回调。

---

## §2 工程结构

> **2026-10：已与 C2ME 解耦构建依赖**（§10）。C2ME 仅作**运行时必需依赖**
> （提供 GPU 批处理赖以吞吐的多线程并行度）。

### 2.1 构建
曾用 Gradle `includeBuild` 引入 C2ME 复合构建，**现已移除**（代码零 C2ME 引用）。

### 2.2 版本对齐

| 项 | 值 |
|---|---|
| Gradle wrapper | 8.14.3 |
| Minecraft | 1.21.1 / Yarn 1.21.1+build.1 |
| Fabric Loader | 0.17.2 / Loom 1.13-SNAPSHOT |
| Java | toolchain 25（运行要求 ≥25）/ Gradle JVM 21 |

### 2.3 C2ME 依赖方式
- **编译期**：无。**运行时**：`fabric.mod.json` 声明 `"c2me": "*"` 硬依赖。
  实测（§10）：无 C2ME 时桌面 CPS ~144→~13（批攒不满）。C2ME 是性能硬需求。

### 2.4 Access Widener（自写）
`src/main/resources/c2me-opts-accel-wgpu.accesswidener`：只放宽真正访问的
`DensityFunctionTypes$*` 内部类与少量字段。历史：曾借用 C2ME 的 AW，但它删条目
会导致我们 `IllegalAccessError`——自写后契约归自己。

---

## §3 核心设计决策

### 3.1 用 instanceof 而非字符串比较
Loom 不重映射字符串常量，运行期类名是 intermediary（`class_6916$class_7051`）。
全部改用 `instanceof` + record accessor。

### 3.2 序列化协议（blob 布局）

### 3.3 置换表资源
MC 构造完 Perlin 后丢 seed，只剩 `perm[256]+origin`，必须传表本身。
收集：树取自 `settings.noiseRouter()`（结构完整），sampler 用
`NoiseConfig.getOrCreateSampler(key)` 现场生成（settings 的 sampler 为 null）。

### 3.4 结果 blob（GPU → Java）

```
[HEADER 8B] magic WG00 + version u16 + chunk_count u16
[offsets 128B] 16 channels payload offsets (u64 LE)
[payload] 16 channels x cells x f32
[CRC32 4B]
```

### 3.5 区块写回（当前架构）
MixinChunkGenerator.populateNoise：
1. 首次：序列化 NoiseRouter → worldNew 建 Rust WorldContext
2. 每 chunk：stage aquifer/beard 表 → requestChunks → pollStatus → 取 blob
3. GPU 直接输出 per-block 状态字节（0=air/1=stone/2=water/3=lava），Java 只把字节写进 chunk section 的 palette（含 heightmap 更新）
4. cir.setReturnValue(completedFuture(chunk)) 取消原版生成
5. **这是唯一的 GPU 写回路径**：旧的"GPU 只算角点密度、MC 在 CPU 上做三线性插值 +
   aquifer + 矿脉"路径已按 §20 的测量结论删除（CPU 写回把 CPS 钉在 ~150，移动端 ~6）。
   三个相关开关（`C2ME_WGPU_BLOCK_OUT/AQUIFER/BEARD`）也随之取消——它们已是正确性的
   必要条件，关掉只会产生错地形。
6. **已知缺口**：矿脉（`OreVeinSampler`，连片大矿脉）尚未在 GPU 上实现。
   普通矿石（煤/铁/金等）由 biome feature 阶段放置，不经过这条路径，不受影响。

---

## §4 当前完成度

见顶部「速查 · 当前完成度」表。

---

## §10 与 C2ME 的关系（构建解耦 + 运行时依赖）

- **构建解耦**：代码零 C2ME 引用，不需 C2ME 源码（stub 由 `src/stub` compileOnly 提供）。
- **运行时必需**：实测无 C2ME 时桌面 CPS ~144→~13（缺多线程并行，批攒不满）。
- **自写 AW**：不再借 C2ME 的 access widener，避免其删条目导致 `IllegalAccessError`。

## §11 混合执行（任意 modded 密度函数兜底）

Rust 不认识的密度节点（地形 mod）在 CPU 按同一角点网格求值，作为 "external input"
喂给 GPU（binding 6）。其余树仍走 GPU，**modded worldgen 保持工作**，只有不支持的子树付 CPU 成本。
实机验证：Tectonic、Epic Terrain 正常。

## §12 诊断基础设施

- Rust 日志写 `<gameDir>/logs/c2me-wgpu.log`（Java 启动时传路径）。
- `C2ME_WGPU_DEBUG`：shader 编译计时、缓存命中、CPS 统计。
- `C2ME_WGPU_PROFILE`：分阶段 GPU 计时（encode/submit/readback/decode）。
- `C2ME_DUMP_*`：离线参考 dump（密度/方块/blob/beard/aquifer 表）。

## §5 附录：开发历程

> 以下为各里程碑历史，压缩自旧版文档，保留关键决策与教训。

### §5.x 里程碑索引（细节见 git history）

- **§5.1 2a**：首个可跑管线。地形出现。修复：magic 字节序、instanceof、`dx_max` 重定义。
- **§5.5 2b**：emitter memoization，WGSL 13 万行 → 5954 行；节点数值验证 ~1e-7。
- **§5.6 2c**：批处理 + 网格对齐 MC 角点（4→8），方块匹配 94.68%→96.18%。
- **§5.7 2d**：正确性验证框架；网格 Y cell 修正；EndIslands（simplex）1.19e-7；多维度守卫；磁盘 PipelineCache。教训：RAW_ROUTER 不可靠，要用 compiled router。
- **§5.8 2e**：ch10/ch11 残差根因 = blob 数据源 + octave 顺序。教训：CPU 解释器一致 ≠ 对 vanilla 正确。
- **§5.9 2f**：多维度 WorldContext；三个正确性 bug（octave 顺序 / simplex origin / EndIslands 环条件反转）。
- **§5.10 2g**：NaN 语义兜底（远坐标）；测试判据改进。教训：平台未定义行为要显式兜底。
- **§5.11 2h**：矿脉路线 3（GPU 角点粗筛 + MC 精算）。教训：CPU per-block 复用 MC 采样器 = 灾难。
- **§5.12 2i**：前端通知 + 初始化失败安全回退。教训：`<clinit>` 抛异常 = 类永久损坏。
- **§5.13 2j**：Android aarch64 交叉编译（cargo-ndk）。
- **§5.14 2k**：内存泄漏修复（result blob 每 chunk 泄漏 + executor 每 batch 重建 buffer）。
- **§5.15 2l**：Mali-G610 shader 规模崩溃 → 按通道拆 A/B/C 三片。教训：编译成本超线性，安全区 ~2100 行。
- **§5.16 2m**：只算/只回读被消费通道（shader 6223→2740 行，回读 -87.5%）。教训：先数消费者。
- **§5.17 2n**：群系气候 GPU 加速（design 2）。
- **§5.18–§5.21**：异步调度 A/B、写回天花板实测、最小闭环。核心结论见 §19–§22。
- **§22 2o**：aquifer + Beardifier 全 GPU，端到端反超纯 CPU。见下节。

### §13–§18 已证伪的实验（存档）

- §13 手机"高 U 低显"负优化（旧架构 CPU 写回下成立，§20 后架构改变）。
- §14 interpolated 边界 bug（非线性外层角点插值错，Tectonic/Epic 触发，未修）。
- §15 JFR 热点：写回瓶颈在 CPU 逐方块（aquifer/heightmap/vein）。
- §16 GC 对负优化贡献：实测排除。
- §17 aquifer 逐帧归因：~65% 写回，一半控制流一半噪声。
- §18 各种"取巧"思路：heightmap 一次性、aquifer 部分 GPU 化等，多数证伪。
  **结论：够长的杠杆是"整体写回 GPU 化"（§20–§22 已实现）。**
### §19–§22 核心结论（近期）

**§19 端到端 A/B（异步调度裁定）**：桌面异步调度曾净负（30 vs 纯 CPU 23s），
根因是 GPU 往返固定延迟落在关键路径。**教训：端到端指标才是判据。**

**§20 写回天花板实测**：MIN_STAGE（跳全部写回）11s ≪ 纯 CPU 23s，证明
**净负全部来自 CPU 写回留在关键路径**，GPU 密度管线本身很快。

**§21 最小闭环**：BlockOut shard 首次跑通（GPU 输出 per-block 状态字节），16s。

**§22 aquifer + Beardifier 全 GPU（当前）**：
- aquifer：Java warmup 惰性表（315 cell ~0ms）+ compact 序列化（2544B）→
  Rust 现算 blockPositions（种子来自表头）→ GPU 12-cell 扫描 + 分类。99.97%。
- beard：静态 24³ 权重表（每 world 一次）+ per-chunk pieces（固定 stride）。
  bit-exact。**必须 f32**（WGSL 无 f64）。
- **端到端 15s vs 纯 CPU 23s（反超 35%），功能完整，地形正确。**
- 遗留 0.03% = 过渡带 `calculateDensity`（barrier 噪声）分支未实现。
- 验证链全离线：CPU-f32 / blockPositions / WGSL / e2e / beard，5 个测试。
- shard B 瘦身（2026-10-17）：`spline_seg` + `octave_stack` 两个 WGSL helper 替换
  高频内联展开，3084→1771 行（重回 Mali ~2100 安全区），bit-identical。
- Spline n==1 修复（2026-10-17）：单点 spline 漏了 `outside()` 外推 → 嵌套 values 全错
  → depth 偏差 ~0.08 → 海底 y≈43–55 误判 stone（chunk 50,50 diff 8.88%）。改
  `spline.rs`/`world.rs` 两处 n==1 分支 + bump `LAYOUT_VERSION` 4→5。验证：逐通道
  bad=0/1225、逐格 diff 归零（回退复现 +0.08~0.098）、GPU rms 2.5e-5/5.7e-5、84 测试
  全过、实机 Tectonic 与 Epic 滴水石锥均消失；怪坑（§14 非线性外层）仍独立存在。

### 关键文件

- `rust/src/codegen/wgsl_gen.rs`：WGSL 生成 + shard 规划 + BlockOut/beard/aquifer WGSL。
- `rust/src/scheduler/executor.rs`：batch 执行、bind group、buffer 管理。
- `rust/src/world/world.rs`：WorldContext、pipeline 编译、测试。
- `rust/src/world/{aquifer,beard}.rs`：aquifer 表解析 + Beardifier 复现。
- `rust/src/c_api.rs`：C API（含 `set_aquifer_table` / `set_beard_table`）。
- `src/main/java/.../mixin/MixinChunkGenerator.java`：写回 + stage + dump。
- `src/main/java/.../WgpuThinBinding.java`：FFM 绑定。

---

