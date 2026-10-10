> ⚠️ **接手者必读（第一件事）**：完整阅读本文件，尤其 **§0 顶层目标与设计约束**。
> 核心铁律：精度恒为 **fp32**，**禁止使用 f64 / shader-f64**（会破坏"多数 GPU 含手机"的顶层目标）；
> 深表达式误差到 ~1e-2 是 fp32 固有上限，**不是 bug**，不要提议改 f64。
> 变量、代码风格与现有 emitter 保持一致；改动前先读懂对应 §。

---
# C2ME WGPU 加速模块 — 进度计划

> 最后更新：2026-10-05
> 状态：**2k 里程碑** —— 桌面（Linux x86_64）+ Android（aarch64；Kirin 980 / 天玑 8100 实测）
> 双端 GPU 地形生成跑通。多维度（Nether/End）、矿脉、EndIslands、内存泄漏修复均已完成；
> **§8 待办已清空**。
> 下一步：按需开新章节（见文件末尾 §8 之后的说明）。

---

## §0 顶层目标与设计约束（不可动摇）

**主要目标：在"多数 GPU"上实现 Minecraft GPU 地形生成加速。**
- 目标平台是 **兼容性优先**：桌面独显/集显 → 笔记本 → **后期可能包括手机 SoC GPU（Adreno / Mali / PowerVR）**。
- 一切技术选型以"能跑在尽可能多的 GPU 上"为最高优先级，性能与精度都服从于此。

**由此派生的硬约束：**
- **计算精度 = fp32（`f32`）**。绝不使用 `f64` / `shader-f64`：
  - WebGPU / 多数移动 GPU **不保证 `f64` 支持**；`shader-f64` 是可选特性，大量设备缺失。
  - 用 `f64` 会让模块在目标平台（尤其手机）上直接无法运行 —— 违背顶层目标。
- **误差预算**：与 Java（全程 `double`）的密度结果偏差，浅表达式应在 ~1e-6，深表达式
  （如 `finalDensity`，含数千次噪声采样 + lerp/FMA 累积）允许到 ~1e-2。
  - **地形判阈对 1e-2 密度差不敏感**（实测地形"与原版别无二致"），故该误差**可接受**。
  - 这不是 bug，是 fp32 的固有上限；**不接受任何"改 f64"的提议**作为常规解法。
- 需要更高精度时，只允许 **fp32 内部补偿**（Kahan 求和、double-double 模拟、局部 offset 坐标系），
  不得引入 `f64`。

---

## §1 项目目标

为 **C2ME（Concurrent Chunk Management Engine）** 提供基于 **WGPU** 的 GPU 加速区块生成模块。

- 将 Minecraft 1.21.1 的 `NoiseRouter` 密度函数在 GPU 上计算。
- **纯 fp32 运算 + 精度补偿**（见 §0），力求与 Java 结果一致。
- Java 端通过 **Panama FFM** 调用 Rust cdylib，5 个下行 C 函数，无上行回调，轮询模型。

---

## §2 工程结构（关键变更）

> **2026-10：已与 C2ME 解耦构建依赖**（详见 §10）。不再使用 C2ME 复合构建、
> 不再借用 C2ME 的 access widener。C2ME 仅作为**运行时必需依赖**保留在
> `fabric.mod.json`（它提供 GPU 批处理赖以吞吐的多线程区块生成并行度）。

### 2.1 构建（无 C2ME 复合构建）*

历史：曾用 Gradle `includeBuild` 把本地 `C2ME-fabric` 作为复合构建引入，
并用 `compileOnly` 引入 `c2me-base` / `c2me-opts-dfc`。**现已全部移除**——
我们的 Java 代码不引用任何 C2ME 类（见 §10），构建不需要 C2ME 源码。

### 2.2 版本对齐

| 项 | 值 |
|---|---|
| Gradle wrapper | 8.14.3 |
| Minecraft | 1.21.1 |
| Yarn | 1.21.1+build.1 |
| Fabric Loader | 0.17.2 |
| Loom | fabric-loom 1.13-SNAPSHOT |
| Java | toolchain 25（编译产物为 Java 25 字节码，运行时要求 ≥25） |
| Gradle JVM | 21 |

### 2.3 C2ME 依赖方式

- **编译期**：无依赖。代码零 C2ME 引用，构建不需 C2ME 源码。
- **运行时**：`fabric.mod.json` 声明 `"c2me": "*"` 为**硬依赖**。
  实测（§10）：无 C2ME 时本 mod 仍能运行，但桌面 CPS 从 ~144 掉到 ~13
  （≈ 手机水平）——因缺少 C2ME 的多线程区块生成，GPU 批处理攒不满批、
  利用率极低。故 C2ME 是**性能上的硬需求**，声明为必需。

### 2.4 Access Widener（自写）

`src/main/resources/c2me-opts-accel-wgpu.accesswidener`（~45 行）：
只放宽我们真正访问的 `DensityFunctionTypes$*` 内部类与少量字段/构造。
`fabric.mod.json` 用 `"accessWidener"` 字段声明它。

历史：曾**直接借用** `c2me-base.accesswidener`（编译期指向 C2ME 源码路径，
运行期靠 C2ME 的 AW 生效）。问题：把我们代码的可访问性契约寄生在 C2ME 上
——C2ME 哪天删掉某个它不再需要的条目，我们就会 `IllegalAccessError`（实测
移除 C2ME 时确曾触发）。自写 AW 后契约归自己，只随 Minecraft 版本变。

---

## §3 核心设计决策

### 3.1 用 instanceof 而非字符串比较（关键教训）

**问题**：Loom 重映射类引用，但**不重映射字符串常量**。运行期 `df.getClass().getName()` 返回 intermediary 名（`class_6916$class_7051`），与代码里硬编码的 Yarn 名（`DensityFunctionTypes$Constant`）永远不相等。

**方案**：`NoiseRouterBlobifier` 全部改用 `instanceof` 模式匹配 + record accessor（`.value()`、`.minInclusive()` 等）。类引用会被 Loom 重映射，运行期可靠。

### 3.2 序列化协议（blob 布局）

```
[HEADER 10B]
  magic   = 4D 43 32 57  ("MC2W", 逐字节写，非 putInt)
  version = u16 LE = 1
  reserved= u32 LE = 0
[NODE_TREE (recursive [tag:u8][payload])]
[SHAPE_CONFIG 16B]
  min_y, height, cell_width, cell_height  (各 i32 LE)
[BIOME_SOURCE]  tag 0xF0 + seed i64（占位）
[PERM_RESOURCE] (可选尾部)
  magic = 0x4D525750 ("PWPM", LE)
  count = u32 LE
  for each: perm[256] (u32 LE ×256) + origin_x/y/z (f64 LE ×3)
```

**注意**：magic 必须逐字节写成 `4D 43 32 57`——Java `putInt(0x4D433257)+LE` 会写成 `57 32 43 4D`，与 Rust 断言不符。

### 3.3 置换表资源（PERM_RESOURCE）

MC 1.21.1 构造完 Perlin 后**原始 seed 就丢了**，只剩 `permutation[256]` + `originX/Y/Z`。因此 Rust 端不能靠 seed 重建，必须接收置换表本身。

Java 端收集链路（全反射，按**类型**而非字段名——字段名也被混淆）：
```
DensityFunctionTypes.Noise.noise() → DensityFunction$Noise
  .noise() → DoublePerlinNoiseSampler          （若为 null，改用下方 key 路径）
  .noiseData().getKey() → NoiseConfig.getOrCreateSampler(key)  ← 现场生成
DoublePerlinNoiseSampler
  ├─ 字段类型为 OctavePerlinNoiseSampler 的（firstSampler/secondSampler）
  │    └─ 字段类型为 PerlinNoiseSampler[] 的（octaveSamplers）
  │         └─ 字段类型为 byte[] 的（permutation）
  └─ originX/Y/Z（public）
```

**关键**：`settings.noiseRouter()` 是**原始未编译树**（结构完整可序列化），但 Noise 的 sampler 为 null；`noiseConfig.getNoiseRouter()` 是 **c2me-opts-dfc 编译后的 JVM 字节码**（不可序列化），但 sampler 已实例化。**方案 A**：树取自 settings，sampler 用 `NoiseConfig.getOrCreateSampler(key)` 现场生成。两者结合。

### 3.4 结果 blob（GPU → Java）

```
[HEADER 8B]   magic "WG00"(LE) + version u16 + chunk_count u16
[offsets 128B] 16 通道各自的 payload 字节偏移（u64 LE）
[payload]      16 通道 × 4096 cell × f32（MC-native stride）
[CRC32 4B]
```

单 chunk = 16 × 16×16×16 × 4 = 262144 字节 payload。

### 3.5 区块写回

`MixinChunkGenerator` 在 `populateNoise` HEAD 处：
1. 首次调用：序列化 NoiseRouter → `worldNew` 建 Rust WorldContext
2. 每 chunk：`requestChunks` → `pollStatus` → `takeResultBlob`
3. 用 **finalDensity 通道（index 11）** 逐 cell 判 `> 0` → STONE，否则 AIR
4. `cir.setReturnValue(completedFuture(chunk))` 取消原版生成

cell 网格 = 4 × 96 × 4 = 1536，每 cell 覆盖 4×4×4 方块。

---

## §4 当前完成度

| 模块 | 状态 |
|------|------|
| 复合构建（include C2ME-fabric） | ✅ 100% |
| Gradle/版本对齐 1.21.1 | ✅ 100% |
| Java 源码移植（mojmap→yarn） | ✅ 100% |
| Rust 原生库编译 + 打包进 jar | ✅ 100% |
| C API / 句柄 / 实例设备 | ✅ 100% |
| AST 反序列化（26 标签） | ✅ 100% |
| 优化 pipeline（4 passes） | ✅ 100% |
| WGSL 生成 + naga 验证 | ✅ 100% |
| GPU pipeline 编译 + dispatch | ✅ 100% |
| 结果 blob 16 通道回读 | ✅ 100% |
| **GPU 算出的 density 写回 Chunk** | ✅ 已生效 |
| **Java→Rust 置换表传输** | ✅ 真实 perm 表 |
| **Noise 节点（多八度 DoublePerlin + DOMAIN_SCALE + origin + amplitude）** | ✅ **数值验证 1.19e-7** |
| **Shift / ShiftA / ShiftB + ShiftedNoise** | ✅ **数值验证 7.45e-9** |
| **Spline（Hermite 树）** | ✅ **数值验证 1.79e-7** |
| **YClampedGradient / SQUEEZE / RegistryEntryHolder 解包** | ✅ 已实现 |
| **emitter memoization（DAG 公共子表达式共享）** | ✅ WGSL 从 13.1 万行降到 5954 行 |
| **地形生成** | ✅ **finalDensity 有正有负，地形出现** |
| 地形边界截断（某坐标外消失） | ❌ 待查 |
| **网格 Y cell 对齐 MC（4→8）** | ✅ 方块匹配 94.68%→96.18% |
| **多维度守卫（非主世界禁用）** | ✅ matchesSettings 判定 |
| **shader 磁盘缓存** | ✅ LZ4 PipelineCache |
| **结构跳过优化（beard）** | ✅ CPS +45% |
| WeirdScaledSampler / InterpolatedNoiseSampler / Aquifer 等 | ❌ 仍占位 0 |

### 已修复的关键 bug（按发现顺序）

1. `modApi` 配置不存在 → 改 `compileOnly`
2. 字符串类名比较失效 → 改 `instanceof`
3. magic 字节序错 → 逐字节写
4. WGSL `dx_max` 重复定义 → 包进词法块 `{ }`
5. `noise_interpolated_sampler` 未定义 → 复用 `emit_cache_interp`
6. 输出缓冲只算 1 通道 → `f32_per_chunk = 16×16×16×16`
7. staging buffer 8MB 不够 → `256KB × max_batch_size`
8. 字段名混淆 → 反射改按**类型**找
9. `DensityFunction$Noise` 包装层 → 解包到 `DoublePerlinNoiseSampler`
10. `settings.noiseRouter()` 的 sampler 为 null → 结合 `NoiseConfig.getOrCreateSampler`
11. **`perm[p + (a+yi) & 255]` 运算符优先级**：`+` 优先于 `&`，`p` 被折进 mask；offset=0 时隐身，offset≠0 时读错表（26 处）
12. **DOMAIN_SCALE 施加对象错**：MC 只作用于 second stack，且是乘坐标；原实现乘进 xzScale 且两栈都乘
13. **amplitude 公式错**：硬编码 0.5，实际是 `(1/6)/(0.1*(1+1/(maxIdx-minIdx+1)))`
14. **多八度频率/振幅**：应每八度 `×2 / ÷2`，原实现乘 lacunarity
15. **origin 完全丢失**：MC 每个 Perlin 采样无条件 `+originX/Y/Z`，原实现没用
16. **ShiftB 坐标 swizzle 自我覆盖**：`cell_y = cell_x*0.25` 读到已被覆盖的 `cell_x` → 先用 `_ox/_oy/_oz` 暂存
17. **RegistryEntryHolder 包装层未解包**：router 通道顶层是 `class_7051[function=Direct{...}]`，真身在里面
18. **`class_5822` 是独立类**：`InterpolatedNoiseSampler` 不在 `DensityFunctionTypes` 里，instanceof 匹配不到
19. **WGSL 负数拼接 `--`**：`-{negative}` 拼成 `--2.0e-2` 解析失败
20. **emitter 无表达式共享**：MC 是 DAG，相同子表达式被重复展开 → 13 万行 WGSL；加 scope-stack memoization 后降到 5954 行

---

## §5 2a 里程碑（已完成）

### 5.1 目标与结果

**目标**：证明"置换表 → GPU → 采样出非零噪声"整条链打通。

**结果**：✅ 达成。`nonZero=159/1536`，值域 `[-0.065, 0]`。

### 5.2 改动清单

| 层 | 改动 |
|---|---|
| Java `NoiseRouterBlobifier` | Noise 节点输出 `TAG_NOISE_SAMPLER(0x84)` + `perm_offset(i32)` + 真实 xz/y scale |
| Rust `ast/mod.rs` | `NoiseSampler.seed:i64` → `perm_offset:i32` |
| Rust `ast/deserialize.rs` | `0x84` 解析新格式：`type:u8 + perm_offset:i32 + xz_scale:f32 + y_scale:f32` |
| Rust `world/constants.rs` | 新增 `PermutationTableSet::from_entries()` 用真实表 |
| Rust `world/world.rs` | 优先用 PERM_RESOURCE 的真实表 |
| Rust `codegen/wgsl_gen.rs` | binding 1 改 `array<u32>`（动态） |
| Rust `codegen/math_lib.wgsl` | perlin 函数改收 `perm_offset:i32`，从 `perm[p + idx]` 查表 |
| Rust `codegen/emitters/mod.rs` | `NoiseSampler` 生成 `noise_perlin3d(perm_offset, cell*scale)` |

### 5.3 关键 bug（本轮修复）

1. **PERM 收集**：`n.noise()` 返回 `DensityFunction$Noise` 包装，需解包到 `DoublePerlinNoiseSampler`（按类型找字段）
2. **sampler 为 null**：`settings.noiseRouter()` 是原始树，Noise 的 sampler 未实例化 → 改用 `NoiseConfig.getOrCreateSampler(key)` 现场生成
3. **cell 数不一致**：Rust 用 `16×16×16=4096`，WGSL/Java 用 `4×96×4=1536` → 统一为 1536
4. **dispatch 范围不足**：`dispatch_workgroups(1,1,1)` 只算 1/24 的 cell → 改为 `(1,24,1)`（按 shape 动态算）
5. **DOMAIN_SCALE 遗漏**：MC 在 `DoublePerlinNoiseSampler` 里乘 `1.0181268882175227` 把整数坐标错开，否则 Perlin 在格点上恒为 0

### 5.4 当前局限（2b 要补的）

- 只用 DoublePerlin 的**第一个八度**（应两层叠加）
- 未做多八度 `amplitudes`/`lacunarity` 累加
- 其余 14 个通道的非 Noise 节点（Spline/Aquifer/Bezier/IntervalSelect…）仍是占位 0
- 结果值域远小于 MC 真实值

---

## §5.5 2b 里程碑（已完成）

### 5.5.1 目标与结果

**目标**：让 GPU 算出接近 MC 的 density，地形不再是虚空。

**结果**：✅ 达成。`finalDensity` 值域从 2a 的 `[-0.065, 0]`（全负→全空气）变为 `[-0.667, +0.275]`，**有正有负，地形正常生成**（石头/空气分布）。

### 5.5.2 新增/修复的节点

| 节点 | 说明 |
|---|---|
| `DoublePerlinNoiseSampler` | 双栈 + 多八度 + DOMAIN_SCALE + amplitude 公式 |
| `PerlinNoiseSampler` origin | 每采样无条件 `+origin`（`maintainPrecision` 之后） |
| `Shift` / `ShiftA` / `ShiftB` | `offsetNoise(coord*0.25) * 4`，坐标轴 swizzle |
| `ShiftedNoise` | 三个偏移噪声改坐标后采样主噪声 |
| `Spline` | 三次 Hermite 插值树（locations/derivatives/values） |
| `YClampedGradient` | `clampedMap(y, fromY, toY, fromValue, toValue)` |
| `SQUEEZE` | `x/2 - x³/24`，专用 tag `0x25`（避免子树重复序列化） |
| `RegistryEntryHolder` | router 通道顶层包装，`function().value()` 解包 |

### 5.5.3 emitter 架构改进：memoization

MC 的 density function 是 **DAG**（有向无环图），不是树——同一子表达式（`depth`、`offset` 等）被数十处引用。

原 emitter 每遇一次引用就完整展开一次子树 → **131064 行 WGSL / 5.7 MB** → naga 验证 36s + 驱动编译 **87 分钟**（或直接崩）。

加入 **scope-stack memoization**（`GenCtx.memo_stack`）后：
- 相同 AST 指纹的子表达式只发射一次，后续复用 `let` 变量
- 作用域栈保证变量在引用处可见（`{` 压栈 / `}` 弹栈）
- `in_shadow` 标志在坐标被遮蔽时（Shift/ShiftedNoise/Cache）禁用 memo，避免坐标错用

**结果：5954 行 / 290 KB**（20 倍缩减），编译时间回到可接受范围。

### 5.5.4 数值验证

全部通过 Rust 内部 GPU-vs-CPU 对照：

| 测试 | 误差 |
|---|---|
| 多八度 DoublePerlin + DOMAIN_SCALE + 非零 origin | 1.19e-7 |
| ShiftedNoise（含 ShiftA/ShiftB） | 7.45e-9 |
| Spline Hermite（4 节点，非退化采样） | 1.79e-7 |

均为 fp32 精度极限。

### 5.5.5 地形正确性突破（2b 收尾）

**问题**：海洋被填成陆地、矿洞变成水洞。

**根因链**（逐一排除后定位）：
1. 密度链本身正确（ch6-10 与 MC 误差 ~1e-5；ch11 在口径统一后匹配）
2. `Cache2D`/`FlatCache` 曾被误改为占用共享内存 + barrier，嵌在
   `CacheInterpolated` 的变长循环内属 WGSL UB → 改为直通
3. 真正的缺口是 **Aquifer**：`density<0 且在海平面下` 应为水而非空气，
   我们只写 STONE/AIR，导致海底暴露、表面规则铺草 → 海洋变陆地

**解法（关键）**：不重写 Aquifer，**复用 MC 自己的 `AquiferSampler`**：
- 新增 `@Invoker` 调 vanilla 私有 `createChunkNoiseSampler(chunk, sa, blender, noiseConfig)`
- 取 `ChunkNoiseSampler.getAquiferSampler()`
- 写回时 `bs = aquifer.apply(pos, gpuDensity)`；`null → 石头`，否则用
  返回值（水/岩浆/空气），**天然与原版一致**

**新增/修复节点**：
| 节点 | 说明 |
|---|---|
| `WeirdScaledSampler` | scaleTunnels/scaleCaves 分段函数 + ysmear perlin |
| `InterpolatedNoiseSampler` | 3 个八度栈 + `perlin_ysmear`（16/16/8 八度）|
| `FlatCache`/`Cache2D` | 改为纯记忆化直通（去 barrier）|
| `CacheInterpolated` | 角点采样 + 中心/角点口径统一 |
| `HALF_NEGATIVE`/`QUARTER_NEGATIVE` | 修正为 `x>0?x:x*k`（原误为无条件乘负）|
| `SQUEEZE` | 补 `clamp(x,-1,1)` |
| `RangeChoice` | 修正边界分支语义 |
| `RegistryEntryHolder` | router 通道顶层解包 |

**已知遗留**：
- **地形边界截断**：某坐标外地形消失（待查 batch origin）

---

## §5.6 2c 里程碑（已完成）—— 批处理 + 结构正确性

对应 commit `ade3a38` + `0bcf87e`。

### 5.6.1 目标与结果

**目标**：让 GPU 真正批量生成（此前每 chunk 一次 dispatch，固定开销
主导），并让结构（村庄）正确落地。

**结果**：✅ 达成。
- 地形肉眼与**原版无区别**
- **结构完全正确**：村庄不埋不悬空、村民不窒息
- **CPS 40~90**（此前单块 38、且 CPU 高）；CPU 明显下降
- 区块边界灰条消失

### 5.6.2 网格对齐 MC 角点采样（`ade3a38`）

MC 的 `ChunkNoiseSampler` 每轴采 `horizontalCellCount + 1` 个**角点**
（x/z = `0,4,8,12,16`，y = `-64..320` 共 97 个），相邻区块共享边界
角点 → 天然连续。

之前只采 `4×96×4` 个 cell，缺边界角点，靠 `cellAt()` 线性外推补——
密度是几十个八度噪声之和，边界非线性的，外推把边缘地形抬高 →
**埋村庄 + 可见接缝**。

改动：
| 位置 | 改动 |
|---|---|
| `wgsl_gen.rs` ×3 prologue | `cell_count_* = 16/cell_width + 1`（y 同理），越界 invocation 用 `return` 守卫 |
| `cache_like.rs` | `CacheInterpolated` 降级为**直通**（dispatch 已是角点网格，每个 invocation 就是一个角点样本；原 5×5×5 共享内存 + barrier 冗余，且 barrier 让越界 early-return 不安全） |
| `executor.rs` / `c_api.rs` | 缓冲 `16 * (5*97*5)` f32/chunk；dispatch 覆盖角点网格 |
| `world.rs` 测试 | 网格常量 `1536 → 2425`，索引 stride 更新 |
| `MixinChunkGenerator.java` | 写回后跟踪 `OCEAN_FLOOR_WG` + `WORLD_SURFACE_WG` 高度图（结构放置读地表高度）；复用 `chunk.getOrCreateChunkNoiseSampler` 替代每 chunk 新建 `ChunkNoiseSampler` |

### 5.6.3 多块 GPU dispatch（`0bcf87e`）

**问题**：Java 侧每 chunk 单独 `requestChunks(单块)`，GPU 每次只算 1 块，
固定开销（buffer/bind/submit/readback）主导 → 吞吐上不去。

**方案**：
- **shader**：新增 binding(4) `positions`（打包 `[cx,cz,...]`）；`gid.z`
  覆盖 `cells_z × num_chunks`，一次 dispatch 生成整批。`chunk_index =
  gid.z / cells_z` 选 per-chunk 原点；输出 **chunk-major**：
  `(chunk*16 + channel)*cells_per_chunk`
- **`bind_group.rs` / `executor.rs`**：上传 positions buffer、绑定
  binding 4、dispatch `cells_z*num_chunks` 个 workgroup
- **`c_api.rs`**：按 `batch.positions`（sorted+deduped）顺序输出，
  保证 batch index ↔ chunk 坐标对应（原 `FxHashMap` 迭代无序，无法映射回坐标）
- **`WgpuBatchCoordinator.java`**（新）：聚合跨 C2ME worker 的请求
  （≤64 块，5ms 窗口），一次 `requestChunks`；mixin 用 `batchIndex`
  从整批 blob 里切出自己的 chunk

### 5.6.4 Beardifier —— 结构地形整形（`0bcf87e`）

**根因**：GPU 密度**缺 `Beardifier` 项**。MC 的 `finalDensity` 含
`BeardifierMarker` 占位符，`ChunkNoiseSampler` 建时按
`structureAccessor` 换成真实 `StructureWeightSampler`——它在结构（村庄等）
周围**压平/填埋地形**，让房子嵌进地面。我们没这一项 → 结构区域该塌的没塌
→ 房子被埋。（这也解释了历史现象：修高度图前"悬空"、之后"一栋悬空"、
后来"被埋"——一直缺这个密度项。）

**解法（复用 MC）**：Java 写回时
```java
StructureWeightSampler sw = StructureWeightSampler.createStructureWeightSampler(
        structureAccessor, chunk.getPos());
double beard = sw.sample(npos);
BlockState bs = aquifer.apply(npos, d + beard);
```
**必须 per-block**：字节码确认 MC 的 `mapAll` 把 `Beardifier` 换成
`StructureWeightSampler` **直接求值**，**不经过** `DensityInterpolator`
角点插值。per-corner 插值会抹平结构地基 → 又悬空。

非结构 chunk 的 piece 列表为空、`sample()` 返 0，无副作用。

### 5.6.5 清理

- 删除 `chdiv` 诊断（口径不可靠：`REF_ROUTER.finalDensity()` 含
  Beardifier，GPU 值不含 → 结构附近必然"失配"，全部误报）
- 删除 per-batch `eprintln!`
- 保留一次性 `ch10` dump

### 5.6.6 benchmark 结果与局限

`gpu_performance_benchmark`（`--nocapture` 可见）：
```
吞吐量 ~186 区块/秒（64 批 × 64 块），GPU 忙碌 99.7%
```
⚠️ **局限**：benchmark 用的简化 AST 只产 ~811 行 shader，真实游戏
router 是 5000+ 行。绝对吞吐量不能直接外推，但「GPU 忙碌占比 / 调度
开销占比」的结论仍成立。

---

## §5.7 2d 里程碑（已完成）—— 正确性验证 + 端地 + 缓存

### 5.7.1 真实 router benchmark 工具链

- benchmark 原先用手搓 811 行 blob，严重失真。新增：游戏侧 dump 真实
  router blob（property/env/marker 触发）→ `src/test/resources/real_router.bin`，
  Rust benchmark 优先读它。实测真实 shader ~171 CPS（vs 手搓 187）。
- 结论：**shader 复杂度不是瓶颈**（真实只慢 8%）。

### 5.7.2 网格 Y cell 修正（`90b33a0`）—— 关键正确性 bug

- MC `GenerationShapeConfig` 的 cell 尺寸 = `size*4`；overworld 用
  `size_vertical=2` → **真实垂直 cell 高 = 8 格（49 个角点）**，不是 4。
- 之前按 4 格采样/回写 → Y 向插值与 vanilla 不一致。方块级对比：
  **94.68% → 96.18%** 匹配。
- 顺带修 `executor.rs` 硬编码 `5*97*5`（改 shape 后回读全 0 的 bug）。

### 5.7.3 可靠验证框架

- **关键教训**：`RAW_ROUTER` 独立 `.sample()` **不可靠**（cache 节点
  返回 0 / 被污染），必须用 **compiled router**（`noiseConfig.getNoiseRouter()`）
  做参照。
- 新增 `real_router_vs_vanilla` 测试：15 通道逐角点对比，噪声原语
  rms ~1e-5 全部 OK；ch10/ch11 有残差（水面）。
- 新增方块级对比（GPU chunk vs vanilla chunk）。

### 5.7.4 动态 ShapeConfig（`ac0735c`）

- grid 参数从 `generationShapeConfig()` 读取（不再硬编码 Overworld），
  为 Nether/End 铺路。Overworld 行为不变。

### 5.7.5 结构跳过优化（`a1670a0`）

- `StructureWeightSampler.sample()` 每 chunk 调 ~98k 次（占写回 ~33%），
  但无结构 chunk 恒返回 0。用 vanilla 同款谓词
  （`getTerrainAdaptation() != NONE`）先查 `getStructureStarts`，空则跳过。
- 实测协调器 **CPS ~43 → ~62（+45%）**。

### 5.7.6 EndIslands（末地岛屿）—— 完整实现 + 验证

- Java 捕获 EndIslands 内部 `SimplexNoiseSampler` 的 `int[512]` perm +
  origin，新增 SIMPLEX_RESOURCE trailer。
- Rust：新 `SimplexEntry` 解析；AST `EndIslands{perm_index}`；新 emitter。
- WGSL：从零移植 MC 2D simplex（int[512] perm、16 梯度 mod 12、
  skew/unskew）+ 25×25 岛屿扫描；新 binding(5)。
- **数值验证**：`end_islands_cpu_vs_gpu` max_diff = **1.19e-7**
  （1746 个非 floor cell，核心命中分支被触发）。

### 5.7.7 多维度守卫（`cb67082`）

- GPU world context 是单例（静态 WORLD_HANDLE，一次初始化）。
  Nether/End 用不同 shape 且 End 需 EndIslands → 走 GPU 会产出错误地形。
- `populateNoise` 加 `matchesSettings(OVERWORLD|LARGE_BIOMES|AMPLIFIED)`
  守卫，其余维度回退 vanilla。
- **真正 per-dimension context 是未来工作**（解锁 Nether/End）。

### 5.7.8 性能与缓存

- **磁盘 PipelineCache**（`caeda8c`）：`$XDG_CACHE_HOME/c2me/wgpu_pipelines/`，
  LZ4 压缩 + 原子写。省 `create_compute_pipeline`（游戏内 ~7s）。
  修复 `compress_prepend_size` 与 `decompress_size_prepended` 不匹配。
- **release 跳过冗余 naga 校验**（`e709d35`）：`create_shader_module`
  内部已跑 naga，我们的 `validate_wgsl` 在 release 是重复。debug-only 后
  省一次完整 naga 解析。
- **局限**：naga 前端翻译（`create_shader_module`，游戏内 ~10s）
  wgpu PipelineCache 不覆盖，仍需每次跑。

### 5.7.9 关键教训

- **插桩污染数据**：`nanoTime` 包住可内联的调用会阻止 JIT 内联，绝对值
  不可信，只能看比例。
- **"肉眼无区别" ≠ 正确**：方块级量化才发现 5% 真实差异。
- **矿脉架构冲突**：`vein_toggle/ridged` 走 `interpolated`（角点+插值），
  且 `vein_gap` 是周期 1 格的高频噪声 → 角点网格无法表达；需独立
  per-block GPU 路径。当前实现净负，已回退。

## §6 下一步：2c 剩余 —— 地形完善

2c 的**核心**（网格对齐、批处理、Beardifier、高度图）已完成，见 §5.6。
以下为剩余项。

### 2c.1 地形边界截断（待查）

**症状**：世界某坐标之外地形突然消失（全部空气）。
**状态**：未复现/未定位。候选方向：`executor.rs` 的 batch origin、
跨批次 cell 世界坐标映射。（注：多块 dispatch 后 origin 已改走
`positions` buffer，`WorldgenParams.batch_origin_x/z` 基本弃用。）

### 2c.2 补齐剩余占位节点

| 节点 | intermediary | 作用 | 状态 |
|---|---|---|---|
| `WeirdScaledSampler` | `class_6944` | 矿脉/洞穴缩放 | ✅ |
| `InterpolatedNoiseSampler` | `class_5822` | 三线性插值噪声 | ✅ |
| `Aquifer` | `AquiferSampler` | 含水层 | ✅ 复用 MC |
| `Beardifier` | `StructureWeightSampler` | 结构地形整形 | ✅ 复用 MC（§5.6.4） |
| `EndIslands` | `0xA0` | 末地岛屿 | ✅ 已实现 + 数值验证 1.19e-7（§5.7） |

### 2c.3 地形完善

- **表面规则**：当前已由 vanilla `buildSurface` 阶段照常处理（我们只
  cancel `populateNoise`，不动表面阶段）——✅ 实测地表正常。
- **矿物 veins**：`vein_toggle/ridged/gap` 三通道已算但**未注入方块**，
  待接。

### 2c.4 性能与缓存

- **benchmark 失真**：用真实 router 跑 benchmark，才有可信吞吐量（§5.6.6）
- **beard 开销**：per-block ~98k 采样/chunk，可考虑只在含结构的 chunk 启用
- ~~shader 编译结果磁盘缓存~~ ✅ 已做（§5.7.8，仅覆盖 driver 阶段）
- WGSL 规模压缩（真实 ~5000 行）

### 2c.5 代码清理（已复核，2026-10）

原清单多数已过时/不适用，逐一核实结论：

- `WorldgenParams.batch_origin_x/z`、`batch_count_x/z`：**不能删**。
  shader 的 per-chunk origin 来自 `positions` buffer（binding 4），
  `batch_count` 用 `arrayLength(&positions)/2`——但这些字段**占据 GPU 结构体
  固定偏移**，删除会破坏后续字段布局。已改为 **LEGACY/RESERVED** 注释，
  仅作布局占位。
- `beardifier.rs`：**保留**。它是 `AstNode::Beardifier` 变体的 emitter
  （返回 0），与 AST enum / deserialize(0xA1) / hash 绑定；真实 blob 不会
  出现该节点（Java 侧序列化成 constant）。无害，不值得连 AST 一起动。
- `math_lib.wgsl` 的 `beardifier_adjust`：**已不存在**（早期已删）。
- `executor.rs::read_results`：`chunk_size_f32` / `channel_offsets` 是
  当前多块拆分的**实际逻辑**，正确，保留。**已删除**其中的 one-shot
  `[chan] min/max/mean` 诊断块（纯调试噪音）。

---

## §7 已知问题与注意事项

1. **C2ME DFC 会注入字段**：日志显示 `DoublePerlinNoiseSampler` 被 C2ME 加了 `c2me$arena`、`c2me$samplerData` 等字段（FFM native 缓存）。我们的 GPU 路径与它的 CPU native 路径并存，需注意不要互相干扰。

2. **诊断日志已清理**：`NoiseRouterBlobifier` 里残留的 `DBG`/`PERM collect`/`UNSUPPORTED NODE` 输出已移除（`UNSUPPORTED NODE` 保留，用于发现漏网节点）。

3. **blob 数据源 = `decompileRouter(noiseConfig.getNoiseRouter())`**（`MixinChunkGenerator`）。
   即把 c2me-opts-dfc 的 `CompiledDensityFunction` 剥回其 `blendingFallback`（原始未编译树），
   GPU 复现的才是 **MC 真正生成地形用的那棵树**。**切勿改回 `settings.noiseRouter()`** —— 那是
   另一棵不同的树（实测 `|settings-compiled|` 在 initialDensity 上达 16.8），会导致地形错误。
   （旧的 `getNoiseRouter(NoiseChunkGenerator)` 私有方法已删除。）

4. **本机 JDK**：`/usr/lib/jvm/java-25-openjdk`（toolchain）+ `/usr/lib/jvm/java-21-openjdk`（Gradle JVM）。

5. **测试实例**：`~/.minecraft/versions/1.21.1-DevMod/`，需 JVM 参数 `--enable-native-access=com.rubidiumlighting.c2me.opts.accel.wgpu`，且用 Java 25 启动。

6. **修改文件时注意**：代码含 `@Mixin`/`@Inject` 等注解，用 `@文件` 协议或 python heredoc 写入时需规避 `@` 前缀被误判为指令。

## §5.8 2e 里程碑 —— ch10/ch11 残差根因定位（已完成）

本轮目标：查清 §5.7.3 遗留的 ch10（`initialDensityWithoutJaggedness`，rms 4.6e-4）
与 ch11（`finalDensity`，rms 8.4e-3）残差。

### 5.8.1 排查过程（三个假设，逐一排除）

1. **cache 语义错？** → 排除。`Cache2D`/`CacheFlat`/`CacheInterpolated` 在 emitter 里
   全是 **pass-through**（逐角点求值），无语义偏差。
2. **深树独有节点 codegen 错？** → 排除。节点类型直方图显示 `Spline`（53 个，ch8）、
   `RangeChoice`/`InterpolatedNoiseSampler`（ch12/13）都已在 PASS 通道出现；ch10 相对 PASS
   通道唯一新增的是 `Min`/`Max`（源自 `Clamp`/`HALF_NEGATIVE` 的 Java 等价改写），核对后语义正确。
3. **blob 数据源错？** → **是真实存在的问题**，但**不是 ch10/11 残差的根因**（见 5.8.2）。

### 5.8.2 blob 数据源修正（保留的关键改动）

- **发现**：blob 从 `settings.noiseRouter()`（RAW）序列化，而 ref 从
  `noiseConfig.getNoiseRouter()`（c2me-opts-dfc compiled）采样。三方 probe
  （settings vs fallback vs compiled）显示：
  `|fallback-compiled| = 0.0`（JIT 不改语义），但 `|settings-compiled| = 16.8 / 0.85`
  （finalDensity）—— **`settings.noiseRouter()` 与真正生成地形用的树不是同一棵**。
- **修复**：新增 `decompileRouter()`，把 compiled 的每个 channel 经反射取
  `CompiledDensityFunction.getFallback()` 剥回原始树，作为 blob 源。
- **效果**：probe 全 0、`sameIdentity=true`；GPU 现在复现的是**正确的树**。
  但 ch10/11 残差**数值几乎不变**（ch10 逐位相同，ch11 8.4e-3 → 1.08e-2），
  证明残差另有原因。

### 5.8.3 决定性实验：CPU-f32 解释器三方对比

在 Rust 里实现 `cpu_ast_eval`（逐位镜像 GPU 的 fp32 语义），对同一棵 blob AST
在角点求值，与 GPU、ref(double) 三方对比：

| 通道 | \|cpu−ref\| rms | \|gpu−ref\| rms | **\|cpu−gpu\| rms** |
|---|---|---|---|
| ch10 | 4.651e-4 | 4.648e-4 | **1.467e-6** |
| ch11 | 1.079e-2 | 1.079e-2 | **8.822e-8** |

### 5.8.4 结论

- **CPU-f32 ≈ GPU**（差异在 f32 末几位，1e-6~1e-8）→ **codegen 完全忠实，无结构性 bug**。
- **CPU/GPU 都远离 double ref**（4.6e-4 / 1.08e-2）→ 残差 **100% 来自 fp32 vs double 精度差**，
  且随表达式深度单调增长（ch0-9 ~1e-6 → ch10 ~5e-4 → ch11 ~1e-2；ch11 含 2915 个噪声采样、
  1888 个 Shift、944 个 Spline，深度放大了累积误差）。
- 依据 **§0**：这是 fp32 的固有上限，**可接受，不是 bug**。地形判阈对此量级不敏感
  （方块级实测"与原版别无二致"）。
- **不接受"改 f64"** 作为常规解法（违背 §0 的"多数 GPU 含手机"目标）。

> ⚠️ **§5.9 修正**：此结论仅对 **ch10** 成立。ch11（含 `InterpolatedNoiseSampler`
> = `old_blended_noise`）的残差里还混着一个**真 bug**——octave 顺序反了（见 §5.9）。
> 当时 CPU 解释器"忠实"地复现了 WGSL 的同一错误，所以 `|cpu-gpu|` 很小，掩盖了 bug
> （又踩了"CPU≈GPU≠对 vanilla"的坑）。下界 ch11 用它暴露出来：rms 9.2e-2、符号相反。

### 5.8.5 复现命令

```bash
cd rust
cargo test --lib cpu_f32_vs_gpu -- --ignored --nocapture   # 三方对比（需 GPU）
cargo test --lib real_router_vs_vanilla -- --nocapture     # GPU vs ref（需 GPU）
```
（`cpu_f32_vs_gpu`、`gpu_performance_benchmark` 均已标 `#[ignore]`，默认 `cargo test` 不跑。）

---

## §5.9 2f 里程碑 —— 多维度 GPU + 三个正确性 bug（已完成）

### 5.9.1 多维度 WorldContext（per-dimension）

- **之前**：单例 `WORLD_HANDLE` / `COORDINATOR` + 静态 `GRID_*`，只加速
  Overworld 家族（`matchesSettings(OVERWORLD|LARGE_BIOMES|AMPLIFIED)`），
  Nether/End 走 vanilla。
- **现在**：`DimState`（worldHandle + coordinator + grid + `defaultBlock` +
  rawRouter），用 `ConcurrentHashMap<ChunkGeneratorSettings, DimState>` 按维度
  缓存，双检锁懒初始化（避免并发首块重复建 context 泄漏 Rust handle）。
- 移除维度守卫；三端共享 `initDimState`（shape 从 `generationShapeConfig()` 读）。
- **默认固体块**用 `settings.defaultBlock()`（Overworld stone / Nether
  netherrack / End end_stone），不再硬编码 `Blocks.STONE`——否则下界被填成石头。

### 5.9.2 blob 数据源 = 真正生成地形的树

- 见 §5.8.2：`decompileRouter(noiseConfig.getNoiseRouter())`，即
  `CompiledDensityFunction.getFallback()` 的原始树（`|fallback−compiled|=0`）。

### 5.9.3 InterpolatedNoiseSampler octave 顺序（关键正确性 bug）

- **根因**：MC `OctavePerlinNoiseSampler.getOctave(i)` 返回
  `octaveSamplers[len-1-i]`（**倒序**）；而普通噪声 `sample()` 循环是**正序**。
  `writeRawOctaveStack`（只被 InterpolatedNoiseSampler 调用）按数组正序存 perm，
  导致喂给 GPU 的 octave↔频率 配对反了。
- **为何主世界没暴露**：主世界 ch0-9 是普通 `DoublePerlin`（正序，正确）；
  ch11 含 `old_blended_noise`（4 个 InterpNoiseSampler），但被其他项掩盖成"精度差"。
  下界 `base_3d_noise` 几乎独占 ch11 → 暴露为符号相反。
- **修复**：`writeRawOctaveStack` 反序遍历 `arr`。Rust/WGSL 侧不动。
- **效果**：下界 finalDensity rms **9.2e-2 → 1.4e-6**（5 个数量级）。

### 5.9.4 SimplexNoiseSampler origin（字符串反射 bug）

- **根因**：`writeSimplexResource` 用 `c.getField("originX")` 字符串反射——
  运行时时 intermediary 映射，字段名不匹配 → 抛 `NoSuchFieldException` →
  catch 写 **512 个 0** → EndIslands 的 simplex perm 全清零 → 末地岛屿错乱。
  （违反项目约定"别字符串反射"。）
- **修复**：新建 `SimplexNoiseSamplerAccessor`（`@Accessor("permutation")`，
  Loom 重映射）取 private perm；`originX/Y/Z` 是 public 字段，直接访问。
- 注册进 `c2me-opts-accel-wgpu.mixins.json`。

### 5.9.5 EndIslands 环形条件反转（关键正确性 bug）

- **根因**：MC 只在 `nX*nX + nZ*nZ > 4096`（**外环**，nX=blockX/16 附近）时
  叠加岛屿凸起；我们写成 `> 4096 就 continue`，正好相反 → 凸起加到了内环、
  外环没有 → **外岛结构（末地城）浮在虚空上**。
- **修复**：WGSL `end_islands` 与 CPU `cpu_end_islands` 同步改为
  `<= 4096 就 continue`（只处理外环）。
- **注意（特性，勿修）**：MC 1.21.1 的距离首项 `100 - sqrt(x*x+z*z)*8` 里
  `x = blockX/8` 是 **int 乘法**（循环内才是 long），`x=46341`（世界坐标≈37万）
  时 `x*x` 溢出 → NaN → **虚空带**。这是原版特性，我们用 i32 会**忠实复现**，
  **不要改成 f32**。

### 5.9.6 确认"怪地形"= 原版行为

- 干净对照（物理移走 wgpu jar 跑 vanilla）：下界悬空方块、末地小岛重叠、
  46341 虚空带——**原版一模一样**。之前的怀疑（"怪 = GPU 错"）不成立。
- 提醒：`mods/` 里 jar 不移走，日志里看 mod 列表就知道跑没跑 GPU（别被
  "看着像原版"误导）。

### 5.9.7 关键教训（补充）

- **"CPU 解释器一致" ≠ "对 vanilla 正确"**：`cpu_ast_eval`/`cpu_end_islands`
  都是照 WGSL 抄的，WGSL 错则 CPU 一起错，测试照样过。真正判据只能是
  **与 MC 原版 dump 对比**（ref 用 `noiseConfig` compiled；结构对照用原版存档）。

---

## §5.10 2g 里程碑 —— NaN 语义 + 判据收尾（已完成）

### 5.10.1 末地远坐标 NaN（B 方案，已接受不完美）

- **现象**：`/tp` 到 370728（≈46341×8）附近，原版是虚空带，我们生成大片实心大陆。
- **根因**：MC `EndIslands.sample` 的距离首项 `x*x`（`x=blockX/8`）是 **int** 乘法，
  `x=46341` 时溢出回绕 → 有时为负 → `Math.sqrt(负数)=NaN` → 密度 NaN → 虚空。
  WGSL 的 `sqrt(负数)` **未定义**（本机 AMD/Mesa 返回 0）→ `100-0=100` → clamp 80 → 实心大陆。
- **修复（B）**：`end_islands` 里检测 `x*x+z*z`（i32 回绕）为负时**显式返回 NaN**。
- **效果**：`erosion`/ch10 的溢出 NaN 正确产生（both=99/93，mismatch 0），虚空带还原。
- **已知不完美**：ch11 `finalDensity` 的 NaN **未传播**（WGSL `squeeze` 的 `clamp(NaN)`
  不遵循 Java 语义）。彻底修需让整条链的 `clamp/max/min` 都按 Java 的 NaN 传播写（A 方案），
  成本高、收益极低（离出生点 37 万格，正常游玩不可达）。**接受。**
- 注意：`Math.sqrt(负数)` 在 MC 是 NaN、在 GPU 未定义——这是**平台差异**，非我们引入。

### 5.10.2 测试判据改进

- `real_router_vs_vanilla` 判据从 `max_diff < 1e-4` 改为 **`rms < 1e-3`**（对齐 §0 的
  fp32 预算）。`max_diff` 仍打印但不作判据——深表达式里个别 cell 因 f32 舍入可略超 1e-4，
  这是 §0 接受范围。**NaN 必须精确匹配**。
- 对比逻辑改为 **NaN-aware**：两边都 NaN 记匹配、单边 NaN 记 mismatch。
- 主世界复测：ch11 rms **1.08e-2 → 2.55e-5**（octave 修复对主世界同样生效，改善 ~420 倍），
  ch10 rms 4.6e-4（fp32 固有，§5.8）。**15 通道全部 OK。**

### 5.10.3 关键教训（再补）

- **平台未定义行为要显式兜底**：`sqrt(负数)`、`clamp(NaN)` 这类 Java 有定义、GPU 未定义的
  运算，是跨平台移植的隐蔽陷阱。
- **极端坐标是压力测试**：37 万格外暴露的 int 溢出 / NaN 语义，正常游玩碰不到，但值得记录。

---

## §5.11 2h 里程碑 —— 矿脉（ore veins）复用（已完成）

### 5.11.1 背景：为什么矿脉无法走角点网格

- MC 矿脉由 `OreVeinSampler.create(veinToggle, veinRidged, veinGap,
  oreRandomDeriver)` 产生，返回 **per-block 的 `BlockStateSampler`**：
  给位置 → 返回矿/矿脉石/`null`（无矿）。它依赖
  **`RandomSplitter.split(x,y,z)` 的确定性随机**，且 `vein_gap` 周期 1 格
  ——角点网格表达不了。

### 5.11.2 路线 1（直接复用，已否决）

- 在 Java 写回时对**每个实心方块**调 `oreVeinSampler.sample(pos)`。
- **实测 CPS 70 → 30（腰斩）**，且每方块重算 3 条 vein 密度（InterpolatedNoiseSampler
  等）——净负，与 §5.7.9 记录一致。
- 另有正确性缺陷：vanilla 传 `ChunkNoiseSampler` 走 **INTERPOLATION**，
  我们传 `UnblendedNoisePos` 走 **NORMAL**，语义不同。

### 5.11.3 路线 3（采用）：GPU 角点粗筛 + MC 精算

- GPU 已输出 `veinToggle`(ch12) 角点值。写回时**三线性插值**出方块的 `vt`，
  用 `|vt| >= 0.4` 粗筛（贴近 MC 的 `|vt|+yClamp<0.4` 判据）。
- **只有通过的方块**才调 MC 的 `OreVeinSampler`（`ChunkNoiseSamplerAccessor`
  取 `blockStateSampler`）。
- **Y 范围守卫**：只在 `-64 <= y <= 56`（铁矿脉 -60..-8、铜矿脉 0..48）内做粗筛，
  其余方块完全跳过插值。
- **实测**：矿脉正常生成（截图可见矿脉石+粗矿块）；CPS 50-87（均值 ~70），
  avgChunk ~4.5-5ms（基线 2.5-3ms）。相比路线 1 的腰斩，**可接受**。

### 5.11.4 已知保留（接受）

- 矿脉**位置未与 vanilla 逐块对齐**：粗筛用插值、精算用 NORMAL，与 vanilla 的
  INTERPOLATION 有细微差异。矿脉只影响矿物、不影响地形骨架，普通玩家不会察觉。
- 彻底对齐需把整条链（含 Xoroshiro128++ 随机）搬到 GPU 第二 dispatch——工作量大，
  收益低，暂不做。

### 5.11.5 关键教训

- **CPU 侧 per-block 复用 MC 采样器 = 性能灾难**（§5.11.2），粗筛是必要的。
- **热路径加 `nanoTime`+原子操作会拖垮性能**（阻碍 JIT 内联 + 线程争用），
  实测把 CPS 打到 17。计数用纯 `incrementAndGet` 尚可。

---

## §5.12 2i 里程碑 —— 前端通知 + 安全回退（已完成）

### 5.12.1 玩家可见通知（聊天栏）

- `ModuleEntryPoint` 从 `boolean` 升级为 **`Status`**：
  `OK / NATIVE_UNAVAILABLE / FFM_DISABLED / INIT_FAILED` + `detail`。
- 新增客户端 mixin `MixinMinecraftClientNotify`（注入 `MinecraftClient.joinWorld`
  的 RETURN）：
  - GPU 不可用时发聊天栏提示（含具体原因：FFM 参数缺失 / 平台无原生库 /
    wgpu 无可用适配器）
  - JVM 未用 ZGC 时提示一次（建议 `-XX:+UseZGC`）
  - **健康时不打扰**（GPU 正常 + ZGC 则静默）

### 5.12.2 GPU 初始化失败的安全回退（关键修复）

- **现象**：`VK_ICD_FILENAMES=/dev/null`（模拟无 Vulkan 驱动）时游戏**卡死**。
- **根因**：`WgpuThinBinding` 的 `<clinit>` 在创建 instance 失败时抛异常 →
  类被 JVM 永久标记为损坏 → 之后任何访问都抛 `ExceptionInInitializerError`
  → 回退守卫本身炸掉 → 异常逃逸到 chunk 生成线程 → 卡死。
- **修复**：`<clinit>` **永不抛出**——MH 句柄改非 final，link/instance 失败
  只记 `available=false`；`isAvailable()` 读该标志。
- `MixinChunkGenerator` 守卫改用 `ModuleEntryPoint.isEnabled()`（安全捕获原因）。
- `ModuleEntryPoint` 用 `WgpuThinBinding.isAvailable()` 区分"库加载了但无适配器"。

### 5.12.3 验证方法（模拟 GPU 不可用，不改代码/不卸驱动）

- 设环境变量 **`VK_ICD_FILENAMES=/dev/null`** 启动：Vulkan loader 找不到 ICD →
  wgpu 枚举不到适配器 → `INIT_FAILED`。
- **注意**：Java 24+ 放宽了 FFM 限制，即使不加 `--enable-native-access` 也能加载
  native，所以"删 FFM 参数"在 Java 25 下**不能**模拟失败。
- 实测：游戏正常回退 CPU、不卡死，聊天栏同时显示 GPU 失败 + ZGC 两条提示。

### 5.12.4 关键教训

- **`<clinit>` 抛异常 = 类永久损坏**：初始化逻辑必须捕获一切、绝不外抛，否则
  连"优雅降级"的守卫都不可用。
- **平台演进**：Java 24+ 的 FFM 宽松策略让"缺参数"不再是失败路径，测试要多想一层。

---

## §5.13 2j 里程碑 —— Android（aarch64）交叉编译（已完成）

### 5.13.1 目标

让同一份 jar 在 **Android 定制 JVM**（与 Linux 发行版尽量一致的虚拟机）上
也能加载 native 并走 GPU 路径。

### 5.13.2 关键约束：Android ≠ Linux

- Android 定制 JVM 的 `os.name` 仍是 `"Linux"`，但 **ABI 不兼容**：
  Android = bionic，桌面 Linux = glibc。**不能**复用 `natives/linux-<arch>/`。
- `NativeLoader` 新增 `detectAndroid()`（`ANDROID_ROOT`/`ANDROID_DATA` 环境变量，
  或 `/system/build.prop` 存在）→ OS 归一化为 **`android`** → 查
  `natives/android-<arch>/`。
- `normalizeArch` 补 `armv7` 映射（`armv7l`/`armeabi-v7a` 等）。

### 5.13.3 gradle 集成

- 新增 `rustBuildAndroid` 任务：调用 **`cargo ndk -t <abi> build --release`**，
  按 ABI 把 `.so` stage 到 `natives/android-<arch>/`。
- **自动启用**：`ANDROID_NDK_HOME` 存在时；`-PskipAndroid` 关闭。
- **ABI 可配**：`-PandroidAbis=arm64-v8a,armeabi-v7a`（默认 `arm64-v8a`）。
- ABI↔Rust target↔arch 映射表内置（arm64-v8a→aarch64，armeabi-v7a→armv7，
  x86_64→x86_64，x86→x86_32）。
- 与现有 `rustBuild`（桌面）并存：一个 jar 可同时含桌面 + Android 多份 `.so`。

### 5.13.4 环境踩坑：NDK 符号链接被写坏

- 症状：`cargo ndk` 报 `Failed to spawn .../clang: Exec format error`。
- 根因：NDK 的 `clang`、`clang++`、`ld`、`lld-link`、`wasm-ld` 等**符号链接**在
  复制/解压时被"去引用"，变成了 8 字节文本文件（内容如 `clang-18`）。
- 修复：对每个小文本文件 `rm` 后 `ln -s <cat 出的目标>` 重建链接。
- 验证：`clang --version` 正常 → `cargo ndk -t arm64-v8a build --release` 通过。

### 5.13.5 实机验证（已通过）

- ✅ **真机成功**：Kirin 980 / Mali-G57 上 GPU 路径跑通，聊天栏只有 ZGC 提示、
  **无**"wgpu 初始化失败"→ `ModuleEntryPoint` 判定 GPU = OK。
- 说明三关全过：`dlopen`（临时目录加载未被 SELinux 挡）、FFM 可用、wgpu
  选到适配器（大概率 Vulkan）。
- 实测 CPS < 20（Mali-G57 + Kirin 980，入门 GPU + 2018 CPU），属合理范围。

### 5.13.6 Android 上的 ZGC 提示已禁用

- **现象**：在 Android 定制 JVM 上启用 ZGC → 整个应用**瞬间闪退**。
- **结论**：ZGC 在安卓定制 JVM 上**不可用**，桌面向的 "建议 -XX:+UseZGC" 在
  安卓上是**有害建议**。
- **修复**：`MixinMinecraftClientNotify` 用 `NativeLoader.IS_ANDROID` 守卫，
  安卓上**不弹** ZGC 提示。

### 5.13.7 当前状态

- ✅ 交叉编译产出 `target/aarch64-linux-android/release/libc2me_wgpu.so`
- ✅ jar 内同时含 `natives/android-aarch64/` 与 `natives/linux-x86_64/`
- ✅ 真机（Kirin 980 / Mali-G57）验证通过，GPU 地形生成正常。

---

## §5.14 2k 里程碑 —— 内存泄漏修复（已完成）

背景：chunky 大规模生成时 RSS 涨到 10GB 后 OOM，CPS 掉到 1 以下。

### 5.14.1 根因一：result blob 每 chunk 泄漏（native，已修）

- `ResultBlob::into_raw_parts` 用 `std::mem::forget(self.data)` 把 Vec **永久泄漏**，
  且全项目**没有任何释放函数**（代码注释自称 "TODO: real ownership via Arc+Drop"）。
- 量级：每 chunk ≈ 16×2425×4 ≈ 155KB；6.5 万 chunk ≈ **10GB**，与观测吻合。
- **修复（方案 A：拷贝后释放）**：
  - Rust：`into_raw_parts` 改转移 `Box<[u8]>`（返回 `*mut u8`）；新增
    `free_raw_parts`；新增 C API `c2me_wgpu_free_result_blob(ptr, len)`。
  - Java：`WgpuThinBinding.takeResultBlob` 把 native blob 拷进 Java 堆 `byte[]`，
    **立即**调 free 释放 native，返回 `MemorySegment.ofArray(copy)`。下游零改动
    （堆 segment 支持 `asByteBuffer()`/`byteSize()`）。
- 效果：RSS 从 10GB 降到**稳定 4.5GB**。

### 5.14.2 根因二：executor 每 batch 重建 buffer（已修）

- `dispatch()` 每 batch 新建 `params_buf`/`positions_buf`/`output_buf`，只有
  `staging` 复用（`StagingChain`）。wgpu 延迟销毁，慢 GPU 上未回收的 buffer 堆积。
- **修复**：三者改为持久 buffer（构造时按 `max_batch_size` 分配一次），
  dispatch 时 `queue.write_buffer` 更新 params/positions，output 直接复用。

### 5.14.3 遗留 OOM 是 Java 堆太小（非本模块问题）

- 修复后仍 OOM，但性质变了：`OutOfMemoryError: Java heap space`，崩在 **C2ME 自己**
  的 `ScopedValueEmulation`（chunk 调度），堆上限仅 3.67GB。
- 把 `-Xmx` 提到 11GB 后：**不崩、CPS 不降** → 确认是堆不够（chunky 2048 = 6.5 万
  chunk 的极限压力测试），非本模块泄漏。
- **结论**：本模块的内存泄漏已修复；大规模生成需给足 Java 堆。

---

## §5.15 2l 里程碑 —— Mali-G610 shader 规模崩溃定位（排查完成，拆分待做）

背景：天玑 8100（Mali-G610 MC6）上 GPU 路径**崩溃**，而 Kirin 980（Mali-G76）
正常。本轮把崩溃根因逐层定位到**驱动编译大规模 compute shader 会设备丢失**。

### 5.15.1 排查链（依次排除，全部有实测证据）

1. **`mapped_at_creation` 驱动 bug？** → 不是。
   最初崩溃报 `Failed to get mapped range ... Buffer 'c2me_wgpu_permutations' is invalid`。
   加 `create_init_buffer`（Android 默认走 `create_buffer` + `queue.write_buffer`）后，
   崩溃点**不变**（仍在 perm buffer），说明问题不在上传方式。
   （注：这个修复本身仍有价值，见 5.15.4。）

2. **PipelineCache？** → 不是主因（**后更正：cache 本身完全可用**）。
   加 `C2ME_WGPU_NO_PIPELINE_CACHE=1` 跳过 `create_pipeline_cache` 后，仍崩
   ——故当时怀疑 cache。**拆分后复测：开 cache 全程无崩溃，见 §5.15.5**。
   真正诱因始终是 shader 规模，与 cache 无关。该开关降级为纯诊断用。

3. **wgpu validation 错误？** → 抓不到。
   装 `device.on_uncaptured_error` + `C2ME_WGPU_DEBUG_LOG` 写文件后确认：
   **崩溃是 Mali 驱动层的硬 abort**，不是 wgpu validation error。
   日志能证明设备在 `create_compute_pipeline` 之后进入**invalid 状态**
   （后续 `create_buffer` 全部返回 invalid buffer）。

4. **ANGLE / 多 Vulkan 实例干扰？** → 排除。
   把 FCL 渲染器从 MG(ANGLE) 换成 Holy GL4ES，编译时间**几乎不变**
   （17.0s → 14.7s），崩溃依旧。

5. **最小 shader 试验** → 锁定规模。
   `C2ME_WGPU_MINIMAL_SHADER=1` 发一个 31 行的结构兼容 shader：
   **编译 8ms、全链路（含 buffer 上传）通过**。
   完整 shader（5383 行）：`create_compute_pipeline` 耗时 13–17s，之后崩。

### 5.15.2 关键数据：编译时间对 shader 规模**超线性**

本地（AMD）用 `C2ME_WGPU_MAX_CHANNELS=N` 测缩放：

| channels | lines | shader_module | compute_pipeline |
|---|---|---|---|
| 1 | 236 | 7.9 ms | 17 ms |
| 4 | 347 | 14.8 ms | 44 ms |
| 8 | 628 | 44 ms | 176 ms |
| 10 | 1650 | 494 ms | 327 ms |
| 11 | 2153 | 905 ms | 400 ms |
| **12** | **4622** | **4.08 s** | **2.69 s** |
| 15 | 4781 | 4.42 s | 2.98 s |

**悬崖在 ch11（finalDensity）**：一个通道就 **+2469 行**，把总量从 2153 →
4622，编译成本跳 5 倍。前 11 个通道合计仅 2153 行。

天玑实测（同一 jar，`MAX_CHANNELS`）：

| 配置 | lines | Mali-G610 编译 | 结果 |
|---|---|---|---|
| 11 通道 | 2153 | **1.64 s** | ✅ 稳定 |
| 12 通道 | 4622 | **15.9 s** | ⚠️ 勉强（曾崩过） |
| 15 通道 | 4781 | 14–17 s | ❌ 设备丢失 |

**结论**：Mali-G610 能编译 ~2100 行（1.6s），但 ~4600 行需 16s 且**不稳定**
（同样规模有时成功有时崩）——崩溃与**编译耗时/热状态**相关，非硬行数。

### 5.15.3 附带发现：shader 规模也拖累执行

同条件（都开性能模式）CPS：

| 配置 | CPS 均值 |
|---|---|
| 11 通道（2153 行） | **13.4** |
| 12 通道（4622 行） | ~10 |

即小 shader **既好编译又好执行**。拆分方案是一箭双雕。

### 5.15.4 已落地的改动（本轮，不含拆分）

- **`create_init_buffer`**：Android 默认 `create_buffer` + `queue.write_buffer`
  （避开部分 Mali 的 `mapped_at_creation` 问题），桌面默认 `create_buffer_init`。
  用 `C2ME_WGPU_MAPPED_AT_CREATION=0/1` 覆盖。
- **`C2ME_WGPU_NO_PIPELINE_CACHE=1`**：跳过 PipelineCache（含条件化
  `Features::PIPELINE_CACHE` 检查与请求）。
- **`device.on_uncaptured_error`**：wgpu 错误经我们的日志输出，而非跨 FFM 边界 abort。
- **`C2ME_WGPU_DEBUG_LOG=<path>`**：Rust 诊断日志写文件（Android 的
  `latest.log` 只抓 stdout，stderr 在崩溃时易丢失）。
- **math_lib 按需裁剪**：只拼接 AST 实际引用到的 helper。
  math_lib 860 → 258 行，总 shader 5383 → 4781 行。**数值零影响**（15 通道
  验证一致，ch11 rms 2.55e-5）。但**未解决 G610 崩溃**（瓶颈在通道正文，不在库）。
- **`C2ME_WGPU_MINIMAL_SHADER=1`**：结构兼容的 31 行 bring-up shader。
- **`C2ME_WGPU_MAX_CHANNELS=N`**：只发前 N 个通道（诊断用）。
- 编译步骤细粒度日志（bind group / pipeline / 每个 buffer）。

### 5.15.5 已完成：按通道拆分 shader（统一拆，桌面+Android）

设计：把 15 通道拆成多片，每片一个独立 `main_entry` + pipeline，
写**同一个 out_buf**（通道索引用全局号，输出布局不变）：

| 片 | 通道 | 计划行数 | **实测行数** | 计划编译 | **Mali-G610 实测** |
|---|---|---|---|---|---|
| A | ch0–10 | 2153 | **2142** | ~1.6 s | **1.15 s** ✅ |
| B | ch11 (finalDensity) | 2469 | **3083** | ~2 s | **9.6–10.2 s** ⚠️ |
| C | ch12–14 (vein×3) | ~160 | **355** | 瞬时 | **0.20 s** ✅ |

**实现**（提交见 git log）：
- `codegen/wgsl_gen`：`Pass` 增 `C`；新增 `Shard { pass, start, end }` +
  `plan_shards()`；`GenCtx.channel_range`；`emit_shader_for_shard()`；
  `MAIN_PASS_C_PROLOGUE`。
- `codegen/emitters/root`：按 `channel_range` 切片，写 out_buf 用**全局通道号**。
- `world/world`：`shader_src` → `shader_srcs: Vec<String>`、`pipeline` →
  `pipelines: Vec<_>`。
- `scheduler/executor`：对多 pipeline 依次 `set_pipeline` + dispatch（bind group 复用）。
- `world/pipeline_cache`：`pass_id(Pass::C) = 3`。
- 新增 `C2ME_WGPU_FORCE_SINGLE=1`（回退到拆前单 shader，兼对照基准）。

**偏差根因**：计划里的行数（2153/2469/160）只算了**通道正文**；每片 shader 会
各自重新带上裁剪后的 `math_lib`（~258 行）+ `BINDINGS_HEADER` + prologue
（~150 行）。A 正文小、加开销后 2142（恰在安全点）；B 正文 2469 + 库/头/序
~614 = **3083**，比安全区大 43% → 编译 9.6–10.2s（安全区 ~1.6s）。

**实机验证（天玑 8100 / Mali-G610）**：
- ✅ **不再崩溃**：15 通道三片全部编译成功、buffer 上传正常、稳定运行。
  对比拆前 15 通道单 shader（14–17s + 随机设备丢失）。
- ✅ **拆分零执行代价**：桌面 RADV 对照（15 通道全量）——
  3 片 **402.5 区块/秒** vs 单 shader **395.5 区块/秒**（差 1.7%，噪声内）。
- ✅ **CPS 非回退**：Android 逐档 `MAX_CHANNELS` 实测
  11ch ≈ **13** → 12ch ≈ **10.9** → 15ch ≈ **7**；
  拆前 11ch 基线 13.4、12ch ~10，与拆后一致。每加通道单调下降 = 真实工作量，
  而非拆分开销（拆前从未在天玑跑通 15ch，无「应达到」的基线）。

**pipeline cache 复测（拆除 `NO_PIPELINE_CACHE` 后，天玑 8100）**：
- ✅ **首次进世界**：三片热编译 A=1.14s / B=9.77s / C=0.23s，
  cache 边编译边落盘（`saved pipeline cache 1697c49a`，260KB 收敛），**无崩溃**。
- ✅ **冷启动**（杀进程重进同世界）：三片全部磁盘 cache 命中——
  A=24.8ms / **B=20.0ms** / C=18.0ms。**B 的 10s 只需付一次。**
- 结论：`C2ME_WGPU_NO_PIPELINE_CACHE` 退役为诊断开关；B 的编译耗时
  由磁盘 cache 缓解，不再是常驻风险。
- math_lib 冗余排查：B 比 A 多 128 行库/头/序，来自 5 个 B 独有 helper
  （`scale_tunnels` / `scale_caves` / `perlin_ysmear` / `interp_noise_sampler`
  / `squeeze`），**全部真被正文调用，无可裁剪冗余**。B 的 3083 行是实打实的。

### 5.15.6 Android 实机数据（更正 §5.13.5）

| 设备 | GPU | 性能模式 | CPS | 内存 | 多人 |
|---|---|---|---|---|---|
| Kirin 980 | **Mali-G76**（此前误记 G57） | 开 | **~14（峰值 20–22）** | 6200 MB 稳定 | 不崩 |
| 天玑 8100 | Mali-G610 MC6 | 开 | ~13（11 通道）/ ~10（12 通道） | — | 需拆分后才能稳 |

> 注：G76 与 G610 是不同架构（Bifrost vs Valhall）。`mapped_at_creation` 的
> 问题只在 G610 观测到，G76 无此问题。
>
> **拆分后三端复测（§5.15.5）**：麒麟 980 裸跑无回归（~14 CPS，峰值 20–22）；
> 桌面 RADV 峰值 144 CPS；天玑 8100 稳定运行。

### 5.15.7 关键教训

- **驱动层崩溃抓不到 stack**：wgpu `on_uncaptured_error` 只覆盖 validation error；
  Mali 的设备丢失表现为"对象创建成功但内部 invalid"，只能靠**分段日志**定位。
- **Android 上 Rust stderr 会丢**：`latest.log` 只抓 stdout；崩溃时缓冲不刷盘。
  必须**写文件**（`C2ME_WGPU_DEBUG_LOG`）才能拿到崩溃前最后的日志。
- **shader 编译成本超线性**：不能线性外推"减 10% 行数 = 减 10% 时间"。
  拆小是唯一可靠解法。
- **同一规模结果随机**（12 通道有时成有时崩）→ 与编译耗时/热状态相关，
  阈值不是固定行数。安全区取 11 通道的 1.6s 量级。


---

## §5.16 2m 里程碑 —— 只算/只回读被消费的通道（2026-10）

### 5.16.1 动机：16 通道里只有 2 个有人读

`MixinChunkGenerator.writeDensityToChunk` 只消费两个通道：
- **ch11 `finalDensity`** —— 实心/空气判定，兼作 `AquiferSampler` 的输入；
- **ch12 `veinToggle`** —— 矿脉粗筛。

`AquiferSampler` 与 `StructureWeightSampler` 都用 MC 自己的 CPU 采样器
（§5.6.4 / §5.5.5），**不读** GPU 的 barrier / fluid / lava 通道。其余
14 个通道算了、传了、却从没有人读。而 `CacheInterpolated` 早已改为直通
（§5.6.2），shard A 当年存在的唯一理由（跨片共享内存）也已消失。

### 5.16.2 第一刀：只 emit 被消费的通道

`wgsl_gen::plan_shards` 从「A=[0,11) + B=[11,12) + C=[12,16)」改为
「B=[11,12) + C=[12,13)」：

| shard | 旧 | 新 |
|---|---|---|
| A [0..11) | 2142–3364 行 | **删除** |
| B [11..12) | 2503 行 | 2503 行（逐字节不变）|
| C | [12..15) 356 行 | [12..13) 237 行 |

实测同一份 `real_router.bin`：
- **旧**：shard A 3364 行 + B 2503 行 + C 356 行 = **6223 行**
- **新**：shard B 2503 行 + C 237 行 = **2740 行（-56%）**

`out_buf` 仍按**全局通道号**写（ch11 落在 slot 11），所以 Java 侧偏移
不用改。小测试 blob（通道数 ≤ 11）自动回退到 `Pass::Single` 单 shader，
保证单元测试仍能触及所有 emitter。

天玑 8100 上 shard A 的 ~1.15s 编译随之消失——那是唯一每次启动都要付的
编译（shard B 有磁盘 cache，§5.15.5）。

### 5.16.3 第二刀：只回读被消费的通道

shader 与 `output_buf` **不动**（16 通道照写，pipeline cache 保持有效），
只改回读：

- `WorldContext::readback_channels()` 从 `shards` 推导要回读的全局通道
  （真实 router = `[11,12]`）。
- executor 的 copy 循环按这些通道从每 chunk 的 `output_buf` 里各拷一段
  到紧凑 staging（ch11+ch12 相邻 → 每 chunk 实际一次 2×CPC 连续拷贝）。
- 结果 blob 从 **16 通道降到 2 通道**：每 chunk 155 KB → **19 KB（-87.5%）**。

`staging` 仍按 16 通道宽度预留（executor 跨 world 复用，Single 模式可能回读
更多通道），但每批只 map/copy `total_bytes`，所以回读节省不受影响。

### 5.16.4 实测（桌面 RADV，真实 router，带 Epic Terrain）

| 指标 | 值 |
|---|---|
| shader 总行数 | **6223 → 2740（-56%）** |
| 回读 blob | 155 KB → **19 KB/chunk（-87.5%）** |
| profile `readback` | ~3.1–4.0 ms（旧 §12.2 记录 5–20 ms）|
| profile `decode` | 0.01 ms（紧凑布局读回未错位）|
| CPS | 稳定段 **75–86**（峰值 86.5）|
| 回退/报错 | **无** |

**地形正确性**：维护者目视确认——连已知的极端坐标"怪坑"（§5.9.6 原版行为）
都在，与 vanilla 一致。

### 5.16.5 附带修复：一个从不断言的测试

`real_router_vs_vanilla` 原来对超出 fp32 预算的通道只 `eprintln!` 警告、
**从不 assert**——等于永远"通过"。本次改为只校验真正被消费的
ch11/ch12，并加 `assert!`。实测 ch11 rms **2.55e-5**、ch12 rms **2.22e-6**，
均在 fp32 预算内。

### 5.16.6 关键教训

- **先数消费者，再决定算多少**：14/16 通道的算力与回读量被浪费了很久，
  因为"16 通道"是早期架构的默认，没人回头核对谁在真正读。
- **裁剪要保住协议**：out_buf 全局槽位不变，只动 emit 范围与回读拷贝，
  Java 布局零改动——这让第一刀几乎无风险。
- **回读与计算可分离优化**：shader 照写 16 通道（cache 有效），只在回读
  阶段挑 2 个通道，两刀互不耦合。
- **假通过的测试比没有测试更危险**：`eprintln` 而非 `assert` 的判据会
  在回归发生时保持沉默。

### 5.16.7 诚实的边界

CPS **不会**因这两刀提升——§15.7 已证 GPU 富余 6 倍，瓶颈是 CPU 逐方块
写回（aquifer/heightmap/vein）。本轮的收益是 **shader 编译时间、回读带宽、
显存/功耗**，不是吞吐。要动 CPS 只能碰 §15.7 定位的三条 CPU 写回路径，
而那三条已逐一验证排除。


---

## §5.17 2n 里程碑 —— 群系气候 GPU 加速（design 2，2026-10）

> **代码位置：`experiment/biome-gpu` 分支。** main 上仅保留本节 + §16 的
> 记录。结论（§16.6）：功能正确但**净负优化**（关键路径延迟），默认关闭。

### 5.17.1 动机与可行性

MC 的群系放置用 `MultiNoiseBiomeSource` + 六参数气候采样
（temperature/humidity/continentalness/erosion/depth/weirdness）。这六个
参数正是 NoiseRouter 的 ch4–9（temperature/vegetation/continents/erosion/
depth/ridges），**emitter 早已存在**，§5.10.2 验证过。GPU 化可行性成立
（C2ME OpenCL 模块也做过同一件事）。

### 5.17.2 关键事实：BIOMES 在 NOISE 之前

从 1.21.1 字节码确认 `ChunkStatus` 依赖链：
EMPTY → STRUCTURE_STARTS → STRUCTURE_REFERENCES → BIOMES → NOISE → …
即 `populateBiomes` **先于** `populateNoise`。群系**不能**搭地形的顺风车
（地形请求还没来）。这决定了架构：**两条独立 GPU 路径**（design 2）。

### 5.17.3 量化公式：一行

MC `MultiNoiseSampler.sample(x,y,z)` 反编译后等价于：

    NoisePos pos = new UnblendedNoisePos(BiomeCoords.toBlock(x), ...);
    float t = (float) temperature.sample(pos);   // 六个通道各一次
    ... createNoiseValuePoint(t,h,c,e,d,w);      // toLong(f) = (long)(f*10000)

`BiomeCoords.toBlock(i) = i << 2`，即**群系格 = 每 4 方块**（三轴）。
没有查表、没有非线性、没有随机——这是"可行"最硬的依据。C2ME OpenCL
承认的"边界偏移一两格"，很可能就是没复现这个 ×10000 截断。

### 5.17.4 实现（Rust）

- 新增 `Pass::D`（群系 shard），emit ch4–9；`ShardMode` 三值：
  `Terrain`（B/C）、`BiomeOnly`（只 D）、`TerrainPlusBiome`（测试用）。
- **Y 网格独立**：群系 Y cell = 4（`BIOME_CELL_HEIGHT`），地形是 8。两段
  在 `out_buf` 分开：terrain 段 `OUT_BUF_CHANNELS=16` 槽/chunk，biome 段
  6 槽/chunk。段基址经 `params._pad` 传入，WGSL 无字面量。
- 新 C API：`c2me_wgpu_world_new_biome` / `c2me_wgpu_request_biomes` /
  `c2me_wgpu_biome_channel_count`。
- **数值验证**（`biome_climate_vs_vanilla`）：GPU 与 vanilla 差
  **1.6e-5 ~ 4.9e-5**（fp32 预算内）；量化 delta ≤ 1（MC 用 double→float，
  我们全程 fp32，±1 固有）。

### 5.17.5 实现（Java）

- `BiomeAccel`：per-dimension biome-only WorldContext + coordinator，门控
  `C2ME_WGPU_BIOME=1`（默认关）。
- `MixinChunkGeneratorBiomes`：inject `populateBiomes` HEAD，用 GPU 六参数
  构造 `NoiseValuePoint` 喂 `mns.getBiomeAtPoint(...)`，复用 MC 自己的群系
  容器填充（`chunk.populateBiomes(supplier, sampler)`）。
- **target 必须是 `NoiseChunkGenerator`，不是 `ChunkGenerator`**：子类
  override 了 `populateBiomes`，挂基类永不触发（第一次测试踩到）。

### 5.17.6 修复的真 bug

1. **`num_chunks` 用错来源**（关键）：shader 里
   `num_chunks = arrayLength(&positions)/2`。但 `positions_buf` 按
   `max_batch_size=64` 持久分配 → `arrayLength` 恒 128 → 恒认为 64 个
   chunk。地形 shard 不受影响（基址不含 num_chunks），但 biome 段基址
   `num_chunks * …` 差一个数量级，读到全 0。改用 `params.batch_count_x`。
2. **`PendingQueue` 全局单例**：terrain 和 biome 共用一个队列 → flush
   互相窃取请求 + `PendingQueue overflow`。改为**每 world 一份**
   （`WorldContext.pending_queue`）。

### 5.17.7 实测（桌面 RADV）

| 指标 | 值 |
|---|---|
| shard D | 1426 行 |
| biome grid | 5×129×5（h=512, cell=4）|
| biome-batch CPS | 70–85 |
| terrain CPS | 65–75 |
| 溢出/回退 | 0 |

群系目视与 vanilla 一致（边界锐利、无系统性偏移）。

### 5.17.8 关键教训

- **override 陷阱**：mixin 挂基类，实际调用的是子类 override。terrain 早已
  挂 `NoiseChunkGenerator`，biome 却挂了 `ChunkGenerator`——同类错误。
- **全局单例在多 world 下必炸**：队列、executor 都必须 per-world 或明确分片。
- **`arrayLength` 反映容量不反映活批**：持久 buffer 场景下不能当批大小用。


## §8 待办（更新）

- ~~ch10/ch11 残差~~ ✅ ch10 = fp32 固有（§5.8）；ch11 的额外残差 = octave 顺序 bug（§5.9.3），已修。
- ~~多维度实例~~ ✅ per-dimension `DimState`，Nether/End 走 GPU（§5.9.1）。
- ~~末地 EndIslands~~ ✅ simplex origin + 环形条件修复（§5.9.4/5.9.5）。
- ~~矿脉 veins~~ ✅ 路线 3：GPU 角点粗筛 + MC OreVeinSampler 精算（§5.11），CPS ~70 可接受。位置未与 vanilla 逐块对齐（接受）。
- ~~末地 ch10 残余残差~~ ✅ 复测：主世界 15 通道全 OK（§5.10.2）；远坐标 NaN 用 B 方案部分还原（§5.10.1，接受）。
- ~~地形边界截断~~ 🟡 **关闭（未复现）**：多块 `positions` 重构后未再观察到；目视与原版一致。低优先，暂不追。
- ~~末地 ch11 NaN 传播~~ 🟡 **接受（不做）**：A 方案（全链 NaN 语义）成本高、收益极低（仅 ~37 万格外触发）；当前 B 方案已还原虚空带。
- ~~代码清理~~ ✅ 已复核（§2c.5）：可清的已清（诊断块、LEGACY 注释），其余为结构布局必需，保留。

- ~~Mali-G610（天玑 8100）shader 规模崩溃~~ ✅ 已按通道拆 shader（§5.15.5）：
  三片 A/B/C，实机不再崩溃，拆分零执行代价，桌面无退化（402.5 vs 395.5 CPS）。
  shard B 首次编译 ~10s 已由磁盘 pipeline cache 缓解（冷启动 ~20ms）；
  B 的 3083 行经查无冗余可裁。**Mali-G610 遗留风险清零。**
- ~~桌面 RADV + 麒麟 980 复测~~ ✅ 桌面峰值 144 CPS；麒麟 980 裸跑无回归
  （~14 CPS，峰值 20–22）。三端全部通过。

- ~~只算/只回读被消费的通道~~ ✅ 已做（§5.16）：shader 6223→2740 行（-56%），
  回读 blob 155KB→19KB/chunk（-87.5%）；CPS 不变（瓶颈是 CPU 写回，§15.7），
  收益在编译时间/带宽/功耗。桌面实测稳定 75–86 CPS，地形目视与 vanilla 一致。

**§8 至此清空。** 后续新方向按需开新章节。

---

## §9 发布

### v0.1.2（待发 Modrinth）

**相对 0.1.1 的新增**：

1. **Tectonic `Invert` 节点直接支持**（§11.2）：映射为 `Div(Constant(1.0), input)`，
   实机验证 GPU 路径启用、地形正确。
2. **编译期与 C2ME 解耦**（§10）：移除复合构建 / `compileOnly` 依赖 /
   借用的 `c2me-base.accesswidener`；自写 `c2me-opts-accel-wgpu.accesswidener`
   （~45 行）；删除 3 个从未引用的库（asyncutil / night-config / mixinsquared）。
   C2ME 仍为运行时硬依赖。
3. **混合执行**（§11）：无法翻译的密度子树交 CPU 在角点网格求值、作为外部
   输入（binding 6）喂给 GPU，不再整体回退。任意 modded 密度函数可共存。
4. **Windows 支持**：build.gradle 检测到 windows rust target 时自动交叉编译
   DLL，jar 现含 `natives/windows-x86_64/c2me_wgpu.dll`。

**三端验证**：
| 平台 | 结果 |
|---|---|
| Linux (RADV) | ✅ 功能正常 |
| Windows (AMD 驱动, mingw DLL) | ✅ 功能正常，原版 CPS avg 89.5 / max 123.8 |
| Android | ⏳ 待测（无设备） |

**tag**：`v0.1.2` → 本轮发布提交。

### v0.1.1（Modrinth 首个公开版本）

> Modrinth 规则：版本上传后**主文件不可替换**。0.1.0 已上传（含下述
> 兼容性 bug），故以 **0.1.1** 重新发布修复后的 jar。

**修复（相对 0.1.0）**：加入 Tectonic 等引入自定义密度函数节点的
数据包时暴露的两个问题——
1. **静默错误地形**：遇到不认识的节点（如 Tectonic 的 `Invert`）时，
   旧代码打印警告后**写常量 0 继续**，若 buffer 恰好够大就会让 GPU
   生成错误地形。当时改为**抛异常硬失败**、干净回退 CPU；
   **后续（§11）升级为「混合执行」**：该子树交 CPU 求值、作为外部输入
   喂给 GPU，不再整体回退。
2. **日志爆炸 + 无提示**：回退后每个区块重试、每次重刷整段节点反射
   dump，`latest.log` 涨到 **615 MB**，且玩家在游戏内看不到任何提示。
   改为：失败维度永久记入 `FAILED_DIMS` 不再重试；`UNSUPPORTED NODE`
   每类只 dump 一次；新增线程安全的 `WgpuFallbackNotice` + client tick
   消费，在聊天栏提示回退原因。
3. 密度树缓冲区 **4 MiB → 32 MiB**（大型数据包溢出会触发
   `BufferOverflowException`）。

**实测**（Tectonic + lithostitched）：日志 **615 MB → 48 KB**，
`UNSUPPORTED NODE` / `world init failed` 各 1 次，聊天栏正确提示。

**遗留**：~~Tectonic 目前仍走 CPU~~ → 已解决：`Invert` 节点直接支持（§11.2），
其余任意未知节点走混合执行（§11）。

### v0.1.0（首个版本，已被 0.1.1 取代）

- **Modrinth**：已提交，**审核中**（首次项目人工审核）。
  - 项目：`C2ME-wgpu-rs`（slug `c2me-wgpu-rs`）。
  - 平台：Fabric 1.21.1，Loader ≥ 0.17.2，Java ≥ 25。
  - 依赖：**C2ME**（required）。
  - 分类：Optimization / Worldgen / Technology；环境 Client + Server；License MIT。
- **CurseForge**：暂不发布。
- **仓库**：已公开 https://gitee.com/rubidium-lighting/C2ME-wgpu-rs
- **tag**：`v0.1.0` → commit `ee5c4c8`；`v0.1.1` → 修复提交（已推 Gitee）。

三端实测：
| 平台 | 结果 |
|---|---|
| 桌面 AMD RADV | 峰值 144 CPS；拆分零退化（402.5 vs 395.5 区块/秒） |
| 天玑 8100（Mali-G610） | 15 通道稳定不崩；磁盘 cache 冷启动 B 从 9.77s → 20ms |
| 麒麟 980（Mali-G76） | 裸跑无回归，~14 CPS（峰值 20–22） |

### 发布流程备忘

1. 构建：`ANDROID_NDK_HOME=<ndk> ./gradlew build`（自动交叉编译 aarch64 native）。
2. 产物：`build/libs/c2me-opts-accel-wgpu-mc1.21.1-<ver>.jar`。
3. 建 tag：`git tag -a v<ver> -m "..."`，push。
4. Modrinth：dashboard → 项目 → Versions → Create version，上传 jar，
   Loader=Fabric、MC=1.21.1、依赖 C2ME=required。

---

## §10 与 C2ME 的关系（构建解耦 + 运行时依赖）

### 10.1 结论

- **编译期：无依赖**。Java 代码零 C2ME 引用（无 `com.ishland` import）；
  已移除复合构建、`compileOnly` 依赖、以及借用的 `c2me-base.accesswidener`。
- **运行时：硬依赖**（`fabric.mod.json` 的 `"c2me": "*"`）。
- **名字保留** `C2ME-wgpu-rs`：定位就是 C2ME 的 GPU 加速配套模块。

### 10.2 为什么运行时仍依赖 C2ME（实测）

移除 C2ME 后（自写 AW 已就位、功能不崩），桌面 RADV 实测：

| 环境 | CPS |
|---|---|
| 桌面 + C2ME | **峰值 144** |
| 桌面 − C2ME | **平均 13.0**（≈ 麒麟 980 的 ~14） |

根因：C2ME 提供**多线程并行区块生成**。没有它，原版基本主线程串行产出
区块请求，`WgpuBatchCoordinator` 攒不满批 → 小 batch 频繁 dispatch →
GPU 利用率极低。**「C2ME 提供并行度，我们提供 GPU 算力」** 缺一不可。

### 10.3 为什么自写 AW（不再借 C2ME 的）

`build.gradle` 曾直接指向 `../C2ME-fabric/c2me-base/.../c2me-base.accesswidener`
（编译期），运行期靠 C2ME 的 AW 生效。问题：**把可访问性契约寄生在 C2ME 上**
——C2ME 一旦删掉某个它不再需要的条目，我们即 `IllegalAccessError`（实测移除
C2ME 时确曾触发）。现改为 `src/main/resources/c2me-opts-accel-wgpu.accesswidener`
（~45 行，只放宽我们真正访问的 `DensityFunctionTypes$*` 与少量字段/构造），
契约归自己，只随 Minecraft 版本变。

### 10.4 为什么不自己实现并行（评估）

C2ME 并行核心模块规模（`.java` 行数）：

| 模块 | 文件 | 行数 | 作用 |
|---|---|---|---|
| FlowSched | 29 | 2784 | 任务调度框架（工作线程池 / 依赖 DAG） |
| c2me-rewrites-chunk-system | 55 | 4169 | 重写 MC 区块系统为并行 |
| c2me-opts-scheduling | 26 | 696 | 调度优化 |
| c2me-opts-worldgen-general | 7 | 191 | 世界生成并行 |
| c2me-base | 107 | 3660 | 基础设施（含调度/线程工具） |

核心合计 **~11,000+ 行**，且深度耦合 MC 内部（替换 `ChunkHolder` /
`ChunkStatus` / `ThreadedAnvilChunkStorage` 状态机），含大量线程安全 bug 修复
mixin。复刻 = 重走 C2ME 的踩坑史 + 接下每个 MC 大版本的维护负担，投入产出比
极差。**保留 C2ME 运行时依赖是理性选择。**

### 10.5 采样器优化已"白嫖"

`settings.noiseRouter()` 是**原始可序列化树**，其 Noise 的 sampler 为 null；
我们用 `NoiseConfig.getOrCreateSampler(key)` 现场生成。这个 sampler 与 C2ME
DFC 用的是**同一个** `NoiseConfig` 产出，**已带 C2ME 的增强**（`c2me$arena`
等注入字段）——我们无需 import C2ME 任何类即可吃到其采样器优化。

注：`noiseConfig.getNoiseRouter()` 是 c2me-opts-dfc **编译后的 JVM 字节码**
（`CompiledDensityFunction`），**不可序列化**——这正是 §3.3 放弃它、改从
`settings` 取树的原因。

---

## §11 混合执行（任意 modded 密度函数的通用兜底）

### 11.1 动机

§5.15.7 的"遇到未知节点整体回退 CPU"过于粗暴：只要地形树里有一个
不认识的节点（如 Tectonic 的 `Invert`），整个维度就失去 GPU 加速。
而实践中未知节点往往是**小包装**（`Invert` 包一个 clamp），真正庞大的
子树（噪声、插值）都是已支持的。

### 11.2 设计：黑盒外部输入

**方案**：无法翻译的子树不再回退，而是当作**黑盒外部输入**——
1. blobify 遇到未知节点 → 登记为「外部输入 #i」，blob 写 `ExternalInput(index)`
2. Java 在**角点网格**（每 chunk 5×97×5）上用 `DensityFunction.sample()`
   求值该子树
3. 上传到 GPU 新 buffer（binding 6）
4. WGSL 里 `ExternalInput(i)` 直接读该 buffer，不计算

GPU 算它懂的，CPU 补它不懂的，**任意改地形的 mod 都能共存**，无需逐个适配。

> 为什么不递归遍历未知节点的子树？因为**节点自身的运算不透明**——
> 知道子节点值，却不知道对它施加什么变换（`1/x`? `clamp`?），无法重建。
> `mapAll` 降级（若节点作者实现了）能覆盖一部分，但不通用、有正确性
> 风险，故不采用；黑盒方案通用且稳。

### 11.3 实现清单

| 层 | 改动 |
|---|---|
| Rust AST | `ExternalInput { index }` + Hash |
| Rust deserialize | tag `0xFD` → `ExternalInput` |
| Rust emitter | `external_input.rs` 读 `external_inputs[]` |
| WGSL / BindGroup | 新 `@binding(6) external_inputs`（read-only） |
| Rust executor | `params._pad[0]` = 外部输入数；惰性分配/上传专用 buffer（无外部输入时绑 4 字节 dummy，避免与 binding 2 的 read-write 存储别名） |
| Rust WorldContext | `set_external_inputs` / `external_count` / `external_data` |
| Rust FFM | `c2me_wgpu_set_external_inputs(world, count, ptr, len)` |
| Java blobify | 未知节点 → 登记 + 写 `ExternalInput(index)` |
| Java coordinator | 逐 chunk/子树/角点求值，dispatch 前上传 |
| opto passes | `ExternalInput` 视为叶子 |

### 11.4 布局

```
external_inputs[] = [chunk0: ext0[G], ext1[G], ...][chunk1: ...]
G = cells_per_chunk (5×97×5)
索引 = chunk_index * (num_ext * G) + i * G + cell_idx
```

### 11.5 实测（桌面 RADV，Tectonic）

- **Invert 直接支持**（默认）：`numExt=0`，外部路径不激活，CPS 正常。
- **强制外部化**（临时禁用 `Invert` 支持）：日志
  `UNSUPPORTED NODE (externalized to CPU): ...Invert` +
  `5 external (CPU-evaluated) density subtree(s)`，**无 CPU 回退**，
  地形正确，CPS ~74。

### 11.6 真实模组测试（2026-10）

尝试用主流 Fabric 地形模组触发外部输入路径，结果：

| 模组 | 结果 |
|---|---|
| Tectonic | 只用 `Invert`（已直接支持），不触发 |
| lithostitched | 库模组，单独装不引入未知节点 |
| More Density Functions (MDF) | 未触发 |
| WorldGen Editor (WGE) | 未触发 |
| Ultimate Scaler (US) | **崩溃**，但与 C2ME 冲突，与我们无关（见下） |

**结论**：主流 Fabric 地形模组基本不引入自定义密度函数节点——
它们大多通过**数据包**组合原版节点。混合执行是"保险"，不是高频路径。

**US 崩溃分析**（记录备查）：US 的 `MixinInterpolatedNoiseSampler` 想用
`@At("STORE")` 注入 `class_5822::modifyBlockX`，但 **C2ME 的
`c2me-opts-natives-math`**（priority 1100 > US 的 1000）已先改写该方法，
导致 US 的注入点定位失败。**冲突双方是 US ↔ C2ME，我们的 mod 不在崩溃链**。

### 11.7 已知边界

- **未在 Android 实测**：无未知节点时走 dummy 路径、与改动前一致；
  有外部输入时的 Android 表现待验证。
- **求值成本**：外部子树越大，CPU 成本越高。实践中未知节点多为小包装，
  成本小；若某 mod 的未知节点包着大子树，可据实再优化。

---

## §12 诊断基础设施（2026-10）

### 12.1 Rust 日志文件（弃用 `C2ME_WGPU_DEBUG_LOG`）

**旧机制**：Rust 诊断靠环境变量 `C2ME_WGPU_DEBUG_LOG=<path>` 指定写文件路径。
Android 上必需（`latest.log` 只抓 Java stdout，Rust stderr 崩溃时丢失）。

**问题**：要多设一个环境变量；且 `wgpu_debug!` 还要求 `C2ME_WGPU_DEBUG`
非空才输出，两个开关易混淆。

**新机制**：Java 启动时通过 FFM `c2me_wgpu_set_log_path` 把
`<gameDir>/logs/c2me-wgpu.log` 传给 Rust（`FabricLoader.getGameDir()`）。
Rust 无路径时回退到 `<cwd>/logs/c2me-wgpu.log`。**不再需要任何环境变量。**

### 12.2 `C2ME_WGPU_PROFILE`（执行阶段计时）

独立于 `C2ME_WGPU_DEBUG` 的开关，输出每个 GPU 批次的阶段耗时
（encode / submit / readback / decode），**每 64 批聚合一行平均**（避免刷屏）。

实测（桌面 RADV，原版，垂直同步关闭）：

| 阶段 | 耗时 | 占比 |
|---|---|---|
| encode | ~0.015 ms | ~0.2% |
| submit | ~0.27 ms | ~3% |
| **readback** | **5–20 ms** | **~90%** |
| ├─ `poll(Wait)`（等 GPU） | 5–20 ms | ← 全在这 |
| └─ `map+copy`（拷贝数据） | 0.03–1 ms | ~0.5% |
| decode | ~0.1–0.9 ms | ~3% |

**关键结论**：瓶颈是 `device.poll(PollType::Wait)` **等待 GPU 完成**（~5ms 固定
开销，批越大越高），**不是**数据回读带宽。曾设想的"只回读 2 通道"优化
（省 87.5% 带宽）**证伪**——`map+copy` 本就 <1 ms。

**推测**：核显上 GPU 被渲染占满，我们的 compute 只能在渲染帧的空隙跑，
还要付同步开销。故在"高 U 低显"设备上 GPU 卸载的收益可能天生有限
（除非有独显 / GPU 空闲 / CPU 被占满）。见 §13。

### 12.3 环境变量现状

| 变量 | 状态 |
|---|---|
| `C2ME_WGPU_DEBUG_LOG` | **退役**（路径改由 Java 传入） |
| `C2ME_WGPU_DEBUG` | 保留（编译/缓存诊断，输出到 c2me-wgpu.log） |
| `C2ME_WGPU_PROFILE` | 新增（阶段计时，独立开关） |

---

## §13 已验证：GPU 卸载在"高 U 低显"设备上是负优化

### 13.1 严格 A/B（麒麟 980 / Mali-G76，Chunky 半径 500，同种子）

| 组 | 配置 | 总区块 | 总时间 | CPS |
|---|---|---|---|---|
| **A** | 纯 C2ME CPU（我们的 mod 禁用） | 4225 | **182 s** | **23.2** |
| **B** | C2ME + GPU mod | 4225 | **254 s** | **16.6** |

**B 比 A 慢约 28%。** 两组同世界、同半径、同种子，仅差我们的 mod。
A 组由维护者手动加 `.disabled` 确认禁用，对比干净。

### 13.2 为什么（与 §12.2 profile 吻合）

GPU 路径 ~90% 时间花在 `poll(Wait)` 等 GPU 完成。麒麟实测每批
`readback ≈ 20 ms`（桌面 RADV 仅 5–8 ms）。核显/移动 GPU 被渲染占满，
我们的 compute 只能抢空隙、还付同步开销——**8 核 A76 纯 CPU 跑噪声反而更快**。

### 13.3 结论与定位

**本 mod 在"强 CPU + 弱 GPU（核显/移动）"设备上是负优化**，而这正是
多数 C2ME 用户的环境。可能受益的场景仅限：
- 独显（GPU 独立于渲染，不与帧竞争）
- GPU 空闲 / 渲染负载低
- CPU 被其他任务占满

C2ME 官方 CPU 路径已被极致优化（DFC 编译字节码 + native math），
这个 GPU 模块在其之上**很难再有优势**。

### 13.4 影响

- **宣发**：不得宣传成"通用加速器"。默认口径应是"实验性，多数设备无收益"。
- **产品**：作为技术验证/实验模块保留，不投入进一步性能优化——
  瓶颈（等 GPU）不是代码能解决的，是硬件与渲染竞争。
- **决策**：不再继续推进此方向。

**结论：已验证的负优化，方向终止。**

---

## §14 Epic Terrain 暴露的 interpolated-边界 bug（已知，未修）

### 14.1 现象

修完 §13 后的两个 bug（stride + 矿脉）后：
- **vanilla 地形**：块级 A/B diff 差异 = **0**（100% 正确）✅
- **Epic Terrain 地形**：出现 **1639 个**我们多填的石头方块（GPU 比 vanilla 多）

### 14.2 根因（运行时树 dump 确认）

Epic 的 `final_density` 运行时结构：

```
BinaryOperation(min)
├── UnaryOperation(squeeze)          ← 外层，非线性
│   └── BinaryOperation(mul)
│       ├── Constant(0.64)
│       └── Wrapper(Interpolated)    ← MC 的插值边界
│           └── BlendDensity(...)
└── RangeChoice(... noodle 洞穴树 ...)
```

- **MC**：只对 `interpolated` **内部**做角点采样+插值；外部（`squeeze`/`mul`/`min`/
  `range_choice`）**逐方块**求值 → `外层(插值(inner))`
- **我们**：对**整棵树**在角点采样，再插值 → `插值(外层(inner))`

`squeeze` 非线性 → 两者不等。**vanilla 结构相同，只是地形平缓未暴露；Epic 放大起伏后暴露。**

### 14.3 为何未修

正确修复需拦截 MC 的 `DensityInterpolator.fill()`，用 GPU 角点值填充 MC 缓冲，
让 MC 自己完成插值 + 外层求值。实测：
- `final_density` 树里有 **8 个** `DensityInterpolator`（含含水层噪声通道）
- 需枚举全部 interpolated 节点、建立实例↔子树映射、改 blobify 协议、处理线程安全

**评估**：4–6 小时的高侵入改动，且本 mod 已确认负优化（§13），投入产出比低。

### 14.4 决定

**不做完整重构。** 保留：
- 实验分支 `experiment/interp-boundary`：含设计文档 `INTERP_REFACTOR.md` +
  hook 验证（`DensityInterpolator.fill` 可达，densities.len=65）+ interpolator 计数诊断
- 主线 `main`：vanilla 100% 正确

Epic 等"interpolated 外包了非线性节点"的模组地形下，本 mod 结果不精确（多填方块）。
**这类地形建议禁用本 mod。**

---

## §15 JFR 热点分析（2026-10，验证 §13 的"写回瓶颈"）

用 JFR（`-XX:StartFlightRecording=...settings=profile`）对游戏跑图 94 秒采样，
零侵入（不像 per-block `nanoTime` 会破坏 JIT）。

### 15.1 热点方法（ExecutionSample）

| 方法 | 占比 | 归属 |
|---|---|---|
| `AquiferSampler$Impl.apply` | **14.73%** | MC 含水层逐方块 |
| `DoublePerlinNoiseSampler.sample` | **14.31%** | 噪声（多由含水层调用） |
| `InvocationShim.invokeDensityFunctionSample` | 6.16% | C2ME DFC 运行时 |
| `ChunkSection.setBlockState` | 5.84% | 写方块 |
| `aquiferExtracted$refreshDistPosIdx` | 3.59% | 含水层内部 |
| `PalettedContainer` 相关 | ~3% | 方块调色板 |
| `StructureWeightSampler.sample` | 2.23% | Beardifier 逐方块 |
| `Heightmap` | 1.58% | 高度图更新 |
| **`writeDensityToChunk`（我们）** | **1.44%** | 循环框架 |

**~45% 在"含水层 + 其噪声"，~13% 在"写方块 + 调色板 + 高度图"，全是 MC 逐方块操作。我们的代码仅 1.44%。**

### 15.2 排除的项（都不是瓶颈）

- **deopt**：4143 次散布数十站点，最高 28 次/站点（≈44/s），Minecraft 多态代码的正常水平。
- **异常**：10406 次，主要是 C2ME 的文件 IO（`UnixException` 7019）、调度
  （`CancellationException` 1420）、反射（`NoSuchMethodError` 1312）——**无一是我们的**。
- **分配**：`NoisePosVanillaInterface` 6.69%（C2ME DFC 包装）；消除我们自己的
  `UnblendedNoisePos` 分配（§MutableNoisePos）后 CPS 无变化。
- **poll/GPU**：执行器 ~79% 时间空闲（批间隔 20ms vs 处理 4ms）→ GPU 不是瓶颈。

### 15.3 结论

**瓶颈 = MC 的逐方块写回（含水层 + 写方块 + 高度图），与原版同构、无法卸载。**
三重确认 §13 的负优化判断：poll / GC / deopt 均非因，唯逐方块写回是。

### 15.4 aquifer 能否搬 GPU？（2026-10，判定：不值）

JFR 显示 aquifer（`AquiferSampler.apply` + 内部 + 噪声 ≈ 25–36%）是写回里最大
的单项，一度是 GPU 卸载的首选目标。**§17 用 JFR 逐帧归因后修正了本节的一个
错误论据**，但结论不变。

> **⚠️ 勘误（2026-10，见 §17）**：本节早期称 aquifer 噪声"逐方块、80× 角点"。
> **实测证伪**：aquifer 的四个噪声调用极少——`DoublePerlinNoiseSampler.sample`
> 的全部样本中只有 **2.5%** 来自 aquifer。aquifer 的真正开销是**逐方块控制流**
> （扫 12 邻居、距离比较、缓存索引维护），**不是计算**。下面的"80×"数字作废，
> 但"不搬"的结论因**更硬的**理由成立。

重新评估"把 aquifer 搬 GPU"：

- **噪声**：aquifer 用 barrier / fluidLevelFloodedness / fluidLevelSpread /
  fluidType 四通道。实测（§17）：
  - spread/type 在**超粗网格**采样（`x/16, y/40, z/16` 与 `x/64, …`）；
  - floodedness 每 cell 一次；barrier 每方块但有 `|intermediate|<=2` 门控；
  - **合计每 chunk 仅几百次**——不是"逐方块 131072"。
- **瓶颈**：aquifer 36.9% 里，~34% 是 `apply` 主体的逐方块逻辑
  （扫邻居、`refreshDistPosIdx` 索引维护、距离计算）——**纯控制流，GPU 搬不动**。
- **随机**：`RandomSplitter`（Xoroshiro128++，128 位）——WGSL 无 128 位整数，
  手机 Mali 不支持。好在每单元一次，可留 CPU。

**结论：不搬。** 理由从"80× 计算代价"改为**"aquifer 是控制流瓶颈，可 GPU 化的
计算只占 2.5%"**——这比原论据更硬：不是"代价高"，而是"**根本没有可搬的东西**"。


### 15.5 最终结论

排除的所有方向：poll/GPU 流水线（GPU 79% 空闲）、GC/分配（消除后无变化）、
deopt/异常（正常）、搬 aquifer（代价 > 收益）。

**唯一瓶颈 = MC 逐方块写回（aquifer + 写方块 + 高度图），与原版同构、搬不动。**
这从 JFR 证据三重确认了 §13 的负优化判断。

### 15.6 后续尝试（全部证伪，2026-10）

在 §15.5 之后又验证了两个方向，均无正收益：

- **surfaceHeightEstimate 预填（对 C2ME OpenCL 思路的可行性验证）**：
  实测 `ChunkNoiseSampler.calculateSurfaceHeightEstimate` 单次 ~79µs、约 200 万次未命中
  （JFR 漏报，因其 sample() 归到叶子帧）。用 GPU 通道 10 预填 `surfaceHeightEstimateCache`
  后，**桌面 CPS 无变化**（51.5→52.3）——**CPU 非端到端瓶颈**，省下的 CPU 被富余吸收。
  （分支 `experiment/surface-height-cache`）

- **异步写回**：`submit()` 立即返回 future，写回在独立线程池（`WRITEBACK_THREADS` 可配）。
  桌面（16 核）：CPS **51.5→68.9（+34%）**，批 1.3→2.5。
  但麒麟 980（8 核）：批同样变大，**CPS 仍 ~12 不变**，且移动新区域时**冻结 62 秒**
  （WRITEBACK_POOL 与 C2ME worker 在少核上互相饥饿，依赖图停摆）。同步版不冻结。
  **会冻游戏的优化不可接受。**
  （分支 `experiment/async-writeback`）

**async + surface-cache 组合**（桌面）：CPS 73.7，与 async 单独（68.9）在噪声内，**无叠加**。

**总账**：所有"省 CPU"方向均无效（CPU 富余）；"流水线"在桌面有效但手机死锁。
瓶颈 = **每批 dispatch 的固定同步延迟**，而解决它（流水线）会引入移动端死锁——两难，无干净解。
**最终结论：该 mod 在目标设备（强 CPU + 弱 GPU）上是净负优化，方向终止。**

### 15.7 写回阶段拆解（2026-10，最终定位）

用诊断开关把"写回"逐段拆开，同构建、同世界：

| 配置 | 跳过 | CPS |
|---|---|---|
| 完整写回 | — | 51.5 |
| 跳 heightmap（NO_HEIGHTMAP） | 2× trackUpdate/方块 | **93.3** |
| 最简（MIN_STAGE） | heightmap + aquifer + beard + vein | **~179** |
| 纯 GPU 算密度（bench，无游戏） | — | 728 |
| bench + 游戏渲染 | — | 327 |

**拆分**：heightmap ≈ +42 CPS，aquifer+beard+vein ≈ +86 CPS。

**结论**：
- GPU 算密度**富余 6 倍**（bench 728 vs 游戏 327，仍远超需求）→ 从不是瓶颈
- **真墙 = CPU 逐方块写回的四个语义必需阶段**：
  - aquifer（水/岩浆）、beard（结构地基）、vein（矿脉）→ +86，**搬不动**（随机/结构语义，§15.4）
  - heightmap（地表/水域）→ +42，**可优化但高风险**：`trackUpdate` 的
    向下扫描分支 + 依赖写入顺序 + 两个谓词（`!isAir` / `blocksMovement`），
    重实现须逐字对齐否则高度图偏 → 地表/结构错。而"省 CPU"方向已被反复证明
    端到端不涨（§15.6 surface-cache）——不值得再赌。

**最终**：该 mod 在目标设备上是净负优化，**所有方向均已逐一验证、排除**。方向终止。


---

## §16 GC 对"高 U 低显"负优化的贡献 —— 实测排除（2026-10）

### 16.1 动机

§13 把"高 U 低显"设备的负优化归因于"GPU 被渲染占满、poll 等待"。但有一个
从未验证的假设：**分配压力在弱 GC（手机）下造成停顿**。§5.17 的 biome 路径
新增了每 chunk 的分配（`float[6*cpc]` ≈ 77 KB/chunk，blob 拷贝 ~1.2 MB/batch），
合计约 **25 MB/s**。若这是主因，手机应受益于"减少分配"。

### 16.2 方法

同一世界、同一构建（开 biome 加速），仅改 GC 配置跑四轮，`-Xlog:gc*` 记录：

| 配置 | GC 次数 | 平均停顿 | 最长停顿 | `Can't keep up` |
|---|---|---|---|---|
| Zulu + ZGC | — | 亚毫秒 | 低 | 2 × 2.0 s |
| Zulu + G1 | 79 | 19.3 ms | 43 ms | 1 × 2.0 s |
| OpenJDK + 默认(G1) | 61 | 21.0 ms | 160 ms | 0 |
| OpenJDK + **SerialGC** | ~301 | **36.6 ms** | **528 ms** | 2 × 2.7 s |

（每轮约 30–150 s 游戏时间，飞行生成新区块。）

### 16.3 结论

1. **GC 实现不是变量**：G1 两轮平均停顿都在 ~20 ms，与 ZGC 体感无差。
   **分配速率才是**（两轮 G1 都是 60–80 次 Young GC，速率一致）。
2. **那个 ~2 s 的 `Can't keep up` 与 GC 无关**：ZGC（亚毫秒停顿）有它，
   SerialGC（528 ms 停顿）也有它，四轮都有。**它另有原因**——最可能是
   Rust executor 是全局单例，terrain / biome 两条路径抢同一把锁、串行化。
3. **SerialGC 是最坏情况**（单线程 STW，avg 36 ms、max 528 ms）——即便
   如此，桌面**体感与 ZGC 无差**。手机 GC 再差 ~3×，也要累积多次才明显；
   而 §15.7 已证 CPU 逐方块写回在手机上同样 ×3–5，**那才是主因**。

### 16.4 对"手动管理内存 / 减少分配"方向的裁定

**收益极低，不做。** 实测证明 GC 不是"高 U 低显"负优化的主要来源；把 25 MB/s
的分配降到接近 0，也不会改变端到端体感。方向终止。

### 16.5 executor 串行假设（实测证伪）

一度怀疑 ~2 s 停顿来自 executor 串行：`get_or_create_executor()` 返回全局
单例 `ExecutorThread`，`request_chunks` / `request_biomes` 都
`executor.lock()` 并**持锁跑完整个 GPU 往返**。biome `avgBatch` 13–26 ms 而
shader 更小，像是排队等锁。

**实测证伪**：拆成 per-world executor（terrain / biome 各一个线程 + buffer）
后，biome `avgBatch` 仍 13–22 ms（没变），`Can't keep up` 反增（2→4）。
executor 拆/合对该停顿**无确定影响**，已回退（无收益 + 多花 ~36 MB native）。

### 16.6 决定性实验：biome 路径 = 2s 停顿的元凶（2026-10）

同飞法、同时长，仅切 `C2ME_WGPU_BIOME`：

| 配置 | `Can't keep up` | terrain CPS |
|---|---|---|
| biome **ON** | **4 次**（~2 s 级）| 49–68 |
| biome **OFF** | **0 次** | **73–91** |

**结论：biome GPU 加速是净负优化。**

- BIOMES 在 NOISE **之前**（§5.17.2）。biome 每 chunk 等一次 GPU 往返
  （avgBatch 14–21 ms），**拉长了整条 chunk 生成链的关键路径** → Server
  tick 停摆。
- 它还**拖累 terrain**（CPS 49–68 → 73–91）：biome 请求占用共享 GPU
  executor 队列，terrain batch 到得更慢。
- **根因认知**：vanilla 的 biome 采样是**纯 CPU 六参数噪声**，本来就快。
  搬到 GPU 换来的不是"更快"，而是**引入 ~14 ms 往返延迟**，还落在关键路径。

这印证 §13/§16 的总判断：**GPU 卸载在关键路径上是负优化**。biome 比地形更
糟——地形的 GPU 计算量真的大（值得省 CPU），biome 计算量本来就小（省不了
多少，却付出往返延迟）。

**处置**：biome 代码移到分支 `experiment/biome-gpu` 保存；main 上不启用
（`C2ME_WGPU_BIOME` 默认关）。作为**技术验证保留**（证明 6 气候参数可精确
复现 1e-5、调用链可接入），不作为性能特性。

### 16.7 关键教训

- **"听起来合理"的假设必须实测**：弱 GC → 卡顿（四轮 GC 数据证伪）、
  executor 串行 → 停顿（拆开实测证伪）。两个"合理"假设都错。
- **关键路径上的延迟 > 吞吐**：GPU 往返的固定延迟，落在 chunk 生成的关键
  路径上，比 CPU 上的批量计算更致命。
- **"能搬" ≠ "值得搬"**：biome 完全可复现、可接入，但计算量太小，搬运的
  延迟成本盖过计算收益。
- **平均停顿 ≠ 端到端体感**：SerialGC avg 36 ms/max 528 ms，体感却与 ZGC
  无差——瓶颈在别处，GC 停顿被吸收。
- **对照实验要切单变量**：biome ON/OFF 这一刀，比之前所有 GC/executor 折腾
  都更能说明问题。

---

## §17 aquifer 逐帧归因 —— 修正 §15.4 的错误论据（2026-10）

### 17.1 动机

§15.4 判定 aquifer 不值得 GPU 化，但其核心论据是"噪声逐方块 131072/chunk
（80× 角点）"。这个数字**从未被直接测量**，只是从"aquifer 用裸 Noise 节点"
推断。在动手前用 JFR 实测验证。

### 17.2 方法

`-XX:StartFlightRecording=settings=profile`，游戏内飞行 60–90 s 生成新区块，
JFR 采样（每 20 ms）后逐帧归因。共 **40473 个 ExecutionSample**（Java）+
9597 个 NativeMethodSample。

### 17.3 结果

**aquifer 总占比（栈中任意深度出现 aquifer 帧）= 14947 / 40473 = 36.9%。**
与维护者的经验记忆一致。

但**逐帧拆开后**（顶层帧，即 CPU 实际执行的方法）：

| 顶层帧 | 占比 | 性质 |
|---|---|---|
| `class_6350$class_5832.method_38317`（aquifer `apply` 主体）| 16.9% | **控制流** |
| `class_6350$class_5832.aquiferExtracted$refreshDistPosIdx` | 4.1% | **索引维护** |
| `class_5216.method_27406`（`DoublePerlinNoiseSampler.sample`）| 15.0% | 噪声 |
| `calculateDensityModified` + `getWaterLevel` | 1.4% | 混合 |
| 其余散布 | ~14% | — |

**关键归因**：把 `DoublePerlinNoiseSampler.sample` 的 6083 个样本按调用栈来源
分类——**只有 147 个（2.4%）来自 aquifer**，5936 个来自地形密度。

Native 采样同理：9597 个 NativeMethodSample 里，只有 **10 个**来自 aquifer。

### 17.4 结论

**aquifer 的 36.9%（后修正为 ~65%）** —— 早期读数只数了 Java `DoublePerlinNoiseSampler`
路径，漏了 C2ME-DFC 编译路径；见 §17.6 勘误。当时误判为"几乎全是控制流"。

- `apply` 主体（16.9%）：扫 12 个邻居 cell、算 `maxDistance`、比较、缓存查找
- `refreshDistPosIdx`（4.1%）：纯索引维护，**零噪声**
- 真正的噪声（Java + native）合计 **~2.5%**

这**修正了 §15.4 的错误论据**：aquifer 噪声不是"逐方块 80×"，而是**极少**。

但**结论不变，且理由更硬**：aquifer 没有"可 GPU 化的计算"——它的成本在
控制流和随机索引上，这些 GPU 搬不动。§15.4 说"不值得搬"，实际是
**"没有可搬的东西"**。

### 17.5 与 biome / GC 实验的一致性

这是本轮第三次独立验证同一规律：

| 方向 | 瓶颈 | 可 GPU 化的部分 |
|---|---|---|
| biome（§16.6）| 关键路径延迟 | 六参数计算本来就小 |
| GC/分配（§16）| 不是端到端瓶颈 | — |
| aquifer（§17）| **逐方块控制流** | 噪声仅 2.5% |

三次都指向 §15.7 的同一结论：**MC 世界生成的瓶颈在"逐方块控制流与语义"，
不在"可批量化的计算"上。** GPU 能加速的永远是后者，而后者从来不是瓶颈。

### 17.6 勘误（2026-10，第二次 JFR，更细归因）

§17.3 的"aquifer 噪声只 2.5%"**是错的**——它只数了
`DoublePerlinNoiseSampler.sample`，**漏掉了 C2ME-DFC 编译路径**。

第二次 JFR（46655 样本）在 `writeDensityToChunk` 子树内逐帧归因：

| 部分 | 占写回 | 说明 |
|---|---|---|
| aquifer（任意深度）| **~65%** | 最大单一来源 |
| └ aquifer 控制流 | ~32% | `apply` + `refreshDistPosIdx` + `extractedCheckFG` |
| └ aquifer 噪声 | ~32% | 经 C2ME-DFC（`InvocationShim` + `DfcCompiled_*`）|
| ChunkSection.setBlockState | 10.6% | 方块写入 |
| Beardifier | 5.1% | |
| **Heightmap** | **2.6%** | 解释 §18 A/B 为何零收益 |
| PalettedContainer | 3.2% | |
| 我们的 writeDensityToChunk | 2.7% | |

**关键证据**：`InvocationShim`（C2ME 编译密度函数）的样本 **5037/5098 = 98.8%
都在 aquifer 栈内**——即那些"DFC"调用几乎全是 **aquifer 自己的**密度函数
（barrier/floodedness/spread/type），不是地形。

**修正后的结论**：aquifer ≈ **一半控制流、一半噪声**（不是 §17.3 的"97.5%
控制流"）。噪声部分**原则上可 GPU 化**（§0 的角点/逐方块方案），控制流部分
（扫 12 邻居、缓存、128 位随机）仍是阻碍。

### 17.7 关键教训

- **"看起来像计算" ≠ "是计算"**：aquifer 用了四个 Noise 节点，看起来是计算
  密集，实测 97.5% 的开销在控制流。
- **归因要到叶子帧**：只看"栈里有 aquifer"得出 36.9%，会误以为值得优化；拆到
  叶子帧才发现噪声只有 2.5%。**归因的粒度决定结论。**
- **JFR 是唯一不污染测量的手段**：AtomicLong 计数会因多线程争用 + 阻碍 JIT
  内联把 CPS 打到 17（§5.11.5 的教训），采样式 JFR 无此问题。
- **勘误要留痕**：§15.4 的错误论据保留在勘误框里，比直接删掉更有价值——它记录了
  "如何从一个合理的推断走向错误结论"。

---

## §18 待验证的"取巧"思路（2026-10）

> 背景：三轮 GPU 尝试（biome/GC/aquifer，§16/§17）都得出"正面硬搬不划算"。
> 换一个角度：**是不是我们现有实现本身就有浪费？** 下列思路按"收益×可行"
> 排序，逐一验证后再决定去留。**先验证，不先写代码。**

### 18.1 线索：heightmap 可能是我们自造的浪费

§15.7 实测：
| 配置 | CPS |
|---|---|
| 完整写回 | 51.5 |
| 跳 heightmap（NO_HEIGHTMAP）| **93.3** |

即"只跳 heightmap"就翻倍。**⚠️ 但这个 +42 的归因后来被 §18.2 的 A/B 两次
实测推翻**（见下）：把 heightmap 更新换成任何更便宜的实现都零收益，说明
+42 并非来自 heightmap 更新本身。当时的"heightmap 是大头"读法存疑。

当时没追问：

**vanilla / C2ME 是怎么更新 heightmap 的？**

- **我们**（§5.6.2，`writeDensityToChunk`）：**逐方块**调
  `oceanFloorWg.trackUpdate(...)` + `worldSurfaceWg.trackUpdate(...)`。
- **C2ME OpenCL**（§17 读到的真实代码）：写完整块数组后，调**一次**
  `Heightmap.populateHeightmaps(chunk, {OCEAN_FLOOR_WG, WORLD_SURFACE_WG})`。
- **vanilla `populateNoise`**：待验证（见 §18.2）。

若 vanilla 也是一次性 populate，则**我们用了一个比 vanilla 更贵的等价实现**，
凭空吃掉可能 ~40 CPS。这不是"要不要 GPU"的问题，是"实现选错了 API"。

### 18.2 待验证清单

**取巧 A：heightmap 改回一次性 populate —— ❌ 已证伪（2026-10）**
- 假设：把逐方块 `trackUpdate` 换成循环后一次 `Heightmap.populateHeightmaps`，
  拿回 §15.7 的 +42 CPS。
- 事实核查：
  - vanilla `populateNoise` **也是**逐方块 `Heightmap.trackUpdate`（我们没错）；
  - C2ME OpenCL 确实用一次性 `populateHeightmaps({OCEAN_FLOOR_WG, WORLD_SURFACE_WG})`；
  - `populateHeightmaps` 实现：每列从顶扫到底，逐深度 `chunk.getBlockState(pos)`。
- **实测（干净 vanilla 地形，Epic/worldgenfeaturefix 禁用）：无提升。**
  write= 大多 40–66 ms/chunk，CPS 75–106，与 §15.7 基线（46–60 ms、70–86）
  持平。
- **认知**：§15.7 的 +42 CPS 来自"**完全不读方块状态**"，不是"少调几个 API"。
  `populateHeightmaps` 仍要逐深度 palette 查询——**成本在 `getBlockState`，
  不在 API 调用次数**。换 API 省不掉。
- 已撤回（避免引入与 vanilla 不同的 heightmap 时机）。

**取巧 B：本地列顶数组替代逐方块 trackUpdate —— ❌ 已证伪（2026-10）**
- 做法：写回热循环里用 `int[256]` 记每列最高匹配方块（世界 y+1），
  热循环内**零 palette 读**（`bs` 已在手），循环后一次写回两个 Heightmap。
  用 `@Invoker` 暴露 private `Heightmap.set`。
- **语义正确**：实测结构无悬空/被埋（"每格只写一次、bottom-up"假设成立）。
- **但零 CPS 提升**：write= 仍 40–60 ms/chunk，CPS 57–99，与基线持平。
- **认知（重要）**：这比 A 更强的负结果——热循环里 palette 读已完全消除，
  仍无收益 ⇒ **`trackUpdate` 不是瓶颈**。这**推翻了 §15.7 的
  "+42 CPS 来自 heightmap"**读法：那 +42 要么测量有误（§15.6 早提过 perf
  数噪声大），要么来自"跳 heightmap"顺带跳掉的别的东西。
- 代码存 `experiment/heightmap-oneshot` 分支。

**取巧 C：只做 4×4 方阵 + estimateSurfaceHeight 预计算**
- 不碰 aquifer 控制流，只把 §17 阻碍1（跨 chunk 依赖）预计算
- 为未来 aquifer GPU 化铺路；本身收益可能有限

**取巧 D：复用 vanilla 的 heightmap / 表面规则路径**
- 只接管密度，把 heightmap 交回 vanilla 某些路径

### 18.3 验证顺序

先 A（最便宜、可能最高收益、零架构改动）→ 再按结果决定 B/C/D。

**A 的判据**：若 vanilla 确为 `populateHeightmaps`，则改我们为同款，实测
CPS 是否从 ~51 回到 ~90 区间。若是，这是本项目第一个"正收益且零 GPU 成本"
的优化。

### 18.4 手机决定性实验：CPU 写回是手机瓶颈，但杠杆必须够长（2026-10）

用 `experiment/writeback-stages` 的 `C2ME_WGPU_MIN_STAGE=1`（跳过
aquifer/beard/vein/heightmap 写回，只写石块）在 Kirin 980（FCL 实例）实测：

| 配置 | write= (ms/chunk) | CPS | readback (ms/batch) |
|---|---|---|---|
| 完整写回 | 88–292 | 6–17 | 11–27 |
| **MIN_STAGE** | **11–21** | **27–41** | **23–38** |

**结论 1：手机降低 CPU 写回确实转化为 CPS。**
write 降 ~14× → CPS 涨 ~2.5×（12 → 33）。与桌面"小改动被吸收"不同，
**手机 CPU 弱、并行收益差，写回直接锁死吞吐**。此前"省 CPU 不转化 CPS"的
概括是错的——正确的规律是**省得多才转化，省得少被吸收**。

**结论 2：杠杆必须够长。**
- 砍掉写回 ~90%（MIN_STAGE）→ 2.5×
- aquifer 噪声只占写回 ~15% → 理论 CPS 12→13，**几乎无感**
- 要显著提升手机，得砍写回的 ~90%，不是 15%

**结论 3：手机的天花板是 GPU 往返，不只是 CPU。**
MIN_STAGE 把 CPU 压到 13 ms/chunk，CPS 仍只 33——因为 **readback（等 GPU）
= 23–38 ms/batch**，而手机 batch 只 ~1.5 chunk。桌面同配置能到 179，因为
桌面 readback 只 3–4 ms。**手机的根本瓶颈是"GPU 往返 30 ms/批 × 小批"。**

### 18.5 aquifer 噪声 GPU 化的最终裁定：❌ 不做

综合 §17.6（aquifer 噪声 = 写回 14.7%）+ §18.4（手机需砍 90% 才有效）：

1. 只搬噪声 → 理论 CPS 12→13，**无感**；
2. 且它**还要增加 GPU 往返**（手机最贵的东西）→ 可能更差；
3. 唯一能让手机大提升的是"**整个写回管线 GPU 化**"（aquifer 控制流 +
   beard + vein + 输出方块状态）= C2ME 那套 **月级工程**，且手机 GPU 往返
   可能仍封顶（§18.4 结论 3）。

**所以：aquifer 噪声 GPU 化（取巧 C 路线）终止。** 手机是真实瓶颈这点首次
被数据证实，但"只搬噪声"这根杠杆太短；够长的杠杆是整体架构重写，超出本
项目范围。

### 18.6 被动请求架构：为什么 batch 攒不起来（2026-10）

观察：executor 79% 空闲（批间隔 20 ms vs 处理 4 ms），但 batch 只有 ~1.4 chunk
（桌面）/ ~1.5（手机）。MAX_BATCH=64 从未达到。

根因：WgpuBatchCoordinator.request() 是**阻塞**的——提交 job 后立刻
`job.future.get()`，直到结果返回。每个 C2ME worker 提交一个请求后就卡在等待
上，于是"并发在途请求数 ≈ worker 线程数"。攒批窗口再大也没用——窗口内不会
有更多请求到达，因为能发请求的 worker 已经在等了。手机 worker 少（~3–6），
batch 上限就 ~3–6。

由此理解 C2ME OpenCL 为什么那样设计：它不是"被动等 chunk 请求凑批"，而是
协调器**主动**为一个 4×4 区域一次性生成 16 个 chunk 的请求。这是请求模型的
改变，不是窗口参数。调大 MAX_WAIT_MS（experiment/c2me-batching 探针）原理上
无效，已确认。

已试过的相关方向（都失败）：
- experiment/c2me-batching：调大攒批窗口 → 原理无效（窗口内无请求可等）。
- experiment/async-writeback（227b4a4）：把**写回**移出 worker 线程 → 桌面
  +34%（batch 1.3→2.5），但 Kirin 980 冻死 62 秒（WRITEBACK_POOL 与 C2ME
  worker 争 8 核，依赖图停摆）。能冻游戏的优化不可接受，废弃。

真正没试过、也是官方核心的：让 worker **非阻塞提交**（异步请求），提交后继续
跑、不等待。风险：在途请求暴涨 → 内存/调度压力，弱 CPU 上可能同样崩（与
async-writeback 同型的坑）。

### 18.6b C2ME 的真实并行模型 + 我们的对应设计（2026-10，源码验证）

读完 C2ME `dev/1.21.11` 源码后，"非阻塞请求 + 写回并行"这条路的机制**完全清楚**了。
它不是"又一个看起来行"的方向——每一步都有 C2ME 对应实现。

C2ME 的模型（源码事实）：
- FlowSched `ScheduledTask.run(releaseLocks)` 只是 `action.get().whenComplete(...)`
  挂回调就返回 → **worker 不阻塞在 task 上**。
- GPU 完成后，写回通过 `schedulingManager.positionedExecutor(pos).execute(writeback)`
  enqueue 回**同一个 `prioritizedScheduler` worker 池**（`WrappingTask`，无独立池）。
- `ScheduledTask` 按 `blockStateWriteRadius` 持有**位置锁**，直到 future 完成才
  `releaseLocks` → 邻 chunk 读不到半成品，并发度由依赖图限制，不线程爆炸。

关键洞察：**高效的核心不是"4×4 方阵"，而是"写回与 worker 共池 + 位置锁"。**
`async-writeback`（§18.6）错在开了独立 `WRITEBACK_POOL`——两池在 8 核上互抢，
依赖图停摆 → 冻 62s。回同池就不会。

我们对应的设计（照搬该模型）：
1. `populateNoise` 提交 GPU 请求后**返回 pending future**（worker 释放 → batch 攒大）。
2. GPU 完成 → 协调器线程 `schedulingManager.positionedExecutor(chunkPos)`
   enqueue 写回任务。
3. 写回在 C2ME worker 池里跑（不新开线程）→ 写方块 → 完成 future。
4. future 完成 → C2ME 释放位置锁 → 邻 chunk 继续。

API 可达性（已确认）：
- `IVanillaChunkManager.c2me$getSchedulingManager()` → `SchedulingManager`
  （`c2me-base`）。
- 我们的 hook 有 `StructureAccessor`，其 `world` 字段 → `ServerWorld`
  → `getChunkManager().chunkLoadingManager` → 拿到该接口。
- `SchedulingManager.positionedExecutor(pos)` 是 public。

两个真实风险：
1. **依赖 C2ME 内部 API**（`c2me-base` 的非公开约定接口 `IVanillaChunkManager`）。
   维护者认为"mod 本名 C2ME-wgpu-rs、运行时硬依赖 C2ME，用它的 API 比 §10 的
   '编译期解耦'更名正言顺"，接受此依赖。
2. **弱 CPU 死锁仍未知**：回同池避免了"两池互抢"，但"pending future →
   位置锁长期持有 → 依赖图等待"在 8 核上会否停摆，**只有实测知道**
   （async-writeback 没验证到这一点）。

工作量：天级（改 chunk 生成主路径 + 接入 C2ME 调度 API），非小时级。
这是目前唯一"机制上说得通、且有 C2ME 先例"的方向。

### 18.7 关键教训

- **"省 CPU 不转化为 CPS"是过度概括**：桌面小改动被并行吸收，但 MIN_STAGE
  （砍 90%）在桌面 51→179、手机 12→33。**转化与否取决于省的量和是否动了
  关键路径，不是"CPU 不是瓶颈"。**
- **手机瓶颈在 GPU 往返，不在 CPU**：readback 30 ms/批 × 小批，是手机 CPS
  的硬顶。桌面 3–4 ms 没有这个问题——**两端瓶颈根本不是一回事**。
- **杠杆长度决定一切**：heightmap 2.6%、aquifer 噪声 15% 都太短；能显著
  改变的是"整段写回"（90%）。

---

## §19 端到端 A/B：异步调度 + 写回池的最终裁定（2026-10）

### 19.1 动机：之前所有"收益"都是 batch 口径

§18.6b 的异步调度报告"桌面 batch 1.3→5、CPS 80→110"，但那 CPS 是
`WgpuBatchCoordinator` 的 **GPU batch 吞吐**（协调器内部指标），**不是端到端
chunk 完成吞吐**。二者可以背离：GPU 流水线跑得快，最终产出仍可能被拖累。

用 Chunky（同世界、中心 61.5,-55.5、半径 500、4225 chunks）做**端到端** A/B，
所有配置用同一世界同一参数。

### 19.2 结果

| 配置 | 总时间 | 端到端 CPS | 备注 |
|---|---|---|---|
| **纯 C2ME CPU**（无本 mod）| **23 s** | **183** | 基准 |
| 写回走 C2ME 池 | 38 s | 110 | §18.6b 默认 |
| 独立写回池 · 2 线程 | 107 s | 45 | 写回饿死 |
| 独立写回池 · 7 线程 | 33 s | 128 | |
| 独立写回池 · 12 线程 | **30 s** | **141** | CPU 90% |

### 19.3 结论

**1. 异步调度在桌面是净负优化：最好的配置仍比纯 CPU 慢约 30%（30 vs 23 s）。**

**2. 独立写回池优于 C2ME 共池**（33 < 38 s）——"写回抢 C2ME worker"**确实是
部分元凶**。（此前一次"无变化"是假象：脚本漏设 `C2ME_WGPU_DEDICATED_WB`，
跑的还是共池。）独立池让协调器 `idle` 从 ~1000 ms 降到 ~0。

**3. 但即使 CPU 榨到 90%、idle 归零，仍慢 30%。** 剩下的差距是**架构性**的：

- 纯 CPU：`populateNoise` 内联执行（几 ms）→ 位置锁秒放 → 流水线全速
- 我们：每 chunk 要等 **GPU 往返（readback ~5 ms）+ 写回** → 位置锁持有更久
  → 并行度被压

**GPU 往返的固定延迟落在 chunk 生成的关键路径上，吃掉的并行度 > 密度卸载
省下的计算。** 这与 §13（手机 16.6 vs 纯 CPU 23.2，慢 28%）方向完全一致。

**4. 加线程治不了。** 12 线程已 CPU 饱和，再加只会过载。

### 19.4 这次裁定的意义

**修正了整个项目的一个核心误读**：§13/§15.7 早已判断"高 U 低显"负优化，但
§18.6b 一度用 batch 口径的"桌面 +30%"覆盖了它。**端到端数据显示：桌面同样
是净负**（慢 30%），与手机同构。

**根因（再次确认）**：C2ME 的 CPU 密度计算已极致优化（DFC 编译字节码 +
native math），GPU 卸载省不了多少，却把固定延迟插进关键路径。§10.2 的
"144 CPS"其实早已暗示——GPU 要超越一个已经很快的 CPU 路径，收益空间极小。

### 19.5 关键教训

- **端到端指标才是判据**：内部 batch 吞吐（GPU 流水线）可以很好看，而最终
  产出更慢。任何"收益"必须用端到端（Chunky/时间）验证，不能用内部计数器。
- **一次"无变化"要先查配置是否真生效**：§18.6b 的"独立池无变化"是脚本漏设
  环境变量导致的假阴性。
- **CPU 饱和 ≠ 高效**：12 线程烧 90% CPU 换来慢 30%，是"用更多资源做更慢的
  事"——优化必须看端到端，不能看资源占用。
- **关键路径延迟 > 吞吐**：与 §16.6（biome）、§17（aquifer）同型——GPU 往返
  的固定延迟落在关键路径上，是 wgpu 架构下的结构性代价。

---

## §20 下一步计划：把"能卸载的都卸载"（2026-10，待做）

### 20.1 动机

§19 的结论是"异步调度 + 写回池"桌面净负（30 vs 23 s）——但它**只卸了密度
计算**，而密度计算不是瓶颈（§17）。写回的四段（§15.7）：

| 写回阶段 | 开销 | 现状 |
|---|---|---|
| aquifer | ~32–48% | CPU（§17）|
| heightmap | ~2.6% | CPU |
| vein | — | CPU（角点粗筛）|
| beard | ~5% | CPU |
| setBlockState | ~10.6% | CPU |

**关键路径延迟**是净负的根因（§19.3）。但**只要 CPU 侧的写回还占大头，
GPU 往返的延迟就"藏"在 CPU 忙碌里**——这与"卸载得越多、GPU 越可能净赚"的
直觉相反，但 §19 的数据支持：MIN_STAGE（跳写回）能到 179 CPS ≈ 纯 CPU，
说明**写回一旦拿掉，GPU 侧几乎不亏**。

### 20.2 计划

**把所有还能搬的写回阶段也搬上 GPU**，目标：CPU 写回只剩"palette 写入 +
heightmap"，把关键路径上 CPU 的部分压到最小：

1. **aquifer 整体 GPU 化**（§17.6 的 ~一半是噪声 + 随机位置）：
   - per-cell 噪声表 GPU 预计算（ch0–3，§5.10.2 已验证可复现）
   - 128 位 Xoroshiro 用 u32 对模拟（§5.17 微基准：3.3× 慢，bit-exact）
   - 控制流（扫 12 邻居、缓存）需 WGSL 移植（§17 的难点）
2. **vein 全 GPU 化**（现在只角点粗筛）
3. **beard / heightmap** 视情况搬或保持
4. 甚至 **GPU 直接输出方块状态字节**（对标 C2ME OpenCL 架构，§17 读到）

### 20.3 风险与前提

- **这不是"再改一刀"**：aquifer 控制流 GPU 化是周级工程，且 §17 已列 3 个
  阻碍（跨 chunk 的 `estimateSurfaceHeight`、两个惰性缓存、全局可变 bool）。
- **必须用端到端指标验收**（§19.5 教训）——Chunky/总时间，不看内部 batch 吞吐。
- **前提**：如果连"全卸载"后仍净负，则这套 wgpu 架构在 C2ME 之上就是不可行，
  方向终止。

### 20.4 当前分支状态（供参考）

所有实验已推 Gitee：
- `experiment/async-writeback`、`experiment/biome-gpu`、`experiment/c2me-batching`
- `experiment/c2me-scheduling`、`experiment/dedicated-writeback`
- `experiment/heightmap-oneshot`、`experiment/interp-boundary`
- `experiment/surface-height-cache`、`experiment/writeback-stages`

### 20.5 写回天花板实测（2026-10，关键）

在流水线协调器（§19 之后，main `23d16a1`）上，用诊断开关测"CPU 写回若全部
移走，端到端能到多少"。Chunky r=500、同世界、同中心：

| 配置 | Total | 开关 |
|---|---|---|
| 纯 C2ME CPU（无本 mod）| **23 s** | — |
| 本 mod，完整写回 | **40 s** | — |
| 本 mod，跳过 aquifer | 32 s | `C2ME_WGPU_SKIP_AQUIFER` |
| **本 mod，跳过全部写回** | **11 s** | `C2ME_WGPU_MIN_STAGE` |

**结论：**

1. **写回占本 mod 40s 里的约 29s**（40−11）。之前判断的"GPU 往返税占一半"
   **被推翻**——MIN_STAGE 下 GPU 路径仅 11s，远低于纯 CPU 23s，说明往返税
   被"批够快 + GPU 富余"吸收了，**没有显现**。
2. **GPU 密度管线本身很快**：写回一移走，11s ≪ 23s。
   **整个净负全部来自"写回留在 CPU"。**
3. 因此 §20 的赌注上调：不是"最多拿回 8s"，而是**写回搬得越多、越接近甚至
   反超纯 CPU**。

**caveat**：MIN_STAGE 写更少的方块（只有 `d>0` 的石头，无水/矿/地表），所以
29s 里有一部分是"少写方块"而非"少跑 aquifer/beard/vein"。**这是天花板，不是
精确值**——真实全卸载后写方块（palette 写入）仍要花时间。

**另一个负面结果**：调大攒批窗口（`C2ME_WGPU_BATCH_WAIT_MS` 5→20）能让批
从 ~5 涨到 ~10，但**端到端不变（44→46s）**——批大小受"请求到达速率"限制，
不是窗口。调窗口是死路。

**下一步**：最现实的第一步是 **aquifer 整体搬 GPU**（§17 的 3 个阻碍），
因为它单项最大（~8s）。

### 20.6 下次开工清单（2026-10-10 记）

1. **§20 aquifer 整体 GPU 化**（写回最大单项，§20.5 测到 ~8s）。第一步：啃
   `estimateSurfaceHeight` 跨 chunk 依赖（§17 阻碍1）。
2. **§14 Epic Terrain interpolated-边界 bug 修复**。方向与 §20 一致（都是
   "接管 MC 密度管线、喂 GPU 数据"），**可能共用脚手架**。修法（§14 已记）：
   拦截 `DensityInterpolator.fill()`，用 GPU 角点值填充 MC 缓冲，让 MC 自己
   完成"插值 + 外层求值"（当前是"整棵树角点插值"，遇非线性外层即错）。
3. **shader 分片是硬性约束**：新增任何 GPU 计算**必须分片成多个 pass**
   （沿用 §5.15.5 的 A/B/C shard 机制）。Mali-G610（天玑 SOC）在单 shader
   过大时会**设备丢失**，且编译成本对规模**超线性**（§5.15.2）。§20 会显著
   增加 shader 体积，务必保持每片在安全区（§5.15.5 实测 ~2100 行 / 编译
   ~1.6s 稳；shard B 3083 行已到 9.6–10.2s，属危险边缘）。
