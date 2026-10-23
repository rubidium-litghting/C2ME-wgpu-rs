# C2ME WGPU 加速模块 — 开发文档

> 最后更新：2026-10-17
> 状态：**0.1.5 发布版**。完整实现 Pass E、GPU 矿脉、100% 对齐原版的 AquiferSampler 逻辑，并重构了 FFM 日志架构与底层防死锁机制。实机验证兼容 Epic Terrain 与 Tectonic。
> 移动端实测：Chunky A/B 测试在 Mali-G610 平板上，端到端总耗时从 3m12s 缩短至 2m19s（+61%）；纯地形生成实时吞吐从 23.52 CPS 提升至 30.65 CPS（+30.3%）。
> 服务端实测 (Headless，无渲染干扰)：Chunky r=500 跑纯原版地形，纯 CPU 耗时 27s (平均 121.1 cps)，GPU 加速耗时 21s (平均 143.6 cps，峰值 215.7 cps)。端到端提速 22%，峰值提速 44.7%。注：纯 CPU 模式下主线程排队（Amdahl 定律）会锁死并发上限，CPU 占用率通常只能达到 70-80%。
> 桌面端到端反超纯 CPU（Chunky r=500：15 s vs 23 s）。GPU 直接输出 per-block 方块状态，CPU 只写 palette。
> 流水线化 A/B（Chunky r=500，原版地形，同一台 RADV RENOIR 核显）：executor 串行 20 s / CPU ~70%；
> 池化输入 buffer + 多批次在途 18 s / CPU ~50%。CPS 中位持平（188），p90 238→254。
> **结论：流水线化省的是 CPU 等待与端到端时间，不是稳定态 CPS**——GPU 执行（readback）仍是瓶颈。
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
| **aquifer 全 GPU（水/岩浆/石分类）** | ✅ 100%（完整 barrier 与 calculateDensity） |
| **Beardifier（结构地基）GPU** | ✅ bit-exact |
| 矿脉 | ✅ 全 GPU（RNG 与方块替换） |
| **端到端反超纯 CPU** | ✅ 15 s vs 23 s (桌面) |
| **移动端端到端提速** | ✅ 34.8 cps vs 21.5 cps (+61%) |
| 史诗地形/构造地形兼容 | ✅ Epic Terrain, Tectonic |
| WeirdScaledSampler | ✅ 完整实现 |

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
24. **跨语言日志需防截断**：Rust 的 `panic` 若触发 `abort`，JVM 会被瞬间带走。务必在 FFI 边界拦截，并用 `sync_all()` 强刷黑匣子文件。
25. **死锁下 `System.exit` 无效**：当主线程死锁时，`System.exit` 的 Shutdown Hook 会被卡死。必须调用 `LogManager.shutdown()` 刷盘后，使用 `Runtime.getRuntime().halt(1)` 强杀进程。
26. **FFM 导出的符号必须带 `#[unsafe(no_mangle)]`**：即便是 `extern "C"` 函数，如果漏掉了这个宏，Rust 编译器依然会改写符号名，导致 Java 端调用 `lookup.find()` 时抛出 `NoSuchElementException`。清理代码时绝不能误删。

27. **低频偶发失败先抓证据，别倒推**：`world::tests` 里 9 个 GPU 测试各自 `ExecutorThread::spawn` 一套约 600 MB buffer，16 线程并行时资源叠加，集显上约 1/5~1/8 概率偶发失败，且无固定 panic 原文。曾误判为显存 OOM——实际 `vulkaninfo` 显示 device-local budget 4.86 GiB、`dmesg` 无 amdgpu OOM 记录，10 次连跑不复现。生产路径 `get_or_create_executor()` 是 `OnceLock<Mutex<>>` 单 executor + 串行，**游戏运行不受影响**。修复方向（暂缓）：测试改用小 `ExecutorConfig`（batch 实际只有 1~3 个 chunk，却按 64 预分配），或加全局 GPU 锁。复现时务必保留完整 stderr + `free -h` + `mem_info_gtt_used` + `dmesg | tail`。

28. **动地形正确性必须跑 `real_router_vs_vanilla`**：它是唯一对照真实 Minecraft 输出（`src/test/resources/real_router.ref.txt`）的测试，因耗时约 2s 已默认 `#[ignore]`。改动 emitter / aquifer / beard / 矿脉 / 反序列化，或任何会改变 WGSL 数值输出的代码后，必须手动跑 `cd rust && cargo test --lib -- --ignored real_router_vs_vanilla`。其余 84 个测试全绿**不能**替代这一步——它们只验证"CPU 参考实现与 GPU 一致"，不验证"GPU 与 vanilla 一致"（见教训 #20）。

### 待办与 0.1.5 规划（2026-10-18 更新）

**0.1.4 已完全解决的历史遗留（存档记录）**
- ✅ **§14 边界 bug**：通过 Pass E 架构（拆解 nonlinear outer 树与 InnerRef 角点）解决，Epic Terrain/Tectonic 地形完全正常。
- ✅ **Barrier 噪声与含水层对齐**：完整移植原版 `calculateDensity`（含高度过滤和全局流体判定），彻底告别全水浮岛。
- ✅ **矿脉全 GPU 化**：完整的 RNG 与矿脉方块替换已实装，无需 CPU 回退。
- ✅ **Shard B 瘦身**：高频内联公式函数化，从 3084 行降到 1771 行。

**0.1.5 目标：代码整理与用户友好化（进入等 Issue 阶段）**
- [x] **代码清洁**：
  - 清理排查期间留下的历史冗余代码、无用 `#[allow(dead_code)]`。
  - 审查并移除重复的 `c2me_wgpu.log` 日志写入逻辑（如果有）。
  - 修复项目内大量的 cargo `unused` 警告。
- [ ] **用户友好化（UX）**：
  - [x] **国际化/本地化 (i18n)**：聊天栏提示已全部迁入 `lang/en_us.json` /
    `lang/zh_cn.json`，通知走 `Text.translatable`。`WgpuFallbackNotice` 队列里存的是
    key + 参数而非拼好的字符串——worker 线程不知道客户端语言。
  - [x] **配置文件系统（替代环境变量）**：`config/c2me-wgpu.json`，手写读写、无第三方
    库。损坏的 JSON 备份为 `.broken` 后重建，越界值 clamp 并回写。本模组已不读取任何
    `C2ME_*` 环境变量。
  - [x] **聊天栏提示**：GPU 回退原因、GC 建议已有。
  - [ ] **GPU 选择**（唯一未落地的功能项）：
    `gpu_index` 字段已存在但未被读取，`create_instance` 也不接收它。需要：
      1. Rust 提供 `c2me_wgpu_list_gpus`，用临时 instance 枚举 adapter，返回带序号的列表。
      2. Java 首次生成配置时把列表以注释形式写进 JSON（仅供人读）。
      3. `create_instance` 接收 `gpu_index`；Rust 按序号选卡，越界或 feature 不足时回退
         默认逻辑（优先集成显卡）。
    - `WgpuInstance` 目前是 `OnceLock` 单例，换卡需要重建 instance——这正好也是热重载
      的前置（见下一条）。
  - [ ] **切换存档时热重载配置**：
    目前配置是启动时读一次，改完必须重启。存档切换是天然的重置点：所有维度都会销毁重建。
    - 配置分三类：A 类影响 instance（`gpu_index`、`disable_pipeline_cache`）、
      B 类影响 world 创建（`mapped_at_creation`）、C 类只影响行为（`debug_*`）。
    - C 类只需重读配置；B 类要销毁 `DimState`；A 类要重建整个 instance。
    - 关键顺序：`shutdown()` 全部 coordinator → 排空 inflight → `world_destroy` →
      清 `DIM_STATES` →（必要时）重建 instance。
    - **前置**：`shutdown()` 已会 join 两个线程并排空 inflight（本轮已做），
      所以同步销毁是安全的，不再需要 5 秒延迟。
    - 风险：重建期间完全无 GPU 能力，必须保证 C2ME 走 CPU 路径。
  - [ ] **状态展示（三项合并做）**：
    - 加载界面提示（正在编译着色器 / 已启用 GPU 名称 / 已回退及原因）
    - 展示当前 GPU 名称与硬件状态（`get_gpu_name` 之前删掉了，需要重新加回，但这次
      不要泄漏 `CString`）
    - F3 调试屏扩展，显示当前 CPS
  - [ ] **ModMenu GUI（软依赖）**：
    - 绝不可将 ModMenu / Cloth Config 设为强制依赖，仅 `modCompileOnly`。
    - **不引入第三方 GUI 库**，用 MC 自带的 `GridWidget` / `DirectionalLayoutWidget` /
      `ScrollableLayout` 手搓。封装 `WgpuRowHelper` 提供 `addBoolean()` / `addIntSlider()`。
    - 翻页式或原生滚动条，防溢出。
    - 没装 ModMenu 时，手改 JSON 必须功能完整。
    - 优先级最低：纯便利性，JSON 已经能用。

- [x] **健壮性**：
  - 增加更多边缘平台（如未测试的 macOS Metal）的预防性回退逻辑。
  - 统一处理未捕获的 GPU DeviceLost，确保游戏不会因为驱动崩溃而卡死。
- [x] **文档梳理**：
  - 整理目前的 Debug 环境变量说明，把不再使用的变量（如 `MIN_STAGE`）彻底从文档中删除。
  - 补充了 FFM Upcall 日志回传机制与防死锁机制的说明。

### 环境变量速查

| 变量 | 值 | 效果 |
|---|---|---|
已全部迁移到 `config/c2me-wgpu.json`，见 §12。本模组不再读取任何 `C2ME_*` 变量。
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

> **2026-10：C2ME 依赖关系**（§10）。构建期通过 Stub 提供接口签名，运行期强依赖调度器
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

## §10 与 C2ME 的关系（构建期 Stub + 运行时强依赖）

- **构建期（Stub 依赖）**：为了不拉取整个 C2ME 源码，我们通过 `src/stub`（`compileOnly`）提供了 `SchedulingManager` 等类的接口签名。Java 代码中依然直接引用了这些类。
- **运行时强依赖**：实测无 C2ME 时桌面 CPS ~144→~13（缺多线程并行，批攒不满）。同时，异步写回架构强依赖 `SchedulingManager.positionedExecutor()` 来执行，若获取不到调度器，GPU 加速将直接回退纯 CPU 路径。
- **自写 AW**：不再借 C2ME 的 access widener，避免其删条目导致 `IllegalAccessError`。

## §11 混合执行（任意 modded 密度函数兜底）

Rust 不认识的密度节点（地形 mod）在 CPU 按同一角点网格求值，作为 "external input"
喂给 GPU（binding 6）。其余树仍走 GPU，**modded worldgen 保持工作**，只有不支持的子树付 CPU 成本。
实机验证：Tectonic、Epic Terrain 正常。

## §12 诊断基础设施

- Rust 日志写 `<gameDir>/logs/c2me-wgpu.log`（Java 启动时传路径）。
所有开关都在 `config/c2me-wgpu.json`，本模组不再读取任何 `C2ME_*` 环境变量：

- `debug_verbose_logging`：shader 编译计时、缓存命中、CPS 统计。
- `debug_stage_timing`：分阶段计时，GPU 侧 `[batch-stage-timing]`（encode/submit/
  readback/decode）与 CPU 侧 `[writeback-timing]`。两组要一起读才有意义。
- `debug_dump_*`：离线参考 dump（密度/方块/blob/beard/aquifer 表）。

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
- §14 interpolated 边界 bug（已通过 Pass E 架构修复）。
- §15 JFR 热点：写回瓶颈在 CPU 逐方块（aquifer/heightmap/vein）。
- §16 GC 对负优化贡献：实测排除。
- §17 aquifer 逐帧归因：~65% 写回，一半控制流一半噪声。
- §18 各种"取巧"思路：heightmap 一次性、aquifer 部分 GPU 化等，多数证伪。
  **结论：够长的杠杆是"整体写回 GPU 化"（§20–§22 已实现）。**
### 日志架构与防死锁机制（0.1.5 新增）

- **FFM Upcall 日志回传**：Rust 侧不再直接向 `STDERR` 打印调试信息。所有的 `wgpu_debug!` 调用会通过 Panama FFM 的回调函数（`upcallStub`）直接传给 Java 端的 SLF4J `LOGGER`，完美合并进 `latest.log`，且格式统一。
- **黑匣子防崩溃日志**：Rust 侧独立保留了对 `c2me-wgpu.log` 的写入。每次写入后执行 `f.flush()`，并在 `setup_panic_hook` 中执行 `f.sync_all()`，确保即便发生 `SIGABRT`（如 `panic=abort` 或 Android 驱动崩溃），底层原因依然能落盘。
- **强制退出防死锁**：`pollStatus` 超时（原生 worker 无响应）时，主线程会永久卡在 C2ME 的 Future 上。此时 `WgpuBatchCoordinator` 调用 `LogManager.shutdown()` 刷出 Java 日志，然后 `Runtime.getRuntime().halt(1)` 强杀进程。批次级失败不走这条路径，见下节。
- **原生库解压策略**：默认将 `.so`/`.dll` 解压到 `<gameDir>/c2me-wgpu-natives/`，避开系统临时目录可能存在的 `noexec` 权限限制。若游戏目录不可写或 `dlopen` 失败，自动清理并回退到系统临时目录（`/tmp`）。同时，解压后自动追加可执行权限，确保 Android 等严格沙箱能正常加载。
- **FFI 边界护甲**：`c2me_wgpu_create_instance` 使用 `extern "C-unwind"`，并用 `std::panic::catch_unwind` 包裹。若 wgpu 底层在初始化时 panic（典型场景：强制指定 `WGPU_BACKEND=gl` 但 EGL 返回 `BadAccess`），可以安全地捕获并回退到 CPU。
- **⚠️ 跨平台崩溃行为差异（重要）**：
  - **桌面端**（Windows/Linux/macOS）：Rust 默认 `panic=unwind`，`catch_unwind` 能有效捕获 wgpu 的 panic，游戏**静默回退到 CPU**，不会闪退。
  - **Android 端**：交叉编译默认 `panic=abort`，`catch_unwind` 失效。wgpu 的 panic 会触发 `SIGABRT`，**JVM 会被瞬间强杀**，游戏直接闪退。此时唯一可依赖的是提前注册的 `setup_panic_hook`，它会在 `abort` 前执行 `sync_all()`，把 panic 堆栈强行写入黑匣子 `c2me-wgpu.log`。
  - **结论**：Android 用户如果设置了不兼容的 `WGPU_BACKEND` 环境变量（如 `gl`），游戏会闪退，但不会丢失崩溃原因。桌面端用户可以放心随意切换后端。

### 运行期错误处理的分层设计

失败按"能否确定状态"分三层处理，不是一刀切：

- **初始化失败 → 回退 CPU**。`c2me_wgpu_create_instance` 是 `extern "C-unwind"` 且被
  `catch_unwind` 包裹，adapter 选择、feature 检查、shader 编译失败都能捕获，Java 侧
  得到 `STATUS_INIT_FAILED`，`ModuleEntryPoint` 记录原因并让该维度走 vanilla/C2ME
  CPU 生成。此时还没分配任何 GPU 资源，回退是干净的。

- **批次失败 → 该维度降级 CPU**。`execute_batch` 返回 `Err`（executor 获取失败、锁
  中毒、单次提交出错）时 `status = STATUS_ERROR`。这是可恢复条件：`complete()` 取走
  handle、标记 `gpuFailed`、把失败批次的 future 全部异常完成。

  但仅仅完成异常 future 是不够的——C2ME 会释放位置锁（`ScheduledTask` 的
  `whenComplete` 对异常完成同样执行），然后**放弃这个 chunk**，它永远停在
  "upgrade to noise" 状态，依赖它的东西全部等死。所以失败回调必须自己把 chunk
  用 vanilla 重新生成：置 `FAILED_DIMS`，再调一次 `populateNoise`，重入时
  `acquireDimState` 看到标记直接返回、不 cancel，vanilla 正常生成。

  `FAILED_DIMS` 的检查必须在 `acquireDimState` **最开头**、缓存查找之前。放在之后
  是初版的 bug：失败回调重入时先拿到缓存里的死 `DimState`，继续往已死的 coordinator
  提交，实测刷了 16 万行错误且地形完全不生成。

  清理有意不做。`shutdown()` 会 join 批处理线程，而失败回调跑在完成线程上——在那里
  调 `shutdown()` 两者互等。失败维度的 GPU 资源留到下次切存档时释放，代价是
  有界的（一个维度的 buffer），优于挂起。

- **超时 / 未知状态 → halt**。`pollStatus` 超过 30 秒仍 PENDING，或返回既非 DONE 也
  非 ERROR 的状态，说明原生 worker 已经不响应了。此时 Java 侧无从判断 device 是否
  还能用，`LogManager.shutdown()` 刷盘后 `Runtime.getRuntime().halt(1)`。

**为什么超时不能降级？** 降级需要能保证"这个 chunk 会被重新生成"。但超时意味着 Rust
侧的 `BatchRequest` 可能永远不完成，Java 侧持有的 handle 无法释放，对应的 chunk 永远
等不到结果。这与批次失败不同——批次失败至少 Rust 侧返回了确定的状态。

**`on_uncaptured_error` 覆盖到哪一层？** wgpu 的 validation error 和 OOM 被它拦截，
只记日志不 panic，`execute_batch` 会返回 `Err` 走降级路径。它**覆盖不到** `device
lost` 和 wgpu 内部断言失败——那两类表现为 panic，最终仍会走到 halt。

**已知未覆盖**：如果 `execute_batch` 内部 panic（而非返回 Err），`status` 停在
`STATUS_PENDING`（panic 发生在 `store` 之前），Java 侧只能等 30 秒超时后 halt。
要覆盖这条需要在 `submit_async` 的回调外再包一层 `catch_unwind`，并引入
`STATUS_PANICKED`。`setup_panic_hook` 已经把堆栈 `sync_all()` 写进
`c2me-wgpu.log`，所以即使走到 halt，崩溃原因也不会丢。

### §23 执行器流水线化（0.1.5）

**问题**：`run_loop` 是 `prepare → submit → poll(Wait) → 读回 → 下一批`。GPU 算的
时候 CPU 在干等，且 `poll(Wait)` 直接挂起线程。

**两处改动**：

1. **poll 驱动**。wgpu 的 `PollType::Poll` 是非阻塞单次检查，且**只有被调用时**
   `map_async` 的回调才会完成。Java 的 `pollStatus` 每 50µs–2ms 调一次，正好可以
   充当心跳——`c2me_wgpu_poll_status` 里顺带 `device.poll(Poll)`，Rust 侧不再需要
   自己阻塞等待。executor 兜底再加一次 `poll(Poll)`，防止 Java 心跳停摆。
2. **输入 buffer 池化**。这是流水线的前提：`params`/`positions`/`aquifer`/`beard`/
   `external` 若是单份，准备批次 N+1 会覆盖批次 N 仍在读的数据。现按
   `ExecutorConfig::pipeline_depth`（默认 3）分配多套，`prepare` 时 acquire、
   `finish` 时归还。
   `output_buf`/`block_buf` **刻意不池化**：每批写入后在同一 command buffer 内
   copy 到 staging，GPU 的提交顺序本身就保证了安全。

**实测（Chunky r=500，原版地形，RADV RENOIR）**：

| | 串行 | 流水线化 |
|---|---|---|
| 端到端 | 20 s | **18 s** |
| CPU 占用 | ~70% | **~50%** |
| CPS 中位 | 188.5 | 188.4 |
| CPS p90 | 238.1 | **254.2** |

**教训**：`batchMs` 在两种模式下口径不同，不可直接对比——串行时它只计 map 段，
流水线化后包含完整在途时间。只有端到端和 CPU 占用是可比的。

**瓶颈仍是 GPU**：`readback` 占每批 90%，是 compute + 拷贝的真实时间，批次再多
也无法重叠。想再提 CPS 只能减少 GPU 工作量（降分辨率、减 shard），代价是地形质量
与移动端兼容性。

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

