# C2ME-wgpu-rs

为 Minecraft 提供由 [wgpu](https://wgpu.rs/) (Vulkan / Metal / DX12 / WebGPU) 驱动的 GPU 加速世界生成。

这是一个 [C2ME](https://modrinth.com/mod/c2me-fabric) 的伴生模组，将区块生成中繁重的噪声 / 密度函数管线卸载至 GPU 计算。

> [English](README_en.md) | **简体中文**

> ⚠️ **实验性功能 — 安装前请阅读。** 
> 本模组现已将整个密度管线（噪声、含水层、结构地基）卸载至 GPU 并直接输出方块状态；CPU 仅负责写入区块调色板。
> 在桌面端 AMD 核显 (RADV) 上，Chunky 端到端测试 (r=500, 4225 区块) 耗时约 15 秒，而纯 CPU 跑 C2ME 约需 23 秒。
> 在移动端 GPU (Mali-G610) 上，同样的测试耗时 2 分 19 秒，而纯 CPU 需 3 分 12 秒（**端到端提速 61%**）。
> 分析 Chunky 的连续生成速率显示，纯地形生成吞吐量提升了 **+30.3%**（30.65 CPS vs 23.52 CPS）。
> 旧版“弱 GPU 可能更慢”的警告已不再适用。

> **兼容性优先。** 
> 本模组刻意基于 **fp32** 构建，以兼容*尽可能广泛的 GPU*——包括核显和移动端 SoC。
> 官方的 C2ME OpenCL 加速模块基于 `fp64`（仅在较新的 Minecraft 版本中可用）；本模组通过覆盖 1.21.1 以及 `fp64`（和 `shader-f64`）无法触及的设备来作为补充。
> 这里的高精度仅通过 fp32 内部补偿来实现；刻意绝不使用 `f64`，因为那会将大多数移动端 GPU 拒之门外。

---

> 💡 **健壮性提示**：如果遇到底层驱动崩溃或 GPU 掉线，游戏会为了防止死锁而安全退出，并将详细的崩溃原因写入 `logs/c2me-wgpu.log`。开发者可通过该文件排查问题。
>
> ⚠️ **跨平台行为差异**：
> - **桌面端**：如果强制指定了不兼容的后端（如 `WGPU_BACKEND=gl`），wgpu 底层 panic 会被安全捕获，游戏**静默回退到 CPU**，不会闪退。
> - **Android 端**：由于 Android 交叉编译默认 `panic=abort`，类似情况会导致**游戏直接闪退**。但崩溃原因会被写入 `logs/c2me-wgpu.log`，请勿在移动端随意指定 `WGPU_BACKEND`。

## 环境要求

| | |
|---|---|
| Minecraft | 1.21.1 (Fabric) |
| Fabric Loader | >= 0.17.2 |
| **C2ME** | **必需** (本模组为伴生模块，不能独立运行) |
| Java | 25+ (本模组基于 Java 25 构建，并使用 Panama FFM API) |
| GPU | 任何支持 wgpu 后端的 GPU |

需要 Java 25 或更高版本：模组针对 Java 25 编译（并通过 Panama FFM API 与其 Rust 核心通信，该 API 从 Java 24 起稳定）。

> **为什么依然叫 "C2ME-wgpu-rs"？** 
> 本模组自带 access widener，并且需要 C2ME。
> 
> **在构建期**，Java 代码直接引用了 C2ME 的类（如 `SchedulingManager`）。为了避免拉取整个 C2ME 源码，我们提供了一个小巧的 **仅编译期 stub**（`src/stub`, `compileOnly` — 不会打包进 jar）。这意味着编译不需要 C2ME 的 jar，也不会锁定任何特定的 C2ME 版本。
> 
> **在运行期**，它强依赖 C2ME 做两件事：
> 1. 多线程区块生成，以保持 GPU 批处理管线满载（有 C2ME 时，桌面端吞吐量可达 ~144 区块/秒；没有它，GPU 会因吃不满而效能骤降至 ~13）。
> 2. 异步写回架构严重依赖 C2ME 的 `SchedulingManager.positionedExecutor()` 在其 worker 池上执行 CPU 任务。如果该 API 不可用，该次运行的 GPU 加速将被完全禁用，退回纯 C2ME 的 CPU 生成（不存在同步 GPU 回退机制）。
> 
> 因此，它依然是一个 **C2ME 伴生模块** —— C2ME 提供并行度和调度器，本模组提供 GPU 算力 —— 名字得以保留。

---

## 安装

1. 为 Minecraft 1.21.1 安装 Fabric Loader。
2. 将 **C2ME** 和 **C2ME-wgpu-rs** 放入 `mods/` 文件夹。
3. 启动游戏。

无需任何配置。如果找到了兼容的 GPU **且** C2ME 的调度 API 可用，地形生成会自动卸载至 GPU；否则，模组保持静默，由纯 C2ME 在 CPU 上生成地形，与往常无异。

---

## 平台支持

| 平台 | 状态 |
|---|---|
| Linux x86_64 | ✅ 已测试 |
| Android aarch64 (自定义 JVM) | ✅ 已测试 (Mali-G610 / 天玑) |
| Windows x86_64 | ✅ 已测试 (AMD, mingw DLL) |
| macOS (Metal) | ⚠️ 有 wgpu 后端，未官方测试 |

在 Android 上，自定义 JVM **不支持** `-XX:+UseZGC`，模组也不会在那里给出该建议。

---

## 工作原理

1. **序列化** — 原版的 `NoiseRouter`（驱动地形的密度函数树）在 Java 侧被序列化为紧凑的二进制 blob。
2. **交接** — 该 blob 通过 Panama FFM API 传给 Rust `cdylib`（无回调，简单的轮询模型）。
3. **编译** — Rust 反序列化该树，运行优化 pass，并生成 WGSL 计算着色器，随后编译为 GPU 管线（并缓存在磁盘上）。
4. **分发** — GPU 一次 dispatch 即可计算整批区块的密度场。为了正确处理模组中非线性的密度函数（例如 `squeeze`），最终的密度树会被拆分（即 "Pass E" 架构）：插值子树在角点求值，而非线性外层在最后的方块分类时逐方块应用。
5. **方块状态** — GPU 自行对每个方块进行分类（空气 / 石头 / 水 / 岩浆），在 GPU 上应用含水层、结构地基（Beardifier）以及**矿脉采样器**（使用原生的 GPU Xoroshiro 随机数生成器）。Java 仅将得到的逐方块状态字节写入区块调色板。

---

## 精度

模组使用 **fp32** 进行计算。与 Minecraft 的 `double` 参考值相比：

- 浅层表达式一致度约为 **1e-6**；
- 深层表达式（数千次噪声采样）最多可漂移约 **1e-2**。

这是 **fp32 的固有极限**，并非 bug。地形阈值对此量级的差异不敏感，且生成的地形与原版在视觉上无法区分。

---

## 已知限制

- 在末地极远坐标（约 37 万格）处，使用部分 NaN 回退机制来复现原版的虚空带。
- **模组密度函数**：GPU 序列化器无法原生理解的定制密度节点（例如来自地形模组的节点）会在 CPU 上以相同的角点网格计算，并作为“外部输入”喂给 GPU。树的其他部分依然在 GPU 上运行，因此模组世界生成保持工作 —— 只有不受支持的子树会承担 CPU 开销。

---

## 环境变量

### GPU 选择（标准 wgpu 变量）

模组遵循 wgpu 自身的环境变量，因此它们的行为与任何其他 wgpu 应用程序一致：

| 变量 | 值 | 效果 |
|---|---|---|
| `WGPU_BACKEND` | `vulkan`, `dx12`, `metal`, `gl`, `webgpu` (逗号分隔) | 限制尝试的图形后端。在某个后端损坏时很有用。注意：`gl` 后端需要 OpenGL 4.3+ 才能支持通用计算 (Compute Shader)。 |
| `WGPU_ADAPTER_NAME` | 子串 (不区分大小写) | 按名称强制指定特定的 GPU，例如 `WGPU_ADAPTER_NAME=radeon`。如果没有匹配项，模组会记录一条警告并回退到其默认选择。 |

如果不使用这些变量，模组会使用 Vulkan / Metal / DX12（取决于操作系统提供的），并优先选择**集成显卡**，从而让独立显卡空闲以用于渲染。

### 诊断与调试导出

仅在排查故障时设置这些变量。它们都在启动时读取一次。

| 变量 | 值 | 效果 |
|---|---|---|
| `C2ME_WGPU_DEBUG` | `1` | 启用详细日志和定期的每秒区块数（`CPS=`）统计。同时将 CPS 打印到聊天 HUD。写入 `<gameDir>/logs/c2me-wgpu.log`。 |
| `C2ME_WGPU_PROFILE` | `1` | 将分阶段的 GPU 批处理计时（encode / submit / readback / decode）写入日志。 |
| `C2ME_WGPU_ASYNC_SCHED` | `0` | 禁用 C2ME worker 池上的异步写回。便于 A/B 基准测试。 |
| `C2ME_WGPU_MAPPED_AT_CREATION` | `0` / `1` | 覆盖缓冲区上传方法。桌面端默认 `1`，Android 端默认 `0`（Mali 驱动拒绝映射路径）。 |
| `C2ME_VANILLA_REF` | `1` | 完全跳过 GPU，让原版生成，然后 dump 方块以供离线对比。 |
| `C2ME_DUMP_BLOCKS` | `1` | 为目标区块 dump 方块状态以供离线对比。 |
| `C2ME_DUMP_BLOB` | 文件路径 | 将序列化的 NoiseRouter blob dump 到指定路径。 |
| `C2ME_DUMP_DIR` | 目录 | dump 文件写入的目录。默认为平台临时目录。 |
| `C2ME_DUMP_CHUNK` | `x,z` 或 `x1,z1,x2,z2` | `C2ME_DUMP_*` 诊断的目标区块。 |
| `C2ME_DUMP_AQUIFER` | `1` | dump 含水层 cell 表。 |
| `C2ME_DUMP_AQ_BLOCKS` | `1` | dump 逐方块含水层参考数据。 |
| `C2ME_DUMP_DENSITY` | `1` | dump GPU 密度值。 |
| `C2ME_DUMP_BEARD` | `1` | dump Beardifier 结构地基数据。 |
| `C2ME_WGPU_FORCE_SINGLE` | `1` | 仅开发用：强制使用单 shard 全通道着色器进行调试。 |
| `C2ME_WGPU_MINIMAL_SHADER` | `1` | 仅开发用：使用最小占位着色器进行启动测试。 |
| `C2ME_WGPU_MAX_CHANNELS` | 整数 | 仅开发用：限制发出的通道数量。 |
| `C2ME_WGPU_FAKE_GPU_MS` | 整数 | 仅开发用：通过注入延迟来模拟较慢的 GPU。 |
| `C2ME_WGPU_NO_PIPELINE_CACHE` | `1` | 仅开发用：绕过磁盘上的着色器管线缓存。 |

blob dump 也可以通过 JVM 属性 `-Dc2me.dumpBlob=<path>` 设置；当该属性未设置时，环境变量仅作为回退。

**原生库解压位置排查**：
默认情况下，模组会将 jar 包内的 `.so` 或 `.dll` 解压到游戏根目录下的 `c2me-wgpu-natives/` 文件夹中。如果该目录无法写入或加载（如遇到 `noexec` 权限限制），模组会自动回退解压到操作系统的临时目录（如 Linux 的 `/tmp`）。
如果你需要排查底层库加载失败的问题，可以在 JVM 启动参数中添加 `-Dc2me.wgpu.preserveNative=true`，模组将把原生库直接解压到游戏当前的工作目录中。

> **注意：** 以 `C2ME_DUMP_` 为前缀的变量以及标记为“仅开发用”的变量严格用于故障排查，正常游玩时无需设置。

> **Rust 侧日志文件：** 原生库将其诊断信息写入 `<gameDir>/logs/c2me-wgpu.log`（路径由 Java 在启动时传入）。这独立于 Minecraft 自身的 `logs/latest.log`，后者仅捕获 Java 日志。

### 缓存位置

| 变量 | 效果 |
|---|---|
| `XDG_CACHE_HOME` | 磁盘着色器管线缓存的基础目录（`$XDG_CACHE_HOME/c2me/wgpu_pipelines/`）。 |
| `HOME` | 当 `XDG_CACHE_HOME` 未设置时的回退（`$HOME/.cache`）。 |

> **注意：** 缓存路径遵循 XDG 约定，这是 Linux 原生特有的。在 Windows 和 macOS 上，这些变量通常未设置或非标准，因此那里的磁盘管线缓存实际上被禁用了。这会导致每次启动都需要一次性的着色器重新编译，但不影响正确性。

### 仅构建期使用

| 变量 | 效果 |
|---|---|
| `ANDROID_NDK_HOME` | 存在时，构建**会自动**为 Android (aarch64) 交叉编译原生库，并将其暂存到 jar 内的 `natives/android-aarch64/` 下。取消设置它（或传入 `-PskipAndroid`）以跳过。 |

---

## 从源码构建

```bash
./gradlew build
```

构建出的模组 jar 会写入 `build/libs/`。

构建需要：
- JDK 25（用于 Java 工具链）和 JDK 21（用于 Gradle JVM）；
- Rust 工具链（原生库作为构建的一部分进行编译）；
- **`cargo-ndk`** *仅当*您为 Android 构建时
  （`cargo install cargo-ndk`，外加一个 Android NDK；当 `ANDROID_NDK_HOME` 未设置时，Android 构建会自动跳过，或通过 `-PskipAndroid` 强制关闭）。

---

## 开发说明

本模组是在**大量 AI 辅助**下开发的 —— 代码、文档和测试大部分是在维护者审查下由 AI 生成的。请对此保持警惕：在依赖它之前，请务必在您自己的硬件上验证行为。欢迎提交 Bug 报告和 Pull Request。

### 运行测试的注意事项

**动地形相关代码后必须额外跑对照测试。** `real_router_vs_vanilla` 是唯一验证 GPU 输出与真实 Minecraft 一致的测试，因耗时约 2 秒已默认忽略。改动 emitter、aquifer、beard、矿脉或反序列化后请手动执行：

```
cd rust && cargo test --lib -- --ignored real_router_vs_vanilla
```

`cargo test` 在集成显卡上可能偶发 GPU 测试失败（约每 5~8 次一次）。原因是测试为每个用例单独创建 executor（各约 600 MB GPU buffer），并行执行时资源叠加；这是测试架构问题，与游戏运行路径无关——游戏内只有一个 executor 且由 Mutex 串行化。若复现，请附完整 `cargo test` 输出以便定位。

---

## 来源说明

本项目是一个独立的、净室（clean-room）重新实现。它是针对 Minecraft 的**运行时行为**（导出的噪声数据和采样的参考值）编写的，而不是针对任何第三方模组的源代码。

一次代码来源审查将本模组的含水层和 Beardifier 实现与 [C2ME](https://modrinth.com/mod/c2me-fabric) 的 OpenCL 加速器进行了比较：没有共享标识符，没有复制的函数，且架构不同（C2ME 逐函数镜像 Minecraft；本模组使用预计算表 + GPU 扫描设计）。在两者看起来相似的地方（例如 `fastInverseSqrt`, `getStructureWeight`），共同的来源是 Minecraft 自己的原版代码，这两个项目都是独立复现的。

范围说明：审查是在特性/架构层面进行的，而不是逐字符差异对比。Minecraft 本身不被重新分发；仅提供原创代码。参见 `LICENSE`。

---

## 许可证

MIT — 参见 [LICENSE](LICENSE)。
