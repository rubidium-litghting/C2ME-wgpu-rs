# C2ME-wgpu-rs

为 Minecraft 提供由 [wgpu](https://wgpu.rs/) (Vulkan / Metal / DX12 / WebGPU) 驱动的 GPU 加速世界生成。

这是一个 [C2ME](https://modrinth.com/mod/c2me-fabric) 的伴生模组，将区块生成中繁重的噪声 / 密度函数管线卸载至 GPU 计算。

> [English](README_en.md) | **简体中文**

> ⚠️ **实验性功能 — 安装前请阅读。** 
> 本模组现已将整个密度管线（噪声、含水层、结构地基）卸载至 GPU 并直接输出方块状态；CPU 仅负责写入区块调色板。
> 在桌面端 AMD 核显 (RADV) 上，Chunky 端到端测试 (r=500, 4225 区块) 耗时约 18 秒，而纯 CPU 跑 C2ME 约需 23 秒。
> 在移动端 GPU (Mali-G610) 上，同样的测试耗时 2 分 19 秒，而纯 CPU 需 3 分 12 秒（**端到端提速 61%**）。
> 分析 Chunky 的连续生成速率显示，纯地形生成吞吐量提升了 **+30.3%**（30.65 CPS vs 23.52 CPS）。
> 旧版“弱 GPU 可能更慢”的警告已不再适用。

> **兼容性优先。** 
> 本模组刻意基于 **fp32** 构建，以兼容*尽可能广泛的 GPU*——包括核显和移动端 SoC。
> 官方的 C2ME OpenCL 加速模块基于 `fp64`（仅在较新的 Minecraft 版本中可用）；本模组通过覆盖 1.21.1 以及 `fp64`（和 `shader-f64`）无法触及的设备来作为补充。
> 这里的高精度仅通过 fp32 内部补偿来实现；刻意绝不使用 `f64`，因为那会将大多数移动端 GPU 拒之门外。

---

> 💡 **健壮性提示**：如果 GPU 在生成中出错，受影响的那个维度会回退到 CPU 生成，游戏继续运行，失败原因写入 `logs/c2me-wgpu.log`。只有当 GPU 完全无响应（提交后 30 秒仍无结果）时，游戏才会为防止死锁而安全退出，并把详细原因写入同一文件。
>
> ⚠️ **跨平台行为差异**：
> - **桌面端**：如果强制指定了不兼容的后端（如 `WGPU_BACKEND=gl`），wgpu 底层 panic 会被安全捕获，游戏**静默回退到 CPU**，不会闪退。
> - **Android 端**：由于 Android 交叉编译默认 `panic=abort`，类似情况会导致**游戏直接闪退**。但崩溃原因会被写入 `logs/c2me-wgpu.log`，请勿在移动端随意指定 `WGPU_BACKEND`。

## 环境要求

| | |
|---|---|
| Minecraft | 1.21.1 (Fabric) |
| Fabric Loader | >= 0.17.2 |
| **C2ME** | **必需**（本模组为伴生模块，不能独立运行）。支持 `0.4.0-alpha.0.29` 到 `0.4.x`（`<0.5.0`）；不在范围内时 Fabric Loader 会在启动前给出明确的依赖错误，而不是让游戏在生成中途崩溃。 |
| Java | 25+ (本模组基于 Java 25 构建，并使用 Panama FFM API) |
| GPU | 任何支持 wgpu 后端、且单个存储缓冲区上限 ≥ 128 MiB 的 GPU（绝大多数 2020 年后的核显与移动端 GPU 都满足） |

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
> 运行期还会对实际加载的 C2ME 做一次形状检查（`C2meCompat`）：对齐批次依赖的
> 内部类签名不匹配时只跳过该功能并记录 ERROR，原有 GPU 路径继续工作。
> 
> 因此，它依然是一个 **C2ME 伴生模块** —— C2ME 提供并行度和调度器，本模组提供 GPU 算力 —— 名字得以保留。

---

## 安装

1. 为 Minecraft 1.21.1 安装 Fabric Loader。
2. 将 **C2ME** 和 **C2ME-wgpu-rs** 放入 `mods/` 文件夹。
3. 启动游戏。

开箱即用，默认配置无需修改。如果找到了兼容的 GPU **且** C2ME 的调度 API 可用，地形生成会自动卸载至 GPU；否则，模组保持静默，由纯 C2ME 在 CPU 上生成地形，与往常无异。
需要调整时改 `config/c2me-wgpu.json`，见下文。

> 💡 **性能建议：搭配光照优化模组**
> 在极限跑图（如旁观者模式全速飞行）或大规模区块预生成时，原版 Minecraft 的单线程光照引擎极易成为新的瓶颈，导致 CPU 主线程卡顿（表现为区块生成跟不上或掉帧）。强烈建议搭配 **ScalableLux**（或同类光照优化模组），以彻底释放 GPU 地形生成的吞吐量。

---

## 平台支持

| 平台 | 状态 |
|---|---|
| Linux x86_64 | ✅ 已测试 |
| Android aarch64 (自定义 JVM) | ✅ 已测试 (Mali-G610 / 天玑) |
| Windows x86_64 | ✅ 已测试 (AMD, mingw DLL) |
| macOS arm64 (Metal) | ⚠️ 可交叉编译并已提供原生库，尚未在真机验证。仅支持 Apple Silicon，不支持 Intel Mac |
| OpenHarmony aarch64 | ⚠️ 可交叉编译并已提供原生库，尚未在真机验证。原生库未签名，需自行处理签名后才能加载 |

在 Android 上，自定义 JVM **不支持** `-XX:+UseZGC`，模组也不会在那里给出该建议。

---

## 工作原理

1. **序列化** — 原版的 `NoiseRouter`（驱动地形的密度函数树）在 Java 侧被序列化为紧凑的二进制 blob。
2. **交接** — 该 blob 通过 Panama FFM API 传给 Rust `cdylib`。调用是下行 C 函数 + 轮询模型；唯一的向上回调用于把 Rust 日志转发到 Java 的 SLF4J。
3. **编译** — Rust 反序列化该树，运行优化 pass，并生成 WGSL 计算着色器，随后编译为 GPU 管线（并缓存在磁盘上）。首次进入某个维度需要编译时，屏幕顶部会显示进度条与已用时间——位置在加载界面之上，不会让人误以为游戏卡死。
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

- **C2ME 内部 API 变动时对齐批次会降级，而不是崩溃**：模组只使用 C2ME 的公开扩展点，但注册的区块状态仍会读取 `ChunkLoadingContext` 等内部类。启动时会做一次字节码形状检查；不匹配时 `[c2me-wgpu-status]` 记录一条 ERROR，跳过 C2ME 状态机里的对齐批次，GPU 仍通过原有的 `populateNoise` 路径工作（吞吐可能略低，地形正确）。跨 `0.5.0` 的 C2ME 构建由加载器直接拒绝，避免 mixin 目标缺失导致启动失败。
- 在末地极远坐标（约 37 万格）处，使用部分 NaN 回退机制来复现原版的虚空带。
- **模组密度函数**：GPU 序列化器无法原生理解的定制密度节点（例如来自地形模组的节点）会在 CPU 上以相同的角点网格计算，并作为“外部输入”喂给 GPU。树的其他部分依然在 GPU 上运行，因此模组世界生成保持工作 —— 只有不受支持的子树会承担 CPU 开销。
- **移动端 GPU 的缓冲区上限**：模组要求显卡支持至少 128 MiB 的单个存储缓冲区（WebGPU 规范的默认下限）。此前的版本要求 256 MiB，把 Adreno 750（骁龙 8 Gen 3）这类只报 128 MiB 的现代移动 GPU 挡在了门外；0.1.6 起门槛已降到 128 MiB。仍不满足的驱动会在启动时被能力检查拒绝，该维度回退 CPU，聊天栏会给出提示。
- **外部输入过多的地形会回退 CPU**：每个无法原生翻译的模组密度节点都要在 CPU 上按角点网格采样再上传给 GPU。上限按**去重后的节点数**计（同一个节点实例被多处引用只算一个）；超过 64 个时，上传量会逼近显卡的缓冲区上限，且 CPU 采样开销远超 GPU 收益，因此模组直接让该维度走 CPU 生成，并在聊天栏提示。
- **NeoForge + Sinytra Connector**：Connector 会把 mod 密度节点交给本模组当作外部输入处理，所以外置节点数取决于 mod 本身。**0.1.7 起按实例去重后，Tectonic 3.0.28 在 Connector 下只有 2 个槽位（732 次引用），832 高的维度可以正常走 GPU**（实测环境：NeoForge 21.1.265 + Connector 2.0.0-beta.17，2026-11-13）；shader 规模与 Fabric 原生持平，地形正确。Connector 仍非官方支持环境，遇到问题请附日志。注意必须使用 **NeoForge 版**的 Tectonic：把 Fabric 版交给 Connector 桥接仍会在加载 Mixin 时崩溃。
- **Windows 没有磁盘管线缓存**：缓存目录按 XDG 约定推导（`$XDG_CACHE_HOME`，回退 `$HOME/.cache`），这两个变量在 Windows 上通常不存在，因此那边的磁盘缓存实际被禁用——每次启动都要重新编译一次着色器。不影响正确性，只影响启动耗时。
- **Intel Mac 不支持**：原生库只提供 `macos-aarch64`。Intel Mac 会因找不到对应库而回退 CPU 生成。
- **Android 上切换图形后端需要重启游戏**：GL 后端创建的设备在销毁时几乎不归还原生内存（实测从约 4 GB RSS 中只释放 15 MiB，而 Vulkan 设备能释放 645 MiB）。因此在 Android 上热切换后端会让新旧两套资源叠加并耗尽内存，模组改为提示重启、不执行热切换。桌面端没有这个现象，仍可热切换。
- **GL 后端不可用时可能泄漏显存**：wgpu 的 GLES 后端在 EGL 上下文创建失败时直接 panic，其展开路径不会释放已分配的原生资源。模组会捕获这个 panic 并保留原设备，游戏不受影响，但每次失败的尝试都会遗留一部分显存。这是 wgpu 上游的行为，模组层面无法回收。

---

## 配置文件

用户设置在 `config/c2me-wgpu.json`。首次启动时自动生成。

**生效时机**：`debug_*` 开头的选项在下次进入世界时即生效；影响 GPU 设备的选项（`gpu_index`、
`disable_pipeline_cache`、`mapped_at_creation`）在**退出存档再进入**时生效，因为那需要重建
整个 wgpu 实例。两者都不需要重启游戏。**例外是 `chunk_batch_size`**：它在 C2ME 状态
注册时读取一次，必须完全重启游戏。

如果文件写坏了（JSON 语法错误），模组会把它改名为 `c2me-wgpu.json.broken` 保留下来，然后生成一份新的默认配置；超出范围的值会被自动修正并回写。

文件里的 `//` 整行注释会被忽略。模组会自动在 `gpu_index` 上方写入一份可用 GPU 清单：

```jsonc
{
  // Available GPUs (use one as gpu_index; -1 = automatic):
  //   0: AMD Radeon Graphics (RADV RENOIR) [Vulkan]
  "gpu_index": -1,
  "disable_pipeline_cache": false,
  "chunk_batch_size": 1,
  "debug_verbose_logging": false,
  "debug_stage_timing": false,
  "debug_height_probe": false,
  "mapped_at_creation": null,
  "debug_vanilla_ref": false,
  "debug_dump_chunk": "",
  "debug_dump_dir": "",
  "debug_dump_blocks": false,
  "debug_dump_aquifer": false,
  "debug_dump_beard": false,
  "debug_dump_blob": ""
}
```

### 常规设置

| 字段 | 默认值 | 说明 |
|---|---|---|
| `gpu_index` | `-1` | 用哪块 GPU 生成地形。`-1` = 自动（核显优先）。序号对应配置文件注释里列出的清单，也可以用 ModMenu 的设置界面选择。越界或该卡不满足要求时会自动回退到自动选择，并在日志里说明原因。 |
| `backend` | `-1` | 图形后端。`-1` = 自动（Vulkan/DX12/Metal，不含 GL）。可选值与当前平台有关，配置文件里会有注释列出。桌面端切换后**退出存档再进入**即生效；Android 上切换需要**重启游戏**（见「已知限制」）。注意 wgpu 的 GL 后端不提供管线缓存功能，选 GL 时需要同时开启 `disable_pipeline_cache`。 |
| `disable_pipeline_cache` | `false` | 绕过磁盘上的着色器管线缓存。仅在驱动于管线缓存处崩溃时才需要开启，代价是每次启动重新编译着色器。 |
| `chunk_batch_size` | `1` | 对齐批次大小：`1` = 逐区块（默认，行为与旧版完全一致），`2` / `4` / `8` = 把 N×N 个区块收敛成一次调度，提高 GPU 批次聚合度。同种子 r=500 实测：逐区块 18 s、2×2 14 s、**4×4 13 s（最快）**、8×8 15 s（批太大，单批延迟吃掉收益）。**改动需要完全重启游戏**，退出存档再进入不会生效。 |
| `mapped_at_creation` | `null` | 缓冲区的上传方式。`null` = 平台默认（桌面用映射路径，Android 用 write_buffer），`true` / `false` 强制指定。写缓冲区多一次拷贝但更保守，这也是 Android 默认走它的原因；映射路径在实测的 Mali-G610 上同样可用，两条路径都能工作。 |

### 调试（`debug_` 前缀）

以下选项用于排查问题，正常游玩时保持默认即可。

| 字段 | 默认值 | 说明 |
|---|---|---|
| `debug_verbose_logging` | `false` | 启用详细诊断日志：着色器编译耗时、管线缓存命中、适配器 workaround，以及定期的每秒区块数（`CPS=`）统计（同时打印到聊天栏）。 |
| `debug_stage_timing` | `false` | 将分阶段计时写入日志：GPU 侧（encode / submit / readback / decode）和 CPU 侧（方块写回区块）。 |
| `debug_height_probe` | `false` | 统计原版 `ChunkGenerator.getHeight` 的调用次数、耗时、去重列数、调用线程，以及调用时 GPU 是否仍有批次在途。用来评估「表面高度缓存」值不值得做（见 `DEVELOPMENT.md` §14）。类加载时读一次，改值需重启。 |

### 离线诊断

用于和原版输出做离线对比，正常游玩时全部保持默认，不会写任何文件。

| 字段 | 默认值 | 说明 |
|---|---|---|
| `debug_vanilla_ref` | `false` | 完全跳过 GPU，让原版生成，再 dump 方块用于对比。 |
| `debug_dump_chunk` | `""` | 目标区块：`"x,z"` 或 `"x1,z1,x2,z2"`。留空则其余 `debug_dump_*` 全部不生效。 |
| `debug_dump_dir` | `""` | dump 文件写入目录。留空 = 平台临时目录。 |
| `debug_dump_blocks` | `false` | dump 目标区块的方块状态。 |
| `debug_dump_aquifer` | `false` | dump 含水层 cell 表。 |
| `debug_dump_beard` | `false` | dump Beardifier 结构与逐方块参考值。 |
| `debug_dump_blob` | `""` | 将序列化的 NoiseRouter blob 写到该路径（`src/test/resources/real_router.bin` 即由此生成）。 |

## 设置界面（ModMenu）

装了 [ModMenu](https://modrinth.com/mod/modmenu) 的话，模组列表里本模组会有一个**设置**按钮，
可以直接改 `gpu_index`、管线缓存、缓冲上传方式以及两个调试开关，不需要手改 JSON。

- ModMenu 是**软依赖**：没装就只按配置文件来，功能完全不受影响。
- 界面完全由 `Config` 字段上的注解生成，不存在"界面和配置文件不同步"的问题。
- 修改后点「完成」即写回文件。`debug_*` 立刻生效，设备相关选项在下一次进入世界时生效。

## 环境变量

本模组自己的所有开关都在配置文件里，不读取任何 `C2ME_*` 环境变量。本节只涉及
**wgpu 库自身**读取的标准变量。

### GPU 选择（标准 wgpu 变量）

模组遵循 wgpu 自身的环境变量，因此它们的行为与任何其他 wgpu 应用程序一致：

| 变量 | 值 | 效果 |
|---|---|---|
| `WGPU_BACKEND` | `vulkan`, `dx12`, `metal`, `gl`, `webgpu` (逗号分隔) | 限制尝试的图形后端。在某个后端损坏时很有用。注意：`gl` 后端需要 OpenGL 4.3+ 才能支持通用计算 (Compute Shader)。 |
| `WGPU_ADAPTER_NAME` | 子串 (不区分大小写) | 按名称强制指定 GPU，例如 `WGPU_ADAPTER_NAME=radeon`。`gpu_index` 是首选方式；此变量留给按名字选择的场景，且只会匹配能通过能力检查的卡。没有匹配项时会记录警告并回退到默认选择。 |

如果不使用这些变量，模组会使用 Vulkan / Metal / DX12（取决于操作系统提供的）。选卡是**能力优先**：每张候选卡都要先通过能力检查（能否跑我们的计算着色器），只有通过的才进入候选池；随后按**集成显卡 → 独立显卡 → 虚拟显卡**的顺序挑选，让独立显卡尽量空闲以用于渲染。软件光栅化器（如 llvmpipe）在枚举阶段就被排除，不会占用序号。

### 原生库与日志

**原生库解压位置排查**：
默认情况下，模组会将 jar 包内的 `.so` 或 `.dll` 解压到游戏根目录下的 `c2me-wgpu-natives/` 文件夹中。如果该目录无法写入或加载（如遇到 `noexec` 权限限制），模组会自动回退解压到操作系统的临时目录（如 Linux 的 `/tmp`）。
如果你需要排查底层库加载失败的问题，可以在 JVM 启动参数中添加 `-Dc2me.wgpu.preserveNative=true`，模组将把原生库直接解压到游戏当前的工作目录中。

> **注意：** 标记为“仅开发用”的配置项以及离线诊断开关严格用于故障排查，正常游玩时无需设置。

> **Rust 侧日志文件：** 原生库将其诊断信息写入 `<gameDir>/logs/c2me-wgpu.log`（路径由 Java 在启动时传入）。这独立于 Minecraft 自身的 `logs/latest.log`，后者仅捕获 Java 日志。

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
- **JDK 25**（Java 工具链）和 **JDK 21**（Gradle JVM）——这两个是硬性要求，缺失会直接失败；
- **Rust 1.85+**（crate 使用 edition 2024）；
- 宿主平台的原生 C 工具链（Linux 上 gcc/clang，Windows 上 MSVC 或 mingw）。

以下平台**按需构建，工具链缺失时自动跳过并打一条日志**，不会导致构建失败：

| 平台 | 触发条件 | 关闭开关 |
|---|---|---|
| Windows | `rustup` 里装有 `x86_64-pc-windows-gnu` 或 `*-msvc` target | — |
| Android | `ANDROID_NDK_HOME` 已设置，且装有 `cargo-ndk` | `-PskipAndroid` |
| OpenHarmony | `OHOS_SDK_NATIVE` 已设置，且装有 `cargo-ohos` | `-PskipOhos` |
| macOS | `OSXCROSS_ROOT` 指向有效的 osxcross（含 `*-apple-darwin*-clang` 与 `MacOSX*.sdk`） | `-PskipMacOS` |

因此一份克隆在只有 JDK + Rust 的机器上也能构建成功，产出的 jar 只含宿主平台的 native。想打包全平台需要自行准备上表中的工具链。

`-PskipRust` 可以完全跳过原生库编译（例如没有 Rust 工具链的 CI）。

---

## 开发说明

本模组是在**大量 AI 辅助**下开发的 —— 代码、文档和测试大部分是在维护者审查下由 AI 生成的。请对此保持警惕：在依赖它之前，请务必在您自己的硬件上验证行为。欢迎提交 Bug 报告和 Pull Request。

### 运行测试的注意事项

**动地形相关代码后必须额外跑对照测试。** `real_router_vs_vanilla` 是唯一验证 GPU 输出与真实 Minecraft 一致的测试，因耗时约 2 秒已默认忽略。改动 emitter、aquifer、beard、矿脉或反序列化后请手动执行：

```
cd rust && cargo test --lib -- --ignored real_router_vs_vanilla
```

`cargo test` 在集成显卡上可能偶发 GPU 测试失败（约每 5~8 次一次）。原因是测试为每个用例单独创建 executor（各约 600 MB GPU buffer），并行执行时资源叠加；这是测试架构问题，与游戏运行路径无关——游戏内只有一个 executor，且按 `pipeline_depth` 有界并发。若复现，请附完整 `cargo test` 输出以便定位。

---

## 来源说明

本项目是一个独立的、净室（clean-room）重新实现。它是针对 Minecraft 的**运行时行为**（导出的噪声数据和采样的参考值）编写的，而不是针对任何第三方模组的源代码。

一次代码来源审查将本模组的含水层和 Beardifier 实现与 [C2ME](https://modrinth.com/mod/c2me-fabric) 的 OpenCL 加速器进行了比较：没有共享标识符，没有复制的函数，且架构不同（C2ME 逐函数镜像 Minecraft；本模组使用预计算表 + GPU 扫描设计）。在两者看起来相似的地方（例如 `fastInverseSqrt`, `getStructureWeight`），共同的来源是 Minecraft 自己的原版代码，这两个项目都是独立复现的。

范围说明：审查是在特性/架构层面进行的，而不是逐字符差异对比。Minecraft 本身不被重新分发；仅提供原创代码。参见 `LICENSE`。

---

## 致谢

特别感谢 **C2ME 作者团队**（[RelativityMC](https://github.com/RelativityMC) 组织及所有贡献者）：

- C2ME 本体提供的多线程区块生成和调度器（`SchedulingManager`），是本模组能够满载运行、将 GPU 算力发挥到极致的绝对基石。
- 官方 C2ME OpenCL 加速模块（`c2me-opts-accel-opencl`）在架构设计、批处理策略和驱动兼容性处理上，为 `c2me-wgpu-rs` 的早期骨架搭建提供了极大的启发（详见 `DEVELOPMENT.md` 中的相关分析与借鉴）。

---

## 许可证

MIT — 参见 [LICENSE](LICENSE)。
