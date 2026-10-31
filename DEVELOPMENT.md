# C2ME WGPU 加速模块 — 开发文档

> 最后更新：2026-10-24
> 状态：**0.1.5 发布版**。完整实现 Pass E、GPU 矿脉、100% 对齐原版的 AquiferSampler 逻辑，并重构了 FFM 日志架构与底层防死锁机制。实机验证兼容 Epic Terrain 与 Tectonic。
> 用户侧已落地：GPU 选择（能力优先，越界自动回退）、切存档热重载配置（冷/热两级，零锁）、
> 聊天栏状态提示、由注解自动生成的 ModMenu 设置界面。
> 移动端实测：Chunky A/B 测试在 Mali-G610 平板上，端到端总耗时从 3m12s 缩短至 2m19s（+61%）；纯地形生成实时吞吐从 23.52 CPS 提升至 30.65 CPS（+30.3%）。
> 服务端实测 (Headless，无渲染干扰)：Chunky r=500 跑纯原版地形，纯 CPU 耗时 27s (平均 121.1 cps)，GPU 加速耗时 21s (平均 143.6 cps，峰值 215.7 cps)。端到端提速 22%，峰值提速 44.7%。注：纯 CPU 模式下主线程排队（Amdahl 定律）会锁死并发上限，CPU 占用率通常只能达到 70-80%。
> 桌面端到端反超纯 CPU（Chunky r=500，RADV RENOIR：**18 s vs 23 s**，流水线化后本机实测）。
> 更早一次桌面实测为 15 s，机器不同，不作为当前基准。GPU 直接输出 per-block 方块状态，CPU 只写 palette。
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
| **端到端反超纯 CPU** | ✅ 18 s vs 23 s (RADV RENOIR，见 §23) |
| **移动端端到端提速** | ✅ 3m12s→2m19s；CPS 23.52→30.65 (+30.3%) |
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
14. **pipeline 缓存键必须跟着代码走**：磁盘缓存键 = AST 哈希 + pass。历史上这里靠手写 `LAYOUT_VERSION` 兜底，**两次都忘了 bump**，且忘记是静默的（旧 pipeline 被复用 → 地形悄悄错）。现已改为把 **WGSL 源码 + `bind_group.rs`** 一起哈希进指纹：改任何影响 shader 或 layout 的代码，key 自动变。新增 codegen 时无需再记得手动改版本号，但**别把动态内容（如时间戳）混进哈希**，那会让缓存永远失效。
15. **shader 编译成本超线性**：拆小是唯一可靠解法（经验安全区 ~2100 行，Mali 单 shader 过大会设备丢失）。
    **2100 行只是经验值，不是保证**：实际复杂度更关键。实测**天玑 8100 的
    Mali 驱动是问题最多的设备**，会对规模/复杂度超限的 shader 直接拒绝编译；
    而**麒麟 980（Mali-G76）在拆分 pass 之前的单体大 shader 上仍能正常工作**，
    说明不同 Mali 世代/驱动版本的容忍度差异极大，不能拿一个行数一刀切。
    行数只能当粗略预警，真实上限只能靠实测。好消息是 lesson #36 的 error
    scope 让这类"驱动拒绝编译"变成可捕获的 `Err` → 该维度回退 CPU，而不是
    死锁或设备丢失。
16. **惰性缓存可预计算**：MC 的惰性表（aquifer waterLevels）在 GPU 上必须整表预计算（Java warmup ~0ms）。
17. **内存泄漏**：共享 map 要 remove-on-consume，不能只 insert。
18. **诊断代码放对位置**：别放在会被短路的分支后（否则是死代码）。

**方法论（最重要）**
19. **端到端指标才是判据**：Chunky 总时间，不看内部 batch 吞吐（§19 教训）。
    **"测试全绿" ≠ "实机可用"**：单元/离线测试只覆盖它断言的那条链路。本次
    "全虚空"事故中 89 个 Rust 测试 + `real_router_vs_vanilla` 全部通过，但游戏
    里一个方块都没有——因为没有任何测试覆盖"GPU 算完 → Java 写回 chunk"这一
    段。凡是改动跨语言边界（Rust 产出 → Java 消费）或涉及驱动行为（shader
    编译上限、后端能力），**必须实机跑一遍再下结论**。旧经验（如"2100 行
    安全"）也只能当假设，芯片/驱动世代不同结论会翻转。
20. **"CPU 解释器一致" ≠ "对 vanilla 正确"**：测试要对照 MC 原版 dump。
21. **先数消费者，再决定算多少**：16 通道里只有 2 个被读，别默认全算。
22. **假通过的测试比没测试更危险**：`eprintln` 而非 `assert` 的判据会沉默。
23. **热路径别用 CAS 循环**：计数直接用 `incrementAndGet`，不要自己写 compare-and-swap 重试。
24. **跨语言日志需防截断**：Rust 的 `panic` 若触发 `abort`，JVM 会被瞬间带走。务必在 FFI 边界拦截，并用 `sync_all()` 强刷黑匣子文件。
25. **死锁下 `System.exit` 无效**：当主线程死锁时，`System.exit` 的 Shutdown Hook 会被卡死。必须调用 `LogManager.shutdown()` 刷盘后，使用 `Runtime.getRuntime().halt(1)` 强杀进程。
26. **FFM 导出的符号必须带 `#[unsafe(no_mangle)]`**：即便是 `extern "C"` 函数，如果漏掉了这个宏，Rust 编译器依然会改写符号名，导致 Java 端调用 `lookup.find()` 时抛出 `NoSuchElementException`。清理代码时绝不能误删。

27. **能力优先，别信 adapter 的自述**：llvmpipe 会把自己标成普通 GPU，靠 `DeviceType`
    过滤挡不住它，结果是一路通过选卡、直到 `check_required` 才失败，并让整个会话掉到
    CPU。正解是每个候选都跑一遍能力检查，只有真正能用的才进选择池；类型只用于排序。
    另外，显式指定的 `gpu_index` 也必须过检查——坏配置应当回退，不该让 instance 创建失败。

28. **低频偶发失败先抓证据，别倒推**：`world::tests` 里 9 个 GPU 测试各自 `ExecutorThread::spawn` 一套约 600 MB buffer，16 线程并行时资源叠加，集显上约 1/5~1/8 概率偶发失败，且无固定 panic 原文。曾误判为显存 OOM——实际 `vulkaninfo` 显示 device-local budget 4.86 GiB、`dmesg` 无 amdgpu OOM 记录，10 次连跑不复现。生产路径 `get_or_create_executor()` 是单个 executor（现已从 `OnceLock` 改为可重置的 `RwLock<Option<Arc<..>>>`，以便换卡时重建；无 Mutex 争用，也早已不是串行），**游戏运行不受影响**。修复方向（暂缓）：测试改用小 `ExecutorConfig`（batch 实际只有 1~3 个 chunk，却按 64 预分配），或加全局 GPU 锁。复现时务必保留完整 stderr + `free -h` + `mem_info_gtt_used` + `dmesg | tail`。

29. **动地形正确性必须跑 `real_router_vs_vanilla`**：它是唯一对照真实 Minecraft 输出（`src/test/resources/real_router.ref.txt`）的测试，因耗时约 2s 已默认 `#[ignore]`。改动 emitter / aquifer / beard / 矿脉 / 反序列化，或任何会改变 WGSL 数值输出的代码后，必须手动跑 `cd rust && cargo test --lib -- --ignored real_router_vs_vanilla`。其余 84 个测试全绿**不能**替代这一步——它们只验证"CPU 参考实现与 GPU 一致"，不验证"GPU 与 vanilla 一致"（见教训 #20）。
30. **重建实例时新旧两份会同时存在**：`rebuild` 先建新的、成功才替换，
    这在桌面端换来"失败可回退"的好处，但在内存紧张的 Android 上两份 GPU 资源
    叠加会顶到 lmkd。实测 12 GB 平板切 GL 后端时系统级挂死，而内存充裕时
    同一操作只是回退。现在按 `MemAvailable` 分流：低于 2 GiB 先释放旧实例再建
    （失败则用默认配置兜底重试），充裕时保留旧实例。Windows 用
    `GlobalMemoryStatusEx`，Linux/Android 读 `/proc/meminfo`，macOS 暂无廉价
    信号、按"紧张"处理。

31. **GL 后端与 `PIPELINE_CACHE` 互斥**：wgpu 的 GLES 后端不提供
    `Features::PIPELINE_CACHE`，而 `check_required` 默认要求它。所以 Android
    上选 GL 必须同时开 `disable_pipeline_cache`，否则会在能力检查处被拒。
    曾把"被拒"误读成"GL 不可用"，实际上打开该开关后 GL 在 Mali-G610 上
    正常出块（CPS 22–29）。能力检查失败时先确认是哪个 feature 缺失，再判断
    后端是否真的不可用。

32. **"释放旧实例"要清三层，少一层就泄漏**：重建 device 时旧资源被三个
    地方持有，逐一验证过才收敛：(a) `ExecutorThread` 持有 `Arc<Device>`
    克隆——原实现用 `Arc::get_mut` 才 shutdown，有在飞批次时返回 `None`
    直接跳过，worker 继续跑；改为 `shutdown(&self)`（`JoinHandle` 包
    `Mutex`）后无条件生效。(b) `pipeline_cache::IN_MEMORY_CACHE` 是进程级
    静态表，存着 `wgpu::ComputePipeline`，每个都引用它创建时的 device；
    该缓存此前从不清理，所以旧 device 的引用计数永远不归零。注意它的键是
    `(fingerprint, pass)`，不含 device 标识，不清理还会跨 device 误用。
    (c) `WgpuInstance` 自身。加了 `[mem]` 诊断日志后才看清：(a)(b) 修好后
    Vulkan 设备 drop 能释放 645 MiB。

33. **drop 不等于归还**：GL 后端创建的设备在 Android 上 drop 时只释放
    15 MiB（同一次测量中 Vulkan 是 645 MiB）。`dropping WgpuInstance`
    正常打印，引用也确实清零，但 EGL context、glow 资源等没有回到系统。
    因此"热切换后端"在 Android 上不可行：新旧两套叠加必然耗尽 12 GB 内存，
    表现为整个平板卡死。现在 Android 上切换后端只提示重启，桌面端照常热切。
    判断依据不能只看"对象是否 drop"，要看进程 RSS。

34. **枚举 adapter 会创建真实上下文，不是只读操作**：`enumerate_adapters`
    在 GLES 后端要 `eglMakeCurrent` 并加载 GL 函数指针。GUI 打开时按配置
    backend 枚举一次，就等于在活设备旁边再造一个实例——Android 上直接冻住
    渲染器（画面静止但操作有响应）。修法：设备创建时把 adapter 列表序列化
    存在 `WgpuInstance` 里，GUI 只读这份缓存，永不触发枚举。

35. **Android 上 `/proc/meminfo` 对应用不可读**：`adb shell` 能读，但应用
    进程内被 SELinux 挡住，`MemAvailable` 恒为 `?`。于是内存感知重建在
    Android 上永远走"先删旧实例"的安全分支——方向正确，但失去了"内存充裕
    时保留回退能力"的一半逻辑。桌面端能读到真实值，两个分支都验证过。

36. **wgpu 的编译/校验失败不走返回值，必须用 error scope 显式捕获**：
    `create_shader_module` 和 `create_compute_pipeline` 遇到 validation error
    都不返回 `Err`，只把错误丢给 `on_uncaptured_error` 打日志，然后照样把
    一个无效对象交给调用方。后果是静默的：invalid pipeline 被提交，输出
    buffer 保持未初始化显存，Java 读到负的 palette 索引 → AIOOBE →
    CompletableFuture 永不完成 → 整个游戏死锁。正解是在调用前后夹
    `device.push_error_scope(ErrorFilter::Validation)` + `pollster::block_on(scope.pop())`，
    有错就提前 return Err；这个 Err 会一路传到维度初始化的 catch，让该维度
    整体回退 CPU，而不是挂死。注意 pipeline cache 只有在 pipeline **确认被
    接受**之后才能落盘，否则会把导致失败的数据持久化。

37. **`ExternalInput` 是角点量，必须像 `InterpolatedNoiseSampler` 一样被 split 提取**：
    Java 侧在 `gx*gy*gz` 角点网格上预先求值 external 子树，WGSL 按 cell 索引
    读回，语义与插值噪声一致。若留在 outer（per-block）树里，BlockOut shard
    会引用它没有绑定的 `external_inputs`，shader 校验失败——而这正是上一条
    的静默路径。该 bug 只在含 modded 密度节点的世界（如暮色森林）暴露。
    **注意 barrierNoise（ch0）也要走 split**：它同样被注入 BlockOut，漏掉它
    会让 Tectonic 这类把 modded 节点塞进 barrier 的世界直接编译失败。

38. **症状变了先看 `git status` / `git diff`，别只信交接文档**：本次排查中
    "死锁"被修好后变成"不卡死但全虚空"，一度被误判为 Rust 侧 pipeline /
    cache / external 索引问题，绕了很久。真因是同一工作区里混着一处
    未提交的误删：`writeBlocksFromGpu` 的写方块三层循环体被清空，GPU 算出
    的方块全被丢弃，chunk 保持全空气。教训：**改了 A 之后症状发生变化，
    第一件事是核对工作区实际改动，而不是顺着文档假设往下推**。

39. **Sinytra Connector 会让世界生成模组的节点类型识别失效**：在 NeoForge
    + Connector 下，Fabric 的 Tectonic 会把大量原版节点包装成其内部的
    `ConfigNoise` 类，导致 `NoiseRouterBlobifier` 的 `instanceof` 全部落空，
    数百个节点被当作 external 外置到 CPU。这不仅让上传 buffer 超过 128 MB
    的 `max_storage_buffer_binding_size` 限制，还会让 CPU 串行采样把 CPS
    拖到个位数。已在维度初始化处加 64 个 external 节点的硬上限，超限直接
    回退 CPU。这是 Connector 的兼容性限制（非原生环境），不做逆向适配。

40. **CPS 随种子（地形复杂度）剧烈波动**：旧的 Chunky 端到端数据（18s vs
    23s、CPS 188）只在特定种子下成立。不同种子的噪声 octave 数量、spline
    分支命中率、aquifer 命中率差异巨大，GPU 工作量完全不同。跨版本/跨环境
    对比必须用同一种子、同视距，否则结论不可靠。视距也是强变量，A/B 测试
    要一起控制。




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
  - [x] **GPU 选择**：
    - Rust 提供 `c2me_wgpu_list_gpus`，用临时 instance 枚举 adapter，返回
      `[{index,name,type,backend,selected?}]` JSON；`DeviceType::Cpu`（llvmpipe 等
      软件光栅化器）在 `enumerate` 阶段就被过滤，不占索引。
    - Java 首次启动时把列表以 `//` 注释形式写进 JSON 的 `gpu_index` 上方，不装
      ModMenu 也能看到序号与卡名。读回时 `stripComments` 只删整行注释，字符串里的
      `//` 原样保留。
    - `create_instance` / `rebuild_instance` 接收 `gpu_index`。选卡是**能力优先**
      而非名字/类型优先：每个候选都过一遍 `features::check_required`，只有真正
      能跑 shader 的才参与选择。越界或不达标都会打日志并回退到自动顺序，坏配置
      不会让整个会话掉到 CPU。
  - [x] **切换存档时热重载配置**：
    - 配置分三类：A 类影响 instance（`gpu_index`、`backend`、
      `disable_pipeline_cache`）、B 类影响 world 创建（`mapped_at_creation`）、
      C 类只影响行为（`debug_*`）。
    - 实现上没有引入任何锁：`debug` / `profile` 是两条 `AtomicBool`（热路径零开销）；
      A/B 两类在重建 instance 时作为 `ColdConfig` 一并替换，因此不存在"全局可变
      配置"这一概念。
    - `WgpuInstance` 从 `OnceLock` 单例改为 `RwLock<Option<Arc<..>>>`，新增
      `rebuild` / `shutdown`；executor 同样可重置。
    - 关键顺序由 `WgpuDimensionRegistry.onSaveSwitch()` 统一执行：
      `destroyAll()`（shutdown 全部 coordinator → world_destroy → 清表）→
      `rebuildFromConfig()`。冷配置没变时只刷新热标志，零成本。
    - 触发点是 `MinecraftClient.joinWorld` 的 HEAD，因此"游戏运行时改配置、
      进存档"也能生效，不必退回标题界面。
    - 变化检测的基准是**磁盘文件**对 `appliedCold`（实际建成的设备用的配置），
      不涉及内存里的 `Config` 单例。GUI 直接改那个单例，拿它当基准会永远
      得出"没变化"。
    - 重建失败时把 `backend` 重置为 `-1`（自动）再回写文件。不复制当前生效
      的后端：那会把会话钉死在某一个具体后端上，而某些机器上它并非最优
      （例如只有 DX12 没有 Vulkan 的 HD 4000）。
  - [x] **状态展示**：
    - 聊天栏：`GPU 加速已启用：<卡名>` / 回退原因，走 `Text.translatable`。
    - 卡名通过 `WgpuStatus` 解析 `list_gpus` 结果里 `selected:true` 的那一项，
      实例重建时失效重取。
    - F3 行**已移除**：聊天栏已经覆盖同样信息，CPS 需要时开
      `debug_verbose_logging` 看日志或聊天栏即可。
  - [x] **ModMenu GUI（软依赖）**：
    - 仅 `modCompileOnly`，`fabric.mod.json` 用独立的 `modmenu` entrypoint +
      `suggests`，没装 ModMenu 时完全不影响。
    - 不引入第三方 GUI 库，全部用原生 widget。
    - **界面由注解反射生成**：`Config` 字段上标 `@ConfigOption(label, kind, order)`，
      屏幕遍历这些字段建控件。加一个设置 = 加一个带注解的字段，GUI 不会与文件
      脱节。`Kind` 支持 `BOOL` / `TRI_STATE`（如 `mapped_at_creation` 的三态）/
      `GPU_INDEX`（自动 + 每张卡的列表）。

- [x] **健壮性**：
  - 增加更多边缘平台（如未测试的 macOS Metal）的预防性回退逻辑。
  - 统一处理未捕获的 GPU DeviceLost，确保游戏不会因为驱动崩溃而卡死。
- [x] **文档梳理**：
  - 整理目前的 Debug 环境变量说明，把不再使用的变量（如 `MIN_STAGE`）彻底从文档中删除。
  - 补充了 FFM Upcall 日志回传机制与防死锁机制的说明。

### 环境变量速查

本模组自己的开关已全部迁移到 `config/c2me-wgpu.json`（见 §12），不再读取任何
`C2ME_*` 环境变量。仅剩 wgpu 库自身的标准变量：

| 变量 | 值 | 效果 |
|---|---|---|
| `WGPU_BACKEND` | 后端名，逗号分隔 | 限制尝试的图形后端 |
| `WGPU_ADAPTER_NAME` | 子串 | 按名称选 GPU。`gpu_index` 现在是首选方式；此变量保留给按名字而非序号选择的场景，且只会匹配能通过能力检查的 adapter |

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
- Java 通过 **Panama FFM** 调用 Rust cdylib，下行 C 函数 + 轮询模型。唯一的
  上行回调用于把 Rust 日志转发到 Java 的 SLF4J（`c2me_wgpu_set_log_callback`）。

---

## §2 工程结构

> **2026-10：C2ME 依赖关系**（§10）。构建期通过 Stub 提供接口签名，运行期强依赖调度器
> （提供 GPU 批处理赖以吞吐的多线程并行度）。

### 2.1 构建
曾用 Gradle `includeBuild` 引入 C2ME 复合构建，**现已移除**。改为在 `src/stub`
提供 `SchedulingManager` / `IVanillaChunkManager` 的接口签名（`compileOnly`，不打进
jar），Java 代码直接引用这些类。详见 §10。

### 2.2 版本对齐

| 项 | 值 |
|---|---|
| Gradle wrapper | 8.14.3 |
| Minecraft | 1.21.1 / Yarn 1.21.1+build.1 |
| Fabric Loader | 0.17.2 / Loom 1.13-SNAPSHOT |
| Java | toolchain 25（运行要求 ≥25）/ Gradle JVM 21 |

### 2.2b 交叉编译 native

`build.gradle` 会按环境自动尝试各平台，缺工具链则跳过并打日志：

| 平台 | 触发条件 | 关闭开关 |
|---|---|---|
| 宿主平台 | 始终（`cargo build --release`） | `-PskipRust` |
| Windows | `rustup` 里装了 `*-pc-windows-*` target | 无（自动检测） |
| Android | `ANDROID_NDK_HOME` 已设置 + `cargo-ndk` | `-PskipAndroid` |
| OpenHarmony | `cargo-ohos` 可用 | `-PskipOhos` |
| macOS | `OSXCROSS_ROOT` 指向有效 osxcross，且其 `bin/` 有 `*-apple-darwin*-clang`、`SDK/` 有 `MacOSX*.sdk` | `-PskipMacOS` |

**macOS 交叉编译**：链接 Metal / Foundation 等系统框架必须有 macOS SDK，所以用
[osxcross](https://github.com/tpoechtrager/osxcross)。构建步骤：

1. 拿 SDK 放进 `osxcross/tarballs/`（文件名须是 `MacOSX*.sdk.tar.xz`）
2. `BUILD_FLAVOR=llvm UNATTENDED=1 ENABLE_ARCHS="arm64" ./build.sh`
   —— 选 `llvm` flavor 是为了绕开 Arch 上没有的 `libtapi`
3. `export OSXCROSS_ROOT=$HOME/osxcross/target`

产物落在 `natives/macos-aarch64/libc2me_wgpu.dylib`。**只支持 Apple Silicon**：
默认 target 是 `aarch64-apple-darwin`，Intel Mac 会找不到库而回退 CPU。

注意 Gradle 的 `env` 是 `HashMap<String, String>`，拼接字符串必须显式 `.toString()`，
否则 Groovy 的 `GString` 会抛 `ClassCastException`。

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
[HEADER 8B]    magic "WG00" + version u16 + chunk_count u16
[offsets 128B] 16 个通道的 payload 偏移 (u64 LE)；未回读的通道为 0
[payload]      仅被回读的通道 × cells × f32
[CRC32 4B]

回读哪些通道由 `WorldContext::readback_channels()` 从 shard 布局推出，实际是
finalDensity(ch11) 与 vein 三通道(ch12–14)。16 通道只是偏移表的上限宽度。

per-block 方块状态走独立的 `block_buf`（u32/block），不经过这个 blob。
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
6. 矿脉（`OreVeinSampler`，连片大矿脉）已在 GPU 上实现：`block_out.wgsl` 里的
   `vein_state()` 复刻了 MC 的采样器，RNG 种子由 `c2me_wgpu_set_ore_seed` 传入。
   普通矿石（煤/铁/金等）仍由 biome feature 阶段放置，不经过这条路径。

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
- **端到端反超纯 CPU，功能完整，地形正确**（当前基准见 §23：18 s vs 23 s）。
- 遗留 0.03% = 过渡带 `calculateDensity`（barrier 噪声）分支未实现。
- 验证链全离线：CPU-f32 / blockPositions / WGSL / e2e / beard，5 个测试。
- shard B 瘦身（2026-10-17）：`spline_seg` + `octave_stack` 两个 WGSL helper 替换
  高频内联展开，3084→1771 行（重回 Mali ~2100 安全区），bit-identical。
- Spline n==1 修复（2026-10-17）：单点 spline 漏了 `outside()` 外推 → 嵌套 values 全错
  → depth 偏差 ~0.08 → 海底 y≈43–55 误判 stone（chunk 50,50 diff 8.88%）。改
  `spline.rs`/`world.rs` 两处 n==1 分支 + bump `LAYOUT_VERSION` 4→5。验证：逐通道
  bad=0/1225、逐格 diff 归零（回退复现 +0.08~0.098）、GPU rms 2.5e-5/5.7e-5、84 测试
  全过、实机 Tectonic 与 Epic 滴水石锥均消失；怪坑（§14 非线性外层）仍独立存在。

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

### 关键文件

- `rust/src/codegen/wgsl_gen.rs`：WGSL 生成 + shard 规划 + BlockOut/beard/aquifer WGSL。
- `rust/src/scheduler/executor.rs`：batch 执行、bind group、buffer 管理。
- `rust/src/world/world.rs`：WorldContext、pipeline 编译、测试。
- `rust/src/world/{aquifer,beard}.rs`：aquifer 表解析 + Beardifier 复现。
- `rust/src/c_api.rs`：C API（含 `set_aquifer_table` / `set_beard_table`）。
- `src/main/java/.../mixin/MixinChunkGenerator.java`：写回 + stage + dump。
- `src/main/java/.../WgpuThinBinding.java`：FFM 绑定。

---

