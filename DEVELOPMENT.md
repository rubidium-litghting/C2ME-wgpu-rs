# C2ME WGPU 加速模块 — 开发文档

> 最后更新：2026-11-14
> 状态：**0.1.5、0.1.6 已发布；0.1.7 已通过 Fabric 与 NeoForge（+Connector，含 Tectonic）实机验证，tag `v0.1.7` 已推送（Release 页面由维护者发布）**。
>
> **已发布 / 已打 tag**
> - **0.1.7**：对齐批次（同种子 4×4 13 s）、shader 编译覆盖层、external 按实例
>   去重、NeoForge + Connector 下的 Tectonic 修正为可走 GPU、批次解码失败的
>   资源泄漏修复。tag `v0.1.7` 已推送；Release 页面由维护者另行发布。
> - **0.1.6**：修掉 modded 世界（暮色森林 / Tectonic）的死锁与全虚空问题，并把
>   shader/pipeline 编译失败从"静默死锁"改为"该维度回退 CPU + 玩家可读提示"。
> - **0.1.5**：完整实现 Pass E、GPU 矿脉、100% 对齐原版的 AquiferSampler 逻辑，
>   并重构 FFM 日志架构与底层防死锁机制；实机验证兼容 Epic Terrain 与 Tectonic
>   （Fabric 原生环境）。用户侧已落地：GPU 选择（能力优先，越界自动回退）、
>   切存档热重载配置（冷/热两级，零锁）、聊天栏状态提示、由注解自动生成的
>   ModMenu 设置界面。
>
> **0.1.7**
> - `alpha.1`：`NewStatusHook` / `NewChunkStatus` 阶段 1 透明委托，r=500 与基线
>   持平（§13）。
> - `alpha.2`：`chunk_batch_size` 对齐批次（2×2 / 4×4 / 8×8）已实现并实机验证
>   （同种子 r=500：14 / 13 / 15 s，4×4 最优、CPS 298，地形正常、无死锁）；
>   默认仍为 1，不改变已验证行为（§13.7）。
> - `alpha.3`：表面高度查询打点 `debug_height_probe`（默认关闭）。表面高度估算
>   据此结案（负结果）：同种子 r=500 仅 696 次 `getHeight`、累计 1.58 s
>   （≈12% 墙钟，约一半与在途 GPU 批次重叠），重复率 6%；纯 CPU 缓存省不到 1%，
>   GPU 预计算要多跑一遍噪声（§14）。探针保留，可随时复测。
> - `alpha.4`：ModMenu 直接提供 `chunk_batch_size` 循环控件（1 / 2 / 4 / 8），
>   不再需要手改 JSON；界面单独提示"需完全重启"。
> - `alpha.5`：C2ME 运行期形状检查 `C2meCompat` + `fabric.mod.json` 上界
>   `<0.5.0`。C2ME 内部 API 变动时跳过对齐状态并保留 `populateNoise` GPU 路径，
>   而不是在生成中抛 `NoSuchMethodError`（§13.6）。
> - `alpha.6`：shader 编译进度覆盖层。首次进入维度时在屏幕顶部显示
>   `正在编译 GPU 着色器… n/N（秒）` 与进度条，注入在 `GameRenderer.render`
>   唯一那处 `DrawContext.draw()` 之前（世界 + HUD + 当前 Screen 之上），因此
>   加载地形界面也看得到。数据走 Rust 侧拉取式快照，见教训 #52。**已实机验证**
>   （清空 `~/.cache/c2me/wgpu_pipelines` 后冷编译，覆盖层在加载地形界面上可见、
>   逐 shard 前进、结束即消失）。
> - `alpha.7`：external payload 改为**随批次携带**。`c2me_wgpu_request_chunks`
>   新增 `(external_count, external_ptr, external_len)`，Rust 把它拷进
>   `Batch.externals`，executor 只读这一份；进程级的 `world.external_data` /
>   `c2me_wgpu_set_external_inputs` 已删除。因此有 CPU 外置节点的维度
>   （暮色森林等）也能走流水线，`WgpuBatchCoordinator` 的串行分支整段移除。
>   payload 长度与 positions 不匹配时直接报错，而不是静默错位地形（教训 #54）。
>   Rust 侧 e2e 测试 `external_payload_travels_with_batch` 覆盖。
> - `alpha.8`：external 节点按**实例身份去重**。密度树是 DAG，序列化器不做
>   memo，同一个子树被引用多少次就写多少个槽——Tectonic 3.0.28 只定义了 7 个
>   `tectonic:config_noise`，却被引用 732 次，于是撞上 64 节点上限、
>   整维度回退 CPU（`too many external density subtrees (732 > 64)`）。
>   现在按实例共享槽位（`sample` 纯函数，同一实例同一角点值必然相同），
>   日志同时打「槽位数 + 引用数」。**已实机确认**：Fabric 与 NeoForge +
>   Connector 下都是 2 槽位 / 732 引用，832 高维度正常跑 GPU、地形正确
>   （2026-11-13）。README/教训 #39/#44 里"Connector 下 Tectonic 必回退 CPU"
>   的旧结论随之作废（教训 #57）。
> - `alpha.9`：executor 缓冲按**实际维度形状**分配（`cells_per_chunk` /
>   `blocks_per_chunk`），形状变大时重建 executor；无法容纳的批次**直接报错**
>   而不是提交。修掉 Tectonic（832 高）地形错乱：BlockOut 的 block buffer 原先
>   按「高度 ≤512」写死，43 chunk 的批次要拷 36.6 MB 出 32 MiB 的 buffer，
>   wgpu 校验失败后**整条 command buffer 被丢弃**，回读到的就是脏数据——
>   表现为悬空板块、被切断的悬崖（教训 #56）。顺带把原版形状（高度 384）的缓冲
>   占用降下来：block buffer 32 MiB→25 MB，每份 staging ≈145→≈115 MB。
>   遗留：批次上限（64）与 external 节点上限（64）仍是代码里的魔数，没有按
>   设备实际限额反推，见下方 0.1.8+ 待办（同一形状下 cell 高=4 时 `b2_out`
>   会超 128 MiB）。
> - `0.1.7` 收尾：可读性整理 + 一处泄漏修复。`executor.rs`
>   的缓冲尺寸算术集中到 `BufferSizes::for_config`，bind group 改用
>   `binding(index, buf)` 简写，beard/aquifer 的重复魔数提为具名常量；
>   `StagingChain::consume` 换成可重入的 `recycle`（任意阶段都可归还，未完成的
>   map 由 `unmap` 取消），`finish_batch` 在解码失败时也归还 staging slot 与
>   input set。原先 `read_results` 的 `?` 会把两者永久借出——shutdown 超时那条
>   路径正好会走到，raw 模式还会对未就绪的 slot 调 `consume`。Java 侧
>   `MAX_EXTERNAL_NODES` 提为类常量并清理全限定名。
>   验证：`cargo test` 91 passed / 7 ignored；五平台 native 全部重建；
>   Fabric 与 NeoForge+Connector 实机通过。
>
> 0.1.7 的全部改动都已合入 `main` 并打了 tag `v0.1.7`；
> `feature/chunk-status-integration` 已全部合入 `main`，该分支于 2026-11-14
> 从本地和 GitHub 删除（内容仍在 `main` 历史里）。
>
> **0.1.8（开发中）**
> - aquifer 预留量改为按 shape 推导：原版 384 高从写死的 1200 降到 315
>   cells/chunk，Tectonic 832 高为 657，修掉约 2000+ 高维度上传时的写越界。
> - aquifer 上传加显式越界检查（失败整批，不再交给 wgpu 校验后丢命令）。
> - 删除从未被引用的 `resources::buffer_pool`，缓冲复用只剩定槽位模型。
>
> **实测基线**（细节见 §9 / §10 / §11 / §13.7；跨组数据不可直接比较，见教训 #45）
> - 桌面端端到端反超纯 CPU（Chunky r=500，RADV RENOIR）：**18 s vs 23 s**
>   （0.1.5 流水线化后；同规模后续记录为 **15 s**，§10）。0.1.7 `4×4` 对齐批次
>   进一步到 **13 s**（§13.7，另一组种子）。GPU 直接输出 per-block 方块状态，
>   CPU 只写 palette。
> - 移动端（Mali-G610 平板）：3m12s → 2m19s（+61%）；纯地形 23.52 → 30.65 CPS（+30.3%）。
> - 服务端 headless（r=500，无渲染干扰）：纯 CPU 27 s（121.1 cps）vs GPU 21 s
>   （143.6 cps，峰值 215.7）；端到端 +22%，峰值 +44.7%。纯 CPU 模式下主线程
>   排队（Amdahl 定律）锁死并发上限，CPU 占用通常只有 70–80%。
> - 流水线化 A/B（r=500，原版地形，同一核显）：executor 串行 20 s / CPU ~70%；
>   池化输入 buffer + 多批次在途 18 s / CPU ~50%。CPS 中位持平（188），p90 238 → 254。
>   **结论：流水线化省的是 CPU 等待与端到端时间，不是稳定态 CPS**——GPU 执行
>   （readback）仍是当时的瓶颈。
>
> 历史细节见「§15 附录：开发历程」。

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
| **端到端反超纯 CPU** | ✅ 18 s vs 23 s (RADV RENOIR，见 §9) |
| **移动端端到端提速** | ✅ 3m12s→2m19s；CPS 23.52→30.65 (+30.3%) |
| 史诗地形/构造地形兼容 | ✅ Epic Terrain, Tectonic |
| WeirdScaledSampler | ✅ 完整实现 |
| **C2ME `NewChunkStatus` 接入（阶段 1）** | ✅ 透明委托，见 §13 |
| **对齐批次（阶段 2，`chunk_batch_size`）** | ✅ 2/4/8 实机验证通过；4×4 最优（13 s / CPS 298），默认仍为 1；ModMenu 可切换，见 §13.7 |

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
    同一份 WGSL 在天玑 8100 上切 **GLES 后端**可正常出块（实测 CPS 22–29），
    说明问题在 Vulkan 路径/指令序列而非 WGSL 本身；但麒麟 980 未复现，
    故按**单机型现象**记录，不推广成"Mali + Vulkan 就提示切 GL"的通用结论。
    行数只能当粗略预警，真实上限只能靠实测。好消息是 lesson #36 的 error
    scope 让这类"驱动拒绝编译"变成可捕获的 `Err` → 该维度回退 CPU，而不是
    死锁或设备丢失。
16. **惰性缓存可预计算**：MC 的惰性表（aquifer waterLevels）在 GPU 上必须整表预计算（Java warmup ~0ms）。
17. **内存泄漏**：共享 map 要 remove-on-consume，不能只 insert。
18. **诊断代码放对位置**：别放在会被短路的分支后（否则是死代码）。

**方法论（最重要）**
19. **端到端指标才是判据**：Chunky 总时间，不看内部 batch 吞吐（§8.1 教训）。
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
    回退 CPU。
    **2026-11-13 更正**：当时"Connector 下 Tectonic 一定回退 CPU"的结论是错的，
    错在把**引用数**当成了节点数（教训 #55）。alpha.8 按实例去重后，**NeoForge
    版 Tectonic 3.0.28 在 Connector 下只有 2 个槽位 / 732 次引用**，832 高的
    维度正常跑 GPU：`sized_for=2625 cells/chunk / 212992 blocks/chunk`，shader
    与 Fabric 原生**逐字节相同**（3069 / 2967 行），地形正常。Connector 仍不是
    官方支持环境，
    但"外部化数量"不再是它的障碍——真正的障碍只有 64 槽位上限本身。

40. **CPS 对种子和视距敏感，但幅度依规模而定**：旧的 Chunky 端到端数据（18s vs
    23s、CPS 188）只在特定种子下成立，不同种子的噪声 octave 数量、spline
    分支命中率、aquifer 命中率确实不同，GPU 工作量也不同。但 2026-11-11 在
    r=1000、固定视距下用两个种子正式对照（§11.8），稳态只差 2.4–2.8%，低于
    重复间波动（pooled SD 3.3–4.0%）。所以：**跨版本/跨环境对比必须同种子、
    同视距**；种子差异不能用来解释两位数百分比，除非重测确认。视距是更强的
    变量，且影响主要集中在启动段（§11.8）。

41. **跨线程交接的队列必须有界，背压落在生产者身上**：`requestAsync` 曾往无界
    `LinkedBlockingQueue` 里 `add`，而 C2ME 提交 chunk 的速度远高于 GPU 退役
    速度（末地 radius 11820 是 ~2.2M chunk），于是几十万个 `Job` 堆在队列里，
    内存单调上涨、GC 压力饿死完成线程，GPU 反而空转等活，吞吐从 ~500 CPS 塌到
    个位数——而且是在跑到百分之几的时候就塌，远早于任务结束。
    修复是 `LinkedBlockingQueue(4096)`（远大于 `MAX_BATCH`，批次仍能瞬间凑满）
    + `offer(job, 50ms)` 轮询循环。要点：
    - **背压该落在提交的 C2ME worker 线程**，不是 GPU 侧；GPU 保持忙碌、内存保持平。
    - **用 `offer` 不用 `put`**：`put` 阻塞在满队列上无法被 `running = false` 唤醒，
      会让 teardown 永久挂起。超时轮询循环才能同时满足"等空位"和"能收摊"。
    - **收摊或中断时要异常完成 future**，让 C2ME 在 CPU 上重新生成这个 chunk，
      而不是让它永远等下去。
    - **判据是规模**：4225 chunk 的测试根本暴露不出来（还没来得及堆积就结束了），
      至少要 6 万+ chunk 才有压力。小规模全绿不能证明队列没问题。
    推广：凡是在线程之间交接请求/任务的容器，都该先问一句"上限是多少、满了谁等"。

42. **部署的 jar 必须晚于修复源码，且要让日志能自证**：本轮出现过一次惊魂——
    0.1.6 的 jar 构建于 02:38，而队列修复的**提交**时间是 02:53，看上去像是
    "号称含修复的实机数据其实不含修复"。真相是源码改于 02:37、构建于 02:38、
    提交于 02:53（本项目一贯"先实机验证，再提交"），jar 里确实有
    `QUEUE_CAPACITY = 4096`。但**光看 git 提交时间会得出相反结论**。
    因此定两条规矩：
    - **核对顺序**：`stat 源文件 mtime` → `stat jar mtime` → `git log` 提交时间。
      构建必须晚于源码改动；提交时间与构建时间的前后关系**不构成判据**。
    - **验证要落到字节码**：`unzip` 出 `WgpuBatchCoordinator.class` 后
      `javap -c -p -constants` 查 `4096` / `offer` / 退场字符串，比看版本号可靠。
    - **日志要能自证**：启动横幅里打 mod 版本（`c2me-opts-accel-wgpu 0.1.6` 已有），
      但队列上限这类修复参数目前不打日志，导致无法从 `latest.log` 反推。重要修复
      应在初始化时打一行"bounded queue, capacity=N"。

43. **能力门槛要贴着规范下限，别拍脑袋抬高**：`max_storage_buffer_binding_size`
    曾硬编码要求 256 MiB。这个数字原本是为了容纳 Tectonic 在 Sinytra Connector
    下暴露的 732 个 external 节点，但那 732 其实是**引用数**：按实例去重后只有
    2 个槽位（alpha.8，教训 #55/#57），256 MiB 的余量从一开始就没有用途。代价是 Adreno 750（骁龙 8 Gen 3）
    这类**正好只报 128 MiB**（WebGPU 规范默认下限）的现代移动 GPU 在启动时就被
    拒绝，玩家只看到"显卡性能不足"，无从判断是门槛而非能力问题。
    0.1.6 起降到 128 MiB。实测数据：桌面 RADV 报 2047 MiB，原版形状下最大的两个
    binding 是 `b2_out`≈85 MiB、`b7_block`≈25 MB（alpha.9 起按形状分配，见
    教训 #56），离 128 仍有距离。**余量最紧的是 Tectonic 这类高维度**：
    832 高、cell_height=4 时 `b2_out`≈92 MiB、`b7_block`≈52 MiB，仍低于
    128 MiB 门槛，但继续加高（或 max_batch 变大）就会顶到门槛。Fabric 原生
    Tectonic 下 `b6_external` 只有几百 KiB，不再是占用大户。
    **定门槛的方法**：先算出实际最大 binding 需求（按 max_batch × 每 chunk
    角点 × 4 字节），再取"不低于规范默认下限、不高于实际需求太多"的值；
    抬高门槛只会无谓地排除硬件。

44. **Connector 的兼容边界比预想的更硬**：曾尝试两种让 Tectonic 在
    NeoForge+Connector 下命中 GPU 的路线，都失败：
    - 在外部化之前主动调一次 `df.apply(no-op visitor)`，期望 mod 节点的
      `mapAll` 把自己降级成原版节点（Tectonic 的 `ConfigNoise` 在
      `smootherScaling=false` 时确实会这么做）。实测无效，`ConfigNoise` 原样
      返回，仍 external 化，仍 732 个。
    - 直接放进 Fabric 版 Tectonic + Fabric 版 Lithostitched 交给 Connector
      桥接。结果 Connector 加载其 Mixin 时直接
      `Failed to execute task .../BiomeMixin.class`，整个游戏崩溃。
    对照实验还确认了一件事：**Connector 不会包装原版密度节点**——纯原版地形
    在 NeoForge+Connector 下 `b6_external`=0、GPU 正常加速。
    **2026-11-13 更正**：这两条路线确实不通，但"Connector 无法完整桥接 Tectonic"
    的结论下早了。改放 **NeoForge 版** Tectonic（ModLauncher 原生加载，不经过
    Connector 的桥接）后一切正常：2 个槽位、832 高、GPU 加速、地形正确。所以
    结论应改为"用对平台构建件即可，不需要逆向适配"；同时 64 节点上限带来的
    CPU 回退在去重后也不再触发（见教训 #39 的更正）。

45. **测试数据必须带种子和可比性声明**：教训 #40 指出 CPS 对种子敏感，但
    这条纪律经常被自己违反。本轮就出过一次：两次 r=500 测试分别测得 20 s 和
    13 s，一度被解读成"跳过 biome 省了 7 秒"，实际上两次的种子是否相同没有
    被记录，而且 20 s 那次本身就偏离了 15 s 的历史基线。**没有种子标注的
    耗时数据，不能用于跨实验比较。**
    规矩：
    - **同一次实验的所有变体必须共用种子**，并在测试记录里写明。
    - **实操**：每次进世界预热时打一次 `/seed`，连同日志一起存到测试记录
      （不进本文档，本文档只记录结论）。Chunky 的 `Total time` 是端到端
      指标，最容易受种子污染。
    - **数据表必须声明可比性**：标明"同组内可比 / 跨组不可比"。历史上的
      0.1.6 实测表就混了不同种子、不同规模和不同维度，标题却写着"同种子"，
      已修正。
    - **视距也必须固定**：`renderDistance` 改变出生点预生成区块数（vd12 约
      25×25，vd6 约 13×13），端到端总耗时差 2–7%，但任务中后段吞吐接近
      （§11.8）。跨视距的 `Total time` 不可直接比较。
    - **内部指标比端到端稳定得多**：`encode/submit/readback/decode` 这类
      毫秒级分阶段数据受种子影响远小于总耗时，可以作为跨实验基准；端到端
      耗时只能在同一批实验内部比。
46. **用官方扩展点接入状态机，别改它的内部表**：`NewStatusHook` 是 C2ME 留给
    外部模块的稳定入口（官方 OpenCL 同样用它）。注入点必须是
    `beforeVanillaStatusRegister` 的 `HEAD`，在原版 delegate 注册前 `add`，
    新状态才会拿到 `statuses.size()`，随后注册的原版 delegate 顺延一位，顺序
    才是 `... BIOMES → 我们的状态 → vanilla NOISE`。放到 `RETURN` 就落在
    NOISE 之后，依赖关系直接反了。
47. **类初始化期间不能依赖"还没注册完"的静态表**：`NewChunkStatus.<clinit>` 里
    调 hook 时，`VANILLA_WORLDGEN_PIPELINE[NOISE]` 仍是 null。任何在构造器里
    解析该表的写法都会拿到 null（或触发二次初始化），必须**惰性解析 + 兜底**
    （本次是 `volatile` 字段 + 首次调用 `fromVanillaStatus`）。同理，构造器里
    不要打"已就绪"日志——真正可用是在第一次 `upgradeToThis`。
48. **优化前先算天花板，别先写 shader**："表面高度估算"从"不做"改成"待评估"
    之后，第一件该做的事是数调用点和调用量，不是动手实现。这一轮数出来的
    上限是每轮 r=500 约 4700 次 `getHeight`、1.4–3.8 s CPU，而墙钟里能拿回
    来的只有压在关键路径上的那部分；同时"从现有 GPU 结果反推高度"被时序、
    beardifier、surface rules 三条独立否掉。**没有分母的优化讨论都是猜。**
49. **"某结构会查高度"≠"它每次都查"：分支条件要从字节码确认**：§14.2 初稿按
    "矿井 placement 是 `spacing=1`，每个区块都是候选"估出一轮 r=500 有
    ≈4700 次 `getHeight`。实测只有 **696 次**——`MineshaftStructure` 的高度
    查询在 `type == MESA` 分支里，普通矿井走
    `StructurePiecesCollector.shiftInto(seaLevel, minY, rng, 10)`。凡是靠
    "结构集合 × placement 概率"推出来的分母，都必须先打点再信。
    反方向也有惊喜：实测单次 **2.2 ms** 是"1/6 区块"模型（≈0.6 ms）的 3–7 倍，
    说明重建 `ChunkNoiseSampler` 的固定开销主导——这一点在写任何代码之前
    都推不出来。
50. **兼容性要看"实际发布的 jar"，不要看分支名；能降级就别只靠版本号**：
    §13.6 曾按分支名断言"26.2 把 `theChunkSystem` 改成了 `chunks`"，实际
    `javap` 打开发布 jar 才发现 26.1/26.2/26.3 都还是 `theChunkSystem`，
    改名发生在 1.21.11 线；1.21.1 的 `0.4.0-alpha.0.30` 也仍然兼容。
    分支名、`gradle.properties` 里的 `mod_version`、甚至 GitHub release 列表
    都可能与用户装的那个 jar 不一致（1.21.1 走的是 Modrinth devbuild 编号）。
    **结论**：① 用 `javap` 核对实际 jar；② 版本号只做粗粒度闸门，真正的
    形状匹配交给启动时反射检查，不匹配就跳过功能并保留降级路径；
    ③ 反射检查代码若在 `Class.forName` 里加载会跑 `<clinit>` 的类，必须用
    `initialize=false`——尤其是这段代码本身就跑在别人的 `<clinit>` 里。

51. **A/B 换 mod 要断言"只装了一个"，并让日志自证**：一次 r=1000 的 A/B 里，
    切换脚本删的是 `c2me-ocl-0.29.jar`，而 CL jar 实际被复制成了 `ocl.jar`，
    于是三次"B 组"同时加载了两套加速器。后果不是 mixin 崩溃，而是静默错测：
    两个模块各自往 `NewStatusHook` 的 HEAD `add` 一个状态，CL 占 `ordinal=6`、
    我们在 `7`；区块先被 CL 推到 NOISE，我们的状态直接短路，**GPU 工作量为零**
    （没有 `shape:` / `first chunk`，也没有绿色"GPU 加速已启用"聊天提示），
    但仍挂着 4×4 的 399 依赖门控，于是同一轮比干净 B 慢 10 s 以上。
    规矩：
    - 切侧后断言 `mods/` 里**恰好一个** `c2me-opts-accel-*` / `ocl.jar`，数量
      不等于 1 直接拒绝开跑；删除时按通配符删，不写死文件名。
    - 跑完校验日志：A 必须有 `Initializing OpenCL`；B 必须有
      `GPU 加速已启用：<卡名>`（`MixinChunkGenerator` 在 `DimState` 建成时发），
      且不得出现 `Initializing OpenCL`。
    - `active: ordinal=6` 才是"只有我们"；`ordinal=7` 意味着前面还有一个加速器
      状态，直接判为污染。
    - **跑得快也要对**：污染批次看起来只是"慢一点"，没有任何崩溃或报错，靠
      端到端数字根本发现不了。

52. **覆盖层要画在 `GameRenderer.render` 里 `DrawContext.draw()` 之前，不是 TAIL、
    更不是 HUD 或聊天栏**：shader 编译发生在首次进入维度的 C2ME worker 上，而那
    正是 `DownloadingTerrainScreen` 占满画面的时候。`InGameHud` 的聊天栏、HUD
    回调和 `Screen` 自己画的内容都在这一层**下面**，所以"编译进度"用聊天提示或
    普通 HUD 回调根本看不见。`GameRenderer.render` 已经把世界、HUD 和当前
    `Screen` 全部画完，注入到它唯一那处 `DrawContext.draw()` 调用之前即可。
    但**不能注入 TAIL**：原版这个方法结尾是
    ```
    ctx.draw();                        // 974
    modelViewStack.popMatrix();        // 977
    RenderSystem.applyModelViewMatrix();// 983
    return;                            // 986 ← TAIL 在这里
    ```
    GUI 的 2D 矩阵（`translate(0,0,-11000)`）在 TAIL 之前已经被弹出，而
    `setOrtho(0,w,h,0,1000,21000)` 的投影还挂着：z=0 的四边形落到近裁剪面
    背面，被**静默裁掉**——不报错、不崩溃、日志干净，看起来就像"覆盖层没触发"。
    注入到 `draw()` 之前，矩阵还活着，而且我们自己排进共享 immediate 缓冲区的
    顶点会被原版紧接着的那次 `draw()` 一起刷掉。教训：判断"我画的东西在哪一
    层"时，要读目标方法的**结尾字节码**，别按方法名想当然。
    数据通道用**拉取式快照**而不是新造一条 upcall：Rust 侧只需一个互斥量
    （`world::progress`）存 `active/done/total`，渲染线程每帧做一次 downcall；
    暂停或最小化时自然零成本，也不会从 worker 线程碰 Java 对象。编译失败时
    靠 RAII guard 在 `?` 提前返回时清标志，否则进度条会永远停在屏幕上。

53. **"地形变了"先确认是不是同一个世界**：A/B 或回归测试里，一次"地形怎么变了"
    的警报最后是空响——两轮测试各自**新建了一个世界**（日志里
    `Deleting level Chunky WGPU`、`Deleting level Chunky`），新世界就是新随机
    种子，地形本来就该不同。判定方法很简单：看 `logs/latest.log` 里的
    `[c2me-wgpu] first chunk [x z]`，如果两轮的出生区块坐标不同，种子几乎必然
    不同，这时任何地形差异都不能算到代码头上；再核对 `[c2me-wgpu] shape:` 是否
    一致，一致说明维度/噪声设置没变，只是种子变了。
    规矩：**地形正确性只在固定种子、或复用同一份存档上比较**；纯粹测耗时/耗时
    回归才可以随便新建世界。同理，临时插桩（例如 sleep）只会改时间线，不会改
    生成结果——先排除"世界/种子/维度"三个变量，再怀疑代码。

54. **跨线程/跨阶段的数据必须由消费它的那次请求拥有，别放在进程级全局状态里**：
    external payload（绑定 6）原本挂在 `WorldContext` 上，靠
    `c2me_wgpu_set_external_inputs` 在每次 `request_chunks` 前覆盖一次。串行
    执行时这没问题；一旦允许流水线（`submit_async` 入队即返回、executor 稍后
    才在 prepare 阶段读它），"入队顺序"和"消费顺序"就解耦了，第二批的 upload
    会覆盖第一批还没消费的数据——地形错位且不报错。这也是当时有 external 节点
    就强制串行的原因。
    判据很简单：**"写入者"和"读取者"不在同一个调用栈、且生命周期需要跨过返回
    点，就不能共享可变全局**。改法是把 payload 作为参数随请求拷进 `Batch`，
    让它的生命周期与那次 GPU 提交绑定；副作用是 FFI 签名变化（旧 native 链接
    失败，属可接受的预发布不兼容），好处是串行分支和随之而来的"外部维度没有
    流水线收益"的例外一并消失。新增测试
    `external_payload_travels_with_batch` 用两个不同 payload 的批次断言回读值
    逐 cell 一致。
    另外：payload 长度必须与最终批次的 chunk 数严格校验（`chunks × slots ×
    cells`）。`flush()` 会排序去重，若调用方按自己的列表构造 payload 而队列里
    恰好还有别人的 chunk，长度对不上就是静默错位——宁可报错回退，也不要生成
    一张"看起来正常但错了一层"的地形。

55. **数"节点"前先想清楚数的是引用还是实例**：Tectonic 3.0.28 在 `latest.log`
    里报 `732 external density subtree(s)`，而它整个 jar 只定义了
    **7 个** `tectonic:config_noise`。原因是密度树是 DAG（注册表里的同一实例
    被许多父节点共享），而序列化器为了简单**不做 memo、逐路径内联**，于是
    "遇到几次未支持节点"= "写几个槽位"，把 2 个实际进入该维度路由的实例数成了
    732 个引用（整个 jar 定义 7 个 `config_noise`，其余不参与本维度），
    直接撞上 64 节点上限，整个维度回退 CPU——看起来像"mod 又没被支持"，
    实际是记账方式错了。
    改法：`IdentityHashMap<DensityFunction,Integer>` 按实例共享槽位。共享是
    **精确**的，不是近似：外置子树只通过 `sample(pos)` 参与计算，纯函数在同一
    角点必然给同一个值；引用它的位置拿到的是同一个 `ExternalInput(index)`。
    代价只有一点：CPU 每个批次仍要把每个**实例**在每 chunk 的角点上采一遍
    （2 × 5×105×5 × 4 chunk ≈ 2 万次采样），与 732 槽位的 770 万次差两个数量级。
    规矩：**遇到"数量爆炸"先分清引用数 / 去重后的实例数 / 真正的工作量**；
    日志里两者都打（`N subtree(s) from M reference(s)`），否则下一次还会误判。
    顺带修正了 64 节点上限的语义：它约束的是槽位（buffer 大小与 CPU 采样量），
    不是引用次数。

56. **持久缓冲的大小必须由"当前维度的形状"算出来，不能写死上界；而 wgpu 的
    校验错误会丢掉整条 command buffer，表现为静默的地形垃圾：Tectonic 进
    世界后地形是悬空板块、被切断的悬崖，日志里没有一条 ERROR 参与世界生成，
    只有两行 `[wgpu-error] Validation Error`：
    `Copy at offset 0 for 36634624 bytes would end up overrunning the bounds of
    the Source buffer of size 33554432`。42 chunk × `16*832*16` 方块 × 4 B =
    36.6 MB，而 `block_buf` 是按旧常量 `16*512*16`（高度 ≤512）分配的 32 MiB。
    **根因不是"算错了"，而是缓冲大小与形状解耦**：`cells_per_chunk` 早就是
    按维度算的，`blocks_per_chunk` 却留了个"足够大"的魔数；任何高度 >512 的
    数据包都会越界。
    两个规矩：
    - **凡是随维度/数据包变化的缓冲，尺寸都从 `world.shape()` 推**，并在
      真正提交前做一次显式的 `need <= have` 校验；容纳不下就返回错误让该批次
      回退，不要提交。
    - **别把 wgpu 的校验错误当成"会被上层看到"的失败**。`Copy` 越界时 wgpu
      拒绝的是**整条** `CommandEncoder`，队列里后续的 dispatch 也一起没了，
      但 `queue.submit` 的返回值并不携带这个失败；CPU 侧照常按 staging 的
      大小去 map，读到的就是上一批的残留或零——于是"GPU 算完了、结果是错的"。
      上游兜底越完善（这里是有循环体被清空的那类 bug 的天然温床），越要保证
      `ErrorFilter::Validation` 的作用域覆盖到提交点，并把回读长度与期望值对账。
    修法：`ExecutorConfig { cells_per_chunk, blocks_per_chunk }` 由
    `get_or_create_executor` 从实际形状传入，executor 只在**变大**时重建
    （旧 worker `shutdown()` 会先排空队列，不丢批次）；`prepare_batch` 再校验
    一次。附带收益：原版形状不再按 512 高分配，`block_buf` 从 32 MiB 降到
    25 MB，每份 staging 从约 145 MB 降到约 115 MB（3 份 + 输入集）。
    新测试 `oversized_shape_is_rejected_not_silently_corrupted` 固定这个契约：
    形状超出时必须 `Err("does not fit the executor buffers")`，而不是提交。

57. **把限制归因于"外部兼容性"之前，先排除自己代码里的计量缺陷**：0.1.6 的
    README 和文档都写着"Tectonic 在 NeoForge+Connector 下会因节点过多回退
    CPU，这是 Connector 的兼容性限制"，NF-1 那行实测记录也是"回退 CPU（732
    external）"。这个结论活了好几个版本，直到 alpha.8 把 external 槽位按实例
    去重、由 732 降到 2，NeoForge + Connector + Tectonic 才被实测证明能正常
    跑 GPU（832 高、shader 3069 行、地形正确，2026-11-13）。
    当时的观察没错（mod 节点确实被当作 external），错的是**归因**：真正撞墙的
    是我们自己"一个引用一个槽位"的记账方式，不是 Connector 的桥接能力。
    这类误判代价很高——它会让一条本可修好的路径被写进 README 当成永久限制，
    并且挡住后续排查（大家都以为"官方不支持，不用查了"）。
    规矩：
    - **写"某某组合/某某平台不支持"之前，先问三件事**：限制是谁报的（我们的
      阈值还是对方的拒绝）、阈值是怎么算的、换个记账口径还成立吗？
    - 这类结论要带**版本号 + 实测数值**（本条就是 `0.1.6 / 732 slots`），
      这样后续修复才能被识别为"推翻了旧结论"，而不是"又出 bug 了"。
    - 结论被推翻时，把**更正**写回原处（教训 #39/#44、NF-1 表、README），
      而不是只在新版本里加一句——否则读旧章节的人仍会得到错误结论。

### 版本记录与 0.1.8+ 待办

**0.1.4 已完全解决的历史遗留（存档记录）**
- ✅ **interpolated 边界 bug**（§15.2.2）：通过 Pass E 架构（拆解 nonlinear outer 树与 InnerRef 角点）解决，Epic Terrain/Tectonic 地形完全正常。
- ✅ **Barrier 噪声与含水层对齐**：完整移植原版 `calculateDensity`（含高度过滤和全局流体判定），彻底告别全水浮岛。
- ✅ **矿脉全 GPU 化**：完整的 RNG 与矿脉方块替换已实装，无需 CPU 回退。
- ✅ **Shard B 瘦身**：高频内联公式函数化，从 3084 行降到 1771 行。

**0.1.5 已完成（已发布）：代码整理与用户友好化**
- [x] **代码清洁**：
  - 清理排查期间留下的历史冗余代码、无用 `#[allow(dead_code)]`。
  - 审查并移除重复的 `c2me_wgpu.log` 日志写入逻辑（如果有）。
  - 修复项目内大量的 cargo `unused` 警告。
- [x] **用户友好化（UX）**：
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
    - 配置分四类：A 类影响 instance（`gpu_index`、`backend`、
      `disable_pipeline_cache`）、B 类影响 world 创建（`mapped_at_creation`）、
      C 类只影响行为（`debug_*`）、D 类在 chunk-status 注册时读取一次
      （`chunk_batch_size`，改动需重进游戏/重启，不能靠切存档热重载）。
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

**0.1.6（已发布）：modded 世界正确性与错误可见性**

- ✅ **暮色森林死锁根因**：`ExternalInput` 是角点量，必须像
  `InterpolatedNoiseSampler` 一样被 split 提取；`barrierNoise`（ch0）同样
  要 split，否则 Tectonic 这类世界会因 BlockOut 引用未绑定的
  `external_inputs` 而编译失败。
- ✅ **shader / pipeline 校验失败不再静默**：两者都用 error scope 捕获，
  失败即让该维度回退 CPU，而不是提交 invalid pipeline 导致死锁。
- ✅ **错误可读化**：`failed validation` 翻译成玩家能懂的中文/英文说明。
- ✅ **提示一致性**："GPU 加速已启用"改由真正初始化成功的维度上报，
  不再与世界进入解耦地抢跑；通知带颜色（绿=正常，黄=降级）。
- ✅ **external 节点上限 64**：超限直接回退 CPU，避免超出
  `max_storage_buffer_binding_size` 并拖垮 CPU 采样。
- ✅ **pipeline 缓存键改为哈希 WGSL 源码 + `bind_group.rs`**，废弃手写的
  `LAYOUT_VERSION`（曾两次忘记 bump 且失败是静默的）。
- ✅ **无界队列泄漏**：`requestAsync` 的队列从无界改为 4096 上限 + 带超时
  `offer` 背压（详见教训 #41）。大规模预生成时内存不再单调上涨，shutdown
  不再超时失败。

**0.1.6 实测数据（Chunky）**

> ⚠️ **本表数据在 HEAD 上测得，不是 `v0.1.6` tag 的构建**：HEAD 含 tag 之后的
> `dc7c006`（readback 去冗余拷贝）、`f3e9c39`、`a283cb1` 等提交。复现发布版
> 数据需 `git checkout v0.1.6` 重新构建。

> ⚠️ **本表不可跨行比较绝对耗时**：各组使用不同种子、不同规模（4225 / 63001 /
> 160801）、不同维度。同组变体之间（如 A-3/A-4 vs A-5，或 NF-2 vs NF-3）才构成
> 有效的 A/B 对照。种子对 CPS 的影响见教训 #40、#45。

| 组 | 实例 | 地形 | 规模 | 耗时 | CPS 中位 | shutdown 失败 | 含队列修复 |
|---|---|---|---|---|---|---|---|
| A-3/A-4 | Fabric | 原版 | 4225 | 18 s | — | — | ❌ |
| A-5 | Fabric | 原版 | 4225 | **15 s** | 206.8（p90 282.2） | **0** | ✅ |
| B-2 | Fabric | 原版 | 4225 | 17 s（wgpu 禁用） | — | — | — |
| A-1/A-2 | Fabric | Tectonic | 4225 | 26–33 s | — | — | ❌ |
| NF-2 | NeoForge | 原版（末地） | 63001 | 1m49s | 516–695 | 0 | ❌ |
| NF-3 | NeoForge | 原版（末地） | 160801 | 4m31s | **599.8**（p90 649.9） | **0** | ✅ |
| NF-1 | NeoForge | Tectonic+Connector | — | alpha.8 前回退 CPU（732 external）；alpha.8 后 **GPU 正常**（2 槽位 / 732 引用） | — | — | — |

> 注：A-5 的 jar 构建于 02:38、源码改于 02:37、提交于 02:53，字节码已确认含
> `QUEUE_CAPACITY = 4096`（见教训 #42）。4225 chunk 规模太小，无法暴露队列问题，
> 含修复的真正压力验证是 NF-3。大规模 GPU vs CPU 对照（B-5，同 4225/400c 口径）
> 尚未跑。

**0.1.7（tag `v0.1.7` 已推送）：C2ME chunk-status 接入**（原
`feature/chunk-status-integration` 分支已全部合入 `main`）

- [x] **阶段 1：打通官方扩展点（透明委托）**。通过 `MixinNewStatusHook` 在 C2ME
  注册原版 `NOISE` delegate 之前插入 `WgpuNoiseStatus`；该状态的
  `upgradeToThis` / `getDependencies` 全部转发给 C2ME 的
  `VanillaWorldGenerationDelegate`，GPU 仍走原有 `populateNoise` →
  `WgpuBatchCoordinator`。详见 §13。
- [x] `fabric.mod.json` 锁到 `"c2me": ">=0.4.0-alpha.0.29 <0.5.0"`（下限是
  `NewStatusHook` 可用的版本，上界防止跨到可能重写 API 的新线）。
- [x] **C2ME 运行期形状检查**：`C2meCompat.verify()` 在注册 `WgpuNoiseStatus`
  前反射核对 `ChunkLoadingContext` 五组件、规范构造器、`TheChunkSystem.getHolder`、
  `ItemHolder`、`ChunkState.reachedStatus`、`NewChunkStatus` 的构造器与两个静态
  方法。不符时打一条 ERROR 并跳过注册，旧的 `populateNoise` GPU 路径继续工作。
  用 `Class.forName(name, false, ...)` 避免在 `NewChunkStatus.<clinit>` 里触发
  其他静态初始化。见 §13.6。
- [x] 构建期 C2ME/FlowSched stub + `compileOnly` RxJava 3.1.12，不打包、不链接 C2ME。
- [x] **阶段 2：对齐空间批次（`chunk_batch_size`）**。把 `N×N`（2/4/8）的
  对齐方阵收敛成一个依赖单元：只有最小角（anchor）走 `upgradeToThis`，
  其余成员依赖 anchor 的本状态；anchor 声明整组的原版 `NOISE` 依赖并集，
  然后**仍通过 C2ME 的 `VanillaWorldGenerationDelegate`** 并行生成每个成员。
  GPU 工作继续留在 `populateNoise` → `WgpuBatchCoordinator`，因此没有第二份
  地形实现。默认 `1` 完全等价于阶段 1。设计见 §13.6。
- [x] **阶段 2 实机验证**：`chunk_batch_size=2` → 4 → 8，同种子 r=500 对比
  地形、CPS 与端到端。结果 14 / **13** / 15 s，4×4 最优（CPS 298）；无死锁、
  无 member 漏生成、地形正常。详见 §13.7。默认值暂不提高，4 作为推荐手动值。
- [x] **shader 编译进度覆盖层**。首次进入维度要编译 shard，桌面端几百毫秒到
  几秒、移动端更久；期间画面卡住却毫无反馈。现在在屏幕顶部画进度条与
  `n/N`、已用秒数，注入在 `GameRenderer.render` 唯一那处 `DrawContext.draw()`
  之前，覆盖世界、HUD 和 `DownloadingTerrainScreen`。进度由 Rust
  `world::progress` 的拉取式快照提供（`c2me_wgpu_compile_progress`），编译
  失败时 RAII guard 会清掉标志，不会留下卡住的进度条。见教训 #52。
  实机验证方式：清空磁盘管线缓存（`rm -rf ~/.cache/c2me/wgpu_pipelines`）后
  进新维度，覆盖层持续 1–3 s；日志里 `compile overlay shown` / `hidden`
  成对出现。**热缓存下只有约 50 ms，看不见是正常的**，不要据此判断功能失效。
- [x] **external payload 随批次携带（alpha.7）**，不再走进程级
  `world.external_data`。原先只有 `pipeline = externalNodes.isEmpty()` 时才流水线：
  有 CPU 外置节点就强制串行。串行不只是慢，更是**数据竞争**——`submit_async`
  入队即返回，executor 稍后才在 prepare 阶段读全局数据，多批次在途时下一批的
  upload 会覆盖前一批尚未消费的内容。现在 payload 随
  `request_chunks(world, positions, external_count, external_ptr, external_len)`
  进入 `Batch.externals`，executor 只读自己这一份（池化的 external buffer 仍
  按需增长，小批次只写前缀）；进程级接口与串行分支均已删除。FFI 签名变更，
  旧 native 会链接失败（模块报 symbol mismatch），随版本号升级。Rust e2e 测试
  `external_payload_travels_with_batch` 覆盖。
- [x] **收尾整理与泄漏修复**：`executor.rs` 的缓冲尺寸算术集中到
  `BufferSizes::for_config`、bind group 改用 `binding()` 简写、beard/aquifer
  魔数提为具名常量；`StagingChain::recycle` 让解码失败时也归还 staging slot 与
  input set，修掉 shutdown 超时路径的泄漏。行为不变，详见顶部 0.1.7 记录。
- [ ] **阶段 3：批内真正共享采样上下文**（官方 CL 的剩余收益）。当前批次只
  收敛调度依赖，每个成员仍各自构建 `ChunkNoiseSampler`；要进一步必须自己
  持有 `Blender` / `StructureAccessor` / sampler，风险远高于阶段 2，暂不做。

**0.1.8（未发版）**
- [x] **aquifer 预留量按 shape 推导**：binding 8 原先写死
  `AQ_MAX_CELLS_PER_CHUNK = 1200`（按原版 512 高估），维度到约 2000+ 高时
  `upload_aquifer_tables` 会写越界。现在 `ShapeConfig::aquifer_cells_per_chunk()`
  由 `size_y = (min_y+height)/12 - min_y/12 + 3` 推出（×9），`ExecutorConfig`
  携带该值，`fits()` 把它纳入 executor 重建条件；同时 `upload_aquifer_tables`
  和缓冲分配共用同一个函数，不会再各算一份。原版 384 高从 1200 降到
  315/chunk，Tectonic 832 高为 657/chunk；日志的 `sized_for` 增加 aquifer cells。
  单测：`aquifer_cells_follow_the_shape`、`aquifer_reservation_tracks_the_shape`。
  实机验证（2026-11-14，Fabric 与 NeoForge+Connector，均 Tectonic 832 高）：
  两边的 `sized_for` 都是 `2625 cells/chunk / 212992 blocks/chunk / 657
  aquifer cells/chunk`，executor 只创建一次，无 validation error、无回退。
- [x] **aquifer 上传加显式越界检查**：`upload_aquifer_tables` 改为返回
  `Result`，写入前用 `aq_p + aq_s` 的实际字节数比对 binding 8 的容量，越界就
  失败整批，而不是交给 wgpu 校验——校验失败会丢掉整条 command buffer，症状是
  脏地形而不是报错。`prepare_batch` 原有的容量检查同时把 aquifer cells/chunk
  纳入。单测：`aquifer_upload_matches_the_buffer_at_max_batch`。
- [x] **删除未被引用的 `resources::buffer_pool`**：核查发现它从未被任何代码
  使用（真正的复用是定槽位的 `InputBufferSet` 与 `StagingChain`），详见下方
  “显存池按用途分类”一条的结论。

**0.1.8+ 待办**
- [ ] **能力上限从设备实际限额反推，别写死在代码里**（alpha.9 遗留的口子）。
  alpha.9 让缓冲尺寸跟着形状走，等于把"能用/不能用"变成形状的函数，但代码里
  仍有一批魔数在替设备做决定：
  - `ExecutorConfig.max_batch_size = 64`（Rust/Java 两侧各一份 `MAX_BATCH`）；
  - `MAX_EXTERNAL_NODES = 64`（`MixinChunkGenerator`）；
  - `features.rs` 的 128 MiB 门槛；
  - 两处尚未校验的硬限额：`max_storage_buffer_binding_size` 与
    `max_buffer_size`。
  **现状数据**（64 chunk 批次，`Limits::default()`）：原版 384 高 →
  `b2_out` 85.3 MiB / block 24 MiB / staging 109 MiB；Tectonic 832 高、cell 4×8 →
  92.3 / 52.0 / 144.3 MiB，对应限额 128 / 128 / 256 MiB，都还在里面。
  但**同一高度若 cell 高是 4**，`b2_out` 就要 192 MiB，直接越过 128 MiB ——
  Tectonic 只是恰好用 4×8 躲过去。越过之后的失败方式正是 alpha.9 修掉的那类：
  wgpu 校验失败、整条 command buffer 被丢弃、回读脏数据。
  **改法**：`batch_limit = min(hard_cap, binding_limit / 每 chunk 绑定字节,
  buffer_limit / 每 chunk 总字节)`，按形状和设备限额算；Java 通过一个只读
  FFM 查询拿到这个上限再攒批（否则 `prepare_batch` 的"容纳不下"会让 Java
  一批一批地失败）。external 的 64 同理，应由每个节点每 chunk 的角点字节数算出。
  **关键前提**：必须用 **device** 的限额，不是 adapter 的。设备是用
  `required_limits = wgpu::Limits::default()` 创建的，adapter 报 2047 MiB 也
  不起作用，真正的 ceiling 就是 WebGPU 默认的 128/256 MiB。
  **关于"临时建 instance 查真实限额"的想法**：方向对，但要注意活动设备的约束——
  Android 上已经有了一个 device 时再建临时 device，会冻住渲染器（`instance/mod.rs`
  里 adapter 列表就是为此改成初始化时构建一次、GUI 不再自行枚举）。所以：
  - 有活动设备时，直接读它的 `device.limits()` / `get_info()`，不新建；
  - 只有在"还没有设备"的阶段（启动能力检查、GUI 选卡）才适合临时实例；
  - 现在启动日志已打出 adapter limits，缺的只是一个 Java 能读到 **device**
    limits 的查询口。
  关联：待测与验证计划第 2 项（65–200 个 external 实例的中间地带）。

### 待测与验证计划（不属于版本功能）

- [ ] Android 实机回归：0.1.6 引入的 external 上限、barrier split、通知时机都
  只在桌面验证过，移动端未测。
- [ ] external 节点数 65–200 的中间地带：阈值 64 原本是拿"4 正常 / 732 回退"
  两端之间拍的，但 732 其实是**引用数**：按实例去重后，Tectonic 3.0.28 在本维度
  只有 2 个槽位（jar 里定义 7 个 `config_noise`，其余不参与该维度），Fabric 与
  NeoForge+Connector 下都已实测能正常跑 GPU（alpha.8，2026-11-13，教训 #55/#57）。
  去重后 65–200 这个区间还没有任何实测样本：需要构造 65–200 个**不同实例**的
  世界来验证上限与性能。上限本身应当由设备限额与每节点每 chunk 的角点字节数
  算出，而不是固定 64（见 0.1.8+ 待办 ·「能力上限从设备实际限额反推」）。
- [x] ~~**Connector 下 shader 膨胀的来源未查清**~~ **已结案（2026-11-13）**：
  膨胀就是 732 个 external 槽位造成的。同一台机器、同一 Tectonic 3.0.28、
  同一 832 高维度，去重到 2 个槽位后 shader 与 Fabric 原生**逐字节相同**
  （shard B 3069 行、Inner 2967 行、字节数也一致），而旧版 Connector 下是
  22546 / 22441 行。
  当时按"一个 `ExternalInput` 只占一行"推算出 732 行、与实际 1.94 万行差 26 倍，
  于是错误地排除了 external 这个原因；实测每个槽位在 inner shard 里带的是一整段
  取值表达式（≈26 行/槽），732 × 26 ≈ 1.9 万行，吻合。
  **教训**：估算"某类节点只占 O(1) 行"之前，先量一行到底有多长。
- [ ] external 维度的吞吐复测：0.1.5 那次 A/B 是端到端 +10%、CPU 占用
  70%→50%；之后 0.1.6 换成有界队列、0.1.7-alpha.7 又把 payload 改成随批次
  携带并删掉了"有外置节点就串行提交"的分支，两次都没重跑。现在 external
  维度也能流水线，理论上收益应大于 0.1.5 那次，需要同种子、同视距的 GPU vs
  CPU 对照（暮色森林或 Tectonic）。在此之前不要对外引用 external 维度的
  提速百分比。

### 借鉴官方 OpenCL 模块的优化点（0.1.8+ 任务池）

官方 CL 模块（见本地 `~/Documents/C2ME/C2ME-fabric/c2me-opts-accel-opencl`）
在架构上比我们走得更远（4x4 空间批处理、全链路 GPU 化）。结合 ARR 协议与
跨平台底线，提炼出以下可借鉴的方向，按**实施难度**排序：

#### 🟢 低难度：内部微调（改动小，收益直接）
- ~~**调色板写入加速（Unsafe 写入）**~~ **(已证伪，勿重试)**
  * 参考官方 `c2me$setUnsafe`，曾实现过一版：新增 `MixinPalettedContainer`
    暴露 `c2me$setUnsafe`，`writeBlocksFromGpu` 改走底层 `PalettedContainer.set(index, value)`，
    每 section 写完再调一次原版 `ChunkSection.calculateCounts()` 重建计数。
  * **实测结果（Chunky r=500，同种子，预热 5s）**：基线 15 s / Rate 298.4，
    改动后 16 s / Rate 299.3。Rate 差异 0.3% 属噪声，**没有收益**；CPS 峰值
    (325 vs 283.7) 是 2 秒窗口的瞬时波动，不作判据。
  * **根因**：`setBlockState` 的记账开销本来就不是瓶颈——实测 `writeback`
    仅 1.4 ms/chunk，分摊到 9 个 worker 线程后墙钟影响可忽略。而新写法把
    **空气方块也计入了成本**：原版 `setBlockState` 对 `v == 0` 的方块直接
    `continue` 跳过、零记账；`calculateCounts()` 却要遍历该 section 全部
    4096 格（其中 40–60% 是空气）重算三个 short 计数。省下的增量更新，
    被"给空气补记账"吃掉，总记账量反而接近翻倍。
  * **教训**：优化前先测量瓶颈在哪。`calculateCounts()` 是给"整体替换
    section"场景设计的，不适合逐方块写入；官方 CL 能用是因为它的瓶颈在
    GPU 往返与 4x4 批处理，写回不在关键路径上。
- ~~**攒批空间排序（伪 4x4 聚类）**~~ **(分析判定无效，未实施)**
  * 官方强行在状态机层面对齐 4x4 区块。咱们不能改 C2ME 状态机。
  * 折中方案（攒批后按坐标排序）**分析后判定无效**：`batch.sort()` 只改
    顺序、不改集合，而这批 chunk 本来就是 C2ME 按空间邻近顺序提交的；
    更重要的是我们的 shader 对每个 chunk 独立计算，**没有跨 chunk 的数据
    复用**，所以顺序对 GPU 工作量没有影响。真要吃 4x4 收益必须改批次
    *选择*（攒满一个对齐区域才发），那是中高难度的事。
  * 注：代码里现有的 `batch.sort()` 保留（Rust 侧要求位置有序），但不是优化点。
  * 真正的 4×4 批次已由 0.1.7 的 `chunk_batch_size` 实现（改的是批次*选择*，
    见 §13.6），本条仅记录当年"只排序"的无效思路。
- ~~**调大 `MAX_WAIT_MS` 让批次攒大**~~ **(已证伪)**
  * 观察到平均批次只有 ~4.5 chunk（655/144），怀疑是 5ms 聚集窗口太短，
    固定开销（encode/submit/poll/readback/FFM）被摊在太少 chunk 上。
  * 实测 15ms：Chunky 总耗时仍 15 s，稳定 Rate 反而从 ~298 掉到 ~250 cps。
    **批次没有变大**——说明喂不满的原因是 C2ME 的提交速率本身，不是聚集
    窗口。单纯加长等待只增加延迟。已回滚到 5ms。
  * 推论：提批次大小要从 C2ME 侧（提交并行度）或改批次选择策略入手，
    在 coordinator 里调参数没用。

#### 🟡 中难度：逻辑重构（需要严格实机验证）
- [x] ~~**显存池按“用途”分类隔离**~~ **核查后不适用（2026-11-14）**：
  * 这条原判断基于 `resources/buffer_pool.rs` 的 `(usage, size)` 键控 LRU 池，
    但该模块自建库起**从未被任何代码引用**（`git log -S` 只找到它自己的创建
    提交，executor 里也从未导入），属于残留的死代码，文档把它当成了在用的实现。
  * 真正在用的复用全是**定槽位**：`executor::InputBufferSet`（params / positions /
    aquifer / beard / external 各是独立字段，尺寸在 executor 建好时定死）和
    `StagingChain` 的 fence 链。任何缓冲都不会按尺寸跨用途传递，所以"输出缓冲
    被当成输入复用"在当前架构里不可能发生，不需要额外分桶。
  * 处理：删除该死模块（`resources/mod.rs` 里写清现在的复用模型）。若以后真要
    引入按尺寸的池，再按用途分桶——别恢复这个键。
- [x] **表面高度估算预计算（Estimate Surface Height）——已实测，结案（§14）**
  * 实测（`debug_height_probe`，同种子 r=500）：**696 次调用 / 1.58 s**，
    0.165 次/区块，单次 2.2 ms，重复率 6%，约一半与在途 GPU 批次重叠。
  * 初稿的"矿井每区块 1 次 → 4700 次"是错的：`MineshaftStructure` 只在
    `type == MESA` 分支查高度，普通矿井用 `shiftInto`。**教训 #49**。
  * 结论：纯 CPU 记忆化省不到 1%；GPU 懒算被单列 readback 延迟吃掉；GPU 区域
    预计算要多跑一遍噪声。**不做**，探针保留可随时复测。
  * "从现有 GPU 结果反推高度"另有三条硬伤（时序 / beardifier / surface
    rules），见 §14.3。

#### 🔴 高难度：核心功能突破（长远目标）
- [ ] **把生物群系（Biome）拉上 GPU——前提已变，需重新评估**
  * 原结论（§15.1 的 2n，2026-10，代码在 `experiment/biome-gpu` 分支）是**净负优化**：
    biome 在 NOISE 之前，GPU 每次 14ms 的往返延迟直接卡在关键路径上，还抢占
    地形队列。这个实测结论**对"独立 GPU 路径"依然成立**。
  * 但当时的架构前提已经变了：
    1. ~~biome 必须走独立的 GPU 请求~~ → 官方 CL 的做法是把 BIOME 和 NOISE
       放进**同一个批次、同一次提交**，没有额外往返。这个模式我们现在才具备
       实现条件（需要自定义 `NewChunkStatus`）。
    2. ~~抢占地形队列~~ → 有界队列 + `pipeline_depth=3` 后，队列纪律比当时好，
       但共享 executor 的争抢是否消失尚未验证。
  * **补充实测**（2026-11）：跳过 `populateBiomes` 后跑 r=500，`inflight` 纹丝
    不动（1.0–1.3/4），CPS 峰值仅 +4%。说明 biome 的计算量本身不是上游瓶颈，
    **收益上限不高**。
  * 结论：**从"证伪"改为"收益有限但机制可行"**。§15.1 的 2n 失败的是实现方式，
    不是这件事本身。若重做，必须走"合并进同一批次"，绝不能重走独立路径。
- [ ] **多 Kernel 流水线（暂缓/仅评估）**
  * 官方用 Event 串联 5 个独立 Kernel。咱们目前是一个 Compute Pass 跑多个 Shard。
  * **不建议盲目拆分**：Vulkan 的提交和同步开销可能吃掉拆分收益。除非实机
    发现某个 Shard 明显拖后腿，否则维持现状。

> **⚖️ 架构边界（2026-11 修订，原为"兼容性红线"）**
>
> **仍成立的：**
> - **绝不放弃 Rust + FFM + Vulkan 架构去碰 JNI / OpenCL。** 这是跨平台
>   （fp32、Mac、Android）的根基，没有任何理由动摇。
>
> **已解除的：原"绝不引入 C2ME ChunkSystem 状态机强依赖"。**
>
> 建立这条红线的理由曾是：我们是伴生模块，C2ME 内部 API 不稳定，绑死会让
> 大版本更新直接崩溃。这个判断在当时是对的。
>
> 解除的三条依据（2026-11 查证）：
> 1. **扩展点是官方提供的，不是 hack**。C2ME 在 `compat/internal_api` 包里
>    暴露了 `NewStatusHook.beforeVanillaStatusRegister(ArrayList<NewChunkStatus>,
>    ChunkStatus)`，官方 OpenCL 模块用的就是它。
> 2. **三个版本结构一致**。查证 1.21.1（0.4.0-a.0.29）、1.21.11（0.4.0-a.0.28）、
>    26.2（0.4.2-a.0.60）的 chunk-system jar：`NewStatusHook`、
>    `compat/internal_api` 包、`VanillaWorldGenerationDelegate extends
>    NewChunkStatus`、`upgradeToThis` 返回 RxJava `Completable` —— 全部一致。
>    写一套集成代码可以跨版本复用，只需处理命名空间差异。
> 3. **旧路径已经走不通了**。1.21.11 起 `populateNoise` 的返回值从
>    `CompletableFuture<Chunk>` 变成同步 `Chunk`，且签名改为
>    `(Blender, StructureAccessor, NoiseConfig, Chunk, int, int)`。C2ME 自己也
>    改成了只 redirect 内部 executor（`MixinNoiseChunkGenerator` 仅有两个
>    `redirect*Executor` 方法）。继续绑在 `populateNoise` 上等于放弃所有新版本。
>
> 解除的代价，必须写清楚：
> - 依赖 `ChunkLoadingContext` / `ChunkState` / `NewChunkStatus` / `ItemHolder`
>   / `KeyStatusPair` 等内部类；
> - **C2ME 大版本更新时可能启动失败**（不是降级，是崩溃）；
> - 必须锁定 C2ME 最低版本，并在 `fabric.mod.json` 里声明。
>
> 因此执行方式：**分支开发 + 强制最低 C2ME 版本 + 每版本实机验证**，
> main 分支在验证通过前不变。
>
> **2026-11 补充（0.1.7-alpha.5）**：新增运行期形状检查 `C2meCompat`
> （不符则跳过注册、保留 `populateNoise` 路径），并把 `fabric.mod.json`
> 依赖改为 `>=0.4.0-alpha.0.29 <0.5.0`。原先"大版本更新直接崩溃"降级为
> "0.4.x 内部差异可降级，跨 0.5.0 由加载器提前拦截"。见 §13.6。

### wgpu 库自身的环境变量（仅排查用）

**本模组的全部开关都在 `config/c2me-wgpu.json`，不读任何 `C2ME_*` 环境变量**
（0.1.5 起已彻底迁移）。下面两个是 wgpu 库自己的标准变量，模组只是"顺带"受
它们影响，正常游玩不需要设置：

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

> **2026-10：C2ME 依赖关系**（§5）。构建期通过 Stub 提供接口签名，运行期强依赖调度器
> （提供 GPU 批处理赖以吞吐的多线程并行度）。

### 2.1 构建
曾用 Gradle `includeBuild` 引入 C2ME 复合构建，**现已移除**。改为在 `src/stub`
提供 `SchedulingManager` / `IVanillaChunkManager` 的接口签名（`compileOnly`，不打进
jar），Java 代码直接引用这些类。详见 §5。

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
- **编译期**：`src/stub` 提供 `NewChunkStatus` / `NewStatusHook` /
  `SchedulingManager` 等接口签名（`compileOnly`），另加
  `compileOnly io.reactivex.rxjava3:rxjava:3.1.12` 仅为类型检查；C2ME 与
  RxJava 都不打包、不链接。
- **运行时**：`fabric.mod.json` 声明
  `"c2me": ">=0.4.0-alpha.0.29 <0.5.0"` 硬依赖（下限即 `NewStatusHook`
  可用的版本；上界防止跨到可能重写 API 的新线，见 §13.6）。实测（§5）：
  无 C2ME 时桌面 CPS ~144→~13（批攒不满）。C2ME 是性能硬需求。
- **兼容性红线**：只用 C2ME 的公开扩展点（`NewStatusHook` /
  `NewChunkStatus`），不 mixin 它的内部实现。**运行期**由
  `C2meCompat.verify()` 在注册前核对 `ChunkLoadingContext` / FlowSched 的
  字节码形状；不符时只打 ERROR 并跳过对齐状态，旧的 `populateNoise` GPU
  路径继续可用（不再直接 `NoSuchMethodError`）。见 §13.6。
  **编译期** stub 只保证源码能编过，不等于运行期一定匹配——这是原来
  "编译期会直接失败"说法的修正。
- **许可证红线（2026-11 补充）**：本地的 `c2me-opts-accel-opencl` 模块是
  `All-Rights-Reserved`（`LICENSE-ARR.txt`）。反编译只允许用来确认**行为事实**
  （例如"官方确实通过 `NewStatusHook` 做批处理"）和公开 API 的存在/签名，
  **不得复制代码、常量、命名、注释或类结构**。本项目的实现只从 MIT 的
  C2ME 公开 API、原版 `ChunkGenerationSteps` 语义和本项目实机数据独立推导。
  需要对照时，读 `C2ME-fabric` 仓库（MIT）而不是 CL 的字节码。

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
下行（Java → Rust）是固定参数的 C 函数，外加 `c2me_wgpu_set_aquifer_table` /
`c2me_wgpu_set_beard_table` 推入的整表数据；external（CPU 外置节点）的值不再
走单独 setter，而是作为 `c2me_wgpu_request_chunks` 的参数随该批次一起传入
（0.1.7-alpha.7，见教训 #54）；
上行（GPU → Java）只有一个聚合 blob，布局见 §3.4。旧的 per-chunk `ResultBlob`
map 已不在生产路径上（§12）。

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
   aquifer + 矿脉"路径已按 §8.2 的测量结论删除（CPU 写回把 CPS 钉在 ~150，移动端 ~6）。
   三个相关开关（`C2ME_WGPU_BLOCK_OUT/AQUIFER/BEARD`）也随之取消——它们已是正确性的
   必要条件，关掉只会产生错地形。
6. 矿脉（`OreVeinSampler`，连片大矿脉）已在 GPU 上实现：`block_out.wgsl` 里的
   `vein_state()` 复刻了 MC 的采样器，RNG 种子由 `c2me_wgpu_set_ore_seed` 传入。
   普通矿石（煤/铁/金等）仍由 biome feature 阶段放置，不经过这条路径。
7. **异步调度**（0.1.5 起）：`populateNoise` 提交 GPU 请求后先返回 pending
   future，C2ME worker 不阻塞；GPU 完成后由
   `SchedulingManager.positionedExecutor(pos)` 把写回任务投回同一个 worker 池，
   future 完成时才释放位置锁。历史设计记录见 §15.6 迁移表的 18.6b 行。

---

## §4 当前完成度

见顶部「速查 · 当前完成度」表。

---

## §5 与 C2ME 的关系（构建期 Stub + 运行时强依赖）

- **构建期（Stub 依赖）**：为了不拉取整个 C2ME 源码，我们通过 `src/stub`（`compileOnly`）提供了 `SchedulingManager` 等类的接口签名。Java 代码中依然直接引用了这些类。
- **扩展点（0.1.7-alpha.1 起）**：chunk status 走 C2ME 的
  `com.ishland.c2me.rewrites.chunksystem.common.compat.internal_api.NewStatusHook`，
  由 `MixinNewStatusHook` 注入；`NewChunkStatus` 与
  `VanillaWorldGenerationDelegate` 同样只存在于 stub。**没有 mixin 任何 C2ME
  内部实现类**，升级时只受公开接口签名影响。
- **运行时强依赖**：实测无 C2ME 时桌面 CPS ~144→~13（缺多线程并行，批攒不满）。同时，异步写回架构强依赖 `SchedulingManager.positionedExecutor()` 来执行，若获取不到调度器，GPU 加速将直接回退纯 CPU 路径。
- **自写 AW**：不再借 C2ME 的 access widener，避免其删条目导致 `IllegalAccessError`。

---

## §6 混合执行（任意 modded 密度函数兜底）

Rust 不认识的密度节点（地形 mod）在 CPU 按同一角点网格求值，作为 "external input"
喂给 GPU（binding 6）。其余树仍走 GPU，**modded worldgen 保持工作**，只有不支持的子树付 CPU 成本。
实机验证：Tectonic、Epic Terrain 正常。

---

## §7 诊断基础设施

- Rust 日志写 `<gameDir>/logs/c2me-wgpu.log`（Java 启动时传路径）。

所有开关都在 `config/c2me-wgpu.json`，本模组不再读取任何 `C2ME_*` 环境变量：

- `debug_verbose_logging`：shader 编译计时、缓存命中、CPS 统计。
- `debug_stage_timing`：分阶段计时，GPU 侧 `[batch-stage-timing]`（encode/submit/
  readback/decode）与 CPU 侧 `[writeback-timing]`。两组要一起读才有意义。
- `debug_height_probe`：`getHeight` 调用点打点（次数/耗时/重复率/GPU 是否在途），
  输出到 `c2me-wgpu-height`。默认关闭，`HeightProbe` 类加载时读一次，改值需重启。
  用途与判读见 §14.5。
- `debug_dump_*`：离线参考 dump（密度/方块/blob/beard/aquifer 表）。

---

## §8 核心结论（0.1.5 基线）

> 这四条是 0.1.5 架构定型的依据；后续实测见 §9–§14。

### 8.1 端到端 A/B：异步调度裁定

桌面异步调度曾净负（30 vs 纯 CPU 23s），根因是 GPU 往返固定延迟落在关键路径。
**教训：端到端指标才是判据。**

### 8.2 写回天花板实测

MIN_STAGE（跳全部写回）11s ≪ 纯 CPU 23s，证明**净负全部来自 CPU 写回留在关键
路径**，GPU 密度管线本身很快。

### 8.3 最小闭环：BlockOut shard

BlockOut shard 首次跑通（GPU 输出 per-block 状态字节），16s。

### 8.4 aquifer + Beardifier 全 GPU

- aquifer：Java warmup 惰性表（315 cell ~0ms）+ compact 序列化（2544B）→
  Rust 现算 blockPositions（种子来自表头）→ GPU 12-cell 扫描 + 分类。99.97%。
- beard：静态 24³ 权重表（每 world 一次）+ per-chunk pieces（固定 stride）。
  bit-exact。**必须 f32**（WGSL 无 f64）。
- **端到端反超纯 CPU，功能完整，地形正确**（当前基准见 §9：18 s vs 23 s）。
- 遗留 0.03% = 过渡带 `calculateDensity`（barrier 噪声）分支未实现。
- 验证链全离线：CPU-f32 / blockPositions / WGSL / e2e / beard，5 个测试。
- shard B 瘦身（2026-10-17）：`spline_seg` + `octave_stack` 两个 WGSL helper 替换
  高频内联展开，3084→1771 行（重回 Mali ~2100 安全区），bit-identical。
- Spline n==1 修复（2026-10-17）：单点 spline 漏了 `outside()` 外推 → 嵌套 values 全错
  → depth 偏差 ~0.08 → 海底 y≈43–55 误判 stone（chunk 50,50 diff 8.88%）。改
  `spline.rs`/`world.rs` 两处 n==1 分支 + bump `LAYOUT_VERSION` 4→5。验证：逐通道
  bad=0/1225、逐格 diff 归零（回退复现 +0.08~0.098）、GPU rms 2.5e-5/5.7e-5、84 测试
  全过、实机 Tectonic 与 Epic 滴水石锥均消失；怪坑（§15.2.2 非线性外层）仍独立存在。

---

## §9 执行器流水线化（0.1.5）

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
   注（0.1.7-alpha.7）：external 的**数据源**不再是被覆盖的进程级全局，而是
   `Batch.externals`（随请求拷入）；池化的 `external` buffer 只是承载它的容器，
   因此有 CPU 外置节点的维度也能安全地多批次在途（教训 #54）。

**实测（Chunky r=500，原版地形，RADV RENOIR）**：

| | 串行 | 流水线化 |
|---|---|---|
| 端到端 | 20 s | **18 s** |
| CPU 占用 | ~70% | **~50%** |
| CPS 中位 | 188.5 | 188.4 |
| CPS p90 | 238.1 | **254.2** |

**教训**：`batchMs` 在两种模式下口径不同，不可直接对比——串行时它只计 map 段，
流水线化后包含完整在途时间。只有端到端和 CPU 占用是可比的。

**教训（写回路径）**：写回（`writeback`）实测仅约 1.4 ms/chunk，且由 9 个
C2ME worker 线程并行承担，不是端到端瓶颈。曾尝试用底层调色板直写 +
`calculateCounts()` 重建计数来加速它，实测零收益（Rate 298.4 → 299.3，属噪声），
因为原版 `setBlockState` 对空气方块是零成本跳过，而 `calculateCounts()`
要对整个 section 的 4096 格（含大量空气）重扫一遍。**优化非瓶颈路径不会
有收益；判断瓶颈要看实测拆分，不要看"哪里代码看着低效"。**

**瓶颈仍是 GPU**：`readback` 占每批 90%，是 compute + 拷贝的真实时间，批次再多
也无法重叠。想再提 CPS 只能减少 GPU 工作量（降分辨率、减 shard），代价是地形质量
与移动端兼容性。

---

## §10 极限 A/B 测试：WGPU vs 官方 OpenCL (1.21.1 同台竞技)

**测试环境**：1.21.1 Fabric，AMD Cezanne 核显 (`gfx90c`)，都开启 ScalableLux，Chunky r=500 (4225 区块)，进图预热 5 秒后启动。

| 指标 | WGPU (Vulkan) | 官方 OpenCL |
|---|---|---|
| **总耗时** | **15 s** | **13 s** |
| **Chunky 峰值 Rate** | 317.6 cps | 339.1 cps |
| **模组内部 CPS 峰值** | 301.1 | 未打印 |
| **环境配置成本** | 零 | 极高（需手动隔离 `rusticl` 防黑屏） |

**核心结论**：
1. **OpenCL 的 2 秒优势**：底层通过 C 语言原生指针直写共享内存，免去了 WGPU 跨越 FFM（Java ↔ Rust）边界和 ResultBlob 回读的通信税。
2. **WGPU 的架构优势**：做到了真正的开箱即用。OpenCL 在 Arch Linux 上因 ROCm 7 与 Mesa 冲突，会直接把 GPU 干挂（TDR 重置），普通玩家难以自行解决。
3. **光照优化的决定性作用**：此前未装 ScalableLux 时，原版单线程光照成为瓶颈，WGPU 的 CPS 中位被压在 206；装上后峰值直接飙到 301，说明 GPU 算力完全充足，只是被主线程卡住了下发。

> **后续修正（2026-11）**：结论 1 里的"2 秒通信税"有一部分其实是我们自己的
> decode 冗余拷贝，§12 去掉后同规模差距收窄；r=1000 的最新复测见 §11.7–§11.8。本表
> 保留为 0.1.5 时点的记录，不与后续数据混用。

---

## §11 性能天花板定位：r=1000 大规模实验（0.1.6 之后）

为区分"代码效率问题"与"硬件瓶颈"，把规模从 r=500 拉到 **r=1000（16129 chunks）**，
稀释冷启动与 Chunky 收尾 I/O 的占比，并开启 `debug_stage_timing`。

### 11.1 分阶段计时：CPU 侧已无空间

稳态（每 64 批一行）的分段耗时：

| 阶段 | 耗时 | 占比 |
|---|---|---|
| encode | 0.03 ms | 0.2% |
| submit | 0.32 ms | 2.3% |
| **readback** | **11–14 ms** | **~82%** |
| decode | 1.6 ms | ~11% |
| total | 14–17 ms/批 | 100% |

**结论**：`encode + submit = 0.35 ms`，CPU 侧的构建命令、bind group、FFM 调用
**已到极限**，Rust/FFM 架构本身没有问题。`readback` 是 GPU 真实计算 + 显存回读
时间，属于硬件天花板。

### 11.2 关键异常：GPU 一直没吃饱

- 平均批次只有 **3~10 个 chunk**（上限 `MAX_BATCH = 64`）
- `inflight` 只有 **0.9~1.7 / 4**，流水线远未打满
- `readback` 全程稳定在 8.6–19.5 ms，**没有随时间变长**——排除了热降频与资源泄漏

三者指向同一结论：**不是我们的队列堵了，是 C2ME 上游提交 chunk 的速率不够。**
r=1000 时 Chunky 跑到外围区域，结构生成 / 生物群系 / 光照这些仍在 CPU 上的
阶段变重，就喂不满 GPU。日志里 A 组的 CPS 从 305 掉到 212 再爬回 268，正是
这个上游波动的映射。

### 11.3 窗口参数已从两端夹死

`MAX_WAIT_MS`（攒批等待窗口）做了三档实测，r=1000、同种子、预热 5s：

| 窗口 | 总耗时 | 平均批次 | inflight | CPS 峰值 | Chunky Rate 均值 |
|---|---|---|---|---|---|
| **1 ms** | 1m05s | 3.4 | 1.3–1.7 | **346.8** | 239.3 |
| **5 ms**（保留） | 1m04s | 5.0 | 0.9–1.3 | 325 | **248.1** |
| 15 ms | 15s(r=500) | 未变大 | — | — | ~250（r=500 口径） |

- **1ms**：流水线更饱满（inflight ↑）、瞬时 CPS 创新高，但批次碎到 3.4，
  每批固定开销吃掉了收益，总耗时反而 +1s。
- **15ms**：批次根本没变大（证明瓶颈不在窗口），只增加延迟。
- **5ms** 是两端之间的最优点。

**结论：调窗口参数只是在"首块延迟"与"批次摊销"之间来回移动，动不了天花板。**

### 11.4 与官方 OpenCL 的同规模对比（r=1000）

| 指标 | WGPU | OpenCL | 差异 |
|---|---|---|---|
| 总耗时 | 1m04s | 1m02s | +3.2% |
| Chunky Rate 均值 | 248.1 | 271.2 | -8.5% |
| Chunky Rate 中位 | 239.7 | 276.7 | -13.4% |
| 众数区间 | 220–239 | 260–279 | -40 |
| Rate 标准差 | 39.7 | 24.8 | 波动大 60% |

r=500 时是 **15 s vs 13 s**（约 13%）；r=1000 时总耗时差距缩到 3.2%
（1m04s vs 1m02s），但稳态 CPS 仍差 8.5%。
**这 8% 是架构性的**：JNI 原生指针 vs FFM、4x4 空间批处理 vs 散装批次、
Biome 也在 GPU 上 vs 仍在 CPU。要追平需要碰 C2ME 状态机——**当时**这条被划为
兼容性红线，故判为不做。（2026-11 该红线已解除并已落地：4x4 对齐批次由
0.1.7-alpha.2 的 `chunk_batch_size` 实现，见 §13.6 / §13.7；Biome 仍未上 GPU。）

> 本表与 §11.7–§11.8 处于不同电源状态（`low-power` vs `performance`）、不同构建，
> 绝对耗时不可互比，只作历史对照。

### 11.5 本轮证伪的优化尝试（全部记录，避免重试）

| 尝试 | 结果 |
|---|---|
| 调色板 Unsafe 直写 + `calculateCounts()` | 中性（Rate 298.4 vs 299.3）。根因：`writeback` 仅 1.4ms/chunk 本就不是瓶颈，而 `calculateCounts()` 要对含大量空气的 4096 格全扫，把省下的增量记账吃回去了。 |
| `MAX_WAIT_MS` 5→15ms | 负优化：批次未变大，只增加延迟。 |
| `MAX_WAIT_MS` 5→1ms | 净持平：流水线更满但批次更碎，总耗时 +1s。 |
| 攒批空间排序（伪 4x4） | 分析判定无效：`sort` 只改顺序不改集合，且 shader 对每个 chunk 独立计算，无跨 chunk 数据复用。 |

**通用教训**：优化前先测量瓶颈。本轮所有失败尝试都在打"批次太小、固定开销太贵"
这个假设，但分阶段数据证明真正的瓶颈是 **C2ME 的提交速率**，不在我们这一侧。
判断依据必须是实测拆分（`encode/submit/readback/decode`），不是"代码看着低效"。

### 11.6 剩余的理论空间

- ~~**decode 占 11%（1.6ms）**~~ **已在 §12 解决，降到 0.47ms**。
- **poll 轮询下沉到 Rust**：减少每批的 FFM 调用次数，属于结构性改动，收益未知。
- **真正的 4x4 批次选择**：需要改 C2ME 状态机。**当时**因兼容性红线判为不做；
  2026-11 红线解除后已实现为 0.1.7 的 `chunk_batch_size` 对齐批次（§13.6），
  r=500 同种子下 `4×4` 为 13 s / CPS 298（§13.7）。剩余空间在"批内共享采样
  上下文"（阶段 3，未做）。

### 11.7 2026-11-10 复测（0.1.7-alpha.5，performance profile）

机器重启后 `platform_profile` 从 `low-power` 变为 `performance`，绝对耗时与
§11.4 不可直接比较；这一轮只换加速器 jar（A = 官方 OpenCL 0.4.0-alpha.0.29，
B = 本模组 0.1.7-alpha.5），同实例、同种子 `-836645977851536298`、同中心与
r=1000，每次全新世界 + 完整重启，A/B 交错并冷却 60 s。GPU 占用取自
`/sys/class/drm/card1/device/gpu_busy_percent`（含渲染线程）。

260 fps（默认设置）：

| 侧 | 总耗时 | Chunky CPS 中位 | GPU busy 中位 |
|---|---|---|---|
| A OpenCL | 59 s / 59 s | 292 / 296 | 97% / 98% |
| B WGPU | 59 s / 57 s | 289 / 302 | 85% / 86% |

30 fps（把渲染从 GPU 上摘掉）：

| 侧 | 总耗时 | Chunky CPS 中位 | GPU busy 中位 |
|---|---|---|---|
| A OpenCL | 42 s / 42 s | 408 / 406 | 76% |
| B WGPU | 55 s / 56 s | 309 / 302 | 54% / 53% |

- **260 fps 打平不是机制相同**：A 的 GPU 被渲染 + 计算占满（97–98%），B 只用到
  85–86%，说明 fp32 噪声阶段本身有余力。
- **放开 GPU 后差距反而出现**：A 降到 42 s，B 只到 55–56 s，且 B 的 GPU 近一半
  时间空闲。r=1000 时我们的天花板在 CPU 侧——批次提交 / readback / 写回
  （`writeback-timing` 实测 1.9–2.6 ms/chunk），不是噪声 kernel。
- 另一个电源状态（`low-power`，重启前）各 5 轮：A 74/80/94/86/90 s，
  B 82/89/83/82/84 s；均值无差异（Welch p≈0.84），但 A 的方差约为 B 的
  **7.4 倍**（CV 9.4% vs 3.5%），与"B 的 GPU 有余力、对机器状态不敏感"一致。
- 完整数据、污染批次说明与复现脚本记录在本地未跟踪的报告
  `.agents/reports/2026-11-10-ocl-vs-wgpu-r1000.md`（不随仓库发布）。

> 方法论：换 jar 的 A/B 必须断言 `mods/` 里**恰好一个**加速器，并在日志里
> 校验（A 出现 `Initializing OpenCL`；B 出现 `GPU 加速已启用` 且不出现前者）。
> 本轮有三次 B 组因为 jar 改名漏删而同时装了两套，见教训 #51。

### 11.8 2026-11-11 种子 × 视距正交 A/B（r=1000）

为回答"视距 / 种子到底影响多大"，用 2 种子 × 2 视距 × 3 重复、共 12 轮做了
正交对照。条件：同实例（AMD Renoir iGPU / RADV）、同空模板、`chunk_batch_size=4`、
Vulkan、260 fps、`simulationDistance=5`、窗口固定 854×480；只替换 `level.dat`
里 `WorldGenSettings.seed` 的 8 字节（已校验其余字节完全一致）。每轮删世界、
完整重启、`/chunky radius 1000` 跑满 16129 区块，并断言 `mods/` 里恰好一个
加速器。种子 S1 = `-836645977851536298`（与 §11.7 相同），S2 =
`7179558554521327105`；视距 6 / 12。四个条件按轮次交替顺序执行。

| 条件 | 总耗时 s | 全程 CPS 中位 | 最后 15 s CPS |
|---|---|---|---|
| S1 / vd12 | 58 / 56–59 | 296.5 | 263.6 / 255–268 |
| S1 / vd6 | 59 / 59–63 | 275.3 | 272.4 / 259–281 |
| S2 / vd12 | 56 / 55–59 | 288.1 | 286.9 / 268–289 |
| S2 / vd6 | 60 / 58–65 | 268.6 | 275.5 / 257–287 |

按 Chunky 进度分段（3 次重复的中位 CPS）：

| 条件 | 5–20% | 20–35% | 35–50% | 50–65% | 65–80% | 80–95% |
|---|---|---|---|---|---|---|
| S1 / vd12 | 262.6 | 291.2 | 302.2 | 299.2 | 300.2 | 263.6 |
| S1 / vd6 | 211.1 | 248.9 | 267.5 | 294.2 | 296.6 | 272.4 |
| S2 / vd12 | 219.8 | 258.3 | 287.6 | 291.0 | 292.8 | 284.9 |
| S2 / vd6 | 184.9 | 223.1 | 249.8 | 274.9 | 275.1 | 277.4 |

- **视距有稳定影响，但它是启动段偏差，不是稳态算力差**：vd12 在每一轮都不慢于
  同种子的 vd6，中位总耗时少 1 s（S1）到 4 s（S2），即 2–7%；35–50% 进度段高
  13–15%，50–80% 只高 1–6%，最后 15 s 方向翻转（S1 上 vd12 反而低 3%）。早期
  （5–20%）差距最大（19–24%），因为 vd12 进世界时预生成约 25×25 出生点区块，
  vd6 只有约 13×13，Chunky 起步就把这部分已生成区块快速计入进度。
- **种子在 r=1000 的稳态影响很小**：同视距下 S1 与 S2 全程 CPS 只差 2.4–2.8%
  （方向一致，S1 略快），小于组内重复波动（pooled SD 3.3–4.0%）。教训 #40
  里"18 s vs 23 s、CPS 188"的剧烈差异无法在 r=1000 稳态复现，那组旧数据很可能
  混入了视距、启动段或版本差异。只有两个种子，不足以断言"种子永远不重要"，
  但不应再用它解释两位数百分比的差距。
- **未验证的假设**：低视距下区块更早被卸载/写盘，保存 I/O 与生成竞争，才使
  后段反而变慢。要确认需要 I/O 采样或 `forceLoadExistingChunks` 对照，本轮
  没有做，不要当成结论引用。
- **方法论**：跨视距的 `Total time` 不可直接比较（见教训 #45）；A/B 要么固定
  视距，要么只比任务中后段的稳态吞吐。同一批 12 轮内重复间 CV 约 2–5%，且
  末轮整体偏慢（顺序漂移），所以只做组内相对比较。
- 完整 12 轮数据、GPU 占用采样与复现脚本记录在本地未跟踪的报告
  `.agents/reports/2026-11-11-seed-vd-ab.md`（不随仓库发布）。

---

## §12 readback 路径去冗余拷贝（0.1.6 之后）

### 12.1 问题：同一份数据被拷了六遍

`finish_batch` 到 Java 拿到的 blob 之间，密度数据和方块数据各自经历了一串
完全冗余的转换：

密度（每批 ~1 MiB）：
```
staging bytes
  → Vec<f32>                      (read_results 解码)
    → 每 chunk 一个 ResultBlob     (from_f32_data: 拷贝 + CRC32)
      → decode_blob_f32s()         (把刚序列化的 blob 又解析回 Vec<f32>)
        → all_f32s.extend          (拼接)
          → from_f32_data          (再序列化 + 再算 CRC)
```

方块（每批 ~24 MiB）：
```
staging bytes
  → Vec<u32>                      (逐 u32 转换)
    → Vec<u8>                     (flat_map to_le_bytes)
      → ResultBlob::from_bytes    (再拷一次)
```

关键在于：**staging 里躺着的字节，就是 Java 要的 payload 布局**
（chunk-major、通道连续、f32 LE）。中间的编码-解码往返没有任何语义变化，
纯粹是历史包袱——`ExecutedBatch.results` 的 per-chunk blob map 是旧架构
（Java 逐 chunk 消费）的产物，现在 Java 只取一个聚合 blob。

### 12.2 改法

- `ResultBlob::from_raw_payload(payload, offsets, chunk_count)`：直接用现成
  字节拼 blob（头 + offsets + payload + CRC），不做类型转换。
- `ExecutedBatch` 增加 `raw_density: Vec<u8>` / `raw_block: Vec<u8>`，
  由 `finish_batch` 直接从 staging 切片得到。
- `finish_batch(prepared, raw)` 按调用方分流：
  - `raw = true`：生产路径（`submit_async` → C API 回调），返回原始字节。
  - `raw = false`：`request_chunks` 同步路径，保留 per-chunk blob map，
    供 89 个离线测试读取。
- `SubmitBatch` 消息带上 `raw` 标志，随 `in_flight` 队列一起传递。
- 删掉 `decode_blob_f32s`（不再需要）。

**踩到的坑**：最初只给 `execute_batch`（测试用）传 `raw = false`，
但测试实际走的是 `request_chunks`，它同样经过 `run_loop` 的生产分支，
导致 10 个测试因 `results` 为空而失败。修正方式是把 `raw` 提升为
`SubmitBatch` 消息的一部分，由两个入口各自声明——**这提醒了一件事：
"synchronous" 和 "test-only" 不是同一回事，判断调用路径要看它最终走哪个循环。**

### 12.3 实测

Chunky r=1000（16129 chunks），同种子、同预热：

| 阶段 | §11（改前） | §12（改后） | 变化 |
|---|---|---|---|
| encode | 0.03 ms | 0.03 ms | — |
| submit | 0.32 ms | 0.31 ms | — |
| readback | 11–14 ms | 9.5–11.8 ms | -1~2 ms |
| **decode** | **1.6 ms** | **0.45–0.50 ms** | **-72%** |
| **total** | 14–17 ms | **10.6–13.0 ms** | **-23%** |

端到端：总耗时 **1m04s → 1m02s**，CPS 峰值 325 → 342.6。

**readback 也跟着降了**：executor 线程不再被 decode 占着，映射检查更及时。

### 12.4 对 §11 结论的影响

§11 判定真正的瓶颈是 C2ME 提交速率（`inflight` 长期只有 0.9–1.1 / 4），
这一点没变。但 **decode 从 1.6ms 降到 0.47ms 后，WGPU 的总耗时追平了
官方 OpenCL（都是 1m02s）**——说明之前那 2 秒差距里，有一部分其实是
我们自己制造的拷贝开销，而不是架构差距。

**教训**：在断言"差距是架构性的"之前，先确认自己这边没有低级浪费。
`total` 里的每一项都要问一句"这份数据真的需要换个形状吗"。

---

## §13 C2ME `NewChunkStatus` 接入（0.1.7-alpha.1，阶段 1）

### 13.1 为什么要接

0.1.6 之前，GPU 加速的入口是 `MixinChunkGenerator.populateNoise`：vanilla
`NOISE` 状态跑到一半，我们拦下密度计算，再 `cir.setReturnValue`。这条路能跑，
但天花板受 C2ME 状态机限制（§11：`inflight` 长期 0.9–1.1 / 4）。官方 OpenCL
模块走得更远：它注册自己的 `NewChunkStatus`，直接持有批次、做 4x4 空间批处理。
要拿到同样的收益，必须先证明我们能安全地进入这条状态机。

### 13.2 阶段 1 只做透明委托

`WgpuNoiseStatus` 插在 `BIOMES` 与 `NOISE` 之间，但**不自己生成地形**：
`upgradeToThis` / `getDependencies` 原样转发给 C2ME 的
`VanillaWorldGenerationDelegate(NOISE)`。于是 GPU 路径保持
`populateNoise` → `WgpuBatchCoordinator` 不变，这一轮只验证三件事：

1. `NewStatusHook` 注入点确实在 vanilla NOISE 之前；
2. 自定义 ordinal 与依赖声明能被调度器接受；
3. 现有地形和性能没有被破坏。

**为什么不直接提交**：vanilla 生成步骤手里有 `Blender`、`StructureAccessor`
和已构建的 `ChunkNoiseSampler`，aquifer/beardifier staging 依赖它们。在状态里
自建一套等于写第二份世界生成，错了很难归因。阶段 1 把"钩子错了"和"生成错了"
分开，阶段 2 再接管数据。

### 13.3 实现要点

- `MixinNewStatusHook` 注入 `beforeVanillaStatusRegister` 的 `HEAD`；
  `vanillaStatus == ChunkStatus.NOISE` 时执行
  `statuses.add(new WgpuNoiseStatus(statuses.size()))`。`remap = false`——
  目标是 C2ME 的类，不是 Minecraft。
- 顺序由插入时机决定：新状态拿 `statuses.size()`，随后注册的 vanilla
  delegate 顺延一位，得到 `... BIOMES → WgpuNoiseStatus → vanilla NOISE ...`。
- `vanillaNoise` **惰性解析**：hook 执行时 `NewChunkStatus.<clinit>` 还没跑完，
  `VANILLA_WORLDGEN_PIPELINE[NOISE]` 是 null。字段声明为 `volatile`，首次
  `upgradeToThis` / `getDependencies` 时用 `fromVanillaStatus(NOISE)` 取。
- 兜底：若解析不到（理论上不可能），记录 warn 并返回
  `Completable.complete()` / `EMPTY_DEPENDENCIES`，让调度器继续走真正的
  vanilla delegate，而不是死锁。
- `ANNOUNCED` 保证 "active" 日志只打一次，避免每个 chunk 刷屏。
- 升级/降级钩子（`postUpgradeToThis` / `preDowngradeFromThis` /
  `downgradeFromThis`）返回 `Completable.complete()`：本状态不持有资源。

### 13.4 实机验证（2026-11-07）

环境：1.21.1 Fabric，RADV RENOIR，种子 `-836645977851536298`，Chunky r=500。
（种子仅用于本次记录；跨实验比较仍按教训 #45 标注可比性。）

| 检查项 | 结果 |
|---|---|
| hook 注册 | `[c2me-wgpu-status] active: ordinal=6`，全程仅 1 次 |
| r=500 端到端 | 4225 chunks / **15 s**，与现有基线持平 |
| CPS | 稳态 260–299，峰值 298.5 |
| 崩溃 / 异常 | 无（仅 flite、离线认证等环境噪声） |
| 地形 | 正常，无空洞 / 错位切面 |
| 退出 | 全部维度保存成功 |

**结论**：持平而非提升是预期结果——这一阶段 GPU 工作与 0.1.6 完全相同，
只是多包了一层状态。它证明的是扩展点、顺序和依赖声明正确，不能证明性能。

### 13.5 阶段 2 为什么没有接管生成

原计划在阶段 2 里自己构建 `ChunkNoiseSampler` / `Blender` / `StructureAccessor`，
但那样等于写第二份世界生成器：aquifer、beardifier、写回半径都要逐项对齐，
出错时很难区分是"调度错了"还是"采样错了"。因此阶段 2 改为**只收敛调度**：

- 生成仍然由 C2ME 的 `VanillaWorldGenerationDelegate` 完成；
- GPU 仍然由 `populateNoise` → `WgpuBatchCoordinator` 服务；
- 批次只改变"整组什么时候算就绪、由谁触发"。

这样任何批次逻辑的 bug 都只可能表现为**慢**或**回退到逐块生成**，而不是错地形。
`MixinChunkGenerator` 的写回路径继续保留，不删除。

### 13.6 对齐批次设计（`chunk_batch_size`）

配置字段 `chunk_batch_size`：`1`（默认，透明委托）/ `2` / `4` / `8`，必须是
2 的幂；非法值在 `Config#clamp` 里回落为 1。状态注册时读取一次（每 JVM 一次），
改动必须完全重启 Minecraft，只退出存档再进入无效。实机结果里 **4×4 最优**
（§13.7），作为推荐值；默认保持 1，保证未验证场景行为不变。ModMenu 设置界面
提供 `1 / 2 / 4 / 8` 的循环控件（0.1.7-alpha.4 起），不再需要手改 JSON；界面
底部单独提示"需完全重启"。

**分组与对齐**。以 `mask = N-1` 判定最小角（anchor）：

```
isAnchor(x, z)  =  (x & mask) == 0 && (z & mask) == 0
anchorOf(x, z)  =  (x & ~mask, z & ~mask)
```

一个组覆盖 `[anchor.x, anchor.x+N-1] × [anchor.z, anchor.z+N-1]`。负坐标按
二进制补码同样成立（例如 `-1 & 3 = 3`，`-4 & 3 = 0`）。

**anchor 的依赖并集**。原版 `NOISE` 生成步骤的 `directDependencies()` 是
`[BIOMES(0..1), STRUCTURE_STARTS(2..8)]`（半径 8）。组的并集是每个成员各扩
8 格后的方形并集，即相对 anchor 的 `[-8, N+7]²`，大小
`(N+16)²-1`（N=2 → 323，N=4 → 399，N=8 → 575）。对偏移 `(x,z)`：

```
dx = max(0, -x, x-(N-1))
dz = max(0, -z, z-(N-1))
d  = max(dx, dz)          // 到组方阵的 Chebyshev 距离
status = fromVanillaStatus(directDependencies.get(d))
```

`d=0..1` 得到 `BIOMES`，`d=2..8` 得到 `STRUCTURE_STARTS`，与单块完全一致
（N=1 时逐项等于原版 `NOISE` 的依赖集）。

**非 anchor**。`getDependencies` 只声明一条：`anchor @ 本状态`。它的
`upgradeToThis` 直接 `Completable.complete()`，因为 anchor 完成时已经替整组
写好了 `ChunkState`。

**anchor 的升级**。按组遍历成员：

1. 自己用当前 `ctx` 调 `vanilla.upgradeToThis(ctx, cancellable)`；
2. 其余成员通过 `ctx.theChunkSystem().getHolder(pos)` 取 `ItemHolder`，
   用 `new ChunkLoadingContext(memberHolder, ctx.tacs(), ctx.schedulingManager(),
   ctx.theChunkSystem(), ctx.dependencies())` 构造成员上下文，再调同一个
   vanilla delegate；
3. 用 `Completable.merge` 同时订阅。原版 `NOISE` 的 `blockStateWriteRadius`
   是 0，所以成员的锁集互不重叠，C2ME 调度器可以并行执行；GPU coordinator
   因此能在同一个 5 ms 窗口里收到整组请求（N=8 正好等于 `MAX_BATCH=64`）。

成员上下文里的 `dependencies()` 只是占位：已确认
`VanillaWorldGenerationDelegate.upgradeToThis` 不读取该字段，区域与锁都由
它自己按 `theChunkSystem` / `schedulingManager` 重建。

**跳过与兜底**（这是正确性的关键）：

- 任何成员只要 `ChunkState.reachedStatus() >= NOISE` 就跳过；
- 取不到 holder / `ChunkState` 的成员跳过并 warn；
- anchor 自己已经 `>= NOISE` 时整组直接完成；
- 被跳过的成员**不会漏生成**：本状态之后紧跟的原版 `NOISE` delegate 会检查
  自己的 `ChunkState`，未到的照常逐块生成。也就是说本状态是纯优化层，
  原版 delegate 仍是最终保证。
- 已经 `>= NOISE` 的 holder 返回 `EMPTY_DEPENDENCIES`，不再无谓等待 anchor。

**版本敏感点**。本实现绑定的是 C2ME 1.21.1 补丁线
`ChunkLoadingContext.holder()/tacs()/schedulingManager()/theChunkSystem()/dependencies()`
五组件 record + FlowSched `ItemHolder.getItem()/getKey()` 的字节码形状。

查证记录（2026-11，全部以**实际发布的 jar** 为准，不看分支名）：

| C2ME 构建 | `ChunkLoadingContext` 第 4 组件 | 结论 |
|---|---|---|
| 1.21.1 `0.4.0-alpha.0.29`（实机基线） | `TheChunkSystem theChunkSystem` | 可用 |
| 1.21.1 `0.4.0-alpha.0.30`（当前最新） | `TheChunkSystem theChunkSystem` | 形状兼容（`javap` + 离线检查；未实机） |
| `dev/1.21.11` 线 | `BoundedRegionArray<AbstractChunkHolder> chunks` | 不可用 |

即：**改名发生在 1.21.11 线，不在 26.2 线**。之前"26.2 把组件改成 `chunks`"
的说法是误读分支名；26.1/26.2/26.3 的 `dev/*` 头仍叫 `theChunkSystem`。1.21.1
补丁线到 `0.4.0-alpha.0.30` 为止都没有改，但它是**滚动 devbuild**，下一个小
版本随时可能跟进改名。

因此加了两层保护：

1. **运行期形状检查**（`C2meCompat.verify()`，注册前调用）。用
   `Class.forName(name, false, ...)`（**不初始化**，因为这段代码正跑在
   `NewChunkStatus.<clinit>` 里）反射核对上面五个组件名/类型、规范构造器、
   `TheChunkSystem.getHolder(Object)`、`ItemHolder.getItem()/getKey()`、
   `ChunkState.reachedStatus()`、`NewChunkStatus(int, ChunkStatus)`、
   `relativeToAbsoluteDependencies`、`fromVanillaStatus`。任何一项不符：
   `[c2me-wgpu-status]` 打一条 ERROR，**不注册** `WgpuNoiseStatus`，
   `populateNoise` 的旧 GPU 路径继续工作——降级，不是崩溃。
2. **`fabric.mod.json` 上界** `>=0.4.0-alpha.0.29 <0.5.0`。上界故意放松到
   0.5.0：`0.4.0-alpha.0.N` 是滚动构建号，卡死具体 N 会让每个 C2ME devbuild
   都要跟一个本模组发版；具体形状差异交给第 1 层处理。0.5.0 视为可能重写
   API 契约的新线，先由加载器给出清晰的不兼容提示（也覆盖"`NewStatusHook`
   本身被移除、mixin 无法应用"这种运行期检查来不及处理的情况）。
   放宽步骤：`javap` 核对新 jar 的 `C2meCompat` 所列签名 → 实机跑一轮
   r=500 → 改 `fabric.mod.json` 并更新本表。

**离线验证**（2026-11-10，不启动 Minecraft）：用真实的 1.21.1 中间名 MC jar +
C2ME `0.4.0-alpha.0.29` / `0.4.0-alpha.0.30` 的嵌套 jar 运行
`C2meCompat.mismatch()`，两者都返回 `null`；再用一个合成 record（第 4 组件改名
`chunks`）遮挡 `ChunkLoadingContext`，返回
`ChunkLoadingContext component 3 is 'chunks', expected 'theChunkSystem'`。
这一步同时确认了 `Class.forName(..., false, ...)` 的必要性：用会触发初始化的
`Class.forName(name)` 版本在同一环境直接抛 `ExceptionInInitializerError`，
因为 `ItemHolder` / `TheChunkSystem` 的 `<clinit>` 不是为在
`NewChunkStatus.<clinit>` 内部被调用而设计的。

### 13.7 阶段 2 实机验证结果（2026-11-08）

测试条件：同一个世界、种子 `-836645977851536298`，每档都**删除并重建世界**
（保证真正重新生成），Chunky r=500（4225 chunks）。四轮中心分别为
`-6.5,-3.5` / `6.5,-1.5` / `-3.5,-7.5` / `1.5,-5.5`——最大相差 13 格
（小于一个区块），按约定视为同一区域。

| batch | 启动日志 | 总耗时 | 稳态 CPS 中位 | CPS 峰值 | 每批平均 chunk |
|---|---|---|---|---|---|
| 1 | `batch_size=1` | 18 s | 266 | 306 | ~4.1 |
| 2 | `batch_size=2 (... 323 dependencies ...)` | 14 s | 247 | 309 | 7.8 |
| **4** | `batch_size=4 (... 399 dependencies ...)` | **13 s** | **298** | 330 | 15.3 |
| 8 | `batch_size=8 (... 575 dependencies ...)` | 15 s | 260 | 372 | 28.8 |

- **地形**：四档均由用户目视确认正常（无空洞、断层、错位切面）。未做逐方块
  dump 对比；批次逻辑只收敛调度，生成仍走原版 delegate，这是目视检查足够
  的原因。
- **稳定性**：四轮均无 `batch member ... has no holder/ChunkState` 警告、
  无 `NoSuchMethodError` / `ClassCastException` / 调度器异常，区块持续完成，
  三存档正常保存。仅有的 ERROR 是认证 / Realms / 旁白环境噪声。
- **批次聚合确实生效**：`c2me-wgpu.log` 的每批平均 chunk 数随 N 单调上升
  （4.1 → 7.8 → 15.3 → 28.8），说明 anchor 把整组一起交给了 GPU。
- **4×4 最优**：13 s / CPS 298。8×8 反而回落到 15 s / CPS 260，虽然峰值最高
  （372）：`c2me-wgpu.log` 显示 8×8 的单批耗时升到 58–93 ms（readback 占大头），
  而每批只有 ~29 个 chunk，已经接近 `MAX_BATCH=64` 的收益上限，属于"批太大、
  攒批等待和单批延迟吃掉收益"。
- batch=1 的 18 s 明显高于历史 15 s，但历史数据是不同种子；本表四轮同种子，
  组内可比（教训 #45）。因此结论只取组内相对值：**4 > 2 ≈ 8 > 1**。

**结论**：`chunk_batch_size` 保留默认 `1`（不改变已验证行为），把 **4** 作为
推荐的手动值写进 §13.6。0.1.7-alpha.4 起不再讨论"改默认值"：默认保持 1 以保证
未验证场景行为不变，把切换权交给 ModMenu 的循环控件（§13.6）。

**操作注意**：`chunk_batch_size` 在 `NewChunkStatus.<clinit>` 时读取一次，
每 JVM 只执行一次，所以改值必须**完全重启 Minecraft**，只退出存档再进入
不会生效。该字段也不会自动写进旧配置文件（GSON 缺字段时用默认值），需手动
添加；值非法时 `Config#clamp` 回落为 1。

回退：把值改回 `1` 并重启即可，不需要换 jar。

---

## §14 表面高度估算（Estimate Surface Height）评估（2026-11-08）

> 结论先行：**已实测，结案（负结果）**。一轮 r=500 只有 696 次 `getHeight`、
> 累计 1.58 s（≈12% 墙钟，且约一半与 GPU 在途重叠），重复率 6%；纯 CPU 缓存
> 省不到 1%，GPU 预计算要多跑一遍噪声，都不划算。测量方法与数据见 §14.2 /
> §14.5。

### 14.1 原版 `getHeight` 的成本模型

依据 1.21.1 named jar 的 `NoiseChunkGenerator#sampleHeightmap` 字节码（行为
事实，未参考任何非 MIT 实现）：

- 每次调用新建一个 `ChunkNoiseSampler`，横向 cell 数固定为 **1**
  （构造器第一个参数 `iconst_1`），垂直方向按 `GenerationShapeConfig` trim 后
  的高度切 cell。主世界 384 高、cell 8 → 48 层 → 密度网格点
  `2 × 49 × 2 = 196`；满区块是 `5 × 49 × 5 = 1225`。即**一次高度查询 ≈ 1/6
  个区块的噪声求值**。
- 传入的是 `Beardifier.INSTANCE`（空 beardifier）和
  `Blender.getNoBlending()`：**高度与 blending 无关，也看不到已放置的结构**，
  只由种子 + `NoiseConfig` 决定。这一点后面 §14.5 的缓存方案直接受益。
- `AquiferSampler.FluidLevelSampler` 与区块生成共用同一个工厂，含水层参与
  判定；但 **surface rules 不参与**（`buildSurface` 在更晚的状态）。
- 求值方式是从世界顶部**逐格**向下 `sampleBlockState()`，命中谓词即
  `stopInterpolation()` 返回；`WORLD_SURFACE_WG` 的谓词是
  `Heightmap.NOT_AIR`——水、树叶、雪都算"非空气"。
- `getHeightOnGround` / `getHeightInGround` 只是 `getHeight` 的 ±1 包装
  （一个 `iconst_1; isub`）。

按 §11 的纯 CPU 端到端（r=500 / 4225 chunks / 27 s ≈ 6.4 ms per chunk）反推，
密度部分约 3–4 ms/chunk，则一次 `getHeight` 粗估 **0.3–0.8 ms**
（196/1225 的网格点，加最多 384 次逐格插值与含水层判定）。
**实测是 2.2 ms（§14.2），比这个模型高 3–7 倍**：每次调用重建
`ChunkNoiseSampler` / 插值器的固定开销主导，网格点数量反而不是主要成本。

### 14.2 谁在调用，以及实测结果（2026-11-08）

| 调用点 | 每次调用数 | 备注 |
|---|---|---|
| `Structure.getCornerHeights`（`getAverageCornerHeights` / `getMinCornerHeight`） | 4 × `getHeightInGround` | 大多数结构选址的公共路径 |
| `StructurePoolBasedGenerator` / `StructurePoolGenerator` | 2 × `getHeightOnGround` / 锚点 | 拼图结构（村庄、前哨站、试炼密室、远古城市…） |
| `MineshaftStructure#getStructurePosition` | 1 × `getHeight`，**仅 MESA 分支** | 见下方勘误 |
| `RuinedPortalStructure` | 依地形适应性分支 | 少数 |

**勘误（本节初稿错了）**：初稿按"矿井 `spacing=1` → 每个区块 1 次"估出
≈4700 次/轮。字节码显示 `MineshaftStructure` 只在 `type == MESA` 分支里调
`getHeight`，普通矿井走 `StructurePiecesCollector.shiftInto(seaLevel, minY, rng, 10)`
——恶地以外的矿井**一次都不查**。这个错误只有实测能纠正。

另外两条仍然是事实，但不改变量级：

1. 高度采样发生在生物群系过滤之前（`getStructurePosition(ctx).filter(isBiomeValid)`，
   `canGenerate` 也只查 placement），生物群系不匹配的候选照样付采样钱。
2. 主世界里真正在付钱的是稀疏集合（村庄 / 前哨站 / 试炼密室 / 远古城市 /
   废弃传送门…），每个结构起始约 4 次角点。

**实测**（`debug_height_probe`，同种子 `-836645977851536298`，r=500 / 4225
chunks，`chunk_batch_size=4`）：

| 指标 | 实测 | 初稿估计 |
|---|---|---|
| 调用次数 | **696**（0.165 次/区块） | 4700（1.1 次/区块） |
| 类型分布 | `WORLD_SURFACE_WG` 577 / `OCEAN_FLOOR_WG` 119 | — |
| 平均单次耗时 | **2.2 ms**（max 46.75 ms，首次调用） | 0.3–0.8 ms |
| 累计 | **1.58 s** | 1.4–3.8 s |
| 去重后重复率 | **6%** | — |
| 调用时 GPU 有批次在途 | **~50%** | — |

两个数字值得单独记：

- **单次 2.2 ms 是"1/6 个区块"模型的 3–7 倍**。196 个网格点本身只值 ~0.6 ms，
  多出来的是每次调用重建 `ChunkNoiseSampler` + 插值器 + 遍历 density 树图的
  固定开销——**固定成本主导，不是算术量主导**。这直接否掉了"按列懒算"的便宜
  路径：要么整区域预计算，要么不查。
- 调用量只有 0.165 次/区块。真正的成本来自每个结构起始的 4 次角点，矿井几乎
  不出钱；这也意味着"每区块缓存 16×16 高度"能命中的调用本来就很少。

（同轮 Chunky 总耗时 25 s、稳态 CPS 中位 118，明显差于同配置的 13 s / 298。
复测（探针关闭）**17 s / CPS 中位 238**，`c2me-wgpu.log` 显示首窗
4.8 ms/chunk、之后稳定在 2.8–3.2 ms/chunk —— 25 s 那次是流水线爬升期被拉长，
属环境波动，不是探针开销（探针每轮总计 <2 ms）。高度查询的累计值与墙钟无关，
但占比按 13 s 算才是保守口径，即 **1.58 s ≈ 12%**；按 17 s 算是 9%。）

### 14.3 为什么不能直接拿现有 GPU 结果反推

三个独立原因，任何一个都足以否掉"扫一遍 block_out 取顶部"的方案：

1. **时序**：`NOISE` 依赖半径 8 的 `STRUCTURE_STARTS`（§13.6 的依赖表）。
   结构选址发生时，目标列的 `NOISE` 还没跑——而这正是我们要回答的问题。
2. **Beardifier**：`sampleHeightmap` 用 `Beardifier.INSTANCE`（空），
   `populateNoise` 用真实 `StructureAccessor` 的 beardifier。拿区块结果反推会
   在结构附近产生 1–2 格偏差，直接影响结构 Y。
3. **语义**：`WORLD_SURFACE_WG` 是"密度 + 含水层"后的最高非空气方块，不含
   surface rules；只有能证明 block_out 与 vanilla `populateNoise` 逐块一致，
   才谈得上等价推导（目前只有目视验证，见 §13.7）。

所以"区块 NOISE 完成后缓存 16×16 高度"这条路命中率≈0：等缓存可用时，该区块
的结构选址早已发生。

### 14.4 真要做的话，最小可行形状

1. **GPU 新增高度入口**：输入列坐标数组，输出每列高度（4 B/列）。可复用现有
   router 求值，但语义必须逐项对齐 `sampleHeightmap`：trim 后的 shape、
   `Beardifier.INSTANCE`、同一 `FluidLevelSampler`、从顶向下取第一个非空气。
2. **必须提前算**：在结构选址之前完成，只能按区域做（例如 anchor 组的 N×N
   区块 + 一圈 margin），或按 `STRUCTURE_STARTS` 的波次推测性预取。
3. **必须攒批**：单列一次 dispatch 的 readback 延迟（0.3–1 ms）会吃光收益，
   需要复用 `WgpuBatchCoordinator` 的攒批机制。
4. **必须留 CPU 兜底**：miss 直接走原版 `getHeight`，正确性由 fallback 保证。
5. **必须重用**：预计算的列数据要能被随后同区块的 `NOISE` 复用，否则等于多跑
   一遍噪声（整区域预计算 ≈ 一次完整 NOISE，量级 +10–20 s），净负。

第 5 条是关键：**不重用结果的区域预计算，成本高于它省下的 CPU。**

### 14.5 建议的执行顺序

1. **先测量，不写 shader**。✅ 已实现（0.1.7-alpha.3）：
   - 开关 `debug_height_probe`（默认 `false`；`HeightProbe` 类加载时读一次，
     改值需重启游戏）。
   - `MixinNoiseChunkGeneratorHeight` 在 `getHeight` 的 `HEAD` / `RETURN` 打点，
     角点高度、矿井、拼图锚点三条路径最终都走 `getHeight`，因此都能覆盖。
   - 统计：调用次数、累计/平均/最大 nanos、去重列数与重复率、按
     `Heightmap.Type` 的分布、调用线程 Top-4，以及**调用时 GPU 是否仍有批次
     在途**（`WgpuBatchCoordinator.activeRequests()`）。
   - 每 256 次调用或 2 s 输出一行，logger 名 `c2me-wgpu-height`：
     ```
     [c2me-wgpu-height] calls=N (+窗口) unique=M dup=X% avg=..ms max=..ms
                        total=..s gpuBusy=..% types=[WORLD_SURFACE_WG:..] threads=[..]
     ```
   - 判读：`total` 是**上限**；`gpuBusy` 高说明这些 CPU 时间本来就与 GPU
     重叠，墙钟收益远小于 `total`；`dup` 决定纯 CPU 记忆化能省多少。
   - 判据不变：累计 < 3% 墙钟 → 结案；>5% 才往下走。
2. **若确实可观，先做便宜的 CPU 记忆化**：`(x, z, type) → height`，每维度一个
   `Long2IntMap`（键为打包的 `x<<32|z` 加 type）。
   - 精确性天然成立：§14.1 已确认高度与 blending 无关、只由种子决定，同种子下
     永不失效。
   - 命中率预计不高（结构角点基本不重合），但实现零风险，可作为第一刀。
3. **只有 (1) 显示 >5% 且 (2) 收益不够时**，才做 §14.4 的 GPU 预计算，并且必须
   先设计结果重用。

**测量结果（2026-11-08，0.1.7-alpha.3）**：一轮 696 次 / **1.58 s**（按 13 s 口径
约 12% 墙钟），重复率仅 6%，约一半调用与在途 GPU 批次重叠。据此：

- 纯 CPU 记忆化最多省 `6% × 1.58 s ≈ 0.1 s`（<1%），不值得加缓存；
- GPU 懒算不可行：列坐标在结构选址内部才确定，单列 dispatch 的 readback 延迟
  （0.3–1 ms）与 2.2 ms 的 CPU 单次成本同量级，还要额外承担精确性风险；
- GPU 区域预计算 = 每个区块多跑一遍噪声（+10–20 s），而上限只有 1.58 s，
  回收不了成本。

**结论：结案（负结果）**。`debug_height_probe` 保留为诊断开关（默认 `false`），
以后视距 / 维度 / 结构集变化时可以一键复测，不需要再写代码。

### 14.6 `FindTopSurface` 的历史结论（保留）

顺带查清了 `FindTopSurface`：它是 1.18 之前的旧节点，**1.21.1 的 NoiseRouter
里根本不存在**，Java 侧只定义 `TAG_FIND_TOP_SURF` 却从不写入，真实 router
测试数据里也搜不到。它的 WGSL emitter 曾返回一个未定义的占位字符串（会生成
非法 shader），现已改为显式返回 `Err`，让该维度干净回退 CPU。反序列化路径
保留，仅为了兼容旧 blob 与单元测试。

---

## §15 附录：开发历程

> 以下为各里程碑历史，压缩自旧版文档，保留关键决策与教训。

### 15.1 里程碑索引（细节见 git history）

- **2a**：首个可跑管线。地形出现。修复：magic 字节序、instanceof、`dx_max` 重定义。
- **2b**：emitter memoization，WGSL 13 万行 → 5954 行；节点数值验证 ~1e-7。
- **2c**：批处理 + 网格对齐 MC 角点（4→8），方块匹配 94.68%→96.18%。
- **2d**：正确性验证框架；网格 Y cell 修正；EndIslands（simplex）1.19e-7；多维度守卫；磁盘 PipelineCache。教训：RAW_ROUTER 不可靠，要用 compiled router。
- **2e**：ch10/ch11 残差根因 = blob 数据源 + octave 顺序。教训：CPU 解释器一致 ≠ 对 vanilla 正确。
- **2f**：多维度 WorldContext；三个正确性 bug（octave 顺序 / simplex origin / EndIslands 环条件反转）。
- **2g**：NaN 语义兜底（远坐标）；测试判据改进。教训：平台未定义行为要显式兜底。
- **2h**：矿脉路线 3（GPU 角点粗筛 + MC 精算）。教训：CPU per-block 复用 MC 采样器 = 灾难。
- **2i**：前端通知 + 初始化失败安全回退。教训：`<clinit>` 抛异常 = 类永久损坏。
- **2j**：Android aarch64 交叉编译（cargo-ndk）。
- **2k**：内存泄漏修复（result blob 每 chunk 泄漏 + executor 每 batch 重建 buffer）。
- **2l**：Mali-G610 shader 规模崩溃 → 按通道拆 A/B/C 三片。教训：编译成本超线性，安全区 ~2100 行。
- **2m**：只算/只回读被消费通道（shader 6223→2740 行，回读 -87.5%）。教训：先数消费者。
- **2n**：群系气候 GPU 加速（design 2）。
- **异步调度 / 写回 / 最小闭环**：异步调度 A/B、写回天花板实测、最小闭环。核心结论见 §8.1–§8.3。
- **2o**：aquifer + Beardifier 全 GPU，端到端反超纯 CPU。见 §8.4。

### 15.2 已证伪的实验（存档）

#### 15.2.1 手机"高 U 低显"负优化

旧架构 CPU 写回下成立；§8.2 的测量之后架构改变。

#### 15.2.2 interpolated 边界 bug

已通过 Pass E 架构修复（见 §8.4）。

#### 15.2.3 JFR 热点：写回瓶颈在 CPU 逐方块

瓶颈在 aquifer / heightmap / vein。

#### 15.2.4 GC 对负优化贡献

实测排除。

#### 15.2.5 aquifer 逐帧归因

~65% 写回，一半控制流一半噪声。

#### 15.2.6 各种"取巧"思路

heightmap 一次性、aquifer 部分 GPU 化等，多数证伪。

**结论：够长的杠杆是"整体写回 GPU 化"（§8.2–§8.4 已实现）。**

### 15.3 日志架构与防死锁机制（0.1.5 新增）

- **FFM Upcall 日志回传**：Rust 侧不再直接向 `STDERR` 打印调试信息。所有的 `wgpu_debug!` 调用会通过 Panama FFM 的回调函数（`upcallStub`）直接传给 Java 端的 SLF4J `LOGGER`，完美合并进 `latest.log`，且格式统一。
- **黑匣子防崩溃日志**：Rust 侧独立保留了对 `c2me-wgpu.log` 的写入。每次写入后执行 `f.flush()`，并在 `setup_panic_hook` 中执行 `f.sync_all()`，确保即便发生 `SIGABRT`（如 `panic=abort` 或 Android 驱动崩溃），底层原因依然能落盘。
- **强制退出防死锁**：`pollStatus` 超时（原生 worker 无响应）时，主线程会永久卡在 C2ME 的 Future 上。此时 `WgpuBatchCoordinator` 调用 `LogManager.shutdown()` 刷出 Java 日志，然后 `Runtime.getRuntime().halt(1)` 强杀进程。批次级失败不走这条路径，见下节。
- **原生库解压策略**：默认将 `.so`/`.dll` 解压到 `<gameDir>/c2me-wgpu-natives/`，避开系统临时目录可能存在的 `noexec` 权限限制。若游戏目录不可写或 `dlopen` 失败，自动清理并回退到系统临时目录（`/tmp`）。同时，解压后自动追加可执行权限，确保 Android 等严格沙箱能正常加载。
- **FFI 边界护甲**：`c2me_wgpu_create_instance` 使用 `extern "C-unwind"`，并用 `std::panic::catch_unwind` 包裹。若 wgpu 底层在初始化时 panic（典型场景：强制指定 `WGPU_BACKEND=gl` 但 EGL 返回 `BadAccess`），可以安全地捕获并回退到 CPU。
- **⚠️ 跨平台崩溃行为差异（重要）**：
  - **桌面端**（Windows/Linux/macOS）：Rust 默认 `panic=unwind`，`catch_unwind` 能有效捕获 wgpu 的 panic，游戏**静默回退到 CPU**，不会闪退。
  - **Android 端**：交叉编译默认 `panic=abort`，`catch_unwind` 失效。wgpu 的 panic 会触发 `SIGABRT`，**JVM 会被瞬间强杀**，游戏直接闪退。此时唯一可依赖的是提前注册的 `setup_panic_hook`，它会在 `abort` 前执行 `sync_all()`，把 panic 堆栈强行写入黑匣子 `c2me-wgpu.log`。
  - **结论**：Android 用户如果设置了不兼容的 `WGPU_BACKEND` 环境变量（如 `gl`），游戏会闪退，但不会丢失崩溃原因。桌面端用户可以放心随意切换后端。

### 15.4 运行期错误处理的分层设计

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

### 15.5 关键文件

- `rust/src/codegen/wgsl_gen.rs`：WGSL 生成 + shard 规划 + BlockOut/beard/aquifer WGSL。
- `rust/src/scheduler/executor.rs`：batch 执行、bind group、buffer 管理。
- `rust/src/world/world.rs`：WorldContext、pipeline 编译、测试。
- `rust/src/world/{aquifer,beard}.rs`：aquifer 表解析 + Beardifier 复现。
- `rust/src/c_api.rs`：C API（含 `set_aquifer_table` / `set_beard_table`）。
- `src/main/java/.../mixin/MixinChunkGenerator.java`：写回 + stage + dump。
- `src/main/java/.../chunkstatus/WgpuNoiseStatus.java`：阶段 1 的透明委托状态。
- `src/main/java/.../mixin/MixinNewStatusHook.java`：在 C2ME 注册原版 NOISE 前插入状态。
- `src/stub/java/com/ishland/...`：C2ME / FlowSched 的编译期接口签名（不打包）。
- `src/main/java/.../WgpuThinBinding.java`：FFM 绑定。

### 15.6 章节编号迁移表

2026-11 把原先"追加式"的 §5 附录拆开、按顺序重编。历史重写中删掉的编号
（如 §10.2、§18.6b）也一并列出，方便旧报告与提交记录对照。

| 旧编号 | 新编号 | 说明 |
|---|---|---|
| §0–§4 | §0–§4 | 未变 |
| §10 | §5 | 与 C2ME 的关系（构建期 Stub + 运行时强依赖） |
| §11 | §6 | 混合执行（modded 密度函数兜底） |
| §12 | §7 | 诊断基础设施 |
| §19 | §8.1 | 端到端 A/B：异步调度裁定 |
| §20 | §8.2 | 写回天花板实测 |
| §21 | §8.3 | 最小闭环：BlockOut shard |
| §22 | §8.4 | aquifer + Beardifier 全 GPU |
| §23 | §9 | 执行器流水线化 |
| §24 | §10 | 极限 A/B：WGPU vs 官方 OpenCL |
| §25 / §25.x | §11 / §11.x | r=1000 性能天花板 |
| §26 / §26.x | §12 / §12.x | readback 去冗余拷贝 |
| §27 / §27.x | §13 / §13.x | C2ME `NewChunkStatus` 接入 |
| §28 / §28.x | §14 / §14.x | 表面高度估算 |
| §5 附录 | §15 | 开发历程 |
| §5.x（里程碑） | §15.1 的 2a–2o | 例：§5.17 → §15.1 的 2n |
| §5.18–§5.21 | §8.1–§8.3 | 异步 / 写回 / 最小闭环 |
| §13–§18 | §15.2.1–§15.2.6 | 已证伪实验（存档） |
| §10.2 | §5 | 运行时依赖 C2ME 的实测（原节已删） |
| §18.6 / §18.6b | §3.5 | 异步调度设计（原节已删，见 §3.5 第 7 条） |

> 代码注释里的旧编号已按同一张表改写。注意代码旧注释里的 §23 长期指
> Beardifier（而不是文档旧的 §23 流水线），现在统一落到 §8.4。
