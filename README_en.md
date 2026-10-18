> **English** | [简体中文](README.md)

# C2ME-wgpu-rs

GPU-accelerated world generation for Minecraft, powered by
[wgpu](https://wgpu.rs/) (Vulkan / Metal / DX12 / WebGPU).

A companion mod for [C2ME](https://modrinth.com/mod/c2me-fabric)
that offloads the heavy noise / density-function pipeline of chunk generation
onto the GPU.

> ⚠️ **Experimental — read before installing.** The mod now offloads the
> whole density pipeline (noise, aquifer, structure shaping) to the GPU and
> emits block states directly; the CPU only writes the chunk palette. On a
> desktop AMD iGPU (RADV), an end-to-end Chunky run (r=500, 4225 chunks)
> finishes in ~15 s vs ~23 s for plain C2ME on the CPU. On a mobile GPU
> (Mali-G610), the same workload completes in 2m19s vs 3m12s on the CPU
> (a **61% end-to-end speedup**). Analyzing Chunky's continuous generation
> rate shows a **+30.3% throughput increase** (30.65 CPS vs 23.52 CPS).
> The old "weak-GPU can be slower" caveat no longer applies.

> **Compatibility-first.** This mod is deliberately built on **fp32** so it can
> run on the *widest possible range of GPUs* — including integrated graphics
> and mobile SoCs. The official C2ME OpenCL accelerator targets `fp64` (available
> on newer Minecraft versions); this mod complements it by covering 1.21.1 and
> devices that `fp64` (and `shader-f64`) cannot reach. Higher precision here is
> only ever obtained through fp32-internal compensation; `f64` is intentionally
> never used, because it would exclude most mobile GPUs.

---

> 💡 **Resilience note**: If the GPU driver crashes or the device is lost, the game will safely halt to prevent deadlocks, and will log the detailed cause to `logs/c2me-wgpu.log` for offline diagnostics.
>
> ⚠️ **Cross-platform behavior**:
> - **Desktop**: If an incompatible backend is forced (e.g. `WGPU_BACKEND=gl`), the wgpu panic is safely caught and the game **silently falls back to CPU** without crashing.
> - **Android**: Since cross-compilation defaults to `panic=abort`, the same situation causes an **immediate crash**. However, the panic reason will be written to `logs/c2me-wgpu.log`. Please do not force `WGPU_BACKEND` on mobile devices.

## Requirements

| | |
|---|---|
| Minecraft | 1.21.1 (Fabric) |
| Fabric Loader | >= 0.17.2 |
| **C2ME** | **required** (this is a companion module, not standalone) |
| Java | 25+ (the mod is built against Java 25 and uses the Panama FFM API) |
| GPU | Any GPU with a working wgpu backend |

Java 25 or newer is required: the mod is compiled for Java 25 (and talks to
its Rust core through the Panama FFM API, which is stable from Java 24).

> **Why still named "C2ME-wgpu-rs"?** This mod ships its own access widener
> and requires C2ME. 
> 
> **At build time**, the Java code directly references C2ME classes (like
> `SchedulingManager`). To avoid pulling the entire C2ME source, we provide
> a small **compile-only stub** (`src/stub`, `compileOnly` — not packaged
> into the jar). This means no C2ME jar is needed to compile, and no C2ME
> version is pinned.
> 
> **At runtime**, it strongly depends on C2ME for two things:
> 1. The multi-threaded chunk generation that keeps the GPU batch pipeline fed
>    (with C2ME, desktop throughput reaches ~144 chunks/s; without it, the GPU
>    starves and collapses to ~13).
> 2. The asynchronous write-back architecture heavily relies on C2ME's
>    `SchedulingManager.positionedExecutor()` to run CPU tasks on its worker
>    pool. If this API is unavailable, GPU acceleration is completely disabled
>    for that run, and plain C2ME generates terrain on the CPU (there is no
>    synchronous GPU fallback).
> 
> So it remains a **C2ME companion module** — C2ME supplies the parallelism and
> the scheduler, this mod supplies the GPU compute — and keeps the name.

---

## Installation

1. Install Fabric Loader for Minecraft 1.21.1.
2. Put **C2ME** and **C2ME-wgpu-rs** into your `mods/` folder.
3. Launch the game.

No configuration is needed. If a compatible GPU is found **and** C2ME's
scheduling API is available, terrain generation is offloaded automatically;
otherwise the mod stays silent and plain C2ME generates terrain on the CPU,
exactly as usual.

---

## Platform support

| Platform | Status |
|---|---|
| Linux x86_64 | ✅ tested |
| Android aarch64 (custom JVM) | ✅ tested (Mali-G610 / Dimensity) |
| Windows x86_64 | ✅ tested (AMD, mingw DLL) |
| macOS (Metal) | ⚠️ wgpu backend available, not officially tested |

On Android, `-XX:+UseZGC` is **not** supported by the custom JVM and the mod
will not suggest it there.

---

## How it works

1. **Serialize** — the vanilla `NoiseRouter` (the density-function tree that
   drives terrain) is serialized into a compact binary blob on the Java side.
2. **Hand off** — the blob is passed to a Rust `cdylib` through the Panama FFM
   API (no callbacks, a simple poll model).
3. **Compile** — Rust deserializes the tree, runs optimization passes, and
   generates a WGSL compute shader, which is compiled to a GPU pipeline
   (and cached on disk).
4. **Dispatch** — the GPU evaluates the density field for a whole batch of
   chunks. To correctly handle modded non-linear density functions (e.g.,
   `squeeze`), the final density tree is split (the "Pass E" architecture):
   interpolated subtrees are evaluated at cell corners, while non-linear wrappers
   are applied per-block during the final classification.
5. **Block states** — the GPU classifies every block (air / stone / water /
   lava) itself, applying the aquifer, the Beardifier (structure shaping), and
   the **Ore Vein sampler** (using a native GPU Xoroshiro RNG) on the GPU.
   Java only writes the resulting per-block state bytes into the chunk's palette.

---

## Accuracy

The mod computes in **fp32**. Compared to Minecraft's `double` reference:

- shallow expressions agree to about **1e-6**;
- deep expressions (thousands of noise samples) can drift up to about **1e-2**.

This is the **inherent limit of fp32**, not a bug. Terrain thresholds are not
sensitive to differences of this magnitude, and the generated terrain is
visually indistinguishable from vanilla. Higher precision is only ever obtained
through fp32-internal compensation — `f64` is intentionally never used, because
it would exclude most mobile GPUs.

---

## Known limitations

- At extremely far End coordinates (~370k blocks), a partial NaN fallback is
  used to reproduce vanilla's void bands.
- **Modded density functions**: a custom density node the GPU serializer does
  not natively understand (e.g. from a terrain mod) is evaluated on the CPU at
  the same corner grid and fed to the GPU as an "external input". The rest of
  the tree still runs on the GPU, so modded worldgen keeps working — only the
  unsupported subtree pays a CPU cost.

---

## Environment variables

### GPU selection (standard wgpu variables)

The mod honours wgpu's own environment variables, so they behave the same as
in any other wgpu application:

| Variable | Values | Effect |
|---|---|---|
| `WGPU_BACKEND` | `vulkan`, `dx12`, `metal`, `gl`, `webgpu` (comma-separated) | Restrict the graphics backends tried. Useful when one backend is broken. Note: The `gl` backend requires OpenGL 4.3+ for compute shader support. |
| `WGPU_ADAPTER_NAME` | substring (case-insensitive) | Force a specific GPU by name, e.g. `WGPU_ADAPTER_NAME=radeon`. If nothing matches, the mod logs a warning and falls back to its default choice. |

Without any of these, the mod uses Vulkan / Metal / DX12 (whichever the OS
provides) and prefers the **integrated GPU**, so the discrete GPU stays free
for rendering.

### Diagnostics and debug exports

Set these only when troubleshooting. All of them are read once at startup.

| Variable | Value | Effect |
|---|---|---|
| `C2ME_WGPU_DEBUG` | `1` | Enables verbose logging and periodic chunk-per-second (`CPS=`) stats. Also prints CPS to the chat HUD. Written to `<gameDir>/logs/c2me-wgpu.log`. |
| `C2ME_WGPU_PROFILE` | `1` | Emits per-stage GPU batch timing (encode / submit / readback / decode) to the log. |
| `C2ME_WGPU_ASYNC_SCHED` | `0` | Disables async write-back on C2ME's worker pool. Useful for A/B benchmarking. |
| `C2ME_WGPU_MAPPED_AT_CREATION` | `0` / `1` | Override buffer upload method. Default `1` on desktop, `0` on Android (Mali rejects mapped path). |
| `C2ME_VANILLA_REF` | `1` | Skip GPU entirely, let vanilla generate, then dump blocks for offline comparison. |
| `C2ME_DUMP_BLOCKS` | `1` | Dump block states for the target chunk for offline comparison. |
| `C2ME_DUMP_BLOB` | file path | Dump the serialized NoiseRouter blob to the given path. |
| `C2ME_DUMP_DIR` | directory | Where dump files are written. Defaults to platform temp dir. |
| `C2ME_DUMP_CHUNK` | `x,z` or `x1,z1,x2,z2` | Target chunk(s) for the `C2ME_DUMP_*` diagnostics. |
| `C2ME_DUMP_AQUIFER` | `1` | Dump the aquifer cell table. |
| `C2ME_DUMP_AQ_BLOCKS` | `1` | Dump per-block aquifer reference data. |
| `C2ME_DUMP_DENSITY` | `1` | Dump GPU density values. |
| `C2ME_DUMP_BEARD` | `1` | Dump Beardifier structure shaping data. |
| `C2ME_WGPU_FORCE_SINGLE` | `1` | Development only: forces a single-shard all-channel shader for debugging. |
| `C2ME_WGPU_MINIMAL_SHADER` | `1` | Development only: uses a minimal stub shader for bring-up testing. |
| `C2ME_WGPU_MAX_CHANNELS` | integer | Development only: caps the number of emitted channels. |
| `C2ME_WGPU_FAKE_GPU_MS` | integer | Development only: simulates a slower GPU by injecting delay. |
| `C2ME_WGPU_NO_PIPELINE_CACHE` | `1` | Development only: bypasses the on-disk shader pipeline cache. |

The blob dump can also be set with the JVM property `-Dc2me.dumpBlob=<path>`;
the environment variable is only a fallback when the property is unset.

**Native extraction location**:
By default, the mod extracts the `.so` or `.dll` from the jar into a `c2me-wgpu-natives/` folder in the game root directory. If that directory is not writable or loading fails (e.g., `noexec` mount), it automatically falls back to the OS temporary directory (e.g., `/tmp` on Linux).
If you need to debug native library loading failures, you can add `-Dc2me.wgpu.preserveNative=true` to your JVM launch arguments. The mod will extract the native library directly to the game's working directory instead.

> **Note:** variables prefixed with `C2ME_DUMP_` and those marked "Development
> only" are strictly for troubleshooting and do not need to be set during
> normal gameplay.

> **Rust-side log file:** the native library writes its diagnostics to
> `<gameDir>/logs/c2me-wgpu.log` (the path is passed from Java at startup).
> This is separate from Minecraft's own `logs/latest.log`, which only captures
> Java logging.

### Cache location

| Variable | Effect |
|---|---|
| `XDG_CACHE_HOME` | Base directory for the on-disk shader pipeline cache (`$XDG_CACHE_HOME/c2me/wgpu_pipelines/`). |
| `HOME` | Fallback when `XDG_CACHE_HOME` is unset (`$HOME/.cache`). |

> **Note:** the cache path follows the XDG convention, which is native to Linux
> only. On Windows and macOS these variables are usually unset or non-standard,
> so the on-disk pipeline cache is effectively disabled there. This costs a
> one-off shader recompile per launch and does not affect correctness.

### Build-time only

| Variable | Effect |
|---|---|
| `ANDROID_NDK_HOME` | When present, the build **automatically** cross-compiles the native library for Android (aarch64) and stages it into the jar under `natives/android-aarch64/`. Unset it (or pass `-PskipAndroid`) to skip. |

---

## Building from source

```bash
./gradlew build
```

The built mod jar is written to `build/libs/`.

Building requires:
- JDK 25 (for the Java toolchain) and JDK 21 (for the Gradle JVM);
- a Rust toolchain (the native library is compiled as part of the build);
- **`cargo-ndk`** *only if* you build for Android
  (`cargo install cargo-ndk`, plus an Android NDK; the Android build is
  skipped automatically when `ANDROID_NDK_HOME` is not set,
  or forced off with `-PskipAndroid`).

---

## Development note

This mod was developed with heavy **AI assistance** — the code, documentation
and tests were largely AI-generated under maintainer review. Treat it
accordingly: verify behaviour on your own hardware before relying on it.
Bug reports and pull requests are welcome.

---

## Provenance

This project is an independent, clean-room reimplementation. It was written
against Minecraft's **runtime behaviour** (dumped noise data and sampled
reference values), not against any third-party mod's source code.

A code-origin review compared this mod's aquifer and Beardifier implementation
against [C2ME](https://modrinth.com/mod/c2me-fabric)'s OpenCL accelerator:
no shared identifiers, no copied functions, and a different architecture
(C2ME mirrors Minecraft function-by-function; this mod uses a
precomputed-table + GPU-scan design). Where the two look similar
(e.g. `fastInverseSqrt`, `getStructureWeight`), the common source is
Minecraft's own vanilla code, which both projects independently reproduce.

Scope note: the review was at the feature/architecture level, not a
character-by-character diff. Minecraft itself is not redistributed; only
original code is shipped. See `LICENSE`.

---

## License

MIT — see [LICENSE](LICENSE).
