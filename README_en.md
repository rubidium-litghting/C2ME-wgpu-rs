# C2ME-wgpu-rs

GPU-accelerated world generation for Minecraft, powered by [wgpu](https://wgpu.rs/) (Vulkan / Metal / DX12 / WebGPU).

This is a companion mod for [C2ME](https://modrinth.com/mod/c2me-fabric) that offloads the heavy noise / density-function pipeline of chunk generation to the GPU.

> **English** | [简体中文](README.md)

> ⚠️ **Experimental — read before installing.**
> The mod now offloads the entire density pipeline (noise, aquifer, beardifier) to the GPU and emits block states directly; the CPU only writes the chunk palette.
> On a desktop AMD integrated GPU (RADV), the end-to-end Chunky test (r=500, 4225 chunks) takes about 18 s, against roughly 23 s for C2ME on CPU alone.
> On a mobile GPU (Mali-G610), the same test takes 2 min 19 s versus 3 min 12 s on CPU (**61% end-to-end speedup**).
> Chunky's sustained generation rate shows terrain throughput up **+30.3%** (30.65 CPS vs 23.52 CPS).
> The older "a weak GPU can be slower" warning no longer applies.

> **Compatibility first.**
> The mod is deliberately built on **fp32** to cover *as many GPUs as possible* — integrated GPUs and mobile SoCs included.
> The official C2ME OpenCL module is built on `fp64` (only available on newer Minecraft versions); this mod complements it by covering 1.21.1 and the devices `fp64` (and `shader-f64`) cannot reach.
> The precision here comes purely from fp32 internal compensation; `f64` is deliberately never used, because it would shut most mobile GPUs out.

---

> 💡 **Robustness note**: if the GPU fails during generation, the affected dimension falls back to CPU generation, the game keeps running, and the reason is written to `logs/c2me-wgpu.log`. Only when the GPU stops responding entirely (no result 30 s after submission) does the game exit safely to avoid a deadlock, again writing the details to that file.
>
> ⚠️ **Cross-platform behaviour differs**:
> - **Desktop**: if an incompatible backend is forced (e.g. `WGPU_BACKEND=gl`), the wgpu panic is caught safely and the game **silently falls back to CPU** without crashing.
> - **Android**: because the cross-compiled build defaults to `panic=abort`, the same situation **crashes the game**. The cause is still written to `logs/c2me-wgpu.log`; do not set `WGPU_BACKEND` arbitrarily on mobile.

## Requirements

| | |
|---|---|
| Minecraft | 1.21.1 (Fabric) |
| Fabric Loader | >= 0.17.2 |
| **C2ME** | **Required** (this is a companion module and cannot run standalone) |
| Java | 25+ (built on Java 25, uses the Panama FFM API) |
| GPU | Any GPU with a wgpu backend and a per-storage-buffer limit of **at least 128 MiB** (almost every integrated and mobile GPU released after 2020 qualifies) |

Java 25 or newer is required: the mod is compiled for Java 25 and talks to its Rust core through the Panama FFM API (stable since Java 24).

> **Why is it still called "C2ME-wgpu-rs"?**
> The mod ships its own access widener and requires C2ME.
>
> **At build time**, the Java code references C2ME classes directly (e.g. `SchedulingManager`). To avoid pulling in the whole C2ME source tree we provide a small **compile-time-only stub** (`src/stub`, `compileOnly` — not packaged into the jar). Compiling therefore needs neither the C2ME jar nor a pinned C2ME version.
>
> **At runtime** it hard-depends on C2ME for two things:
> 1. Multithreaded chunk generation, to keep the GPU batch pipeline saturated (with C2ME, desktop throughput reaches ~144 chunks/s; without it the GPU starves and drops to ~13).
> 2. The async write-back architecture relies heavily on C2ME's `SchedulingManager.positionedExecutor()` to run CPU tasks on its worker pool. If that API is unavailable, GPU acceleration is disabled for the run entirely and generation reverts to plain C2ME on the CPU (there is no synchronous GPU fallback).
>
> It therefore remains a **C2ME companion module** — C2ME provides parallelism and the scheduler, this mod provides the GPU compute — and the name is kept.

---

## Installation

1. Install Fabric Loader for Minecraft 1.21.1.
2. Drop **C2ME** and **C2ME-wgpu-rs** into the `mods/` folder.
3. Launch the game.

It works out of the box; the default config needs no changes. If a compatible GPU **and** C2ME's scheduling API are found, terrain generation is offloaded to the GPU automatically; otherwise the mod stays silent and plain C2ME generates terrain on the CPU exactly as before.
To tune it, edit `config/c2me-wgpu.json` — see below.

> 💡 **Performance tip: pair it with a lighting optimization mod**
> During extreme fast travel (e.g., spectator mode) or large-scale chunk pregeneration, vanilla Minecraft's single-threaded lighting engine can easily become the new bottleneck. This causes main-thread stutters, making it seem like chunk generation can't keep up. We strongly recommend **ScalableLux** (or a similar lighting optimization mod) to fully unleash the GPU terrain generation throughput.

---

## Platform support

| Platform | Status |
|---|---|
| Linux x86_64 | ✅ Tested |
| Android aarch64 (custom JVM) | ✅ Tested (Mali-G610 / Dimensity) |
| Windows x86_64 | ✅ Tested (AMD, mingw DLL) |
| macOS arm64 (Metal) | ⚠️ Cross-compiles and ships a native library; not yet verified on real hardware. Apple Silicon only, Intel Macs unsupported |
| OpenHarmony aarch64 | ⚠️ Cross-compiles and ships a native library; not yet verified on real hardware. The library is unsigned and must be signed before it can be loaded |

On Android, custom JVMs do **not** support `-XX:+UseZGC`, and the mod will not suggest it there.

---

## How it works

1. **Serialisation** — vanilla's `NoiseRouter` (the density-function tree that drives terrain) is serialised on the Java side into a compact binary blob.
2. **Hand-off** — that blob is passed to a Rust `cdylib` over the Panama FFM API. The calls follow a down-call C function + polling model; the only up-call forwards Rust logs to Java's SLF4J.
3. **Compilation** — Rust deserialises the tree, runs optimisation passes, and generates WGSL compute shaders, which are then compiled into GPU pipelines (and cached on disk). When a dimension needs compiling for the first time, a progress bar with elapsed time appears at the top of the screen — above the loading screen, so the client never looks frozen.
4. **Dispatch** — one GPU dispatch computes the density field for a whole batch of chunks. To handle non-linear density functions from mods correctly (e.g. `squeeze`), the final density tree is split (the "Pass E" architecture): interpolated subtrees are evaluated on the corner grid, while the non-linear outer layers are applied per block during final classification.
5. **Block states** — the GPU classifies each block itself (air / stone / water / lava), applying the aquifer, the beardifier (structure base) and the **ore-vein sampler** (with a native GPU Xoroshiro RNG) on the GPU. Java only writes the resulting per-block state bytes into the chunk palette.

---

## Accuracy

The mod computes in **fp32**. Against Minecraft's `double` reference values:

- shallow expressions agree to about **1e-6**;
- deep expressions (thousands of noise samples) can drift by up to about **1e-2**.

That is the **inherent limit of fp32**, not a bug. Terrain thresholds are insensitive to differences of this magnitude, and the generated terrain is visually indistinguishable from vanilla.

---

## Known limitations

- At extreme End coordinates (around 370,000 blocks) a partial NaN fallback is used to reproduce vanilla's void band.
- **Modded density functions**: custom density nodes the GPU serialiser cannot understand natively (e.g. from terrain mods) are evaluated on the CPU over the same corner grid and fed to the GPU as "external inputs". The rest of the tree still runs on the GPU, so modded world generation keeps working — only the unsupported subtrees pay a CPU cost.
- **Mobile GPU buffer limits**: the mod requires a per-storage-buffer limit of at least 128 MiB (the WebGPU default minimum). Earlier versions required 256 MiB, which shut out modern mobile GPUs that report exactly 128 MiB, such as the Adreno 750 (Snapdragon 8 Gen 3); the threshold has been lowered to 128 MiB as of 0.1.6. Drivers that still fail the check are rejected at start-up, the dimension falls back to CPU, and the chat log explains why.
- **Worlds with too many external inputs fall back to CPU**: every modded density node the serialiser cannot translate natively is sampled on the CPU over the corner grid and uploaded to the GPU. The limit counts **distinct nodes after deduplication**: the density tree is a DAG, so one instance reached through many paths only costs one slot, but each slot still costs per-chunk CPU sampling. Above 64 of them the upload approaches the GPU's buffer limit and the CPU sampling cost far outweighs the GPU gain, so the mod sends that dimension down the CPU path and says so in chat.
- **Under NeoForge + Sinytra Connector, some terrain mods trigger the limit above**: Connector wraps Fabric mods' density nodes in its own classes, which the mod cannot recognise, so they are all treated as external inputs. Tectonic, for example, accelerates fine on native Fabric (3.0.28 has 7 distinct nodes after deduplication) but falls back to CPU under Connector because of the node count. This is a Connector compatibility limit; the game still runs normally.
- **No on-disk pipeline cache on Windows**: the cache directory follows the XDG convention (`$XDG_CACHE_HOME`, falling back to `$HOME/.cache`). Those variables are usually unset on Windows, so the disk cache is effectively disabled there — shaders are recompiled on every launch. This does not affect correctness, only start-up time.
- **Intel Macs are unsupported**: only a `macos-aarch64` native is shipped. An Intel Mac fails to find a matching library and falls back to CPU generation.
- **Switching the graphics backend on Android requires a restart**: a device created by the GL backend releases almost none of its native memory when dropped (measured 15 MiB freed out of ~4 GB RSS, against 645 MiB for a Vulkan device). Hot-switching on Android therefore stacks both sets of resources and exhausts memory, so the mod asks for a restart instead. Desktop does not show this behaviour and still hot-swaps.
- **A failing GL backend can leak GPU memory**: wgpu's GLES backend panics when EGL context creation fails, and the unwinding path does not release the native resources it already allocated. The mod catches that panic and keeps the previous device, so the game is unaffected, but each failed attempt leaves some memory behind. This is upstream wgpu behaviour and cannot be reclaimed from the mod.

---

## Configuration

User settings live in `config/c2me-wgpu.json`, generated automatically on first launch.

**When changes take effect**: options prefixed with `debug_` apply the next time you enter a world; options that affect the GPU device (`gpu_index`, `disable_pipeline_cache`, `mapped_at_creation`) apply on **leaving and re-entering the world**, because they require rebuilding the whole wgpu instance. Neither requires restarting the game.

If the file is corrupted (invalid JSON), the mod renames it to `c2me-wgpu.json.broken` for inspection and writes a fresh default config; out-of-range values are clamped and written back.

Whole-line `//` comments in the file are ignored. The mod automatically writes a list of available GPUs above `gpu_index`:

```jsonc
{
  // Available GPUs (use one as gpu_index; -1 = automatic):
  //   0: AMD Radeon Graphics (RADV RENOIR) [Vulkan]
  "gpu_index": -1,
  "disable_pipeline_cache": false,
  "debug_verbose_logging": false,
  "debug_stage_timing": false,
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

### General

| Field | Default | Description |
|---|---|---|
| `gpu_index` | `-1` | Which GPU generates terrain. `-1` = automatic (integrated first). The index matches the list in the config file comments; it can also be picked in the ModMenu settings screen. An out-of-range index or a card that fails the checks falls back to automatic selection, with the reason logged. |
| `backend` | `-1` | Graphics backend. `-1` = automatic (Vulkan/DX12/Metal, excluding GL). Available values depend on the platform and are listed in the config file comments. On desktop a change applies on **leaving and re-entering the world**; on Android it requires a **game restart** (see "Known limitations"). Note that wgpu's GL backend does not offer pipeline caching, so choosing GL requires `disable_pipeline_cache` as well. |
| `disable_pipeline_cache` | `false` | Bypass the on-disk shader pipeline cache. Only needed if the driver crashes on pipeline caching; the cost is recompiling shaders on every launch. |
| `chunk_batch_size` | `1` | Aligned batch size: `1` = one chunk at a time (default, behaves exactly like earlier versions), `2` / `4` / `8` = collapse N×N chunks into one scheduling unit so the GPU sees larger batches. Same-seed r=500 measurements: per-chunk 18 s, 2×2 14 s, **4×4 13 s (fastest)**, 8×8 15 s (batches too large, per-batch latency eats the gain). **Requires a full game restart**; leaving and re-entering the world is not enough. |
| `mapped_at_creation` | `null` | How buffers are uploaded. `null` = platform default (mapped path on desktop, `write_buffer` on Android); `true` / `false` force a specific path. Writing a buffer costs one extra copy but is more conservative, which is why Android defaults to it; the mapped path also works on the Mali-G610 we tested, so both paths are functional. |

### Debugging (`debug_` prefix)

These options are for troubleshooting; leave them at their defaults for normal play.

| Field | Default | Description |
|---|---|---|
| `debug_verbose_logging` | `false` | Enables detailed diagnostics: shader compile times, pipeline-cache hits, adapter workarounds, and periodic chunks-per-second (`CPS=`) stats (also printed to chat). |
| `debug_stage_timing` | `false` | Writes per-stage timings to the log: GPU side (encode / submit / readback / decode) and CPU side (block write-back into the chunk). |
| `debug_height_probe` | `false` | Counts vanilla `ChunkGenerator.getHeight` calls, their cost, distinct columns and calling threads, and whether GPU batches were still in flight. Used to decide whether caching surface height is worth it (see `DEVELOPMENT.md` §14). Read once at class load; a change needs a restart. |

### Offline diagnostics

Used to compare against vanilla output offline. Leave everything at defaults for normal play; no files are written.

| Field | Default | Description |
|---|---|---|
| `debug_vanilla_ref` | `false` | Skip the GPU entirely, let vanilla generate, then dump blocks for comparison. |
| `debug_dump_chunk` | `""` | Target chunk(s): `"x,z"` or `"x1,z1,x2,z2"`. If empty, all other `debug_dump_*` options are inactive. |
| `debug_dump_dir` | `""` | Directory for dump files. Empty = platform temp directory. |
| `debug_dump_blocks` | `false` | Dump the target chunks' block states. |
| `debug_dump_aquifer` | `false` | Dump the aquifer cell table. |
| `debug_dump_beard` | `false` | Dump the beardifier structures and per-block reference values. |
| `debug_dump_blob` | `""` | Write the serialised NoiseRouter blob to this path (`src/test/resources/real_router.bin` is generated this way). |

## Settings screen (ModMenu)

If [ModMenu](https://modrinth.com/mod/modmenu) is installed, this mod gets a **Settings** button in the mod list, where `gpu_index`, the pipeline cache, the buffer upload mode and the two debug switches can be changed without editing JSON.

- ModMenu is a **soft dependency**: without it, the config file alone governs everything and nothing is lost.
- The screen is generated entirely from annotations on `Config` fields, so it can never fall out of sync with the config file.
- Clicking "Done" writes the changes back to the file. `debug_*` applies immediately; device-related options apply the next time you enter a world.

## Environment variables

All of this mod's own switches live in the config file; it reads no `C2ME_*` environment variables. This section covers only the standard variables read by the **wgpu library** itself.

### GPU selection (standard wgpu variables)

The mod honours wgpu's own environment variables, so they behave exactly as they do for any other wgpu application:

| Variable | Value | Effect |
|---|---|---|
| `WGPU_BACKEND` | `vulkan`, `dx12`, `metal`, `gl`, `webgpu` (comma-separated) | Restricts the backends tried. Useful when one backend is broken. Note: the `gl` backend needs OpenGL 4.3+ for compute shaders. |
| `WGPU_ADAPTER_NAME` | substring (case-insensitive) | Forces a GPU by name, e.g. `WGPU_ADAPTER_NAME=radeon`. `gpu_index` is the preferred way; this variable is for name-based selection and only matches cards that pass the capability checks. If nothing matches, a warning is logged and the default selection is used. |

Without these variables the mod uses Vulkan / Metal / DX12 (whichever the OS offers). Adapter selection is **capability-first**: every candidate must pass the capability checks (can it run our compute shaders), and only passing adapters enter the pool; they are then ordered **integrated → discrete → virtual**, keeping the discrete GPU as free as possible for rendering. Software rasterisers (such as llvmpipe) are filtered out during enumeration and never occupy an index.

### Native library and logs

**Locating the extracted native library**:
By default the mod extracts the bundled `.so` or `.dll` into a `c2me-wgpu-natives/` folder in the game directory. If that directory cannot be written or loaded (e.g. a `noexec` mount), the mod automatically falls back to the OS temp directory (e.g. `/tmp` on Linux).
To debug native library load failures, add `-Dc2me.wgpu.preserveNative=true` to the JVM arguments and the mod will extract the library directly into the game's working directory.

> **Note:** options marked "development only" and the offline diagnostic switches are strictly for troubleshooting and need not be set for normal play.

> **Rust-side log file:** the native library writes its diagnostics to `<gameDir>/logs/c2me-wgpu.log` (the path is passed in by Java at start-up). This is separate from Minecraft's own `logs/latest.log`, which only captures Java logging.

### Build-time only

| Variable | Effect |
|---|---|
| `ANDROID_NDK_HOME` | When set, the build **automatically** cross-compiles the native library for Android (aarch64) and stages it under `natives/android-aarch64/` inside the jar. Unset it (or pass `-PskipAndroid`) to skip. |

---

## Building from source

```bash
./gradlew build
```

The built mod jar is written to `build/libs/`.

Building requires:
- **JDK 25** (Java toolchain) and **JDK 21** (Gradle JVM) — both are hard requirements and a missing one fails the build;
- **Rust 1.85+** (the crate uses edition 2024);
- a native C toolchain for the host platform (gcc/clang on Linux, MSVC or mingw on Windows).

The following platforms are **built on demand and skipped with a log line if the toolchain is missing**, never failing the build:

| Platform | Trigger | Disable switch |
|---|---|---|
| Windows | `rustup` has an `x86_64-pc-windows-gnu` or `*-msvc` target | — |
| Android | `ANDROID_NDK_HOME` set and `cargo-ndk` installed | `-PskipAndroid` |
| OpenHarmony | `OHOS_SDK_NATIVE` set and `cargo-ohos` installed | `-PskipOhos` |
| macOS | `OSXCROSS_ROOT` points at a valid osxcross (with `*-apple-darwin*-clang` and `MacOSX*.sdk`) | `-PskipMacOS` |

A clone therefore builds successfully on a machine with only JDK + Rust, and the resulting jar contains only the host platform's native. Packaging all platforms requires the toolchains listed above.

`-PskipRust` skips native compilation entirely (e.g. CI without a Rust toolchain).

---

## Development note

This mod was developed with **heavy AI assistance** — most of the code, documentation and tests were generated by AI under the maintainer's review. Treat it accordingly: verify behaviour on your own hardware before relying on it. Bug reports and pull requests are welcome.

### Notes on running the tests

**You must run the reference test after touching terrain code.** `real_router_vs_vanilla` is the only test that verifies GPU output against real Minecraft; it is ignored by default because it takes about 2 s. After changing the emitter, aquifer, beard, ore veins or deserialisation, run:

```
cd rust && cargo test --lib -- --ignored real_router_vs_vanilla
```

`cargo test` may intermittently fail GPU tests on integrated graphics (roughly once every 5–8 runs). The cause is that each test case creates its own executor (~600 MB of GPU buffers each), and running in parallel stacks those resources; this is a test-architecture issue unrelated to the in-game path — the game has a single executor with bounded concurrency set by `pipeline_depth`. If you can reproduce it, attach the full `cargo test` output.

---

## Provenance

This project is an independent, clean-room reimplementation. It was written against Minecraft's **runtime behaviour** (exported noise data and sampled reference values), not against any third-party mod's source code.

A code-provenance review compared this mod's aquifer and beardifier implementations with [C2ME](https://modrinth.com/mod/c2me-fabric)'s OpenCL accelerator: no shared identifiers, no copied functions, and a different architecture (C2ME mirrors Minecraft function by function; this mod uses precomputed tables + a GPU-scan design). Where the two look similar (e.g. `fastInverseSqrt`, `getStructureWeight`), the common source is Minecraft's own vanilla code, which both projects reproduce independently.

Scope note: the review was done at the feature/architecture level, not as a character-by-character diff. Minecraft itself is not redistributed; only original code is provided. See `LICENSE`.

---

## Acknowledgements

We would like to express our sincere gratitude to the **C2ME author team** ([RelativityMC](https://github.com/RelativityMC) and all contributors):

- C2ME's multithreaded chunk generation and scheduler (`SchedulingManager`) are the absolute foundation that allows this mod to run saturated and maximize GPU compute.
- The official C2ME OpenCL acceleration module (`c2me-opts-accel-opencl`) provided immense inspiration for building the early skeleton of `c2me-wgpu-rs`, particularly its architectural design, batching strategies, and driver compatibility workarounds (see the relevant analysis and borrowed ideas in `DEVELOPMENT.md`).

---

## License

MIT — see [LICENSE](LICENSE).