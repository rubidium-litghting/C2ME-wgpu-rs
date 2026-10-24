# C2ME-wgpu-rs

GPU-accelerated world generation for Minecraft, powered by [wgpu](https://wgpu.rs/) (Vulkan / Metal / DX12 / WebGPU).

This is a companion mod for [C2ME](https://modrinth.com/mod/c2me-fabric). It offloads the heavy noise / density-function pipeline of chunk generation to the GPU.

> [简体中文](README.md) | **English**

> ⚠️ **Experimental — read before installing.**
> The mod offloads the entire density pipeline (noise, aquifers, structure shaping) to the GPU and emits block states directly; the CPU only writes the chunk palette.
> On a desktop AMD integrated GPU (RADV), an end-to-end Chunky run (r=500, 4225 chunks) takes about 18 seconds, versus roughly 23 seconds for plain C2ME on the CPU.
> On a mobile GPU (Mali-G610) the same test takes 2m 19s versus 3m 12s on the CPU (**61% end-to-end speed-up**).
> Analysing Chunky's continuous generation rate shows a **+30.3%** terrain-generation throughput gain (30.65 CPS vs 23.52 CPS).
> The old "weak GPUs may be slower" warning no longer applies.

> **Compatibility first.**
> The mod is deliberately built on **fp32** so it runs on *as many GPUs as possible* — integrated GPUs and mobile SoCs included.
> The official C2ME OpenCL accelerator uses `fp64` (available only on newer Minecraft versions); this mod complements it by covering 1.21.1 and every device where `fp64` (and `shader-f64`) is unavailable.
> Precision here is recovered through fp32 internal compensation only; `f64` is deliberately never used, because it would lock out most mobile GPUs.

---

> 💡 **Resilience note**: if the GPU fails during generation, the affected dimension falls back to CPU generation and the game keeps running; the cause is written to `logs/c2me-wgpu.log`. The game only halts to prevent a deadlock when the GPU stops responding entirely (30 seconds with no result for a submitted batch), logging the reason to the same file.
>
> ⚠️ **Cross-platform behaviour**:
> - **Desktop**: if an incompatible backend is forced (e.g. `WGPU_BACKEND=gl`), the wgpu panic is caught and the game **silently falls back to CPU** without crashing.
> - **Android**: because cross-compilation defaults to `panic=abort`, the same situation makes the **game crash**. The reason is still written to `logs/c2me-wgpu.log`. Do not force `WGPU_BACKEND` on mobile devices.

## Requirements

| | |
|---|---|
| Minecraft | 1.21.1 (Fabric) |
| Fabric Loader | >= 0.17.2 |
| **C2ME** | **Required** (this is a companion mod and cannot run standalone) |
| Java | 25+ (built against Java 25, using the Panama FFM API) |
| GPU | Any GPU with a wgpu backend |

Java 25 or newer is required: the mod is compiled for Java 25 and talks to its Rust core through the Panama FFM API, stable since Java 24.

> **Why is it still called "C2ME-wgpu-rs"?**
> The mod ships its own access widener and requires C2ME.
>
> **At build time**, the Java code references C2ME classes directly (e.g. `SchedulingManager`). To avoid pulling in the whole C2ME source, a small **compile-only stub** is provided (`src/stub`, `compileOnly` — not packaged into the jar). Building therefore needs no C2ME jar and does not pin any specific C2ME version.
>
> **At runtime**, it hard-depends on C2ME for two things:
> 1. Multi-threaded chunk generation, to keep the GPU batch pipeline full (with C2ME, desktop throughput reaches ~144 chunks/s; without it the GPU starves and drops to ~13).
> 2. The async write-back architecture relies on C2ME's `SchedulingManager.positionedExecutor()` to run CPU tasks on its worker pool. If that API is unavailable, GPU acceleration is disabled for the run and generation falls back to plain C2ME on the CPU (there is no synchronous GPU fallback).
>
> It therefore remains a **C2ME companion module** — C2ME provides the parallelism and the scheduler, this mod provides the GPU — and the name stays.

---

## Installation

1. Install Fabric Loader for Minecraft 1.21.1.
2. Put **C2ME** and **C2ME-wgpu-rs** into the `mods/` folder.
3. Start the game.

It works out of the box; the default configuration needs no changes. If a compatible GPU **and** C2ME's scheduling API are available, terrain generation is offloaded to the GPU automatically; otherwise the mod stays silent and plain C2ME generates terrain on the CPU as usual.
To tune anything, edit `config/c2me-wgpu.json` — see below.

---

## Platform support

| Platform | Status |
|---|---|
| Linux x86_64 | ✅ Tested |
| Android aarch64 (custom JVM) | ✅ Tested (Mali-G610 / Dimensity) |
| Windows x86_64 | ✅ Tested (AMD, mingw DLL) |
| macOS arm64 (Metal) | ⚠️ Cross-compilable; native library is shipped, not yet verified on real hardware. Apple Silicon only, Intel Macs unsupported |
| OpenHarmony aarch64 | ⚠️ Cross-compilable; native library is shipped, not yet verified on real hardware. The library is unsigned and must be signed before it can be loaded |

On Android, the custom JVM does **not** support `-XX:+UseZGC`, and the mod does not suggest it there.

---

## How it works

1. **Serialisation** — the vanilla `NoiseRouter` (the density-function tree that drives terrain) is serialised on the Java side into a compact binary blob.
2. **Hand-off** — the blob crosses the Panama FFM API into the Rust `cdylib`. Calls are downcall C functions with a polling model; the single upcall exists only to forward Rust log lines to Java's SLF4J.
3. **Compilation** — Rust deserialises the tree, runs the optimisation passes, emits WGSL compute shaders and compiles them into GPU pipelines (cached on disk).
4. **Dispatch** — one GPU dispatch evaluates the density field for a whole batch of chunks. To handle non-linear density functions from mods correctly (e.g. `squeeze`), the final tree is split (the "Pass E" architecture): interpolated subtrees are evaluated on the corner grid, while the non-linear outer tree is applied per block during the final classification.
5. **Block states** — the GPU classifies every block itself (air / stone / water / lava), applying aquifers, Beardifier structure shaping and the **ore-vein sampler** (with a native GPU Xoroshiro RNG). Java only writes the resulting per-block state bytes into the chunk palette.

---

## Accuracy

The mod computes in **fp32**. Compared with Minecraft's `double` reference:

- shallow expressions agree to about **1e-6**;
- deep expressions (thousands of noise samples) drift by up to about **1e-2**.

That is the **inherent limit of fp32**, not a bug. Terrain thresholds are insensitive to differences of that size, and the generated terrain is visually indistinguishable from vanilla.

---

## Known limitations

- At extreme End coordinates (around 370,000 blocks) a partial NaN fallback is used to reproduce vanilla's void band.
- **Modded density functions**: custom density nodes the GPU serialiser cannot understand natively (e.g. from terrain mods) are evaluated on the CPU over the same corner grid and fed to the GPU as "external inputs". The rest of the tree still runs on the GPU, so modded world generation keeps working — only the unsupported subtrees pay a CPU cost.
- **No on-disk pipeline cache on Windows**: the cache directory follows the XDG convention (`$XDG_CACHE_HOME`, falling back to `$HOME/.cache`). Those variables are usually unset on Windows, so the disk cache is effectively disabled there — shaders are recompiled on every launch. This does not affect correctness, only start-up time.
- **Intel Macs are unsupported**: only a `macos-aarch64` native is shipped. An Intel Mac fails to find a matching library and falls back to CPU generation.

---

## Configuration

User settings live in `config/c2me-wgpu.json`. The file is created on first launch.

**When changes apply:** `debug_*` options take effect the next time you enter a world. Options
that affect the GPU device (`gpu_index`, `disable_pipeline_cache`, `mapped_at_creation`) take
effect when you **leave the save and re-enter**, because they require rebuilding the whole wgpu
instance. Neither requires restarting the game.

If the file is malformed (invalid JSON) the mod renames it to `c2me-wgpu.json.broken` for inspection and writes a fresh default; out-of-range values are corrected and written back.

Whole-line `//` comments are ignored. The mod writes a list of available GPUs above `gpu_index`:

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
| `gpu_index` | `-1` | Which GPU generates terrain. `-1` = automatic (integrated first). The index refers to the list written as a comment in this file; ModMenu offers the same choice as a dropdown. An out-of-range or unsuitable index falls back to automatic selection and says why in the log. |
| `backend` | `-1` | Graphics backend. `-1` = automatic (Vulkan/DX12/Metal, excluding GL). The available values depend on the platform and are listed as a comment in the config file. Switching backends rebuilds the wgpu instance, so it takes effect after **leaving the save and re-entering**. Note that wgpu's GL backend does not provide a pipeline cache, so choosing GL requires enabling `disable_pipeline_cache` as well. |
| `disable_pipeline_cache` | `false` | Bypass the on-disk shader pipeline cache. Only needed if a driver crashes inside the pipeline cache; the cost is recompiling shaders every launch. |
| `mapped_at_creation` | `null` | How buffers are uploaded. `null` = platform default (mapped on desktop, write_buffer on Android); `true` / `false` forces the path. write_buffer costs one extra copy but is the more conservative choice, which is why Android defaults to it; the mapped path also works on the Mali-G610 tested here, so both are usable. |

### Debugging (`debug_` prefix)

For troubleshooting; leave at their defaults during normal play.

| Field | Default | Description |
|---|---|---|
| `debug_verbose_logging` | `false` | Verbose diagnostic logging: shader compile timings, pipeline-cache hits, adapter workarounds, plus periodic chunk-per-second (`CPS=`) stats (also printed to the chat HUD). |
| `debug_stage_timing` | `false` | Emit per-stage timing to the log: GPU side (encode / submit / readback / decode) and CPU side (block write-back into the chunk). |

### Offline diagnostics

For comparing against vanilla output offline. All default to off; a normal session writes nothing.

| Field | Default | Description |
|---|---|---|
| `debug_vanilla_ref` | `false` | Skip the GPU entirely, let vanilla generate, then dump blocks for comparison. |
| `debug_dump_chunk` | `""` | Target chunk(s): `"x,z"` or `"x1,z1,x2,z2"`. Empty disables every other `debug_dump_*` switch. |
| `debug_dump_dir` | `""` | Where dump files go. Empty = platform temp dir. |
| `debug_dump_blocks` | `false` | Dump block states of the target chunk(s). |
| `debug_dump_aquifer` | `false` | Dump the aquifer cell table. |
| `debug_dump_beard` | `false` | Dump Beardifier structure data and per-block reference values. |
| `debug_dump_blob` | `""` | Write the serialised NoiseRouter blob here (this is how `src/test/resources/real_router.bin` is produced). |

## Settings screen (ModMenu)

With [ModMenu](https://modrinth.com/mod/modmenu) installed, this mod gets a **Settings** button
in the mod list, where `gpu_index`, the pipeline cache, the buffer upload path and both debug
switches can be changed without editing JSON by hand.

- ModMenu is a **soft dependency**: without it the config file alone is fully functional.
- The screen is generated from annotations on the `Config` fields, so it cannot drift out of
  sync with the file.
- Changes are written when you press "Done". `debug_*` applies immediately; device-related
  options apply the next time you enter a world.

## Environment variables

Every switch of this mod lives in the config file; it reads no `C2ME_*` environment variables. This section covers only the standard variables read by **wgpu itself**.

### GPU selection (standard wgpu variables)

The mod honours wgpu's own environment variables, so they behave the same as in any other wgpu application:

| Variable | Values | Effect |
|---|---|---|
| `WGPU_BACKEND` | `vulkan`, `dx12`, `metal`, `gl`, `webgpu` (comma-separated) | Restrict the graphics backends tried. Useful when one backend is broken. Note: the `gl` backend requires OpenGL 4.3+ for compute shader support. |
| `WGPU_ADAPTER_NAME` | substring (case-insensitive) | Force a specific GPU by name, e.g. `WGPU_ADAPTER_NAME=radeon`. If nothing matches, the mod logs a warning and falls back to its default choice. |

Without any of these, the mod uses Vulkan / Metal / DX12 (whichever the OS provides) and prefers the **integrated GPU**, so the discrete GPU stays free for rendering.

### Native library and logs

**Troubleshooting native extraction**:
By default the mod extracts the `.so` or `.dll` from the jar into a `c2me-wgpu-natives/` folder in the game root directory. If that directory is not writable or loading fails (e.g. a `noexec` mount), it falls back to the OS temporary directory (e.g. `/tmp` on Linux).
If you need to debug native library loading failures, add `-Dc2me.wgpu.preserveNative=true` to your JVM arguments; the mod will extract the native library directly into the game's working directory.

> **Note:** configuration fields marked "Development only", and the offline diagnostics group, are strictly for troubleshooting and do not need to be set during normal gameplay.

> **Rust-side log file:** the native library writes its diagnostics to `<gameDir>/logs/c2me-wgpu.log` (the path is passed from Java at startup). This is separate from Minecraft's own `logs/latest.log`, which only captures Java logging.

### Build-time only

| Variable | Effect |
|---|---|
| `ANDROID_NDK_HOME` | When set, the build **automatically** cross-compiles the native library for Android (aarch64) and stages it under `natives/android-aarch64/` inside the jar. Unset it (or pass `-PskipAndroid`) to skip. |

---

## Building from source

```bash
./gradlew build
```

The mod jar is written to `build/libs/`.

Building requires:
- JDK 25 (for the Java toolchain) and JDK 21 (for the Gradle JVM);
- a Rust toolchain (the native library is compiled as part of the build);
- **`cargo-ndk`** *only* when building for Android
  (`cargo install cargo-ndk`, plus an Android NDK; with `ANDROID_NDK_HOME` unset the Android build is skipped automatically, or force it off with `-PskipAndroid`).

---

## Development note

This mod was developed with **extensive AI assistance** — the code, documentation and tests are largely AI-generated under maintainer review. Treat it with that in mind: verify behaviour on your own hardware before relying on it. Bug reports and pull requests are welcome.

### Notes on running the tests

**After touching anything terrain-related, run the comparison test.** `real_router_vs_vanilla` is the only test that checks GPU output against real Minecraft; it takes about 2 seconds and is therefore ignored by default. After changing an emitter, aquifer, beard, ore veins or deserialisation, run:

```
cd rust && cargo test --lib -- --ignored real_router_vs_vanilla
```

`cargo test` can occasionally fail a GPU test on integrated graphics (roughly once every 5–8 runs). Each test case creates its own executor (~600 MB of GPU buffers), and running them in parallel stacks those allocations; this is a test-architecture issue unrelated to the in-game path, where a single executor runs with bounded concurrency set by `pipeline_depth`. If it reproduces, please attach the full `cargo test` output.

---

## Provenance

This project is an independent, clean-room reimplementation. It was written against Minecraft's **runtime behaviour** (exported noise data and sampled reference values), not against the source of any third-party mod.

A code-provenance review compared this mod's aquifer and Beardifier implementations with [C2ME](https://modrinth.com/mod/c2me-fabric)'s OpenCL accelerator: no shared identifiers, no copied functions, and different architectures (C2ME mirrors Minecraft function by function; this mod uses precomputed tables plus GPU scan designs). Where the two look similar (e.g. `fastInverseSqrt`, `getStructureWeight`), the common source is Minecraft's own vanilla code, which both projects reimplemented independently.

Scope note: the review was done at the feature/architecture level, not as a character-by-character diff. Minecraft itself is not redistributed; only original code is provided. See `LICENSE`.

---

## License

MIT — see [LICENSE](LICENSE).
