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
> finishes in ~15 s vs ~23 s for plain C2ME on the CPU. Behaviour on
> **weak/mobile GPUs has not been re-benchmarked** under this new
> architecture — the older "strong-CPU / weak-GPU can be slower" caveat
> applied to the previous CPU-write-back design. Benchmark on your own
> hardware before relying on it.

> **Compatibility-first.** This mod is deliberately built on **fp32** so it can
> run on the *widest possible range of GPUs* — including integrated graphics
> and mobile SoCs. The official C2ME accelerator targets `fp64` via OpenCL;
> this mod complements it by covering devices that fp64 (and `shader-f64`)
> cannot reach.

---

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
> and does **not** link against C2ME at build time: the C2ME members it calls
> (`SchedulingManager`) are declared by a small **compile-only stub**
> (`src/stub`, `compileOnly` — not packaged into the jar), so no C2ME jar is
> needed to build and no C2ME version is pinned. It depends on C2ME at
> **runtime**: for the
> multi-threaded chunk generation that keeps the GPU batch pipeline fed (with
> C2ME, desktop throughput reaches ~144 chunks/s; without it, the GPU starves
> and collapses to ~13, roughly phone level), and — when `C2ME_WGPU_ASYNC_SCHED`
> is on (default) — to run the CPU write-back on C2ME's worker pool, so chunk
> requests never block a worker (see "How it works"). If the C2ME scheduling
> API is unavailable, GPU acceleration is simply left off for that run and
> plain C2ME generates terrain on the CPU (there is no synchronous GPU
> fallback). So it remains a **C2ME companion module** — C2ME supplies the
> parallelism, this mod supplies the GPU compute — and keeps the name.

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
| Android aarch64 (custom JVM) | ✅ tested (Kirin 980) |
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
   chunks in one dispatch.
5. **Block states** — the GPU classifies every block (air / stone / water /
   lava) itself, applying the aquifer and the Beardifier (structure shaping)
   on the GPU. Java only writes the resulting per-block state bytes into the
   chunk's palette.

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

- **Ore veins** are not block-exact with vanilla. They are pre-filtered on the
  GPU and finished with Minecraft's own sampler; this only affects ore
  placement, never the terrain itself.
- At extremely far End coordinates (~370k blocks), a partial NaN fallback is
  used to reproduce vanilla's void bands.
- **Modded density functions**: a custom density node the GPU serializer does
  not natively understand (e.g. from a terrain mod) is evaluated on the CPU at
  the same corner grid and fed to the GPU as an "external input". The rest of
  the tree still runs on the GPU, so modded worldgen keeps working — only the
  unsupported subtree pays a CPU cost.
- **Weak/mobile GPUs not yet re-benchmarked** under the block-state
  architecture. The old "queued behind the renderer" caveat applied to the
  previous CPU-write-back design; whether it still holds is unverified.

---

## Environment variables

### GPU selection (standard wgpu variables)

The mod honours wgpu's own environment variables, so they behave the same as
in any other wgpu application:

| Variable | Values | Effect |
|---|---|---|
| `WGPU_BACKEND` | `vulkan`, `dx12`, `metal`, `gl`, `webgpu` (comma-separated) | Restrict the graphics backends tried. Useful when one backend is broken (e.g. `WGPU_BACKEND=gl`). |
| `WGPU_ADAPTER_NAME` | substring (case-insensitive) | Force a specific GPU by name, e.g. `WGPU_ADAPTER_NAME=radeon`. If nothing matches, the mod logs a warning and falls back to its default choice. |

Without any of these, the mod uses Vulkan / Metal / DX12 (whichever the OS
provides) and prefers the **integrated GPU**, so the discrete GPU stays free
for rendering.

### Diagnostics and debug exports

Set these only when troubleshooting. All of them are read once at startup.

| Variable | Value | Effect |
|---|---|---|
| `C2ME_WGPU_DEBUG` | any non-empty value | Enables verbose logging: shader-compile timings, pipeline-cache hits/misses, and periodic chunk-per-second (`CPS=`) throughput stats. Written to `<gameDir>/logs/c2me-wgpu.log`. Silent by default. |
| `C2ME_WGPU_PROFILE` | any non-empty value | Emits per-stage GPU batch timing (encode / submit / readback / decode), aggregated as one averaged line every 64 batches. Written to `<gameDir>/logs/c2me-wgpu.log`. Independent of `C2ME_WGPU_DEBUG`. |
| `C2ME_VANILLA_REF` | `1` | Skip the GPU entirely, let vanilla generate the chunk, then dump its blocks for offline comparison against GPU output. |
| `C2ME_DUMP_DENSITY` | `1` | Dump the GPU density values of the chunk at (0, 0). |
| `C2ME_DUMP_BLOCKS` | `1` | Dump the block states of the chunk at (0, 0), for offline comparison against vanilla. |
| `C2ME_DUMP_BLOB` | file path | Dump the serialized NoiseRouter blob to the given path. |
| `C2ME_WGPU_BLOCK_OUT` | `1` | GPU emits per-block block states (core path; on by default in the dev setup). |
| `C2ME_WGPU_AQUIFER` | `1` | GPU aquifer (water / lava / stone classification). |
| `C2ME_WGPU_BEARD` | `1` | GPU Beardifier (structure terrain shaping). |
| `C2ME_WGPU_MAPPED_AT_CREATION` | `0` / `1` | Override how init-time buffers are uploaded. Default is `1` on desktop (fast mapped upload) and `0` on Android (Mali-G610 drivers reject the mapped path). Set `0` to force `queue.write_buffer`, `1` to force the mapped upload. |

The blob dump can also be set with the JVM property `-Dc2me.dumpBlob=<path>`;
the environment variable is only a fallback when the property is unset.

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
