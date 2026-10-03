/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2026 RubidiumLighting
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

//! C2ME wgpu-rs GPU world generation accelerator.
//!
//! Architecture (90% Rust / <350 lines Java):
//!   Java (Panama FFM 5 downcalls, no upcall)
//!       │ handle + poll model
//!       ▼
//!   c_api.rs ─► instance / ast / codegen / world / scheduler / resources / result
//!
//! Java side must call:  create_instance → world_new → request_chunks → poll* → take_result_blob
//!
//! All numeric density values are pure `f32`. See 计划.txt §5 for the
//! three precision compensation mechanisms (offset coords, Kahan sum, i32 boundaries).

pub mod c_api;
pub mod error;

pub mod instance;
pub mod ast;
pub mod codegen;
pub mod world;
pub mod scheduler;
pub mod resources;
pub mod result;

pub use error::{C2meError, Result};

/// Whether verbose GPU/codegen diagnostics are enabled.
///
/// Gated by the `C2ME_WGPU_DEBUG` environment variable (any non-empty value).
/// Used by the `wgpu_debug!` macro for shader-compile / pipeline-cache logs
/// that would otherwise spam the game log on every launch.
pub(crate) fn debug_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var_os("C2ME_WGPU_DEBUG").is_some_and(|v| !v.is_empty())
    })
}

/// Whether to bypass wgpu's `PipelineCache` entirely.
///
/// Set `C2ME_WGPU_NO_PIPELINE_CACHE=1` to disable the on-disk/in-memory
/// `wgpu::PipelineCache` and skip requesting `Features::PIPELINE_CACHE`.
/// Some Vulkan drivers (observed on Mali-G610 / Dimensity 8100) report the
/// feature but abort inside `create_pipeline_cache` or the subsequent
/// `create_compute_pipeline`, so this switch lets such devices still run
/// (at the cost of recompiling the shader each launch).
pub(crate) fn no_pipeline_cache() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var_os("C2ME_WGPU_NO_PIPELINE_CACHE").is_some_and(|v| !v.is_empty())
    })
}

/// Append one line to the file named by `C2ME_WGPU_DEBUG_LOG`, if set.
///
/// Android's `latest.log` only captures stdout (Java logging), and stderr
/// from the native library is easily lost when the process aborts. Writing
/// directly to a file guarantees the last diagnostic lines survive a crash.
/// Each call opens/appends/closes so nothing stays buffered.
pub(crate) fn debug_log_line(msg: &str) {
    let Some(path) = std::env::var_os("C2ME_WGPU_DEBUG_LOG") else { return };
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{msg}");
    }
}

/// Whether to substitute a minimal bring-up shader for the real one.
///
/// Set `C2ME_WGPU_MINIMAL_SHADER=1` to emit a tiny compute shader that keeps
/// the exact same bind group layout + entry point signature but does almost
/// nothing. Used to tell "the driver cannot handle our 5k-line shader" apart
/// from "the driver cannot run wgpu compute shaders at all".
pub(crate) fn minimal_shader() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var_os("C2ME_WGPU_MINIMAL_SHADER").is_some_and(|v| !v.is_empty())
    })
}

/// Force single-shader (pre-split) mode. Set `C2ME_WGPU_FORCE_SINGLE=1`
/// to compile one `main_entry` shader over all channels instead of the
/// A/B/C shard split. Benchmarking switch + emergency fallback.
pub(crate) fn force_single() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var_os("C2ME_WGPU_FORCE_SINGLE").is_some_and(|v| !v.is_empty())
    })
}

/// Test-only: limit how many router channels are emitted (0 = all).
///
/// Set `C2ME_WGPU_MAX_CHANNELS=N` to emit only the first N channels. Used to
/// measure how shader-compile cost scales with shader size.
pub(crate) fn max_channels() -> usize {
    static FLAG: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("C2ME_WGPU_MAX_CHANNELS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    })
}

/// `eprintln!` that only prints when `C2ME_WGPU_DEBUG` is set.
///
/// Set `C2ME_WGPU_DEBUG=1` to see shader-compile timings, pipeline-cache
/// hits/misses and other codegen diagnostics. Silent by default.
/// Set `C2ME_WGPU_DEBUG_LOG=/path/to/file` to also append every line to
/// that file (useful on Android, where stderr is hard to capture).
#[macro_export]
macro_rules! wgpu_debug {
    ($($arg:tt)*) => {
        if $crate::debug_enabled() {
            let __msg = format!($($arg)*);
            eprintln!("{__msg}");
            $crate::debug_log_line(&__msg);
        }
    };
}
