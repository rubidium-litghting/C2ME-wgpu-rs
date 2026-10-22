// SPDX-License-Identifier: MIT

//! C2ME wgpu-rs GPU world generation accelerator.
//!
//! Architecture (90% Rust / <350 lines Java):
//!   Java (Panama FFM downcalls; one upcall for Rust->Java log forwarding)
//!       │ handle + poll model
//!       ▼
//!   c_api.rs ─► instance / ast / codegen / world / scheduler / resources / result
//!
//! Java side must call:  create_instance → world_new → request_chunks → poll* → take_result_blob
//!
//! All numeric density values are pure `f32`. Three precision compensation
//! mechanisms are used (offset coords, Kahan sum, i32 boundaries).

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

/// Runtime configuration, supplied by the Java side from `config/c2me-wgpu.json`.
///
/// Previously each of these was an independent `C2ME_WGPU_*` environment
/// variable, read once into a `OnceLock`. That made hot-reload impossible and
/// spread the schema across two languages. Now the Java side parses the file
/// and pushes the whole struct through one C API call; the rest of this crate
/// only ever reads these fields.
#[derive(Clone, Debug)]
pub struct Config {
    pub debug: bool,
    pub disable_pipeline_cache: bool,
    pub profile: bool,
    /// `None` = platform default (mapped on desktop, write_buffer on Android).
    /// `Some(true)` forces mapped-at-creation, `Some(false)` forces write_buffer.
    pub mapped_at_creation: Option<bool>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            debug: false,
            disable_pipeline_cache: false,
            profile: false,
            mapped_at_creation: None,
        }
    }
}

static CONFIG: std::sync::OnceLock<Config> = std::sync::OnceLock::new();

/// Store the config. Only the first call takes effect: the value is written
/// once from `c2me_wgpu_update_config` before any GPU work starts, and is
/// read-only afterwards, so an uncontended `OnceLock` load is all a getter
/// costs. Hot-reload will need this replaced with something mutable.
pub(crate) fn set_config(c: Config) {
    let _ = CONFIG.set(c);
}

pub(crate) fn config() -> &'static Config {
    CONFIG.get_or_init(Config::default)
}

/// Whether verbose GPU/codegen diagnostics are enabled.
///
/// Enabled by the `debug_verbose_logging` config field.
/// Used by the `wgpu_debug!` macro for shader-compile / pipeline-cache logs
/// that would otherwise spam the game log on every launch.
pub(crate) fn debug_enabled() -> bool {
    config().debug
}

/// Whether to bypass wgpu's `PipelineCache` entirely.
///
/// Set `disable_pipeline_cache` in the config file to disable the on-disk
/// and in-memory `wgpu::PipelineCache` and skip requesting `Features::PIPELINE_CACHE`.
/// Some Vulkan drivers (observed on Mali-G610 / Dimensity 8100) report the
/// feature but abort inside `create_pipeline_cache` or the subsequent
/// `create_compute_pipeline`, so this switch lets such devices still run
/// (at the cost of recompiling the shader each launch).
pub(crate) fn no_pipeline_cache() -> bool {
    config().disable_pipeline_cache
}

/// Installs a panic hook that writes the panic message to the "black box" log
/// and forces it to disk *before* the process aborts. This is critical for
/// `panic=abort` environments (like Android cross-compiles) where the JVM
/// would otherwise be killed without a chance to log the cause.
pub(crate) fn setup_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let msg = format!("[FATAL PANIC] {}", info);
        let path = log_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(f, "{msg}");
            let _ = f.flush();
            // Force OS to write to disk immediately.
            let _ = f.sync_all();
        }
        
        // Try to notify Java as well, though it may be too late.
        if let Some(cb) = LOG_CALLBACK.get() {
            let bytes = msg.as_bytes();
            cb(bytes.as_ptr(), bytes.len());
        }
    }));
}

/// Path to the Rust-side log file, set once by Java through
/// `c2me_wgpu_set_log_path`. Falls back to `<cwd>/logs/c2me-wgpu.log`
/// (the launcher's working directory is the game dir) when unset.
static LOG_PATH: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();

pub(crate) fn set_log_path(path: &str) {
    let _ = LOG_PATH.set(std::path::PathBuf::from(path));
}

fn log_path() -> std::path::PathBuf {
    LOG_PATH.get().cloned().unwrap_or_else(|| {
        std::env::current_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join("logs")
            .join("c2me-wgpu.log")
    })
}

/// Append one line to the Rust log file. The path is supplied by Java
/// (or defaults to `<cwd>/logs/c2me-wgpu.log`).
///
/// **Why an independent file instead of standard Java logging?**
/// GPU driver crashes (DeviceLost) or Rust segfaults can abort the process
/// instantly. If we routed logs through the Java SLF4J async queue, the last
/// few seconds of diagnostic information would be lost. This synchronous
/// file append acts as a crash-resistant "black box" for low-level debugging.
/// High-level status is still reported back to Java via the C API.
pub type LogCallback = extern "C" fn(*const u8, usize);
static LOG_CALLBACK: std::sync::OnceLock<LogCallback> = std::sync::OnceLock::new();

pub(crate) fn set_log_callback(cb: LogCallback) {
    let _ = LOG_CALLBACK.set(cb);
}

pub(crate) fn debug_log_line(msg: &str) {
    let path = log_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{msg}");
        let _ = f.flush(); // Force userspace buffer to OS
    }
    
    // Also notify Java if a callback is registered.
    if let Some(cb) = LOG_CALLBACK.get() {
        let bytes = msg.as_bytes();
        cb(bytes.as_ptr(), bytes.len());
    }
}

/// Whether to emit per-stage execution timing (encode / submit / fence /
/// map / decode). Enabled by the `debug_stage_timing` config field.
/// Separate from `debug_verbose_logging` so profiling can be turned on without
/// the compile-time / cache log noise.
pub(crate) fn profile_enabled() -> bool {
    config().profile
}

/// Print a profiling line to stderr AND the debug log file. Unlike
/// `wgpu_debug!` this does NOT require `debug_verbose_logging`; the caller
/// gates it on `profile_enabled()`.
pub(crate) fn profile_log(msg: &str) {
    eprintln!("{msg}");
    debug_log_line(msg);
}

/// `eprintln!` that only prints when `debug_verbose_logging` is set.
///
/// Enable it to see shader-compile timings, pipeline-cache hits/misses and
/// other codegen diagnostics. Silent by default.
/// Every line is also appended to the Rust log file (see
/// [`debug_log_line`]), whose path Java supplies at startup.
#[macro_export]
macro_rules! wgpu_debug {
    ($($arg:tt)*) => {
        if $crate::debug_enabled() {
            let __msg = format!($($arg)*);
            $crate::debug_log_line(&__msg);
        }
    };
}
