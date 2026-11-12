// SPDX-License-Identifier: MIT

//! Cross-thread shader-compile progress.
//!
//! Pipeline compilation runs on a C2ME worker thread (inside `world_new`),
//! while the client renders on the main thread. The overlay has to answer
//! "is something compiling right now, and how far along is it?" without
//! blocking either side, so this module is just a tiny mutex-protected
//! snapshot.
//!
//! ## Why not an FFM upcall
//!
//! The existing `LOG_CALLBACK` is fire-and-forget text. Progress has
//! different semantics: the client only needs the *latest* state, and a
//! dropped update is harmless as long as the next poll sees the final
//! value. A pull model (`c2me_wgpu_compile_progress`) also means the Rust
//! side never has to touch Java objects from a worker thread, and a client
//! that is not rendering (paused, minimized) costs nothing.
//!
//! ## Overlapping compiles
//!
//! Two dimensions can in principle be initialised by different workers at
//! the same time. `active` is therefore a counter; `done`/`total` track the
//! most recent compile. The overlay only claims "compiling" while the
//! counter is non-zero, and a single-digit shard count makes the shared
//! `done`/`total` pair a cosmetic imprecision at worst.

use std::sync::Mutex;

#[derive(Clone, Copy, Default)]
struct Snapshot {
    active: u32,
    done: u32,
    total: u32,
}

static STATE: Mutex<Snapshot> = Mutex::new(Snapshot {
    active: 0,
    done: 0,
    total: 0,
});

/// RAII guard: marks compilation active for its lifetime.
///
/// Dropping it clears the flag even when a shard fails and `compile_pipeline`
/// returns early through `?`, so a failed compile cannot leave the overlay
/// stuck on screen forever.
pub struct CompileProgress {
    done: u32,
}

impl CompileProgress {
    /// Begin a compile of `total` shards. `total == 0` is treated as "unknown"
    /// and still reported as active so the overlay can show a spinner rather
    /// than nothing.
    pub fn start(total: usize) -> Self {
        let total = total.min(u32::MAX as usize) as u32;
        if let Ok(mut s) = STATE.lock() {
            s.active = s.active.saturating_add(1);
            s.done = 0;
            s.total = total;
        }
        Self { done: 0 }
    }

    /// Record that one more shard finished.
    pub fn step(&mut self) {
        self.done = self.done.saturating_add(1);
        if let Ok(mut s) = STATE.lock() {
            s.done = self.done;
        }
    }
}

impl Drop for CompileProgress {
    fn drop(&mut self) {
        if let Ok(mut s) = STATE.lock() {
            s.active = s.active.saturating_sub(1);
            if s.active == 0 {
                s.done = 0;
            }
        }
    }
}

/// Snapshot as `(active, done, total)`.
pub fn snapshot() -> (bool, u32, u32) {
    match STATE.lock() {
        Ok(s) => (s.active > 0, s.done, s.total),
        // A poisoned mutex means a compile panicked while holding it. Report
        // "inactive" rather than leaving the overlay pinned on screen.
        Err(_) => (false, 0, 0),
    }
}
