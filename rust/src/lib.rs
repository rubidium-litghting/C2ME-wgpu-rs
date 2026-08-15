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
