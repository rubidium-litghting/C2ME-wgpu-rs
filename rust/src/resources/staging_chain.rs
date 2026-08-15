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

//! Three-phase staging-buffer fence tracking to prevent a newer dispatch
//! reading an in-flight older result (R07 risk).
//!
//! State machine per staging buffer:
//!   A = GPU computation in flight (buffer locked as OUTPUT)
//!   B = dispatch completed, CPU can map_read
//!   C = CPU consumed + unmapped → return to buffer_pool.
//!
//! We always call device.poll(Maintain::Wait) before transitioning A→B,
//! plus a debug_assert that the fence was actually reached.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagingPhase { A, B, C }

pub struct StagingChain {
    // TODO: Vec<(wgpu::Buffer, StagingPhase, Option<wgpu::BufferSlice>)>
}
