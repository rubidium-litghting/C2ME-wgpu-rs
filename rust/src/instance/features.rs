/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2021-2026 RubidiumLighting
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

//! Capability check — MUST PASS or we refuse to use the adapter.
//!
//! 4 mandatory checks (计划.txt §2 Instance 模块):
//!   1. max_storage_buffer_binding_size ≥ 256 MiB
//!   2. max_compute_workgroups_per_dimension ≥ 65535
//!   3. max_buffer_address_space_bits ≥ 31  (i.e. ≥ 2 GiB addressable)
//!   4. Features::PIPELINE_CACHE is present
//!
//! Explicitly NOT required: SHADER_F64 — we are pure fp32.

use crate::Result;

/// Run all 4 mandatory capability checks against an adapter.
/// Returns Ok on pass, Err(MissingCapability(name)) on first failure.
pub fn check_required() -> Result<()> {
    // TODO skeleton: take adapter limits + features and check:
    //   if adapter.limits().max_storage_buffer_binding_size < 256 * 1024 * 1024 -> fail
    //   if adapter.limits().max_compute_workgroups_per_dimension < 65535 -> fail
    //   if adapter.limits().max_buffer_address_space_bits < 31 -> fail
    //   if !adapter.features().contains(Features::PIPELINE_CACHE) -> fail
    Ok(())
}
