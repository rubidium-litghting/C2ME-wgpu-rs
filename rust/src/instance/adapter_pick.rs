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

//! Adapter selection policy: DiscreteGPU > IntegratedGPU > VirtualGpu > CPU.
//!
//! User can override via `C2ME_WGPU_ADAPTER_NAME` env var (substring match).

use crate::Result;

/// Pick the "best" adapter from a wgpu::Instance.
///
/// # Panics
/// Skeleton-only, does no I/O yet.
pub fn pick_best() -> Result<()> {
    // TODO skeleton:
    //   let adapters = instance.enumerate_adapters();
    //   if let Some(name) = env::var("C2ME_WGPU_ADAPTER_NAME") -> match first adapter whose info.name contains name
    //   else -> group by type, pick in order DiscreteGpu > IntegratedGpu > VirtualGpu > Cpu
    //   if none -> return Err(RequestAdapter("no adapters"))
    Ok(())
}
