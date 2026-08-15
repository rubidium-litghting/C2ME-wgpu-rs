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

//! Instance module: discover GPU adapters, pick best device, check features,
//! apply vendor-specific workarounds.

pub mod adapter_pick;
pub mod features;
pub mod workarounds;

use crate::Result;

/// Singleton wgpu instance + picked device + queue.
///
/// WgpuInstance owns the wgpu::Instance, wgpu::Adapter, wgpu::Device and
/// wgpu::Queue. It is created once by `c2me_wgpu_create_instance` and shared
/// across all WorldContexts (one per Minecraft world).
pub struct WgpuInstance {
    /// Placeholder only. Real impl stores:
    ///   pub instance: wgpu::Instance,
    ///   pub adapter: wgpu::Adapter,
    ///   pub device: wgpu::Device,
    ///   pub queue: wgpu::Queue,
    ///   pub adapter_info: wgpu::AdapterInfo,
    ///   pub workarounds: workarounds::Workarounds,
    _priv: (),
}

impl WgpuInstance {
    /// Create a new instance: enumerate adapters, pick best (Discrete >
    /// Integrated > CPU), verify required features, apply workarounds.
    ///
    /// Skeleton: this is a no-op stub returning Ok. Real impl runs
    ///   pollster::block_on(instance.request_adapter(...)) etc.
    pub fn new() -> Result<Self> {
        // TODO:
        //   1. build wgpu::Instance with default Backends (PRIMARY: Vulkan/Metal/DX12)
        //   2. call adapter_pick::pick_best(instance)
        //   3. call features::check_required(&adapter) -> return MissingCapability if fail
        //   4. adapter.request_device with required_features/limits
        //   5. workarounds::detect(&adapter.get_info())
        Ok(Self { _priv: () })
    }
}
