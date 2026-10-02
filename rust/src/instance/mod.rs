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

use std::sync::OnceLock;
use wgpu::wgt::DeviceDescriptor;
use crate::Result;
use workarounds::Workarounds;

static INSTANCE: OnceLock<WgpuInstance> = OnceLock::new();

/// Singleton wgpu instance + picked device + queue.
///
/// WgpuInstance owns the wgpu::Instance, wgpu::Adapter, wgpu::Device and
/// wgpu::Queue. It is created once by `c2me_wgpu_create_instance` and shared
/// across all WorldContexts (one per Minecraft world).
pub struct WgpuInstance {
    /// Placeholder only. Real impl stores:
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pub workarounds: workarounds::Workarounds,
    _priv: (),
}

impl WgpuInstance {
    /// 获取全局单例。已初始化则直接返回引用，未初始化则进行初始化。
    /// 初始化失败时返回 Err，下次调用会重新尝试。
    pub fn instance() -> Result<&'static Self> {
        if let Some(inst) = INSTANCE.get() {
            return Ok(inst);
        }
        let inst = Self::init()?;
        // 竞争情况：两个线程同时 init，set 最多一个成功，失败者的 inst 被 drop
        let _ = INSTANCE.set(inst);
        Ok(INSTANCE.get().expect("INSTANCE must be Some after set"))
    }

    fn init() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY.with_env(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = adapter_pick::pick_best(&instance)?;
        
        features::check_required(&adapter)?;
        let adapter_info = adapter.get_info();
        let workarounds = Workarounds::detect(&adapter_info);
        if workarounds.any() {
            crate::wgpu_debug!("[c2me-wgpu-rs] enabled workarounds for {} ({}): {:?}", adapter_info.name, adapter_info.driver, workarounds);
        }

        let (device, queue) = pollster::block_on(
        adapter.request_device(&DeviceDescriptor{
            required_features: wgpu::Features::PIPELINE_CACHE,
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            label: None,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))?;
        
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            adapter_info,
            workarounds,
            _priv: (),
        })
    }
}
