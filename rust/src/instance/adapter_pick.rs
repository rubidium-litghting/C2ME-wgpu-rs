// SPDX-License-Identifier: MIT

//! Adapter selection policy: IntegratedGPU > DiscreteGPU > VirtualGpu.
//!
//! 设计意图：优先用核显生成地形，独显留给 Minecraft 渲染管线，避免 GPU
//! 资源争用导致渲染掉帧。仅在没有核显时才回退到独显。
//!
//! 不回退到 CPU 后端：C2ME-fabric 主项目本身已有成熟的 CPU 优化路径，
//! 本 wgpu 加速模块只负责 GPU 加速。没有可用 GPU 时应返回 Err 让上层
//! 自动回退到 C2ME 的 CPU 路径，而不是用 wgpu 的 CPU 后端（那个走的是
//! software rasterizer，比原生 CPU 路径慢得多）。
//!
//! User can override via the standard wgpu `WGPU_ADAPTER_NAME` env var
//! (case-insensitive substring match). If set but no adapter matches, we log
//! and fall back to the priority order below rather than aborting.
//!
//! NOTE: every environment variable here will eventually be replaced by the
//! 0.1.5 config file / GUI. The reads are kept as-is so the migration can be
//! done variable-by-variable against a known-good reference.

use crate::Result;
use std::env;

/// Pick the "best" adapter from a wgpu::Instance.
///
/// Order: explicit `WGPU_ADAPTER_NAME` (substring, case-insensitive) >
/// IntegratedGpu > DiscreteGpu > VirtualGpu > any non-CPU adapter.
///
/// Returns `Err` if no adapter is usable, so the caller can fall back to
/// C2ME's CPU path.
pub fn pick_best(instance: &wgpu::Instance) -> Result<wgpu::Adapter> {
    // Enumerate with `all()` rather than `PRIMARY`: when the user sets
    // `WGPU_BACKEND=gl`, the instance only enables GL, and `PRIMARY`
    // (Vulkan/Metal/DX12/WebGPU) would filter those adapters out entirely.
    // The instance itself only ever yields adapters for the backends it was
    // created with, so `all()` cannot introduce unwanted ones.
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    if let Ok(name) = env::var("WGPU_ADAPTER_NAME") {
        let needle = name.to_lowercase();
        if let Some(adapter) = adapters.iter().find(|&adp| {
            adp.get_info().name.to_lowercase().contains(&needle)
        }) {
            return Ok(adapter.clone());
        } else {
            let msg = format!(
                "[c2me-wgpu-rs] WGPU_ADAPTER_NAME=\"{name}\" matched no adapter; using default priority order");
            #[cfg(test)]
            eprintln!("{msg}");
            #[cfg(not(test))]
            crate::debug_log_line(&msg);
        }
    }
    let priority_order = [
        wgpu::DeviceType::IntegratedGpu,
        wgpu::DeviceType::DiscreteGpu,
        wgpu::DeviceType::VirtualGpu,
    ];

    for target_type in priority_order {
        if let Some(adapter) = adapters.iter().find(|adp| {
            adp.get_info().device_type == target_type
        }) {
            return Ok(adapter.clone());
        }
    }

    // Last resort: the GL backend often reports `DeviceType::Other` (it cannot
    // reliably classify the GPU), so the priority list above filters it out.
    // If the user explicitly asked for such a backend (e.g. WGPU_BACKEND=gl),
    // honour that by taking the first non-CPU adapter. CPU/software adapters
    // (llvmpipe, etc.) stay excluded — C2ME's own CPU path is faster.
    if let Some(adapter) = adapters.iter().find(|adp| {
        adp.get_info().device_type != wgpu::DeviceType::Cpu
    }) {
        crate::wgpu_debug!(
            "[c2me-wgpu-rs] no typed adapter matched; using first non-CPU: {} ({:?})",
            adapter.get_info().name, adapter.get_info().device_type
        );
        return Ok(adapter.clone());
    }

    Err(anyhow::anyhow!("No suitable adapter found"))
}