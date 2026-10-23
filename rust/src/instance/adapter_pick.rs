// SPDX-License-Identifier: MIT

//! Adapter selection policy: capability first, then IntegratedGpu >
//! DiscreteGpu > VirtualGpu.
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
//! (case-insensitive substring match) or the `gpu_index` config field. If set
//! but nothing matches, we log and fall back to the priority order below
//! rather than aborting.

use crate::Result;
use std::env;

/// Enumerate the adapters the instance can see, in a stable order.
///
/// `all()` rather than `PRIMARY`: when the user sets `WGPU_BACKEND=gl`, the
/// instance only enables GL, and `PRIMARY` (Vulkan/Metal/DX12/WebGPU) would
/// filter those adapters out entirely. The instance itself only ever yields
/// adapters for the backends it was created with, so `all()` cannot introduce
/// unwanted ones.
///
/// Adapters that report `DeviceType::Cpu` are dropped here, because the index
/// this function returns is persisted in the config file as `gpu_index` and
/// shown in the GUI: a software rasterizer must not occupy a slot. Any adapter
/// that mislabels itself is still caught by the capability probe in
/// [`pick_best`], so this filter is a convenience, not the safety net.
pub fn enumerate(instance: &wgpu::Instance) -> Vec<wgpu::Adapter> {
    pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        .into_iter()
        .filter(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
        .collect()
}

/// Pick the "best" adapter that can actually run our shaders.
///
/// Selection is capability-driven, not name- or type-driven. Adapter metadata
/// lies in practice: Mesa's llvmpipe has been observed to advertise itself as a
/// regular GPU, and a `device_type` filter alone therefore lets a software
/// rasterizer through only to fail on `max_storage_buffer_binding_size`. Every
/// candidate is therefore probed with
/// [`features::check_required`](crate::instance::features::check_required) and
/// only passing adapters are eligible for any branch below.
///
/// Order: explicit `gpu_index` (>= 0) > `WGPU_ADAPTER_NAME` (substring,
/// case-insensitive) > IntegratedGpu > DiscreteGpu > VirtualGpu > any capable
/// adapter.
///
/// `gpu_index` is the position in [`enumerate`]; -1 means automatic. An
/// out-of-range *or incapable* index is logged and falls through to the
/// automatic order, so a stale config cannot make the mod unusable.
///
/// Returns `Err` if no adapter is usable, so the caller can fall back to
/// C2ME's CPU path.
pub fn pick_best(
    instance: &wgpu::Instance,
    gpu_index: i32,
    disable_pipeline_cache: bool,
) -> Result<wgpu::Adapter> {
    let adapters = enumerate(instance);

    // Probe once per adapter. The checks only read limits/features, so this is
    // cheap and keeps the branches below from re-testing the same candidate.
    let capable: Vec<(usize, &wgpu::Adapter)> = adapters
        .iter()
        .enumerate()
        .filter(|(_, a)| {
            crate::instance::features::check_required(a, disable_pipeline_cache).is_ok()
        })
        .collect();

    if gpu_index >= 0 {
        let idx = gpu_index as usize;
        match adapters.get(idx) {
            Some(a) => match crate::instance::features::check_required(a, disable_pipeline_cache) {
                Ok(()) => return Ok(a.clone()),
                Err(e) => {
                    crate::debug_log_line(&format!(
                        "[c2me-wgpu-rs] gpu_index={gpu_index} ({}) rejected: {e}; falling back to automatic selection",
                        a.get_info().name
                    ));
                }
            },
            None => {
                crate::debug_log_line(&format!(
                    "[c2me-wgpu-rs] gpu_index={gpu_index} out of range ({} adapters); using automatic selection",
                    adapters.len()
                ));
            }
        }
    }

    if let Ok(name) = env::var("WGPU_ADAPTER_NAME") {
        let needle = name.to_lowercase();
        // Search the capable set, not all adapters: honouring the name must
        // not resurrect an adapter that cannot run the shaders.
        if let Some((_, a)) = capable
            .iter()
            .find(|(_, a)| a.get_info().name.to_lowercase().contains(&needle))
        {
            return Ok((*a).clone());
        }
        let msg = format!(
            "[c2me-wgpu-rs] WGPU_ADAPTER_NAME=\"{name}\" matched no usable adapter; using default priority order"
        );
        #[cfg(test)]
        eprintln!("{msg}");
        #[cfg(not(test))]
        crate::debug_log_line(&msg);
    }

    let priority_order = [
        wgpu::DeviceType::IntegratedGpu,
        wgpu::DeviceType::DiscreteGpu,
        wgpu::DeviceType::VirtualGpu,
    ];

    for target_type in priority_order {
        if let Some((_, a)) = capable
            .iter()
            .find(|(_, a)| a.get_info().device_type == target_type)
        {
            return Ok((*a).clone());
        }
    }

    // Last resort: the GL backend often reports `DeviceType::Other`, so the
    // priority list above filters it out. Take any adapter that passed the
    // capability checks — software rasterizers are excluded by construction.
    if let Some((_, a)) = capable.first() {
        crate::wgpu_debug!(
            "[c2me-wgpu-rs] no typed adapter matched; using first capable: {} ({:?})",
            a.get_info().name,
            a.get_info().device_type
        );
        return Ok((*a).clone());
    }

    // Nothing passed. Report which limit the first adapter is short of, so the
    // log explains the fallback instead of a bare "no suitable adapter".
    if let Some(a) = adapters.first() {
        if let Err(e) = crate::instance::features::check_required(a, disable_pipeline_cache) {
            let msg = format!(
                "[c2me-wgpu-rs] no capable adapter; {} rejected: {e}",
                a.get_info().name
            );
            crate::debug_log_line(&msg);
            return Err(anyhow::anyhow!("{e}"));
        }
    }

    Err(anyhow::anyhow!("No suitable adapter found"))
}
