//! Vendor/driver-specific workaround flags.
//!
//! 这些 workaround 都是基于实测发现的 driver bug，必须按 vendor/device/driver
//! 精确匹配，避免误伤正常硬件。
//!
//!   - Intel UHD 620 on Mesa: fma(a,b,c) != a*b+c，会导致 Perlin 噪声漂移
//!   - Mesa RADV (AMD): storage buffer 大小未对齐 256 字节时偶发读到上一次
//!     dispatch 的脏数据
//!   - Apple Metal 老款独显: 单 buffer 绑定上限可能 < 256 MiB
//!   - 部分 NVIDIA Windows 驱动: dispatch 间未显式 clear_buffer 会读到脏数据

use wgpu::AdapterInfo;

/// Workaround flags enabled for the current adapter.
#[derive(Debug, Default, Clone, Copy)]
pub struct Workarounds {
    /// Intel UHD 620 + Mesa: codegen 时禁用 `fma()`，改用 `(a*b)+c` 两条指令。
    pub disable_storage_fma: bool,
    /// Mesa RADV (AMD Vulkan): resources 分配 buffer 时把 size 向上对齐到 256 字节。
    pub radv_round_buf_256: bool,
    /// Apple Metal 独显: 当单个 storage buffer 需求超过设备上限时自动拆分绑定。
    pub metal_split_large_buffers: bool,
    /// 部分 NVIDIA 驱动: scheduler 每次 dispatch 前必须 `clear_buffer`。
    pub force_clear_between_dispatches: bool,
}

impl Workarounds {
    // PCI Vendor IDs
    const VENDOR_INTEL: u32 = 0x8086;
    const VENDOR_NVIDIA: u32 = 0x10DE;
    const VENDOR_AMD: u32 = 0x1002;
    //const VENDOR_APPLE: u32 = 0x106B;

    // Intel UHD Graphics 620 (Kaby Lake-R) PCI Device ID
    const INTEL_UHD_620_DEVICE_ID: u32 = 0x5916;

    /// Detect workarounds from adapter info.
    ///
    /// 匹配逻辑要尽量保守：只针对确切观察到的 driver bug 启用 workaround，
    /// 避免对正常硬件产生性能/精度影响。
    pub fn detect(info: &AdapterInfo) -> Self {
        let driver_lower = info.driver.to_lowercase();
        let driver_info_lower = info.driver_info.to_lowercase();
        let is_mesa = driver_lower.contains("mesa") || driver_info_lower.contains("mesa");
        let is_radv = driver_lower.contains("radv") || driver_info_lower.contains("radv");
        let is_metal = info.backend == wgpu::Backend::Metal;

        Self {
            // 精确匹配 Intel UHD 620 + Mesa（不波及 Intel 其它型号）
            disable_storage_fma: info.vendor == Self::VENDOR_INTEL
                && info.device == Self::INTEL_UHD_620_DEVICE_ID
                && is_mesa,

            // RADV = AMD Vulkan Mesa driver
            radv_round_buf_256: info.vendor == Self::VENDOR_AMD && is_radv,

            // Metal backend 在老款 Intel Mac 独显上单 binding 上限可能 < 256 MiB
            metal_split_large_buffers: is_metal
                && info.device_type == wgpu::DeviceType::DiscreteGpu,

            // 保守策略：NVIDIA + Vulkan/DX12 都启用，实测问题在 535-545 驱动版本
            force_clear_between_dispatches: info.vendor == Self::VENDOR_NVIDIA
                && matches!(info.backend, wgpu::Backend::Dx12 | wgpu::Backend::Vulkan),
        }
    }

    /// 任一 workaround 启用即返回 true，用于启动日志。
    pub fn any(&self) -> bool {
        self.disable_storage_fma
            || self.radv_round_buf_256
            || self.metal_split_large_buffers
            || self.force_clear_between_dispatches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_info(vendor: usize, device: usize, driver: &str, backend: wgpu::Backend) -> AdapterInfo {
        AdapterInfo {
            name: "fake".into(),
            vendor: vendor as u32,
            device: device as u32,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: driver.into(),
            driver_info: String::new(),
            backend,
            device_pci_bus_id: String::new(),
            subgroup_min_size: 0,
            subgroup_max_size: 0,
            transient_saves_memory: None,
            limit_bucket: None,
        }
    }

    #[test]
    fn detect_intel_uhd_620_mesa() {
        let info = fake_info(0x8086, 0x5916, "Mesa Intel", wgpu::Backend::Vulkan);
        let w = Workarounds::detect(&info);
        assert!(w.disable_storage_fma);
        assert!(!w.radv_round_buf_256);
    }

    #[test]
    fn detect_amd_radv() {
        let info = fake_info(0x1002, 0x73BF, "radv", wgpu::Backend::Vulkan);
        let w = Workarounds::detect(&info);
        assert!(w.radv_round_buf_256);
        assert!(!w.disable_storage_fma);
    }

    #[test]
    fn no_workaround_for_normal_gpu() {
        let info = fake_info(0x10DE, 0x2684, "NVIDIA 550.0", wgpu::Backend::Vulkan);
        let w = Workarounds::detect(&info);
        assert!(w.force_clear_between_dispatches); // NVIDIA+Vulkan 启用 clear
        assert!(!w.disable_storage_fma);
        assert!(!w.radv_round_buf_256);
    }
}