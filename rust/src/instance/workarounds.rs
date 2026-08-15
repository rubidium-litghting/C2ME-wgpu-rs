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

//! Vendor/driver-specific workaround flags.
//!
//! Current known workarounds (计划.txt §2):
//!   - Intel UHD 620 on Mesa: disable native storage-buffer FMA
//!     (fma(a,b,c) does not equal a*b+c on that driver; causes Perlin drift)
//!   - Mesa RADV (AMD): round storage buffer sizes up to 256 bytes
//!     to avoid occasional reads of stale data from last dispatch.

/// Workaround flags enabled for the current adapter.
#[derive(Debug, Default, Clone, Copy)]
pub struct Workarounds {
    /// Intel UHD 620 + Mesa: force `(a*b)+c` two-instruction instead of fma().
    pub disable_storage_fma: bool,
    /// Mesa RADV: round buffer sizes up to next multiple of 256 bytes.
    pub radv_round_buf_256: bool,
}

impl Workarounds {
    /// Detect workarounds from adapter info (vendor/device name + driver description).
    pub fn detect(_adapter_name: &str, _driver: &str) -> Self {
        // TODO skeleton:
        //   let is_intel_uhd_620 = adapter.vendor == 0x8086 && adapter.device == 0x5916
        //                        && driver_desc.contains("mesa");
        //   let is_radv = driver_desc.to_lowercase().contains("radv");
        Self::default()
    }
}
