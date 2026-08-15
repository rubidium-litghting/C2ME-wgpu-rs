//! Build script: embed math_lib.wgsl source string at compile time
//! so the binary is self-contained (no extra files to ship).

use std::env;
use std::path::Path;

fn main() {
    // Rerun if math_lib.wgsl changes
    println!("cargo:rerun-if-changed=src/codegen/math_lib.wgsl");

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("math_lib_wgsl.rs");

    // We'll use include_str! directly in the module instead; this build.rs
    // currently just declares the rerun-if-changed dep.
    let _ = dest_path;
}
