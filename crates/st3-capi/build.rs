// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Generate the C header `st3.h` from the crate's `#[repr(C)]` types and
//! `extern "C"` functions with cbindgen.
//!
//! The header is written into `OUT_DIR`, and its directory is exported to the
//! crate and its tests via the `ST3_HEADER_DIR` compile-time environment
//! variable so the C harness can `#include` it. A copy is committed at
//! `include/st3.h` for consumers, who cannot locate `OUT_DIR`; `make header`
//! refreshes it and the `committed_header_is_current` test keeps it honest.

use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by cargo");
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR is set by cargo");
    let header_path = PathBuf::from(&out_dir).join("st3.h");

    // Regenerate when the sources or the cbindgen config change.
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=build.rs");

    let config = cbindgen::Config::from_root_or_default(&crate_dir);
    let bindings = cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .expect("cbindgen generates st3.h");
    bindings.write_to_file(&header_path);

    // Let the crate (and its tests) locate the generated header.
    println!("cargo:rustc-env=ST3_HEADER_DIR={out_dir}");
}
