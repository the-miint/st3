// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Compile and run the documented C usage example (`examples/usage.c`) against
//! the built `libst3` shared library, so the worked example a downstream C
//! consumer copies from is guaranteed to keep compiling and running.
//!
//! This mirrors the end-to-end harness runner (`capi_harness.rs`): compile with
//! `cc`, including the cbindgen-generated `st3.h` (its directory is exported by
//! the build script as `ST3_HEADER_DIR`) and linking `-lst3`, then run under
//! `LD_LIBRARY_PATH`. If no C compiler is present the test skips gracefully so
//! the gate stays green on minimal environments. The two runners are kept
//! separate rather than factored: `capi_harness.rs` is the asserting harness,
//! this is the example gate, and they may diverge.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Whether a working C compiler is available.
fn have_cc(cc: &str) -> bool {
    Command::new(cc)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn c_usage_example_compiles_and_runs() {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    if !have_cc(&cc) {
        eprintln!("no C compiler ('{cc}') found; skipping the C usage example");
        return;
    }

    let header_dir = env!("ST3_HEADER_DIR");
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let example_c = PathBuf::from(manifest_dir).join("examples").join("usage.c");
    assert!(example_c.exists(), "missing {}", example_c.display());

    // The test binary lives at target/<profile>/deps/<name>-<hash>; the shared
    // library is two levels up at target/<profile>/libst3.so.
    let exe = std::env::current_exe().expect("current_exe");
    let profile_dir: &Path = exe
        .parent()
        .and_then(|p| p.parent())
        .expect("target/<profile> directory");
    let lib = profile_dir.join("libst3.so");
    assert!(
        lib.exists(),
        "libst3.so not found at {}; is st3-capi built as a cdylib?",
        lib.display()
    );

    let bin = profile_dir.join("st3_c_example");

    // Compile the example against the generated header and the shared library.
    let compile = Command::new(&cc)
        .arg("-std=c11")
        .arg("-Wall")
        .arg("-Wextra")
        .arg(format!("-I{header_dir}"))
        .arg(&example_c)
        .arg("-o")
        .arg(&bin)
        .arg(format!("-L{}", profile_dir.display()))
        .arg("-lst3")
        .arg("-lm")
        .output()
        .expect("failed to invoke the C compiler");
    assert!(
        compile.status.success(),
        "compiling usage.c failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr)
    );

    // Run it with the shared library on the loader path.
    let run = Command::new(&bin)
        .env("LD_LIBRARY_PATH", profile_dir)
        .output()
        .expect("failed to run the compiled example");
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "C usage example exited with {:?}:\nstdout:\n{stdout}\nstderr:\n{stderr}",
        run.status.code()
    );
    assert!(
        stdout.contains("usage example OK"),
        "C usage example did not report success:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}
